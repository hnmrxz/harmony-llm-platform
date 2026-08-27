#include "ZipReader.h"

#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#include <utility>

#include "Inflate.h"

namespace hllm {

namespace {
constexpr std::uint32_t kEocdSig = 0x06054b50;
constexpr std::uint32_t kCentralSig = 0x02014b50;
constexpr std::uint32_t kLocalSig = 0x04034b50;
constexpr std::uint32_t kMaxComment = 65535;

bool ReadAll(const std::string& path, std::string& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 0) {
        fclose(f);
        return false;
    }
    out.resize(static_cast<std::size_t>(len));
    if (len > 0 && fread(&out[0], 1, static_cast<std::size_t>(len), f) !=
                       static_cast<std::size_t>(len)) {
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

std::uint16_t U16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}
std::uint32_t U32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3U] << 24));
}
std::uint16_t U16At(const std::string& s, std::size_t i) {
    return U16(reinterpret_cast<const std::uint8_t*>(&s[i]));
}
std::uint32_t U32At(const std::string& s, std::size_t i) {
    return U32(reinterpret_cast<const std::uint8_t*>(&s[i]));
}

std::string NormalizeName(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (char c : raw) {
        if (c == '\\') {
            c = '/';
        }
        out.push_back(c);
    }
    return out;
}

bool IsSafeRelative(const std::string& name) {
    if (name.empty() || name.front() == '/') {
        return false;
    }
    std::size_t start = 0;
    while (start <= name.size()) {
        std::size_t end = name.find('/', start);
        if (end == std::string::npos) {
            end = name.size();
        }
        if (name.substr(start, end - start) == "..") {
            return false;
        }
        start = end + 1;
        if (start > name.size()) {
            break;
        }
    }
    return true;
}

bool WriteBytes(const std::string& absPath, const std::string& data) {
    FILE* f = fopen(absPath.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    if (!data.empty() && fwrite(data.data(), 1, data.size(), f) != data.size()) {
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

bool MkDirs(const std::string& absPath) {
    std::size_t pos = 1;  // keep the leading '/'
    while ((pos = absPath.find('/', pos)) != std::string::npos) {
        std::string dir = absPath.substr(0, pos);
        if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
        ++pos;
    }
    return true;
}

}  // namespace

bool ZipReader::Open(const std::string& path) {
    if (!ReadAll(path, archiveBytes_)) {
        return false;
    }
    entries_.clear();

    const std::string& data = archiveBytes_;
    if (data.size() < 22) {
        return false;
    }
    std::size_t searchStart = data.size() > kMaxComment + 22 ? data.size() - kMaxComment - 22 : 0;
    std::size_t eocd = std::string::npos;
    for (std::size_t i = data.size() - 22; i >= searchStart && i != std::string::npos; --i) {
        if (U32At(data, i) == kEocdSig) {
            eocd = i;
            break;
        }
        if (i == searchStart) {
            break;
        }
    }
    if (eocd == std::string::npos) {
        return false;
    }

    std::uint16_t entryCount = U16At(data, eocd + 10);
    std::uint32_t cdSize = U32At(data, eocd + 12);
    std::uint32_t cdOffset = U32At(data, eocd + 16);
    if (entryCount == 0xFFFF || cdSize == 0xFFFFFFFF || cdOffset == 0xFFFFFFFF) {
        return false;  // ZIP64 unsupported by this reference reader
    }
    (void)cdSize;

    std::size_t pos = cdOffset;
    for (std::uint16_t i = 0; i < entryCount; ++i) {
        if (pos + 46 > data.size() || U32At(data, pos) != kCentralSig) {
            return false;
        }
        Entry entry;
        entry.method = U16At(data, pos + 10);
        entry.compressedSize = U32At(data, pos + 20);
        entry.uncompressedSize = U32At(data, pos + 24);
        std::uint16_t nameLen = U16At(data, pos + 28);
        std::uint16_t extraLen = U16At(data, pos + 30);
        std::uint16_t commentLen = U16At(data, pos + 32);
        std::uint32_t localOffset = U32At(data, pos + 42);
        entry.name = NormalizeName(data.substr(pos + 46, nameLen));
        entry.localOffset = localOffset;
        entry.zip64 = (entry.compressedSize == 0xFFFFFFFF || entry.uncompressedSize == 0xFFFFFFFF);
        entries_.push_back(std::move(entry));
        pos += 46 + nameLen + extraLen + commentLen;
    }
    return true;
}

bool ZipReader::ReadEntry(const Entry& entry, std::string& out) const {
    if (entry.zip64) {
        return false;
    }
    if (entry.localOffset + 30 > archiveBytes_.size() ||
        U32At(archiveBytes_, entry.localOffset) != kLocalSig) {
        return false;
    }
    const std::string& data = archiveBytes_;
    std::size_t lp = entry.localOffset;
    std::uint16_t nameLen = U16At(data, lp + 26);
    std::uint16_t extraLen = U16At(data, lp + 28);
    std::uint32_t compressedSize = U32At(data, lp + 18);
    std::size_t dataStart = lp + 30 + nameLen + extraLen;
    if (dataStart + compressedSize > data.size()) {
        return false;
    }
    const std::uint8_t* payload = reinterpret_cast<const std::uint8_t*>(&data[dataStart]);
    if (entry.method == 0) {
        out.assign(reinterpret_cast<const char*>(payload), compressedSize);
        return true;
    }
    if (entry.method == 8) {
        return Inflate(payload, compressedSize, out);
    }
    return false;  // unsupported compression method
}

bool ZipReader::Extract(const Entry& entry, const std::string& destDir,
                        std::string* writtenPath) const {
    if (!IsSafeRelative(entry.name)) {
        return false;
    }
    std::string absPath = destDir;
    if (!absPath.empty() && absPath.back() != '/') {
        absPath.push_back('/');
    }
    absPath += entry.name;
    if (!MkDirs(absPath)) {
        return false;
    }
    std::string bytes;
    if (!ReadEntry(entry, bytes)) {
        return false;
    }
    if (!WriteBytes(absPath, bytes)) {
        return false;
    }
    if (writtenPath != nullptr) {
        *writtenPath = absPath;
    }
    return true;
}

bool ZipReader::ExtractAll(const std::string& destDir) const {
    for (const auto& entry : entries_) {
        if (!entry.name.empty() && entry.name.back() == '/') {
            continue;  // directory entry
        }
        if (!Extract(entry, destDir)) {
            return false;
        }
    }
    return true;
}

}  // namespace hllm
