#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hllm {

/*
 * Minimal ZIP archive reader used to unpack a `.hllm` package. Supports stored
 * (method 0) and deflated (method 8) entries, which is what the Ubuntu
 * Converter emits (`zipfile.ZIP_DEFLATED`).
 *
 * Every extraction path is treated as untrusted: absolute paths and entries
 * that climb above the extraction root (via `..`) are rejected. The runtime
 * never executes content from the archive.
 *
 * ZIP64 archives (a >4GB 27B model may trigger them) are detected but this
 * reference reader only supports standard 32-bit sizes; the device build may
 * substitute libarchive/zlib for full ZIP64 support.
 */
class ZipReader {
public:
    struct Entry {
        std::string name;
        std::uint16_t method = 0;  // 0=stored, 8=deflate
        std::uint32_t compressedSize = 0;
        std::uint32_t uncompressedSize = 0;
        std::uint32_t localOffset = 0;
        bool zip64 = false;
    };

    /* Parse the archive at `path` and fill `entries_`. */
    bool Open(const std::string& path);

    const std::vector<Entry>& entries() const { return entries_; }

    /* Decompress a single entry into `out`. */
    bool ReadEntry(const Entry& entry, std::string& out) const;

    /* Extract one entry to `destDir`, safely joining the entry name. */
    bool Extract(const Entry& entry, const std::string& destDir,
                 std::string* writtenPath = nullptr) const;

    /* Extract every entry to `destDir`, skipping directory entries. */
    bool ExtractAll(const std::string& destDir) const;

private:
    std::vector<Entry> entries_;
    std::string archiveBytes_;
};

}  // namespace hllm
