#include "PackageReader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <utility>

#include "Json.h"
#include "Sha256.h"

namespace hllm {

namespace {

using json::Value;

bool FileExists(const std::string& absPath, std::uint64_t* sizeOut = nullptr) {
    struct stat st;
    if (::stat(absPath.c_str(), &st) != 0) {
        return false;
    }
    if (!S_ISREG(st.st_mode)) {
        return false;
    }
    if (sizeOut != nullptr) {
        *sizeOut = static_cast<std::uint64_t>(st.st_size);
    }
    return true;
}

/*
 * Normalize a package-relative path: reject absolute paths, and resolve '.'
 * and '..' segments. Returns false if the path climbs above the root.
 */
bool NormalizeRelPath(const std::string& relPath, std::string& outNormalized) {
    if (relPath.empty() || relPath.front() == '/') {
        return false;  // absolute paths are rejected per spec
    }
    std::vector<std::string> parts;
    std::istringstream stream(relPath);
    std::string seg;
    while (std::getline(stream, seg, '/')) {
        if (seg.empty() || seg == ".") {
            continue;
        }
        if (seg == "..") {
            if (parts.empty()) {
                return false;  // climbs above root
            }
            parts.pop_back();
            continue;
        }
        parts.push_back(seg);
    }
    std::ostringstream out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            out << '/';
        }
        out << parts[i];
    }
    outNormalized = out.str();
    return true;
}

std::string GetString(const Value* v, const std::string& key) {
    return v != nullptr && v->is_object() ? v->get(key)->as_string() : std::string();
}

std::int64_t GetInt(const Value* v, const std::string& key, std::int64_t fallback = 0) {
    if (v == nullptr || !v->is_object()) {
        return fallback;
    }
    const Value* child = v->get(key);
    return child != nullptr ? child->as_int64(fallback) : fallback;
}

}  // namespace

bool PackageReader::SafeJoin(const std::string& rootDir, const std::string& relPath,
                             std::string& outAbs) {
    return SafeJoinImpl(rootDir, relPath, outAbs);
}

bool PackageReader::SafeJoinImpl(const std::string& rootDir, const std::string& relPath,
                                 std::string& outAbs) {
    std::string normalized;
    if (!NormalizeRelPath(relPath, normalized)) {
        return false;
    }
    std::string base = rootDir;
    while (!base.empty() && base.back() == '/') {
        base.pop_back();
    }
    outAbs = base + (normalized.empty() ? "" : "/" + normalized);

    // Containment check: the joined path must start with `base`.
    if (outAbs.size() < base.size() ||
        outAbs.compare(0, base.size(), base) != 0) {
        return false;
    }
    return true;
}

bool PackageReader::ReadFile(const std::string& absPath, std::string& outBytes) {
    std::uint64_t size = 0;
    if (!FileExists(absPath, &size)) {
        return false;
    }
    int fd = ::open(absPath.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    std::string bytes(size, '\0');
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        ssize_t n = ::read(fd, &bytes[offset], bytes.size() - offset);
        if (n <= 0) {
            ::close(fd);
            return false;
        }
        offset += static_cast<std::size_t>(n);
    }
    ::close(fd);
    outBytes = std::move(bytes);
    return true;
}

bool PackageReader::ReadManifest(const std::string& rootDir, Manifest& manifest,
                                 std::vector<std::string>& errors) {
    std::string manifestPath;
    if (!SafeJoinImpl(rootDir, "manifest.json", manifestPath)) {
        errors.push_back("manifest.json path is invalid");
        return false;
    }
    std::string bytes;
    if (!ReadFile(manifestPath, bytes)) {
        errors.push_back("manifest.json not found or unreadable");
        return false;
    }
    auto tree = json::Parse(bytes);
    if (tree == nullptr || !tree->is_object()) {
        errors.push_back("manifest.json is not valid JSON");
        return false;
    }

    manifest = Manifest{};
    manifest.schemaVersion = GetString(tree.get(), "schema_version");

    const Value* model = tree->get("model");
    manifest.model.name = GetString(model, "name");
    manifest.model.family = GetString(model, "family");
    manifest.model.architecture = GetString(model, "architecture");
    manifest.model.source_type = GetString(model, "source_type");
    manifest.model.source_id = GetString(model, "source_id");
    manifest.model.revision = GetString(model, "revision");

    const Value* quant = tree->get("quantization");
    manifest.quantization.type = GetString(quant, "type");
    manifest.quantization.bits = GetInt(quant, "bits");
    manifest.quantization.group_size = GetInt(quant, "group_size");

    const Value* target = tree->get("target");
    manifest.target.backend = GetString(target, "backend");
    manifest.target.chip = GetString(target, "chip");
    manifest.target.runtime_version = GetString(target, "runtime_version");

    const Value* runtime = tree->get("runtime");
    manifest.runtime.contextLength = GetInt(runtime, "context_length");
    manifest.runtime.minimumMemoryMb = GetInt(runtime, "minimum_memory_mb");

    const Value* llmEngine = tree->get("llm_engine");
    if (llmEngine != nullptr && llmEngine->is_object()) {
        manifest.llmEngine.engine_type = GetString(llmEngine, "engine_type");
        manifest.llmEngine.kv_cache_max_len = GetInt(llmEngine, "kv_cache_max_len");
        manifest.llmEngine.prefill_len = GetInt(llmEngine, "prefill_len");
        manifest.llmEngine.max_io_tokens = GetInt(llmEngine, "max_io_tokens");
        manifest.llmEngine.vocab_size = GetInt(llmEngine, "vocab_size");
        manifest.llmEngine.hidden_size = GetInt(llmEngine, "hidden_size");
        manifest.llmEngine.num_hidden_layers = GetInt(llmEngine, "num_hidden_layers");
        manifest.llmEngine.num_attention_kv_heads = GetInt(llmEngine, "num_attention_kv_heads");
        manifest.llmEngine.num_attention_head_dims = GetInt(llmEngine, "num_attention_head_dims");
        manifest.llmEngine.max_position_embeddings = GetInt(llmEngine, "max_position_embeddings");
        manifest.llmEngine.embedding_input_type = GetString(llmEngine, "embedding_input_type");
    }

    const Value* build = tree->get("build");
    manifest.build.converter_version = GetString(build, "converter_version");
    manifest.build.git_commit = GetString(build, "git_commit");
    manifest.build.python_version = GetString(build, "python_version");

    manifest.artifacts.clear();
    const Value* artifacts = tree->get("artifacts");
    if (artifacts != nullptr && artifacts->is_array()) {
        for (std::size_t i = 0; i < artifacts->size(); ++i) {
            const Value* entry = artifacts->at(i);
            if (entry == nullptr || !entry->is_object()) {
                continue;
            }
            Artifact artifact;
            artifact.type = GetString(entry, "type");
            artifact.path = GetString(entry, "path");
            artifact.sha256 = GetString(entry, "sha256");
            artifact.size = static_cast<std::uint64_t>(GetInt(entry, "size"));
            if (!artifact.path.empty()) {
                manifest.artifacts.push_back(std::move(artifact));
            }
        }
    }

    if (manifest.schemaVersion.empty() || manifest.schemaVersion != "1.0") {
        errors.push_back("unsupported manifest schema_version: '" + manifest.schemaVersion + "'");
        return false;
    }
    return true;
}

bool PackageReader::VerifyIntegrity(const std::string& rootDir, const Manifest& manifest,
                                    std::vector<std::string>& errors) {
    for (const auto& artifact : manifest.artifacts) {
        std::string absPath;
        if (!SafeJoinImpl(rootDir, artifact.path, absPath)) {
            errors.push_back("invalid artifact path: " + artifact.path);
            continue;
        }
        if (!FileExists(absPath)) {
            errors.push_back("missing artifact: " + artifact.path);
            continue;
        }
        if (artifact.sha256.empty()) {
            continue;  // integrity not declared; treat as unknown, not a hard failure
        }
        std::string bytes;
        if (!ReadFile(absPath, bytes)) {
            errors.push_back("unreadable artifact: " + artifact.path);
            continue;
        }
        const std::string actual = Sha256Hex(bytes);
        if (actual != artifact.sha256) {
            errors.push_back("checksum mismatch: " + artifact.path);
        }
    }
    return errors.empty();
}

bool PackageReader::LocateEngineFiles(const std::string& rootDir, const Manifest& manifest,
                                      EngineFiles& out, std::vector<std::string>& errors) {
    for (const auto& artifact : manifest.artifacts) {
        std::string absPath;
        if (!SafeJoinImpl(rootDir, artifact.path, absPath)) {
            continue;
        }
        const std::string base = artifact.path.substr(artifact.path.find_last_of('/') + 1);
        if (base == "context.json") {
            out.contextJsonPath = absPath;
        } else if (base == "executor.json") {
            out.executorJsonPath = absPath;
        } else if (base == "tokenizer.json") {
            out.tokenizerJsonPath = absPath;
        } else if (artifact.path.size() > 4 && artifact.path.substr(artifact.path.size() - 4) == ".omc") {
            out.omcPath = absPath;
        } else if (base == "SubGraph_0.weight") {
            // weight directory = parent of the weight file (engine expects dir).
            const std::size_t slash = absPath.find_last_of('/');
            out.weightDir = slash == std::string::npos ? absPath : absPath.substr(0, slash);
        }
    }
    return !out.omcPath.empty() && !out.contextJsonPath.empty() && !out.executorJsonPath.empty();
}

}  // namespace hllm
