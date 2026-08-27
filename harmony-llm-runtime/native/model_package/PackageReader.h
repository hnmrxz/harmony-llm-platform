#pragma once

#include <string>
#include <vector>

#include "Manifest.h"

namespace hllm {

/*
 * PackageReader validates an extracted `.hllm` package on disk.
 *
 * A `.hllm` is a ZIP archive; the runtime extracts it to a temporary install
 * directory (using the platform archive API) and then calls this module to:
 *   1. Parse and validate manifest.json,
 *   2. Verify every declared artifact's SHA-256 and path,
 *   3. Locate the CANN LLM Engine files (omc / weight dir / embedding /
 *      context.json / executor.json / tokenizer.json).
 *
 * All paths are treated as untrusted data: an artifact path that escapes the
 * package root (absolute, or containing `..`) is rejected.
 */

struct EngineFiles {
    std::string contextJsonPath;
    std::string executorJsonPath;
    std::string omcPath;
    std::string weightDir;
    std::string tokenizerJsonPath;
};

class PackageReader {
public:
    /* Parse manifest.json under `rootDir`. On success returns true and fills
     * `manifest`; otherwise returns false and appends reasons to `errors`. */
    static bool ReadManifest(const std::string& rootDir, Manifest& manifest,
                             std::vector<std::string>& errors);

    /* Verify every artifact referenced by `manifest` exists on disk, stays
     * under `rootDir`, and matches the declared SHA-256. Safe on success. */
    static bool VerifyIntegrity(const std::string& rootDir, const Manifest& manifest,
                                std::vector<std::string>& errors);

    /* Locate the engine files referenced by the manifest + config, using the
     * executor.json's model/weight paths when present. */
    static bool LocateEngineFiles(const std::string& rootDir, const Manifest& manifest,
                                  EngineFiles& out, std::vector<std::string>& errors);

    /* Read a whole file into `outBytes`. Returns false on I/O error. */
    static bool ReadFile(const std::string& absPath, std::string& outBytes);

    /* Join `rootDir` with a package-relative `relPath` after rejecting any
     * traversal (absolute path, or a path that climbs above rootDir). */
    static bool SafeJoin(const std::string& rootDir, const std::string& relPath,
                         std::string& outAbs);

private:
    static bool SafeJoinImpl(const std::string& rootDir, const std::string& relPath,
                             std::string& outAbs);
};

}  // namespace hllm
