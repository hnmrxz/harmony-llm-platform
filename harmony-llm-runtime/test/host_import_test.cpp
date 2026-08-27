// Host-side test of the Runtime import pipeline:
//   .hllm (deflate ZIP) -> ZipReader::ExtractAll -> PackageReader::ReadManifest
//   -> VerifyIntegrity -> LocateEngineFiles.
// Built with a plain host g++ against native/ (no OHOS NAPI dependency).
#include <cstdio>
#include <string>
#include <vector>

#include "model_package/ZipReader.h"
#include "model_package/PackageReader.h"
#include "model_package/Manifest.h"

using namespace hllm;

static int failures = 0;

static void check(bool cond, const char* what) {
    std::printf("  %s %s\n", cond ? "[ OK ]" : "[FAIL]", what);
    if (!cond) {
        ++failures;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <test_model.hllm>\n", argv[0]);
        return 2;
    }
    const std::string package = argv[1];
    const std::string installDir = package + ".d";

    // 1. Unpack
    ZipReader zip;
    check(zip.Open(package), "zip.Open");
    bool extractOk = zip.ExtractAll(installDir);
    check(extractOk, "zip.ExtractAll");

    // 2. Parse + verify manifest
    Manifest manifest;
    std::vector<std::string> errors;
    bool manifestOk = PackageReader::ReadManifest(installDir, manifest, errors);
    check(manifestOk && errors.empty(), "package manifest parse");
    if (!manifestOk) {
        for (const auto& e : errors) {
            std::printf("    err: %s\n", e.c_str());
        }
        return 1;
    }
    check(manifest.schemaVersion == "1.0", "schema_version == 1.0");
    check(manifest.target.chip == "kirinx90", "target chip == kirinx90");

    errors.clear();
    bool integrityOk = PackageReader::VerifyIntegrity(installDir, manifest, errors);
    check(integrityOk, "artifact integrity (existence + sha256)");
    for (const auto& e : errors) {
        std::printf("    err: %s\n", e.c_str());
    }

    // 3. Locate engine files
    EngineFiles engineFiles;
    errors.clear();
    bool located = PackageReader::LocateEngineFiles(installDir, manifest, engineFiles, errors);
    check(located, "locate engine files");
    check(!engineFiles.omcPath.empty(), "omc path located");
    check(!engineFiles.contextJsonPath.empty(), "context.json located");
    check(!engineFiles.executorJsonPath.empty(), "executor.json located");
    check(!engineFiles.tokenizerJsonPath.empty(), "tokenizer.json located");

    if (failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
