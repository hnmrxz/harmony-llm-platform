// Host-side test of the Runtime import pipeline:
//   .hllm (deflate ZIP) -> ZipReader::ExtractAll -> PackageReader::ReadManifest
//   -> VerifyIntegrity -> LocateEngineFiles.
// Built with a plain host g++ against native/ (no OHOS NAPI dependency).
#include <sys/stat.h>
#include <cstdio>
#include <string>
#include <vector>

#include "model_package/ZipReader.h"
#include "model_package/PackageReader.h"
#include "model_package/Manifest.h"
#include "runtime/ModelManager.h"

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

    // 4. ModelManager: import into a durable store and drive the state machine.
    const std::string store = package + ".store";
    ModelManager manager(store);
    std::string modelId;
    errors.clear();
    bool imported = manager.ImportPackage(package, modelId, errors);
    check(imported, "modelmanager import + install");
    for (const auto& e : errors) {
        std::printf("    err: %s\n", e.c_str());
    }
    check(!modelId.empty(), "stable model id assigned");

    InstalledModel model;
    check(manager.GetModel(modelId, model), "retrieve installed model");
    check(model.state == ModelState::Installed, "state == INSTALLED after import");
    check(model.name == "Qwen3-8B", "installed model name");

    // Advance the lifecycle state and confirm persistence.
    check(manager.SetState(modelId, ModelState::Ready), "set state READY");
    ModelManager reloaded(store);
    InstalledModel after;
    check(reloaded.GetModel(modelId, after) && after.state == ModelState::Ready,
          "state persisted across store reload");

    EngineFiles loadedFiles;
    check(reloaded.GetEngineFiles(modelId, loadedFiles) && !loadedFiles.omcPath.empty(),
          "loaded engine files from store");

    // 5. ImportFolder from an extracted .hllm (folder with manifest.json).
    const std::string extractedDir = package + ".ex";
    ZipReader z2;
    z2.Open(package);
    z2.ExtractAll(extractedDir);
    ModelManager folderStore(store + ".folder");
    std::string folderId;
    errors.clear();
    check(folderStore.ImportFolder(extractedDir, folderId, errors),
          "importfolder(extracted .hllm folder)");
    for (const auto& e : errors) {
        std::printf("    err: %s\n", e.c_str());
    }
    check(!folderId.empty() && folderStore.GetEngineFiles(folderId, loadedFiles) &&
              !loadedFiles.omcPath.empty(),
          "folder model loaded from store");

    // 6. ImportFolder from a raw folder of pre-converted CANN files (no manifest).
    const std::string rawDir = package + ".raw";
    ::mkdir(rawDir.c_str(), 0755);
    auto writeFile = [&](const std::string& rel, const char* data, size_t n) {
        FILE* f = fopen((rawDir + "/" + rel).c_str(), "wb");
        fwrite(data, 1, n, f);
        fclose(f);
    };
    writeFile("qwen.omc", "OMC", 3);
    writeFile("context.json", "{}", 2);
    writeFile("executor.json", "{}", 2);
    writeFile("tokenizer.json", "{}", 2);
    ModelManager rawStore(store + ".raw");
    std::string rawId;
    errors.clear();
    check(rawStore.ImportFolder(rawDir, rawId, errors), "importfolder(raw CANN files folder)");
    check(!rawId.empty(), "raw folder model id assigned");
    // The generated manifest must exist in the installed copy.
    EngineFiles rawFiles;
    check(rawStore.GetEngineFiles(rawId, rawFiles) && !rawFiles.contextJsonPath.empty(),
          "raw folder engine files located");

    if (failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
