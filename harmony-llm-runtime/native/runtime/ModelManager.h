#pragma once

#include <map>
#include <string>
#include <vector>

#include "../model_package/Manifest.h"
#include "../model_package/PackageReader.h"
#include "ModelState.h"

namespace hllm {

/*
 * A model registered in the runtime's model store, with the engine files the
 * native runtime needs to load it.
 */
struct InstalledModel {
    std::string id;
    std::string name;
    std::string family;
    std::string targetChip;
    ModelState state = ModelState::Unset;
    EngineFiles files;
};

/*
 * ModelManager is the durable model store on device. It imports a `.hllm`
 * (unpack + verify), atomically installs it under the install root, records the
 * lifecycle state, and exposes the engine files for the runtime to load.
 *
 * The index is persisted as JSON under `<installRoot>/models.json`.
 */
class ModelManager {
public:
    explicit ModelManager(const std::string& installRoot);

    /* Unpack, verify and atomically install a `.hllm`. On success fills
     * `modelId` (a stable id derived from the manifest name) and sets state. */
    bool ImportPackage(const std::string& hllmPath, std::string& modelId,
                       std::vector<std::string>& errors);

    bool SetState(const std::string& modelId, ModelState state);

    bool GetModel(const std::string& modelId, InstalledModel& out) const;
    bool GetEngineFiles(const std::string& modelId, EngineFiles& out) const;
    std::vector<InstalledModel> List() const;

private:
    bool LoadIndex();
    bool SaveIndex();
    static std::string SanitizeId(const std::string& name);

    std::string installRoot_;
    std::map<std::string, InstalledModel> models_;
};

}  // namespace hllm
