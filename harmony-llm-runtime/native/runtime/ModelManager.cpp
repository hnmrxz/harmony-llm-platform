#include "ModelManager.h"

#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <utility>

#include "../model_package/Json.h"
#include "../model_package/ZipReader.h"

namespace hllm {

namespace {

bool DirExists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool MkDir(const std::string& p) {
    return ::mkdir(p.c_str(), 0755) == 0 || errno == EEXIST;
}

// Recursive remove without shelling out (safe even if a path looks odd).
bool RmRecursive(const std::string& p) {
    struct stat st;
    if (::lstat(p.c_str(), &st) != 0) {
        return true;  // doesn't exist
    }
    if (!S_ISDIR(st.st_mode)) {
        return ::unlink(p.c_str()) == 0;
    }
    DIR* dir = ::opendir(p.c_str());
    if (dir == nullptr) {
        return false;
    }
    struct dirent* entry;
    while ((entry = ::readdir(dir)) != nullptr) {
        if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        std::string child = p + "/" + entry->d_name;
        RmRecursive(child);
    }
    ::closedir(dir);
    return ::rmdir(p.c_str()) == 0;
}

std::string JoinPath(const std::string& base, const std::string& rel) {
    if (base.empty() || base.back() == '/') {
        return base + rel;
    }
    return base + "/" + rel;
}

}  // namespace

const char* ModelStateName(ModelState state) {
    switch (state) {
        case ModelState::Unset: return "UNSET";
        case ModelState::Imported: return "IMPORTED";
        case ModelState::Validating: return "VALIDATING";
        case ModelState::Installed: return "INSTALLED";
        case ModelState::Ready: return "READY";
        case ModelState::Running: return "RUNNING";
        case ModelState::Error: return "ERROR";
        case ModelState::Incompatible: return "INCOMPATIBLE";
    }
    return "UNSET";
}

std::string ModelManager::SanitizeId(const std::string& name) {
    std::string out;
    out.reserve(name.size());
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-') {
            out.push_back(std::tolower(static_cast<unsigned char>(c)));
        } else {
            out.push_back('_');
        }
    }
    while (!out.empty() && out.front() == '_') {
        out.erase(out.begin());
    }
    return out.empty() ? "model" : out;
}

ModelManager::ModelManager(const std::string& installRoot) : installRoot_(installRoot) {
    MkDir(installRoot_);
    MkDir(JoinPath(installRoot_, "installed"));
    LoadIndex();
}

bool ModelManager::LoadIndex() {
    models_.clear();
    std::string indexPath = JoinPath(installRoot_, "models.json");
    std::string bytes;
    if (!PackageReader::ReadFile(indexPath, bytes)) {
        return true;  // no index yet
    }
    auto tree = json::Parse(bytes);
    if (tree == nullptr || !tree->is_object()) {
        return false;
    }
    for (const auto& kv : tree->members()) {
        const json::Value* v = kv.second.get();
        if (v == nullptr || !v->is_object()) {
            continue;
        }
        InstalledModel model;
        model.id = kv.first;
        model.name = v->get("name") ? v->get("name")->as_string() : model.id;
        model.family = v->get("family") ? v->get("family")->as_string() : "";
        model.targetChip = v->get("targetChip") ? v->get("targetChip")->as_string() : "";
        if (const json::Value* st = v->get("state")) {
            std::string s = st->as_string();
            if (s == "INSTALLED") model.state = ModelState::Installed;
            else if (s == "READY") model.state = ModelState::Ready;
            else if (s == "RUNNING") model.state = ModelState::Running;
            else if (s == "ERROR") model.state = ModelState::Error;
            else if (s == "INCOMPATIBLE") model.state = ModelState::Incompatible;
            else if (s == "IMPORTED") model.state = ModelState::Imported;
            else if (s == "VALIDATING") model.state = ModelState::Validating;
        }
        models_.emplace(model.id, std::move(model));
    }
    return true;
}

bool ModelManager::SaveIndex() {
    json::Value root = json::Value::Object();
    for (const auto& kv : models_) {
        json::Value entry = json::Value::Object();
        entry.set("name", json::Value::String(kv.second.name));
        entry.set("family", json::Value::String(kv.second.family));
        entry.set("targetChip", json::Value::String(kv.second.targetChip));
        entry.set("state", json::Value::String(ModelStateName(kv.second.state)));
        root.set(kv.first, std::move(entry));
    }
    std::string indexPath = JoinPath(installRoot_, "models.json");
    FILE* f = fopen(indexPath.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    std::string data = json::Serialize(root);
    bool ok = fwrite(data.data(), 1, data.size(), f) == data.size();
    fclose(f);
    return ok;
}

bool ModelManager::ImportPackage(const std::string& hllmPath, std::string& modelId,
                                 std::vector<std::string>& errors) {
    // Unpack to a fresh staging directory.
    std::string base = JoinPath(installRoot_, "staging");
    MkDir(base);
    std::string staging = JoinPath(base, ".stage");
    RmRecursive(staging);
    if (!MkDir(staging)) {
        errors.push_back("cannot create staging directory");
        return false;
    }

    ZipReader zip;
    if (!zip.Open(hllmPath)) {
        errors.push_back("failed to open .hllm archive");
        return false;
    }
    if (!zip.ExtractAll(staging)) {
        errors.push_back("failed to unpack .hllm archive (unsupported entry or bad archive)");
        RmRecursive(staging);
        return false;
    }

    Manifest manifest;
    if (!PackageReader::ReadManifest(staging, manifest, errors)) {
        RmRecursive(staging);
        return false;
    }
    errors.clear();
    if (!PackageReader::VerifyIntegrity(staging, manifest, errors)) {
        RmRecursive(staging);
        return false;
    }

    modelId = SanitizeId(manifest.model.name);
    std::string finalDir = JoinPath(JoinPath(installRoot_, "installed"), modelId);

    // Atomic switch: move an existing install aside, then move the staging dir
    // into place; drop the stale copy afterwards.
    std::string backup = JoinPath(installRoot_, ".old_" + modelId);
    RmRecursive(backup);
    if (DirExists(finalDir)) {
        if (::rename(finalDir.c_str(), backup.c_str()) != 0) {
            errors.push_back("cannot replace existing install");
            RmRecursive(staging);
            return false;
        }
    }
    if (::rename(staging.c_str(), finalDir.c_str()) != 0) {
        errors.push_back("cannot install model directory");
        // Restore the previous install if it was moved aside.
        if (DirExists(backup)) {
            ::rename(backup.c_str(), finalDir.c_str());
        }
        RmRecursive(staging);
        return false;
    }
    RmRecursive(backup);

    InstalledModel model;
    model.id = modelId;
    model.name = manifest.model.name;
    model.family = manifest.model.family;
    model.targetChip = manifest.target.chip;
    model.state = ModelState::Installed;
    models_[modelId] = model;
    SaveIndex();
    return true;
}

bool ModelManager::SetState(const std::string& modelId, ModelState state) {
    auto it = models_.find(modelId);
    if (it == models_.end()) {
        return false;
    }
    it->second.state = state;
    return SaveIndex();
}

bool ModelManager::GetModel(const std::string& modelId, InstalledModel& out) const {
    auto it = models_.find(modelId);
    if (it == models_.end()) {
        return false;
    }
    out = it->second;
    return true;
}

bool ModelManager::GetEngineFiles(const std::string& modelId, EngineFiles& out) const {
    auto it = models_.find(modelId);
    if (it == models_.end()) {
        return false;
    }
    std::string installDir = JoinPath(JoinPath(installRoot_, "installed"), modelId);
    Manifest manifest;
    std::vector<std::string> errors;
    if (!PackageReader::ReadManifest(installDir, manifest, errors)) {
        return false;
    }
    return PackageReader::LocateEngineFiles(installDir, manifest, out, errors);
}

std::vector<InstalledModel> ModelManager::List() const {
    std::vector<InstalledModel> out;
    out.reserve(models_.size());
    for (const auto& kv : models_) {
        out.push_back(kv.second);
    }
    return out;
}

}  // namespace hllm
