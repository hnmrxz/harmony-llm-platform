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
#include "../model_package/Sha256.h"
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

bool FileIsRegular(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

void CollectFiles(const std::string& dir, const std::string& relPrefix,
                  std::vector<std::string>& out) {
    DIR* d = ::opendir(dir.c_str());
    if (d == nullptr) {
        return;
    }
    struct dirent* e;
    while ((e = ::readdir(d)) != nullptr) {
        if (std::strcmp(e->d_name, ".") == 0 || std::strcmp(e->d_name, "..") == 0) {
            continue;
        }
        std::string child = dir + "/" + e->d_name;
        std::string rel = relPrefix.empty() ? e->d_name : relPrefix + "/" + e->d_name;
        struct stat st;
        if (::lstat(child.c_str(), &st) != 0) {
            continue;
        }
        if (S_ISDIR(st.st_mode)) {
            CollectFiles(child, rel, out);
        } else if (S_ISREG(st.st_mode)) {
            out.push_back(rel);
        }
    }
    ::closedir(d);
}

bool CopyDirRecursive(const std::string& src, const std::string& dst) {
    if (!MkDir(dst)) {
        return false;
    }
    DIR* d = ::opendir(src.c_str());
    if (d == nullptr) {
        return false;
    }
    struct dirent* e;
    while ((e = ::readdir(d)) != nullptr) {
        if (std::strcmp(e->d_name, ".") == 0 || std::strcmp(e->d_name, "..") == 0) {
            continue;
        }
        std::string s = src + "/" + e->d_name;
        std::string dt = dst + "/" + e->d_name;
        struct stat st;
        if (::lstat(s.c_str(), &st) != 0) {
            ::closedir(d);
            return false;
        }
        if (S_ISDIR(st.st_mode)) {
            if (!CopyDirRecursive(s, dt)) {
                ::closedir(d);
                return false;
            }
        } else if (S_ISREG(st.st_mode)) {
            std::string bytes;
            if (!PackageReader::ReadFile(s, bytes)) {
                ::closedir(d);
                return false;
            }
            FILE* f = fopen(dt.c_str(), "wb");
            if (f == nullptr) {
                ::closedir(d);
                return false;
            }
            bool ok = bytes.empty() || fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
            fclose(f);
            if (!ok) {
                ::closedir(d);
                return false;
            }
        }
    }
    ::closedir(d);
    return true;
}

bool WriteFileText(const std::string& p, const std::string& data) {
    FILE* f = fopen(p.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    bool ok = data.empty() || fwrite(data.data(), 1, data.size(), f) == data.size();
    fclose(f);
    return ok;
}

std::string SerializeManifest(const Manifest& m) {
    json::Value root = json::Value::Object();
    root.set("schema_version", json::Value::String("1.0"));
    json::Value model = json::Value::Object();
    model.set("name", json::Value::String(m.model.name));
    model.set("family", json::Value::String(m.model.family));
    model.set("architecture", json::Value::String(m.model.architecture));
    root.set("model", std::move(model));
    json::Value target = json::Value::Object();
    target.set("backend", json::Value::String(m.target.backend));
    target.set("chip", json::Value::String(m.target.chip));
    target.set("runtime_version", json::Value::String(m.target.runtime_version));
    root.set("target", std::move(target));
    json::Value arts = json::Value::Array();
    for (const auto& a : m.artifacts) {
        json::Value ja = json::Value::Object();
        ja.set("type", json::Value::String(a.type));
        ja.set("path", json::Value::String(a.path));
        ja.set("sha256", json::Value::String(a.sha256));
        ja.set("size", json::Value::Number(static_cast<double>(a.size)));
        arts.push(std::move(ja));
    }
    root.set("artifacts", std::move(arts));
    return json::Serialize(root);
}

bool BuildManifestFromFolder(const std::string& folder, Manifest& manifest,
                             std::vector<std::string>& errors) {
    std::vector<std::string> files;
    CollectFiles(folder, "", files);
    bool hasOmc = false;
    bool hasCtx = false;
    bool hasExec = false;
    bool hasTok = false;
    std::size_t slash = folder.find_last_of('/');
    std::string name = slash == std::string::npos ? folder : folder.substr(slash + 1);

    manifest = Manifest{};
    manifest.schemaVersion = "1.0";
    manifest.model.name = name;
    manifest.model.family = "prepacked";
    manifest.model.architecture = "prepacked";
    manifest.target.backend = "cann_llm_engine";

    for (const auto& rel : files) {
        std::string abs = JoinPath(folder, rel);
        std::string bytes;
        if (!PackageReader::ReadFile(abs, bytes)) {
            errors.push_back("unreadable: " + rel);
            return false;
        }
        Artifact a;
        a.type = "resource";
        a.path = rel;
        a.sha256 = Sha256Hex(bytes);
        a.size = bytes.size();
        bool isOmc = rel.size() > 4 && rel.substr(rel.size() - 4) == ".omc";
        if (isOmc) {
            a.type = "model";
            hasOmc = true;
        }
        if (rel == "context.json") {
            hasCtx = true;
        } else if (rel == "executor.json") {
            hasExec = true;
        } else if (rel == "tokenizer.json") {
            hasTok = true;
        }
        manifest.artifacts.push_back(std::move(a));
    }

    if (!hasOmc || !hasCtx || !hasExec || !hasTok) {
        errors.push_back(
            "folder missing required CANN engine files (.omc / context.json / "
            "executor.json / tokenizer.json)");
        return false;
    }
    return true;
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

bool ModelManager::ImportFolder(const std::string& folderPath, std::string& modelId,
                                std::vector<std::string>& errors) {
    if (!DirExists(folderPath)) {
        errors.push_back("folder does not exist");
        return false;
    }
    Manifest manifest;
    bool hasManifest = FileIsRegular(JoinPath(folderPath, "manifest.json"));
    if (hasManifest) {
        // An extracted .hllm (manifest.json + artifacts).
        if (!PackageReader::ReadManifest(folderPath, manifest, errors)) {
            return false;
        }
        errors.clear();
        if (!PackageReader::VerifyIntegrity(folderPath, manifest, errors)) {
            return false;
        }
    } else {
        // A folder of pre-converted CANN engine files: build a manifest.
        if (!BuildManifestFromFolder(folderPath, manifest, errors)) {
            return false;
        }
    }

    modelId = SanitizeId(manifest.model.name);
    std::string finalDir = JoinPath(JoinPath(installRoot_, "installed"), modelId);
    std::string backup = JoinPath(installRoot_, ".old_" + modelId);
    RmRecursive(backup);
    if (DirExists(finalDir)) {
        if (::rename(finalDir.c_str(), backup.c_str()) != 0) {
            errors.push_back("cannot replace existing install");
            return false;
        }
    }
    if (!CopyDirRecursive(folderPath, finalDir)) {
        errors.push_back("cannot install folder");
        if (DirExists(backup)) {
            ::rename(backup.c_str(), finalDir.c_str());
        }
        return false;
    }
    if (!hasManifest) {
        if (!WriteFileText(JoinPath(finalDir, "manifest.json"), SerializeManifest(manifest))) {
            errors.push_back("cannot write manifest to installed copy");
            return false;
        }
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
