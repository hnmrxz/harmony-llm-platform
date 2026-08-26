#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hllm {

struct Artifact {
    std::string type;
    std::string path;
    std::string sha256;
    std::uint64_t size = 0;
};

struct Target {
    std::string backend;
    std::string chip;
    std::string runtimeVersion;
};

struct RuntimeRequirements {
    std::int64_t contextLength = 0;
    std::int64_t minimumMemoryMb = 0;
};

struct Manifest {
    std::string schemaVersion;
    std::string name;
    std::string family;
    std::string architecture;
    Target target;
    RuntimeRequirements runtime;
    std::vector<Artifact> artifacts;
};

} // namespace hllm
