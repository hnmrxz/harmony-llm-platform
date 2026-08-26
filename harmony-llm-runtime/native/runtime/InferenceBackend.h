#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hllm {

struct DeviceProfile {
    std::string osVersion;
    std::string chip;
    std::string runtimeVersion;
    std::uint64_t availableMemoryBytes = 0;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual bool IsCompatible(const DeviceProfile& device,
                              const std::string& targetChip,
                              const std::string& requiredRuntime) const = 0;
    virtual bool LoadOfflineModel(const std::vector<std::uint8_t>& modelBuffer) = 0;
    virtual bool Run(const std::vector<std::int64_t>& inputIds,
                     std::vector<float>& output) = 0;
    virtual void Unload() = 0;
};

} // namespace hllm
