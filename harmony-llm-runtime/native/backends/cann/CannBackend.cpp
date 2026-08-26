#include "../../runtime/InferenceBackend.h"

namespace hllm {

/*
 * The HarmonyOS-specific implementation is intentionally isolated here.
 *
 * Current Huawei documentation describes the native offline-model path using
 * NNRt/CANN APIs: obtain device IDs, construct an offline-model compilation
 * instance from a model buffer, then run the executor. The exact SDK headers
 * and signatures are tied to the installed HarmonyOS/DevEco SDK and therefore
 * must be bound in the device build rather than guessed in portable code.
 *
 * See docs/hllm-package-spec.md for the package contract.
 */

class CannBackend final : public InferenceBackend {
public:
    bool IsCompatible(const DeviceProfile& device,
                      const std::string& targetChip,
                      const std::string& requiredRuntime) const override
    {
        return !device.chip.empty() && device.chip == targetChip &&
               (requiredRuntime.empty() || device.runtimeVersion == requiredRuntime);
    }

    bool LoadOfflineModel(const std::vector<std::uint8_t>& /*modelBuffer*/) override
    {
        // Bind OH_NNCompilation_ConstructWithOfflineModelBuffer and related
        // lifecycle APIs in the HarmonyOS SDK-specific implementation.
        return false;
    }

    bool Run(const std::vector<std::int64_t>& /*inputIds*/,
             std::vector<float>& /*output*/) override
    {
        // Bind OH_NNExecutor_RunSync after the model compilation has completed.
        return false;
    }

    void Unload() override {}
};

} // namespace hllm
