#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace hllm {

/*
 * DeviceProfile: read from the target HarmonyOS device (via HDC param / system
 * APIs), not guessed from a marketing name.
 */
struct DeviceProfile {
    std::string osVersion;
    std::string chip;
    std::string runtimeVersion;
    std::uint64_t availableMemoryBytes = 0;
};

/*
 * Metrics are produced by the CANN LLM Engine itself, so the runtime does not
 * need to replicate timing or token accounting.
 */
struct GenerationMetrics {
    double totalTimeMs = 0.0;
    double prefillTimeMs = 0.0;
    double decodeTimeMs = 0.0;
    std::uint64_t inputTokenCount = 0;
    std::uint64_t outputTokenCount = 0;
    std::uint64_t decodeNum = 0;
};

/*
 * Callbacks used to stream generated text to the UI layer. `onToken` fires for
 * every produced token (the engine invokes it from its worker thread). `onDone`
 * carries the full accumulated text. `onError` carries a human-readable reason.
 */
struct GenerationCallbacks {
    std::function<void(const std::string& token)> onToken;
    std::function<void(const std::string& fullText)> onDone;
    std::function<void(const std::string& error)> onError;
};

/*
 * InferenceBackend is the boundary the ArkTS layer talks to (through NAPI).
 *
 * The concrete backend (CannEngineBackend) delegates to the CANN LLM Engine
 * (`libhiai_llm_engine.so`), which already implements tokenize -> prefill ->
 * decode -> sampling -> detokenize plus KV-cache and memory management. The
 * runtime only has to supply the text of context.json and executor.json.
 */
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    /* Hardware/runtime compatibility gate, checked before load. */
    virtual bool IsCompatible(const DeviceProfile& device,
                              const std::string& targetChip,
                              const std::string& requiredRuntime) const = 0;

    /*
     * Build the engine context and executor from the raw JSON text. The engine
     * reads the .omc, weight, embedding and tokenizer files referenced by
     * executor.json at this point, so the model files must already be on disk.
     */
    virtual bool LoadModel(const std::string& contextJson,
                           const std::string& executorJson) = 0;

    /* Synchronous generation; returns accumulated text. */
    virtual bool Generate(const std::string& prompt, std::string& outText) = 0;

    /* Asynchronous generation with streaming callbacks. */
    virtual bool GenerateAsync(const std::string& prompt,
                               const GenerationCallbacks& callbacks) = 0;

    /* Request cancellation; the engine aborts at the next safe point. */
    virtual bool Cancel() = 0;

    /* Copy the accumulated generation (last async run). */
    virtual bool GetAllGeneration(std::string& outText) const = 0;

    /* Copy the metrics of the last run. */
    virtual bool GetMetrics(GenerationMetrics& out) const = 0;

    /* Release the engine context and executor. */
    virtual void Unload() = 0;
};

/*
 * Create the CANN LLM Engine-backed backend. The concrete class is defined in
 * native/backends/cann/CannBackend.cpp. The NAPI layer and any host app acquire
 * a backend through this factory rather than constructing it directly.
 */
std::unique_ptr<InferenceBackend> CreateCannEngineBackend();

}  // namespace hllm
