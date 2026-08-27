#include "../../runtime/InferenceBackend.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "../../runtime/LLMEngineApi.h"

namespace hllm {

class CannEngineBackend;  // defined below; callbacks only need a pointer.

namespace {

/*
 * The CANN LLM Engine callbacks carry only `const LLMEngine_Context*`. Because
 * a CannEngineBackend owns exactly one context, we keep a small registry that
 * maps the context pointer back to the owning backend instance. The callbacks
 * are C function pointers, so they cannot capture `this`; the registry lets a
 * callback reach the right backend's GenerationCallbacks.
 */
std::mutex g_registryMutex;
std::unordered_map<const LLMEngine_Context*, CannEngineBackend*> g_byContext;

void RegisterContext(const LLMEngine_Context* ctx, CannEngineBackend* backend) {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    g_byContext[ctx] = backend;
}

void UnregisterContext(const LLMEngine_Context* ctx) {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    g_byContext.erase(ctx);
}

CannEngineBackend* FindBackend(const LLMEngine_Context* ctx) {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    auto it = g_byContext.find(ctx);
    return it == g_byContext.end() ? nullptr : it->second;
}

}  // namespace

class CannEngineBackend final : public InferenceBackend {
public:
    bool IsCompatible(const DeviceProfile& device,
                      const std::string& targetChip,
                      const std::string& requiredRuntime) const override
    {
        if (device.chip.empty()) {
            return false;
        }
        if (targetChip.empty() || device.chip != targetChip) {
            // targetChip is usually the `target.chip` from the manifest's
            // `llm_engine`-oriented target identity.
            return false;
        }
        return requiredRuntime.empty() || device.runtimeVersion == requiredRuntime;
    }

    bool LoadModel(const std::string& contextJson,
                   const std::string& executorJson) override
    {
        Unload();

        contextJson_ = contextJson;
        executorJson_ = executorJson;

        LLMEngine_Context* ctx = LLMEngine_Context_CreateFromContextJson(contextJson.c_str());
        if (ctx == nullptr) {
            return false;
        }
        LLMEngine_Executor* executor = LLMEngine_Executor_CreateFromJson(executorJson.c_str());
        if (executor == nullptr) {
            LLMEngine_Context_Destroy(&ctx);
            return false;
        }

        context_ = ctx;
        executor_ = executor;
        RegisterContext(context_, this);
        return true;
    }

    bool Generate(const std::string& prompt, std::string& outText) override
    {
        if (!Ready()) {
            return false;
        }

        LLMEngine_Prompt* p = LLMEngine_Prompt_Create();
        if (p == nullptr) {
            return false;
        }
        if (LLMEngine_Prompt_SetText(p, prompt.c_str()) != LLMEngine_SUCCESS) {
            LLMEngine_Prompt_Destroy(&p);
            return false;
        }
        const auto status = LLMEngine_Executor_LLM_Generate(executor_, context_, p);
        LLMEngine_Prompt_Destroy(&p);
        if (status != LLMEngine_SUCCESS) {
            return false;
        }
        return GetAllGeneration(outText);
    }

    bool GenerateAsync(const std::string& prompt,
                       const GenerationCallbacks& callbacks) override
    {
        if (!Ready()) {
            return false;
        }

        callbacks_ = callbacks;
        LLMEngine_Status st;

        // The engine invokes these from its worker thread; forward through the
        // registry to `this` and copy the produced text out before returning.
        st = LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(context_, &OnSomeToken);
        if (st != LLMEngine_SUCCESS) {
            return false;
        }
        st = LLMEngine_Context_SetOnAllTokensGenerateDoneFunc(context_, &OnAllDone);
        if (st != LLMEngine_SUCCESS) {
            return false;
        }
        st = LLMEngine_Context_SetOnGenerateAsyncFailed(context_, &OnFailed);
        if (st != LLMEngine_SUCCESS) {
            return false;
        }

        LLMEngine_Prompt* p = LLMEngine_Prompt_Create();
        if (p == nullptr) {
            return false;
        }
        if (LLMEngine_Prompt_SetText(p, prompt.c_str()) != LLMEngine_SUCCESS) {
            LLMEngine_Prompt_Destroy(&p);
            return false;
        }
        st = LLMEngine_Executor_LLM_GenerateAsync(executor_, context_, p);
        LLMEngine_Prompt_Destroy(&p);
        return st == LLMEngine_SUCCESS;
    }

    bool Cancel() override
    {
        // The engine aborts at the next safe point. For this engine surface
        // there is no documented synchronous cancel; clearing the callbacks
        // makes the UI stop consuming tokens, and a subsequent Unload() frees
        // the engine. Kept as a distinct step so the UI can signal intent.
        callbacks_ = GenerationCallbacks{};
        return true;
    }

    bool GetAllGeneration(std::string& outText) const override
    {
        if (context_ == nullptr) {
            return false;
        }
        std::uint32_t len = 0;
        if (LLMEngine_Context_GetAllGenerationLen(context_, &len) != LLMEngine_SUCCESS) {
            return false;
        }
        std::string text(len, '\0');
        if (len > 0 &&
            LLMEngine_Context_GetAllGeneration(context_, text.data(), len) != LLMEngine_SUCCESS) {
            return false;
        }
        outText = std::move(text);
        return true;
    }

    bool GetMetrics(GenerationMetrics& out) const override
    {
        if (context_ == nullptr) {
            return false;
        }
        GenerationMetrics m;
        LLMEngine_Context_GetTotalTimeMs(context_, &m.totalTimeMs);
        LLMEngine_Context_GetPrefillTimeMs(context_, &m.prefillTimeMs);
        LLMEngine_Context_GetDecodeTimeMs(context_, &m.decodeTimeMs);
        LLMEngine_Context_GetInputTokenCount(context_, &m.inputTokenCount);
        LLMEngine_Context_GetOutputTokenCount(context_, &m.outputTokenCount);
        LLMEngine_Context_GetDecodeNum(context_, &m.decodeNum);
        out = m;
        return true;
    }

    void Unload() override
    {
        if (context_ != nullptr) {
            UnregisterContext(context_);
            LLMEngine_Context_Destroy(&context_);
        }
        if (executor_ != nullptr) {
            LLMEngine_Executor_Deinit(executor_);
            LLMEngine_Executor_Destroy(&executor_);
        }
        callbacks_ = GenerationCallbacks{};
    }

private:
    bool Ready() const { return context_ != nullptr && executor_ != nullptr; }

    static void OnSomeToken(const LLMEngine_Context* ctx) {
        auto* backend = FindBackend(ctx);
        if (backend == nullptr || !backend->callbacks_.onToken) {
            return;
        }
        std::uint32_t len = 0;
        if (LLMEngine_Context_GetOneGenerationLen(ctx, &len) != LLMEngine_SUCCESS || len == 0) {
            return;
        }
        std::string token(len, '\0');
        if (LLMEngine_Context_GetOneGeneration(ctx, token.data(), len) != LLMEngine_SUCCESS) {
            return;
        }
        backend->callbacks_.onToken(std::move(token));
    }

    static void OnAllDone(const LLMEngine_Context* ctx) {
        auto* backend = FindBackend(ctx);
        if (backend == nullptr) {
            return;
        }
        std::string full;
        if (backend->GetAllGeneration(full) && backend->callbacks_.onDone) {
            backend->callbacks_.onDone(full);
        }
    }

    static void OnFailed(const LLMEngine_Context* ctx) {
        auto* backend = FindBackend(ctx);
        if (backend == nullptr) {
            return;
        }
        if (backend->callbacks_.onError) {
            backend->callbacks_.onError("LLMEngine generation failed");
        }
    }

    LLMEngine_Context* context_ = nullptr;
    LLMEngine_Executor* executor_ = nullptr;
    std::string contextJson_;
    std::string executorJson_;
    GenerationCallbacks callbacks_;
};

std::unique_ptr<InferenceBackend> CreateCannEngineBackend() {
    return std::make_unique<CannEngineBackend>();
}

}  // namespace hllm
