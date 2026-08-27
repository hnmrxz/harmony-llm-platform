/*
 * LLMEngineStub.cpp — no-op definitions of the CANN LLM Engine C-API.
 *
 * This translation unit is compiled ONLY when the real `libhiai_llm_engine.so`
 * is not present (e.g. a host/CI SDK build). It lets the native module link and
 * the app build so the toolchain/SDK path can be validated without the CANN
 * Kit. On a device build with the DDK headers + `.so`, the real engine is used
 * instead (see CMakeLists.txt).
 */
#include "runtime/LLMEngineApi.h"

extern "C" {

LLMEngine_Context* LLMEngine_Context_CreateFromContextJson(const char*) { return nullptr; }
LLMEngine_Executor* LLMEngine_Executor_CreateFromJson(const char*) { return nullptr; }
LLMEngine_Status LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(
    LLMEngine_Context*, LLMEngine_OnTokenDone) { return LLMEngine_FAILED; }
LLMEngine_Status LLMEngine_Context_SetOnAllTokensGenerateDoneFunc(
    LLMEngine_Context*, LLMEngine_OnAllTokensDone) { return LLMEngine_FAILED; }
LLMEngine_Status LLMEngine_Context_SetOnGenerateAsyncFailed(
    LLMEngine_Context*, LLMEngine_OnGenerateFailed) { return LLMEngine_FAILED; }

LLMEngine_Prompt* LLMEngine_Prompt_Create(void) { return nullptr; }
LLMEngine_Status LLMEngine_Prompt_SetText(LLMEngine_Prompt*, const char*) { return LLMEngine_FAILED; }
LLMEngine_Status LLMEngine_Prompt_Destroy(LLMEngine_Prompt** prompt) {
    // A stub never allocates, so there is nothing to free; guard against a
    // double-free on a (hypothetical) non-null pointer only.
    return LLMEngine_SUCCESS;
}

LLMEngine_Status LLMEngine_Executor_LLM_Generate(LLMEngine_Executor*, LLMEngine_Context*,
                                                 LLMEngine_Prompt*) { return LLMEngine_FAILED; }
LLMEngine_Status LLMEngine_Executor_LLM_GenerateAsync(LLMEngine_Executor*, LLMEngine_Context*,
                                                      LLMEngine_Prompt*) { return LLMEngine_FAILED; }

LLMEngine_Status LLMEngine_Context_GetOneGenerationLen(const LLMEngine_Context*, std::uint32_t*) {
    return LLMEngine_FAILED;
}
LLMEngine_Status LLMEngine_Context_GetOneGeneration(const LLMEngine_Context*, char*, std::uint32_t) {
    return LLMEngine_FAILED;
}
LLMEngine_Status LLMEngine_Context_GetAllGenerationLen(const LLMEngine_Context*, std::uint32_t*) {
    return LLMEngine_FAILED;
}
LLMEngine_Status LLMEngine_Context_GetAllGeneration(const LLMEngine_Context*, char*, std::uint32_t) {
    return LLMEngine_FAILED;
}

LLMEngine_Status LLMEngine_Context_GetTotalTimeMs(const LLMEngine_Context*, double*) {
    return LLMEngine_FAILED;
}
LLMEngine_Status LLMEngine_Context_GetPrefillTimeMs(const LLMEngine_Context*, double*) {
    return LLMEngine_FAILED;
}
LLMEngine_Status LLMEngine_Context_GetDecodeTimeMs(const LLMEngine_Context*, double*) {
    return LLMEngine_FAILED;
}
LLMEngine_Status LLMEngine_Context_GetInputTokenCount(const LLMEngine_Context*, std::uint64_t*) {
    return LLMEngine_FAILED;
}
LLMEngine_Status LLMEngine_Context_GetOutputTokenCount(const LLMEngine_Context*, std::uint64_t*) {
    return LLMEngine_FAILED;
}
LLMEngine_Status LLMEngine_Context_GetDecodeNum(const LLMEngine_Context*, std::uint64_t*) {
    return LLMEngine_FAILED;
}

LLMEngine_Status LLMEngine_Executor_Deinit(LLMEngine_Executor*) { return LLMEngine_SUCCESS; }
LLMEngine_Status LLMEngine_Executor_Destroy(LLMEngine_Executor** executor) {
    if (executor != nullptr) {
        *executor = nullptr;
    }
    return LLMEngine_SUCCESS;
}
LLMEngine_Status LLMEngine_Context_Destroy(LLMEngine_Context** ctx) {
    if (ctx != nullptr) {
        *ctx = nullptr;
    }
    return LLMEngine_SUCCESS;
}

}  // extern "C"
