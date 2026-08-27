#pragma once

/*
 * LLMEngineApi.h — the C-API surface of Huawei's CANN LLM Engine.
 *
 * This is the contract the native runtime layer is coded against. In a real
 * DevEco device build, this file is either replaced by, or must be consistent
 * with, the DDK headers shipped with the installed CANN Kit:
 *
 *   lm_engine_model_info.h
 *   cann_llm_engine_context.h
 *   cann_llm_engine_executor.h
 *   llm_engine_c_api_export.h
 *   llm_engine_context_base.h
 *   llm_engine_executor_base.h
 *   llm_engine_return_types.h
 *
 * The exact signatures above are tied to the installed SDK. The declared
 * (opaque) handles and the function names/signatures in this file mirror the
 * usage in the official sample:
 *   CANN_LLM/CANN_LLM_Engine_Demo/CANNLLMEngineDemoNext/.../llm_demo.cpp
 *
 * On an actual device, link against `libhiai_llm_engine.so` and add the DDK
 * `include` directory (and, for 64-bit, the `libs/arm64-v8a` / `lib64` folder)
 * to the CMake target. See README.md.
 */

#include <cstdint>

#if defined(_WIN32) || defined(__OHOS__)
#define HLLM_ENGINE_API __attribute__((visibility("default")))
#else
#define HLLM_ENGINE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Opaque engine handles -------------------------------------------- */
typedef struct LLMEngine_Context LLMEngine_Context;
typedef struct LLMEngine_Executor LLMEngine_Executor;
typedef struct LLMEngine_Prompt LLMEngine_Prompt;

/* ---- Status ----------------------------------------------------------- */
enum LLMEngine_Status {
    LLMEngine_SUCCESS = 0,
    LLMEngine_FAILED = 1,
    LLMEngine_ILLEGAL_PARAM = 2,
    LLMEngine_UNKNOWN = 3,
};

/*
 * Engine lifecycle context. `engine_type` in context.json is currently the
 * autograd/autoregressive mode; the context carries generation + sampling
 * options and the streaming callbacks.
 */

/* Build a context from the text of context.json (the caller owns the bytes). */
HLLM_ENGINE_API
LLMEngine_Context* LLMEngine_Context_CreateFromContextJson(const char* json);

/* Build an executor from the text of executor.json (the caller owns the bytes). */
HLLM_ENGINE_API
LLMEngine_Executor* LLMEngine_Executor_CreateFromJson(const char* json);

/*
 * Streaming / completion / failure callbacks. All are invoked on the engine's
 * worker thread; the returned text buffer is only valid inside the callback.
 */
typedef void (*LLMEngine_OnTokenDone)(const LLMEngine_Context* ctx);
typedef void (*LLMEngine_OnAllTokensDone)(const LLMEngine_Context* ctx);
typedef void (*LLMEngine_OnGenerateFailed)(const LLMEngine_Context* ctx);

HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(
    LLMEngine_Context* ctx, LLMEngine_OnTokenDone fn);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_SetOnAllTokensGenerateDoneFunc(
    LLMEngine_Context* ctx, LLMEngine_OnAllTokensDone fn);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_SetOnGenerateAsyncFailed(
    LLMEngine_Context* ctx, LLMEngine_OnGenerateFailed fn);

/* ---- Prompt ----------------------------------------------------------- */
HLLM_ENGINE_API
LLMEngine_Prompt* LLMEngine_Prompt_Create(void);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Prompt_SetText(LLMEngine_Prompt* prompt, const char* text);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Prompt_Destroy(LLMEngine_Prompt** prompt);

/* ---- Generation ------------------------------------------------------- */
/* Synchronous: blocks until the whole generation is complete. */
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Executor_LLM_Generate(LLMEngine_Executor* executor,
                                                 LLMEngine_Context* ctx,
                                                 LLMEngine_Prompt* prompt);
/* Asynchronous: streams via the callbacks registered on the context. */
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Executor_LLM_GenerateAsync(LLMEngine_Executor* executor,
                                                      LLMEngine_Context* ctx,
                                                      LLMEngine_Prompt* prompt);

/* ---- Reading the streamed / accumulated generation -------------------- */
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_GetOneGenerationLen(const LLMEngine_Context* ctx,
                                                       std::uint32_t* len);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_GetOneGeneration(const LLMEngine_Context* ctx,
                                                    char* buffer, std::uint32_t len);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_GetAllGenerationLen(const LLMEngine_Context* ctx,
                                                       std::uint32_t* len);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_GetAllGeneration(const LLMEngine_Context* ctx,
                                                    char* buffer, std::uint32_t len);

/* ---- Metrics (all optional; a missing metric returns LLMEngine_FAILED) - */
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_GetTotalTimeMs(const LLMEngine_Context* ctx, double* ms);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_GetPrefillTimeMs(const LLMEngine_Context* ctx, double* ms);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_GetDecodeTimeMs(const LLMEngine_Context* ctx, double* ms);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_GetInputTokenCount(const LLMEngine_Context* ctx,
                                                      std::uint64_t* count);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_GetOutputTokenCount(const LLMEngine_Context* ctx,
                                                       std::uint64_t* count);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_GetDecodeNum(const LLMEngine_Context* ctx,
                                                std::uint64_t* count);

/* ---- Teardown --------------------------------------------------------- */
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Executor_Deinit(LLMEngine_Executor* executor);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Executor_Destroy(LLMEngine_Executor** executor);
HLLM_ENGINE_API
LLMEngine_Status LLMEngine_Context_Destroy(LLMEngine_Context** ctx);

#ifdef __cplusplus
}  // extern "C"
#endif
