#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hllm {

/*
 * Typed representations of the CANN LLM Engine's `context.json` and
 * `executor.json`. These are the configuration files the engine consumes
 * directly; they are shipped inside the `.hllm` package. The runtime reads them
 * for compatibility checks and forwards their raw JSON text to the engine.
 *
 * Field set follows the official template in the CANN LLM solution guide
 * (section 6.2). Only the fields the runtime needs for compatibility/benchmark
 * decisions are modelled here; the raw JSON text is forwarded verbatim.
 */

/* ---- context.json ----------------------------------------------------- */
struct SamplerConfig {
    bool do_sample = true;
    std::int64_t seed = 0;
    std::int64_t top_k = 0;
    double top_p = 0.0;
    double temperature = 0.0;
    double repetition_penalty = 1.0;
};

struct GenerateOptions {
    std::int64_t callback_freq = 0;
    std::int64_t max_gen_tokens = 0;
    std::vector<std::string> stop_sequence;
    std::int64_t init_token_len = 0;
};

struct ContextConfig {
    std::int64_t version = 1;
    std::string engine_type = "autoregressive";
    GenerateOptions generate_options;
    SamplerConfig sampler;
};

/* ---- executor.json ---------------------------------------------------- */
struct LlmConfig {
    std::int64_t bos_token_id = -1;
    std::int64_t eos_token_id = -1;
    std::int64_t kv_cache_max_len = 0;
    std::int64_t sliding_window_len = 0;
    std::int64_t max_position_embeddings = 0;
    std::int64_t num_attention_kv_heads = 0;
    std::int64_t num_attention_head_dims = 0;
    std::int64_t num_hidden_layers = 0;
    std::int64_t prefill_len = 0;
    std::int64_t decode_len = 1;
    std::int64_t vocab_size = 0;
    std::int64_t vocab_real_size = 0;
    bool use_output_pos = false;
    std::int64_t max_io_tokens = 0;
    std::int64_t hidden_size = 0;
    std::string embedding_weights;
    std::string embedding_dequant_scale;
    std::string embedding_input_type = "int8";
};

struct TokenizerConfig {
    std::string type;  // spm | hmb | xlmroberta | bpe | qwen | qwen2 | glm
    std::string path;
};

struct AutoregressiveConfig {
    std::string model_path;   // <model>.omc
    std::string weight_path;  // directory holding SubGraph_0.weight
};

struct ExecutorConfig {
    std::int64_t version = 1;
    std::string engine_type = "autoregressive";
    LlmConfig llm_config;
    TokenizerConfig tokenizer;
    AutoregressiveConfig autoregressive;

    /* The raw JSON bytes, forwarded to LLMEngine_Executor_CreateFromJson. */
    std::string rawJson;
};

}  // namespace hllm
