#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hllm {

/*
 * Mirror of the HLLM manifest produced by the Ubuntu converter
 * (src/hllm/schema/manifest.py). The runtime only needs the subset needed for
 * compatibility checks and import validation; unknown fields are ignored.
 *
 * `llmEngine` fields carry the CANN LLM Engine's runtime hint (engine_type,
 * KV cache length, prefill length, max I/O tokens, vocab size...), which the
 * runtime compares against DeviceProfile to decide COMPATIBLE / WITH_LIMITS /
 * INCOMPATIBLE before loading.
 */

struct Artifact {
    std::string type;         // "model" | "resource"
    std::string path;
    std::string sha256;
    std::uint64_t size = 0;
};

struct ModelInfo {
    std::string name;
    std::string family;
    std::string architecture;
    std::string source_type;  // "huggingface" | "local"
    std::string source_id;
    std::string revision;
};

struct QuantizationInfo {
    std::string type;        // "cann_4bit" | "fp8" | ...
    std::int64_t bits = 0;
    std::int64_t group_size = 0;
};

struct Target {
    std::string backend;     // "cann" | "cann_llm_engine"
    std::string chip;
    std::string runtime_version;
};

struct RuntimeRequirements {
    std::int64_t contextLength = 0;
    std::int64_t minimumMemoryMb = 0;
};

/* ---- CANN LLM Engine runtime hint ------------------------------------- */
struct LlmEngineRuntime {
    std::string engine_type = "autoregressive";
    std::int64_t kv_cache_max_len = 0;
    std::int64_t prefill_len = 0;
    std::int64_t max_io_tokens = 0;
    std::int64_t vocab_size = 0;
    std::int64_t hidden_size = 0;
    std::int64_t num_hidden_layers = 0;
    std::int64_t num_attention_kv_heads = 0;
    std::int64_t num_attention_head_dims = 0;
    std::int64_t max_position_embeddings = 0;
    std::string embedding_input_type;
};

struct BuildInfo {
    std::string converter_version;
    std::string git_commit;
    std::string python_version;
};

struct Manifest {
    std::string schemaVersion;
    ModelInfo model;
    QuantizationInfo quantization;
    Target target;
    RuntimeRequirements runtime;
    LlmEngineRuntime llmEngine;
    BuildInfo build;
    std::vector<Artifact> artifacts;
};

}  // namespace hllm
