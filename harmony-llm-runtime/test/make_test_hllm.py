#!/usr/bin/env python3
"""Create a synthetic .hllm package for the host import test.

The package mirrors the CANN LLM Engine file set the Runtime consumes:

    manifest.json
    model/qwen.omc
    model/SubGraph_0.weight
    model/qwen_64_2048.embedding_weights
    model/qwen_64_2048.embedding_dequant_scale
    config/context.json
    config/executor.json
    model/tokenizer.json

The archive is deflated so it exercises the Runtime's bundled DEFLATE inflater.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    package = root / "test_model.hllm"
    if package.exists():
        package.unlink()

    artifacts: dict[str, bytes] = {
        "model/qwen.omc": b"OMC-MODEL-BYTES",
        "model/SubGraph_0.weight": b"WEIGHT-BYTES",
        "model/qwen_64_2048.embedding_weights": b"EMBEDDING-WEIGHTS",
        "model/qwen_64_2048.embedding_dequant_scale": b"EMBEDDING-SCALE",
        "config/context.json": json.dumps({
            "version": 1,
            "engine_type": "autoregressive",
            "generate_options": {"callback_freq": 2, "max_gen_tokens": 600, "stop_sequence": ["<|im_end|>"], "init_token_len": 0},
            "sampler": {"do_sample": True, "seed": 99, "top-k": 16, "top-p": 0.95, "temperature": 0.6, "repetition_penalty": 1.2},
        }, ensure_ascii=False).encode("utf-8"),
        "config/executor.json": json.dumps({
            "version": 1,
            "engine_type": "autoregressive",
            "llm_config": {
                "bos_token_id": 151643, "eos_token_id": 151643, "kv_cache_max_len": 2048,
                "sliding_window_len": 0, "max_position_embeddings": 32768,
                "num_attention_kv_heads": 4, "num_attention_head_dims": 128,
                "num_hidden_layers": 4, "prefill_len": 64, "decode_len": 1,
                "vocab_size": 152064, "vocab_real_size": 152064, "use_output_pos": False,
                "max_io_tokens": 4096, "hidden_size": 3584,
                "embedding_weights": "qwen_64_2048.embedding_weights",
                "embedding_dequant_scale": "qwen_64_2048.embedding_dequant_scale",
                "embedding_input_type": "int8",
            },
            "tokenizer": {"type": "qwen", "path": "tokenizer.json"},
            "autoregressive": {"model_path": "qwen.omc", "weight_path": "."},
        }, ensure_ascii=False).encode("utf-8"),
        "model/tokenizer.json": b'{"tokenizer": "qwen"}',
    }

    manifest = {
        "schema_version": "1.0",
        "model": {"name": "Qwen3-8B", "family": "qwen3", "architecture": "Qwen3ForCausalLM",
                  "source_type": "huggingface", "source_id": "Qwen/Qwen3-8B", "revision": "test"},
        "quantization": {"type": "cann_4bit", "bits": 4, "group_size": 128},
        "target": {"backend": "cann_llm_engine", "chip": "kirinx90", "runtime_version": "9.1.0"},
        "runtime": {"context_length": 32768, "minimum_memory_mb": 4096},
        "llm_engine": {"engine_type": "autoregressive", "kv_cache_max_len": 2048, "prefill_len": 64,
                        "max_io_tokens": 4096, "vocab_size": 152064, "hidden_size": 3584,
                        "num_hidden_layers": 4, "num_attention_kv_heads": 4,
                        "num_attention_head_dims": 128, "max_position_embeddings": 32768,
                        "embedding_input_type": "int8"},
        "build": {"converter_version": "0.1.0", "python_version": "3.14"},
        "artifacts": [
            {"type": "model", "path": p, "sha256": sha256(d), "size": len(d)}
            for p, d in artifacts.items()
        ],
    }
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for path, data in artifacts.items():
            archive.writestr(path, data)

    print(f"created {package} ({package.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
