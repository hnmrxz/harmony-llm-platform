import json
from pathlib import Path

import pytest

from hllm.models.layout import extract_layout, load_layout


def test_extract_layout_from_root_config() -> None:
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 3584,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 152064,
        "max_position_embeddings": 32768,
        "bos_token_id": 151643,
        "eos_token_id": 151643,
    }
    layout = extract_layout(config)
    assert layout.hidden_size == 3584
    assert layout.num_hidden_layers == 28
    assert layout.num_attention_heads == 28
    assert layout.num_attention_kv_heads == 4
    assert layout.num_attention_head_dims == 128
    assert layout.vocab_size == 152064
    assert layout.max_position_embeddings == 32768
    assert layout.bos_token_id == 151643


def test_extract_layout_uses_text_config_for_multimodal() -> None:
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 4096,
            "num_hidden_layers": 28,
            "num_attention_heads": 36,
            "num_key_value_heads": 4,
            "vocab_size": 152064,
            "max_position_embeddings": 131072,
        },
        "vision_config": {"image_size": 896},
    }
    layout = extract_layout(config)
    assert layout.hidden_size == 4096
    # head_dim falls back to hidden_size // num_attention_heads
    assert layout.num_attention_head_dims == 4096 // 36
    assert layout.max_position_embeddings == 131072


def test_layout_requires_geometry() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        extract_layout({"model_type": "qwen3"})


def test_load_layout_reads_config_json(tmp_path: Path) -> None:
    config = {
        "hidden_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "vocab_size": 100000,
        "max_position_embeddings": 8192,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    layout = load_layout(tmp_path)
    assert layout.hidden_size == 2048
    assert layout.num_hidden_layers == 12


def test_load_layout_missing_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_layout(tmp_path)
