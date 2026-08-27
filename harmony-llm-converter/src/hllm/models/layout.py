"""Extract the LLM model geometry needed to drive the official CANN OMG/LLM
Engine configuration.

The official CANN LLM solution requires the per-model geometry to build the
OMG `--input_shape` / `--dynamic_dims` / `--input_type` / `--output_type` and
the device-side `executor.json`. This module reads those values from the model's
`config.json` (or its `text_config`, for multimodal models) without loading any
weights.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelLayout:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_attention_kv_heads: int
    num_attention_head_dims: int
    vocab_size: int
    max_position_embeddings: int
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    sliding_window_len: int = 0
    torch_dtype: str | None = None

    @property
    def kv_cache_max_len(self) -> int:
        """Default KV cache length; the official executor uses 2048/4096. We use
        max_position_embeddings as an upper bound, capped to a supported value."""
        return self.max_position_embeddings or 2048


def _int_config(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return config


def extract_layout(config: dict[str, Any]) -> ModelLayout:
    """Derive the model geometry from a parsed config.json."""
    text_cfg = _text_config(config)
    hidden_size = _int_config(text_cfg, "hidden_size")
    num_layers = _int_config(text_cfg, "num_hidden_layers")
    num_heads = _int_config(text_cfg, "num_attention_heads")
    if hidden_size is None or num_layers is None or num_heads is None:
        raise ValueError(
            "cannot determine model layout: config must define hidden_size, "
            "num_hidden_layers and num_attention_heads (in config.json or text_config)"
        )

    num_kv_heads = _int_config(text_cfg, "num_key_value_heads") or num_heads
    head_dim = _int_config(text_cfg, "head_dim") or (hidden_size // num_heads)
    vocab_size = _int_config(text_cfg, "vocab_size")
    max_pos = _int_config(
        text_cfg, "max_position_embeddings", "max_seq_len", "max_sequence_length"
    ) or _int_config(config, "max_position_embeddings")
    if vocab_size is None:
        raise ValueError("cannot determine model layout: config must define vocab_size")

    dtype = text_cfg.get("torch_dtype") or config.get("torch_dtype")
    bos = _int_config(text_cfg, "bos_token_id")
    eos = _int_config(text_cfg, "eos_token_id")
    sliding = _int_config(config, "sliding_window") or 0

    return ModelLayout(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_attention_kv_heads=num_kv_heads,
        num_attention_head_dims=head_dim,
        vocab_size=vocab_size,
        max_position_embeddings=max_pos or 2048,
        bos_token_id=bos,
        eos_token_id=eos,
        sliding_window_len=sliding,
        torch_dtype=str(dtype) if dtype is not None else None,
    )


def load_layout(model_dir: str | Path) -> ModelLayout:
    """Load and parse config.json from a local model directory."""
    root = Path(model_dir).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json: {root}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"expected JSON object: {config_path}")
    return extract_layout(config)
