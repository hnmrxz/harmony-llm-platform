"""Model metadata detection without loading heavyweight model weights."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelMetadata:
    name: str
    model_type: str | None
    architecture: str | None
    dtype: str | None
    parameter_count: int | None
    context_length: int | None
    has_tokenizer: bool
    has_chat_template: bool
    is_multimodal: bool = False
    text_config: dict[str, Any] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def inspect_model(path: str | Path) -> ModelMetadata:
    """Inspect a local Hugging Face-style model directory without loading weights."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model directory not found: {root}")

    config_path = root / "config.json"
    if not config_path.exists():
        raise ValueError(f"missing config.json: {root}")

    config = _read_json(config_path)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = None

    architectures = config.get("architectures") or []
    architecture = architectures[0] if architectures else None
    effective = text_config or config
    dtype = effective.get("torch_dtype", effective.get("dtype"))
    context_length = (
        effective.get("max_position_embeddings")
        or effective.get("max_seq_len")
        or effective.get("max_sequence_length")
        or config.get("max_position_embeddings")
    )

    parameter_count = config.get("num_parameters") or effective.get("num_parameters")
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    has_tokenizer = any((root / name).exists() for name in tokenizer_files)

    tokenizer_config = root / "tokenizer_config.json"
    has_chat_template = False
    if tokenizer_config.exists():
        tokenizer = _read_json(tokenizer_config)
        has_chat_template = bool(tokenizer.get("chat_template"))

    return ModelMetadata(
        name=root.name,
        model_type=config.get("model_type"),
        architecture=architecture,
        dtype=str(dtype) if dtype is not None else None,
        parameter_count=int(parameter_count) if parameter_count is not None else None,
        context_length=int(context_length) if context_length is not None else None,
        has_tokenizer=has_tokenizer,
        has_chat_template=has_chat_template,
        is_multimodal=bool(config.get("image_token_id") or config.get("video_token_id")),
        text_config=text_config,
    )
