"""Qwen model-family helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hllm.models.detector import ModelMetadata


@dataclass(frozen=True, slots=True)
class QwenCapabilities:
    family: str
    text_generation: bool
    vision: bool
    video: bool


def normalize_qwen_family(metadata: ModelMetadata, config: dict[str, Any]) -> QwenCapabilities:
    model_type = (metadata.model_type or "").lower()
    architecture = (metadata.architecture or "").lower()
    family = "qwen3_5" if "qwen3_5" in model_type or "qwen3_5" in architecture else "qwen3"
    vision = "conditionalgeneration" in architecture or bool(config.get("vision_config"))
    video = bool(config.get("video_token_id")) or bool(config.get("video_config"))
    return QwenCapabilities(family, True, vision, video)


def validate_qwen_target(metadata: ModelMetadata) -> None:
    architecture = (metadata.architecture or "").lower()
    if not architecture.startswith("qwen"):
        raise ValueError("Qwen adapter received a non-Qwen architecture")
