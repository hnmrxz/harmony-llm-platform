from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from hllm.models.detector import ModelMetadata


@dataclass(frozen=True, slots=True)
class QwenCapabilities:
    family: str
    text_generation: bool
    vision: bool
    video: bool


class QwenAdapter:
    family: ClassVar[str] = "qwen"

    def supports(self, metadata: ModelMetadata) -> bool:
        model_type = (metadata.model_type or "").lower()
        architecture = (metadata.architecture or "").lower()
        return model_type.startswith("qwen") or architecture.startswith("qwen")

    def capabilities(self, metadata: ModelMetadata) -> QwenCapabilities:
        self.validate(metadata)
        model_type = (metadata.model_type or "").lower()
        architecture = (metadata.architecture or "").lower()
        family = "qwen3_5" if "qwen3_5" in model_type or "qwen3_5" in architecture else "qwen3"
        return QwenCapabilities(
            family=family,
            text_generation=True,
            vision=metadata.is_multimodal or "conditionalgeneration" in architecture,
            video=bool(metadata.text_config and metadata.text_config.get("video_config")),
        )

    def validate(self, metadata: ModelMetadata) -> None:
        if not self.supports(metadata):
            raise ValueError("Qwen adapter received a non-Qwen architecture")

    def normalize(self, metadata: ModelMetadata, config: dict[str, Any] | None = None) -> QwenCapabilities:
        self.validate(metadata)
        return self.capabilities(metadata)
