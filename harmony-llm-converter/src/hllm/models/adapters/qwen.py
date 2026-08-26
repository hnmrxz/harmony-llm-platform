"""Qwen family adapter, including Qwen3.5/Qwen3.8 metadata layouts."""

from __future__ import annotations

from hllm.models.detector import ModelMetadata


class QwenAdapter:
    family = "qwen"

    def supports(self, metadata: ModelMetadata) -> bool:
        model_type = (metadata.model_type or "").lower()
        architecture = (metadata.architecture or "").lower()
        return model_type.startswith("qwen") or architecture.startswith("qwen")

    @staticmethod
    def is_qwen3_8(metadata: ModelMetadata) -> bool:
        return (metadata.model_type or "").lower() in {"qwen3_5", "qwen3_8"}
