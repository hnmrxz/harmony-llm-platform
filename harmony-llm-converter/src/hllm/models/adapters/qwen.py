"""Qwen-family model adapter.

The adapter deliberately only identifies the family for now. Export and CANN
specific behavior will be added after the first hardware-validated pipeline.
"""

from __future__ import annotations

from hllm.models.detector import ModelMetadata


class QwenAdapter:
    family = "qwen"

    def supports(self, metadata: ModelMetadata) -> bool:
        model_type = (metadata.model_type or "").lower()
        architecture = (metadata.architecture or "").lower()
        return model_type.startswith("qwen") or architecture.startswith("qwen")
