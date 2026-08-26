"""Registry for model-family adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hllm.models.detector import ModelMetadata


class ModelAdapter(Protocol):
    family: str

    def supports(self, metadata: ModelMetadata) -> bool: ...


@dataclass(frozen=True, slots=True)
class AdapterMatch:
    family: str
    adapter: ModelAdapter


class ModelRegistry:
    def __init__(self) -> None:
        self._adapters: list[ModelAdapter] = []

    def register(self, adapter: ModelAdapter) -> None:
        if any(item.family == adapter.family for item in self._adapters):
            raise ValueError(f"adapter already registered: {adapter.family}")
        self._adapters.append(adapter)

    def resolve(self, metadata: ModelMetadata) -> AdapterMatch:
        for adapter in self._adapters:
            if adapter.supports(metadata):
                return AdapterMatch(adapter.family, adapter)
        raise ValueError(
            f"unsupported model architecture: {metadata.architecture or metadata.model_type or 'unknown'}"
        )

    def families(self) -> tuple[str, ...]:
        return tuple(adapter.family for adapter in self._adapters)
