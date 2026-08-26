"""Typed representation of the HLLM package manifest.

The manifest is the stable contract between the Ubuntu converter and the
HarmonyOS runtime. Keep it intentionally independent from converter internals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Artifact:
    type: str
    path: str
    sha256: str | None = None
    size: int | None = None


@dataclass(slots=True)
class ModelInfo:
    name: str
    family: str
    architecture: str
    source_type: str = "local"
    source_id: str | None = None
    revision: str | None = None


@dataclass(slots=True)
class QuantizationInfo:
    type: str
    bits: int | None = None
    group_size: int | None = None


@dataclass(slots=True)
class TargetInfo:
    backend: str
    chip: str
    runtime_version: str | None = None


@dataclass(slots=True)
class RuntimeInfo:
    context_length: int | None = None
    minimum_memory_mb: int | None = None


@dataclass(slots=True)
class BuildInfo:
    converter_version: str
    git_commit: str | None = None
    python_version: str | None = None
    pytorch_version: str | None = None
    transformers_version: str | None = None
    onnx_version: str | None = None
    cann_version: str | None = None


@dataclass(slots=True)
class Manifest:
    schema_version: str
    model: ModelInfo
    quantization: QuantizationInfo
    target: TargetInfo
    runtime: RuntimeInfo
    build: BuildInfo
    artifacts: list[Artifact] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the public JSON representation."""
        return asdict(self)
