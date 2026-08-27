"""Pipeline public types."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Stage(StrEnum):
    DOWNLOAD = "download"
    INSPECT = "inspect"
    QUANTIZE = "quantize"
    EXPORT = "export"
    CANN_CONVERT = "cann_convert"
    VALIDATE = "validate"
    PACKAGE = "package"


@dataclass(slots=True)
class BuildOptions:
    source: str
    output_dir: Path
    target_chip: str
    quantization: str = "int4"
    context_length: int | None = None
    revision: str | None = None


@dataclass(slots=True)
class BuildState:
    options: BuildOptions
    stage: Stage | None = None
    model_dir: Path | None = None
    artifacts: list[Path] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)

    def enter(self, stage: Stage) -> None:
        self.stage = stage
        self.logs.append(f"stage={stage.value}")

    def record(self, message: str) -> None:
        self.logs.append(message)


__all__ = ["BuildOptions", "BuildState", "Stage"]
