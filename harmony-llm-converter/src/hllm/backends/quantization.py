"""Quantization backend contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    method: str = "cann_4bit"
    bits: int = 4
    group_size: int | None = None


class Quantizer(Protocol):
    def quantize(self, model_dir: Path, output_dir: Path, config: QuantizationConfig) -> None: ...


class ExternalQuantizer:
    """Adapter for a vendor/validated quantization executable.

    The converter keeps quantization orchestration separate from the vendor
    implementation so the latter can be upgraded without changing the HLLM API.
    """
    def __init__(self, command: tuple[str, ...]) -> None:
        self.command = command

    def quantize(self, model_dir: Path, output_dir: Path, config: QuantizationConfig) -> None:
        raise NotImplementedError(
            "Wire a validated CANN Kit quantization command here; do not infer vendor flags."
        )
