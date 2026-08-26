"""Preflight planning for large-model conversion jobs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    parameters: int
    source_weight_bytes: int
    estimated_quantized_bytes: int
    estimated_package_bytes: int
    recommended_ram_bytes: int
    recommended_disk_bytes: int


def estimate_resources(parameters: int, bits: int = 4) -> ResourceEstimate:
    if parameters <= 0:
        raise ValueError("parameters must be positive")
    if bits <= 0 or bits > 16:
        raise ValueError("bits must be between 1 and 16")
    source = (parameters * 2)  # conservative FP16 baseline
    quantized = (parameters * bits + 7) // 8
    # Leave headroom for scales, tokenizer, manifests and conversion workspace.
    package = int(quantized * 1.15)
    ram = max(source * 2, 8 * 1024**3)
    disk = source + quantized * 2 + 20 * 1024**3
    return ResourceEstimate(parameters, source, quantized, package, ram, disk)


def can_start(*, available_ram: int, available_disk: int, estimate: ResourceEstimate) -> bool:
    return available_ram >= estimate.recommended_ram_bytes and available_disk >= estimate.recommended_disk_bytes
