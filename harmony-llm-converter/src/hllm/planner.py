from __future__ import annotations

from dataclasses import dataclass

from hllm.models.weights import WeightInventory


_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    parameters: int
    source_weight_bytes: int
    estimated_quantized_bytes: int
    estimated_package_bytes: int
    recommended_ram_bytes: int
    recommended_disk_bytes: int


def _ram_requirement(source_bytes: int, *, multiplier: float = 1.5, floor_bytes: int = 4 * _GIB) -> int:
    """Estimate conversion RAM without imposing an arbitrary 8 GiB floor.

    The result is a planning guardrail, not a benchmark-derived requirement.
    Individual quantizers/exporters may override this at the profile level.
    """
    return max(floor_bytes, int(source_bytes * multiplier))


def _disk_requirement(source_bytes: int, quantized_bytes: int, *, workspace_bytes: int = 20 * _GIB) -> int:
    return source_bytes + quantized_bytes * 2 + workspace_bytes


def estimate_resources(parameters: int, bits: int = 4) -> ResourceEstimate:
    if parameters <= 0:
        raise ValueError("parameters must be positive")
    if not 1 <= bits <= 16:
        raise ValueError("bits must be between 1 and 16")
    source = parameters * 2
    quantized = (parameters * bits + 7) // 8
    package = int(quantized * 1.15)
    ram = _ram_requirement(source)
    disk = _disk_requirement(source, quantized)
    return ResourceEstimate(parameters, source, quantized, package, ram, disk)


def estimate_from_weights(inventory: WeightInventory, bits: int = 4) -> ResourceEstimate | None:
    if inventory.total_bytes <= 0:
        return None
    if not 1 <= bits <= 16:
        raise ValueError("bits must be between 1 and 16")
    source = inventory.total_bytes
    quantized = max(1, (source * bits + 15) // 16)
    package = int(quantized * 1.20)
    ram = _ram_requirement(source)
    disk = _disk_requirement(source, quantized)
    return ResourceEstimate(
        inventory.parameter_count or 0,
        source,
        quantized,
        package,
        ram,
        disk,
    )


def can_start(*, available_ram: int, available_disk: int, estimate: ResourceEstimate) -> bool:
    if available_ram <= 0 or available_disk <= 0:
        return False
    return (
        available_ram >= estimate.recommended_ram_bytes
        and available_disk >= estimate.recommended_disk_bytes
    )
