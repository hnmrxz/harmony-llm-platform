from __future__ import annotations

from hllm.models.weights import WeightInventory
from hllm.planner import ResourceEstimate


def estimate_from_weights(inventory: WeightInventory, bits: int = 4) -> ResourceEstimate | None:
    if inventory.total_bytes <= 0:
        return None
    # Source bytes are authoritative for real downloaded files. The INT4
    # estimate is derived from the source FP16/BF16 weight byte density.
    source = inventory.total_bytes
    quantized = max(1, (source * bits + 15) // 16)
    package = int(quantized * 1.20)
    ram = max(source + 8 * 1024**3, 8 * 1024**3)
    disk = source + quantized * 2 + 20 * 1024**3
    return ResourceEstimate(
        parameters=inventory.parameter_count or 0,
        source_weight_bytes=source,
        estimated_quantized_bytes=quantized,
        estimated_package_bytes=package,
        recommended_ram_bytes=ram,
        recommended_disk_bytes=disk,
    )
