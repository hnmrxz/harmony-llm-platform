from hllm.models.weights import WeightInventory
from hllm.planner import estimate_from_weights, estimate_resources


def test_parameter_estimate() -> None:
    estimate = estimate_resources(27_000_000_000, 4)
    assert estimate.estimated_quantized_bytes >= 13_500_000_000
    assert estimate.recommended_disk_bytes > estimate.source_weight_bytes


def test_weight_estimate_uses_downloaded_size() -> None:
    inventory = WeightInventory(None, 55_000_000_000, 100, 18)
    estimate = estimate_from_weights(inventory, 4)
    assert estimate is not None
    assert estimate.source_weight_bytes == 55_000_000_000
    assert estimate.estimated_quantized_bytes == 13_750_000_000
