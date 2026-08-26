"""Inspect Hugging Face weight indexes without loading tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WeightInventory:
    parameter_count: int | None
    total_bytes: int
    tensor_count: int
    shard_count: int

    @property
    def estimated_int4_weight_bytes(self) -> int:
        """Lower-bound packed INT4 size before scales/metadata overhead."""
        if self.parameter_count is None:
            return 0
        return (self.parameter_count + 1) // 2


def inspect_weight_index(model_dir: str | Path) -> WeightInventory:
    root = Path(model_dir).expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-file safetensors cannot reliably expose parameter count without
        # loading the header; keep this API conservative for now.
        return WeightInventory(None, 0, 0, 1 if any(root.glob("*.safetensors")) else 0)

    with index_path.open("r", encoding="utf-8") as handle:
        index: dict[str, Any] = json.load(handle)

    weight_map = index.get("weight_map", {})
    metadata = index.get("metadata", {})
    total_bytes = int(metadata.get("total_size", 0) or 0)
    shards = set(weight_map.values())
    return WeightInventory(
        parameter_count=None,
        total_bytes=total_bytes,
        tensor_count=len(weight_map),
        shard_count=len(shards),
    )
