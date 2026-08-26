"""Inspect Hugging Face weight indexes without loading tensors."""
from __future__ import annotations

import json
import struct
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
        if self.parameter_count is None:
            return 0
        return (self.parameter_count + 1) // 2


def _safetensors_header(path: Path) -> tuple[int, int]:
    """Return parameter count and tensor count from a safetensors header only."""
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len > 100 * 1024 * 1024:
            raise ValueError(f"unreasonable safetensors header: {path}")
        header = json.loads(handle.read(header_len))
    parameters = 0
    tensors = 0
    for key, value in header.items():
        if key == "__metadata__":
            continue
        shape = value.get("shape", [])
        count = 1
        for dim in shape:
            count *= int(dim)
        parameters += count
        tensors += 1
    return parameters, tensors


def inspect_weight_index(model_dir: str | Path) -> WeightInventory:
    root = Path(model_dir).expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            index: dict[str, Any] = json.load(handle)
        weight_map = index.get("weight_map", {})
        metadata = index.get("metadata", {})
        shard_names = sorted(set(weight_map.values()))
    else:
        weight_map = {}
        metadata = {}
        shard_names = sorted(path.name for path in root.glob("*.safetensors"))

    total_bytes = int(metadata.get("total_size", 0) or 0)
    if total_bytes <= 0:
        total_bytes = sum((root / name).stat().st_size for name in shard_names if (root / name).is_file())

    parameter_count = 0
    tensor_count = 0
    for shard_name in shard_names:
        shard = root / shard_name
        if not shard.is_file():
            raise ValueError(f"missing safetensors shard: {shard_name}")
        try:
            shard_params, shard_tensors = _safetensors_header(shard)
        except (OSError, ValueError, json.JSONDecodeError, struct.error):
            shard_params, shard_tensors = 0, 0
        parameter_count += shard_params
        tensor_count += shard_tensors

    return WeightInventory(
        parameter_count=parameter_count or None,
        total_bytes=total_bytes,
        tensor_count=tensor_count or len(weight_map),
        shard_count=len(shard_names),
    )
