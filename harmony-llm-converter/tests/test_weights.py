import json
import struct
from pathlib import Path

from hllm.models.weights import inspect_weight_index


def _write_safetensors_header(path: Path) -> None:
    header = {
        "a": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
        "b": {"dtype": "BF16", "shape": [4], "data_offsets": [12, 20]},
    }
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"x" * 20)


def test_parameter_count_from_safetensors_header(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_safetensors_header(model / "model.safetensors")
    inventory = inspect_weight_index(model)
    assert inventory.parameter_count == 10
    assert inventory.tensor_count == 2
    assert inventory.shard_count == 1
