import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from hllm.validation import validate_hllm


def _make_package(path: Path, payload: bytes = b"model-data") -> None:
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "artifacts": [{"type": "model", "path": "model/model.om", "sha256": digest, "size": len(payload)}],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("model/model.om", payload)


def test_validate_hllm_success(tmp_path: Path) -> None:
    package = tmp_path / "model.hllm"
    _make_package(package)
    assert validate_hllm(package)["schema_version"] == "1.0"


def test_validate_hllm_detects_checksum(tmp_path: Path) -> None:
    package = tmp_path / "model.hllm"
    _make_package(package)
    with zipfile.ZipFile(package, "a") as archive:
        archive.writestr("model/model.om", b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_hllm(package)
