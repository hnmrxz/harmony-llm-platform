from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_hllm(path: str | Path) -> dict:
    package = Path(path).expanduser().resolve()
    if package.suffix != ".hllm" or not package.is_file():
        raise ValueError("expected an existing .hllm package")
    with zipfile.ZipFile(package, "r") as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise ValueError("missing manifest.json")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != "1.0":
            raise ValueError("unsupported HLLM schema_version")
        for artifact in manifest.get("artifacts", []):
            name = artifact["path"]
            if name not in names:
                raise ValueError(f"missing artifact: {name}")
            expected = artifact.get("sha256")
            if expected:
                actual = sha256_bytes(archive.read(name))
                if actual != expected:
                    raise ValueError(f"checksum mismatch: {name}")
        return manifest
