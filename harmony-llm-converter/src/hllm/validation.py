from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def _sha256_stream(stream, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def validate_hllm(path: str | Path) -> dict:
    package = Path(path).expanduser().resolve()
    if package.suffix != ".hllm" or not package.is_file():
        raise ValueError("expected an existing .hllm package")
    with zipfile.ZipFile(package, "r") as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise ValueError("missing manifest.json")
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid manifest.json") from exc
        if manifest.get("schema_version") != "1.0":
            raise ValueError("unsupported HLLM schema_version")
        for artifact in manifest.get("artifacts", []):
            name = artifact.get("path")
            if not isinstance(name, str) or not name or name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"invalid artifact path: {name!r}")
            if name not in names:
                raise ValueError(f"missing artifact: {name}")
            expected = artifact.get("sha256")
            if expected:
                with archive.open(name, "r") as stream:
                    actual = _sha256_stream(stream)
                if actual != expected:
                    raise ValueError(f"checksum mismatch: {name}")
        return manifest
