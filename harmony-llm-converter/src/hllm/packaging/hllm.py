"""Create the portable .hllm archive."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterable

from hllm.schema.manifest import Artifact, Manifest


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def package_hllm(
    output: str | Path,
    manifest: Manifest,
    files: Iterable[tuple[Path, str]],
) -> Path:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = list(files)
    manifest.artifacts.clear()
    for source, archive_path in entries:
        if not source.is_file():
            raise FileNotFoundError(source)
        manifest.artifacts.append(
            Artifact(
                type="model" if archive_path.startswith("model/") else "resource",
                path=archive_path,
                sha256=sha256_file(source),
                size=source.stat().st_size,
            )
        )

    manifest_json = json.dumps(
        manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_json)
        for source, archive_path in entries:
            archive.write(source, archive_path)
    return destination


def validate_hllm(path: str | Path) -> dict:
    """Backward-compatible wrapper; validation.py is the single implementation."""
    from hllm.validation import validate_hllm as validate

    return validate(path)
