"""Create and validate the portable .hllm archive."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterable

from hllm.schema.manifest import Manifest


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
    """Package files and a manifest into a deterministic-enough HLLM archive."""
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    entries = list(files)
    manifest.artifacts.clear()
    for source, archive_path in entries:
        if not source.is_file():
            raise FileNotFoundError(source)
        manifest.artifacts.append(
            # Size and checksum are computed before writing so Runtime can
            # verify the exact artifact it is about to install.
            __import__("hllm.schema.manifest", fromlist=["Artifact"]).Artifact(
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
    """Perform structural validation without executing model code."""
    package = Path(path).expanduser().resolve()
    if not package.is_file():
        raise FileNotFoundError(package)
    if package.suffix != ".hllm":
        raise ValueError("HLLM package must use the .hllm extension")

    with zipfile.ZipFile(package, "r") as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise ValueError("missing manifest.json")
        manifest = json.loads(archive.read("manifest.json"))
        for artifact in manifest.get("artifacts", []):
            artifact_path = artifact["path"]
            if artifact_path not in names:
                raise ValueError(f"missing artifact: {artifact_path}")
            expected = artifact.get("sha256")
            if expected:
                actual = hashlib.sha256(archive.read(artifact_path)).hexdigest()
                if actual != expected:
                    raise ValueError(f"checksum mismatch: {artifact_path}")
        return manifest
