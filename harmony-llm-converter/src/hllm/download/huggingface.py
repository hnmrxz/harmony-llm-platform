"""Hugging Face Hub integration kept behind a small adapter."""

from __future__ import annotations

import os
from pathlib import Path


def download_model(
    repo_id: str,
    output_dir: str | Path,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download a Hugging Face repository into ``output_dir``.

    ``huggingface_hub`` is imported lazily so local inspection and doctor
    commands can work without the Hub dependency.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face support is not installed; install the 'huggingface' extra."
        ) from exc

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    effective_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "token": effective_token,
        "local_dir": str(target),
    }

    # ``local_dir_use_symlinks`` was removed from newer huggingface_hub
    # releases. Do not pass it so the downloader remains compatible with
    # current HF Hub 1.x installations.
    local_path = snapshot_download(**kwargs)
    return Path(local_path).resolve()
