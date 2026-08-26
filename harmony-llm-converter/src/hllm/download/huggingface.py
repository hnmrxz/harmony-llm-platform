"""Hugging Face Hub integration kept behind a small adapter."""

from __future__ import annotations

from pathlib import Path


def download_model(
    repo_id: str,
    output_dir: str | Path,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download a Hugging Face repository into ``output_dir``.

    The optional dependency is imported lazily so ``hllm doctor`` and local
    metadata operations can work without installing the Hub client.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face support is not installed; install the 'huggingface'
            extra."
        ) from exc

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return Path(local_path).resolve()
