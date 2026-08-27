"""Build explicit HarmonyOS OMG/OMC command lines."""
from __future__ import annotations

from pathlib import Path

from hllm.config import BuildProfile


def prepare_omg_output(output: Path) -> Path:
    """Prepare an OMG output prefix without pre-creating the result file.

    The DDK OMG tool expects ``--output`` to be a new output prefix.  Older
    GraphEngine builds validate the path with ``realpath`` but do not require
    the leaf itself to exist.  Pre-creating the leaf (for example with
    ``touch``) causes the same DDK builds to reject the output as invalid.
    Existing regular files/symlinks from a previous failed build are removed;
    directories are rejected because they are never valid OMG output leaves.
    """
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.is_dir():
            raise ValueError(f"OMG output must be a file prefix, not a directory: {output}")
        output.unlink()
    return output


def build_omg_command(profile: BuildProfile, *, model: Path, output: Path) -> tuple[str, ...]:
    if profile.conversion_tool != "omg":
        raise ValueError(f"OMG command requested for conversion tool '{profile.conversion_tool}'")
    if profile.framework != 5:
        raise ValueError(f"Kirin ONNX conversion requires framework=5, got {profile.framework}")
    if profile.cann_target != "omc":
        raise ValueError(f"HarmonyOS Kirin build requires target=omc, got {profile.cann_target}")
    platform = (profile.platform or "").strip()
    if not platform:
        raise ValueError("Kirin OMC profile requires cann.platform (for example: kirinx90)")
    model_path = model.expanduser().resolve()
    output_path = prepare_omg_output(output)
    return (
        "omg",
        f"--model={model_path}",
        "--framework=5",
        f"--output={output_path}",
        "--target=omc",
        f"--platform={platform}",
    )
