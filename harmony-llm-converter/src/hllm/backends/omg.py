"""Build explicit HarmonyOS OMG/OMC command lines."""
from __future__ import annotations

from pathlib import Path

from hllm.config import BuildProfile


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
    return (
        "omg",
        f"--model={model}",
        "--framework=5",
        f"--output={output}",
        "--target=omc",
        f"--platform={platform}",
    )
