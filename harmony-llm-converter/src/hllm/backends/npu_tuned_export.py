"""Official CANN LLM NPU-tuned export boundary.

The CANN LLM solution does not use a generic torch ONNX export. Instead it
exports the model with NPU-affinity-adapted ``npu_tuned_model`` modeling files:

    * fill ``model_info_target.yaml`` with the model architecture + path
    * run ``export_model_single_<arch>.py`` (its ``info_path`` points at the yaml)

The result is a *friendly* ONNX (+ pb weights) whose input/output signature
matches what the subsequent OMG step (see ``cann_omg``) expects — a graph with
``input_embed`` / ``attention_mask`` / ``position_ids`` / ``past_key*`` /
``past_value*`` / ``new_kv_cache_pos`` / ``embed_scales`` inputs.

This module renders the yaml and the export command; it does not execute the
GPU toolchain itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class NpuTunedExportSpec:
    model_dir: Path
    output_dir: Path
    model_arch: str
    info_path: Path
    export_script: str | None = None
    onnx_name: str = "model"

    @property
    def script(self) -> str:
        """Use the architecture-specific export script when not overridden."""
        if self.export_script:
            return self.export_script
        arch = self.model_arch.lower()
        if "qwen" in arch:
            return "export_model_single_qwen2.py"
        return "export_model_single.py"


@dataclass(frozen=True, slots=True)
class NpuTunedExportArtifacts:
    onnx: tuple[str, ...] = ("model.onnx",)
    weights: tuple[str, ...] = ("model.onnx.data",)


def build_export_model_info_yaml(spec: NpuTunedExportSpec) -> str:
    """Render the official ``model_info_target.yaml``.

    The original file in the CANN sample uses keys such as ``model_arch`` and a
    path to the source model; convert the exact fields to the ones the DDK's
    export script reads. If a field name differs, pass a custom yaml via the
    profile's ``export.command`` instead of relying on this default render.
    """
    model_dir = spec.model_dir.expanduser().resolve()
    output_dir = spec.output_dir.expanduser().resolve()
    return f"""\
model_arch: {spec.model_arch}
model_path: {model_dir}
output_dir: {output_dir}
onnx_name: {spec.onnx_name}
"""


def build_export_command(spec: NpuTunedExportSpec) -> tuple[str, ...]:
    """Return the export command. ``info_path`` is passed as a CLI argument when
    the script supports it; otherwise it must be wired into the script."""
    script = spec.script
    # The sample hardcodes info_path inside the script; passing it as an arg is
    # the portable default for drives that support it.
    return ("python", script, f"--info-path={spec.info_path.expanduser().resolve()}")
