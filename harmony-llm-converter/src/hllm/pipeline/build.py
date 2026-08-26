from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from hllm.config import BuildProfile, load_profile
from hllm.models.detector import inspect_model
from hllm.models.registry import ModelRegistry
from hllm.models.adapters import default_registry
from hllm.models.weights import inspect_weight_index
from hllm.planner import estimate_from_weights
from hllm.pipeline import BuildOptions, Stage
from hllm.pipeline.runner import PipelineRunner


@dataclass(frozen=True, slots=True)
class BuildRequest:
    source: str
    target_chip: str
    quantization: str
    context_length: int | None
    output_dir: Path
    profile: Path | None
    dry_run: bool


def _profile_from_request(request: BuildRequest) -> BuildProfile:
    if request.profile:
        return load_profile(request.profile)
    return BuildProfile(
        model_source=request.source,
        target_chip=request.target_chip,
        quantization=request.quantization,
        bits=4 if request.quantization == "int4" else 8,
        context_length=request.context_length,
        output_dir=request.output_dir,
    )


def build(request: BuildRequest) -> dict:
    profile = _profile_from_request(request)
    source = Path(request.source).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("build requires a local model directory; run `hllm download` first")

    metadata = inspect_model(source)
    registry: ModelRegistry = default_registry()
    match = registry.resolve(metadata)
    inventory = inspect_weight_index(source)
    estimate = estimate_from_weights(inventory, bits=profile.bits)
    if estimate is not None:
        available_disk = shutil.disk_usage(profile.output_dir).free
        if available_disk < estimate.recommended_disk_bytes:
            return {
                "status": "failed",
                "stage": Stage.INSPECT.value,
                "error": "INSUFFICIENT_DISK",
                "required_bytes": estimate.recommended_disk_bytes,
                "available_bytes": available_disk,
            }

    options = BuildOptions(
        source=str(source),
        output_dir=profile.output_dir,
        target_chip=profile.target_chip,
        quantization=profile.quantization,
        context_length=profile.context_length,
    )
    runner = PipelineRunner(options)
    runner.prepare()
    runner.set_source(source)
    runner.enter(Stage.INSPECT)
    runner.state.record(f"adapter={match.family}")
    runner.state.record(f"model_type={metadata.model_type}")
    runner.state.record(f"architecture={metadata.architecture}")
    runner.state.record(f"multimodal={metadata.is_multimodal}")

    if request.dry_run:
        for stage in (Stage.QUANTIZE, Stage.EXPORT, Stage.CANN_CONVERT, Stage.VALIDATE, Stage.PACKAGE):
            runner.enter(stage)
        return {
            "status": "success",
            "dry_run": True,
            "model": metadata.name,
            "adapter": match.family,
            "inventory": asdict(inventory),
            "resource_estimate": asdict(estimate) if estimate else None,
            "stages": [stage.value for stage in Stage],
        }

    raise RuntimeError(
        "Real conversion execution requires a validated quantization/export/CANN profile. "
        "Use --profile with commands validated against the installed CANN Kit release."
    )
