from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from hllm.config import BuildProfile, load_profile
from hllm.models.adapters import default_registry
from hllm.models.detector import inspect_model
from hllm.models.weights import inspect_weight_index
from hllm.packaging.hllm import package_hllm
from hllm.pipeline import BuildOptions, Stage
from hllm.pipeline.executor import ExecutionContext, StageExecutor
from hllm.pipeline.runner import PipelineRunner
from hllm.planner import estimate_from_weights
from hllm.schema.manifest import BuildInfo, Manifest, ModelInfo, QuantizationInfo, RuntimeInfo, TargetInfo


@dataclass(frozen=True, slots=True)
class BuildRequest:
    source: str
    target_chip: str
    quantization: str
    context_length: int | None
    output_dir: Path
    profile: Path | None
    dry_run: bool


def _default_profile(request: BuildRequest) -> BuildProfile:
    return BuildProfile(
        model_source=request.source,
        target_chip=request.target_chip,
        quantization=request.quantization,
        bits=4 if request.quantization == "int4" else 8,
        context_length=request.context_length,
        output_dir=request.output_dir,
    )


def build(request: BuildRequest) -> dict:
    profile = load_profile(request.profile) if request.profile else _default_profile(request)
    source = Path(request.source).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("build requires a local model directory; run `hllm download` first")

    metadata = inspect_model(source)
    adapter = default_registry().resolve(metadata).family
    inventory = inspect_weight_index(source)
    estimate = estimate_from_weights(inventory, bits=profile.bits)

    profile.output_dir.mkdir(parents=True, exist_ok=True)
    available_disk = shutil.disk_usage(profile.output_dir).free
    if estimate and available_disk < estimate.recommended_disk_bytes:
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
    runner.state.record(f"adapter={adapter}")

    if request.dry_run:
        for stage in (Stage.QUANTIZE, Stage.EXPORT, Stage.CANN_CONVERT, Stage.VALIDATE, Stage.PACKAGE):
            runner.enter(stage)
        return {"status": "success", "dry_run": True, "model": metadata.name, "adapter": adapter,
                "inventory": asdict(inventory), "resource_estimate": asdict(estimate) if estimate else None,
                "stages": [stage.value for stage in Stage]}

    context = ExecutionContext(
        model=source,
        work=runner.work_dir,
        output=runner.dist_dir,
        target=profile.target_chip,
        quant=profile.quantization,
    )
    executor = StageExecutor(context)

    runner.enter(Stage.QUANTIZE)
    executor.run(profile.quantization_commands)

    runner.enter(Stage.EXPORT)
    executor.run((profile.export_command,) if profile.export_command else ())

    runner.enter(Stage.CANN_CONVERT)
    executor.run(profile.cann_commands)

    final_dir = runner.work_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    # A profile is required to place the device-ready CANN artifact under
    # work/final. Tokenizers/config are copied here so Runtime remains
    # independent from the original Hugging Face checkout.
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "chat_template.json"):
        src = source / name
        if src.is_file():
            destination = final_dir / "tokenizer" / name if name.startswith("token") or name == "special_tokens_map.json" else final_dir / "config" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, destination)
    config_file = source / "config.json"
    if config_file.is_file():
        (final_dir / "config").mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_file, final_dir / "config" / "model.json")

    model_files = [(path, f"model/{path.relative_to(final_dir).as_posix()}") for path in final_dir.rglob("*") if path.is_file()]
    if not model_files:
        raise RuntimeError("no final artifacts found under work/final")

    runner.enter(Stage.VALIDATE)
    runner.enter(Stage.PACKAGE)
    manifest = Manifest(
        schema_version="1.0",
        model=ModelInfo(name=metadata.name, family=adapter, architecture=metadata.architecture or "unknown", source_type="local", source_id=str(source)),
        quantization=QuantizationInfo(type=profile.quantization, bits=profile.bits),
        target=TargetInfo(backend="cann", chip=profile.target_chip),
        runtime=RuntimeInfo(context_length=profile.context_length),
        build=BuildInfo(converter_version="0.1.0"),
    )
    package_name = f"{metadata.name}-{profile.target_chip}-{profile.quantization}.hllm"
    package_path = runner.dist_dir / package_name
    package_hllm(package_path, manifest, model_files)
    (runner.dist_dir / "build.json").write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "success", "package": str(package_path), "manifest": manifest.to_dict()}
