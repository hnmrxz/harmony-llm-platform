from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from hllm.backends.commands import ExternalCommandError
from hllm.config import BuildProfile, load_profile
from hllm.download.huggingface import download_model
from hllm.models.adapters import default_registry
from hllm.models.detector import inspect_model
from hllm.models.weights import inspect_weight_index
from hllm.packaging.artifacts import assemble_artifacts
from hllm.packaging.hllm import package_hllm
from hllm.pipeline import BuildOptions, Stage
from hllm.pipeline.executor import ExecutionContext, StageExecutor
from hllm.pipeline.runner import PipelineRunner
from hllm.planner import can_start, estimate_from_weights
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
    model_cache_dir: Path = Path("./models")


def _default_profile(request: BuildRequest) -> BuildProfile:
    return BuildProfile(
        model_source=request.source,
        model_family=None,
        input_quantization=None,
        preferred_path=None,
        fallback_path=None,
        target_chip=request.target_chip,
        runtime_version=None,
        quantization=request.quantization,
        bits=4 if request.quantization == "int4" else 8,
        context_length=request.context_length,
        output_dir=request.output_dir,
    )


def _is_repo_id(value: str) -> bool:
    return "/" in value and not Path(value).expanduser().exists()


def _resolve_source(request: BuildRequest) -> tuple[Path, str | None]:
    source = Path(request.source).expanduser()
    if source.is_dir():
        return source.resolve(), None
    if _is_repo_id(request.source):
        destination = request.model_cache_dir.expanduser() / request.source.replace("/", "__")
        return download_model(request.source, destination), request.source
    raise ValueError("model must be a local directory or a Hugging Face repo id")


def build(request: BuildRequest) -> dict:
    profile = load_profile(request.profile) if request.profile else _default_profile(request)
    try:
        source, source_id = _resolve_source(request)
        metadata = inspect_model(source)
        match = default_registry().resolve(metadata)
        match.adapter.validate(metadata)
        inventory = inspect_weight_index(source)

        if metadata.is_fp8 and not profile.supports_fp8_input:
            return {"status": "failed", "stage": Stage.INSPECT.value, "error": "UNSUPPORTED_FP8_TARGET"}
        if profile.input_quantization == "fp8" and not metadata.is_fp8:
            return {"status": "failed", "stage": Stage.INSPECT.value, "error": "INPUT_QUANTIZATION_MISMATCH"}

        bits_for_plan = 8 if metadata.is_fp8 else profile.bits
        estimate = estimate_from_weights(inventory, bits=bits_for_plan)
        profile.output_dir.mkdir(parents=True, exist_ok=True)
        available_disk = shutil.disk_usage(profile.output_dir).free
        import psutil
        available_ram = psutil.virtual_memory().available
        if estimate and not can_start(available_ram=available_ram, available_disk=available_disk, estimate=estimate):
            error = "INSUFFICIENT_DISK" if available_disk < estimate.recommended_disk_bytes else "INSUFFICIENT_RAM"
            return {"status": "failed", "stage": Stage.INSPECT.value, "error": error,
                    "required_ram_bytes": estimate.recommended_ram_bytes, "available_ram_bytes": available_ram,
                    "required_disk_bytes": estimate.recommended_disk_bytes, "available_disk_bytes": available_disk}

        options = BuildOptions(source=str(source), output_dir=profile.output_dir, target_chip=profile.target_chip,
                               quantization=profile.quantization, context_length=profile.context_length,
                               revision=source_id)
        runner = PipelineRunner(options)
        runner.prepare(); runner.set_source(source); runner.enter(Stage.INSPECT)
        runner.state.record(f"adapter={match.family}")
        runner.state.record(f"dtype={metadata.dtype}")
        runner.state.record(f"parameters={metadata.parameter_count or inventory.parameter_count}")
        runner.state.record(f"available_ram={available_ram}")
        runner.state.record(f"available_disk={available_disk}")

        if request.dry_run:
            stages = [Stage.DOWNLOAD, Stage.INSPECT]
            if not (metadata.is_fp8 and profile.supports_fp8_input): stages.append(Stage.QUANTIZE)
            stages.extend([Stage.EXPORT, Stage.CANN_CONVERT, Stage.VALIDATE, Stage.PACKAGE])
            return {"status": "success", "dry_run": True, "model": metadata.name, "adapter": match.family,
                    "dtype": metadata.dtype, "inventory": asdict(inventory),
                    "resource_estimate": asdict(estimate) if estimate else None,
                    "stages": [stage.value for stage in stages], "logs": runner.state.logs}

        context = ExecutionContext(model=source, work=runner.work_dir, output=runner.dist_dir,
                                   target=profile.target_chip, quant=profile.quantization)
        executor = StageExecutor(context, logger=runner.state.record)
        if not (metadata.is_fp8 and profile.supports_fp8_input):
            runner.enter(Stage.QUANTIZE); executor.run(profile.quantization_commands)
        else:
            runner.state.record("skip=quantize reason=fp8_input")

        runner.enter(Stage.EXPORT)
        if profile.export_command: executor.run((profile.export_command,))
        else: runner.state.record("skip=export reason=no_profile_command")

        runner.enter(Stage.CANN_CONVERT); executor.run(profile.cann_commands)
        model_files = assemble_artifacts(source, runner.work_dir / "final")
        if not model_files:
            raise RuntimeError("no final artifacts found under work/final; target profile must create a deployable artifact")

        runner.enter(Stage.VALIDATE)
        manifest = Manifest(
            schema_version="1.0",
            model=ModelInfo(name=metadata.name, family=match.family, architecture=metadata.architecture or "unknown",
                            source_type="huggingface" if source_id else "local", source_id=source_id or str(source)),
            quantization=QuantizationInfo(type="fp8" if metadata.is_fp8 else profile.quantization, bits=8 if metadata.is_fp8 else profile.bits),
            target=TargetInfo(backend="cann", chip=profile.target_chip, runtime_version=profile.runtime_version),
            runtime=RuntimeInfo(context_length=profile.context_length),
            build=BuildInfo(converter_version="0.1.0", python_version=sys.version.split()[0]),
        )
        runner.enter(Stage.PACKAGE)
        package_name = f"{metadata.name}-{profile.target_chip}-{'fp8' if metadata.is_fp8 else profile.quantization}.hllm"
        package_path = runner.dist_dir / package_name
        package_hllm(package_path, manifest, model_files)
        from hllm.validation import validate_hllm
        validate_hllm(package_path)
        result = {"status": "success", "package": str(package_path), "manifest": manifest.to_dict(), "logs": runner.state.logs}
        (runner.dist_dir / "build.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except ExternalCommandError as exc:
        return {"status": "failed", "error": "EXTERNAL_COMMAND_FAILED", "command": list(exc.result.argv),
                "exit_code": exc.result.returncode, "stderr": exc.result.stderr, "stdout": exc.result.stdout}
