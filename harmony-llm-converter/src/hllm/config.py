from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ALLOWED_BUILD_KEYS = {"model", "pipeline", "quantization", "export", "cann", "runtime", "output"}


@dataclass(frozen=True, slots=True)
class BuildProfile:
    model_source: str
    model_family: str | None
    input_quantization: str | None
    preferred_path: str | None
    fallback_path: str | None
    target_chip: str
    runtime_version: str | None
    quantization: str
    bits: int
    context_length: int | None
    output_dir: Path
    quantization_workspace: Path | None = None
    conversion_workspace: Path | None = None
    cann_commands: tuple[tuple[str, ...], ...] = ()
    export_command: tuple[str, ...] = ()
    quantization_commands: tuple[tuple[str, ...], ...] = ()
    capabilities: tuple[tuple[str, bool], ...] = ()

    @property
    def supports_fp8_input(self) -> bool:
        return self.input_quantization == "fp8" or self.preferred_path == "fp8_to_cann"

    @property
    def capability_map(self) -> dict[str, bool]:
        return dict(self.capabilities)


@dataclass(frozen=True, slots=True)
class TargetProfile:
    platform: str
    soc_version: str
    runtime_model_format: str
    cann_version: str | None
    conversion_tool: str
    framework: int
    target: str


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"profile section '{name}' must be an object")
    return value


def _validate_schema(data: dict[str, Any], schema_name: str, message: str) -> None:
    schema_path = Path(__file__).resolve().parents[3] / "docs" / schema_name
    if not schema_path.is_file():
        return
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("jsonschema is required to load profiles") from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: list(e.path))
    if errors:
        path = ".".join(str(p) for p in errors[0].path) or "$"
        raise ValueError(f"{message} at {path}: {errors[0].message}")


def _validate_build_shape(data: dict[str, Any]) -> None:
    unknown = set(data) - _ALLOWED_BUILD_KEYS
    if unknown:
        raise ValueError(f"unknown build profile keys: {sorted(unknown)}")
    _validate_schema(data, "build-profile.schema.json", "build profile schema error")


def load_profile(path: str | Path) -> BuildProfile:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("install PyYAML to use YAML build profiles") from exc
    data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("build profile root must be a mapping")
    _validate_build_shape(data)
    model, pipeline = _section(data, "model"), _section(data, "pipeline")
    quant, runtime = _section(data, "quantization"), _section(data, "runtime")
    output, cann, export = _section(data, "output"), _section(data, "cann"), _section(data, "export")
    source = str(model.get("source", "")).strip()
    target_chip = str(cann.get("target_chip", "")).strip()
    if not source or not target_chip:
        raise ValueError("build profile requires model.source and cann.target_chip")
    capabilities = runtime.get("capabilities", {})
    if not isinstance(capabilities, dict):
        raise ValueError("runtime.capabilities must be an object")
    return BuildProfile(
        model_source=source,
        model_family=str(model.get("family")) if model.get("family") else None,
        input_quantization=str(model.get("input_quantization")) if model.get("input_quantization") else None,
        preferred_path=str(pipeline.get("preferred_path")) if pipeline.get("preferred_path") else None,
        fallback_path=str(pipeline.get("fallback_path")) if pipeline.get("fallback_path") else None,
        target_chip=target_chip,
        runtime_version=str(cann.get("runtime_version")) if cann.get("runtime_version") else None,
        quantization=str(quant.get("method", "cann_4bit")), bits=int(quant.get("bits", 4)),
        context_length=int(runtime["context_length"]) if runtime.get("context_length") else None,
        output_dir=Path(output.get("directory", "./dist")).expanduser(),
        quantization_workspace=Path(quant["workspace"]).expanduser() if quant.get("workspace") else None,
        conversion_workspace=Path(cann["workspace"]).expanduser() if cann.get("workspace") else None,
        cann_commands=tuple(tuple(map(str, cmd)) for cmd in cann.get("commands", [])),
        export_command=tuple(map(str, export.get("command", []))),
        quantization_commands=tuple(tuple(map(str, cmd)) for cmd in quant.get("commands", [])),
        capabilities=tuple((str(k), bool(v)) for k, v in capabilities.items()),
    )


def load_target_profile(path: str | Path) -> TargetProfile:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("install PyYAML to use YAML target profiles") from exc
    data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("target profile root must be a mapping")
    _validate_schema(data, "target-profile.schema.json", "target profile schema error")
    target, cann = _section(data, "target"), _section(data, "cann")
    return TargetProfile(
        platform=str(target["platform"]), soc_version=str(target["soc_version"]),
        runtime_model_format=str(target["runtime_model_format"]),
        cann_version=str(cann.get("version")) if cann.get("version") else None,
        conversion_tool=str(cann.get("conversion_tool", "omg")), framework=int(cann.get("framework", 5)),
        target=str(cann.get("target", "omc")),
    )
