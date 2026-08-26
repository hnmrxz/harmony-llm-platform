from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BuildProfile:
    model_source: str
    target_chip: str
    quantization: str
    bits: int
    context_length: int | None
    output_dir: Path
    quantization_workspace: Path | None = None
    conversion_workspace: Path | None = None
    cann_commands: tuple[tuple[str, ...], ...] = ()
    export_command: tuple[str, ...] = ()
    quantization_commands: tuple[tuple[str, ...], ...] = ()


def load_profile(path: str | Path) -> BuildProfile:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("install PyYAML to use YAML build profiles") from exc

    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    model = data.get("model", {})
    quant = data.get("quantization", {})
    runtime = data.get("runtime", {})
    output = data.get("output", {})
    cann = data.get("cann", {})
    export = data.get("export", {})
    return BuildProfile(
        model_source=str(model.get("source", "")),
        target_chip=str(cann.get("target_chip", "")),
        quantization=str(quant.get("method", "cann_4bit")),
        bits=int(quant.get("bits", 4)),
        context_length=int(runtime["context_length"]) if runtime.get("context_length") else None,
        output_dir=Path(output.get("directory", "./dist")).expanduser(),
        quantization_workspace=Path(quant["workspace"]).expanduser() if quant.get("workspace") else None,
        conversion_workspace=Path(cann["workspace"]).expanduser() if cann.get("workspace") else None,
        cann_commands=tuple(tuple(map(str, cmd)) for cmd in cann.get("commands", [])),
        export_command=tuple(map(str, export.get("command", []))),
        quantization_commands=tuple(tuple(map(str, cmd)) for cmd in quant.get("commands", [])),
    )
