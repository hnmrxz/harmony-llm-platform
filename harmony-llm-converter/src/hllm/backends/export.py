"""ONNX export boundary.

Actual export flags are model/framework/version-specific, so the converter
accepts an explicit command template rather than guessing them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hllm.backends.commands import CommandResult, run_command


@dataclass(frozen=True, slots=True)
class ExportProfile:
    name: str
    command: tuple[str, ...]


class OnnxExporter:
    def __init__(self, profile: ExportProfile) -> None:
        self.profile = profile

    def export(self, *, model_dir: Path, output_dir: Path, dry_run: bool = False) -> CommandResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = tuple(
            item.replace("{model}", str(model_dir))
            .replace("{output}", str(output_dir))
            for item in self.profile.command
        )
        return run_command(argv, cwd=model_dir, dry_run=dry_run)
