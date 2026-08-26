from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Callable, Iterable

from hllm.backends.commands import CommandResult, run_command


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    model: Path
    work: Path
    output: Path
    target: str
    quant: str

    def render(self, value: str) -> str:
        replacements = {
            "model": str(self.model),
            "work": str(self.work),
            "output": str(self.output),
            "target": self.target,
            "quant": self.quant,
        }
        for key, replacement in replacements.items():
            value = value.replace("{" + key + "}", replacement)
        return Template(value).safe_substitute(replacements)


class StageExecutor:
    def __init__(self, context: ExecutionContext, dry_run: bool = False, logger: Callable[[str], None] | None = None) -> None:
        self.context = context
        self.dry_run = dry_run
        self.logger = logger

    def run(self, commands: Iterable[Iterable[str]]) -> list[CommandResult]:
        results: list[CommandResult] = []
        for command in commands:
            rendered = tuple(self.context.render(str(arg)) for arg in command)
            self.logger and self.logger(f"command={shlex.join(rendered)}")
            result = run_command(rendered, cwd=self.context.work, dry_run=self.dry_run)
            results.append(result)
            self.logger and self.logger(f"returncode={result.returncode}")
            if result.stdout:
                self.logger and self.logger(f"stdout={result.stdout.rstrip()}")
            if result.stderr:
                self.logger and self.logger(f"stderr={result.stderr.rstrip()}")
        return results

    @staticmethod
    def command_for_shell(script: str) -> tuple[str, ...]:
        return ("bash", "-lc", script)
