from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from hllm.backends.commands import CommandResult, run_command


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    model: Path
    work: Path
    output: Path
    target: str
    quant: str

    def render(self, value: str) -> str:
        return value.format(
            model=self.model,
            work=self.work,
            output=self.output,
            target=self.target,
            quant=self.quant,
        )


class StageExecutor:
    def __init__(self, context: ExecutionContext, dry_run: bool = False) -> None:
        self.context = context
        self.dry_run = dry_run

    def run(self, commands: Iterable[Iterable[str]]) -> list[CommandResult]:
        results: list[CommandResult] = []
        for command in commands:
            rendered = tuple(self.context.render(str(arg)) for arg in command)
            results.append(run_command(rendered, cwd=self.context.work, dry_run=self.dry_run))
        return results

    @staticmethod
    def command_for_shell(script: str) -> tuple[str, ...]:
        return ("bash", "-lc", script)
