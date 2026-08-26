"""Run external toolchain commands with captured logs and explicit failures."""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class ExternalCommandError(RuntimeError):
    def __init__(self, result: CommandResult) -> None:
        self.result = result
        super().__init__(
            f"command failed ({result.returncode}): {shlex.join(result.argv)}\n{result.stderr.strip()}"
        )


def run_command(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
) -> CommandResult:
    args = tuple(str(item) for item in argv)
    if not args:
        raise ValueError("empty command")
    if dry_run:
        return CommandResult(args, 0, "", "")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    result = CommandResult(args, completed.returncode, completed.stdout, completed.stderr)
    if result.returncode != 0:
        raise ExternalCommandError(result)
    return result
