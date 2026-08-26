"""Small subprocess boundary for CANN/quantization toolchains."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def run_command(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> CommandResult:
    """Run an explicitly constructed command and capture stdout/stderr.

    Commands are passed as argv arrays rather than shell strings to avoid shell
    interpretation. Backend-specific code is responsible for constructing the
    arguments appropriate to the installed CANN Kit/tool version.
    """
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    result = CommandResult(tuple(command), process.returncode, process.stdout, process.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"external command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return result
