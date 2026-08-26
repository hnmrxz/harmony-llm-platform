"""CANN command backend.

This module deliberately does not guess vendor-tool flags. CANN Kit releases
and target devices can differ, so a target profile supplies the exact command.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CannProfile:
    name: str
    target_chip: str
    commands: tuple[tuple[str, ...], ...]


class CannBackend:
    def __init__(self, profile: CannProfile) -> None:
        self.profile = profile

    def validate(self) -> None:
        if not self.profile.commands:
            raise ValueError(f"CANN profile has no conversion commands: {self.profile.name}")

    def run(self, *, workdir: Path) -> None:
        self.validate()
        for argv in self.profile.commands:
            subprocess.run(list(argv), cwd=workdir, check=True)
