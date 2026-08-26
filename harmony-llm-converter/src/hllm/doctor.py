"""Environment diagnostics for the Ubuntu conversion host."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_checks() -> list[Check]:
    checks: list[Check] = []
    checks.append(Check("OS", platform.system() == "Linux", platform.platform()))
    checks.append(Check("Python", sys.version_info >= (3, 10), platform.python_version()))

    for module in ("torch", "transformers", "onnx", "huggingface_hub"):
        installed = importlib.util.find_spec(module) is not None
        checks.append(Check(module, installed, "installed" if installed else "not installed"))

    cann_tools = ["atc", "omg"]
    for tool in cann_tools:
        location = shutil.which(tool)
        checks.append(Check(tool, location is not None, location or "not found in PATH"))

    return checks
