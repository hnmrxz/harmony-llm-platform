"""Environment diagnostics for the Ubuntu conversion host."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _candidate_cann_roots() -> list[Path]:
    candidates: list[Path] = []
    env_root = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    home = Path.home()
    candidates.extend(
        [
            Path("/usr/local/Ascend/cann"),
            Path("/usr/local/Ascend/ascend-toolkit"),
            home / "Ascend/cann",
            home / "Ascend/ascend-toolkit",
        ]
    )
    return list(dict.fromkeys(candidates))


def _find_tool(name: str, candidates: list[Path]) -> str | None:
    path = shutil.which(name)
    if path:
        return path
    for root in candidates:
        possible = [
            root / "bin" / name,
            root / "latest" / "bin" / name,
        ]
        for candidate in possible:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def run_checks() -> list[Check]:
    checks: list[Check] = [
        Check("OS", platform.system() == "Linux", platform.platform()),
        Check("Python", sys.version_info >= (3, 10), platform.python_version()),
    ]

    for module in ("torch", "transformers", "onnx", "huggingface_hub"):
        installed = importlib.util.find_spec(module) is not None
        checks.append(Check(module, installed, "installed" if installed else "not installed"))

    candidates = _candidate_cann_roots()
    atc = _find_tool("atc", candidates)
    checks.append(
        Check(
            "atc",
            atc is not None,
            atc or "not found; source CANN set_env.sh or install CANN Toolkit",
        )
    )

    omg = _find_tool("omg", candidates)
    checks.append(
        Check(
            "omg",
            omg is not None,
            omg or "not found (optional for some CANN conversion profiles)",
            required=False,
        )
    )

    set_env = next(
        (root / "set_env.sh" for root in candidates if (root / "set_env.sh").is_file()),
        None,
    )
    checks.append(
        Check(
            "CANN environment",
            set_env is not None or bool(os.environ.get("ASCEND_HOME_PATH")),
            str(set_env) if set_env else "not detected; source the installed CANN set_env.sh",
        )
    )
    return checks
