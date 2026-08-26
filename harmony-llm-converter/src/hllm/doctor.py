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


def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    env_values = (
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        os.environ.get("ASCEND_HOME_PATH"),
        os.environ.get("DDK_INSTALL_PATH"),
    )
    for value in env_values:
        if value:
            candidates.append(Path(value).expanduser())

    home = Path.home()
    candidates.extend(
        [
            Path("/usr/local/Ascend/cann"),
            Path("/usr/local/Ascend/ascend-toolkit"),
            home / "Ascend/cann",
            home / "Ascend/ascend-toolkit",
            home / "Ascend/cann-9.1.0",
        ]
    )
    return list(dict.fromkeys(candidates))


def _find_tool(name: str, roots: list[Path]) -> str | None:
    path = shutil.which(name)
    if path:
        return path
    for root in roots:
        candidates = [
            root / "bin" / name,
            root / "latest" / "bin" / name,
            root / "tools" / name,
            root / "tools" / "tools_omg" / name,
        ]
        for candidate in candidates:
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

    roots = _candidate_roots()
    atc = _find_tool("atc", roots)
    checks.append(
        Check(
            "ATC",
            atc is not None,
            atc or "not found; source CANN set_env.sh or install CANN Toolkit",
        )
    )

    omg = _find_tool("omg", roots)
    checks.append(
        Check(
            "OMG",
            omg is not None,
            omg
            or "not found; required for HarmonyOS/Kirin OMC conversion profiles, "
               "not required for ATC-only Ascend profiles",
            required=False,
        )
    )

    set_env = next(
        (root / "set_env.sh" for root in roots if (root / "set_env.sh").is_file()),
        None,
    )
    checks.append(
        Check(
            "CANN environment",
            set_env is not None or bool(os.environ.get("ASCEND_HOME_PATH")),
            str(set_env) if set_env else "not detected; source installed CANN set_env.sh",
        )
    )

    ddk_tools_omg = [root / "tools" / "tools_omg" for root in roots]
    ddk_root = next((p for p in ddk_tools_omg if p.is_dir()), None)
    checks.append(
        Check(
            "HarmonyOS DDK tools_omg",
            ddk_root is not None,
            str(ddk_root) if ddk_root else "not detected",
            required=False,
        )
    )

    return checks
