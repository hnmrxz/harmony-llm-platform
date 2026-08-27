"""Tests for the external command runner boundary."""
from __future__ import annotations

import pytest

from hllm.backends.commands import ExternalCommandError, run_command


def test_missing_binary_raises_structured_failure() -> None:
    with pytest.raises(ExternalCommandError) as excinfo:
        run_command(["hllm-definitely-not-a-real-tool-xyz", "--flag"])
    assert excinfo.value.result.returncode == 127
    assert "command not found" in excinfo.value.result.stderr
    assert "hllm-definitely-not-a-real-tool-xyz" in str(excinfo.value)


def test_dry_run_skips_execution() -> None:
    result = run_command(["hllm-definitely-not-a-real-tool-xyz"], dry_run=True)
    assert result.returncode == 0
    assert result.stdout == ""


def test_timeout_surfaces_as_external_error() -> None:
    with pytest.raises(ExternalCommandError) as excinfo:
        run_command(["bash", "-c", "sleep 5"], timeout=1)
    assert excinfo.value.result.returncode == 124


def test_nonzero_exit_raises_with_captured_output() -> None:
    with pytest.raises(ExternalCommandError) as excinfo:
        run_command(["bash", "-c", "echo boom >&2; exit 3"])
    assert excinfo.value.result.returncode == 3
    assert "boom" in excinfo.value.result.stderr
