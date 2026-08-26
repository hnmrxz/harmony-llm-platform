from pathlib import Path

import pytest

from hllm.config import load_profile, load_target_profile


def test_load_build_profile_and_fp8(tmp_path: Path) -> None:
    profile = tmp_path / "build.yaml"
    profile.write_text(
        """\nmodel:\n  source: Qwen/Qwen3.8-27B-FP8\n  family: qwen3.8\n  input_quantization: fp8\npipeline:\n  preferred_path: fp8_to_cann\nquantization:\n  method: cann_4bit\n  bits: 4\ncann:\n  target_chip: kirin9020\nruntime:\n  context_length: 8192\noutput:\n  directory: ./dist\n""",
        encoding="utf-8",
    )
    loaded = load_profile(profile)
    assert loaded.target_chip == "kirin9020"
    assert loaded.supports_fp8_input
    assert loaded.input_quantization == "fp8"


def test_unknown_build_profile_key_rejected(tmp_path: Path) -> None:
    profile = tmp_path / "bad.yaml"
    profile.write_text("model:\n  source: x\nextra: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown build profile keys"):
        load_profile(profile)


def test_target_profile_is_separate_schema(tmp_path: Path) -> None:
    profile = tmp_path / "target.yaml"
    profile.write_text(
        "target:\n  platform: kirin9020\n  soc_version: kirin9020\n  runtime_model_format: omc\ncann:\n  version: '9.1.0'\n  conversion_tool: omg\n  framework: 5\n  target: omc\n",
        encoding="utf-8",
    )
    loaded = load_target_profile(profile)
    assert loaded.platform == "kirin9020"
    assert loaded.conversion_tool == "omg"
