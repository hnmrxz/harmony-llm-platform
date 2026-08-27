import json
from pathlib import Path

from hllm.pipeline.build import BuildRequest, build


def _model(root: Path, dtype: str = "bfloat16") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "torch_dtype": dtype}),
        encoding="utf-8",
    )
    return root


def test_build_dry_run(tmp_path: Path) -> None:
    model = _model(tmp_path / "model")
    result = build(BuildRequest(str(model), "kirin9020", "int4", None, tmp_path / "out", None, True))
    assert result["status"] == "success"
    assert "quantize" in result["stages"]


def test_fp8_without_profile_is_rejected(tmp_path: Path) -> None:
    model = _model(tmp_path / "model", "float8_e4m3fn")
    result = build(BuildRequest(str(model), "kirin9020", "int4", None, tmp_path / "out", None, True))
    assert result["status"] == "failed"
    assert result["error"] == "UNSUPPORTED_FP8_TARGET"
