from pathlib import Path

from hllm.backends import dopt
from hllm.backends.dopt import build_config_yaml, build_dopt_config_json, build_run_script


def _config(tmp_path: Path) -> dopt.DoptConfig:
    return dopt.DoptConfig(
        model_path=tmp_path / "model",
        dopt_libs=tmp_path / "dopt_pytorch_py3",
        dataset="dataset.json",
    )


def test_config_yaml_renders_expected_keys(tmp_path: Path) -> None:
    text = build_config_yaml(_config(tmp_path))
    assert "train_files: dataset.json" in text
    assert "quant_param_2: false" in text
    assert "embedding_separate: true" in text
    assert "fp16: true" in text
    assert "Qwen3DecoderLayer" in text
    assert "learning_rate: !!float 1e-4" in text


def test_config_yaml_quant_param_2_flag(tmp_path: Path) -> None:
    cfg = dopt.DoptConfig(
        model_path=tmp_path / "model",
        dopt_libs=tmp_path / "dopt",
        quant_param_2=True,
    )
    assert "quant_param_2: true" in build_config_yaml(cfg)


def test_run_script_contains_stage_dispatch(tmp_path: Path) -> None:
    script = build_run_script(_config(tmp_path))
    assert "opt_main.py" in script
    assert "--quant-stage $quant_stage" in script
    assert "quant_stage=$1" in script
    assert "block_size=128" in script
    assert "CUDA_VISIBLE_DEVICES=0" in script
    assert "DEVICE=cuda" in script


def test_stage_command_runs_run_sh(tmp_path: Path) -> None:
    cmd = dopt.build_dopt_stage_command(_config(tmp_path), "stage2")
    assert cmd == ("bash", "-c", "sh run.sh stage2")


def test_dopt_config_json_uses_recommended_strategies() -> None:
    text = build_dopt_config_json()
    assert "Quant_Embed_MinMax" in text
    assert "Quant_act_weight_eco" in text
    assert "Quant_lm_head" in text
    assert '"bit": 4' in text
    assert '"group_size": 64' in text
