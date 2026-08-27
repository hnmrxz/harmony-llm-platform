from pathlib import Path

from hllm.backends.npu_tuned_export import (
    NpuTunedExportSpec,
    build_export_command,
    build_export_model_info_yaml,
)


def _spec(tmp_path: Path) -> NpuTunedExportSpec:
    return NpuTunedExportSpec(
        model_dir=tmp_path / "model",
        output_dir=tmp_path / "export",
        model_arch="Qwen3ForCausalLM",
        info_path=tmp_path / "model_info_target.yaml",
    )


def test_yaml_renders_arch_and_paths(tmp_path: Path) -> None:
    yaml = build_export_model_info_yaml(_spec(tmp_path))
    assert "model_arch: Qwen3ForCausalLM" in yaml
    assert str((tmp_path / "model").resolve()) in yaml
    assert str((tmp_path / "export").resolve()) in yaml
    assert "onnx_name: model" in yaml


def test_export_command_uses_qwen_script(tmp_path: Path) -> None:
    command = build_export_command(_spec(tmp_path))
    assert command[0] == "python"
    assert command[1] == "export_model_single_qwen2.py"
    assert f"--info-path={(tmp_path / 'model_info_target.yaml').resolve()}" in command


def test_export_command_uses_overridden_script(tmp_path: Path) -> None:
    spec = NpuTunedExportSpec(
        model_dir=tmp_path / "model",
        output_dir=tmp_path / "export",
        model_arch="OtherModel",
        info_path=tmp_path / "info.yaml",
        export_script="export_model_single_glm.py",
    )
    command = build_export_command(spec)
    assert command[1] == "export_model_single_glm.py"
