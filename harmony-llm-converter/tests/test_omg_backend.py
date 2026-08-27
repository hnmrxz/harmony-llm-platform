from pathlib import Path

import pytest

from hllm.backends import omg
from hllm.backends.omg import build_omg_command, prepare_omg_output
from hllm.config import BuildProfile


def _profile() -> BuildProfile:
    return BuildProfile(
        model_source="Qwen/Qwen3-0.6B",
        model_family="qwen3",
        input_quantization=None,
        preferred_path=None,
        fallback_path=None,
        target_chip="KirinX90",
        platform="kirinx90",
        runtime_version=None,
        quantization="int4",
        bits=4,
        context_length=8192,
        output_dir=Path("build"),
    )


def test_prepare_omg_output_leaves_new_prefix_absent(tmp_path: Path) -> None:
    output = prepare_omg_output(tmp_path / "out" / "model")
    assert output.is_absolute()
    assert output.parent.is_dir()
    assert not output.exists()


def test_prepare_omg_output_removes_stale_file(tmp_path: Path) -> None:
    output = tmp_path / "out" / "model"
    output.parent.mkdir(parents=True)
    output.write_text("stale", encoding="utf-8")

    prepared = prepare_omg_output(output)
    assert prepared == output.resolve()
    assert not prepared.exists()


def test_prepare_omg_output_rejects_directory(tmp_path: Path) -> None:
    output = tmp_path / "out" / "model"
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="file prefix"):
        prepare_omg_output(output)


def test_build_omg_command_normalizes_model_before_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        omg,
        "_normalize_external_data_metadata",
        lambda path: calls.append(("external", path)) or {"changed": 0},
    )
    monkeypatch.setattr(
        omg,
        "normalize_onnx_node_names",
        lambda path: calls.append(("names", path)) or {"changed": 0},
    )
    monkeypatch.setattr(
        omg,
        "make_kirin_omg_compatible",
        lambda path: calls.append(("kirin", path)) or {"fold": {}, "gather": {}, "unsqueeze_expand": {}},
    )

    command = build_omg_command(_profile(), model=model, output=tmp_path / "out" / "model")

    assert calls == [
        ("external", model.resolve()),
        ("names", model.resolve()),
        ("kirin", model.resolve()),
    ]
    assert command[0] == "omg"
    assert "--framework=5" in command
    assert "--target=omc" in command
    assert "--platform=kirinx90" in command
    output_arg = next(arg for arg in command if arg.startswith("--output="))
    assert Path(output_arg.split("=", 1)[1]).is_absolute()
    assert not Path(output_arg.split("=", 1)[1]).exists()
