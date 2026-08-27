from pathlib import Path

import pytest

from hllm.backends.cann_omg import OmgSpec, build_official_omg_command
from hllm.models.layout import ModelLayout


def _layout() -> ModelLayout:
    return ModelLayout(
        hidden_size=3584,
        num_hidden_layers=4,
        num_attention_heads=28,
        num_attention_kv_heads=4,
        num_attention_head_dims=128,
        vocab_size=152064,
        max_position_embeddings=32768,
    )


def _spec(tmp_path: Path) -> OmgSpec:
    return OmgSpec(
        model=tmp_path / "model.onnx",
        output=tmp_path / "out" / "model",
        layout=_layout(),
        platform="kirinx90",
        quant_params_file=tmp_path / "quant_params_file",
    )


def test_build_official_omg_command_has_core_flags(tmp_path: Path) -> None:
    argv = build_official_omg_command(_spec(tmp_path))
    text = " ".join(argv)
    assert argv[0] == "omg"
    assert "--framework=5" in argv
    assert "--target=omc" in argv
    assert "--platform=kirinx90" in argv
    assert "--save_weights_as_external_data=true" in argv
    assert any(a.startswith("--compress_conf=") for a in argv)
    assert any(a.startswith("--input_shape=") for a in argv)
    assert any(a.startswith("--dynamic_dims=") for a in argv)
    assert any(a.startswith("--input_type=") for a in argv)
    assert any(a.startswith("--output_type=") for a in argv)


def test_input_shape_has_one_kv_pair_per_layer(tmp_path: Path) -> None:
    argv = build_official_omg_command(_spec(tmp_path))
    input_shape = next(a.split("=", 1)[1] for a in argv if a.startswith("--input_shape="))
    # 4 layers -> 4 past_key_in + 4 past_value_in entries.
    assert input_shape.count("past_key_in") == 4
    assert input_shape.count("past_value_in") == 4
    assert "new_kv_cache_pos:-1" in input_shape
    assert "embed_scales:1,-1,1" in input_shape


def test_output_type_has_logits_and_past_per_layer(tmp_path: Path) -> None:
    argv = build_official_omg_command(_spec(tmp_path))
    output_type = next(a.split("=", 1)[1] for a in argv if a.startswith("--output_type="))
    assert "lm_logits:FP32" in output_type
    assert output_type.count("past_key") == 4
    assert output_type.count("past_value") == 4


def test_dynamic_dims_fixed_to_five(tmp_path: Path) -> None:
    argv = build_official_omg_command(_spec(tmp_path))
    dynamic = next(a.split("=", 1)[1] for a in argv if a.startswith("--dynamic_dims="))
    assert dynamic == "1,1,1,1,1;64,64,64,64,64"


def test_official_omg_rejects_directory_output(tmp_path: Path) -> None:
    output = tmp_path / "out" / "model"
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="file prefix"):
        build_official_omg_command(_spec(tmp_path))


def test_no_compress_conf_when_absent(tmp_path: Path) -> None:
    spec = OmgSpec(
        model=tmp_path / "model.onnx",
        output=tmp_path / "out" / "model",
        layout=_layout(),
        platform="kirinx90",
        quant_params_file=None,
    )
    argv = build_official_omg_command(spec)
    assert not any(a.startswith("--compress_conf=") for a in argv)
