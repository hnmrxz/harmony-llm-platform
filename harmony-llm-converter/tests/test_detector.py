import json
from pathlib import Path

from hllm.models.detector import inspect_model


def test_detector_handles_nested_text_config_and_fp8(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"torch_dtype": "float8_e4m3fn", "max_position_embeddings": 32768},
            "vision_config": {"hidden_size": 1024},
        }),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    metadata = inspect_model(tmp_path)
    assert metadata.dtype == "fp8_e4m3fn"
    assert metadata.is_fp8
    assert metadata.context_length == 32768
    assert metadata.is_multimodal
    assert metadata.has_chat_template
