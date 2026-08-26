import ast
from pathlib import Path


def test_huggingface_downloader_is_valid_python() -> None:
    path = Path(__file__).parents[1] / "src/hllm/download/huggingface.py"
    ast.parse(path.read_text(encoding="utf-8"))
