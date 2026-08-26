from pathlib import Path

from hllm.pipeline.executor import ExecutionContext


def test_render_preserves_literal_braces() -> None:
    context = ExecutionContext(Path("/tmp/model/{x}"), Path("/tmp/work"), Path("/tmp/out"), "kirin9020", "int4")
    assert context.render("${model}/config/{x}") == "/tmp/model/{x}/config/{x}"


def test_render_supports_legacy_placeholders() -> None:
    context = ExecutionContext(Path("/tmp/model"), Path("/tmp/work"), Path("/tmp/out"), "kirin9020", "int4")
    assert context.render("{work}/{target}/{quant}") == "/tmp/work/kirin9020/int4"
