from pathlib import Path

import onnx

from hllm.backends.onnx import audit_onnx


def _minimal_model(path: Path, opset: int = 11) -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["x"], ["y"], name="identity")
    model = helper.make_model(
        helper.make_graph([node], "g", [x], [y]),
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    onnx.save(model, path)


def test_audit_static_opset11(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    _minimal_model(path)
    report = audit_onnx(path, expected_opset=11, expected_ir=path and None, require_static=True)
    assert report["ok"]
    assert report["opset"] == 11
    assert report["dynamic_inputs"] == []


def test_audit_rejects_wrong_opset(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    _minimal_model(path, opset=17)
    report = audit_onnx(path, expected_opset=11, require_static=True)
    assert not report["ok"]
    assert any("opset=17" in error for error in report["errors"])
