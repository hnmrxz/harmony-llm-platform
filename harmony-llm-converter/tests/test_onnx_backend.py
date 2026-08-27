from pathlib import Path

import onnx

from hllm.backends.onnx import _normalize_external_data_metadata, audit_onnx, normalize_onnx_node_names


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
    report = audit_onnx(path, expected_opset=11, require_static=True)
    assert report["ok"]
    assert report["opset"] == 11
    assert report["dynamic_inputs"] == []


def test_audit_rejects_wrong_opset(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    _minimal_model(path, opset=17)
    report = audit_onnx(path, expected_opset=11, require_static=True)
    assert not report["ok"]
    assert any("opset=17" in error for error in report["errors"])


def test_audit_external_data_by_model_path(tmp_path: Path) -> None:
    from onnx import TensorProto, helper

    path = tmp_path / "model.onnx"
    data_path = tmp_path / "weights.bin"
    data_path.write_bytes((1.0).hex().encode())

    tensor = helper.make_tensor("w", TensorProto.FLOAT, [1], [1.0])
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key = "location"
    entry.value = data_path.name
    entry = tensor.external_data.add()
    entry.key = "length"
    entry.value = "4"
    entry = tensor.external_data.add()
    entry.key = "offset"
    entry.value = "0"
    node = helper.make_node("Identity", ["w"], ["y"], name="identity")
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    model = helper.make_model(
        helper.make_graph([node], "g", [], [y], initializer=[tensor]),
        opset_imports=[helper.make_operatorsetid("", 11)],
    )
    onnx.save(model, path)

    # The audit/checker must validate external data relative to the ONNX file,
    # not against an unloaded ModelProto passed directly to check_model().
    report = audit_onnx(path)
    assert report["ok"]
    assert report["external_locations"] == [data_path.name]


def test_audit_rejects_missing_external_data(tmp_path: Path) -> None:
    from onnx import TensorProto, helper

    path = tmp_path / "model.onnx"
    tensor = helper.make_tensor("w", TensorProto.FLOAT, [1], [1.0])
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key = "location"
    entry.value = "missing.bin"
    node = helper.make_node("Identity", ["w"], ["y"], name="identity")
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    model = helper.make_model(
        helper.make_graph([node], "g", [], [y], initializer=[tensor]),
        opset_imports=[helper.make_operatorsetid("", 11)],
    )
    onnx.save(model, path)

    report = audit_onnx(path)
    assert not report["ok"]
    assert any("ONNX_EXTERNAL_DATA_MISSING" in error for error in report["errors"])


def test_normalize_external_data_adds_explicit_range(tmp_path: Path) -> None:
    from onnx import TensorProto, helper

    path = tmp_path / "model.onnx"
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"12345678")
    tensor = helper.make_tensor("w", TensorProto.FLOAT, [2], [1.0, 2.0])
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key = "location"
    entry.value = weights.name
    model = helper.make_model(
        helper.make_graph([], "g", [], [], initializer=[tensor]),
        opset_imports=[helper.make_operatorsetid("", 11)],
    )
    onnx.save(model, path)

    result = _normalize_external_data_metadata(path)
    assert result["changed"] == 2
    normalized = onnx.load(path, load_external_data=False)
    metadata = {entry.key: entry.value for entry in normalized.graph.initializer[0].external_data}
    assert metadata["location"] == weights.name
    assert metadata["offset"] == "0"
    assert metadata["length"] == str(weights.stat().st_size)
    assert weights.read_bytes() == b"12345678"


def test_normalize_onnx_node_names_makes_names_short_unique(tmp_path: Path) -> None:
    from onnx import TensorProto, helper

    path = tmp_path / "model.onnx"
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
    nodes = [
        helper.make_node("Identity", ["x"], ["y"], name="/wrapped/model/Identity_0"),
        helper.make_node("Identity", ["y"], ["z"], name="/wrapped/model/Identity_0"),
        helper.make_node("Identity", ["z"], ["out"], name=""),
    ]
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])
    model = helper.make_model(
        helper.make_graph(nodes, "g", [x], [out]),
        opset_imports=[helper.make_operatorsetid("", 11)],
    )
    onnx.save(model, path)

    result = normalize_onnx_node_names(path)
    assert result["changed"] == 3

    normalized = onnx.load(path, load_external_data=False)
    names = [node.name for node in normalized.graph.node]
    assert names == ["n000000_Identity", "n000001_Identity", "n000002_Identity"]
    assert len(names) == len(set(names))
    assert [list(node.input) for node in normalized.graph.node] == [["x"], ["y"], ["z"]]
    assert [list(node.output) for node in normalized.graph.node] == [["y"], ["z"], ["out"]]
