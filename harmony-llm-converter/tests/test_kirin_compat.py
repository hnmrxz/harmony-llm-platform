"""Unit tests for KirinX90 OMG graph-compatibility rewrites."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from hllm.backends.kirin_compat import (
    KirinCompatError,
    _retype_gather_indices,
    _rewrite_unsqueeze_expand,
    _static_fold,
    make_kirin_omg_compatible,
)


def _build_model(nodes, inputs, outputs, inits=()):
    g = helper.make_graph(list(nodes), "t", list(inputs), list(outputs), list(inits))
    m = helper.make_model(g, opset_imports=[helper.make_operatorsetid("", 11)])
    m.ir_version = 6
    return m


def _vi(name, elem, dims):
    return helper.make_tensor_value_info(name, elem, dims)


def test_static_fold_materializes_mask_cone_and_drops_input() -> None:
    """COS fed by Constant folds away; attention_mask dropped once unconsumed."""
    nodes = [
        helper.make_node("Constant", [], ["cshape"],
                         value=numpy_helper.from_array(np.array([2], dtype=np.int64))),
        helper.make_node("ConstantOfShape", ["cshape"], ["cos_out"],
                         value=numpy_helper.from_array(np.array([True], dtype=bool))),
        helper.make_node("Cast", ["attention_mask"], ["mask_u8"], to=TensorProto.UINT8),
        helper.make_node("Identity", ["x"], ["y"]),
        helper.make_node("Identity", ["cos_out"], ["cos_y"]),
    ]
    model = _build_model(
        nodes,
        [_vi("attention_mask", TensorProto.INT64, [2]), _vi("x", TensorProto.FLOAT, [])],
        [_vi("y", TensorProto.FLOAT, []), _vi("cos_y", TensorProto.BOOL, [2])],
    )
    stats = _static_fold(model)
    g = model.graph
    ops = {n.op_type for n in g.node}
    assert "ConstantOfShape" not in ops
    # attention_mask is fully consumed by the fold and disappears from inputs.
    assert [v.name for v in g.input] == ["x"]
    init_by_name = {i.name: i for i in g.initializer}
    folded = numpy_helper.to_array(init_by_name["cos_y"])
    assert folded.dtype == np.bool_
    assert bool(folded.reshape(-1)[0]) is True
    assert stats["folded_nodes"] >= 3
    assert stats["materialized"] >= 1


def test_static_fold_keeps_models_without_feeds() -> None:
    """Without a declared static feed nothing is folded."""
    node = helper.make_node("Cast", ["ids"], ["u8"], to=TensorProto.UINT8)
    model = _build_model(
        [node],
        [_vi("ids", TensorProto.INT64, [1, 4])],
        [_vi("u8", TensorProto.UINT8, [1, 4])],
    )
    stats = _static_fold(model)
    assert stats["folded_nodes"] == 0
    assert len(model.graph.node) == 1
    assert [v.name for v in model.graph.input] == ["ids"]


def test_retype_gather_indices_shares_cast_and_keeps_other_consumers() -> None:
    idx_src = helper.make_node("Identity", ["ids"], ["idx"])
    g1 = helper.make_node("Gather", ["w", "idx"], ["o1"], axis=0)
    add = helper.make_node("Add", ["idx", "idx"], ["summed"])
    g2 = helper.make_node("Gather", ["w", "idx"], ["o2"], axis=0)
    model = _build_model(
        [idx_src, g1, add, g2],
        [
            _vi("w", TensorProto.FLOAT, [8, 4]),
            _vi("ids", TensorProto.INT64, [2]),
        ],
        [
            _vi("o1", TensorProto.FLOAT, [2, 4]),
            _vi("o2", TensorProto.FLOAT, [2, 4]),
            _vi("summed", TensorProto.INT64, [2]),
        ],
        [numpy_helper.from_array(np.zeros((8, 4), dtype=np.float32), name="w")],
    )
    stats = _retype_gather_indices(model)
    assert stats["gather_rewired"] == 2
    casts = [n for n in model.graph.node if n.op_type == "Cast"]
    assert len(casts) == 1
    assert casts[0].output[0] == "idx_hllm_i32"
    gathers = [n for n in model.graph.node if n.op_type == "Gather"]
    assert all(n.input[1] == "idx_hllm_i32" for n in gathers)
    add_node = next(n for n in model.graph.node if n.op_type == "Add")
    assert add_node.input == ["idx", "idx"]


def test_rewrite_unsqueeze_expand_with_inference() -> None:
    unsq = helper.make_node("Unsqueeze", ["vec"], ["mat"], axes=[0], name="unsq")
    exp = helper.make_node("Expand", ["scalar", "shape_3"], ["row"], name="exp")
    model = _build_model(
        [unsq, exp],
        [_vi("vec", TensorProto.FLOAT, [4]), _vi("scalar", TensorProto.FLOAT, [])],
        [_vi("mat", TensorProto.FLOAT, [1, 4]), _vi("row", TensorProto.FLOAT, [3])],
        [
            numpy_helper.from_array(np.array(1.5, dtype=np.float32), name="scalar"),
            numpy_helper.from_array(np.array([3], dtype=np.int64), name="shape_3"),
        ],
    )
    stats = _rewrite_unsqueeze_expand(model)
    assert stats["unsqueeze"] == 1
    assert stats["expand"] == 1
    ops = {n.op_type for n in model.graph.node}
    assert ops == {"Reshape", "Mul"}
    inits = {i.name: i for i in model.graph.initializer}
    shapes = [numpy_helper.to_array(v) for k, v in inits.items() if k.endswith("_shape")]
    ones = [numpy_helper.to_array(v) for k, v in inits.items() if k.endswith("_ones")]
    assert any(np.array_equal(s.reshape(-1), np.array([1, 4])) for s in shapes)
    assert any(np.array_equal(o.reshape(-1), np.ones(3, dtype=np.float32)) for o in ones)


def test_make_compatible_roundtrip_preserves_externals(tmp_path: Path) -> None:
    data_file = tmp_path / "weights.bin"
    weights = np.arange(8, dtype=np.float32)
    data_file.write_bytes(weights.tobytes())

    t = onnx.TensorProto()
    t.name = "w"
    t.data_type = TensorProto.FLOAT
    t.dims.extend([8])
    t.data_location = onnx.TensorProto.EXTERNAL
    entry = t.external_data.add()
    entry.key = "location"
    entry.value = data_file.name

    cast = helper.make_node("Cast", ["ids"], ["ids32"], to=TensorProto.INT32)
    gather = helper.make_node("Gather", ["w", "ids32"], ["out"], axis=0)
    model = _build_model(
        [cast, gather],
        [_vi("ids", TensorProto.INT64, [1, 2])],
        [_vi("out", TensorProto.FLOAT, [1, 2, 8])],
        [t],
    )
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    before = data_file.read_bytes()

    report = make_kirin_omg_compatible(path)

    assert data_file.read_bytes() == before
    reloaded = onnx.load(str(path), load_external_data=False)
    ext_init = next(i for i in reloaded.graph.initializer if i.name == "w")
    assert ext_init.external_data, "external reference must be preserved"
    assert set(report) == {"fold", "gather", "unsqueeze_expand"}
    onnx.checker.check_model(str(path))


def test_make_compatible_flags_dangling_graph(tmp_path: Path) -> None:
    broken = tmp_path / "broken.onnx"
    node = helper.make_node("Identity", ["ghost"], ["out"])
    model = _build_model([node], [], [_vi("out", TensorProto.FLOAT, [2])])
    onnx.save(model, str(broken))

    # The dangling input is outside any foldable cone; our post-pass validator
    # must reject the result instead of writing a corrupt file.
    with pytest.raises((KirinCompatError, ValueError)):
        make_kirin_omg_compatible(broken)
