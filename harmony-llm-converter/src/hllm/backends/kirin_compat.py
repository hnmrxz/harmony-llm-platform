"""Graph-compatibility rewrites for the KirinX90 legacy OMG parser.

The DDK 6.0.x ``libai_fmk_onnx_parser.so`` rejects several patterns emitted by
torch.onnx.export. Empirically confirmed against the shipped binary:

1. Any ``ConstantOfShape`` fed by a Constant aborts ONNX import inside
   ``UpdateUserSetNodeNames`` ("opIndex != context.tensorOperator.end()"),
   regardless of node naming.
2. GE ``GatherV2D`` refuses INT64 index tensors (INT32 required).
3. The NPU front-end kernel store lacks ``ExpandDims``/``BroadcastTo``, so the
   model-compatibility check fails on every Unsqueeze/Expand.

For ``cann_static`` exports every tensor shape is a compile-time constant, so
all three patterns can be rewritten value-identically:

1. Everything reachable from declared static feeds (currently only the all-ones
   ``attention_mask``) is evaluated with NumPy and materialized as initializers;
   the folded graph input disappears when nothing else consumes it. The result
   is an omc that always behaves like an unpadded prefill — which matches the
   documented Prefill-only smoke-test scope of ``cann_static``.
2. A single Cast-to-INT32 is inserted per INT64 index tensor before every
   Gather that consumes it; unrelated consumers keep seeing the original
   INT64 tensor.
3. Unsqueeze becomes Reshape with a constant shape vector and Expand becomes a
   Mul against constant ones, both derived from shape inference.

All passes are metadata-only: models are loaded with
``load_external_data=False`` and re-serialized without touching external weight
bytes, matching the other OMG-boundary normalizations in this package.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

FeedBuilder = Callable[[list[int]], "np.ndarray"]

_STATIC_FEEDS: dict[str, FeedBuilder] = {}


def _ones_int64(dims: list[int]) -> "np.ndarray":
    return np.ones(dims, dtype=np.int64)


_STATIC_FEEDS["attention_mask"] = _ones_int64


class KirinCompatError(RuntimeError):
    """Raised when a rewrite pass would leave the graph structurally invalid."""


_FOLDABLE_OPS = {
    "Cast", "Constant", "ConstantOfShape", "Shape", "Gather", "Flatten",
    "Mul", "Add", "Sub", "Reshape", "Concat", "Where", "And", "Or",
    "Equal", "Expand", "Unsqueeze", "Squeeze", "Slice", "Neg",
    "Transpose", "Identity",
}

_CAST_TARGETS = {
    TensorProto.FLOAT: np.float32,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT8: np.int8,
    TensorProto.UINT16: np.uint16,
    TensorProto.INT16: np.int16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.BOOL: np.bool_,
    TensorProto.FLOAT16: np.float16,
}


def _attr_tensor(node: onnx.NodeProto, name: str):
    for attr in node.attribute:
        if attr.name == name and attr.type == onnx.AttributeProto.TENSOR:
            return numpy_helper.to_array(attr.t)
    return None


def _evaluate(node: onnx.NodeProto, inputs: list):
    op = node.op_type
    if op == "Cast":
        to = next(a.i for a in node.attribute if a.name == "to")
        return [inputs[0].astype(_CAST_TARGETS[to])]
    if op == "Constant":
        value = _attr_tensor(node, "value")
        return [value] if value is not None else None
    if op == "ConstantOfShape":
        value = _attr_tensor(node, "value")
        fill = value.reshape(-1)[0] if value is not None and value.size else np.float32(0)
        dtype = value.dtype if value is not None else np.float32
        dims = [int(x) for x in inputs[0].reshape(-1)]
        return [np.full(dims, fill, dtype=dtype)]
    if op == "Shape":
        return [np.array(inputs[0].shape, dtype=np.int64)]
    if op == "Gather":
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
        return [np.take(inputs[0], inputs[1], axis=axis)]
    if op == "Flatten":
        axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
        x = inputs[0]
        lead = int(np.prod(x.shape[:axis])) if axis > 0 else x.size
        return [x.reshape(lead, -1)]
    if op == "Mul":
        return [inputs[0] * inputs[1]]
    if op == "Add":
        return [inputs[0] + inputs[1]]
    if op == "Sub":
        return [inputs[0] - inputs[1]]
    if op == "Neg":
        return [-inputs[0]]
    if op == "Reshape":
        return [inputs[0].reshape([int(x) for x in inputs[1].reshape(-1)])]
    if op == "Concat":
        axis = next(a.i for a in node.attribute if a.name == "axis")
        return [np.concatenate(list(inputs), axis=axis)]
    if op == "Where":
        return [np.where(inputs[0], inputs[1], inputs[2])]
    if op == "And":
        return [np.logical_and(inputs[0], inputs[1])]
    if op == "Or":
        return [np.logical_or(inputs[0], inputs[1])]
    if op == "Equal":
        return [inputs[0] == inputs[1]]
    if op == "Expand":
        return [np.broadcast_to(inputs[0], [int(x) for x in inputs[1].reshape(-1)])]
    if op == "Unsqueeze":
        axes = next(a.ints for a in node.attribute if a.name == "axes")
        return [np.expand_dims(inputs[0], list(axes))]
    if op == "Squeeze":
        axes_attr = next((a.ints for a in node.attribute if a.name == "axes"), None)
        return [np.squeeze(inputs[0], tuple(axes_attr) if axes_attr is not None else None)]
    if op == "Slice":
        data = inputs[0]
        starts = inputs[1].reshape(-1)
        ends = inputs[2].reshape(-1)
        axes = inputs[3].reshape(-1)
        steps = inputs[4].reshape(-1) if len(inputs) > 4 else np.ones_like(starts)
        key = [slice(None)] * data.ndim
        for start, end, axis, step in zip(starts, ends, axes, steps):
            key[int(axis)] = slice(int(start), int(end), int(step))
        return [data[tuple(key)]]
    if op == "Transpose":
        perm = next((a.ints for a in node.attribute if a.name == "perm"), None)
        return [np.transpose(inputs[0], None if perm is None else list(perm))]
    if op == "Identity":
        return [inputs[0]]
    return None


def _static_fold(model: onnx.ModelProto) -> dict:
    """Evaluate feed-rooted constant cones into initializers; drop fed inputs."""
    g = model.graph

    init_map = {i.name: numpy_helper.to_array(i) for i in g.initializer if not i.external_data}
    known = dict(init_map)
    input_vi = {v.name: v for v in g.input}
    for name, build in _STATIC_FEEDS.items():
        vi = input_vi.get(name)
        if vi is None:
            continue
        tt = vi.type.tensor_type
        if not tt.HasField("shape") or any(
            not d.HasField("dim_value") or d.dim_value <= 0 for d in tt.shape.dim
        ):
            # Dynamic export profile: nothing safe to materialize for this feed.
            continue
        dims = [d.dim_value for d in tt.shape.dim]
        known[name] = build(dims)

    stats = {"folded_nodes": 0, "eval_skipped": 0}
    while True:
        replaced = {}
        survivors = []
        for n in g.node:
            if n.op_type in _FOLDABLE_OPS and n.output and all(
                (i in known) for i in n.input if i
            ):
                try:
                    outs = _evaluate(n, [known[i] for i in n.input if i])
                except Exception:  # noqa: BLE001 - leave op for kernel-level handling
                    outs = None
                    stats["eval_skipped"] += 1
                if (
                    outs is not None
                    and len(outs) == len(n.output)
                    and all(v is not None for v in outs)
                ):
                    for out, value in zip(n.output, outs):
                        replaced[out] = value
                    continue
            survivors.append(n)
        removed_now = len(g.node) - len(survivors)
        if not replaced:
            break
        del g.node[:]
        g.node.extend(survivors)
        known.update(replaced)
        stats["folded_nodes"] += removed_now

    used = {i for n in g.node for i in n.input if i} | {v.name for v in g.output}
    existing = {i.name for i in g.initializer}
    added = 0
    for tensor in sorted(used & set(known)):
        if tensor not in existing:
            g.initializer.append(numpy_helper.from_array(known[tensor], name=tensor))
            existing.add(tensor)
            added += 1
    stats["materialized"] = added

    node_inputs = {i for n in g.node for i in n.input if i}
    dropped_feeds = [
        name for name in _STATIC_FEEDS
        if name in input_vi and name not in node_inputs
    ]
    if dropped_feeds:
        kept_inputs = [v for v in g.input if v.name not in dropped_feeds]
        del g.input[:]
        g.input.extend(kept_inputs)
    stats["dropped_feeds"] = dropped_feeds
    return stats


def _retype_gather_indices(model: onnx.ModelProto) -> dict:
    g = model.graph
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False).graph
        dtypes = {
            vi.name: vi.type.tensor_type.elem_type
            for vi in list(inferred.value_info) + list(inferred.input) + list(inferred.output)
        }
        for i in g.initializer:
            dtypes.setdefault(i.name, i.data_type)
    except Exception:  # noqa: BLE001 - fall back to declaration-level knowledge
        dtypes = {i.name: i.data_type for i in g.initializer}
        dtypes.update({v.name: v.type.tensor_type.elem_type for v in g.input})

    casts = {}
    rewired = 0
    new_nodes = []
    for n in g.node:
        if n.op_type == "Gather" and len(n.input) >= 2:
            idx_t = n.input[1]
            if idx_t and dtypes.get(idx_t) == TensorProto.INT64:
                new_name = casts.get(idx_t)
                if new_name is None:
                    new_name = f"{idx_t}_hllm_i32"
                    cast = helper.make_node(
                        "Cast", [idx_t], [new_name],
                        to=TensorProto.INT32, name=f"{new_name}_cast",
                    )
                    new_nodes.append(cast)
                    casts[idx_t] = new_name
                n.input[1] = new_name
                rewired += 1
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    return {"gather_rewired": rewired}


def _rewrite_unsqueeze_expand(model: onnx.ModelProto) -> dict:
    g = model.graph
    inferred = shape_inference.infer_shapes(model, strict_mode=False).graph
    dims_of = {}
    elem_of = {}
    for vi in list(inferred.value_info) + list(inferred.input) + list(inferred.output):
        tt = vi.type.tensor_type
        if tt.HasField("shape") and all(d.HasField("dim_value") for d in tt.shape.dim):
            dims_of[vi.name] = [d.dim_value for d in tt.shape.dim]
            elem_of[vi.name] = tt.elem_type

    taken = {i.name for i in g.initializer}

    def uniq(name: str) -> str:
        base = name
        counter = len(taken)
        while name in taken:
            counter += 1
            name = f"{base}_{counter}"
        taken.add(name)
        return name

    stats = {"unsqueeze": 0, "expand": 0, "skipped_dynamic": 0}
    out_nodes = []
    for n in g.node:
        if n.op_type == "Unsqueeze":
            src = n.input[0] if n.input else ""
            out_t = n.output[0] if n.output else ""
            if src not in dims_of or not out_t:
                stats["skipped_dynamic"] += 1
                out_nodes.append(n)
                continue
            axes = sorted(
                int(a) + (len(dims_of[src]) + 1 if int(a) < 0 else 0)
                for att in n.attribute if att.name == "axes"
                for a in att.ints
            )
            dims = list(dims_of[src])
            for ax in axes:
                dims.insert(ax, 1)
            shape_init = uniq(f"{out_t.strip('/').replace('/', '_')}_shape")
            g.initializer.append(
                numpy_helper.from_array(np.array(dims, dtype=np.int64), name=shape_init)
            )
            out_nodes.append(helper.make_node("Reshape", [src, shape_init], list(n.output)))
            stats["unsqueeze"] += 1
        elif n.op_type == "Expand":
            src = n.input[0] if n.input else ""
            out_t = n.output[0] if n.output else ""
            if src not in dims_of or out_t not in dims_of:
                stats["skipped_dynamic"] += 1
                out_nodes.append(n)
                continue
            ones_init = uniq(f"{out_t.strip('/').replace('/', '_')}_ones")
            dtype = _CAST_TARGETS.get(elem_of.get(src, TensorProto.FLOAT), np.float32)
            g.initializer.append(
                numpy_helper.from_array(np.ones(dims_of[out_t], dtype=dtype), name=ones_init)
            )
            out_nodes.append(helper.make_node("Mul", [src, ones_init], list(n.output)))
            stats["expand"] += 1
        else:
            out_nodes.append(n)
    del g.node[:]
    g.node.extend(out_nodes)
    return stats


def _validate_no_dangling(model: onnx.ModelProto) -> None:
    g = model.graph
    produced = {o for n in g.node for o in n.output if o}
    known = produced | {i.name for i in g.initializer} | {v.name for v in g.input}
    dangling = [
        (n.op_type, i)
        for n in g.node
        for i in n.input
        if i and i not in known
    ]
    if dangling:
        raise KirinCompatError(f"dangling tensors after rewrite: {dangling[:5]}")


def make_kirin_omg_compatible(path) -> dict:
    """Apply all KirinX90 OMG compatibility rewrites to an ONNX file, in place.

    Returns per-pass statistics suitable for build logs. External weight bytes
    are never loaded or rewritten.
    """
    model_path = Path(path).expanduser().resolve()
    model = onnx.load(str(model_path), load_external_data=False)

    fold_stats = _static_fold(model)
    gather_stats = _retype_gather_indices(model)
    expand_stats = _rewrite_unsqueeze_expand(model)
    _validate_no_dangling(model)

    model_path.write_bytes(model.SerializeToString())
    return {
        "fold": fold_stats,
        "gather": gather_stats,
        "unsqueeze_expand": expand_stats,
    }
