"""Qwen ONNX exporters and CANN-oriented ONNX validation helpers."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _export_with_torch(
    model_dir: Path,
    output: Path,
    *,
    opset: int,
    batch_size: int,
    sequence_length: int,
    precision: str,
    external_data: bool,
) -> Path:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("PyTorch and Transformers are required for ONNX export") from exc

    kwargs: dict[str, Any] = {
        "dtype": torch.float32 if precision == "fp32" else "auto",
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    if precision == "fp32":
        model = model.float()

    device = next(model.parameters()).device
    input_ids = torch.ones((batch_size, sequence_length), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(sequence_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    class LogitsWrapper(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
        ) -> torch.Tensor:
            out = self.wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=False,
            )
            return out[0]

    wrapper = LogitsWrapper(model)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, position_ids),
            str(output),
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
            external_data=external_data,
        )
    return output


def _simplify_static(path: Path) -> None:
    """Simplify only self-contained ONNX; never rewrite unloaded external-data tensors."""
    try:
        import onnx
        from onnxsim import simplify
    except ImportError as exc:
        raise RuntimeError("static export with inline weights requires onnxsim") from exc
    model = onnx.load(str(path), load_external_data=True)
    simplified, ok = simplify(model, perform_optimization=True, skip_fuse_bn=True)
    if not ok:
        raise RuntimeError("ONNX_SIMPLIFY_FAILED: onnxsim returned check=false")
    onnx.save_model(simplified, str(path), save_as_external_data=False)


def _set_ir_version_preserving_external_data(path: Path, ir_version: int | None) -> None:
    """Change only ModelProto metadata without dereferencing external tensor payloads."""
    if ir_version is None:
        return
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    model.ir_version = ir_version
    path.write_bytes(model.SerializeToString())


def _collect_external_locations(path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    locations: list[str] = []
    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location":
                locations.append(entry.value)
                break
    return sorted(set(locations))


def _validate_external_files(path: Path) -> None:
    locations = _collect_external_locations(path)
    root = path.parent.resolve()
    missing: list[str] = []
    for location in locations:
        candidate = (root / location).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"ONNX_EXTERNAL_DATA_ESCAPE: {location}") from exc
        if not candidate.is_file():
            missing.append(location)
    if missing:
        raise RuntimeError(f"ONNX_EXTERNAL_DATA_MISSING: {missing}")


def export_qwen_onnx(
    model_dir: Path,
    output: Path,
    *,
    mode: str = "generic",
    opset: int = 17,
    ir_version: int | None = None,
    batch_size: int = 1,
    sequence_length: int = 4,
    precision: str = "auto",
    external_data: bool = True,
) -> Path:
    """Export Qwen for diagnosis or a conservative CANN static smoke test."""
    model_dir = model_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if mode not in {"generic", "cann_static"}:
        raise ValueError(f"unsupported ONNX export mode: {mode}")
    if batch_size < 1 or sequence_length < 1:
        raise ValueError("batch_size and sequence_length must be positive")
    if precision not in {"auto", "fp32"}:
        raise ValueError(f"unsupported export precision: {precision}")
    if mode == "cann_static" and opset < 11:
        raise ValueError("cann_static requires opset >= 11")

    _export_with_torch(
        model_dir,
        output,
        opset=opset,
        batch_size=batch_size,
        sequence_length=sequence_length,
        precision=precision,
        external_data=external_data,
    )

    if mode == "cann_static":
        if external_data:
            # Do not run onnxsim or onnx.save on a model containing unloaded external tensors.
            # Preserve every exporter-created weight file byte-for-byte.
            _validate_external_files(output)
            _set_ir_version_preserving_external_data(output, ir_version)
        else:
            _simplify_static(output)
            _set_ir_version_preserving_external_data(output, ir_version)
    return output


def audit_onnx(
    path: str | Path,
    *,
    expected_opset: int | None = None,
    expected_ir: int | None = None,
    require_static: bool = False,
) -> dict[str, Any]:
    """Audit ONNX metadata and external data without loading tensor payloads in Python."""
    import onnx

    model_path = Path(path).expanduser().resolve()
    model = onnx.load(str(model_path), load_external_data=False)

    # Validate external-data references ourselves first. Then ask ONNX checker to
    # resolve the model from its path so it can locate external payloads relative
    # to the protobuf file. Passing an unloaded ModelProto to check_model() is
    # invalid for external-data models and raises the exact "should be stored in
    # ... but it is not regular file" ValidationError seen in production.
    _validate_external_files(model_path)
    onnx.checker.check_model(str(model_path), full_check=False)

    dynamic_inputs: list[str] = []
    for value in model.graph.input:
        if value.type.HasField("tensor_type") and value.type.tensor_type.HasField("shape"):
            if any(dim.dim_param for dim in value.type.tensor_type.shape.dim):
                dynamic_inputs.append(value.name)

    opset = next((item.version for item in model.opset_import if item.domain == ""), None)
    node_counts = Counter(node.op_type for node in model.graph.node)
    external = sum(bool(initializer.external_data) for initializer in model.graph.initializer)
    external_locations = _collect_external_locations(model_path)

    errors: list[str] = []
    if expected_opset is not None and opset != expected_opset:
        errors.append(f"opset={opset}, expected={expected_opset}")
    if expected_ir is not None and model.ir_version != expected_ir:
        errors.append(f"ir_version={model.ir_version}, expected={expected_ir}")
    if require_static and dynamic_inputs:
        errors.append(f"dynamic_inputs={dynamic_inputs}")

    return {
        "path": str(model_path),
        "ir_version": model.ir_version,
        "opset": opset,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "external_initializers": external,
        "dynamic_inputs": dynamic_inputs,
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
        "op_types": dict(node_counts),
        "external_locations": external_locations,
        "ok": not errors,
        "errors": errors,
    }


def write_audit(path: str | Path, audit: dict[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
