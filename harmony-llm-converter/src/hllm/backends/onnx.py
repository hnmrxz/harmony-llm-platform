"""ONNX export helpers for decoder-only Qwen models.

The exporter is intentionally conservative: it exports a forward graph for
logits with dynamic batch/sequence dimensions. Vendor-specific OMC conversion
can then consume the resulting ONNX file.
"""
from __future__ import annotations

from pathlib import Path


def export_qwen_onnx(model_dir: Path, output: Path, *, opset: int = 17) -> Path:
    """Export a local Qwen causal-LM to ONNX.

    ``optimum`` is not required. The implementation uses the Transformers model
    and ``torch.onnx.export``. Large models should normally be exported on a
    GPU-equipped conversion host; this function is primarily the generic
    integration point used by the build pipeline.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("PyTorch and Transformers are required for ONNX export") from exc

    model_dir = model_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    device = next(model.parameters()).device
    input_ids = torch.ones((1, 4), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    class LogitsWrapper(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return self.wrapped(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    wrapper = LogitsWrapper(model)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(output),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=opset,
            dynamo=False,
        )
    return output
