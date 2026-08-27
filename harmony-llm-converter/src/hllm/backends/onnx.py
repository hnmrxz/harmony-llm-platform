"""ONNX export helpers for decoder-only Qwen models.

The exporter uses a conservative legacy Torch ONNX path because the converter
supports Python environments whose PyTorch versions predate the newer exporter.
Qwen is forced onto an eager-attention, FP32, no-cache graph to avoid complex
intermediate values that the legacy exporter cannot lower reliably.
"""
from __future__ import annotations

from pathlib import Path


def export_qwen_onnx(model_dir: Path, output: Path, *, opset: int = 17) -> Path:
    """Export a local Qwen causal-LM to a deployable ONNX graph."""
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("PyTorch and Transformers are required for ONNX export") from exc

    model_dir = model_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    load_kwargs = {
        "torch_dtype": torch.float32,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            attn_implementation="eager",
            **load_kwargs,
        )
    except TypeError:
        # Compatibility with Transformers releases that do not expose
        # attn_implementation on from_pretrained.
        model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
    model = model.float()
    model.eval()

    device = next(model.parameters()).device
    input_ids = torch.ones((1, 4), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(4, device=device, dtype=torch.long).unsqueeze(0)

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
            return self.wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=False,
            )[0]

    wrapper = LogitsWrapper(model)
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (input_ids, attention_mask, position_ids),
                str(output),
                input_names=["input_ids", "attention_mask", "position_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "position_ids": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                opset_version=opset,
                dynamo=False,
            )
    except RuntimeError as exc:
        message = str(exc)
        if "ComplexDouble" in message or "complex" in message.lower():
            raise RuntimeError(
                "ONNX_EXPORT_UNSUPPORTED: legacy Torch ONNX exporter encountered "
                "a complex tensor in the Qwen graph even with eager/FP32/no-cache "
                "export. Upgrade the export backend or use a validated pre-converted "
                "ONNX model for this Transformers/PyTorch combination."
            ) from exc
        raise
    return output
