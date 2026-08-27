"""Build the official HarmonyOS CANN OMG conversion command.

The CANN LLM solution converts a NPU-friendly ONNX into a device .omc by
calling OMG with a rich, graph-aware parameter set:

    omg --framework=5 --model=<onnx> --output=<prefix> \
        --compress_conf=<quant_params_file> \
        --input_shape="input_embed:1,-1,<vocab>;attention_mask:1,1,-1,<ctx>;...
                       past_key_inN:<kv>,2,1,<head_dim>;...;new_kv_cache_pos:-1;
                       embed_scales:1,-1,1" \
        --dynamic_dims="1,1,1,1,1;64,64,64,64,64" \
        --input_type="past_key_inN:FP16;..." \
        --output_type="lm_logits:FP32;past_keyN:FP16;..." \
        --save_weights_as_external_data=true \
        --platform=<platform> --target=omc

The parameter values depend on the geometry of the model graph produced by the
NPU-tuned export (tensor names, KV-cache dimensions, head/layer counts). This
module derives everything possible from the model layout; the exact tensor
names and static KV-cache length are configurable because they must match the
specific exported graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hllm.models.layout import ModelLayout


@dataclass(frozen=True, slots=True)
class OmgSpec:
    model: Path
    output: Path
    layout: ModelLayout
    platform: str
    quant_params_file: Path | None = None
    framework: int = 5
    target: str = "omc"
    kv_cache_max_len: int | None = None
    input_embed_name: str = "input_embed"
    logits_name: str = "lm_logits"
    kv_time_dim: int = 2  # official past_key/past_value shape uses 2 here
    dynamic_dims: tuple[int, int] = (1, 64)  # (min, max) for the dynamic inputs
    kv_dtype: str = "FP16"
    logits_dtype: str = "FP32"
    save_weights_as_external_data: bool = True

    @property
    def num_layers(self) -> int:
        return self.layout.num_hidden_layers

    @property
    def head_dim(self) -> int:
        return self.layout.num_attention_head_dims

    @property
    def cache_len(self) -> int:
        return int(self.kv_cache_max_len or self.layout.kv_cache_max_len)


def _join_entries(entries: list[str]) -> str:
    return ";".join(entries)


def _input_shape(spec: OmgSpec) -> str:
    layout = spec.layout
    entries: list[str] = [
        f"{spec.input_embed_name}:1,-1,{layout.vocab_size}",
        f"attention_mask:1,1,-1,{spec.cache_len}",
        f"position_ids:1,-1,",
    ]
    for i in range(spec.num_layers):
        entries.append(f"past_key_in{i}:{spec.cache_len},{spec.kv_time_dim},1,{spec.head_dim}")
    for i in range(spec.num_layers):
        entries.append(f"past_value_in{i}:{spec.cache_len},{spec.kv_time_dim},1,{spec.head_dim}")
    entries.append("new_kv_cache_pos:-1")
    entries.append("embed_scales:1,-1,1")
    return _join_entries(entries)


def _dynamic_dims(spec: OmgSpec) -> str:
    # Five dynamic inputs: input_embed, attention_mask, position_ids,
    # new_kv_cache_pos, embed_scales (each with (min, max) sequence dims).
    lo, hi = spec.dynamic_dims
    return f"{lo},{lo},{lo},{lo},{lo};{hi},{hi},{hi},{hi},{hi}"


def _input_types(spec: OmgSpec) -> str:
    entries: list[str] = []
    for i in range(spec.num_layers):
        entries.append(f"past_key_in{i}:{spec.kv_dtype}")
        entries.append(f"past_value_in{i}:{spec.kv_dtype}")
    return _join_entries(entries)


def _output_types(spec: OmgSpec) -> str:
    entries: list[str] = [f"{spec.logits_name}:{spec.logits_dtype}"]
    for i in range(spec.num_layers):
        entries.append(f"past_key{i}:{spec.kv_dtype}")
        entries.append(f"past_value{i}:{spec.kv_dtype}")
    return _join_entries(entries)


def build_official_omg_command(spec: OmgSpec) -> tuple[str, ...]:
    """Return the full OMG argv for a NPU-friendly ONNX -> .omc conversion.

    `spec.model` must be absolute; `spec.output` is used verbatim as the OMG
    output prefix. No external-data mutation is performed here.
    """
    model = spec.model.expanduser().resolve()
    output = spec.output.expanduser().resolve()
    if output.exists():
        if output.is_dir():
            raise ValueError(f"OMG output must be a file prefix, not a directory: {output}")
        output.unlink()

    argv = [
        "omg",
        f"--model={model}",
        f"--framework={spec.framework}",
        f"--output={output}",
        f"--target={spec.target}",
        f"--platform={spec.platform}",
        f"--input_shape={_input_shape(spec)}",
        f"--dynamic_dims={_dynamic_dims(spec)}",
        f"--input_type={_input_types(spec)}",
        f"--output_type={_output_types(spec)}",
    ]
    if spec.quant_params_file is not None:
        argv.append(f"--compress_conf={spec.quant_params_file.expanduser().resolve()}")
    if spec.save_weights_as_external_data:
        argv.append("--save_weights_as_external_data=true")
    return tuple(argv)
