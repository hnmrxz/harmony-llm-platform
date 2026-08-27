"""Official CANN LLM dopt three-stage quantization wrapper.

The CANN LLM solution quantizes a PyTorch model on a GPU in three stages
(`run.sh stage1 | stage2 | stage3`):

    stage1  weight quantization      -> dopt_config.json + trained_quant_weight.pth
    stage2  activation quantization   -> trained.pth
    stage3  quant-param extraction    -> fake_quant_weight.pth + quant_params_file
                                          + embedding_weights + embedding_quant_scale

This module models the official `config.yaml` + `run.sh` scaffolding and the
per-stage command so the converter can drive the toolchain without guessing
flags. It does not execute the GPU toolchain itself; the build pipeline runs
the generated stage commands.

The `dopt_config.json` that stage1 emits needs a manual quant-strategy edit
(decode `Quant_act_weight_eco`, lm_head `Quant_lm_head`, embedding
`Quant_Embed_MinMax`); see `build_dopt_config()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DoptStage = Literal["stage1", "stage2", "stage3"]


@dataclass(frozen=True, slots=True)
class DoptConfig:
    model_path: Path
    dopt_libs: Path
    device: str = "cuda"
    cuda_visible_devices: str = "0"
    testcase: str = "output_dir"
    block_size: int = 128
    dataset: str = "dataset.json"
    train_samples: int = 1024
    ptq_samples: int = 1024
    cutoff_len: int = 128
    calibration_samples: int = 256
    quant_param_2: bool = False
    embedding_separate: bool = True
    fp16: bool = True
    no_split_module_classes: tuple[str, ...] = (
        "Qwen3DecoderLayer",
        "Qwen2DecoderLayer",
        "GlmDecoderLayer",
        "LlamaDecoderLayer",
        "HunYuanDecoderLayer",
    )

    @property
    def output_dir(self) -> Path:
        return self.testcase / "train_output"  # relative; mirror the official script


@dataclass(frozen=True, slots=True)
class DoptArtifacts:
    """The files each stage is expected to produce (used for bookkeeping)."""

    stage1: tuple[str, ...] = ("dopt_config.json", "trained_quant_weight.pth")
    stage2: tuple[str, ...] = ("trained.pth",)
    stage3: tuple[str, ...] = (
        "fake_quant_weight.pth",
        "quant_params_file",
        "embedding_weights",
        "embedding_quant_scale",
    )


def build_config_yaml(config: DoptConfig) -> str:
    """Render the official `config.yaml` used by `opt_main.py`."""
    no_split = "\n".join(f"    - {cls}" for cls in config.no_split_module_classes)
    return f"""\
kd:
  enable: False
  loss: mse
  micro_batch_size: 2
  gradient_accumulation_steps: 4
  weight_decay: 0.0
  warmup_steps: 10
  num_epochs: 3
  learning_rate: !!float 1e-4
  eval_step: 1
  logging_step: 50
  lr_scheduler_type: cosine
  trainable_keys:
    - quant_alpha
    - norm
  no_split_module_classes:
{no_split}
dataset:
  train_files: {config.dataset}
  train_samples: {config.train_samples}
  ptq_samples: {config.ptq_samples}

extra_training_config:
  fp16: {str(config.fp16).lower()}

cutoff_len: {config.cutoff_len}
num_samples: {config.calibration_samples}
quant_param_2: {str(config.quant_param_2).lower()}
embedding_separate: {str(config.embedding_separate).lower()}
lm_head_size:
"""


def build_run_script(config: DoptConfig) -> str:
    """Render the official `run.sh` that dispatches the three stages."""
    return f"""\
#!/bin/bash
qlibs='{config.dopt_libs}'
export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=0
export PYTHONPATH=${{qlibs}}:$PYTHONPATH

export DEVICE={config.device}
export CUDA_VISIBLE_DEVICES={config.cuda_visible_devices}

ROOT=.
testcase='{config.testcase}'
RUN_FILE=${{qlibs}}/dopt/dopt_lm/opt_main.py
output_dir=${{ROOT}}/${{testcase}}/train_output
mkdir -p ${{output_dir}}
cp ${{ROOT}}/config.yaml ${{output_dir}}

model_path='{config.model_path}'
dopt_config=./${{testcase}}/dopt_config.json
quant_stage=$1
block_size={config.block_size}

python -u \\
    ${{RUN_FILE}} --model-path ${{model_path}} \\
    --dopt-config ${{dopt_config}} \\
    --optimize-config ${{ROOT}}/config.yaml \\
    --quant-stage $quant_stage \\
    --block-size ${{block_size}} \\
    --output-dir ${{output_dir}} 2>&1 | tee ${{output_dir}}/logs.log
"""


def build_dopt_stage_command(config: DoptConfig, stage: DoptStage) -> tuple[str, ...]:
    """Return the `bash run.sh <stage>` argv with the surrounding env."""
    return ("bash", "-c", f"sh run.sh {stage}")


def build_dopt_config_json(
    *,
    embed_quant_strategy: str = "Quant_Embed_MinMax",
    decoder_quant_strategy: str = "Quant_act_weight_eco",
    lm_head_quant_strategy: str = "Quant_lm_head",
    weight_bits: int = 4,
    weight_group_size: int = 64,
    input_bits: int = 16,
) -> str:
    """Render a recommended `dopt_config.json` for the user to copy stage1's
    output and edit `quant_strategy`. This is the recommended configuration from
    the official guide (decode/lm_head/embedding strategies, 4-bit weights,
    group 64-128, 16-bit activations)."""
    return f"""\
{{
    "layer_stategy": {{
        "model.embed_tokens": {{
            "type": "<class 'torch.nn.modules.sparse.Embedding'>",
            "quant_strategy": "{embed_quant_strategy}"
        }},
        "model.layers.0.self_attn.q_proj": {{
            "type": "<class 'torch.nn.modules.linear.Linear'>",
            "quant_strategy": "{decoder_quant_strategy}",
            "weight" : {{
                "bit": {weight_bits},
                "group_size": {weight_group_size}
            }},
            "input": {{
                "bit": {input_bits}
            }}
        }},
        "lm_head": {{
            "type": "<class 'torch.nn.modules.linear.Linear'>",
            "quant_strategy": "{lm_head_quant_strategy}"
        }}
    }}
}}
"""
