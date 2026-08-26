# HarmonyOS LLM Converter

Ubuntu 端的大语言模型生产工具链。

本项目负责将 Hugging Face 或本地的大语言模型，经过检测、适配、量化、ONNX 导出、CANN 转换和验证后，生成可以直接交付 HarmonyOS Runtime 使用的最终 `.hllm` 模型包。

> **本项目只负责模型生产，不负责 HarmonyOS 端 UI 和最终设备推理。**

## Target Model Range

第一阶段的能力目标不是只支持一个固定模型，而是：

> **Qwen/Qwen3.8-27B 及参数规模更小、且经过 CANN Kit 验证的 Qwen/LLM 模型。**

Qwen3.8-27B 当前 Hugging Face 模型是 27B 参数、原生 262,144 context，并且是带视觉能力的模型；其仓库权重约 55.6 GB。citeturn0search4turn0search5

因此 Converter 必须从第一天就支持 **分片权重**，不能把“单个 safetensors 文件”作为模型假设。

对于 Qwen3.8-27B，首个可交付目标建议定义为：

```text
Qwen/Qwen3.8-27B
        ↓
Hugging Face snapshot
        ↓
metadata + shard inventory
        ↓
CANN Kit 4-bit quantization
        ↓
ONNX export
        ↓
CANN target conversion
        ↓
validation
        ↓
.hllm
```

> **注意：27B 的最终可运行性不是由“模型大小”单独决定，而是由目标 HarmonyOS 设备的 NPU、内存、CANN Runtime/Kit 版本、算子支持和量化结果共同决定。** HarmonyOS 的 Neural Network Runtime 对 NPU 有硬件依赖，且不支持模拟器。citeturn0search3

## Core Goal

建立一条稳定、可复现、可扩展的模型生产流水线：

```text
Hugging Face / Local Model
        ↓
Download / Import
        ↓
Verify
        ↓
Inspect
        ↓
Model Adapter
        ↓
Quantize
        ↓
ONNX Export
        ↓
CANN Convert
        ↓
Validate
        ↓
Package
        ↓
.hllm
```

## Responsibilities

### Supported

- Hugging Face 模型下载
- 本地模型导入
- 分片权重识别
- 模型文件校验
- 模型架构识别
- 模型家族 Adapter
- 量化
- ONNX 导出
- CANN 模型转换
- CANN 转换日志采集
- 产物完整性验证
- 基础推理验证
- `.hllm` 模型包生成
- CLI
- 后续 Web UI

### Not Supported Here

- HarmonyOS ArkUI
- HarmonyOS App 业务逻辑
- 最终设备模型安装
- 用户聊天 UI
- 设备侧 NPU 推理

## Recommended Environment

目标环境为 Ubuntu。具体 Python、PyTorch、Transformers、ONNX 和 CANN 版本必须通过实际目标设备/Kit 版本建立兼容矩阵后固定。

**重要：当前华为 CANN Kit 的 LLM 4-bit 量化流程要求量化阶段运行在 GPU/CUDA 环境。** 因此“Ubuntu”是操作系统要求，不意味着普通无 GPU 的 Ubuntu 主机就能完成 27B 量化。citeturn0search1

建议项目提供环境检查命令，在执行转换前检查：

```text
OS
Python
CUDA / GPU
PyTorch
Transformers
ONNX / ONNX Runtime
CANN Kit
CANN toolkit / compiler tools
available disk space
available memory
```

转换器不得静默使用不兼容版本；版本信息必须进入构建记录。

## Project Layout

```text
harmony-llm-converter/
├── README.md
├── hllm/
│   ├── cli/
│   ├── download/
│   ├── models/
│   │   └── adapters/
│   ├── quantization/
│   ├── export/
│   ├── cann/
│   ├── validation/
│   ├── packaging/
│   └── schema/
├── configs/
├── scripts/
├── tests/
├── docs/
└── pyproject.toml
```

## CLI Direction

第一版 CLI 建议支持以下命令：

```bash
hllm doctor
hllm download Qwen/Qwen3.8-27B
hllm inspect ./Qwen3.8-27B
hllm quantize ./Qwen3.8-27B --method cann-4bit
hllm export ./Qwen3.8-27B --format onnx
hllm convert ./Qwen3.8-27B --target <chip>
hllm validate ./build/model
hllm package ./build/model --output ./dist/model.hllm
```

一键流程目标：

```bash
hllm build Qwen/Qwen3.8-27B \
  --target <chip> \
  --quant cann-4bit \
  --output ./dist
```

一键流程必须保持阶段化日志，不能把所有失败都包装成一个模糊的 `build failed`。

## Model Adapter Architecture

模型适配器是本项目的核心扩展机制：

```text
Model Detector
      ↓
Model Registry
      ↓
Adapter
 ┌────┼────┬────┐
Qwen Llama DeepSeek GLM ...
      ↓
Normalized Model Pipeline
```

Adapter 应负责处理模型家族差异，例如：

- architecture detection
- config normalization
- tokenizer
- chat template
- multimodal capability
- ONNX export requirements
- quantization requirements
- CANN conversion requirements

转换 Pipeline 不应针对单个模型写大量条件分支。

### Qwen3.8 特别说明

当前 Qwen3.8-27B 的 Hugging Face `config.json` 使用 `qwen3_5` model type，并以 `Qwen3_5ForConditionalGeneration` architecture 描述，同时存在 `text_config`。因此检测器不能只读取顶层 `max_position_embeddings`、`dtype` 等字段。项目已经增加 nested `text_config` 处理和 Qwen3.5/Qwen3.8 识别路径。citeturn0search10

它还是多模态模型，因此 **文本 LLM 路径与视觉/视频路径必须分开验证**；不能因为语言模型部分能转换，就宣称完整 VLM 已经支持。

## Conversion Pipeline

### 1. Download

优先支持 Hugging Face Hub，并保留本地模型输入能力。

下载阶段需要记录：

- repository ID
- revision / commit
- 下载文件
- 文件大小
- SHA256（适用时）

### 2. Inspect

读取并分析：

- architecture
- model type
- nested text config
- parameter count
- dtype
- tokenizer
- chat template
- context length
- multimodal capability
- weight index / shard count
- required files

### 3. Quantization

首个目标使用 **CANN Kit 的 4-bit LLM 量化流程**，而不是自行发明一套“看起来像 INT4”的算法。

华为当前文档给出的流程为三个阶段：

```text
Stage 1: Weight Quantization
        ↓
Stage 2: Activation Quantization
        ↓
Stage 3: Quantization Parameter Extraction
```

其产物包括 `trained_quant_weight.pth`、`trained.pth`、`fake_quant_weight.pth` 以及量化参数；其中 `fake_quant_weight.pth` 用于 ONNX 导出。citeturn0search0turn0search1

当前官方参数示例还包含：

```text
ptq_samples: 1024
cutoff_len: 128
num_samples: 256
embedding_separate: true
quant_param_2: hardware-dependent
```

其中 `quant_param_2` 与目标硬件有关，因此 Converter **绝不能固定写死**；必须由 target profile 决定。citeturn0search0

### 4. ONNX Export

将经过适配和量化参数处理的模型导出为符合 CANN 转换要求的中间产物。

导出过程必须保存：

- export config
- input/output metadata
- dynamic shape information
- opset
- export logs

### 5. CANN Conversion

CANN 转换是最终设备模型生产阶段。

所有目标芯片相关配置必须显式记录：

```text
target chip
runtime version
compiler/tool version
operator compatibility
shape constraints
```

不要在代码中猜测某个芯片名称或固定 `atc` 参数。CANN backend 使用显式 target profile 和 argv command runner。

### 6. Validation

至少验证：

```text
Artifact exists
Checksum
Manifest
Tokenizer
Model metadata
Expected input/output
Basic inference
```

如果转换环境具备对应运行条件，应执行最小推理测试。

对于 27B，建议增加：

```text
FP/BF16 reference output
↓
Quantized simulation output
↓
Converted model output
```

进行固定 calibration prompt 的回归比较。

### 7. Packaging

生成统一 `.hllm` 包。

建议结构：

```text
model.hllm
├── manifest.json
├── model/
├── tokenizer/
├── config/
└── integrity/
```

具体 CANN 模型文件扩展名不应成为上层 API 的固定约束，由 `manifest.json` 的 artifact 描述实际文件。

## 27B Storage Planning

Qwen3.8-27B 当前仓库权重约 55.6 GB。理论上纯 4-bit 权重下限约为 13.5 GB（27B × 4 bit），但真实部署包还需要量化 scale/zero-point、embedding、lm_head、runtime metadata、可能的额外权重以及安装临时空间，因此**不能把 13.5 GB 当作实际最终模型大小**。citeturn0search4

Converter 应在开始构建前计算：

```text
source size
+ temporary quantization workspace
+ ONNX workspace
+ CANN conversion workspace
+ final artifact
+ safety margin
```

然后拒绝明显不具备足够磁盘空间的构建。

## HLLM Manifest

Manifest 是 Converter 与 HarmonyOS Runtime 之间的正式合同。

至少应描述：

```json
{
  "schema_version": "1.0",
  "model": {},
  "quantization": {},
  "target": {},
  "runtime": {},
  "tokenizer": {},
  "artifacts": {},
  "build": {}
}
```

其中 `build` 应包含转换器和关键依赖版本，以支持问题定位和复现。

## Build Reproducibility

每次构建都应产生独立 build record，例如：

```text
build-info.json
```

至少记录：

- converter version
- git commit
- Python version
- CUDA version
- PyTorch version
- Transformers version
- ONNX version
- CANN version
- target chip
- quantization parameters
- context length
- source model revision
- timestamp

## Output Contract

成功构建后的唯一交付物是：

```text
<model>-<target>-<quantization>.hllm
```

HarmonyOS Runtime 不应该依赖 Converter 的工作目录、Python 环境或临时文件。

## Failure Handling

每个阶段必须产生明确错误码和日志：

```text
DOWNLOAD_FAILED
MODEL_INVALID
UNSUPPORTED_ARCHITECTURE
QUANTIZATION_FAILED
ONNX_EXPORT_FAILED
CANN_CONVERSION_FAILED
VALIDATION_FAILED
PACKAGE_FAILED
INSUFFICIENT_DISK
UNSUPPORTED_TARGET
UNSUPPORTED_OPERATOR
```

失败后保留中间产物和日志，方便开发者定位问题；生产模式可以提供自动清理选项。

## Roadmap

### Phase 1 — Qwen3.8-27B / Smaller LLM

- Ubuntu environment doctor
- Hugging Face sharded download
- Qwen3/Qwen3.5/Qwen3.8 metadata adapter
- CANN 4-bit three-stage quantization integration
- ONNX export
- target-specific CANN conversion
- `.hllm` package
- automated validation

### Phase 2 — Model Ecosystem

- Qwen family expansion
- Llama
- DeepSeek
- GLM
- more CANN-supported architectures

### Phase 3 — Conversion Infrastructure

- batch conversion
- conversion queue
- Remote Worker
- model cache
- build reuse
- benchmark
- target compatibility database

### Phase 4 — Multimodal

- Qwen3.8 vision path
- video input path
- multimodal tokenizer/processor assets
- HarmonyOS multimodal Runtime contract

## Important Constraint

**不要把 HarmonyOS Runtime 的实现细节反向耦合到模型转换器。**

Converter 的目标只有一个：

> 生产一个符合公开 HLLM Package Contract、并声明明确硬件/运行时要求的可部署模型包。

只要 `.hllm` 合同稳定，Converter 内部实现可以持续演进。