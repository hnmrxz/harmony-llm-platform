# HarmonyOS LLM Converter

Ubuntu 端的大语言模型生产工具链。

本项目负责将 Hugging Face 或本地的大语言模型，经过检测、适配、量化、ONNX 导出、CANN 转换和验证后，生成可以直接交付 HarmonyOS Runtime 使用的最终 `.hllm` 模型包。

> **本项目只负责模型生产，不负责 HarmonyOS 端 UI 和最终设备推理。**

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

目标环境为 Ubuntu，具体 Python、PyTorch、Transformers、ONNX 和 CANN 版本以实际验证后的兼容矩阵为准。

建议项目提供环境检查命令，在执行转换前检查：

```text
OS
Python
pip / uv / conda
PyTorch
Transformers
ONNX / ONNX Runtime
CANN
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
hllm download Qwen/Qwen3-1.7B
hllm inspect ./Qwen3-1.7B
hllm quantize ./Qwen3-1.7B --method int4
hllm export ./Qwen3-1.7B --format onnx
hllm convert ./Qwen3-1.7B --target <chip>
hllm validate ./build/model
hllm package ./build/model --output ./dist/model.hllm
```

同时提供一键流程：

```bash
hllm build Qwen/Qwen3-1.7B \
  --target <chip> \
  --quant int4 \
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
- ONNX export requirements
- quantization requirements
- CANN conversion requirements

转换 Pipeline 不应针对单个模型写大量条件分支。

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
- parameter count
- dtype
- tokenizer
- chat template
- context length
- required files

### 3. Quantization

第一阶段重点支持 INT4，并将量化实现抽象为独立 backend。

未来可扩展：

- INT8
- BF16/FP16 保持精度
- 不同 group size
- calibration dataset

### 4. ONNX Export

将经过适配的模型导出为符合 CANN 转换要求的中间产物。

导出过程必须保存：

- export config
- input/output metadata
- dynamic shape information
- opset
- export logs

### 5. CANN Conversion

CANN 转换是最终设备模型生产阶段。

所有目标芯片相关配置必须显式记录，不允许使用不可追踪的默认值。

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
```

失败后保留中间产物和日志，方便开发者定位问题；生产模式可以提供自动清理选项。

## Roadmap

### Phase 1

- Ubuntu 环境检测
- Hugging Face 下载
- Qwen3 Adapter
- INT4
- ONNX 导出
- CANN 转换
- `.hllm` 第一版
- 自动验证

### Phase 2

- Qwen 系列扩展
- Llama
- DeepSeek
- GLM
- CLI 完善
- Web UI

### Phase 3

- 批量转换
- 转换队列
- Remote Worker
- 模型缓存
- 构建复用
- Benchmark

### Phase 4

- 多芯片目标
- 更多量化方案
- 模型注册中心
- CI 自动构建

## Important Constraint

**不要把 HarmonyOS Runtime 的实现细节反向耦合到模型转换器。**

Converter 的目标只有一个：

> 生产一个符合公开 HLLM Package Contract、并声明明确硬件/运行时要求的可部署模型包。

只要 `.hllm` 合同稳定，Converter 内部实现可以持续演进。