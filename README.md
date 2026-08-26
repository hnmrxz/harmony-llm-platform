# HarmonyOS LLM Platform

面向 HarmonyOS 电脑、平板及后续鸿蒙设备的本地大语言模型转换、交付与 NPU 推理平台。

## Architecture

```text
Hugging Face / Local Model
          │
          ▼
┌──────────────────────────────┐
│ Ubuntu Converter             │
│ Download / Inspect           │
│ Plan / Quantize / Export     │
│ CANN Convert / Validate      │
│ Package                      │
└──────────────┬───────────────┘
               │
               ▼
             .hllm
               │
               ▼
┌──────────────────────────────┐
│ HarmonyOS Runtime            │
│ Import / Verify / Install    │
│ Tokenizer / Native Runtime   │
│ CANN / NNRt / NPU             │
└──────────────────────────────┘
```

仓库严格保持两个子项目解耦：Ubuntu 负责生产模型，HarmonyOS 只消费已经转换好的 `.hllm`。

## Projects

### `harmony-llm-converter`

Ubuntu 模型生产链：

```text
Hugging Face
 → Download
 → Inspect
 → Resource Plan
 → Model Adapter
 → CANN 4-bit Quantization
 → ONNX Export
 → Target-specific CANN Conversion
 → Validation
 → .hllm Package
```

当前已经具备：

- Hugging Face 下载入口
- Qwen/Qwen3/Qwen3.5/Qwen3.8 家族识别基础
- 分片 safetensors 清单检查
- 大模型磁盘资源预估
- Qwen Adapter Registry
- CANN 三阶段量化命令编排
- ONNX 外部导出命令编排
- CANN 外部转换命令编排
- `.hllm` 打包与 SHA-256 校验
- CLI
- GitHub Actions 基础测试
- Qwen3.8-27B INT4 profile 模板

华为当前 CANN Kit 文档明确提供 LLM 4-bit 三阶段量化流程：weight quantization、activation quantization、quantization parameter extraction，并要求在 GPU/CUDA 环境执行。citeturn565728search0turn565728search4

### `harmony-llm-runtime`

HarmonyOS 模型消费端：

- `.hllm` 导入
- Manifest / SHA-256 校验
- 设备能力检查
- 安装与模型管理
- Tokenizer
- Native inference backend
- CANN/NNRt 离线模型接入边界
- 流式推理接口设计

华为当前文档说明离线模型的加载、编译和推理主要在 native 层实现；NNRt 支持针对特定 AI 硬件的 offline model 推理，但该模型不能跨 AI 硬件直接复用。citeturn624475search0turn624475search2

## HLLM Package

两端唯一核心合同：

```text
.hllm
├── manifest.json
└── model/
    ├── device-specific offline model
    ├── tokenizer/
    └── config/
```

规范文件：

- `docs/hllm-package-spec.md`
- `docs/hllm-manifest.schema.json`

## Target Model Range

项目目标包括：

- Qwen3 / Qwen3.5 / Qwen3.8
- **Qwen/Qwen3.8-27B** 作为当前核心目标
- 27B 以下模型

Qwen3.8-27B 当前 Hugging Face 仓库约为 55.6 GB 权重，并采用分片 safetensors；Converter 因此不允许假设单个权重文件，也不会为了读取 metadata 而完整加载模型。citeturn621287search5

> 实际能否在某一台 HarmonyOS 设备上运行，最终由模型包目标芯片、CANN/NNRt 版本、NPU 内存、上下文长度和实际设备能力共同决定。

## Usage Direction

```bash
hllm doctor
hllm download Qwen/Qwen3.8-27B --output ./models
hllm inspect ./models/Qwen__Qwen3.8-27B
hllm build ./models/Qwen__Qwen3.8-27B \
  --target <validated-kirin-target> \
  --quant int4 \
  --profile ./configs/qwen3.8-27b-int4.example.yaml
```

生产转换必须使用经过实际 CANN Kit / 目标设备验证的 profile。仓库不会凭空硬编码未验证的芯片参数或厂商私有命令行选项。

## Current State

已经进入 **MVP 实现阶段**，不是单纯架构文档仓库：

1. Converter 基础代码已建立。
2. HLLM Package 基础实现已建立。
3. Build Planner / Stage Runner / External Command Executor 已建立。
4. Qwen3.8-27B profile 已建立。
5. HarmonyOS native Runtime backend boundary 已建立。

尚未宣称完成的部分是：

- 指定真实 Kirin 目标芯片上的 CANN 量化成功记录
- 指定真实 HarmonyOS 电脑/平板上的 NPU 端到端验证
- 面向具体 DevEco/CANN SDK 版本的最终 native API 绑定

这些必须在真实 Ubuntu+CANN 和 HarmonyOS 真机环境中验证后才能标记为 PASS。

## Development Principle

不要为了“支持任意 Hugging Face 模型”而把模型转换工具链塞进 HarmonyOS。

```text
Ubuntu:
  Model → Quantize → Export → CANN → .hllm

HarmonyOS:
  .hllm → Verify → Install → Native Runtime → NPU
```
