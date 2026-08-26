# HarmonyOS LLM Platform

面向 HarmonyOS 电脑、平板及后续鸿蒙设备的本地大语言模型转换、分发与推理平台。

本仓库采用 **模型生产端与模型消费端彻底分离** 的架构：

- `harmony-llm-converter/`：运行在 Ubuntu，负责从 Hugging Face 或本地模型开始，完成模型检测、量化、ONNX 导出、CANN 转换、验证和最终模型包生成。
- `harmony-llm-runtime/`：运行在 HarmonyOS 设备，负责导入模型包、兼容性检查、安装、Tokenizer、CANN/NPU 推理和用户交互。

> 核心原则：**Ubuntu 负责生产模型，HarmonyOS 负责消费模型。**

## Architecture

```text
Hugging Face / Local Model
          │
          ▼
┌──────────────────────────────┐
│ Ubuntu LLM Converter         │
│                              │
│ Download → Inspect           │
│ → Quantize → ONNX            │
│ → CANN Convert → Validate    │
│ → Package                    │
└──────────────┬───────────────┘
               │
               ▼
        HarmonyOS LLM Package
              (.hllm)
               │
               ▼
┌──────────────────────────────┐
│ HarmonyOS LLM Runtime        │
│                              │
│ Import → Verify → Install    │
│ → Tokenizer → Inference      │
│ → CANN / NPU → Chat          │
└──────────────────────────────┘
```

## Repository Structure

```text
harmony-llm-platform/
├── README.md
├── harmony-llm-converter/
│   └── README.md
└── harmony-llm-runtime/
    └── README.md
```

## Design Principles

1. **转换与运行解耦**：HarmonyOS 端不承担 Hugging Face 下载、PyTorch、量化、ONNX 导出和 CANN 转换。
2. **统一模型包**：两端通过稳定的 `.hllm` 模型包格式和 `manifest.json` 交互。
3. **设备能力驱动**：模型包声明目标芯片、CANN 后端、精度、上下文长度等要求，HarmonyOS 端在安装前执行兼容性检查。
4. **可扩展模型适配器**：模型转换器通过 Adapter 支持 Qwen、Llama、DeepSeek、GLM 等模型家族，而不是把模型逻辑硬编码进主流程。
5. **可复现构建**：转换产物记录 Converter、CANN、Python、PyTorch、Transformers 等版本及构建参数。
6. **运行时轻量化**：HarmonyOS 端只携带运行所需组件，不依赖完整 Python/PyTorch/Transformers 转换环境。

## Target Workflow

### Model Production

```text
Model Source
  → Download / Import
  → Inspect
  → Model Adapter
  → Quantization
  → ONNX Export
  → CANN Conversion
  → Validation
  → HLLM Packaging
```

### Device Deployment

```text
.hllm
  → Import
  → Integrity Check
  → Manifest Parse
  → Device Compatibility Check
  → Install
  → Load Runtime
  → Tokenize
  → CANN/NPU Inference
  → Stream Output
```

## Scope

### Converter

- Hugging Face 模型下载
- 本地模型导入
- 模型架构识别
- 模型 Adapter
- 量化
- ONNX 导出
- CANN 模型转换
- 转换日志与构建信息
- 模型完整性验证
- `.hllm` 打包
- CLI / Web UI

### Runtime

- `.hllm` 文件导入
- 本地/局域网模型传输
- Manifest 解析
- 设备能力检测
- 模型兼容性检查
- 模型安装、删除、升级
- Tokenizer 与 Chat Template
- CANN/NPU 推理
- 流式文本输出
- 本地模型管理

## Non-Goals

当前阶段 HarmonyOS Runtime **不负责**：

- Hugging Face 模型下载
- PyTorch 模型转换
- 模型量化
- ONNX 导出
- CANN 编译/转换
- 在设备端建立完整 Python 模型转换环境

## Development Roadmap

### Phase 1 — End-to-End MVP

- Qwen3 小型模型
- INT4
- Ubuntu 转换链路
- `.hllm` 第一版
- HarmonyOS 导入
- CANN/NPU 基础推理
- 流式聊天

### Phase 2 — Model Ecosystem

- Qwen 系列扩展
- Llama
- DeepSeek
- GLM
- 更多量化策略
- 更多目标芯片

### Phase 3 — Distribution

- 局域网模型传输
- 模型仓库
- 远程转换 Worker
- 模型版本管理
- Benchmark
- 自动兼容性分析

### Phase 4 — AI Runtime Platform

逐步扩展到 VLM、Embedding、Reranker、ASR、TTS 和 Agent 等端侧 AI 工作负载。

## Project Status

当前仓库处于 **架构初始化阶段**。README、模型包协议和两个子项目的职责边界先行固定，代码实现按端到端 MVP 推进。

## License

License 将在项目进入正式代码开发阶段后确定。