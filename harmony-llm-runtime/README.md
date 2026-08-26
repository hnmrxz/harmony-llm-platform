# HarmonyOS LLM Runtime

运行在 HarmonyOS 电脑、平板及后续兼容设备上的本地大语言模型运行与管理应用。

本项目是 `harmony-llm-platform` 的模型消费端，只接收已经由 Ubuntu Converter 生成的 `.hllm` 模型包。

> **本项目不负责模型转换。HarmonyOS 端只负责模型导入、验证、安装、管理和本地推理。**

## Core Goal

让用户能够把一个已经完成转换的 `.hllm` 模型文件导入 HarmonyOS 设备，然后直接使用设备侧 CANN/NPU 完成本地大语言模型推理。

```text
.hllm
  ↓
Import
  ↓
Integrity Check
  ↓
Manifest Parse
  ↓
Device Compatibility Check
  ↓
Install
  ↓
Tokenizer
  ↓
CANN / NPU Runtime
  ↓
Streaming Inference
  ↓
Chat UI
```

## Strict Boundary

HarmonyOS Runtime **不包含**：

- Hugging Face 下载
- Transformers
- PyTorch 模型转换
- 模型量化
- ONNX 导出
- CANN 模型转换/编译
- 为模型转换而部署完整 Python 环境

这些职责全部属于 `harmony-llm-converter/`。

Runtime 唯一关心的是：

```text
给我一个 .hllm
→ 我判断能不能在当前设备运行
→ 能就安装
→ 然后调用本地 NPU 推理
```

## Supported Model Input

第一版只接受统一模型包：

```text
*.hllm
```

不建议直接让 Runtime 暴露以下底层文件作为普通用户入口：

```text
*.onnx
*.om
*.omc
*.bin
*.safetensors
```

如果用户只有这些文件，应先通过 Ubuntu Converter 生成 `.hllm`。

## Runtime Architecture

```text
┌──────────────────────────────────────┐
│ HarmonyOS UI / ArkUI                 │
│                                      │
│ Home / Models / Import / Chat        │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Application Services                 │
│                                      │
│ ModelService / RuntimeService        │
│ DeviceService / TransferService      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Native Runtime Layer                 │
│                                      │
│ ModelLoader / Tokenizer / Sampler    │
│ KV Cache / Inference                 │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ CANN / Hardware Backend              │
└──────────────────┬───────────────────┘
                   │
                   ▼
                 NPU
```

ArkTS 负责界面和应用业务；Native/C++ 层负责模型加载、Tokenizer、推理以及与底层 AI Runtime 的连接。

## Model Import

第一阶段支持：

### Local Import

```text
选择 .hllm
   ↓
复制/安装到模型目录
```

### LAN Transfer

后续可以让 Ubuntu Converter 提供局域网服务：

```text
Ubuntu
  │
  │ LAN
  ▼
HarmonyOS Runtime
```

用户可以直接从 Ubuntu 的模型列表选择已经生成的模型，无需手工复制文件。

### HTTP/HTTPS

后续可以扩展远程模型服务器，但必须经过完整性校验后才能安装。

## Model Installation

导入模型后严格按照以下顺序处理：

```text
1. File exists
2. Archive readable
3. Manifest readable
4. Schema supported
5. Artifact checksum valid
6. Model metadata valid
7. Tokenizer files valid
8. Device compatibility valid
9. Storage capacity valid
10. Install
```

任何一步失败，都不能进入运行状态。

## Device Compatibility

Runtime 必须先获取设备能力信息，再判断模型是否可以安装/运行。

建议抽象为：

```text
DeviceProfile
├── HarmonyOS version
├── chip
├── NPU availability
├── memory
├── storage
├── CANN/runtime version
└── supported model formats
```

模型 Manifest 则声明：

```text
ModelRequirements
├── backend
├── target chip
├── runtime version
├── minimum memory
├── context length
└── artifact format
```

然后执行：

```text
DeviceProfile
      ×
ModelRequirements
      ↓
Compatibility Result
```

结果至少分为：

```text
COMPATIBLE
COMPATIBLE_WITH_LIMITS
INCOMPATIBLE
```

对于不兼容模型，应明确告诉用户原因，而不是只显示“模型无法运行”。

## Model Manager

模型中心负责：

- 已安装模型
- 模型详情
- 模型大小
- 量化方式
- 目标芯片
- 上下文长度
- Runtime/CANN 要求
- 启动/停止
- 删除
- 升级
- 重新验证

建议模型状态：

```text
IMPORTED
VALIDATING
INSTALLED
READY
RUNNING
ERROR
INCOMPATIBLE
```

## Tokenizer

Tokenizer 是 Runtime 的一等组件。

模型包必须包含运行所需的 tokenizer 与相关配置，并保留模型自己的 Chat Template 信息。

推理链路：

```text
User Message
    ↓
Chat Template
    ↓
Tokenizer
    ↓
Input IDs
    ↓
CANN/NPU
    ↓
Token IDs
    ↓
Tokenizer Decode
    ↓
Streaming Text
```

Runtime 不应该假设所有模型使用相同的 Prompt 格式。

## Inference Runtime

第一阶段重点实现：

- 单会话聊天
- 流式 token 输出
- temperature
- top-p
- max tokens
- stop conditions
- context management
- KV cache
- cancel generation

推理层应该保持与 UI 解耦：

```text
Chat UI
   ↓
Inference API
   ↓
Runtime
   ↓
CANN
```

## Security and Integrity

模型包安装前必须验证：

- SHA256 / Manifest integrity
- Artifact presence
- Manifest schema
- Model metadata

后续应支持模型包签名，例如：

```text
.hllm
  ↓
Signature Verify
  ↓
Trusted Publisher
  ↓
Install
```

不应默认执行模型包中任何脚本或可执行文件。

## Storage Management

模型可能非常大，因此 Model Manager 必须具备：

- 可用空间检测
- 安装前空间预估
- 临时文件管理
- 原子安装
- 失败回滚
- 模型删除
- 缓存清理

不要在模型尚未完成校验时覆盖现有可用模型。

## Error Model

Runtime 应提供结构化错误，例如：

```text
PACKAGE_INVALID
MANIFEST_UNSUPPORTED
CHECKSUM_FAILED
TOKENIZER_INVALID
DEVICE_UNSUPPORTED
CANN_VERSION_UNSUPPORTED
INSUFFICIENT_MEMORY
INSUFFICIENT_STORAGE
MODEL_LOAD_FAILED
INFERENCE_FAILED
```

用户界面同时显示人类可读的解释和必要的诊断信息。

## Recommended UI

### Home

```text
HarmonyOS LLM

Installed Models

Qwen3-1.7B INT4
✓ Ready

[Chat]

[Import Model]
```

### Import

```text
Import HarmonyOS Model

[ Select .hllm ]

Model: Qwen3-1.7B
Target: Kirin ...
Quantization: INT4

Compatibility: Checking...

[Install]
```

### Model Detail

```text
Qwen3-1.7B

INT4
Target: ...
Context: 8192

Device: Compatible
Storage: OK
Runtime: OK

[Start]
[Delete]
```

### Chat

```text
Model: Qwen3-1.7B INT4

User: 你好

Assistant: 你好！……

[Stop]
```

## Project Layout

```text
harmony-llm-runtime/
├── README.md
├── entry/
├── ets/
│   ├── pages/
│   ├── components/
│   └── services/
├── native/
│   ├── runtime/
│   └── backends/
│       └── cann/
├── resources/
└── tests/
```

实际 HarmonyOS 工程结构以当前 SDK/DevEco Studio 工程模板和目标 HarmonyOS API 版本为准。

## Roadmap

### Phase 1 — Runtime MVP

- `.hllm` 导入
- Manifest 解析
- SHA256 校验
- 设备能力检测
- 模型安装
- Tokenizer
- CANN/NPU 基础推理
- 流式聊天

### Phase 2 — Model Management

- 模型列表
- 模型详情
- 删除/升级
- 存储管理
- 失败回滚
- 更完整的兼容性诊断

### Phase 3 — Transfer

- Ubuntu Converter 局域网发现
- LAN 模型传输
- HTTP/HTTPS 模型服务器
- 断点续传

### Phase 4 — Runtime Ecosystem

- 多模型并存
- 多会话
- Benchmark
- 更多模型类型
- VLM / Embedding / Reranker 等运行时能力

## Compatibility Contract

本项目与 Ubuntu Converter 的唯一核心耦合点是：

> **HLLM Package Contract**

Converter 可以持续更换量化、ONNX、CANN 转换实现；Runtime 可以持续更换 UI 和内部运行时实现，只要双方遵守同一模型包规范，就可以独立演进。

## Important Constraint

不要为了支持“任意 Hugging Face 模型”而把模型转换逻辑塞入 HarmonyOS App。

正确的边界是：

```text
Ubuntu:
任意受支持的模型
        ↓
最终可部署 .hllm

HarmonyOS:
.hllm
        ↓
验证
        ↓
运行
```

这样可以保持 HarmonyOS App 体积、依赖和运行环境可控，同时让模型转换链路可以在 Ubuntu 上持续迭代。