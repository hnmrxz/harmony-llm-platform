# HarmonyOS LLM Converter

Ubuntu 端的大语言模型生产工具链。

本项目负责将 Hugging Face 或本地的大语言模型，经过检测、适配、资源规划、量化、ONNX 导出、CANN 转换和验证后，生成可以直接交付 HarmonyOS Runtime 使用的最终 `.hllm` 模型包。

> **本项目只负责模型生产，不负责 HarmonyOS 端 UI 和最终设备推理。**

## 目标模型范围

当前首要目标：

> **Qwen/Qwen3.8-27B 及参数规模更小、且经过目标 CANN/设备验证的模型。**

Qwen3.8-27B 属于大模型/多模态模型，因此第一阶段要优先跑通文本 LLM 路径；视觉/视频能力必须单独验证，不能因为语言模型部分成功就宣称完整 VLM 已支持。

## 一、Ubuntu 环境准备

推荐使用独立虚拟环境：

```bash
cd harmony-llm-converter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[huggingface,quantization,export,dev]'
```

执行环境检查：

```bash
hllm doctor
```

重点确认：

```text
Linux
Python >= 3.10
NVIDIA GPU / CUDA（用于 CANN LLM 量化流程）
PyTorch
Transformers
ONNX / ONNX Runtime
Hugging Face Hub
CANN Kit / toolkit / compiler tools
足够的 RAM
足够的磁盘空间
```

具体 Python、PyTorch、Transformers、ONNX、CANN Kit 和 CUDA 版本必须通过实际目标环境验证后固定。不要因为 `pip install` 成功就认为整个转换链兼容。

## 二、完整转换流程

完整链条：

```text
Hugging Face / Local Model
        ↓
Download / Import
        ↓
Inspect
        ↓
Resource Plan
        ↓
Model Adapter
        ↓
CANN 4-bit Quantization
        ↓
ONNX Export
        ↓
Target-specific CANN Conversion
        ↓
Validation
        ↓
HLLM Packaging
        ↓
.hllm
```

## 三、从 Hugging Face 下载模型

例如：

```bash
hllm download Qwen/Qwen3.8-27B \
  --output ./models
```

建议固定 revision，便于复现：

```bash
hllm download Qwen/Qwen3.8-27B \
  --revision <commit-or-tag> \
  --output ./models
```

下载目录建议保持：

```text
models/
└── Qwen__Qwen3.8-27B/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── model.safetensors.index.json
    ├── model-00001-of-....safetensors
    └── ...
```

不要删除 `index.json`、tokenizer 或 `config.json`，后续模型检测和打包都需要它们。

## 四、模型检测

```bash
hllm inspect ./models/Qwen__Qwen3.8-27B
```

检查至少包含：

- model type
- architecture
- nested `text_config`
- dtype
- context length
- tokenizer
- chat template
- multimodal capability
- weight index
- shard count

Qwen3.8 可能通过 `text_config` 描述真正的文本模型配置，因此不能只读取顶层配置。

### 权重分片检查

大模型通常采用多个 safetensors shard。Converter 应读取：

```text
model.safetensors.index.json
```

而不是假定：

```text
model.safetensors
```

检查重点：

```text
total weight size
shard count
tensor count
```

## 五、资源规划

在量化之前必须做资源预估：

```text
源权重
+ 量化 workspace
+ ONNX workspace
+ CANN workspace
+ 最终模型包
+ 安全余量
```

27B 模型即使采用 INT4，最终模型也不会等于“参数量 × 4bit”的理论下限，因为还会存在量化参数、scale、部分未量化结构、tokenizer、manifest 和转换临时空间。

推荐先准备远大于最终包体的临时磁盘空间，并在生产环境加入独立的 workspace 分区。

## 六、Qwen Adapter

Converter 通过 Model Registry 选择 Adapter：

```text
Detector
  ↓
Registry
  ↓
Qwen Adapter
  ↓
Normalized Model
```

Adapter 负责：

- 架构识别
- 配置归一化
- tokenizer 资产识别
- chat template
- multimodal capability
- ONNX 导出输入
- 量化输入要求
- CANN 目标要求

新增模型时不要在主 Pipeline 中增加大量 `if model == ...`。

## 七、CANN 4-bit 量化

第一阶段使用华为 CANN Kit 的 LLM 4bit 路径，并把它抽象为三个明确 stage：

```text
Stage 1
Weight Quantization
        ↓
Stage 2
Activation Quantization
        ↓
Stage 3
Quantization Parameter Extraction
```

每个 stage 都必须：

- 有独立日志
- 有明确输入输出目录
- 有失败状态
- 有版本记录
- 有产物存在性检查

### Profile 原则

不要在 Python 代码里写死厂商参数。使用 target/build profile：

```text
configs/
└── qwen3.8-27b-int4.example.yaml
```

生产 profile 至少包含：

```yaml
model:
  source: Qwen/Qwen3.8-27B

quantization:
  method: cann_4bit
  bits: 4

cann:
  target_chip: <validated-chip>
  runtime_version: <validated-version>
  profile: <validated-profile>
```

`quant_param_2` 等硬件相关参数必须由 profile 决定，不得全局硬编码。

## 八、ONNX 导出

量化/模拟量化产物进入 ONNX Export Backend。

导出必须记录：

```text
opset
input names
output names
input shapes
output shapes
dynamic dimensions
export tool version
export command
stdout/stderr
```

并将导出目录视为临时 artifact，不直接当作最终 Runtime 输入。

## 九、CANN 目标转换

CANN Backend 接收一个显式 target profile：

```text
Model / ONNX
    ↓
Target Profile
    ├── chip
    ├── CANN version
    ├── runtime version
    ├── compiler command
    └── shape/operator constraints
```

执行失败时，必须保留：

```text
command
exit code
stdout
stderr
working directory
profile
```

不要把 CANN 命令包装成一个无法诊断的 `conversion failed`。

## 十、验证

至少完成：

```text
文件存在
↓
Manifest
↓
SHA-256
↓
Tokenizer
↓
Artifact metadata
↓
目标格式
↓
最小推理验证
```

更高等级的验证：

```text
Reference model
      ↓
Calibration prompts
      ↓
Quantized simulation
      ↓
Converted CANN model
      ↓
Compare outputs
```

对于 27B，建议保存固定 regression prompt 集合，确保每次 CANN/量化 profile 变化后都能比较结果。

## 十一、生成 `.hllm`

最终输出：

```bash
hllm package ./build/artifacts \
  --output ./dist/Qwen3.8-27B-<target>-int4.hllm
```

`.hllm` 内包含：

```text
manifest.json
model/
tokenizer/
config/
integrity/
```

每一个模型 artifact 都应该记录：

```text
path
size
sha256
type
```

## 十二、一键构建

目标命令：

```bash
hllm build ./models/Qwen__Qwen3.8-27B \
  --target <validated-target-chip> \
  --quant int4 \
  --profile ./configs/qwen3.8-27b-int4.example.yaml \
  --output ./dist
```

一键构建应依次输出：

```text
[download]
[inspect]
[plan]
[quantize]
[export]
[cann_convert]
[validate]
[package]
```

每一阶段失败都必须停止后续阶段，避免产生“看起来成功但模型实际不可用”的包。

## 十三、已有转换产物

如果其他环境已经完成模型转换，可以跳过 Converter 的量化/导出步骤，直接整理已有离线模型、tokenizer 和 metadata，然后按 HLLM Package Contract 生成 `.hllm`。

```text
已有离线模型
   ↓
manifest
   ↓
checksum
   ↓
.hllm
```

这也是项目面对“某些电脑无法完成模型转换”时的备用路线。

## 十四、命令参考

```bash
# 环境检查
hllm doctor

# 下载
hllm download Qwen/Qwen3.8-27B --output ./models

# 检查模型
hllm inspect ./models/Qwen__Qwen3.8-27B

# 构建
hllm build ./models/Qwen__Qwen3.8-27B \
  --target <target> \
  --quant int4 \
  --profile ./configs/qwen3.8-27b-int4.example.yaml \
  --output ./dist

# 验证
hllm validate ./dist/model.hllm
```

CLI 参数和具体 profile 以代码当前版本为准；README 中的 `<target>` 是占位符，不是可直接用于 CANN 的芯片名称。

## 十五、目录与日志

推荐生产目录：

```text
workspace/
├── models/
├── builds/
│   └── <job-id>/
│       ├── inspect/
│       ├── quantization/
│       ├── onnx/
│       ├── cann/
│       ├── validation/
│       └── logs/
└── dist/
```

失败任务不要马上删除 workspace，否则无法诊断 CANN 失败。

## 十六、故障排查

### 下载失败

检查：

```text
HF token
revision
network
磁盘
Hugging Face Hub version
```

### `UNSUPPORTED_ARCHITECTURE`

检查 `config.json` 的：

```text
model_type
architectures
text_config
```

### `INSUFFICIENT_DISK`

清理：

```text
旧 build workspace
ONNX 中间产物
失败的 CANN 产物
Hugging Face 缓存
```

然后重新执行 planner。

### CANN 转换失败

不要只复制最后一行错误。应收集：

```text
CANN version
profile
target chip
command
stdout
stderr
input ONNX metadata
shape configuration
```

### `.hllm` 无法在 HarmonyOS 安装

优先检查：

```text
manifest.target
manifest.runtime
manifest.artifacts
SHA256
设备能力
NPU memory
Runtime version
```

## 十七、开发与测试

```bash
source .venv/bin/activate
python -m pytest
python -m compileall src
```

建议每新增一个：

```text
Model Adapter
Target Profile
Quantization Backend
Export Backend
```

同时补充对应测试。

## 十八、开发原则

- 不把 CANN 私有命令硬编码进通用 Pipeline。
- 不把模型转换依赖塞进 HarmonyOS Runtime。
- 不把未验证 target profile 标成 production-ready。
- 不把“成功生成文件”当成“模型可运行”。
- 真实硬件验证结果必须可追溯到 profile、CANN 版本和构建记录。
