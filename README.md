# HarmonyOS LLM Platform

面向 HarmonyOS 电脑、平板及后续鸿蒙设备的本地大语言模型转换、交付与 NPU 推理平台。

## 项目定位

本项目严格拆分为两个独立子项目：

- `harmony-llm-converter/`：运行在 Ubuntu，负责从 Hugging Face / 本地模型开始，完成模型分析、资源规划、量化、ONNX 导出、CANN 转换、验证和 `.hllm` 打包。
- `harmony-llm-runtime/`：运行在 HarmonyOS 设备，只消费已经转换好的 `.hllm`，负责导入、校验、安装、设备兼容性判断、Tokenizer 与本地 NPU 推理。

核心原则：

> **Ubuntu 负责生产模型，HarmonyOS 负责消费模型。**

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
│ CANN / NNRt / NPU            │
└──────────────────────────────┘
```

## 当前目标模型

当前首要目标为：

- Qwen3 / Qwen3.5 / Qwen3.8
- **Qwen/Qwen3.8-27B**
- **Qwen/Qwen3.8-27B-FP8（官方 FP8 输入源）**
- 参数规模小于 27B 的受支持模型

Qwen 官方已经在 Hugging Face 提供 `Qwen/Qwen3.8-27B-FP8`。该仓库提供 FP8 量化后的 Transformers 格式权重，采用 fine-grained FP8、block size 128，并标注 Apache-2.0。它可以直接被 Transformers、vLLM、SGLang 等框架使用，但它仍然不是 HarmonyOS/CANN 的最终离线模型。citeturn450986view0turn450986view1

因此 Converter 现在提供两条入口：

```text
路径 A：官方 FP8
Qwen/Qwen3.8-27B-FP8
        ↓
Inspect
        ↓
FP8 → 已验证 CANN Target Profile
        ↓
Validation
        ↓
.hllm

路径 B：原始模型
Qwen/Qwen3.8-27B
        ↓
Inspect
        ↓
已验证量化/导出流程
        ↓
ONNX
        ↓
CANN
        ↓
Validation
        ↓
.hllm
```

> **FP8 是优先验证的 27B 输入格式，但是否可以直接进入目标 CANN 转换链，必须由具体 CANN Kit 版本与目标芯片 profile 实测确认。**

## 仓库结构

```text
harmony-llm-platform/
├── README.md
├── LICENSE
├── docs/
│   ├── hllm-package-spec.md
│   ├── hllm-manifest.schema.json
│   └── qwen3.8-27b-fp8.md
├── harmony-llm-converter/
│   ├── README.md
│   ├── pyproject.toml
│   ├── configs/
│   ├── src/
│   └── tests/
└── harmony-llm-runtime/
    ├── README.md
    ├── entry/
    ├── ets/
    └── native/
```

## 快速开始

### 1. Ubuntu 安装 Converter

进入转换器目录：

```bash
cd harmony-llm-converter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[huggingface,quantization,export,dev]'
```

检查环境：

```bash
hllm doctor
```

必须重点确认：

```text
Linux
Python >= 3.10
GPU/CUDA（用于 CANN LLM 量化；FP8→CANN 也必须按目标 profile 验证）
PyTorch
Transformers
ONNX
CANN Kit / 工具链
磁盘空间
内存
```

具体 CANN 版本必须按实际目标设备建立验证矩阵；仓库不假定任何未验证的厂商命令行参数。

### 2. 下载模型

#### 官方 FP8 路径（27B 推荐先验证）

```bash
hllm download Qwen/Qwen3.8-27B-FP8 \
  --output ./models
```

检查：

```bash
hllm inspect ./models/Qwen__Qwen3.8-27B-FP8
```

#### 原始模型路径

```bash
hllm download Qwen/Qwen3.8-27B \
  --output ./models
```

### 3. 构建 `.hllm`

使用已经经过真实 CANN/目标设备验证的 target profile。

FP8 输入：

```bash
hllm build ./models/Qwen__Qwen3.8-27B-FP8 \
  --target <validated-target-chip> \
  --profile ./configs/qwen3.8-27b-fp8.example.yaml \
  --output ./dist
```

原始模型：

```bash
hllm build ./models/Qwen__Qwen3.8-27B \
  --target <validated-target-chip> \
  --quant int4 \
  --profile ./configs/qwen3.8-27b-int4.example.yaml \
  --output ./dist
```

建议先运行 planner / dry-run（如当前 CLI 版本支持）：

```bash
hllm build ./models/Qwen__Qwen3.8-27B-FP8 \
  --target <validated-target-chip> \
  --profile ./configs/qwen3.8-27b-fp8.example.yaml \
  --output ./dist \
  --dry-run
```

转换阶段：

```text
Download
  ↓
Inspect
  ↓
Resource Plan
  ↓
Model Adapter
  ↓
FP8 conversion / validated quantization path
  ↓
ONNX Export（需要时）
  ↓
Target-specific CANN Conversion
  ↓
Validation
  ↓
.hllm Packaging
```

完成后得到：

```text
./dist/
└── <model>-<target>-<profile>.hllm
```

### 4. 验证模型包

```bash
hllm validate ./dist/<model>-<target>-<profile>.hllm
```

至少检查：

```text
manifest.json
artifact presence
SHA-256
model metadata
tokenizer resources
target profile
```

### 5. 导入 HarmonyOS

将 `.hllm` 传到鸿蒙电脑或平板，然后在 Runtime 中：

```text
模型中心
  → 导入模型
  → 选择 .hllm
  → 完整性校验
  → 设备兼容性检查
  → 安装
  → 启动
```

Runtime 不需要 Hugging Face、PyTorch、Transformers 或模型转换工具链。

## 手工转换与已有模型导入

如果 Ubuntu 上无法为目标设备完成模型转换，也可以直接使用已经转换好的 `.hllm`：

```text
已有 CANN 离线模型/产物
        ↓
按 HLLM Package Contract 打包
        ↓
.hllm
        ↓
HarmonyOS Runtime
```

因此平台有两条合法输入路径：

```text
路径 A：
Hugging Face → Converter → .hllm → Runtime

路径 B：
已有转换产物 → HLLM Package → Runtime
```

## `.hllm` 模型包

两端唯一核心合同：

```text
.hllm
├── manifest.json
├── model/
├── tokenizer/
├── config/
└── integrity/
```

规范文件：

- `docs/hllm-package-spec.md`
- `docs/hllm-manifest.schema.json`
- `docs/qwen3.8-27b-fp8.md`

Converter 可以更换量化/导出/CANN 实现，只要生成的 `.hllm` 符合协议，Runtime 就不需要改动。

## 设备兼容性

最终能否运行由以下因素共同决定：

```text
Model Package
      ×
Target Chip
      ×
CANN / NNRt Version
      ×
NPU Memory
      ×
Context Length
      ×
Operator Support
      ↓
Compatibility Result
```

因此请不要仅根据“模型只有 4bit”或“模型已经是 FP8”判断某台鸿蒙电脑或平板能够运行。

## 常见问题

### `hllm doctor` 找不到 CANN 工具

检查 CANN 安装路径及 `PATH`/环境脚本，然后重新打开 shell 并执行：

```bash
which atc
which omg
hllm doctor
```

实际所需工具以对应 CANN Kit profile 为准。

### FP8 模型为什么还需要 Converter？

`Qwen/Qwen3.8-27B-FP8` 是官方 Hugging Face Transformers 格式的 FP8 权重，不是 HarmonyOS/CANN 的最终设备离线模型。必须根据目标芯片执行已验证的 CANN 转换，并最终封装成 `.hllm`。citeturn450986view0

### FP8 能否直接转换成 CANN？

不能在仓库层面假设“所有 CANN 版本、所有目标芯片都支持直接 FP8→设备模型”。必须由 target profile 显式声明并经过真实硬件验证。未验证时应返回 `UNSUPPORTED_TARGET` 或要求采用其他已验证路径。

### 27B 转换前磁盘不足

27B 原始/FP8 权重都属于大型模型，量化、ONNX 和 CANN 转换还可能需要额外临时空间。不要只按最终模型理论大小准备磁盘。

### 模型导入 Runtime 后显示不兼容

重点检查：

```text
manifest.target.chip
manifest.target.runtime_version
manifest.runtime.context_length
artifact format
NPU memory
```

### 已有 `.om/.omc` 但 Runtime 不接受

这是预期设计。Runtime 的普通用户入口是 `.hllm`，底层离线模型必须先按 HLLM Package Contract 描述并打包。

### Qwen3.8-27B 能转换但视觉输入不能用

Qwen3.8-27B 是原生视觉语言模型，官方模型卡同时描述文本、图像和视频能力。当前项目优先锁定文本 LLM 路径，再单独推进多模态 profile；文本路径 PASS 不等于 VLM 完整 PASS。citeturn450986view0

## 开发与验证

Converter：

```bash
cd harmony-llm-converter
source .venv/bin/activate
python -m pytest
python -m compileall src
```

开发模式安装：

```bash
python -m pip install -e '.[dev]'
```

建议每增加一个模型 Adapter 或 Target Profile，都同时增加：

```text
metadata test
planner test
command/profile test
package validation test
```

真实 CANN 转换和真实 HarmonyOS NPU 运行必须在对应硬件环境验证，CI 不会伪造 PASS。

## 当前状态

已经进入 **MVP 实现阶段**，包括：

- Converter 基础代码
- Hugging Face 下载
- Qwen 家族检测基础
- 分片 safetensors inventory
- Build Planner
- Pipeline Runner
- 外部命令执行边界
- CANN 4bit 量化阶段建模
- ONNX/CANN Backend
- 官方 Qwen3.8-27B-FP8 输入 profile
- `.hllm` 打包与校验
- HLLM Manifest Contract
- HarmonyOS Native Runtime 边界
- 基础 CI

尚未宣称完成的内容：

- 某一具体 Kirin 芯片的最终 CANN 转换 PASS
- 某一具体 HarmonyOS 电脑/平板上的 NPU 端到端 PASS
- Qwen3.8-27B FP8→特定 Kirin 目标的生产级直通路径
- Qwen3.8-27B 视觉/视频路径的生产级支持

## 贡献原则

- 不把厂商私有参数硬编码到通用 Pipeline。
- 不把模型转换依赖塞入 HarmonyOS Runtime。
- 不跳过 Manifest、checksum 和兼容性检查。
- 不把未经过真实设备验证的 profile 标记为 production-ready。

## License

本项目采用 **GNU General Public License v3.0 (GPL-3.0)**。

完整许可证文本见 [`LICENSE`](./LICENSE)。

> 外部模型仓库拥有其自己的许可证。`Qwen/Qwen3.8-27B-FP8` 当前在 Hugging Face 标注为 Apache-2.0；使用该模型时应同时遵守模型仓库的许可与使用要求。citeturn456562search1
