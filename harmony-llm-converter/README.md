# HarmonyOS LLM Converter

Ubuntu 端的大语言模型生产工具链。

本项目负责将 Hugging Face 或本地模型经过检测、资源规划、量化/FP8 处理、ONNX 导出、CANN/OMG 转换、验证后生成 `.hllm`。

> **Ubuntu 负责模型生产；HarmonyOS Runtime 只消费最终 `.hllm`。**

## 目标模型

当前首要目标：

- Qwen3 / Qwen3.5 / Qwen3.8
- `Qwen/Qwen3.8-27B`
- `Qwen/Qwen3.8-27B-FP8`
- 参数规模更小且通过目标 CANN/设备验证的模型

## 1. 安装

```bash
cd harmony-llm-converter
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[huggingface,quantization,export,dev]'
```

不强制使用虚拟环境。系统 Python 用户级安装也可以，但不建议使用 `sudo pip` 污染系统 Python。

检查：

```bash
hllm doctor
```

重点：

```text
Python / PyTorch / Transformers / ONNX
CANN Toolkit / ATC
HarmonyOS DDK tools_omg（Kirin/OMC 路径需要）
CUDA/GPU（量化阶段如 profile 要求）
RAM / Disk
```

## 2. Build Profile 与 Target Profile 已分离

这是当前版本的重要设计：**不要用一个 YAML 同时表达“模型构建参数”和“目标芯片能力”。**

### Build Profile

用于定义一次模型构建：

```text
configs/qwen3.8-27b-int4.example.yaml
configs/qwen3.8-27b-fp8.example.yaml
```

schema：

```text
docs/build-profile.schema.json
```

主要包括：

- `model.source`
- `model.input_quantization`
- `pipeline.preferred_path`
- `quantization.*`
- `export.*`
- `cann.target_chip`
- `runtime.*`
- `output.*`

### Target Profile

用于定义设备/芯片目标：

```text
configs/kirin9020-cann91-llm.example.yaml
configs/kirinx90-cann91-llm.example.yaml
```

schema：

```text
docs/target-profile.schema.json
```

通过 `load_target_profile()` 单独加载。这样 `target.platform / soc_version / runtime_model_format` 不会再被误当成 BuildProfile 字段。

## 3. 下载模型

```bash
hllm download Qwen/Qwen3-0.6B --output ./models
```

27B：

```bash
hllm download Qwen/Qwen3.8-27B-FP8 --output ./models
```

也可以直接在 `build` 中传 Hugging Face repo ID，Converter 会自动下载：

```bash
hllm build Qwen/Qwen3-0.6B \
  --target <validated-target> \
  --profile ./configs/<profile>.yaml \
  --output ./build
```

建议生产环境固定 revision；模型缓存目录可通过：

```bash
--model-cache ./models
```

## 4. 检查模型

```bash
hllm inspect ./models/Qwen__Qwen3-0.6B
```

检测包括：

- model type
- architecture
- nested `text_config`
- dtype
- FP8 输入识别
- context length
- tokenizer
- chat template
- multimodal capability
- safetensors shard inventory
- header-level parameter count

不会为了读取 metadata 把整个大模型权重加载进内存。

## 5. FP8 路径

官方 FP8 模型必须显式声明：

```yaml
model:
  input_quantization: fp8

pipeline:
  preferred_path: fp8_to_cann
```

Converter 的行为：

```text
输入模型 = FP8
      ↓
profile 支持 FP8？
 ├─ YES → 跳过通用 QUANTIZE Stage
 │          ↓
 │        EXPORT / CANN
 └─ NO  → UNSUPPORTED_FP8_TARGET
```

不会把 FP8 静默当成 BF16/FP16。

## 6. INT4 路径

```text
BF16/FP16
  ↓
CANN 4-bit Quantization
  ↓
Stage 1: Weight
  ↓
Stage 2: Activation
  ↓
Stage 3: Quant Params
  ↓
ONNX
  ↓
CANN/OMG
  ↓
.hllm
```

厂商命令和参数必须放在 profile，而不是写死进 Python 主 Pipeline。

## 7. Qwen3.8-27B-FP8 示例

```bash
hllm build ./models/Qwen__Qwen3.8-27B-FP8 \
  --target <validated-target-chip> \
  --profile ./configs/qwen3.8-27b-fp8.example.yaml \
  --output ./build/qwen3.8-27b-fp8
```

先执行 dry-run：

```bash
hllm build ./models/Qwen__Qwen3.8-27B-FP8 \
  --target <validated-target-chip> \
  --profile ./configs/qwen3.8-27b-fp8.example.yaml \
  --output ./build/qwen3.8-27b-fp8 \
  --dry-run
```

## 8. 小模型冒烟测试

推荐首先测试：

```text
Qwen3-0.6B
↓
inspect
↓
dry-run build
↓
真实 ONNX/CANN/OMG
↓
.hllm
```

先跑：

```bash
hllm download Qwen/Qwen3-0.6B --output ./models
hllm inspect ./models/Qwen__Qwen3-0.6B
hllm build ./models/Qwen__Qwen3-0.6B \
  --target <validated-target-chip> \
  --profile ./configs/<validated-profile>.yaml \
  --dry-run
```

确认基础链路稳定后，再逐步升级到 1.7B、4B、8B/14B，最后验证 27B。

## 9. 资源规划

Converter 在真正执行前检查：

```text
available RAM
available disk
source weight size
estimated quantized size
conversion workspace
```

磁盘不足返回：

```text
INSUFFICIENT_DISK
```

RAM 不足返回：

```text
INSUFFICIENT_RAM
```

对 27B 模型，不要按最终 INT4 理论大小准备磁盘；量化、ONNX、CANN/OMG 中间产物需要额外 workspace。

## 10. 命令失败诊断

外部 CANN/量化命令统一经过 command runner：

```text
command
exit_code
stdout
stderr
```

失败返回：

```text
EXTERNAL_COMMAND_FAILED
```

不要只看一句 `conversion failed`，完整日志会进入 build state / build.json。

## 11. HLLM 打包与验证

最终：

```bash
hllm validate ./build/.../model.hllm
```

验证：

- `.hllm` 后缀
- manifest
- schema version
- artifact presence
- SHA-256
- 禁止绝对路径和 `..` 路径

大文件 SHA-256 采用流式读取，不会把几十 GB 模型一次性读进内存。

## 12. 开发测试

安装开发依赖：

```bash
python3 -m pip install -e '.[dev]'
```

运行：

```bash
pytest
python -m compileall src
```

当前测试覆盖：

- build profile schema
- target profile schema
- Qwen/FP8 detection
- safetensors header parameter counting
- planner
- build dry-run / FP8 gate
- HLLM checksum corruption
- safe command rendering
- manifest serialization
- Hugging Face module syntax

## 13. 真实硬件验证边界

WSL2 + x86_64 CANN Toolkit 可以用于模型分析/转换开发，但不等于 Kirin NPU。

最终 PASS 必须分层：

```text
Level 1  Python / dependencies
Level 2  CANN / ATC / DDK tools
Level 3  target offline model generation
Level 4  HarmonyOS real-device NPU inference
```

`Target Profile` 只有在 Level 3/4 验证完成后才能标记 production-ready。
