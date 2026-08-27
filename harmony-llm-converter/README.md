# HarmonyOS LLM Converter

Ubuntu 端的大语言模型生产工具链。

本项目负责将 Hugging Face 或本地模型经过检测、资源规划、量化/FP8 处理、CANN-oriented ONNX 导出、OMG/OMC 转换、验证后生成 `.hllm`。

> **Ubuntu 负责模型生产；HarmonyOS Runtime 只消费最终 `.hllm`。**

## 当前目标

首要目标设备：**Kirin X90 Plus**。

当前模型目标：

- `Qwen/Qwen3-0.6B`（转换链冒烟测试）
- Qwen3/Qwen3.x 中不大于 27B 的受支持模型
- `Qwen/Qwen3.8-27B-FP8`

产品名称 `Kirin X90 Plus`、CANN target 和 OMG platform 是三个不同层级。当前安装的 X90 plugin 明确使用 `kirinx90` 作为 SoC/platform 标识；最终 production profile 仍应通过目标设备 HDC 信息验证。

## 1. 安装

```bash
cd harmony-llm-converter
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[huggingface,quantization,export,dev]'
```

不强制使用虚拟环境。系统 Python 用户级安装也可以；不要使用 `sudo pip` 污染系统 Python。

检查：

```bash
hllm doctor
```

重点：Python、PyTorch、Transformers、ONNX、CANN/ATC、HarmonyOS DDK `tools_omg`、CUDA（GPU profile 需要）、RAM、Disk。

## 2. CUDA

CUDA 不是所有 Converter 操作的硬依赖。CPU-only 可以完成下载、inspect、资源规划、dry-run、manifest/HLLM 验证。

当前 WSL2 参考环境：

```text
PyTorch 2.13.0 + CUDA 12.6
```

安装：

```bash
python3 -m pip install -r requirements/cuda126.txt \
  --index-url https://download.pytorch.org/whl/cu126
```

验证：

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name)
    print("VRAM MiB:", p.total_memory // (1024**2))
PY
```

WSL2 使用 Windows 主机 NVIDIA 驱动，不要在 WSL 内重复安装 Linux NVIDIA Driver。2GB MX450 只适合环境/小模型冒烟测试，不适合 Qwen3.8-27B 的生产量化。

## 3. CANN / HarmonyOS DDK

普通 CANN Toolkit 提供通用转换工具（如 `atc`）。Kirin/HarmonyOS OMC 路径需要对应 DDK 的 `tools_omg` 与目标 platform plugin。

检查：

```bash
which atc
which omg
hllm doctor
```

本项目当前验证的目录结构：

```text
$HOME/Ascend/cann/
$HOME/Ascend/ddk-6.0.1.0/tools/tools_omg/
$HOME/Ascend/ddk-6.0.1.0/tools/platform/kirinx90/
```

## 4. Build Profile / Target Profile

Build Profile 定义一次模型构建；Target Profile 定义设备/芯片能力。不要混用。

示例：

```text
configs/qwen3-0.6b-kirinx90.example.yaml
configs/qwen3.8-27b-fp8.example.yaml
configs/kirinx90-cann91-llm.example.yaml
```

## 5. Qwen3 CANN-oriented ONNX export

默认 Qwen 自动导出不是简单的“任意 Transformers forward → 动态 ONNX”。针对 OMG/OMC 冒烟测试，使用：

```yaml
export:
  mode: cann_static
  precision: fp32
  opset: 11
  ir_version: 6
  batch_size: 1
  sequence_length: 4
  external_data: true
```

行为：

```text
Qwen3
  ↓
静态 batch/sequence
  ↓
显式 position_ids
  ↓
eager attention
  ↓
Torch ONNX export
  ↓
onnxsim 静态图简化
  ↓
统一 external-data
  ↓
ONNX preflight
  ↓
OMG/OMC
```

原始 `generic` ONNX 模式仍保留用于兼容性诊断，但不应作为 Kirin OMC production path。

每次 build 都会生成：

```text
work/export/model.onnx
work/export/model.onnx.data（启用 external_data 时）
work/export/onnx-audit.json
```

如果工作目录里存在旧的、与 profile 不匹配的 ONNX，构建会自动判定为 stale 并重新导出，不会复用旧的 opset17/dynamic 图。

## 6. 下载模型

```bash
hllm download Qwen/Qwen3-0.6B --output ./models
```

也可以直接：

```bash
hllm build Qwen/Qwen3-0.6B \
  --target KirinX90 \
  --profile ./configs/qwen3-0.6b-kirinx90.example.yaml \
  --output ./build
```

生产环境建议固定 Hugging Face revision。

## 7. 模型检查

```bash
hllm inspect ./models/Qwen__Qwen3-0.6B
```

会检查 model type、architecture、nested `text_config`、dtype/FP8、context length、tokenizer、chat template、multimodal capability 和 safetensors inventory，并通过权重 header 统计参数量。

## 8. Qwen3-0.6B 冒烟测试

```bash
hllm build ./models/Qwen__Qwen3-0.6B \
  --target KirinX90 \
  --profile ./configs/qwen3-0.6b-kirinx90.example.yaml \
  --output ./build/qwen3-0.6b \
  --dry-run
```

dry-run 只验证 pipeline/resource/profile，不执行外部转换。

真实构建：

```bash
hllm build ./models/Qwen__Qwen3-0.6B \
  --target KirinX90 \
  --profile ./configs/qwen3-0.6b-kirinx90.example.yaml \
  --output ./build/qwen3-0.6b
```

## 9. OMG/OMC 调用

当没有自定义 `cann.commands` 时，Kirin profile 会自动生成：

```text
omg
  --framework=5
  --target=omc
  --platform=kirinx90
```

输出路径使用绝对路径，父目录会在执行前创建。不要通过 `touch` 预创建 output 文件；OMG 应自行生成最终离线模型。

## 10. ONNX preflight

在 OMG 前自动检查：

- ONNX checker
- opset
- IR version
- 静态输入 shape
- external-data location
- external-data 文件是否存在
- node/initializer/input/output inventory

不满足 profile 的模型会在进入 OMG 前失败并返回 `ONNX_PREFLIGHT_FAILED`，避免把确定不兼容的图交给厂商 parser。

## 11. FP8

FP8 输入必须显式声明：

```yaml
model:
  input_quantization: fp8
pipeline:
  preferred_path: fp8_to_cann
```

不支持时返回 `UNSUPPORTED_FP8_TARGET`，不会静默转换成 BF16/FP16。

目标模型：

```text
Qwen/Qwen3.8-27B-FP8
```

## 12. INT4

INT4 生产路径应使用 CANN-compatible LLM quantization backend，而不是把普通 Python 整数化逻辑写进主 Pipeline。需要量化的 profile 应显式提供 quantization workspace/commands。

```text
BF16/FP16
  ↓
LLM quantization
  ↓
quantized weights / parameters
  ↓
CANN-oriented export
  ↓
OMG/OMC
  ↓
.hllm
```

## 13. Prefill / Decode / KV Cache

当前 `cann_static` 是 **Prefill-only smoke-test exporter**，目标是先验证 Qwen3 graph 能被当前 DDK/OMG/KirinX90 接受；它不是最终高性能 LLM runtime graph。

生产 LLM exporter 的下一阶段必须提供：

```text
Prefill graph
Decode graph
KV cache inputs/outputs
```

在此之前不会把“单一 full-forward logits ONNX”标记为 production-ready。

## 14. 资源规划

Build 前估算 source weights、quantized size、RAM、disk。真实 build 资源不足会返回 `INSUFFICIENT_RAM` / `INSUFFICIENT_DISK`；dry-run 只输出 warning，不阻断规划验证。

## 15. HLLM 打包与验证

```bash
hllm validate ./build/.../model.hllm
```

验证 manifest、schema、artifact presence、SHA-256 和路径安全。大文件使用流式 SHA-256，不会一次性读入内存。

## 16. X90 Plus 真机确认

```bash
hdc list targets
hdc -t <SN> shell param get ohos.boot.chiptype
```

以设备返回值校准最终 `soc_version`/platform，不以市场产品名猜测。

## 17. 测试

```bash
pytest
python -m compileall src
```

重点测试 profile schema、Qwen detection、FP8 gating、safetensors counting、planner、ONNX audit、HLLM integrity 和 command runner。

## 18. 验证等级

```text
Level 1  Python / dependencies
Level 2  CANN / ATC / DDK / OMG
Level 3  Kirin target offline model generation
Level 4  HarmonyOS X90 Plus real-device NPU inference
```

只有 Level 3/4 通过后，Target Profile 才能标记 `production-ready`。
