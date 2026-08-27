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

WSL2 使用 Windows 主机 NVIDIA 驱动，不要在 WSL 内重复安装 Linux NVIDIA Driver。2GB MX450 只适合 CUDA 环境和小模型冒烟测试，不适合作为 Qwen3.8-27B 的生产量化 GPU。

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

针对 OMG/OMC 冒烟测试，Qwen profile 使用静态输入和低版本 opset：

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
external-data 保持原始文件布局
  ↓
ModelProto 元数据安全处理
  ↓
ONNX preflight
  ↓
OMG/OMC
```

`generic` ONNX 模式仍保留用于通用兼容性诊断，但不作为 Kirin OMC production path。

### External Data 规则

大模型权重可能必须使用 ONNX external data。项目不会对“未加载到内存中的 external tensor”执行二次 `onnx.save()`、`onnxsim` 重写或伪造 placeholder 文件；这样可以避免：

```text
ValidationError:
Data of TensorProto ... should be stored in ... but it is not regular file
```

同时避免为 27B 模型把几十 GB 权重一次性载入 RAM。

`onnx-audit.json` 会检查每个 `external_data.location` 是否存在、是否越界到模型目录之外。

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

然后执行真实构建：

```bash
hllm build ./models/Qwen__Qwen3-0.6B \
  --target KirinX90 \
  --profile ./configs/qwen3-0.6b-kirinx90.example.yaml \
  --output ./build/qwen3-0.6b
```

构建会审计工作目录中的旧 ONNX；与 profile 的 opset、IR 或静态要求不一致时自动重新导出，不会继续使用旧动态模型。

## 9. OMG/OMC 调用

未配置自定义 `cann.commands` 时自动生成：

```text
omg
  --framework=5
  --target=omc
  --platform=kirinx90
```

输出目录提前创建，但最终模型文件必须由 OMG/OMC 实际生成。

## 10. ONNX preflight

在 OMG 前检查：ONNX checker、opset、IR version、静态输入、external-data location、external-data 文件存在性、node/initializer/input/output inventory。

不满足 profile 的模型会在进入 OMG 前失败并报告 `ONNX_PREFLIGHT_FAILED`。

## 11. FP8

FP8 输入必须显式声明：

```yaml
model:
  input_quantization: fp8
pipeline:
  preferred_path: fp8_to_cann
```

不支持时返回 `UNSUPPORTED_FP8_TARGET`，不会静默转换成 BF16/FP16。

## 12. INT4

INT4 生产路径应使用 CANN-compatible LLM quantization backend，而不是把普通整数化逻辑写进主 Pipeline。

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

当前 `cann_static` 是 **Prefill-only smoke-test exporter**，只用于验证当前 Qwen3 graph 与 DDK/OMG/KirinX90 的兼容性，不代表最终高性能 runtime graph。

生产 LLM exporter 后续必须提供：

```text
Prefill graph
Decode graph
KV cache inputs/outputs
```

在此之前不标记 production-ready。

## 14. 资源规划

Build 前估算 source weights、quantized size、RAM、disk。资源不足返回 `INSUFFICIENT_RAM` / `INSUFFICIENT_DISK`；dry-run 只输出 warning。

## 15. HLLM 打包与验证

```bash
hllm validate ./build/.../model.hllm
```

验证 manifest、schema、artifact presence、SHA-256 和路径安全。大文件使用流式 SHA-256。

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
