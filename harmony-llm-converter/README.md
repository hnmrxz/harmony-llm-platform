# HarmonyOS LLM Converter

Ubuntu 端的大语言模型生产工具链。

本项目负责将 Hugging Face 或本地模型经过检测、资源规划、量化/FP8 处理、ONNX 导出、CANN/OMG 转换、验证后生成 `.hllm`。

> **Ubuntu 负责模型生产；HarmonyOS Runtime 只消费最终 `.hllm`。**

## 当前目标

首要目标设备：**Kirin X90 Plus**。

当前模型目标：

- `Qwen/Qwen3-0.6B`（冒烟测试）
- Qwen3 / Qwen3.5 / Qwen3.8 小于等于 27B 的受支持模型
- `Qwen/Qwen3.8-27B`
- `Qwen/Qwen3.8-27B-FP8`

产品名称 `Kirin X90 Plus` 与 CANN/离线模型的工具链标识可能不是同一个字符串；真实设备应通过 HDC 获取芯片标识，再确定 production Target Profile。

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
CUDA/GPU（仅 GPU-required profile 需要）
RAM / Disk
```

## 2. NVIDIA CUDA 环境

CUDA 不是所有 Converter 操作的硬依赖。以下操作可以 CPU-only 执行：

```text
HF 下载
inspect
资源规划
dry-run
manifest / .hllm 验证与打包
```

量化 profile 如果明确要求 CUDA，则必须有可用 GPU。

### WSL2 + CUDA 12.6

项目提供参考文件：

```text
requirements/cuda126.txt
```

使用官方 PyTorch CUDA 12.6 wheel index：

```bash
python3 -m pip install -r requirements/cuda126.txt \
  --index-url https://download.pytorch.org/whl/cu126
```

然后：

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

WSL2 使用 Windows 主机 NVIDIA 驱动，不要在 WSL 内重复安装 Linux NVIDIA Driver。

> GeForce MX450 2GB 可以用于 CUDA 环境冒烟测试，但不适合作为 Qwen3.8-27B FP8/INT4 的生产量化 GPU。大模型生产转换应使用有足够 GPU VRAM 和系统 RAM 的 Ubuntu 环境。

## 3. CANN / HarmonyOS DDK

普通 CANN Toolkit 提供通用转换工具（例如 `atc`）。Kirin/HarmonyOS 最终 OMC 路径还需要对应 HarmonyOS DDK 的 `tools_omg`。

检查：

```bash
which atc
find "$HOME" -type f -path '*tools_omg*/omg' 2>/dev/null
hllm doctor
```

如果 `ATC` 正常但 `OMG`/`tools_omg` 不存在，不要把 `.om` 误认为已经生成了最终 Kirin 模型。

## 4. Build Profile 与 Target Profile

不要用一个 YAML 同时表达“模型构建参数”和“目标设备能力”。

### Build Profile

定义一次具体构建，例如：

```text
configs/qwen3-0.6b-kirinx90.example.yaml
configs/qwen3.8-27b-fp8.example.yaml
```

### Target Profile

定义设备/芯片目标，例如：

```text
configs/kirinx90-cann91-llm.example.yaml
```

真实 production profile 必须经过目标设备 Level 3/4 验证后再启用。

## 5. 下载模型

```bash
hllm download Qwen/Qwen3-0.6B --output ./models
```

27B FP8：

```bash
hllm download Qwen/Qwen3.8-27B-FP8 --output ./models
```

也可以直接构建 Hugging Face repo ID：

```bash
hllm build Qwen/Qwen3-0.6B \
  --target KirinX90 \
  --profile ./configs/qwen3-0.6b-kirinx90.example.yaml \
  --output ./build
```

生产环境建议固定 revision，并使用：

```bash
--model-cache ./models
```

## 6. 模型检查

```bash
hllm inspect ./models/Qwen__Qwen3-0.6B
```

检测包括：

- model type
- architecture
- nested `text_config`
- dtype / FP8
- context length
- tokenizer
- chat template
- multimodal capability
- safetensors shard inventory
- header-level parameter count

不会为了读取 metadata 把整个模型权重加载进内存。

## 7. Qwen3-0.6B 冒烟测试

第一条实际开发链建议固定为：

```bash
hllm download Qwen/Qwen3-0.6B --output ./models

hllm inspect ./models/Qwen__Qwen3-0.6B

hllm build ./models/Qwen__Qwen3-0.6B \
  --target KirinX90 \
  --profile ./configs/qwen3-0.6b-kirinx90.example.yaml \
  --output ./build/qwen3-0.6b \
  --dry-run
```

`--dry-run` 会执行：

```text
DOWNLOAD
INSPECT
PLAN
QUANTIZE（非 FP8 时展示）
EXPORT
CANN_CONVERT
VALIDATE
PACKAGE
```

但不会执行任何外部转换命令。

## 8. FP8 路径

官方 FP8 输入必须显式声明：

```yaml
model:
  input_quantization: fp8

pipeline:
  preferred_path: fp8_to_cann
```

行为：

```text
输入 = FP8
  ↓
profile 支持 FP8？
 ├─ YES → 跳过通用 QUANTIZE
 └─ NO  → UNSUPPORTED_FP8_TARGET
```

不会把 FP8 静默当成 BF16/FP16。

目标模型：

```text
Qwen/Qwen3.8-27B-FP8
```

## 9. INT4 路径

```text
BF16/FP16
  ↓
4-bit Quantization backend
  ↓
Weight quantization
  ↓
Activation quantization
  ↓
Quantization parameters
  ↓
ONNX
  ↓
CANN/OMG target conversion
  ↓
.hllm
```

厂商命令和参数必须放进 profile/backend，不写死进主 Pipeline。

## 10. 资源规划

Build 前检查：

```text
source weight size
estimated quantized size
available RAM
available disk
```

不足时真实 build 会返回：

```text
INSUFFICIENT_RAM
INSUFFICIENT_DISK
```

`--dry-run` 不会因为资源不足而失败，而是输出：

```json
"resource_ok": false,
"resource_warning": "INSUFFICIENT_RAM"
```

这允许在没有足够硬件的开发机上先验证 pipeline。

## 11. 外部命令失败诊断

CANN/量化/导出命令通过统一 command runner 执行并记录：

```text
command
exit_code
stdout
stderr
```

失败：

```text
EXTERNAL_COMMAND_FAILED
```

## 12. HLLM 打包与验证

最终：

```bash
hllm validate ./build/.../model.hllm
```

验证：

- manifest
- schema version
- artifact presence
- SHA-256
- 路径安全

模型 artifact 使用流式 SHA-256，避免把几十 GB 文件一次性读进 RAM。

## 13. Kirin X90 Plus 真机确认

连接实际 HarmonyOS 电脑后：

```bash
hdc list targets
```

再：

```bash
hdc -t <SN> shell param get ohos.boot.chiptype
```

把返回值作为最终 `soc_version/platform` 的依据，不要仅凭产品宣传名称猜测 CANN target。

## 14. 测试

```bash
pytest
python -m compileall src
```

重点覆盖：

- profile schema
- Qwen detection
- FP8 gating
- safetensors header counting
- planner
- build dry-run
- HLLM integrity
- command execution

## 15. 验证等级

```text
Level 1  Python / dependencies
Level 2  CANN / ATC / HarmonyOS DDK
Level 3  Kirin target offline model generation
Level 4  HarmonyOS X90 Plus real-device NPU inference
```

只有 Level 3/4 验证成功后，Target Profile 才能标记 `production-ready`。
