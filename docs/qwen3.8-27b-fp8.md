# Qwen3.8-27B-FP8 作为 Converter 输入源

Qwen 官方在 Hugging Face 提供了 `Qwen/Qwen3.8-27B-FP8`。该仓库提供 FP8 量化后的 Transformers 格式权重，采用 fine-grained FP8、block size 128，并可直接被 Transformers、vLLM、SGLang 等框架使用。详见：

https://huggingface.co/Qwen/Qwen3.8-27B-FP8

## 为什么将 FP8 纳入本项目

Converter 的目标是生产 HarmonyOS/CANN 可部署模型，而不是强制所有输入都从 BF16 原始权重开始。

因此现在定义两条 Qwen3.8 输入路径：

```text
路径 A：官方 FP8
Qwen/Qwen3.8-27B-FP8
        ↓
Inspect
        ↓
FP8 → CANN target conversion / required intermediate processing
        ↓
Validation
        ↓
.hllm
```

以及：

```text
路径 B：原始模型
Qwen/Qwen3.8-27B
        ↓
Inspect
        ↓
CANN 量化 / required intermediate processing
        ↓
ONNX
        ↓
CANN conversion
        ↓
Validation
        ↓
.hllm
```

实际选择哪条路径，由目标 CANN Kit、目标芯片和已验证的转换 profile 决定。

## FP8 不是最终 HarmonyOS 模型

`Qwen/Qwen3.8-27B-FP8` 是 Hugging Face Transformers 格式的 FP8 权重，不等价于 HarmonyOS/CANN 的设备离线模型。

因此：

```text
Qwen3.8-27B-FP8
      ≠
HarmonyOS .hllm
```

仍然必须经过本项目的：

```text
target inspection
↓
conversion
↓
validation
↓
HLLM packaging
```

## FP8 路径的优先级

对于 27B 目标，默认推荐先验证官方 FP8 输入路径，因为它可以避免直接从 BF16 权重重新执行完整量化流程；但只有在目标 CANN Kit/芯片 profile 已验证支持该 FP8 输入路径时，才能将其标记为生产级流程。

如果目标 profile 不支持 FP8 直接转换，Converter 必须自动回退到经过验证的其他输入路径，或明确提示用户先使用官方/受支持的中间转换方式。

## 资源规划

FP8 输入仍然属于大型模型工作负载。不要把“FP8”误认为“可以忽略 GPU、RAM、磁盘和转换 workspace”。

Build Planner 必须同时考虑：

- 模型下载大小
- 临时解包空间
- 转换 workspace
- ONNX workspace
- CANN workspace
- 最终 `.hllm` 大小
- 校验副本/回滚空间

## 模型能力

Qwen3.8-27B 是视觉语言模型，支持文本、图像和视频能力。当前 Converter 的第一阶段重点仍然是**文本语言模型路径**；FP8 源模型的视觉/视频处理器和相关资源必须单独进入 multimodal profile，不能因为语言部分可转换就宣称完整 VLM 部署成功。

## 典型命令

```bash
hllm doctor

hllm download Qwen/Qwen3.8-27B-FP8 \
  --output ./models

hllm inspect ./models/Qwen__Qwen3.8-27B-FP8

hllm build ./models/Qwen__Qwen3.8-27B-FP8 \
  --target <validated-kirin-target> \
  --profile ./configs/qwen3.8-27b-fp8.example.yaml \
  --output ./dist
```

## License

该 Hugging Face 模型仓库当前标注为 `apache-2.0`。本项目的 GPL-3.0 许可证仅适用于本项目源代码，不改变外部模型仓库自身的许可证要求。
