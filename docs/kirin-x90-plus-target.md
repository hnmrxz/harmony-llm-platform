# Kirin X90 Plus Target

本项目的目标电脑芯片为 **Kirin X90 Plus**。

## CANN Target Identity

截至当前官方 CANN Kit 文档，工具链的 AI 处理器类型公开支持列表使用 **KirinX90** 这一系列标识，并没有在当前文档中把 Kirin X90 Plus 作为独立的 `soc_version` 标识列出。官方文档要求在真实 HarmonyOS 设备上通过 HDC 查询实际芯片类型：

```bash
hdc list targets
hdc -t <SN> shell param get ohos.boot.chiptype
```

因此 Converter 中：

```text
产品名称：Kirin X90 Plus
CANN platform/soc identity：以真实设备返回值为准
默认候选：KirinX90
```

**不得仅凭产品营销名称把 `KirinX90` 直接视为已经验证的 X90 Plus target。** 在真实设备上查询到的 `ohos.boot.chiptype` 应优先于 profile 中的猜测值。

## Smoke Test Profile

使用：

```text
configs/qwen3-0.6b-kirinx90.example.yaml
```

执行：

```bash
hllm build ./models/Qwen__Qwen3-0.6B \
  --target KirinX90 \
  --profile ./configs/qwen3-0.6b-kirinx90.example.yaml \
  --output ./build \
  --dry-run
```

真实转换前，先把 profile 的 target identity 替换成设备实际返回值，并确认匹配的 HarmonyOS DDK/OMG 工具版本。
