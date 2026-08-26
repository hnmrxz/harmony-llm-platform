# HarmonyOS LLM Runtime

运行在 HarmonyOS 电脑、平板及后续兼容设备上的本地大语言模型运行与管理应用。

本项目是 `harmony-llm-platform` 的模型消费端，只接收已经由 Ubuntu Converter 生成的 `.hllm` 模型包。

> **本项目不负责模型转换。HarmonyOS 端只负责模型导入、验证、安装、管理和本地推理。**

## 一、用户完整使用流程

```text
Ubuntu
  ↓
生成 .hllm
  ↓
复制 / 上传到 HarmonyOS
  ↓
Runtime 导入
  ↓
SHA-256 / Manifest 校验
  ↓
设备兼容性检查
  ↓
安装
  ↓
Tokenizer / Prompt
  ↓
Native Runtime
  ↓
CANN / NNRt / NPU
  ↓
流式输出
```

Runtime 用户不需要安装：

```text
Python
PyTorch
Transformers
ONNX
CANN Converter
```

## 二、支持的输入

普通用户入口只接受：

```text
*.hllm
```

不建议直接把以下文件交给 Runtime：

```text
*.safetensors
*.onnx
*.om
*.omc
*.bin
```

这些底层文件应先由 Ubuntu Converter 整理成符合 HLLM Package Contract 的 `.hllm`。

## 三、导入模型

### 本地文件导入

典型流程：

```text
模型中心
 → 导入模型
 → 选择 .hllm
 → 校验
 → 安装
```

导入时不得直接覆盖正在使用的模型。

推荐安装流程：

```text
Temporary Import
      ↓
Integrity Check
      ↓
Compatibility Check
      ↓
Atomic Install
      ↓
READY
```

### 从 Ubuntu 传输

后续支持局域网模式：

```text
Ubuntu Converter
       │
       │ LAN
       ▼
HarmonyOS Runtime
```

Ubuntu 负责提供已完成的 `.hllm`；HarmonyOS 只下载该文件并执行与本地导入相同的校验流程。

## 四、安装前检查

Runtime 必须按顺序执行：

```text
1. 文件存在
2. HLLM archive 可读取
3. manifest.json 可解析
4. schema_version 支持
5. artifact 存在
6. SHA-256 正确
7. tokenizer 资源完整
8. target chip 匹配
9. Runtime/CANN 版本匹配
10. NPU / memory / storage 资源满足要求
```

任何一步失败，模型不能进入 `READY`。

## 五、模型状态

推荐状态机：

```text
IMPORTED
   ↓
VALIDATING
   ↓
INSTALLED
   ↓
READY
   ↓
RUNNING
   ↓
READY
```

异常状态：

```text
ERROR
INCOMPATIBLE
```

模型升级建议采用双目录或临时目录：

```text
model-v1
model-v2.tmp
    ↓
validate
    ↓
atomic switch
    ↓
model-v2
```

这样新模型失败时不会破坏旧模型。

## 六、设备能力检测

安装和启动前都应获取 `DeviceProfile`：

```text
DeviceProfile
├── HarmonyOS version
├── device model
├── chip
├── NPU availability
├── RAM
├── storage
├── runtime/CANN version
└── supported offline model formats
```

模型包则提供 `ModelRequirements`：

```text
ModelRequirements
├── backend
├── target chip
├── runtime version
├── minimum memory
├── context length
└── artifact format
```

最终结果：

```text
COMPATIBLE
COMPATIBLE_WITH_LIMITS
INCOMPATIBLE
```

错误必须说明原因，例如：

```text
目标芯片不匹配
CANN Runtime 版本过低
NPU 内存不足
上下文长度超出当前设备建议值
不支持所需 artifact format
```

## 七、Qwen3.8-27B 运行注意事项

Qwen3.8-27B 是本项目当前核心目标，但它是大模型/多模态模型。

第一阶段应将能力明确拆成：

```text
Text LLM
  └── 第一优先级

Vision
  └── 后续独立验证

Video
  └── 后续独立验证
```

因此：

> Runtime 能够运行 Qwen3.8-27B 的文本路径，不等于视觉和视频输入路径已经支持。

另外，是否可以运行 27B 不由“27B”这个数字单独决定，而取决于：

```text
target chip
NPU memory
runtime version
quantization
context length
operator support
```

## 八、Tokenizer 与 Chat Template

Tokenizer 是 Runtime 的核心组件，不可以只依赖默认规则。

模型包应提供：

```text
tokenizer.json / tokenizer.model
tokenizer_config.json
special tokens
chat template
model config
```

完整链路：

```text
User Message
      ↓
Conversation State
      ↓
Chat Template
      ↓
Tokenizer
      ↓
Input IDs
      ↓
Native Runtime
      ↓
CANN / NPU
      ↓
Token IDs
      ↓
Tokenizer Decode
      ↓
Streaming Text
```

每个模型的 Chat Template 都可能不同，因此 Runtime 不能硬编码 Qwen prompt 给所有模型共用。

## 九、推理 API

Native Runtime 建议提供稳定接口：

```text
loadModel(modelId)
unloadModel(modelId)
createSession(modelId)
appendMessage(sessionId, role, content)
generate(sessionId, options)
cancel(sessionId)
reset(sessionId)
```

生成参数至少支持：

```text
temperature
top_p
max_tokens
stop sequences
context length
```

UI 与 Native Runtime 通过异步事件流通信：

```text
GENERATE_STARTED
TOKEN
TOKEN
TOKEN
GENERATE_COMPLETED
```

错误事件：

```text
MODEL_LOAD_FAILED
OUT_OF_MEMORY
UNSUPPORTED_OPERATOR
RUNTIME_ERROR
CANCELLED
```

## 十、HarmonyOS 工程边界

推荐：

```text
ArkUI / ArkTS
     ↓
Application Services
     ↓
NAPI / Native Bridge
     ↓
C++ Runtime
     ↓
CANN / NNRt Backend
     ↓
NPU
```

ArkTS 负责：

- 页面
- 模型列表
- 导入
- 安装进度
- Chat UI
- 设置
- 错误展示

C++/Native 负责：

- HLLM package access
- tokenizer bridge
- model loader
- tensor/session management
- generation
- KV cache
- CANN/NNRt calls

不要把模型转换逻辑放进 ArkTS。

## 十一、模型管理

模型中心至少显示：

```text
Model Name
Model Family
Quantization
Package Size
Target Chip
Runtime Version
Context Length
Compatibility
Status
```

操作：

```text
导入
验证
安装
启动
停止
删除
升级
重新验证
```

## 十二、存储管理

模型文件可能非常大，因此安装流程必须使用临时空间和原子切换：

```text
.hllm
 ↓
/tmp/import/model.hllm
 ↓
verify
 ↓
extract
 ↓
verify artifacts
 ↓
install
 ↓
READY
```

必须检查：

- package size
- extracted size
- free space
- temporary space

删除模型时确认模型没有处于 `RUNNING`。

## 十三、完整性和安全

Runtime 必须验证：

```text
manifest.json
schema version
artifact presence
SHA-256
```

后续增加数字签名：

```text
.hllm
 ↓
Signature Verify
 ↓
Trusted Publisher
 ↓
Install
```

绝不因为模型包中存在某个脚本文件而自动执行该脚本。

## 十四、常见导入问题

### `PACKAGE_INVALID`

通常检查：

```text
.hllm 是否完整
manifest.json 是否存在
ZIP 是否损坏
```

### `CHECKSUM_FAILED`

模型传输过程中被改变或包生成时的 artifact 已经不同。

重新从 Converter 复制 `.hllm`，不要手工替换包内文件。

### `DEVICE_UNSUPPORTED`

检查：

```text
manifest.target.chip
DeviceProfile.chip
```

### `CANN_VERSION_UNSUPPORTED`

模型是在特定版本 CANN 环境转换的，Runtime 的离线模型兼容条件可能不同。应使用目标设备对应的 target profile 重新转换。

### `INSUFFICIENT_MEMORY`

可尝试：

```text
更小模型
更低 context length
更合适的量化 profile
```

但不能仅靠减少 UI 配置绕过模型本身的 NPU 内存需求。

## 十五、模型启动

推荐模型启动流程：

```text
Select Model
    ↓
Re-check Compatibility
    ↓
Load Tokenizer
    ↓
Load Offline Model
    ↓
Create Session
    ↓
READY
```

真正开始生成时：

```text
Prompt
 ↓
Tokenize
 ↓
Runtime Execute
 ↓
NPU
 ↓
Decode
 ↓
UI Stream
```

## 十六、性能与 Benchmark

Runtime 应记录：

```text
first token latency
prompt tokens/sec
generation tokens/sec
peak memory
context length
load time
model size
```

不要只记录“聊天是否成功”。

后续可以对同一个 `.hllm` 在不同 HarmonyOS 设备上建立 benchmark 表。

## 十七、开发与测试

HarmonyOS Runtime 开发时至少建立：

```text
Package Parser Test
Manifest Test
Checksum Test
Compatibility Test
Model Manager Test
Tokenizer Test
Native Backend Test
Inference Test
```

其中 Native/CANN 真机测试不能由模拟器结果代替。真实 NPU 兼容性必须使用对应设备验证。

## 十八、当前项目边界

### Runtime 做

```text
.hllm
 ↓
Verify
 ↓
Install
 ↓
Tokenizer
 ↓
Native Runtime
 ↓
CANN / NNRt / NPU
```

### Runtime 不做

```text
Hugging Face
 ↓
Transformers
 ↓
Quantization
 ↓
ONNX Export
 ↓
CANN Convert
```

这些全部由 Ubuntu Converter 完成。

## 十九、推荐用户操作手册

普通用户：

```text
1. 获取一个 .hllm
2. 打开 HarmonyOS LLM Runtime
3. 进入“模型中心”
4. 点击“导入模型”
5. 选择 .hllm
6. 等待完整性检查和兼容性检查
7. 安装
8. 打开模型
9. 新建会话
10. 开始本地对话
```

开发者：

```text
1. 在 Ubuntu 部署 Converter
2. 下载 Hugging Face 模型
3. inspect 模型
4. 使用已验证 target profile
5. 执行 INT4 / ONNX / CANN 转换
6. validate
7. 生成 .hllm
8. 传到 HarmonyOS 真机
9. 导入
10. 做设备兼容性与推理 benchmark
```

## 二十、Compatibility Contract

本项目与 Ubuntu Converter 的唯一核心耦合点是：

> **HLLM Package Contract**

Converter 可以持续更换量化、ONNX、CANN 转换实现；Runtime 可以持续更换 UI 和内部运行时实现，只要双方遵守同一模型包规范，就可以独立演进。

## 相关文档

- `../README.md`
- `../docs/hllm-package-spec.md`
- `../docs/hllm-manifest.schema.json`
