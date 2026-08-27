# HarmonyOS LLM Runtime

运行在 HarmonyOS 电脑、平板及后续兼容设备上的本地大语言模型运行与管理应用。

本项目是 `harmony-llm-platform` 的模型消费端，只接收已经由 Ubuntu Converter 生成的 `.hllm` 模型包。

> **本项目不负责模型转换。HarmonyOS 端只负责模型导入、验证、安装、管理和本地推理。**

## 一、与 CANN LLM Engine 的关系

本 Runtime 的推理层直接使用华为官方 **CANN LLM Engine**（`libhiai_llm_engine.so` + `llm_engine_c_api_export.h` 等头文件）。

LLM Engine 已经封装好完整的大模型计算链路：

```text
文本 → 分词 → LLM prefill → LLM Decoder → 输出采样 → 分词解码 → 文本
```

并内置**内存复用、KV Cache 管理、投机推理、端云协同、多模态拓展、性能指标**等能力。因此 Runtime 不需要自研 tokenizer、prefill/decode 循环、KV cache 或采样。

对照官方示例（`CANN_LLM_Engine_Demo/CANNLLMEngineDemoNext/.../llm_demo.cpp`），Runtime 端只需要：

```text
1. 读 context.json    → LLMEngine_Context_CreateFromContextJson(text)
2. 读 executor.json   → LLMEngine_Executor_CreateFromJson(text)
3. LLMEngine_Prompt_SetText(prompt)  ← 纯文本（不再需要 tokenizer）
4. LLMEngine_Executor_LLM_GenerateAsync(executor, context, prompt)
5. 通过 LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc 回调拿到流式 token
6. 通过 LLMEngine_Context_Get*TimeMs / Get*TokenCount 拿到内置指标
```

> Runtime 唯一需要自处理的只是**聊天模板**：把用户消息按模型 `chat_template` 渲染成一段文本，再交给 `LLMEngine_Prompt_SetText`。这是纯字符串层，不涉及 tokenizer。

### 与旧设计的区别

历史版本把推理接口建模为**低阶 NNRt**（`OH_NNCompilation_ConstructWithOfflineModelBuffer` + `OH_NNExecutor_RunSync`），需要 Runtime 自己实现完整 LLM 链路。现在改为 LLM Engine 生命周期接口，大幅简化：

| 项 | 旧（低阶 NNRt） | 现（LLM Engine） |
|---|---|---|
| 接口 | `LoadOfflineModel(buffer)` / `Run(inputIds, output)` | `LoadModel(contextJson, executorJson)` / `GenerateAsync(prompt, callbacks)` |
| tokenizer | 自研 | 引擎内部（经 `executor.json` 的 `tokenizer.path/type`） |
| prefill/decode | 自研 | 内置 |
| KV cache | 自研 | 内置 |
| 采样 | 自研 | `context.json` 的 `sampler` 配置 |
| 指标 | 自研计时 | `Get*TimeMs` / `Get*TokenCount` |

## 二、仓库结构

```text
harmony-llm-runtime/
├── README.md
├── AppScope/
│   └── app.json5
├── build-profile.json5
├── hvigorfile.ts
├── oh-package.json5
├── entry/
│   ├── build-profile.json5
│   ├── hvigorfile.ts
│   ├── oh-package.json5
│   └── src/main/
│       ├── module.json5
│       ├── cpp/
│       │   ├── CMakeLists.txt              # 链接 libhiai_llm_engine.so 与 native 源码
│       │   ├── LLMEngineModule.cpp         # NAPI 桥（loadmodel/modelinfer/answerget/deinitmodel）
│       │   └── types/libentry/
│       │       ├── Index.d.ts              # ArkTS 侧类型声明
│       │       └── oh-package.json5
│       ├── ets/
│       │   ├── entryability/EntryAbility.ets
│       │   └── pages/Index.ets             # Chat UI + 模型加载
│       └── resources/base/
│           ├── element/string.json
│           ├── element/color.json
│           └── profile/main_pages.json
└── native/
    ├── runtime/
    │   ├── InferenceBackend.h              # 抽象推理接口（LLM Engine 生命周期）
    │   └── LLMEngineApi.h                  # LLM Engine C-API 契约
    ├── backends/cann/
    │   └── CannBackend.cpp                 # LLM Engine 实现
    ├── test/
    │   ├── make_test_hllm.py               # 生成一个 deflate 压缩的 .hllm 测试包
    │   ├── host_import_test.cpp            # 主机侧导入流程测试（无需 OHOS/NAPI）
    │   └── run_host_test.sh
    └── native/
        ├── runtime/
        │   ├── InferenceBackend.h          # 抽象推理接口（LLM Engine 生命周期）
        │   └── LLMEngineApi.h              # LLM Engine C-API 契约
        ├── backends/cann/
        │   └── CannBackend.cpp             # LLM Engine 实现
        └── model_package/
            ├── Manifest.h
            ├── EngineConfig.h              # context.json / executor.json 类型
            ├── ZipReader.h / .cpp          # .hllm（ZIP）解包（stored + deflate）
            ├── Inflate.h / .cpp            # 自包含 DEFLATE 解压（无需 zlib）
            ├── PackageReader.h / .cpp      # .hllm 校验 / 定位引擎文件
            ├── Json.h / .cpp               # 轻量 JSON 解析
            └── Sha256.h / .cpp             # 完整性校验用 SHA-256
```

## 三、模型包（.hllm）

Runtime 消费的 `.hllm` 取自 Ubuntu Converter，内部按 CANN LLM Engine 需要的文件集组织：

```text
model.hllm
├── manifest.json
├── model/
│   ├── <model>.omc
│   ├── SubGraph_0.weight
│   ├── <model>_<len>_<ctx>.embedding_weights
│   └── <model>_<len>_<ctx>.embedding_dequant_scale
└── config/
    ├── context.json      # Engine 生成/采样配置
    ├── executor.json     # .omc / 权重目录 / tokenizer 描述
    └── tokenizer.json
```

`manifest.json` 的 `artifacts[]` 逐条声明 path + sha256 + size；`target.backend` 记为 `cann_llm_engine`；`runtime.llm_engine` 携带 `engine_type`、`kv_cache_max_len`、`prefill_len`、`max_io_tokens`、`vocab_size`、`hidden_size`、`num_hidden_layers`、注意力头数/维度、`max_position_embeddings`、`embedding_input_type` 等，供设备兼容性判断使用。

## 四、原生推理层

### `native/runtime/LLMEngineApi.h`

声明 LLM Engine 的 C-API（opaque 句柄 + 函数），与官方示例 `llm_demo.cpp` 的调用一致。真机构建时，该文件需与所装 CANN Kit 的 DDK 头文件保持一致（或直接替换为真实头文件）。

### `native/runtime/InferenceBackend.h`

抽象推理接口：

```cpp
struct DeviceProfile { osVersion; chip; runtimeVersion; availableMemoryBytes; };
struct GenerationMetrics { totalTimeMs; prefillTimeMs; decodeTimeMs; inputTokenCount; outputTokenCount; decodeNum; };
struct GenerationCallbacks { onToken; onDone; onError; };

class InferenceBackend {
    virtual bool IsCompatible(device, targetChip, requiredRuntime) const = 0;
    virtual bool LoadModel(contextJson, executorJson) = 0;
    virtual bool Generate(prompt, outText) = 0;
    virtual bool GenerateAsync(prompt, callbacks) = 0;
    virtual bool Cancel() = 0;
    virtual bool GetAllGeneration(outText) const = 0;
    virtual bool GetMetrics(out) const = 0;
    virtual void Unload() = 0;
};
```

### `native/backends/cann/CannBackend.cpp`

实现上述接口，直接调用 `LLMEngineApi.h`。引擎回调只携带 `const LLMEngine_Context*`，因此用一个小的全局注册表把 context 指回所属 backend，从而把 `onToken/onDone/onError` 转发到 UI。

### `native/model_package/`

导入链路：

```text
.hllm
  ├─ ZipReader.ExtractAll()   解包（stored + deflate）
  │    ├─ Inflate.h/.cpp      自包含 DEFLATE 解压（不依赖 zlib）
  │    └─ 拒绝路径穿越（绝对路径 / ..）
  ├─ PackageReader.ReadManifest()  解析 manifest.json（用 Json.h）
  ├─ PackageReader.VerifyIntegrity()  每个 artifact 存在 + SHA-256 匹配
  └─ PackageReader.LocateEngineFiles()  定位 .omc / 权重目录 / embedding /
                                        context.json / executor.json / tokenizer.json
```

`manifest.json` 校验：

```text
可解析
schema_version == 1.0
每个 artifact 路径不越界（无绝对路径 / ..）
每个 artifact 存在 + SHA-256 匹配
```

`Json.h/.cpp`、`Sha256.h/.cpp`、`Inflate.h/.cpp` 均为无依赖实现；真机可用平台 JSON / 密码学 / archive 库替换而不影响调用方。

### 主机侧导入测试（无需 OHOS）

`native/` 与 `test/` 可在普通 Linux 上用宿主 g++ 直接编译验证完整导入流程：

```bash
cd harmony-llm-runtime
bash test/run_host_test.sh
```

该测试用 Python 生成一个 deflate 压缩的 `.hllm`，用 ZipReader 解包、PackageReader 校验并定位引擎文件，全部通过即证明导入链路正确。

## 五、NAPI 桥（entry/src/main/cpp/LLMEngineModule.cpp）

ArkTS 通过 `libentry.so` 调用：

```ts
llmEngine.loadmodel(contextJsonPath, executorJsonPath): string   // 加载 .omc 并返回状态
llmEngine.modelinfer(question): void                              // 异步生成
llmEngine.answerget((token: string) => void): void                // 注册流式回调
llmEngine.deinitmodel(): void                                     // 释放引擎
```

流式 token 通过 `napi_threadsafe_function` 从引擎工作线程安全回到 JS 主线程。

## 六、接入 CANN Kit（真机）

参考官方 DemoNext，在 `entry/src/main/cpp` 下准备：

```text
cpp/
├── include/
│   ├── lm_engine_model_info.h
│   ├── cann_llm_engine_context.h
│   ├── cann_llm_engine_executor.h
│   ├── llm_engine_c_api_export.h
│   ├── llm_engine_context_base.h
│   ├── llm_engine_executor_base.h
│   └── llm_engine_return_types.h
└── lib64/
    └── libhiai_llm_engine.so
```

`entry/build-profile.json5` 的 `externalNativeOptions.path` 已指向 `./src/main/cpp/CMakeLists.txt`；`CMakeLists.txt` 已把 `include/` 加入头文件路径并链接 `libhiai_llm_engine.so`。

## 七、模型导入流程（运行时机）

```text
选择 .hllm
    ↓
临时解包到 /tmp/import
    ↓
PackageReader：manifest / SHA-256 / 路径校验
    ↓
设备兼容性检查（DeviceProfile × model requirements）
    ↓
原子安装（model-v1 → model-v2.tmp → validate → switch）
    ↓
loadmodel(context.json, executor.json)
    ↓
READY
```

启动会话后：

```text
用户输入
    ↓ 应用 chat template（字符串层）
    ↓ LLMEngine_Prompt_SetText
    ↓ LLMEngine_Executor_LLM_GenerateAsync
    ↓ LLM Engine 内部：分词 → prefill → decode → 采样 → 解码
    ↓ onSomeToken 回调 → NAPI → ArkTS 流式展示
```

## 八、设备兼容性

最终能否运行由以下因素共同决定：

```text
模型包（.omc / 精度 / 词表）
      ×  目标芯片
      ×  CANN / LLM Engine 版本
      ×  NPU 内存（kv_cache_max_len、max_io_tokens、prefill_len、max_position_embeddings）
      ×  operator 支持
      ↓
COMPATIBLE / COMPATIBLE_WITH_LIMITS / INCOMPATIBLE
```

## 九、构建

```bash
# 在 DevEco Studio（Windows / Mac）打开 harmony-llm-runtime 并同步构建
# 或者，在已配置 SDK 的环境：
devecocli build --modules entry --build-mode debug
```

> devecocli 在本机为 `1.2.0-stable`，但其 DevEco Studio / SDK 自动检测目前在 **Linux 不支持**，真机构建/运行需在 DevEco Studio 或已正确配置 SDK 的进程中执行。

## 十、模型管理 / 状态机 / 安全

模型中心至少显示：名称、家族、量化、包大小、目标芯片、运行时版本、上下文长度、兼容性、状态。

推荐状态机：

```text
IMPORTED → VALIDATING → INSTALLED → READY → RUNNING → READY
异常：ERROR / INCOMPATIBLE
```

安全原则：

- `.hllm` 是数据，不是可执行代码。
- 拒绝路径穿越、绝对路径、重复 manifest 条目、未声明可执行内容。
- 绝不因为包内存在某个脚本文件而自动执行它。
- SHA-256 校验通过前不得安装。
- 删除模型前确认没有处于 `RUNNING`。

## 十一、相关文档

- `../README.md`
- `../docs/hllm-package-spec.md`
- `../docs/optimization-against-cann-llm-guide.md`（对照官方解决方案的优化与简化分析）
