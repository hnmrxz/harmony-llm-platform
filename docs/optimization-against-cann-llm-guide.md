# 对照《CANN LLM 大语言模型解决方案》的优化与简化分析

本文对照官方指南《CANN LLM 大语言模型解决方案》：

<https://gitcode.com/HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp/blob/master/CANN_LLM/CANN_LLM_Engine_Guide/CANN%20LLM%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88.md>

以及配套示例工程 `CANN_LLM_Engine_Demo/CANNLLMEngineDemoNext` 中的 `llm_demo.cpp`，对本项目（`harmony-llm-platform`）当前实现进行逐项对照，指出**可以优化与简化的点**。

> 本文只做分析与建议，不替未经验证的参数“背书”。凡是官方指南标注需要真实硬件验证的路径，本文一律标注为待验证，不宣称生产可用。

---

## 0. 结论摘要

当前项目存在一个根本性的**架构错位**：

- **Converter 侧**：项目走的是“通用 torch ONNX 导出 + 自研 `kirin_compat` 图重写 + 极简 OMG 参数”的路径。而官方方案走的是“**dopt 三段式量化 → NPU 亲和导出的友好 ONNX → 带完整输入/输出签名与量化配置的 OMG**”的路径。两者本质不同。
- **Runtime 侧**：项目当前把推理接口建模为低阶 NNRt（`OH_NNCompilation_ConstructWithOfflineModelBuffer` + `OH_NNExecutor_RunSync`），需要 Runtime 自己实现完整 LLM 链路（prefill/decode/kv cache/tokenizer）。而官方**直接提供 `LLM Engine`**（`libhiai_llm_engine.so` + `llm_engine_c_api_export.h` 等头文件），一句话即可完成“文本 → token → prefill → 采样 → token→文本”的整条链，并内置内存复用、KV Cache 管理、投机推理、多模态拓展。

对应用的几个关键事实（来自官方指南）：

1. **模型支持范围**（指南第 1 节）：`Qwen2.5-1.5B`、`DeepSeek-R1-Distill-Qwen-1.5B`、`Glm-1.5b`、`Qwen2.5-7B-Instruct`、`Qwen3-8B`。**`kirin9020` 仅支持前两个，`kirinx90` 支持全部五个。**
2. **`Qwen3.8-27B` 不在官方支持列表内。** 这直接挑战本项目当前“首要目标为 Qwen3.8-27B”的设定。
3. **硬件支持**：`kirin 9020`、`kirin x90`。
4. **Runtime 消费的文件集**（指南第 6.2 节）：`yourmodel.omc`、`SubGraph_0.weight`、`embedding_weights`、`embedding_dequant_scale`、`context.json`、`executor.json`、`tokenizer.json`。
5. **量化是 GPU 三段式**（指南第 2 节）：`stage1` 权重量化 → `stage2` 激活量化 → `stage3` 量化参数提取，产出 `dopt_config.json`、`trained.pth`、`fake_quant_weight.pth`、`quant_params_file`、`embedding_weights`、`embedding_quant_scale`。**只有 GPU 环境。**
6. **NPU 亲和适配**（指南第 3 节）：官方在 `npu_tuned_model` 下修改了各模型的 `modeling` 文件，让导出图适配 NPU。

---

## 1. Converter 侧：与官方流程的差距与简化点

### 1.1 官方转换链 vs 当前实现

官方（指南第 2、3、4 节）：

```text
HF 源模型
   │
   ├─ dopt 三段式量化（GPU）───────────► dopt_config.json
   │                                     trained.pth
   │                                     fake_quant_weight.pth
   │                                     quant_params_file（量化系数）
   │                                     embedding_weights / embedding_quant_scale
   │
   ├─ NPU 亲和导出的友好 ONNX + pb 权重 ──►（用 npu_tuned_model 的 modeling 文件）
   │
   └─ OMG ─────────────────────────────► .omc + SubGraph_0.weight
             --framework=5
             --compress_conf <quant_params_file>
             --input_shape="input_embed:1,-1,...;attention_mask:...;past_key_inN:...;..."
             --dynamic_dims="1,1,...;64,64,..."
             --input_type="past_key_inN:FP16;..."
             --output_type="lm_logits:FP32;past_keyN:FP16;..."
             --save_weights_as_external_data=true
             --platform=kirinx90
             --target=omc
```

当前实现（`backends/omg.py`）：

```text
omg --model=<onnx> --framework=5 --output=<prefix> --target=omc --platform=kirinx90
```

**缺失的关键参数：**

- `--compress_conf`（量化系数文件路径）。官方必须把 `dopt` 阶段产出的 `quant_params_file` 传给 OMG，才能得到真正的量化端侧模型。
- `--input_shape` / `--dynamic_dims`：定义 prefill / decode 的输入签名与动态档位。**没有它，OMG 无法导出可被 LLM Engine 消费的 OMC。**
- `--input_type` / `--output_type`：KV cache（`past_key/past_value`）用 FP16、`lm_logits` 用 FP32 等类型约束。
- `--save_weights_as_external_data=true`：把权重作为外置 `SubGraph_0.weight`。

**简化方向：** `backends/omg.py` 应改为按官方 `to_omc.sh` 的模板生成完整命令，而不是只带 5 个参数。这些参数来自模型/量化信息，应由 profile / 模型元数据驱动，而不是在代码里硬编码。

### 1.2 自研 `kirin_compat.py` 与 `onnx.py` 的图重写，官方方案不需要

当前项目为了“让通用 torch 导出图能被 KirinX90 的 `libai_fmk_onnx_parser.so` 解析”，写了大量图重写逻辑（`backends/kirin_compat.py` 365 行 + `onnx.py` 的 `normalize_onnx_node_names` / 外部数据元数据补全）：

- `_static_fold`：对常量锥做 NumPy 折叠，把 `attention_mask` 等折叠成 initializer。
- `_retype_gather_indices`：把 Gather 的 INT64 索引转 INT32。
- `_rewrite_unsqueeze_expand`：把 Unsqueeze→Reshape、Expand→Mul。
- `normalize_onnx_node_names`：用首输出张量作为节点名。
- 外部数据 offset/length 补全。

**为什么这些是“补丁”而非“正解”：** 官方用 **NPU 亲和适配后的 `modeling` 文件**做导出，产出的 ONNX 本来就是为 OMG/端侧消费优化的；同时官方 OMG 命令带完整 `--input_shape/--dynamic_dims/--input_type/--output_type`。也就是说，**图结构“天然”兼容**，不需要这些值等价重写。

**简化方向（高收益）：**

- 放弃自研通用导出 + `kirin_compat` 图重写，改用官方 `npu_tuned_model` 的导出脚本/建模文件（`export_model_single_qwen2.py` + `model_info_target.yaml`）产出友好 ONNX。
- `kirin_compat.py`、`onnx.py` 中的 `_static_fold` / `_retype_gather_indices` / `_rewrite_unsqueeze_expand` / `normalize_onnx_node_names` / 外部数据补全，**都可以整体删掉**。这能大幅简化 `onnx.py` 和 `backends/kirin_compat.py`。

> 注意：这条路上官方导出脚本依赖其 `npu_tuned_model` 目录与依赖环境（`cann_lm_engine/requirements.txt`，且**量化和导出均为 GPU 环境**）。因此在无真实 CANN Kit / GPU 环境下，当前自研路径至少能作为“诊断”保留；但不应把它当作生产转换链。建议以官方工具链为唯一生产路径，把自研图重写降级为“如官方路径不可用的回退”。

### 1.3 量化：官方是 GPU 三段式 dopt，项目目前只有“命令占位”

当前 `backends/quantization.py`：

```python
class ExternalQuantizer:
    def quantize(self, model_dir, output_dir, config):
        raise NotImplementedError("Wire a validated CANN Kit quantization command here; do not infer vendor flags.")
```

即：量化能力尚未落地，只是占位。

官方指南明确：

- 量化**仅支持 GPU**（`DEVICE=cuda`）。
- 三段式脚本 `run.sh stage1/2/3`，中间产物依次为 `dopt_config.json`（**需要手工修改 `quant_strategy`**）→ `trained_quant_weight.pth` → `trained.pth` → `fake_quant_weight.pth + quant_params_file + embedding_weights + embedding_quant_scale`。
- 推荐策略：decode 层 `Quant_act_weight_eco`、lm_head 层 `Quant_lm_head`、embedding 层 `Quant_Embed_MinMax`；权重 4bit、激活 16bit、group 128；`quant_param_2`：kirinx90 默认 false、kirin9020 默认 true；`embedding_separate` 控制 embedding 是否单独存 bin。

**简化方向：** 把 `ExternalQuantizer` 落实为对官方 `dopt_lm/opt_main.py` 三段式脚本的封装（`run.sh stage1/2/3`），并把 `dopt_config.json` 的策略生成、以及 `--compress_conf` 所需的 `quant_params_file` 作为阶段产物记录。`config.py` 的量化 profile 应为三段式暴露 `dopt_config.json` 生成与编辑参数。

### 1.4 FP8 输入路径是“推测性”的

项目 README / docs 把 `Qwen/Qwen3.8-27B-FP8` 作为“优先验证的 27B 输入源”，并定义 `fp8_to_cann` 路径。

官方指南**没有描述 FP8 直通转换链**。官方量化是 dopt 从原始 PyTorch 权重做 16-4 grouplinear。FP8 是 Transformers/vLLM/SGLang 侧的重量化表示，不等于 CANN 端侧输入。

**影响：**

- `fp8_to_cann` 作为“已验证”路径目前没有官方依据，属于推测。
- 若仍要支持 27B，需要先实测 FP8 权重能否被 dopt 加载，或走“FP8 → 反量化 → BF16 → dopt 4bit”的中间路径；这两者都需要真实硬件/GPU 验证。

**简化方向：** 明确标注 `fp8_to_cann` 为 `UNVALIDATED`，不作为默认首选；在官方支持范围内，优先用**原始权重 + dopt 三段式 + NPU 亲和导出 + OMG**这条官方路径拿到首个端到端 PASS。FP8 只作为后续独立验证项。

### 1.5 目标模型集的重新定位

本项目把 `Qwen3.8-27B` 作为“首要目标”。但官方指南当前支持列表最高为 **`Qwen3-8B`**（kirin x90），且**没有 27B**。

**建议：**

- 第一阶段把“首个端到端 PASS”的目标改为官方支持列表内的模型，例如 **`Qwen3-8B`（kirin x90）** 或 **`Qwen2.5-7B-Instruct`（kirin x90）**。这些模型的转换参数、`.omc` 输入签名、LLM Engine 配置（`executor.json`）官方文档都有明确值，最容易被验证。
- `Qwen3.8-27B` 降级为“超大模型 & 多模态探路”，作为独立、待验证的 stretch goal。
- 视觉/视频能力（指南也说明当前 LLM Engine 提供多模态拓展但仍以文本为主）应继续严格与文本路径分离。

---

## 2. Runtime 侧：改用官方 LLM Engine API，大幅简化

### 2.1 当前实现建模的是低阶 NNRt，需要重写

当前 `native/runtime/InferenceBackend.h` + `native/backends/cann/CannBackend.cpp`：

```cpp
virtual bool LoadOfflineModel(const std::vector<std::uint8_t>& modelBuffer) = 0;
virtual bool Run(const std::vector<std::int64_t>& inputIds, std::vector<float>& output) = 0;
```

注释里也写明计划绑定 `OH_NNCompilation_ConstructWithOfflineModelBuffer` 和 `OH_NNExecutor_RunSync`。这是**低阶 NNRt 离线模型编译 + 手动执行**的路径，意味着 Runtime 要自己实现：

- tokenizer 编码
- prefill / decode 循环
- KV cache 管理
- 采样（top-k/top-p/temperature）
- token→text 解码
- 多步串联与内存复用

**这是巨大的不必要复杂度。** 官方 `LLM Engine` 已经把这些全部封装好。

### 2.2 官方推荐集成的形态（来自 `llm_demo.cpp`）

官方 DemoNext 的调用链：

```cpp
// 1. 从 context.json 创建上下文
g_LLMEngineContext = LLMEngine_Context_CreateFromContextJson(jsonStr);

// 2. 从 executor.json 创建执行器（该文件内指定 .omc、权重目录、tokenizer 路径与类型）
g_LLMEngineExecutor = LLMEngine_Executor_CreateFromJson(jsonStr);

// 3. 构造 prompt
mllmPrompt = LLMEngine_Prompt_Create();
LLMEngine_Prompt_SetText(mllmPrompt, prompt);

// 4a. 同步：文本→token→prefill→decode→采样→token→文本，一次拿全量
LLMEngine_Executor_LLM_Generate(executor, context, prompt);
LLMEngine_Context_GetAllGenerationLen / GetAllGeneration  // 拿全量结果

// 4b. 异步（推荐，用于流式）：
LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(ctx, onSomeToken);   // 每个 token 回调
LLMEngine_Context_SetOnAllTokensGenerateDoneFunc(ctx, onAllDone);     // 完成回调
LLMEngine_Context_SetOnGenerateAsyncFailed(ctx, onFailed);            // 失败回调
LLMEngine_Executor_LLM_GenerateAsync(executor, context, prompt);

// 5. 性能指标（内置）
LLMEngine_Context_GetTotalTimeMs / GetPrefillTimeMs / GetDecodeTimeMs
LLMEngine_Context_GetInputTokenCount / GetOutputTokenCount / GetDecodeNum

// 6. 释放
LLMEngine_Executor_Deinit / Destroy, LLMEngine_Context_Destroy, LLMEngine_Prompt_Destroy
```

**结论：** Runtime 端只需要：

- 解析 `.hllm`，取出官方 7 文件集（`.omc`、`SubGraph_0.weight`、`embedding_weights`、`embedding_dequant_scale`、`context.json`、`executor.json`、`tokenizer.json`）。
- 读 `context.json` / `executor.json` 传给 `CreateFromContextJson` / `CreateFromJson`。
- 用文本 prompt 调用 `LLMEngine_Executor_LLM_GenerateAsync`，通过两个回调把流式 token 推给 UI。
- 不再需要自研 tokenizer、prefill/decode、KV cache、采样。

（唯一需要自处理的只是**聊天模板**：把用户消息按模型 `chat_template` 渲染成一段文本再交给 `LLMEngine_Prompt_SetText`。这是纯字符串层面，不涉及 tokenizer。）

### 2.3 架构图（面向电脑端 Runtime）

```text
ArkTS 页面（模型中心 / Chat UI / 设置）
        │ NAPI（loadmodel / modelinfer / answerget / deinitmodel）
        ▼
Native Runtime Engine（C++）
        │ 解析 .hllm（manifest/校验/解包），读 context.json & executor.json
        ▼
LLM Engine C-API（libhiai_llm_engine.so，由 DDK 提供）
        │ 自动完成 tokenize → prefill → decode → 采样 → detokenize
        ▼
NPU
```

### 2.4 Runtime 优化/简化建议清单

1. **删除自研 NNRt 低阶建模**（`LoadOfflineModel(modelBuffer)` / `Run(inputIds, output)`），改为 LLM Engine 生命周期接口：`LoadModel(contextJson, executorJson)` / `GenerateAsync(prompt, callbacks)` / `GetMetrics` / `Unload`。
2. **用官方 7 文件集作为 `model/` 布局**，并把 `context.json` / `executor.json` 直接放进 `.hllm`（作为 `config/` 或 `model/`），而不是让 Runtime 回去拼接。
3. **tokenizer 由 LLM Engine 内部处理**（通过 `executor.json` 的 `tokenizer.path` / `tokenizer.type`）。Runtime 只需应用 chat template。
4. **context.json / executor.json 语义进入 HLLM Manifest**：`manifest.runtime` 应包含 `llm_engine` 相关字段（如 `engine_type=autoregressive`、KV cache 长度、词表大小、hidden_size、num_hidden_layers、head 数/维度等），以支撑设备兼容性判断（例如 `kv_cache_max_len`、`prefill_len`、`max_io_tokens` 与 NPU 内存的匹配）。
5. **性能指标直接取 LLM Engine 内置**（`GetTotalTimeMs` / `GetPrefillTimeMs` / `GetDecodeTimeMs` / `GetInputTokenCount` / `GetOutputTokenCount` / `GetDecodeNum`），不必自建计时与 token 统计。

---

## 3. HLLM Package Contract 的对齐

当前 HLLM 规范（`docs/hllm-package-spec.md`、`packaging/hllm.py`、`schema/manifest.py`）把 `model/` 视为“设备相关离线模型文件”，并另设 `tokenizer/`、`config/`。

建议与官方 LLM Engine 消费的 7 文件集对齐（作为一组新的、官方向 target profile）：

```text
model.hllm
├── manifest.json
├── model/
│   ├── <model>.omc
│   ├── SubGraph_0.weight
│   ├── <model>_<len>_<ctx>.embedding_weights
│   └── <model>_<len>_<ctx>.embedding_dequant_scale
└── config/
    ├── context.json
    ├── executor.json
    └── tokenizer.json     # 或放到 model/，遵循 executor.json 的 tokenizer.path
```

`manifest.json` 的 `artifacts[]` 逐条声明 path + sha256 + size；`target.backend` 应为 `cann_llm_engine`；`runtime` 增加 `llm_engine` 配置（engine_type、kv_cache_max_len、prefill_len、max_io_tokens、vocab_size、hidden_size、num_hidden_layers、num_attention_kv_heads、num_attention_head_dims、max_position_embeddings、embedding_input_type）。

> 这些都是“官方值”，需要在真机验证后落地；但结构化字段应现在就在 manifest 中定义，避免后续返工。

---

## 4. 迁移与验证路线（建议顺序）

第一阶段（最快拿到端到端 PASS）：
1. 选官方支持模型：`Qwen3-8B`（kirin x90）。
2. GPU 环境跑官方 `dopt` 三段式量化，得到 `quant_params_file` 与 `embedding_*`。
3. 用官方 `npu_tuned_model` 导出友好 ONNX。
4. 用 `to_omc.sh` 对应命令（带 `--compress_conf`、`--input_shape/--dynamic_dims/--input_type/--output_type`）得到 `.omc` + `SubGraph_0.weight`。
5. 拼 7 文件集 → `.hllm`。
6. Runtime 侧用 `LLM Engine` 消费，验证生成与流式。
7. 真机 benchmark（首 token、prefill tps、decode tps、峰值内存、加载耗时）。

第二阶段：
- 追加支持 `Qwen2.5-7B-Instruct` / `Glm-1.5b` 等官方列表模型。
- 只有在官方列表内模型全部 PASS 后，再评估 `Qwen3.8-27B` 与多模态。

---

## 5. 可立即落地（无需硬件）的改动

这些改动不依赖真机/GPU 即可完成，且能显著减少“误导性复杂度”：

1. 删除/降级 `backends/kirin_compat.py` 的图重写逻辑，改为注释说明“官方 NPU 亲和导出路径无需此重写；本文件仅作为非官方回退保留”。
2. `backends/omg.py` 扩展为**按官方模板生成完整 OMG 命令**（`--compress_conf` / `--input_shape` / `--dynamic_dims` / `--input_type` / `--output_type` / `--save_weights_as_external_data`），参数由 profile/模型元数据提供。
3. `backends/quantization.py` 落实为对官方 `dopt` 三段式脚本的封装。
4. Runtime 侧：`InferenceBackend.h` 改为 LLM Engine 生命周期接口，`CannBackend.cpp` 用 `LLM Engine` 实现，并新增 `PackageReader`（`.hllm` 校验/解包）与 `context.json`/`executor.json` 解析。
5. Manifest 追加 `llm_engine` 运行时字段与 7 文件集 artifact 说明。
6. 把 `fp8_to_cann` 标注为 `UNVALIDATED`；把目标模型首位改为官方支持列表内的模型（如 `Qwen3-8B`）。

---

## 6. 参考文献

- CANN LLM 大语言模型解决方案指南（原文，GitCode）：<https://gitcode.com/HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp/blob/master/CANN_LLM/CANN_LLM_Engine_Guide/CANN%20LLM%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88.md>
- 同仓库（Gitee 镜像）：<https://gitee.com/harmonyos_samples/cannkit_samplecode_lm_engine_cpp>
- `CANNLLMEngineDemoNext/entry/src/main/cpp/llm_demo.cpp`（LLM Engine C-API 调用示例）
- LLM Engine API 文档：<https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/api/cannkit-llm-engine>
