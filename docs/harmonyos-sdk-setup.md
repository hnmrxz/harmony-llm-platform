# HarmonyOS SDK 安装与构建环境说明

本项目电脑端 Runtime（`harmony-llm-runtime`）需要在 HarmonyOS 侧编译，前提是有一份可用的 **HarmonyOS SDK**。

## 1. 当前本机状态（已核实）

本机 `devecocli` 已安装：

```text
@deveco/deveco-cli   1.2.0-stable
路径：~/.nvm/versions/node/v24.19.0/bin/devecocli
```

但**本机没有 HarmonyOS SDK**，且该 CLI 在 Linux 上无法自动检测 DevEco Studio：

```text
$ devecocli device list
Error: Linux is not fully supported yet for automatic DevEco Studio detection.

$ devecocli create --app-name Test ...     # 同样报错
$ devecocli build ...                       # 需要 DevEco Studio / SDK 检测
```

原因：该 CLI 的 `findDevEcoStudio()` 只检测 **Windows 注册表** 与 **macOS /Applications**，在 Linux 直接抛错，且**没有环境变量可绕过**（`DEVECO_HOME` / `DEVECO_PATH` 只用于 skills/MCP 配置，不影响 DevEco Studio 检测）。

另外，**SDK 本身**与 Command Line Tools 的下载在华为开发者站点需要**华为开发者账号登录**。

结论：**在这台 Linux 主机上无法用 devecocli 完成真机构建/运行**。需要一台装有 DevEco Studio（Windows / Mac，含 SDK）的机器，或在该类机器上完成 SDK 安装后构建。

## 2. 推荐路径（A）：DevEco Studio（Windows / Mac）

1. 安装 DevEco Studio（含 SDK 管理器）。
2. 打开本仓库 `harmony-llm-runtime`。
3. 在 SDK 管理器中选择匹配的 SDK / API 版本。
4. 同步依赖（ohpm install）后构建。
5. 复制 CANN Kit 的头文件与 `libhiai_llm_engine.so` 到 `entry/src/main/cpp/include`、`entry/src/main/cpp/lib64`（见 Runtime README 第六节）。

`build-profile.json5` 的 `targetSdkVersion` / `compatibleSdkVersion` 需改为你实际安装的 SDK 版本。

## 3. 推荐路径（B）：Command Line Tools + ohsdkmgr（Linux/CI）

华为官方提供 Command Line Tools，内含 `ohsdkmgr`（SDK 管理器），可在无 DevEco Studio 的机器上装 SDK。

```bash
# 1. 从华为开发者站点下载 DevEco Studio Command Line Tools（需华为账号登录）
#    https://developer.huawei.com/consumer/cn/doc/deveco-studio/ide-commandline-get
#    解压后包含 ohsdkmgr / hvigor / ohpm 等。

# 2. 用 ohsdkmgr 安装目标 API 的 SDK，例如：
./ohsdkmgr install --product HarmonyOS --version <api-version>
./ohsdkmgr sdk-version                       # 查看已安装版本
# 详细用法见：
#   https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/ide-command-line-ohsdkmgr

# 3. 导出 SDK 根目录，devecocli 的 build/run 会用 DEVECO_SDK_HOME
export DEVECO_SDK_HOME=/path/to/ohos-sdk
```

> 说明：`ohsdkmgr install` 需要华为账号登录（下载与授权）。这一授权动作必须由你本人完成，我无法代做。

## 4. 安装完成后验证

```bash
devecocli --version
devecocli device list                 # 应能列出真机/模拟器
cd harmony-llm-runtime
devecocli build --modules entry --build-mode debug
```

## 5. 本机可独立验证的部分（无需 SDK）

- Converter：`cd harmony-llm-converter && source .venv/bin/activate && pytest`
- Runtime 原生导入链路：`cd harmony-llm-runtime && bash test/run_host_test.sh`
- 原生 C++ 语法检查：`c++ -std=c++17 -fsyntax-only -I native <file>.cpp`

这两条已在本机通过：Runtime 导入链路（zip 解包 → manifest → SHA-256 → 定位引擎文件）在 ASan/UBSan 下全绿；Converter 官方路径（layout / OMG 参数 / dopt）测试通过。
