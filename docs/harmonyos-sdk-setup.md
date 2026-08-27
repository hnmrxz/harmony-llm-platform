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

## 3. 已确认可用的 Linux 路径：Command Line Tools（SDK 内嵌）

> 你已提供 `commandline-tools-linux-x64-6.1.1.300.zip`（HarmonyOS 6.1.1 / API 24），**SDK 内嵌**在归档中，无需单独的 `ohsdkmgr`。

```bash
# 1. 解压（约 6.5 GB）
unzip commandline-tools-linux-x64-6.1.1.300.zip -d /home/jat/Develop/deveco-clt

# 2. 配置环境（或 source harmony-llm-runtime/build-env.sh）
export DEVECO_SDK_HOME=/home/jat/Develop/deveco-clt/command-line-tools/sdk
export DEVECO_NODE_HOME=/home/jat/Develop/deveco-clt/command-line-tools/tool/node
export PATH=/home/jat/Develop/deveco-clt/command-line-tools/bin:$DEVECO_NODE_HOME/bin:$PATH

# 3. 安装依赖（ohpm 走公开仓库）
cd harmony-llm-runtime
ohpm install

# 4. 构建
hvigorw --mode project -p product=default -p buildMode=debug assembleApp
```

**已实测：** 原生模块（OHOS Clang 交叉编译）与 ArkTS 均在 Linux 上编译通过；唯一需要的是最终 HAP 打包/签名用的 **Java**（`spawn java ENOENT`）。请安装一个 JRE/JDK 并让 `java` 在 PATH 上（DevEco Studio 自带，CLT 不附带）。

> 注意：`devecocli`（DevEco Studio 检测）在 Linux 上不可用；直接用上面这套 `hvigorw` / `ohpm` 即可。`devecocli` 只在 Windows/Mac 下用于真机/模拟器。

## 4. 版本与产物

- 工具链版本：`HarmonyOS 6.1.1.300`（hvigor 6.24.4、ohpm 6.1.2.285、API 24）。
- 项目 `build-profile.json5` 已设为 `targetSdkVersion`/`compatibleSdkVersion = 6.1.1(24)`。
- 产物：`entry/build/default/outputs/default/**/entry-default-signed.hap`。

## 5. 本机可独立验证的部分（无需 SDK）

- Converter：`cd harmony-llm-converter && source .venv/bin/activate && pytest`
- Runtime 原生导入链路：`cd harmony-llm-runtime && bash test/run_host_test.sh`
- 原生 C++ 语法检查：`c++ -std=c++17 -fsyntax-only -I native <file>.cpp`

这两条已在本机通过：Runtime 导入链路（zip 解包 → manifest → SHA-256 → 定位引擎文件）在 ASan/UBSan 下全绿；Converter 官方路径（layout / OMG 参数 / dopt）测试通过。
