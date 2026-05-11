# OSH26 Android llama.cpp 项目交接文档

## 1. 当前项目目标

本项目服务于 OSH-26 课程项目中的 Android 本地 Agent Runtime。目标是在 Android 环境中跑通一个轻量推理引擎原型，为后续 Agent 提供本地小模型推理能力。

当前阶段的重点不是立刻改 llama.cpp 内核，而是先完成最小闭环：

- Android App 能启动。
- JNI 能调用 native C++。
- native 层能流式返回 token。
- 能在模拟器上完成 UI、callback、cancel、日志链路。
- 后续再接入 llama.cpp 真正加载 GGUF 模型。

一句话目标：

**先把 Android/JNI 推理通道跑通，再把 llama.cpp 接进来，最后逐步扩展缓存、调度和性能实验。**

## 2. 当前仓库与目录

llama.cpp fork 仓库位于：

```text
D:\Programs\llama
```

用户新建的空 Android C++ 项目位于：

```text
C:\Users\18259\AndroidStudioProjects\llamamodifiedforOSH26
```

建议把 Android 项目合并进 llama fork，作为 llama.cpp 仓库内的示例工程：

```text
D:\Programs\llama\
  examples\
    android_osh26\
      settings.gradle.kts
      build.gradle.kts
      app\
        build.gradle.kts
        src\main\cpp\
          CMakeLists.txt
          native-lib.cpp
```

合并后，Android Studio 应打开：

```text
D:\Programs\llama\examples\android_osh26
```

不要把 llama.cpp 再复制到 Android 项目里。`D:\Programs\llama` 已经是主仓库，Android demo 应该直接引用同仓库里的 llama.cpp 源码。

## 3. 当前 git 状态

`D:\Programs\llama` 是 git 仓库，当前分支状态曾检查为：

```text
## master...repo-b/master
```

Android Studio 项目目录 `C:\Users\18259\AndroidStudioProjects\llamamodifiedforOSH26` 当前不是 git 仓库。

因此推荐由 `D:\Programs\llama` 统一管理源码、Android demo 和后续改动。

## 4. 推荐合并命令

在 PowerShell 中执行：

```powershell
New-Item -ItemType Directory -Force D:\Programs\llama\examples\android_osh26

Copy-Item C:\Users\18259\AndroidStudioProjects\llamamodifiedforOSH26\* `
  D:\Programs\llama\examples\android_osh26 `
  -Recurse `
  -Exclude .gradle,build,local.properties
```

复制完成后，在 `D:\Programs\llama` 下检查：

```powershell
git status --short
```

预期看到：

```text
?? examples/android_osh26/
```

## 5. Android 开发环境建议

当前阶段使用 Android 模拟器即可，真机或开发板留到性能评测阶段。

推荐模拟器配置：

```text
Device: Pixel 7 或 Pixel 8
API: 35 或 36
ABI: x86_64
Image: Google APIs
```

模拟器适合验证：

- UI。
- JNI。
- native library 加载。
- token callback。
- cancel/release。
- Android Service 或 Activity 生命周期。
- 日志输出。

模拟器不适合做真实性能数据：

- tokens/s。
- TTFT。
- Vulkan/GPU 推理。
- ARM big.LITTLE 调度。
- 温升、降频、功耗。

## 6. llama.cpp 版本策略

当前建议：

- 不追最新 master 的频繁变化。
- 在 fork 中固定一个可工作的 commit/tag。
- 先保证 Android demo 能稳定编译。

之前讨论的建议版本是：

```text
llama.cpp release b8833
```

如果当前 fork 已经基于其他版本，也可以先沿用当前版本，避免一开始引入升级变量。后续需要记录：

```text
llama.cpp source: ggml-org/llama.cpp fork
base version/commit: 待确认
```

## 7. 第一阶段技术路线

第一阶段只做 mock 推理闭环，不急着接真实模型。

目标链路：

```text
Android UI
  -> Java/Kotlin 调用 JNI
  -> native-lib.cpp
  -> mock token loop
  -> callback token
  -> UI 逐步显示
  -> cancel 可中断
  -> logcat 输出 mock TTFT / tokens/s
```

这样可以先把 Android 和 JNI 的问题单独解决，避免 UI、CMake、llama.cpp 推理问题混在一起。

建议 JNI 接口：

```text
loadModel(path: String): Boolean
generate(prompt: String, callback): void
cancel(): void
release(): void
getEngineStats(): String
```

mock 阶段 `loadModel()` 可以只检查路径或直接返回成功，`generate()` 每隔几十到几百毫秒返回一个假 token。

## 8. 第二阶段接入 llama.cpp

mock 链路跑通后，再修改 Android CMake，引用仓库根目录的 llama.cpp。

Android Gradle 需要向 CMake 传入 llama 根目录。示例：

```kotlin
defaultConfig {
    externalNativeBuild {
        cmake {
            arguments += "-DLLAMA_ROOT=${rootDir}/../.."
        }
    }
}
```

在 `examples/android_osh26/app/src/main/cpp/CMakeLists.txt` 中再接入：

```cmake
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_ROOT} llama-build)
```

然后将 app 的 native library 链接到 llama.cpp 对应 target。具体 target 名称需要根据当前 llama.cpp CMake 实际导出的 target 确认，常见可能包括：

```text
llama
ggml
common
```

第一版只启用 CPU backend。Vulkan 暂时不接。

## 9. 推荐模型与测试策略

真接模型时，优先选择小模型 GGUF：

```text
Qwen2.5-0.5B-Instruct GGUF Q4_K_M
Qwen3-0.6B GGUF Q4_K_M
TinyLlama-1.1B GGUF Q4_K_M
```

模拟器只做功能 smoke test，不记录真实性能。

真实评测阶段建议使用：

- 老高通真机，8GB RAM 起步。
- Snapdragon 865 / 870 / 888 均可。
- 优先 OnePlus 8T、Redmi K40 / POCO F3、小米 10。

## 10. 后续研究路线

baseline 跑通后，逐步扩展以下能力：

1. Trace Metrics：记录 TTFT、tokens/s、decode latency、内存等。
2. Prefix Cache：复用 system prompt 和工具模板前缀。
3. Scheduler Lite：支持 2 到 4 个小并发请求。
4. KV Block Manager：做 KV cache 分页、回收和复用。
5. Structured Tool Calling：约束 JSON 输出，服务 Agent 工具调用。
6. Vulkan Backend：在真机或开发板上评估 GPU 加速收益。

推荐优先级：

```text
Baseline > Trace Metrics > Prefix Cache > Scheduler Lite > KV Paging > Tool Calling > Vulkan
```

## 11. 报告中的核心表述

项目表达建议：

```text
本项目基于 llama.cpp/ggml 构建 Android 端轻量推理引擎，吸收 vLLM 在 KV cache 管理、连续批处理和前缀复用方面的设计思想，为本地 Agent Runtime 提供低延迟、可观测、可扩展的小模型推理能力。
```

路线表达：

```text
先用 CPU 跑稳 Android 推理闭环，再逐步加入缓存复用、小并发调度和 Vulkan 增强。
```

技术亮点表达：

```text
用 vLLM 的推理智慧提升端侧效率，用 llama.cpp 的工程底座完成 Android 落地，用可观测指标证明每一步优化的价值。
```

## 12. 新 Codex 终端的建议下一步

新终端进入：

```text
D:\Programs\llama
```

然后执行：

```powershell
git status --short --branch
```

如果 Android 项目还没有复制进来，执行第 4 节的合并命令。

复制后先不要接 llama.cpp，优先完成：

- Android Studio 能打开 `examples/android_osh26`。
- Gradle sync 成功。
- 模拟器能安装并启动 app。
- `native-lib.cpp` 中 mock token 流能通过 JNI 返回到 UI。
- logcat 能看到基础性能日志。

完成这些后，再进入 llama.cpp CMake 接入阶段。

