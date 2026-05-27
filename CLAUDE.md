# CLAUDE.md

## 项目概述

本仓库是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 fork，服务于 **OSH26 课程项目** 的 Android 本地 Agent Runtime。目标是在 Android 设备上运行轻量推理引擎，为 Agent 提供本地小模型推理能力。

当前分支: `osh26-runtime`

## 当前进展

- **已完成**: 红米 K40 真机 CPU 推理跑通（Qwen3-0.6B GGUF），支持流式输出、cancel、release、HTTP server（兼容 OpenAI API）
- **进行中**: Vulkan GPU 推理 —— CMake 已配置 `GGML_VULKAN ON`，vendor 了 Vulkan-Hpp + SPIRV-Headers，shader 编译通过，当前卡在链接阶段 `vkGetPhysicalDeviceFeatures2` 符号缺失；已把 `minSdk` 从 24 提到 26 但问题尚未解决

## 目录结构

```
D:\Programs\llama
├── examples/android_osh26/          # Android 示例工程（唯一关注的子项目）
│   ├── app/
│   │   ├── build.gradle.kts          # minSdk=26, targetSdk=36, 传 LLAMA_ROOT 给 CMake
│   │   └── src/main/
│   │       ├── cpp/
│   │       │   ├── CMakeLists.txt    # Android CMake，GGML_VULKAN=ON，链接 llama
│   │       │   ├── native-lib.cpp    # JNI 入口
│   │       │   ├── osh26_engine.cpp  # 推理引擎核心：加载、生成、取消、统计
│   │       │   └── osh26_engine.h    # ComputeBackend + SchedulerLite 接口
│   │       └── java/org/osh26/llama/
│   │           ├── MainActivity.java # 主 Activity：加载模型、生成、HTTP server 控制
│   │           ├── LlamaNative.java  # JNI native 方法声明
│   │           └── LlmHttpServer.java# 内嵌 HTTP server（127.0.0.1:8000，兼容 /v1/chat/completions）
│   └── settings.gradle.kts
├── ggml/src/ggml-vulkan/            # ggml Vulkan 后端（shader + C++ 实现）
├── vendor/
│   ├── vulkan-headers/include/      # Vendored Khronos Vulkan-Hpp (vulkan.hpp 等)
│   └── spirv-headers/include/       # Vendored SPIR-V headers
├── src/                             # llama.cpp 核心
├── common/                          # llama.cpp 公共代码
├── AGENTS.md                        # Codex 协作说明（旧版，可能过时）
├── OSH26_HANDOFF.md                 # 项目交接文档
└── vllm_core_android_llamacpp_plan.md # vLLM 能力移植方案（长期路线）
```

## 架构

```
Android UI (MainActivity)
  → LlamaNative (JNI)
    → native-lib.cpp (JNI bridge)
      → osh26::SchedulerLite → ComputeBackend (osh26_engine.cpp)
        → llama.cpp / ggml
          → CPU backend (working)
          → Vulkan backend (WIP, linking broken)
```

### JNI 接口

| 方法 | 功能 |
|------|------|
| `loadModel(path)` | 加载 GGUF 模型，返回状态字符串 |
| `generateStream(prompt, callback, ...)` | 流式生成，回调 onToken/onComplete/onError |
| `generateBlockingJson(prompt, ...)` | 阻塞生成，返回 JSON |
| `cancel()` | 取消当前生成 |
| `release()` | 释放模型和上下文 |
| `getEngineStats()` | 返回引擎状态 JSON（backend, devices, tokens/s 等）|

### ComputeBackend 关键逻辑

- `loadModel()`: 优先尝试 Vulkan（`n_gpu_layers=-1`），失败自动回退 CPU（`n_gpu_layers=0`）
- `generate()`: Qwen3 模板拼接 → tokenize → decode loop → sampler → callback
- 模型路径: `{filesDir}/models/qwen3-0.6b.gguf`（app 私有目录）
- 备选路径: `/sdcard/Android/data/org.osh26.llama/files/models/`

### HTTP API

内嵌 HTTP server 监听 `127.0.0.1:8000`：
- `GET /health` — 引擎状态
- `GET /v1/models` — 模型列表
- `POST /load_model` — 加载模型 `{"path": "..."}`
- `POST /v1/chat/completions` — 兼容 OpenAI chat completions（支持 stream）

## Vulkan 构建链

1. CMake 变量: `GGML_VULKAN ON`，其余 backend 均 OFF
2. 第三方头: `vendor/vulkan-headers/include` + `vendor/spirv-headers/include`
3. glslc: 从 NDK `shader-tools/windows-x86_64/glslc.exe` 显式指定
4. Android CMake 把 vendor headers 注入 `ggml-vulkan` target 的 include path

### 已知构建问题

- **minSdk 24 → 26**: `vkGetPhysicalDeviceFeatures2` 需要 Vulkan 1.1，API 24 的 `libvulkan.so` 不包含。已提升 minSdk 到 26（`app/build.gradle.kts`），但可能还需要在 CMake 侧清理缓存重建
- **glslc 找不到**: CMake 默认找不到 NDK 的 glslc，已在 CMakeLists.txt 中显式设置 `Vulkan_GLSLC_EXECUTABLE`
- **vulkan.hpp 缺失**: Android NDK 只有 C 头，已 vendoring Khronos Vulkan-Hpp 到 `vendor/vulkan-headers/`

## 常用命令

```powershell
# 构建 Android APK
cd examples\android_osh26
.\gradlew.bat assembleDebug

# 清理构建缓存后重建（Vulkan 构建问题时常需要）
Remove-Item -Recurse -Force examples\android_osh26\app\.cxx
.\gradlew.bat assembleDebug

# 查看 git 状态
git status --short --branch

# 安装到真机（红米 K40）
adb install -r examples\android_osh26\app\build\outputs\apk\debug\app-debug.apk

# 推送模型到真机
adb push qwen3-0.6b.gguf /data/local/tmp/
adb shell "run-as org.osh26.llama cp /data/local/tmp/qwen3-0.6b.gguf /data/data/org.osh26.llama/files/models/"
```

## 开发原则

- 修改集中在 `examples/android_osh26`，除非确实需要改 llama.cpp 核心
- Android demo 直接引用仓库根目录的 llama.cpp/ggml 源码，不复制
- CPU-first: 先保证 CPU 推理稳定，Vulkan 是增量增强
- 不提交 `.gradle`、`build`、`local.properties`、`.cxx`、IDE 临时文件
- 真机测试设备: 红米 K40 (Snapdragon 870, 8GB+ RAM, Adreno 650)

## ✅ 已验证的事实
- GPU MUL_MAT 完全正确：Q、K、V、O、Gate、Up、Down 所有 matmul 与 CPU reference 误差 < 1e-6。不要再验证。
- 权重加载正确：GGUF tensor 名字匹配，first value 非零。

## 🔴 禁止尝试的方案
- **ggml-cpu hook**: 多次尝试均崩溃。多线程环境下 Vulkan command pool 不是线程安全的，即使加 mutex 也不可靠。不要再用 extern function pointer hook MUL_MAT 的方案。
- **ggml backend 注册**: scheduler 不给我们的 backend 分配层，多次尝试无效。

## 长期路线

详见 `vllm_core_android_llamacpp_plan.md`。概括：

```
Baseline → Trace Metrics → Prefix Cache → Scheduler Lite → KV Paging → Tool Calling → Vulkan
```

当前在 Baseline 阶段，即将完成 Vulkan 接入（Baseline 增强项）。