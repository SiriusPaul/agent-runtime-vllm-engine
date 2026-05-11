# Codex 协作说明

## 项目概况

当前仓库是 `llama.cpp` fork，根目录为：

```text
D:\Programs\llama
```

当前重点是 `examples/android_osh26` Android 示例工程，用于 OSH26 项目的本地 Agent Runtime 原型。近期目标是先跑通 Android / JNI / native mock 推理链路，再逐步接入真正的 llama.cpp GGUF 推理。

## 当前工作重点

- Android App 能正常启动。
- Kotlin/Java 层能调用 JNI。
- native C++ 层能返回 mock token。
- UI 能展示流式输出。
- 支持 cancel、release、日志和基础状态查看。
- mock 链路稳定后，再把 llama.cpp CMake target 接入 Android 工程。

## 目录入口

- Android 示例工程：`examples/android_osh26`
- Android native 入口：`examples/android_osh26/app/src/main/cpp/native-lib.cpp`
- Android CMake：`examples/android_osh26/app/src/main/cpp/CMakeLists.txt`
- App 主界面：`examples/android_osh26/app/src/main/java/com/example/llamamodifiedforosh26/MainActivity.kt`
- llama.cpp 核心源码：`src`
- 公共代码：`common`
- 文档：`docs`
- 测试：`tests`

## 开发原则

- 先保证最小闭环可运行，再扩展真实模型加载、缓存、调度和性能实验。
- Android demo 应直接引用本仓库中的 llama.cpp 源码，不要再复制一份 llama.cpp 到 Android 项目内。
- 修改应尽量集中在 `examples/android_osh26`，除非确实需要改动 llama.cpp 核心。
- 不要提交 `.gradle`、`build`、`local.properties`、IDE 临时文件等本地生成内容。
- 遇到已有未提交改动时，先确认改动意图，不要覆盖或回滚用户已有工作。

## 常用检查命令

```powershell
git status --short
```

```powershell
cd examples\android_osh26
.\gradlew.bat assembleDebug
```

如果 Gradle 需要联网下载依赖，可能需要用户批准网络访问。

## 后续建议路线

1. 整理 Android 示例工程的 `.gitignore` 和当前未跟踪文件。
2. 实现 JNI mock engine：`loadModel`、`generate`、`cancel`、`release`、`getEngineStats`。
3. 在 UI 中展示 prompt 输入、流式 token 输出、取消按钮和日志状态。
4. 在模拟器上验证启动、回调、取消、释放和日志链路。
5. mock 链路稳定后，再接入 llama.cpp CMake 和真实 GGUF 模型加载。
