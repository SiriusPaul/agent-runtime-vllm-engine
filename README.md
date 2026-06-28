# OSH26 Android 本地大模型运行时

本仓库是基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的实验性 fork，当前主要工作集中在 `examples/android_osh26`：在 Android 真机上构建一套可独立运行的本地大模型推理应用，并针对移动端 Vulkan GPU、Qwen3 GGUF 模型、KV Cache、长上下文和低延迟流式输出进行优化。

项目目前已经完成从 Android UI、JNI、llama.cpp 模型解析到自研 Vulkan 推理后端的完整链路。应用既可以直接在手机界面中加载模型和对话，也可以通过本机 HTTP 服务使用 OpenAI 兼容接口调用。

> 当前默认模型为 `Qwen3-1.7B Q8_0`，默认文件名为 `qwen3-1.7b-q8_0.gguf`。

## 项目成果

### 已完成能力

- Android 原生应用，可在真机上完成模型导入、加载、流式生成、取消和缓存重置。
- Java 层通过 JNI 调用 C++ 推理引擎。
- 直接复用本仓库中的 llama.cpp，不在 Android 工程内复制另一份源码。
- 支持 llama.cpp CPU 推理路径和 OSH26 自研 Vulkan GPU Runtime。
- 支持 Qwen3-0.6B 和 Qwen3-1.7B GGUF 模型。
- 支持 Q8_0 权重加载、Q8 动态激活量化和 W8A8 计算。
- Q8 模型使用单份权重驻留，避免同时保留重复的 F32 展开权重。
- Vulkan 生产模式下可跳过 llama.cpp CPU context，降低大模型的重复内存占用。
- GPU prefill、GPU decode attention、GPU KV Cache 更新和 GPU LM head。
- 支持 8K 上下文。
- 支持分页式前缀 KV Cache、LRU 管理和跨请求前缀复用。
- 支持 prefill 分块，避免中等长度 prompt 在 GPU prefill 阶段卡住。
- 支持流式 token 回调及 OpenAI 兼容的 SSE 流式响应。
- 内置 `/health` 性能和状态观测接口。
- 提供 Vulkan 正确性、稳定性和性能验证脚本。

### 当前性能

测试设备为项目开发期间使用的 Android 13、ARM64 真机。性能会随 SoC、GPU 驱动、温度和上下文长度变化。

| 模型 | 优化前 CPU TPS | 当前 median TPS | 提升 | Median decode |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q8_0 | 0.4127 | 5.7498 | +1293.2% | 131.288 ms |
| Qwen3-1.7B Q8_0 | 0.2513 | 3.6408 | +1352.4% | 226.241 ms |

最新提升来自 decode 阶段的 Q8 GEMV fast path：当 `nt == 1` 时，Q/K/V/O/Gate/Up/Down 投影使用 subgroup `GEMV_Q8` kernel，不再使用针对矩阵批处理设计的 `MMQ8` kernel。

两个 Q8 模型均完成 15-request 自动验证：

- 重复请求 token ID 稳定。
- `attention_fallback_layers=0`。
- `last_logits_sanity_ok=true`。
- descriptor allocation median 为 0。
- 未出现 Vulkan device loss、native crash 或输出非有限值。


## 系统架构

```text
Android UI / OpenAI-compatible HTTP API
                    |
                    v
              Java JNI facade
                    |
                    v
              OSH26 C++ Engine
              /              \
             v                v
    llama.cpp CPU path   OSH26 Vulkan Runtime
                              |
          +-------------------+-------------------+
          |                   |                   |
       Q8 GEMV          GPU Attention        GPU LM Head
          |                   |                   |
          +------------- GPU KV Cache ------------+
```

### 推理路径

1. llama.cpp 负责 GGUF 元数据、词表、tokenize、detokenize 和 sampler。
2. OSH26 Runtime 从 GGUF 加载模型权重并建立 Vulkan buffer。
3. Prefill 按 prompt 长度分块执行，当前短分块上限为 32 tokens。
4. Decode 阶段使用 GPU Q8 projection、GPU attention 和 GPU KV Cache。
5. LM head 在 GPU 上计算并归并 top-k candidate。
6. CPU 从 candidate 中执行 llama.cpp sampler，并将生成 token 流式返回 UI 或 HTTP 客户端。

## KV Cache 机制与创新

本项目中的 KV Cache 分为两个层次。第一层是单次请求内的 GPU KV Cache，用于自回归 decode；第二层是跨请求的分页前缀缓存，用于复用重复 system prompt、对话模板或公共上下文的 prefill 结果。

| 层次 | 生命周期 | 主要作用 | 当前实现 |
| --- | --- | --- | --- |
| 活跃 KV Cache | 单次生成请求 | 保存已处理 token 的 Key/Value，decode 时只计算新 token | FP16 GPU buffer，支持 8K context |
| 分页前缀缓存 | 跨多个请求 | 恢复公共 prompt 前缀的 KV，跳过对应 prefill | 16-token page、LRU、GPU 复制、严格 token 前缀匹配 |

### 请求内 KV Cache

Transformer 在生成第 `t` 个 token 时，需要让新 Query 关注此前所有 token。若不缓存历史 Key/Value，每生成一个 token 都要重新计算完整前缀，计算量会随序列长度快速增长。OSH26 在 prefill 时将每层生成的 K/V 直接写入 GPU cache，decode 时只为新 token 计算 Q/K/V，并让 attention 读取历史 cache：

```text
Prefill(prompt)
  -> 计算 prompt 的 K/V
  -> 写入 GPU active KV cache

Decode(token_t)
  -> 只计算 token_t 的 Q/K/V
  -> Q_t 读取 [0, t] 范围的历史 K/V
  -> 新 K/V 追加到 cache
```

Key 和 Value 采用不同的 GPU 排布，以匹配 attention 的实际读取方向：

```text
Key   [kvHeadNum, headDim / 4, maxLen].vec4
Value [kvHeadNum, maxLen, headDim / 4].vec4
```

这种布局减少了 shader 中的地址转换和非连续访问。KV update、attention 和后续 decode 均在 Vulkan 路径内完成，不需要每 token 将完整 KV 搬回 CPU。

### 跨请求分页前缀缓存

仅有请求内 KV Cache 仍会在每次新请求中重复计算 system prompt 和公共上下文。项目额外维护一个独立于 active KV cache 的 GPU 前缀页池，在请求结束后保存可复用的 KV page，在后续请求开始前恢复最长公共前缀。

```text
新请求 token
  -> 查找严格相同的最长 token 前缀
  -> 将可复用长度向下对齐到 16-token page
  -> 从 GPU prefix pool 恢复 K/V 到 active KV cache
  -> 只 prefill 未命中的后缀
  -> decode
  -> 将页对齐前缀写回 prefix pool
```

当前策略有以下正确性约束：

- 只接受 token id 完全一致的公共前缀，不做文本、模糊或语义匹配。
- 可复用长度按 16 tokens 向下对齐，避免维护任意长度的碎片页。
- 始终保留 prompt 的最后一个 token 重新计算，确保 logits 和生成边界来自本次 forward。
- 动态条目默认最多缓存 128 tokens；subagent pinned 条目默认最多缓存 256 tokens。
- 前缀页池默认 32 pages，可通过 Android property 回退到更小容量。
- Debug correctness、非 Vulkan 路径或不支持的状态会自动跳过前缀复用。

### Stateless Subagent Cache

端侧 Agent 的 subagent 通常不依赖历史上下文，只依赖一段固定 system prompt 和当前任务。HTTP `/v1/chat/completions` 会识别包含 `YAML action flow` 和 `tool: return` 的 system prompt，并进入 `stateless_subagent` 模式：

```text
system: 固定 subagent 工具协议
user:   当前任务
```

该模式只缓存固定 system prompt 和 chat template 的 user-role 起始部分，不缓存每次 user task，避免一次性任务污染 cache。第一次请求完成后写入 pinned prefix，后续相同 system prompt、不同 user task 的请求可以直接恢复这段 KV。

可用属性：

```powershell
adb shell setprop debug.osh26.prefix_cache_pages 32
adb shell setprop debug.osh26.prefix_cache_max_entry_tokens 128
adb shell setprop debug.osh26.prefix_cache_max_pinned_tokens 256
adb shell setprop debug.osh26.prefix_cache_max_entries 8
```

### 缓存管理与调度创新

| 设计 | 基础实现常见做法 | 本项目实现 | 直接收益 |
| --- | --- | --- | --- |
| 最长可复用前缀 | 只匹配完整 prompt 或最近条目 | 遍历缓存并选择 page 对齐后复用 token 数最多的条目 | 尽可能减少实际 prefill 长度 |
| 逻辑条目与物理页分离 | 每条记录持有连续大块缓存 | 条目只记录 token 和 page slot，物理页由共享池管理 | 回收简单，内存上限明确 |
| LRU 与去重 | 缓存满后整体清空或重复存储 | 相同前缀更新已有条目；不足时逐条淘汰最久未使用项并回收页 | 提高有限 GPU 内存的命中效率 |
| Subagent pinned prefix | 每次重复计算固定工具协议 | 固定 system prompt 写入 pinned entry，不参与普通淘汰 | 降低 subagent 首 token 等待 |
| Cache-aware scheduling | 请求严格 FIFO 执行 | 队列优先选择可复用前缀最长的请求，同分时按 FIFO | 批量请求中优先兑现缓存收益，同时保留公平性 |
| GPU resident store/restore | KV 经 CPU 暂存和恢复 | active cache 与 prefix pool 之间使用 GPU buffer copy，可选批量复制 | 降低 CPU 往返和 host bookkeeping |
| 可观测性 | 只记录是否命中 | 暴露 hit ratio、复用 token/page、淘汰、碎片率及 restore/store 时间 | 可以量化缓存对 TTFT 的实际贡献 |

前缀缓存的主要收益是降低重复上下文的 TTFT，而不是直接提高单个 decode token 的算力上限。命中后，`n_pos` 从已恢复的 token 数继续推进，GPU 只执行剩余后缀的 prefill。对于固定 system prompt、多轮 Agent 模板和重复知识前缀，这一机制能够把大量重复矩阵计算替换为页级 GPU copy。

`/health` 提供以下关键字段用于验证：

- `last_prefix_cache_hit`、`last_prefix_tokens`、`last_prefix_restore_ms`。
- `prefix_cache_hits`、`prefix_cache_misses`、`prefix_cache_evictions`。
- `prefix_cache_reuse_tokens`、`prefix_cache_reuse_blocks`。
- `prefix_cache_hit_ratio`、`prefix_cache_block_reuse_ratio`、`prefix_cache_fragmentation`。
- `prefix_cache_pool_pages`、`prefix_cache_used_pages`、`prefix_cache_free_pages`。
- `last_prompt_mode`、`subagent_prefix_tokens`、`subagent_prefix_warm_state`。
- `prefix_cache_pinned_entries`、`prefix_cache_dynamic_entries`。

### 当前默认优化

- Q8 prefill/decode：开启。
- Decode Q8 GEMV：默认开启。
- LM head Q8 tied-weight path：开启。
- Descriptor cache：开启。
- 前缀 KV Cache：开启。
- Single submit：保留实验开关，默认关闭。

可通过以下方式临时关闭 decode GEMV，回退到旧的 `MMQ8` decode 路径：

```powershell
adb shell setprop debug.osh26.decode_q8_gemv 0
```

恢复默认：

```powershell
adb shell setprop debug.osh26.decode_q8_gemv 1
```

## 仓库结构

```text
.
|-- common/                         llama.cpp 公共工具和模型辅助代码
|-- docs/                           llama.cpp 与后端相关文档
|-- examples/
|   `-- android_osh26/              OSH26 Android 应用和 Vulkan Runtime
|       |-- app/
|       |   |-- build.gradle.kts    Android app 构建配置
|       |   `-- src/main/
|       |       |-- AndroidManifest.xml
|       |       |-- java/org/osh26/llama/
|       |       |   |-- MainActivity.java   UI、模型导入和流式对话
|       |       |   |-- LlamaNative.java    JNI 接口声明
|       |       |   `-- LlmHttpServer.java  本机 HTTP/OpenAI API
|       |       `-- cpp/
|       |           |-- native-lib.cpp       JNI native 入口
|       |           |-- osh26_engine.cpp     调度、采样、缓存和生成流程
|       |           |-- osh26_vk_gpu.c       Vulkan 推理核心
|       |           |-- osh26_vk_gpu.h       Vulkan Runtime API 和统计
|       |           |-- vk_wrapper/          Vulkan 动态加载封装
|       |           |-- *.comp               GLSL compute shader
|       |           `-- *.spv.h              编译后的 SPIR-V 头文件
|       |-- generate_spv_headers.ps1         Shader 编译和头文件生成
|       |-- verify-vulkan-runtime.ps1        真机自动验证脚本
|       |-- PERFORMANCE_NOTES.md              性能实验和基线记录
|       |-- gradlew / gradlew.bat             Gradle Wrapper
|       `-- settings.gradle.kts
|-- ggml/                           llama.cpp 张量和后端基础设施
|-- src/                            llama.cpp 核心实现
|-- tests/                          上游及公共测试
|-- CMakeLists.txt                  根 CMake 构建入口
`-- README.md                       本文档
```

Android CMake 通过 `LLAMA_ROOT` 直接引用仓库根目录：

```kotlin
arguments += "-DLLAMA_ROOT=${rootDir}/../.."
```

因此 `examples/android_osh26` 必须保留在当前仓库层级中，不能单独复制出去构建。

## 环境要求

### 开发机

- Windows 10/11。
- Android Studio 或 Android SDK Command-line Tools。
- Android SDK Platform 36。
- Android NDK，支持 ARM64。
- CMake 3.22.1。
- JDK 11 或兼容版本。
- PowerShell 5.1 或更高版本。
- ADB 可用。

### Android 设备

- Android 8.0/API 26 或更高。
- ARM64-v8a。
- 支持 Vulkan Compute，至少Vulkan1.1。
- 有足够的可用内存和存储空间。

Qwen3-1.7B Q8_0 文件约 1.8 GB，加载后还需要权重、KV Cache、activation、LM head buffer 和驱动侧资源。建议使用内存较充足的机器，并避免同时运行高内存应用。

## 构建与安装

### 1. 检查设备

```powershell
adb devices
```

应看到状态为 `device` 的 Android 设备：

```text
List of devices attached
xxxxxxxx    device
```

### 2. 进入 Android 工程

```powershell
cd examples\android_osh26
```

### 3. 构建 Debug APK

```powershell
.\gradlew.bat assembleDebug
```

生成的 APK 位于：

```text
examples/android_osh26/app/build/outputs/apk/debug/app-debug.apk
```

### 4. 安装到设备

```powershell
.\gradlew.bat installDebug
```

也可以直接使用 ADB：

```powershell
adb install -r -t .\app\build\outputs\apk\debug\app-debug.apk
```

### 5. 启动应用

```powershell
adb shell am start -n org.osh26.llama/.MainActivity
```

## 准备和加载模型

应用默认寻找：

```text
/data/data/org.osh26.llama/files/models/qwen3-1.7b-q8_0.gguf
```

### 方式一：在 UI 中导入

1. 启动应用。
2. 点击模型导入按钮。
3. 从 Android 文件选择器中选择 `.gguf` 文件。
4. 应用会将模型复制到自己的 `files/models` 目录。
5. 确认模型路径后点击 `Load Model`。

这是日常使用最简单的方式。

### 方式二：通过 ADB 部署

先将模型上传到临时目录：

```powershell
adb push D:\Models\qwen3-1.7b-q8_0.gguf /data/local/tmp/qwen3-1.7b-q8_0.gguf
adb shell chmod 644 /data/local/tmp/qwen3-1.7b-q8_0.gguf
```

创建应用模型目录并复制文件：

```powershell
adb shell run-as org.osh26.llama mkdir -p files/models
adb shell run-as org.osh26.llama cp /data/local/tmp/qwen3-1.7b-q8_0.gguf files/models/qwen3-1.7b-q8_0.gguf
```

检查文件：

```powershell
adb shell run-as org.osh26.llama ls -lh files/models
```

项目验证脚本识别的默认模型文件如下：

| 模型 | 设备路径 |
| --- | --- |
| Qwen3-0.6B full | `/data/data/org.osh26.llama/files/models/qwen3-0.6b.gguf` |
| Qwen3-0.6B Q8_0 | `/data/data/org.osh26.llama/files/models/qwen3-0.6b-base-q8_0.gguf` |
| Qwen3-1.7B Q8_0 | `/data/data/org.osh26.llama/files/models/qwen3-1.7b-q8_0.gguf` |

模型文件不应提交到 Git。

## 开始推理

### 使用 Android UI

1. 启动应用。
2. 确认模型路径。
3. 点击 `Load Model` 并等待加载完成。
4. 在消息输入框中输入问题。
5. 点击发送按钮。
6. UI 会先显示 tokenize、prefill 和 first-token 状态，随后逐 token 展示回答。

界面还提供：

- 取消当前生成。
- 重置 KV Cache 和会话上下文。
- 启动或停止本机 HTTP 服务。
- 查看 engine、Vulkan、缓存和性能状态。

### 使用 OpenAI 兼容 HTTP API

应用启动时默认监听设备本机：

```text
127.0.0.1:8000
```

服务仅绑定 loopback，不直接暴露到局域网。开发机通过 ADB forward 访问：

```powershell
adb forward tcp:8000 tcp:8000
```

#### 检查服务状态

```powershell
curl.exe http://127.0.0.1:8000/health
```

#### 加载 Vulkan 模型

PowerShell：

```powershell
$body = @{
    path = "/data/data/org.osh26.llama/files/models/qwen3-1.7b-q8_0.gguf"
    backend = "vulkan"
    n_gpu_layers = -1
    debug_correctness = $false
} | ConvertTo-Json -Compress

Invoke-RestMethod `
    -Method Post `
    -Uri http://127.0.0.1:8000/load_model `
    -ContentType "application/json" `
    -Body $body
```

#### 非流式对话

```powershell
$body = @{
    model = "local-gguf"
    max_tokens = 128
    temperature = 0.6
    top_p = 0.95
    seed = 51966
    thinking = $false
    messages = @(
        @{
            role = "user"
            content = "请简单介绍本地大模型推理的优势。"
        }
    )
} | ConvertTo-Json -Depth 6 -Compress

Invoke-RestMethod `
    -Method Post `
    -Uri http://127.0.0.1:8000/v1/chat/completions `
    -ContentType "application/json" `
    -Body $body
```

#### curl 示例

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"local-gguf","max_tokens":64,"temperature":0.6,"top_p":0.95,"thinking":false,"messages":[{"role":"user","content":"Explain local inference briefly."}]}'
```

请求中加入 `"stream": true` 可获得 SSE 流式响应。

### 其他 HTTP 接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` | Engine、Vulkan、缓存和性能状态 |
| GET | `/v1/models` | 返回 `local-gguf` 模型列表 |
| POST | `/load_model` | 配置后端并加载 GGUF |
| POST | `/v1/chat/completions` | OpenAI 兼容对话接口 |
| POST | `/cancel` | 请求取消生成 |
| POST | `/reset_cache` | 清空 KV Cache |
| POST | `/benchmark/quant_gemv` | 量化 GEMV benchmark |
| POST | `/benchmark/quant_gemm` | 量化 GEMM benchmark |
| POST | `/benchmark/q8_gemm` | Q8 GEMM benchmark |

## 运行验证

### 基础健康检查

```powershell
adb forward tcp:8000 tcp:8000
curl.exe -s http://127.0.0.1:8000/health
```

重点检查：

```text
engine.model_loaded
engine.running
engine.last_error
engine.last_ttft_ms
engine.last_tokens_per_second
engine.last_logits_sanity_ok
engine.vulkan.ready
engine.vulkan.attention_fallback_layers
engine.vulkan.decode_q8_gemv_enabled
engine.vulkan.last_decode_q8_gemv_used
```

### 自动验证脚本

在 Android 工程目录执行：

```powershell
cd examples\android_osh26
```

验证 Qwen3-1.7B Q8_0：

```powershell
$env:OSH26_SINGLE_SUBMIT = "1"
.\verify-vulkan-runtime.ps1 -RunModel17BQ8 -DecodeQ8Gemv -SkipInstall
```

验证 Qwen3-0.6B Q8_0：

```powershell
$env:OSH26_SINGLE_SUBMIT = "1"
.\verify-vulkan-runtime.ps1 -RunModel06BQ8 -DecodeQ8Gemv -SkipInstall
```

同时验证两个 Q8 模型：

```powershell
$env:OSH26_SINGLE_SUBMIT = "1"
.\verify-vulkan-runtime.ps1 `
    -RunModel06BQ8 `
    -RunModel17BQ8 `
    -DecodeQ8Gemv `
    -SkipInstall
```

若尚未安装最新 APK，去掉 `-SkipInstall`。

验证结果保存到：

```text
examples/android_osh26/verify-results/verify-results-YYYYMMDD-HHMMSS.json
```

脚本还会向以下文件追加性能摘要：

```text
examples/android_osh26/PERFORMANCE_NOTES.md
```

### 验证门槛

当前 Q8 生产路径主要检查：

- Vulkan Runtime 正常加载。
- 无 native/Vulkan error。
- `attention_fallback_layers=0`。
- top-k token 合法、无重复且 logit 按降序排列。
- `last_logits_sanity_ok=true`。
- 重复请求 token ID 稳定。
- descriptor allocation median 为 0。
- 开启 decode GEMV 时 `last_decode_q8_gemv_used=true`。
- 1.7B decode GEMV median TPS 不低于 2.55。
- TPS、TTFT 和 LM head 不超过基线允许的回退范围。

### 查看日志

```powershell
adb logcat -d -t 500 `
  OSH26GPU:I `
  OSH26Llama:I `
  OSH26Main:I `
  OSH26HTTP:I `
  AndroidRuntime:E `
  "*:S"
```

清空日志后重新复现：

```powershell
adb logcat -c
```

## Shader 开发

GLSL shader 位于：

```text
examples/android_osh26/app/src/main/cpp/*.comp
```

修改 shader 后需要重新生成 SPIR-V 头文件：

```powershell
.\examples\android_osh26\generate_spv_headers.ps1
```

随后重新构建 APK：

```powershell
cd examples\android_osh26
.\gradlew.bat assembleDebug
```

不要只修改 `.comp` 而遗漏对应的 `.spv.h`，否则设备上运行的仍可能是旧 shader。

## 开发阶段


| 日期 | 阶段 | 主要工作 | 阶段结果 |
| --- | --- | --- | --- |
| 2026-05-11 | Android 工程初始化 | 创建 Gradle/App/Manifest/native 工程结构，确定 `org.osh26.llama` 包名和 ARM64 目标 | 建立 Android、JNI 和 CMake 的最小工程骨架 |
| 2026-05-12 | 模拟器推理与 API | 在模拟器运行 Qwen3-0.6B，打通 Java/JNI/llama.cpp，增加 thinking 配置和 OpenAI 兼容接口 | UI 与 `/v1/chat/completions` 均可触发本地生成 |
| 2026-05-14 | 真机 CPU 推理 | 将 GGUF 加载和生成迁移到 ARM64 设备，验证 CPU backend 和 UI 输出 | 完成真实设备上的端到端 CPU 基线 |
| 2026-05-17 至 05-18 | Vulkan 初步接入 | 增加 backend 和 GPU layer 配置，首次加载 Android Vulkan 路径 | 模型可加载，但暴露输出乱码和数值正确性问题 |
| 2026-05-24 | 正确性排查 | 检查 token、logit、detokenize 和 GPU 中间结果，补充诊断日志 | 建立后续 kernel 对比和错误定位基础 |
| 2026-05-27 | 自研 Vulkan Runtime | 增加 Qwen3 compute shader，适配 Vulkan 动态加载和 GPU attention，建立 buffer/pipeline/descriptor/command 基础设施 | 自研 GPU 推理主干成形 |
| 2026-06-02 | 输出修复 | 修复乱码、token 和数值链路问题 | GPU 输出恢复为可用文本 |
| 2026-06-03 | KV Cache、Prefill、LM Head | 去除生产路径 CPU 对比；增加 GPU active KV cache、LRU 缓存块、chunked prefill 和 LM head 优化；并行化 matmul/shader | 速度提升到约 1.5 TPS，并具备请求内历史 K/V 复用能力 |
| 2026-06-04 | GPU Prefill 与稳定性 | 将 prefill 主路径迁移到 GPU，优化 TTFT，修复生成卡住问题 | Prefill 和 decode GPU 链路趋于稳定 |
| 2026-06-06 | 原生 Q8_0 | 支持 Q8_0 权重、动态 activation quantization 和 W8A8 kernel；优化 LM head/submit；建立量化正确性验证 | 降低权重带宽与加载开销，为更多量化格式打下基础 |
| 2026-06-07 | Q8 Decode、分页前缀缓存、8K | Decode 全面接入原生 Q8_0；实现严格 token 前缀匹配、16-token page、LRU 回收和跨请求 KV 复用；扩展 8K context | 公共 system prompt 可跳过重复 prefill，并完成超过 1024 tokens 验证 |
| 2026-06-08 | 1.7B 与内存优化 | Q8 权重改为单份驻留，模型维度参数化，支持 Qwen3-1.7B；移除 Vulkan 生产模式的重复 CPU 权重/context | 默认 UI 模型切换为 Qwen3-1.7B Q8_0，显著控制内存占用 |
| 2026-06-09 | Submit、Descriptor、Profiling | 合并 decode layer command，引入 descriptor cache 和 single-submit 实验；补全 GPU timestamp；将短 prefill chunk 限制为 32 | 稳态 descriptor allocation 降为 0，定位并修复 1.7B 中等长度 prefill 卡住问题 |
| 2026-06-10 | Decode Q8 GEMV | 为 `nt == 1` 的 Q/K/V/O/Gate/Up/Down 启用 subgroup Q8 GEMV，补充各 decode 阶段 profiling | 0.6B median 达 5.7498 TPS，1.7B median 达 3.0408 TPS，并将该路径设为默认 |

## 当前瓶颈与后续方向

在 1.7B Q8 decode GEMV 路径中，当前单 token GPU 时间主要分布为：

| 阶段 | Median GPU 时间 |
| --- | ---: |
| FFN Gate/Up | 62.2 ms |
| LM Head | 50.3 ms |
| FFN Down | 37.7 ms |
| Attention | 28.1 ms |
| QKV | 23.3 ms |
| O Projection | 10.8 ms |

下一阶段优先级：

1. 优化或融合 FFN Gate/Up 与 activation quantization。
2. 优化 FFN Down 的 Q8 GEMV 内存访问。
3. 继续降低 LM head 的 wall time 和 host wait。
4. 评估 attention 在上下文增长后的带宽瓶颈。
5. 保持 Q8 正确性和稳定性门槛，再考虑 Q4_K/Q6_K。

## 已知限制

- 当前 Android APK 仅构建 `arm64-v8a`。
- 自研 Vulkan Runtime 主要针对当前支持的 Qwen3 结构和维度。
- 8K 上下文可用，但长上下文 prefill 仍然较慢且内存压力较高。
- Single submit 是实验路径，默认关闭。
- GPU sampler 尚未实现，每 token 仍需 CPU 读取 top-k candidate 并采样。
- 不同 Vulkan 驱动的 subgroup、内存和 timestamp 行为可能不同。
- Q8 GEMV 使用并行归约，浮点累加顺序与旧 GEMM kernel 不同；生成结果应以新路径自身的稳定性和正确性门槛验证。

## 清理与注意事项

- 不要提交 GGUF 模型。
- 不要提交 `.gradle`、`build`、`local.properties` 或 IDE 临时文件。
- 不要覆盖工作区中已有的未提交实验。
- 性能测试前应确认设备温度、后台应用和 Vulkan property 状态。
- verifier 运行结束后会关闭 single-submit，并恢复 decode Q8 GEMV 默认开启状态。
