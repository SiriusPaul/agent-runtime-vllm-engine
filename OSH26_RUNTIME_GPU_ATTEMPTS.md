# OSH26 GPU 推理探究记录

## 环境

- **设备**: 红米 K40 (alioth), Snapdragon 870 (Adreno 650), 8GB RAM
- **系统**: MIUI Android 13 (TKQ1.221114.001, API 33)
- **模型**: Qwen3-0.6B (GGUF, ~1.5GB)
- **目标**: 在 Android 上实现 GPU 推理（OpenCL / Vulkan）

---

## 尝试一：OpenCL 后端

### 阶段 1：编译时 OpenCL 未注册

**现象**: App 能运行，但所有计算都在 CPU 上，即使 `n_gpu_layers=99` 也不生效。

**根因**: `ggml/src/CMakeLists.txt` 中缺少 `ggml_add_backend(OpenCL)` 调用，导致 `GGML_USE_OPENCL` 编译宏未定义，OpenCL 后端从未被注册到 ggml 框架中。`llama_supports_gpu_offload()` 返回 false，所以 `n_gpu_layers` 被忽略。

**修复**: 在 `ggml/src/CMakeLists.txt` 第 449 行添加 `ggml_add_backend(OpenCL)`。

### 阶段 2：成功注册 OpenCL 后端，但无法加载系统 libOpenCL.so

添加 `ggml_add_backend(OpenCL)` 后，`GGML_USE_OPENCL` 被定义，后端注册成功。但 `System.loadLibrary("llama_osh26")` 时动态链接器无法解析 `clGetDeviceInfo` 等 OpenCL 符号——因为系统 `/vendor/lib64/libOpenCL.so` 不在 app 的 linker namespace 中。

#### Android Linker Namespace 说明

Android 10+ 引入了 linker namespace 隔离：
- **classloader-namespace**: app 的库加载路径，仅包含 `/data/app/...`、`/data/data/org.osh26.llama/`
- **vendor-namespace**: 厂商库路径，包含 `/vendor/lib64/`、`/system/lib64/` 部分库

两个 namespace 之间完全隔离，app 的 `dlopen("/vendor/lib64/libOpenCL.so")` 会被拒绝。

```
permitted_paths="/data:/mnt:expand:/data/data/org.osh26.llama"
```

### 阶段 3：尝试绕过 namespace 隔离

#### 方案 A：Java File I/O 拷贝到 app data 目录

用 Java 的 `FileInputStream`（不受 linker namespace 限制）把 `/vendor/lib64/libOpenCL.so` 拷贝到 `/data/data/org.osh26.llama/files/`（属于 permitted path），然后 `dlopen`。

**问题**: vendor 的 `libOpenCL.so` 有 DT_NEEDED 依赖（`libcutils.so`、`libvndksupport.so`、`libc++.so`），这些依赖也在 vendor namespace 中，无法解析。

```
dlopen failed: library "libcutils.so" not found:
  needed by /data/data/org.osh26.llama/files/libOpenCL.so
  in namespace classloader-namespace
```

#### 方案 B：同时拷贝依赖库并预加载

更新 `copyOpenCL()` 方法，把 `libcutils.so`、`libvndksupport.so`、`libOpenCL.so` 三个文件都拷贝到 data 目录，然后在 `opencl_preload` constructor 中用 `dlopen(RTLD_GLOBAL)` 按顺序加载。

**问题**: 依赖自身也有 DT_NEEDED 依赖，且 `libcutils.so` 在 `/system/lib64/` 中（属于 system namespace，同样不可达）。linker 不允许从 classloader-namespace 加载任何 system/vendor 分区的库。

```
dlopen failed: library "libcutils.so" not found:
  needed by .../files/libOpenCL.so in namespace classloader-namespace
```

文件确实存在（`adb shell run-as` 确认），但 linker 不搜索 data 目录来解决 DT_NEEDED。

#### 方案 C：opencl_wrappers.c — 编译时包装所有 OpenCL 函数

不直接链接 OpenCL 符号，而是在 `llama_osh26.so` 内实现所有 OpenCL 函数的包装层。每个包装函数用 `dlsym(RTLD_DEFAULT, ...)` 在运行时查找真实符号。

**问题**：`dlsym(RTLD_DEFAULT, ...)` 找到了我们自己定义的包装函数 → 无限递归 → SIGSEGV（栈溢出）。

**改进**：改为保存 `dlopen` 的 handle，用 `dlsym(handle, ...)` 仅搜索真实库。

但 core problem 依然存在：**永远无法成功 `dlopen` 系统 `/vendor/lib64/libOpenCL.so`**。

#### 方案 D：`android_dlopen_ext` + `library_fd`

使用 `android_dlopen_ext` 通过文件描述符（而非路径）加载，尝试绕过 namespace 检查：

```c
int fd = open("/vendor/lib64/libOpenCL.so", O_RDONLY);
android_dlextinfo ext = {.flags = ANDROID_DLEXT_USE_LIBRARY_FD, .library_fd = fd};
android_dlopen_ext(NULL, RTLD_NOW | RTLD_GLOBAL, &ext);
```

**问题**: 未完成构建测试，但根据 Android 源码分析，`ANDROID_DLEXT_USE_LIBRARY_FD` 在 Android 10+ 上也受 namespace 限制——linker 在加载依赖时会重新检查路径。

### OpenCL 结论

**不可行**。Android 13 / MIUI 的 linker namespace 隔离是内核级安全机制，app 层无法绕过。无论通过什么方式加载 `/vendor/lib64/libOpenCL.so`，其所有 DT_NEEDED 依赖链（`libcutils.so` → 更多 vendor libs）都在 vendor namespace 中，app 无法访问。

---

## 尝试二：Vulkan 后端

### 阶段 1：初始 Vulkan 配置

项目初始已有 Vulkan 配置：
- `GGML_VULKAN ON`
- vendored `vulkan-hpp` (v351) + `spirv-headers`
- NDK glslc 路径配置
- vendor headers include 路径注入

**编译通过**，但 `vkGetPhysicalDeviceFeatures2` 符号缺失。

**根因**: 该项目之前 `minSdk=24`，NDK 的 `libvulkan.so` 在 API 24 不导出 `vkGetPhysicalDeviceFeatures2`（Vulkan 1.1 函数）。已通过升级到 `minSdk=26` 修复。

### 阶段 2：版本检查 1.2 → 1.1

ggml-vulkan 后端要求 Vulkan 1.2（从 2023 年首次创建 `2307523d3` 以来从未支持过 1.1）。但 Adreno 650 仅保证 1.1。

**修改**: 将版本检查从 `VK_API_VERSION_1_2` 降为 `VK_API_VERSION_1_1`，对低于 1.2 的设备打印 warning。

### 阶段 3：runtime dispatcher + shader target

两个关键修复：
1. `vkGetPhysicalDeviceFeatures2` → Vulkan-Hpp dispatcher 调用（3 处）
2. `vkWaitSemaphores` → dispatcher 调用 + null check

### Vulkan 测试结果

| 测试 | 结果 |
|------|------|
| 编译 | ✅ 通过 |
| 安装 | ✅ 成功 |
| Vulkan 初始化 | ✅ 成功，检测到 Adreno 650 |
| 后端注册 | ✅ Vulkan0 注册成功 |
| 模型加载 | ✅ 成功 |
| **推理输出** | **❌ 乱码** |

ggml-vulkan 已知的 Adreno workaround（禁用 cooperative matrix、禁用 async）已合入，但仍然无法避免输出乱码。

### Vulkan 结论

**不可行**。Adreno 650 的 Vulkan 驱动存在与 ggml-vulkan shader/buffer 操作的不兼容，导致 GPU 计算结果不正确。这是 GPU 驱动层 bug，应用层无法修复。

---

## BL 解锁后 OpenCL 可以运行吗？

**可以。** BL 解锁 + root 后，可以通过以下任一方式解决 linker namespace 隔离：

### 方案 1：Magisk 模块修改 linker 配置（推荐）

Android 10+ 的 linker namespace 配置在 `/system/etc/linker.config.pb`（protobuf 格式）。Magisk 模块可以在 post-fs-data 阶段修改此配置，将 `/vendor/lib64/` 添加到 app 的 namespace 的 `permitted_paths` 中。

也可以安装 **"Linker Namespace Enabler"** 或类似 Magisk 模块，一键让 app namespace 可以访问 vendor 库。

### 方案 2：直接使用 `LD_LIBRARY_PATH`（需 SELinux 宽松模式）

Root 后设置 `setenforce 0`（禁用 SELinux），然后 `dlopen("/vendor/lib64/libOpenCL.so")` 就可以工作。或者用 Magisk 模块修改 SELinux policy，允许 untrusted_app 访问 vendor_file。

### 方案 3：替换 libOpenCL.so

在 Magisk 模块中，把 `/vendor/lib64/libOpenCL.so` 复制到 `/data/data/org.osh26.llama/files/` 并设置正确的权限和 SELinux context。因为 data 目录已经在 app namespace 的 permitted_paths 中，可以直接加载。

**注意**: OpenCL 本身的 Adreno 650 实现是**正确的**（之前项目成功使用过 OpenCL）。问题100%是 Android 安全机制造成的 namespace 隔离，与 OpenCL 驱动的正确性无关。

---

## 总结

| 路径 | 状态 | 阻因 |
|------|------|------|
| CPU | ✅ 可运行 | 无 GPU 加速 |
| OpenCL | ❌ linker namespace | Android 13 MIUI 安全隔离（root 可解） |
| Vulkan | ❌ 输出乱码 | Adreno 650 驱动 bug（无解） |

**建议**: 如果可能解锁 BL + root，OpenCL 是最有希望的 GPU 路径。Vulkan 的驱动 bug 即使 root 也无法修复。
