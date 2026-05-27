# 1. 把 GGUF 文件从电脑推送到手机临时目录
  
  adb push qwen3-0.6b.gguf /data/local/tmp/

# 2. 复制到 app 私有目录（app 才能读）

  adb shell "run-as org.osh26.llama cp /data/local/tmp/qwen3-0.6b.gguf /data/data/org.osh26.llama/files/models/"

# 或者直接推送到可读目录再用 app 加载

  adb push qwen3-0.6b.gguf /sdcard/Android/data/org.osh26.llama/files/models/


最终架构
  
GGUF文件 ──(一次性加载)──→ GPU Pool (一个大 VkBuffer, ~2.5GB)
                            │ HOST_VISIBLE | HOST_COHERENT
                            │ CPU 和 GPU 共享同一块物理内存
                            ├── Weights (F32, ~2GB, 永不移动)
                            ├── KV Cache (~116MB)
                            └── Activations (~15MB, 复用)

推理过程 (每个token, 零copy):
  CPU: embedding lookup → pool
  [28层]:
    CPU: RMS norm (pool mapped ptr)
    GPU: MUL_MAT Q/K/V (pool buffer binding)
    CPU: RoPE, KV cache, attention
    GPU: MUL_MAT O/gate/up/down
    CPU: SiLU, residual add
    CPU: RMS norm → GPU: MUL_MAT LM head
    CPU: argmax sample

  ┌──────────────┬────────────────────────────────────────┐
  │     特性     │                  实现                   │
  ├──────────────┼────────────────────────────────────────┤
  │ 权重加载     │ GGUF 直接读取 (不依赖 llama internals)   │
  ├──────────────┼────────────────────────────────────────┤
  │ 显存管理     │ 一个大 VkBuffer, bump allocator          │
  ├──────────────┼────────────────────────────────────────┤
  │ CPU↔GPU 通信 │ 零 copy (统一内存)                       │
  ├──────────────┼────────────────────────────────────────┤
  │ 线程安全     │ pthread mutex                            │
  ├──────────────┼────────────────────────────────────────┤
  │ ggml backend │ 已移除 — 不注册, 不参与调度              │
  └──────────────┴────────────────────────────────────────┘
