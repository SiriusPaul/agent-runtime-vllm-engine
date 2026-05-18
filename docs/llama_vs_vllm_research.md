# llama.cpp 与 vLLM 底层实现差异调研

---

## 1. 模型表示与加载

### 1.1 文件格式

| | llama.cpp | vLLM |
|---|---|---|
| 主格式 | **GGUF**(单文件,header + KV metadata + tensors) | HF safetensors / pytorch bin / 自定义 |
| 量化嵌入 | 量化权重 **直接写入** GGUF | 权重通常 FP16/BF16;量化方案 (AWQ/GPTQ/FP8) 单独标注 |
| 元数据 | KV 字典(tokenizer、模板、架构、超参一站式) | `config.json` + `tokenizer_config.json` 分散 |
| Tokenizer | **内置在 GGUF**,无外部依赖 | 加载时调 `transformers` / HF `tokenizers` |
| mmap | 默认支持 ([src/llama-mmap.cpp](src/llama-mmap.cpp)) | 弱(HF 默认全量加载到显存/内存) |

GGUF 的"自包含"特性对 Android **非常重要**:推理引擎不需要 Python 和 HF 依赖,APK 里只塞一个 .gguf 就能跑。

### 1.2 权重张量到模型的绑定

- **llama.cpp**:`llama-model-loader.cpp` 读 GGUF → 在 `ggml_context` 注册每个 tensor → `llama-model.cpp` 按架构(`LLM_ARCH_LLAMA` / `LLM_ARCH_QWEN3` / ...)填到 `llama_layer` 结构。
- **vLLM**:`vllm/model_executor/models/<arch>.py` 用 PyTorch `nn.Module` 拼模型;权重加载走 `weight_loader` 回调,把 HF 权重映射到 vLLM 的 layer。

### 1.3 量化

| 格式 | llama.cpp | vLLM |
|---|---|---|
| FP16 / BF16 | ✅ | ✅ |
| INT8 weight-only | Q8_0 | ✅ (W8A16) |
| INT4 weight-only | Q4_0 / Q4_K_M / IQ4_XS | AWQ / GPTQ |
| 2~3 bit | Q2_K / IQ2_* / IQ3_* / Q3_K | 实验性,生态弱 |
| **FP8 KV / Activation** | F16/BF16 KV(部分量化 K/V dtype) | ✅(FP8 KV、FP8 W8A8) |
| **混合粒度** | K-quants:每 256 元素一个 super-block | per-channel / per-group |

**对 Android 的意义**:llama.cpp 的 Q4_K_M / Q5_K_M / IQ4_XS 是端侧主力,在 0.5B~7B 模型上几乎是默认选择。这是 vLLM 不便覆盖的区域。

---

## 2. 张量后端与算子

| | llama.cpp / ggml | vLLM |
|---|---|---|
| 张量库 | **ggml** (C, 自研) | PyTorch + 自定义 CUDA/Triton |
| 后端注册 | `ggml_backend_*`(CPU/CUDA/Vulkan/Metal/SYCL/CANN/MUSA/OpenCL/RPC) | CUDA backend,辅以 ROCm/CPU/TPU(子项目) |
| 算子定义方式 | C 函数 + 后端特化(SIMD/CUDA kernel) | Python op + 底层 Triton/CUDA |
| 计算图 | `ggml_cgraph` 静态构建,`ggml_backend_sched` 算子级调度到后端 | PyTorch eager + `torch.compile` + 自定义 attention kernel |
| 图复用 | 支持(`llm_graph_result` 重放,见 [src/llama-context.h:341](src/llama-context.h#L341) `gf_res_prev`) | **CUDA Graph** 捕获 decode 路径 |
| 线程模型 | 主线程 + ggml threadpool;`n_threads` / `n_threads_batch` 分别可控 | Python 进程 + CUDA stream;Worker process per GPU |

**Vulkan 实情**:`ggml/src/ggml-vulkan` 已经有 GEMM、RoPE、RMSNorm、SoftMax、Conv、Flash-Attention 等核心算子,在 Snapdragon 8 Gen2/Gen3 上 tokens/s 可超过 CPU 2-4× (具体随模型/驱动差异极大)。但 **Android Vulkan 驱动碎片化严重**,白名单机型外建议先 CPU 兜底。

---

## 3. KV 缓存 ★

### 3.1 vLLM 的 PagedAttention

vLLM 的核心创新:**像操作系统管理虚拟内存那样管理 KV cache**。

```text
全局 KV 池:在 GPU 显存上预分配 N 个 KV block
  ┌────┬────┬────┬────┬────┬────┐
  │ B0 │ B1 │ B2 │ B3 │ B4 │ ...│   每个 block 容纳 16 token (默认)
  └────┴────┴────┴────┴────┴────┘

每个 sequence 维护一个 "block table"(逻辑 → 物理):
  seq_A.block_table = [B2, B0, B4]      # 长度 = ceil(seq_len / block_size)
  seq_B.block_table = [B2, B3]          # 注意:B2 与 A 共享!ref_count[B2] = 2

attention kernel 输入额外接收 block_table,内部:
  for each token position p in seq:
    block_id = block_table[p / block_size]
    offset   = p % block_size
    读取 KV_pool[block_id][offset]
```

关键性质:
1. **零外部碎片**:所有分配都是 block 粒度
2. **跨序列共享**:相同前缀的 block 物理上同一份
3. **写时复制 (CoW)**:共享 block 被一方"扩展"时,引用计数 >1 才复制
4. **抢占/回收**:scheduler 可以把低优请求的 block "swap-out" 到 CPU 内存或直接重新计算

PagedAttention kernel 必须按 `[B*H, head_dim]` 的形式从离散 block 里 gather,因此是一个**专门写的 CUDA kernel**(`vllm/csrc/attention/paged_attention_v1.cu`、`v2.cu`),不是标准 FlashAttention。

### 3.2 llama.cpp 当前 KV 实现

[src/llama-kv-cache.h](src/llama-kv-cache.h):

```cpp
struct llama_kv_cache : public llama_memory_i {
    std::vector<kv_layer> layers;       // 每层一个 K, V 大张量
    std::vector<llama_kv_cells> v_cells;// 每 stream 一组 cell 元数据
    std::vector<uint32_t> v_heads;      // 每 stream 一个 ring-buffer 头
    std::vector<uint32_t> seq_to_stream;
    uint32_t n_seq_max, n_stream;
    // SWA 支持
    uint32_t n_swa;
    llama_swa_type swa_type;
};
```

形状上 (per layer, per stream):
```
K: [n_embd_k_gqa, n_ctx]    连续大张量
V: [n_embd_v_gqa, n_ctx]    (或 transpose 后 [n_ctx, n_embd_v_gqa])
```

每次 `find_slot(ubatch, cont)` 在 v_cells 里找一段空位,返回 `slot_info { idxs }` — 这就是 "本 ubatch 的每个 token 写到第几个 cell"。

**和 vLLM 的对比**:

| 维度 | llama.cpp 现状 | vLLM | 差距判定 |
|---|---|---|---|
| 分配粒度 | 单 cell(== 单 token) | block (默认 16 token) | 粒度更细,理论上无内部碎片;但**外部碎片**明显 |
| 跨序列共享 | 只有 `seq_cp` 全量复制 | 块级 ref_count 共享 | 缺自动前缀复用 |
| 长会话回收 | `seq_rm` 标 free + ring buffer | 释放整 block 入池 | 类似,但 llama.cpp 无显式池 |
| 多 stream | ✅ (`n_stream` + `seq_to_stream`) | 通过 block_table 间接 | **llama.cpp 已有 stream 概念,可作为 paged 改造的脚手架** |
| SWA | ✅ ([src/llama-kv-cache-iswa.h](src/llama-kv-cache-iswa.h)) | ✅ | 持平 |
| Mamba/SSM 状态 | ✅ ([src/llama-memory-recurrent.h](src/llama-memory-recurrent.h)) | 实验性 | llama.cpp 略好 |
| 写入图节点 | `cpy_k(ctx, k_cur, k_idxs, il, sinfo)` 已经是"按索引写" | gather-scatter via custom kernel | **llama.cpp 接口已经够通用**,只是 idxs 当前简单 |

### 3.3 改造路线

llama.cpp 的 `llama_memory_i` 抽象 ([src/llama-memory.h:68](src/llama-memory.h#L68)) 是天然插点:

```cpp
struct llama_memory_i {
    virtual llama_memory_context_ptr init_batch(...) = 0;  // 切 ubatch + 找 slot
    virtual bool seq_rm/seq_cp/seq_keep/seq_add/seq_div(...) = 0;
    // state I/O
};
```

**最小改造路径**:
1. 新增 `llama_kv_cache_paged : public llama_memory_i`
2. 内部用 `block_table[seq_id] = [block_id...]` + `free_blocks` 列表
3. `init_batch` 时给每个 ubatch token 算出 `(block_id, offset)`,封装进新的 `slot_info`
4. `cpy_k` / `cpy_v` 用 `ggml_set_rows`(已存在算子)做 gather/scatter
5. `seq_cp` 改成增 ref_count,不实际拷贝
6. `get_k/get_v` 在建图时把 block_table 喂进 attention 算子

**Attention 算子怎么办**?
- vLLM 必须写专门的 paged attention kernel,因为 KV 物理离散。
- ggml 已经有 `ggml_set_rows` / `ggml_get_rows`,可以 **先聚合再算**:每个 ubatch 把所需的 block 物理 gather 出一段连续 KV,丢给现有 FlashAttention/标准 Attention。代价是多一次拷贝,但避免改 attention kernel。
- 进阶:在 ggml-vulkan 后端写一个 paged attention shader(可参考 vllm 的 v2 kernel)。

**结论**:llama.cpp 的接口允许我们用 **"聚合-计算-散播"** 的笨办法先把 paged 跑起来,再逐后端优化 kernel。这是 vLLM 不会有的灵活性(它必须 day-1 就写 kernel)。

---

## 4. 批处理与请求调度 

### 4.1 vLLM 的 Continuous Batching

vLLM 的调度器是 **iteration-level**:

```python
# 伪代码:vllm/core/scheduler.py
while engine_running:
    scheduled = []
    # (a) 收集 prefill 候选
    for req in waiting_queue:
        if can_allocate(req.prompt_tokens):
            scheduled.append(("prefill", req))

    # (b) 收集 decode 候选(所有 running 请求,每个吃 1 token)
    for req in running_queue:
        scheduled.append(("decode", req))

    # (c) 一次性算 (chunked prefill + decode 混合 batch)
    forward(scheduled)  # 一次 forward = 一个 iteration

    # (d) 把 EOS / max_tokens 命中的请求出队
    for req in scheduled:
        if req.done: finish(req)
```

关键性质:
- **不等长拼批**:同一 forward 里可以有 A 的 prefill (1500 token) + B 的 decode (1 token) + C 的 prefill chunk (256 token)
- **Chunked prefill**:超长 prompt 切片喂入,避免一次 prefill 阻塞所有 decode
- **抢占**:显存吃紧时可以把低优请求的 KV 换出
- **背压**:waiting → running 的转移取决于 KV 池剩余

### 4.2 llama.cpp 的"批处理"

[include/llama.h:235](include/llama.h#L235) 的 `llama_batch` 已经支持:

```c
typedef struct llama_batch {
    int32_t       n_tokens;
    llama_token  * token;       // [n_tokens]
    llama_pos    * pos;         // [n_tokens]
    int32_t      * n_seq_id;    // [n_tokens]
    llama_seq_id ** seq_id;     // [n_tokens][n_seq_id[i]]
    int8_t       * logits;      // [n_tokens] 是否输出
} llama_batch;
```

也就是说 **一个 batch 里可以放任意多个序列的任意位置的 token**,每 token 独立标 `pos` 和 `seq_id`。这已经是 continuous batching 的**数据结构基础**。

**但是**,llama.cpp **没有** 一个常驻调度线程把多请求合成 batch。`llama_decode` 是同步阻塞 API:
- 调用者负责拼 batch、负责按 ubatch 切分、负责处理 EOS。
- `llama-server`([tools/server/](tools/server/) 或 [examples/server/](examples/server/))实现了一个**朴素**版的请求级调度,但远不如 vLLM 精细。

### 4.3 改造路线

把 vLLM Scheduler 思想搬到端上 **不需要改 llama.cpp 任何代码**,只需要在上层写一个"engine":

```cpp
// 在 examples/android_osh26/app/src/main/cpp/osh26_engine.cpp 里写
class Engine {
    llama_context * ctx;
    std::deque<Request> waiting, running;
    std::thread loop;

    void iteration() {
        std::vector<llama_token> tokens;
        std::vector<llama_pos>   poss;
        std::vector<llama_seq_id> seqs;
        std::vector<int8_t>      logits;

        // 1. waiting → running:为 prefill 找 KV 空间
        admit_new();
        // 2. 给每个 running 请求加 1 个 decode token + chunked prefill
        build_batch(tokens, poss, seqs, logits);

        llama_batch batch = {...};
        llama_decode(ctx, batch);

        // 3. 取 logits,采样,回调 token
        dispatch_outputs();
    }
};
```

针对 Android 端的简化:
- **N ≤ 4 并发**:不做复杂优先级,简单 round-robin + "交互优先" 队列即可
- **不做抢占**:KV 满了就拒新请求 (`waiting` 排队等)
- **chunked prefill 阈值** = `n_ubatch`(已存在)

---

## 5. Attention Kernel

| | llama.cpp (ggml) | vLLM |
|---|---|---|
| Prefill kernel | `ggml_flash_attn_ext` (CPU/GPU 都有) | `flash_attn_varlen_func` (FlashAttention-2/3) |
| Decode kernel | 同上(单 token 当退化情况) | **paged_attention_v1 / v2** (专门为离散 KV 写) |
| GQA / MQA | ✅ 通过 `n_head_kv` 参数 | ✅ |
| SWA | ✅ | ✅ |
| ALiBi | ✅ | ✅ |
| RoPE 变体 | NORM / NEOX / MROPE / IMROPE / VISION | NORM / NEOX (主流) |
| Causal mask | 算子内置 | kernel 内置 |
| Multi-query batch | 通过 `kq_mask` 显式控制 | 通过 `block_tables` + `cu_seqlens` 隐式 |

**判断**:
- llama.cpp 在算子覆盖度上**不输** vLLM,且每个后端(尤其 Vulkan)都已经实现 flash-attention。
- 真正区别是 **vLLM 的 attention 直接从 paged KV 算**,llama.cpp 需要先 gather。
- 移动端上 gather 的开销 vs 写专用 shader 的工程量,**先 gather 是合理的工程选择**。

---

## 6. 采样与结构化输出

### 6.1 标准采样

| | llama.cpp | vLLM |
|---|---|---|
| Top-k / Top-p / Min-p | ✅ ([include/llama.h:1305-1311](include/llama.h#L1305-L1311)) | ✅ |
| Temperature / Dynamic-temp | ✅ ([include/llama.h:1317-1320](include/llama.h#L1317-L1320)) | ✅ |
| Repetition penalty | ✅ `init_penalties` | ✅ |
| Mirostat v1/v2 | ✅ | ✅ |
| Typical / XTC / Top-n-σ | ✅ | 部分 |
| DRY | ✅ ([include/llama.h:1392](include/llama.h#L1392)) | 不内置 |
| Logit bias | ✅ | ✅ |
| 采样位置 | CPU 默认,**实验性**后端采样钩子([include/llama.h:1232-1252](include/llama.h#L1232-L1252)) | CPU(Python) + 实验性 `SamplerOutput` GPU 采样 |
| 链式组合 | `llama_sampler_chain_add` | `SamplingParams` 单对象 |

### 6.2 Structured Generation / Tool Calling

| | llama.cpp | vLLM |
|---|---|---|
| GBNF 语法 | ✅ ([include/llama.h:1355](include/llama.h#L1355) `init_grammar`) | ✅ via `outlines` / `lm-format-enforcer` / `guidance` |
| JSON Schema → 语法 | ✅ ([common/json-schema-to-grammar.h](common/json-schema-to-grammar.h)) | ✅ via outlines |
| Lazy grammar(只在触发模式后约束) | ✅ ([include/llama.h:1371](include/llama.h#L1371) `grammar_lazy_patterns`) | ✅ |
| LL-Guidance (微软) | 可选集成 ([common/llguidance.cpp](common/llguidance.cpp)) | ✅ |
| Tool calling 模板 | [common/chat.cpp](common/chat.cpp) 内置多种 chat template + 工具调用解析器 | server 层处理 |

**对项目意义**:
- 不需要从零搭 tool calling — `llama_sampler_init_grammar_lazy_patterns` + JSON Schema → GBNF 是 30 行代码的事
- 触发模式可以是 `<tool_call>` 或具体模型的工具调用 tag

---

## 7. Prefix Cache

| | llama.cpp | vLLM |
|---|---|---|
| 自动前缀缓存 | ❌ | ✅ Automatic Prefix Caching (APC) |
| 命中方式 | 调用者手动 `llama_memory_seq_cp` | block 级 hash(prefix_hash → block_id) |
| 跨请求复用 | 要自己维护 token 序列 → seq_id 的映射 | 内置 |
| Tokenizer 版本绑定 | 需手动检查 | hash 已隐含 token id |
| 失效策略 | LRU 由调用者决定 | LRU + ref_count |

**移植路径**(在 paged KV 之上):
1. 把每个 block 完成写入时,算 `hash(token_ids_in_block, prefix_hash_of_prev_block)`,存到 `block_id_by_hash` 字典
2. 新请求 prefill 时,逐 block 尝试命中;命中则增 ref_count、跳过这段计算
3. 命中边界对齐 block_size,未对齐部分照常 prefill
4. LRU 淘汰:`free_blocks` 优先回收 `ref_count == 0` 且最久未访问的

这一层**不依赖 llama.cpp 源码改动**,完全做在自家 paged memory 子类里。

---

## 8. Speculative Decoding / 推理加速进阶

| | llama.cpp | vLLM |
|---|---|---|
| Draft model speculative | ✅ ([common/speculative.h](common/speculative.h)) | ✅ |
| Medusa / Eagle / MTP | 部分 | ✅(多种 head 实现) |
| n-gram speculative | ✅ ([common/ngram-cache.h](common/ngram-cache.h)) | ✅ |
| Multi-step decoding | 通过 batch 多 token 隐式支持 | 显式 `num_scheduler_steps` |
| Chunked prefill | 通过 `n_ubatch` 切分 | ✅ 调度器原生 |
| **CUDA Graph / 图复用** | 图节点级复用 ([src/llama-context.h:341](src/llama-context.h#L341)) | CUDA Graph 捕获 |
| Tensor parallel | layer split / row split (单进程多 GPU) | TP via NCCL |
| Pipeline parallel | ❌(简易) | ✅ |
| Expert parallel (MoE) | 单 GPU MoE,无 EP | ✅ |

**Android 端项目用得到的**:n-gram speculative(轻量、无需第二个模型);chunked prefill(直接走 `n_ubatch`)。

---

## 9. 内存模型与显存/内存预算

| | llama.cpp | vLLM |
|---|---|---|
| KV 预算计算 | `n_ctx × n_seq_max × n_layer × (k_size + v_size)` 一次性 allocate | `gpu_memory_utilization` 比例,扣除权重和激活后剩余全做 KV 池 |
| 量化 KV | 通过 `type_k` / `type_v` 选 dtype (F16/Q8_0/Q5_1/IQ4_NL...) | FP8 KV(H100 起) |
| KV offload 到主机内存 | 显式(部分后端) | ✅ (CPU offload + swap) |
| Out-of-memory 行为 | `llama_decode` 返回 1 (No KV slot) | scheduler 抢占低优,或 reject |
| **mmap 权重** | ✅ 默认 | 弱 |

**Android 实情**:
- 8GB RAM 设备上,Qwen2.5-0.5B-Q4_K_M 模型本身 ~400MB,KV 在 4k context 下 ~50MB 量级,完全可控
- 关键预算是 **n_ctx × n_seq_max**(决定 KV 大小)+ 权重 + ggml compute buffer 三块
- 量化 K/V (`type_k = GGML_TYPE_Q8_0`) 可再省 2×,但在小模型上质量损失需评测

---

## 10. Tokenizer

| | llama.cpp | vLLM |
|---|---|---|
| 实现 | C++ 内置 ([src/llama-vocab.h](src/llama-vocab.h)),支持 SPM/BPE/WPM/UGM/RWKV/PLaMo2 | 调 HF `tokenizers`(Rust + Python 绑定) |
| 依赖 | 0(token 表在 GGUF 内) | tokenizer.json + Python |
| 性能 | 单线程 C++,通常足够端侧 | Rust 极快 |
| 一致性 | 与 HF 对齐做了大量 corner case 测试,偶有差异 | 直接是 HF 实现 |

**对 Android 重要**:不需要带 Python 解释器和 Rust 库 — 这是把推理引擎塞进 APK 的关键前提。

---

## 11. 多 GPU / 分布式

| | llama.cpp | vLLM |
|---|---|---|
| 单进程多 GPU | layer split / row split / tensor split([include/llama.h:194](include/llama.h#L194) `llama_split_mode`) | Worker per GPU |
| Tensor parallel(行/列切矩阵) | 仅 row split 接近,但有限 | ✅ Megatron-style |
| Pipeline parallel | ❌ | ✅ |
| Expert parallel | ❌ | ✅ |
| 通信库 | ggml 内部 + RPC backend | NCCL |

**对端侧无关** — Android 单 GPU 场景。

---

## 12. 服务化与可观测性

| | llama.cpp (`llama-server`) | vLLM |
|---|---|---|
| HTTP API | OpenAI-compatible | OpenAI-compatible |
| Streaming | SSE | SSE |
| 指标 | 基础 perf counters([include/llama.h:1504](include/llama.h#L1504) `llama_perf_context_data`) | Prometheus metrics,丰富 |
| 多用户 | 简单调度 | 完整调度器 |
| 部署形态 | 单可执行 (~5MB to ~50MB 含后端) | Python + CUDA runtime |

**端侧不需要 HTTP**,但 `llama_perf_context_data` 远不够 — 我们需要按请求记录:
- TTFT (Time-To-First-Token)
- per-token decode latency (中位数 / P95)
- prefill tokens/s vs decode tokens/s
- KV 占用率、prefix cache 命中率
- 温度 / 主频(`/sys/class/thermal/`)
- 主线程是否被抢占

这些都是项目 `trace_metrics` 模块要实现的内容。

---

## 13. 端到端时延来源对比

一次"用户输入 → 收到首 token"在两边的成本拆解:

```text
                 llama.cpp on Android                vLLM on H100
                 ─────────────────────                ────────────
  HTTP/IPC          ~0 (JNI)                          ~1-3 ms
  Tokenize          0.1-1 ms (C++)                    0.5-2 ms
  Schedule admit    < 0.1 ms (直接 decode)            1-5 ms (调度器)
  KV alloc          ~0 (ring buffer)                  0.1-1 ms (block 分配)
  Prefill compute   主要成本 (CPU/Vulkan)             极快 (TP + FlashAttn)
  Sample (first)    0.1-0.5 ms                        0.5-2 ms
  ─────────────────────────────────────────────────────────────────
  TTFT 主导项       prefill 计算                      调度 + 通信
```

端侧的瓶颈始终是 **prefill 算力**,vLLM 的诸多优化(调度、缓存、抢占)在单用户、prompt 不长的端侧场景**收益有限**。所以项目优先级:

1. **Prefix Cache**(系统 prompt 复用 → TTFT 立刻降一个数量级)
2. **KV 量化 / Paging**(让更长 context 可用 → 用户体验)
3. **Vulkan**(让 prefill 算力提升 → 直接砍 TTFT)
4. Continuous batching(只在 N>1 并发时有价值 — Agent runtime 多任务并发时才需要)

---

## 14. 借鉴而非移植 — vLLM 思想到 llama.cpp 的精确映射

| vLLM 概念 | llama.cpp 插点 | 改造性质 |
|---|---|---|
| Paged KV + block table | 新增 `llama_memory_i` 子类,KV gather/scatter 用 `ggml_set_rows` | **新代码,不动主干** |
| Continuous batching | 在 `llama_context` 之上写 engine 调度线程,合多请求成一个 `llama_batch` | **纯应用层** |
| Chunked prefill | 直接调整 `n_ubatch` + 调度器分片 | **配置** |
| Automatic prefix cache | 在新 paged memory 里加 `hash → block_id` 索引 | **新代码** |
| Structured generation | `llama_sampler_init_grammar_lazy_patterns` + `json-schema-to-grammar` | **配置** |
| Speculative decoding | `common/speculative.{cpp,h}` + `common/ngram-cache.{cpp,h}` | **现成** |
| KV swap-out / 抢占 | `llama_state_seq_get_data_ext` 写到主机内存 / 文件 | **新代码** |
| FP8 KV | 通过 `cparams.type_k` / `type_v` 选量化类型 | **配置** |
| Metrics | 自己写 `trace_metrics` 模块,挂在 engine 而不是 llama_context | **新代码** |

llama.cpp 已经提供了 vLLM 思想需要的所有**接缝**,差的是把 vLLM 设计具体化的**那层薄薄的 C++ 中间件**。这一层的代码量、复杂度,适合一个课程项目周期。

---

## 14. 总结

| 维度           | llama.cpp                                          | vLLM                                      |
| -------------- | -------------------------------------------------- | ----------------------------------------- |
| 角色           | **嵌入式推理库** + CLI/HTTP demo                   | **多租户推理服务器**                      |
| 目标硬件       | 任意 (CPU 第一,Vulkan/Metal/CUDA/SYCL 增量)        | 数据中心 GPU(CUDA 第一)                   |
| 语言/产物      | C++17,单 `.so` / `.a`,可静态链接进 APP             | Python + CUDA/Triton/C++,需 Python 运行时 |
| 调度粒度       | 同步 `llama_decode`,一次只跑一个 batch             | iteration-level 调度器,token 粒度抢占     |
| KV 形态        | 整块预分配 + ring-buffer slot                      | **Paged**:固定大小 block + 块表           |
| 跨序列前缀复用 | 手动 `seq_cp`                                      | **自动**前缀缓存(hash 命中)               |
| 量化深度       | 极丰富(Q2~Q8 + IQ + K-quants + TQ + MXFP4 + NVFP4) | 主要 FP16/BF16 + AWQ/GPTQ/FP8             |
| 分布式         | 单进程多 GPU(layer/row split)                      | TP + PP + EP,NCCL 通信                    |



## 15. 立项建议:Android 端实施优先级

按"投入产出比 × 与课程演示价值"排序:

### Phase A — Baseline & Telemetry(1 周)
- llama.cpp Android 编译跑通(CPU 后端)
- 接入 Qwen2.5-0.5B-Q4_K_M
- 自写 `trace_metrics`:TTFT / tokens/s / 内存 / 温度
- **Deliverable**:首张性能基线表

### Phase B — Prefix Cache(2-3 周)★ ROI 最高
- 用 `llama_memory_seq_cp` 实现"系统 prompt 复用"
- 上层维护 `hash(prompt_prefix) → seq_id` 表
- 命中场景的 TTFT 应降到原来的 1/5 ~ 1/10
- **Deliverable**:命中/未命中 TTFT 对比柱状图

### Phase C — Paged KV(3-4 周)
- 实现 `llama_kv_cache_paged` 作为 `llama_memory_i` 子类
- block size = 16/32 可配
- 实现 ref_count + CoW
- 单序列性能不应回退;长会话内存稳定性应提升
- **Deliverable**:长会话 OOM 触发率对比

### Phase D — Scheduler Lite + Structured Output(2 周)
- engine 层 continuous batching(N ≤ 4)
- `grammar_lazy_patterns` 接入 JSON tool call
- **Deliverable**:并发场景 P95 对比 + tool call 成功率

### Phase E — Vulkan(2-3 周,风险高)
- 开启 `GGML_VULKAN=ON`,机型白名单
- 评估 GEMM/FlashAttn 算子收益
- **Deliverable**:CPU vs Vulkan tokens/s & 能耗对比

