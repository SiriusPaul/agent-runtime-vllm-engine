# OSH26 Runtime PPT Source

## Deck Goal

Build a course-project presentation for OSH26 Runtime, an Android-side local Agent Runtime prototype based on a llama.cpp fork. The message is: OSH26 Runtime is not a duplicate of llama.cpp; it uses llama.cpp as the model/kernel base and adds Android deployment, Agent-oriented lifecycle, vLLM-style memory optimization, small-concurrency scheduling, streaming, cancellation, health telemetry, and target-device GPU optimization.

## Core Positioning

OSH26 Runtime is a lightweight Android Agent Runtime framework:

- Reuses llama.cpp/GGUF/tokenizer as the proven model base.
- Adds Android JNI and HTTP bridge for app/agent integration.
- Adds streaming token output, cancel, release, and runtime health state.
- Adds vLLM-inspired Prefix KV Cache and paged-style reuse for repeated system prompts and tool templates.
- Adds Q8 W8A8 Vulkan path for prefill acceleration on Snapdragon 870 / Adreno 650.
- Adds benchmark and health telemetry for TTFT, TPS, submit count, LM-head latency, descriptor allocation, and memory.

## Suggested Slide Outline

1. Cover: OSH26 Runtime - Android Agent Runtime for local LLM deployment.
2. Problem: endpoint Agent inference is more than running a model.
3. Why vanilla llama.cpp is not enough for Agent Runtime.
4. Our architecture: llama.cpp base + Android runtime layer + optimized backend.
5. Android integration closed loop: JNI, stream, cancel, release, health, OpenAI-style HTTP API.
6. vLLM ideas adapted to Android: Prefix KV Cache, Scheduler Lite, chunked prefill, telemetry.
7. Performance chart: Q8 W8A8 prefill reduces TTFT from 7361.61 ms to 788.56 ms.
8. Performance chart: native Q8 average TTFT around 717.259 ms.
9. Performance chart: Prefix KV Cache reduces exact-repeat TTFT from 1902.63 ms to 613.658 ms.
10. 8K context and long-context validation for Agent memory.
11. Latest LM-head / descriptor optimization: descriptor allocation reaches 0, submit median 2.
12. Why not llama.cpp native OpenCL.
13. Why not llama.cpp native Vulkan.
14. Comparison with MNN and vLLM.
15. Novelty and project contribution.
16. Summary and defense answers.

## Refined Deck 2026-07-01

The deck was expanded from 16 to 24 pages and rewritten in Chinese for the OSH26 course defense. The revised focus is: endpoint deployment is an operating-system-level problem, and OSH26 differs from plain llama.cpp by adding Android lifecycle control, KV/prefix memory management, request scheduling, Adreno-specific Q8 Vulkan execution, and verifier-gated real-device optimization.

New content added:

- Endpoint deployment importance: offline use, privacy, low latency, local permission control.
- Git-history roadmap: CPU inference, OpenAI API, failed native Vulkan, custom GPU runtime, KV cache, Q8 prefill, native Q8, 8K context, single weight residency, 1.7B support, decode Q8 GEMV, stateless subagent cache.
- Operating-system concepts embedded into the technical explanation: working set, page cache, pinned/dynamic entries, LRU, fragmentation, short-job preference, aging, backpressure, critical sections, control plane/data plane split, driver boundary.
- More explicit failure analysis for OpenCL and native ggml Vulkan.
- More direct differentiation against llama.cpp, vLLM, and MNN.
- Action Fabric linkage: upper layer makes Agent execution schedulable; OSH26 Runtime makes local model serving schedulable.

## Key Data

### Q8 W8A8 Prefill

- Baseline F16 prefill TTFT: 7361.61 ms.
- Q8 prefill TTFT: 788.56 ms.
- Improvement: 9.34x.
- Same sampled token IDs in deterministic prompts.

### Native Q8_0 GGUF

- Native Q8 average repeated TTFT: 717.259 ms.
- Preserves approximately 0.7-0.84 s TTFT achieved by converted Q8 path.
- Native Q8 avoids repeated load-time F16 to Q8 conversion.

### Prefix KV Cache

- Cold Question A TTFT: 1902.63 ms.
- Exact repeat A TTFT: 613.658 ms.
- Improvement: 3.10x, 67.7% TTFT reduction.
- Cold TPS: 3.383.
- Exact repeat TPS: 4.734.
- Shared-prefix B TTFT: 1288.87 ms.
- Shared-prefix B TPS: 3.891.

### 8K Context

- Context increased from 1024 to 8192 tokens.
- Cold A TTFT: 1348.14 ms.
- Exact repeat A TTFT: 460.893 ms.
- Exact repeat A TPS: 5.232.
- Long request successfully prefetched 5488 user tokens in 43 chunks of 128 tokens, then generated 8 tokens.

### Latest LM Head and Descriptor Cache

- 0.6B-Q8_0: median TTFT 392.802 ms, TPS 4.7800, LM head 162.996 ms, submit median 2, descriptor allocation median 0.
- 1.7B-Q8_0: median TTFT 841.991 ms, TPS 2.2154, LM head 363.742 ms, submit median 2, descriptor allocation median 0.

## Why Not Native OpenCL

OpenCL on Android is not a stable public NDK deployment path. On the target Redmi K40 / Android 13 / MIUI device, the system vendor libOpenCL.so and its DT_NEEDED dependencies are isolated by Android linker namespaces. Attempts to copy libOpenCL.so or preload dependencies still fail because vendor/system dependencies remain outside the app namespace. Root or Magisk-level namespace changes might make OpenCL possible, but that is not acceptable for a deployable course demo or general Android Agent Runtime.

## Why Not Native ggml Vulkan

The target device is Snapdragon 870 / Adreno 650. Native ggml-vulkan assumes newer Vulkan capability, while Adreno 650 only guarantees Vulkan 1.1. After lowering version checks and using runtime dispatch, initialization and model load succeeded, but generation produced corrupted output. That means the issue is not only API availability but shader/buffer correctness on this driver stack. OSH26 therefore uses a controlled Vulkan 1.1 path with device-specific kernels and correctness gates, with CPU fallback where needed.

## Comparison

### llama.cpp

Strength: excellent GGUF, tokenizer, CPU backend, model ecosystem.
Gap for our goal: not a complete Android Agent Runtime; scheduling, prefix cache, small concurrency, health telemetry, and app lifecycle are left to upper layers.

### vLLM

Strength: server-side scheduling, PagedAttention, prefix caching, high-throughput CUDA serving.
Gap for Android: depends on Python/CUDA/Triton/server assumptions. OSH26 borrows the memory and scheduling ideas, not the full server stack.

### MNN

Strength: mature mobile inference framework and optimized operators.
Gap for our goal: more graph/operator oriented; less direct support for GGUF/llama.cpp workflow, Agent request lifecycle, KV reuse, prompt cache, tool-call output, and OpenAI-style local service interface.

## Defense Message

This is not repeated wheel-building. The project contributes an Android Agent Runtime layer above a proven model base, plus a target-device optimized GPU and cache path. The work includes Android integration, JNI/HTTP API, streaming/cancel/release lifecycle, Q8 W8A8 prefill acceleration, Prefix KV Cache, 8K context validation, descriptor allocation optimization, real-device benchmarking, and failure analysis for OpenCL/Vulkan deployment paths.
