# osh26_agent_runtime_ppt - Design Spec

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | osh26_agent_runtime_ppt |
| **Canvas Format** | PPT 16:9 (1280 x 720) |
| **Page Count** | 16 |
| **Design Style** | Top Consulting + dark tech academic defense |
| **Target Audience** | OSH26 course instructors and teaching assistants |
| **Use Case** | Course project defense presentation |
| **Created Date** | 2026-06-09 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280 x 720 px |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 56 px, top 44 px, bottom 36 px |
| **Content Area** | 1168 x 620 px |

---

## III. Visual Theme

### Theme Style

- **Style**: conclusion-first technical defense deck
- **Theme**: dark technology theme
- **Tone**: rigorous, data-backed, engineering-oriented

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#0B1020` | Main page background |
| **Secondary bg** | `#111827` | Panels and cards |
| **Panel 2** | `#16213A` | Alternate panels |
| **Primary** | `#2563EB` | Structure, blue bars, key claims |
| **Primary 2** | `#60A5FA` | Secondary blue marks |
| **Accent** | `#22C55E` | Positive results and improvements |
| **Secondary accent** | `#A78BFA` | vLLM-inspired mechanisms |
| **Warning** | `#EF4444` | Infeasible backend paths |
| **Body text** | `#F8FAFC` | Main text |
| **Secondary text** | `#CBD5E1` | Body support text |
| **Muted text** | `#64748B` | Captions and references |
| **Border/divider** | `#1F2A44` | Card borders and chart grids |
| **Grid** | `#334155` | Chart axis and baseline |
| **White** | `#FFFFFF` | Strong contrast labels |

---

## IV. Typography System

### Font Plan

**Typography direction**: PPT-safe technical sans with monospace for metrics.

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | Microsoft YaHei | Arial | sans-serif |
| **Body** | Microsoft YaHei | Arial | sans-serif |
| **Emphasis** | Microsoft YaHei | Arial | sans-serif |
| **Code** | - | Consolas, Courier New | monospace |

**Per-role font stacks**

- Title: `"Microsoft YaHei", Arial, sans-serif`
- Body: `"Microsoft YaHei", Arial, sans-serif`
- Emphasis: `"Microsoft YaHei", Arial, sans-serif`
- Code: `Consolas, monospace`

### Font Size Hierarchy

**Baseline**: Body font size = 18 px.

| Purpose | Size |
| ------- | ---- |
| Cover title | 58 px |
| Section title | 44 px |
| Page title | 32 px |
| Subtitle | 24 px |
| Body | 18 px |
| Annotation | 13 px |
| Footnote | 11 px |

---

## V. Layout Principles

- Conclusion-first page titles.
- Dark background with restrained cards and clear chart regions.
- Dense pages use 2-column or dashboard layouts.
- Breathing pages use one large claim and minimal support text.
- Every data chart contains explicit comparison and a short "so what" statement.

---

## VI. Icon Strategy

- **Library**: `tabler-outline`
- **Stroke width**: 2
- **Inventory**: cpu, device-mobile, database, chart-bar, layers, route, shield, alert-triangle, rocket, server, code, activity
- Icons are optional in the first version; visual hierarchy relies mainly on text, bars, and structured shapes.

---

## VII. Visualization Reference List

| Page | Visualization | Type |
| ---- | ------------- | ---- |
| P07 | Q8 TTFT before/after | Bar chart |
| P08 | Native Q8 repeated TTFT | KPI + compact bar |
| P09 | Prefix cache TTFT and TPS | Grouped bar chart |
| P11 | LM-head latest benchmark | Two-model bar chart |
| P14 | Alternative framework comparison | Matrix |

---

## VIII. Image Resource List

No external images are required in this first draft. The deck uses editable SVG shapes and charts.

---

## IX. Content Outline

### P01 - Cover
Title: OSH26 Runtime
Content: Android Agent Runtime for local LLM deployment. Message: faster, memory-aware, concurrent-ready, and easier to deploy than raw llama.cpp for endpoint agents.

### P02 - Executive Summary
Title: We turn llama.cpp from a model runner into an Android Agent Runtime.
Content: Three claims: Android integration closed loop; vLLM-style memory and scheduling ideas; real-device performance evidence.

### P03 - Endpoint Agent Problem
Title: Endpoint Agent inference is a runtime problem, not just a model problem.
Content: Agents need streaming, cancellation, long context, prompt reuse, concurrent tasks, telemetry, and robust backend selection.

### P04 - Why Vanilla llama.cpp Is Not Enough
Title: llama.cpp is the right base, but not the complete endpoint Agent framework.
Content: It provides GGUF, tokenizer, CPU backend and ecosystem; OSH26 adds Android lifecycle, service API, scheduler, cache, telemetry, and device-specific GPU path.

### P05 - Architecture
Title: OSH26 Runtime adds a thin Agent-oriented layer above llama.cpp.
Content: App/Agent to JNI/HTTP bridge to Scheduler Lite to Prefix KV Cache to OSH26 Vulkan/CPU backend to streaming callback and health telemetry.

### P06 - Android Integration
Title: The Android loop is already closed from UI to native generation.
Content: loadModel, generateStream, cancel, release, getEngineStats, /health and /v1/chat/completions.

### P07 - Q8 W8A8 Prefill
Title: Q8 W8A8 prefill cuts first-token latency by 9.34x on device.
Content: F16 prefill TTFT 7361.61 ms versus Q8 prefill TTFT 788.56 ms.

### P08 - Native Q8 Deployment
Title: Native Q8 keeps sub-second TTFT while removing repeated conversion cost.
Content: Average repeated TTFT 717.259 ms, preserving the 0.7 to 0.84 second range.

### P09 - Prefix KV Cache
Title: Prefix KV Cache converts repeated Agent prompts into immediate TTFT gains.
Content: Exact-repeat TTFT falls from 1902.63 ms to 613.658 ms; TPS improves from 3.383 to 4.734.

### P10 - 8K Context
Title: The runtime supports longer Agent memory beyond the previous 1024-token boundary.
Content: 8192-token context validated; 5488-token prompt prefetched in chunks and generated valid output.

### P11 - LM-head and Descriptor Cache
Title: Latest LM-head path reaches zero hot descriptor allocation.
Content: 0.6B median TTFT 392.802 ms, 1.7B median TTFT 841.991 ms, submit median 2 and descriptor allocation median 0.

### P12 - Why Not OpenCL
Title: OpenCL is blocked by Android deployment constraints, not by lack of effort.
Content: vendor libOpenCL.so and DT_NEEDED dependencies are isolated by linker namespace; root-level workarounds are not acceptable for deployable runtime.

### P13 - Why Not Native ggml Vulkan
Title: Native ggml Vulkan is too fragile for the target Adreno stack.
Content: Vulkan 1.2 assumptions conflict with Adreno 650 Vulkan 1.1 guarantee; lowered checks still produced corrupted output, so OSH26 uses controlled Vulkan 1.1 kernels and correctness gates.

### P14 - Alternatives
Title: OSH26 occupies the gap between model libraries, server runtimes, and mobile graph engines.
Content: llama.cpp is the base, vLLM is the inspiration, MNN is a mobile operator framework; OSH26 is the endpoint Agent runtime layer.

### P15 - Novelty
Title: The contribution is system integration plus endpoint-specific optimization.
Content: JNI/HTTP service, streaming/cancel lifecycle, Q8 Vulkan path, Prefix KV Cache, 8K context, benchmark gates, backend failure analysis.

### P16 - Closing
Title: OSH26 Runtime makes local Android Agents deployable.
Content: Faster first token, reusable memory, small-concurrency readiness, and real-device evidence make the project more than a wrapper around llama.cpp.

---

## X. Speaker Notes Guidance

Use concise Chinese narration, conclusion first. Each notes page should explain why the slide matters for the course project and how it answers concerns about repeated wheel-building and novelty.

---

## XI. Technical Constraints

- All pages are SVG with viewBox `0 0 1280 720`.
- Use only colors and typography in `spec_lock.md`.
- Use editable SVG shapes and charts, no external images.
- Avoid banned SVG features: style tags, class attributes, foreignObject, masks, scripts, and group opacity.

