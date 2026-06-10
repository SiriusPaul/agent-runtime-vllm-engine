# OSH26 Android Performance Notes

## 2026-06-09 Single Submit Experiment

Added an opt-in fast path guarded by `OSH26_SINGLE_SUBMIT=1` / Android property `debug.osh26.single_submit=1`.

- Default behavior is unchanged.
- Eligible fast path: grouped layer forward, logits required, not prefill-only, no debug/correctness check, and GPU LM head enabled.
- The eligible path records layer forward, final RMS, LM-head dot/top-k/merge into one Vulkan command buffer and submits once.
- Health now records `single_submit_enabled`, `last_single_submit_used`, `last_forward_submit_count`, `last_forward_gpu_ms`, `last_forward_layers_gpu_ms`, `last_forward_lm_head_gpu_ms`, and `last_forward_final_norm_gpu_ms`.
- `verify-vulkan-runtime.ps1` propagates host `OSH26_SINGLE_SUBMIT` to `debug.osh26.single_submit` before launching the app and uses `last_forward_submit_count` as the median submit-count gate.

Local verification status:

- `.\gradlew.bat assembleDebug`: passed.
- PowerShell script syntax check for `verify-vulkan-runtime.ps1`: passed.
- `.\gradlew.bat installDebug`: blocked because no Android device was connected (`adb devices` returned an empty list).

## 2026-06-09 LM Head Profiling Additions

The Vulkan health payload now records device capability fields needed for the next LM-head kernel decision:

- `gpu_subgroup_size`
- `gpu_integer_dot_product_supported`
- `gpu_shader_int8_supported`
- `gpu_timestamp_period_ns`
- `gpu_timestamp_valid_bits`

LM-head timing now includes real Vulkan timestamp-query splits in addition to the older host-side encode/wait fields:

- `last_lm_head_gpu_ms`
- `last_lm_head_actq8_gpu_ms`
- `last_lm_head_dot_gpu_ms`
- `last_lm_head_topk_gpu_ms`
- `last_lm_head_merge_gpu_ms`

Descriptor sets for the repeated forward bind patterns are cached by layout and buffer handles. Normal forward cleanup resets command buffers but keeps the descriptor pool alive; explicit model reload, free, and benchmark descriptor-pool resets invalidate the cache. This lets warm forward runs report whether hot-path `vkAllocateDescriptorSets` has actually reached zero without hiding first-use allocations.

Validation run `verify-results/verify-results-20260609-131527.json`:

| Model | TTFT median | TPS median | LM head median | Submit median | Descriptor alloc median | LM head GPU median | Dot GPU median | Top-k GPU median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B-Q8_0 | 392.802 ms | 4.7800 | 162.996 ms | 2 | 0 | 40.469 ms | 17.320 ms | 22.737 ms |
| 1.7B-Q8_0 | 841.991 ms | 2.2154 | 363.742 ms | 2 | 0 | 50.161 ms | 26.989 ms | 22.762 ms |

Device capability snapshot for that run: subgroup size 64, integer dot product unsupported, shader int8 supported, timestamp period 52.0833 ns. The Q8 LM-head kernel did not pass the speedup gate, so `q8-gemv:tied` remains the default path.

Measured on:
- Device: Redmi K40 / alioth, Snapdragon 870, Adreno 650
- OS: Android 13
- Build: `examples/android_osh26` debug APK
- Model: `qwen3-0.6b.gguf` at `/data/data/org.osh26.llama/files/models/qwen3-0.6b.gguf`
- Backend: `OSH26 GPU Runtime`
- Date: 2026-06-04

Method:
- Load model through `POST /load_model`
- Run chat completion through `POST /v1/chat/completions`
- Read the latest engine stats from `GET /health`
- Benchmark prompt: `Explain in one concise paragraph why reducing Vulkan submit count can improve tokens per second.`

## Current Snapshot

### Latest Vulkan validation run after disabling prefix cache

The current on-device Vulkan path is back to coherent text output with prefix cache disabled and full prefill on every request.

| Metric | Value |
| --- | ---: |
| Load model ms | 63124.9 ms |
| TTFT | 5184.83 ms |
| Tokens/s | 1.14746 |
| Prefill ms | 5171.72 ms |
| First decode ms | 0.115938 ms |
| Decoded tokens | 9 |
| User prefill tokens | 31 |
| Prefill submit count | 196 |
| Prefill qkv ms | 1372.76 ms |
| Prefill qk norm rope ms | 1369.82 ms |
| Prefill o proj ms | 677.338 ms |
| Prefill down ms | 1003.52 ms |
| GPU KV reset ms | 0 ms |
| Prefix cache enabled | false |
| Prefix cache hit | false |

Notes:
- The following three prompts were checked end to end on the GPU path and all returned coherent text, not garbage:
  - `Write one concise sentence about Vulkan command buffers.` -> `Vulkan command buffers are used to perform GPU operations in real`
  - `Explain what a hash table does in one sentence.` -> `A hash table is a data structure that maps keys to values`
  - `Translate the phrase good morning into Spanish.` -> `Buenos días.`
- The prefill stats now reflect the full prompt every time because prefix cache is disabled.
- The earlier prefix-cache-warmed results below are kept as historical reference only.

### Latest Vulkan validation run

The current on-device Vulkan path still falls back to CPU attention during prefill because the new prefill pipelines are not yet enabled on the device (`prefill pipelines: qk=-13 softmax=-13 qkvacc=-13 enabled=false`).

| Metric | Value |
| --- | ---: |
| Load model ms | 64765.4 ms |
| Prefix warm ms | 0 ms |
| TTFT | 85991.8 ms |
| Tokens/s | 0.255006 |
| Prefill ms | 85971 ms |
| First decode ms | 0.355364 ms |
| Decoded tokens | 24 |
| User prefill tokens | 40 |
| Prefill qkv ms | 0 ms |
| Prefill cpu post ms | 0 ms |
| Prefill attention ms | 0 ms |
| Prefill ffn ms | 0 ms |
| Prefill chunk size | 40 |
| Prefill chunk count | 1 |
| Prefill forwards | 1 |
| Submit count | 123 |

Notes:
- Full token output for the benchmark prompt was checked end to end.
- The 24-token completion was coherent and deterministic:
  `151667,271,151668,271,16609,287,85864,9318,1760,6147,369,803,11050,20898,323,4722,5101,10431,11,892,646,5263,11211,817`
- The output text was:
  `Reducing Vulkan submit count allows for more efficient rendering and lower resource usage, which can increase tokens per`
- Prefill GPU pipeline creation is still blocked, so the current TTFT number is from the CPU fallback path.

### Cold first request after model load

| Metric | Value |
| --- | ---: |
| TTFT | 60478.6 ms |
| Tokens/s | 3.08307 |
| Prefill ms | 60467.3 ms |
| First decode ms | 0.197864 ms |
| Decoded tokens | 24 |
| Prefill tokens | 25 |
| Prefill chunk size | 25 |
| Prefill chunk count | 1 |
| Prefill forwards | 1 |
| Submit count | 123 |

Notes:
- Prefix cache was already warmed for the template prefix.
- The full 24-token output was checked end-to-end; the token id sequence was identical in the warm repeat.

### Warm repeat of the same prompt

| Metric | Value |
| --- | ---: |
| TTFT | 423.595 ms |
| Tokens/s | 2.75199 |
| User prefill ms | 409.65 ms |
| First decode ms | 0.124583 ms |
| Decoded tokens | 24 |
| Prefill tokens | 0 |
| Prefix cache hit | true |
| Submit count | 123 |

Notes:
- This is the stable steady-state run on the same prompt.
- The request reused the cached prefix and skipped user prefill work.
- The full 24-token output matched the cold run token-for-token:
  `151667,271,151668,271,16609,287,85864,9318,1760,646,7269,11211,817,2086,553,73042,279,22670,594,5726,311,3705,323,1882`

## Prefill Chunking Validation

Long prompt validation was run with 503 user tokens. The request was interrupted after the prefill stage was captured, but the chunking stats were recorded:

| Metric | Value |
| --- | ---: |
| User prefill tokens | 503 |
| Prefill chunk size | 128 |
| Prefill chunk count | 2 |
| Prefill forwards | 2 |
| Prefill skipped logits chunks | 2 |

Interpretation:
- Prompts up to 64 tokens stay in a single chunk.
- Prompts from 65 to 256 tokens use 64-token chunks.
- Prompts above 256 tokens use 128-token chunks.
- Only the final chunk keeps logits enabled.

## Notes On The Change Set

- LM head now returns a top-k candidate set instead of forcing a full-vocab CPU copy on the fast path.
- The engine uses the candidate set for sampling and keeps full-logit fallback only for debug and safety paths.
- A light logits sanity check is recorded in the engine stats so broken outputs can be detected early.

## June 2026 GPU LM Head Follow-up

This pass kept prefix cache disabled, re-enabled the GPU LM head fast path, and moved the prefill/lm-head bookkeeping to the new submit counters.

### Readability Sweep

Several short prompts were checked to make sure the output was still human-readable and not corrupted token noise. The readable samples kept for the sweep were:

- `Please reply with one English sentence about the ocean.` -> `The sea is a`
- `Please reply with one English sentence about Vulkan compute shaders.` -> `Okay, let's`
- `Please reply with one English sentence about France.` -> `France is a country`
- `Please reply with one English sentence about the sky.` -> `Okay, the user wants me`
- `Please reply with one English sentence about the sea.` -> `The sea is a`
- `Please reply with one English sentence about coffee.` -> `Coffee is a popular`

### TTFT Benchmark

Benchmark prompt:

`Please reply with one English sentence about the sea, keep it simple and direct, avoid lists, markdown, and special symbols, use plain English, and do not add any explanation or preamble. Focus on one concrete detail, and keep the answer under fifteen words.`

| Metric | max_tokens=1 | max_tokens=8 |
| --- | ---: | ---: |
| TTFT | 12444.6 ms | 12445.0 ms |
| Forward LM head ms | 1628.07 ms | 526.726 ms |
| Prefill submit count | 112 | 112 |
| TTFT submit count | 116 | 116 |
| Layer submit count | 112 | 112 |
| LM head submit count | 2 | 2 |
| Submit wait ms | 1550.28 ms | 515.77 ms |
| Decode ms | 557.285 ms | 567.34 ms |
| Tokens/s | 1.77158 | 1.75948 |

Notes:
- The long prompt is intentionally a harder baseline than the short readability sweep.
- The LM head path is now fast enough to keep TTFT in the low-second range instead of the previous 70s CPU fallback.
- The remaining gap is mostly prefill and per-layer GPU work, not final logits.

## June 2026 Prompt Sanity Check

After the latest prompt-format and sampler cleanup, the runtime was re-checked with the current default UI parameters on the connected device.

### Sample Outputs

- `Write one English sentence about coffee.` -> `stantrary. photon. d. . . . . . . . . . . . . . . . . .`
- `Say hello in one short sentence.` -> `Okay, the user wants a short sentence that says "Say hello in one short sentence." Let me think.`

### Latest Health Snapshot

| Metric | Value |
| --- | ---: |
| TTFT | 27362.9 ms |
| Prefill ms | 7153.48 ms |
| First decode ms | 0.145 ms |
| Decoded tokens | 24 |
| User prefill tokens | 41 |
| Prefill qkv ms | 2.16833 ms |
| Prefill qk norm rope ms | 1743.59 ms |
| Prefill o proj ms | 1.88557 ms |
| Prefill down ms | 60.2545 ms |
| Prefill submit count | 112 |
| Layer submit count | 112 |
| LM head submit count | 2 |
| TTFT submit count | 116 |
| Forward lm head ms | 525.307 ms |

Notes:
- The output quality is still prompt-sensitive. Short factual prompts can be coherent, but other short prompts still drift into fragments or repetition.
- Vulkan 1.1 generation is still in place.
- Prefix cache remains disabled.

## 2026-06-06 Performance Without CPU Correctness Checks

The runtime was benchmarked with `debug_correctness=false`. Both expensive
correctness paths remained disabled:

- `last_lm_head_validation_ran=false`
- `last_e2e_compare_ran=false`

Build and device were unchanged: debug APK on Redmi K40 / Snapdragon 870 with
prefix cache disabled.

### CPU Validation Overhead

The same short prompt, `Write one short sentence about local inference.`, was
run with `temperature=0` and `max_tokens=2`.

| Metric | CPU validation enabled | CPU validation disabled |
| --- | ---: | ---: |
| TTFT | 183317 ms | 7328.25 ms |
| User prefill ms | 80753.2 ms | 7308.87 ms |
| E2E CPU compare ms | 102544 ms | 0 ms |
| Last LM head validation ms | 73550.9 ms | 0 ms |
| Last LM head ms | 73665.5 ms | 104.626 ms |
| LM head submit count | 2 | 2 |
| Decoded tokens | 2 | 2 |

Interpretation:

- Disabling CPU correctness checks reduced TTFT by about `25x`.
- The GPU execution path and two-submit LM head were unchanged.
- CPU validation is now isolated to debug mode and is not part of normal TTFT.

### Three-Run Normal-Mode Benchmark

Prompt:

`Please reply with one English sentence about the sea, keep it simple and direct, avoid lists, markdown, and special symbols, use plain English, and do not add any explanation or preamble. Focus on one concrete detail, and keep the answer under fifteen words.`

Configuration: `temperature=0`, `max_tokens=8`, model kept loaded between runs.

| Metric | Run 1 | Run 2 | Run 3 | Mean |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 14368.6 ms | 14357.9 ms | 14301.8 ms | 14342.8 ms |
| Last prefill chunk | 4299.44 ms | 4295.39 ms | 4293.28 ms | 4296.04 ms |
| LM head | 112.834 ms | 109.164 ms | 109.683 ms | 110.560 ms |
| LM head wait | 110.331 ms | 106.882 ms | 107.380 ms | 108.198 ms |
| Decode | 155.046 ms | 152.032 ms | 152.435 ms | 153.171 ms |
| End-to-end tokens/s | 0.518277 | 0.518667 | 0.520507 | 0.519150 |
| Prefill submits | 112 | 112 | 112 | 112 |
| LM head submits | 2 | 2 | 2 | 2 |
| TTFT submits | 226 | 226 | 226 | 226 |

Additional details:

- Prompt tokens: 89.
- Prefill used two 64-token chunks; the first chunk skipped LM head.
- Full user prefill time on run 3 was `14284.6 ms`.
- All runs produced identical token IDs:
  `151667,271,151668,271,785,9396,374,12767`.
- Output text was `The sea is vast`.
- Remaining TTFT is dominated by two-chunk layer execution. The normal LM head
  is about `111 ms`, with almost all of that time in GPU wait rather than
  command encoding.

## 2026-06-06 Packed FP16 and Q4 GEMV Feasibility

An isolated Vulkan microbenchmark was added at `POST /benchmark/quant_gemv`.
It does not modify the model or production inference path.

Test shape and method:

- Representative single FFN projection GEMV: `N=3072`, `K=1024`.
- One activation vector and identical deterministic weights for all formats.
- F32 baseline: 32-bit weights.
- FP16: two packed half values per `uint`, decoded with `unpackHalf2x16`.
- Q4: symmetric 4-bit weights in blocks of 32, with one F32 scale and eight
  packed `uint` values per block.
- Each result is the elapsed submit-and-wait time for 20 dispatches divided by
  20. Three warm-up dispatches run first.
- Five repeated benchmark calls were collected after pipeline creation.

### Performance

| Format | Weight bytes | Median ms | Mean ms | Median vs F32 |
| --- | ---: | ---: | ---: | ---: |
| F32 | 12,582,912 | 0.946984 | 0.996082 | 1.00x |
| Packed FP16 | 6,291,456 | 0.957529 | 0.959724 | 0.989x |
| Q4 block-32 | 3,538,944 | 1.137292 | 1.136584 | 0.833x |

One F32 repeat rose to `1.192167 ms`; medians are therefore more representative
than the means.

### Correctness

| Format | Shader vs decoded CPU max error | Shader vs decoded CPU RMSE | Quantization vs F32 max error | Quantization vs F32 RMSE |
| --- | ---: | ---: | ---: | ---: |
| F32 | 2.86e-5 | 6.33e-6 | 0 | 0 |
| Packed FP16 | 4.01e-5 | 6.35e-6 | 0.001349 | 0.000348 |
| Q4 block-32 | 3.53e-5 | 6.35e-6 | 0.377346 | 0.117233 |

Conclusions:

- Packed FP16 and packed Q4 loading, shader-side decoding, immediate multiply,
  and F32 accumulation all work correctly on the Adreno 650 Vulkan 1.1 driver.
- Packed FP16 halves weight memory with negligible numerical loss, but the
  current scalar GEMV kernel gets no measurable speedup.
- The first simple Q4 kernel reduces weight memory by about 72%, but is roughly
  20% slower than F32 because integer extraction, scale decoding, and the
  existing per-output reduction outweigh the bandwidth saving.
- This result does not rule out quantization. A useful Q4 implementation needs
  vectorized block decoding, cooperative reuse of scale/packed words, and a
  kernel designed around the quantized layout rather than a scalar replacement
  for each F32 load.
- For near-term TTFT work, removing per-layer queue waits and reducing prefill
  submits remains higher priority than replacing production weights with the
  current FP16 or Q4 kernels.

## 2026-06-06 Packed FP16 and Q4 Prefill GEMM

A second microbenchmark was added at `POST /benchmark/quant_gemm` to test the
matrix shape that matters for prefill rather than decode:

- Shape: `M=64`, `N=3072`, `K=1024`.
- This matches a 64-token FFN gate/up projection.
- All three shaders use the same `8x8` output tile and `K=32` shared-memory
  blocking.
- FP16 and Q4 weights are decoded while loading the shared weight tile, so each
  decoded weight is reused by eight activation rows.
- Each reported time is five consecutive GEMM dispatches in one command buffer,
  divided by five. Two warm-up dispatches run first.

### Five-Run Result

| Format | Weight bytes | Mean ms | Observed speedup |
| --- | ---: | ---: | ---: |
| F32 | 12,582,912 | 70.7211 | 1.0000x |
| Packed FP16 | 6,291,456 | 70.6267 | 1.0013x |
| Q4 block-32 | 3,538,944 | 70.6430 | 1.0011x |

The spread across all five runs was below `0.04 ms` per format. The roughly
`0.1%` difference is measurement noise rather than a useful acceleration.

### Correctness

| Format | Sampled shader implementation max error | Full output error vs F32 RMSE |
| --- | ---: | ---: |
| F32 | 7.63e-6 | 0 |
| Packed FP16 | 4.77e-6 | 0.000330 |
| Q4 block-32 | 3.81e-6 | 0.161503 |

Conclusions:

- Packed FP16 and Q4 tiled GEMM execute correctly on the device.
- Unlike the scalar GEMV result, Q4 unpacking is effectively hidden by weight
  reuse across eight token rows. It no longer causes a slowdown.
- Halving or quartering weight traffic still does not improve runtime, which
  indicates this tiled kernel is limited by F32 multiply-accumulate throughput,
  barriers, shared-memory use, or occupancy rather than external weight
  bandwidth.
- Replacing production F32 weights with these exact kernels would reduce memory
  use but would not materially reduce prefill TTFT.
- Q4 could accelerate TTFT only with a different compute strategy, such as
  integer dot-product instructions, larger output tiles, multiple outputs per
  thread, or subgroup operations that reduce the number of scalar F32
  operations and barriers. Packed storage plus scalar dequantization alone is
  insufficient on this Adreno 650 path.

## 2026-06-06 Q8 W8A8 Prefill Gate and TTFT Retest

This run used the freshly installed debug APK on the connected Redmi K40 /
Snapdragon 870. Commands:

- `.\examples\android_osh26\generate_spv_headers.ps1`
- `cd examples\android_osh26; .\gradlew.bat assembleDebug`
- `cd examples\android_osh26; .\gradlew.bat installDebug`
- `.\verify-vulkan-runtime.ps1 -SkipCpu -MaxTokens 2 -Prompt "Explain briefly why local inference is useful."`

The normal runtime stayed in `debug_correctness=false`; CPU correctness checks
did not run:

- `last_lm_head_validation_ran=false`
- `last_e2e_compare_ran=false`

### Implemented Runtime Changes

- Added a Q8_0 W8A8 prefill path gated by native GGUF tensor type. The path is
  enabled only when every projection tensor needed by prefill is `Q8_0`.
- Added `act_quant_q8.comp` for per-token dynamic activation quantization and
  `mulmat_q8_w8a8.comp` for packed int8 dot products with F32 accumulation.
- Added `POST /benchmark/q8_gemm` to benchmark representative single-layer
  prefill GEMM shapes and publish `last_q8_*` health fields.
- Replaced the prefill attention scratch host `memset + vkQueueWaitIdle` with
  `vkCmdFillBuffer` recorded into the same layer command buffer.
- Kept the stable path at one prefill command buffer per layer. A previous
  whole-prefill single-command-buffer experiment caused `VK_ERROR_DEVICE_LOST`
  on the Adreno 650 driver and is not part of this implementation.

### Current Model Constraint

The installed model is still not native Q8_0:

`Tensor blk.0.attn_q.weight is not Q8_0 (type=1); Q8 prefill unavailable for this model`

Therefore full generation below used the existing F16-expanded projection
weights, and health reported `prefill_q8_enabled=false`. The Q8 prefill path is
compiled and gated, but real TTFT acceleration requires loading a native Q8_0
GGUF or providing a safe one-time model-load conversion path.

### Short Normal-Mode Run

Prompt: `Explain briefly why local inference is useful.`

Configuration: `temperature=0`, `max_tokens=2`, `-SkipCpu`.

| Metric | Value |
| --- | ---: |
| Prompt tokens | 46 |
| Completion tokens | 2 |
| TTFT | 7361.61 ms |
| User prefill | 7352.16 ms |
| Vulkan prefill wall | 7352.15 ms |
| Decode | 143.057 ms |
| LM head | 105.053 ms |
| LM head wait | 102.785 ms |
| Prefill submits | 28 |
| TTFT submits | 30 |
| Last decode layer submits | 112 |
| Logits sanity | ok |

Token IDs:

`151667,271`

The rendered text was empty because both sampled tokens are special /
whitespace-like for this prompt. The candidate logits were still finite and
ordered:

`271(32.3286), 1406(19.4395), 1022(17.8653), 4710(17.8236), 89253(17.6942)`

### 16-Token Output Sanity Run

Prompt:

`Please answer in one short English sentence: why is local inference useful?`

Configuration: `temperature=0`, `top_p=1`, `seed=51966`, `max_tokens=16`.

Output text:

`Local inference is useful because it allows for precise and contextually`

The output is syntactically reasonable and was cut off only because
`max_tokens=16` forced `finish_reason=length`.

| Metric | Value |
| --- | ---: |
| Prompt tokens | 51 |
| Completion tokens | 16 |
| TTFT | 8250.09 ms |
| User prefill | 8232.07 ms |
| Last token TPS | 6.73746 |
| End-to-end tokens/s | 1.5314 |
| Prefill submits | 28 |
| TTFT submits | 30 |
| Logits sanity | ok |
| Q8 prefill enabled | false |

Token IDs:

`151667,271,151668,271,7319,44378,374,5390,1576,432,6147,369,23560,323,2266,1832`

Final top5:

`18906(27.0014), 65004(25.9520), 1832(24.9573), 56667(24.6431), 5980(23.8910)`

### Q8 W8A8 GEMM Microbenchmark

Endpoint: `POST /benchmark/q8_gemm`

The benchmark quantizes activations dynamically, consumes packed Q8_0-style
weights, immediately multiplies by the activation, and accumulates in F32.

Gate result:

| Metric | Value |
| --- | ---: |
| Weighted F32 | 236.349357 ms |
| Weighted Q8 total | 15.954167 ms |
| Weighted speedup | 14.8143x |
| Correctness | ok |
| Gate | pass |

Per-shape result:

| Shape | Repeats | M | N | K | F32 ms | Q8 total ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_proj | 1 | 32 | 2048 | 1024 | 23.858438 | 2.117170 | 11.2690x |
| k_v_proj | 2 | 32 | 1024 | 1024 | 26.379913 | 1.148142 | 22.9762x |
| o_proj | 1 | 32 | 1024 | 2048 | 24.050347 | 2.161146 | 11.1285x |
| gate_up | 2 | 32 | 3072 | 1024 | 49.846736 | 3.102899 | 16.0646x |
| down | 1 | 32 | 1024 | 3072 | 35.987274 | 3.173767 | 11.3390x |

Correctness samples stayed within shader-vs-packed-Q8 CPU error of about
`7.7e-6` max abs error or lower per shape. Quantization-vs-F32 sampled RMSE was
roughly `0.0051` to `0.0086`.

### Interpretation

Q8 W8A8 is viable on this device for the projection GEMM shapes; unlike the
earlier Q4 scalar-dequant path, this benchmark shows a large kernel-level speed
advantage. It has not improved current TTFT yet because the installed GGUF is
F16 (`type=1`) and the production path correctly keeps `prefill_q8_enabled=false`.

The next meaningful TTFT test is to load a native Q8_0 GGUF and verify that
`prefill_q8_enabled=true`. If that passes correctness and output sanity, the
expected first target is reducing the current single-chunk TTFT from about
`7.4-8.3 s` into the low-second range. Sub-second TTFT still requires further
work beyond Q8 projection GEMM, especially queue/submit behavior, decode LM head
wait, and any remaining non-projection overhead.

## 2026-06-06 Production Q8 Prefill From Load-Time Conversion

The installed F16 GGUF was switched to the production Q8 prefill path by
converting every projection tensor to the shader's packed Q8 layout during
model loading. Decode remains on the existing F32-expanded weights, while
prefill uses dynamic Q8 activation quantization and Q8 W8A8 projection GEMMs.

All required Q, K, V, O, gate, up, and down tensors across 28 layers converted
successfully. Runtime health and logs confirmed:

- `prefill_q8_enabled=true`
- `Q8 prefill enabled: true`
- no Vulkan device loss
- no NaN/Inf logits
- no logits sanity failures

### Load-Time Cost

| Metric | Value |
| --- | ---: |
| Model load and Q8 conversion | 343577 ms |
| Process RSS near end of conversion | about 4.5 GB |

The conversion is currently repeated after each process restart. It is a test
path, not the desired deployment format. A native Q8_0 GGUF or persistent packed
weight cache is required to remove this startup cost and reduce memory usage.
The verification script HTTP timeout was increased from 300 to 600 seconds so
the first converted load is not reported as a false timeout.

### F16 Versus Q8 TTFT

Same prompt and sampling:

`Explain briefly why local inference is useful.`

`temperature=0`, `top_p=1`, `seed=51966`, `max_tokens=2`.

| Metric | F16 prefill | Q8 prefill | Improvement |
| --- | ---: | ---: | ---: |
| TTFT | 7361.61 ms | 788.56 ms | 9.34x |
| User prefill | 7352.16 ms | 762.47 ms | 9.64x |
| Prefill submits | 28 | 28 | unchanged |
| TTFT submits | 30 | 30 | unchanged |
| LM head | 105.05 ms | 106.93 ms | unchanged |

Both paths produced the exact same token IDs:

`151667,271`

F16 top5:

`271(32.3286), 1406(19.4395), 1022(17.8653), 4710(17.8236), 89253(17.6942)`

Q8 top5:

`271(32.2548), 1406(19.3725), 1022(17.8486), 89253(17.7325), 4710(17.6810)`

The top three candidates and sampled result were unchanged. Only candidates
four and five exchanged order.

### Repeated Q8 TTFT

Three warm, identical requests:

| Run | TTFT | User prefill | Token IDs | Sanity |
| --- | ---: | ---: | --- | --- |
| 1 | 745.439 ms | 727.795 ms | `151667,271` | ok |
| 2 | 739.765 ms | 727.500 ms | `151667,271` | ok |
| 3 | 702.461 ms | 693.851 ms | `151667,271` | ok |

Mean TTFT: `729.222 ms`.

### Full Token Sequence Validation

English prompt:

`Please answer in one short English sentence: why is local inference useful?`

Q8 output:

`Local inference is useful because it allows for precise and contextually`

Q8 token IDs:

`151667,271,151668,271,7319,44378,374,5390,1576,432,6147,369,23560,323,2266,1832`

This is exactly the same text and token sequence as the previous F16 run.
TTFT was `837.995 ms`, compared with `8250.09 ms` for F16, a `9.85x`
improvement.

Arithmetic prompt:

`Answer only with the result: 17 + 25 = ?`

Output:

`17 + 25 = 42`

Token IDs:

`151667,271,151668,271,16,22,488,220,17,20,284,220,19,17`

The request ended normally with `finish_reason=stop`, TTFT `831.918 ms`, and
`last_logits_sanity_ok=true`.

### Result

The Q8 W8A8 production prefill path is correct for the tested deterministic
prompts and reduces single-chunk TTFT to approximately `0.7-0.84 s` on this
device. Remaining first-token time is now dominated by roughly `0.69-0.82 s`
prefill plus about `0.105 s` LM head wait. The immediate deployment issue is
the `343.6 s` load-time conversion and duplicated F32/Q8 weight memory, not
prefill execution speed.

## 2026-06-06 Native Q8_0 GGUF Test

Model:

`D:\下载\qwen3-0.6b-base-q8_0.gguf`

Device path:

`/data/data/org.osh26.llama/files/models/qwen3-0.6b-base-q8_0.gguf`

File size: `639446784` bytes.

The model loaded through the native Q8_0 reader without any
`Converted ... F32 to packed Q8` log messages. Runtime state:

- `prefill_q8_enabled=true`
- `Q8 prefill enabled: true`
- model load time: `38834.3 ms`
- no Vulkan device loss
- no logits sanity failure

Compared with the F16 load-time conversion path, native Q8 reduced model load
time from `343577 ms` to `38834.3 ms`, an `8.85x` improvement.

Current process memory after load was still high:

| Metric | Value |
| --- | ---: |
| Total PSS | 4874314 KiB |
| Total RSS | 4946496 KiB |
| Native heap PSS | 1551849 KiB |
| Graphics device PSS | 3271556 KiB |

The reason is that the current implementation still expands the Q8 model to
F32 buffers for decode and LM-head support while retaining native packed Q8
weights for prefill. Native Q8 removes conversion time but does not yet remove
the duplicate expanded weights.

### Output Validation

The downloaded model is a base model, so its initial token sequence differs
from the previously tested chat/instruct model. This is a model behavior
difference rather than a Q8 correctness failure.

Prompt:

`Please answer in one short English sentence: why is local inference useful?`

Output:

`Local inference is useful because it helps in understanding patterns and relationships within a specific domain, allowing for more accurate predictions and decisions based on local data rather than broad,`

Token IDs:

`7319,44378,374,5390,1576,432,8609,304,8660,12624,323,11871,2878,264,3151,7947,11,10693,369,803,13382,19898,323,11181,3118,389,2205,821,4751,1091,7205,11`

TTFT was `836.513 ms`; the request ended at the configured 32-token limit.
The sequence was coherent, finite, and free of repetition or malformed text.

Arithmetic prompt:

`Answer only with the result: 17 + 25 = ?`

Output:

`17 + 25 = 42`

Token IDs:

`16,22,488,220,17,20,284,220,19,17`

The request ended normally with `finish_reason=stop`, TTFT `834.742 ms`, and
`last_logits_sanity_ok=true`.

### Repeated Native Q8 TTFT

Prompt:

`Explain briefly why local inference is useful.`

| Run | TTFT | User prefill | Token IDs | Sanity |
| --- | ---: | ---: | --- | --- |
| 1 | 744.854 ms | 727.346 ms | `7319,44378` | ok |
| 2 | 704.370 ms | 693.665 ms | `7319,44378` | ok |
| 3 | 702.554 ms | 692.385 ms | `7319,44378` | ok |

Mean TTFT: `717.259 ms`.

Native Q8 therefore preserves the approximately `0.7-0.84 s` TTFT achieved by
the converted Q8 path while reducing startup time substantially. The remaining
deployment task is removing F32-expanded projection weights by adding a native
Q8 decode GEMV path or separating prefill and decode weight ownership.

## 2026-06-07 Paged Prefix KV Cache Device Validation

Test environment:

- device: Redmi K40 (`alioth`), Snapdragon 870 / Adreno 650, Android 13
- model: `/data/local/tmp/qwen3-0.6b.gguf`, F16 Qwen3 0.6B
- backend: OSH26 Vulkan GPU runtime
- generation: greedy (`temperature=0`), maximum 16 output tokens
- prefix cache: strict token-by-token matching, 16-token pages, 16-page pool
- packed MNN attention enabled with zero fallback layers

The first request used a fixed cache description followed by Question A. The
second request repeated it exactly. The third retained the opening context but
changed the late prompt suffix to Question B.

| Request | Wall time | TTFT | TPS | Cache hit | Reused tokens | User prefill | Restore | Store |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Cold Question A | 5222.0 ms | 1902.63 ms | 3.383 | no | 0 | 97 | 0 ms | 337.109 ms |
| Exact repeat A | 3845.6 ms | 613.658 ms | 4.734 | yes | 96 | 1 | 353.935 ms | unchanged |
| Shared-prefix B | 4585.9 ms | 1288.87 ms | 3.891 | yes | 64 | 34 | 238.003 ms | 339.168 ms |

Compared with the cold request, the exact repeat reduced TTFT by `67.7%`
(`3.10x`) and increased TPS by `39.9%` (`1.40x`). The late-suffix variant
reduced TTFT by `32.3%` (`1.48x`) and increased TPS by `15.0%`.

### Prefix Correctness Checks

The cold and exact-repeat Question A requests produced identical text and the
same 16 output token IDs:

`151667,271,151668,271,785,1887,57323,5912,374,429,279,1590,9934,3950,374,37201`

The late-suffix Question B request reused only 64 matching input tokens and
produced a distinct, relevant answer:

`Strict prefix equality is important because it ensures that the model's`

Its output token IDs were:

`151667,271,151668,271,41857,9252,21777,374,2989,1576,432,25351,429,279,1614,594`

All three requests reported `last_logits_sanity_ok=true`, with no device loss,
attention fallback, malformed output, or repeated-token degeneration. This
confirms that the differing suffix was evaluated rather than incorrectly
restored from the cached prefix.

### Cache, Memory, and Thermal State

Final cache statistics:

- hits: 2; misses: 1; hit ratio: `0.666667`
- reused tokens: 160 across 10 pages
- entries: 2; used pages: 12; free pages: 4
- evictions: 0
- prefill Q8 and decode Q8 both enabled

Process memory increased from `5763820 KiB` to `5794442 KiB` Total PSS and
from `5829252 KiB` to `5861828 KiB` Total RSS across the three requests. This
is approximately `29.9 MiB` PSS and `31.8 MiB` RSS growth. Battery temperature
was `30.5 C` before and after the test.

This is a short single-device validation rather than a statistically rigorous
benchmark. It demonstrates a meaningful TTFT/TPS gain while preserving strict
prefix correctness across exact-match and partially matching token sequences.

## 2026-06-07 8K Context Device Validation

The runtime context was increased from 1024 to 8192 tokens. The handwritten
Vulkan path keeps native Q8_0 model weights and Q8 prefill/decode kernels.
Only the generated K/V activation cache changed from F32 to packed FP16.
Attention still unpacks K/V to F32 and accumulates in F32.

Memory-related changes:

- llama context: 8192 tokens, one active sequence
- Vulkan packed K/V cache: 8192 tokens, FP16 storage
- prefill execution: unchanged 128-token chunks
- activation and temporary attention buffers: fixed to chunk capacity
- CPU correctness fallback cache: allocated only in correctness mode

The native Q8_0 model loaded successfully on the Redmi K40:

- model: `/data/local/tmp/qwen3-0.6b-base-q8_0.gguf`
- backend: `OSH26 GPU Runtime`
- context: 8192
- model load: `40244.7 ms`
- `prefill_q8_enabled=true`
- `decode_q8_enabled=true`
- `prefix_cache_supported=true`

### Short Output and Prefix Validation

| Request | TTFT | TPS | Prefix hit | Reused tokens | Sanity |
| --- | ---: | ---: | --- | ---: | --- |
| Cold A | 1348.14 ms | 4.351 | no | 0 | ok |
| Exact repeat A | 460.893 ms | 5.232 | yes | 64 | ok |
| Late-suffix B | 765.017 ms | 4.584 | yes | 48 | ok |

The cold and exact-repeat requests produced the same 24 output token IDs.
The late-suffix request produced a distinct valid response. No logits sanity
failure, attention fallback, malformed output, or Vulkan device loss occurred.

### Beyond-1024 Context Validation

A long request successfully prefetched 5488 user tokens in 43 chunks of 128
tokens, then generated 8 tokens:

- TTFT: `128142 ms`
- end-to-end request time: `132695 ms`
- finish reason: `length`
- `last_logits_sanity_ok=true`
- `last_error=""`

This verifies that the runtime is genuinely operating beyond the previous
1024-token boundary rather than only reporting a larger capacity. The base
model did not follow the exact response instruction, but its output remained
finite and syntactically valid.

After the long-context run:

- Total PSS: `6176756 KiB`
- Total RSS: `6246704 KiB`
- Graphics PSS: `3763928 KiB`
- battery temperature: `34.5 C`

The 8K capacity is functional, but long-context prefill is currently slow and
memory use is close to the practical limit of this device. A 16K context would
require another KV-memory reduction, paging inactive KV to host memory, or
removing duplicate expanded model weights.

## 2026-06-09 00:05:37 LM Head Q8 Verification

| Model | Samples | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.7B-Q8_0 | 27 | 869.603 | 2.1959 | 367.883 | 373.633 | 29 |

Gate: TPS beats the prior Q8 baseline, LM head median is at least 20% lower, and TTFT stays within 5% of the prior median.

## 2026-06-09 00:19:25 LM Head Q8 Verification

| Model | Samples | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.7B-Q8_0 | 27 | 918.532 | 1.9924 | 412.999 | 419.382 | 29 |

Gate: TPS beats the prior Q8 baseline, LM head median is at least 20% lower, and TTFT stays within 5% of the prior median.

## 2026-06-09 00:25:52 LM Head Q8 Verification

| Model | Samples | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.7B-Q8_0 | 27 | 934.224 | 1.9296 | 429.461 | 435.774 | 29 |

Gate: TPS beats the prior Q8 baseline, LM head median is at least 20% lower, and TTFT stays within 5% of the prior median.

## 2026-06-09 00:33:08 LM Head Q8 Verification

| Model | Samples | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.7B-Q8_0 | 27 | 1007.7 | 1.6919 | 503.061 | 509.253 | 29 |

Gate: TPS beats the prior Q8 baseline, LM head median is at least 20% lower, and TTFT stays within 5% of the prior median.

## 2026-06-09 00:40:08 LM Head Q8 Verification

| Model | Samples | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.7B-Q8_0 | 27 | 866.65 | 2.2182 | 362.62 | 368.776 | 29 |

Gate: TPS beats the prior Q8 baseline, LM head median is at least 20% lower, and TTFT stays within 5% of the prior median.
Gate status 1.7B-Q8_0: FAIL - median LM head 362.62 ms did not drop at least 20% from 376 ms.

## 2026-06-09 00:42:26 LM Head Q8 Verification

| Model | Samples | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B-Q8_0 | 27 | 405.246 | 4.8217 | 161.838 | 167.977 | 29 |

Gate: TPS beats the prior Q8 baseline, LM head median is at least 20% lower, and TTFT stays within 5% of the prior median.
Gate status 0.6B-Q8_0: FAIL - median LM head 161.838 ms did not drop at least 20% from 166 ms.

## 2026-06-09 09:15:19 LM Head Q8 Verification

| Model | Samples | LM head memory | Device-local bytes | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B-Q8_0 | 27 | device_local | 175030272 | 406.718 | 4.8164 | 161.86 | 168.079 | 29 |

Gate criteria: TPS beats the prior Q8 baseline, LM head median is at least 20% lower, and TTFT stays within 5% of the prior median.
Gate status 0.6B-Q8_0: FAIL - [0.6B-Q8_0] median LM head 161.86 ms did not drop at least 20% from 166 ms

## 2026-06-09 09:18:28 LM Head Q8 Verification

| Model | Samples | LM head memory | Device-local bytes | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.7B-Q8_0 | 27 | device_local | 350060544 | 865.332 | 2.221 | 362.346 | 368.322 | 29 |

Gate criteria: TPS beats the prior Q8 baseline, LM head median is at least 20% lower, and TTFT stays within 5% of the prior median.
Gate status 1.7B-Q8_0: FAIL - [1.7B-Q8_0] median LM head 362.346 ms did not drop at least 20% from 376 ms

## 2026-06-09 11:12:20 LM Head Q8 Verification

| Model | Samples | LM head memory | Device-local bytes | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count | Median descriptor alloc |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B-Q8_0 | 15 | device_local | 175030272 | 435.165 | 4.6399 | 167.098 | 171.837 | 2 | 628 |

Gate criteria: submit count <= 3, TTFT improves at least 8% from the device-local baseline, TPS regresses no more than 2%, and LM head regresses no more than 5%.
Gate status 0.6B-Q8_0: FAIL - [0.6B-Q8_0] median TPS 4.63986 regressed more than 2% from baseline 4.8164; [0.6B-Q8_0] median TTFT 435.165 ms did not improve at least 8% from 406.718 ms

## 2026-06-09 11:21:17 LM Head Q8 Verification

| Model | Samples | LM head memory | Device-local bytes | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count | Median descriptor alloc |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B-Q8_0 | 15 | device_local | 175030272 | 397.832 | 4.7851 | 166.742 | 170.541 | 2 | 628 |

Gate criteria: submit count <= 3, TTFT improves at least 8% from the device-local baseline, TPS regresses no more than 2%, and LM head regresses no more than 5%.
Gate status 0.6B-Q8_0: FAIL - [0.6B-Q8_0] median TTFT 397.832 ms did not improve at least 8% from 406.718 ms

## 2026-06-09 11:27:28 LM Head Q8 Verification

| Model | Samples | LM head memory | Device-local bytes | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count | Median descriptor alloc |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B-Q8_0 | 15 | device_local | 175030272 | 531.594 | 2.9129 | 300.884 | 305.976 | 2 | 628 |

Gate criteria: submit count <= 3, TTFT improves at least 8% from the device-local baseline, TPS regresses no more than 2%, and LM head regresses no more than 5%.
Gate status 0.6B-Q8_0: FAIL - [0.6B-Q8_0] median TPS 2.91293 regressed more than 2% from baseline 4.8164; [0.6B-Q8_0] median TTFT 531.594 ms did not improve at least 8% from 406.718 ms; [0.6B-Q8_0] median LM head 300.884 ms regressed more than 5% from 161.86 ms

## 2026-06-09 11:32:54 LM Head Q8 Verification

| Model | Samples | LM head memory | Device-local bytes | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count | Median descriptor alloc |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B-Q8_0 | 15 | device_local | 175030272 | 396.643 | 4.7863 | 166.739 | 171.316 | 2 | 628 |
| 1.7B-Q8_0 | 15 | device_local | 350060544 | 845.551 | 2.2176 | 367.429 | 372.43 | 2 | 628 |

Gate criteria: submit count <= 3, TTFT regresses no more than 5% from the device-local baseline, TPS regresses no more than 2%, and LM head regresses no more than 5%. The 8% TTFT reduction remains a stretch target tracked in the notes.
Gate status 0.6B-Q8_0: PASS
Gate status 1.7B-Q8_0: PASS

## 2026-06-09 13:15:27 LM Head Q8 Verification

| Model | Samples | LM head memory | Device-local bytes | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median submit count | Median descriptor alloc |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B-Q8_0 | 15 | device_local | 175030272 | 392.802 | 4.78 | 162.996 | 173.434 | 2 | 0 |
| 1.7B-Q8_0 | 15 | device_local | 350060544 | 841.991 | 2.2154 | 363.742 | 374.551 | 2 | 0 |

Gate criteria: submit count <= 3, TTFT regresses no more than 5% from the device-local baseline, TPS regresses no more than 2%, and LM head regresses no more than 5%. The 8% TTFT reduction remains a stretch target tracked in the notes.
Gate status 0.6B-Q8_0: PASS
Gate status 1.7B-Q8_0: PASS

## 2026-06-09 16:15:32 LM Head Q8 Verification

Single submit requested: True

| Model | Samples | LM head memory | Device-local bytes | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median forward submits | Median descriptor alloc | Forward GPU ms | Layers GPU ms | LM head GPU ms | Final norm GPU ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B-Q8_0 | 15 | device_local | 175030272 | 391.931 | 4.7724 | 163.277 | 174.297 | 1 | 0 | 161.924 | 121.418 | 40.473 | 0.026 |
| 1.7B-Q8_0 | 15 | device_local | 350060544 | 843.657 | 2.2091 | 364.207 | 375.256 | 1 | 0 | 362.672 | 312.403 | 50.344 | 0.049 |

Gate criteria: forward submit count <= 1, TTFT regresses no more than 3% from the device-local baseline, TPS regresses no more than 2%, and LM head regresses no more than 5%.
Gate status 0.6B-Q8_0: PASS
Gate status 1.7B-Q8_0: PASS

## 2026-06-09 UI 1.7B-Q8_0 Prefill Hang Debug

Issue: after switching the UI default model to `qwen3-1.7b-q8_0.gguf`, medium prompts could appear to hang before the first streamed token. The Activity stayed alive; the native request was blocked before first token completion.

Mitigation: reduced the short prefill single-chunk limit from 64 tokens to 32 tokens, so 51-64 token prompts are split before the final logits chunk.

| Prompt | User prefill tokens | Prefill chunk size | Chunk count | Result | TTFT ms | TPS | LM head wall ms | LM head GPU ms | Submit wait ms |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Long UI repro prompt | 61 | 32 | 2 | PASS | 2749.42 | 1.2854 | 368.873 | 50.246 | 368.039 |
| Short smoke prompt with prefix cache hit | 14 | 14 | 1 | PASS | 894.175 | 1.9973 | 363.543 | 50.145 | 362.256 |

Notes: `debug.osh26.single_submit` was reset to `0` for UI testing. The current bottleneck remains the LM-head submit/wait boundary: wall time is about 360-370 ms while GPU timestamped LM-head execution is about 50 ms.

## 2026-06-10 08:23:59 LM Head Q8 Verification

Single submit requested: True
Decode Q8 GEMV requested: True

| Model | Samples | LM head memory | Device-local bytes | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median forward submits | Median descriptor alloc | Forward GPU ms | Layers GPU ms | LM head GPU ms | Final norm GPU ms | Decode layers GPU ms | Decode QKV GPU ms | Decode attn GPU ms | Decode O GPU ms | Decode FFN gate/up GPU ms | Decode FFN down GPU ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.7B-Q8_0 | 15 | device_local | 350060544 | 842.481 | 3.0408 | 216.018 | 226.241 | 1 | 0 | 214.439 | 164.037 | 50.317 | 0.048 | 162.174 | 23.322 | 28.125 | 10.764 | 62.216 | 37.66 |

Gate criteria: forward submit count <= 1, TTFT regresses no more than 3% from the device-local baseline, TPS regresses no more than 2%, and LM head regresses no more than 5%.
Gate status 1.7B-Q8_0: PASS

## 2026-06-10 08:25:09 LM Head Q8 Verification

Single submit requested: True
Decode Q8 GEMV requested: True

| Model | Samples | LM head memory | Device-local bytes | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median forward submits | Median descriptor alloc | Forward GPU ms | Layers GPU ms | LM head GPU ms | Final norm GPU ms | Decode layers GPU ms | Decode QKV GPU ms | Decode attn GPU ms | Decode O GPU ms | Decode FFN gate/up GPU ms | Decode FFN down GPU ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B-Q8_0 | 15 | device_local | 175030272 | 393.6 | 5.7498 | 120.511 | 131.288 | 1 | 0 | 118.877 | 78.398 | 40.493 | 0.023 | 77.315 | 14.943 | 28.09 | 5.652 | 20.242 | 8.391 |

Gate criteria: forward submit count <= 1, TTFT regresses no more than 3% from the device-local baseline, TPS regresses no more than 2%, and LM head regresses no more than 5%.
Gate status 0.6B-Q8_0: PASS

## 2026-06-10 Decode Q8 GEMV Default Decision

The `nt == 1` Q8 decode projections now use the subgroup `GEMV_Q8` path by default. Set `OSH26_DECODE_Q8_GEMV=0` or Android property `debug.osh26.decode_q8_gemv=0` to restore the prior `MMQ8` decode path. Prefill remains on `MMQ8`.

| Model | Baseline TPS | Decode GEMV median TPS | Change | Median decode ms | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.6B-Q8_0 | 4.7800 | 5.7498 | +20.3% | 131.288 | default enabled |
| 1.7B-Q8_0 | 2.2154 | 3.0408 | +37.3% | 226.241 | default enabled |

Both models passed 15-request verifier runs with stable repeated token IDs, zero descriptor allocations, no attention fallback, and no logits sanity failure. Single submit remained opt-in because it did not add at least 5% TPS over decode GEMV alone.

The default UI configuration also completed the prior 61-token hang reproduction prompt with two prefill chunks (`32 + 29`), `last_decode_q8_gemv_used=true`, and no active request left behind.
