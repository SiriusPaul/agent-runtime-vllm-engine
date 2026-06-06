# OSH26 Android Performance Notes

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
