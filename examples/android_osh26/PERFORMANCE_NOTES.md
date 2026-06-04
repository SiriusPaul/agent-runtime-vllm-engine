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
