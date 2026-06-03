# OSH26 Android Performance Notes

Measured on:
- Device: Redmi K40 / alioth, Snapdragon 870, Adreno 650
- OS: Android 13
- Build: `examples/android_osh26` debug APK
- Model: `qwen3-0.6b.gguf` at `/data/data/org.osh26.llama/files/models/qwen3-0.6b.gguf`
- Backend: `OSH26 GPU Runtime`
- Date: 2026-06-03

Method:
- Load model through `POST /load_model`
- Run chat completion through `POST /v1/chat/completions`
- Read the latest engine stats from `GET /health`

## Current Snapshot

### Cold first request after model load

| Metric | Value |
| --- | ---: |
| TTFT | 50079.4 ms |
| Tokens/s | 0.41211 |
| Prefill ms | 50063 ms |
| First decode ms | 3.92599 ms |
| Decoded tokens | 24 |
| Prefill tokens | 22 |
| Prefill chunk size | 22 |
| Prefill chunk count | 1 |
| Prefill forwards | 1 |
| Submit count | 95 |

Notes:
- Prefix cache was already warmed for the template prefix.
- This is the first user completion after model load, so it still includes the cold per-request overhead.

### Warm repeat of the same prompt

| Metric | Value |
| --- | ---: |
| TTFT | 408.46 ms |
| Tokens/s | 2.69988 |
| User prefill ms | 396.111 ms |
| First decode ms | 0.119063 ms |
| Decoded tokens | 24 |
| Prefill tokens | 0 |
| Prefix cache hit | true |
| Submit count | 95 |

Notes:
- This is the stable steady-state run on the same prompt.
- The request reused the cached prefix and skipped user prefill work.

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
