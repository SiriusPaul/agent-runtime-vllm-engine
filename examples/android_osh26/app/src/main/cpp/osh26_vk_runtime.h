#pragma once
#include <stdbool.h>
#include <stdint.h>

// Standalone GPU runtime for Qwen3-0.6B inference on Adreno 650.
// Manages its own GPU memory pool (weights, KV cache, activations).
// Bypasses ggml's backend scheduler entirely.
//
// Architecture assumptions (Qwen3-0.6B):
//   hidden_dim=1024, intermediate_dim=3072, num_layers=28
//   num_heads=16, num_kv_heads=8, head_dim=128 (Q:2048, K:1024, V:1024)

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;
struct llama_vocab;
struct llama_context;

// Initialize Vulkan device and create shader pipelines.
// Returns 0 on success, -1 on failure.
int osh26_vk_rt_init(void);

// Load model weights from an already-loaded llama_model into GPU buffers.
// Weights are dequantized to F32 on the CPU then uploaded to GPU.
// Returns 0 on success, -1 on failure.
int osh26_vk_rt_load_model(struct llama_model * model);

// Run one forward pass (prompt processing or single-token decode).
// - tokens: array of token IDs to process
// - n_tokens: number of tokens (batch size)
// - pos: starting position in the KV cache
// - ctx: llama_context for vocab and sampling
// Returns 0 on success, -1 on failure.
int osh26_vk_rt_forward(const int * tokens, int n_tokens, int pos,
                         struct llama_context * ctx);

// Sample the next token from the output logits on GPU.
// Returns the sampled token ID, or -1 on failure.
int osh26_vk_rt_sample(struct llama_context * ctx, float temperature, float top_p, int seed);

// Get the logits buffer (on CPU, after GPU→CPU copy). Size = vocab_size.
// Returns NULL if not available.
const float * osh26_vk_rt_get_logits(void);

// Free all GPU resources.
void osh26_vk_rt_free(void);

// Check if runtime is initialized and model is loaded.
bool osh26_vk_rt_ready(void);

#ifdef __cplusplus
}
#endif
