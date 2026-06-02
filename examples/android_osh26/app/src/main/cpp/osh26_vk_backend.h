#pragma once
#include <stdbool.h>
#include <stdint.h>

// Minimal Vulkan compute backend for OSH26.
// Uses only C Vulkan API (no Vulkan-Hpp). Implements just the
// operations needed for Qwen3-0.6B inference on Adreno 650.
//
// Usage: osh26_vk_init() registers the backend with ggml.
// Then configure with n_gpu_layers > 0 to offload to GPU.

#ifdef __cplusplus
extern "C" {
#endif

// Initialize and register the minimal Vulkan backend with ggml.
// Returns 0 on success, -1 on failure.
int osh26_vk_init(void);

struct osh26_vk_stats {
    bool ready;
    bool registered;
    uint64_t graph_compute_calls;
    uint64_t mul_mat_dispatches;
    uint64_t rms_norm_dispatches;
    uint64_t buffers_allocated;
    uint64_t buffers_freed;
    uint64_t current_buffer_bytes;
    uint64_t peak_buffer_bytes;
    uint64_t expanded_tensor_uploads;
    uint64_t expanded_upload_bytes;
    uint64_t f16_uploads;
    uint64_t bf16_uploads;
    uint64_t q4_k_uploads;
    uint64_t q6_k_uploads;
};

// Copy cumulative backend counters into out. Returns 0 on success.
int osh26_vk_get_stats(struct osh26_vk_stats * out);

#ifdef __cplusplus
}
#endif
