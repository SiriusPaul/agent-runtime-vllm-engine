#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

struct osh26_vk_stats {
    bool ready, registered;
    uint64_t graph_compute_calls, mul_mat_dispatches, rms_norm_dispatches;
    uint64_t buffers_allocated, buffers_freed, current_buffer_bytes, peak_buffer_bytes;
    uint64_t expanded_tensor_uploads, expanded_upload_bytes, f16_uploads, bf16_uploads, q4_k_uploads, q6_k_uploads;
    bool mnn_attention_enabled;
    float last_attention_max_abs_err;
    uint32_t attention_fallback_layers;
    int last_logits_top5[5];
    float last_logits_top5_values[5];
};

int osh26_vk_gpu_init(void);
int osh26_vk_gpu_load_model(const char * model_path);
int osh26_vk_gpu_forward(const int * tokens, int n_tokens, int pos);
const float * osh26_vk_gpu_logits(void);
bool osh26_vk_gpu_ready(void);
void osh26_vk_gpu_free(void);
int osh26_vk_get_stats(struct osh26_vk_stats * out);

#ifdef __cplusplus
}
#endif
