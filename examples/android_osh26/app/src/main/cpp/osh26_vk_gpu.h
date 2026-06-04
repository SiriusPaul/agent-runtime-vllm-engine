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
    bool mnn_prefill_attention_enabled;
    bool debug_correctness;
    float last_attention_max_abs_err;
    uint32_t attention_fallback_layers;
    uint64_t last_forward_submit_count;
    double last_forward_layers_ms;
    double last_forward_attention_ms;
    double last_forward_kv_update_ms;
    double last_forward_lm_head_ms;
    double last_prefill_qkv_ms;
    double last_prefill_cpu_post_ms;
    double last_prefill_attention_ms;
    double last_prefill_ffn_ms;
    int last_logits_top5[5];
    float last_logits_top5_values[5];
    double last_prefill_ms;
    double last_decode_ms;
    double last_lm_head_ms;
    double last_token_tps;
    bool gpu_lm_head_enabled;
    float last_lm_head_max_abs_err;
    int last_lm_head_ref_top5[5];
};

struct osh26_vk_candidate {
    int token;
    float logit;
};

typedef enum osh26_vk_forward_flags {
    OSH26_FORWARD_NEED_LOGITS = 1u << 0,
    OSH26_FORWARD_PREFILL_ONLY = 1u << 1,
    OSH26_FORWARD_DEBUG_CHECK = 1u << 2,
} osh26_vk_forward_flags_t;

int osh26_vk_gpu_init(void);
int osh26_vk_gpu_load_model(const char * model_path);
int osh26_vk_gpu_forward_ex(const int * tokens, int n_tokens, int pos, uint32_t flags);
static inline int osh26_vk_gpu_forward(const int * tokens, int n_tokens, int pos) {
    return osh26_vk_gpu_forward_ex(tokens, n_tokens, pos, OSH26_FORWARD_NEED_LOGITS);
}
const float * osh26_vk_gpu_logits(void);
int osh26_vk_gpu_collect_topk(struct osh26_vk_candidate * out, int max_out);
bool osh26_vk_gpu_ready(void);
void osh26_vk_gpu_free(void);
int osh26_vk_get_stats(struct osh26_vk_stats * out);
void osh26_vk_gpu_set_debug_correctness(bool enabled);

#ifdef __cplusplus
}
#endif
