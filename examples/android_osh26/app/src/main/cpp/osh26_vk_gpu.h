#pragma once
#include <stdbool.h>
#include <stddef.h>
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
    bool prefill_q8_enabled;
    bool decode_q8_enabled;
    bool debug_correctness;
    float last_attention_max_abs_err;
    uint32_t attention_fallback_layers;
    uint64_t last_forward_submit_count;
    uint64_t last_prefill_submit_count;
    uint64_t last_layer_submit_count;
    uint64_t last_lm_head_submit_count;
    uint64_t last_ttft_submit_count;
    double last_forward_layers_ms;
    double last_forward_attention_ms;
    double last_forward_kv_update_ms;
    double last_forward_lm_head_ms;
    double last_prefill_qkv_ms;
    double last_prefill_qk_norm_rope_ms;
    double last_prefill_o_proj_ms;
    double last_prefill_down_ms;
    double last_prefill_cpu_post_ms;
    double last_prefill_attention_ms;
    double last_prefill_ffn_gate_up_silu_ms;
    double last_submit_wait_ms;
    int last_logits_top5[5];
    float last_logits_top5_values[5];
    double last_prefill_ms;
    double last_decode_ms;
    double last_lm_head_ms;
    double last_lm_head_gemv_ms;
    double last_lm_head_local_topk_ms;
    double last_lm_head_merge_ms;
    double last_lm_head_wait_ms;
    double last_token_tps;
    bool gpu_lm_head_enabled;
    bool last_q8_benchmark_ran;
    bool last_q8_gate_pass;
    double last_q8_weighted_f32_ms;
    double last_q8_weighted_total_ms;
    double last_q8_weighted_speedup;
    bool last_lm_head_validation_ran;
    bool last_lm_head_validation_ok;
    int last_lm_head_validation_stage;
    float last_lm_head_matched_logit_max_abs_err;
    bool last_lm_head_top1_match;
    int last_lm_head_top5_overlap;
    int last_lm_head_top20_overlap;
    float last_lm_head_cpu_top1_margin;
    double last_lm_head_validation_ms;
    int last_lm_head_ref_top5[5];
};

struct osh26_vk_candidate {
    int token;
    float logit;
};

#define OSH26_VK_PREFIX_CACHE_PAGE_TOKENS 16
#define OSH26_VK_PREFIX_CACHE_POOL_PAGES 16

typedef enum osh26_vk_forward_flags {
    OSH26_FORWARD_NEED_LOGITS = 1u << 0,
    OSH26_FORWARD_PREFILL_ONLY = 1u << 1,
    OSH26_FORWARD_DEBUG_CHECK = 1u << 2,
    OSH26_FORWARD_VALIDATE_PREFILL = 1u << 3,
    OSH26_FORWARD_VALIDATE_FIRST_DECODE = 1u << 4,
} osh26_vk_forward_flags_t;

int osh26_vk_gpu_init(void);
int osh26_vk_gpu_load_model(const char * model_path);
int osh26_vk_gpu_forward_ex(const int * tokens, int n_tokens, int pos, uint32_t flags);
static inline int osh26_vk_gpu_forward(const int * tokens, int n_tokens, int pos) {
    return osh26_vk_gpu_forward_ex(tokens, n_tokens, pos, OSH26_FORWARD_NEED_LOGITS);
}
int osh26_vk_gpu_collect_topk(struct osh26_vk_candidate * out, int max_out);
int osh26_vk_gpu_quant_benchmark(char * out_json, size_t out_size);
int osh26_vk_gpu_quant_gemm_benchmark(char * out_json, size_t out_size);
int osh26_vk_gpu_q8_gemm_benchmark(char * out_json, size_t out_size);
bool osh26_vk_gpu_ready(void);
int osh26_vk_gpu_reset_cache(void);
bool osh26_vk_gpu_prefix_cache_supported(void);
int osh26_vk_gpu_prefix_cache_store_page(int page_slot, int src_token);
int osh26_vk_gpu_prefix_cache_restore_page(int page_slot, int dst_token);
int osh26_vk_gpu_prefix_cache_clear(void);
void osh26_vk_gpu_free(void);
int osh26_vk_get_stats(struct osh26_vk_stats * out);
void osh26_vk_gpu_set_debug_correctness(bool enabled);

#ifdef __cplusplus
}
#endif
