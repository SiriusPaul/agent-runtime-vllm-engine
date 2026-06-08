/* All-GPU inference (except attention). Individual VkBuffer per tensor.
   Adapted from MNN VulkanBuffer pattern. */
#include "osh26_vk_gpu.h"
#include "ggml.h"
#include "gguf.h"
#include <android/log.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "vk_wrapper/vulkan_wrapper.h"
#include "mulmat_tiled.spv.h"
#include "mulmat_reduce.spv.h"
#include "gemv_reduce.spv.h"
#include "gemv_fp16_packed.spv.h"
#include "gemv_q4_packed.spv.h"
#include "mulmat_fp16_packed.spv.h"
#include "mulmat_q4_packed.spv.h"
#include "act_quant_q8.spv.h"
#include "mulmat_q8_w8a8.spv.h"
#include "lm_head_topk_local.spv.h"
#include "lm_head_topk_merge.spv.h"
#include "rms_norm.spv.h"
#include "rope.spv.h"
#include "rope_neox.spv.h"
#include "softmax_gpu.spv.h"
#include "silu_mul.spv.h"
#include "add.spv.h"
#include "attn_decode_q1.spv.h"
#include "attn_kvcache.spv.h"
#include "attention_prefill_kblock_qk.spv.h"
#include "attention_prefill_kblock_softmax_online.spv.h"
#include "attention_prefill_kblock_qkv_acc.spv.h"
#include "attention_prefill_kblock_finalize.spv.h"

#define TAG "OSH26GPU"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__)

#define HDIM (g_hdim)
#define IDIM (g_idim)
#define N_LAY 28
#define N_HD  16
#define N_KVH 8
#define HD    128
#define QDIM  (N_HD*HD)
#define KVD   (N_KVH*HD)
#define VOCAB 151936
/* HDIM and IDIM are runtime variables, set by load_qwen3_model_spec() */
static int g_hdim = 0;   /* hidden_dim: 1024 for 0.6B, 2048 for 1.7B */
static int g_idim = 0;   /* intermediate_dim: 3072 for 0.6B, 6144 for 1.7B */
#define MAX_S 8192
#define MAX_FORWARD_TOKENS 128
#define HOST_KV_MAX_S 1024
#define PREFILL_ATTN_BLOCK_TOKENS 128
#define HEAD_SHARD 16384
#define HEAD_SHARDS ((VOCAB + HEAD_SHARD - 1) / HEAD_SHARD)
#define LM_HEAD_LOCAL_TOPK 32
#define LM_HEAD_GLOBAL_TOPK 32
#define LM_HEAD_TOP1_MARGIN_REQUIRED 1.0e-3f
#define SUBMIT_POOL_CAP 512
#define F32(n) ((VkDeviceSize)(n)*4)
#define PREFIX_PAGE_TOKENS OSH26_VK_PREFIX_CACHE_PAGE_TOKENS
#define PREFIX_POOL_PAGES OSH26_VK_PREFIX_CACHE_POOL_PAGES

/* Individual buffer (MNN-style: each tensor gets own VkBuffer, offset always 0) */
typedef struct { VkBuffer B; VkDeviceMemory M; float *P; VkDeviceSize size; } VkBuf;

/* Forward declarations (defined later, used by buf_alloc/buf_free) */
static VkDevice D; static VkPhysicalDeviceMemoryProperties MP;

static void buf_free(VkBuf *b){if(b->P){vkUnmapMemory(D,b->M);b->P=NULL;}if(b->M){vkFreeMemory(D,b->M,0);b->M=0;}if(b->B){vkDestroyBuffer(D,b->B,0);b->B=0;}b->size=0;}
static bool buf_alloc(VkBuf *b,VkDeviceSize sz){memset(b,0,sizeof(*b));b->size=sz;
    VkBufferCreateInfo ci={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,0,0,sz,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_SHARING_MODE_EXCLUSIVE,0,0};
    if(vkCreateBuffer(D,&ci,0,&b->B))return false;
    VkMemoryRequirements mr;vkGetBufferMemoryRequirements(D,b->B,&mr);
    uint32_t mt=UINT32_MAX;for(uint32_t i=0;i<MP.memoryTypeCount;i++)if((mr.memoryTypeBits&(1u<<i))&&(MP.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){mt=i;break;}
    if(mt==UINT32_MAX){vkDestroyBuffer(D,b->B,0);return false;}
    VkMemoryAllocateInfo mai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,0,mr.size,mt};
    if(vkAllocateMemory(D,&mai,0,&b->M)){vkDestroyBuffer(D,b->B,0);return false;}
    vkBindBufferMemory(D,b->B,b->M,0);vkMapMemory(D,b->M,0,sz,0,(void**)&b->P);return true;}
static void buf_flush(VkBuf*b){(void)b;}
static void buf_inv(VkBuf*b){(void)b;}

/* Dummy buffer for unused descriptor bindings */
static VkBuf B_Dummy;

/* Weights: one VkBuf per tensor per layer */
static VkBuf W_ra[N_LAY],W_Q[N_LAY],W_K[N_LAY],W_V[N_LAY],W_O[N_LAY];
static VkBuf W_Qn[N_LAY],W_Kn[N_LAY];
static VkBuf W_rf[N_LAY],W_Gate[N_LAY],W_Up[N_LAY],W_Down[N_LAY];
static VkBuf WQ_Q[N_LAY],WQ_K[N_LAY],WQ_V[N_LAY],WQ_O[N_LAY];
static VkBuf WQ_Gate[N_LAY],WQ_Up[N_LAY],WQ_Down[N_LAY];
static VkBuf W_Fnorm,W_HeadShard[HEAD_SHARDS]; static float *Emb;

/* Activation buffers */
static VkBuf B_KV,B_Hid,B_Hid2,B_Qb,B_Kb,B_Vb,B_Sc,B_Att,B_PrefillOAcc,B_Gat,B_Up,B_Dwn,B_Tmp,B_Q8In,B_Last,B_LogPart,B_LmShardTopk,B_LmTopk;
static VkBuf B_KCache[N_LAY],B_VCache[N_LAY],B_AttnConst,B_KVConst;
static VkBuf B_KPrefixPool[N_LAY],B_VPrefixPool[N_LAY];
static VkBuf B_KVUpdateConst[N_LAY],B_AttnRunConst[N_LAY];

static VkInstance V;static VkQueue Q;static VkCommandPool CP;static VkDescriptorPool DP,DP_Lm;
static VkDescriptorSet DS_LmLocal,DS_LmMerge;
static VkDescriptorSetLayout DSL;static VkPipelineLayout PL;
static VkDescriptorSetLayout DSL_Dec;static VkPipelineLayout PL_Dec;
static VkDescriptorSetLayout DSL_KV;static VkPipelineLayout PL_KV;
static VkDescriptorSetLayout DSL_AttnQK;static VkPipelineLayout PL_AttnQK;
static VkDescriptorSetLayout DSL_AttnSoftmax;static VkPipelineLayout PL_AttnSoftmax;
static VkDescriptorSetLayout DSL_AttnQKVAcc;static VkPipelineLayout PL_AttnQKVAcc;
static VkDescriptorSetLayout DSL_AttnFinalize;static VkPipelineLayout PL_AttnFinalize;
static VkPipeline P_MMt,P_MMr,P_GMV,P_GMVFp16,P_GMVQ4,P_MMFp16,P_MMQ4,P_ActQ8,P_MMQ8,P_RMS,P_RoPE,P_RoPENeox,P_SMax,P_SiLU,P_Add,P_AttnDec,P_KVUpdate,P_AttnPrefillQK,P_AttnPrefillSoftmax,P_AttnPrefillQKVAcc,P_AttnPrefillFinalize,P_LmTopKLocal,P_LmTopKMerge;
static uint32_t QFI;
static pthread_mutex_t Mtx=PTHREAD_MUTEX_INITIALIZER;
static bool vk_ok,mdl_ok;
static bool g_mnn_attention_enabled;
static bool g_mnn_prefill_attention_enabled;
static bool g_debug_correctness;
static float g_last_attention_max_abs_err;
static uint32_t g_attention_fallback_layers;
static int g_last_logits_top5[5];
static float g_last_logits_top5_values[5];
static int g_last_logits_topk_ids[LM_HEAD_GLOBAL_TOPK];
static float g_last_logits_topk_values[LM_HEAD_GLOBAL_TOPK];
static int g_last_logits_topk_count;
static uint64_t g_last_layer_submit_count;
static uint64_t g_last_lm_head_submit_count;
static uint64_t g_last_ttft_submit_count;
static double g_last_submit_wait_ms;
static double g_last_prefill_ms;
static double g_last_decode_ms;
static double g_last_lm_head_ms;
static double g_last_lm_head_gemv_ms;
static double g_last_lm_head_local_topk_ms;
static double g_last_lm_head_merge_ms;
static double g_last_lm_head_wait_ms;
static double g_last_token_tps;
static uint64_t g_last_forward_submit_count;
static uint64_t g_last_prefill_submit_count;
static double g_last_forward_layers_ms;
static double g_last_forward_attention_ms;
static double g_last_forward_kv_update_ms;
static double g_last_forward_lm_head_ms;
static double g_last_prefill_qkv_ms;
static double g_last_prefill_qk_norm_rope_ms;
static double g_last_prefill_o_proj_ms;
static double g_last_prefill_down_ms;
static double g_last_prefill_cpu_post_ms;
static double g_last_prefill_attention_ms;
static double g_last_prefill_ffn_gate_up_silu_ms;
static bool g_prefill_q8_enabled;
static bool g_decode_q8_enabled;
static bool g_q8_only_mode;
static bool g_embedding_head_shared;
static uint64_t g_resident_f32_matrix_bytes;
static bool g_gpu_lm_head_enabled=true;
static bool g_last_q8_benchmark_ran;
static bool g_last_q8_gate_pass;
static double g_last_q8_weighted_f32_ms;
static double g_last_q8_weighted_total_ms;
static double g_last_q8_weighted_speedup;
static bool g_last_lm_head_validation_ran;
static bool g_last_lm_head_validation_ok;
static int g_last_lm_head_validation_stage;
static float g_last_lm_head_matched_logit_max_abs_err;
static bool g_last_lm_head_top1_match;
static int g_last_lm_head_top5_overlap;
static int g_last_lm_head_top20_overlap;
static float g_last_lm_head_cpu_top1_margin;
static double g_last_lm_head_validation_ms;
static int g_last_lm_head_ref_top5[5];
static bool g_current_forward_is_prefill;
static uint32_t g_submit_cursor;
static VkCommandBuffer g_submit_cbs[SUBMIT_POOL_CAP];
static enum { SUBMIT_PHASE_NONE = 0, SUBMIT_PHASE_LAYER = 1, SUBMIT_PHASE_LM_HEAD = 2 } g_current_submit_phase;

typedef struct {
    int32_t s0[4];
    int32_t s1[4];
    int32_t s2[4];
    float f0[4];
} AttnConst;

static double now_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec*1000.0+(double)ts.tv_nsec/1000000.0;
}

static void q8_pack_weight_cpu(const float *w,uint32_t *wq,uint32_t n,uint32_t k);

static void topk_insert(int * ids, float * values, int * count, int limit, int token, float logit) {
    if (!isfinite(logit) || limit <= 0 || token < 0 || token >= VOCAB) {
        return;
    }
    for (int i = 0; i < *count; ++i) {
        if (ids[i] == token) {
            return;
        }
    }
    if (*count < limit) {
        int i = *count;
        while (i > 0 && logit > values[i - 1]) {
            ids[i] = ids[i - 1];
            values[i] = values[i - 1];
            --i;
        }
        ids[i] = token;
        values[i] = logit;
        *count += 1;
        return;
    }
    if (logit <= values[limit - 1]) {
        return;
    }
    int i = limit - 1;
    while (i > 0 && logit > values[i - 1]) {
        ids[i] = ids[i - 1];
        values[i] = values[i - 1];
        --i;
    }
    ids[i] = token;
    values[i] = logit;
}

static int topk_overlap_count(const int *a, int a_count, const int *b, int b_count) {
    int overlap = 0;
    for (int i = 0; i < a_count; ++i) {
        for (int j = 0; j < b_count; ++j) {
            if (a[i] == b[j]) {
                overlap++;
                break;
            }
        }
    }
    return overlap;
}

static bool topk_candidates_valid(const int *ids, const float *values, int count) {
    if (count != LM_HEAD_GLOBAL_TOPK) {
        return false;
    }
    for (int i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= VOCAB || !isfinite(values[i])) {
            return false;
        }
        if (i > 0 && values[i] > values[i - 1]) {
            return false;
        }
        for (int j = 0; j < i; ++j) {
            if (ids[i] == ids[j]) {
                return false;
            }
        }
    }
    return true;
}

static bool gguf_get_u32_checked(struct gguf_context *g, const char *key, uint32_t *out) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) {
        LOGE("GGUF key missing or not uint32: %s", key);
        return false;
    }
    *out = gguf_get_val_u32(g, id);
    return true;
}

static bool gguf_check_u32(struct gguf_context *g, const char *key, uint32_t expected) {
    uint32_t actual = 0;
    if (!gguf_get_u32_checked(g, key, &actual)) {
        return false;
    }
    if (actual != expected) {
        LOGE("GGUF key mismatch: %s expected=%u actual=%u", key, expected, actual);
        return false;
    }
    return true;
}

static bool gguf_check_tensor_elements(struct gguf_context *g, const char *name, int64_t expected) {
    int64_t id = gguf_find_tensor(g, name);
    if (id < 0) {
        LOGE("GGUF tensor missing: %s", name);
        return false;
    }
    enum ggml_type type = gguf_get_tensor_type(g, id);
    size_t tsize = gguf_get_tensor_size(g, id);
    int64_t actual = 0;
    if (type == GGML_TYPE_F32) {
        actual = (int64_t)(tsize / sizeof(float));
    } else if (type == GGML_TYPE_F16) {
        actual = (int64_t)(tsize / sizeof(ggml_fp16_t));
    } else {
        actual = (int64_t)(tsize / ggml_type_size(type)) * (int64_t)ggml_blck_size(type);
    }
    if (actual != expected) {
        LOGE("GGUF tensor size mismatch: %s expected_elements=%lld actual_elements=%lld",
             name, (long long)expected, (long long)actual);
        return false;
    }
    return true;
}

static bool load_qwen3_model_spec(struct gguf_context *g, int *hdim_out, int *idim_out) {
    int64_t arch_id = gguf_find_key(g, "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(g, arch_id) != GGUF_TYPE_STRING) {
        LOGE("GGUF key missing or not string: general.architecture");
        return false;
    }
    const char *arch = gguf_get_val_str(g, arch_id);
    if (arch == NULL || strcmp(arch, "qwen3") != 0) {
        LOGE("GGUF architecture mismatch: expected=qwen3 actual=%s", arch ? arch : "(null)");
        return false;
    }

    uint32_t block_count = 0, head_count = 0, head_count_kv = 0;
    uint32_t key_length = 0, value_length = 0;
    uint32_t embedding_length = 0, feed_forward_length = 0;
    if (!gguf_get_u32_checked(g, "qwen3.block_count", &block_count) ||
        block_count != N_LAY) {
        LOGE("GGUF block_count must be %d, got %u", N_LAY, block_count);
        return false;
    }
    if (!gguf_get_u32_checked(g, "qwen3.attention.head_count", &head_count) ||
        head_count != N_HD) {
        LOGE("GGUF head_count must be %d, got %u", N_HD, head_count);
        return false;
    }
    if (!gguf_get_u32_checked(g, "qwen3.attention.head_count_kv", &head_count_kv) ||
        head_count_kv != N_KVH) {
        LOGE("GGUF head_count_kv must be %d, got %u", N_KVH, head_count_kv);
        return false;
    }
    if (!gguf_get_u32_checked(g, "qwen3.attention.key_length", &key_length) ||
        key_length != HD) {
        LOGE("GGUF key_length must be %d, got %u", HD, key_length);
        return false;
    }
    if (!gguf_get_u32_checked(g, "qwen3.attention.value_length", &value_length) ||
        value_length != HD) {
        LOGE("GGUF value_length must be %d, got %u", HD, value_length);
        return false;
    }
    if (!gguf_get_u32_checked(g, "qwen3.embedding_length", &embedding_length)) {
        return false;
    }
    if (!gguf_get_u32_checked(g, "qwen3.feed_forward_length", &feed_forward_length)) {
        return false;
    }

    /* Validate supported profiles */
    bool profile_ok = false;
    if (embedding_length == 1024 && feed_forward_length == 3072) {
        profile_ok = true;  /* 0.6B */
    } else if (embedding_length == 2048 && feed_forward_length == 6144) {
        profile_ok = true;  /* 1.7B */
    }
    if (!profile_ok) {
        LOGE("Unsupported Qwen3 profile: hidden=%u FFN=%u. Supported: 0.6B (1024/3072), 1.7B (2048/6144)",
             embedding_length, feed_forward_length);
        return false;
    }

    if (!gguf_check_tensor_elements(g, "token_embd.weight", (int64_t)VOCAB * embedding_length)) {
        return false;
    }

    *hdim_out = (int)embedding_length;
    *idim_out = (int)feed_forward_length;
    LOGI("Model spec: hidden_dim=%d intermediate_dim=%d (profile=%s)",
         *hdim_out, *idim_out,
         embedding_length == 1024 ? "0.6B" : "1.7B");
    return true;
}

static void compute_lm_head_cpu_topk(const VkBuf *head_src, int *ids, float *values, int *count, int limit) {
    *count = 0;
    const float *head_in = head_src->P;
    for (int s = 0; s < HEAD_SHARDS; ++s) {
        int base = s * HEAD_SHARD;
        int nv = VOCAB - base;
        if (nv > HEAD_SHARD) {
            nv = HEAD_SHARD;
        }
        for (int v = 0; v < nv; ++v) {
            const float *w = W_HeadShard[s].P + (VkDeviceSize)v * HDIM;
            float dot = 0.0f;
            for (int k = 0; k < HDIM; ++k) {
                dot += w[k] * head_in[k];
            }
            topk_insert(ids, values, count, limit, base + v, dot);
        }
    }
}

static void rope_neox_cpu(float * x, int nt, int nh, int pos, int hd) {
    const int half = hd / 2;
    for (int t = 0; t < nt; ++t) {
        const int p = pos + t;
        for (int h = 0; h < nh; ++h) {
            float * row = x + (t * nh + h) * hd;
            for (int i = 0; i < half; ++i) {
                const float theta = 1.0f / powf(1e6f, (2.0f * (float) i) / (float) hd);
                const float c = cosf((float) p * theta);
                const float s = sinf((float) p * theta);
                const float x0 = row[i];
                const float x1 = row[i + half];
                row[i]        = x0 * c - x1 * s;
                row[i + half] = x0 * s + x1 * c;
            }
        }
    }
}

static VkCommandBuffer CB(void){
    if (g_submit_cursor >= SUBMIT_POOL_CAP) {
        LOGE("submit pool exhausted (%u)", g_submit_cursor);
        return VK_NULL_HANDLE;
    }
    VkCommandBuffer cb = g_submit_cbs[g_submit_cursor];
    if (cb == VK_NULL_HANDLE) {
        LOGE("submit buffer slot %u not initialized", g_submit_cursor);
        return VK_NULL_HANDLE;
    }
    VkCommandBufferBeginInfo b={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,0,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,0};
    VkResult r = vkBeginCommandBuffer(cb, &b);
    if (r != VK_SUCCESS) {
        LOGE("vkBeginCommandBuffer failed: %d", (int)r);
        return VK_NULL_HANDLE;
    }
    return cb;
}
static void SubmitRecord(VkCommandBuffer cb, bool wait_now){
    if (cb == VK_NULL_HANDLE) {
        return;
    }
    const uint32_t slot = g_submit_cursor++;
    if (slot >= SUBMIT_POOL_CAP) {
        LOGE("submit slot overflow %u", slot);
        return;
    }
    vkEndCommandBuffer(cb);
    VkSubmitInfo s={VK_STRUCTURE_TYPE_SUBMIT_INFO,0,0,0,0,1,&cb,0,0};
    VkResult qr = vkQueueSubmit(Q,1,&s,VK_NULL_HANDLE);
    if (qr != VK_SUCCESS) {
        LOGE("vkQueueSubmit failed: %d", (int)qr);
    }
    g_last_forward_submit_count++;
    if (g_current_submit_phase == SUBMIT_PHASE_LAYER) {
        g_last_layer_submit_count++;
        if (g_current_forward_is_prefill) {
            g_last_prefill_submit_count++;
        }
    } else if (g_current_submit_phase == SUBMIT_PHASE_LM_HEAD) {
        g_last_lm_head_submit_count++;
    }
    if (wait_now) {
        const double wait_start_ms = now_ms();
        VkResult wr = vkQueueWaitIdle(Q);
        g_last_submit_wait_ms += now_ms() - wait_start_ms;
        if (wr != VK_SUCCESS) {
            LOGE("vkQueueWaitIdle failed: %d", (int)wr);
        }
    }
}
static void Sub(VkCommandBuffer cb){ SubmitRecord(cb, true); }
static void SubmitNoWait(VkCommandBuffer cb){ SubmitRecord(cb, false); }
static void WaitSubmittedForHostAccess(const char *reason){
    (void)reason;
    if (g_submit_cursor > 0) {
        const double wait_start_ms = now_ms();
        VkResult wr = vkQueueWaitIdle(Q);
        g_last_submit_wait_ms += now_ms() - wait_start_ms;
        if (wr != VK_SUCCESS) {
            LOGE("vkQueueWaitIdle before host access failed: %d", (int)wr);
        }
    }
}
static void WaitAndRecycleAtEnd(void){
    if (g_submit_cursor > 0) {
        const double wait_start_ms = now_ms();
        VkResult wr = vkQueueWaitIdle(Q);
        g_last_submit_wait_ms += now_ms() - wait_start_ms;
        if (wr != VK_SUCCESS) {
            LOGE("vkQueueWaitIdle failed: %d", (int)wr);
        }
        if (CP) {
            VkResult cr = vkResetCommandPool(D, CP, 0);
            if (cr != VK_SUCCESS) {
                LOGE("vkResetCommandPool failed: %d", (int)cr);
            }
        }
        vkResetDescriptorPool(D, DP, 0);
        g_submit_cursor = 0;
    }
    g_current_submit_phase = SUBMIT_PHASE_NONE;
}

/* Bind buffers to descriptor set. NULL buffer → use dummy */
static void BIND(VkCommandBuffer cb,VkBuf*b0,VkBuf*b1,VkBuf*b2,const uint32_t pc[4]){
    if(!b0)b0=&B_Dummy;if(!b1)b1=&B_Dummy;if(!b2)b2=&B_Dummy;
    VkDescriptorSet ds=0;VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP,1,&DSL};
    VkResult ar=vkAllocateDescriptorSets(D,&da,&ds);if(ar!=VK_SUCCESS){LOGE("DS alloc fail!");return;}
    VkDescriptorBufferInfo bi[3]={{b0->B,0,b0->size},{b1->B,0,b1->size},{b2->B,0,b2->size}};
    VkWriteDescriptorSet wr[3];memset(wr,0,sizeof(wr));
    for(int j=0;j<3;j++){wr[j].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;wr[j].dstSet=ds;wr[j].dstBinding=j;wr[j].descriptorCount=1;wr[j].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;wr[j].pBufferInfo=&bi[j];}
    vkUpdateDescriptorSets(D,3,wr,0,0);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL,0,1,&ds,0,0);
    vkCmdPushConstants(cb,PL,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc);
}

static bool update_lm_descriptor_set(VkDescriptorSet ds,VkBuf*b0,VkBuf*b1){
    VkBuf*b2=&B_Dummy;
    VkDescriptorBufferInfo bi[3]={{b0->B,0,b0->size},{b1->B,0,b1->size},{b2->B,0,b2->size}};
    VkWriteDescriptorSet wr[3];memset(wr,0,sizeof(wr));
    for(int j=0;j<3;j++){wr[j].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;wr[j].dstSet=ds;wr[j].dstBinding=j;wr[j].descriptorCount=1;wr[j].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;wr[j].pBufferInfo=&bi[j];}
    vkUpdateDescriptorSets(D,3,wr,0,0);
    return true;
}

static bool create_lm_descriptor_sets(void){
    if(!DP_Lm||!DSL)return false;
    vkResetDescriptorPool(D,DP_Lm,0);
    DS_LmLocal=VK_NULL_HANDLE;DS_LmMerge=VK_NULL_HANDLE;
    VkDescriptorSetLayout layouts[2]={DSL,DSL};
    VkDescriptorSet sets[2]={VK_NULL_HANDLE,VK_NULL_HANDLE};
    VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP_Lm,2,layouts};
    VkResult ar=vkAllocateDescriptorSets(D,&da,sets);
    if(ar!=VK_SUCCESS){LOGE("LM head descriptor allocation failed: %d",(int)ar);return false;}
    DS_LmLocal=sets[0];DS_LmMerge=sets[1];
    update_lm_descriptor_set(DS_LmLocal,&B_LogPart,&B_LmShardTopk);
    update_lm_descriptor_set(DS_LmMerge,&B_LmShardTopk,&B_LmTopk);
    return true;
}

static void BIND_LM(VkCommandBuffer cb,VkDescriptorSet ds,const uint32_t pc[4]){
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL,0,1,&ds,0,0);
    vkCmdPushConstants(cb,PL,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc);
}

static void BIND_DEC(VkCommandBuffer cb,VkBuf*b0,VkBuf*b1,VkBuf*b2,VkBuf*b3,VkBuf*b4,VkBuf*b5,VkBuf*b6,VkBuf*b7){
    VkBuf* b[8]={b0,b1,b2,b3,b4,b5,b6,b7};
    for(int i=0;i<8;i++)if(!b[i])b[i]=&B_Dummy;
    VkDescriptorSet ds=0;VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP,1,&DSL_Dec};
    VkResult ar=vkAllocateDescriptorSets(D,&da,&ds);if(ar!=VK_SUCCESS){LOGE("DEC DS alloc fail!");return;}
    VkDescriptorBufferInfo bi[8];
    VkWriteDescriptorSet wr[8];memset(wr,0,sizeof(wr));
    for(int j=0;j<8;j++){
        bi[j]=(VkDescriptorBufferInfo){b[j]->B,0,b[j]->size};
        wr[j].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[j].dstSet=ds;wr[j].dstBinding=(uint32_t)j;wr[j].descriptorCount=1;
        wr[j].descriptorType=(j==7)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[j].pBufferInfo=&bi[j];
    }
    vkUpdateDescriptorSets(D,8,wr,0,0);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL_Dec,0,1,&ds,0,0);
}

static void BIND_KV(VkCommandBuffer cb,VkBuf*b0,VkBuf*b1,VkBuf*b2,VkBuf*b3,VkBuf*b4){
    VkBuf* b[5]={b0,b1,b2,b3,b4};
    for(int i=0;i<5;i++)if(!b[i])b[i]=&B_Dummy;
    VkDescriptorSet ds=0;VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP,1,&DSL_KV};
    VkResult ar=vkAllocateDescriptorSets(D,&da,&ds);if(ar!=VK_SUCCESS){LOGE("KV DS alloc fail!");return;}
    VkDescriptorBufferInfo bi[5];
    VkWriteDescriptorSet wr[5];memset(wr,0,sizeof(wr));
    for(int j=0;j<5;j++){
        bi[j]=(VkDescriptorBufferInfo){b[j]->B,0,b[j]->size};
        wr[j].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[j].dstSet=ds;wr[j].dstBinding=(uint32_t)j;wr[j].descriptorCount=1;
        wr[j].descriptorType=(j==4)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[j].pBufferInfo=&bi[j];
    }
    vkUpdateDescriptorSets(D,5,wr,0,0);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL_KV,0,1,&ds,0,0);
}

static void BIND_PREFILL_QK(VkCommandBuffer cb,VkBuf*out,VkBuf*query,VkBuf*cache_key,VkBuf*mask,VkBuf*const_buf){
    VkBuf* b[5]={out,query,cache_key,mask,const_buf};
    for(int i=0;i<5;i++)if(!b[i])b[i]=&B_Dummy;
    VkDescriptorSet ds=0;VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP,1,&DSL_AttnQK};
    VkResult ar=vkAllocateDescriptorSets(D,&da,&ds);if(ar!=VK_SUCCESS){LOGE("ATTN QK DS alloc fail!");return;}
    VkDescriptorBufferInfo bi[5];
    VkWriteDescriptorSet wr[5];memset(wr,0,sizeof(wr));
    for(int j=0;j<5;j++){
        bi[j]=(VkDescriptorBufferInfo){b[j]->B,0,b[j]->size};
        wr[j].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[j].dstSet=ds;wr[j].dstBinding=(uint32_t)j;wr[j].descriptorCount=1;
        wr[j].descriptorType=(j==4)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[j].pBufferInfo=&bi[j];
    }
    vkUpdateDescriptorSets(D,5,wr,0,0);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL_AttnQK,0,1,&ds,0,0);
}

static void BIND_PREFILL_SOFTMAX(VkCommandBuffer cb,VkBuf*w,VkBuf*qk,VkBuf*m,VkBuf*l,VkBuf*alpha,VkBuf*const_buf){
    VkBuf* b[6]={w,qk,m,l,alpha,const_buf};
    for(int i=0;i<6;i++)if(!b[i])b[i]=&B_Dummy;
    VkDescriptorSet ds=0;VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP,1,&DSL_AttnSoftmax};
    VkResult ar=vkAllocateDescriptorSets(D,&da,&ds);if(ar!=VK_SUCCESS){LOGE("ATTN softmax DS alloc fail!");return;}
    VkDescriptorBufferInfo bi[6];
    VkWriteDescriptorSet wr[6];memset(wr,0,sizeof(wr));
    for(int j=0;j<6;j++){
        bi[j]=(VkDescriptorBufferInfo){b[j]->B,0,b[j]->size};
        wr[j].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[j].dstSet=ds;wr[j].dstBinding=(uint32_t)j;wr[j].descriptorCount=1;
        wr[j].descriptorType=(j==5)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[j].pBufferInfo=&bi[j];
    }
    vkUpdateDescriptorSets(D,6,wr,0,0);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL_AttnSoftmax,0,1,&ds,0,0);
}

static void BIND_PREFILL_QKV_ACC(VkCommandBuffer cb,VkBuf*out_acc,VkBuf*w,VkBuf*cache_value,VkBuf*alpha,VkBuf*const_buf){
    VkBuf* b[5]={out_acc,w,cache_value,alpha,const_buf};
    for(int i=0;i<5;i++)if(!b[i])b[i]=&B_Dummy;
    VkDescriptorSet ds=0;VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP,1,&DSL_AttnQKVAcc};
    VkResult ar=vkAllocateDescriptorSets(D,&da,&ds);if(ar!=VK_SUCCESS){LOGE("ATTN qkv acc DS alloc fail!");return;}
    VkDescriptorBufferInfo bi[5];
    VkWriteDescriptorSet wr[5];memset(wr,0,sizeof(wr));
    for(int j=0;j<5;j++){
        bi[j]=(VkDescriptorBufferInfo){b[j]->B,0,b[j]->size};
        wr[j].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[j].dstSet=ds;wr[j].dstBinding=(uint32_t)j;wr[j].descriptorCount=1;
        wr[j].descriptorType=(j==4)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[j].pBufferInfo=&bi[j];
    }
    vkUpdateDescriptorSets(D,5,wr,0,0);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL_AttnQKVAcc,0,1,&ds,0,0);
}

static void BIND_PREFILL_FINALIZE(VkCommandBuffer cb,VkBuf*out,VkBuf*out_acc,VkBuf*l,VkBuf*const_buf){
    VkBuf* b[4]={out,out_acc,l,const_buf};
    for(int i=0;i<4;i++)if(!b[i])b[i]=&B_Dummy;
    VkDescriptorSet ds=0;VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP,1,&DSL_AttnFinalize};
    VkResult ar=vkAllocateDescriptorSets(D,&da,&ds);if(ar!=VK_SUCCESS){LOGE("ATTN finalize DS alloc fail!");return;}
    VkDescriptorBufferInfo bi[4];
    VkWriteDescriptorSet wr[4];memset(wr,0,sizeof(wr));
    for(int j=0;j<4;j++){
        bi[j]=(VkDescriptorBufferInfo){b[j]->B,0,b[j]->size};
        wr[j].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[j].dstSet=ds;wr[j].dstBinding=(uint32_t)j;wr[j].descriptorCount=1;
        wr[j].descriptorType=(j==3)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[j].pBufferInfo=&bi[j];
    }
    vkUpdateDescriptorSets(D,4,wr,0,0);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL_AttnFinalize,0,1,&ds,0,0);
}

/* Dispatch macros — pipeline FIRST (MNN order: bind pipeline, then descriptors) */
#define RMS(cb,rows,cols,x,w,y) do{uint32_t p[4]={rows,cols,0,0};vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_RMS);BIND(cb,x,w,y,p);vkCmdDispatch(cb,(uint32_t)(rows),1,1);}while(0)
#define BARRIER(cb) do{VkMemoryBarrier mb_={VK_STRUCTURE_TYPE_MEMORY_BARRIER,0,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT};vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb_,0,0,0,0);}while(0)
#define CLEAR_FLOAT_BUF(cb,buf,nfloats) do{ \
    VkMemoryBarrier mb0_={VK_STRUCTURE_TYPE_MEMORY_BARRIER,0,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT}; \
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&mb0_,0,0,0,0); \
    vkCmdFillBuffer(cb,(buf)->B,0,F32((VkDeviceSize)(nfloats)),0u); \
    VkMemoryBarrier mb1_={VK_STRUCTURE_TYPE_MEMORY_BARRIER,0,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT}; \
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb1_,0,0,0,0); \
}while(0)
#define MM(cb,w,x,y,M,N,K) do{uint32_t p[4]={M,N,K,K};if((M)==1){vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_GMV);}else if((M)<=4&&(N)<=65535){vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_MMr);}else{vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_MMt);}BIND(cb,w,x,y,p);if((M)==1){vkCmdDispatch(cb,(N),1,1);}else if((M)<=4&&(N)<=65535){vkCmdDispatch(cb,(N),(M),1);}else{vkCmdDispatch(cb,((N)+7)/8,((M)+7)/8,1);}}while(0)
#define ACT_Q8(cb,x,q,M,K) do{uint32_t p[4]={M,K,K,(((K)+31u)/32u)};vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_ActQ8);BIND(cb,x,q,&B_Dummy,p);vkCmdDispatch(cb,(((K)+31u)/32u),(M),1);BARRIER(cb);}while(0)
#define MMQ8(cb,wq,q,y,M,N,K) do{uint32_t p[4]={M,N,K,K};vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_MMQ8);BIND(cb,wq,q,y,p);vkCmdDispatch(cb,((N)+7u)/8u,((M)+7u)/8u,1);}while(0)
#define ROPE(cb,x,nt,nh,posv,hd) do{uint32_t p[4]={nt,nh,(uint32_t)(posv),hd};vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_RoPE);BIND(cb,x,NULL,NULL,p);uint32_t tot_=(nt)*(nh)*((hd)/2);vkCmdDispatch(cb,(tot_+63)/64,1,1);}while(0)
#define ROPE_NEOX(cb,x,nt,nh,posv,hd) do{uint32_t p[4]={nt,nh,(uint32_t)(posv),hd};vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_RoPENeox);BIND(cb,x,NULL,NULL,p);uint32_t tot_=(nt)*(nh)*((hd)/2);vkCmdDispatch(cb,(tot_+63)/64,1,1);}while(0)
#define ADD(cb,y,x,n) do{uint32_t p[4]={n,0,0,0};vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_Add);BIND(cb,y,x,NULL,p);vkCmdDispatch(cb,((n)+63)/64,1,1);}while(0)
#define SILU(cb,a,b,z,n) do{uint32_t p[4]={n,0,0,0};vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_SiLU);BIND(cb,a,b,z,p);vkCmdDispatch(cb,((n)+63)/64,1,1);}while(0)
#define DEC_ATTN(cb,out,q,kcache,vcache,cbuf) do{vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_AttnDec);BIND_DEC(cb,out,q,NULL,NULL,kcache,vcache,NULL,cbuf);vkCmdDispatch(cb,N_HD,1,1);}while(0)
#define KV_UPDATE(cb,k,v,kcache,vcache,cbuf,ntok) do{vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_KVUpdate);BIND_KV(cb,k,v,kcache,vcache,cbuf);vkCmdDispatch(cb,(HD/4+7)/8,(ntok),N_KVH);}while(0)

static void pack_mnn_kv_cache(int l,int pos,int nt,const float*kb,const float*vb){
    const int d4s=HD/4;
    ggml_fp16_t *kc=(ggml_fp16_t*)B_KCache[l].P,*vc=(ggml_fp16_t*)B_VCache[l].P;
    for(int t=0;t<nt;t++){
        int token=pos+t;
        if(token<0||token>=MAX_S)continue;
        for(int kvh=0;kvh<N_KVH;kvh++){
            const float *krow=kb+(t*N_KVH+kvh)*HD;
            const float *vrow=vb+(t*N_KVH+kvh)*HD;
            for(int d4=0;d4<d4s;d4++){
                ggml_fp16_t *kd=kc+(((kvh*d4s+d4)*MAX_S+token)*4);
                ggml_fp16_t *vd=vc+(((kvh*MAX_S+token)*d4s+d4)*4);
                for(int lane=0;lane<4;lane++){
                    kd[lane]=ggml_fp32_to_fp16(krow[d4*4+lane]);
                    vd[lane]=ggml_fp32_to_fp16(vrow[d4*4+lane]);
                }
            }
        }
    }
    buf_flush(&B_KCache[l]);buf_flush(&B_VCache[l]);
}

static bool prefix_page_args_valid(int page_slot,int token){
    return mdl_ok
        && page_slot >= 0
        && page_slot < PREFIX_POOL_PAGES
        && token >= 0
        && token + PREFIX_PAGE_TOKENS <= MAX_S
        && B_KPrefixPool[0].P != NULL
        && B_VPrefixPool[0].P != NULL;
}

static size_t kcache_vec_offset(int kvh,int d4,int token,int stride){
    return (size_t)(((kvh*(HD/4)+d4)*stride+token)*4);
}

static size_t vcache_vec_offset(int kvh,int token,int d4,int stride){
    return (size_t)(((kvh*stride+token)*(HD/4)+d4)*4);
}

static void copy_packed_kv_page(VkBuf *dst_k,VkBuf *dst_v,int dst_token,int dst_stride,const VkBuf *src_k,const VkBuf *src_v,int src_token,int src_stride){
    const int d4s=HD/4;
    for(int kvh=0;kvh<N_KVH;kvh++){
        for(int d4=0;d4<d4s;d4++){
            ggml_fp16_t *dk=(ggml_fp16_t*)dst_k->P+kcache_vec_offset(kvh,d4,dst_token,dst_stride);
            const ggml_fp16_t *sk=(const ggml_fp16_t*)src_k->P+kcache_vec_offset(kvh,d4,src_token,src_stride);
            memcpy(dk,sk,(size_t)PREFIX_PAGE_TOKENS*4*sizeof(ggml_fp16_t));
        }
        for(int t=0;t<PREFIX_PAGE_TOKENS;t++){
            ggml_fp16_t *dv=(ggml_fp16_t*)dst_v->P+vcache_vec_offset(kvh,dst_token+t,0,dst_stride);
            const ggml_fp16_t *sv=(const ggml_fp16_t*)src_v->P+vcache_vec_offset(kvh,src_token+t,0,src_stride);
            memcpy(dv,sv,(size_t)d4s*4*sizeof(ggml_fp16_t));
        }
    }
}

static void cpu_attention_ref(int nt,int pos,int l,const float*qb,const float*kv,float*sc,float*out){
    int kvo=l*2*HOST_KV_MAX_S*KVD;
    int slen=pos+nt;
    memset(sc,0,nt*N_HD*HOST_KV_MAX_S*sizeof(float));
    float isd=1.0f/sqrtf((float)HD);
    for(int h=0;h<N_HD;h++){
        int kvh=h*N_KVH/N_HD;
        for(int t=0;t<nt;t++){
            const float*qt=qb+t*QDIM+h*HD;
            float*sr=sc+(t*N_HD+h)*HOST_KV_MAX_S;
            int lim=(nt==1)?slen:(pos+t+1);
            for(int s=0;s<lim;s++){
                const float*ks=kv+kvo+s*KVD+kvh*HD;
                float dot=0;
                for(int d=0;d<HD;d++)dot+=qt[d]*ks[d];
                sr[s]=dot*isd;
            }
        }
    }
    for(int r=0;r<nt*N_HD;r++){
        float*row=sc+r*HOST_KV_MAX_S;
        int nc=(nt==1)?slen:(pos+(r/N_HD)+1);
        float mx=row[0];
        for(int c=1;c<nc;c++)if(row[c]>mx)mx=row[c];
        float sum=0;
        for(int c=0;c<nc;c++){row[c]=expf(row[c]-mx);sum+=row[c];}
        for(int c=0;c<nc;c++)row[c]/=sum;
    }
    memset(out,0,nt*QDIM*sizeof(float));
    for(int h=0;h<N_HD;h++){
        int kvh=h*N_KVH/N_HD;
        for(int t=0;t<nt;t++){
            float*oh=out+t*QDIM+h*HD;
            const float*sr=sc+(t*N_HD+h)*HOST_KV_MAX_S;
            for(int s=0;s<slen;s++){
                const float*vs=kv+kvo+HOST_KV_MAX_S*KVD+s*KVD+kvh*HD;
                float wgt=sr[s];
                for(int d=0;d<HD;d++)oh[d]+=wgt*vs[d];
            }
        }
    }
}

static bool prefill_attention_gpu_record(VkCommandBuffer cb, int l, int nt, int pos) {
    const bool verbose_prefill = g_debug_correctness;
    if (cb == VK_NULL_HANDLE) {
        LOGE("PREFILL stage: attention command buffer is null");
        return false;
    }
    if (!g_mnn_prefill_attention_enabled) {
        if (l == 0) {
            LOGE("PREFILL stage: attention disabled (pipeline creation failed)");
        }
        return false;
    }
    const int total_len = pos + nt;
    const int block_len = PREFILL_ATTN_BLOCK_TOKENS;
    const int block_len4 = ((block_len + 3) / 4) * 4;
    const int block_len4_4 = block_len4 / 4;
    const int q4_count = (nt + 3) / 4;
    const int q2_count = (nt + 1) / 2;
    const int d4_size = HD / 4;
    AttnConst ac = {{nt, block_len, N_HD, N_KVH}, {HD, N_HD / N_KVH, pos, total_len}, {nt, total_len, 2, MAX_S}, {1.0f / sqrtf((float)HD), 0, 0, 0}};
    if (l == 0 && verbose_prefill) {
        LOGI("PREFILL stage: layer0 attention begin");
        LOGI("PREFILL stage: ptrs att=%p oacc=%p up=%p dwn=%p tmp=%p attconst=%p",
             (void *)B_Att.P, (void *)B_PrefillOAcc.P, (void *)B_Up.P, (void *)B_Dwn.P, (void *)B_Tmp.P, (void *)B_AttnRunConst[l].P);
    }
    memcpy(B_AttnRunConst[l].P, &ac, sizeof(ac));
    if (l == 0 && verbose_prefill) {
        LOGI("PREFILL stage: layer0 attention const copied");
    }
    buf_flush(&B_AttnRunConst[l]);
    if (l == 0 && verbose_prefill) {
        LOGI("PREFILL stage: layer0 attention const flushed");
    }
    if (l == 0 && verbose_prefill) {
        LOGI("PREFILL stage: layer0 zero B_PrefillOAcc");
    }
    CLEAR_FLOAT_BUF(cb, &B_PrefillOAcc, (VkDeviceSize)nt * QDIM);
    if (l == 0 && verbose_prefill) {
        LOGI("PREFILL stage: layer0 zero B_Up");
    }
    CLEAR_FLOAT_BUF(cb, &B_Up, (VkDeviceSize)nt * N_HD);
    if (l == 0 && verbose_prefill) {
        LOGI("PREFILL stage: layer0 zero B_Dwn");
    }
    CLEAR_FLOAT_BUF(cb, &B_Dwn, (VkDeviceSize)nt * N_HD);
    if (l == 0 && verbose_prefill) {
        LOGI("PREFILL stage: layer0 zero B_Tmp");
    }
    CLEAR_FLOAT_BUF(cb, &B_Tmp, (VkDeviceSize)nt * N_HD);

    const double start_ms = now_ms();
    if (l == 0 && verbose_prefill) {
        LOGI("PREFILL stage: layer0 attention CB begin");
    }
    for (int k_start = 0; k_start < total_len; k_start += block_len) {
        const int cur_block = (total_len - k_start < block_len) ? (total_len - k_start) : block_len;
        const uint32_t qk_pc[4] = {(uint32_t)k_start, (uint32_t)cur_block, 0, 0};
        if (l == 0 && k_start == 0 && verbose_prefill) {
            LOGI("PREFILL stage: layer0 qk bind begin");
        }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P_AttnPrefillQK);
        BIND_PREFILL_QK(cb, &B_Sc, &B_Qb, &B_KCache[l], &B_Dummy, &B_AttnRunConst[l]);
        if (l == 0 && k_start == 0 && verbose_prefill) {
            LOGI("PREFILL stage: layer0 qk bind done");
        }
        vkCmdPushConstants(cb, PL_AttnQK, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, qk_pc);
        if (l == 0 && k_start == 0 && verbose_prefill) {
            LOGI("PREFILL stage: layer0 qk push done");
        }
        vkCmdDispatch(cb, (uint32_t)((block_len4_4 + 7) / 8), (uint32_t)((q4_count + 7) / 8), N_HD);
        if (l == 0 && k_start == 0 && verbose_prefill) {
            LOGI("PREFILL stage: layer0 qk dispatch done");
        }
        BARRIER(cb);
        if (l == 0 && k_start == 0 && verbose_prefill) {
            LOGI("PREFILL stage: layer0 qk barrier done");
        }

        const uint32_t softmax_pc[4] = {(uint32_t)cur_block, 0, 0, 0};
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P_AttnPrefillSoftmax);
        BIND_PREFILL_SOFTMAX(cb, &B_Gat, &B_Sc, &B_Up, &B_Dwn, &B_Tmp, &B_AttnRunConst[l]);
        vkCmdPushConstants(cb, PL_AttnSoftmax, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, softmax_pc);
        vkCmdDispatch(cb, N_HD, (uint32_t)nt, 1);
        BARRIER(cb);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P_AttnPrefillQKVAcc);
        BIND_PREFILL_QKV_ACC(cb, &B_PrefillOAcc, &B_Gat, &B_VCache[l], &B_Tmp, &B_AttnRunConst[l]);
        vkCmdPushConstants(cb, PL_AttnQKVAcc, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, qk_pc);
        vkCmdDispatch(cb, (uint32_t)((d4_size + 7) / 8), (uint32_t)((q2_count + 7) / 8), N_HD);
        BARRIER(cb);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P_AttnPrefillFinalize);
    BIND_PREFILL_FINALIZE(cb, &B_Att, &B_PrefillOAcc, &B_Dwn, &B_AttnRunConst[l]);
    vkCmdDispatch(cb, (uint32_t)((d4_size + 7) / 8), (uint32_t)((q2_count + 7) / 8), N_HD);
    BARRIER(cb);
    if (l == 0 && verbose_prefill) {
        LOGI("PREFILL stage: layer0 attention CB recorded");
    }
    const double elapsed_ms = now_ms() - start_ms;
    g_last_prefill_attention_ms += elapsed_ms;
    g_last_forward_attention_ms += elapsed_ms;
    return true;
}

static bool prefill_attention_gpu(int l, int nt, int pos) {
    VkCommandBuffer cb = CB();
    if (!prefill_attention_gpu_record(cb, l, nt, pos)) {
        return false;
    }
    SubmitNoWait(cb);
    return true;
}

/* ---- Init ---- */
int osh26_vk_gpu_init(void){if(vk_ok)return 0;
    if(!InitVulkan()){LOGE("InitVulkan failed");return -1;}
    VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO,0,"OSH26",1,"OSH26",1,VK_API_VERSION_1_1};
    VkInstanceCreateInfo ci={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,0,0,&ai,0,0,0,0};
    if(vkCreateInstance(&ci,0,&V))return -1;
    uint32_t nd=0;vkEnumeratePhysicalDevices(V,&nd,0);VkPhysicalDevice*pd=calloc(nd,sizeof(*pd));vkEnumeratePhysicalDevices(V,&nd,pd);VkPhysicalDevice ph=pd[0];free(pd);
    VkPhysicalDeviceProperties pdp;vkGetPhysicalDeviceProperties(ph,&pdp);vkGetPhysicalDeviceMemoryProperties(ph,&MP);LOGI("GPU: %s",pdp.deviceName);
    uint32_t qn=0;vkGetPhysicalDeviceQueueFamilyProperties(ph,&qn,0);VkQueueFamilyProperties*qp=calloc(qn,sizeof(*qp));vkGetPhysicalDeviceQueueFamilyProperties(ph,&qn,qp);
    for(uint32_t i=0;i<qn;i++)if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){QFI=i;break;}free(qp);
    float pr=1;VkDeviceQueueCreateInfo dq={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,0,0,QFI,1,&pr};
    VkDeviceCreateInfo dc={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,0,0,1,&dq,0,0,0,0};
    if(vkCreateDevice(ph,&dc,0,&D))return -1;vkGetDeviceQueue(D,QFI,0,&Q);
    VkCommandPoolCreateInfo cp={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,0,VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,QFI};vkCreateCommandPool(D,&cp,0,&CP);
    VkDescriptorPoolSize ds[2]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1048576},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,262144}};
    VkDescriptorPoolCreateInfo dp={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,0,VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,262144,2,ds};vkCreateDescriptorPool(D,&dp,0,&DP);
    VkDescriptorPoolSize lm_ds={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6};
    VkDescriptorPoolCreateInfo lm_dp={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,0,0,2,1,&lm_ds};
    if(vkCreateDescriptorPool(D,&lm_dp,0,&DP_Lm)!=VK_SUCCESS){LOGE("LM head descriptor pool creation failed");return -1;}
    VkCommandBufferAllocateInfo cba={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,0,CP,VK_COMMAND_BUFFER_LEVEL_PRIMARY,SUBMIT_POOL_CAP};
    if (vkAllocateCommandBuffers(D, &cba, g_submit_cbs) != VK_SUCCESS) {
        LOGE("preallocate command buffers failed");
        return -1;
    }
    VkDescriptorSetLayoutBinding bd[3];for(int i=0;i<3;i++)bd[i]=(VkDescriptorSetLayoutBinding){i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
    VkDescriptorSetLayoutCreateInfo dl={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,3,bd};vkCreateDescriptorSetLayout(D,&dl,0,&DSL);
    VkPushConstantRange pc={VK_SHADER_STAGE_COMPUTE_BIT,0,16};
    VkPipelineLayoutCreateInfo pl={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&DSL,1,&pc};vkCreatePipelineLayout(D,&pl,0,&PL);
    VkDescriptorSetLayoutBinding bdd[8];for(int i=0;i<8;i++)bdd[i]=(VkDescriptorSetLayoutBinding){(uint32_t)i,(i==7)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
    VkDescriptorSetLayoutCreateInfo dld={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,8,bdd};vkCreateDescriptorSetLayout(D,&dld,0,&DSL_Dec);
    VkPipelineLayoutCreateInfo pld={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&DSL_Dec,0,0};vkCreatePipelineLayout(D,&pld,0,&PL_Dec);
    VkDescriptorSetLayoutBinding bdk[5];for(int i=0;i<5;i++)bdk[i]=(VkDescriptorSetLayoutBinding){(uint32_t)i,(i==4)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
    VkDescriptorSetLayoutCreateInfo dlk={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,5,bdk};vkCreateDescriptorSetLayout(D,&dlk,0,&DSL_KV);
    VkPipelineLayoutCreateInfo plk={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&DSL_KV,0,0};vkCreatePipelineLayout(D,&plk,0,&PL_KV);
    VkDescriptorSetLayoutBinding bdqk[5];for(int i=0;i<5;i++)bdqk[i]=(VkDescriptorSetLayoutBinding){(uint32_t)i,(i==4)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
    VkDescriptorSetLayoutCreateInfo dlqk={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,5,bdqk};VkResult dlr1=vkCreateDescriptorSetLayout(D,&dlqk,0,&DSL_AttnQK);
    VkDescriptorSetLayoutBinding bdsm[6];for(int i=0;i<6;i++)bdsm[i]=(VkDescriptorSetLayoutBinding){(uint32_t)i,(i==5)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
    VkDescriptorSetLayoutCreateInfo dlsm={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,6,bdsm};VkResult dlr2=vkCreateDescriptorSetLayout(D,&dlsm,0,&DSL_AttnSoftmax);
    VkDescriptorSetLayoutBinding bdqa[5];for(int i=0;i<5;i++)bdqa[i]=(VkDescriptorSetLayoutBinding){(uint32_t)i,(i==4)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
    VkDescriptorSetLayoutCreateInfo dlqa={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,5,bdqa};VkResult dlr3=vkCreateDescriptorSetLayout(D,&dlqa,0,&DSL_AttnQKVAcc);
    VkDescriptorSetLayoutBinding bdfn[4];for(int i=0;i<4;i++)bdfn[i]=(VkDescriptorSetLayoutBinding){(uint32_t)i,(i==3)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
    VkDescriptorSetLayoutCreateInfo dlfn={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,4,bdfn};VkResult dlr4=vkCreateDescriptorSetLayout(D,&dlfn,0,&DSL_AttnFinalize);
    VkPushConstantRange pc8={VK_SHADER_STAGE_COMPUTE_BIT,0,8};
    VkPipelineLayoutCreateInfo plqk={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&DSL_AttnQK,1,&pc8};VkResult plr1=vkCreatePipelineLayout(D,&plqk,0,&PL_AttnQK);
    VkPipelineLayoutCreateInfo plsm={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&DSL_AttnSoftmax,1,&pc8};VkResult plr2=vkCreatePipelineLayout(D,&plsm,0,&PL_AttnSoftmax);
    VkPipelineLayoutCreateInfo plqa={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&DSL_AttnQKVAcc,1,&pc8};VkResult plr3=vkCreatePipelineLayout(D,&plqa,0,&PL_AttnQKVAcc);
    VkPipelineLayoutCreateInfo plfn={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&DSL_AttnFinalize,0,0};VkResult plr4=vkCreatePipelineLayout(D,&plfn,0,&PL_AttnFinalize);
    LOGI("prefill layouts: dsl=%d/%d/%d/%d pl=%d/%d/%d/%d", (int)dlr1, (int)dlr2, (int)dlr3, (int)dlr4, (int)plr1, (int)plr2, (int)plr3, (int)plr4);
    { VkShaderModuleCreateInfo s={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,0,0,0,0};VkShaderModule m;
      s.codeSize=examples_android_osh26_app_src_main_cpp_mulmat_tiled_spv_len;s.pCode=(const uint32_t*)examples_android_osh26_app_src_main_cpp_mulmat_tiled_spv;vkCreateShaderModule(D,&s,0,&m);VkComputePipelineCreateInfo pi={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,0,0,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,m,"main",0},PL,0,(uint32_t)-1};vkCreateComputePipelines(D,0,1,&pi,0,&P_MMt);
      s.codeSize=examples_android_osh26_app_src_main_cpp_mulmat_reduce_spv_len;s.pCode=(const uint32_t*)examples_android_osh26_app_src_main_cpp_mulmat_reduce_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_MMr);
      s.codeSize=_tmp_gemv_reduce_spv_len;s.pCode=(const uint32_t*)_tmp_gemv_reduce_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_GMV);
      s.codeSize=_tmp_gemv_fp16_packed_spv_len;s.pCode=(const uint32_t*)_tmp_gemv_fp16_packed_spv;VkResult fp16_sm=vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;VkResult fp16_p=fp16_sm==VK_SUCCESS?vkCreateComputePipelines(D,0,1,&pi,0,&P_GMVFp16):fp16_sm;
      s.codeSize=_tmp_gemv_q4_packed_spv_len;s.pCode=(const uint32_t*)_tmp_gemv_q4_packed_spv;VkResult q4_sm=vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;VkResult q4_p=q4_sm==VK_SUCCESS?vkCreateComputePipelines(D,0,1,&pi,0,&P_GMVQ4):q4_sm;
      if(fp16_p!=VK_SUCCESS||q4_p!=VK_SUCCESS){LOGE("quant benchmark pipeline creation failed: fp16=%d q4=%d",(int)fp16_p,(int)q4_p);return -1;}
      s.codeSize=_tmp_mulmat_fp16_packed_spv_len;s.pCode=(const uint32_t*)_tmp_mulmat_fp16_packed_spv;VkResult mm_fp16_sm=vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;VkResult mm_fp16_p=mm_fp16_sm==VK_SUCCESS?vkCreateComputePipelines(D,0,1,&pi,0,&P_MMFp16):mm_fp16_sm;
      s.codeSize=_tmp_mulmat_q4_packed_spv_len;s.pCode=(const uint32_t*)_tmp_mulmat_q4_packed_spv;VkResult mm_q4_sm=vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;VkResult mm_q4_p=mm_q4_sm==VK_SUCCESS?vkCreateComputePipelines(D,0,1,&pi,0,&P_MMQ4):mm_q4_sm;
      if(mm_fp16_p!=VK_SUCCESS||mm_q4_p!=VK_SUCCESS){LOGE("quant GEMM pipeline creation failed: fp16=%d q4=%d",(int)mm_fp16_p,(int)mm_q4_p);return -1;}
      s.codeSize=_tmp_act_quant_q8_spv_len;s.pCode=(const uint32_t*)_tmp_act_quant_q8_spv;VkResult act_q8_sm=vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;VkResult act_q8_p=act_q8_sm==VK_SUCCESS?vkCreateComputePipelines(D,0,1,&pi,0,&P_ActQ8):act_q8_sm;
      s.codeSize=_tmp_mulmat_q8_w8a8_spv_len;s.pCode=(const uint32_t*)_tmp_mulmat_q8_w8a8_spv;VkResult mm_q8_sm=vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;VkResult mm_q8_p=mm_q8_sm==VK_SUCCESS?vkCreateComputePipelines(D,0,1,&pi,0,&P_MMQ8):mm_q8_sm;
      if(act_q8_p!=VK_SUCCESS||mm_q8_p!=VK_SUCCESS){LOGE("Q8 W8A8 benchmark pipeline creation failed: act=%d gemm=%d",(int)act_q8_p,(int)mm_q8_p);return -1;}
      s.codeSize=_tmp_rms_norm_spv_len;s.pCode=(const uint32_t*)_tmp_rms_norm_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_RMS);
      s.codeSize=_tmp_rope_spv_len;s.pCode=(const uint32_t*)_tmp_rope_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_RoPE);
      s.codeSize=_tmp_rope_neox_spv_len;s.pCode=(const uint32_t*)_tmp_rope_neox_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_RoPENeox);
      s.codeSize=_tmp_softmax_gpu_spv_len;s.pCode=(const uint32_t*)_tmp_softmax_gpu_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_SMax);
      s.codeSize=_tmp_silu_mul_spv_len;s.pCode=(const uint32_t*)_tmp_silu_mul_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_SiLU);
      s.codeSize=_tmp_add_spv_len;s.pCode=(const uint32_t*)_tmp_add_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_Add);
      s.codeSize=_tmp_lm_head_topk_local_spv_len;s.pCode=(const uint32_t*)_tmp_lm_head_topk_local_spv;VkResult lm_sm1=vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;VkResult lm_p1=lm_sm1==VK_SUCCESS?vkCreateComputePipelines(D,0,1,&pi,0,&P_LmTopKLocal):lm_sm1;
      s.codeSize=_tmp_lm_head_topk_merge_spv_len;s.pCode=(const uint32_t*)_tmp_lm_head_topk_merge_spv;VkResult lm_sm2=vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;VkResult lm_p2=lm_sm2==VK_SUCCESS?vkCreateComputePipelines(D,0,1,&pi,0,&P_LmTopKMerge):lm_sm2;
      if(lm_p1!=VK_SUCCESS||lm_p2!=VK_SUCCESS){LOGE("LM head topK pipeline creation failed: local=%d merge=%d",(int)lm_p1,(int)lm_p2);return -1;}
      s.codeSize=_tmp_attn_decode_q1_spv_len;s.pCode=(const uint32_t*)_tmp_attn_decode_q1_spv;vkCreateShaderModule(D,&s,0,&m);VkComputePipelineCreateInfo pid={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,0,0,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,m,"main",0},PL_Dec,0,(uint32_t)-1};if(vkCreateComputePipelines(D,0,1,&pid,0,&P_AttnDec)==VK_SUCCESS)g_mnn_attention_enabled=true; }
    { VkShaderModuleCreateInfo s={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,0,0,0,0};VkShaderModule m;
      VkResult sm1,sm2,sm3,sm4,r1,r2,r3,r4;
      s.codeSize=_tmp_attention_prefill_kblock_qk_spv_len;s.pCode=(const uint32_t*)_tmp_attention_prefill_kblock_qk_spv;sm1=vkCreateShaderModule(D,&s,0,&m);VkComputePipelineCreateInfo piqk={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,0,0,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,m,"main",0},PL_AttnQK,0,(uint32_t)-1};r1=vkCreateComputePipelines(D,0,1,&piqk,0,&P_AttnPrefillQK);
      s.codeSize=_tmp_attention_prefill_kblock_softmax_online_spv_len;s.pCode=(const uint32_t*)_tmp_attention_prefill_kblock_softmax_online_spv;sm2=vkCreateShaderModule(D,&s,0,&m);VkComputePipelineCreateInfo pism={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,0,0,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,m,"main",0},PL_AttnSoftmax,0,(uint32_t)-1};r2=vkCreateComputePipelines(D,0,1,&pism,0,&P_AttnPrefillSoftmax);
      s.codeSize=_tmp_attention_prefill_kblock_qkv_acc_spv_len;s.pCode=(const uint32_t*)_tmp_attention_prefill_kblock_qkv_acc_spv;sm3=vkCreateShaderModule(D,&s,0,&m);VkComputePipelineCreateInfo piqa={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,0,0,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,m,"main",0},PL_AttnQKVAcc,0,(uint32_t)-1};r3=vkCreateComputePipelines(D,0,1,&piqa,0,&P_AttnPrefillQKVAcc);
      s.codeSize=_tmp_attention_prefill_kblock_finalize_spv_len;s.pCode=(const uint32_t*)_tmp_attention_prefill_kblock_finalize_spv;sm4=vkCreateShaderModule(D,&s,0,&m);VkComputePipelineCreateInfo pifn={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,0,0,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,m,"main",0},PL_AttnFinalize,0,(uint32_t)-1};r4=vkCreateComputePipelines(D,0,1,&pifn,0,&P_AttnPrefillFinalize);
      LOGI("prefill shader modules: qk=%d softmax=%d qkvacc=%d final=%d", (int)sm1, (int)sm2, (int)sm3, (int)sm4);
      g_mnn_prefill_attention_enabled = (r1 == VK_SUCCESS && r2 == VK_SUCCESS && r3 == VK_SUCCESS && r4 == VK_SUCCESS);
      LOGI("prefill pipelines: qk=%d softmax=%d qkvacc=%d final=%d enabled=%s", (int)r1, (int)r2, (int)r3, (int)r4, g_mnn_prefill_attention_enabled ? "true" : "false"); }
    { VkShaderModuleCreateInfo s={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,0,0,_tmp_attn_kvcache_spv_len,(const uint32_t*)_tmp_attn_kvcache_spv};VkShaderModule m;vkCreateShaderModule(D,&s,0,&m);VkComputePipelineCreateInfo pik={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,0,0,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,m,"main",0},PL_KV,0,(uint32_t)-1};vkCreateComputePipelines(D,0,1,&pik,0,&P_KVUpdate); }
    buf_alloc(&B_Dummy,256);
    buf_alloc(&B_AttnConst,sizeof(AttnConst));
    buf_alloc(&B_KVConst,sizeof(AttnConst));
    vk_ok=true;LOGI("GPU ready (mnn_attention=%s)",g_mnn_attention_enabled?"true":"false");return 0;}

/* ---- Model loading ---- */
static bool read_gguf(struct gguf_context*g,FILE*f,const char*name,float*dst){
    int idx=gguf_find_tensor(g,name);if(idx<0){LOGE("Tensor not found: %s",name);return false;}
    size_t off=gguf_get_data_offset(g)+gguf_get_tensor_offset(g,idx),tsize=gguf_get_tensor_size(g,idx);
    enum ggml_type type=gguf_get_tensor_type(g,idx);int64_t n_el;int blck=(int)ggml_blck_size(type);
    if(type==GGML_TYPE_F32)n_el=(int64_t)(tsize/sizeof(float));else if(type==GGML_TYPE_F16)n_el=(int64_t)(tsize/sizeof(ggml_fp16_t));
    else n_el=(int64_t)(tsize/ggml_type_size(type))*blck;
    fseek(f,(long)off,SEEK_SET);
    if(type==GGML_TYPE_F32){fread(dst,sizeof(float),(size_t)n_el,f);}
    else if(type==GGML_TYPE_F16){ggml_fp16_t*buf=(ggml_fp16_t*)malloc(tsize);fread(buf,1,tsize,f);ggml_fp16_to_fp32_row(buf,dst,n_el);free(buf);}
    else{void*buf=malloc(tsize);fread(buf,1,tsize,f);ggml_get_type_traits(type)->to_float(buf,dst,n_el);free(buf);}
    return true;}
static bool read_gguf_q8_packed(struct gguf_context*g,FILE*f,const char*name,VkBuf*dst,uint32_t n,uint32_t k){
    int idx=gguf_find_tensor(g,name);if(idx<0){LOGE("Tensor not found: %s",name);return false;}
    enum ggml_type type=gguf_get_tensor_type(g,idx);
    if(type!=GGML_TYPE_Q8_0){LOGI("Tensor %s is not Q8_0 (type=%d); Q8 prefill unavailable for this model",name,(int)type);return false;}
    if((k%32u)!=0u){LOGE("Tensor %s K is not divisible by 32: %u",name,k);return false;}
    const uint32_t blocks=k/32u;
    size_t expected=(size_t)n*(size_t)blocks*(sizeof(ggml_fp16_t)+32u);
    size_t tsize=gguf_get_tensor_size(g,idx);
    if(tsize!=expected){LOGE("Tensor %s Q8_0 size mismatch expected=%zu actual=%zu",name,expected,tsize);return false;}
    const VkDeviceSize words=(VkDeviceSize)n*blocks*9u;
    if(!buf_alloc(dst,words*sizeof(uint32_t))){LOGE("alloc q8 %s",name);return false;}
    uint32_t*out=(uint32_t*)dst->P;
    size_t off=gguf_get_data_offset(g)+gguf_get_tensor_offset(g,idx);
    fseek(f,(long)off,SEEK_SET);
    for(uint32_t row=0;row<n;row++){
        for(uint32_t block=0;block<blocks;block++){
            ggml_fp16_t h=0;
            uint8_t qs[32];
            if(fread(&h,1,sizeof(h),f)!=sizeof(h)||fread(qs,1,sizeof(qs),f)!=sizeof(qs)){
                LOGE("read q8 tensor failed: %s",name);
                return false;
            }
            float scale=ggml_fp16_to_fp32(h);
            uint32_t scale_bits;
            memcpy(&scale_bits,&scale,sizeof(scale_bits));
            out[row*blocks+block]=scale_bits;
            const uint32_t word_base=n*blocks+(row*blocks+block)*8u;
            for(uint32_t wi=0;wi<8u;wi++){
                uint32_t packed=0u;
                for(uint32_t t=0;t<4u;t++){
                    packed|=((uint32_t)qs[wi*4u+t])<<(t*8u);
                }
                out[word_base+wi]=packed;
            }
        }
    }
    buf_flush(dst);
    return true;
}
static bool pack_f32_q8_buffer(const char*name,const VkBuf*src,VkBuf*dst,uint32_t n,uint32_t k){
    if(src==NULL||src->P==NULL||dst==NULL||n==0u||k==0u||(k%32u)!=0u){
        LOGE("Invalid F32 to Q8 pack request for %s",name);
        return false;
    }
    const VkDeviceSize words=(VkDeviceSize)n*(k/32u)*9u;
    if(!buf_alloc(dst,words*sizeof(uint32_t))){
        LOGE("alloc converted q8 %s",name);
        return false;
    }
    const double start_ms=now_ms();
    q8_pack_weight_cpu(src->P,(uint32_t*)dst->P,n,k);
    buf_flush(dst);
    LOGI("Converted %s F32 to packed Q8 in %.2f ms",name,now_ms()-start_ms);
    return true;
}
#define LOAD_BUF(name,buf,nelem) do{if(!buf_alloc(&(buf),((VkDeviceSize)(nelem))*4)){LOGE("alloc %s",name);return -1;}if(!read_gguf(gctx,f,name,(buf).P)){LOGE("read %s",name);return -1;}buf_flush(&(buf));}while(0)
#define TRY_LOAD_Q8(name,qbuf,fbuf,n,k) do{ \
    if(q8_load_ok){ \
        int q8_idx_=gguf_find_tensor(gctx,name); \
        enum ggml_type q8_type_=q8_idx_>=0?gguf_get_tensor_type(gctx,q8_idx_):GGML_TYPE_COUNT; \
        bool q8_ok_=q8_type_==GGML_TYPE_Q8_0 \
            ? read_gguf_q8_packed(gctx,f,name,&(qbuf),(uint32_t)(n),(uint32_t)(k)) \
            : pack_f32_q8_buffer(name,&(fbuf),&(qbuf),(uint32_t)(n),(uint32_t)(k)); \
        if(!q8_ok_)q8_load_ok=false; \
    } \
}while(0)
int osh26_vk_gpu_load_model(const char*path){if(!vk_ok||!path)return -1;
    g_last_attention_max_abs_err=0.0f;g_attention_fallback_layers=0;
    memset(g_last_logits_top5,0,sizeof(g_last_logits_top5));memset(g_last_logits_top5_values,0,sizeof(g_last_logits_top5_values));
    memset(g_last_logits_topk_ids,0,sizeof(g_last_logits_topk_ids));memset(g_last_logits_topk_values,0,sizeof(g_last_logits_topk_values));g_last_logits_topk_count=0;
    g_last_prefill_ms=0.0;g_last_decode_ms=0.0;g_last_lm_head_ms=0.0;g_last_lm_head_gemv_ms=0.0;g_last_lm_head_local_topk_ms=0.0;g_last_lm_head_merge_ms=0.0;g_last_lm_head_wait_ms=0.0;g_last_token_tps=0.0;
    g_last_lm_head_validation_ran=false;g_last_lm_head_validation_ok=false;g_last_lm_head_validation_stage=0;g_last_lm_head_matched_logit_max_abs_err=0.0f;g_last_lm_head_top1_match=false;g_last_lm_head_top5_overlap=0;g_last_lm_head_top20_overlap=0;g_last_lm_head_cpu_top1_margin=0.0f;g_last_lm_head_validation_ms=0.0;memset(g_last_lm_head_ref_top5,0,sizeof(g_last_lm_head_ref_top5));
    g_last_prefill_submit_count=0;g_last_layer_submit_count=0;g_last_lm_head_submit_count=0;g_last_ttft_submit_count=0;g_last_submit_wait_ms=0.0;
    g_last_prefill_qkv_ms=0.0;g_last_prefill_qk_norm_rope_ms=0.0;g_last_prefill_o_proj_ms=0.0;g_last_prefill_down_ms=0.0;
    g_last_prefill_cpu_post_ms=0.0;g_last_prefill_attention_ms=0.0;g_last_prefill_ffn_gate_up_silu_ms=0.0;
    g_last_forward_submit_count=0;g_last_forward_layers_ms=0.0;g_last_forward_attention_ms=0.0;g_last_forward_kv_update_ms=0.0;g_last_forward_lm_head_ms=0.0;g_last_logits_topk_count=0;
    g_q8_only_mode=false; g_embedding_head_shared=false; g_resident_f32_matrix_bytes=0;
    g_submit_cursor=0; g_current_submit_phase = SUBMIT_PHASE_NONE;
    g_current_forward_is_prefill=false;
    if(!B_Dummy.B && !buf_alloc(&B_Dummy,256)){LOGE("alloc dummy");return -1;}
    if(!B_AttnConst.B && !buf_alloc(&B_AttnConst,sizeof(AttnConst))){LOGE("alloc attn const");return -1;}
    if(!B_KVConst.B && !buf_alloc(&B_KVConst,sizeof(AttnConst))){LOGE("alloc kv const");return -1;}
    FILE*f=fopen(path,"rb");if(!f){LOGE("open %s",path);return -1;}
    struct gguf_init_params gp={true,NULL};struct gguf_context*gctx=gguf_init_from_file(path,gp);if(!gctx){fclose(f);return -1;}
    if(!load_qwen3_model_spec(gctx, &g_hdim, &g_idim)){gguf_free(gctx);fclose(f);return -1;}
    LOGI("Runtime dims: HDIM=%d IDIM=%d", HDIM, IDIM);
    Emb=(float*)malloc(((VkDeviceSize)VOCAB*HDIM)*4);if(!read_gguf(gctx,f,"token_embd.weight",Emb)){free(Emb);gguf_free(gctx);fclose(f);return -1;}
    char n[128];
    bool q8_load_ok=true;
    /* Pre-scan: detect if all 7 projection tensors per layer are native GGML_TYPE_Q8_0 */
    bool all_proj_q8 = true;
    static const char *proj_names[] = {
        "blk.%d.attn_q.weight", "blk.%d.attn_k.weight", "blk.%d.attn_v.weight",
        "blk.%d.attn_output.weight", "blk.%d.ffn_gate.weight",
        "blk.%d.ffn_up.weight", "blk.%d.ffn_down.weight"
    };
    for (int l = 0; l < N_LAY && all_proj_q8; l++) {
        for (int ti = 0; ti < 7; ti++) {
            snprintf(n, sizeof(n), proj_names[ti], l);
            int idx = gguf_find_tensor(gctx, n);
            if (idx < 0 || gguf_get_tensor_type(gctx, idx) != GGML_TYPE_Q8_0) {
                all_proj_q8 = false;
                break;
            }
        }
    }
    g_q8_only_mode = all_proj_q8;
    g_resident_f32_matrix_bytes = 0;
    LOGI("Model mode: %s (all projection tensors are Q8_0)", g_q8_only_mode ? "Q8_ONLY" : "HYBRID_F32_Q8");
    for(int l=0;l<N_LAY;l++){
        snprintf(n,sizeof(n),"blk.%d.attn_norm.weight",l);LOAD_BUF(n,W_ra[l],HDIM);
        if (g_q8_only_mode) {
            /* Q8_ONLY: skip F32 projection tensors, load Q8 packed directly */
            snprintf(n,sizeof(n),"blk.%d.attn_q.weight",l);
            if(!read_gguf_q8_packed(gctx,f,n,&WQ_Q[l],QDIM,HDIM)){LOGE("Q8 load %s",n);gguf_free(gctx);fclose(f);return -1;}
            snprintf(n,sizeof(n),"blk.%d.attn_k.weight",l);
            if(!read_gguf_q8_packed(gctx,f,n,&WQ_K[l],KVD,HDIM)){LOGE("Q8 load %s",n);gguf_free(gctx);fclose(f);return -1;}
            snprintf(n,sizeof(n),"blk.%d.attn_v.weight",l);
            if(!read_gguf_q8_packed(gctx,f,n,&WQ_V[l],KVD,HDIM)){LOGE("Q8 load %s",n);gguf_free(gctx);fclose(f);return -1;}
            snprintf(n,sizeof(n),"blk.%d.attn_output.weight",l);
            if(!read_gguf_q8_packed(gctx,f,n,&WQ_O[l],HDIM,QDIM)){LOGE("Q8 load %s",n);gguf_free(gctx);fclose(f);return -1;}
        } else {
            snprintf(n,sizeof(n),"blk.%d.attn_q.weight",l);LOAD_BUF(n,W_Q[l],QDIM*HDIM);TRY_LOAD_Q8(n,WQ_Q[l],W_Q[l],QDIM,HDIM);
            snprintf(n,sizeof(n),"blk.%d.attn_k.weight",l);LOAD_BUF(n,W_K[l],KVD*HDIM);TRY_LOAD_Q8(n,WQ_K[l],W_K[l],KVD,HDIM);
            snprintf(n,sizeof(n),"blk.%d.attn_v.weight",l);LOAD_BUF(n,W_V[l],KVD*HDIM);TRY_LOAD_Q8(n,WQ_V[l],W_V[l],KVD,HDIM);
            snprintf(n,sizeof(n),"blk.%d.attn_output.weight",l);LOAD_BUF(n,W_O[l],HDIM*QDIM);TRY_LOAD_Q8(n,WQ_O[l],W_O[l],HDIM,QDIM);
        }
        snprintf(n,sizeof(n),"blk.%d.attn_q_norm.weight",l);LOAD_BUF(n,W_Qn[l],HD);
        snprintf(n,sizeof(n),"blk.%d.attn_k_norm.weight",l);LOAD_BUF(n,W_Kn[l],HD);
        snprintf(n,sizeof(n),"blk.%d.ffn_norm.weight",l);LOAD_BUF(n,W_rf[l],HDIM);
        if (g_q8_only_mode) {
            snprintf(n,sizeof(n),"blk.%d.ffn_gate.weight",l);
            if(!read_gguf_q8_packed(gctx,f,n,&WQ_Gate[l],IDIM,HDIM)){LOGE("Q8 load %s",n);gguf_free(gctx);fclose(f);return -1;}
            snprintf(n,sizeof(n),"blk.%d.ffn_up.weight",l);
            if(!read_gguf_q8_packed(gctx,f,n,&WQ_Up[l],IDIM,HDIM)){LOGE("Q8 load %s",n);gguf_free(gctx);fclose(f);return -1;}
            snprintf(n,sizeof(n),"blk.%d.ffn_down.weight",l);
            if(!read_gguf_q8_packed(gctx,f,n,&WQ_Down[l],HDIM,IDIM)){LOGE("Q8 load %s",n);gguf_free(gctx);fclose(f);return -1;}
        } else {
            snprintf(n,sizeof(n),"blk.%d.ffn_gate.weight",l);LOAD_BUF(n,W_Gate[l],IDIM*HDIM);TRY_LOAD_Q8(n,WQ_Gate[l],W_Gate[l],IDIM,HDIM);
            snprintf(n,sizeof(n),"blk.%d.ffn_up.weight",l);LOAD_BUF(n,W_Up[l],IDIM*HDIM);TRY_LOAD_Q8(n,WQ_Up[l],W_Up[l],IDIM,HDIM);
            snprintf(n,sizeof(n),"blk.%d.ffn_down.weight",l);LOAD_BUF(n,W_Down[l],HDIM*IDIM);TRY_LOAD_Q8(n,WQ_Down[l],W_Down[l],HDIM,IDIM);
        }
    }
    if (!g_q8_only_mode) {
        /* Calculate resident F32 matrix bytes (projection weights only) */
        g_resident_f32_matrix_bytes = 0;
        for (int l = 0; l < N_LAY; l++) {
            g_resident_f32_matrix_bytes += (VkDeviceSize)(QDIM*HDIM) * 4;  /* Q */
            g_resident_f32_matrix_bytes += (VkDeviceSize)(KVD*HDIM) * 4;  /* K */
            g_resident_f32_matrix_bytes += (VkDeviceSize)(KVD*HDIM) * 4;  /* V */
            g_resident_f32_matrix_bytes += (VkDeviceSize)(HDIM*QDIM) * 4;  /* O */
            g_resident_f32_matrix_bytes += (VkDeviceSize)(IDIM*HDIM) * 4;  /* Gate */
            g_resident_f32_matrix_bytes += (VkDeviceSize)(IDIM*HDIM) * 4;  /* Up */
            g_resident_f32_matrix_bytes += (VkDeviceSize)(HDIM*IDIM) * 4;  /* Down */
        }
    }
    LOAD_BUF("output_norm.weight",W_Fnorm,HDIM);
    float *head_tmp=(float*)malloc(F32((VkDeviceSize)VOCAB*HDIM));
    if(!head_tmp){LOGE("alloc lm head tmp");gguf_free(gctx);fclose(f);return -1;}
    if(gguf_find_tensor(gctx,"output.weight")>=0){
        if(!read_gguf(gctx,f,"output.weight",head_tmp)){free(head_tmp);gguf_free(gctx);fclose(f);return -1;}
    }else{
        memcpy(head_tmp,Emb,F32((VkDeviceSize)VOCAB*HDIM));
    }
    for(int s=0;s<HEAD_SHARDS;s++){
        int base=s*HEAD_SHARD;
        int nv=VOCAB-base;
        if(nv>HEAD_SHARD)nv=HEAD_SHARD;
        if(!buf_alloc(&W_HeadShard[s],F32((VkDeviceSize)nv*HDIM))){LOGE("alloc lm head shard %d",s);free(head_tmp);gguf_free(gctx);fclose(f);return -1;}
        memcpy(W_HeadShard[s].P,head_tmp+(VkDeviceSize)base*HDIM,F32((VkDeviceSize)nv*HDIM));
        buf_flush(&W_HeadShard[s]);
    }
    free(head_tmp);
    gguf_free(gctx);fclose(f);
#define ALLOC_ZERO_BUF(buf, bytes, label) do { \
        if(!buf_alloc(&(buf),(bytes))){LOGE("alloc %s",label);osh26_vk_gpu_free();return -1;} \
        memset((buf).P,0,(buf).size);buf_flush(&(buf)); \
    } while(0)
#define ALLOC_BUF(buf, bytes, label) do { \
        if(!buf_alloc(&(buf),(bytes))){LOGE("alloc %s",label);osh26_vk_gpu_free();return -1;} \
    } while(0)
    if(g_debug_correctness){
        ALLOC_ZERO_BUF(B_KV,((VkDeviceSize)N_LAY*2*HOST_KV_MAX_S*KVD)*4,"B_KV");
    }
    for(int l=0;l<N_LAY;l++){
        ALLOC_ZERO_BUF(B_KCache[l],((VkDeviceSize)N_KVH*HD*MAX_S)*2,"B_KCache");
        ALLOC_ZERO_BUF(B_VCache[l],((VkDeviceSize)N_KVH*HD*MAX_S)*2,"B_VCache");
        ALLOC_ZERO_BUF(B_KPrefixPool[l],((VkDeviceSize)N_KVH*HD*PREFIX_POOL_PAGES*PREFIX_PAGE_TOKENS)*2,"B_KPrefixPool");
        ALLOC_ZERO_BUF(B_VPrefixPool[l],((VkDeviceSize)N_KVH*HD*PREFIX_POOL_PAGES*PREFIX_PAGE_TOKENS)*2,"B_VPrefixPool");
        ALLOC_ZERO_BUF(B_KVUpdateConst[l],sizeof(AttnConst),"B_KVUpdateConst");
        ALLOC_ZERO_BUF(B_AttnRunConst[l],sizeof(AttnConst),"B_AttnRunConst");
    }
    ALLOC_BUF(B_Hid,((VkDeviceSize)MAX_FORWARD_TOKENS*HDIM)*4,"B_Hid");
    ALLOC_BUF(B_Hid2,((VkDeviceSize)MAX_FORWARD_TOKENS*HDIM)*4,"B_Hid2");
    ALLOC_BUF(B_Qb,((VkDeviceSize)MAX_FORWARD_TOKENS*QDIM)*4,"B_Qb");
    ALLOC_BUF(B_Kb,((VkDeviceSize)MAX_FORWARD_TOKENS*KVD)*4,"B_Kb");
    ALLOC_BUF(B_Vb,((VkDeviceSize)MAX_FORWARD_TOKENS*KVD)*4,"B_Vb");
    ALLOC_BUF(B_Sc,((VkDeviceSize)N_HD*MAX_FORWARD_TOKENS*(g_debug_correctness?HOST_KV_MAX_S:PREFILL_ATTN_BLOCK_TOKENS))*4,"B_Sc");
    ALLOC_BUF(B_Att,((VkDeviceSize)MAX_FORWARD_TOKENS*QDIM)*4,"B_Att");
    ALLOC_BUF(B_PrefillOAcc,((VkDeviceSize)MAX_FORWARD_TOKENS*QDIM)*4,"B_PrefillOAcc");
    ALLOC_BUF(B_Gat,((VkDeviceSize)MAX_FORWARD_TOKENS*IDIM)*4,"B_Gat");
    ALLOC_BUF(B_Up,((VkDeviceSize)MAX_FORWARD_TOKENS*IDIM)*4,"B_Up");
    ALLOC_BUF(B_Dwn,((VkDeviceSize)MAX_FORWARD_TOKENS*IDIM)*4,"B_Dwn");
    ALLOC_BUF(B_Tmp,((VkDeviceSize)MAX_FORWARD_TOKENS*HDIM)*4,"B_Tmp");
    ALLOC_BUF(B_Q8In,((VkDeviceSize)MAX_FORWARD_TOKENS*((IDIM+31)/32)*9)*sizeof(uint32_t),"B_Q8In");
    ALLOC_BUF(B_Last,((VkDeviceSize)HDIM)*4,"B_Last");
    ALLOC_BUF(B_LogPart,((VkDeviceSize)HEAD_SHARD)*4,"B_LogPart");
    ALLOC_BUF(B_LmShardTopk,((VkDeviceSize)HEAD_SHARDS*LM_HEAD_LOCAL_TOPK*2)*4,"B_LmShardTopk");
    ALLOC_BUF(B_LmTopk,((VkDeviceSize)LM_HEAD_GLOBAL_TOPK*2)*4,"B_LmTopk");
    if(!create_lm_descriptor_sets()){LOGE("create LM head descriptors");osh26_vk_gpu_free();return -1;}
#undef ALLOC_BUF
#undef ALLOC_ZERO_BUF
    g_prefill_q8_enabled=q8_load_ok;
    g_decode_q8_enabled=q8_load_ok;
    LOGI("Q8 prefill/decode enabled: %s",g_prefill_q8_enabled?"true":"false");
    LOGI("Model mode: %s (resident_f32_matrix_bytes=%llu)",
         g_q8_only_mode?"Q8_ONLY":"HYBRID",
         (unsigned long long)g_resident_f32_matrix_bytes);
    mdl_ok=true;LOGI("Model loaded");return 0;}
#undef TRY_LOAD_Q8

/* ---- Forward pass ---- */
int osh26_vk_gpu_forward_ex(const int*tokens,int nt,int pos,uint32_t flags){if(!mdl_ok)return -1;if(!tokens||nt<=0||nt>MAX_FORWARD_TOKENS||pos<0||pos>MAX_S-nt){LOGE("forward range invalid nt=%d pos=%d max_context=%d",nt,pos,MAX_S);return -1;}const bool need_logits=(flags&OSH26_FORWARD_NEED_LOGITS)!=0;const bool prefill_only=(flags&OSH26_FORWARD_PREFILL_ONLY)!=0;const int validation_stage=(flags&OSH26_FORWARD_VALIDATE_PREFILL)?1:((flags&OSH26_FORWARD_VALIDATE_FIRST_DECODE)?2:0);const bool correctness_check=g_debug_correctness&&validation_stage!=0;const bool debug_check=((flags&OSH26_FORWARD_DEBUG_CHECK)!=0);const bool needs_cpu_attention=correctness_check||debug_check||!g_mnn_attention_enabled||g_attention_fallback_layers!=0;if(needs_cpu_attention&&B_KV.P==NULL){LOGE("CPU attention fallback is unavailable in the 8K fast configuration");return -1;}if(needs_cpu_attention&&pos>HOST_KV_MAX_S-nt){LOGE("CPU attention fallback supports at most %d tokens",HOST_KV_MAX_S);return -1;}
    if(g_q8_only_mode && (debug_check || (correctness_check && nt > 1))){
        LOGE("Q8_ONLY mode: debug/correctness check incompatible (no F32 projection weights allocated). Use F16 model for debug mode.");
        pthread_mutex_unlock(&Mtx);return -1;
    }static int fc=0;if((correctness_check||debug_check)&&++fc<=3)LOGI("forward#%d nt=%d pos=%d flags=0x%x",fc,nt,pos,flags);pthread_mutex_lock(&Mtx);
    if(pos==0){g_last_lm_head_validation_ran=false;g_last_lm_head_validation_ok=false;g_last_lm_head_validation_stage=0;g_last_lm_head_matched_logit_max_abs_err=0.0f;g_last_lm_head_top1_match=false;g_last_lm_head_top5_overlap=0;g_last_lm_head_top20_overlap=0;g_last_lm_head_cpu_top1_margin=0.0f;g_last_lm_head_validation_ms=0.0;g_last_ttft_submit_count=0;memset(g_last_lm_head_ref_top5,0,sizeof(g_last_lm_head_ref_top5));}
    g_last_forward_submit_count=0;g_last_forward_layers_ms=0.0;g_last_forward_attention_ms=0.0;g_last_forward_kv_update_ms=0.0;g_last_forward_lm_head_ms=0.0;
    if (nt > 1) {
        g_last_prefill_submit_count=0;g_last_prefill_qkv_ms=0.0;g_last_prefill_qk_norm_rope_ms=0.0;g_last_prefill_o_proj_ms=0.0;g_last_prefill_down_ms=0.0;g_last_prefill_cpu_post_ms=0.0;g_last_prefill_attention_ms=0.0;g_last_prefill_ffn_gate_up_silu_ms=0.0;
    }
    g_last_layer_submit_count=0;g_last_lm_head_submit_count=0;g_last_submit_wait_ms=0.0;
    g_submit_cursor=0;
    g_current_forward_is_prefill = (nt > 1);
    g_current_submit_phase = SUBMIT_PHASE_LAYER;
    const bool verbose_prefill = correctness_check || debug_check;
    const double forward_start_ms=now_ms();
    float*hidden=B_Hid.P,*hnorm=B_Hid2.P,*qb=B_Qb.P,*kb=B_Kb.P,*vb=B_Vb.P,*sc=B_Sc.P,*att=B_Att.P,*gate=B_Gat.P,*up=B_Up.P,*dwn=B_Dwn.P,*kv=B_KV.P,*tmp=B_Tmp.P;
    for(int i=0;i<nt;i++){int tok=tokens[i];if(tok<0||tok>=VOCAB)tok=0;memcpy(hidden+i*HDIM,Emb+tok*HDIM,HDIM*sizeof(float));}
    buf_flush(&B_Hid);
    int do_diag=(nt==1 && debug_check); /* expensive decode diagnostics */
    for(int l=0;l<N_LAY;l++){
        VkCommandBuffer prefill_cb = VK_NULL_HANDLE;
        const bool decode_gpu_attention = (nt == 1 && !debug_check && g_mnn_attention_enabled && ((g_attention_fallback_layers & (1u << l)) == 0));
        const bool decode_grouped_layer = decode_gpu_attention && !correctness_check && !g_debug_correctness;
        const bool decode_q8_layer = decode_grouped_layer && g_decode_q8_enabled;
        VkCommandBuffer decode_cb = VK_NULL_HANDLE;
        if (nt > 1 && !debug_check) {
            prefill_cb = CB();
            if (prefill_cb == VK_NULL_HANDLE) {
                g_current_forward_is_prefill=false;
                pthread_mutex_unlock(&Mtx);
                return -1;
            }
        } else if (decode_grouped_layer) {
            decode_cb = CB();
            if (decode_cb == VK_NULL_HANDLE) {
                g_current_forward_is_prefill=false;
                pthread_mutex_unlock(&Mtx);
                return -1;
            }
        }
        /* --- RMS attn + Q,K,V projection --- */
        if (nt > 1 && !debug_check) {
          if (l == 0 && verbose_prefill) {
              LOGI("PREFILL stage: layer0 qkv begin");
          }
          const double prefill_qkv_start_ms = now_ms();
          VkCommandBuffer cb1 = prefill_cb;
          RMS(cb1,nt,HDIM,&B_Hid,&W_ra[l],&B_Hid2);
          BARRIER(cb1);
          if(g_prefill_q8_enabled){
            ACT_Q8(cb1,&B_Hid2,&B_Q8In,nt,HDIM);
            MMQ8(cb1,&WQ_Q[l],&B_Q8In,&B_Qb,nt,QDIM,HDIM);
            MMQ8(cb1,&WQ_K[l],&B_Q8In,&B_Kb,nt,KVD,HDIM);
            MMQ8(cb1,&WQ_V[l],&B_Q8In,&B_Vb,nt,KVD,HDIM);
          }else{
            MM(cb1,&W_Q[l],&B_Hid2,&B_Qb,nt,QDIM,HDIM);
            MM(cb1,&W_K[l],&B_Hid2,&B_Kb,nt,KVD,HDIM);
            MM(cb1,&W_V[l],&B_Hid2,&B_Vb,nt,KVD,HDIM);
          }
          BARRIER(cb1);
          const double prefill_qk_norm_rope_start_ms = now_ms();
          RMS(cb1,nt * N_HD,HD,&B_Qb,&W_Qn[l],&B_Qb);
          RMS(cb1,nt * N_KVH,HD,&B_Kb,&W_Kn[l],&B_Kb);
          BARRIER(cb1);
          ROPE_NEOX(cb1,&B_Qb,nt,N_HD,pos,HD);
          ROPE_NEOX(cb1,&B_Kb,nt,N_KVH,pos,HD);
          BARRIER(cb1);
          g_last_prefill_qkv_ms += prefill_qk_norm_rope_start_ms - prefill_qkv_start_ms;
          g_last_prefill_qk_norm_rope_ms += now_ms() - prefill_qk_norm_rope_start_ms;
          if (l == 0 && verbose_prefill) {
              LOGI("PREFILL stage: layer0 qkv recorded");
          }
        } else if (nt == 1 && !debug_check) {
          VkCommandBuffer cb1=decode_grouped_layer?decode_cb:CB();
          RMS(cb1,nt,HDIM,&B_Hid,&W_ra[l],&B_Hid2);
          BARRIER(cb1);
          if(decode_q8_layer){
            ACT_Q8(cb1,&B_Hid2,&B_Q8In,nt,HDIM);
            MMQ8(cb1,&WQ_Q[l],&B_Q8In,&B_Qb,nt,QDIM,HDIM);
            MMQ8(cb1,&WQ_K[l],&B_Q8In,&B_Kb,nt,KVD,HDIM);
            MMQ8(cb1,&WQ_V[l],&B_Q8In,&B_Vb,nt,KVD,HDIM);
          }else{
            MM(cb1,&W_Q[l],&B_Hid2,&B_Qb,nt,QDIM,HDIM);
            MM(cb1,&W_K[l],&B_Hid2,&B_Kb,nt,KVD,HDIM);
            MM(cb1,&W_V[l],&B_Hid2,&B_Vb,nt,KVD,HDIM);
          }
          BARRIER(cb1);
          RMS(cb1,N_HD,HD,&B_Qb,&W_Qn[l],&B_Qb);
          RMS(cb1,N_KVH,HD,&B_Kb,&W_Kn[l],&B_Kb);
          BARRIER(cb1);
          ROPE_NEOX(cb1,&B_Qb,nt,N_HD,pos,HD);
          ROPE_NEOX(cb1,&B_Kb,nt,N_KVH,pos,HD);
          BARRIER(cb1);
          if(!decode_grouped_layer)SubmitNoWait(cb1);
        }else{
          VkCommandBuffer cb1a=CB();
          RMS(cb1a,nt,HDIM,&B_Hid,&W_ra[l],&B_Hid2);
          SubmitNoWait(cb1a);
        }
        if(do_diag)buf_inv(&B_Hid2);
        if(do_diag && l==0){float*cpu=B_Hid.P,*w=W_ra[0].P;float ss=0;for(int i=0;i<HDIM;i++){float v=cpu[i];ss+=v*v;}float inv=1.0f/sqrtf(ss/(float)HDIM+1e-6f);
         float ecpu0=cpu[0]*inv*w[0],egpu0=hnorm[0];LOGI("D01 RMS: err=%.2e",(double)fabsf(ecpu0-egpu0));}
        if(nt==1 && !decode_grouped_layer){buf_inv(&B_Qb);buf_inv(&B_Kb);}
        if(do_diag && (l==0||l==1||l==27)){float*qw=W_Q[l].P,*kw=W_K[l].P;float cq0=0,ck0=0;
         for(int k=0;k<HDIM;k++){cq0+=qw[k]*hnorm[k];ck0+=kw[k]*hnorm[k];}
         LOGI("L%d Q0 e=%.1e K0 e=%.1e",l,(double)fabsf(cq0-qb[0]),(double)fabsf(ck0-kb[0]));}

        if(nt==1 && !debug_check && !decode_grouped_layer){
          buf_inv(&B_Kb);buf_inv(&B_Vb);
        }

        /* --- KV cache + attention --- */
        { int kvo=l*2*HOST_KV_MAX_S*KVD;bool att_on_host=false;bool att_done=false;const double kv_update_start_ms=now_ms();
          if(nt>1 && !debug_check){
              AttnConst kc={{nt,nt,N_HD,N_KVH},{HD,N_HD/N_KVH,pos,pos+nt},{nt,pos+nt,2,MAX_S},{1.0f/sqrtf((float)HD),0,0,0}};
              memcpy(B_KVUpdateConst[l].P,&kc,sizeof(kc));buf_flush(&B_KVUpdateConst[l]);
              if (l == 0 && verbose_prefill) {
                  LOGI("PREFILL stage: layer0 kv update begin");
              }
              { VkCommandBuffer cbk=prefill_cb;
                KV_UPDATE(cbk,&B_Kb,&B_Vb,&B_KCache[l],&B_VCache[l],&B_KVUpdateConst[l],nt);
                BARRIER(cbk);
              }
              if (l == 0 && verbose_prefill) {
                  LOGI("PREFILL stage: layer0 kv update recorded");
              }
          }else if(decode_gpu_attention){
              AttnConst kc={{1,nt,N_HD,N_KVH},{HD,N_HD/N_KVH,pos,pos+nt},{0,0,0,MAX_S},{1.0f/sqrtf((float)HD),0,0,0}};
              memcpy(B_KVUpdateConst[l].P,&kc,sizeof(kc));buf_flush(&B_KVUpdateConst[l]);
              { VkCommandBuffer cbk=decode_grouped_layer?decode_cb:CB();
                KV_UPDATE(cbk,&B_Kb,&B_Vb,&B_KCache[l],&B_VCache[l],&B_KVUpdateConst[l],nt);
                BARRIER(cbk);
                if(!decode_grouped_layer)SubmitNoWait(cbk);
              }
          }else{
              memcpy(kv+kvo+pos*KVD,kb,nt*KVD*sizeof(float));memcpy(kv+kvo+HOST_KV_MAX_S*KVD+pos*KVD,vb,nt*KVD*sizeof(float));
              pack_mnn_kv_cache(l,pos,nt,kb,vb);
          }
          g_last_forward_kv_update_ms += now_ms()-kv_update_start_ms;
          if (nt>1 && !debug_check) {
              if (!prefill_attention_gpu_record(prefill_cb, l, nt, pos)) {
                  LOGE("PREFILL stage: layer%d GPU attention failed", l);
                  vkEndCommandBuffer(prefill_cb);
                  vkResetCommandPool(D, CP, 0);
                  vkResetDescriptorPool(D, DP, 0);
                  g_submit_cursor = 0;
                  g_current_forward_is_prefill=false;
                  pthread_mutex_unlock(&Mtx);
                  return -1;
              } else {
                  att_done = true;
                  if (l == 0 && verbose_prefill) {
                      LOGI("PREFILL stage: layer0 attention done");
                  }
              }
          } else {
              const double attention_start_ms=now_ms();
              if(!att_done && decode_gpu_attention){
                  AttnConst ac={{1,pos+1,N_HD,N_KVH},{HD,N_HD/N_KVH,pos,pos+1},{0,0,0,MAX_S},{1.0f/sqrtf((float)HD),0,0,0}};
                  memcpy(B_AttnRunConst[l].P,&ac,sizeof(ac));buf_flush(&B_AttnRunConst[l]);
                  { VkCommandBuffer cba=decode_grouped_layer?decode_cb:CB();DEC_ATTN(cba,&B_Att,&B_Qb,&B_KCache[l],&B_VCache[l],&B_AttnRunConst[l]);BARRIER(cba);if(!decode_grouped_layer)SubmitNoWait(cba); }
                  if(debug_check){
                      cpu_attention_ref(nt,pos,l,qb,kv,sc,tmp);
                      buf_inv(&B_Att);
                      float maxe=0.0f;for(int i=0;i<QDIM;i++){float e=fabsf(att[i]-tmp[i]);if(e>maxe)maxe=e;}
                      g_last_attention_max_abs_err=maxe;
                      if(maxe>1e-3f){
                          g_attention_fallback_layers|=(1u<<l);
                          memcpy(att,tmp,nt*QDIM*sizeof(float));att_on_host=true;
                          LOGE("L%d MNN decode attention fallback max_abs_err=%.3e",l,(double)maxe);
                      }else if(do_diag && (l==0||l==27)){
                          LOGI("L%d MNN decode attention max_abs_err=%.3e",l,(double)maxe);
                      }
                  }
              }else{
                  cpu_attention_ref(nt,pos,l,qb,kv,sc,tmp);
                  memcpy(att,tmp,nt*QDIM*sizeof(float));att_on_host=true;
              }
              g_last_forward_attention_ms += now_ms()-attention_start_ms;
              if(att_on_host)buf_flush(&B_Att);
          }
        }
        /* GPU output projection reads B_Att for both prefill and decode. */
        if(do_diag && l==0){int kvo=0;int slen=pos+nt;
         /* CPU manual attention for head=0, token=0, dim=0 */
         float*score_cpu=(float*)malloc(slen*sizeof(float));
         int kvh0=0;float*att_cpu=(float*)calloc(QDIM,sizeof(float));
         for(int s=0;s<slen;s++){float dot=0;for(int d=0;d<HD;d++)dot+=qb[d]*kv[kvo+s*KVD+d];score_cpu[s]=dot/sqrtf((float)HD);}
         float mx=score_cpu[0];for(int s=1;s<slen;s++)if(score_cpu[s]>mx)mx=score_cpu[s];
         float sum_w=0;for(int s=0;s<slen;s++){float w=expf(score_cpu[s]-mx);sum_w+=w;score_cpu[s]=w;}
         for(int s=0;s<slen;s++)score_cpu[s]/=sum_w;
         for(int s=0;s<slen;s++){float w=score_cpu[s];for(int d=0;d<HD;d++)att_cpu[d]+=w*kv[kvo+HOST_KV_MAX_S*KVD+s*KVD+d];}
         LOGI("D08 ATTN: gpu[0]=%.6f cpu[0]=%.6f err=%.2e",(double)att[0],(double)att_cpu[0],(double)fabsf(att[0]-att_cpu[0]));
         free(score_cpu);free(att_cpu);}

        if(nt==1 && !debug_check){
          VkCommandBuffer cb2=decode_grouped_layer?decode_cb:CB();
          if(decode_q8_layer){
            ACT_Q8(cb2,&B_Att,&B_Q8In,nt,QDIM);
            MMQ8(cb2,&WQ_O[l],&B_Q8In,&B_Tmp,nt,HDIM,QDIM);
          }else{
            MM(cb2,&W_O[l],&B_Att,&B_Tmp,nt,HDIM,QDIM);
          }
          BARRIER(cb2);
          ADD(cb2,&B_Hid,&B_Tmp,nt*HDIM);
          BARRIER(cb2);
          RMS(cb2,nt,HDIM,&B_Hid,&W_rf[l],&B_Hid2);
          BARRIER(cb2);
          if(decode_q8_layer){
            ACT_Q8(cb2,&B_Hid2,&B_Q8In,nt,HDIM);
            MMQ8(cb2,&WQ_Gate[l],&B_Q8In,&B_Gat,nt,IDIM,HDIM);
            MMQ8(cb2,&WQ_Up[l],&B_Q8In,&B_Up,nt,IDIM,HDIM);
          }else{
            MM(cb2,&W_Gate[l],&B_Hid2,&B_Gat,nt,IDIM,HDIM);
            MM(cb2,&W_Up[l],&B_Hid2,&B_Up,nt,IDIM,HDIM);
          }
          BARRIER(cb2);
          SILU(cb2,&B_Gat,&B_Up,&B_Dwn,nt*IDIM);
          BARRIER(cb2);
          if(decode_q8_layer){
            ACT_Q8(cb2,&B_Dwn,&B_Q8In,nt,IDIM);
            MMQ8(cb2,&WQ_Down[l],&B_Q8In,&B_Tmp,nt,HDIM,IDIM);
          }else{
            MM(cb2,&W_Down[l],&B_Dwn,&B_Tmp,nt,HDIM,IDIM);
          }
          BARRIER(cb2);
          ADD(cb2,&B_Hid,&B_Tmp,nt*HDIM);
          BARRIER(cb2);
          SubmitNoWait(cb2);
        }else if (nt > 1 && !debug_check) {
          const double prefill_o_proj_start_ms = now_ms();
          VkCommandBuffer cb2=prefill_cb;
          if(g_prefill_q8_enabled){
            ACT_Q8(cb2,&B_Att,&B_Q8In,nt,QDIM);
            MMQ8(cb2,&WQ_O[l],&B_Q8In,&B_Tmp,nt,HDIM,QDIM);
          }else{
            MM(cb2,&W_O[l],&B_Att,&B_Tmp,nt,HDIM,QDIM);
          }
          BARRIER(cb2);
          ADD(cb2,&B_Hid,&B_Tmp,nt*HDIM);
          const double prefill_ffn_start_ms = now_ms();
          BARRIER(cb2);
          RMS(cb2,nt,HDIM,&B_Hid,&W_rf[l],&B_Hid2);
          BARRIER(cb2);
          if(g_prefill_q8_enabled){
            ACT_Q8(cb2,&B_Hid2,&B_Q8In,nt,HDIM);
            MMQ8(cb2,&WQ_Gate[l],&B_Q8In,&B_Gat,nt,IDIM,HDIM);
            MMQ8(cb2,&WQ_Up[l],&B_Q8In,&B_Up,nt,IDIM,HDIM);
          }else{
            MM(cb2,&W_Gate[l],&B_Hid2,&B_Gat,nt,IDIM,HDIM);
            MM(cb2,&W_Up[l],&B_Hid2,&B_Up,nt,IDIM,HDIM);
          }
          BARRIER(cb2);
          SILU(cb2,&B_Gat,&B_Up,&B_Dwn,nt*IDIM);
          const double prefill_down_start_ms = now_ms();
          BARRIER(cb2);
          if(g_prefill_q8_enabled){
            ACT_Q8(cb2,&B_Dwn,&B_Q8In,nt,IDIM);
            MMQ8(cb2,&WQ_Down[l],&B_Q8In,&B_Tmp,nt,HDIM,IDIM);
          }else{
            MM(cb2,&W_Down[l],&B_Dwn,&B_Tmp,nt,HDIM,IDIM);
          }
          BARRIER(cb2);
          ADD(cb2,&B_Hid,&B_Tmp,nt*HDIM);
          BARRIER(cb2);
          SubmitNoWait(cb2);
          g_last_prefill_o_proj_ms += prefill_ffn_start_ms - prefill_o_proj_start_ms;
          g_last_prefill_ffn_gate_up_silu_ms += prefill_down_start_ms - prefill_ffn_start_ms;
          g_last_prefill_down_ms += now_ms() - prefill_down_start_ms;
          if (l == 0 && verbose_prefill) {
              LOGI("PREFILL stage: layer0 ffn recorded");
          }
        }else{
        /* --- Submit 2a: O projection + residual --- */
        { float oh0=B_Hid.P[0];
          VkCommandBuffer cb2a=CB();
          MM(cb2a,&W_O[l],&B_Att,&B_Tmp,nt,HDIM,QDIM);
          BARRIER(cb2a);
          ADD(cb2a,&B_Hid,&B_Tmp,nt*HDIM);
          BARRIER(cb2a);
          SubmitNoWait(cb2a);
          if(do_diag&&l==0){float*ow=W_O[0].P;float co=0;for(int k=0;k<QDIM;k++)co+=ow[k]*att[k];
           LOGI("D09 Omat err=%.2e",(double)fabsf(co-B_Tmp.P[0]));
           LOGI("D10 ResA err=%.2e",(double)fabsf((oh0+B_Tmp.P[0])-B_Hid.P[0]));}
        }
        double prefill_ffn_start_ms = 0.0;
        if (nt > 1 && !debug_check) {
            prefill_ffn_start_ms = now_ms();
        }
        /* --- Submit 2b: RMS_FFN + Gate + Up + SiLU --- */
        { VkCommandBuffer cb2b=CB();
          RMS(cb2b,nt,HDIM,&B_Hid,&W_rf[l],&B_Hid2);
          BARRIER(cb2b);
          MM(cb2b,&W_Gate[l],&B_Hid2,&B_Gat,nt,IDIM,HDIM);
          MM(cb2b,&W_Up[l],&B_Hid2,&B_Up,nt,IDIM,HDIM);
          BARRIER(cb2b);
          SILU(cb2b,&B_Gat,&B_Up,&B_Dwn,nt*IDIM);
          BARRIER(cb2b);
          SubmitNoWait(cb2b); }
        if(do_diag&&l==0){buf_inv(&B_Dwn);buf_inv(&B_Gat);buf_inv(&B_Hid2);
         float*cpu=B_Hid.P,*w=W_rf[0].P;float ss=0;for(int i=0;i<HDIM;i++){float v=cpu[i];ss+=v*v;}float inv=1.0f/sqrtf(ss/(float)HDIM+1e-6f);
         float e11=cpu[0]*inv*w[0];LOGI("D11 RMSf err=%.2e",(double)fabsf(e11-B_Hid2.P[0]));
         float*gw=W_Gate[0].P;float cg=0;for(int k=0;k<HDIM;k++)cg+=gw[k]*B_Hid2.P[k];
         LOGI("D12 Gate err=%.2e",(double)fabsf(cg-B_Gat.P[0]));
         float*uw=W_Up[0].P;float cu=0;for(int k=0;k<HDIM;k++)cu+=uw[k]*B_Hid2.P[k];
         LOGI("D13 Up   err=%.2e",(double)fabsf(cu-B_Up.P[0]));
         float g0=B_Gat.P[0],u0=B_Up.P[0],silu=g0/(1.0f+expf(-g0));
         LOGI("D14 SiLU err=%.2e",(double)fabsf(silu*u0-B_Dwn.P[0]));}

        double prefill_down_start_ms = 0.0;
        if (nt > 1 && !debug_check) {
            prefill_down_start_ms = now_ms();
        }
        /* --- Submit 2c: Down projection + residual --- */
        { float ph0=B_Hid.P[0];
          VkCommandBuffer cb2c=CB();
          MM(cb2c,&W_Down[l],&B_Dwn,&B_Tmp,nt,HDIM,IDIM);
          BARRIER(cb2c);
          ADD(cb2c,&B_Hid,&B_Tmp,nt*HDIM);
          BARRIER(cb2c);
          SubmitNoWait(cb2c);
          if(do_diag&&l==0){float*dw=W_Down[0].P;float cd=0;for(int k=0;k<IDIM;k++)cd+=dw[k]*B_Dwn.P[k];
           LOGI("D15 Down err=%.2e",(double)fabsf(cd-B_Tmp.P[0]));
           LOGI("D16 ResF err=%.2e",(double)fabsf((ph0+B_Tmp.P[0])-B_Hid.P[0]));}
        }
        if (prefill_down_start_ms > 0.0) {
            g_last_prefill_down_ms += now_ms() - prefill_down_start_ms;
        }
        if (prefill_ffn_start_ms > 0.0) {
            g_last_prefill_ffn_gate_up_silu_ms += prefill_down_start_ms - prefill_ffn_start_ms;
        }
        if (nt > 1 && !debug_check && l == 0 && verbose_prefill) {
            LOGI("PREFILL stage: layer0 ffn done");
        }
        }
    }
    g_last_forward_layers_ms=now_ms()-forward_start_ms;
    if(prefill_only || !need_logits){
        g_last_prefill_ms=g_last_forward_layers_ms;
        g_current_forward_is_prefill=false;
        WaitAndRecycleAtEnd();
        g_last_ttft_submit_count+=g_last_forward_submit_count;
        pthread_mutex_unlock(&Mtx);return 0;
    }
    if(debug_check){
        if(nt==1)LOGI("L27hid: %.4f %.4f %.4f %.4f",(double)hidden[0],(double)hidden[1],(double)hidden[2],(double)hidden[3]);
        else LOGI("PREFILL L27hid(last): %.4f %.4f %.4f %.4f",
              (double)hidden[(nt-1)*HDIM + 0], (double)hidden[(nt-1)*HDIM + 1],
              (double)hidden[(nt-1)*HDIM + 2], (double)hidden[(nt-1)*HDIM + 3]);
    }
    g_current_forward_is_prefill = false;
    /* --- Final RMS + LM head --- */
    g_current_submit_phase = SUBMIT_PHASE_LM_HEAD;
    const double lm_head_start_ms=now_ms();
    g_last_lm_head_gemv_ms=0.0;g_last_lm_head_local_topk_ms=0.0;g_last_lm_head_merge_ms=0.0;g_last_lm_head_wait_ms=0.0;
    VkBuf *head_src = &B_Hid2;
    VkCommandBuffer lm_cb=CB();
    if(lm_cb==VK_NULL_HANDLE){
      g_current_forward_is_prefill=false;
      pthread_mutex_unlock(&Mtx);
      return -1;
    }
    if(nt>1){
        VkBufferCopy rgn={F32((uint32_t)(nt-1)*HDIM),0,F32(HDIM)};
        vkCmdCopyBuffer(lm_cb,B_Hid.B,B_Last.B,1,&rgn);
        VkMemoryBarrier mb={VK_STRUCTURE_TYPE_MEMORY_BARRIER,0,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(lm_cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,0,0,0);
        RMS(lm_cb,1,HDIM,&B_Last,&W_Fnorm,&B_Hid2);
    }else{
        RMS(lm_cb,1,HDIM,&B_Hid,&W_Fnorm,&B_Hid2);
    }
    BARRIER(lm_cb);
    if(g_gpu_lm_head_enabled){
      g_last_logits_topk_count = 0;
      {
        for(int s=0;s<HEAD_SHARDS;s++){
          int base=s*HEAD_SHARD;
          int nv=VOCAB-base;
          if(nv>HEAD_SHARD)nv=HEAD_SHARD;
          const double gemv_encode_start=now_ms();
          MM(lm_cb,&W_HeadShard[s],head_src,&B_LogPart,1,nv,HDIM);
          g_last_lm_head_gemv_ms+=now_ms()-gemv_encode_start;
          BARRIER(lm_cb);
          uint32_t p_local[4]={(uint32_t)nv,(uint32_t)LM_HEAD_LOCAL_TOPK,(uint32_t)base,(uint32_t)(s*LM_HEAD_LOCAL_TOPK)};
          const double local_encode_start=now_ms();
          vkCmdBindPipeline(lm_cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_LmTopKLocal);
          BIND_LM(lm_cb,DS_LmLocal,p_local);
          vkCmdDispatch(lm_cb,1,1,1);
          g_last_lm_head_local_topk_ms+=now_ms()-local_encode_start;
          BARRIER(lm_cb);
        }
        uint32_t p_merge[4]={(uint32_t)(HEAD_SHARDS*LM_HEAD_LOCAL_TOPK),(uint32_t)LM_HEAD_GLOBAL_TOPK,0,(1u<<31)};
        const double merge_encode_start=now_ms();
        vkCmdBindPipeline(lm_cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_LmTopKMerge);
        BIND_LM(lm_cb,DS_LmMerge,p_merge);
        vkCmdDispatch(lm_cb,1,1,1);
        g_last_lm_head_merge_ms=now_ms()-merge_encode_start;
        BARRIER(lm_cb);
        SubmitNoWait(lm_cb); }
      const double lm_wait_start=now_ms();
      WaitAndRecycleAtEnd();
      g_last_lm_head_wait_ms=now_ms()-lm_wait_start;
      buf_inv(&B_LmTopk);
      const uint32_t *topk_u=(const uint32_t*)B_LmTopk.P;
      for (int i = 0; i < LM_HEAD_GLOBAL_TOPK; ++i) {
          int tok = (int)topk_u[i * 2 + 0];
          float logit;
          memcpy(&logit, &topk_u[i * 2 + 1], sizeof(float));
          topk_insert(g_last_logits_topk_ids, g_last_logits_topk_values, &g_last_logits_topk_count, LM_HEAD_GLOBAL_TOPK, tok, logit);
      }
      if (correctness_check) {
        const double validation_start_ms=now_ms();
        int ref_ids[LM_HEAD_GLOBAL_TOPK];
        float ref_values[LM_HEAD_GLOBAL_TOPK];
        int ref_count = 0;
        buf_inv(head_src);
        compute_lm_head_cpu_topk(head_src, ref_ids, ref_values, &ref_count, LM_HEAD_GLOBAL_TOPK);
        for (int i = 0; i < 5; ++i) {
            g_last_lm_head_ref_top5[i] = (i < ref_count) ? ref_ids[i] : 0;
        }
        if (ref_count >= 2) {
            g_last_lm_head_cpu_top1_margin = ref_values[0] - ref_values[1];
        }
        g_last_lm_head_validation_ran = true;
        g_last_lm_head_validation_stage = validation_stage;
        g_last_lm_head_matched_logit_max_abs_err = 0.0f;
        for (int i = 0; i < ref_count; ++i) {
            for (int j = 0; j < g_last_logits_topk_count; ++j) {
                if (ref_ids[i] == g_last_logits_topk_ids[j]) {
                    const float err=fabsf(ref_values[i]-g_last_logits_topk_values[j]);
                    if(err>g_last_lm_head_matched_logit_max_abs_err)g_last_lm_head_matched_logit_max_abs_err=err;
                    break;
                }
            }
        }
        g_last_lm_head_top1_match = (ref_count > 0 && g_last_logits_topk_count > 0 && ref_ids[0] == g_last_logits_topk_ids[0]);
        g_last_lm_head_top5_overlap = topk_overlap_count(ref_ids,ref_count<5?ref_count:5,g_last_logits_topk_ids,g_last_logits_topk_count<5?g_last_logits_topk_count:5);
        g_last_lm_head_top20_overlap = topk_overlap_count(ref_ids,ref_count<20?ref_count:20,g_last_logits_topk_ids,g_last_logits_topk_count<20?g_last_logits_topk_count:20);
        g_last_lm_head_validation_ok = topk_candidates_valid(g_last_logits_topk_ids,g_last_logits_topk_values,g_last_logits_topk_count);
        if (g_last_lm_head_cpu_top1_margin >= LM_HEAD_TOP1_MARGIN_REQUIRED && !g_last_lm_head_top1_match) {
            g_last_lm_head_validation_ok = false;
        }
        if (g_last_lm_head_top5_overlap < 4 || g_last_lm_head_top20_overlap < 18) {
            g_last_lm_head_validation_ok = false;
        }
        g_last_lm_head_validation_ms=now_ms()-validation_start_ms;
        LOGI("LM_HEAD topk check: stage=%s ok=%s cpu_top1=%d gpu_top1=%d cpu_margin=%.4f top5=%d/5 top20=%d/20 matched_max_err=%.3e validation_ms=%.2f",
             validation_stage==1?"prefill":"first_decode",
             g_last_lm_head_validation_ok ? "true" : "false",
             ref_count > 0 ? ref_ids[0] : -1,
             g_last_logits_topk_count > 0 ? g_last_logits_topk_ids[0] : -1,
             (double)g_last_lm_head_cpu_top1_margin,
             g_last_lm_head_top5_overlap,
             g_last_lm_head_top20_overlap,
             (double)g_last_lm_head_matched_logit_max_abs_err,
             g_last_lm_head_validation_ms);
        if (!g_last_lm_head_validation_ok) {
            LOGE("LM_HEAD validation mismatch recorded; generation continues");
        }
      }
    } else {
      SubmitNoWait(lm_cb);
      WaitAndRecycleAtEnd();
      buf_inv(head_src);
      compute_lm_head_cpu_topk(head_src, g_last_logits_topk_ids, g_last_logits_topk_values, &g_last_logits_topk_count, LM_HEAD_GLOBAL_TOPK);
      for (int i = 0; i < 5; ++i) {
          g_last_lm_head_ref_top5[i] = (i < g_last_logits_topk_count) ? g_last_logits_topk_ids[i] : 0;
      }
    }
    g_last_lm_head_ms=now_ms()-lm_head_start_ms;
    g_last_forward_lm_head_ms=g_last_lm_head_ms;
    if(nt>1)g_last_ttft_submit_count+=g_last_forward_submit_count;
    {
     int top5[5]={0};float top5v[5];
     if (g_last_logits_topk_count > 0) {
         const int top_count = g_last_logits_topk_count < 5 ? g_last_logits_topk_count : 5;
         for (int i = 0; i < top_count; ++i) {
             top5[i] = g_last_logits_topk_ids[i];
             top5v[i] = g_last_logits_topk_values[i];
         }
         for (int i = top_count; i < 5; ++i) {
             top5[i] = 0;
             top5v[i] = -1e30f;
         }
     }
     memcpy(g_last_logits_top5,top5,sizeof(top5));memcpy(g_last_logits_top5_values,top5v,sizeof(top5v));
     if(debug_check&&nt==1)
     LOGI("LOGITS top5: %d(%.1f) %d(%.1f) %d(%.1f) %d(%.1f) %d(%.1f)",top5[0],(double)top5v[0],top5[1],(double)top5v[1],top5[2],(double)top5v[2],top5[3],(double)top5v[3],top5[4],(double)top5v[4]);}
    if(debug_check&&nt>1){const int*top5=g_last_logits_top5;const float*top5v=g_last_logits_top5_values;
     LOGI("PREFILL LOGITS top5: %d(%.1f) %d(%.1f) %d(%.1f) %d(%.1f) %d(%.1f)",top5[0],(double)top5v[0],top5[1],(double)top5v[1],top5[2],(double)top5v[2],top5[3],(double)top5v[3],top5[4],(double)top5v[4]);}
    const double forward_ms=now_ms()-forward_start_ms;
    if(nt>1){
        g_last_prefill_ms=forward_ms;
    }else{
        g_last_decode_ms=forward_ms;
        g_last_token_tps=forward_ms>0.0 ? 1000.0/forward_ms : 0.0;
    }
    g_current_forward_is_prefill=false;
    pthread_mutex_unlock(&Mtx);return 0;}

static bool benchmark_descriptor_set(VkBuf *w,VkBuf *x,VkBuf *y,VkDescriptorSet *out){
    VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP,1,&DSL};
    if(vkAllocateDescriptorSets(D,&da,out)!=VK_SUCCESS)return false;
    VkDescriptorBufferInfo bi[3]={{w->B,0,w->size},{x->B,0,x->size},{y->B,0,y->size}};
    VkWriteDescriptorSet wr[3];memset(wr,0,sizeof(wr));
    for(int i=0;i<3;i++){wr[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;wr[i].dstSet=*out;wr[i].dstBinding=(uint32_t)i;wr[i].descriptorCount=1;wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;wr[i].pBufferInfo=&bi[i];}
    vkUpdateDescriptorSets(D,3,wr,0,0);
    return true;
}

static double benchmark_gemv_pipeline(VkPipeline pipeline,VkDescriptorSet ds,uint32_t n,uint32_t k,int iterations){
    VkCommandBuffer cb=g_submit_cbs[0];
    VkCommandBufferBeginInfo begin={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,0,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,0};
    if(vkBeginCommandBuffer(cb,&begin)!=VK_SUCCESS)return -1.0;
    uint32_t pc[4]={1,n,k,k};
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL,0,1,&ds,0,0);
    vkCmdPushConstants(cb,PL,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc);
    for(int i=0;i<iterations;i++){
        vkCmdDispatch(cb,n,1,1);
        if(i+1<iterations)BARRIER(cb);
    }
    if(vkEndCommandBuffer(cb)!=VK_SUCCESS)return -1.0;
    VkSubmitInfo submit={VK_STRUCTURE_TYPE_SUBMIT_INFO,0,0,0,0,1,&cb,0,0};
    const double start=now_ms();
    if(vkQueueSubmit(Q,1,&submit,VK_NULL_HANDLE)!=VK_SUCCESS)return -1.0;
    if(vkQueueWaitIdle(Q)!=VK_SUCCESS)return -1.0;
    const double elapsed=now_ms()-start;
    vkResetCommandPool(D,CP,0);
    return elapsed/(double)iterations;
}

static double benchmark_gemm_pipeline(VkPipeline pipeline,VkDescriptorSet ds,uint32_t m,uint32_t n,uint32_t k,int iterations){
    VkCommandBuffer cb=g_submit_cbs[0];
    VkCommandBufferBeginInfo begin={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,0,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,0};
    if(vkBeginCommandBuffer(cb,&begin)!=VK_SUCCESS)return -1.0;
    uint32_t pc[4]={m,n,k,k};
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL,0,1,&ds,0,0);
    vkCmdPushConstants(cb,PL,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc);
    for(int i=0;i<iterations;i++){
        vkCmdDispatch(cb,(n+7u)/8u,(m+7u)/8u,1);
        if(i+1<iterations)BARRIER(cb);
    }
    if(vkEndCommandBuffer(cb)!=VK_SUCCESS)return -1.0;
    VkSubmitInfo submit={VK_STRUCTURE_TYPE_SUBMIT_INFO,0,0,0,0,1,&cb,0,0};
    const double start=now_ms();
    if(vkQueueSubmit(Q,1,&submit,VK_NULL_HANDLE)!=VK_SUCCESS)return -1.0;
    if(vkQueueWaitIdle(Q)!=VK_SUCCESS)return -1.0;
    const double elapsed=now_ms()-start;
    vkResetCommandPool(D,CP,0);
    return elapsed/(double)iterations;
}

static double benchmark_act_q8_pipeline(VkDescriptorSet ds,uint32_t m,uint32_t k,int iterations){
    VkCommandBuffer cb=g_submit_cbs[0];
    VkCommandBufferBeginInfo begin={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,0,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,0};
    if(vkBeginCommandBuffer(cb,&begin)!=VK_SUCCESS)return -1.0;
    const uint32_t blocks=(k+31u)/32u;
    uint32_t pc[4]={m,k,k,blocks};
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_ActQ8);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL,0,1,&ds,0,0);
    vkCmdPushConstants(cb,PL,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc);
    for(int i=0;i<iterations;i++){
        vkCmdDispatch(cb,blocks,m,1);
        if(i+1<iterations)BARRIER(cb);
    }
    if(vkEndCommandBuffer(cb)!=VK_SUCCESS)return -1.0;
    VkSubmitInfo submit={VK_STRUCTURE_TYPE_SUBMIT_INFO,0,0,0,0,1,&cb,0,0};
    const double start=now_ms();
    if(vkQueueSubmit(Q,1,&submit,VK_NULL_HANDLE)!=VK_SUCCESS)return -1.0;
    if(vkQueueWaitIdle(Q)!=VK_SUCCESS)return -1.0;
    const double elapsed=now_ms()-start;
    vkResetCommandPool(D,CP,0);
    return elapsed/(double)iterations;
}

static double benchmark_q8_gemm_pipeline(VkDescriptorSet ds,uint32_t m,uint32_t n,uint32_t k,int iterations){
    VkCommandBuffer cb=g_submit_cbs[0];
    VkCommandBufferBeginInfo begin={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,0,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,0};
    if(vkBeginCommandBuffer(cb,&begin)!=VK_SUCCESS)return -1.0;
    uint32_t pc[4]={m,n,k,k};
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_MMQ8);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL,0,1,&ds,0,0);
    vkCmdPushConstants(cb,PL,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc);
    for(int i=0;i<iterations;i++){
        vkCmdDispatch(cb,(n+7u)/8u,(m+7u)/8u,1);
        if(i+1<iterations)BARRIER(cb);
    }
    if(vkEndCommandBuffer(cb)!=VK_SUCCESS)return -1.0;
    VkSubmitInfo submit={VK_STRUCTURE_TYPE_SUBMIT_INFO,0,0,0,0,1,&cb,0,0};
    const double start=now_ms();
    if(vkQueueSubmit(Q,1,&submit,VK_NULL_HANDLE)!=VK_SUCCESS)return -1.0;
    if(vkQueueWaitIdle(Q)!=VK_SUCCESS)return -1.0;
    const double elapsed=now_ms()-start;
    vkResetCommandPool(D,CP,0);
    return elapsed/(double)iterations;
}

static double benchmark_q8_total_pipeline(VkDescriptorSet ds_act,VkDescriptorSet ds_q8,uint32_t m,uint32_t n,uint32_t k,int iterations){
    VkCommandBuffer cb=g_submit_cbs[0];
    VkCommandBufferBeginInfo begin={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,0,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,0};
    if(vkBeginCommandBuffer(cb,&begin)!=VK_SUCCESS)return -1.0;
    const uint32_t blocks=(k+31u)/32u;
    uint32_t pc_act[4]={m,k,k,blocks};
    uint32_t pc_mm[4]={m,n,k,k};
    for(int i=0;i<iterations;i++){
        vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_ActQ8);
        vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL,0,1,&ds_act,0,0);
        vkCmdPushConstants(cb,PL,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc_act);
        vkCmdDispatch(cb,blocks,m,1);
        BARRIER(cb);
        vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_MMQ8);
        vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL,0,1,&ds_q8,0,0);
        vkCmdPushConstants(cb,PL,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc_mm);
        vkCmdDispatch(cb,(n+7u)/8u,(m+7u)/8u,1);
        if(i+1<iterations)BARRIER(cb);
    }
    if(vkEndCommandBuffer(cb)!=VK_SUCCESS)return -1.0;
    VkSubmitInfo submit={VK_STRUCTURE_TYPE_SUBMIT_INFO,0,0,0,0,1,&cb,0,0};
    const double start=now_ms();
    if(vkQueueSubmit(Q,1,&submit,VK_NULL_HANDLE)!=VK_SUCCESS)return -1.0;
    if(vkQueueWaitIdle(Q)!=VK_SUCCESS)return -1.0;
    const double elapsed=now_ms()-start;
    vkResetCommandPool(D,CP,0);
    return elapsed/(double)iterations;
}

static void benchmark_error(const float *actual,const float *reference,int n,float *max_abs,float *rmse){
    double sum_sq=0.0;float maxe=0.0f;
    for(int i=0;i<n;i++){float e=fabsf(actual[i]-reference[i]);if(e>maxe)maxe=e;sum_sq+=(double)e*(double)e;}
    *max_abs=maxe;*rmse=(float)sqrt(sum_sq/(double)n);
}

int osh26_vk_gpu_quant_benchmark(char *out_json,size_t out_size){
    if(out_json==NULL||out_size==0)return -1;
    out_json[0]='\0';
    if(!vk_ok){snprintf(out_json,out_size,"{\"ok\":false,\"error\":\"Vulkan is not ready\"}");return -1;}
    pthread_mutex_lock(&Mtx);
    vkQueueWaitIdle(Q);vkResetCommandPool(D,CP,0);vkResetDescriptorPool(D,DP,0);g_submit_cursor=0;

    const uint32_t n=3072,k=1024,blocks=k/32;
    const int warmup_iterations=3,iterations=20;
    VkBuf wf32={0},wfp16={0},wq4={0},x={0},y={0};
    float *ref_f32=NULL,*ref_fp16=NULL,*ref_q4=NULL,*gpu_f32=NULL,*gpu_fp16=NULL,*gpu_q4=NULL;
    bool ok=buf_alloc(&wf32,F32((VkDeviceSize)n*k))&&
            buf_alloc(&wfp16,(VkDeviceSize)n*k*2)&&
            buf_alloc(&wq4,(VkDeviceSize)n*blocks*9*4)&&
            buf_alloc(&x,F32(k))&&buf_alloc(&y,F32(n));
    if(!ok){snprintf(out_json,out_size,"{\"ok\":false,\"error\":\"benchmark buffer allocation failed\"}");goto cleanup;}

    ref_f32=(float*)calloc(n,sizeof(float));ref_fp16=(float*)calloc(n,sizeof(float));ref_q4=(float*)calloc(n,sizeof(float));
    gpu_f32=(float*)malloc(F32(n));gpu_fp16=(float*)malloc(F32(n));gpu_q4=(float*)malloc(F32(n));
    if(!ref_f32||!ref_fp16||!ref_q4||!gpu_f32||!gpu_fp16||!gpu_q4){snprintf(out_json,out_size,"{\"ok\":false,\"error\":\"benchmark host allocation failed\"}");goto cleanup;}

    for(uint32_t i=0;i<k;i++)x.P[i]=sinf((float)i*0.017f)*0.75f+cosf((float)i*0.003f)*0.25f;
    uint32_t *fp16_words=(uint32_t*)wfp16.P,*q4_words=(uint32_t*)wq4.P;
    for(uint32_t row=0;row<n;row++){
        for(uint32_t col=0;col<k;col++){
            float w=sinf((float)(row*13u+col*7u)*0.0013f)*0.12f+cosf((float)(row+col*3u)*0.007f)*0.03f;
            wf32.P[(VkDeviceSize)row*k+col]=w;
            ref_f32[row]+=w*x.P[col];
            ggml_fp16_t h=ggml_fp32_to_fp16(w);
            uint32_t word_index=(row*k+col)>>1u;
            if((col&1u)==0u)fp16_words[word_index]=(uint32_t)h;else fp16_words[word_index]|=((uint32_t)h)<<16u;
            ref_fp16[row]+=ggml_fp16_to_fp32(h)*x.P[col];
        }
        for(uint32_t block=0;block<blocks;block++){
            float maxabs=0.0f;
            for(uint32_t j=0;j<32;j++){float w=wf32.P[(VkDeviceSize)row*k+block*32+j];float a=fabsf(w);if(a>maxabs)maxabs=a;}
            float scale=maxabs>0.0f?maxabs/7.0f:1.0f;
            uint32_t base=(row*blocks+block)*9u;memcpy(&q4_words[base],&scale,sizeof(scale));
            for(uint32_t j=0;j<8;j++)q4_words[base+1+j]=0u;
            for(uint32_t j=0;j<32;j++){
                float w=wf32.P[(VkDeviceSize)row*k+block*32+j];
                int q=(int)lrintf(w/scale);if(q<-7)q=-7;if(q>7)q=7;
                uint32_t enc=(uint32_t)(q+8),word=j>>3u,shift=(j&7u)*4u;
                q4_words[base+1+word]|=enc<<shift;
                ref_q4[row]+=(float)q*scale*x.P[block*32+j];
            }
        }
    }

    VkDescriptorSet ds_f32=VK_NULL_HANDLE,ds_fp16=VK_NULL_HANDLE,ds_q4=VK_NULL_HANDLE;
    if(!benchmark_descriptor_set(&wf32,&x,&y,&ds_f32)||!benchmark_descriptor_set(&wfp16,&x,&y,&ds_fp16)||!benchmark_descriptor_set(&wq4,&x,&y,&ds_q4)){
        snprintf(out_json,out_size,"{\"ok\":false,\"error\":\"benchmark descriptor allocation failed\"}");goto cleanup;
    }
    benchmark_gemv_pipeline(P_GMV,ds_f32,n,k,warmup_iterations);
    benchmark_gemv_pipeline(P_GMVFp16,ds_fp16,n,k,warmup_iterations);
    benchmark_gemv_pipeline(P_GMVQ4,ds_q4,n,k,warmup_iterations);

    double f32_ms=benchmark_gemv_pipeline(P_GMV,ds_f32,n,k,iterations);memcpy(gpu_f32,y.P,F32(n));
    double fp16_ms=benchmark_gemv_pipeline(P_GMVFp16,ds_fp16,n,k,iterations);memcpy(gpu_fp16,y.P,F32(n));
    double q4_ms=benchmark_gemv_pipeline(P_GMVQ4,ds_q4,n,k,iterations);memcpy(gpu_q4,y.P,F32(n));
    float f32_impl_max,f32_impl_rmse,fp16_impl_max,fp16_impl_rmse,q4_impl_max,q4_impl_rmse,fp16_quant_max,fp16_quant_rmse,q4_quant_max,q4_quant_rmse;
    benchmark_error(gpu_f32,ref_f32,n,&f32_impl_max,&f32_impl_rmse);
    benchmark_error(gpu_fp16,ref_fp16,n,&fp16_impl_max,&fp16_impl_rmse);
    benchmark_error(gpu_q4,ref_q4,n,&q4_impl_max,&q4_impl_rmse);
    benchmark_error(ref_fp16,ref_f32,n,&fp16_quant_max,&fp16_quant_rmse);
    benchmark_error(ref_q4,ref_f32,n,&q4_quant_max,&q4_quant_rmse);
    snprintf(out_json,out_size,
        "{\"ok\":true,\"shape\":{\"n\":%u,\"k\":%u},\"iterations\":%d,"
        "\"f32\":{\"bytes\":%llu,\"ms\":%.6f,\"impl_max_abs_err\":%.8g,\"impl_rmse\":%.8g},"
        "\"fp16_packed\":{\"bytes\":%llu,\"ms\":%.6f,\"speedup\":%.4f,\"impl_max_abs_err\":%.8g,\"impl_rmse\":%.8g,\"quant_max_abs_err\":%.8g,\"quant_rmse\":%.8g},"
        "\"q4_block32\":{\"bytes\":%llu,\"ms\":%.6f,\"speedup\":%.4f,\"impl_max_abs_err\":%.8g,\"impl_rmse\":%.8g,\"quant_max_abs_err\":%.8g,\"quant_rmse\":%.8g}}",
        n,k,iterations,(unsigned long long)wf32.size,f32_ms,(double)f32_impl_max,(double)f32_impl_rmse,
        (unsigned long long)wfp16.size,fp16_ms,f32_ms/fp16_ms,(double)fp16_impl_max,(double)fp16_impl_rmse,(double)fp16_quant_max,(double)fp16_quant_rmse,
        (unsigned long long)wq4.size,q4_ms,f32_ms/q4_ms,(double)q4_impl_max,(double)q4_impl_rmse,(double)q4_quant_max,(double)q4_quant_rmse);

cleanup:
    free(gpu_q4);free(gpu_fp16);free(gpu_f32);free(ref_q4);free(ref_fp16);free(ref_f32);
    buf_free(&y);buf_free(&x);buf_free(&wq4);buf_free(&wfp16);buf_free(&wf32);
    vkResetDescriptorPool(D,DP,0);g_submit_cursor=0;
    pthread_mutex_unlock(&Mtx);
    return strstr(out_json,"\"ok\":true")?0:-1;
}

int osh26_vk_gpu_quant_gemm_benchmark(char *out_json,size_t out_size){
    if(out_json==NULL||out_size==0)return -1;
    out_json[0]='\0';
    if(!vk_ok){snprintf(out_json,out_size,"{\"ok\":false,\"error\":\"Vulkan is not ready\"}");return -1;}
    pthread_mutex_lock(&Mtx);
    vkQueueWaitIdle(Q);vkResetCommandPool(D,CP,0);vkResetDescriptorPool(D,DP,0);g_submit_cursor=0;

    const uint32_t m=64,n=3072,k=1024,blocks=k/32,total=m*n;
    const int warmup_iterations=2,iterations=5,sample_count=64;
    VkBuf wf32={0},wfp16={0},wq4={0},x={0},y={0};
    float *gpu_f32=NULL,*gpu_fp16=NULL,*gpu_q4=NULL;
    bool ok=buf_alloc(&wf32,F32((VkDeviceSize)n*k))&&
            buf_alloc(&wfp16,(VkDeviceSize)n*k*2)&&
            buf_alloc(&wq4,(VkDeviceSize)n*blocks*9*4)&&
            buf_alloc(&x,F32((VkDeviceSize)m*k))&&buf_alloc(&y,F32(total));
    if(!ok){snprintf(out_json,out_size,"{\"ok\":false,\"error\":\"GEMM benchmark buffer allocation failed\"}");goto cleanup;}
    gpu_f32=(float*)malloc(F32(total));gpu_fp16=(float*)malloc(F32(total));gpu_q4=(float*)malloc(F32(total));
    if(!gpu_f32||!gpu_fp16||!gpu_q4){snprintf(out_json,out_size,"{\"ok\":false,\"error\":\"GEMM benchmark host allocation failed\"}");goto cleanup;}

    for(uint32_t row=0;row<m;row++)for(uint32_t col=0;col<k;col++)x.P[(VkDeviceSize)row*k+col]=sinf((float)(row*17u+col)*0.013f)*0.75f+cosf((float)(row+col*5u)*0.003f)*0.25f;
    uint32_t *fp16_words=(uint32_t*)wfp16.P,*q4_words=(uint32_t*)wq4.P;
    for(uint32_t row=0;row<n;row++){
        for(uint32_t col=0;col<k;col++){
            float w=sinf((float)(row*13u+col*7u)*0.0013f)*0.12f+cosf((float)(row+col*3u)*0.007f)*0.03f;
            wf32.P[(VkDeviceSize)row*k+col]=w;
            ggml_fp16_t h=ggml_fp32_to_fp16(w);
            uint32_t word_index=(row*k+col)>>1u;
            if((col&1u)==0u)fp16_words[word_index]=(uint32_t)h;else fp16_words[word_index]|=((uint32_t)h)<<16u;
        }
        for(uint32_t block=0;block<blocks;block++){
            float maxabs=0.0f;
            for(uint32_t j=0;j<32;j++){float w=wf32.P[(VkDeviceSize)row*k+block*32+j];float a=fabsf(w);if(a>maxabs)maxabs=a;}
            float scale=maxabs>0.0f?maxabs/7.0f:1.0f;
            uint32_t base=(row*blocks+block)*9u;memcpy(&q4_words[base],&scale,sizeof(scale));
            for(uint32_t j=0;j<8;j++)q4_words[base+1+j]=0u;
            for(uint32_t j=0;j<32;j++){
                float w=wf32.P[(VkDeviceSize)row*k+block*32+j];
                int q=(int)lrintf(w/scale);if(q<-7)q=-7;if(q>7)q=7;
                q4_words[base+1+(j>>3u)]|=((uint32_t)(q+8))<<((j&7u)*4u);
            }
        }
    }

    VkDescriptorSet ds_f32=VK_NULL_HANDLE,ds_fp16=VK_NULL_HANDLE,ds_q4=VK_NULL_HANDLE;
    if(!benchmark_descriptor_set(&wf32,&x,&y,&ds_f32)||!benchmark_descriptor_set(&wfp16,&x,&y,&ds_fp16)||!benchmark_descriptor_set(&wq4,&x,&y,&ds_q4)){
        snprintf(out_json,out_size,"{\"ok\":false,\"error\":\"GEMM benchmark descriptor allocation failed\"}");goto cleanup;
    }
    benchmark_gemm_pipeline(P_MMt,ds_f32,m,n,k,warmup_iterations);
    benchmark_gemm_pipeline(P_MMFp16,ds_fp16,m,n,k,warmup_iterations);
    benchmark_gemm_pipeline(P_MMQ4,ds_q4,m,n,k,warmup_iterations);

    double f32_ms=benchmark_gemm_pipeline(P_MMt,ds_f32,m,n,k,iterations);memcpy(gpu_f32,y.P,F32(total));
    double fp16_ms=benchmark_gemm_pipeline(P_MMFp16,ds_fp16,m,n,k,iterations);memcpy(gpu_fp16,y.P,F32(total));
    double q4_ms=benchmark_gemm_pipeline(P_MMQ4,ds_q4,m,n,k,iterations);memcpy(gpu_q4,y.P,F32(total));
    float fp16_vs_f32_max,fp16_vs_f32_rmse,q4_vs_f32_max,q4_vs_f32_rmse;
    benchmark_error(gpu_fp16,gpu_f32,(int)total,&fp16_vs_f32_max,&fp16_vs_f32_rmse);
    benchmark_error(gpu_q4,gpu_f32,(int)total,&q4_vs_f32_max,&q4_vs_f32_rmse);

    float f32_sample_max=0.0f,fp16_sample_max=0.0f,q4_sample_max=0.0f;
    double f32_sample_sq=0.0,fp16_sample_sq=0.0,q4_sample_sq=0.0;
    for(int sample=0;sample<sample_count;sample++){
        uint32_t out_index=(uint32_t)(((uint64_t)sample*(uint64_t)(total-1))/(uint64_t)(sample_count-1));
        uint32_t out_m=out_index/n,out_n=out_index%n;
        float ref_f32=0.0f,ref_fp16=0.0f,ref_q4=0.0f;
        for(uint32_t col=0;col<k;col++){
            float activation=x.P[(VkDeviceSize)out_m*k+col];
            float weight=wf32.P[(VkDeviceSize)out_n*k+col];
            ref_f32+=weight*activation;
            uint32_t packed_half=fp16_words[(out_n*k+col)>>1u];
            ggml_fp16_t half=(ggml_fp16_t)(((col&1u)==0u)?(packed_half&0xffffu):(packed_half>>16u));
            ref_fp16+=ggml_fp16_to_fp32(half)*activation;
            uint32_t block=col>>5u,in_block=col&31u,base=(out_n*blocks+block)*9u;
            float scale;memcpy(&scale,&q4_words[base],sizeof(scale));
            uint32_t packed=q4_words[base+1+(in_block>>3u)];
            int q=(int)((packed>>((in_block&7u)*4u))&15u)-8;
            ref_q4+=(float)q*scale*activation;
        }
        float e0=fabsf(gpu_f32[out_index]-ref_f32),e1=fabsf(gpu_fp16[out_index]-ref_fp16),e2=fabsf(gpu_q4[out_index]-ref_q4);
        if(e0>f32_sample_max)f32_sample_max=e0;if(e1>fp16_sample_max)fp16_sample_max=e1;if(e2>q4_sample_max)q4_sample_max=e2;
        f32_sample_sq+=(double)e0*e0;fp16_sample_sq+=(double)e1*e1;q4_sample_sq+=(double)e2*e2;
    }
    float f32_sample_rmse=(float)sqrt(f32_sample_sq/sample_count),fp16_sample_rmse=(float)sqrt(fp16_sample_sq/sample_count),q4_sample_rmse=(float)sqrt(q4_sample_sq/sample_count);
    snprintf(out_json,out_size,
        "{\"ok\":true,\"shape\":{\"m\":%u,\"n\":%u,\"k\":%u},\"iterations\":%d,\"sampled_outputs\":%d,"
        "\"f32\":{\"bytes\":%llu,\"ms\":%.6f,\"sample_impl_max_abs_err\":%.8g,\"sample_impl_rmse\":%.8g},"
        "\"fp16_packed\":{\"bytes\":%llu,\"ms\":%.6f,\"speedup\":%.4f,\"sample_impl_max_abs_err\":%.8g,\"sample_impl_rmse\":%.8g,\"vs_f32_max_abs_err\":%.8g,\"vs_f32_rmse\":%.8g},"
        "\"q4_block32\":{\"bytes\":%llu,\"ms\":%.6f,\"speedup\":%.4f,\"sample_impl_max_abs_err\":%.8g,\"sample_impl_rmse\":%.8g,\"vs_f32_max_abs_err\":%.8g,\"vs_f32_rmse\":%.8g}}",
        m,n,k,iterations,sample_count,(unsigned long long)wf32.size,f32_ms,(double)f32_sample_max,(double)f32_sample_rmse,
        (unsigned long long)wfp16.size,fp16_ms,f32_ms/fp16_ms,(double)fp16_sample_max,(double)fp16_sample_rmse,(double)fp16_vs_f32_max,(double)fp16_vs_f32_rmse,
        (unsigned long long)wq4.size,q4_ms,f32_ms/q4_ms,(double)q4_sample_max,(double)q4_sample_rmse,(double)q4_vs_f32_max,(double)q4_vs_f32_rmse);

cleanup:
    free(gpu_q4);free(gpu_fp16);free(gpu_f32);
    buf_free(&y);buf_free(&x);buf_free(&wq4);buf_free(&wfp16);buf_free(&wf32);
    vkResetDescriptorPool(D,DP,0);g_submit_cursor=0;
    pthread_mutex_unlock(&Mtx);
    return strstr(out_json,"\"ok\":true")?0:-1;
}

static int unpack_i8_cpu(uint32_t word,uint32_t idx){
    uint32_t b=(word>>(idx*8u))&255u;
    return (b&128u)?((int)b-256):(int)b;
}

static void q8_pack_weight_cpu(const float *w,uint32_t *wq,uint32_t n,uint32_t k){
    const uint32_t blocks=(k+31u)/32u;
    const uint32_t words_offset=n*blocks;
    memset(wq,0,(size_t)n*blocks*9u*sizeof(uint32_t));
    for(uint32_t row=0;row<n;row++){
        for(uint32_t block=0;block<blocks;block++){
            float maxabs=0.0f;
            for(uint32_t j=0;j<32u;j++){
                const uint32_t col=block*32u+j;
                const float v=(col<k)?w[(VkDeviceSize)row*k+col]:0.0f;
                const float a=fabsf(v);
                if(a>maxabs)maxabs=a;
            }
            float scale=fmaxf(maxabs/127.0f,1.0e-12f);
            uint32_t scale_bits;
            memcpy(&scale_bits,&scale,sizeof(scale_bits));
            wq[row*blocks+block]=scale_bits;
            const uint32_t word_base=words_offset+(row*blocks+block)*8u;
            for(uint32_t wi=0;wi<8u;wi++){
                uint32_t packed=0u;
                for(uint32_t t=0;t<4u;t++){
                    const uint32_t col=block*32u+wi*4u+t;
                    const float v=(col<k)?w[(VkDeviceSize)row*k+col]:0.0f;
                    int qi=(int)lrintf(fmaxf(-127.0f,fminf(127.0f,v/scale)));
                    packed|=(uint32_t)(qi&255)<<(t*8u);
                }
                wq[word_base+wi]=packed;
            }
        }
    }
}

static float q8_ref_dot_cpu(const uint32_t *wq,const uint32_t *xq,uint32_t m_count,uint32_t n_count,uint32_t k,uint32_t row,uint32_t col){
    (void)m_count;
    const uint32_t blocks=(k+31u)/32u;
    const uint32_t w_words_offset=n_count*blocks;
    const uint32_t x_words_offset=m_count*blocks;
    float sum=0.0f;
    for(uint32_t block=0;block<blocks;block++){
        uint32_t w_scale_bits=wq[col*blocks+block];
        uint32_t x_scale_bits=xq[row*blocks+block];
        float ws,xs;
        memcpy(&ws,&w_scale_bits,sizeof(ws));
        memcpy(&xs,&x_scale_bits,sizeof(xs));
        int dot=0;
        const uint32_t wbase=w_words_offset+(col*blocks+block)*8u;
        const uint32_t xbase=x_words_offset+(row*blocks+block)*8u;
        for(uint32_t wi=0;wi<8u;wi++){
            const uint32_t ww=wq[wbase+wi];
            const uint32_t xw=xq[xbase+wi];
            dot+=unpack_i8_cpu(ww,0u)*unpack_i8_cpu(xw,0u);
            dot+=unpack_i8_cpu(ww,1u)*unpack_i8_cpu(xw,1u);
            dot+=unpack_i8_cpu(ww,2u)*unpack_i8_cpu(xw,2u);
            dot+=unpack_i8_cpu(ww,3u)*unpack_i8_cpu(xw,3u);
        }
        sum+=(float)dot*ws*xs;
    }
    return sum;
}

typedef struct {
    const char *name;
    uint32_t m,n,k,repeats;
    double f32_ms,quant_ms,q8_ms,q8_total_ms,speedup;
    float sample_impl_max_abs_err,sample_impl_rmse;
    float sample_vs_f32_max_abs_err,sample_vs_f32_rmse;
    bool ok;
    char json[1024];
} Q8ShapeBench;

static bool q8_run_shape(Q8ShapeBench *r,char *err,size_t err_size){
    const uint32_t m=r->m,n=r->n,k=r->k,blocks=(k+31u)/32u,total=m*n;
    const int warmup_iterations=1,iterations=3,sample_count=64;
    VkBuf wf32={0},wq={0},x={0},xq={0},y={0};
    bool success=false;
    const VkDeviceSize q_weight_words=(VkDeviceSize)n*blocks*9u;
    const VkDeviceSize q_act_words=(VkDeviceSize)m*blocks*9u;
    if(!buf_alloc(&wf32,F32((VkDeviceSize)n*k))||
       !buf_alloc(&wq,q_weight_words*sizeof(uint32_t))||
       !buf_alloc(&x,F32((VkDeviceSize)m*k))||
       !buf_alloc(&xq,q_act_words*sizeof(uint32_t))||
       !buf_alloc(&y,F32(total))){
        snprintf(err,err_size,"Q8 shape %s buffer allocation failed",r->name);
        goto cleanup;
    }
    for(uint32_t row=0;row<m;row++){
        for(uint32_t col=0;col<k;col++){
            x.P[(VkDeviceSize)row*k+col]=sinf((float)(row*17u+col)*0.013f)*0.75f+cosf((float)(row+col*5u)*0.003f)*0.25f;
        }
    }
    for(uint32_t row=0;row<n;row++){
        for(uint32_t col=0;col<k;col++){
            wf32.P[(VkDeviceSize)row*k+col]=sinf((float)(row*13u+col*7u)*0.0013f)*0.12f+cosf((float)(row+col*3u)*0.007f)*0.03f;
        }
    }
    q8_pack_weight_cpu(wf32.P,(uint32_t*)wq.P,n,k);
    VkDescriptorSet ds_f32=VK_NULL_HANDLE,ds_act=VK_NULL_HANDLE,ds_q8=VK_NULL_HANDLE;
    vkResetDescriptorPool(D,DP,0);
    if(!benchmark_descriptor_set(&wf32,&x,&y,&ds_f32)||
       !benchmark_descriptor_set(&x,&xq,&y,&ds_act)||
       !benchmark_descriptor_set(&wq,&xq,&y,&ds_q8)){
        snprintf(err,err_size,"Q8 shape %s descriptor allocation failed",r->name);
        goto cleanup;
    }
    benchmark_gemm_pipeline(P_MMt,ds_f32,m,n,k,warmup_iterations);
    benchmark_q8_total_pipeline(ds_act,ds_q8,m,n,k,warmup_iterations);
    r->f32_ms=benchmark_gemm_pipeline(P_MMt,ds_f32,m,n,k,iterations);
    r->quant_ms=benchmark_act_q8_pipeline(ds_act,m,k,iterations);
    r->q8_ms=benchmark_q8_gemm_pipeline(ds_q8,m,n,k,iterations);
    r->q8_total_ms=benchmark_q8_total_pipeline(ds_act,ds_q8,m,n,k,iterations);
    if(r->f32_ms<=0.0||r->quant_ms<=0.0||r->q8_ms<=0.0||r->q8_total_ms<=0.0){
        snprintf(err,err_size,"Q8 shape %s benchmark dispatch failed",r->name);
        goto cleanup;
    }
    r->speedup=r->f32_ms/r->q8_total_ms;
    {
        float impl_max=0.0f,vs_f32_max=0.0f;
        double impl_sq=0.0,vs_f32_sq=0.0;
        for(int sample=0;sample<sample_count;sample++){
            uint32_t out_index=(uint32_t)(((uint64_t)sample*(uint64_t)(total-1))/(uint64_t)(sample_count-1));
            uint32_t out_m=out_index/n,out_n=out_index%n;
            float ref_q8=q8_ref_dot_cpu((const uint32_t*)wq.P,(const uint32_t*)xq.P,m,n,k,out_m,out_n);
            float ref_f32=0.0f;
            for(uint32_t col=0;col<k;col++){
                ref_f32+=wf32.P[(VkDeviceSize)out_n*k+col]*x.P[(VkDeviceSize)out_m*k+col];
            }
            const float gpu=y.P[out_index];
            const float e_impl=fabsf(gpu-ref_q8);
            const float e_f32=fabsf(gpu-ref_f32);
            if(e_impl>impl_max)impl_max=e_impl;
            if(e_f32>vs_f32_max)vs_f32_max=e_f32;
            impl_sq+=(double)e_impl*(double)e_impl;
            vs_f32_sq+=(double)e_f32*(double)e_f32;
        }
        r->sample_impl_max_abs_err=impl_max;
        r->sample_impl_rmse=(float)sqrt(impl_sq/(double)sample_count);
        r->sample_vs_f32_max_abs_err=vs_f32_max;
        r->sample_vs_f32_rmse=(float)sqrt(vs_f32_sq/(double)sample_count);
    }
    r->ok=(r->sample_impl_max_abs_err<1.0e-3f||r->sample_impl_rmse<1.0e-4f);
    snprintf(r->json,sizeof(r->json),
        "{\"name\":\"%s\",\"repeats\":%u,\"shape\":{\"m\":%u,\"n\":%u,\"k\":%u},"
        "\"f32_ms\":%.6f,\"q8\":{\"quant_ms\":%.6f,\"gemm_ms\":%.6f,\"total_ms\":%.6f,"
        "\"speedup\":%.4f,\"sample_impl_max_abs_err\":%.8g,\"sample_impl_rmse\":%.8g,"
        "\"sample_vs_f32_max_abs_err\":%.8g,\"sample_vs_f32_rmse\":%.8g,\"ok\":%s}}",
        r->name,r->repeats,m,n,k,r->f32_ms,r->quant_ms,r->q8_ms,r->q8_total_ms,r->speedup,
        (double)r->sample_impl_max_abs_err,(double)r->sample_impl_rmse,
        (double)r->sample_vs_f32_max_abs_err,(double)r->sample_vs_f32_rmse,r->ok?"true":"false");
    success=true;

cleanup:
    buf_free(&y);buf_free(&xq);buf_free(&x);buf_free(&wq);buf_free(&wf32);
    vkResetDescriptorPool(D,DP,0);
    return success;
}

int osh26_vk_gpu_q8_gemm_benchmark(char *out_json,size_t out_size){
    if(out_json==NULL||out_size==0)return -1;
    out_json[0]='\0';
    if(!vk_ok){snprintf(out_json,out_size,"{\"ok\":false,\"error\":\"Vulkan is not ready\"}");return -1;}
    pthread_mutex_lock(&Mtx);
    vkQueueWaitIdle(Q);vkResetCommandPool(D,CP,0);vkResetDescriptorPool(D,DP,0);g_submit_cursor=0;

    Q8ShapeBench shapes[]={
        {.name="q_proj",.m=32,.n=2048,.k=1024,.repeats=1},
        {.name="k_v_proj",.m=32,.n=1024,.k=1024,.repeats=2},
        {.name="o_proj",.m=32,.n=1024,.k=2048,.repeats=1},
        {.name="gate_up",.m=32,.n=3072,.k=1024,.repeats=2},
        {.name="down",.m=32,.n=1024,.k=3072,.repeats=1},
    };
    const int shape_count=(int)(sizeof(shapes)/sizeof(shapes[0]));
    char err[256]={0};
    double weighted_f32=0.0,weighted_q8=0.0;
    bool all_ok=true;
    for(int i=0;i<shape_count;i++){
        if(!q8_run_shape(&shapes[i],err,sizeof(err))){
            snprintf(out_json,out_size,"{\"ok\":false,\"error\":\"%s\"}",err[0]?err:"Q8 benchmark failed");
            pthread_mutex_unlock(&Mtx);
            return -1;
        }
        weighted_f32+=shapes[i].f32_ms*(double)shapes[i].repeats;
        weighted_q8+=shapes[i].q8_total_ms*(double)shapes[i].repeats;
        if(!shapes[i].ok)all_ok=false;
    }
    const double weighted_speedup=weighted_q8>0.0?weighted_f32/weighted_q8:0.0;
    const bool gate_pass=all_ok&&weighted_speedup>=2.0;
    g_last_q8_benchmark_ran=true;
    g_last_q8_gate_pass=gate_pass;
    g_last_q8_weighted_f32_ms=weighted_f32;
    g_last_q8_weighted_total_ms=weighted_q8;
    g_last_q8_weighted_speedup=weighted_speedup;
    char shape_json[5200];
    size_t off=0;
    off+=(size_t)snprintf(shape_json+off,sizeof(shape_json)-off,"[");
    for(int i=0;i<shape_count&&off<sizeof(shape_json);i++){
        off+=(size_t)snprintf(shape_json+off,sizeof(shape_json)-off,"%s%s",i?",":"",shapes[i].json);
    }
    if(off<sizeof(shape_json)){
        snprintf(shape_json+off,sizeof(shape_json)-off,"]");
    }else{
        shape_json[sizeof(shape_json)-2]=']';
        shape_json[sizeof(shape_json)-1]='\0';
    }
    snprintf(out_json,out_size,
        "{\"ok\":true,\"method\":\"Q8_0 W8A8 dynamic activation quantization\","
        "\"prompt_tokens\":32,\"iterations\":3,\"gate\":{\"required_speedup\":2.0,"
        "\"weighted_f32_ms\":%.6f,\"weighted_q8_total_ms\":%.6f,\"weighted_speedup\":%.4f,"
        "\"correctness_ok\":%s,\"pass\":%s},\"shapes\":%s}",
        weighted_f32,weighted_q8,weighted_speedup,all_ok?"true":"false",gate_pass?"true":"false",shape_json);
    vkResetDescriptorPool(D,DP,0);g_submit_cursor=0;
    pthread_mutex_unlock(&Mtx);
    return 0;
}

int osh26_vk_gpu_collect_topk(struct osh26_vk_candidate *out, int max_out){
    if (g_last_logits_topk_count <= 0) {
        return 0;
    }
    if (out == NULL || max_out <= 0) {
        return g_last_logits_topk_count;
    }
    const int n = g_last_logits_topk_count < max_out ? g_last_logits_topk_count : max_out;
    for (int i = 0; i < n; ++i) {
        out[i].token = g_last_logits_topk_ids[i];
        out[i].logit = g_last_logits_topk_values[i];
    }
    return n;
}
bool osh26_vk_gpu_ready(void){return vk_ok&&mdl_ok;}

bool osh26_vk_gpu_prefix_cache_supported(void){
    if(!vk_ok||!mdl_ok)return false;
    if(!g_mnn_attention_enabled||!g_mnn_prefill_attention_enabled)return false;
    if(g_debug_correctness||g_attention_fallback_layers!=0)return false;
    return B_KPrefixPool[0].P!=NULL&&B_VPrefixPool[0].P!=NULL;
}

int osh26_vk_gpu_reset_cache(void){
    if (!vk_ok) {
        return 0;
    }
    for(int l=0;l<N_LAY;l++){
        if (B_KCache[l].P != NULL) {
            memset(B_KCache[l].P, 0, (size_t)B_KCache[l].size);
            buf_flush(&B_KCache[l]);
        }
        if (B_VCache[l].P != NULL) {
            memset(B_VCache[l].P, 0, (size_t)B_VCache[l].size);
            buf_flush(&B_VCache[l]);
        }
    }
    return 0;
}

int osh26_vk_gpu_prefix_cache_store_page(int page_slot,int src_token){
    pthread_mutex_lock(&Mtx);
    if(!osh26_vk_gpu_prefix_cache_supported()||!prefix_page_args_valid(page_slot,src_token)){
        pthread_mutex_unlock(&Mtx);
        return -1;
    }
    const int pool_stride=PREFIX_POOL_PAGES*PREFIX_PAGE_TOKENS;
    const int pool_token=page_slot*PREFIX_PAGE_TOKENS;
    for(int l=0;l<N_LAY;l++){
        copy_packed_kv_page(&B_KPrefixPool[l],&B_VPrefixPool[l],pool_token,pool_stride,&B_KCache[l],&B_VCache[l],src_token,MAX_S);
        buf_flush(&B_KPrefixPool[l]);
        buf_flush(&B_VPrefixPool[l]);
    }
    pthread_mutex_unlock(&Mtx);
    return 0;
}

int osh26_vk_gpu_prefix_cache_restore_page(int page_slot,int dst_token){
    pthread_mutex_lock(&Mtx);
    if(!osh26_vk_gpu_prefix_cache_supported()||!prefix_page_args_valid(page_slot,dst_token)){
        pthread_mutex_unlock(&Mtx);
        return -1;
    }
    const int pool_stride=PREFIX_POOL_PAGES*PREFIX_PAGE_TOKENS;
    const int pool_token=page_slot*PREFIX_PAGE_TOKENS;
    for(int l=0;l<N_LAY;l++){
        copy_packed_kv_page(&B_KCache[l],&B_VCache[l],dst_token,MAX_S,&B_KPrefixPool[l],&B_VPrefixPool[l],pool_token,pool_stride);
        buf_flush(&B_KCache[l]);
        buf_flush(&B_VCache[l]);
    }
    pthread_mutex_unlock(&Mtx);
    return 0;
}

int osh26_vk_gpu_prefix_cache_clear(void){
    if(!vk_ok){
        return 0;
    }
    pthread_mutex_lock(&Mtx);
    for(int l=0;l<N_LAY;l++){
        if(B_KPrefixPool[l].P!=NULL){
            memset(B_KPrefixPool[l].P,0,(size_t)B_KPrefixPool[l].size);
            buf_flush(&B_KPrefixPool[l]);
        }
        if(B_VPrefixPool[l].P!=NULL){
            memset(B_VPrefixPool[l].P,0,(size_t)B_VPrefixPool[l].size);
            buf_flush(&B_VPrefixPool[l]);
        }
    }
    pthread_mutex_unlock(&Mtx);
    return 0;
}

void osh26_vk_gpu_free(void){
    if(DP_Lm)vkResetDescriptorPool(D,DP_Lm,0);DS_LmLocal=VK_NULL_HANDLE;DS_LmMerge=VK_NULL_HANDLE;
    buf_free(&B_LmTopk);buf_free(&B_LmShardTopk);buf_free(&B_LogPart);buf_free(&B_Last);buf_free(&B_Q8In);buf_free(&B_Tmp);buf_free(&B_Dwn);buf_free(&B_Up);buf_free(&B_Gat);
    buf_free(&B_PrefillOAcc);buf_free(&B_Att);buf_free(&B_Sc);buf_free(&B_Vb);buf_free(&B_Kb);buf_free(&B_Qb);
    buf_free(&B_Hid2);buf_free(&B_Hid);buf_free(&B_KV);
    buf_free(&B_KVConst);buf_free(&B_AttnConst);
    for(int l=0;l<N_LAY;l++){buf_free(&B_AttnRunConst[l]);buf_free(&B_KVUpdateConst[l]);buf_free(&B_VPrefixPool[l]);buf_free(&B_KPrefixPool[l]);buf_free(&B_VCache[l]);buf_free(&B_KCache[l]);}
    for(int s=0;s<HEAD_SHARDS;s++)buf_free(&W_HeadShard[s]);
    buf_free(&W_Fnorm);
    for(int l=0;l<N_LAY;l++){buf_free(&WQ_Down[l]);buf_free(&WQ_Up[l]);buf_free(&WQ_Gate[l]);buf_free(&WQ_O[l]);buf_free(&WQ_V[l]);buf_free(&WQ_K[l]);buf_free(&WQ_Q[l]);buf_free(&W_Down[l]);buf_free(&W_Up[l]);buf_free(&W_Gate[l]);buf_free(&W_rf[l]);buf_free(&W_Kn[l]);buf_free(&W_Qn[l]);buf_free(&W_O[l]);buf_free(&W_V[l]);buf_free(&W_K[l]);buf_free(&W_Q[l]);buf_free(&W_ra[l]);}
    if(Emb){free(Emb);Emb=NULL;}
    mdl_ok=false;g_prefill_q8_enabled=false;g_decode_q8_enabled=false;
}
int osh26_vk_get_stats(struct osh26_vk_stats*o){if(!o)return -1;memset(o,0,sizeof(*o));o->ready=vk_ok;o->registered=mdl_ok;o->mnn_attention_enabled=g_mnn_attention_enabled;o->mnn_prefill_attention_enabled=g_mnn_prefill_attention_enabled;o->prefill_q8_enabled=g_prefill_q8_enabled;o->decode_q8_enabled=g_decode_q8_enabled;o->q8_only_mode=g_q8_only_mode;o->embedding_head_shared=g_embedding_head_shared;o->resident_f32_matrix_bytes=g_resident_f32_matrix_bytes;o->debug_correctness=g_debug_correctness;o->last_attention_max_abs_err=g_last_attention_max_abs_err;o->attention_fallback_layers=g_attention_fallback_layers;o->last_forward_submit_count=g_last_forward_submit_count;o->last_prefill_submit_count=g_last_prefill_submit_count;o->last_layer_submit_count=g_last_layer_submit_count;o->last_lm_head_submit_count=g_last_lm_head_submit_count;o->last_ttft_submit_count=g_last_ttft_submit_count;o->last_forward_layers_ms=g_last_forward_layers_ms;o->last_forward_attention_ms=g_last_forward_attention_ms;o->last_forward_kv_update_ms=g_last_forward_kv_update_ms;o->last_forward_lm_head_ms=g_last_forward_lm_head_ms;o->last_prefill_qkv_ms=g_last_prefill_qkv_ms;o->last_prefill_qk_norm_rope_ms=g_last_prefill_qk_norm_rope_ms;o->last_prefill_o_proj_ms=g_last_prefill_o_proj_ms;o->last_prefill_down_ms=g_last_prefill_down_ms;o->last_prefill_cpu_post_ms=g_last_prefill_cpu_post_ms;o->last_prefill_attention_ms=g_last_prefill_attention_ms;o->last_prefill_ffn_gate_up_silu_ms=g_last_prefill_ffn_gate_up_silu_ms;o->last_submit_wait_ms=g_last_submit_wait_ms;o->last_prefill_ms=g_last_prefill_ms;o->last_decode_ms=g_last_decode_ms;o->last_lm_head_ms=g_last_lm_head_ms;o->last_lm_head_gemv_ms=g_last_lm_head_gemv_ms;o->last_lm_head_local_topk_ms=g_last_lm_head_local_topk_ms;o->last_lm_head_merge_ms=g_last_lm_head_merge_ms;o->last_lm_head_wait_ms=g_last_lm_head_wait_ms;o->last_token_tps=g_last_token_tps;o->gpu_lm_head_enabled=g_gpu_lm_head_enabled;o->last_q8_benchmark_ran=g_last_q8_benchmark_ran;o->last_q8_gate_pass=g_last_q8_gate_pass;o->last_q8_weighted_f32_ms=g_last_q8_weighted_f32_ms;o->last_q8_weighted_total_ms=g_last_q8_weighted_total_ms;o->last_q8_weighted_speedup=g_last_q8_weighted_speedup;o->last_lm_head_validation_ran=g_last_lm_head_validation_ran;o->last_lm_head_validation_ok=g_last_lm_head_validation_ok;o->last_lm_head_validation_stage=g_last_lm_head_validation_stage;o->last_lm_head_matched_logit_max_abs_err=g_last_lm_head_matched_logit_max_abs_err;o->last_lm_head_top1_match=g_last_lm_head_top1_match;o->last_lm_head_top5_overlap=g_last_lm_head_top5_overlap;o->last_lm_head_top20_overlap=g_last_lm_head_top20_overlap;o->last_lm_head_cpu_top1_margin=g_last_lm_head_cpu_top1_margin;o->last_lm_head_validation_ms=g_last_lm_head_validation_ms;memcpy(o->last_lm_head_ref_top5,g_last_lm_head_ref_top5,sizeof(g_last_lm_head_ref_top5));memcpy(o->last_logits_top5,g_last_logits_top5,sizeof(g_last_logits_top5));memcpy(o->last_logits_top5_values,g_last_logits_top5_values,sizeof(g_last_logits_top5_values));return 0;}
void osh26_vk_gpu_set_debug_correctness(bool enabled){g_debug_correctness=enabled;if(!enabled){g_last_attention_max_abs_err=0.0f;g_last_lm_head_validation_ran=false;g_last_lm_head_validation_ok=false;g_last_lm_head_validation_stage=0;}}
