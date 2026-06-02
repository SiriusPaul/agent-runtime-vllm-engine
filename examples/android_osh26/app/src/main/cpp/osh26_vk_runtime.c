#include "osh26_vk_runtime.h"

#include <android/log.h>
#include <dlfcn.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "ggml.h"
#include "gguf.h"

#include "mulmat_tiled.spv.h"
#include "mulmat_reduce.spv.h"
#include "rms_norm_test.spv.h"

#define TAG "OSH26VkRT"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Model architecture constants (Qwen3-0.6B)
#define HIDDEN_DIM  1024
#define INTERM_DIM  3072
#define N_LAYERS    28
#define N_HEADS     16
#define N_KV_HEADS  8
#define HEAD_DIM    128
#define Q_DIM       (N_HEADS * HEAD_DIM)     // 2048
#define KV_DIM      (N_KV_HEADS * HEAD_DIM)  // 1024
#define VOCAB_SIZE  151936
#define MAX_SEQ_LEN 512

// GPU memory pool
typedef struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
    void *         mapped;
    VkDeviceSize   size;
    VkDeviceSize   used;
} gpu_pool_t;

// Per-layer weight offsets (in floats from pool base)
typedef struct {
    uint32_t rms_attn;     // HIDDEN_DIM floats
    uint32_t q_weight;     // Q_DIM * HIDDEN_DIM
    uint32_t k_weight;     // KV_DIM * HIDDEN_DIM
    uint32_t v_weight;     // KV_DIM * HIDDEN_DIM
    uint32_t o_weight;     // HIDDEN_DIM * Q_DIM
    uint32_t rms_ffn;      // HIDDEN_DIM
    uint32_t gate_weight;  // INTERM_DIM * HIDDEN_DIM
    uint32_t up_weight;    // INTERM_DIM * HIDDEN_DIM
    uint32_t down_weight;  // HIDDEN_DIM * INTERM_DIM
} layer_weights_t;

// GPU runtime state
static struct {
    bool     ready;
    bool     model_loaded;

    VkInstance       inst;
    VkPhysicalDevice  phys;
    VkDevice          dev;
    VkQueue           queue;
    uint32_t          qfi;
    VkPhysicalDeviceMemoryProperties mem_props;
    VkCommandPool     cmd_pool;
    VkDescriptorPool  desc_pool;
    VkDescriptorSetLayout mm_dsl;
    VkPipelineLayout  mm_layout;
    VkPipeline        mm_tiled;
    VkPipeline        mm_reduce;
    VkPipeline        rms_norm_pipe;
    VkPipelineLayout  rms_norm_layout;
    VkDescriptorSetLayout rms_dsl;

    gpu_pool_t pool;        // main GPU memory pool
    layer_weights_t layers[N_LAYERS];
    uint32_t emb_offset;    // embedding table: VOCAB_SIZE * HIDDEN_DIM
    uint32_t final_norm;    // HIDDEN_DIM
    uint32_t lm_head;       // VOCAB_SIZE * HIDDEN_DIM

    // KV cache offsets within pool
    uint32_t kv_cache;      // N_LAYERS * 2 * MAX_SEQ_LEN * N_KV_HEADS * HEAD_DIM
    int      kv_pos;        // current write position

    // Activation buffer offsets (reused each forward pass)
    uint32_t act_hidden;    // MAX_SEQ_LEN * HIDDEN_DIM
    uint32_t act_q;         // MAX_SEQ_LEN * Q_DIM
    uint32_t act_k;         // MAX_SEQ_LEN * KV_DIM
    uint32_t act_v;         // MAX_SEQ_LEN * KV_DIM
    uint32_t act_scores;    // N_HEADS * MAX_SEQ_LEN * MAX_SEQ_LEN
    uint32_t act_attn_out;  // MAX_SEQ_LEN * Q_DIM
    uint32_t act_gate;      // MAX_SEQ_LEN * INTERM_DIM
    uint32_t act_up;        // MAX_SEQ_LEN * INTERM_DIM
    uint32_t act_down;      // MAX_SEQ_LEN * HIDDEN_DIM
    uint32_t logits;        // MAX_SEQ_LEN * VOCAB_SIZE

    float * cpu_logits;     // CPU-side logits buffer (VOCAB_SIZE floats)
} g_rt;

#define F32_BYTES(n)  ((VkDeviceSize)(n) * sizeof(float))
#define F32_SIZE(n)   ((uint32_t)(n))

static uint32_t find_mem_type(uint32_t bits, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < g_rt.mem_props.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (g_rt.mem_props.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    return UINT32_MAX;
}

// ---- Memory pool ----

static bool pool_init(gpu_pool_t * p, VkDeviceSize size) {
    p->size = size;
    p->used = 0;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(g_rt.dev, &bci, NULL, &p->buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_rt.dev, p->buf, &mr);
    uint32_t mt = find_mem_type(mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) return false;
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = mt,
    };
    if (vkAllocateMemory(g_rt.dev, &mai, NULL, &p->mem) != VK_SUCCESS) return false;
    vkBindBufferMemory(g_rt.dev, p->buf, p->mem, 0);
    vkMapMemory(g_rt.dev, p->mem, 0, size, 0, &p->mapped);
    LOGI("Pool allocated: %.1f MB", (double)size / 1048576.0);
    return true;
}

// Allocate from pool, returns offset in floats
static uint32_t pool_alloc(gpu_pool_t * p, VkDeviceSize bytes) {
    uint32_t off = (uint32_t)(p->used / sizeof(float));
    p->used += bytes;
    if (p->used > p->size) {
        LOGE("Pool OOM! need=%zu have=%zu", (size_t)bytes, (size_t)p->size);
        return 0;
    }
    return off;
}

static float * pool_ptr(gpu_pool_t * p, uint32_t offset) {
    return (float *)((uint8_t *)p->mapped + (VkDeviceSize)offset * sizeof(float));
}

// ---- Shader helpers ----

static VkShaderModule load_shader(const unsigned char * data, size_t len) {
    VkShaderModuleCreateInfo sm = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = len,
        .pCode = (const uint32_t *)data,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    vkCreateShaderModule(g_rt.dev, &sm, NULL, &shader);
    return shader;
}

// ---- Command dispatch ----

static VkCommandBuffer begin_cb(void) {
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = g_rt.cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    vkAllocateCommandBuffers(g_rt.dev, &ai, &cb);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

static bool end_cb_and_submit(VkCommandBuffer cb) {
    vkEndCommandBuffer(cb);
    VkFence fence;
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(g_rt.dev, &fci, NULL, &fence) != VK_SUCCESS) return false;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    VkResult r = vkQueueSubmit(g_rt.queue, 1, &si, fence);
    if (r == VK_SUCCESS) r = vkWaitForFences(g_rt.dev, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(g_rt.dev, fence, NULL);
    vkFreeCommandBuffers(g_rt.dev, g_rt.cmd_pool, 1, &cb);
    return r == VK_SUCCESS;
}

// Dispatch a MUL_MAT on a single command buffer.
// All tensors are sub-buffers of g_rt.pool at given float offsets.
static void cmd_mul_mat(VkCommandBuffer cb, uint32_t w_off, uint32_t x_off, uint32_t y_off,
                         uint32_t M, uint32_t N, uint32_t K) {
    VkDescriptorSet ds;
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = g_rt.desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &g_rt.mm_dsl,
    };
    vkAllocateDescriptorSets(g_rt.dev, &dsai, &ds);

    VkDescriptorBufferInfo infos[3] = {
        { g_rt.pool.buf, F32_BYTES(w_off), VK_WHOLE_SIZE },
        { g_rt.pool.buf, F32_BYTES(x_off), VK_WHOLE_SIZE },
        { g_rt.pool.buf, F32_BYTES(y_off), VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet writes[3];
    memset(writes, 0, sizeof(writes));
    for (int j = 0; j < 3; j++) {
        writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[j].dstSet = ds; writes[j].dstBinding = j;
        writes[j].descriptorCount = 1;
        writes[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[j].pBufferInfo = &infos[j];
    }
    vkUpdateDescriptorSets(g_rt.dev, 3, writes, 0, NULL);

    uint32_t pc[4] = { M, N, K, K };  // stride_a = K (contiguous)
    vkCmdPushConstants(cb, g_rt.mm_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_rt.mm_layout, 0, 1, &ds, 0, NULL);
    if (M <= 4) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_rt.mm_reduce);
        vkCmdDispatch(cb, N, M, 1);
    } else {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_rt.mm_tiled);
        vkCmdDispatch(cb, (N + 7) / 8, (M + 7) / 8, 1);
    }
}

// ---- Vulkan init ----

int osh26_vk_rt_init(void) {
    if (g_rt.ready) return 0;

    void * lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) { LOGE("dlopen vulkan failed"); return -1; }

    VkApplicationInfo ai = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "OSH26RT",
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    if (vkCreateInstance(&ici, NULL, &g_rt.inst) != VK_SUCCESS) { LOGE("vkCreateInstance"); return -1; }

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(g_rt.inst, &nd, NULL);
    VkPhysicalDevice * pds = calloc(nd, sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(g_rt.inst, &nd, pds);
    g_rt.phys = pds[0];
    free(pds);

    VkPhysicalDeviceProperties pdp;
    vkGetPhysicalDeviceProperties(g_rt.phys, &pdp);
    vkGetPhysicalDeviceMemoryProperties(g_rt.phys, &g_rt.mem_props);
    LOGI("GPU: %s", pdp.deviceName);

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_rt.phys, &qn, NULL);
    VkQueueFamilyProperties * qps = calloc(qn, sizeof(*qps));
    vkGetPhysicalDeviceQueueFamilyProperties(g_rt.phys, &qn, qps);
    for (uint32_t i = 0; i < qn; i++) {
        if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_rt.qfi = i; break; }
    }
    free(qps);

    float p = 1.0f;
    VkDeviceQueueCreateInfo dq = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = g_rt.qfi, .queueCount = 1, .pQueuePriorities = &p,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &dq,
    };
    if (vkCreateDevice(g_rt.phys, &dci, NULL, &g_rt.dev) != VK_SUCCESS) { LOGE("vkCreateDevice"); return -1; }
    vkGetDeviceQueue(g_rt.dev, g_rt.qfi, 0, &g_rt.queue);

    // Command pool
    VkCommandPoolCreateInfo cmd_pool_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = g_rt.qfi,
    };
    vkCreateCommandPool(g_rt.dev, &cmd_pool_ci, NULL, &g_rt.cmd_pool);

    // Descriptor pool
    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096 };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1024, .poolSizeCount = 1, .pPoolSizes = &dps,
    };
    vkCreateDescriptorPool(g_rt.dev, &dpci, NULL, &g_rt.desc_pool);

    // MUL_MAT descriptor set layout
    VkDescriptorSetLayoutBinding mm_bind[3];
    for (int i = 0; i < 3; i++)
        mm_bind[i] = (VkDescriptorSetLayoutBinding){ i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = mm_bind,
    };
    vkCreateDescriptorSetLayout(g_rt.dev, &dslci, NULL, &g_rt.mm_dsl);

    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &g_rt.mm_dsl,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
    };
    vkCreatePipelineLayout(g_rt.dev, &plci, NULL, &g_rt.mm_layout);

    // Load shaders
    VkShaderModule mm_tiled_shader = load_shader(
        examples_android_osh26_app_src_main_cpp_mulmat_tiled_spv,
        examples_android_osh26_app_src_main_cpp_mulmat_tiled_spv_len);
    VkShaderModule mm_reduce_shader = load_shader(
        examples_android_osh26_app_src_main_cpp_mulmat_reduce_spv,
        examples_android_osh26_app_src_main_cpp_mulmat_reduce_spv_len);

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .pName = "main" },
        .layout = g_rt.mm_layout,
    };
    cpci.stage.module = mm_tiled_shader;
    vkCreateComputePipelines(g_rt.dev, NULL, 1, &cpci, NULL, &g_rt.mm_tiled);
    cpci.stage.module = mm_reduce_shader;
    vkCreateComputePipelines(g_rt.dev, NULL, 1, &cpci, NULL, &g_rt.mm_reduce);

    // RMS_NORM pipeline
    VkDescriptorSetLayoutBinding rms_bind[2] = {
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo rms_dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = rms_bind,
    };
    vkCreateDescriptorSetLayout(g_rt.dev, &rms_dslci, NULL, &g_rt.rms_dsl);
    VkPipelineLayoutCreateInfo rms_plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &g_rt.rms_dsl,
    };
    vkCreatePipelineLayout(g_rt.dev, &rms_plci, NULL, &g_rt.rms_norm_layout);

    VkShaderModule rms_shader = load_shader(_tmp_rms_norm_test_spv, _tmp_rms_norm_test_spv_len);
    const uint32_t rms_ncols = 1024;
    VkSpecializationMapEntry spec_e = { 0, 0, sizeof(rms_ncols) };
    VkSpecializationInfo spec_i = { 1, &spec_e, sizeof(rms_ncols), &rms_ncols };
    VkComputePipelineCreateInfo rms_cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = rms_shader, .pName = "main",
                   .pSpecializationInfo = &spec_i },
        .layout = g_rt.rms_norm_layout,
    };
    vkCreateComputePipelines(g_rt.dev, NULL, 1, &rms_cpci, NULL, &g_rt.rms_norm_pipe);

    g_rt.ready = true;
    LOGI("Vulkan runtime init done");
    return 0;
}

// ---- Model loading ----

// Copy a ggml tensor's F32 data to GPU pool. Returns float offset in pool.
static uint32_t upload_tensor(const struct ggml_tensor * t) {
    int64_t n = ggml_nelements(t);
    VkDeviceSize bytes = F32_BYTES(n);
    uint32_t off = pool_alloc(&g_rt.pool, bytes);
    float * dst = pool_ptr(&g_rt.pool, off);
    if (t->type == GGML_TYPE_F32) {
        memcpy(dst, t->data, (size_t)bytes);
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)t->data, dst, n);
    } else if (t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q6_K) {
        const struct ggml_type_traits * tr = ggml_get_type_traits(t->type);
        tr->to_float(t->data, dst, n);
    }
    return off;
}

// Look up a tensor by name in the llama model
static const struct ggml_tensor * get_tensor(const struct llama_model * model, const char * name) {
    return llama_model_get_tensor(model, name);
}

int osh26_vk_rt_load_model(struct llama_model * model) {
    if (!g_rt.ready || !model) return -1;

    // Compute total GPU memory needed
    VkDeviceSize weight_bytes = 0;
    weight_bytes += F32_BYTES(VOCAB_SIZE * HIDDEN_DIM);  // embedding
    for (int l = 0; l < N_LAYERS; l++) {
        weight_bytes += F32_BYTES(HIDDEN_DIM);             // rms_attn
        weight_bytes += F32_BYTES(Q_DIM * HIDDEN_DIM);     // q
        weight_bytes += F32_BYTES(KV_DIM * HIDDEN_DIM);    // k
        weight_bytes += F32_BYTES(KV_DIM * HIDDEN_DIM);    // v
        weight_bytes += F32_BYTES(HIDDEN_DIM * Q_DIM);     // o
        weight_bytes += F32_BYTES(HIDDEN_DIM);             // rms_ffn
        weight_bytes += F32_BYTES(INTERM_DIM * HIDDEN_DIM);// gate
        weight_bytes += F32_BYTES(INTERM_DIM * HIDDEN_DIM);// up
        weight_bytes += F32_BYTES(HIDDEN_DIM * INTERM_DIM);// down
    }
    weight_bytes += F32_BYTES(HIDDEN_DIM);                  // final_norm
    weight_bytes += F32_BYTES(VOCAB_SIZE * HIDDEN_DIM);     // lm_head

    VkDeviceSize kv_bytes = F32_BYTES(N_LAYERS * 2 * MAX_SEQ_LEN * N_KV_HEADS * HEAD_DIM);
    VkDeviceSize act_bytes = 0;
    act_bytes += F32_BYTES(MAX_SEQ_LEN * HIDDEN_DIM);       // hidden
    act_bytes += F32_BYTES(MAX_SEQ_LEN * Q_DIM);            // q
    act_bytes += F32_BYTES(MAX_SEQ_LEN * KV_DIM);           // k
    act_bytes += F32_BYTES(MAX_SEQ_LEN * KV_DIM);           // v
    act_bytes += F32_BYTES(N_HEADS * MAX_SEQ_LEN * MAX_SEQ_LEN); // scores
    act_bytes += F32_BYTES(MAX_SEQ_LEN * Q_DIM);            // attn_out
    act_bytes += F32_BYTES(MAX_SEQ_LEN * INTERM_DIM);       // gate
    act_bytes += F32_BYTES(MAX_SEQ_LEN * INTERM_DIM);       // up
    act_bytes += F32_BYTES(MAX_SEQ_LEN * HIDDEN_DIM);       // down
    act_bytes += F32_BYTES(MAX_SEQ_LEN * VOCAB_SIZE);       // logits

    VkDeviceSize total = weight_bytes + kv_bytes + act_bytes;
    LOGI("GPU memory plan: weights=%.0fMB kv=%.0fMB act=%.0fMB total=%.0fMB",
         (double)weight_bytes / 1048576.0, (double)kv_bytes / 1048576.0,
         (double)act_bytes / 1048576.0, (double)total / 1048576.0);

    if (!pool_init(&g_rt.pool, total)) {
        LOGE("Failed to allocate GPU pool");
        return -1;
    }

    // Load embedding
    {
        const struct ggml_tensor * t = get_tensor(model, "token_embd.weight");
        if (!t) t = get_tensor(model, "model.embed_tokens.weight");
        if (!t) { LOGE("Embedding tensor not found"); return -1; }
        g_rt.emb_offset = upload_tensor(t);
        LOGI("Embedding: %dx%d at offset %u", (int)t->ne[0], (int)t->ne[1], g_rt.emb_offset);
    }

    // Load per-layer weights
    char name[256];
    for (int l = 0; l < N_LAYERS; l++) {
        layer_weights_t * w = &g_rt.layers[l];

        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        const struct ggml_tensor * t = get_tensor(model, name);
        if (!t) { LOGE("Tensor %s not found", name); return -1; }
        w->rms_attn = upload_tensor(t);

        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        t = get_tensor(model, name);
        if (!t) { LOGE("Tensor %s not found", name); return -1; }
        w->q_weight = upload_tensor(t);

        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        t = get_tensor(model, name);
        if (!t) { LOGE("Tensor %s not found", name); return -1; }
        w->k_weight = upload_tensor(t);

        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        t = get_tensor(model, name);
        if (!t) { LOGE("Tensor %s not found", name); return -1; }
        w->v_weight = upload_tensor(t);

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        t = get_tensor(model, name);
        if (!t) { LOGE("Tensor %s not found", name); return -1; }
        w->o_weight = upload_tensor(t);

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        t = get_tensor(model, name);
        if (!t) { LOGE("Tensor %s not found", name); return -1; }
        w->rms_ffn = upload_tensor(t);

        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        t = get_tensor(model, name);
        if (!t) { LOGE("Tensor %s not found", name); return -1; }
        w->gate_weight = upload_tensor(t);

        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        t = get_tensor(model, name);
        if (!t) { LOGE("Tensor %s not found", name); return -1; }
        w->up_weight = upload_tensor(t);

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        t = get_tensor(model, name);
        if (!t) { LOGE("Tensor %s not found", name); return -1; }
        w->down_weight = upload_tensor(t);
    }

    // Final norm
    {
        const struct ggml_tensor * t = get_tensor(model, "output_norm.weight");
        if (!t) t = get_tensor(model, "model.norm.weight");
        if (!t) { LOGE("Final norm tensor not found"); return -1; }
        g_rt.final_norm = upload_tensor(t);
    }

    // LM head
    {
        const struct ggml_tensor * t = get_tensor(model, "output.weight");
        if (!t) t = get_tensor(model, "lm_head.weight");
        if (!t) {
            // Tied with embedding
            g_rt.lm_head = g_rt.emb_offset;
        } else {
            g_rt.lm_head = upload_tensor(t);
        }
    }

    // Activation buffers
    g_rt.act_hidden   = pool_alloc(&g_rt.pool, F32_BYTES(MAX_SEQ_LEN * HIDDEN_DIM));
    g_rt.act_q        = pool_alloc(&g_rt.pool, F32_BYTES(MAX_SEQ_LEN * Q_DIM));
    g_rt.act_k        = pool_alloc(&g_rt.pool, F32_BYTES(MAX_SEQ_LEN * KV_DIM));
    g_rt.act_v        = pool_alloc(&g_rt.pool, F32_BYTES(MAX_SEQ_LEN * KV_DIM));
    g_rt.act_scores   = pool_alloc(&g_rt.pool, F32_BYTES(N_HEADS * MAX_SEQ_LEN * MAX_SEQ_LEN));
    g_rt.act_attn_out = pool_alloc(&g_rt.pool, F32_BYTES(MAX_SEQ_LEN * Q_DIM));
    g_rt.act_gate     = pool_alloc(&g_rt.pool, F32_BYTES(MAX_SEQ_LEN * INTERM_DIM));
    g_rt.act_up       = pool_alloc(&g_rt.pool, F32_BYTES(MAX_SEQ_LEN * INTERM_DIM));
    g_rt.act_down     = pool_alloc(&g_rt.pool, F32_BYTES(MAX_SEQ_LEN * HIDDEN_DIM));
    g_rt.logits       = pool_alloc(&g_rt.pool, F32_BYTES(VOCAB_SIZE));  // 1 token worth

    // KV cache
    g_rt.kv_cache = pool_alloc(&g_rt.pool, kv_bytes);
    g_rt.kv_pos = 0;

    // CPU logits buffer
    g_rt.cpu_logits = calloc(VOCAB_SIZE, sizeof(float));

    g_rt.model_loaded = true;
    LOGI("Model loaded: pool used=%.1fMB", (double)g_rt.pool.used / 1048576.0);
    return 0;
}

// ---- CPU-side compute helpers (run on CPU for correctness) ----

static void rms_norm_cpu(float * y, const float * x, const float * w, int rows, int ncols) {
    const float eps = 1e-6f;
    for (int r = 0; r < rows; r++) {
        const float * xr = x + r * ncols;
        float * yr = y + r * ncols;
        float ss = 0.0f;
        for (int c = 0; c < ncols; c++) ss += xr[c] * xr[c];
        ss = 1.0f / sqrtf(ss / (float)ncols + eps);
        for (int c = 0; c < ncols; c++) yr[c] = xr[c] * ss * w[c];
    }
}

static void rope_cpu(float * q, int q_heads, float * k, int kv_heads,
                     int n_tokens, int pos, int head_dim) {
    // Standard RoPE with theta=10000000^(-2i/d), base frequency 1000000
    const float theta_base = 1000000.0f;
    for (int t = 0; t < n_tokens; t++) {
        int p = pos + t;
        for (int h = 0; h < q_heads; h++) {
            float * qh = q + t * q_heads * head_dim + h * head_dim;
            for (int i = 0; i < head_dim; i += 2) {
                float theta = 1.0f / powf(theta_base, (float)i / (float)head_dim);
                float cos_t = cosf((float)p * theta);
                float sin_t = sinf((float)p * theta);
                float q0 = qh[i], q1 = qh[i + 1];
                qh[i]     = q0 * cos_t - q1 * sin_t;
                qh[i + 1] = q0 * sin_t + q1 * cos_t;
            }
        }
        for (int h = 0; h < kv_heads; h++) {
            float * kh = k + t * kv_heads * head_dim + h * head_dim;
            for (int i = 0; i < head_dim; i += 2) {
                float theta = 1.0f / powf(theta_base, (float)i / (float)head_dim);
                float cos_t = cosf((float)p * theta);
                float sin_t = sinf((float)p * theta);
                float k0 = kh[i], k1 = kh[i + 1];
                kh[i]     = k0 * cos_t - k1 * sin_t;
                kh[i + 1] = k0 * sin_t + k1 * cos_t;
            }
        }
    }
}

static void softmax_cpu(float * x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float * row = x + r * cols;
        float mx = row[0];
        for (int c = 1; c < cols; c++) if (row[c] > mx) mx = row[c];
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) { row[c] = expf(row[c] - mx); sum += row[c]; }
        for (int c = 0; c < cols; c++) row[c] /= sum;
    }
}

static void silu_cpu(float * x, int n) {
    for (int i = 0; i < n; i++) {
        float sx = 1.0f / (1.0f + expf(-x[i]));
        x[i] = x[i] * sx;
    }
}

static void elem_mul_cpu(float * z, const float * x, const float * y, int n) {
    for (int i = 0; i < n; i++) z[i] = x[i] * y[i];
}

static void elem_add_cpu(float * y, const float * x, int n) {
    for (int i = 0; i < n; i++) y[i] += x[i];
}

// ---- Forward pass ----
// GPU: MUL_MAT (the heavy ops)
// CPU: embedding, RMS norm, RoPE, attention, element-wise ops

int osh26_vk_rt_forward(const int * tokens, int n_tokens, int pos,
                         struct llama_context * ctx) {
    if (!g_rt.model_loaded) return -1;

    VkCommandBuffer cb = begin_cb();

    // ---- Embedding lookup (CPU) ----
    float * emb = pool_ptr(&g_rt.pool, g_rt.emb_offset);
    float * hidden = pool_ptr(&g_rt.pool, g_rt.act_hidden);
    for (int i = 0; i < n_tokens; i++) {
        int tok = tokens[i];
        if (tok < 0 || tok >= VOCAB_SIZE) tok = 0;
        memcpy(hidden + i * HIDDEN_DIM, emb + tok * HIDDEN_DIM,
               HIDDEN_DIM * sizeof(float));
    }

    float * q_buf  = pool_ptr(&g_rt.pool, g_rt.act_q);
    float * k_buf  = pool_ptr(&g_rt.pool, g_rt.act_k);
    float * v_buf  = pool_ptr(&g_rt.pool, g_rt.act_v);
    float * scores = pool_ptr(&g_rt.pool, g_rt.act_scores);
    float * attn_o = pool_ptr(&g_rt.pool, g_rt.act_attn_out);
    float * gate   = pool_ptr(&g_rt.pool, g_rt.act_gate);
    float * up     = pool_ptr(&g_rt.pool, g_rt.act_up);
    float * down   = pool_ptr(&g_rt.pool, g_rt.act_down);

    // ---- Process each layer (GPU MUL_MAT + CPU everything else) ----
    for (int l = 0; l < N_LAYERS; l++) {
        layer_weights_t * w = &g_rt.layers[l];

        // RMS Norm (attention) — CPU
        float * rms_w = pool_ptr(&g_rt.pool, w->rms_attn);
        rms_norm_cpu(hidden, hidden, rms_w, n_tokens, HIDDEN_DIM);

        // QKV projections — GPU
        cmd_mul_mat(cb, w->q_weight, g_rt.act_hidden, g_rt.act_q, n_tokens, Q_DIM, HIDDEN_DIM);
        cmd_mul_mat(cb, w->k_weight, g_rt.act_hidden, g_rt.act_k, n_tokens, KV_DIM, HIDDEN_DIM);
        cmd_mul_mat(cb, w->v_weight, g_rt.act_hidden, g_rt.act_v, n_tokens, KV_DIM, HIDDEN_DIM);

        if (!end_cb_and_submit(cb)) { LOGE("Layer %d QKV submit failed", l); return -1; }

        // RoPE — CPU
        rope_cpu(q_buf, N_HEADS, k_buf, N_KV_HEADS, n_tokens, pos, HEAD_DIM);

        // Store K,V in cache
        float * kv = pool_ptr(&g_rt.pool, g_rt.kv_cache);
        int kv_off = l * 2 * MAX_SEQ_LEN * N_KV_HEADS * HEAD_DIM;
        memcpy(kv + kv_off + pos * KV_DIM, k_buf, n_tokens * KV_DIM * sizeof(float));
        memcpy(kv + kv_off + MAX_SEQ_LEN * KV_DIM + pos * KV_DIM,
               v_buf, n_tokens * KV_DIM * sizeof(float));

        // Attention: QK^T, softmax, PV — CPU
        int seq_len = pos + n_tokens;
        float inv_sqrt_dh = 1.0f / sqrtf((float)HEAD_DIM);
        memset(scores, 0, n_tokens * N_HEADS * MAX_SEQ_LEN * sizeof(float));
        for (int h = 0; h < N_HEADS; h++) {
            int kv_h = h * N_KV_HEADS / N_HEADS;
            for (int t = 0; t < n_tokens; t++) {
                const float * qt = q_buf + t * Q_DIM + h * HEAD_DIM;
                float * score_row = scores + (t * N_HEADS + h) * MAX_SEQ_LEN;
                for (int s = 0; s < seq_len; s++) {
                    const float * ks = kv + kv_off + s * KV_DIM + kv_h * HEAD_DIM;
                    float dot = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) dot += qt[d] * ks[d];
                    score_row[s] = dot * inv_sqrt_dh;
                }
            }
        }

        softmax_cpu(scores, n_tokens * N_HEADS, seq_len);

        memset(attn_o, 0, n_tokens * Q_DIM * sizeof(float));
        for (int h = 0; h < N_HEADS; h++) {
            int kv_h = h * N_KV_HEADS / N_HEADS;
            for (int t = 0; t < n_tokens; t++) {
                float * oh = attn_o + t * Q_DIM + h * HEAD_DIM;
                const float * sr = scores + (t * N_HEADS + h) * MAX_SEQ_LEN;
                for (int s = 0; s < seq_len; s++) {
                    const float * vs = kv + kv_off + MAX_SEQ_LEN * KV_DIM + s * KV_DIM + kv_h * HEAD_DIM;
                    float wgt = sr[s];
                    for (int d = 0; d < HEAD_DIM; d++) oh[d] += wgt * vs[d];
                }
            }
        }

        // Output projection — GPU
        cb = begin_cb();
        cmd_mul_mat(cb, w->o_weight, g_rt.act_attn_out, g_rt.act_attn_out, n_tokens, HIDDEN_DIM, Q_DIM);
        if (!end_cb_and_submit(cb)) { LOGE("Layer %d O submit failed", l); return -1; }
        elem_add_cpu(hidden, attn_o, n_tokens * HIDDEN_DIM);

        // RMS Norm (FFN) — CPU
        rms_w = pool_ptr(&g_rt.pool, w->rms_ffn);
        rms_norm_cpu(hidden, hidden, rms_w, n_tokens, HIDDEN_DIM);

        // FFN gate + up — GPU
        cb = begin_cb();
        cmd_mul_mat(cb, w->gate_weight, g_rt.act_hidden, g_rt.act_gate, n_tokens, INTERM_DIM, HIDDEN_DIM);
        cmd_mul_mat(cb, w->up_weight,   g_rt.act_hidden, g_rt.act_up,   n_tokens, INTERM_DIM, HIDDEN_DIM);
        if (!end_cb_and_submit(cb)) { LOGE("Layer %d gate/up submit failed", l); return -1; }

        // SiLU + gate*up — CPU
        silu_cpu(gate, n_tokens * INTERM_DIM);
        elem_mul_cpu(down, gate, up, n_tokens * INTERM_DIM);

        // Down projection — GPU
        cb = begin_cb();
        cmd_mul_mat(cb, w->down_weight, g_rt.act_down, g_rt.act_hidden, n_tokens, HIDDEN_DIM, INTERM_DIM);
        if (!end_cb_and_submit(cb)) { LOGE("Layer %d down submit failed", l); return -1; }
        elem_add_cpu(hidden, down, n_tokens * HIDDEN_DIM);

        cb = begin_cb();
    }

    // Final RMS Norm — CPU
    {
        float * fn_w = pool_ptr(&g_rt.pool, g_rt.final_norm);
        rms_norm_cpu(hidden, hidden, fn_w, n_tokens, HIDDEN_DIM);
    }

    // LM head — GPU
    cmd_mul_mat(cb, g_rt.lm_head, g_rt.act_hidden, g_rt.logits, n_tokens, VOCAB_SIZE, HIDDEN_DIM);
    if (!end_cb_and_submit(cb)) { LOGE("LM head submit failed"); return -1; }

    // Copy last token's logits to CPU
    float * gpu_logits = pool_ptr(&g_rt.pool, g_rt.logits);
    memcpy(g_rt.cpu_logits, gpu_logits + (n_tokens - 1) * VOCAB_SIZE,
           VOCAB_SIZE * sizeof(float));

    g_rt.kv_pos += n_tokens;
    return 0;
}

// ---- Sampling ----

int osh26_vk_rt_sample(struct llama_context * ctx, float temperature, float top_p, int seed) {
    if (!g_rt.cpu_logits) return -1;

    const struct llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    // Create sampler chain
    struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = false;
    struct llama_sampler * smpl = llama_sampler_chain_init(sp);

    // Apply temperature and top_p
    // (Simplified: just use llama's sampler on our logits)
    // For now, use argmax
    int best = 0;
    float best_val = g_rt.cpu_logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (g_rt.cpu_logits[i] > best_val) {
            best_val = g_rt.cpu_logits[i];
            best = i;
        }
    }

    llama_sampler_free(smpl);
    return best;
}

const float * osh26_vk_rt_get_logits(void) {
    return g_rt.cpu_logits;
}

bool osh26_vk_rt_ready(void) {
    return g_rt.ready && g_rt.model_loaded;
}

void osh26_vk_rt_free(void) {
    if (g_rt.cpu_logits) { free(g_rt.cpu_logits); g_rt.cpu_logits = NULL; }
    if (g_rt.pool.mapped) { vkUnmapMemory(g_rt.dev, g_rt.pool.mem); }
    if (g_rt.pool.mem) { vkFreeMemory(g_rt.dev, g_rt.pool.mem, NULL); }
    if (g_rt.pool.buf) { vkDestroyBuffer(g_rt.dev, g_rt.pool.buf, NULL); }
    g_rt.model_loaded = false;
    g_rt.ready = false;
}
