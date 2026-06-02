#include "osh26_vk_backend.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <android/log.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "mulmat_ggml.spv.h"
#include "mulmat_tiled.spv.h"
#include "mulmat_reduce.spv.h"
#include "rms_norm_test.spv.h"

#define TAG "OSH26Vk"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define OSH26_ENABLE_GPU_RMS_NORM 0

#define CHECK_VK(expr) do { \
    VkResult _res = (expr); \
    if (_res != VK_SUCCESS) { \
        LOGE("%s failed: %d", #expr, (int) _res); \
        return false; \
    } \
} while (0)

static VkInstance g_instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_physical_device = VK_NULL_HANDLE;
static VkDevice g_device = VK_NULL_HANDLE;
static VkQueue g_queue = VK_NULL_HANDLE;
static uint32_t g_queue_family = 0;
static VkPhysicalDeviceMemoryProperties g_mem_props;
static VkCommandPool g_command_pool = VK_NULL_HANDLE;
static VkDescriptorPool g_descriptor_pool = VK_NULL_HANDLE;
static VkDescriptorSetLayout g_mul_mat_dsl = VK_NULL_HANDLE;
static VkPipelineLayout g_mul_mat_layout = VK_NULL_HANDLE;
static VkPipeline g_mul_mat_pipeline = VK_NULL_HANDLE;
static VkShaderModule g_mul_mat_shader = VK_NULL_HANDLE;
static VkPipelineLayout g_mul_mat_tiled_layout = VK_NULL_HANDLE;
static VkPipeline g_mul_mat_tiled_pipeline = VK_NULL_HANDLE;
static VkShaderModule g_mul_mat_tiled_shader = VK_NULL_HANDLE;
static VkPipeline g_mul_mat_reduce_pipeline = VK_NULL_HANDLE;
static VkShaderModule g_mul_mat_reduce_shader = VK_NULL_HANDLE;
static VkDescriptorSetLayout g_rms_norm_dsl = VK_NULL_HANDLE;
static VkPipelineLayout g_rms_norm_layout = VK_NULL_HANDLE;
static VkPipeline g_rms_norm_pipeline = VK_NULL_HANDLE;
static VkShaderModule g_rms_norm_shader = VK_NULL_HANDLE;
static bool g_ready = false;
static bool g_registered = false;
static struct ggml_backend_reg g_reg;
static struct ggml_backend_device g_dev;
static struct ggml_backend_buffer_type g_buft;

struct osh26_vk_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void * mapped;
    size_t size;
};

struct osh26_vk_backend_context {
    int unused;
};

static bool load_vulkan_symbols(void);
static bool init_vulkan(void);
static bool create_static_resources(void);

static uint32_t find_memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < g_mem_props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (g_mem_props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

static bool tensor_is_plain_f32(const struct ggml_tensor * t) {
    return t != NULL && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

static bool tensor_is_plain_floatish(const struct ggml_tensor * t) {
    return t != NULL &&
           (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 ||
            t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_Q4_K ||
            t->type == GGML_TYPE_Q6_K);
}

static bool type_is_f32_expanded(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 ||
           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K;
}

static bool tensor_byte_range_to_float_range(
        const struct ggml_tensor * tensor, size_t offset, size_t size,
        size_t * first, size_t * count) {
    const int64_t n = ggml_nelements(tensor);
    if (n < 0) return false;
    if (tensor->type == GGML_TYPE_F16) {
        if (offset % sizeof(ggml_fp16_t) != 0 || size % sizeof(ggml_fp16_t) != 0) return false;
        *first = offset / sizeof(ggml_fp16_t);
        *count = size / sizeof(ggml_fp16_t);
    } else if (tensor->type == GGML_TYPE_BF16) {
        if (offset % sizeof(ggml_bf16_t) != 0 || size % sizeof(ggml_bf16_t) != 0) return false;
        *first = offset / sizeof(ggml_bf16_t);
        *count = size / sizeof(ggml_bf16_t);
    } else {
        const struct ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
        if (traits == NULL || traits->to_float == NULL || traits->type_size == 0 || traits->blck_size <= 0)
            return false;
        if (offset % traits->type_size != 0 || size % traits->type_size != 0) return false;
        *first = (offset / traits->type_size) * (size_t) traits->blck_size;
        *count = (size / traits->type_size) * (size_t) traits->blck_size;
    }
    return *first <= (size_t) n && *count <= (size_t) n - *first;
}

static bool is_osh26_vk_buft(ggml_backend_buffer_type_t buft) {
    return buft != NULL && buft->iface.get_name == g_buft.iface.get_name;
}

static bool is_osh26_vk_buffer(ggml_backend_buffer_t buffer) {
    return buffer != NULL && is_osh26_vk_buft(buffer->buft);
}

static struct osh26_vk_buffer * tensor_vk_buffer(const struct ggml_tensor * t) {
    if (t == NULL) return NULL;
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    if (!is_osh26_vk_buffer(buffer)) return NULL;
    return (struct osh26_vk_buffer *) buffer->context;
}

static VkDeviceSize tensor_vk_offset(const struct ggml_tensor * t) {
    struct osh26_vk_buffer * ctx = tensor_vk_buffer(t);
    return (VkDeviceSize) ((const uint8_t *) t->data - (const uint8_t *) ctx->mapped);
}

static bool op_is_supported_mul_mat(const struct ggml_tensor * op) {
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];
    if (!tensor_is_plain_floatish(a) || !tensor_is_plain_floatish(b) || op->type != GGML_TYPE_F32)
        return false;
    if (a->ne[0] != b->ne[0]) return false;
    if (!ggml_is_contiguous(op)) return false;
    return true;
}

static bool op_is_supported_rms_norm(const struct ggml_tensor * op) {
    const struct ggml_tensor * src = op->src[0];
    if (!tensor_is_plain_f32(src) || !tensor_is_plain_f32(op)) return false;
    float eps = 0.0f; memcpy(&eps, op->op_params, sizeof(eps));
    if (fabsf(eps - 1.0e-6f) > 1.0e-9f) return false;
    if (src->ne[0] <= 0 || src->ne[0] > UINT32_MAX) return false;
    if (src->ne[0] != 1024) return false;
    if (ggml_nelements(src) / src->ne[0] > UINT32_MAX) return false;
    return true;
}

// ---- Backend interface stubs (minimal) ----
static const char * osh26_vk_backend_get_name(ggml_backend_t b) { GGML_UNUSED(b); return "OSH26_Vulkan"; }
static void osh26_vk_backend_free(ggml_backend_t b) { if (b) { free(b->context); free(b); } }
static void osh26_vk_synchronize(ggml_backend_t b) { GGML_UNUSED(b); if (g_device) vkDeviceWaitIdle(g_device); }
static const char * osh26_vk_buft_get_name(ggml_backend_buffer_type_t b) { GGML_UNUSED(b); return "OSH26_Vulkan"; }
static void * osh26_vk_buffer_get_base(ggml_backend_buffer_t b) { return ((struct osh26_vk_buffer *)b->context)->mapped; }
static void osh26_vk_buffer_free(ggml_backend_buffer_t b) {
    struct osh26_vk_buffer * ctx = (struct osh26_vk_buffer *)b->context;
    if (!ctx) return;
    if (ctx->mapped) vkUnmapMemory(g_device, ctx->memory);
    if (ctx->buffer)  vkDestroyBuffer(g_device, ctx->buffer, NULL);
    if (ctx->memory)  vkFreeMemory(g_device, ctx->memory, NULL);
    free(ctx);
}
static void osh26_vk_buffer_memset_tensor(ggml_backend_buffer_t buf, struct ggml_tensor * t, uint8_t v, size_t off, size_t sz) {
    GGML_UNUSED(buf);
    if (type_is_f32_expanded(t->type)) {
        size_t first, count;
        if (!tensor_byte_range_to_float_range(t, off, sz, &first, &count)) return;
        memset((float *)t->data + first, v == 0 ? 0 : -1, count * sizeof(float));
    } else {
        memset((uint8_t *)t->data + off, v, sz);
    }
}
static void osh26_vk_buffer_set_tensor(ggml_backend_buffer_t buf, struct ggml_tensor * t, const void * d, size_t off, size_t sz) {
    GGML_UNUSED(buf);
    if (t->type == GGML_TYPE_F16) {
        size_t first, count;
        if (!tensor_byte_range_to_float_range(t, off, sz, &first, &count)) return;
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)d, (float *)t->data + first, (int64_t)count);
    } else if (t->type == GGML_TYPE_BF16) {
        size_t first, count;
        if (!tensor_byte_range_to_float_range(t, off, sz, &first, &count)) return;
        const ggml_bf16_t * src = (const ggml_bf16_t *)d;
        float * dst = (float *)t->data + first;
        for (size_t i = 0; i < count; i++) dst[i] = ggml_bf16_to_fp32(src[i]);
    } else if (t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q6_K) {
        size_t first, count;
        if (!tensor_byte_range_to_float_range(t, off, sz, &first, &count)) return;
        ggml_get_type_traits(t->type)->to_float(d, (float *)t->data + first, (int64_t)count);
    } else {
        memcpy((uint8_t *)t->data + off, d, sz);
    }
}
static void osh26_vk_buffer_get_tensor(ggml_backend_buffer_t buf, const struct ggml_tensor * t, void * d, size_t off, size_t sz) {
    GGML_UNUSED(buf);
    if (t->type == GGML_TYPE_F16) {
        size_t first, count;
        if (!tensor_byte_range_to_float_range(t, off, sz, &first, &count)) return;
        const float * src = (const float *)t->data + first;
        ggml_fp16_t * dst = (ggml_fp16_t *)d;
        for (size_t i = 0; i < count; i++) dst[i] = ggml_fp32_to_fp16(src[i]);
    } else if (t->type == GGML_TYPE_BF16) {
        size_t first, count;
        if (!tensor_byte_range_to_float_range(t, off, sz, &first, &count)) return;
        const float * src = (const float *)t->data + first;
        ggml_bf16_t * dst = (ggml_bf16_t *)d;
        for (size_t i = 0; i < count; i++) dst[i] = ggml_fp32_to_bf16(src[i]);
    } else if (t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q6_K) {
        size_t first, count;
        if (!tensor_byte_range_to_float_range(t, off, sz, &first, &count)) return;
        const struct ggml_type_traits * tr = ggml_get_type_traits(t->type);
        if (tr->from_float_ref) tr->from_float_ref((const float *)t->data + first, d, (int64_t)count);
    } else {
        memcpy(d, (const uint8_t *)t->data + off, sz);
    }
}
static bool osh26_vk_buffer_cpy_tensor(ggml_backend_buffer_t buf, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_UNUSED(buf);
    if (!src || !dst || !src->data || !dst->data) return false;
    if (type_is_f32_expanded(dst->type)) {
        ggml_backend_buffer_t sb = src->view_src ? src->view_src->buffer : src->buffer;
        if (is_osh26_vk_buffer(sb))
            memcpy(dst->data, src->data, (size_t)ggml_nelements(src) * sizeof(float));
        else
            osh26_vk_buffer_set_tensor(buf, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }
    memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;
}
static void osh26_vk_buffer_clear(ggml_backend_buffer_t b, uint8_t v) {
    memset(((struct osh26_vk_buffer *)b->context)->mapped, v, ((struct osh26_vk_buffer *)b->context)->size);
}
static const struct ggml_backend_buffer_i osh26_vk_buffer_iface = {
    osh26_vk_buffer_free, NULL /*get_base*/, NULL /*init_tensor*/,
    osh26_vk_buffer_memset_tensor, osh26_vk_buffer_set_tensor, osh26_vk_buffer_get_tensor,
    NULL, NULL, osh26_vk_buffer_cpy_tensor, osh26_vk_buffer_clear, NULL,
};
static ggml_backend_buffer_t osh26_vk_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (!g_ready && !init_vulkan()) return NULL;
    struct osh26_vk_buffer * ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->size = size;
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    if (vkCreateBuffer(g_device, &bci, NULL, &ctx->buffer) != VK_SUCCESS) { free(ctx); return NULL; }
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_device, ctx->buffer, &mr);
    uint32_t mt = find_memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) { vkDestroyBuffer(g_device, ctx->buffer, NULL); free(ctx); return NULL; }
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mt };
    if (vkAllocateMemory(g_device, &mai, NULL, &ctx->memory) != VK_SUCCESS) { vkDestroyBuffer(g_device, ctx->buffer, NULL); free(ctx); return NULL; }
    if (vkBindBufferMemory(g_device, ctx->buffer, ctx->memory, 0) != VK_SUCCESS || vkMapMemory(g_device, ctx->memory, 0, size, 0, &ctx->mapped) != VK_SUCCESS) {
        vkFreeMemory(g_device, ctx->memory, NULL); vkDestroyBuffer(g_device, ctx->buffer, NULL); free(ctx); return NULL;
    }
    return ggml_backend_buffer_init(buft, osh26_vk_buffer_iface, ctx, size);
}
static size_t osh26_vk_buft_get_alignment(ggml_backend_buffer_type_t b) { GGML_UNUSED(b); return 256; }
static size_t osh26_vk_buft_get_alloc_size(ggml_backend_buffer_type_t b, const struct ggml_tensor * t) {
    GGML_UNUSED(b);
    return type_is_f32_expanded(t->type) ? (size_t)ggml_nelements(t) * sizeof(float) : ggml_nbytes(t);
}
static bool osh26_vk_buft_is_host(ggml_backend_buffer_type_t b) { GGML_UNUSED(b); return false; }
static ggml_backend_t osh26_vk_device_init_backend(ggml_backend_dev_t dev, const char * p) {
    GGML_UNUSED(p);
    if (!g_ready && !init_vulkan()) return NULL;
    struct osh26_vk_backend_context * ctx = calloc(1, sizeof(*ctx));
    ggml_backend_t be = calloc(1, sizeof(struct ggml_backend));
    if (!ctx || !be) { free(ctx); free(be); return NULL; }
    static ggml_guid guid = { 0x4f,0x53,0x48,0x32,0x36,0x2d,0x56,0x4b,0x2d,0x4d,0x56,0x50,0,0,0,1 };
    static const struct ggml_backend_i iface = {
        osh26_vk_backend_get_name, osh26_vk_backend_free, NULL, NULL, NULL, NULL, NULL,
        osh26_vk_synchronize, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    };
    be->guid = &guid; be->iface = iface; be->device = dev; be->context = ctx;
    return be;
}
static const char * osh26_vk_device_get_name(ggml_backend_dev_t d) { GGML_UNUSED(d); return "OSH26_Vulkan"; }
static const char * osh26_vk_device_get_description(ggml_backend_dev_t d) { GGML_UNUSED(d); return "OSH26 custom Vulkan"; }
static void osh26_vk_device_get_memory(ggml_backend_dev_t d, size_t * f, size_t * t) { GGML_UNUSED(d); *f = *t = 1ULL<<30; }
static enum ggml_backend_dev_type osh26_vk_device_get_type(ggml_backend_dev_t d) { GGML_UNUSED(d); return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void osh26_vk_device_get_props(ggml_backend_dev_t d, struct ggml_backend_dev_props * p) {
    p->name = "OSH26 Vulkan"; p->description = "GPU"; p->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    p->memory_free = p->memory_total = 1ULL<<30;
    p->caps.async = false; p->caps.host_buffer = false;
    p->caps.buffer_from_host_ptr = false; p->caps.events = false;
}
static ggml_backend_buffer_type_t osh26_vk_device_get_buffer_type(ggml_backend_dev_t d) { GGML_UNUSED(d); return &g_buft; }
static bool osh26_vk_device_supports_op(ggml_backend_dev_t d, const struct ggml_tensor * op) {
    GGML_UNUSED(d); if (!g_ready) return false;
    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: return true;
        case GGML_OP_MUL_MAT: return op_is_supported_mul_mat(op);
        case GGML_OP_RMS_NORM: return false;
        default: return false;
    }
}
static bool osh26_vk_device_supports_buft(ggml_backend_dev_t d, ggml_backend_buffer_type_t b) { GGML_UNUSED(d); return is_osh26_vk_buft(b); }
static bool osh26_vk_device_offload_op(ggml_backend_dev_t d, const struct ggml_tensor * op) {
    GGML_UNUSED(d);
    if (!g_ready) return false;
    if (op->op == GGML_OP_MUL_MAT && op_is_supported_mul_mat(op)) return true;
    if (op->src[0] == NULL && tensor_is_plain_floatish(op)) return true;
    return false;
}
static const char * osh26_vk_reg_get_name(ggml_backend_reg_t r) { GGML_UNUSED(r); return "OSH26_Vulkan"; }
static size_t osh26_vk_reg_get_device_count(ggml_backend_reg_t r) { GGML_UNUSED(r); return 1; }
static ggml_backend_dev_t osh26_vk_reg_get_device(ggml_backend_reg_t r, size_t i) { GGML_UNUSED(r); GGML_ASSERT(i == 0); return &g_dev; }

static enum ggml_status run_command(VkCommandBuffer cb) {
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(g_device, &fci, NULL, &fence) != VK_SUCCESS) return GGML_STATUS_FAILED;
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
    VkResult res = vkQueueSubmit(g_queue, 1, &si, fence);
    if (res == VK_SUCCESS) res = vkWaitForFences(g_device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(g_device, fence, NULL);
    return res == VK_SUCCESS ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static bool create_shader(const unsigned char * data, size_t len, VkShaderModule * shader) {
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                       .codeSize = len, .pCode = (const uint32_t *)data };
    return vkCreateShaderModule(g_device, &smci, NULL, shader) == VK_SUCCESS;
}

static bool create_static_resources(void) {
    VkCommandPoolCreateInfo cmd_pool_ci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_queue_family };
    if (vkCreateCommandPool(g_device, &cmd_pool_ci, NULL, &g_command_pool) != VK_SUCCESS) return false;

    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096 };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, .maxSets = 1024, .poolSizeCount = 1, .pPoolSizes = &dps };
    if (vkCreateDescriptorPool(g_device, &dpci, NULL, &g_descriptor_pool) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding mm_bindings[3] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo mm_dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = mm_bindings };
    if (vkCreateDescriptorSetLayout(g_device, &mm_dslci, NULL, &g_mul_mat_dsl) != VK_SUCCESS) return false;

    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
    VkPipelineLayoutCreateInfo mm_plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &g_mul_mat_dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    if (vkCreatePipelineLayout(g_device, &mm_plci, NULL, &g_mul_mat_layout) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding rms_bindings[2] = {
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo rms_dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = rms_bindings };
    if (vkCreateDescriptorSetLayout(g_device, &rms_dslci, NULL, &g_rms_norm_dsl) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo rms_plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &g_rms_norm_dsl };
    if (vkCreatePipelineLayout(g_device, &rms_plci, NULL, &g_rms_norm_layout) != VK_SUCCESS) return false;

    if (!create_shader(examples_android_osh26_app_src_main_cpp_mulmat_ggml_spv, examples_android_osh26_app_src_main_cpp_mulmat_ggml_spv_len, &g_mul_mat_shader) ||
        !create_shader(examples_android_osh26_app_src_main_cpp_mulmat_tiled_spv, examples_android_osh26_app_src_main_cpp_mulmat_tiled_spv_len, &g_mul_mat_tiled_shader) ||
        !create_shader(examples_android_osh26_app_src_main_cpp_mulmat_reduce_spv, examples_android_osh26_app_src_main_cpp_mulmat_reduce_spv_len, &g_mul_mat_reduce_shader) ||
        !create_shader(_tmp_rms_norm_test_spv, _tmp_rms_norm_test_spv_len, &g_rms_norm_shader))
        return false;

    VkComputePipelineCreateInfo mm_cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = g_mul_mat_shader, .pName = "main" },
        .layout = g_mul_mat_layout };
    if (vkCreateComputePipelines(g_device, NULL, 1, &mm_cpci, NULL, &g_mul_mat_pipeline) != VK_SUCCESS) return false;

    {
        VkPipelineLayoutCreateInfo tiled_plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &g_mul_mat_dsl, .pushConstantRangeCount = 1,
            .pPushConstantRanges = &(VkPushConstantRange){ VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 } };
        if (vkCreatePipelineLayout(g_device, &tiled_plci, NULL, &g_mul_mat_tiled_layout) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo tiled_cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = g_mul_mat_tiled_shader, .pName = "main" },
            .layout = g_mul_mat_tiled_layout };
        if (vkCreateComputePipelines(g_device, NULL, 1, &tiled_cpci, NULL, &g_mul_mat_tiled_pipeline) != VK_SUCCESS) return false;

        VkComputePipelineCreateInfo reduce_cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = g_mul_mat_reduce_shader, .pName = "main" },
            .layout = g_mul_mat_tiled_layout };
        if (vkCreateComputePipelines(g_device, NULL, 1, &reduce_cpci, NULL, &g_mul_mat_reduce_pipeline) != VK_SUCCESS) return false;
    }

    const uint32_t rms_ncols = 1024;
    VkSpecializationMapEntry spec_entry = { 0, 0, sizeof(rms_ncols) };
    VkSpecializationInfo spec_info = { 1, &spec_entry, sizeof(rms_ncols), &rms_ncols };
    VkComputePipelineCreateInfo rms_cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = g_rms_norm_shader, .pName = "main",
                   .pSpecializationInfo = &spec_info },
        .layout = g_rms_norm_layout };
    if (vkCreateComputePipelines(g_device, NULL, 1, &rms_cpci, NULL, &g_rms_norm_pipeline) != VK_SUCCESS) return false;
    return true;
}

static bool load_vulkan_symbols(void) {
    void * lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) { lib = dlopen("/system/lib64/libvulkan.so", RTLD_NOW | RTLD_LOCAL); }
    if (!lib) { LOGE("dlopen libvulkan failed: %s", dlerror()); return false; }
    return true;
}

static bool init_vulkan(void) {
    if (g_ready) return true;
    if (!load_vulkan_symbols()) return false;

    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "OSH26",
                              .applicationVersion = 1, .pEngineName = "OSH26Vk", .engineVersion = 1,
                              .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    CHECK_VK(vkCreateInstance(&ici, NULL, &g_instance));

    uint32_t n_devices = 0;
    CHECK_VK(vkEnumeratePhysicalDevices(g_instance, &n_devices, NULL));
    if (n_devices == 0) { LOGE("no Vulkan physical devices"); return false; }
    VkPhysicalDevice * devices = calloc(n_devices, sizeof(*devices));
    if (!devices) return false;
    vkEnumeratePhysicalDevices(g_instance, &n_devices, devices);
    g_physical_device = devices[0];
    free(devices);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_physical_device, &props);
    vkGetPhysicalDeviceMemoryProperties(g_physical_device, &g_mem_props);
    LOGI("using Vulkan device: %s", props.deviceName);

    uint32_t n_queues = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_physical_device, &n_queues, NULL);
    VkQueueFamilyProperties * queues = calloc(n_queues, sizeof(*queues));
    if (!queues) return false;
    vkGetPhysicalDeviceQueueFamilyProperties(g_physical_device, &n_queues, queues);
    bool found = false;
    for (uint32_t i = 0; i < n_queues; ++i) {
        if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_queue_family = i; found = true; break; }
    }
    free(queues);
    if (!found) { LOGE("no Vulkan compute queue"); return false; }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo dqci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = g_queue_family, .queueCount = 1, .pQueuePriorities = &priority };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &dqci };
    CHECK_VK(vkCreateDevice(g_physical_device, &dci, NULL, &g_device));
    vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);

    if (!create_static_resources()) return false;

    g_ready = true;
    LOGI("custom Vulkan backend ready");
    return true;
}

// ============= GPU MUL_MAT HOOK (with mutex + persistent temp buffers) =============

struct ggml_compute_params;
bool (*g_osh26_vk_hook_mul_mat)(const struct ggml_compute_params *, struct ggml_tensor *) = NULL;
static pthread_mutex_t g_hook_mutex = PTHREAD_MUTEX_INITIALIZER;

static struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    void * mapped;
    VkDeviceSize size;
} g_temp_buf[3];

static bool temp_buf_ensure(int i, VkDeviceSize bytes) {
    if (g_temp_buf[i].size >= bytes) return true;
    if (g_temp_buf[i].mapped) vkUnmapMemory(g_device, g_temp_buf[i].mem);
    if (g_temp_buf[i].mem)   vkFreeMemory(g_device, g_temp_buf[i].mem, NULL);
    if (g_temp_buf[i].buf)   vkDestroyBuffer(g_device, g_temp_buf[i].buf, NULL);
    memset(&g_temp_buf[i], 0, sizeof(g_temp_buf[i]));

    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    if (vkCreateBuffer(g_device, &bci, NULL, &g_temp_buf[i].buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_device, g_temp_buf[i].buf, &mr);
    uint32_t mt = find_memory_type(mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) return false;
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mt };
    if (vkAllocateMemory(g_device, &mai, NULL, &g_temp_buf[i].mem) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(g_device, g_temp_buf[i].buf, g_temp_buf[i].mem, 0) != VK_SUCCESS) return false;
    if (vkMapMemory(g_device, g_temp_buf[i].mem, 0, bytes, 0, &g_temp_buf[i].mapped) != VK_SUCCESS) return false;
    g_temp_buf[i].size = bytes;
    return true;
}

static void copy_in(float * dst, const struct ggml_tensor * src, int64_t rows, int64_t cols) {
    if (src->type == GGML_TYPE_F32 && ggml_is_contiguous(src)) {
        memcpy(dst, src->data, (size_t)(rows * cols * sizeof(float)));
    } else if (src->type == GGML_TYPE_F16 && ggml_is_contiguous(src)) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)src->data, dst, rows * cols);
    } else {
        for (int64_t r = 0; r < rows; r++) {
            const float * sr = (const float *)((const uint8_t *)src->data + r * src->nb[1]);
            if (src->type == GGML_TYPE_F32)
                memcpy(dst + r * cols, sr, cols * sizeof(float));
            else
                for (int64_t c = 0; c < cols; c++) dst[r * cols + c] = sr[c];
        }
    }
}

static void copy_out(struct ggml_tensor * dst, const float * src, int64_t rows, int64_t cols) {
    if (ggml_is_contiguous(dst)) {
        memcpy(dst->data, src, (size_t)(rows * cols * sizeof(float)));
    } else {
        for (int64_t r = 0; r < rows; r++) {
            float * dr = (float *)((uint8_t *)dst->data + r * dst->nb[1]);
            memcpy(dr, src + r * cols, cols * sizeof(float));
        }
    }
}

static bool osh26_vk_hook_mul_mat_impl(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    (void)params;
    if (!g_ready) return false;
    if (dst->op != GGML_OP_MUL_MAT) return false;
    struct ggml_tensor * a = dst->src[0], * b = dst->src[1];
    if (!a || !b) return false;
    if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16) return false;
    if (b->type != GGML_TYPE_F32) return false;
    if (dst->type != GGML_TYPE_F32) return false;
    int64_t M = dst->ne[1], N = dst->ne[0], K = a->ne[0];
    if (M <= 0 || N <= 0 || K <= 0) return false;

    bool ok = false;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    pthread_mutex_lock(&g_hook_mutex);

    VkDeviceSize Asz = (VkDeviceSize)(M * K * sizeof(float));
    VkDeviceSize Bsz = (VkDeviceSize)(N * K * sizeof(float));
    VkDeviceSize Csz = (VkDeviceSize)(M * N * sizeof(float));

    if (!temp_buf_ensure(0, Asz) || !temp_buf_ensure(1, Bsz) || !temp_buf_ensure(2, Csz)) goto done;

    copy_in((float *)g_temp_buf[0].mapped, a, M, K);
    memcpy(g_temp_buf[1].mapped, b->data, (size_t)Bsz);

    {
        VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            NULL, g_descriptor_pool, 1, &g_mul_mat_dsl };
        if (vkAllocateDescriptorSets(g_device, &dsai, &ds) != VK_SUCCESS) goto done;
    }
    {
        VkDescriptorBufferInfo dbi[3] = {
            { g_temp_buf[1].buf, 0, VK_WHOLE_SIZE },
            { g_temp_buf[0].buf, 0, VK_WHOLE_SIZE },
            { g_temp_buf[2].buf, 0, VK_WHOLE_SIZE },
        };
        VkWriteDescriptorSet w[3]; memset(w, 0, sizeof(w));
        for (int j = 0; j < 3; j++) {
            w[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[j].dstSet = ds; w[j].dstBinding = j; w[j].descriptorCount = 1;
            w[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[j].pBufferInfo = &dbi[j];
        }
        vkUpdateDescriptorSets(g_device, 3, w, 0, NULL);
    }
    {
        VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            NULL, g_command_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
        if (vkAllocateCommandBuffers(g_device, &ai, &cb) != VK_SUCCESS) goto done;
    }

    {
        uint32_t pc[4] = { (uint32_t)M, (uint32_t)N, (uint32_t)K, (uint32_t)K };
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
        vkBeginCommandBuffer(cb, &bi);
        if (M <= 4) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_mul_mat_reduce_pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_mul_mat_tiled_layout, 0, 1, &ds, 0, NULL);
            vkCmdPushConstants(cb, g_mul_mat_tiled_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatch(cb, (uint32_t)N, (uint32_t)M, 1);
        } else {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_mul_mat_tiled_pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_mul_mat_tiled_layout, 0, 1, &ds, 0, NULL);
            vkCmdPushConstants(cb, g_mul_mat_tiled_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatch(cb, ((uint32_t)N + 7) / 8, ((uint32_t)M + 7) / 8, 1);
        }
        vkEndCommandBuffer(cb);
    }

    {
        VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
        vkCreateFence(g_device, &fci, NULL, &fence);
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL };
        if (vkQueueSubmit(g_queue, 1, &si, fence) == VK_SUCCESS)
            vkWaitForFences(g_device, 1, &fence, VK_TRUE, UINT64_MAX);
    }

    copy_out(dst, (const float *)g_temp_buf[2].mapped, M, N);
    ok = true;

done:
    if (fence != VK_NULL_HANDLE) vkDestroyFence(g_device, fence, NULL);
    if (cb != VK_NULL_HANDLE) vkFreeCommandBuffers(g_device, g_command_pool, 1, &cb);
    if (ds != VK_NULL_HANDLE) vkFreeDescriptorSets(g_device, g_descriptor_pool, 1, &ds);
    pthread_mutex_unlock(&g_hook_mutex);
    return ok;
}

// ============= Public API =============

int osh26_vk_init(void) {
    if (g_registered) return g_ready ? 0 : -1;
    if (!init_vulkan()) { LOGE("custom Vulkan init failed"); return -1; }

    g_osh26_vk_hook_mul_mat = osh26_vk_hook_mul_mat_impl;

    g_buft.iface.get_name = osh26_vk_buft_get_name;
    g_buft.iface.alloc_buffer = osh26_vk_buft_alloc_buffer;
    g_buft.iface.get_alignment = osh26_vk_buft_get_alignment;
    g_buft.iface.get_max_size = NULL;
    g_buft.iface.get_alloc_size = osh26_vk_buft_get_alloc_size;
    g_buft.iface.is_host = osh26_vk_buft_is_host;
    g_buft.device = &g_dev;
    g_buft.context = NULL;

    static const struct ggml_backend_device_i dev_iface = {
        osh26_vk_device_get_name, osh26_vk_device_get_description,
        osh26_vk_device_get_memory, osh26_vk_device_get_type, osh26_vk_device_get_props,
        osh26_vk_device_init_backend, osh26_vk_device_get_buffer_type,
        NULL, NULL, osh26_vk_device_supports_op, osh26_vk_device_supports_buft,
        osh26_vk_device_offload_op, NULL, NULL, NULL,
    };
    static const struct ggml_backend_reg_i reg_iface = {
        osh26_vk_reg_get_name, osh26_vk_reg_get_device_count, osh26_vk_reg_get_device, NULL,
    };

    g_reg.api_version = GGML_BACKEND_API_VERSION;
    g_reg.iface = reg_iface;
    g_reg.context = NULL;
    g_dev.iface = dev_iface;
    g_dev.reg = &g_reg;
    g_dev.context = NULL;
    ggml_backend_register(&g_reg);
    g_registered = true;
    LOGI("custom Vulkan backend registered");
    return 0;
}

int osh26_vk_get_stats(struct osh26_vk_stats * out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    out->ready = g_ready;
    out->registered = g_registered;
    return 0;
}
