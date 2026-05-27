/* Vulkan buffer wrapper — adapted from MNN VulkanBuffer */
#include "vk_buffer.h"
#include <android/log.h>
#include <stdlib.h>
#include <string.h>
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "VKBUF", __VA_ARGS__)

/* Extern: physical device is set by osh26_vk_gpu_init */
extern VkPhysicalDevice g_vk_phy;
extern VkDevice         g_vk_dev;
extern VkPhysicalDeviceMemoryProperties g_vk_mp;

static uint32_t find_mem_type(uint32_t type_bits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < g_vk_mp.memoryTypeCount; i++)
        if ((type_bits & (1u << i)) && (g_vk_mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return UINT32_MAX;
}

bool vk_buf_alloc_ex(VkBuf *b, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_flags) {
    memset(b, 0, sizeof(*b));
    b->size = size;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0, size, usage,
                              VK_SHARING_MODE_EXCLUSIVE, 0, 0};
    if (vkCreateBuffer(g_vk_dev, &bci, 0, &b->buf) != VK_SUCCESS) { LOGE("vkCreateBuffer"); return false; }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_vk_dev, b->buf, &mr);
    uint32_t mt = find_mem_type(mr.memoryTypeBits, mem_flags);
    if (mt == UINT32_MAX) { vkDestroyBuffer(g_vk_dev, b->buf, 0); LOGE("no mem type"); return false; }
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, mr.size, mt};
    if (vkAllocateMemory(g_vk_dev, &mai, 0, &b->mem) != VK_SUCCESS) { vkDestroyBuffer(g_vk_dev, b->buf, 0); LOGE("vkAllocateMemory"); return false; }
    vkBindBufferMemory(g_vk_dev, b->buf, b->mem, 0);
    b->coherent = (mem_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    if (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) vk_buf_map(b);
    return true;
}

bool vk_buf_alloc(VkBuf *b, VkDeviceSize size) {
    return vk_buf_alloc_ex(b, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void vk_buf_free(VkBuf *b) {
    if (b->mapped) { vkUnmapMemory(g_vk_dev, b->mem); b->mapped = NULL; }
    if (b->mem)    { vkFreeMemory(g_vk_dev, b->mem, 0); b->mem = VK_NULL_HANDLE; }
    if (b->buf)    { vkDestroyBuffer(g_vk_dev, b->buf, 0); b->buf = VK_NULL_HANDLE; }
    b->size = 0;
}

void *vk_buf_map(VkBuf *b) {
    if (b->mapped) return b->mapped;
    if (vkMapMemory(g_vk_dev, b->mem, 0, b->size, 0, &b->mapped) != VK_SUCCESS) return NULL;
    return b->mapped;
}

void vk_buf_unmap(VkBuf *b) { if (b->mapped) { vkUnmapMemory(g_vk_dev, b->mem); b->mapped = NULL; } }

void vk_buf_flush(VkBuf *b, VkDeviceSize off, VkDeviceSize len) {
    if (b->coherent) return;
    VkMappedMemoryRange r = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, 0, b->mem, off, len ? len : b->size};
    vkFlushMappedMemoryRanges(g_vk_dev, 1, &r);
}

void vk_buf_invalidate(VkBuf *b, VkDeviceSize off, VkDeviceSize len) {
    if (b->coherent) return;
    VkMappedMemoryRange r = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, 0, b->mem, off, len ? len : b->size};
    vkInvalidateMappedMemoryRanges(g_vk_dev, 1, &r);
}
