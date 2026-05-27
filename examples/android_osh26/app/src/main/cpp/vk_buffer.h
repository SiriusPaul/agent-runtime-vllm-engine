/* Minimal Vulkan buffer wrapper — adapted from MNN VulkanBuffer */
#ifndef OSH26_VK_BUFFER_H
#define OSH26_VK_BUFFER_H
#include "vk_wrapper/vulkan_wrapper.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
    size_t         size;
    void          *mapped;   /* persistently mapped if host-visible */
    bool           coherent; /* HOST_COHERENT → no explicit flush needed */
} VkBuf;

/* Allocate a device-local + host-visible buffer (STORAGE_BUFFER, zero-copy) */
bool vk_buf_alloc(VkBuf *b, VkDeviceSize size);
/* Allocate with custom usage + memory flags */
bool vk_buf_alloc_ex(VkBuf *b, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_flags);
void vk_buf_free(VkBuf *b);
/* Map for CPU access (no-op if already mapped). Returns pointer. */
void *vk_buf_map(VkBuf *b);
void  vk_buf_unmap(VkBuf *b);
/* Flush writes / invalidate reads (no-op for coherent memory) */
void  vk_buf_flush(VkBuf *b, VkDeviceSize off, VkDeviceSize len);
void  vk_buf_invalidate(VkBuf *b, VkDeviceSize off, VkDeviceSize len);
#endif
