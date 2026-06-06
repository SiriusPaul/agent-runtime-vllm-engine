/* Vulkan function pointer loader — adapted from MNN (orig. Google Inc, Apache 2.0) */
#include "vulkan_wrapper.h"
#include <dlfcn.h>
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "VKWRAP", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "VKWRAP", __VA_ARGS__)

#define LOAD(fn) vk##fn = reinterpret_cast<PFN_vk##fn>(dlsym(lib, "vk" #fn))

int InitVulkan(void) {
    static int done = 0;
    if (done) return 1;
    void *lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) { LOGE("dlopen libvulkan.so: %s", dlerror()); return 0; }
    /* Instance */
    LOAD(CreateInstance); LOAD(DestroyInstance); LOAD(EnumeratePhysicalDevices);
    LOAD(GetPhysicalDeviceFeatures); LOAD(GetPhysicalDeviceFeatures2);
    LOAD(GetPhysicalDeviceFormatProperties); LOAD(GetPhysicalDeviceProperties);
    LOAD(GetPhysicalDeviceQueueFamilyProperties); LOAD(GetPhysicalDeviceMemoryProperties);
    LOAD(GetInstanceProcAddr); LOAD(GetDeviceProcAddr);
    LOAD(EnumerateDeviceExtensionProperties);
    LOAD(CreateDevice); LOAD(DestroyDevice); LOAD(GetDeviceQueue);
    LOAD(QueueSubmit); LOAD(QueueWaitIdle); LOAD(DeviceWaitIdle);
    /* Memory */
    LOAD(AllocateMemory); LOAD(FreeMemory); LOAD(MapMemory); LOAD(UnmapMemory);
    LOAD(FlushMappedMemoryRanges); LOAD(InvalidateMappedMemoryRanges);
    LOAD(BindBufferMemory); LOAD(GetBufferMemoryRequirements);
    /* Buffer */
    LOAD(CreateBuffer); LOAD(DestroyBuffer);
    /* Fence */
    LOAD(CreateFence); LOAD(DestroyFence); LOAD(WaitForFences); LOAD(ResetFences);
    /* Shader & Pipeline */
    LOAD(CreateShaderModule); LOAD(DestroyShaderModule);
    LOAD(CreateComputePipelines); LOAD(DestroyPipeline);
    LOAD(CreatePipelineLayout); LOAD(DestroyPipelineLayout);
    /* Descriptor */
    LOAD(CreateDescriptorSetLayout); LOAD(DestroyDescriptorSetLayout);
    LOAD(CreateDescriptorPool); LOAD(DestroyDescriptorPool);
    LOAD(ResetDescriptorPool);
    LOAD(AllocateDescriptorSets); LOAD(FreeDescriptorSets); LOAD(UpdateDescriptorSets);
    /* Command */
    LOAD(CreateCommandPool); LOAD(DestroyCommandPool); LOAD(ResetCommandPool);
    LOAD(AllocateCommandBuffers); LOAD(FreeCommandBuffers);
    LOAD(BeginCommandBuffer); LOAD(EndCommandBuffer);
    LOAD(CmdBindPipeline); LOAD(CmdBindDescriptorSets);
    LOAD(CmdDispatch); LOAD(CmdPipelineBarrier); LOAD(CmdPushConstants);
    LOAD(CmdCopyBuffer);
    LOGI("Vulkan functions loaded OK");
    done = 1; return 1;
}

/* Function pointer definitions */
PFN_vkCreateInstance vkCreateInstance;
PFN_vkDestroyInstance vkDestroyInstance;
PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;
PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures;
PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2;
PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties;
PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr;
PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties;
PFN_vkCreateDevice vkCreateDevice;
PFN_vkDestroyDevice vkDestroyDevice;
PFN_vkGetDeviceQueue vkGetDeviceQueue;
PFN_vkQueueSubmit vkQueueSubmit;
PFN_vkQueueWaitIdle vkQueueWaitIdle;
PFN_vkDeviceWaitIdle vkDeviceWaitIdle;
PFN_vkAllocateMemory vkAllocateMemory;
PFN_vkFreeMemory vkFreeMemory;
PFN_vkMapMemory vkMapMemory;
PFN_vkUnmapMemory vkUnmapMemory;
PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges;
PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges;
PFN_vkBindBufferMemory vkBindBufferMemory;
PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
PFN_vkCreateBuffer vkCreateBuffer;
PFN_vkDestroyBuffer vkDestroyBuffer;
PFN_vkCreateFence vkCreateFence;
PFN_vkDestroyFence vkDestroyFence;
PFN_vkWaitForFences vkWaitForFences;
PFN_vkResetFences vkResetFences;
PFN_vkCreateShaderModule vkCreateShaderModule;
PFN_vkDestroyShaderModule vkDestroyShaderModule;
PFN_vkCreateComputePipelines vkCreateComputePipelines;
PFN_vkDestroyPipeline vkDestroyPipeline;
PFN_vkCreatePipelineLayout vkCreatePipelineLayout;
PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout;
PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout;
PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout;
PFN_vkCreateDescriptorPool vkCreateDescriptorPool;
PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool;
PFN_vkResetDescriptorPool vkResetDescriptorPool;
PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets;
PFN_vkFreeDescriptorSets vkFreeDescriptorSets;
PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets;
PFN_vkCreateCommandPool vkCreateCommandPool;
PFN_vkDestroyCommandPool vkDestroyCommandPool;
PFN_vkResetCommandPool vkResetCommandPool;
PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
PFN_vkFreeCommandBuffers vkFreeCommandBuffers;
PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
PFN_vkEndCommandBuffer vkEndCommandBuffer;
PFN_vkCmdBindPipeline vkCmdBindPipeline;
PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets;
PFN_vkCmdDispatch vkCmdDispatch;
PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
PFN_vkCmdPushConstants vkCmdPushConstants;
PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
