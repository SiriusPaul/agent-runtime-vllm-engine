/* Minimal Vulkan test: device creation + buffer upload/download + basic compute */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include "test_shader.h"

static PFN_vkGetInstanceProcAddr pGI;

#define TRY(n, inst) PFN_##n n = (PFN_##n)pGI((VkInstance)inst, #n); if (!n) { __android_log_print(ANDROID_LOG_ERROR, "TestVk", "%s not found", #n); return -1; }
#define IGLOBAL(n) PFN_##n n = (PFN_##n)pGI(NULL, #n); if (!n) { __android_log_print(ANDROID_LOG_ERROR, "TestVk", "%s not found(global)", #n); return -1; }

static uint32_t mem_type(VkPhysicalDeviceMemoryProperties *mp, uint32_t tb, VkMemoryPropertyFlags f) {
    for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
        if ((tb & (1u<<i)) && (mp->memoryTypes[i].propertyFlags & f) == f) return i;
    return UINT32_MAX;
}

int run_vulkan_test(void) {
    __android_log_print(ANDROID_LOG_INFO, "TestVk", "=== START ===");

    void *lib = dlopen("/system/lib64/libvulkan.so", RTLD_NOW);
    if (!lib) { __android_log_print(ANDROID_LOG_ERROR, "TestVk", "dlopen: %s", dlerror()); return -1; }
    __android_log_print(ANDROID_LOG_INFO, "TestVk", "libvulkan loaded");

    pGI = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    if (!pGI) { __android_log_print(ANDROID_LOG_ERROR, "TestVk", "no GIProcAddr"); return -1; }

    /* Step 1: Instance (only vkCreateInstance can be loaded with NULL) */
    IGLOBAL(vkCreateInstance);
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "vktest", 1, "", 0, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &ai, 0, NULL, 0, NULL};
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { __android_log_print(ANDROID_LOG_ERROR, "TestVk", "createInst"); return -2; }
    __android_log_print(ANDROID_LOG_INFO, "TestVk", "Instance OK");

    /* Now load rest with instance handle */
    TRY(vkEnumeratePhysicalDevices, inst); TRY(vkGetPhysicalDeviceProperties, inst);

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(inst, &nd, NULL);
    VkPhysicalDevice *pds = calloc(nd, sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(inst, &nd, pds);
    VkPhysicalDeviceProperties pdp;
    vkGetPhysicalDeviceProperties(pds[0], &pdp);
    VkPhysicalDevice pd = pds[0];
    __android_log_print(ANDROID_LOG_INFO, "TestVk", "Device: %s v%d.%d", pdp.deviceName,
        VK_VERSION_MAJOR(pdp.apiVersion), VK_VERSION_MINOR(pdp.apiVersion));
    free(pds);

    /* Step 2: Device + queue */
    TRY(vkGetPhysicalDeviceQueueFamilyProperties, inst);
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
    VkQueueFamilyProperties *qps = calloc(qn, sizeof(*qps));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qps);
    int qfi = -1;
    for (uint32_t i = 0; i < qn; i++) if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = (int)i; break; }
    if (qfi < 0) { __android_log_print(ANDROID_LOG_ERROR, "TestVk", "no compute q"); free(qps); return -3; }
    __android_log_print(ANDROID_LOG_INFO, "TestVk", "qfi=%d", qfi);
    free(qps);

    TRY(vkCreateDevice, inst); TRY(vkGetDeviceQueue, inst);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dq = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, (uint32_t)qfi, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &dq, 0, NULL, 0, NULL};
    VkDevice dev;
    if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { __android_log_print(ANDROID_LOG_ERROR, "TestVk", "createDev"); return -4; }
    __android_log_print(ANDROID_LOG_INFO, "TestVk", "Device OK");

    /* Step 3: Memory + buffer test (no compute shader) */
    TRY(vkGetPhysicalDeviceMemoryProperties, inst); TRY(vkCreateBuffer, inst); TRY(vkGetBufferMemoryRequirements, inst);
    TRY(vkAllocateMemory, inst); TRY(vkBindBufferMemory, inst); TRY(vkMapMemory, inst); TRY(vkUnmapMemory, inst);
    VkPhysicalDeviceMemoryProperties memp;
    vkGetPhysicalDeviceMemoryProperties(pd, &memp);

    const VkDeviceSize sz = 256;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, sz,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkBuffer tb;
    vkCreateBuffer(dev, &bci, NULL, &tb);
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, tb, &mr);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size,
        mem_type(&memp, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VkDeviceMemory tm;
    vkAllocateMemory(dev, &mai, NULL, &tm);
    vkBindBufferMemory(dev, tb, tm, 0);

    /* Write test pattern to GPU and read back */
    float *ptr;
    vkMapMemory(dev, tm, 0, sz, 0, (void**)&ptr);
    for (int i = 0; i < 16; i++) ptr[i] = (float)i;
    vkUnmapMemory(dev, tm);
    vkMapMemory(dev, tm, 0, sz, 0, (void**)&ptr);
    int mem_err = 0;
    for (int i = 0; i < 16; i++) if (ptr[i] != (float)i) { mem_err++; __android_log_print(ANDROID_LOG_ERROR, "TestVk", "MEM[%d]=%f!=%d", i, ptr[i], i); }
    if (!mem_err) __android_log_print(ANDROID_LOG_INFO, "TestVk", "Buffer R/W: PASS");
    else __android_log_print(ANDROID_LOG_ERROR, "TestVk", "Buffer R/W: %d errors", mem_err);
    vkUnmapMemory(dev, tm);

    /* Step 4: Compute pipeline test */
    TRY(vkCreateShaderModule, inst); TRY(vkCreateDescriptorSetLayout, inst); TRY(vkCreatePipelineLayout, inst);
    TRY(vkCreateComputePipelines, inst); TRY(vkCreateDescriptorPool, inst); TRY(vkAllocateDescriptorSets, inst);
    TRY(vkUpdateDescriptorSets, inst); TRY(vkCreateCommandPool, inst); TRY(vkAllocateCommandBuffers, inst);
    TRY(vkBeginCommandBuffer, inst); TRY(vkEndCommandBuffer, inst); TRY(vkCmdBindPipeline, inst);
    TRY(vkCmdBindDescriptorSets, inst); TRY(vkCmdDispatch, inst); TRY(vkQueueSubmit, inst);
    TRY(vkCreateFence, inst); TRY(vkWaitForFences, inst); TRY(vkFlushMappedMemoryRanges, inst);

    /* Create shader from SPIR-V */
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0,
        add_spv_len, (const uint32_t*)add_spv};
    VkShaderModule shader;
    if (vkCreateShaderModule(dev, &sm, NULL, &shader) != VK_SUCCESS)
        { __android_log_print(ANDROID_LOG_ERROR, "TestVk", "shader create fail"); return -5; }

    /* 3 buffers: A, B, C */
    const uint32_t N = 1024;
    const VkDeviceSize Nsz = N * sizeof(float);
    VkBuffer bufs[3]; VkDeviceMemory mems[3];
    for (int i = 0; i < 3; i++) {
        vkCreateBuffer(dev, &(VkBufferCreateInfo){VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, Nsz,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_SHARING_MODE_EXCLUSIVE, 0, NULL}, NULL, &bufs[i]);
        vkGetBufferMemoryRequirements(dev, bufs[i], &mr);
        vkAllocateMemory(dev, &(VkMemoryAllocateInfo){VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size,
            mem_type(&memp, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)}, NULL, &mems[i]);
        vkBindBufferMemory(dev, bufs[i], mems[i], 0);
    }

    /* Fill A[i]=i, B[i]=i*2 */
    float *a, *b;
    vkMapMemory(dev, mems[0], 0, Nsz, 0, (void**)&a);
    vkMapMemory(dev, mems[1], 0, Nsz, 0, (void**)&b);
    for (uint32_t i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(i*2.0f); }
    vkUnmapMemory(dev, mems[0]); vkUnmapMemory(dev, mems[1]);

    /* Pipeline */
    VkDescriptorSetLayoutBinding bnd[3];
    for (int i = 0; i < 3; i++) bnd[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayout dsl;
    vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, bnd}, NULL, &dsl);
    VkPipelineLayout pl;
    vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 0, NULL}, NULL, &pl);
    VkPipeline pipe;
    VkResult pr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", NULL},
        pl, VK_NULL_HANDLE, -1}, NULL, &pipe);
    __android_log_print(ANDROID_LOG_INFO, "TestVk", "Pipeline: %d", pr);

    /* Descriptor set */
    VkDescriptorPool dp;
    vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1,
        &(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}}, NULL, &dp);
    VkDescriptorSet ds;
    vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl}, &ds);

    VkDescriptorBufferInfo dbi[3];
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        dbi[i] = (VkDescriptorBufferInfo){bufs[i], 0, Nsz};
        wds[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi[i], NULL};
    }
    vkUpdateDescriptorSets(dev, 3, wds, 0, NULL);

    /* Command buffer */
    VkCommandPool cp;
    vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, (uint32_t)qfi}, NULL, &cp);
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1}, &cb);

    vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, 0, NULL});
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, N/64, 1, 1);
    vkEndCommandBuffer(cb);

    /* Submit */
    VkFence f;
    vkCreateFence(dev, &(VkFenceCreateInfo){VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0}, NULL, &f);
    VkQueue q;
    vkGetDeviceQueue(dev, qfi, 0, &q);
    __android_log_print(ANDROID_LOG_INFO, "TestVk", "Submit: %d", vkQueueSubmit(q, 1,
        &(VkSubmitInfo){VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL}, f));
    __android_log_print(ANDROID_LOG_INFO, "TestVk", "WaitFence: %d", vkWaitForFences(dev, 1, &f, VK_TRUE, UINT64_MAX));

    /* Verify */
    float *c;
    vkMapMemory(dev, mems[2], 0, Nsz, 0, (void**)&c);
    int errs = 0;
    for (uint32_t i = 0; i < N; i++) {
        float ex = (float)i + (float)(i*2);
        if (c[i] != ex) { if (errs < 8) __android_log_print(ANDROID_LOG_ERROR, "TestVk", "FAIL[%d]: exp %.0f got %.6f", i, ex, c[i]); errs++; }
    }
    if (!errs) __android_log_print(ANDROID_LOG_INFO, "TestVk", "COMPUTE: PASS (%d ok)", N);
    else __android_log_print(ANDROID_LOG_ERROR, "TestVk", "COMPUTE: %d/%d ERRORS", errs, N);
    vkUnmapMemory(dev, mems[2]);

    // Cleanup ptr casts
    TRY(vkDestroyFence, inst); TRY(vkDestroyCommandPool, inst); TRY(vkDestroyDescriptorPool, inst);
    TRY(vkDestroyDescriptorSetLayout, inst); TRY(vkDestroyPipelineLayout, inst); TRY(vkDestroyPipeline, inst);
    TRY(vkDestroyShaderModule, inst); TRY(vkDestroyBuffer, inst); TRY(vkFreeMemory, inst);
    vkDestroyFence(dev, f, NULL);
    vkDestroyCommandPool(dev, cp, NULL);
    vkDestroyDescriptorPool(dev, dp, NULL);
    vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL);
    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyShaderModule(dev, shader, NULL);
    for (int i = 0; i < 3; i++) { vkDestroyBuffer(dev, bufs[i], NULL); vkFreeMemory(dev, mems[i], NULL); }
    vkDestroyBuffer(dev, tb, NULL); vkFreeMemory(dev, tm, NULL);

    __android_log_print(ANDROID_LOG_INFO, "TestVk", "=== DONE ===");
    return 0;
}
