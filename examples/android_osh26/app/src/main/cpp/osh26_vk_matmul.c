/* Minimal Vulkan MUL_MAT ggml backend. Pure C API. ~200 lines. */
#define VK_NO_PROTOTYPES

#include "ggml.h"
#include "ggml-backend.h"
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include "mulmat_test.spv.h"

#define TAG "VkMM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static VkInstance       g_inst;
static VkPhysicalDevice g_phys;
static VkDevice         g_dev;
static VkQueue          g_queue;
static uint32_t         g_qfi;
static VkPhysicalDeviceMemoryProperties g_mp;
static VkCommandPool    g_cp;
static VkDescriptorPool g_dp;
static VkShaderModule   g_shader_mm;
static VkPipelineLayout g_pl_mm;
static VkPipeline       g_pipe_mm;
static VkDescriptorSetLayout g_dsl;

#define DECL(fn) static PFN_##fn pfn_##fn
DECL(vkGetInstanceProcAddr);
DECL(vkCreateInstance); DECL(vkEnumeratePhysicalDevices);
DECL(vkGetPhysicalDeviceProperties); DECL(vkCreateDevice); DECL(vkGetDeviceQueue);
DECL(vkAllocateMemory); DECL(vkFreeMemory); DECL(vkMapMemory); DECL(vkUnmapMemory);
DECL(vkCreateBuffer); DECL(vkDestroyBuffer);
DECL(vkGetBufferMemoryRequirements); DECL(vkBindBufferMemory);
DECL(vkCreateShaderModule); DECL(vkDestroyShaderModule);
DECL(vkCreateDescriptorSetLayout); DECL(vkDestroyDescriptorSetLayout);
DECL(vkCreatePipelineLayout); DECL(vkDestroyPipelineLayout);
DECL(vkCreateComputePipelines); DECL(vkDestroyPipeline);
DECL(vkCreateDescriptorPool); DECL(vkDestroyDescriptorPool);
DECL(vkAllocateDescriptorSets); DECL(vkUpdateDescriptorSets);
DECL(vkCreateCommandPool); DECL(vkDestroyCommandPool);
DECL(vkAllocateCommandBuffers); DECL(vkFreeCommandBuffers);
DECL(vkBeginCommandBuffer); DECL(vkEndCommandBuffer);
DECL(vkCmdBindPipeline); DECL(vkCmdBindDescriptorSets); DECL(vkCmdDispatch);
DECL(vkCmdPushConstants); DECL(vkQueueSubmit); DECL(vkCreateFence);
DECL(vkDestroyFence); DECL(vkWaitForFences);
DECL(vkGetPhysicalDeviceMemoryProperties);
DECL(vkGetPhysicalDeviceQueueFamilyProperties);
#undef DECL

#define CALL(fn) pfn_##fn
#define LOAD(fn) pfn_##fn = (PFN_##fn)CALL(vkGetInstanceProcAddr)(g_inst, #fn)

static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags f) {
    for(uint32_t i=0;i<g_mp.memoryTypeCount;i++)
        if((bits&(1u<<i))&&(g_mp.memoryTypes[i].propertyFlags&f)==f) return i;
    return UINT32_MAX;
}

static void make_buf(VkDeviceSize sz, VkBuffer *b, VkDeviceMemory *m) {
    VkBufferCreateInfo bci={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,0,0,sz,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_SHARING_MODE_EXCLUSIVE,0,0};
    CALL(vkCreateBuffer)(g_dev,&bci,0,b);
    VkMemoryRequirements mr; CALL(vkGetBufferMemoryRequirements)(g_dev,*b,&mr);
    VkMemoryAllocateInfo mai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,0,mr.size,
        memtype(mr.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    CALL(vkAllocateMemory)(g_dev,&mai,0,m); CALL(vkBindBufferMemory)(g_dev,*b,*m,0);
}

int osh26_vk_matmul_init(void) {
    LOGI("=== Init ===");
    void *lib=dlopen("/system/lib64/libvulkan.so",RTLD_NOW);
    if(!lib){LOGE("dlopen:%s",dlerror());return -1;}
    CALL(vkGetInstanceProcAddr)=dlsym(lib,"vkGetInstanceProcAddr");
    if(!CALL(vkGetInstanceProcAddr)){LOGE("no pGI");return -1;}

    /* Global funcs with NULL */
    pfn_vkCreateInstance = (PFN_vkCreateInstance)CALL(vkGetInstanceProcAddr)(NULL,"vkCreateInstance");
    VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO,0,"vkmm",1,"",0,VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,0,0,&ai,0,0,0,0};
    if(CALL(vkCreateInstance)(&ici,0,&g_inst)!=VK_SUCCESS){LOGE("createInst");return -1;}

    LOAD(vkEnumeratePhysicalDevices); LOAD(vkGetPhysicalDeviceProperties);
    LOAD(vkCreateDevice); LOAD(vkGetDeviceQueue); LOAD(vkAllocateMemory); LOAD(vkFreeMemory);
    LOAD(vkMapMemory); LOAD(vkUnmapMemory); LOAD(vkCreateBuffer); LOAD(vkDestroyBuffer);
    LOAD(vkGetBufferMemoryRequirements); LOAD(vkBindBufferMemory);
    LOAD(vkCreateShaderModule); LOAD(vkDestroyShaderModule);
    LOAD(vkCreateDescriptorSetLayout); LOAD(vkDestroyDescriptorSetLayout);
    LOAD(vkCreatePipelineLayout); LOAD(vkDestroyPipelineLayout);
    LOAD(vkCreateComputePipelines); LOAD(vkDestroyPipeline);
    LOAD(vkCreateDescriptorPool); LOAD(vkDestroyDescriptorPool);
    LOAD(vkAllocateDescriptorSets); LOAD(vkUpdateDescriptorSets);
    LOAD(vkCreateCommandPool); LOAD(vkDestroyCommandPool);
    LOAD(vkAllocateCommandBuffers); LOAD(vkFreeCommandBuffers);
    LOAD(vkBeginCommandBuffer); LOAD(vkEndCommandBuffer);
    LOAD(vkCmdBindPipeline); LOAD(vkCmdBindDescriptorSets); LOAD(vkCmdDispatch); LOAD(vkCmdPushConstants);
    LOAD(vkQueueSubmit); LOAD(vkCreateFence); LOAD(vkDestroyFence); LOAD(vkWaitForFences);
    LOAD(vkGetPhysicalDeviceMemoryProperties); LOAD(vkGetPhysicalDeviceQueueFamilyProperties);

    uint32_t nd=0; CALL(vkEnumeratePhysicalDevices)(g_inst,&nd,0);
    VkPhysicalDevice *pds=calloc(nd,sizeof(VkPhysicalDevice));
    CALL(vkEnumeratePhysicalDevices)(g_inst,&nd,pds); g_phys=pds[0];
    VkPhysicalDeviceProperties pdp; CALL(vkGetPhysicalDeviceProperties)(g_phys,&pdp);
    LOGI("Device: %s v%d.%d",pdp.deviceName,VK_VERSION_MAJOR(pdp.apiVersion),VK_VERSION_MINOR(pdp.apiVersion));
    free(pds); CALL(vkGetPhysicalDeviceMemoryProperties)(g_phys,&g_mp);

    uint32_t qn=0; CALL(vkGetPhysicalDeviceQueueFamilyProperties)(g_phys,&qn,0);
    VkQueueFamilyProperties *qps=calloc(qn,sizeof(*qps));
    CALL(vkGetPhysicalDeviceQueueFamilyProperties)(g_phys,&qn,qps);
    for(uint32_t i=0;i<qn;i++) if(qps[i].queueFlags&VK_QUEUE_COMPUTE_BIT){g_qfi=i;break;}
    free(qps); LOGI("qfi=%u",g_qfi);

    {float p=1; VkDeviceQueueCreateInfo dq={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,0,0,g_qfi,1,&p};
     VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,0,0,1,&dq,0,0,0,0};
     CALL(vkCreateDevice)(g_phys,&dci,0,&g_dev);}
    CALL(vkGetDeviceQueue)(g_dev,g_qfi,0,&g_queue);

    VkCommandPoolCreateInfo cpi={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,0,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,g_qfi};
    CALL(vkCreateCommandPool)(g_dev,&cpi,0,&g_cp);
    VkDescriptorPoolSize dps[]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,300}};
    VkDescriptorPoolCreateInfo dpi={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,0,0,100,1,dps};
    CALL(vkCreateDescriptorPool)(g_dev,&dpi,0,&g_dp);

    /* Load MUL_MAT shader */
    VkShaderModuleCreateInfo sm={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,0,0,
        _tmp_mulmat_test_spv_len,(const uint32_t*)_tmp_mulmat_test_spv};
    CALL(vkCreateShaderModule)(g_dev,&sm,0,&g_shader_mm);

    VkDescriptorSetLayoutBinding bnd[3];
    for(int i=0;i<3;i++) bnd[i]=(VkDescriptorSetLayoutBinding){i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
    VkDescriptorSetLayoutCreateInfo dlc={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,3,bnd};
    CALL(vkCreateDescriptorSetLayout)(g_dev,&dlc,0,&g_dsl);
    VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,12};
    VkPipelineLayoutCreateInfo plc={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&g_dsl,1,&pcr};
    CALL(vkCreatePipelineLayout)(g_dev,&plc,0,&g_pl_mm);
    VkComputePipelineCreateInfo cpci={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,0,0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,g_shader_mm,"main",0},
        g_pl_mm,0,-1};
    CALL(vkCreateComputePipelines)(g_dev,0,1,&cpci,0,&g_pipe_mm);
    g_vk_matmul_ready = 1;
    LOGI("=== Ready ===");
    return 0;
}

int g_vk_matmul_ready = 0;

    {
        extern void register_backend(ggml_backend_reg_t reg);
        #include "ggml-backend-impl.h"

        static struct { ggml_backend_dev_t dev; } ctx;
        static struct ggml_backend_device_i dev_i = {}; /* zero-init all to nullptr */

        dev_i.get_name   = [](ggml_backend_dev_t)->const char*{ return "OSH26_Vulkan"; };
        dev_i.get_description = [](ggml_backend_dev_t)->const char*{ return "Minimal GPU (MUL_MAT only)"; };
        dev_i.get_memory = [](ggml_backend_dev_t, size_t*f,size_t*t){ *f=*t=1024*1024*1024; };
        dev_i.get_type   = [](ggml_backend_dev_t)->ggml_backend_dev_type{ return GGML_BACKEND_DEVICE_TYPE_GPU; };
        dev_i.get_props  = [](ggml_backend_dev_t, ggml_backend_dev_props *p){
            p->name="OSH26 Vulkan"; p->description="GPU"; p->type=GGML_BACKEND_DEVICE_TYPE_GPU;
            p->memory_free=p->memory_total=1024*1024*1024;
            p->caps={false,false,false,false};
        };
        dev_i.supports_op = [](ggml_backend_dev_t, const ggml_tensor *op)->bool{
            return op->op==GGML_OP_MUL_MAT;
        };
        dev_i.get_buffer_type = [](ggml_backend_dev_t d)->ggml_backend_buffer_type_t{
            static struct ggml_backend_buffer_type_i i;
            static ggml_backend_buffer_type bt;
            i.get_name = [](ggml_backend_buffer_type_t)->const char*{ return "VK_BUF"; };
            i.alloc_buffer = [](ggml_backend_buffer_type_t, size_t sz)->ggml_backend_buffer_t{
                struct { VkBuffer vb; VkDeviceMemory vm; } *h = calloc(1, sizeof(*h)+sz);
                make_buf(sz,&h->vb,&h->vm);
                CALL(vkMapMemory)(g_dev,h->vm,0,sz,0,(void**)&((char*)h)[sizeof(*h)]);
                return (ggml_backend_buffer_t){0,NULL,h,sz,(ggml_backend_buffer_type_t)&bt};
            };
            i.free_buffer = [](ggml_backend_buffer_t b){
                auto *h=(decltype((struct{int x;})*)nullptr)b.context;
                CALL(vkDestroyBuffer)(g_dev,((typeof(h))b.context)->vb,0);
                CALL(vkFreeMemory)(g_dev,((typeof(h))b.context)->vm,0); free(b.context);
            };
            i.get_alignment = [](ggml_backend_buffer_type_t)->size_t{ return 256; };
            bt.iface=&i; bt.context=&bt;
            return &bt;
        };
        dev_i.get_host_buffer_type = NULL;
        dev_i.supports_buft = NULL;
        dev_i.init_backend = [](ggml_backend_dev_t dev, const char *d)->ggml_backend_t{
            static struct { ggml_backend_t b; ggml_backend_dev_t dev; } be;
            static struct ggml_backend_i iface;
            iface.graph_compute = [](ggml_backend_t b, ggml_cgraph *cg)->ggml_status{
                for(int i=0;i<cg->n_nodes;i++){
                    ggml_tensor *t=cg->nodes[i];
                    if(t->op!=GGML_OP_MUL_MAT||!ggml_backend_buffer_is_vk(t->buffer)) continue;
                    ggml_tensor *a=t->src[0], *b_=t->src[1];
                    int64_t M=t->ne[1], N=t->ne[0], K=a->ne[0];
                    /* data follows the buffer header */
                    float *da=(float*)((char*)a->buffer->context+32);
                    float *db=(float*)((char*)b_->buffer->context+32);
                    float *dc=(float*)((char*)t->buffer->context+32);
                    osh26_vk_matmul((int)M,(int)N,(int)K,da,db,dc);
                }
                return GGML_STATUS_SUCCESS;
            };
            iface.free_backend = [](ggml_backend_t){};
            be.b.iface=&iface; be.b.device=dev; be.b.context=&be; be.dev=dev;
            return &be.b;
            UNUSED(d);
        };
        dev_i.offload_op = [](ggml_backend_dev_t, const ggml_tensor *op)->bool{
            return op->op==GGML_OP_MUL_MAT;
        };
        dev_i.buffer_from_host_ptr = NULL;
        dev_i.event_new = NULL; dev_i.event_free = NULL;
        dev_i.event_record = NULL; dev_i.event_wait = NULL; dev_i.event_synchronize = NULL;

        ctx.dev = (ggml_backend_dev_t){GGML_BACKEND_API_VERSION, &dev_i, &ctx};
        static ggml_backend_reg reg = {GGML_BACKEND_API_VERSION, NULL, NULL};
        register_backend(&reg, nullptr);
        static ggml_backend_dev_t *devs[] = {&ctx.dev, NULL};
        reg = (ggml_backend_reg_t){GGML_BACKEND_API_VERSION, NULL, devs};
    }
    LOGI("=== Registered ===");
    return 0;
}

int osh26_vk_matmul(int M, int N, int K,
    const float *a_data, const float *b_data, float *c_data)
{
    VkDeviceSize Asz=M*K*sizeof(float), Bsz=K*N*sizeof(float), Csz=M*N*sizeof(float);
    VkBuffer ba,bb,bc; VkDeviceMemory ma,mb,mc;
    make_buf(Asz,&ba,&ma); make_buf(Bsz,&bb,&mb); make_buf(Csz,&bc,&mc);

    { void *p; CALL(vkMapMemory)(g_dev,ma,0,Asz,0,&p); memcpy(p,a_data,Asz); CALL(vkUnmapMemory)(g_dev,ma); }
    { void *p; CALL(vkMapMemory)(g_dev,mb,0,Bsz,0,&p); memcpy(p,b_data,Bsz); CALL(vkUnmapMemory)(g_dev,mb); }

    VkCommandBuffer cb; VkCommandBufferAllocateInfo cba={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,0,g_cp,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1};
    CALL(vkAllocateCommandBuffers)(g_dev,&cba,&cb);
    CALL(vkBeginCommandBuffer)(cb,&(VkCommandBufferBeginInfo){VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,0,0,0});

    VkDescriptorSet ds; VkDescriptorSetAllocateInfo dsa={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,g_dp,1,&g_dsl};
    CALL(vkAllocateDescriptorSets)(g_dev,&dsa,&ds);
    VkDescriptorBufferInfo dbi[3]={{ba,0,Asz},{bb,0,Bsz},{bc,0,Csz}};
    VkWriteDescriptorSet w[3]; for(int i=0;i<3;i++) w[i]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,ds,i,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&dbi[i],0};
    CALL(vkUpdateDescriptorSets)(g_dev,3,w,0,0);

    uint32_t pc[3]={(uint32_t)M,(uint32_t)N,(uint32_t)K};
    CALL(vkCmdPushConstants)(cb,g_pl_mm,VK_SHADER_STAGE_COMPUTE_BIT,0,12,pc);
    CALL(vkCmdBindPipeline)(cb,VK_PIPELINE_BIND_POINT_COMPUTE,g_pipe_mm);
    CALL(vkCmdBindDescriptorSets)(cb,VK_PIPELINE_BIND_POINT_COMPUTE,g_pl_mm,0,1,&ds,0,0);
    CALL(vkCmdDispatch)(cb,(M+15)/16,(N+15)/16,1);
    CALL(vkEndCommandBuffer)(cb);
    VkFence f; CALL(vkCreateFence)(g_dev,&(VkFenceCreateInfo){VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,0,0},0,&f);
    CALL(vkQueueSubmit)(g_queue,1,&(VkSubmitInfo){VK_STRUCTURE_TYPE_SUBMIT_INFO,0,0,0,0,1,&cb,0,0},f);
    CALL(vkWaitForFences)(g_dev,1,&f,VK_TRUE,UINT64_MAX); CALL(vkDestroyFence)(g_dev,f,0);

    { void *p; CALL(vkMapMemory)(g_dev,mc,0,Csz,0,&p); memcpy(c_data,p,Csz); CALL(vkUnmapMemory)(g_dev,mc); }
    CALL(vkDestroyBuffer)(g_dev,ba,0);CALL(vkFreeMemory)(g_dev,ma,0);
    CALL(vkDestroyBuffer)(g_dev,bb,0);CALL(vkFreeMemory)(g_dev,mb,0);
    CALL(vkDestroyBuffer)(g_dev,bc,0);CALL(vkFreeMemory)(g_dev,mc,0);
    CALL(vkFreeCommandBuffers)(g_dev,g_cp,1,&cb);
    return 0;
}
