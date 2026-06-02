/* Minimal GPU MUL_MAT hook. Intercepts MUL_MAT in ggml-cpu, runs on GPU.
   Mutex-protected, persistent temp buffers, descriptor cleanup.
   SAME proven approach as standalone test (osh26_vk_matmul.c). */

#include "osh26_vk_hook.h"
#include "ggml.h"
#include <android/log.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "mulmat_tiled.spv.h"
#include "mulmat_reduce.spv.h"
#define TAG "OSH26_HOOK"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__)

static VkInstance       V; static VkDevice D; static VkQueue Q;
static VkCommandPool    CP; static VkDescriptorPool DP;
static VkDescriptorSetLayout DSL; static VkPipelineLayout PL;
static VkPipeline       PR, PT; static uint32_t QFI;
static VkPhysicalDeviceMemoryProperties MP;
extern bool (*g_osh26_vk_hook_mul_mat)(const void*,void*);
static bool hook_impl(const void*params,void*tensor);
static pthread_mutex_t  Mtx = PTHREAD_MUTEX_INITIALIZER;
static bool             ok;
static int              n_calls, n_gpu;

static uint32_t mt(uint32_t b,VkMemoryPropertyFlags f){
    for(uint32_t i=0;i<MP.memoryTypeCount;i++)if((b&(1u<<i))&&(MP.memoryTypes[i].propertyFlags&f)==f)return i;
    return UINT32_MAX;
}
// Persistent temp buffers
static struct {VkBuffer B;VkDeviceMemory M;void*P;VkDeviceSize S;} TB[3];
static bool grow(int i,VkDeviceSize sz){
    if(TB[i].S>=sz)return true;
    if(TB[i].P)vkUnmapMemory(D,TB[i].M);
    if(TB[i].M)vkFreeMemory(D,TB[i].M,0);
    if(TB[i].B)vkDestroyBuffer(D,TB[i].B,0);
    memset(&TB[i],0,sizeof(TB[i]));
    VkBufferCreateInfo b={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,0,0,sz,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_SHARING_MODE_EXCLUSIVE,0,0};
    if(vkCreateBuffer(D,&b,0,&TB[i].B))return false;
    VkMemoryRequirements mr;vkGetBufferMemoryRequirements(D,TB[i].B,&mr);
    uint32_t m=mt(mr.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if(m==UINT32_MAX)return false;
    VkMemoryAllocateInfo a={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,0,mr.size,m};
    if(vkAllocateMemory(D,&a,0,&TB[i].M))return false;
    vkBindBufferMemory(D,TB[i].B,TB[i].M,0);
    vkMapMemory(D,TB[i].M,0,sz,0,&TB[i].P);
    TB[i].S=sz;return true;
}
// Copy tensor to contiguous f32 buffer
static void cpin(float*d,const struct ggml_tensor*s,int64_t R,int64_t C){
    if(s->type==GGML_TYPE_F32&&ggml_is_contiguous(s))memcpy(d,s->data,R*C*sizeof(float));
    else for(int64_t r=0;r<R;r++){const float*sr=(const float*)((const uint8_t*)s->data+r*s->nb[1]);memcpy(d+r*C,sr,C*sizeof(float));}
}
static void cpout(struct ggml_tensor*d,const float*s,int64_t R,int64_t C){
    if(ggml_is_contiguous(d))memcpy(d->data,s,R*C*sizeof(float));
    else for(int64_t r=0;r<R;r++){float*dr=(float*)((uint8_t*)d->data+r*d->nb[1]);memcpy(dr,s+r*C,C*sizeof(float));}
}

int osh26_vk_hook_init(void){if(ok)return 0;
    void*lib=dlopen("libvulkan.so",RTLD_NOW);if(!lib)return -1;
    VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO,0,"OSH26Hook",1,"OSH26",1,VK_API_VERSION_1_1};
    VkInstanceCreateInfo ci={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,0,0,&ai,0,0,0,0};
    if(vkCreateInstance(&ci,0,&V))return -1;
    uint32_t nd=0;vkEnumeratePhysicalDevices(V,&nd,0);
    VkPhysicalDevice*pd=calloc(nd,sizeof(*pd));vkEnumeratePhysicalDevices(V,&nd,pd);
    VkPhysicalDevice ph=pd[0];free(pd);
    VkPhysicalDeviceProperties pdp;vkGetPhysicalDeviceProperties(ph,&pdp);
    vkGetPhysicalDeviceMemoryProperties(ph,&MP);LOGI("GPU:%s",pdp.deviceName);
    uint32_t qn=0;vkGetPhysicalDeviceQueueFamilyProperties(ph,&qn,0);
    VkQueueFamilyProperties*qp=calloc(qn,sizeof(*qp));vkGetPhysicalDeviceQueueFamilyProperties(ph,&qn,qp);
    for(uint32_t i=0;i<qn;i++)if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){QFI=i;break;}free(qp);
    float pr=1;VkDeviceQueueCreateInfo dq={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,0,0,QFI,1,&pr};
    VkDeviceCreateInfo dc={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,0,0,1,&dq,0,0,0,0};
    if(vkCreateDevice(ph,&dc,0,&D))return -1;vkGetDeviceQueue(D,QFI,0,&Q);
    VkCommandPoolCreateInfo cp={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,0,VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,QFI};
    vkCreateCommandPool(D,&cp,0,&CP);
    VkDescriptorPoolSize ds={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4096};
    VkDescriptorPoolCreateInfo dp={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,0,VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,1024,1,&ds};
    vkCreateDescriptorPool(D,&dp,0,&DP);
    VkDescriptorSetLayoutBinding bd[3];
    for(int i=0;i<3;i++)bd[i]=(VkDescriptorSetLayoutBinding){i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
    VkDescriptorSetLayoutCreateInfo dl={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,3,bd};
    vkCreateDescriptorSetLayout(D,&dl,0,&DSL);
    VkPushConstantRange pc={VK_SHADER_STAGE_COMPUTE_BIT,0,16};
    VkPipelineLayoutCreateInfo pl={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&DSL,1,&pc};
    vkCreatePipelineLayout(D,&pl,0,&PL);
    VkShaderModuleCreateInfo sm={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,0,0,
        examples_android_osh26_app_src_main_cpp_mulmat_tiled_spv_len,(const uint32_t*)examples_android_osh26_app_src_main_cpp_mulmat_tiled_spv};
    VkShaderModule m1,m2;vkCreateShaderModule(D,&sm,0,&m1);
    sm.codeSize=examples_android_osh26_app_src_main_cpp_mulmat_reduce_spv_len;sm.pCode=(const uint32_t*)examples_android_osh26_app_src_main_cpp_mulmat_reduce_spv;vkCreateShaderModule(D,&sm,0,&m2);
    VkComputePipelineCreateInfo pi={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,0,0,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,m1,"main",0},PL,0,(uint32_t)-1};
    vkCreateComputePipelines(D,0,1,&pi,0,&PT);pi.stage.module=m2;vkCreateComputePipelines(D,0,1,&pi,0,&PR);
    ok=true;g_osh26_vk_hook_mul_mat=hook_impl;LOGI("Hook ready");return 0;}

bool osh26_vk_hook_ready(void){return ok;}
void osh26_vk_hook_free(void){}

// The hook implementation
static bool hook_impl(const void*params,void*tensor){
    (void)params;if(!ok)return false;
    struct ggml_tensor*d=(struct ggml_tensor*)tensor;
    if(d->op!=GGML_OP_MUL_MAT)return false;
    struct ggml_tensor*a=d->src[0],*b=d->src[1];
    if(!a||!b||b->type!=GGML_TYPE_F32||d->type!=GGML_TYPE_F32)return false;
    int64_t M=d->ne[1],N=d->ne[0],K=a->ne[0];
    if(M<=0||N<=0||K<=0)return false;

    bool r=false;VkDescriptorSet ds=0;VkCommandBuffer cb=0;VkFence f=0;
    pthread_mutex_lock(&Mtx);
    n_calls++;

    VkDeviceSize Az=(VkDeviceSize)(M*K*4),Bz=(VkDeviceSize)(N*K*4),Cz=(VkDeviceSize)(M*N*4);
    if(!grow(0,Az)||!grow(1,Bz)||!grow(2,Cz))goto done;
    cpin((float*)TB[0].P,a,M,K);
    cpin((float*)TB[1].P,b,N,K);

    {VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP,1,&DSL};
     if(vkAllocateDescriptorSets(D,&da,&ds)){vkResetDescriptorPool(D,DP,0);vkAllocateDescriptorSets(D,&da,&ds);}}
    {VkDescriptorBufferInfo bi[3]={{TB[1].B,0,VK_WHOLE_SIZE},{TB[0].B,0,VK_WHOLE_SIZE},{TB[2].B,0,VK_WHOLE_SIZE}};
     VkWriteDescriptorSet wr[3];memset(wr,0,sizeof(wr));
     for(int j=0;j<3;j++){wr[j].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;wr[j].dstSet=ds;wr[j].dstBinding=j;wr[j].descriptorCount=1;wr[j].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;wr[j].pBufferInfo=&bi[j];}
     vkUpdateDescriptorSets(D,3,wr,0,0);}
    {VkCommandBufferAllocateInfo ca={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,0,CP,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1};
     if(vkAllocateCommandBuffers(D,&ca,&cb))goto done;}
    {uint32_t pc[4]={(uint32_t)M,(uint32_t)N,(uint32_t)K,(uint32_t)K};
     VkCommandBufferBeginInfo bi={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,0,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,0};
     vkBeginCommandBuffer(cb,&bi);
     if(M<=4){vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PR);vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL,0,1,&ds,0,0);vkCmdPushConstants(cb,PL,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc);vkCmdDispatch(cb,(uint32_t)N,(uint32_t)M,1);}
     else{vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PT);vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL,0,1,&ds,0,0);vkCmdPushConstants(cb,PL,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc);vkCmdDispatch(cb,((uint32_t)N+7)/8,((uint32_t)M+7)/8,1);}
     vkEndCommandBuffer(cb);}
    {VkFenceCreateInfo fi={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,0,0};vkCreateFence(D,&fi,0,&f);
     VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO,0,0,0,0,1,&cb,0,0};
     if(vkQueueSubmit(Q,1,&si,f)==VK_SUCCESS)vkWaitForFences(D,1,&f,VK_TRUE,UINT64_MAX);}
    cpout(d,(const float*)TB[2].P,M,N);
    n_gpu++;r=true;
done:
    if(f)vkDestroyFence(D,f,0);
    if(cb)vkFreeCommandBuffers(D,CP,1,&cb);
    if(ds)vkFreeDescriptorSets(D,DP,1,&ds);
    pthread_mutex_unlock(&Mtx);
    if(n_calls<=3||(n_calls%100==0))LOGI("call#%d gpu=%d cpu=%d",n_calls,n_gpu,n_calls-n_gpu);
    return r;
}
