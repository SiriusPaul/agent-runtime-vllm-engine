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
#include "vk_wrapper/vulkan_wrapper.h"
#include "mulmat_tiled.spv.h"
#include "mulmat_reduce.spv.h"
#include "rms_norm.spv.h"
#include "rope.spv.h"
#include "softmax_gpu.spv.h"
#include "silu_mul.spv.h"
#include "add.spv.h"
#include "attn_decode_q1.spv.h"

#define TAG "OSH26GPU"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__)

#define HDIM 1024
#define IDIM 3072
#define N_LAY 28
#define N_HD  16
#define N_KVH 8
#define HD    128
#define QDIM  (N_HD*HD)
#define KVD   (N_KVH*HD)
#define VOCAB 151936
#define MAX_S 512
#define F32(n) ((VkDeviceSize)(n)*4)

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
static void buf_flush(VkBuf*b){VkMappedMemoryRange r={VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,0,b->M,0,b->size};vkFlushMappedMemoryRanges(D,1,&r);}
static void buf_inv(VkBuf*b){VkMappedMemoryRange r={VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,0,b->M,0,b->size};vkInvalidateMappedMemoryRanges(D,1,&r);}

/* Dummy buffer for unused descriptor bindings */
static VkBuf B_Dummy;

/* Weights: one VkBuf per tensor per layer */
static VkBuf W_ra[N_LAY],W_Q[N_LAY],W_K[N_LAY],W_V[N_LAY],W_O[N_LAY];
static VkBuf W_Qn[N_LAY],W_Kn[N_LAY];
static VkBuf W_rf[N_LAY],W_Gate[N_LAY],W_Up[N_LAY],W_Down[N_LAY];
static VkBuf W_Fnorm,W_Head; static float *Emb;

/* Activation buffers */
static VkBuf B_KV,B_Hid,B_Hid2,B_Qb,B_Kb,B_Vb,B_Sc,B_Att,B_Gat,B_Up,B_Dwn,B_Tmp,B_Log;
static VkBuf B_KCache[N_LAY],B_VCache[N_LAY],B_AttnConst;

static VkInstance V;static VkQueue Q;static VkCommandPool CP;static VkDescriptorPool DP;
static VkDescriptorSetLayout DSL;static VkPipelineLayout PL;
static VkDescriptorSetLayout DSL_Dec;static VkPipelineLayout PL_Dec;
static VkPipeline P_MMt,P_MMr,P_RMS,P_RoPE,P_SMax,P_SiLU,P_Add,P_AttnDec;
static uint32_t QFI;
static pthread_mutex_t Mtx=PTHREAD_MUTEX_INITIALIZER;
static bool vk_ok,mdl_ok;
static bool g_mnn_attention_enabled;
static float g_last_attention_max_abs_err;
static uint32_t g_attention_fallback_layers;
static int g_last_logits_top5[5];
static float g_last_logits_top5_values[5];

typedef struct {
    int32_t s0[4];
    int32_t s1[4];
    int32_t s2[4];
    float f0[4];
} AttnConst;

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

static VkCommandBuffer CB(void){VkCommandBuffer cb;VkCommandBufferAllocateInfo a={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,0,CP,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1};vkAllocateCommandBuffers(D,&a,&cb);VkCommandBufferBeginInfo b={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,0,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,0};vkBeginCommandBuffer(cb,&b);return cb;}
static VkDescriptorSet g_ds[16];static int g_n;
static void Sub(VkCommandBuffer cb){vkEndCommandBuffer(cb);VkFence f;VkFenceCreateInfo fi={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,0,0};vkCreateFence(D,&fi,0,&f);VkSubmitInfo s={VK_STRUCTURE_TYPE_SUBMIT_INFO,0,0,0,0,1,&cb,0,0};vkQueueSubmit(Q,1,&s,f);vkWaitForFences(D,1,&f,VK_TRUE,UINT64_MAX);vkDestroyFence(D,f,0);vkFreeCommandBuffers(D,CP,1,&cb);if(g_n>0){vkFreeDescriptorSets(D,DP,(uint32_t)g_n,g_ds);g_n=0;}}

/* Bind buffers to descriptor set. NULL buffer → use dummy */
static void BIND(VkCommandBuffer cb,VkBuf*b0,VkBuf*b1,VkBuf*b2,const uint32_t pc[4]){
    if(!b0)b0=&B_Dummy;if(!b1)b1=&B_Dummy;if(!b2)b2=&B_Dummy;
    VkDescriptorSet ds=0;VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP,1,&DSL};
    VkResult ar=vkAllocateDescriptorSets(D,&da,&ds);if(ar!=VK_SUCCESS){LOGE("DS alloc fail!");return;}
    if(g_n<16)g_ds[g_n++]=ds;
    VkDescriptorBufferInfo bi[3]={{b0->B,0,b0->size},{b1->B,0,b1->size},{b2->B,0,b2->size}};
    VkWriteDescriptorSet wr[3];memset(wr,0,sizeof(wr));
    for(int j=0;j<3;j++){wr[j].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;wr[j].dstSet=ds;wr[j].dstBinding=j;wr[j].descriptorCount=1;wr[j].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;wr[j].pBufferInfo=&bi[j];}
    vkUpdateDescriptorSets(D,3,wr,0,0);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL,0,1,&ds,0,0);
    vkCmdPushConstants(cb,PL,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc);
}

static void BIND_DEC(VkCommandBuffer cb,VkBuf*b0,VkBuf*b1,VkBuf*b2,VkBuf*b3,VkBuf*b4,VkBuf*b5,VkBuf*b6,VkBuf*b7){
    VkBuf* b[8]={b0,b1,b2,b3,b4,b5,b6,b7};
    for(int i=0;i<8;i++)if(!b[i])b[i]=&B_Dummy;
    VkDescriptorSet ds=0;VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP,1,&DSL_Dec};
    VkResult ar=vkAllocateDescriptorSets(D,&da,&ds);if(ar!=VK_SUCCESS){LOGE("DEC DS alloc fail!");return;}
    if(g_n<16)g_ds[g_n++]=ds;
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

/* Dispatch macros — pipeline FIRST (MNN order: bind pipeline, then descriptors) */
#define RMS(cb,rows,cols,x,w,y) do{uint32_t p[4]={rows,cols,0,0};vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_RMS);BIND(cb,x,w,y,p);vkCmdDispatch(cb,(uint32_t)(rows),1,1);}while(0)
#define BARRIER(cb) do{VkMemoryBarrier mb_={VK_STRUCTURE_TYPE_MEMORY_BARRIER,0,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT};vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb_,0,0,0,0);}while(0)
#define MM(cb,w,x,y,M,N,K) do{uint32_t p[4]={M,N,K,K};if((M)<=4&&(N)<=65535){vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_MMr);}else{vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_MMt);}BIND(cb,w,x,y,p);if((M)<=4&&(N)<=65535){vkCmdDispatch(cb,(N),(M),1);}else{vkCmdDispatch(cb,((N)+7)/8,((M)+7)/8,1);}}while(0)
#define ROPE(cb,x,nt,nh,posv,hd) do{uint32_t p[4]={nt,nh,(uint32_t)(posv),hd};vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_RoPE);BIND(cb,x,NULL,NULL,p);uint32_t tot_=(nt)*(nh)*((hd)/2);vkCmdDispatch(cb,(tot_+63)/64,1,1);}while(0)
#define ADD(cb,y,x,n) do{uint32_t p[4]={n,0,0,0};vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_Add);BIND(cb,y,x,NULL,p);vkCmdDispatch(cb,((n)+63)/64,1,1);}while(0)
#define SILU(cb,a,b,z,n) do{uint32_t p[4]={n,0,0,0};vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_SiLU);BIND(cb,a,b,z,p);vkCmdDispatch(cb,((n)+63)/64,1,1);}while(0)
#define DEC_ATTN(cb,out,q,kcache,vcache,cbuf) do{vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P_AttnDec);BIND_DEC(cb,out,q,NULL,NULL,kcache,vcache,NULL,cbuf);vkCmdDispatch(cb,N_HD,1,1);}while(0)

static void pack_mnn_kv_cache(int l,int pos,int nt,const float*kb,const float*vb){
    const int d4s=HD/4;
    float *kc=B_KCache[l].P,*vc=B_VCache[l].P;
    for(int t=0;t<nt;t++){
        int token=pos+t;
        if(token<0||token>=MAX_S)continue;
        for(int kvh=0;kvh<N_KVH;kvh++){
            const float *krow=kb+(t*N_KVH+kvh)*HD;
            const float *vrow=vb+(t*N_KVH+kvh)*HD;
            for(int d4=0;d4<d4s;d4++){
                memcpy(kc+(((kvh*d4s+d4)*MAX_S+token)*4),krow+d4*4,4*sizeof(float));
                memcpy(vc+(((kvh*MAX_S+token)*d4s+d4)*4),vrow+d4*4,4*sizeof(float));
            }
        }
    }
    buf_flush(&B_KCache[l]);buf_flush(&B_VCache[l]);
}

static void cpu_attention_ref(int nt,int pos,int l,const float*qb,const float*kv,float*sc,float*out){
    int kvo=l*2*MAX_S*KVD;
    int slen=pos+nt;
    memset(sc,0,nt*N_HD*MAX_S*sizeof(float));
    float isd=1.0f/sqrtf((float)HD);
    for(int h=0;h<N_HD;h++){
        int kvh=h*N_KVH/N_HD;
        for(int t=0;t<nt;t++){
            const float*qt=qb+t*QDIM+h*HD;
            float*sr=sc+(t*N_HD+h)*MAX_S;
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
        float*row=sc+r*MAX_S;
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
            const float*sr=sc+(t*N_HD+h)*MAX_S;
            for(int s=0;s<slen;s++){
                const float*vs=kv+kvo+MAX_S*KVD+s*KVD+kvh*HD;
                float wgt=sr[s];
                for(int d=0;d<HD;d++)oh[d]+=wgt*vs[d];
            }
        }
    }
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
    VkDescriptorSetLayoutBinding bd[3];for(int i=0;i<3;i++)bd[i]=(VkDescriptorSetLayoutBinding){i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
    VkDescriptorSetLayoutCreateInfo dl={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,3,bd};vkCreateDescriptorSetLayout(D,&dl,0,&DSL);
    VkPushConstantRange pc={VK_SHADER_STAGE_COMPUTE_BIT,0,16};
    VkPipelineLayoutCreateInfo pl={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&DSL,1,&pc};vkCreatePipelineLayout(D,&pl,0,&PL);
    VkDescriptorSetLayoutBinding bdd[8];for(int i=0;i<8;i++)bdd[i]=(VkDescriptorSetLayoutBinding){(uint32_t)i,(i==7)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
    VkDescriptorSetLayoutCreateInfo dld={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,8,bdd};vkCreateDescriptorSetLayout(D,&dld,0,&DSL_Dec);
    VkPipelineLayoutCreateInfo pld={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&DSL_Dec,0,0};vkCreatePipelineLayout(D,&pld,0,&PL_Dec);
    { VkShaderModuleCreateInfo s={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,0,0,0,0};VkShaderModule m;
      s.codeSize=examples_android_osh26_app_src_main_cpp_mulmat_tiled_spv_len;s.pCode=(const uint32_t*)examples_android_osh26_app_src_main_cpp_mulmat_tiled_spv;vkCreateShaderModule(D,&s,0,&m);VkComputePipelineCreateInfo pi={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,0,0,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,m,"main",0},PL,0,(uint32_t)-1};vkCreateComputePipelines(D,0,1,&pi,0,&P_MMt);
      s.codeSize=examples_android_osh26_app_src_main_cpp_mulmat_reduce_spv_len;s.pCode=(const uint32_t*)examples_android_osh26_app_src_main_cpp_mulmat_reduce_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_MMr);
      s.codeSize=_tmp_rms_norm_spv_len;s.pCode=(const uint32_t*)_tmp_rms_norm_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_RMS);
      s.codeSize=_tmp_rope_spv_len;s.pCode=(const uint32_t*)_tmp_rope_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_RoPE);
      s.codeSize=_tmp_softmax_gpu_spv_len;s.pCode=(const uint32_t*)_tmp_softmax_gpu_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_SMax);
      s.codeSize=_tmp_silu_mul_spv_len;s.pCode=(const uint32_t*)_tmp_silu_mul_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_SiLU);
      s.codeSize=_tmp_add_spv_len;s.pCode=(const uint32_t*)_tmp_add_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_Add);
      s.codeSize=_tmp_attn_decode_q1_spv_len;s.pCode=(const uint32_t*)_tmp_attn_decode_q1_spv;vkCreateShaderModule(D,&s,0,&m);VkComputePipelineCreateInfo pid={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,0,0,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,m,"main",0},PL_Dec,0,(uint32_t)-1};if(vkCreateComputePipelines(D,0,1,&pid,0,&P_AttnDec)==VK_SUCCESS)g_mnn_attention_enabled=true; }
    buf_alloc(&B_Dummy,256);
    buf_alloc(&B_AttnConst,sizeof(AttnConst));
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
#define LOAD_BUF(name,buf,nelem) do{if(!buf_alloc(&(buf),((VkDeviceSize)(nelem))*4)){LOGE("alloc %s",name);return -1;}if(!read_gguf(gctx,f,name,(buf).P)){LOGE("read %s",name);return -1;}buf_flush(&(buf));}while(0)
int osh26_vk_gpu_load_model(const char*path){if(!vk_ok||!path)return -1;
    if(!B_Dummy.B && !buf_alloc(&B_Dummy,256)){LOGE("alloc dummy");return -1;}
    FILE*f=fopen(path,"rb");if(!f){LOGE("open %s",path);return -1;}
    struct gguf_init_params gp={true,NULL};struct gguf_context*gctx=gguf_init_from_file(path,gp);if(!gctx){fclose(f);return -1;}
    Emb=(float*)malloc(((VkDeviceSize)VOCAB*HDIM)*4);if(!read_gguf(gctx,f,"token_embd.weight",Emb)){free(Emb);gguf_free(gctx);fclose(f);return -1;}
    char n[128];
    for(int l=0;l<N_LAY;l++){
        snprintf(n,sizeof(n),"blk.%d.attn_norm.weight",l);LOAD_BUF(n,W_ra[l],HDIM);
        snprintf(n,sizeof(n),"blk.%d.attn_q.weight",l);LOAD_BUF(n,W_Q[l],QDIM*HDIM);
        snprintf(n,sizeof(n),"blk.%d.attn_k.weight",l);LOAD_BUF(n,W_K[l],KVD*HDIM);
        snprintf(n,sizeof(n),"blk.%d.attn_v.weight",l);LOAD_BUF(n,W_V[l],KVD*HDIM);
        snprintf(n,sizeof(n),"blk.%d.attn_output.weight",l);LOAD_BUF(n,W_O[l],HDIM*QDIM);
        snprintf(n,sizeof(n),"blk.%d.attn_q_norm.weight",l);LOAD_BUF(n,W_Qn[l],HD);
        snprintf(n,sizeof(n),"blk.%d.attn_k_norm.weight",l);LOAD_BUF(n,W_Kn[l],HD);
        snprintf(n,sizeof(n),"blk.%d.ffn_norm.weight",l);LOAD_BUF(n,W_rf[l],HDIM);
        snprintf(n,sizeof(n),"blk.%d.ffn_gate.weight",l);LOAD_BUF(n,W_Gate[l],IDIM*HDIM);
        snprintf(n,sizeof(n),"blk.%d.ffn_up.weight",l);LOAD_BUF(n,W_Up[l],IDIM*HDIM);
        snprintf(n,sizeof(n),"blk.%d.ffn_down.weight",l);LOAD_BUF(n,W_Down[l],HDIM*IDIM);
    }
    LOAD_BUF("output_norm.weight",W_Fnorm,HDIM);
    if(gguf_find_tensor(gctx,"output.weight")>=0)LOAD_BUF("output.weight",W_Head,VOCAB*HDIM);
    else{buf_alloc(&W_Head,((VkDeviceSize)VOCAB*HDIM)*4);memcpy(W_Head.P,Emb,((VkDeviceSize)VOCAB*HDIM)*4);buf_flush(&W_Head);}
    gguf_free(gctx);fclose(f);
    buf_alloc(&B_KV,((VkDeviceSize)N_LAY*2*MAX_S*KVD)*4);memset(B_KV.P,0,B_KV.size);buf_flush(&B_KV);
    for(int l=0;l<N_LAY;l++){
        buf_alloc(&B_KCache[l],((VkDeviceSize)N_KVH*HD*MAX_S)*4);memset(B_KCache[l].P,0,B_KCache[l].size);buf_flush(&B_KCache[l]);
        buf_alloc(&B_VCache[l],((VkDeviceSize)N_KVH*HD*MAX_S)*4);memset(B_VCache[l].P,0,B_VCache[l].size);buf_flush(&B_VCache[l]);
    }
    buf_alloc(&B_Hid,((VkDeviceSize)MAX_S*HDIM)*4);
    buf_alloc(&B_Hid2,((VkDeviceSize)MAX_S*HDIM)*4);
    buf_alloc(&B_Qb,((VkDeviceSize)MAX_S*QDIM)*4);
    buf_alloc(&B_Kb,((VkDeviceSize)MAX_S*KVD)*4);
    buf_alloc(&B_Vb,((VkDeviceSize)MAX_S*KVD)*4);
    buf_alloc(&B_Sc,((VkDeviceSize)N_HD*MAX_S*MAX_S)*4);
    buf_alloc(&B_Att,((VkDeviceSize)MAX_S*QDIM)*4);
    buf_alloc(&B_Gat,((VkDeviceSize)MAX_S*IDIM)*4);
    buf_alloc(&B_Up,((VkDeviceSize)MAX_S*IDIM)*4);
    buf_alloc(&B_Dwn,((VkDeviceSize)MAX_S*IDIM)*4);
    buf_alloc(&B_Tmp,((VkDeviceSize)MAX_S*HDIM)*4);
    buf_alloc(&B_Log,((VkDeviceSize)VOCAB)*4);
    mdl_ok=true;LOGI("Model loaded");return 0;}

/* ---- Forward pass ---- */
int osh26_vk_gpu_forward(const int*tokens,int nt,int pos){if(!mdl_ok)return -1;static int fc=0;if(++fc<=3)LOGI("forward#%d nt=%d pos=%d",fc,nt,pos);pthread_mutex_lock(&Mtx);
    float*hidden=B_Hid.P,*hnorm=B_Hid2.P,*qb=B_Qb.P,*kb=B_Kb.P,*vb=B_Vb.P,*sc=B_Sc.P,*att=B_Att.P,*gate=B_Gat.P,*up=B_Up.P,*dwn=B_Dwn.P,*kv=B_KV.P,*tmp=B_Tmp.P;
    for(int i=0;i<nt;i++){int tok=tokens[i];if(tok<0||tok>=VOCAB)tok=0;memcpy(hidden+i*HDIM,Emb+tok*HDIM,HDIM*sizeof(float));}
    buf_flush(&B_Hid);
    int do_diag=(nt==1); /* diag for all layers on decode */
    for(int l=0;l<N_LAY;l++){
        /* --- Submit 1a: RMS attn --- */
        { VkCommandBuffer cb1a=CB();
          RMS(cb1a,nt,HDIM,&B_Hid,&W_ra[l],&B_Hid2);
          Sub(cb1a); }
        if(nt==1)buf_inv(&B_Hid2);
        if(do_diag && l==0){float*cpu=B_Hid.P,*w=W_ra[0].P;float ss=0;for(int i=0;i<HDIM;i++){float v=cpu[i];ss+=v*v;}float inv=1.0f/sqrtf(ss/(float)HDIM+1e-6f);
         float ecpu0=cpu[0]*inv*w[0],egpu0=hnorm[0];LOGI("D01 RMS: err=%.2e",(double)fabsf(ecpu0-egpu0));}

        /* --- Submit 1b: Q,K,V matmul (no RoPE) --- */
        { VkCommandBuffer cb1b=CB();
          MM(cb1b,&W_Q[l],&B_Hid2,&B_Qb,nt,QDIM,HDIM);
          MM(cb1b,&W_K[l],&B_Hid2,&B_Kb,nt,KVD,HDIM);
          MM(cb1b,&W_V[l],&B_Hid2,&B_Vb,nt,KVD,HDIM);
          Sub(cb1b); }
        if(nt==1){buf_inv(&B_Qb);buf_inv(&B_Kb);}
        if(do_diag && (l==0||l==1||l==27)){float*qw=W_Q[l].P,*kw=W_K[l].P;float cq0=0,ck0=0;
         for(int k=0;k<HDIM;k++){cq0+=qw[k]*hnorm[k];ck0+=kw[k]*hnorm[k];}
         LOGI("L%d Q0 e=%.1e K0 e=%.1e",l,(double)fabsf(cq0-qb[0]),(double)fabsf(ck0-kb[0]));}

        /* --- CPU: per-head Q/K RMSNorm required by Qwen3 --- */
        {
          buf_inv(&B_Qb);buf_inv(&B_Kb);
          for(int t=0;t<nt;t++){
            for(int h=0;h<N_HD;h++){
                float *qh = qb + t*QDIM + h*HD;
                float ss = 0.0f;
                for(int d=0;d<HD;d++){ float v = qh[d]; ss += v*v; }
                float inv = 1.0f/sqrtf(ss/(float)HD + 1e-6f);
                for(int d=0;d<HD;d++) qh[d] = qh[d]*inv*W_Qn[l].P[d];
            }
            for(int h=0;h<N_KVH;h++){
                float *kh = kb + t*KVD + h*HD;
                float ss = 0.0f;
                for(int d=0;d<HD;d++){ float v = kh[d]; ss += v*v; }
                float inv = 1.0f/sqrtf(ss/(float)HD + 1e-6f);
                for(int d=0;d<HD;d++) kh[d] = kh[d]*inv*W_Kn[l].P[d];
            }
          }
          buf_flush(&B_Qb);buf_flush(&B_Kb);
          if(do_diag && l==0){
            float ssq = 0.0f;
            for(int d=0;d<HD;d++){ float v = qb[d]; ssq += v*v; }
            LOGI("D04 QNorm: q_rms=%.4f q0=%.4f k0=%.4f",(double)sqrtf(ssq/(float)HD),(double)qb[0],(double)kb[0]);
          }
        }

        /* --- CPU: Qwen3 uses NEOX RoPE layout, not adjacent even/odd pairs --- */
        { float qpre0 = qb[0], qpre64 = qb[HD/2];
          rope_neox_cpu(qb, nt, N_HD, pos, HD);
          rope_neox_cpu(kb, nt, N_KVH, pos, HD);
          buf_flush(&B_Qb);buf_flush(&B_Kb);
          buf_inv(&B_Vb);
          if(do_diag && l==0){
           float th0=1.0f/powf(1e6f,0.0f/128.0f);
           float c=cosf((float)pos*th0),s=sinf((float)pos*th0);
           float r0=qpre0*c-qpre64*s; /* NEOX pairs dim 0 with dim 64 */
           LOGI("D05 RoPE: pos=%d preQ=%.4f/%.4f exp=%.4f got=%.4f err=%.2e",pos,(double)qpre0,(double)qpre64,(double)r0,(double)qb[0],(double)fabsf(r0-qb[0]));}
        }

        /* --- KV cache + MNN-style decode attention with CPU correctness gate --- */
        { int kvo=l*2*MAX_S*KVD;memcpy(kv+kvo+pos*KVD,kb,nt*KVD*sizeof(float));memcpy(kv+kvo+MAX_S*KVD+pos*KVD,vb,nt*KVD*sizeof(float));
          pack_mnn_kv_cache(l,pos,nt,kb,vb);
          cpu_attention_ref(nt,pos,l,qb,kv,sc,tmp);
          if(nt==1 && g_mnn_attention_enabled && ((g_attention_fallback_layers&(1u<<l))==0)){
              AttnConst ac={{1,pos+1,N_HD,N_KVH},{HD,N_HD/N_KVH,pos,pos+1},{0,0,0,MAX_S},{1.0f/sqrtf((float)HD),0,0,0}};
              memcpy(B_AttnConst.P,&ac,sizeof(ac));buf_flush(&B_AttnConst);
              { VkCommandBuffer cba=CB();DEC_ATTN(cba,&B_Att,&B_Qb,&B_KCache[l],&B_VCache[l],&B_AttnConst);Sub(cba); }
              buf_inv(&B_Att);
              float maxe=0.0f;for(int i=0;i<HD;i++){float e=fabsf(att[i]-tmp[i]);if(e>maxe)maxe=e;}
              g_last_attention_max_abs_err=maxe;
              if(maxe>1e-3f){
                  g_attention_fallback_layers|=(1u<<l);
                  memcpy(att,tmp,nt*QDIM*sizeof(float));
                  LOGE("L%d MNN decode attention fallback max_abs_err=%.3e",l,(double)maxe);
              }else if(do_diag && (l==0||l==27)){
                  LOGI("L%d MNN decode attention max_abs_err=%.3e",l,(double)maxe);
              }
          }else{
              memcpy(att,tmp,nt*QDIM*sizeof(float));
          } }
        /* GPU output projection reads B_Att for both prefill and decode. */
        buf_flush(&B_Att);
        if(do_diag && l==0){int kvo=0;int slen=pos+nt;
         /* CPU manual attention for head=0, token=0, dim=0 */
         float*score_cpu=(float*)malloc(slen*sizeof(float));
         int kvh0=0;float*att_cpu=(float*)calloc(QDIM,sizeof(float));
         for(int s=0;s<slen;s++){float dot=0;for(int d=0;d<HD;d++)dot+=qb[d]*kv[kvo+s*KVD+d];score_cpu[s]=dot/sqrtf((float)HD);}
         float mx=score_cpu[0];for(int s=1;s<slen;s++)if(score_cpu[s]>mx)mx=score_cpu[s];
         float sum_w=0;for(int s=0;s<slen;s++){float w=expf(score_cpu[s]-mx);sum_w+=w;score_cpu[s]=w;}
         for(int s=0;s<slen;s++)score_cpu[s]/=sum_w;
         for(int s=0;s<slen;s++){float w=score_cpu[s];for(int d=0;d<HD;d++)att_cpu[d]+=w*kv[kvo+MAX_S*KVD+s*KVD+d];}
         LOGI("D08 ATTN: gpu[0]=%.6f cpu[0]=%.6f err=%.2e",(double)att[0],(double)att_cpu[0],(double)fabsf(att[0]-att_cpu[0]));
         free(score_cpu);free(att_cpu);}

        /* --- Submit 2a: O projection + residual --- */
        { float oh0=B_Hid.P[0];
          VkCommandBuffer cb2a=CB();
          MM(cb2a,&W_O[l],&B_Att,&B_Tmp,nt,HDIM,QDIM);
          BARRIER(cb2a);
          ADD(cb2a,&B_Hid,&B_Tmp,nt*HDIM);
          Sub(cb2a);
          if(do_diag&&l==0){float*ow=W_O[0].P;float co=0;for(int k=0;k<QDIM;k++)co+=ow[k]*att[k];
           LOGI("D09 Omat err=%.2e",(double)fabsf(co-B_Tmp.P[0]));
           LOGI("D10 ResA err=%.2e",(double)fabsf((oh0+B_Tmp.P[0])-B_Hid.P[0]));}
        }

        /* --- Submit 2b: RMS_FFN + Gate + Up + SiLU --- */
        { VkCommandBuffer cb2b=CB();
          RMS(cb2b,nt,HDIM,&B_Hid,&W_rf[l],&B_Hid2);
          BARRIER(cb2b);
          MM(cb2b,&W_Gate[l],&B_Hid2,&B_Gat,nt,IDIM,HDIM);
          MM(cb2b,&W_Up[l],&B_Hid2,&B_Up,nt,IDIM,HDIM);
          BARRIER(cb2b);
          SILU(cb2b,&B_Gat,&B_Up,&B_Dwn,nt*IDIM);
          Sub(cb2b); }
        if(do_diag&&l==0){buf_inv(&B_Dwn);buf_inv(&B_Gat);buf_inv(&B_Hid2);
         float*cpu=B_Hid.P,*w=W_rf[0].P;float ss=0;for(int i=0;i<HDIM;i++){float v=cpu[i];ss+=v*v;}float inv=1.0f/sqrtf(ss/(float)HDIM+1e-6f);
         float e11=cpu[0]*inv*w[0];LOGI("D11 RMSf err=%.2e",(double)fabsf(e11-B_Hid2.P[0]));
         float*gw=W_Gate[0].P;float cg=0;for(int k=0;k<HDIM;k++)cg+=gw[k]*B_Hid2.P[k];
         LOGI("D12 Gate err=%.2e",(double)fabsf(cg-B_Gat.P[0]));
         float*uw=W_Up[0].P;float cu=0;for(int k=0;k<HDIM;k++)cu+=uw[k]*B_Hid2.P[k];
         LOGI("D13 Up   err=%.2e",(double)fabsf(cu-B_Up.P[0]));
         float g0=B_Gat.P[0],u0=B_Up.P[0],silu=g0/(1.0f+expf(-g0));
         LOGI("D14 SiLU err=%.2e",(double)fabsf(silu*u0-B_Dwn.P[0]));}

        /* --- Submit 2c: Down projection + residual --- */
        { float ph0=B_Hid.P[0];
          VkCommandBuffer cb2c=CB();
          MM(cb2c,&W_Down[l],&B_Dwn,&B_Tmp,nt,HDIM,IDIM);
          BARRIER(cb2c);
          ADD(cb2c,&B_Hid,&B_Tmp,nt*HDIM);
          Sub(cb2c);
          if(do_diag&&l==0){float*dw=W_Down[0].P;float cd=0;for(int k=0;k<IDIM;k++)cd+=dw[k]*B_Dwn.P[k];
           LOGI("D15 Down err=%.2e",(double)fabsf(cd-B_Tmp.P[0]));
           LOGI("D16 ResF err=%.2e",(double)fabsf((ph0+B_Tmp.P[0])-B_Hid.P[0]));}
        }
    }
    if(nt==1)LOGI("L27hid: %.4f %.4f %.4f %.4f",(double)hidden[0],(double)hidden[1],(double)hidden[2],(double)hidden[3]);
    else LOGI("PREFILL L27hid(last): %.4f %.4f %.4f %.4f",
              (double)hidden[(nt-1)*HDIM + 0], (double)hidden[(nt-1)*HDIM + 1],
              (double)hidden[(nt-1)*HDIM + 2], (double)hidden[(nt-1)*HDIM + 3]);
    /* --- Final RMS + LM head --- */
    { VkCommandBuffer cbf=CB();
      RMS(cbf,nt,HDIM,&B_Hid,&W_Fnorm,&B_Hid2);
      BARRIER(cbf);
      if(nt>1){VkBufferCopy rgn={F32((uint32_t)(nt-1)*HDIM),0,F32(HDIM)};vkCmdCopyBuffer(cbf,B_Hid2.B,B_Tmp.B,1,&rgn);}
      Sub(cbf); }
    buf_inv(&B_Hid2);
    buf_inv(&B_Tmp);
    {
     const float *head_in = (nt>1) ? B_Tmp.P : B_Hid2.P;
     for(int v=0;v<VOCAB;v++){
      const float *w = W_Head.P + (VkDeviceSize)v*HDIM;
      float dot = 0.0f;
      for(int k=0;k<HDIM;k++) dot += w[k] * head_in[k];
     B_Log.P[v] = dot;
     }
    }
    {const float*logits=B_Log.P;
     int top5[5]={0};float top5v[5];
     for(int j=0;j<5;j++){float mx=-1e30f;int ti=0;
      for(int i=0;i<VOCAB;i++){bool dup=false;for(int k=0;k<j;k++)if(i==top5[k]){dup=true;break;}if(!dup&&logits[i]>mx){mx=logits[i];ti=i;}}
      top5[j]=ti;top5v[j]=mx;}
     memcpy(g_last_logits_top5,top5,sizeof(top5));memcpy(g_last_logits_top5_values,top5v,sizeof(top5v));
     if(nt==1)
     LOGI("LOGITS top5: %d(%.1f) %d(%.1f) %d(%.1f) %d(%.1f) %d(%.1f)",top5[0],(double)top5v[0],top5[1],(double)top5v[1],top5[2],(double)top5v[2],top5[3],(double)top5v[3],top5[4],(double)top5v[4]);}
    if(nt>1){const int*top5=g_last_logits_top5;const float*top5v=g_last_logits_top5_values;
     LOGI("PREFILL LOGITS top5: %d(%.1f) %d(%.1f) %d(%.1f) %d(%.1f) %d(%.1f)",top5[0],(double)top5v[0],top5[1],(double)top5v[1],top5[2],(double)top5v[2],top5[3],(double)top5v[3],top5[4],(double)top5v[4]);}
    pthread_mutex_unlock(&Mtx);return 0;}

const float*osh26_vk_gpu_logits(void){return B_Log.P;}
bool osh26_vk_gpu_ready(void){return vk_ok&&mdl_ok;}
void osh26_vk_gpu_free(void){
    buf_free(&B_Log);buf_free(&B_Tmp);buf_free(&B_Dwn);buf_free(&B_Up);buf_free(&B_Gat);
    buf_free(&B_Att);buf_free(&B_Sc);buf_free(&B_Vb);buf_free(&B_Kb);buf_free(&B_Qb);
    buf_free(&B_Hid2);buf_free(&B_Hid);buf_free(&B_KV);
    for(int l=0;l<N_LAY;l++){buf_free(&B_VCache[l]);buf_free(&B_KCache[l]);}
    buf_free(&W_Head);buf_free(&W_Fnorm);
    for(int l=0;l<N_LAY;l++){buf_free(&W_Down[l]);buf_free(&W_Up[l]);buf_free(&W_Gate[l]);buf_free(&W_rf[l]);buf_free(&W_Kn[l]);buf_free(&W_Qn[l]);buf_free(&W_O[l]);buf_free(&W_V[l]);buf_free(&W_K[l]);buf_free(&W_Q[l]);buf_free(&W_ra[l]);}
    if(Emb){free(Emb);Emb=NULL;}
    mdl_ok=false;
}
int osh26_vk_get_stats(struct osh26_vk_stats*o){if(!o)return -1;memset(o,0,sizeof(*o));o->ready=vk_ok;o->registered=mdl_ok;o->mnn_attention_enabled=g_mnn_attention_enabled;o->last_attention_max_abs_err=g_last_attention_max_abs_err;o->attention_fallback_layers=g_attention_fallback_layers;memcpy(o->last_logits_top5,g_last_logits_top5,sizeof(g_last_logits_top5));memcpy(o->last_logits_top5_values,g_last_logits_top5_values,sizeof(g_last_logits_top5_values));return 0;}
