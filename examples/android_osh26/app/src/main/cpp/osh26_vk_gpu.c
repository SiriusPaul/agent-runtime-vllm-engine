/* All-GPU inference (except attention). 2 Submits/layer.
   Pool: HOST_VISIBLE|HOST_COHERENT VkBuffer. CPU does KV cache + attention. */

#include "osh26_vk_gpu.h"
#include "ggml.h"
#include "gguf.h"
#include <android/log.h>
#include <dlfcn.h>
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

static struct { VkBuffer B;VkDeviceMemory M;float*P;VkDeviceSize S,U; } GP;
static VkDeviceSize g_max_sbr=256*1024*1024;
static uint32_t A(size_t b){GP.U=(GP.U+255)&~(VkDeviceSize)255;uint32_t o=(uint32_t)(GP.U/4);GP.U+=b;return o;}
static float* Px(uint32_t o){return GP.P+o;}

static VkInstance V;static VkDevice D;static VkQueue Q;static VkCommandPool CP;static VkDescriptorPool DP;
static VkDescriptorSetLayout DSL;static VkPipelineLayout PL;
static VkPipeline P_MMt,P_MMr,P_RMS,P_RoPE,P_SMax,P_SiLU,P_Add;
static uint32_t QFI;static VkPhysicalDeviceMemoryProperties MP;
static pthread_mutex_t Mtx=PTHREAD_MUTEX_INITIALIZER;
static bool vk_ok,mdl_ok;
static uint32_t MT(uint32_t b,VkMemoryPropertyFlags f){for(uint32_t i=0;i<MP.memoryTypeCount;i++)if((b&(1u<<i))&&(MP.memoryTypes[i].propertyFlags&f)==f)return i;return UINT32_MAX;}

static struct {uint32_t ra,Q,K,V,O,rf,Gate,Up,Down;} L[N_LAY];
static uint32_t Fnorm,Head;static float*Emb;
static uint32_t KV,Hid,Hid2,Qb,Kb,Vb,Sc,Att,Gat,Up_,Dwn,Tmp,Log;
static VkCommandBuffer CB(void){VkCommandBuffer cb;VkCommandBufferAllocateInfo a={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,0,CP,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1};vkAllocateCommandBuffers(D,&a,&cb);VkCommandBufferBeginInfo b={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,0,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,0};vkBeginCommandBuffer(cb,&b);return cb;}
static VkDescriptorSet g_ds[16];static int g_n;
static void Sub(VkCommandBuffer cb){vkEndCommandBuffer(cb);VkFence f;VkFenceCreateInfo fi={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,0,0};vkCreateFence(D,&fi,0,&f);VkSubmitInfo s={VK_STRUCTURE_TYPE_SUBMIT_INFO,0,0,0,0,1,&cb,0,0};vkQueueSubmit(Q,1,&s,f);vkWaitForFences(D,1,&f,VK_TRUE,UINT64_MAX);vkDestroyFence(D,f,0);vkFreeCommandBuffers(D,CP,1,&cb);if(g_n>0){vkFreeDescriptorSets(D,DP,(uint32_t)g_n,g_ds);g_n=0;}}

// Allocate descriptor set, bind 1-3 buffers at pool offsets, push 16B constants
static void BIND(VkCommandBuffer cb,uint32_t b0,uint32_t b1,uint32_t b2,const uint32_t pc[4]){
    VkDescriptorSet ds=0;VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,DP,1,&DSL};
    if(vkAllocateDescriptorSets(D,&da,&ds)){vkResetDescriptorPool(D,DP,0);vkAllocateDescriptorSets(D,&da,&ds);}
    if(g_n<16)g_ds[g_n++]=ds;
    VkDeviceSize r0=GP.S-F32(b0),r1=GP.S-F32(b1),r2=GP.S-F32(b2);if(r0>g_max_sbr)r0=g_max_sbr;if(r1>g_max_sbr)r1=g_max_sbr;if(r2>g_max_sbr)r2=g_max_sbr;VkDescriptorBufferInfo bi[3]={{GP.B,F32(b0),r0},{GP.B,F32(b1),r1},{GP.B,F32(b2),r2}};
    VkWriteDescriptorSet wr[3];memset(wr,0,sizeof(wr));
    for(int j=0;j<3;j++){wr[j].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;wr[j].dstSet=ds;wr[j].dstBinding=j;wr[j].descriptorCount=1;wr[j].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;wr[j].pBufferInfo=&bi[j];}
    vkUpdateDescriptorSets(D,3,wr,0,0);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,PL,0,1,&ds,0,0);
    vkCmdPushConstants(cb,PL,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc);
}
#define BIND1(cb,a,pc)   BIND(cb,a,0,0,pc)

// Shorthand dispatchers (record into current command buffer)
// Dispatch macros (all record into `cb`)
#define RMS(cb,rows,cols,x,w,y) do{uint32_t p[4]={rows,cols,0,0};BIND((cb),x,w,y,p);vkCmdBindPipeline((cb),VK_PIPELINE_BIND_POINT_COMPUTE,P_RMS);vkCmdDispatch((cb),((rows)+63)/64,1,1);}while(0)
#define BARRIER(cb) do{VkMemoryBarrier mb_={VK_STRUCTURE_TYPE_MEMORY_BARRIER,0,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT};vkCmdPipelineBarrier((cb),VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb_,0,0,0,0);}while(0)
#define MM(cb,w,x,y,M,N,K) do{uint32_t p[4]={M,N,K,K};BIND((cb),w,x,y,p);if((M)<=4){vkCmdBindPipeline((cb),VK_PIPELINE_BIND_POINT_COMPUTE,P_MMr);vkCmdDispatch((cb),(N),(M),1);}else{vkCmdBindPipeline((cb),VK_PIPELINE_BIND_POINT_COMPUTE,P_MMt);vkCmdDispatch((cb),((N)+7)/8,((M)+7)/8,1);}}while(0)
#define ROPE(cb,x,nt,nh,posv,hd) do{uint32_t p[4]={nt,nh,(uint32_t)(posv),hd};BIND1((cb),x,p);vkCmdBindPipeline((cb),VK_PIPELINE_BIND_POINT_COMPUTE,P_RoPE);uint32_t tot_=(nt)*(nh)*((hd)/2);vkCmdDispatch((cb),(tot_+63)/64,1,1);}while(0)
#define ADD(cb,y,x,n) do{uint32_t p[4]={n,0,0,0};BIND((cb),y,x,0,p);vkCmdBindPipeline((cb),VK_PIPELINE_BIND_POINT_COMPUTE,P_Add);vkCmdDispatch((cb),((n)+63)/64,1,1);}while(0)
#define SILU(cb,a,b,z,n) do{uint32_t p[4]={n,0,0,0};BIND((cb),a,b,z,p);vkCmdBindPipeline((cb),VK_PIPELINE_BIND_POINT_COMPUTE,P_SiLU);vkCmdDispatch((cb),((n)+63)/64,1,1);}while(0)

// ---- CPU: attention (complex GQA, keep on CPU for now) ----
static void cpu_attn(float*qb,float*kb,float*vb,float*kv,float*sc,float*att,int nt,int pos){
    float isd=1.0f/sqrtf((float)HD);int slen=pos+nt;
    for(int l=0;l<N_LAY;l++){int kvo=l*2*MAX_S*KVD;
        memset(sc,0,nt*N_HD*MAX_S*sizeof(float));
        for(int h=0;h<N_HD;h++){int kvh=h*N_KVH/N_HD;
            for(int t=0;t<nt;t++){const float*qt=qb+t*QDIM+h*HD;float*sr=sc+(t*N_HD+h)*MAX_S;
                int lim=(nt==1)?slen:(pos+t+1);
                for(int s=0;s<lim;s++){const float*ks=kv+kvo+s*KVD+kvh*HD;float dot=0;
                    for(int d=0;d<HD;d++)dot+=qt[d]*ks[d];sr[s]=dot*isd;}}}
        for(int r=0;r<nt*N_HD;r++){float*row=sc+r*MAX_S;int nc=(nt==1)?slen:(pos+(r/N_HD)+1);
            float mx=row[0];for(int c=1;c<nc;c++)if(row[c]>mx)mx=row[c];float sum=0;
            for(int c=0;c<nc;c++){row[c]=expf(row[c]-mx);sum+=row[c];}
            for(int c=0;c<nc;c++)row[c]/=sum;}
        memset(att,0,nt*QDIM*sizeof(float));
        for(int h=0;h<N_HD;h++){int kvh=h*N_KVH/N_HD;
            for(int t=0;t<nt;t++){float*oh=att+t*QDIM+h*HD;const float*sr=sc+(t*N_HD+h)*MAX_S;
                for(int s=0;s<slen;s++){const float*vs=kv+kvo+MAX_S*KVD+s*KVD+kvh*HD;float wgt=sr[s];
                    for(int d=0;d<HD;d++)oh[d]+=wgt*vs[d];}}}
        // This is wrong — cpu_attn handles ALL layers at once. Need per-layer.
    }
}

// ---- Init ----
int osh26_vk_gpu_init(void){if(vk_ok)return 0;
    if(!InitVulkan()){LOGE("InitVulkan failed");return -1;}
    VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO,0,"OSH26",1,"OSH26",1,VK_API_VERSION_1_1};
    VkInstanceCreateInfo ci={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,0,0,&ai,0,0,0,0};
    if(vkCreateInstance(&ci,0,&V))return -1;
    uint32_t nd=0;vkEnumeratePhysicalDevices(V,&nd,0);VkPhysicalDevice*pd=calloc(nd,sizeof(*pd));vkEnumeratePhysicalDevices(V,&nd,pd);VkPhysicalDevice ph=pd[0];free(pd);
    VkPhysicalDeviceProperties pdp;vkGetPhysicalDeviceProperties(ph,&pdp);vkGetPhysicalDeviceMemoryProperties(ph,&MP);g_max_sbr=pdp.limits.maxStorageBufferRange;LOGI("GPU: %s (maxSBR=%zuMB)",pdp.deviceName,(size_t)(g_max_sbr/1024/1024));
    uint32_t qn=0;vkGetPhysicalDeviceQueueFamilyProperties(ph,&qn,0);VkQueueFamilyProperties*qp=calloc(qn,sizeof(*qp));vkGetPhysicalDeviceQueueFamilyProperties(ph,&qn,qp);
    for(uint32_t i=0;i<qn;i++)if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){QFI=i;break;}free(qp);
    float pr=1;VkDeviceQueueCreateInfo dq={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,0,0,QFI,1,&pr};
    VkDeviceCreateInfo dc={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,0,0,1,&dq,0,0,0,0};
    if(vkCreateDevice(ph,&dc,0,&D))return -1;vkGetDeviceQueue(D,QFI,0,&Q);
    VkCommandPoolCreateInfo cp={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,0,VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,QFI};vkCreateCommandPool(D,&cp,0,&CP);
    VkDescriptorPoolSize ds={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4096};
    VkDescriptorPoolCreateInfo dp={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,0,VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,1024,1,&ds};vkCreateDescriptorPool(D,&dp,0,&DP);
    VkDescriptorSetLayoutBinding bd[3];for(int i=0;i<3;i++)bd[i]=(VkDescriptorSetLayoutBinding){i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
    VkDescriptorSetLayoutCreateInfo dl={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,3,bd};vkCreateDescriptorSetLayout(D,&dl,0,&DSL);
    VkPushConstantRange pc={VK_SHADER_STAGE_COMPUTE_BIT,0,16};
    VkPipelineLayoutCreateInfo pl={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&DSL,1,&pc};vkCreatePipelineLayout(D,&pl,0,&PL);
    { VkShaderModuleCreateInfo s={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,0,0,0,0};VkShaderModule m;
      s.codeSize=examples_android_osh26_app_src_main_cpp_mulmat_tiled_spv_len;s.pCode=(const uint32_t*)examples_android_osh26_app_src_main_cpp_mulmat_tiled_spv;vkCreateShaderModule(D,&s,0,&m);VkComputePipelineCreateInfo pi={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,0,0,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,m,"main",0},PL,0,(uint32_t)-1};vkCreateComputePipelines(D,0,1,&pi,0,&P_MMt);
      s.codeSize=examples_android_osh26_app_src_main_cpp_mulmat_reduce_spv_len;s.pCode=(const uint32_t*)examples_android_osh26_app_src_main_cpp_mulmat_reduce_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_MMr);
      s.codeSize=_tmp_rms_norm_spv_len;s.pCode=(const uint32_t*)_tmp_rms_norm_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_RMS);
      s.codeSize=_tmp_rope_spv_len;s.pCode=(const uint32_t*)_tmp_rope_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_RoPE);
      s.codeSize=_tmp_softmax_gpu_spv_len;s.pCode=(const uint32_t*)_tmp_softmax_gpu_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_SMax);
      s.codeSize=_tmp_silu_mul_spv_len;s.pCode=(const uint32_t*)_tmp_silu_mul_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_SiLU);
      s.codeSize=_tmp_add_spv_len;s.pCode=(const uint32_t*)_tmp_add_spv;vkCreateShaderModule(D,&s,0,&m);pi.stage.module=m;vkCreateComputePipelines(D,0,1,&pi,0,&P_Add); }
    vk_ok=true;LOGI("GPU ready (7 pipelines)");return 0;}

// ---- Model loading (unchanged) ----
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
int osh26_vk_gpu_load_model(const char*path){if(!vk_ok||!path)return -1;
    FILE*f=fopen(path,"rb");if(!f){LOGE("open %s",path);return -1;}
    struct gguf_init_params gp={true,NULL};struct gguf_context*gctx=gguf_init_from_file(path,gp);if(!gctx){fclose(f);return -1;}
    Emb=(float*)malloc(F32(VOCAB*HDIM));if(!read_gguf(gctx,f,"token_embd.weight",Emb)){free(Emb);gguf_free(gctx);fclose(f);return -1;}
    VkDeviceSize wb=0;for(int l=0;l<N_LAY;l++)wb+=F32(HDIM+QDIM*HDIM+KVD*HDIM+KVD*HDIM+HDIM*QDIM+HDIM+IDIM*HDIM+IDIM*HDIM+HDIM*IDIM);wb+=F32(HDIM+VOCAB*HDIM);
    VkDeviceSize kvb=F32(N_LAY*2*MAX_S*KVD);
    VkDeviceSize ab=F32(MAX_S*HDIM*2+MAX_S*QDIM+MAX_S*KVD*2+N_HD*MAX_S*MAX_S+MAX_S*QDIM+MAX_S*IDIM*2+MAX_S*IDIM+MAX_S*HDIM+VOCAB);
    VkDeviceSize total=(VkDeviceSize)((double)(wb+kvb+ab)*1.15);
    LOGI("Pool: %.0fMB +15%% = %.0fMB",(double)(wb+kvb+ab)/1e6,(double)total/1e6);
    GP.S=total;VkBufferCreateInfo bci={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,0,0,total,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_SHARING_MODE_EXCLUSIVE,0,0};
    if(vkCreateBuffer(D,&bci,0,&GP.B)){LOGE("vkCreateBuffer");free(Emb);gguf_free(gctx);fclose(f);return -1;}
    VkMemoryRequirements mr;vkGetBufferMemoryRequirements(D,GP.B,&mr);uint32_t mt=MT(mr.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if(mt==UINT32_MAX){free(Emb);gguf_free(gctx);fclose(f);return -1;}
    VkMemoryAllocateInfo mai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,0,mr.size,mt};
    if(vkAllocateMemory(D,&mai,0,&GP.M)){free(Emb);gguf_free(gctx);fclose(f);return -1;}
    vkBindBufferMemory(D,GP.B,GP.M,0);vkMapMemory(D,GP.M,0,total,0,(void**)&GP.P);
#define LOAD(name,off,nelem) do{off=A(F32(nelem));if(!read_gguf(gctx,f,name,Px(off))){free(Emb);gguf_free(gctx);fclose(f);return -1;}}while(0)
    // Emb already loaded to CPU via read_gguf above; not in GPU pool
    char n[128];for(int l=0;l<N_LAY;l++){snprintf(n,sizeof(n),"blk.%d.attn_norm.weight",l);LOAD(n,L[l].ra,HDIM);snprintf(n,sizeof(n),"blk.%d.attn_q.weight",l);LOAD(n,L[l].Q,QDIM*HDIM);snprintf(n,sizeof(n),"blk.%d.attn_k.weight",l);LOAD(n,L[l].K,KVD*HDIM);snprintf(n,sizeof(n),"blk.%d.attn_v.weight",l);LOAD(n,L[l].V,KVD*HDIM);snprintf(n,sizeof(n),"blk.%d.attn_output.weight",l);LOAD(n,L[l].O,HDIM*QDIM);snprintf(n,sizeof(n),"blk.%d.ffn_norm.weight",l);LOAD(n,L[l].rf,HDIM);snprintf(n,sizeof(n),"blk.%d.ffn_gate.weight",l);LOAD(n,L[l].Gate,IDIM*HDIM);snprintf(n,sizeof(n),"blk.%d.ffn_up.weight",l);LOAD(n,L[l].Up,IDIM*HDIM);snprintf(n,sizeof(n),"blk.%d.ffn_down.weight",l);LOAD(n,L[l].Down,HDIM*IDIM);}
    LOAD("output_norm.weight",Fnorm,HDIM);
    if(gguf_find_tensor(gctx,"output.weight")>=0)LOAD("output.weight",Head,VOCAB*HDIM);
    else{Head=A(F32(VOCAB*HDIM));memcpy(Px(Head),Emb,F32(VOCAB*HDIM));}
#undef LOAD
    gguf_free(gctx);fclose(f);
    KV=A(kvb);memset(Px(KV),0,(size_t)kvb);Hid=A(F32(MAX_S*HDIM));Hid2=A(F32(MAX_S*HDIM));Qb=A(F32(MAX_S*QDIM));Kb=A(F32(MAX_S*KVD));Vb=A(F32(MAX_S*KVD));Sc=A(F32(N_HD*MAX_S*MAX_S));Att=A(F32(MAX_S*QDIM));Gat=A(F32(MAX_S*IDIM));Up_=A(F32(MAX_S*IDIM));Dwn=A(F32(MAX_S*IDIM));Tmp=A(F32(MAX_S*HDIM));Log=A(F32(VOCAB));
    mdl_ok=true;LOGI("Model loaded: %.1f/%.1f MB",(double)GP.U/1e6,(double)GP.S/1e6);return 0;}

// ---- Forward pass (2 Submits/layer) ----
int osh26_vk_gpu_forward(const int*tokens,int nt,int pos){if(!mdl_ok)return -1;static int fc=0;if(++fc<=3)LOGI("forward#%d nt=%d pos=%d",fc,nt,pos);pthread_mutex_lock(&Mtx);
    float*hidden=Px(Hid),*hnorm=Px(Hid2),*qb=Px(Qb),*kb=Px(Kb),*vb=Px(Vb),*sc=Px(Sc),*att=Px(Att),*gate=Px(Gat),*up=Px(Up_),*dwn=Px(Dwn),*kv=Px(KV),*tmp=Px(Tmp);
    for(int i=0;i<nt;i++){int tok=tokens[i];if(tok<0||tok>=VOCAB)tok=0;memcpy(hidden+i*HDIM,Emb+tok*HDIM,HDIM*sizeof(float));}
    LOGI("DIAG emb: tok[0]=%d hid[0..3]=%.4f %.4f %.4f %.4f",tokens[0],(double)hidden[0],(double)hidden[1],(double)hidden[2],(double)hidden[3]);
    for(int l=0;l<N_LAY;l++){
        // ---- Submit 1: RMS + Q,K,V + RoPE ----
        { VkCommandBuffer cb1=CB();
          {uint32_t p[4]={nt,HDIM,0,0};BIND(cb1,Hid,L[l].ra,Hid2,p);vkCmdBindPipeline(cb1,VK_PIPELINE_BIND_POINT_COMPUTE,P_RMS);vkCmdDispatch(cb1,(uint32_t)nt,1,1);}
          BARRIER(cb1);
          MM(cb1,L[l].Q,Hid2,Qb,nt,QDIM,HDIM);
          MM(cb1,L[l].K,Hid2,Kb,nt,KVD,HDIM);
          MM(cb1,L[l].V,Hid2,Vb,nt,KVD,HDIM);
          BARRIER(cb1);
          {uint32_t p[4]={nt,N_HD,(uint32_t)pos,HD};BIND1(cb1,Qb,p);vkCmdBindPipeline(cb1,VK_PIPELINE_BIND_POINT_COMPUTE,P_RoPE);uint32_t tot=nt*N_HD*(HD/2);vkCmdDispatch(cb1,(tot+63)/64,1,1);}
          {uint32_t p[4]={nt,N_KVH,(uint32_t)pos,HD};BIND1(cb1,Kb,p);vkCmdBindPipeline(cb1,VK_PIPELINE_BIND_POINT_COMPUTE,P_RoPE);uint32_t tot=nt*N_KVH*(HD/2);vkCmdDispatch(cb1,(tot+63)/64,1,1);}
          Sub(cb1); }
        if(l==0&&nt==1){LOGI("DIAG L0 RMS ok: hnorm[0..3]=%.4f %.4f %.4f %.4f",(double)hnorm[0],(double)hnorm[1],(double)hnorm[2],(double)hnorm[3]);}

        // CPU: KV cache + Attention
        { int kvo=l*2*MAX_S*KVD;memcpy(kv+kvo+pos*KVD,kb,nt*KVD*sizeof(float));memcpy(kv+kvo+MAX_S*KVD+pos*KVD,vb,nt*KVD*sizeof(float));
          int slen=pos+nt;memset(sc,0,nt*N_HD*MAX_S*sizeof(float));
          float isd=1.0f/sqrtf((float)HD);
          for(int h=0;h<N_HD;h++){int kvh=h*N_KVH/N_HD;for(int t=0;t<nt;t++){const float*qt=qb+t*QDIM+h*HD;float*sr=sc+(t*N_HD+h)*MAX_S;int lim=(nt==1)?slen:(pos+t+1);for(int s=0;s<lim;s++){const float*ks=kv+kvo+s*KVD+kvh*HD;float dot=0;for(int d=0;d<HD;d++)dot+=qt[d]*ks[d];sr[s]=dot*isd;}}}
          for(int r=0;r<nt*N_HD;r++){float*row=sc+r*MAX_S;int nc=(nt==1)?slen:(pos+(r/N_HD)+1);float mx=row[0];for(int c=1;c<nc;c++)if(row[c]>mx)mx=row[c];float sum=0;for(int c=0;c<nc;c++){row[c]=expf(row[c]-mx);sum+=row[c];}for(int c=0;c<nc;c++)row[c]/=sum;}
          memset(att,0,nt*QDIM*sizeof(float));for(int h=0;h<N_HD;h++){int kvh=h*N_KVH/N_HD;for(int t=0;t<nt;t++){float*oh=att+t*QDIM+h*HD;const float*sr=sc+(t*N_HD+h)*MAX_S;for(int s=0;s<slen;s++){const float*vs=kv+kvo+MAX_S*KVD+s*KVD+kvh*HD;float wgt=sr[s];for(int d=0;d<HD;d++)oh[d]+=wgt*vs[d];}}} }

        // ---- Submit 2: O + residual + RMS_FFN + Gate + Up + SiLU + Down + residual ----
        { VkCommandBuffer cb2=CB();
          MM(cb2,L[l].O,Att,Tmp,nt,HDIM,QDIM);
          BARRIER(cb2);
          {uint32_t p[4]={nt*HDIM,0,0,0};BIND(cb2,Hid,Tmp,0,p);vkCmdBindPipeline(cb2,VK_PIPELINE_BIND_POINT_COMPUTE,P_Add);vkCmdDispatch(cb2,(nt*HDIM+63)/64,1,1);}
          BARRIER(cb2);
          {uint32_t p[4]={nt,HDIM,0,0};BIND(cb2,Hid,L[l].rf,Hid2,p);vkCmdBindPipeline(cb2,VK_PIPELINE_BIND_POINT_COMPUTE,P_RMS);vkCmdDispatch(cb2,(uint32_t)nt,1,1);}
          BARRIER(cb2);
          MM(cb2,L[l].Gate,Hid2,Gat,nt,IDIM,HDIM);
          MM(cb2,L[l].Up,Hid2,Up_,nt,IDIM,HDIM);
          {uint32_t p[4]={nt*IDIM,0,0,0};BIND(cb2,Gat,Up_,Dwn,p);vkCmdBindPipeline(cb2,VK_PIPELINE_BIND_POINT_COMPUTE,P_SiLU);vkCmdDispatch(cb2,(nt*IDIM+63)/64,1,1);}
          BARRIER(cb2);
          MM(cb2,L[l].Down,Dwn,Tmp,nt,HDIM,IDIM);
          BARRIER(cb2);
          {uint32_t p[4]={nt*HDIM,0,0,0};BIND(cb2,Hid,Tmp,0,p);vkCmdBindPipeline(cb2,VK_PIPELINE_BIND_POINT_COMPUTE,P_Add);vkCmdDispatch(cb2,(nt*HDIM+63)/64,1,1);}
          Sub(cb2); }
        if(l==0&&nt==1){LOGI("DIAG L0 FFN ok: hidden[0..3]=%.4f %.4f %.4f %.4f",(double)hidden[0],(double)hidden[1],(double)hidden[2],(double)hidden[3]);}
    }
    { VkCommandBuffer cbf=CB();
      {uint32_t p[4]={nt,HDIM,0,0};BIND(cbf,Hid,Fnorm,Hid2,p);vkCmdBindPipeline(cbf,VK_PIPELINE_BIND_POINT_COMPUTE,P_RMS);vkCmdDispatch(cbf,(uint32_t)nt,1,1);}
      BARRIER(cbf);
      MM(cbf,Head,Hid2+(nt-1)*HDIM,Log,1,VOCAB,HDIM);
      Sub(cbf); }
    pthread_mutex_unlock(&Mtx);return 0;}

const float*osh26_vk_gpu_logits(void){return Px(Log);}
bool osh26_vk_gpu_ready(void){return vk_ok&&mdl_ok;}
void osh26_vk_gpu_free(void){if(GP.P){vkUnmapMemory(D,GP.M);GP.P=NULL;}if(GP.M){vkFreeMemory(D,GP.M,0);}if(GP.B){vkDestroyBuffer(D,GP.B,0);}if(Emb){free(Emb);Emb=NULL;}mdl_ok=false;}
int osh26_vk_get_stats(struct osh26_vk_stats*o){if(!o)return -1;memset(o,0,sizeof(*o));o->ready=vk_ok;o->registered=mdl_ok;return 0;}
