# 01_cover

1. OSH26 Runtime

开场直接讲：我们的贡献不是模型算法，而是把本地模型做成 Android 上可部署、可观测、可调度的端侧系统。


# 02_endpoint_importance

2. 为什么端侧部署重要

把端侧部署的价值讲清楚：不是为了炫技，而是 Agent 要接触本地系统能力，云端不能替代。


# 03_runtime_problem

3. 端侧推理是 Runtime 问题

这一页把问题从模型拉回系统：端侧部署要有调度、内存、可观测和恢复机制。


# 04_llamacpp_boundary

4. 与 llama.cpp 的边界

强调边界：我们不是重复造 tokenizer 或模型格式，而是围绕 Android Agent 场景做系统层增强。


# 05_git_roadmap

5. 优化路线图

这一页直接引用 git 历史脉络，告诉老师我们做了很多系统工程，不是只做页面包装。


# 06_architecture

6. 总体架构

把系统拆开：上层是 Agent 可调用的服务，下层是可控的 GPU/CPU 执行和 KV 内存。


# 07_android_lifecycle

7. Android 生命周期闭环

强调 Android 集成很实际：UI、JNI、native、HTTP、取消、释放、健康状态都在同一个闭环。


# 08_opencl_failure

8. 为什么不用 OpenCL

讲 OpenCL 的失败时要讲系统边界：linker namespace、vendor 依赖、部署性，不是单纯性能选择。


# 09_native_vulkan_failure

9. 为什么不用原生 ggml Vulkan

强调原生 Vulkan 的风险：不是跑不起来那么简单，而是跑起来后输出不可信。


# 10_custom_gpu_runtime

10. 自研 GPU Runtime

这页正面讲重写后端：不是为了炫耀，而是原生路径在目标设备上不可控。


# 11_memory_architecture

11. 统一内存架构

把内存架构讲成 OS 课知识：控制面/数据面，工作集，设备驻留，减少复制。


# 12_kv_cache_os

12. KV Cache 管理

这一页把 KV cache 和 OS 里的虚拟内存/页缓存类比起来，是课程知识的自然融入。


# 13_stateless_subagent

13. Agent 专用 Prefix Cache

这页体现针对 Agent 做了专门优化，不是通用聊天缓存。


# 14_scheduler

14. 请求调度

把调度讲实：不是只有 KV cache，还有队列、取消、背压、aging。


# 15_q8_prefill

15. Q8 Prefill

用实测数据讲 Q8 prefill。注意强调 first-token，而不是泛泛讲吞吐。


# 16_decode_gemv

16. Decode Q8 GEMV

git 历史里 decode Q8 GEMV 是明显增益点，要单独给一页。


# 17_lm_head_descriptor

17. LM head 与 descriptor cache

讲系统指标：submit、descriptor、GPU timestamp，这些都是 OS/驱动边界上的优化。


# 18_context_8k

18. 8K Context

8K context 要讲成 Agent memory 需求，结合 chunking 和内存工作集。


# 19_model_residency

19. 权重驻留与 1.7B

结合 git 的 Step 1/2/3 讲内存驻留，体现不是只做 0.6B demo。


# 20_benchmark_gates

20. Benchmark Gates

这一页证明我们有工程闭环，而不是只跑了一次样例。


# 21_comparison

21. 框架对比

对比要讲清楚我们的定位：不是全能框架，而是端侧 Agent 这条链路打穿。


# 22_action_fabric_link

22. 与 Action Fabric 合流

和前面 Action Fabric 结合，说明端侧部署不是孤立部分。


# 23_teacher_takeaway

23. 老师应该看到什么

这页可以作为答辩前的总结：课程知识点都自然嵌在系统实现里。


# 24_closing

24. 结尾

结尾回到主张：端侧 AI 框架的重点是部署系统能力，而不是单次推理。
