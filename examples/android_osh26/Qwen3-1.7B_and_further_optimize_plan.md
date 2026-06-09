# Qwen3-1.7B 兼容与内存性能优化计划

## 总体策略

  采用最小风险顺序：

  1. 先建立严格真机正确性门禁。
  2. 在 Qwen3-0.6B 上实现 Q8_0 单份权重驻留。
  3. 将仅有差异的 hidden/FFN 维度参数化，启用 Qwen3-1.7B。
  4. 消除 llama.cpp CPU 模型/context 重复驻留。
  5. 再做 DEVICE_LOCAL、LM head 和提交调度优化。

  Qwen3-1.7B Q8_0 必须在 Redmi K40 真机通过；1.7B 全量模型可在加载前因内存不足明确拒绝。模型文件为"D:\下载\GPT-5-Distill-Qwen3-1.7B-Instruct.Q8_0.gguf"。

## 注意事项

  1. 严格按照步骤一步一步执行。
  2. 每次完成一个步骤严格按照要求进行**真机上的测试验收**（使用8000端口转发进行测试），同时如果性能有改动，补充性能数据到性能报告 D:\Programs\llama\examples\android_osh26\PERFORMANCE_NOTES.md 中（0.6B模型的数据接着写，1.7B模型的数据另开章节来对比）
  3. VULKAN 版本保持为1.1，不准引入更高版本的特性，一旦不兼容立刻回退
  4. 最小化改动，不准改动与当前步骤无关的代码，尽量复用已有的代码，不准重新造轮子
  5. 仔细想好改动的目的再改代码
  6. 每次改完代码自己先review一遍确保没有低级错误再继续
  7. 每次有大改动，保证验证通过后git commit
  8. 一旦多次实现都有问题，直接回退重来，多参考llama.cpp和MNN的源代码（在上级文件夹中）

## 分步实施

### 步骤 0：建立不可跳过的真机基线

  修改 examples/android_osh26/verify-vulkan-runtime.ps1：

- 支持模型矩阵：0.6B 全量、0.6B Q8_0、1.7B Q8_0。
- 每个模型先运行 llama.cpp CPU，再运行 Vulkan。
- 固定 temperature=0、top_p=1、固定 seed。
- 固定测试输入：
  - 英文单词指令。
  - 英文事实问答。
  - 中译英。
  - 中文算术。
  - 中文短句生成。
  - 简单代码生成。
  - 300 token 以上长 prompt。
  - 同一 prompt 连续运行三次。

- 保存每次请求的 token IDs、top-5、健康状态、TTFT、TPS和内存统计为 JSON。
- 任何断言失败立即返回非零退出码。

  修改 osh26_engine.cpp 的 debug correctness 路径：

- CPU 与 GPU 使用相同 prompt 和相同已选 token 前缀。
- 比较 prefill 后及前 8 个 decode 位置。
- 每个位置记录 CPU/GPU top-1、top-5、top-20及 margin。
- CPU 比较仅在 debug_correctness=true 时运行。

  统一正确性标准：

- attention_fallback_layers == 0。
- 不允许 NaN、非法 token、重复 token 候选或连续空白退化。
- CPU top-1 margin ≥ 1e-3 时，GPU top-1必须完全一致。
- margin < 1e-3 时，GPU token 必须位于 CPU top-5。
- 每个位置 top-5 overlap ≥ 4/5，top-20 overlap ≥ 18/20。
- 同一输入三次运行的完整 token ID 序列必须一致。
- 每个实现步骤完成后，都必须重新执行全部可用模型的上述真机测试。

### 步骤 1：0.6B Q8_0 单份权重驻留

  修改 osh26_vk_gpu.c 的模型加载区域：

- 将模型模式明确分为：
  - FULL：大型矩阵走现有 F32 计算路径。
  - Q8_0：所有大型矩阵必须是 GGUF Q8_0。

- 禁止 Q4、混合量化和加载时由 F32 转 Q8。
- Q8_0 模式直接调用 read_gguf_q8_packed()，不得先调用 LOAD_BUF()。
- Q8_0 模式不分配 W_Q/W_K/W_V/W_O/W_Gate/W_Up/W_Down。
- 任何大型矩阵不是 Q8_0 时，在分配大 buffer 前终止加载。
- 移除全局 q8_load_ok 的部分成功状态，加载失败时释放本次已分配的全部 buffer。
- RMSNorm、Q/K norm 保持 F32，因体积很小且不属于大型权重。

  同文件处理 embedding 和 LM head：

- embedding 以 Q8_0 packed 数据保存，按 token 行解量化到 B_Hid。
- tied LM head 与 embedding 共享同一份 Q8_0 权重。
- 存在独立 output.weight 时单独保存 Q8_0。
- LM head 先复用现有 W8A8 kernel，输入只动态量化一次，再按 shard 计算 top-k。
- CPU debug top-k 改为从 Q8_0 权重计算，不能依赖 F32 head。

  修改 osh26_vk_gpu.h：

- 增加模型存储模式、Q8/F32常驻字节数、embedding/head共享状态统计字段。

  本步真机门禁：

- 0.6B 全量和 Q8_0各运行全部测试输入。
- Q8_0 模式要求 resident_f32_matrix_bytes == 0。
- tied 模型要求 embedding_head_shared == true。
- 0.6B Q8_0进程内存目标不超过 2.2 GiB。
- 正确性必须满足步骤 0 全部标准。

### 步骤 2：参数化并启用 Qwen3-1.7B

  修改 osh26_vk_gpu.c 顶部固定参数：

- 新增内部 Osh26ModelSpec，保存：
  - hidden_dim
  - intermediate_dim
  - q_dim
  - kv_dim
  - layer_count
  - head_count
  - kv_head_count
  - head_dim
  - vocab_size
  - context_size

- 保持共同常量不动态化：两种模型均为28层、16 Q heads、8 KV heads、head dim 128、vocab 151936。
- 仅将真正不同的 HDIM 和 IDIM 替换为运行时 hidden_dim 和 intermediate_dim。
- QDIM/KVD由 spec 计算，所有 buffer 大小、矩阵 shape、循环跨度和 LM head stride 使用 spec。
- 静态层数组可继续保留28项，避免扩大重构范围。

  替换 validate_qwen3_06b_gguf()：

- 改为 load_qwen3_model_spec()。
- 仅接受两个精确 profile：
  - 0.6B：hidden 1024、FFN 3072。
  - 1.7B：hidden 2048、FFN 6144。

- 其他 Qwen3 profile 返回包含实际维度的明确错误。
- 在任何大内存分配前完成 metadata 和 tensor shape 验证。

  Shader 原则：

- 现有 matmul、RMS、RoPE、attention shader 已通过 push constants 接收尺寸，不为1.7B复制 shader。
- 仅修正 dispatch 和 buffer stride 的宿主参数。
- B_Q8In按 max(hidden_dim, intermediate_dim, q_dim)分配。

  本步真机门禁：

- 0.6B 全量、0.6B Q8_0、1.7B Q8_0均运行完整输入集。
- 1.7B Q8_0必须完成加载、prefill和至少16 token decode。
- 每个输入执行 CPU/GPU前8位置锁步比较。
- 1.7B Q8_0进程内存目标不超过4 GiB（不包括KV Cache）。
- 0.6B所有 token 和性能不得相对步骤1退化。

### 步骤 3：消除 llama.cpp CPU 重复驻留

  修改 osh26_engine.cpp 模型加载逻辑：

- model_params.use_mmap = true，避免完整复制 GGUF。
- Vulkan且debug_correctness=false时不创建 llama context。
- CPU backend或debug_correctness=true时才创建 CPU context。
- Vulkan生产路径继续保留 llama model/vocab，用于 tokenize、token text、EOG和sampler。
- CPU context改为延迟创建和及时释放。
- 1.7B全量模型在预计内存超过设备预算时，加载前返回明确错误，不进入部分分配状态。

  修改健康状态输出：

- 增加模型 profile、量化模式、CPU context 是否存在、mmap状态。
- 修正 kv_cache_device，准确显示是否存在 CPU KV。
- 暴露 Q8/F32权重、KV、activation、prefix cache和总 buffer 字节数。

  本步真机门禁：

- 三个模型路径重新运行完整输入集。
- Vulkan非 debug 模式必须显示 cpu_context_active=false。
- debug模式仍必须完成 CPU/GPU锁步比较。
- 相对步骤2，Q8模式内存必须明显下降，且 token 标准完全不降低。

### 步骤 4：权重迁移到 DEVICE_LOCAL

  修改 osh26_vk_gpu.c 的 VkBuf 和分配函数：

- 区分 immutable weight、host-visible staging、activation/KV buffer。
- Q8大型权重使用 DEVICE_LOCAL。
- 新增一个可复用 staging buffer，分块上传权重。
- 上传完成后立即释放或复用 staging，不长期映射权重。
- 激活、调试输出和少量常量暂时保持 HOST_VISIBLE，控制改动范围。
- 实现真实的 allocation/current/peak byte统计。

  本步真机门禁：

- 三种模型重新执行完整输入集。
- 所有正确性标准不变。
- 设备端连续加载/释放模型5次，不允许崩溃或持续增长。
- 连续运行30次请求，peak memory稳定。
- 中位 TTFT/TPS相对步骤3不得退化超过5%。

### 步骤 5：进一步性能优化

  修改 osh26_vk_gpu.c：

- 模型加载后预创建每层 descriptor set，不在每个算子调用时分配和更新。
- decode整层继续使用单 command buffer，并进一步减少跨层 submit。
- 将全局 BARRIER替换为精确的 buffer barrier，先只处理明确的生产者/消费者关系。
- Q8 activation 在 Q/K/V之间复用一次；gate/up之间复用一次。
- benchmark结果必须实际控制Q8性能状态，不再只记录。

  新增 gemv_q8_packed.comp，并修改 generate_spv_headers.ps1：

- 为 M=1 decode和LM head提供专用Q8 GEMV。
- 使用 subgroup reduction；不改变Q8_0存储格式。
- prefill继续使用现有Q8 GEMM。

  更新 PERFORMANCE_NOTES.md：

- 分别记录0.6B和1.7B的模型加载时间、常驻/峰值内存、TTFT、decode TPS、submit count和正确性结果。
- 只记录真实测量值，不填写推测数据。

  本步真机门禁：

- 完整正确性矩阵全部通过。
- 每个模型每个输入运行3次，使用中位数比较。
- 任何单项优化若造成token门禁失败，必须单独回滚。
- Q8 decode TPS不得低于步骤4。
- TTFT不得退化超过5%。
- submit count应下降；若未下降，该调度优化不合入。

## 最终验收

- Qwen3-0.6B全量和Q8_0均无功能回归。
- Qwen3-1.7B Q8_0可在真机稳定完成短、中文、代码和长prompt推理。
- Q8_0模式不存在任何大型F32权重副本。
- GPU生产模式不存在CPU context和CPU KV重复驻留。
- 不支持的量化格式及1.7B全量内存不足均在大分配前明确失败。
- 所有步骤都有对应真机JSON结果和PERFORMANCE_NOTES.md记录。
