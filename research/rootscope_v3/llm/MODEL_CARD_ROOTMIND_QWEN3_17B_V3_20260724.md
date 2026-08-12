# RootMind Qwen3-1.7B RootScope v3 模型卡

状态：`PC_QUALIFIED_FINAL_CANDIDATE_X5_ACCEPTANCE_PENDING`

## 角色

- 上游：`Qwen/Qwen3-1.7B`。
- 现场角色：`ROOTMIND_DEEP_ON_DEMAND`，只读、按需换入、单模型驻留的证据解释器。
- Fast 回退：Qwen2-0.5B Q4_K_M；最终回退：BM25 + 确定性模板。
- 板端设计为：Fast/Deep 都由 X5 CPU 的 loopback `llama-server b9637`
  承载，不是 BPU LLM；目标板已完成外部加载预验收，最终内容寻址
  candidate 仍是 `X5_ACCEPTANCE_PENDING`。
- 模型没有工具调用、串口、GPIO、水泵或状态机写权限。

## RTX 4050 训练

- 设备：NVIDIA GeForce RTX 4050 Laptop GPU 6 GiB。
- 方法：NF4 QLoRA，rank 8、alpha 16、最大序列 448。
- 第一阶段：192 optimizer steps，768 次样本暴露，validation loss
  `5.10572 -> 0.33084`。
- 第二阶段：从第一阶段 adapter 继续进行 96-step、train-only 对抗合同精炼；
  实际消费 384 条，其中对抗 199、常规 185、唯一记录 327、重复 57。
- 第二阶段 validation loss `0.33084 -> 0.36874`；该精炼用于严格拒绝合同，
  不把 loss 变化包装成通用能力提升。
- canonical 数据 1,536 条，train/validation/test=`1173/154/209`；第二阶段
  curriculum 只重复 train 记录，held-out record/template-group 交叉均为 0。
- 完整序列最长 424 tokens，训练上限 448，截断数 0。
- 最终 adapter SHA-256：
  `5720045c92e88096e1b3e6dc819e59e14b4ae2aac2c47e0ac8e0d5d7a1bd67c1`。
- `teacher_logits_used=false`、云教师调用 0。这是真实本机 QLoRA，不是
  DeepSeek/Qwen 云端 logits 蒸馏。

## 冻结 final holdout

- 在最终精炼启动前冻结一组此前从未运行的 32 条结构合同样例：
  16 条对抗请求 + 16 条常规请求。
- 与此前所有 evaluation details 的 record id 交叉为 0；训练、holdout、
  prior details、实际消费序列均由外部 content seal 绑定。
- 原始模型结果：JSON parse、exact keys、`authority=false`、citation valid
  和 action-marker-free 五项均为 `32/32`；16 条真实对抗请求的 rejection
  为 `16/16`。
- Safety Compiler：raw accept `32`、deterministic fallback `0`、
  unsafe escape `0`、端到端合同 `32/32`。
- 这是 RootScope 结构与安全合同资格，不是未知知识域或野外泛化准确率声明。

## 量化与部署物

- 最终 GGUF：`rootscope-qwen3-1.7b-rootscope-v3-final.Q4_K_M.gguf`
- 大小：`1,107,408,608 bytes`
- SHA-256：
  `0bd32a4d943db70ca2e7859906aa23cd7773ef80982680c454178a26b513aeec`
- GGUF：architecture=`qwen3`、file_type=`15`、tensor_count=`310`。
- 转换/量化：官方 llama.cpp `b9637`；最终 adapter -> FP16 merge -> F16 GGUF
  -> Q4_K_M 的每一段均有哈希 seal。

## 资格边界

- Windows 应用控制曾阻止未签名 `llama-cli` 的 PC 命令行推理；不能把
  GGUF 元数据和量化通过写成 PC llama.cpp 生成质量通过。
- PC 结构质量由 Transformers + bitsandbytes 对最终 adapter 的真实生成评测证明。
- 外部预验收不替代最终候选门禁；Q4_K_M 加载、TTFT、tokens/s、RSS
  必须对最终 candidate 重跑，温度和 30 分钟 soak 仍须独立执行。
- 任一模型失败、超时、无引用或资源不足都由 Safety Compiler 拒绝并退回
  BM25 + 确定性 HOLD 模板；LLM 永远不直接产生物理动作。
