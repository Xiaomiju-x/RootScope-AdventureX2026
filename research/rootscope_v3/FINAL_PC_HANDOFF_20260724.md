# RootScope v3 最终 PC 交接

状态：`PC_COMPLETE_X5_FINAL_CANDIDATE_ACCEPTANCE_PENDING`

## 1. 交付结论

RootScope v3 的 PC 侧算法、模型、知识库、安全合同、离线依赖、发布构建与板端
资格脚本已经收口。指定 RDK X5 4GB 已经上电并完成固定 `rootscope-x5`
身份核验与外部预验收；下一步是把重新构建的唯一内容寻址 candidate
StageOnly 上传，再对该候选执行板端离线软件验收与原子激活。

外部预验收不是最终候选验收。本结论不表示最终 candidate 已激活、
实时相机已通过或水泵闭环已完成。

## 2. 最终算法栈

### RootSight

- 植物类别证据 + 置信/OOD/HOLD；
- 黄光、固定曝光/白平衡和未知卡门禁已写入板端资格流程；
- before/after 湿润变化、目标格/邻格选择性与跨域交叉检查；
- BPU 以 canonical `hrt_model_exec` 为数值 oracle，并使用 persistent native
  `libdnn` worker 做 43 样本单次加载回放。外部预验收记录不冒充最终
  candidate acceptance；Python `hbm_runtime` 只保留为非权威负路径观测。

PC 回执只证明冻结静态 CPU 参考 4/4 和 2 个 deterministic wetting fixture；
`cpu_bpu_compared=0`，live camera 与最终 candidate BPU 资格仍待重新验收。

### RootMind

- Fast：Qwen2-0.5B Q4_K_M；
- Deep：RootScope Qwen3-1.7B Q4_K_M；
- 最终回退：BM25 + deterministic HOLD template；
- `ONE_MODEL_AT_A_TIME`；外部预验收已经证明模型可在目标板加载，最终
  candidate 仍须让 Fast/Deep 在 X5 CPU loopback `llama-server` 重跑；
- Safety Compiler 采用 reject-and-deterministically-replace，LLM 无 tool/serial/
  GPIO/pump/state-machine authority。

最终未见 holdout：16 对抗 + 16 常规；JSON、字段、零权限、引用和动作标记
五项均 32/32，对抗请求拒绝为 16/16。Safety Compiler raw accept=32、fallback=0、
unsafe escape=0。

### RAG2 与 Plant2Action

- 17 个来源、42 chunks、64 gold、36 forbidden；
- BM25 PC R@5 92.19%，hard top3 84.09%，forbidden R@5 94.44%，
  citation escape 0；
- 默认 runtime 为 `bm25_runtime.py`，Dense challenger 未晋级、不打包；
- Plant2Action 只生成有上限、带证据与 reason code 的 proposal；
- Resource Broker、System Coordinator 和 19 类故障矩阵 fail-closed。

## 3. 最终模型

- adapter SHA-256：
  `5720045c92e88096e1b3e6dc819e59e14b4ae2aac2c47e0ac8e0d5d7a1bd67c1`
- Q4_K_M：
  `rootscope_v3/models/llm/rootscope-qwen3-1.7b-rootscope-v3-final.Q4_K_M.gguf`
- bytes：`1,107,408,608`
- SHA-256：
  `0bd32a4d943db70ca2e7859906aa23cd7773ef80982680c454178a26b513aeec`
- GGUF：Qwen3、file type 15、310 tensors。

训练是真实 RTX4050 staged QLoRA：192-step 基础 SFT + 96-step train-only
对抗合同精炼；teacher logits 未使用、云教师调用为 0。训练、holdout、最终评测、
merge 和 GGUF 均有独立 content seal。

## 4. 最终候选部署顺序

1. PC 保持当前网络设置不变；用户确认 X5 与 PC 同一现场 LAN。
2. 固定 SSH alias `rootscope-x5` 必须通过 StrictHostKeyChecking、ED25519
   主机键以及 hostname/machine-id/serial/WLAN MAC/`aarch64` 精确身份核验。
3. 在最终 release 目录执行 StageOnly：

```powershell
.\tools\release_v3\deploy_rootscope_v3_to_x5.ps1 `
  -SshAlias rootscope-x5 `
  -ReleaseDirectory <output\releases\最终目录> `
  -StageOnly
```

4. 检查 stage receipt 为 `STAGED_ONLY_CURRENT_UNCHANGED`。
5. 去掉 `-StageOnly`：脚本先对 candidate 路径运行离线 bootstrap、CPU ONNX、
   BM25、persistent BPU 43、Fast/Deep loopback smoke；全部通过后才原子激活。
6. 软件激活后继续做资源 soak、live camera 和硬件闭环，不能把离线 software
   acceptance 当作整机验收。

## 5. 最终候选验收前必须保留为 pending 的门禁

- 指定 X5 身份复核；
- CPU replay；
- persistent native `libdnn` 43 样本 BPU replay（并与 canonical
  `hrt_model_exec` oracle 对齐）；
- RootMind Fast/Deep 的加载、TTFT、tokens/s、RSS；
- 内存、CMA、温度和 30 分钟 soak；
- USB 实时相机、现场偏黄灯光、固定曝光/白平衡、未知卡/OOD；
- STM32、USB-TTL、ACK、sequence、watchdog、急停；
- 真实称重、目标根区湿润、邻格串水；
- `physical_completion=false`，直到真实闭环逐项通过。

## 6. 禁止的答辩表述

- 不说“v3 已部署到 X5”；
- 不说“BPU 植物模型已经板端资格化”；
- 不说“LLM 跑在 BPU”或“多个大模型并发常驻”；
- 不说“实时相机、STM32、水泵闭环已通过”；
- 不说“32/32 等于野外或未知知识域 100% 泛化”；
- 不说“完成了 DeepSeek/Qwen 云 logits 蒸馏”。

准确口径是：PC 完成、目标板已上电并做过外部预验收，最终内容寻址 candidate
仍须经过身份复核、板端资格、原子激活和独立 live/resource 门禁。
