# RootScope v3 执行期增量

状态：`HISTORICAL_EVENT_LOG_FINAL_CANDIDATE_X5_ACCEPTANCE_PENDING`

| 时间 | 工作包 | 新增内容 | 证据/路径 | 真实性边界 |
|---|---|---|---|---|
| 2026-07-24 | E0 | 冻结 v2 回滚哈希、模型/数据/教师/依赖注册表、五套评测 schema 与零权限候选骨架 | `E0_HANDOFF_20260724.md`、`evidence/e0_verification_receipt_20260724.json` | PC-only；X5 断电 |
| 2026-07-24 | E1/E2 | 新增 `hbm_runtime` adapter、Resource Broker、Plant2Action 合同、System Coordinator、19 类故障矩阵、Decision Receipt 与零权限 Safety Compiler | `../rootscope/app/runtime_v3/`、`action_v3/`、`system_v3/`、`rootmind_v3/` | proposal-only；未打开串口/GPIO/泵 |
| 2026-07-24 | E3 | 构建 17 来源/42 chunks 的 RAG2；64 gold、36 forbidden；BM25 通过部署门，Dense challenger 被拒绝 | `rag2/`、`evidence/rag2_bm25_evaluation_20260724.json` | PC 检索资格；X5 FTS5/soak 待上电 |
| 2026-07-24 | E4 | RootSight 静态 CPU 参考 4/4、未知输入拒绝、2 个 deterministic wetting fixture | `evaluations/vision_pc_static_20260724.json` | 不是 live camera，也不是本轮 X5 CPU/BPU parity |
| 2026-07-24 | E5 | 下载官方 Qwen3-1.7B；RTX4050 完成 192-step QLoRA + 96-step train-only 对抗精炼；冻结未见 final holdout 32/32 | `llm/`、`models/llm/rootscope_qwen3_17b_qlora_final_v6_adv96_bound/`、`evaluations/llm_pc_final_holdout_v6_20260724.json` | 教师调用 0、teacher logits false；结构合同，不是未知域泛化 |
| 2026-07-24 | E5 审计 | 首个未充分绑定的 refinement 在 step 51 主动终止；随后补齐 curriculum、父 adapter、消费序列、holdout、evaluation 与 GGUF 外部 content seals | `evaluations/llm_refinement_aborted_v5_20260724.json`、`evidence/llm_*_v6_20260724.json` | 中止运行不进入 release、不冒充成果 |
| 2026-07-24 | E6 | 最终 adapter 合并、官方 llama.cpp b9637 转换并量化为 Q4_K_M；生成离线 wheelhouse、可信 stage/verify/accept/activate 工具 | `models/llm/rootscope-qwen3-1.7b-rootscope-v3-final.Q4_K_M.gguf`、`../tools/release_v3/` | PC 量化/元数据通过；X5 load/latency/RSS/soak 待上电 |
| 2026-07-24 | E7 | 目标 X5 上电并完成外部只读预验收；新增 canonical HRT oracle、persistent native libdnn、RootMind Deep b9637 兼容门与 acceptance-bound 事务激活 | `../rootscope/app/runtime_v3/native/`、`../tools/release_v3/` | 外部预验收不是最终 candidate acceptance；live/resource/STM32/physical 仍独立 |

此前 2026-07-23 已完成的 RootScope v2/Omega 增量仍由
`../rootscope/BUILT_DURING_EVENT.md` 记录，本文件不重写其历史。
