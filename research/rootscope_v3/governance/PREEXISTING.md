# RootScope v3 启动前资产披露

冻结时点：2026-07-24  
状态：`PREEXISTING_FROZEN_FOR_V3`

本文件只记录 RootScope v3 执行开始前已经存在的资产。逐项机器状态见
`../registries/`；本文件不把计划目标升级为完成事实。

## 已存在并有证据的 RootScope 基线

- v2 不可变运行候选：
  `output/releases/rootscope_competition_runtime_v2_candidate_20260723`；
  77 个包内文件，tar SHA-256 为
  `03ca7b8d9ff8b691f1fd61dc696601ba30f494377a0b2a3cfadb66c19478ed94`。
- seed17 ResNet18 CPU ONNX 四分类模型，X5 上以
  `CPUExecutionProvider` 作为主链/审计链运行过。
- r7 ResNet18 Bayes-e `.bin`，已在 X5 的 canonical
  `hrt_model_exec` 路径执行，状态仍为
  `SHADOW_CANDIDATE_NOT_DEFAULT`。
- Qwen2 0.5B Q4_K_M GGUF 和 ARM64 llama-server b9637；板端只读、
  loopback、单模型单 slot 运行过，不是三模型并发集群。
- SQLite FTS5/BM25 RAG v1：15 个 source、24 个 chunk；带 citation
  allowlist、forbidden-query 和零权限门。
- Omega/RB-VoE 领域层：Evidence DAG、Hybrid Belief State、
  Counterfactual Failure Core、H=1/H=2 Risk-Bounded Value of Evidence、
  Deterministic Safety Compiler、Decision Receipt、Truth Ribbon、
  Claim Ledger、proposal-only DR-MPC 和 15 类故障注入。
- 已有的授权候选图、78 张 machine-curated provisional 数据、
  20 张笔记本纸卡采集图、三张登记演示卡和 unknown 负例。
- 已有 aarch64 CPU wheelhouse、BPU 系统包隔离方法、确定性 USTAR
  构建、哈希绑定、安装/校验/回滚和证据脚本。

## 来自冻结 XRD 项目的可复用方法

只允许复用架构与工程方法，包括 llama.cpp 运行方法、CPU/BPU
异构调度经验、RAG/Claim Ledger、RB-VoE、证据哈希和 fail-closed
思想。XRD 材料/NIR 领域模型、数据、实测和物理闭环不能冒充
RootScope 结果；冻结 XRD 文件本身不在 v3 工作根内修改。

## v3 启动前尚未完成

- 没有最终黄光、固定支架、X5 USB 相机的正式 live 资格；
- 没有合格的 persistent `hbm_runtime` 植物模型主链；
- 没有 RootScope 专用 0.8B/1.7B/2B 蒸馏 adapter 或量化资格；
- 没有 60+ gold / 30+ forbidden 的 RAG 2.0 冻结评测集；
- 没有灌溉前后湿润前沿的真实配对物理数据；
- 没有 STM32F4 USB-TTL 身份、ACK、泵、HX711 或真实出水证据；
- 没有自主灌溉物理闭环。

权威基线详见 `baseline_v2_snapshot.json`。旧
`rootscope/PREEXISTING.md` 中按较早时点书写的“尚不存在”段落只保留
历史意义；v3 判断以本文件、注册表和哈希快照为准。
