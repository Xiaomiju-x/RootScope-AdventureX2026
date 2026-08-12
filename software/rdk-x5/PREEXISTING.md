# RootScope 赛前预制资产披露

更新：2026-07-23（E0 v3 冻结）  
用途：记录 AdventureX 正式开始前已经存在或完成的资产。最终披露口径以 Ultimate Guide 或主办方书面要求为准。

## 2026-07-23 现场开发前的权威补充

以下资产均在本轮 RootScope-Ω v3 现场开发启动前已经存在，必须作为赛前/先期资产披露，不能计入本轮现场新增：

- RootScope H0-H12 独立软件基线、Fake F407、唯一串口 writer、追加式证据链、视觉质量门、湿润验证和本地只读 Dashboard；
- machine-curated provisional v3 的 78 张实验图片、seed17 CPU ONNX、aarch64 离线 wheelhouse、三张登记演示卡及 PC 双路径夹具；
- Qwen2 0.5B Q4_K_M GGUF、ARM64 `llama-server` b9637 交付件和默认禁用/loopback-only 的只读 LLM 配置；
- 不可变 v1/v2 X5 离线交付件，其中完整 v2 tar 的 SHA-256 为 `e6627685170252004d118bf77a690a9f89ad3afa274910697554d0f5cc8c3ebb`，独立组合审计为 50/50 PASS；
- BPU support-only 适配器、mapper 实验和 r3-r7 失败证据；截至冻结时仍为 `selected_bin=null`，没有合格的 RootScope BPU `.bin`；
- 已有训练、数据审计、release、安装、回滚和资格化脚本及其证据。

本轮开发前的逐文件哈希、不可变 release 哈希和新 X5 只读身份基线见
`evidence/e0_v3_20260723/`。其中 source composition root 为
`14801659154ab9229563dfc7fddb9c59c7d35f817d50ac70349e2ac31f28a493`，
release composition root 为
`4ce4978034323f37d650ee29a369e82ed5bc714f7739c5fc3d682a4a1944a28f`。

下文保留 2026-07-16 时点的历史记录。其“尚不存在”结论只描述当时时点；若与本节冲突，以本节和 E0 v3 哈希清单为准。

| 资产 | 原位置 | RootScope 使用方式 | 是否直接进入作品 |
|---|---|---|---:|
| XRD `0xAA55` 帧、ACK、心跳和身份门方法 | `embodied_brain/` 的 F407/ROS 串口实现 | 只读参考；RootScope 已用独立业务类型与测试重新实现 | 否 |
| XRD 追加式 JSONL/SHA-256 证据方法 | `predict_engine/persistence.py` 等 | 只读参考；RootScope 使用独立 schema 和状态根 | 否 |
| XRD Horizon BPU 导出/运行经验 | XRD 视觉与部署工具 | 后续只参考 checker/mapper/hbm_runtime 流程，不复制未资格化模型 | 否 |
| XRD Dashboard/本地 LLM 降级思想 | `dashboard.py` 等 | 只复用 fail-closed、只读解释和模板降级原则 | 否 |
| RootScope 产品、电气、机械和 72H 文档 | `adventurex/` | 项目赛前设计与现场执行基线 | 是，文档 |
| 旧 33 行 Commons 种子数据 | `adventurex/datasets/desert_plants_v1/` | 永久打印留出和迁移兼容输入；完整性 PASS，但 NOT_TRAIN_READY | 否 |
| RootScope H0-H12 软件基线 | `adventurex/rootscope/` | 赛前 fixture、协议、状态机、视觉/称重和证据链基线 | 是，须披露 |
| E0 数据合同 v2、审计器与迁移回执 | `rootscope/configs/`、`rootscope/training/` | 冻结八分区、永久留出、PTQ、许可证和最终光学数据门 | 是，须披露 |
| E0 生产 runtime、物理边界、唯一 writer、release/rollback schema | `rootscope/app/` | 赛前软件 fail-closed 骨架；E0 不含可用物理端口 opener | 是，须披露 |
| 1,333 张 Wikimedia 合法候选及人工审阅队列 | `datasets/desert_plants_wikimedia_staging_e0/` | 赛前人工审核输入；全部 `UNASSIGNED_DO_NOT_TRAIN` | 否，待人审后另建 A1 根 |
| E0 规则、X5 capsule、训练主机、物理交接和依赖偏差审计 | `rootscope/evidence/e0/` | 赛前证据与阻塞清单 | 是，文档/回执 |

## 2026-07-16 已完成但不能冒充现场新增

- class contract v2、lock 和 v1→v2 migration receipt；
- 旧 33 行数据的完整性投影审计；
- E0 生产 runtime、串口抽象/唯一 writer、release/rollback 严格合同；
- Wikimedia 采集器、许可证白名单、完整性审计器、隔离事务、人工审阅队列和四类联系表；
- 1,333 张机器筛选候选，类别 300/300/300/433，六个 required unknown 场景各 40；
- RootScope 185/185、候选工具 27/27 回归。

## 明确不存在的赛前成品

截至本记录：

- 没有 RootScope 训练 checkpoint；
- 没有 RootScope ONNX；
- 没有 RootScope Horizon BPU `.bin` 或 X5 actual replay；
- 没有 RootScope GGUF/RAG 成品和资格回执；
- 没有 final-optics printed/local/site 数据根；
- 没有真实 RootScope F407/USB-TTL/泵/称重/湿润闭环证据；
- 没有 exact-twin X5 离线安装、回滚或冷启动证明。

XRD 真车、双 X5 或双臂的既有真机结果不得转写成 RootScope 真机证据。后续 E1-E4 的 checkpoint、ONNX、BPU bin、GGUF、知识库、依赖包和 release 都必须继续追加确切路径、SHA-256、许可证和创建时间；现场新增另记入 `BUILT_DURING_EVENT.md`。
