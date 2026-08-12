# AdventureX 现场新增记录

状态：`IN_PROGRESS`。

本文件只在 AdventureX 官方创造窗口开始后记录现场新增内容。当前赛前准备不得倒填为“72 小时从零完成”。

| 时间 | 负责人 | 新增/修改 | 证据路径 | 验收结果 |
|---|---|---|---|---|
| 2026-07-23 14:36 +08:00 | 算法负责人 / Codex | 冻结 RootScope-Ω v3 开发前的 AdventureX 源码、不可变 release 与新 X5 身份边界；未触碰 XRD runtime、相机、串口、泵或 PC 网络 | `evidence/e0_v3_20260723/` | `E0_COMPLETE_PRE_V3`；source root=`14801659154ab9229563dfc7fddb9c59c7d35f817d50ac70349e2ac31f28a493`；release root=`4ce4978034323f37d650ee29a369e82ed5bc714f7739c5fc3d682a4a1944a28f` |
| 2026-07-23 14:46-15:23 +08:00 | 算法负责人 / Codex | 新增 zero-authority Ω Evidence Core、SQLite FTS5/BM25 + Claim Ledger + 三只读逻辑角色、Hybrid Belief、Counterfactual Failure Core、H=2 RB-VoE、Safety Compiler、Decision Receipt、Truth Ribbon、proposal-only DR-MPC、15 类故障注入和 loopback-only Dashboard | `app/omega/`、`app/omega_knowledge/`、`app/omega_runtime/`、`configs/omega/`、`tests/test_omega_*.py`、`tests/omega_knowledge/`、`evidence/omega_v3_20260723/omega_selected_tests_receipt.json` | 当前选定 Ω 测试 `81/81 PASS`；知识链 `26/26 PASS`；Windows SQLite 关闭后重开通过；测试观察回执 SHA=`9848573f...adbb` |
| 2026-07-23 14:53-15:23 +08:00 | 算法负责人 / Codex | 生成 5 场景锁定回放与 DR-MPC/故障注入本地评估；全部为 `SIMULATION_ONLY`，无相机、串口、泵、网络或物理完成主张 | `evidence/omega_v3_20260723/locked_replay_report.json`、`evidence/omega_v3_20260723/algorithm_evaluation.json` | 回放 `5/5`；DR-MPC `4/4`；故障 `15`；`unsafe_accept=0`；`physical_command_count=0`；文件 SHA 分别为 `220c564a...8caf`、`fa18304d...88ca` |
| 2026-07-23 15:25-15:28 +08:00 | 算法负责人 / Codex | 新增一个常驻 loopback 模型端点串行服务三个只读逻辑角色的 adapter；无 tool/串口/GPIO/状态机/泵接口，失败逐角色确定性降级 | `app/omega_runtime/loopback_llm_cluster.py`、`tests/test_omega_loopback_llm_cluster.py` | Python loopback HTTP fixture `4/4 PASS`；不是实际 Qwen/llama.cpp 或 X5 资格证明 |
| 2026-07-23 15:27 +08:00 | 算法负责人 / Codex | 不改写原 81-test 回执，追加网络边界更正：Dashboard 测试使用本机回环 HTTP，正确口径为 touched loopback、未访问外网、未修改 PC 网络 | `evidence/omega_v3_20260723/omega_selected_tests_receipt_network_boundary_correction.json` | SHA=`59a4c3d3...c9e4`；`loopback_http_touched=true`、`external_network_touched=false` |
| 2026-07-23 15:13-16:00 +08:00 | 算法负责人 / Codex | 在固定 SSH alias 与严格身份复核下，把不可变 v2 bundle 安装到全新 RDK X5；完成 CPU ONNX 模拟输入、Qwen2/llama-server 最小前台 loopback、BPU import-only，随后恢复无进程、端口关闭、无 gate 状态 | `evidence/new_x5_20260723/` | 合并回执 SHA=`4a81fc22...462e6`；CPU 与最小 LLM smoke 通过；厂商 MobileNet 仅 support；`selected_bin=null`；v2 顶层 app 缺件已如实记录，未改原 tar |
| 2026-07-23 15:30-16:20 +08:00 | 算法负责人 / Codex | 新增实验视觉 Energy/max-probability/质量门/pooled-marginal conformal abstention、通用 BPU 辅助 probe、四张显式哈希图片的新 X5 CPU 回放合同；没有重跑 holdout，也没有把厂商 MobileNet 写成植物模型 | `app/omega_vision/`、`app/omega_bpu_aux/`、`configs/omega/vision_board_replay_new_x5_20260723.json`、`evidence/omega_vision_v3_20260723/` | 原视觉观察 SHA=`7d2bfe59...788fc`；真实性 addendum SHA=`bddfb2e4...c191`；本地视觉/板端合同 `15/15 PASS`；正式 coverage/模型/BPU/相机资格均为 false |
| 2026-07-23 16:20-16:35 +08:00 | 算法负责人 / Codex | 新增 deterministic USTAR v3 delta build/audit/板端 verify helper，并修复 Truth Ribbon 页面静态真值缺陷：实际 CPU 后端、fallback reason、authority 与 physical claim 现在都从证据 API 校验后渲染，缺字段显示 UNVERIFIED | `tools/release/`、`app/omega_runtime/static/index.html`、`tests/test_omega_dashboard.py` | candidate 工具 `8/8 PASS`；核心 `102/102 PASS`；正式 candidate 尚未创建 |
| 2026-07-23 16:35-16:43 +08:00 | 算法负责人 / Codex | 冻结四图 board replay 的完整 config SHA、校准/provenance/PC reference canonical hash；跨架构容差预冻结为 `1e-4`；完成新目录 replay、算法评估及桌面/移动浏览器验收 | `app/omega_vision/board_replay.py`、`configs/omega/vision_board_replay_new_x5_20260723.json`、`evidence/omega_v3_candidate_local_20260723/` | 本地总计 `125/125 PASS`；5/5 replay；DR-MPC 4/4；15 faults/0 unsafe accept/0 physical command；QA receipt SHA=`a867b80f...d368` |

| 2026-07-24 00:00-04:54 +08:00 | 算法负责人 / Codex | 在独立 `rootscope_v3/` 中完成 RootScope v3 PC 算法、RAG、RootMind、Safety Compiler、发布工具链与审计 seal；训练 Qwen3-1.7B 最终 adapter 并生成 Q4_K_M | `../rootscope_v3/`、`../tools/release_v3/` | final holdout 五项通用合同 `32/32`，真实对抗请求拒绝 `16/16`；Safety Compiler raw accept=32/fallback=0/unsafe escape=0；X5 断电，身份/CPU/BPU/资源/live camera/STM32/物理闭环仍 pending |

候选发布已进入 `PC_COMPLETE_X5_POWER_PENDING`：v3 PC 训练、冻结 holdout、静态视觉、
BM25、零权限合同与离线发布工具均已完成；这不升级为 X5 或物理完成。板端
`hbm_runtime` 43 样本、Q4_K_M CPU smoke、资源 soak、实时 USB 相机、STM32/USB-TTL
和真实根区湿润闭环必须等 X5 与硬件上电后逐项验收。
