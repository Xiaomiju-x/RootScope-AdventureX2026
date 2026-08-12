# RootScope RDK X5 软件

本目录是 AdventureX RootScope 的独立运行工程。它不从 XRD 主目录 runtime import，也不包含 Nav2、SLAM、LiDAR、深度相机、底盘、升降台或机械臂代码。

当前软件状态：`H0-H12 LOCAL BASELINE + E4 EXPERIMENTAL DELIVERY + RootScope-Ω v3 LOCAL VERIFIED / X5 v2 BASE VERIFIED / ZERO AUTHORITY / CANDIDATE PREPACKAGE`。

2026-07-23 的 Ω v3 最新实现、证据哈希与严格边界见 [`OMEGA_V3_IMPLEMENTATION_STATUS.md`](OMEGA_V3_IMPLEMENTATION_STATUS.md)。新 X5 已完成不可变 v2 基线、CPU ONNX 模拟输入和 Qwen2/`llama-server` 最小前台回环 smoke；Ω v3 仍未上板，三逻辑角色、显式图片 CPU/OOD 与通用 BPU probe 必须等候选包部署后另做板端回放。

H12 已完成纯内存状态机、协议、Fake F407、视觉质量/湿润验证、证据链与本地 Dashboard 夹具基线。它没有打开或枚举真实串口，没有连接设备网络，也没有触碰 RDK X5、F407、泵、HX711 或 UVC；完整边界与 H24 门禁见 [`H12_IMPLEMENTATION_STATUS.md`](H12_IMPLEMENTATION_STATUS.md)。

## 2026-07-23 RootScope-Ω v3 本地增量

- `app/omega/`：Evidence DAG、Hybrid Belief State、Counterfactual Failure Core 与 H=1/H=2 Risk-Bounded Value of Evidence；
- `app/omega_knowledge/`：SQLite FTS5/BM25、不可变 Claim Ledger、引用 allowlist、注入防护，以及 `EVIDENCE_EXPLAINER` / `SAFETY_AUDITOR` / `DEFENSE_QA` 三个只读逻辑角色；
- `app/omega_runtime/`：Edge profile、确定性 Safety Compiler、Decision Receipt、Truth Ribbon、locked replay、proposal-only DR-MPC、15 类故障注入、loopback-only Dashboard，以及“一常驻模型、三逻辑角色串行共享”的 loopback LLM adapter；
- `app/omega_vision/`：seed17 CPU ONNX 的 Energy、max-probability、图像质量和 pooled-marginal conformal abstention，以及面向新 X5 的四张显式哈希图片回放；Mahalanobis 因 ONNX 不暴露有效 embedding 而明确跳过；
- `app/omega_bpu_aux/`：固定厂商 MobileNetV2 的通用 ImageNet 数值描述 probe；它不是植物分类器，不能进入 Safety Compiler；
- 锁定回放 `5/5` 匹配，DR-MPC `4/4` 匹配，15 类故障注入 `unsafe_accept=0`、物理命令数为 0；
- 最新本地核心测试 `102/102 PASS`，视觉/OOD/板端回放合同 `15/15 PASS`，deterministic candidate 工具 `8/8 PASS`；
- 当前**本地锁定回放**实际 profile 是 `SAFE_CPU`；本地 LLM 为确定性 fallback，BPU 模型未资格化。

冻结证据：

- `evidence/omega_v3_20260723/locked_replay_report.json`，SHA-256=`220c564a738dfa6f4284e4ebc5fc9195bbfd76c5510866ae7fc09706a6448caf`
- `evidence/omega_v3_20260723/algorithm_evaluation.json`，SHA-256=`fa18304d9e875deb1726d945d39f29c2fa83551a67ed6c051b9b0c576cf788ca`
- `evidence/omega_v3_20260723/omega_selected_tests_receipt.json`，SHA-256=`9848573fd356baab9aa2735e788878430d93cab7bd77670957ecfd8ff4d3adbb`
- 上述测试回执网络边界更正，SHA-256=`59a4c3d34f5f33632668a0fb2b232547fd6e5aba23a3141f582ca809d827c9e4`：测试使用 loopback HTTP，但未访问外网、未修改 PC 网络
- `evidence/omega_vision_v3_20260723/vision_consolidated.json`，SHA-256=`7d2bfe5914fd4f06ebc82d53537954af0d299f1f570f55c4917a17a94dd788fc`
- 视觉真实性边界更正 `vision_truth_boundary_addendum.json`，SHA-256=`bddfb2e407d33c0d50f44f037250e8464ab4265deb9354b0e7a48d27c6acc191`
- 新 X5 v2 合并观察 `evidence/new_x5_20260723/consolidated_new_x5_receipt.json`，SHA-256=`4a81fc22a68ff245097e6cf5266b7e6ccb06a9c9f7d50c4005bff940580462e6`

最终 v3 candidate 尚未打包；合入门禁见 [`OMEGA_V3_CANDIDATE_RELEASE_CHECKLIST.md`](OMEGA_V3_CANDIDATE_RELEASE_CHECKLIST.md)。

## 2026-07-17 算法交付增量

- 机器整理 provisional v3 共 78 张，实验划分为 train/val/print/creator-holdout=`55/9/6/8`；它仍不是人工审核或正式 data lock。RTX4050 三种子实验选中 seed17，val=`8/9`、print=`3/6`，正式小样本拒绝门仍拒绝全部类别，因此模型状态固定为 `MACHINE_CURATED_EXPERIMENTAL_NOT_QUALIFIED`。
- seed17 已导出并哈希绑定为 CPU ONNX，clean-X5 胶囊复现训练预处理；CPython 3.10/aarch64 的 11-wheel 离线候选 wheelhouse、OpenCV 和当前用户安装器已进入确定性 core 包。
- 三张训练参考（草丛、灌木、幼树）已登记为 `DEMO_REFERENCE_NOT_HOLDOUT_ONCE_REGISTERED`；沙丘 unknown 保持未登记。PC 模拟相机帧中三张正例双路径共识通过、unknown 拒绝，审计 `40/40 PASS`；这不是 UVC、X5、holdout 或泛化精度证据。
- XRD 来源的 Qwen2 0.5B GGUF 已独立分包并锁 SHA；aarch64 `llama-server` 与该 GGUF 已在新 X5 完成一次最小前台 loopback smoke 并在结束后关闭端口。它仍默认禁用、没有长稳资格，且只有只读解释权限。
- Bayes-e 的 256 条 train-only calibration、mapper 配置和独立审计已准备；尚未启动 Docker/WSL、没有 `.bin`，所以 `bpu_compiled=false`。
- 最终 RootScope 回归为 `282/282 PASS`；X5 core/LLM 交付路径与现场命令见 [`deploy/x5/ROOTSCOPE_X5_OFFLINE_RUNBOOK_ZH.md`](deploy/x5/ROOTSCOPE_X5_OFFLINE_RUNBOOK_ZH.md)。

## 权限边界

- 状态机是唯一任务命令所有者；
- F407 负责三泵互斥、称重、硬超时和安全输入；
- BPU/标签只提供形态输入，不直接控制泵；
- 本地 LLM 只读证据，不拥有串口、状态机写权限或复位权限；
- 绿色完成只允许 `TARGET_WETTING_VERIFIED`。

## 本地检查

在本目录运行：

```powershell
$py='python'
& $py -m unittest discover -s tests -v
& $py -m app.web.fixture_server --port 8765
```

打开 `http://127.0.0.1:8765`，点击“启动夹具演示”可运行完整的纯内存闭环。该入口只接受 `SIMULATION_ONLY / FAKE_F407` 配置；“复位夹具”只复位页面视图；“模拟急停”在此入口故意没有动作回调并返回 `ACTION_UNAVAILABLE_IN_CURRENT_MODE`。三者都不会触碰真实串口或设备。

若只想检查默认锁定页、完全不注册模拟动作，可运行 `& $py -m app.web.server --port 8765`。

## 目录

```text
app/          状态机、串口、视觉、证据和本地页面
configs/      类别、Profile、串口身份和冻结阈值
training/     数据审计及后续训练/导出工具
tests/        stdlib unittest 本地验收
evidence/     可重建的本地验收报告
deploy/       后续 X5 systemd 与健康检查
```

真串口、泵、称重和相机参数必须等硬件/机械负责人提交实物审计后再启用。

`deploy/x5/` 同时保留 v1 clean-X5 零权限骨架和 v2 现场离线组合包合同。v2 已在新 X5 按原 tar 哈希安装并完成 CPU 模拟输入、最小 LLM 前台回环和 BPU import-only 观察；安装时发现的顶层 `app` 缺件由单命令、非持久 shim 绕过，原 tar 与安装树未改写。`selected_bin=null`，没有 RootScope 植物 BPU 模型。Ω v3 仍须通过新的不可变 delta 候选部署与板端回放，当前不能写成 v3 已上板或物理闭环完成。v2 分阶段合同见 [`deploy/x5/ROOTSCOPE_X5_FIELD_BUNDLE_V2_RUNBOOK_ZH.md`](deploy/x5/ROOTSCOPE_X5_FIELD_BUNDLE_V2_RUNBOOK_ZH.md)。

## H24 前的冻结项

- 物理 USB-TTL 适配器必须采用单一 writer，并通过 heartbeat 不插入命令事务、E-stop 最高优先级和真实线序审计；合同见 [`BLOCKERS.md`](BLOCKERS.md)。
- 先完成一泵 + HX711 + 急停的单通道连续 3 次实物闭环，再扩三通道；在此之前不得把模拟结果写成物理完成。
- 正式 A1 数据集仍为 `NOT_TRAIN_READY`；当前训练只属于 machine-curated experimental 线。三张植物打印源已转为演示模板而不再是 holdout，实际 UVC 复拍仍 pending，`print_eligible=false`。
- BPU 与本地 LLM 排在安全串口、称重和湿润实物证据之后；没有实际 `hobot_dnn` 日志时不得声称 BPU 已部署。
