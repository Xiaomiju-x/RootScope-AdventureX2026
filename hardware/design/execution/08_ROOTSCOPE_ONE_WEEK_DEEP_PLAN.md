# RootScope 赛前一周 + 现场 72 小时 v2.0 深度执行计划

> 状态：v2.0 已获队长确认；E0 于 2026-07-16 完成软件收口  
> E0 结论：`E0_SOFTWARE_READY_A1_BLOCKED`  
> 工作目录：仓库根目录  
> 产品：固定式 RootScope 根区灌溉舱  
> 当前能力：`SIMULATED_ONLY`

## 1. 冻结判断

RootScope 不是移动小车，也不是迷你版 XRD 机器人。它是一台固定式沙地根区灌溉舱：固定 UVC 观察三张植物卡/三层沙芯，RDK X5 做视觉、状态机、证据和只读解释，STM32F407 独占泵、称重和安全输入，三路微型泵向 Z1/Z2/Z3 定点供水。

本周不加入底盘、导航、SLAM、LiDAR、IMU、深度相机、ROS 2 重构、多智能体或云端控制。评委能在 60–75 秒内亲手选择目标、看到单泵动作、质量曲线、目标层湿润、邻层排斥、故障锁存和证据回执，才是主线。

```text
固定 UVC ──> CPU 画质/几何门 ──> BPU 四分类候选 ──> 安全 admission
                                                        │
只读 LLM/RAG <── 结构化事实/来源 <── 证据链 <── 状态机 <──┤
                                                        │
STM32F407 <── 唯一 USB-TTL writer <─────────────────────┘
   ├── 三泵互斥/硬超时
   ├── HX711 定量停泵
   └── 急停/漏水/卡泵/失联 fail-closed
```

## 2. 四人职责

| 角色 | 主责 | 每日必须交付 |
|---|---|---|
| A 队长/算法 | RDK X5 全部代码、数据合同、CPU/BPU、状态机、串口唯一 writer、Dashboard、LLM/RAG、release | 可执行代码、测试、SHA、失败清单、队友任务卡 |
| B 硬件 | F407、泵驱动、电源/保险、HX711、急停/安全输入、线束与真实接线 | 型号/引脚/电气测量、台架日志、照片、备件 |
| C 机械 | 固定舱、干湿隔离、接水盘、卡片位、相机/灯架、三层沙芯、软管和运输锁 | 尺寸图、装配照、承重/漏水/运输恢复记录 |
| D 运维宣讲 | 规则、资产披露、人审/许可、runbook、口播、展位和证据归档 | Ultimate Guide 闭环、审阅签名、盲演、提交包 |

队长每晚只看四个根：软件测试根、数据/模型根、硬件安全根、机械/演示根。没有可复核证据的“完成”不进入次日计划。

## 3. 权威边界

- F407 是泵、安全输入、HX711、心跳失联和硬超时的最终执行权威。
- X5 状态机决定任务 admission 和证据完整性，但不能绕过 F407 ACK/TASK_RESULT。
- 只有一个串口 owner/writer；HTTP、视觉、LLM、heartbeat 都不能直接写串口。
- BPU 只给四类可见形态与拒答置信，不直接决定发泵，也不推断真实根深或真实需水量。
- LLM/RAG 只读解释结构化事实和本地来源；无串口、状态机、Clear E-stop 或剂量修改权限。
- Dashboard 只展示和受控触发，不拥有 GPIO/串口直控能力。

完成等级严格为：`SIMULATED_ONLY → ACTUATOR_ACK → MASS_LOSS_VERIFIED → TARGET_WETTING_VERIFIED`。任何冲突进入 `ABORTED_LOCKED`。赛前软件不得冒充现场真机等级。

## 4. 分阶段硬门

### E0：合同、候选数据、生产边界

已完成：

- class contract v2、八分区、永久打印留出、train-only PTQ 和最终光学收据；
- 旧 33 行数据兼容迁移与 `NOT_TRAIN_READY` 审计；
- 生产 runtime、物理端口禁用边界、唯一 writer 和 release/rollback 合同；
- 1,333 张 Wikimedia 候选，300/300/300/433，六类 unknown 各 40；
- 严格完整性审计、确定性人工审阅队列和四类联系表；
- RootScope 185/185、候选工具 27/27 测试通过。

E0 没有训练模型、打开真实端口、枚举设备、连接 X5/F407、烧录或湿运行。A1 仍被人工审核、最终光学与 printed/local/site 域阻塞。

### A1：`DATA_LOCKED`

必须同时满足：

1. 三个目标类各至少 100 个经人工批准的独立自然 source group；
2. unknown 至少 120 个批准组，覆盖至少六种负样本家族；
3. 每类 printed_train 12 组，按 6/2/2/2 分配，至少四个最终光学复拍/组；
4. 每类 printed_test 4 组、print_demo 2 组，至少四个复拍/组；
5. site_acceptance 每类 1 张封存卡 + 10 个封存 unknown 场景；
6. B/C/D 对 UVC、灯光、距离、纸张、打印机、几何和 USB 口三方签名；
7. 八分区 source group 两两不交叉，永久留出不可迁移，PTQ 仅来自 train；
8. 许可、坏图、SHA、近重复、系列合并和 split 泄漏审计全部 PASS。

人工审阅时必须合并同作者连拍/同场景系列；1 个 Commons 页面不自动等于 1 个独立组。

### E1：RTX 4050 训练和 CPU/ONNX 锁定

A1 通过后才启动：

- 训练一套轻量四分类权重，首选 MobileNetV3-Small 级别；三固定 seed；
- validation 负责选模，calibration 冻结拒答/温度/阈值；test 只在模型锁定后一次性打开；
- 同一 checkpoint 导出静态 ONNX，冻结 resize/crop/normalization/labels/reject policy；
- CPU ONNX 作为板端诊断/降级，不能因“没量化”就宣称更准确；
- 生成 checkpoint、ONNX、训练 capsule、数据根、代码根和 model card SHA。

### E2：Horizon BPU

- 使用 A1 冻结的 train-only `ptq_calibration`；
- checker → mapper/makertbin → 算子落点/日志 → `.bin`；
- PC/x86 golden replay 先过，再在授权的同 SKU X5 上跑 actual `hbm_runtime`；
- 对比 CPU/ONNX 与 BPU 的逐样本输出、拒答、漂移和延迟；
- 资格失败则 BPU 保持 shadow，正式感知退到 tag/template，不在杭州重新转换。

### E3：本地 LLM/RAG 与 Dashboard

- 冻结单一 GGUF、许可证、SQLite FTS/BM25 知识库、来源 SHA；
- 20 个黄金问答、20 个禁权问题、JSON Schema 和确定性模板；
- LLM 只解释“看到了什么、为什么拒绝、泵/质量/湿润证据是否一致”；
- 与视觉/core 并行压测，资源不足默认禁用 LLM，模板始终可用；
- profile 固定为 `FULL_BPU_LLM`、`BPU_TEMPLATE`、`TAG_TEMPLATE`，每个 profile 独立成包。

### E4：离线 release 与 exact-twin X5

- release 包含 app、冻结模型、KB、纯 Python wheelhouse、systemd/udev、安装/验收/回滚和 golden replay；
- 应用包不混入系统 `.deb`，系统镜像/SDK/hbm_runtime 由不可变 capsule 单独绑定；
- 完成 TARGET_ENROLLMENT、三次干净安装、冷启动、restart/SIGKILL、2h soak、回滚再安装；
- 每次启动必须 pump OFF、`COMMISSIONING_LOCKED`、唯一 writer、无旧任务恢复；
- 目标是拿到同型新 X5 后，通过一次离线 SSH 安装和一次只读验收进入 locked 状态，不是随机新板无校准即可湿运行。

### E5：杭州现场目标绑定

- 只绑定现场新增根：板卡/USB/UVC 身份、homography/ROI、称重、泵、安全输入、最终卡片和几何；
- B+C 双签后按“指示灯 → 单泵/称重/接水盘 → 单通道 3/3 → 三通道”逐级授权；
- 不修改 portable 模型、KB、阈值或依赖；任何变化产生独立 commissioning root；
- 未取得液体书面允许时，使用密闭收集或封闭湿润 coupon，并相应收缩宣讲。

### E6：整机资格与冻结

- 故障注入：急停、漏水、失联、ACK 冲突、称重失效、卡泵、证据缺失、BPU/LLM 不可用；
- 30-run、非开发者盲演、运输后恢复和两套备份介质；
- 绑定 `PORTABLE_RELEASE_ROOT + COMMISSIONING_ROOT` 后冻结；
- 最终讲稿只描述真机实测等级，不把 shadow、fixture 或机器候选写成正式能力。

## 5. 赛前七天并行表

| 日程 | A 算法 | B 硬件 | C 机械 | D 运维宣讲 | 当日硬门 |
|---|---|---|---|---|---|
| D-7/E0 | 合同/runtime/候选链 | BOM/板型/电源核对 | 尺寸/干湿区冻结 | 规则/资产披露 | `E0_SOFTWARE_READY_A1_BLOCKED` |
| D-6/A1 | 人审、分组、最终光学采集 | UVC/灯/USB、电源台架 | 相机架/卡槽/沙芯 | 许可复核与签名 | `A1_DATA_LOCKED` |
| D-5/E1 | RTX4050 训练、ONNX | F407/FW/单泵/HX711 | R1 装配/承重 | 模型卡与证据索引 | `A2_MODEL_LOCKED` |
| D-4/E2 | Horizon 转换/golden | 三泵互斥/急停 | 单通道湿测/漏水 | 失败口径 | BPU qualified 或 shadow |
| D-3/E3 | LLM/RAG/Dashboard | 线束/备件 | 三通道/运输锁 | 20+20 问答、盲演 | profile 冻结 |
| D-2/E4 | 离线 release/twin | 整机电气 soak | 运输恢复 2 次 | 安装/回滚 runbook | A4 profile PASS |
| D-1 | 只修 blocker | 备件包 | 排空/包装 | 讲稿/提交/介质 | 不再训练/转换 |

任何硬门未通过，后续只允许走已定义降级 profile，不允许靠临场堆模型或改阈值掩盖。

## 6. 最重要的现场演示

1. 评委选择一张目标卡；Dashboard 显示真实 backend、类别/拒答和来源。
2. X5 生成 task，唯一 writer 发往 F407；只允许目标泵动作。
3. HX711 到目标质量或硬超时停止；状态机等待新鲜 TASK_RESULT。
4. UVC 复拍验证目标层变化与邻层排斥；任何冲突进入 `ABORTED_LOCKED`。
5. Dashboard 展示原始帧、ACK、质量曲线、湿润证据和 SHA 链；LLM只读解释。
6. 主动制造一次故障，让评委看到拒绝、锁存和人工恢复，而不是只看顺利 demo。

## 7. 当前下一步

E0 已完成。现在不应训练模型；应先：

1. A+D 审完 1,333 条队列，按系列合并并保留足够批准组；
2. B+C+D 当天冻结最终光学收据；
3. A+C+D 采 printed/local/site 域并跑 v2 审计；
4. B 完成真实型号、引脚、电源、泵/HX711/急停台架；
5. D 取得 Ultimate Guide/书面边界；
6. 仅在 `A1_DATA_LOCKED` 根生成后启动 E1 训练。

E0 证据权威入口：`adventurex/rootscope/evidence/e0/E0_EXECUTION_STATUS.md`。
