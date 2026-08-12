# 05｜XRD 资产复用边界与 AdventureX 新增工作

## 1. 原则

复用 XRD 的“可靠基础设施”，不复制 XRD 的车辆形态和业务语义。所有预制代码、模型、板卡与测试资产都要在 `PREEXISTING.md` 中如实列出；AdventureX 现场新增的灌溉硬件、协议类型、状态机、视觉配置和测试数据另列 `BUILT_DURING_EVENT.md`。

截至本方案冻结时，AdventureX 2026 公开规则是否允许何种程度的预制代码/硬件仍需以 Ultimate Guide、Portal 或主办方书面回复为准。未确认前，不把“可复用”理解为“可直接计入现场成果”。

## 2. 可直接复用的设计资产

| XRD 资产 | RootScope 用法 | 证据位置 | 复用限制 |
|---|---|---|---|
| `0xAA55` 帧、长度、checksum、ACK 解析骨架 | X5–F407 USB-TTL 通信 | `embodied_brain/ros2_ws/src/my_robot_drivers/include/my_robot_drivers/serial_protocol.hpp` | 只复用帧层；灌溉业务消息重新定义 |
| `EMERGENCY_STOP=0x10`、`CLEAR_ESTOP=0x11`、`HEARTBEAT=0xFF` 的锁存思想 | 急停、显式复位、失联关泵 | 同上及 `serial_f407_node.cpp` | 需重新验证 RootScope 固件，不能沿用车载真机结论 |
| `SAFETY_STATE`、`FIRMWARE_INFO` 和精确身份门 | 上位机只接受预期固件/能力 | 同上 | RootScope 必须使用新 build ID/capabilities |
| USB 串口设备固定别名思路 | `/dev/F407`，避免 ttyUSB 顺序漂移 | `embodied_brain/README.md`、现有 udev 规则 | VID/PID/serial 按新适配器实测 |
| 执行器服务与 heartbeat 分离、ACK 后再改本地状态 | 泵服务不阻塞心跳，clear-estop 必须有 F407 ACK | `serial_f407_node.cpp` | 不复制 lift/electromagnet 业务代码 |
| 结构化完成等级与物理证据门 | 区分模拟、ACK、称重、目标 ROI | `my_robot_msgs/action/DispatchTask.action`、`pickup_physical_evidence_v1_20260711.md` | RootScope 使用自己的字段与证据来源 |
| 追加式 JSONL、SHA-256 绑定和本地离线证据 | 每轮灌溉回执与版本追踪 | `embodied_brain/docs/data_loop_mcap_manifest.md`、`predict_engine/persistence.py` | 只能称“本地可追溯/篡改可检测”，不是第三方审计或签名 |
| Dashboard 的实时状态、告警、WebSocket 和离线页面模式 | RootScope 单页看板 | `dashboard.py`、`my_robot_dashboard` | 页面和术语必须重做，不保留车载拓扑 |
| fail-closed 测试方法 | 断线、急停、重复帧、过期数据、身份错配 | `embodied_brain/docs/embodied_v3_acceptance_20260709.md` | 车载 PASS 不能转移为 RootScope PASS |

## 3. 可以复用思想但不复制实现

- 任务 admission gate、原子占用和取消语义；
- 物理证据与任务 ID、阶段起始时间、校准版本绑定；
- “真实输入/fixture/回放” provenance 标记；
- 服务重启后默认锁定、旧任务不恢复；
- 失败注入与独立审计表；
- UI 中只按证据等级显示完成，不靠字符串猜测。

这些都要重写成固定式泵水业务，不能保留 `cmd_vel`、Nav2、lift、pickup、electromagnet 等词。

## 4. 必须新写/新做

### 4.1 F407 固件

必须新建 RootScope 独立工程或清晰的新构建目标，不在原车固件中塞条件编译并冒充已验证。至少包括：

- 三泵 GPIO 与硬件互斥；
- HX711 称重采样、稳定窗与原始值上报；
- 急停、漏水、卡匣、Start/Reset 输入；
- 泵硬超时、心跳看门狗、上电锁定；
- 新固件协议版本、能力位、构建 ID；
- 三路泵/称重/安全输入的台架测试。

### 4.2 新业务消息（建议冻结后不再改 type）

在保留帧层的前提下，为 RootScope 分配独立类型并建立单一 ICD。建议语义如下，具体 type 值由 A/B 在 H0–H4 查重后冻结：

| 方向 | 消息 | 最小字段 |
|---|---|---|
| X5→F407 | `ARM_TASK` | task_id/seq、channel、target_mass、hard_timeout、config_hash 摘要 |
| X5→F407 | `ABORT_TASK` | task_id/seq、reason |
| X5→F407 | `EMERGENCY_STOP` | 无 payload，复用语义 |
| X5→F407 | `CLEAR_ESTOP` | 显式受控复位，复用语义 |
| X5→F407 | `HEARTBEAT` | seq/上位机状态摘要 |
| F407→X5 | `IRRIGATION_TELEMETRY` | task_id、三泵状态、raw/filtered mass、输入位、uptime |
| F407→X5 | `SAFETY_STATE` | estop、leak、cartridge、watchdog、blocked_count |
| F407→X5 | `FIRMWARE_INFO` | protocol、capabilities、build_id、hw_variant |
| F407→X5 | `ACK/ERROR` | ack_for_type、seq、status/reason |

`ARM_TASK` 只能授权一个通道；F407 不接受“泵掩码=多位”的命令。浮点的端序、单位、范围和无效值必须在 ICD 中固定，不能由两端各自猜测。

### 4.3 X5 应用

- `rootscope_core`：状态机、admission、幂等 task_id 和故障恢复；
- `rootscope_serial`：0xAA55 parser、ACK、身份、新鲜度与命令重放拒绝；
- `rootscope_vision`：场景卡 ROI、前后沙体 ROI、遮挡/过曝检查；
- `rootscope_ui`：固定装置页面，不出现地图、车速、里程计或导航；
- `rootscope_evidence`：基线/结果摘要、称重、命令/ACK、安全门、版本和终态记录。

P0 可使用单进程 Python 服务加独立串口线程和看门狗，不要求把系统拆成七个 ROS 2 包。若使用 ROS 2，也只为现有部署便利，不把 ROS 2 本身当创新点。

### 4.4 机械、标定和数据

- 三路卡匣、水路、称重托盘、漏水盘和固定光学全部是 AdventureX 新资产；
- 每只泵、每个卡匣、同一批沙的标定数据必须现场记录来源；
- BPU 训练/验证数据若来自赛前，也需披露；若是现场采集，记录时间和划分；
- 真实完成等级只由 RootScope 本机证据决定，不能引用 XRD 车载测试数字。

## 5. 明确禁止复用到主线的 XRD 部分

- `cmd_vel`、`/cmd_vel_safe`、底盘电机 BSP；
- IMU、里程计、LiDAR、Astra 深度相机；
- SLAM、AMCL、Nav2、Collision Monitor、Lab-FSD、MPPI；
- 升降台、探针滑台、电磁铁、pickup、机械臂；
- 双 X5 调度、AI 脑材料预测、XRD/PL 模型；
- 车载 Cockpit 中的地图、速度、车辆拓扑和任何车载实测指标。

这些模块即使在仓库里存在，也不进入 RootScope 运行进程、不出现在海报“本次技术栈”中、不占用 72 小时调试时间。

## 6. 对外披露模板

### 预制基础

“我们赛前已有一套自研的 RDK X5–STM32 通信、安全状态和本地证据基础设施。本次按赛事规则披露并复用了帧协议、心跳、固件身份、急停锁存和记录方法。”

### 本次新增

“AdventureX 期间我们围绕固定式 RootScope 新完成三通道灌溉硬件、F407 灌溉固件、称重闭环、透明卡匣、固定视觉验证、场景交互和整机故障测试。”

### 不该说

“整个系统都是 72 小时从零完成”或“XRD 已验证过，所以 RootScope 也已验证”。

## 7. `PREEXISTING.md` 最小字段

- 资产名称与仓库路径；
- 创建时间/已有版本或 commit；
- 本次使用方式；
- 本次修改摘要；
- 许可证/团队所有权；
- 是否进入最终二进制、演示或只作参考；
- 对应赛事规则/主办方确认记录。
