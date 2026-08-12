# Security Policy / 安全政策

RootScope 同时包含普通软件、模型流水线和真实执行器原型代码。安全问题可能影响凭据、供应链、隐私，也可能导致电机、水泵或继电器进入不安全状态。

## 报告漏洞

请使用 GitHub 的 [Private vulnerability reporting](https://github.com/Xiaomiju-x/RootScope-AdventureX2026/security/advisories/new) 私下报告。不要先开公开 Issue，也不要附上真实凭据、私网拓扑、设备唯一身份或可能导致物理动作的完整利用脚本。

报告建议包含：

- 受影响 commit/tag 和文件；
- 漏洞类型与最小复现（使用 fixture/模拟器）；
- 预期影响和建议修复；
- 是否可能触达串口、GPIO、固件、泵、电机、发布资产或 CI secret；
- 可公开披露的范围和时间建议。

维护者会尽力在 7 天内确认、30 天内给出评估或缓解方案，但这是志愿项目，不承诺 SLA。修复发布后再协调公开披露。

## 支持范围

| 版本 | 安全更新 |
|---|---|
| 最新 `1.x` release / `main` | 支持 |
| 旧 tag、比赛现场冻结镜像、个人 fork | 不主动支持 |

上游 RDK OS、STM32 HAL/CMSIS、模型运行时、Python 依赖和第三方模型的漏洞由各上游处理；请同时遵循其安全公告。

## 高风险边界

以下行为必须默认拒绝：

- 凭据、API key、token、私钥或云/VPS 配置进入 Git、日志、回执、模型或媒体；
- 真实私网地址、MAC、machine-id、boot-id、序列号、用户名目录被公开；
- 导入模块、网页、LLM/RAG 或网络请求直接获得执行器写权限；
- 身份/版本/capabilities 不一致时仍解锁；
- 心跳丢失、UART fault、硬超时、STOP 或进程崩溃后输出保持开启；
- 无界开泵、自动解锁、自动物理重试、无传感器自动回升；
- 未经校验的模型/固件/发布包进入物理链；
- 未授权数据、人像、GPS、胸牌、二维码或屏幕内容公开。

## 物理安全

本项目不是工业产品或安全认证控制器。真实复现至少需要：

- 能物理切断执行器功率的急停/总断开；
- 保险、合适线径、隔离/驱动、浪涌与续流保护；
- 机械限位、护罩、防夹和卡滞处理；
- 漏水防护、接水盘和干湿区隔离；
- STM32 独立看门狗、硬超时、锁存关闭；
- 分级上电、假负载、单执行器、湿式验收；
- 训练过的操作员全程监护。

软件 `HOLD`、MCU ACK 或网页状态不能替代物理关断和现场观察。发生异常时先切断执行器动力，再收集证据。

## Secret 响应

如果 secret 已经提交：

1. 立即在提供方撤销/轮换，**不要等待 Git 历史清理**；
2. 检查使用日志和关联权限；
3. 私下通知维护者；
4. 用最小范围方式清理历史并重新审计；
5. 不在公开 Issue 粘贴 secret，即使已失效。

Git 历史重写会影响所有 clone，应由维护者协调。删除当前文件不能使历史 secret 失效。

## English summary

Report vulnerabilities privately through GitHub Security Advisories. Do not publish credentials, device identity, private topology, or actuator-ready exploits. RootScope is not a safety-certified controller. Hardware reproduction requires independent power isolation, emergency stop, watchdog/timeouts, mechanical limits, leak protection, staged tests, and a trained operator. Rotate exposed secrets immediately; deleting a file or rewriting history is not a substitute for revocation.
