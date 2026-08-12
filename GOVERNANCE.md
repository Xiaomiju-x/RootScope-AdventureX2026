# Governance / 项目治理

RootScope 是 AdventureX 2026 赛后维护的开源档案。治理优先级依次是：人身/硬件安全、可验证真实性、许可与隐私、可复现性、兼容性和新功能。

## 角色

- **Maintainers**：维护 roadmap、release、仓库设置与最终技术决策。
- **Code owners**：审阅指定目录，对安全、许可或领域合同提出阻断意见。
- **Contributors**：通过 Issue、PR、复现报告和审阅参与。

当前 maintainer / CODEOWNER 为 `@Xiaomiju-x`。项目可通过透明 PR 更新角色。

## 决策方式

日常修复由 maintainer 在 PR 中审阅并合并。较大改动需先有设计 Issue，至少说明：问题、非目标、接口、失败方向、测试/复现计划、许可/隐私和迁移影响。

优先通过公开讨论形成粗略共识；无法达成时由 maintainer 记录理由后决定。以下原则不是投票可绕过的：

- 不虚构或扩大比赛、性能、BPU、物理验证结论；
- 不公开凭据、私有设备身份或无权再分发资产；
- 不让 LLM/RAG 获得物理执行权；
- 不移除 fail-closed、锁存、超时、人工确认等安全边界；
- 不把固定答辩卡写成开放世界农艺能力。

## 审阅要求

| 改动 | 最低审阅 |
|---|---|
| 文档、拼写、普通测试 | 1 位 maintainer/CODEOWNER |
| 运行时行为、模型合同、CI/release | 1 位 CODEOWNER + 通过 CI/审计 |
| 固件、串口、GPIO、物理动作 | 设计 Issue、失败路径测试、无动作证据、CODEOWNER 明确批准 |
| 数据/模型/媒体加入 | 来源、许可、哈希、隐私审计和 CODEOWNER 批准 |
| 安全修复 | 私下协调、测试、发布与披露计划 |

紧急 secret 撤销或明显危险内容下架可由 maintainer 先处理再补公开记录。

## Release

- 语义化版本：不兼容合同变更为 major，新能力为 minor，兼容修复为 patch。
- Release 由受保护的 commit/tag 构建，附变更、SHA-256、迁移、安全和真实性边界。
- 模型/固件/大资产采用内容寻址 manifest；release 不包含缓存或无权再分发材料。
- 一个失败的资格结果保持失败，不通过重写 receipt 或降低阈值“凑通过”。

## 项目范围

项目接受赛后复现、测试、文档、安全、教育和明确隔离的研究改进。它不恢复原比赛设备在线服务，不为 XRD 或其他项目自动部署，也不承诺量产、田间或商业支持。

## 行为与申诉

所有参与者受 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) 约束。对审阅或治理决定有异议，可在原 Issue/PR 提出一次基于证据的复议；涉及隐私或行为事件时使用私下渠道。maintainer 应公开记录非敏感结论和理由。

## English summary

Safety, truthful evidence, licensing/privacy, and reproducibility take precedence over features. Routine changes require one maintainer/CODEOWNER review; runtime/release changes require CI and audit; actuator or asset changes require explicit design and evidence review. Maintainers decide after public discussion when consensus is not reached, but cannot waive the project’s truth, privacy, licensing, or fail-closed principles.
