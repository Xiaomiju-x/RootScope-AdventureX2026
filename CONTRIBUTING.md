# Contributing to RootScope / 参与贡献

感谢你帮助 RootScope 变得更安全、更清楚、更容易复现。我们欢迎文档、测试、跨平台兼容、数据治理、模型卡、硬件设计审阅和 fail-closed 逻辑改进。

项目是已完成的 AdventureX 2026 获奖原型。维护目标是复现、审计和教育，不会为了“功能更多”牺牲真实性或执行器安全。

## 开始之前

1. 阅读 [README](README.md)、[架构](docs/ARCHITECTURE.md)、[安全政策](SECURITY.md) 和 [治理](GOVERNANCE.md)。
2. 搜索已有 Issue/PR；较大改动先开提案。
3. 不要在 Issue、PR、commit、日志或附件中放入账号密码、token、私钥、真实私网地址、MAC、设备 ID、用户名绝对路径或未获授权的人像/数据。
4. 不要建议跳过心跳、锁存、身份核验、硬超时、限位、人工确认或最终关泵。
5. 真实硬件测试需要在自己的设备、受控现场和人工监护下进行；本仓库不授权操作比赛归档设备。

## 本地开发

```bash
git clone https://github.com/Xiaomiju-x/RootScope-AdventureX2026.git
cd RootScope-AdventureX2026
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
python tools/audit_public_release.py
```

保持提交小而聚焦。推荐分支名：`fix/...`、`docs/...`、`test/...`、`feat/...`。提交信息清楚说明意图和真实性边界；不要求把多人协作压缩成一个巨大 commit。

## 改动要求

### 代码

- 新行为必须有测试，尤其是缺失、冲突、过期、OOD、超时和恢复路径。
- 保持 LLM/RAG 零物理权限；自由文本不能进入档位或协议字段。
- 默认配置必须是 fixture/disabled/fail-closed，不包含真实设备路径。
- 设备 I/O 使用显式接口并支持假实现；导入模块不应产生物理副作用。
- 物理动作失败不能自动重试。
- 不为通过测试而降低安全断言或吞掉异常。

### 文档与主张

- 区分 PC、X5 CPU、actual BPU、回放、shadow、sim-only 和真实物理观察。
- 固定卡结果不能表述为开放世界精度；步数不能表述为长度或根深。
- benchmark 附环境、提交、输入/模型哈希、样本量和限制。
- 改动行为时同步 README/指南/schema/示例。

### 数据、模型与媒体

- 数据必须有来源、作者、许可证、source group、哈希和再分发权限。
- `rights_approved=false` 或许可不清的素材不能进 PR。
- 模型大文件先提案；必须有模型卡、上游修订、SHA-256、运行时、验证边界和许可证。
- 使用 Git LFS 跟踪批准的大文件，禁止提交 checkpoint/cache/基础模型重复副本。
- 人物素材需确认发布同意并移除 EXIF/GPS、位置水印、胸牌、二维码和敏感屏幕。

### 固件/硬件

- 清楚说明目标板、工具链、引脚、电气有效电平和上电默认态。
- 修改 V15 安全合同需要设计说明、主机侧测试、无动作台架回执和人工审阅。
- 不接受默认开泵、无界开泵、自动解锁、自动回升或移除看门狗的改动。
- 不把 GPIO 直接驱动电机/泵/继电器线圈当成可接受接线。

## Pull request 清单

- [ ] 改动范围单一，关联 Issue/设计讨论；
- [ ] 测试通过，新增行为有失败路径覆盖；
- [ ] `python tools/audit_public_release.py` 通过；
- [ ] 没有凭据、私有身份、绝对路径或不明许可资产；
- [ ] 文档、schema、模型卡/数据卡已同步；
- [ ] 结果表述包含证据等级和限制；
- [ ] 如涉及硬件，明确“未上板/只读/无动作/真实动作”的实际状态；
- [ ] 确认贡献可按仓库许可证分发。

维护者可能要求拆分 PR、增加回执或保留失败状态。合并通常需要至少一位 CODEOWNER 审阅；影响物理权限、许可或公开主张的改动可能要求额外审阅。

## English summary

Contributions are welcome when they improve reproducibility, tests, documentation, governance, or fail-closed safety. Never submit credentials, device identity, private topology, unlicensed data, or unsanitized people/media. New behavior needs tests—especially failure paths. LLM/RAG must remain unable to operate hardware, and physical failures must never trigger automatic retries. Run the test suite and public-release audit before every PR.
