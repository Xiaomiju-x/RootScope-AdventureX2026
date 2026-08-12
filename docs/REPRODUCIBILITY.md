# 复现协议 / Reproducibility

RootScope 采用“主张—资产—命令—回执—边界”绑定的复现方式。目标不是让复现者照抄一个脚本就通电，而是让每个结论都能追溯到明确版本，并区分软件结果、固定输入板端结果和真实物理观察。

## 1. 证据等级

| 等级 | 环境 | 可以说明 | 不能说明 |
|---|---|---|---|
| R0 静态审计 | Git 文件、哈希、schema | 资产完整、合同一致 | 代码实际运行 |
| R1 PC 软件 | 合成夹具、单测、CPU 回放 | 逻辑和固定输入结果 | X5/BPU 或物理性能 |
| R2 X5 CPU | 板端 CPU 固定输入 | 板端依赖和 CPU 输出 | BPU 执行 |
| R3 actual BPU | X5 上真实 BPU backend | 指定输入/模型在 BPU 执行 | 开放世界泛化或物理成功 |
| R4 无动作硬件 | 相机/STM32 只读、执行器断电 | 身份、协议、安全态 | 电机方向、泵出水 |
| R5 受控物理 | 人工监护的单次运动/通水 | 该次动作与最终状态 | 长期可靠性/无人值守 |

结果必须使用达到的最低准确等级。例如 PC mapper 编译估计不能写成“X5 真机延迟”，MCU ACK 不能代替现场对实际出水的观察。

## 2. 冻结输入

一次复现至少冻结：

- Git commit/tag；
- Python、OS、RDK runtime、BPU 工具链和编译器版本；
- 模型、模板、RAG pack、固件 HEX 的 SHA-256；
- 数据/评测 manifest SHA-256；
- 相机格式、光学配置和物理摆位版本；
- 运行命令、UTC 时间、退出码和原始输出；
- 最终安全状态与人工观察字段。

禁止在失败后静默修改阈值、输入或期望值，再沿用原 receipt ID。

## 3. 一键软件基线

```bash
python -m pip install -e ".[dev]"
pytest
python tools/audit_public_release.py
rootscope-public examples/grass_agree.json
rootscope-public examples/conflict_hold.json
rootscope-public examples/ood_hold.json
```

该基线是 R1，设备和模型二进制均非必需。保存输出并确认重复运行的规范化回执哈希一致（时间字段存在的完整工程回执除外）。

## 4. 研究合同复现

```bash
python -m pytest research/rootscope_v3/tests
python research/rootscope_v3/tools/verify_e0.py --adventurex-root research
```

再按模块运行：

- 视觉：`research/rootscope_v3/evaluations/run_vision_evaluation.py`；
- RAG2：`research/rootscope_v3/rag2/audit_rag2.py`；
- RootMind：`research/rootscope_v3/llm/evaluate_qwen3_17b_qlora.py`；
- 安全编译器：`research/rootscope_v3/llm/evaluate_safety_compiler_v3.py`；
- 发布门：`pipelines/release_v3/verify_pc_gates_v3.py`。

这些脚本来自比赛归档，部分需要模型/数据资产或历史格式。执行前通过配置参数指向复现者本地、已校验的相对资产目录；不要把归档中的旧绝对路径写回仓库。

## 5. 视觉复现

### 固定输入

1. 校验 CPU ONNX、BPU `.bin`、模板和四张输入图哈希。
2. 锁定预处理：224×224、RGB、归一化与输出 class order。
3. 对 CPU 路径保存 logits/decision/latency。
4. 在 X5 上对同样输入运行 actual BPU backend，记录加载和执行日志。
5. 比较完整输出，不只比较 top-1。
6. 几何复核单独记录模板、匹配点、inliers 和失败原因。

### 实时相机

固定相机、镜头、纸张、打印机、灯光、距离与曝光后，至少覆盖四张已知卡、未知卡、空场、手遮挡、模糊、反光和混合目标。按独立摆位/会话统计，不能把连续帧当独立样本。

## 6. RAG/LLM 复现

- RAG 报告 corpus/allowlist/gold/forbidden 的哈希和来源数量；
- 检索指标区分普通问题与 forbidden 问题；
- LLM 严格验证完整 JSON、回答字段、拒答字段和引用，不用“看起来合理”人工代替；
- 记录超时、解析失败、无引用和 BM25/HOLD 降级；
- 固定任务 holdout 的结果不能宣传为通用大模型能力；
- 证明 LLM 输出不进入动作档位或串口字段。

## 7. X5/BPU 复现

按 [RDK X5 部署](RDK_X5_DEPLOYMENT.md) 的 G0–G6 晋级。actual BPU 证据必须包含板端运行时/硬件标识的非敏感版本信息、输入/模型哈希、命令退出码和输出比较；PC 转换报告单独保存。

既有公开结论为两条固定回放路径 43/43、mean cosine 1.0。复现结果不同是需要分析的新事实，而不是通过复制旧 receipt 修正。

## 8. 固件与物理链复现

1. 按 [STM32 构建](STM32_BUILD.md) 构建并记录工具链/HEX 哈希；
2. 断开执行器动力，只读核验 V15 身份和输出关闭；
3. 无泵/假负载验证心跳、STOP、UART fault 和硬超时；
4. 人工确认危险区、回顶、供电和接水；
5. 每档只执行一次，不自动重试；
6. 单独记录 MCU 回执与现场实际运动/出水观察；
7. 最终独立查询输出关闭和锁存；
8. 下一档必须等待上一档明确通过。

最终比赛映射为纯沙 0、草丛 1024、灌木 1536、幼树 2048；固定注水为 5 秒。复现者不得把这些步数换算为未测量的物理深度。

## 9. Receipt 最小结构

```json
{
  "schema": "rootscope.reproduction.receipt.v1",
  "claim_id": "example-fixed-input-only",
  "evidence_level": "R1_PC_SOFTWARE",
  "git_commit": "<40_HEX_COMMIT>",
  "inputs": [{"path": "<RELATIVE_PATH>", "sha256": "<64_HEX>"}],
  "environment": {"python": "<VERSION>", "platform": "<NON_SENSITIVE>"},
  "command": ["python", "<ENTRYPOINT>"],
  "exit_code": 0,
  "result": {"passed": true},
  "physical_action_authority": false,
  "limitations": ["fixed inputs only"]
}
```

公开 receipt 不包含用户名、绝对路径、私网 IP、MAC、machine-id、boot-id、设备序列号或 token。必要的设备一致性可以用一次性不可逆哈希并在私有记录中验证，公开稿只保留非敏感合同。

## 10. 复现报告清单

- [ ] 资产清单和所有 SHA-256；
- [ ] 精确 commit/tag；
- [ ] 无未跟踪配置或本地补丁；
- [ ] 环境与工具链版本；
- [ ] 命令、退出码、原始输出；
- [ ] 失败样例和拒绝路径；
- [ ] 证据等级；
- [ ] 人工观察与机器回执分栏；
- [ ] 最终输出关闭/锁存；
- [ ] 限制与不可外推结论。

## English summary

Reproduction is claim-bound and tiered from static audit (R0) through supervised physical trials (R5). Freeze the commit, environment, model/data/firmware hashes, exact command, raw output, and limitations. Never describe PC compilation as BPU execution, a fixed-input replay as open-world accuracy, or an MCU acknowledgement as observed water delivery. Public receipts are sanitized and contain no device identity or private topology.
