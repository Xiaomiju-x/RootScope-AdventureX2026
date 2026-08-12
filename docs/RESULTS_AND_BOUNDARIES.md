# 结果与真实性边界 / Results & Truth Boundaries

## 1. 比赛结果

RootScope 在 **AdventureX 2026 D-Robotics「Give AI a Body」赛道**获得**银奖、最终第 2 名**。项目的准确名称是“RootScope 固定式根区灌溉舱”。

公开媒体包含赛事舞台远景和团队获奖合影。舞台远景只说明现场/赛道背景，不能单独证明具体名次；团队合影中的银地瓜奖牌和两台 RDK X5 奖品提供视觉佐证。仓库不依据奖品数量、舞台文字或照片自行推导其他奖项等级。

## 2. 最终作品事实

- 最终形态是固定式灌溉舱，不是移动小车。
- Scout Mini 轮式底盘只被复用为承载架/供电底座，轮子和导航不在决赛控制链。
- RDK X5 负责固定相机视觉、双证据、CPU 本地 LLM/RAG、确定性档位和证据。
- STM32F103C8T6 V15 负责 PA0–PA3 单向探针、PB6 单泵、心跳、硬超时和锁存。
- 固定映射是纯沙 `0`、草丛 `1024`、灌木 `1536`、幼树 `2048` 步。
- 探针没有自动回升或顶部传感器，每轮由操作员手动回顶。
- 比赛演示的定时注水为 5 秒。

## 3. 已验证结果

| 主张 | 证据范围 | 准确表述 |
|---|---|---|
| 四卡视觉 | 受控现场的四张固定打印卡 | 4/4 固定卡双证据资格通过 |
| 同场景静态样本 | 冻结同场景 holdout | 8/8；不能作为独立开放世界测试 |
| 三档探针 | 受控、逐档、人工回顶 | 1024/1536/2048 相对步数分别完成验证 |
| 完整草丛链 | 两次现场事务 | 识别 → 1024 步下降 → 5 秒注水 → 关泵锁存 |
| BPU | 冻结输入、指定 X5/runtime | 两条 actual BPU 回放路径各 43/43；记录 mean cosine 1.0 |
| RootMind | X5 CPU、按需加载 | Fast/Deep/BM25-HOLD 只读解释路径运行 |
| RAG2 快照 | 冻结 corpus/gold/forbidden | BM25 Recall@5 92.19%、hard Top-3 84.09%、Forbidden Recall@5 94.44%、Citation Escape 0 |
| 失败方向 | 未知/冲突/过期/断联/超时 | HOLD/STOP、无自动物理重试、最终输出关闭 |

这些是冻结的项目快照，不是持续在线监控。每个百分比必须与对应 corpus、评测脚本、manifest 和 receipt 一起解释。

## 4. 不能外推

| 已观察 | 不能外推为 |
|---|---|
| 固定打印卡 4/4 | 任意野外植物 100% 准确率 |
| 1024/1536/2048 步 | 毫米、厘米、土层或生物根深 |
| 5 秒泵任务 | 准确水量、节水率或最佳农艺剂量 |
| MCU ACK/状态 | 肉眼已见出水、无漏水或机械必然成功 |
| actual BPU 43/43 | 开放世界泛化、持续延迟或物理动作权威 |
| LLM 固定任务结果 | 通用问答、自由生成或农艺专家能力 |
| 两次完整链成功 | 长期可靠性、无人值守或量产能力 |

## 5. 摄像头与性能用语

“4K”只描述相机能力；稳定现场演示链使用的分辨率/帧率应以具体回执为准（已有叙述为 1080p30）。任何延迟/内存/温度数字都是特定快照，需附环境、样本量和时间，不写成永久性能。

LLM 运行在 CPU；BPU 用于视觉。三类 RootMind 角色按需换入，不是并发常驻集群。

## 6. 公开证据

清理后的回执位于 [`evidence/public/`](../evidence/public/)。公开视频和图片在 [`assets/media/`](../assets/media/)，解释见 [`DEMO.md`](DEMO.md)。公开 receipt 删除真实 IP、MAC、machine-id、boot-id、序列号、绝对路径和凭据；清理不改变技术结果字段。

证据使用方法见 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)。如果媒体、机器回执和人工观察不一致，应保留差异并降低结论，不得选择性删除失败证据。

## 7. 不作出的主张

- 全球第一、唯一方案或其他团队没有；
- 完全自主、全自动回升、无人值守；
- 田间长期验证、量产、工业安全认证；
- 已验证节水百分比、植物生长改善或农艺处方；
- 远程网页直接控制泵/电机；
- 固定答辩卡等同真实沙漠植物；
- 编译器估算等同 actual X5 性能。

## English summary

RootScope won the Silver Award and finished second in the AdventureX 2026 D-Robotics “Give AI a Body” track. Its verified scope is a stationary, controlled-card prototype: four fixed cards, three relative downward presets, two complete supervised grass-card irrigation chains, on-device fixed-input BPU replays, and read-only CPU LLM/RAG. These results do not establish open-world plant accuracy, physical depth, water savings, agronomic benefit, unattended reliability, or safety certification.
