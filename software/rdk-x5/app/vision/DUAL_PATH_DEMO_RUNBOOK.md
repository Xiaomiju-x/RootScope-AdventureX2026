# RootScope 双路径视觉证据现场流程

## 当前边界

这条链路只面向三天现场的**已登记打印卡片演示**：seed17 CPU ONNX 给出四类 raw top1 假设，AKAZE/ORB + RANSAC 单独验证某一张已登记模板。只有“恰好一个模板通过几何 + 类别一致 + 显式 demo 阈值通过”时，才显示 `EXPERIMENTAL_KNOWN_CARD_CONSENSUS`。

它不证明开放世界植物识别能力。seed17 的固定状态为 `MACHINE_CURATED_EXPERIMENTAL_NOT_QUALIFIED`；正式 per-class rejection gate 因样本量不足仍然拒绝全部。raw top1 只能标成 `DEMO_HYPOTHESIS`。双路径共识也不能控制泵、串口或状态机，所有 authority 字段固定为 `false`。

当前已冻结的实验注册表为 [`known_card_template_registry.frozen.experimental.json`](known_card_template_registry.frozen.experimental.json)：只登记 `163498042`（草丛）、`68787114`（灌木）和 `92774234`（幼树）三张训练参考，角色统一为 `DEMO_REFERENCE_NOT_HOLDOUT_ONCE_REGISTERED`。它们不是 holdout 或泛化证据；`157364276` 沙丘负例保持未登记。独立登记收据位于 `adventurex/evidence/rootscope_demo_template_registry_receipt_20260717.json`。空注册表示例仍保留为失败路径夹具。

## 赛前采集与冻结

1. 当前三张植物原图与来源信息已经按 SHA 冻结；到场先复核注册表、登记收据和打印 PDF 的哈希。`157364276` 是沙丘/无目标负例，只用于验证系统能够拒绝，**禁止登记为模板**。
2. 每个类别先只打印一张，保留统一白边和不重复的卡片编号。不要把两个打印件登记为同一 `template_id`，也不要登记相同 raw SHA。
3. 用现场实际 UVC 相机完成开发采集：正视、左右倾角、远近、局部反光、正常亮度。开发帧只用于调几何阈值，不计作独立测试证据。
4. 当前打印源已逐字节复制到 `known_card_templates/`，严格注册表状态为 `FROZEN_EXPERIMENTAL_DEMO_REFERENCES`，并绑定原数据记录 ID、来源 manifest、原始页面和 attribution。若换图、裁剪或改文件，必须用登记工具重建注册表与收据，不能手改哈希。
5. 一旦图片进入注册表，其角色立即变成 `DEMO_REFERENCE_NOT_HOLDOUT_ONCE_REGISTERED`。如果选择 `38233728`、`66745979` 或任何现有 holdout，必须先从 holdout 评估口径和分母中移除，再冻结新 manifest/receipt；禁止同时宣称“模板”和“未见 holdout”。
6. 冻结注册表 raw SHA、每张模板 raw SHA、模型 SHA、阈值 JSON、matcher config 与代码版本。冻结后不再用测试帧调阈值。
7. 重新采集一组未参与调参的 UVC 复拍：每张卡至少覆盖正视、倾斜、距离和光照。逐帧保存原图和双路径 JSON；报告通过、拒绝、多模板冲突和类别分歧，不删失败帧。

## 现场运行

先离线自检 clean-X5 capsule。下面命令只读一个已有 RGB 文件，不打开摄像头、串口或网络：

```bash
cd /opt/rootscope/current
/opt/rootscope/venv/bin/python3 -m app.vision.dual_path_demo \
  --query /opt/rootscope/evidence/capture_001.png \
  --registry /opt/rootscope/current/app/vision/known_card_template_registry.frozen.experimental.json \
  --capsule-config /opt/rootscope/current/deploy/x5/capsule_config.seed17_cpu_experimental.json \
  --thresholds-json /opt/rootscope/current/app/vision/dual_path_demo.thresholds.example.json \
  --matcher-config-json /opt/rootscope/current/app/vision/card_geometric_matcher.config.example.json \
  --output-json /opt/rootscope/evidence/capture_001.dual_path.json
```

退出码 `0` 只表示生成了实验性已知卡片共识，`2` 表示拒绝，`1` 表示合同或运行错误。三个退出码都不允许触发灌溉。

## 必须演示的失败模式

- 未登记图或完全无纹理图：`NO_REGISTERED_TEMPLATE_GEOMETRIC_PASS`；
- 两张登记模板同时通过：`MULTIPLE_REGISTERED_TEMPLATES_GEOMETRIC_PASS`；
- 语义 top1 与模板类别不同：`SEMANTIC_TEMPLATE_CLASS_DISAGREEMENT`；
- top1 概率或 margin 不足：`EXPERIMENTAL_SEMANTIC_DEMO_THRESHOLD_NOT_MET`；
- 模板文件被替换、路径逃逸、注册表多余字段或重复 ID/SHA：加载阶段 fail-closed；
- 任何几何结果声称 authority：该模板不能计为通过。

最终讲解口径应是：“语义模型提出类别假设，几何支路确认这是登记过的打印卡片；两条证据一致才显示实验性共识。系统仍保持人工确认与零执行权限。”
