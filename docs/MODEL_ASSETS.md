# 模型资产 / Model Assets

本仓库区分三类东西：项目生成且允许再分发的最终资产、需要从上游获取的基础模型、仅保留合同/回执而不再分发的研究中间物。不要用文件名推断许可或资格；以当前 release 的 manifest、模型卡和 SHA-256 为准。

## 1. 最终运行角色

| 角色 | 典型格式/运行时 | 作用 | 物理权限 |
|---|---|---|---|
| RootSight CPU | ONNX / ONNX Runtime CPU | 固定答辩卡语义证据与回退 | 无 |
| RootSight BPU | Horizon `.bin` / Bayes-e | 固定输入资格、视觉执行/辅助证据 | 无 |
| 几何复核 | 模板 + AKAZE/RANSAC | 独立几何证据 | 无 |
| RootMind Fast | 量化 GGUF / CPU | 快速结构化解释 | 无 |
| RootMind Deep | Qwen3-1.7B 适配器/量化部署物 / CPU | 领域解释与引用 | 无 |
| BM25/HOLD | SQLite/JSON/NumPy | 检索与确定性降级 | 无 |

Fast、Deep、BM25/HOLD 是按需换入的逻辑微集群，不是多模型同时常驻。LLM 运行在 CPU；BPU 用于视觉。任何模型输出都不能直接编码 STM32 命令。

## 2. 资产目录规则

正式大文件放在 `model-assets/`，建议结构如下：

```text
model-assets/
  MANIFEST.json
  vision-cpu/
    MODEL_CARD.md
    *.onnx
  vision-bpu/
    MODEL_CARD.md
    *.bin
    conversion-receipt.json
  rootmind-adapter/
    MODEL_CARD.md
    adapter_config.json
    adapter_model.safetensors
```

每个可执行资产必须记录：

- 逻辑 ID、角色、格式、字节数和 SHA-256；
- 上游模型及其精确版本/提交；
- 训练/转换代码与参数入口；
- 输入形状、颜色顺序、归一化、输出语义；
- 目标硬件/运行时/工具链；
- 已验证、未验证和明确禁止的表述；
- 许可和再分发依据；
- `physical_authority: false`。

Git LFS 只解决大文件传输，不解决许可。下载后运行：

```bash
git lfs pull
git lfs ls-files
sha256sum model-assets/<ASSET>
```

将计算结果与 `MANIFEST.json` 对比。LFS 指针文本、缺失对象或哈希不符都不能进入复现。

## 3. 视觉模型

项目模型注册表保存在 [`research/rootscope_v3/registries/models.v1.json`](../research/rootscope_v3/registries/models.v1.json)。其中一些路径是开发归档的逻辑定位，不保证当前仓库直接包含对应二进制。

已有冻结登记包括：

- ResNet18 CPU ONNX：固定 224×224 RGB 输入，作为视觉主链/审计回退；
- ResNet18 Bayes-e `.bin`：针对 X5 的量化转换物，BPU 角色无动作权限；
- 固定卡模板：与打印卡和光学布置绑定。

43/43、mean cosine 1.0 只描述冻结回放样本和指定运行时。它不能推出自然植物精度、实时端到端精度或物理成功率。

## 4. RootMind 与第三方基础模型

本项目开发中使用了 Qwen 系列和 BGE 系列上游资产。基础模型通常体积大，且其许可证、使用政策和下载来源由上游维护；本仓库不把基础模型或合并后的冗余副本当作自研资产重复上传。

复现者应：

1. 阅读模型卡中记录的上游仓库与精确版本；
2. 直接从官方/上游来源下载；
3. 独立接受其许可证与使用条款；
4. 校验上游哈希/修订；
5. 再应用本仓提供的适配器或量化/合并流水线；
6. 保存自己的转换回执。

不要把 DeepSeek/Qwen 等离线教师输出视为 ground truth。RootMind 评测只适用于固定合同任务，不能宣传为通用自由生成能力。

## 5. 转换与复现入口

| 目标 | 入口 |
|---|---|
| 视觉训练 | `pipelines/training/train_rootscope_answer_cards.py` |
| 数据质量分析 | `pipelines/training/analyze_rootscope_v3_model_quality.py` |
| BPU staging/审计 | `pipelines/bpu/` |
| CPU/BPU 回放输入 | `pipelines/evaluation/build_rootscope_cpu_bpu_replay_inputs.py` |
| RootMind QLoRA | `research/rootscope_v3/llm/train_qwen3_17b_qlora.py` |
| 合并/封印 | `research/rootscope_v3/llm/merge_qwen3_17b_adapter.py`、`seal_final_gguf_v3.py` |
| RAG2 | `research/rootscope_v3/rag2/` |
| 发布构建/验收 | `pipelines/release_v3/` |

转换前先检查脚本的输入合同；开发归档中的绝对路径、缓存目录或设备配置不能原样复用。

## 6. 不发布的模型内容

- 上游基础权重的重复副本；
- 训练缓存、优化器状态、临时 checkpoint、合并前后冗余副本；
- 没有清楚许可证/来源的模型；
- 包含隐私或凭据的训练回执；
- 只编译但未做板端语义验收却被命名为“已验证”的模型；
- 为复现不必要的旧候选和失败中间物。

“完整开源”在此指完整源码、合同、可合法分发的最终产物与再生流程，不等于复制整个开发缓存。

## English summary

Every executable model asset must have a model card, SHA-256, byte size, upstream revision, input/output contract, target runtime, validation boundary, license, and `physical_authority: false`. Git LFS does not grant redistribution rights. Base Qwen/BGE models are obtained from their upstream sources; this repository publishes only assets it may redistribute plus the code and receipts needed to recreate the rest. Replay agreement is not open-world accuracy.
