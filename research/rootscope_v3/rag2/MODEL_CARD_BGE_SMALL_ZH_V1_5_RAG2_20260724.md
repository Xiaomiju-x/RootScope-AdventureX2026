# RootScope RAG 2.0 Dense Challenger Model Card

日期：2026-07-24  
状态：`PC_EXECUTED_NOT_SELECTED_X5_PENDING`

## 模型

- 上游：`BAAI/bge-small-zh-v1.5`
- 固定 revision：`7999e1d3359715c523056ef9478215996d62a620`
- 许可：MIT
- 上游 locator：
  `https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/7999e1d3359715c523056ef9478215996d62a620`
- 导出：ONNX opset 17
- 量化：ONNX Runtime dynamic per-channel `QUInt8` 权重
- 选定模型 SHA-256：
  `3f813fea8dbd30a883032affb2b0020add20a1dd31639bd2f6dae0b9e6665f6c`
- 选定模型大小：24,019,671 bytes（22.91 MiB）

## 表示合同

- 最大长度：128 tokens
- 文档：不加 instruction
- 查询：前缀 `为这个句子生成表示以用于检索相关文章：`
- pooling：按 attention mask 做 mean pooling，再 L2 normalize
- 文档矩阵：42 × 512，float16；不是常驻向量数据库
- tokenizer：项目内纯 Python WordPiece，已与固定 Hugging Face
  tokenizer 对中文、英文未知词和公式样例逐 token 对齐

## PC 量化验证

- FP32/UINT8 embedding cosine：min `0.995143`，mean `0.995824`
- batch=2、20 次 UINT8 推理：p50 `3.032 ms`，p95 `3.478 ms`
- RAG 完整查询的 dense p95：`6.271 ms`
- RRF p95：`7.322 ms`
- 加载后的 PC RSS 增量：`66.40 MiB`

这些是 Windows x86_64 PC 单轮实测，不是 RDK X5 数字。

## 冻结检索结论

评测包含 64 个 gold，其中 44 个为语义改写 hard query；另有 36 个
forbidden/authority query。

| 后端 | gold Top-3 | hard Top-3 | gold Top-5 |
|---|---:|---:|---:|
| FTS5/BM25 | 89.06% | 84.09% | 92.19% |
| BGE dense | 76.56% | 75.00% | 87.50% |
| equal-weight RRF | 87.50% | 84.09% | 89.06% |

RRF 相对 BM25 的 hard Top-3 绝对增益为 `0`，配对精确检验
`p=1.0`，且全部 gold Top-3 下降 `1.5625` 个百分点。因此它没有通过
预先写定的“至少 +5 个百分点、p<=0.05、总体不回退”晋级门。

## 部署决策

默认部署继续使用 `SQLite FTS5/BM25 v2`。该 dense 模型只保留为
PC 研究资产，不进入默认 X5 常驻链；也不需要为没有收益的模型占用
4GB X5 的内存。若未来重建独立领域语义数据，只能在新的未见测试集上
重新挑战，不能用本轮 hard set 继续调权重。

## 权限和真实性边界

- `execution_authority=false`
- `physical_authority=false`
- `serial_write=false`
- `pump_command=false`
- 未接触断电 X5、相机、串口、GPIO 或泵
- PC ONNX 成功不等于 X5 兼容、延迟或资源资格
