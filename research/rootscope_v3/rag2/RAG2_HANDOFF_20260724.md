# RootScope RAG 2.0 PC 交接

日期：2026-07-24  
状态：`RAG2_BM25_PC_QUALIFIED_DENSE_NOT_ELIGIBLE_X5_PENDING`

## 结论

RAG 2.0 已在 PC 侧完成可部署前准备，现有 v1 未被覆盖：

- 17 个登记来源；
- 42 个带 locator、license、use boundary 和 hash 的短知识块；
- 64 个 gold，其中 44 个为冻结的语义改写 hard query；
- 36 个越权/事实边界 query；
- SQLite FTS5/BM25 持久索引；
- BAAI BGE small 中文 ONNX UINT8 实际挑战器；
- 引用 allowlist、资源、量化漂移和零权限审计。

最终默认选择是 `SQLite FTS5/BM25 v2`。Dense + RRF 没有提高冻结 hard
query，因此被明确淘汰出默认现场链。这不是“没做向量检索”，而是实际
完成模型下载、导出、量化、执行和配对比较后作出的资源收益决策。

## PC 实测

- BM25 gold recall@1/3/5：`73.44% / 89.06% / 92.19%`
- BM25 hard Top-3：`84.09%`
- BM25 forbidden boundary recall@5：`94.44%`
- BM25 PC p95：`0.695 ms`
- RRF hard Top-3 增益：`0`
- RRF 全量 Top-3 变化：`-1.5625` 个百分点
- 配对精确检验：`p=1.0`
- BGE UINT8：22.91 MiB，PC RRF p95 `7.322 ms`
- BGE FP32/UINT8 embedding cosine min：`0.995143`
- PC 载入 dense 后 RSS 增量：`66.40 MiB`
- citation allowlist 越界：`0`

## 默认部署文件

- `pack/rootscope_rag_sources.v2.json`
- `pack/rootscope_rag_corpus.v2.jsonl`
- `pack/rootscope_rag_citation_allowlist.v2.json`
- `pack/rag2_index.sqlite3`
- `bm25_runtime.py`
- `deploy_selection.v1.json`

`hybrid_index.py` 与 Dense 资产只保留为 PC 消融研究证据；最终选择由
`deploy_selection.v1.json` 固定为 stdlib-only `bm25_runtime.py`，默认 release
不携带 Dense 模型或 embedding。

Dense 研究资产在 `rootscope_v3/models/rag/` 与
`pack/corpus_embeddings.*`，默认 release 可以不携带。

## 验证

```powershell
.\.ai_curation_venv\Scripts\python.exe rootscope_v3\rag2\build_rag2.py
.\.ai_curation_venv\Scripts\python.exe rootscope_v3\rag2\build_index.py
.\.ai_curation_venv\Scripts\python.exe rootscope_v3\rag2\audit_rag2.py
.\.ai_curation_venv\Scripts\python.exe -m unittest discover -s rootscope_v3\rag2\tests -v
```

权威回执：

`evidence/rag2_audit_20260724.json`

## 最终内容寻址候选的剩余门

默认 BM25 路径只需验证：

1. X5 SQLite 是否带 FTS5；
2. 42 行索引完整性与 SHA；
3. 64/36 检索回放；
4. p50/p95、RSS 和 30 分钟只读 soak；
5. 与 RootMind 的 citation allowlist 接口。

Dense 默认不部署，所以 X5 不需要为它安装额外服务或常驻模型。若队长
仍希望展示研究过程，可把模型作为“PC 已实测但未晋级”的消融证据。

## 真实性边界

- 本轮只在 PC 执行；
- X5 已上电并做过外部只读预验收；最终内容寻址 candidate 的 BM25
  不可变数据库验收与资源 soak 仍是
  `FINAL_CANDIDATE_ACCEPTANCE_PENDING`；
- 未打开相机、串口、GPIO、泵或物理设备；
- 未读取或输出任何 API 密钥；
- RAG 与 LLM 的 execution/physical/serial/pump authority 全为 false。
