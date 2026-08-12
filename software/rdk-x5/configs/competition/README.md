# RootScope Competition RAG v1

This directory contains the small, citation-bound knowledge pack used by the
AdventureX RootScope demonstration.  It is intentionally text-only and sized
for SQLite FTS5/BM25 plus one read-only 0.5B local model on the 4 GB RDK X5.

Files:

- `rootscope_rag_sources.v1.json`: source registry and source-domain allowlist.
- `rootscope_rag_corpus.v1.jsonl`: 24 short Chinese knowledge chunks.
- `rootscope_rag_citation_allowlist.v1.json`: the exact citation IDs that may
  leave retrieval.
- `rootscope_rag_gold_qa.v1.jsonl`: 20 positive retrieval/answer checks.
- `rootscope_rag_forbidden_qa.v1.jsonl`: 20 refusal and truth-boundary checks.
- `audit_competition_rag.py`: dependency-free integrity and policy audit.
- `audit_competition_rag_retrieval.py`: executable FTS5/BM25 routing and
  zero-authority response audit.

The pack does not contain actuator commands, credentials, unrelated materials
research content, or a plant-to-water-dose lookup table.  Agricultural formulae
are planning references only.  RootScope local observations remain explicitly
bounded to their evidence snapshots.

Run:

```text
python configs/competition/audit_competition_rag.py
python configs/competition/audit_competition_rag_retrieval.py
```

`--fix-hashes` is reserved for a reviewed content update.  It recomputes only
the deterministic source-binding and chunk-content hashes, then performs the
same full audit.

The retrieval audit builds the actual SQLite FTS5/BM25 index, checks all 20
gold and 20 forbidden questions against the reviewed citations, and exercises
the citation allowlist, command rejection and all-false authority report.  The
release builder requires both audit commands to return `PASS`.
