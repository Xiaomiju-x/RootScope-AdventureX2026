# RootScope Competition LLM/RAG micro-cluster

This package is the memory-bounded presentation path for a 4 GB RDK X5:

- one existing Qwen2 0.5B OpenAI-compatible endpoint on `127.0.0.1`;
- one completion capped at 64 model tokens;
- one compact `e/a/q` JSON object projected into the three logical roles
  `EVIDENCE_EXPLAINER`, `SAFETY_AUDITOR`, and `DEFENSE_QA`;
- deterministic Markdown/JSONL retrieval, strict citation allowlists, prompt
  injection blocking, exact response keys, and an all-false authority capsule;
- no camera, serial, GPIO, pump, state-machine, tool-use, external network, or
  model-service start interface.

The CLI is a client only. It **does not** start `llama-server`:

```bash
python -m app.competition_llm \
  --endpoint http://127.0.0.1:9080 \
  --model-id qwen2-0.5b-q4km-rootscope-competition \
  --model-sha256 <64-lowercase-hex> \
  --api-mode chat \
  --corpus configs/omega/field_knowledge.v1.md \
  --query "解释 RootScope 当前证据、安全边界和 X5 部署状态" \
  --output evidence/competition_llm.json
```

Use `--api-mode completion` for llama.cpp's legacy `/completion` endpoint.
The mode is explicit so a missing route never causes a second inference retry.

JSONL corpus rows use this bounded contract:

```json
{"id":"K01","title":"Product boundary","text":"Source text","locator":"docs/facts.md"}
```

`id` and `text` are required. `title` and `locator` are optional. Any other
key is rejected so the corpus contract stays auditable.

For the reviewed rich pack, use the real SQLite FTS5/BM25 adapter:

```bash
python -m app.competition_llm.competition_rag \
  --model-sha256 <64-lowercase-hex> \
  --corpus configs/competition/rootscope_rag_corpus.v1.jsonl \
  --registry configs/competition/rootscope_rag_sources.v1.json \
  --allowlist configs/competition/rootscope_rag_citation_allowlist.v1.json
```

This path verifies registry bindings, content hashes and the frozen citation
allowlist before building the in-memory `app.omega_knowledge` index. Its report
states `retrieval_backend=SQLITE_FTS5_BM25`; only returned and injection-safe
hits enter the compact LLM prompt. Long citation IDs are represented inside the
64-token completion by `C1..C3` and expanded back to their hash-bound IDs only
after strict validation.

## 4 GB X5 competition orchestrator

`tools/start_x5_competition_runtime_v2.sh` is the foreground orchestrator for
the v2 candidate. It:

- verifies the exact X5 identity, candidate `SHA256SUMS`, r7 bin,
  `llama-server` and Qwen2 GGUF before starting anything;
- imports RootScope Python code from the candidate release itself through
  `PYTHONPATH=$RELEASE_ROOT/rootscope`;
- keeps exactly one Qwen2 0.5B `llama-server` on `127.0.0.1:9080`;
- invokes `app.competition_llm.competition_rag` exactly once and projects that
  single completion into three logical roles;
- keeps the r7 BPU process on a run-scoped AF_UNIX socket;
- optionally enters Competition Live v2 only when `--live` is explicit;
- owns and cleans only the two processes it started.

The orchestrator accepts CLI exit `0` (strict model output accepted) and exit
`2` (deterministic cited fallback). Both are one-call, zero-authority outcomes;
neither enables tools or physical actions. It never registers a service,
creates boot persistence, scans devices, opens serial/GPIO/pump interfaces, or
uses a non-loopback LLM endpoint.

```bash
bash rootscope/tools/start_x5_competition_runtime_v2.sh --no-live
# Only after the USB camera is physically ready:
bash rootscope/tools/start_x5_competition_runtime_v2.sh --live
```

Every invocation creates a new private directory below
`~/.local/share/rootscope-competition-runtime/runs/` containing candidate
checksum verification, dependency evidence, real backend logs, the one-call
three-role report, a ready receipt, and a cleanup receipt.
