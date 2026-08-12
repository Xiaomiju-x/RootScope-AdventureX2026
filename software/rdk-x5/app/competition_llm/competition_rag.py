"""Strict adapter from the rich competition JSONL pack to Omega FTS5/BM25.

The existing :mod:`app.omega_knowledge` store is reused only as an immutable
in-memory retrieval engine. Retrieved rows are the only passages supplied to
the competition LLM.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from app.omega_knowledge import KnowledgeChunk, KnowledgeStore, SourceRecord

from .contracts import (
    MAX_CORPUS_BYTES,
    MAX_CORPUS_CHUNKS,
    MAX_QUERY_CHARS,
    CompetitionLlmError,
    CorpusChunk,
    LoopbackConfig,
    bounded_one_line,
    canonical_bytes,
    sha256_bytes,
)
from .guard import assess_untrusted_text
from .runtime import CompactModel, _run_with_candidates


RETRIEVAL_BACKEND = "SQLITE_FTS5_BM25"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_BINDING_FIELDS = (
    "source_id",
    "publisher",
    "source_type",
    "title",
    "locator",
    "version",
    "license",
    "use_boundary",
    "public_safe",
    "source_sha256",
)
_SOURCE_KEYS = frozenset({*_SOURCE_BINDING_FIELDS, "source_binding_sha256"})
_ROW_KEYS = frozenset(
    {
        "schema",
        "id",
        "source",
        "title",
        "locator",
        "version",
        "license",
        "use_boundary",
        "paragraph",
        "text",
        "content_sha256",
        "citation_id",
        "public_safe",
    }
)
_SOURCE_TYPE_MAP = {
    "OFFICIAL_WEB": "OFFICIAL_WEB",
    "PAPER": "PAPER",
    "DATASET": "DATASET",
    "MANUAL": "MANUAL",
    "LOCAL_FILE": "LOCAL_FILE",
    "LOCAL_EVIDENCE": "LOCAL_FILE",
    "LOCAL_PLAN": "LOCAL_FILE",
    "LOCAL_CODE": "LOCAL_FILE",
}


def _load_json(path: Path, *, maximum: int = MAX_CORPUS_BYTES) -> Any:
    raw = Path(path).read_bytes()
    if not raw or len(raw) > maximum:
        raise CompetitionLlmError(
            f"{Path(path).name} must contain 1..{maximum} bytes"
        )
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompetitionLlmError(
            f"{Path(path).name} must contain valid UTF-8 JSON"
        ) from exc


def _file_sha256(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _source_fingerprint(source: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_bytes(
            {field: source.get(field) for field in _SOURCE_BINDING_FIELDS}
        )
    )


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CompetitionLlmError(f"{name} must be lowercase SHA-256")
    return value


def _load_registry(path: Path) -> tuple[dict[str, Mapping[str, Any]], str]:
    value = _load_json(path)
    expected_top = {
        "schema",
        "generated_at_utc",
        "allowed_web_domains",
        "local_root",
        "sources",
    }
    if not isinstance(value, dict) or set(value) != expected_top:
        raise CompetitionLlmError("competition source registry keys mismatch")
    if value["schema"] != "rootscope.competition.rag-source-registry.v1":
        raise CompetitionLlmError("competition source registry schema mismatch")
    rows = value["sources"]
    if not isinstance(rows, list) or not rows or len(rows) > 128:
        raise CompetitionLlmError("competition source registry size is invalid")
    sources: dict[str, Mapping[str, Any]] = {}
    for index, source in enumerate(rows):
        if not isinstance(source, dict) or set(source) != _SOURCE_KEYS:
            raise CompetitionLlmError(f"registry source {index} keys mismatch")
        source_id = source["source_id"]
        if not isinstance(source_id, str) or not source_id:
            raise CompetitionLlmError(f"registry source {index} id is invalid")
        if source_id in sources:
            raise CompetitionLlmError(f"duplicate registry source: {source_id}")
        if source.get("public_safe") is not True:
            raise CompetitionLlmError(
                f"registry source {source_id} is not public-safe"
            )
        binding = _require_sha256(
            source.get("source_binding_sha256"),
            f"registry source {source_id} binding",
        )
        if binding != _source_fingerprint(source):
            raise CompetitionLlmError(
                f"registry source {source_id} binding hash mismatch"
            )
        if source.get("source_type") not in _SOURCE_TYPE_MAP:
            raise CompetitionLlmError(
                f"registry source {source_id} type is unsupported"
            )
        sources[source_id] = source
    return sources, _file_sha256(path)


def _load_allowlist(path: Path) -> tuple[set[str], str]:
    value = _load_json(path)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "source_ids", "citation_ids"}
        or value.get("schema")
        != "rootscope.competition.rag-citation-allowlist.v1"
    ):
        raise CompetitionLlmError(
            "competition citation allowlist schema mismatch"
        )
    citations = value["citation_ids"]
    if (
        not isinstance(citations, list)
        or not citations
        or any(not isinstance(item, str) for item in citations)
        or len(citations) != len(set(citations))
    ):
        raise CompetitionLlmError("competition citation allowlist is invalid")
    return set(citations), _file_sha256(path)


@dataclass
class CompetitionRagIndex:
    """Owned in-memory FTS5 index with immutable source/citation receipts."""

    store: KnowledgeStore
    chunks: tuple[CorpusChunk, ...]
    corpus_path: Path
    corpus_sha256: str
    registry_sha256: str
    allowlist_sha256: str
    allowlist: frozenset[str]
    integrity: Mapping[str, Any]

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "CompetitionRagIndex":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def search(
        self, query: str, *, limit: int = 3
    ) -> tuple[tuple[CorpusChunk, ...], dict[str, float]]:
        try:
            hits = self.store.search(query, limit=limit)
        except Exception as exc:
            raise CompetitionLlmError(
                f"FTS5/BM25 retrieval rejected query: {type(exc).__name__}"
            ) from exc
        by_id = {chunk.citation_id: chunk for chunk in self.chunks}
        selected: list[CorpusChunk] = []
        scores: dict[str, float] = {}
        for hit in hits:
            if hit.citation_id not in self.allowlist:
                raise CompetitionLlmError(
                    "FTS5/BM25 returned a citation outside the frozen allowlist"
                )
            chunk = by_id.get(hit.citation_id)
            if chunk is None:
                raise CompetitionLlmError(
                    "FTS5/BM25 returned an unbound citation"
                )
            if (
                chunk.source_sha256 != hit.source_sha256
                or chunk.chunk_sha256 != hit.chunk_sha256
            ):
                raise CompetitionLlmError(
                    "FTS5/BM25 hit hash differs from the loaded corpus"
                )
            selected.append(chunk)
            scores[hit.citation_id] = float(hit.bm25_score)
        return tuple(selected), scores


def load_competition_rag(
    *,
    corpus_path: Path,
    registry_path: Path,
    allowlist_path: Path,
) -> CompetitionRagIndex:
    """Validate the rich pack and build the actual Omega FTS5/BM25 index."""

    sources, registry_sha256 = _load_registry(registry_path)
    frozen_allowlist, allowlist_sha256 = _load_allowlist(allowlist_path)
    raw = Path(corpus_path).read_bytes()
    if not raw or len(raw) > MAX_CORPUS_BYTES:
        raise CompetitionLlmError("competition corpus byte size is invalid")
    corpus_sha256 = sha256_bytes(raw)
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise CompetitionLlmError("competition corpus must be UTF-8 JSONL") from exc

    store = KnowledgeStore(":memory:")
    try:
        source_records: dict[str, SourceRecord] = {}
        chunks_by_source: dict[str, list[KnowledgeChunk]] = {}
        report_chunks: list[CorpusChunk] = []
        seen_ids: set[str] = set()
        seen_citations: set[str] = set()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CompetitionLlmError(
                    f"competition corpus line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, dict) or set(row) != _ROW_KEYS:
                raise CompetitionLlmError(
                    f"competition corpus line {line_number} keys mismatch"
                )
            if row["schema"] != "rootscope.competition.rag-chunk.v1":
                raise CompetitionLlmError(
                    f"competition corpus line {line_number} schema mismatch"
                )
            if row["public_safe"] is not True:
                raise CompetitionLlmError(
                    f"competition corpus line {line_number} is not public-safe"
                )
            source_id = row["source"]
            source = sources.get(source_id)
            if source is None:
                raise CompetitionLlmError(
                    f"competition corpus line {line_number} source is unregistered"
                )
            for row_key, source_key in (
                ("title", "title"),
                ("locator", "locator"),
                ("version", "version"),
                ("license", "license"),
                ("use_boundary", "use_boundary"),
            ):
                if row[row_key] != source[source_key]:
                    raise CompetitionLlmError(
                        f"competition corpus line {line_number} source binding mismatch"
                    )
            chunk_id = row["id"]
            paragraph = row["paragraph"]
            text = row["text"]
            citation_id = row["citation_id"]
            if not all(
                isinstance(item, str) and item
                for item in (chunk_id, paragraph, text, citation_id)
            ):
                raise CompetitionLlmError(
                    f"competition corpus line {line_number} text fields are invalid"
                )
            if chunk_id in seen_ids or citation_id in seen_citations:
                raise CompetitionLlmError(
                    f"competition corpus line {line_number} duplicates identity"
                )
            seen_ids.add(chunk_id)
            seen_citations.add(citation_id)
            expected_citation = f"{source_id}#{paragraph}@{chunk_id}"
            if citation_id != expected_citation:
                raise CompetitionLlmError(
                    f"competition corpus line {line_number} citation binding mismatch"
                )
            content_sha256 = _require_sha256(
                row["content_sha256"],
                f"competition corpus line {line_number} content hash",
            )
            if content_sha256 != sha256_bytes(text.strip().encode("utf-8")):
                raise CompetitionLlmError(
                    f"competition corpus line {line_number} content hash mismatch"
                )
            if citation_id not in frozen_allowlist:
                raise CompetitionLlmError(
                    f"competition corpus line {line_number} citation is not allowlisted"
                )
            binding_sha256 = source["source_binding_sha256"]
            if source_id not in source_records:
                source_records[source_id] = SourceRecord(
                    source_id=source_id,
                    title=source["title"],
                    locator=source["locator"],
                    source_type=_SOURCE_TYPE_MAP[source["source_type"]],
                    version=source["version"],
                    license=source["license"],
                    sha256=binding_sha256,
                )
            knowledge_chunk = KnowledgeChunk(
                chunk_id=chunk_id,
                source_id=source_id,
                paragraph_id=paragraph,
                text=text.strip(),
                sha256=content_sha256,
            )
            if knowledge_chunk.citation_id != citation_id:
                raise CompetitionLlmError(
                    f"competition corpus line {line_number} store citation mismatch"
                )
            chunks_by_source.setdefault(source_id, []).append(knowledge_chunk)
            report_chunks.append(
                CorpusChunk(
                    citation_id=citation_id,
                    title=source["title"],
                    text=text.strip(),
                    locator=source["locator"],
                    source_sha256=binding_sha256,
                    chunk_sha256=content_sha256,
                )
            )
            if len(report_chunks) > MAX_CORPUS_CHUNKS:
                raise CompetitionLlmError(
                    f"competition corpus exceeds {MAX_CORPUS_CHUNKS} chunks"
                )
        if not report_chunks:
            raise CompetitionLlmError("competition corpus contains no chunks")
        if seen_citations != frozen_allowlist:
            raise CompetitionLlmError(
                "competition corpus and frozen citation allowlist differ"
            )
        for source_id in sorted(source_records):
            store.add_documents(
                source_records[source_id],
                chunks_by_source[source_id],
            )
        integrity = store.integrity_report()
        if integrity.get("passed") is not True:
            raise CompetitionLlmError("Omega knowledge index integrity check failed")
        return CompetitionRagIndex(
            store=store,
            chunks=tuple(report_chunks),
            corpus_path=Path(corpus_path),
            corpus_sha256=corpus_sha256,
            registry_sha256=registry_sha256,
            allowlist_sha256=allowlist_sha256,
            allowlist=frozenset(frozen_allowlist),
            integrity=integrity,
        )
    except Exception:
        store.close()
        raise


def run_competition_rag_microcluster(
    *,
    query: str,
    corpus_path: Path,
    registry_path: Path,
    allowlist_path: Path,
    config: LoopbackConfig,
    model: CompactModel | None = None,
) -> dict[str, Any]:
    """Run one compact inference using only rich-pack FTS5/BM25 hits."""

    query = bounded_one_line(query, "query", MAX_QUERY_CHARS)
    with load_competition_rag(
        corpus_path=corpus_path,
        registry_path=registry_path,
        allowlist_path=allowlist_path,
    ) as index:
        if assess_untrusted_text(query).blocked:
            candidates: tuple[CorpusChunk, ...] = ()
            scores: dict[str, float] = {}
        else:
            candidates, scores = index.search(query, limit=3)
        details = {
            "indexed_chunk_count": len(index.chunks),
            "candidate_count": len(candidates),
            "bm25_scores": scores,
            "corpus_sha256": index.corpus_sha256,
            "registry_sha256": index.registry_sha256,
            "citation_allowlist_sha256": index.allowlist_sha256,
            "index_integrity_sha256": sha256_bytes(
                canonical_bytes(index.integrity)
            ),
            "index_integrity_passed": index.integrity.get("passed") is True,
            "sqlite_integrity": index.integrity.get("sqlite_integrity"),
            "fts_row_count": index.integrity.get("fts_row_count"),
        }
        return _run_with_candidates(
            query=query,
            corpus_path=corpus_path,
            config=config,
            chunks=index.chunks,
            candidates=candidates,
            model=model,
            retrieval_backend=RETRIEVAL_BACKEND,
            retrieval_details=details,
        )


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    competition = root / "configs/competition"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9080")
    parser.add_argument(
        "--model-id",
        default="qwen2-0.5b-q4km-rootscope-competition",
    )
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--api-mode", choices=("chat", "completion"), default="chat"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=competition / "rootscope_rag_corpus.v1.jsonl",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=competition / "rootscope_rag_sources.v1.json",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=competition / "rootscope_rag_citation_allowlist.v1.json",
    )
    parser.add_argument(
        "--query",
        default="RootScope 的安全边界和 RDK X5 有限资源方案是什么？",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_competition_rag_microcluster(
        query=args.query,
        corpus_path=args.corpus,
        registry_path=args.registry,
        allowlist_path=args.allowlist,
        config=LoopbackConfig(
            endpoint=args.endpoint,
            model_id=args.model_id,
            model_sha256=args.model_sha256,
            timeout_seconds=args.timeout,
            api_mode=args.api_mode,
        ),
    )
    rendered = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0 if report["provenance"]["model_output_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
