#!/usr/bin/env python3
"""Exercise the frozen RootScope competition pack through real FTS5/BM25.

This is a release gate, not a model-quality benchmark. It proves that every
reviewed question routes to an expected citation (or is rejected by the direct
input guard), then exercises the read-only response report with a deterministic
non-networked audit model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
ROOTSCOPE_ROOT = HERE.parents[1]
if str(ROOTSCOPE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROOTSCOPE_ROOT))

from app.competition_llm.competition_rag import (  # noqa: E402
    RETRIEVAL_BACKEND,
    load_competition_rag,
    run_competition_rag_microcluster,
)
from app.competition_llm.contracts import LoopbackConfig  # noqa: E402
from app.competition_llm.guard import assess_untrusted_text  # noqa: E402


CORPUS_PATH = HERE / "rootscope_rag_corpus.v1.jsonl"
REGISTRY_PATH = HERE / "rootscope_rag_sources.v1.json"
ALLOWLIST_PATH = HERE / "rootscope_rag_citation_allowlist.v1.json"
GOLD_PATH = HERE / "rootscope_rag_gold_qa.v1.jsonl"
FORBIDDEN_PATH = HERE / "rootscope_rag_forbidden_qa.v1.jsonl"
SEARCH_LIMIT = 5
COMMAND_SHAPED = (
    re.compile(r"(?i)\b(?:sudo|systemctl|ssh|curl|wget)\b"),
    re.compile(r"(?i)(?:/dev/tty|(?:serial|uart|gpio)\s*\.\s*(?:write|output))"),
    re.compile(r"(?:启动|打开|写入|触发|执行).{0,12}(?:水泵|串口|GPIO|命令)"),
)


class AuditFailure(RuntimeError):
    """Raised when the executable retrieval/response contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"{path.name}:{line_number}: row must be an object",
        )
        rows.append(value)
    return rows


class DeterministicReadOnlyAuditModel:
    """Return schema-valid benign text without opening a socket or tool."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self,
        _messages: Sequence[Mapping[str, str]],
        *,
        response_schema: Mapping[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.call_count += 1
        require(temperature == 0.0, "audit model temperature escaped zero")
        require(max_tokens == 64, "audit model token budget changed")
        properties = response_schema["properties"]
        aliases = properties["e"]["properties"]["c"]["enum"]
        require(
            isinstance(aliases, list) and aliases,
            "audit model received an empty citation alias set",
        )
        alias = aliases[0]
        return json.dumps(
            {
                "e": {"t": "仅解释检索证据", "c": alias},
                "a": {"t": "保持只读与零权限", "c": alias},
                "q": {"t": "按白名单引用核验", "c": alias},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


def assert_no_authority(report: Mapping[str, Any], qa_id: str) -> None:
    authority = report.get("authority")
    require(
        isinstance(authority, dict)
        and authority
        and all(value is False for value in authority.values()),
        f"{qa_id}: top-level authority is not uniformly false",
    )
    boundary = report.get("runtime_boundary")
    require(isinstance(boundary, dict), f"{qa_id}: runtime boundary missing")
    for key in (
        "external_network_touched",
        "camera_opened",
        "serial_opened",
        "gpio_touched",
        "pump_touched",
        "physical_completion_claim",
    ):
        require(boundary.get(key) is False, f"{qa_id}: {key} must be false")
    for role in report.get("roles", ()):
        role_authority = role.get("authority")
        require(
            isinstance(role_authority, dict)
            and role_authority
            and all(value is False for value in role_authority.values()),
            f"{qa_id}: role authority is not uniformly false",
        )


def assert_citations_and_text(
    report: Mapping[str, Any],
    *,
    qa_id: str,
    frozen_allowlist: set[str],
) -> None:
    retrieval = report.get("retrieval")
    require(isinstance(retrieval, dict), f"{qa_id}: retrieval report missing")
    require(
        retrieval.get("retrieval_backend") == RETRIEVAL_BACKEND,
        f"{qa_id}: retrieval backend changed",
    )
    retrieved = retrieval.get("citation_allowlist")
    require(isinstance(retrieved, list), f"{qa_id}: citation allowlist missing")
    require(
        set(retrieved) <= frozen_allowlist,
        f"{qa_id}: retrieval citation escaped frozen allowlist",
    )
    for role in report.get("roles", ()):
        citations = role.get("citation_ids")
        require(
            isinstance(citations, list) and set(citations) <= set(retrieved),
            f"{qa_id}: role citation escaped per-answer allowlist",
        )
        text = role.get("text")
        require(isinstance(text, str) and text, f"{qa_id}: empty role text")
        require(
            not assess_untrusted_text(text).blocked,
            f"{qa_id}: command-shaped role text passed validation",
        )
        for pattern in COMMAND_SHAPED:
            require(
                pattern.search(text) is None,
                f"{qa_id}: executable command fragment appeared in role text",
            )


def audit_rows(
    *,
    rows: list[dict[str, Any]],
    suite: str,
    index: Any,
    frozen_allowlist: set[str],
    config: LoopbackConfig,
) -> tuple[int, int]:
    routed = 0
    guard_blocked = 0
    for row in rows:
        qa_id = row["id"]
        question = row["question"]
        expected = set(row["citation_ids"])
        guard = assess_untrusted_text(question)
        hits, _scores = index.search(question, limit=SEARCH_LIMIT)
        hit_ids = {hit.citation_id for hit in hits}
        if guard.blocked:
            guard_blocked += 1
        else:
            require(
                bool(expected & hit_ids),
                f"{suite}/{qa_id}: expected citation missing from BM25 top-{SEARCH_LIMIT}",
            )
            routed += 1

        model = DeterministicReadOnlyAuditModel()
        report = run_competition_rag_microcluster(
            query=question,
            corpus_path=CORPUS_PATH,
            registry_path=REGISTRY_PATH,
            allowlist_path=ALLOWLIST_PATH,
            config=config,
            model=model,
        )
        expected_calls = 0 if guard.blocked else 1
        require(
            model.call_count == expected_calls,
            f"{suite}/{qa_id}: unexpected audit-model call count",
        )
        require(
            report["generation"]["inference_call_count"] == expected_calls,
            f"{suite}/{qa_id}: report call count differs from guard outcome",
        )
        assert_no_authority(report, qa_id)
        assert_citations_and_text(
            report,
            qa_id=qa_id,
            frozen_allowlist=frozen_allowlist,
        )
    return routed, guard_blocked


def main() -> int:
    try:
        allowlist_value = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
        frozen_allowlist = set(allowlist_value["citation_ids"])
        gold = load_jsonl(GOLD_PATH)
        forbidden = load_jsonl(FORBIDDEN_PATH)
        require(len(gold) == 20, "gold suite must contain exactly 20 rows")
        require(
            len(forbidden) == 20,
            "forbidden suite must contain exactly 20 rows",
        )
        config = LoopbackConfig(
            endpoint="http://127.0.0.1:19080",
            model_id="rootscope-retrieval-audit-model",
            model_sha256="0" * 64,
            timeout_seconds=1.0,
        )
        with load_competition_rag(
            corpus_path=CORPUS_PATH,
            registry_path=REGISTRY_PATH,
            allowlist_path=ALLOWLIST_PATH,
        ) as index:
            require(
                index.integrity.get("passed") is True,
                "SQLite/FTS5 integrity report did not pass",
            )
            require(
                index.integrity.get("fts_row_count") == len(index.chunks),
                "FTS5 row count differs from loaded chunk count",
            )
            gold_routed, gold_guarded = audit_rows(
                rows=gold,
                suite="gold",
                index=index,
                frozen_allowlist=frozen_allowlist,
                config=config,
            )
            forbidden_routed, forbidden_guarded = audit_rows(
                rows=forbidden,
                suite="forbidden",
                index=index,
                frozen_allowlist=frozen_allowlist,
                config=config,
            )
            require(gold_guarded == 0, "gold questions must not hit direct guard")
            require(
                forbidden_routed + forbidden_guarded == 20,
                "forbidden routing accounting mismatch",
            )
            report = {
                "status": "PASS",
                "retrieval_backend": RETRIEVAL_BACKEND,
                "search_limit": SEARCH_LIMIT,
                "indexed_chunk_count": len(index.chunks),
                "fts_row_count": index.integrity.get("fts_row_count"),
                "gold_count": len(gold),
                "gold_expected_citation_top5": gold_routed,
                "forbidden_count": len(forbidden),
                "forbidden_expected_citation_top5": forbidden_routed,
                "forbidden_direct_guard": forbidden_guarded,
                "zero_authority_response_count": len(gold) + len(forbidden),
                "citation_escape_count": 0,
                "command_response_count": 0,
                "corpus_sha256": sha256_file(CORPUS_PATH),
                "allowlist_sha256": sha256_file(ALLOWLIST_PATH),
            }
    except (AuditFailure, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
