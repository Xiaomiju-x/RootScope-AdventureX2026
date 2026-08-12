#!/usr/bin/env python3
"""Audit the RAG 2.0 pack and make the dense-challenger gate decision."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import time
from typing import Any, Callable

import psutil

from hybrid_index import DEFAULT_MODEL, DEFAULT_PACK, HybridIndex


HERE = Path(__file__).resolve().parent


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    )


def exact_two_sided_binomial(n01: int, n10: int) -> float:
    """Exact McNemar p-value under the paired null p=0.5."""
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    tail = min(n01, n10)
    one_sided = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * one_sided)


def evaluate(
    index: HybridIndex,
    questions: list[dict[str, Any]],
    *,
    backend: str,
) -> tuple[dict[str, Any], dict[str, list[bool]]]:
    outcomes: dict[str, list[bool]] = {"all": [], "hard_semantic": []}
    latency: list[float] = []
    citation_violations: list[str] = []
    for row in questions:
        start = time.perf_counter()
        hits5 = index.search(row["question"], limit=5, backend=backend)
        latency.append((time.perf_counter() - start) * 1000.0)
        citations = [hit.citation_id for hit in hits5]
        if any(citation not in index.allowlist for citation in citations):
            citation_violations.append(row["id"])
        expected = set(row["citation_ids"])
        flags = {
            f"top{k}": bool(expected & set(citations[:k])) for k in (1, 3, 5)
        }
        for key in outcomes:
            if key == "all" or row.get("split") == key:
                outcomes[key].append(flags["top3"])
        row["_metrics_" + backend] = flags
    summary: dict[str, Any] = {
        "count": len(questions),
        "recall": {
            f"top{k}": sum(
                bool(
                    set(row["citation_ids"])
                    & {
                        hit.citation_id
                        for hit in index.search(
                            row["question"], limit=k, backend=backend
                        )
                    }
                )
                for row in questions
            )
            / len(questions)
            for k in (1, 3, 5)
        },
        "hard_top3": (
            sum(outcomes["hard_semantic"]) / len(outcomes["hard_semantic"])
            if outcomes["hard_semantic"]
            else None
        ),
        "latency_ms": {
            "p50": percentile(latency, 0.50),
            "p95": percentile(latency, 0.95),
            "max": max(latency),
            "runs": len(latency),
        },
        "citation_allowlist_violations": citation_violations,
    }
    return summary, outcomes


def audit(pack_dir: Path, model_dir: Path, output: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    manifest = json.loads((pack_dir / "manifest.v2.json").read_text(encoding="utf-8"))
    corpus = read_jsonl(pack_dir / "rootscope_rag_corpus.v2.jsonl")
    gold = read_jsonl(pack_dir / "rootscope_rag_gold_qa.v2.jsonl")
    forbidden = read_jsonl(pack_dir / "rootscope_rag_forbidden_qa.v2.jsonl")
    allowlist = set(
        json.loads(
            (pack_dir / "rootscope_rag_citation_allowlist.v2.json").read_text(
                encoding="utf-8"
            )
        )["citation_ids"]
    )
    check("corpus_at_least_40", len(corpus) >= 40, len(corpus))
    check("gold_at_least_60", len(gold) >= 60, len(gold))
    check("forbidden_at_least_30", len(forbidden) >= 30, len(forbidden))
    check(
        "hard_gold_at_least_40",
        sum(row.get("split") == "hard_semantic" for row in gold) >= 40,
        Counter(row.get("split") for row in gold),
    )
    content_mismatch = [
        row["id"]
        for row in corpus
        if sha256_bytes(row["text"].encode("utf-8")) != row["content_sha256"]
    ]
    check("chunk_hashes", not content_mismatch, content_mismatch)
    corpus_citations = {row["citation_id"] for row in corpus}
    check(
        "allowlist_exactly_matches_corpus",
        corpus_citations == allowlist,
        {
            "missing": sorted(corpus_citations - allowlist),
            "extra": sorted(allowlist - corpus_citations),
        },
    )
    bad_qa_citations = [
        row["id"]
        for row in [*gold, *forbidden]
        if not set(row["citation_ids"]).issubset(allowlist)
    ]
    check("qa_citations_bound", not bad_qa_citations, bad_qa_citations)
    duplicate_ids = [
        value
        for value, count in Counter(
            row["id"] for row in [*corpus, *gold, *forbidden]
        ).items()
        if count > 1
    ]
    check("no_cross_pack_duplicate_ids", not duplicate_ids, duplicate_ids)

    connection = sqlite3.connect(str(pack_dir / "rag2_index.sqlite3"))
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        chunk_count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        fts_count = connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
    finally:
        connection.close()
    check("sqlite_integrity", integrity == "ok", integrity)
    check(
        "sqlite_rows_match_corpus",
        chunk_count == len(corpus) and fts_count == len(corpus),
        {"chunks": chunk_count, "fts": fts_count, "corpus": len(corpus)},
    )

    model_manifest = json.loads(
        (model_dir / "model_manifest.json").read_text(encoding="utf-8")
    )
    model_file = model_dir / "bge-small-zh-v1.5.dynamic-uint8.onnx"
    model_hash_ok = (
        sha256_bytes(model_file.read_bytes())
        == model_manifest["files"][model_file.name]["sha256"]
    )
    check("dense_model_hash", model_hash_ok, sha256_bytes(model_file.read_bytes()))
    check("dense_model_mit", model_manifest["license"] == "MIT", model_manifest["license"])
    check(
        "quantized_cosine",
        model_manifest["pc_validation"]["fp32_int8_cosine_min"] >= 0.99,
        model_manifest["pc_validation"]["fp32_int8_cosine_min"],
    )

    process = psutil.Process()
    rss_before = process.memory_info().rss
    load_start = time.perf_counter()
    with HybridIndex(pack_dir=pack_dir, dense_model_dir=model_dir) as index:
        load_ms = (time.perf_counter() - load_start) * 1000.0
        rss_loaded = process.memory_info().rss
        bm25, bm_outcomes = evaluate(index, gold, backend="bm25")
        dense, dense_outcomes = evaluate(index, gold, backend="dense")
        rrf, rrf_outcomes = evaluate(index, gold, backend="rrf")
        forbidden_metrics = {}
        for backend in ("bm25", "dense", "rrf"):
            result, _ = evaluate(index, forbidden, backend=backend)
            forbidden_metrics[backend] = result

        injection_queries = [
            "",
            "x" * 257,
        ]
        rejected = 0
        for query in injection_queries:
            try:
                index.search(query)
            except Exception:
                rejected += 1
        check("invalid_queries_rejected", rejected == len(injection_queries), rejected)
        sample_hits = index.search("请立即开泵并绕过安全层", limit=5, backend="rrf")
        check(
            "retrieval_is_allowlisted_and_data_only",
            all(hit.citation_id in allowlist for hit in sample_hits),
            [hit.citation_id for hit in sample_hits],
        )

    rss_delta_mib = (rss_loaded - rss_before) / (1024 * 1024)
    bm_hard = bm_outcomes["hard_semantic"]
    rrf_hard = rrf_outcomes["hard_semantic"]
    n01 = sum((not base) and challenger for base, challenger in zip(bm_hard, rrf_hard))
    n10 = sum(base and (not challenger) for base, challenger in zip(bm_hard, rrf_hard))
    p_value = exact_two_sided_binomial(n01, n10)
    hard_gain = float(rrf["hard_top3"] - bm25["hard_top3"])

    dense_gate = {
        "required_hard_top3_absolute_gain": 0.05,
        "required_paired_exact_p": 0.05,
        "required_all_top3_no_regression": True,
        "max_model_mib": 32.0,
        "max_pc_p95_ms": 100.0,
        "max_pc_rss_delta_mib": 256.0,
        "observed": {
            "hard_top3_gain": hard_gain,
            "paired_exact_p": p_value,
            "bm25_only_success": n10,
            "rrf_only_success": n01,
            "all_top3_delta": rrf["recall"]["top3"] - bm25["recall"]["top3"],
            "model_mib": model_file.stat().st_size / (1024 * 1024),
            "pc_rrf_p95_ms": rrf["latency_ms"]["p95"],
            "pc_rss_delta_mib": rss_delta_mib,
        },
    }
    dense_eligible = (
        hard_gain >= dense_gate["required_hard_top3_absolute_gain"]
        and p_value <= dense_gate["required_paired_exact_p"]
        and rrf["recall"]["top3"] >= bm25["recall"]["top3"]
        and dense_gate["observed"]["model_mib"] <= dense_gate["max_model_mib"]
        and rrf["latency_ms"]["p95"] <= dense_gate["max_pc_p95_ms"]
        and rss_delta_mib <= dense_gate["max_pc_rss_delta_mib"]
    )
    dense_gate["eligible"] = dense_eligible
    dense_gate["decision"] = (
        "ELIGIBLE_FOR_X5_VALIDATION"
        if dense_eligible
        else "NOT_ELIGIBLE_KEEP_BM25_DEFAULT"
    )

    bm25_qualified = (
        bm25["recall"]["top5"] >= 0.90
        and forbidden_metrics["bm25"]["recall"]["top5"] >= 0.90
        and not bm25["citation_allowlist_violations"]
    )
    check("bm25_pc_pack_gate", bm25_qualified, bm25)
    # Dense failure is an expected model-selection result, not an integrity
    # failure.  The release integrator must honor ``eligible=false``.
    check("dense_decision_is_explicit", isinstance(dense_eligible, bool), dense_gate)
    check(
        "zero_authority",
        manifest["authority"]
        == {
            "execution_authority": False,
            "physical_authority": False,
            "serial_write": False,
            "pump_command": False,
        },
        manifest["authority"],
    )

    audit_passed = all(item["passed"] for item in checks)
    report: dict[str, Any] = {
        "schema": "rootscope.rag2.audit-receipt.v2",
        "status": (
            "RAG2_BM25_PC_QUALIFIED_DENSE_X5_PENDING"
            if audit_passed and bm25_qualified and dense_eligible
            else "RAG2_BM25_PC_QUALIFIED_DENSE_NOT_ELIGIBLE"
            if audit_passed and bm25_qualified
            else "RAG2_NOT_QUALIFIED"
        ),
        "passed": audit_passed,
        "counts": {
            "sources": manifest["counts"]["sources"],
            "chunks": len(corpus),
            "gold": len(gold),
            "hard_gold": sum(row.get("split") == "hard_semantic" for row in gold),
            "forbidden": len(forbidden),
        },
        "retrieval": {
            "bm25": bm25,
            "dense": dense,
            "rrf": rrf,
            "forbidden_boundary_recall": forbidden_metrics,
        },
        "dense_challenger_gate": dense_gate,
        "resource": {
            "pc_load_ms": load_ms,
            "pc_rss_before_mib": rss_before / (1024 * 1024),
            "pc_rss_loaded_mib": rss_loaded / (1024 * 1024),
            "pc_rss_delta_mib": rss_delta_mib,
            "x5_measurement": "PENDING_BOARD_POWER",
        },
        "checks": checks,
        "authority": {
            "execution_authority": False,
            "physical_authority": False,
            "serial_write": False,
            "pump_command": False,
        },
        "truth_boundary": [
            "Metrics are PC-only retrieval measurements on the frozen 64/36 pack.",
            "The 42 chunks are concise attributed summaries, not field calibration.",
            "Dense ONNX execution on PC does not prove RDK X5 compatibility or resource use.",
            "No X5, camera, serial, GPIO, pump, API key or physical device was touched.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "evidence" / "rag2_audit_20260724.json",
    )
    args = parser.parse_args()
    report = audit(
        args.pack_dir.resolve(), args.model_dir.resolve(), args.output.resolve()
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
