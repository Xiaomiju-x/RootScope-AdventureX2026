#!/usr/bin/env python3
"""X5 read-only CPU ONNX and BM25 acceptance without camera or actuators."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.release_root.resolve(strict=True)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "rootscope"))
    from app.edge.capsule import CapsuleConfig
    from app.edge.selftest import run_simulated_selftest
    from rootscope_v3.rag2.bm25_runtime import BM25Index

    capsule_path = (
        root / "rootscope/deploy/x5/capsule_config.seed17_cpu_experimental.json"
    )
    value = json.loads(capsule_path.read_text(encoding="utf-8"))
    value["model"]["path"] = str(root / "models/rootscope_seed17_cpu.onnx")
    config = CapsuleConfig.from_mapping(value)
    cpu = run_simulated_selftest(config)
    if cpu.get("status") != "PASS_CPU_ONNX_SIMULATED_INPUT_NOT_ACCURACY_EVIDENCE":
        raise SystemExit("CPU ONNX simulated replay failed")
    queries = (
        (
            "RootScope 是移动机器人吗",
            "rootscope-field-knowledge-v1#k01@rootscope-product-boundary",
        ),
        (
            "视觉模型能直接打开水泵吗",
            "rootscope-x5-plan-20260723#section-6@rootscope-future-serial-boundary",
        ),
        (
            "内存不足时为什么保持相机和心跳",
            "rootscope-v3-plan-20260724#section-7@resource-protection-order",
        ),
    )
    rag_rows = []
    with BM25Index(root / "rootscope_v3/rag2/pack") as index:
        integrity = index.connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise SystemExit("RAG SQLite integrity check failed")
        for query, expected_top1 in queries:
            hits = index.search(query, limit=5)
            if not hits:
                raise SystemExit(f"RAG returned no citations for {query}")
            if hits[0].citation_id != expected_top1:
                raise SystemExit(
                    f"RAG frozen top-1 citation drift for {query}: "
                    f"actual={hits[0].citation_id} expected={expected_top1}"
                )
            rag_rows.append(
                {
                    "query": query,
                    "expected_top1_citation": expected_top1,
                    "citations": [hit.citation_id for hit in hits],
                    "backend": "SQLITE_FTS5_BM25_V2",
                }
            )
    receipt = {
        "schema": "rootscope.v3.x5-cpu-bm25-acceptance.v1",
        "status": "PASS_X5_CPU_ONNX_AND_BM25_READ_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cpu": cpu,
        "rag": {
            "sqlite_integrity": "ok",
            "database_open_mode": "URI_MODE_RO_IMMUTABLE_1",
            "queries": rag_rows,
        },
        "camera_opened": False,
        "serial_opened": False,
        "gpio_touched": False,
        "pump_touched": False,
        "physical_completion": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "rag_queries": len(rag_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
