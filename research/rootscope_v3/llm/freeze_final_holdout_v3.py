#!/usr/bin/env python3
"""Freeze an unseen, stratified final holdout before the final adapter exists."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--prior-details", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("holdout output already exists")
    if args.cases < 16 or args.cases % 2:
        raise SystemExit("cases must be an even integer of at least 16")
    receipt = json.loads(args.dataset_receipt.read_text(encoding="utf-8"))
    if sha256_file(args.test) != receipt["file_sha256"]["test.jsonl"]:
        raise SystemExit("test split hash mismatch")
    rows = load_jsonl(args.test)
    prior_ids: set[str] = set()
    prior_hashes = {}
    for path in args.prior_details:
        value = json.loads(path.read_text(encoding="utf-8"))
        prior_ids.update(row["record_id"] for row in value["results"])
        prior_hashes[path.as_posix()] = sha256_file(path)
    candidates = [row for row in rows if row["record_id"] not in prior_ids]
    adversarial = [
        row for row in candidates if row["input"].get("adversarial_request") is not None
    ]
    regular = [
        row for row in candidates if row["input"].get("adversarial_request") is None
    ]
    half = args.cases // 2
    if len(adversarial) < half or len(regular) < half:
        raise SystemExit("insufficient unseen stratified holdout rows")
    rng = random.Random(2026072402)
    selected = rng.sample(adversarial, half) + rng.sample(regular, half)
    rng.shuffle(selected)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in selected
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.open("xb").write(payload.encode("utf-8"))
    selected_ids = [row["record_id"] for row in selected]
    if set(selected_ids) & prior_ids:
        raise SystemExit("final holdout overlaps prior evaluation")
    manifest = {
        "schema": "rootscope.v3.final-holdout-freeze.v1",
        "status": "FROZEN_UNSEEN_BEFORE_FINAL_REFINEMENT",
        "seed": 2026072402,
        "source": {
            "test_sha256": sha256_file(args.test),
            "dataset_receipt_sha256": sha256_file(args.dataset_receipt),
            "prior_details": prior_hashes,
            "prior_record_count": len(prior_ids),
        },
        "holdout": {
            "sha256": sha256_file(args.output),
            "record_ids_sha256": hashlib.sha256(
                json.dumps(selected_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "cases": len(selected),
            "adversarial": half,
            "regular": half,
            "overlap_with_prior_count": 0,
        },
        "claim_boundary": (
            "Frozen before the final refinement adapter exists. Knowledge-source "
            "families may overlap the training corpus; this qualifies structured "
            "contract behavior, not unseen-domain knowledge generalization."
        ),
    }
    with args.manifest.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
    print(json.dumps(manifest["holdout"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
