#!/usr/bin/env python3
"""Build a deterministic train-only curriculum for safety-contract refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adversarial-copies", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("curriculum output already exists; refusing mutable overwrite")
    if not 2 <= args.adversarial_copies <= 8:
        raise SystemExit("adversarial-copies must be between 2 and 8")

    train = load_jsonl(args.train)
    validation = load_jsonl(args.validation)
    test = load_jsonl(args.test)
    dataset_receipt = json.loads(args.dataset_receipt.read_text(encoding="utf-8"))
    for name, path in (
        ("train.jsonl", args.train),
        ("validation.jsonl", args.validation),
        ("test.jsonl", args.test),
    ):
        if sha256_file(path) != dataset_receipt["file_sha256"][name]:
            raise SystemExit(f"dataset receipt mismatch: {name}")
    train_ids = {row["record_id"] for row in train}
    held_out_ids = {row["record_id"] for row in [*validation, *test]}
    held_out_groups = {row["template_group"] for row in [*validation, *test]}
    if train_ids & held_out_ids:
        raise SystemExit("source split record-id overlap")
    if {row["template_group"] for row in train} & held_out_groups:
        raise SystemExit("source split template-group overlap")
    adversarial = [
        row for row in train if row["input"].get("adversarial_request") is not None
    ]
    if not adversarial:
        raise SystemExit("no train-only adversarial rows")
    if any(
        "ADVERSARIAL_REQUEST_REJECTED" not in row["output"]["reason_codes"]
        or not row["output"]["proposed_explanation"].startswith("拒绝")
        or row["output"]["authority"] is not False
        for row in adversarial
    ):
        raise SystemExit("train-only adversarial contract is incomplete")

    curriculum = [*train, *(adversarial * (args.adversarial_copies - 1))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in curriculum
    )
    args.output.open("xb").write(payload.encode("utf-8"))
    output_ids = {row["record_id"] for row in curriculum}
    if not output_ids <= train_ids or output_ids & held_out_ids:
        raise SystemExit("curriculum contains held-out records")

    manifest = {
        "schema": "rootscope.v3.train-only-adversarial-curriculum.v1",
        "status": "PASS_TRAIN_ONLY_NO_HELD_OUT_RECORDS",
        "source": {
            "dataset_receipt_sha256": sha256_file(args.dataset_receipt),
            "train_sha256": sha256_file(args.train),
            "validation_sha256": sha256_file(args.validation),
            "test_sha256": sha256_file(args.test),
            "train_unique_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "curriculum": {
            "relative_output": args.output.as_posix(),
            "sha256": sha256_file(args.output),
            "rows": len(curriculum),
            "unique_record_ids": len(output_ids),
            "adversarial_rows": sum(
                row["input"].get("adversarial_request") is not None
                for row in curriculum
            ),
            "adversarial_copies": args.adversarial_copies,
            "held_out_record_count": len(output_ids & held_out_ids),
            "held_out_template_group_count": len(
                {row["template_group"] for row in curriculum} & held_out_groups
            ),
        },
        "truth_boundary": (
            "Curriculum duplicates only original train-split adversarial records; "
            "no validation or test record is copied into training."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
    print(json.dumps(manifest["curriculum"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
