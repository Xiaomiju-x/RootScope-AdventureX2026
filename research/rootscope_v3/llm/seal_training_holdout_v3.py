#!/usr/bin/env python3
"""Create an external immutable-content seal for final training and holdout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--prior-details", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("seal output already exists")
    training_receipt_path = args.adapter_root / "training_receipt.json"
    training = json.loads(training_receipt_path.read_text(encoding="utf-8"))
    if (
        training.get("schema") != "rootscope.v3.qlora-training-receipt.v1"
        or training.get("status") != "PASS_REAL_RTX4050_QLORA_ADAPTER"
    ):
        raise SystemExit("training receipt is not passing")
    artifacts = []
    for relative, declared in sorted(training["artifacts"].items()):
        path = args.adapter_root / relative
        actual = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if (
            actual["bytes"] != declared["bytes"]
            or actual["sha256"] != declared["sha256"]
        ):
            raise SystemExit(f"training artifact mismatch: {relative}")
        artifacts.append(actual)

    manifest = json.loads(args.holdout_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_UNSEEN_BEFORE_FINAL_REFINEMENT":
        raise SystemExit("holdout manifest is not frozen")
    if sha256_file(args.holdout) != manifest["holdout"]["sha256"]:
        raise SystemExit("holdout hash mismatch")
    prior_ids: set[str] = set()
    prior_root_rows = []
    supplied_prior_paths = {path.as_posix() for path in args.prior_details}
    if supplied_prior_paths != set(manifest["source"]["prior_details"]):
        raise SystemExit("prior-details set is not exactly the frozen manifest set")
    for path in args.prior_details:
        expected = manifest["source"]["prior_details"].get(path.as_posix())
        actual = sha256_file(path)
        if expected != actual:
            raise SystemExit(f"prior details mismatch: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        prior_ids.update(row["record_id"] for row in value["results"])
        prior_root_rows.append({"path": path.as_posix(), "sha256": actual})
    holdout_rows = load_jsonl(args.holdout)
    if {row["record_id"] for row in holdout_rows} & prior_ids:
        raise SystemExit("holdout overlaps prior evaluations")
    started = datetime.fromisoformat(
        training["started_at_utc"].replace("Z", "+00:00")
    ).timestamp()
    if (
        args.holdout.stat().st_mtime >= started
        or args.holdout_manifest.stat().st_mtime >= started
    ):
        raise SystemExit("holdout was not frozen before final training started")

    seal = {
        "schema": "rootscope.v3.training-holdout-seal.v1",
        "status": "PASS_EXTERNAL_CONTENT_SEAL",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "training": {
            "receipt_sha256": sha256_file(training_receipt_path),
            "adapter_sha256": sha256_file(
                args.adapter_root / "adapter" / "adapter_model.safetensors"
            ),
            "artifact_count": len(artifacts),
            "artifact_root_sha256": hashlib.sha256(canonical(artifacts)).hexdigest(),
        },
        "holdout": {
            "sha256": sha256_file(args.holdout),
            "manifest_sha256": sha256_file(args.holdout_manifest),
            "cases": len(holdout_rows),
            "overlap_with_prior_count": 0,
            "frozen_before_training_started": True,
        },
        "prior_details_root_sha256": hashlib.sha256(
            canonical(sorted(prior_root_rows, key=lambda item: item["path"]))
        ).hexdigest(),
        "claim_boundary": (
            "This seal binds content and chronology for structured-contract "
            "qualification. It is not an unseen-domain knowledge claim."
        ),
        "authority": {
            "execution_authority": False,
            "serial_write": False,
            "gpio_write": False,
            "pump_command": False,
            "physical_completion": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(canonical(seal))
    print(json.dumps({"status": seal["status"], **seal["training"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
