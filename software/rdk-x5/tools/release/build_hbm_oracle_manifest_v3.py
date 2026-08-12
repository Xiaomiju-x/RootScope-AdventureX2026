#!/usr/bin/env python3
"""Join frozen static43 inputs with actual hrt-model-exec oracle evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--oracle-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = load_jsonl(args.input_manifest)
    results = load_jsonl(args.oracle_results)
    if len(inputs) != 43 or len(results) != 43:
        raise SystemExit("frozen qualification set must contain exactly 43 rows")
    by_path = {row["relative_path"]: row for row in inputs}
    output_rows: list[dict[str, Any]] = []
    for result in results:
        relative = result["relative_path"]
        source = by_path.get(relative)
        if source is None:
            raise SystemExit(f"oracle path missing from input manifest: {relative}")
        if source["sha256"] != result["file_sha256"]:
            raise SystemExit(f"file hash mismatch for {relative}")
        proposal = result["bpu_proposal"]
        if proposal["backend"] != "drobotics.hrt_model_exec@1.24.5/cold-load":
            raise SystemExit(f"row is not canonical hrt evidence: {relative}")
        output_rows.append(
            {
                "schema": "rootscope.hbm-persistent-oracle-row.v1",
                "relative_path": relative,
                "file_sha256": source["sha256"],
                "tensor_sha256": result["input_tensor_sha256"],
                "oracle_backend": proposal["backend"],
                "oracle_logits": proposal["logits"],
                "oracle_top1_index": proposal["top1_index"],
                "oracle_top1_class": proposal["top1_class"],
                "truth_boundary": source["truth_boundary"],
            }
        )
    envelope = {
        "schema": "rootscope.hbm-persistent-oracle-manifest.v1",
        "count": len(output_rows),
        "model_sha256": (
            "4dc4bc24741332bb6bc6da184dd1d6f954ae539ba740330490339d76eb200285"
        ),
        "source_input_manifest_sha256": sha256_file(args.input_manifest),
        "source_oracle_results_sha256": sha256_file(args.oracle_results),
        "qualification_gate": {
            "top1_agreement_min": 43,
            "mean_cosine_min": 0.995,
            "nonfinite_allowed": 0,
            "runtime_errors_allowed": 0,
        },
        "selection_effect": "REPORT_ONLY_NO_RUNTIME_CONFIG_MUTATION",
        "authority": {
            "execution_authority": False,
            "serial_write": False,
            "gpio_write": False,
            "pump_command": False,
            "physical_completion": False,
        },
        "rows": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "count": len(output_rows),
                "output": str(args.output),
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
