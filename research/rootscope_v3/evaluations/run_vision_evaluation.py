#!/usr/bin/env python3
"""Build a truthful PC-only RootSight-Delta audit receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rootscope_v3.vision.group_split import assign_group_splits, audit_group_splits


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adventurex-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.adventurex_root.resolve()
    provisional_manifest = root / "datasets/rootscope_machine_curated_provisional_v3/manifest.jsonl"
    capture_root = root / "captures/laptop_card_session_20260723_205230"
    capture_manifest = capture_root / "captures.jsonl"
    provisional = read_jsonl(provisional_manifest)
    captures = read_jsonl(capture_manifest)
    session_id = capture_root.name
    split_rows = assign_group_splits([{**row, "session_id": session_id} for row in captures])
    split_audit = audit_group_splits(split_rows)
    test_path = Path(__file__).with_name("test_rootsight_delta.py")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", str(test_path)],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=180,
    )
    model_paths = {
        "cpu_seed17_onnx": root / "models/rootscope_seed17_resnet18.onnx",
        "bpu_r7_bin": root / "output/rootscope_bpu_seed17_quant_variant_r7_default_int16_all_nodes/model_output/rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes.bin",
    }
    model_inventory = {
        name: {
            "present": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
            "path": str(path.relative_to(root)) if path.is_file() else str(path),
        }
        for name, path in model_paths.items()
    }
    receipt = {
        "schema_version": "rootscope.rootsight-delta-pc-evaluation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_PC_ONLY_ZERO_AUTHORITY" if proc.returncode == 0 and split_audit["status"] == "PASS" else "FAIL",
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "tests": {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "test_file_sha256": sha256(test_path),
        },
        "data_audit": {
            "provisional_manifest_records": len(provisional),
            "provisional_class_counts": dict(sorted(Counter(row.get("class_id") for row in provisional).items())),
            "provisional_training_eligible_true": sum(row.get("training_eligible") is True for row in provisional),
            "provisional_human_reviewed_true": sum(row.get("human_reviewed") is True for row in provisional),
            "capture_records": len(captures),
            "capture_class_counts": dict(sorted(Counter(row.get("class_id") for row in captures).items())),
            "capture_truth_boundary": sorted({row.get("truth_boundary") for row in captures}),
            "capture_group_split_audit": split_audit,
        },
        "model_inventory": model_inventory,
        "selection": {
            "cpu_primary": "EXISTING_SEED17_FROZEN_NO_REPLACEMENT",
            "bpu": "EXISTING_R7_SHADOW_CANDIDATE",
            "learned_wetting_segmentation": "NOT_TRAINED_NO_PHYSICAL_PAIRED_WETTING_DATA",
            "rootsight_delta": "DETERMINISTIC_CANDIDATE_READY_FOR_X5_STATIC_AND_LIVE_QUALIFICATION",
        },
        "implemented": [
            "reproducible yellow-light/exposure/gamma/blur/perspective/moire/JPEG augmentation",
            "capture-session/source-group split with group/hash leakage audit",
            "optical quality plus logits OOD HOLD",
            "multi-frame temporal median and agreement HOLD",
            "phase-correlation before/after registration",
            "reference-patch color correction and Lab delta",
            "morphology/component filtering, target coverage, neighbor spill, center offset, wetting front",
            "mass/visual consistency gate",
        ],
        "truth_boundary": {
            "x5_powered": False,
            "x5_tested": False,
            "gpu_used": False,
            "physical_wetting_pairs_available": False,
            "accuracy_claim": False,
            "physical_authority": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if receipt["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
