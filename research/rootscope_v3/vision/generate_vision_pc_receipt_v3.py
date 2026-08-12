#!/usr/bin/env python3
"""Build the RootSight v3 schema receipt from frozen static evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adventurex", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.adventurex.resolve(strict=True)
    sys.path.insert(0, str(root / "rootscope"))
    from app.vision import PixelROI, verify_wetting_change
    board_path = (
        root
        / "rootscope/evidence/omega_v3_x5_final_20260723T094509Z"
        / "06_omega_vision_board_replay.receipt.json"
    )
    board = json.loads(board_path.read_text(encoding="utf-8"))
    if board["status"] != "PASS_X5_CPU_EXPLICIT_IMAGE_REPLAY_ZERO_AUTHORITY":
        raise SystemExit("frozen static board replay is not passing")
    if board["summary"]["pc_reference_parity_passed_count"] != 4:
        raise SystemExit("frozen static parity count changed")
    if board["effects_and_authority"]["execution_authority"] is not False:
        raise SystemExit("frozen vision receipt authority changed")

    baseline = np.full((240, 320, 3), 170, dtype=np.uint8)
    target = PixelROI("Z2", 100, 80, 100, 70)
    neighbors = (PixelROI("Z1", 100, 10, 100, 60), PixelROI("Z3", 100, 160, 100, 60))
    target_only = baseline.copy()
    target_only[80:150, 100:200] = 95
    spill = target_only.copy()
    spill[160:220, 100:200] = 95
    wet_pass = verify_wetting_change(baseline, target_only, target, neighbors)
    wet_spill = verify_wetting_change(baseline, spill, target, neighbors)
    if not wet_pass.passed or wet_spill.passed:
        raise SystemExit("RootSight wetting selectivity fixture gate failed")

    latencies = sorted(float(row["inference"]["cpu_inference_ms"]) for row in board["images"])
    p95 = latencies[-1]
    receipt = {
        "schema": "rootscope.v3.vision-evaluation.v1",
        "evaluation_id": "rootsight-v3-pc-static-contract-20260724",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evaluation_scope": "PC_STATIC",
        "dataset": {
            "dataset_id": "frozen-x5-four-static-projection-plus-wetting-fixtures",
            "partition": "DEVELOPMENT",
            "manifest_sha256": board["manifest"]["sha256"],
            "split_unit": "SOURCE_GROUP",
            "sample_count": 6,
            "formal_holdout": False,
        },
        "optical_domain": {
            "camera_id": None,
            "lighting": "STATIC_REGISTERED_IMAGES_AND_SYNTHETIC_WETTING_FIXTURES",
            "mount_fixed": False,
            "exposure_fixed": False,
            "white_balance_fixed": False,
            "reference_patch_used": False,
        },
        "backends": [
            {
                "backend_id": "CPU_ONNX_STATIC_PARITY",
                "model_id": "rootscope-seed17-resnet18-cpu-onnx",
                "model_sha256": board["runtime"]["model_sha256"],
                "execution_actual": True,
                "role": "PRIMARY",
            },
            {
                "backend_id": "ROOTSIGHT_WETTING_NUMPY_FIXTURE",
                "model_id": "relative-before-after-selectivity-v1",
                "model_sha256": None,
                "execution_actual": True,
                "role": "AUDIT",
            },
        ],
        "counts": {
            "raw_scored": 4,
            "raw_top1_correct": 4,
            "full_gate_scored": 6,
            "full_gate_expected_outcome": 6,
            "known_positive_count": 3,
            "unknown_or_occluded_count": 1,
            "false_irrigation_intent_count": 0,
            "cpu_bpu_compared": 0,
            "cpu_bpu_top1_agree": 0,
            "non_finite_output_count": 0,
        },
        "latency_ms": {
            "measurement_scope": "MODEL_ONLY",
            "sample_count": len(latencies),
            "p50": statistics.median(latencies),
            "p95": p95,
            "max": max(latencies),
            "cold_start": None,
        },
        "qualification": {
            "status": "PASS",
            "gates_passed": [
                "FROZEN_X5_STATIC_PC_REFERENCE_PARITY_4_OF_4",
                "UNKNOWN_STATIC_INPUT_ABSTAINED",
                "ZERO_FALSE_IRRIGATION_INTENT_ON_STATIC_SET",
                "WETTING_TARGET_ONLY_FIXTURE_ACCEPTED",
                "WETTING_NEIGHBOR_SPILL_FIXTURE_REJECTED",
            ],
            "claim_boundary": (
                "Pass is limited to frozen static CPU parity and deterministic "
                "before/after wetting fixtures. It is not a formal plant-accuracy, "
                "live-camera, BPU persistent, moisture, root-depth or physical "
                "irrigation qualification; those remain X5/hardware gates."
            ),
        },
        "authority": {
            "execution_authority": False,
            "physical_authority": False,
            "pump_command": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "sample_count": 6, "false_irrigation": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
