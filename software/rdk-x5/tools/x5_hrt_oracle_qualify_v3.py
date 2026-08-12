#!/usr/bin/env python3
"""Qualify the canonical RDK ``hrt_model_exec`` path against static43.

This is a board-side, zero-authority BPU replay.  It never opens a camera,
serial port, GPIO, service manager, or actuator.  The persistent Python HBM
adapter has a separate qualification gate; this script preserves the vendor
CLI as the numerical oracle and fail-closed fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.competition_runtime.bpu_shadow_worker import HrtModelExecR7BpuBackend
from app.competition_runtime.plant_cpu_bpu_replay import rgb_to_bpu_tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--oracle-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hrt-model-exec", default="/usr/sbin/hrt_model_exec", type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    args = parser.parse_args()

    oracle = json.loads(args.oracle_manifest.read_text(encoding="utf-8"))
    if oracle.get("schema") != "rootscope.hbm-persistent-oracle-manifest.v1":
        raise SystemExit("oracle manifest schema mismatch")
    rows_expected = oracle.get("rows")
    if oracle.get("count") != 43 or not isinstance(rows_expected, list):
        raise SystemExit("oracle manifest must contain exactly 43 rows")
    if len(rows_expected) != 43:
        raise SystemExit("oracle manifest row count changed")
    if oracle.get("model_sha256") != args.model_sha256:
        raise SystemExit("oracle/model SHA-256 mismatch")
    if args.output.exists():
        raise SystemExit("refusing to overwrite canonical HRT evidence")
    if args.work_root.exists():
        raise SystemExit("HRT work root must be new")
    args.work_root.mkdir(parents=True, mode=0o700)

    backend = HrtModelExecR7BpuBackend(
        args.model,
        args.model_sha256,
        executable_path=args.hrt_model_exec,
        work_root=args.work_root,
        inference_timeout_s=10.0,
    )
    rows: list[dict[str, Any]] = []
    cosines: list[float] = []
    max_abs_values: list[float] = []
    agreements = 0
    for expected in rows_expected:
        image_path = args.input_root / expected["relative_path"]
        if sha256_file(image_path) != expected["file_sha256"]:
            raise SystemExit(f"image hash mismatch: {expected['relative_path']}")
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"unable to decode: {expected['relative_path']}")
        tensor = rgb_to_bpu_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        tensor_sha = hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
        if tensor_sha != expected["tensor_sha256"]:
            raise SystemExit(f"preprocessing drift: {expected['relative_path']}")

        logits = np.asarray(backend.infer_tensor(tensor), dtype=np.float64)
        oracle_logits = np.asarray(expected["oracle_logits"], dtype=np.float64)
        similarity = cosine(logits, oracle_logits)
        max_abs = float(np.max(np.abs(logits - oracle_logits)))
        actual_top1 = int(np.argmax(logits))
        agree = actual_top1 == int(expected["oracle_top1_index"])
        agreements += int(agree)
        cosines.append(similarity)
        max_abs_values.append(max_abs)
        rows.append(
            {
                "relative_path": expected["relative_path"],
                "tensor_sha256": tensor_sha,
                "oracle_top1_index": expected["oracle_top1_index"],
                "hrt_top1_index": actual_top1,
                "top1_agreement": agree,
                "cosine": similarity,
                "max_abs": max_abs,
                "hrt_inference_evidence": dict(backend.last_inference_evidence),
            }
        )

    gate = oracle["qualification_gate"]
    passed = (
        agreements >= int(gate["top1_agreement_min"])
        and statistics.fmean(cosines) >= float(gate["mean_cosine_min"])
    )
    report = {
        "schema": "rootscope.v3.x5-hrt-oracle-qualification.v1",
        "status": "PASS_X5_CANONICAL_HRT_BPU_ORACLE" if passed else "FAIL_CLOSED",
        "backend_actual": backend.backend_actual,
        "canonical_numerical_oracle": True,
        "persistent_model": False,
        "cold_load_per_inference": True,
        "real_time_qualified": False,
        "model_sha256": backend.model_sha256,
        "hrt_model_exec_sha256": backend.executable_sha256,
        "oracle_manifest_sha256": sha256_file(args.oracle_manifest),
        "count": len(rows),
        "top1_agreement": agreements,
        "mean_cosine": statistics.fmean(cosines),
        "minimum_cosine": min(cosines),
        "maximum_abs_logit_delta": max(max_abs_values),
        "eligible_as_fail_closed_bpu_shadow_fallback": passed,
        "selected_for_primary_runtime": False,
        "rows": rows,
        "authority": {
            "execution_authority": False,
            "serial_write": False,
            "gpio_write": False,
            "pump_command": False,
            "state_machine_write": False,
            "physical_completion": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "count": report["count"],
                "top1_agreement": report["top1_agreement"],
                "mean_cosine": report["mean_cosine"],
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
