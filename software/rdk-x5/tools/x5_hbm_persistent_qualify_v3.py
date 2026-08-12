#!/usr/bin/env python3
"""Qualify one persistent ``hbm_runtime`` input policy against static43.

The script is board-side read-only inference.  It writes a JSON report but
does not select a production backend, change a config, open a camera, or touch
the future STM32/pump path.
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

from app.competition_runtime.plant_cpu_bpu_replay import rgb_to_bpu_tensor
from app.runtime_v3.hbm_runtime_adapter import PersistentHbmR7Adapter


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument(
        "--input-policy",
        choices=("RAW_UINT8", "RGB128_CENTERED_INT8"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    oracle = json.loads(args.oracle_manifest.read_text(encoding="utf-8"))
    if oracle.get("schema") != "rootscope.hbm-persistent-oracle-manifest.v1":
        raise SystemExit("oracle manifest schema mismatch")
    if oracle.get("count") != 43 or len(oracle.get("rows", ())) != 43:
        raise SystemExit("oracle manifest must contain exactly 43 rows")
    if oracle.get("model_sha256") != args.model_sha256:
        raise SystemExit("oracle/model SHA-256 mismatch")
    adapter = PersistentHbmR7Adapter(
        args.model,
        args.model_sha256,
        input_policy=args.input_policy,
    )

    rows: list[dict[str, Any]] = []
    agreements = 0
    cosines: list[float] = []
    max_abs_values: list[float] = []
    for expected in oracle["rows"]:
        image_path = args.input_root / expected["relative_path"]
        if sha256_file(image_path) != expected["file_sha256"]:
            raise SystemExit(f"image hash mismatch: {expected['relative_path']}")
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"unable to decode: {expected['relative_path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = rgb_to_bpu_tensor(rgb)
        tensor_sha = hashlib.sha256(tensor.tobytes()).hexdigest()
        if tensor_sha != expected["tensor_sha256"]:
            raise SystemExit(f"preprocessing drift: {expected['relative_path']}")
        actual = adapter.infer_uint8(tensor)
        actual_logits = np.asarray(actual["logits"], dtype=np.float64)
        oracle_logits = np.asarray(expected["oracle_logits"], dtype=np.float64)
        similarity = cosine(actual_logits, oracle_logits)
        max_abs = float(np.max(np.abs(actual_logits - oracle_logits)))
        agree = actual["top1_index"] == expected["oracle_top1_index"]
        agreements += int(agree)
        cosines.append(similarity)
        max_abs_values.append(max_abs)
        rows.append(
            {
                "relative_path": expected["relative_path"],
                "tensor_sha256": tensor_sha,
                "oracle_top1_index": expected["oracle_top1_index"],
                "hbm_top1_index": actual["top1_index"],
                "top1_agreement": agree,
                "cosine": similarity,
                "max_abs": max_abs,
                "latency_ms": actual["latency_ms"],
                "runtime_tensor_dtype": actual["runtime_tensor_dtype"],
                "runtime_tensor_sha256": actual["runtime_tensor_sha256"],
            }
        )
    gate = oracle["qualification_gate"]
    passed = (
        agreements >= int(gate["top1_agreement_min"])
        and statistics.fmean(cosines) >= float(gate["mean_cosine_min"])
    )
    report = {
        "schema": "rootscope.hbm-persistent-qualification.v1",
        "status": "PASS" if passed else "FAIL_CLOSED",
        "input_policy": args.input_policy,
        "backend_actual": "hbm_runtime.HB_HBMRuntime",
        "persistent_model": True,
        "model_sha256": adapter.model_sha256,
        "oracle_manifest_sha256": sha256_file(args.oracle_manifest),
        "count": len(rows),
        "top1_agreement": agreements,
        "mean_cosine": statistics.fmean(cosines),
        "minimum_cosine": min(cosines),
        "maximum_abs_logit_delta": max(max_abs_values),
        "latency_ms": {
            "median": statistics.median(item["latency_ms"] for item in rows),
            "mean": statistics.fmean(item["latency_ms"] for item in rows),
            "max": max(item["latency_ms"] for item in rows),
        },
        "eligible_for_shadow_runtime": passed,
        "selected_for_runtime": False,
        "selection_effect": "REPORT_ONLY_NO_RUNTIME_CONFIG_MUTATION",
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
    print(json.dumps({key: report[key] for key in (
        "status", "input_policy", "count", "top1_agreement",
        "mean_cosine", "eligible_for_shadow_runtime"
    )}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
