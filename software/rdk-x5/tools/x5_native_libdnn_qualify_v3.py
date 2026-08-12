#!/usr/bin/env python3
"""Qualify the persistent native libdnn valid-shape bridge on an actual X5.

The qualification is a static43, hash-bound, zero-authority BPU replay.  It
does not open a camera, serial device, GPIO, pump, network socket, service
manager, or action state.  It never selects or activates a runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from app.runtime_v3.native_libdnn_adapter import (
    EXPECTED_MODEL_NAME,
    PersistentNativeLibdnnR7Adapter,
    ZERO_AUTHORITY,
)


CMA_ISOLATED_POST_WORKER_THRESHOLD_KIB = 128 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 1.0


def memory_snapshot() -> dict[str, int | None]:
    wanted = ("MemTotal", "MemAvailable", "CmaTotal", "CmaFree")
    values: dict[str, int | None] = {key: None for key in wanted}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, separator, remainder = line.partition(":")
            if separator and key in values:
                values[key] = int(remainder.strip().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return {f"{key}_kib": value for key, value in values.items()}


def process_snapshot(pid: int) -> dict[str, Any]:
    root = Path("/proc") / str(pid)
    result: dict[str, Any] = {"pid": pid, "proc_present": root.exists()}
    try:
        status: dict[str, str] = {}
        for line in (root / "status").read_text(encoding="ascii").splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"Name", "State", "VmRSS", "VmSize", "Threads"}:
                status[key] = value.strip()
        result["status"] = status
    except OSError:
        result["status"] = {}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--compile-contract", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--oracle-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-model-name", default=EXPECTED_MODEL_NAME)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit("refusing to overwrite native libdnn qualification evidence")
    oracle = json.loads(args.oracle_manifest.read_text(encoding="utf-8"))
    if oracle.get("schema") != "rootscope.hbm-persistent-oracle-manifest.v1":
        raise SystemExit("oracle manifest schema mismatch")
    expected_rows = oracle.get("rows")
    if (
        oracle.get("count") != 43
        or not isinstance(expected_rows, list)
        or len(expected_rows) != 43
    ):
        raise SystemExit("oracle manifest must contain exactly 43 rows")
    if oracle.get("model_sha256") != args.model_sha256:
        raise SystemExit("oracle/model SHA-256 mismatch")

    before = memory_snapshot()
    rows: list[dict[str, Any]] = []
    cosines: list[float] = []
    maximum_abs_values: list[float] = []
    agreements = 0
    adapter = PersistentNativeLibdnnR7Adapter(
        args.model,
        args.model_sha256,
        worker_path=args.worker,
        compile_contract_path=args.compile_contract,
        expected_model_name=args.expected_model_name,
    )
    worker_pid = adapter.worker_pid
    loaded_snapshot: dict[str, Any] | None = None
    during: dict[str, int | None] | None = None
    worker_identity = adapter.worker_identity.to_dict()
    try:
        for index, expected in enumerate(expected_rows):
            image_path = args.input_root / expected["relative_path"]
            if sha256_file(image_path) != expected["file_sha256"]:
                raise SystemExit(
                    f"image hash mismatch: {expected['relative_path']}"
                )
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise SystemExit(f"unable to decode: {expected['relative_path']}")
            tensor = rgb_to_bpu_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            tensor_sha256 = hashlib.sha256(
                tensor.tobytes(order="C")
            ).hexdigest()
            if tensor_sha256 != expected["tensor_sha256"]:
                raise SystemExit(
                    f"preprocessing drift: {expected['relative_path']}"
                )

            actual = adapter.infer_uint8(tensor)
            if index == 0:
                loaded_snapshot = process_snapshot(worker_pid)
                during = memory_snapshot()
            actual_logits = np.asarray(actual["logits"], dtype=np.float64)
            oracle_logits = np.asarray(
                expected["oracle_logits"], dtype=np.float64
            )
            similarity = cosine(actual_logits, oracle_logits)
            maximum_abs = float(
                np.max(np.abs(actual_logits - oracle_logits))
            )
            actual_top1 = int(actual["top1_index"])
            agree = actual_top1 == int(expected["oracle_top1_index"])
            agreements += int(agree)
            cosines.append(similarity)
            maximum_abs_values.append(maximum_abs)
            rows.append(
                {
                    "relative_path": expected["relative_path"],
                    "tensor_sha256": tensor_sha256,
                    "oracle_top1_index": expected["oracle_top1_index"],
                    "native_top1_index": actual_top1,
                    "top1_agreement": agree,
                    "cosine": similarity,
                    "max_abs": maximum_abs,
                    "worker_inference_ms": actual["worker_inference_ms"],
                    "roundtrip_latency_ms": actual["roundtrip_latency_ms"],
                    "inference_count_since_load": actual[
                        "inference_count_since_load"
                    ],
                }
            )
    finally:
        worker_exit_code = adapter.close()

    after = memory_snapshot()
    worker_after_close = process_snapshot(worker_pid)
    clean_worker_exit = (
        worker_exit_code == 0 and not bool(worker_after_close["proc_present"])
    )
    gate = oracle.get("qualification_gate")
    if not isinstance(gate, dict):
        raise SystemExit("oracle qualification gate missing")
    mean_cosine = statistics.fmean(cosines)
    numerical_pass = (
        len(rows) == 43
        and agreements >= int(gate["top1_agreement_min"])
        and mean_cosine >= float(gate["mean_cosine_min"])
    )
    passed = numerical_pass and clean_worker_exit
    cma_free_after = after.get("CmaFree_kib")
    isolated_post_worker_threshold_observed = (
        isinstance(cma_free_after, int)
        and cma_free_after >= CMA_ISOLATED_POST_WORKER_THRESHOLD_KIB
    )
    report = {
        "schema": "rootscope.v3.x5-native-libdnn-qualification.v1",
        "status": (
            "PASS_X5_PERSISTENT_NATIVE_LIBDNN"
            if passed
            else "FAIL_CLOSED"
        ),
        "backend_actual": (
            "rootscope.native_libdnn_valid_shape_bridge/libdnn.so"
        ),
        "canonical_oracle_backend": "drobotics.hrt_model_exec",
        "persistent_model": True,
        "cold_load_per_inference": False,
        "single_worker_pid": worker_pid,
        "model_load_count": 1,
        "model_sha256": args.model_sha256,
        "model_name": args.expected_model_name,
        "worker": worker_identity,
        "oracle_manifest_sha256": sha256_file(args.oracle_manifest),
        "count": len(rows),
        "top1_agreement": agreements,
        "mean_cosine": mean_cosine,
        "minimum_cosine": min(cosines),
        "maximum_abs_logit_delta": max(maximum_abs_values),
        "latency_ms": {
            "worker_inference_median": statistics.median(
                item["worker_inference_ms"] for item in rows
            ),
            "worker_inference_mean": statistics.fmean(
                item["worker_inference_ms"] for item in rows
            ),
            "worker_inference_max": max(
                item["worker_inference_ms"] for item in rows
            ),
            "roundtrip_median": statistics.median(
                item["roundtrip_latency_ms"] for item in rows
            ),
            "roundtrip_mean": statistics.fmean(
                item["roundtrip_latency_ms"] for item in rows
            ),
            "roundtrip_max": max(
                item["roundtrip_latency_ms"] for item in rows
            ),
        },
        "worker_lifecycle": {
            "loaded": loaded_snapshot,
            "exit_code_after_stdin_eof": worker_exit_code,
            "after_close": worker_after_close,
            "clean_exit_no_residual_process": clean_worker_exit,
        },
        "memory_kib": {
            "before_worker": before,
            "model_loaded": during,
            "after_worker_exit": after,
        },
        "isolated_post_worker_cma_observation": {
            "threshold_kib": CMA_ISOLATED_POST_WORKER_THRESHOLD_KIB,
            "observed_after_worker_exit_kib": cma_free_after,
            "threshold_observed": isolated_post_worker_threshold_observed,
            "effect_on_this_numerical_qualification": (
                "RECORDED_SEPARATELY_NOT_HIDDEN"
            ),
            "full_stack_or_soak_claim": False,
        },
        "numerical_oracle_gate_passed": numerical_pass,
        "eligible_for_zero_authority_shadow_runtime": passed,
        "selected_for_runtime": False,
        "selection_effect": "REPORT_ONLY_NO_RUNTIME_CONFIG_MUTATION",
        "rows": rows,
        "authority": dict(ZERO_AUTHORITY),
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
                "worker_exit_clean": clean_worker_exit,
                "CmaFree_kib_after": cma_free_after,
                "isolated_post_worker_cma_threshold_observed": (
                    isolated_post_worker_threshold_observed
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
