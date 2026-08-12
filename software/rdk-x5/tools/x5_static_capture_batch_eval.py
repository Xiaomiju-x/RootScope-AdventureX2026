#!/usr/bin/env python3
"""Replay operator-labelled printed-card captures through the frozen X5 CPU stack.

This is a zero-authority, static-file evaluator.  It never opens a camera,
serial port, GPIO, pump, service, or network interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.omega_vision.ood import Calibration, decide, evaluate_quality
from app.vision.card_geometric_matcher import MatcherConfig
from app.vision.dual_path_demo import (
    DemoThresholds,
    build_seed17_runner_from_capsule,
    evaluate_dual_path_demo,
)


FROZEN_HASHES = {
    "capsule": "1e839948e466895e0b416aababb39402380463f9ce125c8aeb59f4087191cb97",
    "model": "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad",
    "registry": "f5328b66c6b385dd081c1a27577d92b5c63fc18dc00d815f7d3d2561bb68e29f",
    "calibration": "e82b196ab627a935fd571bba2b37aa637f040af5ba81fbab5183c1f15aa2e564",
    "thresholds": "877205689ad903207e0bcb5ffabdcbc5f1472c00b8f82e72faeb7cdd7d140fcd",
    "matcher": "9952864e50371675e7ea181cc57f2edd9eadd9189bfa0e9eda1e5cdd8f8ca61a",
}
ZERO_AUTHORITY = {
    "execution_authority": False,
    "physical_authority": False,
    "serial_write": False,
    "gpio_access": False,
    "pump_command": False,
    "state_machine_write": False,
    "irrigation_execution": False,
    "physical_completion": False,
}
EXPECTED_LABELS = ("grass_clump", "low_shrub", "young_tree", "non_target")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return {"path": str(resolved), "sha256": actual, "bytes": resolved.stat().st_size}


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_calibration(path: Path) -> Calibration:
    payload = load_object(path)
    raw = payload.get("calibration")
    if not isinstance(raw, dict):
        raise ValueError("calibration manifest omits calibration")
    normalized = dict(raw)
    for key in ("class_order", "conformal_nonconformity", "calibration_roles"):
        value = normalized.get(key)
        if not isinstance(value, list):
            raise ValueError(f"calibration.{key} must be an array")
        normalized[key] = tuple(value)
    calibration = Calibration(**normalized)
    if calibration.status != "MACHINE_CURATED_EXPERIMENTAL_NOT_QUALIFIED":
        raise ValueError("calibration claim boundary changed")
    return calibration


def load_capture_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"capture manifest line {line_number} is not an object")
        if item.get("class_id") not in EXPECTED_LABELS:
            raise ValueError(f"capture manifest line {line_number} has an invalid label")
        if item.get("truth_boundary") != "OPERATOR_LABELLED_LAPTOP_CAPTURE_NOT_HOLDOUT":
            raise ValueError(f"capture manifest line {line_number} changed truth boundary")
        records.append(item)
    if not records:
        raise ValueError("capture manifest is empty")
    return records


def final_reasons(dual: dict[str, Any], omega: Any) -> list[str]:
    reasons: list[str] = []
    semantic = dual["semantic"]
    geometry = dual["geometry"]
    consensus = dual["consensus"]
    if semantic.get("status") != "DEMO_HYPOTHESIS":
        reasons.append("SEMANTIC_HYPOTHESIS_STATUS_INVALID")
    if omega.decision != "CLASSIFY":
        reasons.append("OMEGA_OOD_ABSTAIN")
        reasons.extend(f"OMEGA_{reason}" for reason in omega.reasons)
    if omega.raw_top1_class == "unknown":
        reasons.append("UNKNOWN_CLASS_FAIL_CLOSED")
    if geometry.get("contract_valid_pass_count") != 1:
        reasons.append("GEOMETRY_NOT_EXACTLY_ONE_REGISTERED_PASS")
    if dual.get("experimental_consensus_passed") is not True:
        reasons.append("DUAL_PATH_CONSENSUS_REJECTED")
    if consensus.get("passed") is not True:
        reasons.append("DUAL_PATH_FINAL_BINDING_REJECTED")
    if omega.predicted_class != consensus.get("selected_template_class"):
        reasons.append("OMEGA_GEOMETRY_CLASS_DISAGREEMENT")
    return list(dict.fromkeys(reasons))


def confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    observed = ("grass_clump", "low_shrub", "young_tree", "unknown")
    output: dict[str, dict[str, int]] = {}
    for expected in EXPECTED_LABELS:
        output[expected] = {
            actual: sum(
                1
                for row in rows
                if row["expected_label"] == expected and row["raw_top1_class"] == actual
            )
            for actual in observed
        }
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--matcher", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    assets = {
        "capsule": require_hash(args.capsule, FROZEN_HASHES["capsule"], "capsule"),
        "model": require_hash(args.model, FROZEN_HASHES["model"], "model"),
        "registry": require_hash(args.registry, FROZEN_HASHES["registry"], "registry"),
        "calibration": require_hash(
            args.calibration, FROZEN_HASHES["calibration"], "calibration"
        ),
        "thresholds": require_hash(
            args.thresholds, FROZEN_HASHES["thresholds"], "thresholds"
        ),
        "matcher": require_hash(args.matcher, FROZEN_HASHES["matcher"], "matcher"),
    }
    input_root = args.input_root.expanduser().resolve(strict=True)
    manifest = args.capture_manifest.expanduser().resolve(strict=True)
    rows = load_capture_manifest(manifest)
    calibration = load_calibration(Path(assets["calibration"]["path"]))
    thresholds = DemoThresholds.from_mapping(
        load_object(Path(assets["thresholds"]["path"]))
    )
    matcher_config = MatcherConfig.from_mapping(
        load_object(Path(assets["matcher"]["path"]))
    )
    runner = build_seed17_runner_from_capsule(
        Path(assets["capsule"]["path"]),
        model_path=Path(assets["model"]["path"]),
    )
    if list(runner.providers) != ["CPUExecutionProvider"]:
        raise ValueError(f"runner is not CPU-only: {runner.providers}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with args.output_jsonl.open("x", encoding="utf-8", newline="\n") as output:
        for index, source in enumerate(rows, start=1):
            relative_path = source.get("relative_path")
            if not isinstance(relative_path, str):
                raise ValueError(f"capture record {index} omits relative_path")
            image_path = (input_root / relative_path).resolve(strict=True)
            if input_root not in image_path.parents:
                raise ValueError(f"capture record {index} escapes input root")
            actual_sha = sha256_file(image_path)
            if actual_sha != source.get("sha256"):
                raise ValueError(f"capture record {index} image SHA-256 mismatch")
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

            item_started = time.perf_counter()
            dual = evaluate_dual_path_demo(
                query_path=image_path,
                runner=runner,
                registry_path=Path(assets["registry"]["path"]),
                thresholds=thresholds,
                matcher_config=matcher_config,
            )
            semantic = dual["semantic"]
            omega = decide(
                semantic["raw_logits"],
                evaluate_quality(rgb),
                calibration,
            )
            reasons = final_reasons(dual, omega)
            final_accepted = not reasons
            display_class = (
                dual["consensus"].get("selected_template_class")
                if final_accepted
                else None
            )
            expected_label = source["class_id"]
            expected_semantic = "unknown" if expected_label == "non_target" else expected_label
            expected_outcome_matched = (
                (not final_accepted and omega.raw_top1_class == "unknown")
                if expected_label == "non_target"
                else (final_accepted and display_class == expected_label)
            )
            result = {
                "schema": "rootscope.x5-static-capture-eval.v1",
                "index": index,
                "relative_path": relative_path,
                "image_sha256": actual_sha,
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "expected_label": expected_label,
                "expected_semantic_label": expected_semantic,
                "raw_top1_class": semantic["raw_top1_class"],
                "raw_top1_probability": semantic["raw_top1_probability"],
                "raw_top1_margin": semantic["raw_top1_margin"],
                "raw_top1_label_matched": semantic["raw_top1_class"]
                == expected_semantic,
                "omega": omega.to_dict(),
                "geometry_pass_count": dual["geometry"]["contract_valid_pass_count"],
                "dual_path_consensus_passed": dual["experimental_consensus_passed"],
                "final_accepted": final_accepted,
                "display_class": display_class,
                "final_reject_reasons": reasons,
                "expected_outcome_matched": expected_outcome_matched,
                "elapsed_ms": round((time.perf_counter() - item_started) * 1000.0, 3),
                "compute_boundary": {
                    "provider": "CPUExecutionProvider",
                    "seed17_cpu_executed": True,
                    "yolo_used": False,
                    "plant_bpu_used": False,
                    "plant_bpu_selected_bin": None,
                },
                "truth_boundary": source["truth_boundary"],
                "authority": dict(ZERO_AUTHORITY),
            }
            output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            results.append(result)

    label_counts = Counter(item["expected_label"] for item in results)
    raw_matches = sum(item["raw_top1_label_matched"] for item in results)
    expected_matches = sum(item["expected_outcome_matched"] for item in results)
    positives = [item for item in results if item["expected_label"] != "non_target"]
    negatives = [item for item in results if item["expected_label"] == "non_target"]
    summary = {
        "schema": "rootscope.x5-static-capture-eval-summary.v1",
        "generated_at_utc": utc_now(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "provider_actual": list(runner.providers),
        "model": {
            **assets["model"],
            "selection": "seed17",
            "model_qualified": False,
            "bpu_ready": False,
            "yolo_used": False,
        },
        "assets": assets,
        "input_manifest": {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            "record_count": len(results),
            "label_counts": dict(sorted(label_counts.items())),
            "truth_boundary": "OPERATOR_LABELLED_LAPTOP_CAPTURE_NOT_HOLDOUT",
        },
        "observations": {
            "raw_top1_label_matches": raw_matches,
            "raw_top1_label_total": len(results),
            "raw_top1_label_agreement": raw_matches / len(results),
            "positive_final_accept_matches": sum(
                item["expected_outcome_matched"] for item in positives
            ),
            "positive_final_accept_total": len(positives),
            "non_target_safe_reject_matches": sum(
                item["expected_outcome_matched"] for item in negatives
            ),
            "non_target_safe_reject_total": len(negatives),
            "full_expected_outcome_matches": expected_matches,
            "full_expected_outcome_total": len(results),
            "confusion_raw_top1": confusion(results),
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "runtime_boundary": {
            "static_files_only": True,
            "camera_opened": False,
            "serial_opened": False,
            "gpio_touched": False,
            "pump_touched": False,
            "persistent_service_started": False,
            "runtime_network_touched": False,
        },
        "claim_boundary": {
            "operator_labelled_batch_observation_only": True,
            "formal_holdout": False,
            "generalization_accuracy": False,
            "model_qualified": False,
            "plant_bpu_inference": False,
            "irrigation_or_physical_completion": False,
        },
        "authority": dict(ZERO_AUTHORITY),
    }
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
