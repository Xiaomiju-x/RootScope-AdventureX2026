"""Build the immutable RootScope Omega seed17 OOD/abstention receipt.

No training occurs here.  The script reads the existing seed17 ONNX and the
78-image machine-curated provisional pack, calibrates only on experimental
train/validation suggestions, then evaluates each holdout exactly once.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image


ROOTSCOPE = Path(__file__).resolve().parents[2]
ADVENTUREX = ROOTSCOPE.parent
if str(ROOTSCOPE) not in sys.path:
    sys.path.insert(0, str(ROOTSCOPE))

from app.omega_vision.ood import Calibration, calibrate, decide, evaluate_quality  # noqa: E402


CLASS_ORDER = ("grass_clump", "low_shrub", "young_tree", "unknown")
TRAIN = "EXPERIMENTAL_TRAIN_SUGGESTION"
VAL = "EXPERIMENTAL_VAL_SUGGESTION"
PRINT = "PRINT_DEMO_HOLDOUT_NOT_TRAIN"
CREATOR = "CREATOR_GROUP_HOLDOUT_NOT_TRAIN"
EXPECTED_ROLE_COUNTS = {TRAIN: 55, VAL: 9, PRINT: 6, CREATOR: 8}
DATASET = ADVENTUREX / "datasets" / "rootscope_machine_curated_provisional_v3"
MODEL = ROOTSCOPE / "deploy" / "x5" / "models" / "rootscope_seed17_cpu_experimental_opset11.onnx"
OUTPUT = ROOTSCOPE / "evidence" / "omega_vision_v3_20260723" / "vision_consolidated.json"


class EvidenceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceError(f"{path} is not a JSON object")
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise EvidenceError(f"manifest line {number} is not an object")
        rows.append(value)
    return rows


def _intersection_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    groups = {
        role: {str(row[field]) for row in rows if row["experimental_split_suggestion"] == role}
        for role in EXPECTED_ROLE_COUNTS
    }
    results: dict[str, int] = {}
    roles = tuple(EXPECTED_ROLE_COUNTS)
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1 :]:
            results[f"{left}__{right}"] = len(groups[left] & groups[right])
    return results


def audit_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = DATASET / "manifest.jsonl"
    receipt_path = DATASET / "receipt.json"
    rows = load_rows(manifest_path)
    receipt = load_json(receipt_path)
    if len(rows) != 78:
        raise EvidenceError(f"expected 78 records, found {len(rows)}")
    role_counts = {
        role: sum(row.get("experimental_split_suggestion") == role for row in rows)
        for role in EXPECTED_ROLE_COUNTS
    }
    if role_counts != EXPECTED_ROLE_COUNTS:
        raise EvidenceError(f"role counts differ: {role_counts}")
    if sha256_file(manifest_path) != receipt.get("manifest_sha256"):
        raise EvidenceError("dataset receipt does not bind manifest")

    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    class_counts = {name: 0 for name in CLASS_ORDER}
    for index, row in enumerate(rows):
        filename = row.get("filename")
        expected_sha = row.get("copied_image_sha256")
        label = row.get("class_id")
        if not isinstance(filename, str) or filename in seen_paths:
            raise EvidenceError(f"invalid/duplicate filename at record {index}")
        image_path = (DATASET / filename).resolve()
        try:
            image_path.relative_to(DATASET.resolve())
        except ValueError as exc:
            raise EvidenceError(f"path escape at record {index}") from exc
        if not image_path.is_file() or sha256_file(image_path) != expected_sha:
            raise EvidenceError(f"image hash mismatch at record {index}")
        if expected_sha in seen_hashes:
            raise EvidenceError(f"duplicate image bytes at record {index}")
        if label not in class_counts:
            raise EvidenceError(f"unknown class at record {index}")
        authority = row.get("authority")
        if (
            row.get("machine_curated_only") is not True
            or row.get("human_reviewed") is not False
            or row.get("training_eligible") is not False
            or row.get("formal_split_assigned") is not False
            or row.get("split") != "UNASSIGNED_DO_NOT_TRAIN"
            or not isinstance(authority, dict)
            or any(value is not False for value in authority.values())
        ):
            raise EvidenceError(f"authority boundary violated at record {index}")
        seen_paths.add(filename)
        seen_hashes.add(expected_sha)
        class_counts[label] += 1

    creator_overlap = _intersection_counts(rows, "creator_group")
    source_overlap = _intersection_counts(rows, "source_group")
    for key, count in creator_overlap.items():
        if PRINT not in key and count != 0:
            raise EvidenceError(f"creator leakage: {key}={count}")
    if any(source_overlap.values()):
        raise EvidenceError(f"source group leakage: {source_overlap}")
    return rows, {
        "status": "PASS_MACHINE_CURATED_PROVISIONAL_IDENTITY_ONLY",
        "record_count": len(rows),
        "class_counts": class_counts,
        "role_counts": role_counts,
        "byte_hash_verified_count": len(seen_hashes),
        "creator_intersection_counts": creator_overlap,
        "source_group_intersection_counts": source_overlap,
        "manifest_sha256": sha256_file(manifest_path),
        "receipt_sha256": sha256_file(receipt_path),
        "sha256sums_sha256": sha256_file(DATASET / "SHA256SUMS"),
        "human_reviewed": False,
        "formal_split_assigned": False,
        "training_eligible": False,
        "data_locked": False,
    }


def preprocess(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    width, height = rgb.size
    short = min(width, height)
    long = max(width, height)
    resized_long = int(256 * long / short)
    resized_size = (256, resized_long) if width <= height else (resized_long, 256)
    resized = rgb.resize(resized_size, resample=Image.Resampling.BILINEAR)
    left = int(round((resized.size[0] - 224) / 2.0))
    top = int(round((resized.size[1] - 224) / 2.0))
    cropped = resized.crop((left, top, left + 224, top + 224))
    array = np.asarray(cropped, dtype=np.float32) / np.float32(255.0)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    return np.transpose((array - mean) / std, (2, 0, 1))[None].astype(np.float32)


def infer(rows: list[dict[str, Any]]) -> tuple[list[list[float]], list[Any], dict[str, Any]]:
    import onnxruntime as ort

    if sha256_file(MODEL) != "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad":
        raise EvidenceError("seed17 ONNX hash mismatch")
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise EvidenceError(f"ONNX provider is not CPU-only: {session.get_providers()}")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != "image"
        or list(inputs[0].shape) != [1, 3, 224, 224]
        or len(outputs) != 1
        or outputs[0].name != "logits"
        or list(outputs[0].shape) != [1, 4]
    ):
        raise EvidenceError("seed17 ONNX I/O contract mismatch")

    logits: list[list[float]] = []
    qualities: list[Any] = []
    for row in rows:
        with Image.open(DATASET / row["filename"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            qualities.append(evaluate_quality(rgb))
            output = np.asarray(session.run(["logits"], {"image": preprocess(image)})[0], dtype=np.float64)
        if output.shape != (1, 4) or not np.isfinite(output).all():
            raise EvidenceError("ONNX output was not finite [1,4] logits")
        logits.append([float(value) for value in output[0]])
    return logits, qualities, {
        "runtime": "onnxruntime",
        "version": ort.__version__,
        "provider_requested": "CPUExecutionProvider",
        "providers_actual": session.get_providers(),
        "output_semantics": "FINAL_CLASSIFIER_LOGITS",
        "output_name": "logits",
        "output_shape": [1, 4],
        "embedding_output_present": False,
    }


def indices_for(rows: list[dict[str, Any]], roles: Iterable[str]) -> list[int]:
    accepted = set(roles)
    return [index for index, row in enumerate(rows) if row["experimental_split_suggestion"] in accepted]


def domain_metrics(
    rows: list[dict[str, Any]],
    logits: list[list[float]],
    qualities: list[Any],
    calibration: Calibration,
    indices: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    correct = 0
    accepted = 0
    accepted_correct = 0
    reason_counts: dict[str, int] = {}
    for index in indices:
        decision = decide(logits[index], qualities[index], calibration)
        label = rows[index]["class_id"]
        raw_correct = decision.raw_top1_class == label
        correct += int(raw_correct)
        if decision.decision == "CLASSIFY":
            accepted += 1
            accepted_correct += int(decision.predicted_class == label)
        for reason in decision.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        records.append(
            {
                "record_index": index,
                "pageid": rows[index].get("pageid"),
                "filename": rows[index]["filename"],
                "role": rows[index]["experimental_split_suggestion"],
                "machine_curated_label_not_human_truth": label,
                "image_sha256": rows[index]["copied_image_sha256"],
                "final_logits": logits[index],
                "decision": decision.to_dict(),
            }
        )
    count = len(indices)
    return {
        "sample_count": count,
        "raw_top1_accuracy_against_machine_curated_labels": correct / count,
        "classified_count": accepted,
        "coverage": accepted / count,
        "selective_accuracy_against_machine_curated_labels": (
            accepted_correct / accepted if accepted else None
        ),
        "abstained_count": count - accepted,
        "abstain_reason_counts": dict(sorted(reason_counts.items())),
    }, records


def build() -> dict[str, Any]:
    rows, dataset_audit = audit_dataset()
    logits, qualities, runtime = infer(rows)
    train_indices = indices_for(rows, (TRAIN,))
    val_indices = indices_for(rows, (VAL,))
    reference_indices = train_indices + val_indices
    class_to_index = {name: index for index, name in enumerate(CLASS_ORDER)}
    calibration = calibrate(
        reference_logits=[logits[index] for index in reference_indices],
        reference_quality=[qualities[index] for index in reference_indices],
        validation_logits=[logits[index] for index in val_indices],
        validation_labels=[class_to_index[rows[index]["class_id"]] for index in val_indices],
        class_order=CLASS_ORDER,
        alpha=0.20,
        temperature=1.0,
    )

    domains: dict[str, Any] = {}
    sample_records: list[dict[str, Any]] = []
    for name, roles in (
        ("experimental_train_suggestion", (TRAIN,)),
        ("experimental_validation_suggestion", (VAL,)),
        ("creator_group_holdout_single_evaluation", (CREATOR,)),
        ("digital_print_source_holdout_single_evaluation", (PRINT,)),
    ):
        metrics, records = domain_metrics(
            rows, logits, qualities, calibration, indices_for(rows, roles)
        )
        domains[name] = metrics
        sample_records.extend(records)

    artifact_hashes = {
        "dataset_manifest": dataset_audit["manifest_sha256"],
        "dataset_receipt": dataset_audit["receipt_sha256"],
        "dataset_sha256sums": dataset_audit["sha256sums_sha256"],
        "seed17_onnx": sha256_file(MODEL),
        "seed17_deployment_manifest": sha256_file(
            ROOTSCOPE / "deploy" / "x5" / "seed17_cpu_deployment_manifest.json"
        ),
        "implementation_ood": sha256_file(ROOTSCOPE / "app" / "omega_vision" / "ood.py"),
        "implementation_evidence_builder": sha256_file(Path(__file__).resolve()),
    }
    report: dict[str, Any] = {
        "schema_version": "rootscope.omega-vision-consolidated.v1",
        "status": "MACHINE_CURATED_EXPERIMENTAL_NOT_QUALIFIED",
        "dataset_identity_audit": dataset_audit,
        "model": {
            "name": "rootscope_seed17_cpu_experimental_opset11",
            "class_order": list(CLASS_ORDER),
            "runtime": runtime,
            "model_candidate": False,
            "model_qualified": False,
            "bpu_ready": False,
            "bpu_compiled_for_this_receipt": False,
            "selected_bin": None,
        },
        "calibration": {
            **calibration.to_dict(),
            "fit_sample_counts": {"train_suggestion": 55, "validation_suggestion": 9},
            "holdout_used_for_thresholds": False,
            "finite_sample_method": (
                "Energy upper/maxprob lower/quality bounds use fixed finite-order statistics "
                "from train+validation; label nonconformity uses validation-only split conformal."
            ),
        },
        "mahalanobis": {
            "status": "SKIPPED_NO_VALID_EMBEDDING_OUTPUT",
            "reason": (
                "The immutable ONNX exposes only final classifier logits [1,4]. "
                "Final logits are not relabeled as penultimate embeddings."
            ),
        },
        "fusion": {
            "rule": (
                "CLASSIFY only when Energy, max probability, all image-quality gates, "
                "singleton split-conformal agreement, and non-unknown top1 all pass; otherwise ABSTAIN."
            ),
            "fail_closed": True,
            "unknown_is_abstain": True,
            "zero_authority": True,
        },
        "domain_evaluations": domains,
        "holdout_protocol": {
            "creator_group_holdout_evaluation_count": 1,
            "digital_print_source_holdout_evaluation_count": 1,
            "used_for_weights_checkpoint_temperature_or_thresholds": False,
            "digital_print_source_is_uvc_recapture": False,
            "physical_print_domain_tested": False,
        },
        "sample_records": sample_records,
        "artifact_sha256": artifact_hashes,
        "execution": {
            "training_executed": False,
            "torch_imported": False,
            "new_rtx4050_candidate_created": False,
            "reason_no_new_candidate": (
                "Existing seed17 immutable ONNX was sufficient; new training was prohibited "
                "for this closeout and local torch loading was blocked by Windows application control."
            ),
            "hardware_touched": False,
            "network_touched": False,
            "x5_touched": False,
            "bpu_used": False,
            "physical_authority": False,
            "execution_authority": False,
        },
        "truth_boundaries": {
            "human_reviewed": False,
            "formal_split_assigned": False,
            "training_eligible": False,
            "data_locked": False,
            "rights_approved": False,
            "print_eligible": False,
            "model_qualified": False,
            "camera_qualified": False,
            "physical_domain_qualified": False,
            "accuracy_values_compare_against_machine_curated_labels_not_human_ground_truth": True,
        },
    }
    report["composition_root_sha256"] = canonical_sha(report)
    return report


def main() -> int:
    report = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise EvidenceError(f"refusing to overwrite single-evaluation receipt: {OUTPUT}")
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha256_file(OUTPUT),
                "composition_root_sha256": report["composition_root_sha256"],
                "status": report["status"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
