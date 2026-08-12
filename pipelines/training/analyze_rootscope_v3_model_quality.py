#!/usr/bin/env python3
"""Independent read-only quality analysis for the frozen RootScope v3 run.

The script never imports the training pipeline and never writes inside the run
or dataset.  It verifies the frozen receipt/hash envelope, reloads every
checkpoint on CPU, reproduces center-crop validation/holdout metrics, and
evaluates a pre-declared equal-logit ensemble.  A deterministic 10-view
five-crop-plus-horizontal-flip result is reported separately as exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


CLASS_NAMES = ("grass_clump", "low_shrub", "young_tree", "unknown")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INPUT_SIZE = 224
SEEDS = (17, 29, 43)
RUN_RELATIVE = "output/rootscope_machine_curated_experimental_runs/v3_rtx4050_multiseed_20260717_r1"
PACK_RELATIVE = "datasets/rootscope_machine_curated_provisional_v3"
TRAIN_ROLE = "EXPERIMENTAL_TRAIN_SUGGESTION"
VAL_ROLE = "EXPERIMENTAL_VAL_SUGGESTION"
PRINT_ROLE = "PRINT_DEMO_HOLDOUT_NOT_TRAIN"
CREATOR_HOLDOUT_ROLE = "CREATOR_GROUP_HOLDOUT_NOT_TRAIN"
NATURAL_DOMAIN = "NATURAL_WEB_VALIDATION"
PRINT_DOMAIN = "DIGITAL_PRINT_SOURCE_HOLDOUT_NOT_UVC_RECAPTURE"
RUN_STATUS = "MACHINE_CURATED_EXPERIMENTAL_MODEL_NOT_QUALIFIED"
PACK_STATUS = "MACHINE_CURATED_EXPERIMENTAL_V3_ONLY_NOT_HUMAN_REVIEWED_NOT_A1_NOT_DATA_LOCKED"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
SUM_RE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")

ROLE_SEMANTICS = {
    TRAIN_ROLE: (
        "Optimization source seen by training; recognition on this role is resubstitution "
        "behavior, not generalization evidence and not a print-domain claim."
    ),
    VAL_ROLE: (
        "Natural-web validation used for checkpoint selection, temperature, and rejection "
        "calibration; it is not an untouched test set."
    ),
    PRINT_ROLE: (
        "Digital source held out from training/validation for a future print demo; this is "
        "not a physical print, UVC recapture, glare, distance, or optics-loop result."
    ),
    CREATOR_HOLDOUT_ROLE: (
        "Creator-group holdout unused by the reported training evaluation; any score here is "
        "post-hoc exploratory only and is not a pre-registered qualification result."
    ),
}

YOUNG_TREE_VISUAL_NOTES = {
    75760716: {
        "visual_observation": (
            "The intended sapling has a thin visible trunk, but it occupies a cluttered woodland "
            "scene with mature trunks, branches, grass, and shrub-like foliage across most of the frame."
        ),
        "likely_failure_mechanism": (
            "Background and multi-plant context dominate the 224-pixel crop, while the target's "
            "single-trunk silhouette is small; this is a source-composition/domain problem."
        ),
        "label_assessment": (
            "The young-tree label is visually plausible for the central plant; the evidence does "
            "not establish a wrong label, but the image is unsuitable as a clean class card."
        ),
    },
    98911085: {
        "visual_observation": (
            "The image is a close, oblique/top-down view of a very small Tamarix sapling.  The full "
            "trunk-crown-ground silhouette and scale cues are weak, and feathery growth reads as a low shrub."
        ),
        "likely_failure_mechanism": (
            "Pose and framing collapse the tree-versus-shrub structural cue; sandy ground occupies much "
            "of the crop and the plant is visually closer to the low-shrub training morphology."
        ),
        "label_assessment": (
            "Metadata and the visible stem support sapling, so a definite label error is not shown; "
            "this is primarily a label-definition/presentation ambiguity and domain issue."
        ),
    },
    180772202: {
        "visual_observation": (
            "A leafless planted tree is isolated against snow with a strong trunk-and-crown silhouette "
            "and a watering bag, but it is not a desert-scene example."
        ),
        "likely_failure_mechanism": (
            "When a seed fails, the most plausible cause is the extreme snow/leafless-domain shift, "
            "not absence of tree structure."
        ),
        "label_assessment": "The young-tree label is visually clear; no label error is indicated.",
    },
    184915021: {
        "visual_observation": (
            "The planted tree has a visible stem and crown but is partly overlaid by a large wire cage, "
            "with grassland and distant vegetation adding texture."
        ),
        "likely_failure_mechanism": (
            "The cage and background introduce strong non-plant edges; any failure is likely an "
            "occlusion/context-domain issue rather than a clearly wrong young-tree label."
        ),
        "label_assessment": "The young-tree label is visually plausible; no definite label error is shown.",
    },
}

# Filled only with analyst-inspected images after the machine-stability gate is
# computed.  The script rejects a selection that is not correct for every seed,
# both center/TTA paths, and both equal-logit ensembles.
VISUALLY_INSPECTED_CARD_PAGEIDS: dict[str, list[int]] = {
    "grass_clump": [163498042, 38233728],
    "low_shrub": [68787114, 66745979],
    "young_tree": [92774234],
    "unknown": [157364276],
}

VISUAL_CARD_NOTES = {
    163498042: "Clean, isolated desert grass tussock with base and radiating blades visible; train role only.",
    38233728: "Dominant fountain-grass clump and stable digital holdout; background vegetation remains a caveat.",
    68787114: "Compact, low, isolated grey-green shrub with clear mound morphology; train role only.",
    66745979: "Large foreground Atriplex shrub, stable digital holdout, with limited background shrub texture.",
    92774234: "Newly planted tree with visible trunk, crown, basin, and scale cues; clearest stable young-tree card, but train role.",
    157364276: "Visually clean plant-absent desert dune scene; strong unknown/non-target card, but train role.",
}


class AnalysisError(RuntimeError):
    """Fail-closed quality-analysis error."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise AnalysisError(f"cannot read JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise AnalysisError(f"blank JSONL line {line_number}: {path}")
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise AnalysisError(f"invalid JSONL line {line_number}: {error}") from error
        if not isinstance(value, dict):
            raise AnalysisError(f"JSONL line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise AnalysisError(f"empty JSONL: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    rows: list[str] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        if path.is_symlink():
            raise AnalysisError(f"symlink in frozen tree: {path}")
        rows.append(f"{path.relative_to(root).as_posix()}\0{sha256_file(path)}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _canonical_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or path.as_posix() != value:
        raise AnalysisError(f"non-canonical relative path: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise AnalysisError(f"unsafe relative path: {value!r}")
    return value


def safe_child(root: Path, relative: str) -> Path:
    relative = _canonical_relative(relative)
    path = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise AnalysisError(f"path escapes root: {relative}") from error
    if path.is_symlink():
        raise AnalysisError(f"symlink target rejected: {relative}")
    return path


def parse_sha256sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = SUM_RE.fullmatch(line)
        if match is None:
            raise AnalysisError(f"invalid SHA256SUMS line {line_number}")
        digest, relative = match.groups()
        relative = _canonical_relative(relative)
        if relative in result or relative == "SHA256SUMS":
            raise AnalysisError(f"duplicate/self-referential SHA256SUMS path: {relative}")
        result[relative] = digest
    if not result:
        raise AnalysisError("empty SHA256SUMS")
    return result


def verify_hash_envelope(run_root: Path) -> dict[str, Any]:
    sums = parse_sha256sums(run_root / "SHA256SUMS")
    actual = {
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(sums) != actual:
        raise AnalysisError(
            f"run SHA256SUMS coverage mismatch: missing={sorted(actual-set(sums))}, stale={sorted(set(sums)-actual)}"
        )
    for relative, expected in sums.items():
        actual_hash = sha256_file(safe_child(run_root, relative))
        if actual_hash != expected:
            raise AnalysisError(f"run hash mismatch for {relative}: {actual_hash} != {expected}")
    return {
        "full_coverage": True,
        "covered_file_count": len(sums),
        "sha256sums_sha256": sha256_file(run_root / "SHA256SUMS"),
        "run_tree_sha256": tree_sha256(run_root),
        "hashes": sums,
    }


def require_false(mapping: Mapping[str, Any], key: str, *, location: str) -> None:
    if mapping.get(key) is not False:
        raise AnalysisError(f"{location}.{key} must be exactly false")


def verify_receipt(run_root: Path, pack_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope = verify_hash_envelope(run_root)
    receipt = load_json(run_root / "run_receipt.json")
    if receipt.get("status") != RUN_STATUS:
        raise AnalysisError("unexpected run status")
    if tuple(receipt.get("seeds", ())) != SEEDS:
        raise AnalysisError("receipt does not contain exactly the fixed seeds 17,29,43")
    if receipt.get("class_order") != list(CLASS_NAMES):
        raise AnalysisError("receipt class order mismatch")
    if receipt.get("input_shape") != [1, 3, INPUT_SIZE, INPUT_SIZE]:
        raise AnalysisError("receipt input shape mismatch")
    for key in (
        "formal_a1_dataset",
        "human_reviewed",
        "rights_approved",
        "rights",
        "training_eligible",
        "print_eligible",
        "data_locked",
        "model_qualified",
        "model_candidate",
        "x5_ready",
        "bpu_compiled",
        "physical_print_tested",
        "uvc_recapture_evaluated",
        "execution_authority",
    ):
        require_false(receipt, key, location="receipt")
    authority = receipt.get("authority")
    if not isinstance(authority, dict) or not authority or any(value is not False for value in authority.values()):
        raise AnalysisError("receipt authority map must be non-empty and entirely false")
    if receipt.get("experimental_model_candidate") is not True:
        raise AnalysisError("receipt experimental_model_candidate must be true")
    if receipt.get("digital_print_holdout_is_uvc_recapture") is not False:
        raise AnalysisError("receipt incorrectly claims UVC recapture")

    artifact_hashes = receipt.get("artifact_hashes_before_receipt")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise AnalysisError("missing artifact_hashes_before_receipt")
    for relative, expected in artifact_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or SHA_RE.fullmatch(expected) is None:
            raise AnalysisError("invalid artifact hash binding in receipt")
        actual = sha256_file(safe_child(run_root, relative))
        if actual != expected:
            raise AnalysisError(f"receipt artifact binding mismatch for {relative}")

    manifest_hash = sha256_file(pack_root / "manifest.jsonl")
    unchanged = receipt.get("input_and_formal_authority_unchanged")
    if not isinstance(unchanged, dict) or unchanged.get("unchanged") is not True:
        raise AnalysisError("receipt missing unchanged input/formal authority record")
    after = unchanged.get("after")
    if not isinstance(after, dict) or after.get("manifest_sha256") != manifest_hash:
        raise AnalysisError("receipt input manifest hash does not match current v3 pack")
    pack_receipt = load_json(pack_root / "receipt.json")
    if pack_receipt.get("status") != PACK_STATUS:
        raise AnalysisError("unexpected v3 pack status")

    seed_results = receipt.get("seed_results")
    if not isinstance(seed_results, list) or [item.get("seed") for item in seed_results] != list(SEEDS):
        raise AnalysisError("receipt seed_results mismatch")
    expected_selected = max(seed_results, key=lambda value: tuple(value["selection_key"]))
    if receipt.get("selected_seed") != expected_selected:
        raise AnalysisError("receipt selected seed is not the natural-validation selection-key winner")
    return receipt, {
        "status": "PASS",
        "full_sha256_coverage": envelope["full_coverage"],
        "hash_covered_file_count": envelope["covered_file_count"],
        "sha256sums_sha256": envelope["sha256sums_sha256"],
        "run_tree_sha256": envelope["run_tree_sha256"],
        "run_receipt_sha256": sha256_file(run_root / "run_receipt.json"),
        "artifact_hashes_before_receipt_verified": True,
        "authority_map_all_false": True,
        "required_non_claim_flags_false": True,
        "selected_seed_recomputed_from_natural_validation_key": expected_selected["seed"],
        "pack_manifest_sha256": manifest_hash,
        "pack_receipt_sha256": sha256_file(pack_root / "receipt.json"),
        "pack_tree_sha256": tree_sha256(pack_root),
    }


def build_center_transform() -> Any:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_tta_views(image: Any) -> list[Any]:
    """Return the pre-declared 10 views: 5 crops and each crop's hflip."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as vision_functional

    resized = transforms.Resize(256, interpolation=InterpolationMode.BILINEAR)(image)
    crops = transforms.FiveCrop(INPUT_SIZE)(resized)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    tensors: list[Any] = []
    for crop in crops:
        tensors.append(normalize(vision_functional.to_tensor(crop)))
        tensors.append(normalize(vision_functional.to_tensor(vision_functional.hflip(crop))))
    if len(tensors) != 10:
        raise AnalysisError("fixed TTA protocol did not produce exactly 10 views")
    return tensors


def build_model() -> Any:
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.avgpool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def load_checkpoint_model(checkpoint_path: Path, *, expected_seed: int, manifest_sha256: str) -> Any:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise AnalysisError(f"checkpoint root is not an object: {checkpoint_path}")
    expected = {
        "schema_version": "rootscope.resnet18_experimental_checkpoint.v1",
        "status": RUN_STATUS,
        "seed": expected_seed,
        "class_order": list(CLASS_NAMES),
        "input_shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
        "architecture": "torchvision.resnet18_fixed_avgpool7x7",
        "input_pack_manifest_sha256": manifest_sha256,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise AnalysisError(f"checkpoint {checkpoint_path.name} field {key!r} mismatch")
    if not isinstance(checkpoint.get("epoch"), int) or checkpoint["epoch"] <= 0:
        raise AnalysisError("checkpoint epoch is invalid")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise AnalysisError("checkpoint state dict missing")
    model = build_model()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_images(pack_root: Path, rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    from PIL import Image, ImageOps

    images: list[Any] = []
    for row in rows:
        path = safe_child(pack_root, str(row["filename"]))
        with Image.open(path) as opened:
            images.append(ImageOps.exif_transpose(opened).convert("RGB"))
    return images


def infer_center(model: Any, tensors: Any, *, batch_size: int = 16) -> Any:
    import torch

    logits: list[Any] = []
    with torch.inference_mode():
        for start in range(0, int(tensors.shape[0]), batch_size):
            logits.append(model(tensors[start : start + batch_size]).detach().cpu())
    return torch.cat(logits)


def infer_tta(model: Any, images: Sequence[Any], *, images_per_batch: int = 8) -> Any:
    import torch

    results: list[Any] = []
    with torch.inference_mode():
        for start in range(0, len(images), images_per_batch):
            group = images[start : start + images_per_batch]
            views = [view for image in group for view in build_tta_views(image)]
            batch = torch.stack(views)
            logits = model(batch).detach().cpu().reshape(len(group), 10, len(CLASS_NAMES)).mean(dim=1)
            results.append(logits)
    return torch.cat(results)


def metrics_from_logits(logits: Any, labels: Any, *, domain: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    if int(labels.numel()) == 0:
        raise AnalysisError(f"empty evaluation domain: {domain}")
    probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[int(truth), int(prediction)] += 1
    per_class: dict[str, Any] = {}
    recalls: list[float] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        support = int((labels == class_index).sum().item())
        correct = int(confusion[class_index, class_index].item())
        recall = correct / support if support else None
        if recall is not None:
            recalls.append(recall)
        per_class[class_name] = {"support": support, "recall": recall}
    confidence = probabilities.max(dim=1).values
    correctness = predictions.eq(labels).float()
    ece = 0.0
    for low in torch.linspace(0.0, 0.9, 10):
        high = low + 0.1
        mask = (confidence >= low) & (
            confidence < high if float(high) < 1.0 else confidence <= 1.0
        )
        if bool(mask.any()):
            fraction = float(mask.float().mean().item())
            ece += fraction * abs(
                float(correctness[mask].mean().item()) - float(confidence[mask].mean().item())
            )
    return {
        "domain": domain,
        "sample_count": int(labels.numel()),
        "accuracy": float(correctness.mean().item()),
        "balanced_accuracy_present_classes": sum(recalls) / len(recalls),
        "all_four_classes_present": all(per_class[name]["support"] > 0 for name in CLASS_NAMES),
        "cross_entropy": float(functional.cross_entropy(logits, labels).item()),
        "ece_10_bin": ece,
        "confusion_matrix_truth_rows_prediction_columns": confusion.tolist(),
        "per_class": per_class,
    }


def role_indices(rows: Sequence[Mapping[str, Any]], role: str) -> list[int]:
    return [index for index, row in enumerate(rows) if row.get("experimental_split_suggestion") == role]


def subset_tensor(tensor: Any, indices: Sequence[int]) -> Any:
    import torch

    return tensor[torch.tensor(indices, dtype=torch.long)]


def compare_reported_metrics(recomputed: Mapping[str, Any], reported: Mapping[str, Any]) -> dict[str, Any]:
    exact_keys = (
        "sample_count",
        "all_four_classes_present",
        "confusion_matrix_truth_rows_prediction_columns",
        "per_class",
    )
    for key in exact_keys:
        if recomputed.get(key) != reported.get(key):
            raise AnalysisError(f"reported metric mismatch for exact field {key}")
    tolerances = {
        "accuracy": 2e-6,
        "balanced_accuracy_present_classes": 1e-12,
        # The receipt was produced by CUDA while this independent replay is
        # deliberately CPU-only; ResNet convolution reduction order creates
        # small floating-point drift even when all discrete predictions match.
        "cross_entropy": 2e-4,
        "ece_10_bin": 2e-4,
    }
    differences: dict[str, float] = {}
    for key, tolerance in tolerances.items():
        difference = abs(float(recomputed[key]) - float(reported[key]))
        differences[key] = difference
        if difference > tolerance:
            raise AnalysisError(
                f"reported metric mismatch for {key}: diff={difference} tolerance={tolerance}"
            )
    return {"status": "PASS", "absolute_differences": differences, "tolerances": tolerances}


def probability_record(
    row: Mapping[str, Any],
    logits: Any,
    *,
    temperature: float = 1.0,
) -> dict[str, Any]:
    import torch

    probabilities = torch.softmax(logits / temperature, dim=0)
    top2 = probabilities.topk(k=2)
    predicted_index = int(top2.indices[0].item())
    truth = str(row["class_id"])
    return {
        "pageid": int(row["pageid"]),
        "filename": str(row["filename"]),
        "title": str(row.get("title", "")),
        "truth_class": truth,
        "role": str(row["experimental_split_suggestion"]),
        "predicted_class": CLASS_NAMES[predicted_index],
        "correct": CLASS_NAMES[predicted_index] == truth,
        "confidence": float(top2.values[0].item()),
        "top1_minus_top2_margin": float((top2.values[0] - top2.values[1]).item()),
        "probabilities_by_class": {
            class_name: float(probabilities[index].item())
            for index, class_name in enumerate(CLASS_NAMES)
        },
    }


def predictions_for_domain(
    rows: Sequence[Mapping[str, Any]],
    logits: Any,
    indices: Sequence[int],
    *,
    temperature: float = 1.0,
) -> list[dict[str, Any]]:
    return [probability_record(rows[index], logits[index], temperature=temperature) for index in indices]


def validate_manifest_rows(rows: Sequence[Mapping[str, Any]], pack_root: Path) -> dict[str, Any]:
    if len(rows) != 78:
        raise AnalysisError(f"expected 78 v3 rows, got {len(rows)}")
    seen_pageids: set[int] = set()
    role_counts: Counter[str] = Counter()
    for row in rows:
        class_id = row.get("class_id")
        role = row.get("experimental_split_suggestion")
        pageid = row.get("pageid")
        if class_id not in CLASS_NAMES or role not in ROLE_SEMANTICS or not isinstance(pageid, int):
            raise AnalysisError("manifest class/role/pageid contract violation")
        if pageid in seen_pageids:
            raise AnalysisError(f"duplicate pageid in manifest: {pageid}")
        seen_pageids.add(pageid)
        image = safe_child(pack_root, str(row["filename"]))
        if sha256_file(image) != row.get("copied_image_sha256"):
            raise AnalysisError(f"manifest image hash mismatch for pageid {pageid}")
        role_counts[str(role)] += 1
    expected_counts = {
        TRAIN_ROLE: 55,
        VAL_ROLE: 9,
        PRINT_ROLE: 6,
        CREATOR_HOLDOUT_ROLE: 8,
    }
    if dict(role_counts) != expected_counts:
        raise AnalysisError(f"role counts mismatch: {dict(role_counts)}")
    return {"row_count": len(rows), "role_counts": expected_counts, "image_hashes_verified": True}


def build_candidate_pool(
    rows: Sequence[Mapping[str, Any]],
    center_by_seed: Mapping[int, Any],
    tta_by_seed: Mapping[int, Any],
    center_ensemble: Any,
    tta_ensemble: Any,
) -> tuple[dict[str, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    import torch

    ranked: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_NAMES}
    by_pageid: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        truth = str(row["class_id"])
        truth_index = CLASS_TO_INDEX[truth]
        center_predictions = {
            str(seed): CLASS_NAMES[int(center_by_seed[seed][index].argmax().item())] for seed in SEEDS
        }
        tta_predictions = {
            str(seed): CLASS_NAMES[int(tta_by_seed[seed][index].argmax().item())] for seed in SEEDS
        }
        center_ensemble_prediction = CLASS_NAMES[int(center_ensemble[index].argmax().item())]
        tta_ensemble_prediction = CLASS_NAMES[int(tta_ensemble[index].argmax().item())]
        all_correct = all(value == truth for value in center_predictions.values()) and all(
            value == truth for value in tta_predictions.values()
        )
        all_correct = all_correct and center_ensemble_prediction == truth and tta_ensemble_prediction == truth
        true_probabilities = [
            float(torch.softmax(center_by_seed[seed][index], dim=0)[truth_index].item()) for seed in SEEDS
        ] + [
            float(torch.softmax(tta_by_seed[seed][index], dim=0)[truth_index].item()) for seed in SEEDS
        ] + [
            float(torch.softmax(center_ensemble[index], dim=0)[truth_index].item()),
            float(torch.softmax(tta_ensemble[index], dim=0)[truth_index].item()),
        ]
        record = {
            "pageid": int(row["pageid"]),
            "filename": str(row["filename"]),
            "title": str(row.get("title", "")),
            "class_id": truth,
            "role": str(row["experimental_split_suggestion"]),
            "role_semantics": ROLE_SEMANTICS[str(row["experimental_split_suggestion"])],
            "stable_correct_all_three_seeds_center_and_tta_and_both_ensembles": all_correct,
            "minimum_true_class_probability_across_eight_fixed_paths": min(true_probabilities),
            "center_predictions_by_seed": center_predictions,
            "tta_predictions_by_seed": tta_predictions,
            "center_equal_logit_ensemble_prediction": center_ensemble_prediction,
            "tta_equal_logit_ensemble_prediction": tta_ensemble_prediction,
        }
        by_pageid[int(row["pageid"])] = record
        if all_correct:
            ranked[truth].append(record)
    for class_name in CLASS_NAMES:
        ranked[class_name].sort(
            key=lambda record: (-record["minimum_true_class_probability_across_eight_fixed_paths"], record["pageid"])
        )
        ranked[class_name] = ranked[class_name][:10]
    return ranked, by_pageid


def build_visual_card_selections(by_pageid: Mapping[int, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for class_name, pageids in VISUALLY_INSPECTED_CARD_PAGEIDS.items():
        selections: list[dict[str, Any]] = []
        for pageid in pageids:
            record = by_pageid.get(pageid)
            if record is None or record.get("class_id") != class_name:
                raise AnalysisError(f"visual card pageid {pageid} is missing or has wrong class")
            if record.get("stable_correct_all_three_seeds_center_and_tta_and_both_ensembles") is not True:
                raise AnalysisError(f"visual card pageid {pageid} does not pass the fixed model-stability gate")
            selection = dict(record)
            selection["analyst_visual_note"] = VISUAL_CARD_NOTES[pageid]
            selection["visual_truth_authority"] = False
            selections.append(selection)
        result[class_name] = selections
    return result


def build_young_tree_analysis(
    rows: Sequence[Mapping[str, Any]],
    center_by_seed: Mapping[int, Any],
    center_ensemble: Any,
    tta_ensemble: Any,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row["class_id"] != "young_tree" or row["experimental_split_suggestion"] not in {VAL_ROLE, PRINT_ROLE}:
            continue
        pageid = int(row["pageid"])
        seed_predictions = {
            str(seed): CLASS_NAMES[int(center_by_seed[seed][index].argmax().item())] for seed in SEEDS
        }
        center_prediction = CLASS_NAMES[int(center_ensemble[index].argmax().item())]
        tta_prediction = CLASS_NAMES[int(tta_ensemble[index].argmax().item())]
        records.append(
            {
                "pageid": pageid,
                "filename": str(row["filename"]),
                "title": str(row.get("title", "")),
                "role": str(row["experimental_split_suggestion"]),
                "center_predictions_by_seed": seed_predictions,
                "center_equal_logit_ensemble_prediction": center_prediction,
                "exploratory_tta_equal_logit_ensemble_prediction": tta_prediction,
                "any_center_seed_young_tree_to_low_shrub": "low_shrub" in seed_predictions.values(),
                "center_ensemble_young_tree_to_low_shrub": center_prediction == "low_shrub",
                "visual_interpretation": YOUNG_TREE_VISUAL_NOTES.get(pageid),
            }
        )
    return {
        "interpretation_authority": "ANALYST_VISUAL_DIAGNOSIS_ONLY_NOT_HUMAN_GROUND_TRUTH_REVIEW",
        "label_error_established": False,
        "dominant_assessment": (
            "The observed young_tree-to-low_shrub errors are better explained by target scale, "
            "crop/composition, viewing pose, occlusion, and source-domain mismatch than by a proven "
            "wrong label.  The print-holdout images are poor clean-card exemplars."
        ),
        "whole_frame_letterbox_implication": (
            "Worth an isolated v4 ablation: the current resize-plus-center-crop can discard whole-plant "
            "silhouette and scale cues.  Letterbox should be compared under frozen seeds/splits against "
            "center-crop without using the print holdout to tune or select."
        ),
        "records": records,
    }


def generate_report(workspace: Path, run_root: Path) -> dict[str, Any]:
    import gc
    import torch

    pack_root = (workspace / PACK_RELATIVE).resolve(strict=True)
    receipt, receipt_verification = verify_receipt(run_root, pack_root)
    rows = load_jsonl(pack_root / "manifest.jsonl")
    manifest_verification = validate_manifest_rows(rows, pack_root)
    labels = torch.tensor([CLASS_TO_INDEX[str(row["class_id"])] for row in rows], dtype=torch.long)
    val_indices = role_indices(rows, VAL_ROLE)
    print_indices = role_indices(rows, PRINT_ROLE)
    images = load_images(pack_root, rows)
    center_transform = build_center_transform()
    center_tensors = torch.stack([center_transform(image) for image in images])
    manifest_sha256 = sha256_file(pack_root / "manifest.jsonl")

    center_by_seed: dict[int, Any] = {}
    tta_by_seed: dict[int, Any] = {}
    per_seed: list[dict[str, Any]] = []
    result_by_seed = {int(item["seed"]): item for item in receipt["seed_results"]}
    for seed in SEEDS:
        seed_dir = run_root / f"seed_{seed:05d}"
        checkpoint_path = seed_dir / "best_checkpoint.pt"
        model = load_checkpoint_model(checkpoint_path, expected_seed=seed, manifest_sha256=manifest_sha256)
        center_logits = infer_center(model, center_tensors)
        tta_logits = infer_tta(model, images)
        center_by_seed[seed] = center_logits
        tta_by_seed[seed] = tta_logits
        calibration = load_json(seed_dir / "calibration.json")
        temperature = float(calibration["temperature"])
        val_logits = subset_tensor(center_logits, val_indices) / temperature
        val_labels = subset_tensor(labels, val_indices)
        print_logits = subset_tensor(center_logits, print_indices) / temperature
        print_labels = subset_tensor(labels, print_indices)
        val_metrics = metrics_from_logits(val_logits, val_labels, domain=NATURAL_DOMAIN)
        print_metrics = metrics_from_logits(print_logits, print_labels, domain=PRINT_DOMAIN)
        reported_metrics = load_json(seed_dir / "metrics.json")
        val_match = compare_reported_metrics(val_metrics, reported_metrics["natural_validation_raw"])
        print_match = compare_reported_metrics(
            print_metrics, reported_metrics["digital_print_source_holdout_raw"]
        )
        if any(
            evidence.get("acceptance_enabled") is not False
            for evidence in calibration["per_predicted_class_evidence"].values()
        ):
            raise AnalysisError("expected all per-class rejection calibration authorities to fail closed")
        per_seed.append(
            {
                "seed": seed,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "temperature_from_natural_validation": temperature,
                "receipt_metric_reproduction": {
                    "natural_validation": val_match,
                    "digital_print_source_holdout": print_match,
                },
                "natural_validation": {
                    "metrics": val_metrics,
                    "per_image": predictions_for_domain(
                        rows, center_logits, val_indices, temperature=temperature
                    ),
                },
                "digital_print_source_holdout": {
                    "metrics": print_metrics,
                    "per_image": predictions_for_domain(
                        rows, center_logits, print_indices, temperature=temperature
                    ),
                },
                "rejection_gate_result": (
                    "ALL_CLASSES_FORCE_REJECT_FROM_NATURAL_VALIDATION_WILSON_SUPPORT; "
                    "therefore no accepted predictions are claimed"
                ),
                "receipt_best_epoch": int(result_by_seed[seed]["best_epoch"]),
            }
        )
        del model
        gc.collect()

    center_ensemble = torch.stack([center_by_seed[seed] for seed in SEEDS]).mean(dim=0)
    tta_ensemble = torch.stack([tta_by_seed[seed] for seed in SEEDS]).mean(dim=0)
    center_val_metrics = metrics_from_logits(
        subset_tensor(center_ensemble, val_indices), subset_tensor(labels, val_indices), domain=NATURAL_DOMAIN
    )
    center_print_metrics = metrics_from_logits(
        subset_tensor(center_ensemble, print_indices), subset_tensor(labels, print_indices), domain=PRINT_DOMAIN
    )
    tta_val_metrics = metrics_from_logits(
        subset_tensor(tta_ensemble, val_indices), subset_tensor(labels, val_indices), domain=NATURAL_DOMAIN
    )
    tta_print_metrics = metrics_from_logits(
        subset_tensor(tta_ensemble, print_indices), subset_tensor(labels, print_indices), domain=PRINT_DOMAIN
    )
    ranked_candidates, candidates_by_pageid = build_candidate_pool(
        rows, center_by_seed, tta_by_seed, center_ensemble, tta_ensemble
    )
    visual_cards = build_visual_card_selections(candidates_by_pageid)
    young_analysis = build_young_tree_analysis(
        rows, center_by_seed, center_ensemble, tta_ensemble
    )
    script_hash = sha256_file(Path(__file__).resolve())
    return {
        "schema_version": "rootscope.independent_v3_model_quality_analysis.v1",
        "status": "PASS_WITH_SEVERE_SMALL_DATA_AND_DOMAIN_LIMITATIONS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": Path(__file__).resolve().relative_to(workspace).as_posix(),
        "analysis_script_sha256": script_hash,
        "run_relative_path": run_root.relative_to(workspace).as_posix(),
        "pack_relative_path": PACK_RELATIVE,
        "analysis_compute_device": "CPU",
        "no_training_executed": True,
        "run_and_dataset_modified": False,
        "receipt_verification": receipt_verification,
        "manifest_verification": manifest_verification,
        "class_order_for_confusion_truth_rows_prediction_columns": list(CLASS_NAMES),
        "fixed_protocol": {
            "center_crop": "Resize shorter edge to 256, center crop 224, ImageNet normalize",
            "equal_logit_ensemble": (
                "Arithmetic mean of the three uncalibrated raw logit vectors from seeds 17,29,43; "
                "equal weights fixed before inspecting either evaluation domain; no ensemble temperature "
                "or rejection threshold fitted."
            ),
            "digital_print_used_for_weights_or_hyperparameters": False,
            "tta": (
                "EXPLORATORY_ONLY: resize shorter edge to 256, take fixed five 224 crops, add horizontal "
                "flip of each crop (10 views), average raw logits within seed, then equal-logit average seeds."
            ),
            "tta_selected_or_tuned_from_print_results": False,
        },
        "role_semantics": ROLE_SEMANTICS,
        "per_seed_center_crop_recomputation": per_seed,
        "fixed_equal_logit_ensemble_center_crop": {
            "temperature_calibrated": False,
            "rejection_calibrated": False,
            "natural_validation": {
                "metrics": center_val_metrics,
                "per_image": predictions_for_domain(rows, center_ensemble, val_indices),
            },
            "digital_print_source_holdout": {
                "metrics": center_print_metrics,
                "per_image": predictions_for_domain(rows, center_ensemble, print_indices),
            },
        },
        "exploratory_fixed_tta_equal_logit_ensemble": {
            "not_a_selected_model": True,
            "natural_validation": {
                "metrics": tta_val_metrics,
                "per_image": predictions_for_domain(rows, tta_ensemble, val_indices),
            },
            "digital_print_source_holdout": {
                "metrics": tta_print_metrics,
                "per_image": predictions_for_domain(rows, tta_ensemble, print_indices),
            },
        },
        "young_tree_to_low_shrub_analysis": young_analysis,
        "model_stable_candidate_pool_top10_per_class": ranked_candidates,
        "analyst_visually_inspected_stable_candidate_cards": visual_cards,
        "candidate_card_limitation": (
            "Training/validation candidates may be useful for a staged demo card but cannot be cited as "
            "untouched performance.  Only PRINT_DEMO_HOLDOUT_NOT_TRAIN is a digital-source holdout, and "
            "even that is not a physical-print or camera-recapture result."
        ),
        "three_day_main_chain_recommendation": {
            "decision": "PRIORITIZE_SEMANTIC_PLUS_KNOWN_CARD_GEOMETRIC_CONSISTENCY_GATE",
            "preferred_over_immediate_letterbox_v4_as_main_chain": True,
            "architecture": (
                "Keep the frozen ResNet18 as the semantic branch.  Add a CPU known-card branch using "
                "AKAZE and/or ORB descriptors, ratio/cross-check filtering, and RANSAC homography.  A "
                "high-confidence known-card result requires semantic class == reference-card class and "
                "all locked geometry gates to pass; disagreement or weak evidence returns REJECT."
            ),
            "geometry_gates_to_lock_before_demo": [
                "minimum mutual/ratio-filtered correspondence count",
                "minimum RANSAC inlier count and inlier ratio",
                "maximum median reprojection error",
                "valid convex projected quadrilateral with plausible area/aspect ratio",
                "unique best template with sufficient separation from runner-up",
            ],
            "why_this_is_better_for_three_days": (
                "The field demo uses a small, known card set, where instance geometry supplies direct, "
                "interpretable evidence and can reject semantic mistakes without retraining on nine "
                "validation images.  The fixed ensemble and fixed TTA did not repair either young-tree "
                "print-holdout failure, so another classifier training round is a higher-variance main-chain bet."
            ),
            "strict_claim_boundary": (
                "The geometry branch verifies identity of a known printed reference; it is not general "
                "plant recognition.  Open-world images remain semantic-only and should follow the frozen "
                "rejection path."
            ),
            "holdout_warning": (
                "If a current PRINT_DEMO_HOLDOUT_NOT_TRAIN image becomes a matching template, it ceases to "
                "be a holdout for that system.  Relabel it operationally as a demo reference and evaluate "
                "with separate untouched camera recaptures; never report template self-match as holdout accuracy."
            ),
            "physical_commissioning_needed": (
                "Thresholds must be locked from separate development recaptures spanning distance, yaw, "
                "pitch, lighting, blur, and partial occlusion, then checked once on untouched recaptures."
            ),
            "letterbox_v4_position": (
                "Useful as a frozen secondary ablation for whole-plant silhouette preservation, but only "
                "after the hybrid rejection main chain works; do not use print results to select v4."
            ),
        },
        "conclusions": {
            "receipt_report_reproduced": True,
            "fixed_ensemble_is_qualification": False,
            "tta_is_exploratory": True,
            "whole_frame_letterbox_v4_ablation_recommended": True,
            "whole_frame_letterbox_v4_three_day_priority": "SECONDARY_AFTER_HYBRID_GATE",
            "reason": (
                "Young-tree errors align with whole-plant silhouette loss and clutter/pose domain shift. "
                "A frozen whole-frame letterbox ablation is technically justified, but it must select on "
                "natural validation only and preserve the digital print holdout as evaluation-only."
            ),
        },
        "authority": {
            "data_locked": False,
            "dataset_manifest_write": False,
            "human_review": False,
            "model_qualification": False,
            "print_eligibility": False,
            "rights_approval": False,
            "split_assignment": False,
            "training_eligibility": False,
            "visual_truth": False,
            "execution_authority": False,
        },
        "formal_a1_dataset": False,
        "human_reviewed": False,
        "rights_approved": False,
        "training_eligible": False,
        "print_eligible": False,
        "data_locked": False,
        "model_candidate": False,
        "model_qualified": False,
        "physical_print_tested": False,
        "uvc_recapture_evaluated": False,
        "x5_ready": False,
        "bpu_compiled": False,
        "project_hardware_touched_by_analysis": False,
        "network_touched_by_analysis": False,
    }


def write_report(path: Path, report: Mapping[str, Any], *, workspace: Path, run_root: Path) -> None:
    path = path.resolve(strict=False)
    try:
        path.relative_to(workspace)
    except ValueError as error:
        raise AnalysisError("evidence output must remain inside AdventureX") from error
    try:
        path.relative_to(run_root)
    except ValueError:
        pass
    else:
        raise AnalysisError("evidence output must not modify the frozen run")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--run", type=Path, default=Path(RUN_RELATIVE))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/rootscope_v3_model_quality_analysis.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        workspace = args.workspace.resolve(strict=True)
        run_root = args.run if args.run.is_absolute() else workspace / args.run
        run_root = run_root.resolve(strict=True)
        output = args.output if args.output.is_absolute() else workspace / args.output
        report = generate_report(workspace, run_root)
        write_report(output, report, workspace=workspace, run_root=run_root)
    except (AnalysisError, OSError, ValueError, KeyError) as error:
        print(f"FAIL_CLOSED: {error}")
        return 2
    summary = {
        "status": report["status"],
        "output": str(output),
        "ensemble_val_accuracy": report["fixed_equal_logit_ensemble_center_crop"]["natural_validation"]["metrics"]["accuracy"],
        "ensemble_print_accuracy": report["fixed_equal_logit_ensemble_center_crop"]["digital_print_source_holdout"]["metrics"]["accuracy"],
        "tta_val_accuracy": report["exploratory_fixed_tta_equal_logit_ensemble"]["natural_validation"]["metrics"]["accuracy"],
        "tta_print_accuracy": report["exploratory_fixed_tta_equal_logit_ensemble"]["digital_print_source_holdout"]["metrics"]["accuracy"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
