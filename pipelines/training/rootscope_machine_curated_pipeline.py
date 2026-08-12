#!/usr/bin/env python3
"""Fail-closed experimental trainer for the RootScope provisional image pack.

This module deliberately does *not* promote the machine-curated pack to A1,
human-reviewed, rights-approved, print-eligible, data-locked, train-ready, or
model-qualified state.  It permits an isolated experiment only after an
explicit CLI acknowledgement, audits every source byte, preserves print-card
holdouts, and records the limitations in every output receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STATUS = "MACHINE_CURATED_EXPERIMENTAL_ONLY_NOT_HUMAN_REVIEWED_NOT_A1_NOT_DATA_LOCKED"
STATUS_V2 = "MACHINE_CURATED_EXPERIMENTAL_V2_ONLY_NOT_HUMAN_REVIEWED_NOT_A1_NOT_DATA_LOCKED"
STATUS_V3 = "MACHINE_CURATED_EXPERIMENTAL_V3_ONLY_NOT_HUMAN_REVIEWED_NOT_A1_NOT_DATA_LOCKED"
MODEL_STATUS = "MACHINE_CURATED_EXPERIMENTAL_MODEL_NOT_QUALIFIED"
SMOKE_STATUS = "MACHINE_CURATED_SMOKE_ONLY_NOT_MODEL_CANDIDATE"
PACK_NAME_V1 = "rootscope_machine_curated_provisional_v1"
PACK_NAME_V2 = "rootscope_machine_curated_provisional_v2"
PACK_NAME_V3 = "rootscope_machine_curated_provisional_v3"
PACK_NAME = PACK_NAME_V3
PACK_CONTRACTS = {
    PACK_NAME_V1: {
        "status": STATUS,
        "asset_schema": "rootscope.machine_curated_provisional_asset.v1",
        "receipt_schema": "rootscope.machine_curated_provisional_receipt.v1",
        "allowed_source_keys": ("E0", "E1", "E2"),
        "requires_machine_visual_review_evidence": False,
    },
    PACK_NAME_V2: {
        "status": STATUS_V2,
        "asset_schema": "rootscope.machine_curated_provisional_asset.v2",
        "receipt_schema": "rootscope.machine_curated_provisional_receipt.v2",
        "allowed_source_keys": ("E0", "E1", "E2"),
        "requires_machine_visual_review_evidence": False,
    },
    PACK_NAME_V3: {
        "status": STATUS_V3,
        "asset_schema": "rootscope.machine_curated_provisional_asset.v3",
        "receipt_schema": "rootscope.machine_curated_provisional_receipt.v3",
        "allowed_source_keys": ("E0", "E1", "E2", "E3", "E4"),
        "requires_machine_visual_review_evidence": True,
    },
}
CLASS_NAMES = ("grass_clump", "low_shrub", "young_tree", "unknown")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INPUT_SIZE = 224
ONNX_OPSET = 11

TRAIN_ROLE = "EXPERIMENTAL_TRAIN_SUGGESTION"
VAL_ROLE = "EXPERIMENTAL_VAL_SUGGESTION"
PRINT_ROLE = "PRINT_DEMO_HOLDOUT_NOT_TRAIN"
CREATOR_HOLDOUT_ROLE = "CREATOR_GROUP_HOLDOUT_NOT_TRAIN"
ALLOWED_ROLES = {TRAIN_ROLE, VAL_ROLE, PRINT_ROLE, CREATOR_HOLDOUT_ROLE}

FALSE_RECORD_FIELDS = (
    "data_locked",
    "human_reviewed",
    "print_eligible",
    "rights_approved",
    "training_eligible",
)
REQUIRED_FALSE_AUTHORITY = (
    "data_locked",
    "dataset_manifest_write",
    "human_review",
    "model_qualification",
    "print_eligibility",
    "rights_approval",
    "split_assignment",
    "training_eligibility",
    "visual_truth",
)
SOURCE_DATASET_NAMES = {
    "E0": "desert_plants_wikimedia_staging_e0",
    "E1": "desert_plants_whole_plant_reacquisition_e1",
    "E2": "desert_plants_young_tree_reacquisition_e2",
    "E3": "desert_plants_young_tree_reacquisition_e3",
    "E4": "desert_plants_young_tree_category_reacquisition_e4",
}
DHASH_ALGORITHM = "rootscope_rgb_center_sample_9x8_v1"
MAX_CROSS_PARTITION_DHASH_DISTANCE = 4
PRINT_EVAL_DOMAIN = "DIGITAL_PRINT_SOURCE_HOLDOUT_NOT_UVC_RECAPTURE"
NATURAL_VAL_DOMAIN = "NATURAL_WEB_VALIDATION"
LONG_TRAIN_MINIMUMS = {
    "grass_clump": {"train": 6, "validation": 2},
    "low_shrub": {"train": 6, "validation": 2},
    "young_tree": {"train": 5, "validation": 2},
    "unknown": {"train": 15, "validation": 2},
}
LONG_TRAIN_SOURCE_MINIMUMS = {
    "grass_clump": {"train": 6, "validation": 2},
    "low_shrub": {"train": 6, "validation": 2},
    "young_tree": {"train": 5, "validation": 2},
    "unknown": {"train": 15, "validation": 2},
}
LONG_TRAIN_CREATOR_MINIMUMS = {
    "grass_clump": {"train": 5, "validation": 2},
    "low_shrub": {"train": 5, "validation": 2},
    "young_tree": {"train": 5, "validation": 2},
    "unknown": {"train": 15, "validation": 2},
}


class GateError(RuntimeError):
    """A fail-closed contract rejected the requested operation."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    rows: list[str] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        rows.append(f"{path.relative_to(root).as_posix()}\0{sha256_file(path)}\n")
    return sha256_bytes("".join(rows).encode("utf-8"))


def payload_root_sha256(root: Path) -> str:
    """Rebuild the pack payload root created before receipt/SHA256SUMS existed."""

    excluded = {"receipt.json", "SHA256SUMS"}
    rows: list[str] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(f"{relative}\0{sha256_file(path)}\n")
    return sha256_bytes("".join(rows).encode("utf-8"))


def image_dhash64(path: Path) -> str:
    """Recompute the collector's exact RGB centre-sample 9x8 dHash."""

    from PIL import Image

    with Image.open(path) as opened:
        opened.load()
        rgb = opened.convert("RGB")
    if rgb.width < 1 or rgb.height < 1:
        raise GateError(f"invalid image dimensions for dHash: {path}")
    bits: list[str] = []
    for y in range(8):
        source_y = min(rgb.height - 1, int(((y + 0.5) * rgb.height) // 8))
        for x in range(8):
            left_x = min(rgb.width - 1, int(((x + 0.5) * rgb.width) // 9))
            right_x = min(rgb.width - 1, int(((x + 1.5) * rgb.width) // 9))
            left = rgb.getpixel((left_x, source_y))
            right = rgb.getpixel((right_x, source_y))
            left_luma = 299 * left[0] + 587 * left[1] + 114 * left[2]
            right_luma = 299 * right[0] + 587 * right[1] + 114 * right[2]
            bits.append("1" if left_luma > right_luma else "0")
    return f"{int(''.join(bits), 2):016x}"


def dhash_distance(left: str, right: str) -> int:
    try:
        if len(left) != 16 or len(right) != 16:
            raise ValueError("dHash must contain 16 hexadecimal characters")
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as error:
        raise GateError(f"invalid dHash value: {left!r}, {right!r}") from error


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise GateError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise GateError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def safe_child(root: Path, relative_value: str, *, must_exist: bool = True) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise GateError(f"unsafe relative path {relative_value!r}")
    root_resolved = root.resolve(strict=True)
    candidate = (root_resolved / relative).resolve(strict=must_exist)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise GateError(f"path escapes root: {relative_value!r}") from error
    return candidate


def assert_source_record_binding(
    row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    *,
    source_root: Path,
    copied_image_path: Path,
    location: str,
) -> str:
    """Bind pack provenance fields to the source manifest and actual image bytes."""

    for field in ("source_group", "creator_group"):
        if row.get(field) != source_row.get(field):
            raise GateError(f"{location} {field} differs from source manifest")
    source_relative = source_row.get("filename")
    if not isinstance(source_relative, str) or row.get("source_image_path") != source_relative:
        raise GateError(f"{location} source_image_path differs from source manifest filename")
    source_image_path = safe_child(source_root, source_relative)
    source_digest = sha256_file(source_image_path)
    copied_digest = sha256_file(copied_image_path)
    if source_row.get("download_sha256") != source_digest:
        raise GateError(f"{location} source manifest download_sha256 differs from source image bytes")
    if row.get("source_image_sha256") != source_digest:
        raise GateError(f"{location} source_image_sha256 differs from source image bytes")
    if copied_digest != source_digest:
        raise GateError(f"{location} source image bytes differ from copied image")
    if source_row.get("dhash64_algorithm") != DHASH_ALGORITHM:
        raise GateError(f"{location} source manifest dHash algorithm mismatch")
    actual_dhash = image_dhash64(source_image_path)
    if source_row.get("dhash64") != actual_dhash:
        raise GateError(f"{location} source manifest dHash differs from recomputed image dHash")
    if row.get("dhash64") != actual_dhash:
        raise GateError(f"{location} copied record dHash differs from recomputed image dHash")
    return actual_dhash


def require_false_authority(value: object, *, location: str) -> None:
    if not isinstance(value, dict):
        raise GateError(f"{location}.authority must be an object")
    missing = [key for key in REQUIRED_FALSE_AUTHORITY if key not in value]
    if missing:
        raise GateError(f"{location}.authority missing keys: {missing}")
    bad = {key: value.get(key) for key in value if value.get(key) is not False}
    if bad:
        raise GateError(f"{location}.authority contains non-false values: {bad}")


def validate_fail_closed_record(
    row: Mapping[str, Any],
    *,
    location: str,
    expected_status: str = STATUS,
    expected_schema: str = "rootscope.machine_curated_provisional_asset.v1",
) -> None:
    if row.get("status") != expected_status:
        raise GateError(f"{location} has unexpected status {row.get('status')!r}")
    if row.get("schema_version") != expected_schema:
        raise GateError(f"{location} has unexpected asset schema")
    for field in FALSE_RECORD_FIELDS:
        if row.get(field) is not False:
            raise GateError(f"{location}.{field} must be exactly false")
    if row.get("machine_curated_only") is not True:
        raise GateError(f"{location}.machine_curated_only must be exactly true")
    if row.get("experimental_training_switch_required") is not True:
        raise GateError(f"{location}.experimental_training_switch_required must be true")
    if row.get("split") != "UNASSIGNED_DO_NOT_TRAIN":
        raise GateError(f"{location}.split must remain UNASSIGNED_DO_NOT_TRAIN")
    require_false_authority(row.get("authority"), location=location)


def parse_sha256sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise GateError(f"invalid SHA256SUMS row {path}:{line_number}")
        digest, relative = parts
        try:
            int(digest, 16)
        except ValueError as error:
            raise GateError(f"invalid SHA-256 at {path}:{line_number}") from error
        if relative in result:
            raise GateError(f"duplicate SHA256SUMS path {relative!r}")
        result[relative] = digest.lower()
    if not result:
        raise GateError("SHA256SUMS is empty")
    return result


def _role_family(role: str) -> str:
    if role == TRAIN_ROLE:
        return "train"
    if role == VAL_ROLE:
        return "natural_validation"
    if role in {PRINT_ROLE, CREATOR_HOLDOUT_ROLE}:
        return "print_isolation"
    raise GateError(f"unknown experimental role {role!r}")


def assert_group_partition_isolation(rows: Sequence[Mapping[str, Any]]) -> None:
    for key in ("source_group", "copied_image_sha256"):
        owners: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            owners[str(row[key])].add(_role_family(str(row["experimental_split_suggestion"])))
        leakage = {value: sorted(families) for value, families in owners.items() if len(families) > 1}
        if leakage:
            raise GateError(f"{key} crosses experimental partitions: {leakage}")

    creator_owners: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        creator_owners[str(row["creator_group"])].add(
            _role_family(str(row["experimental_split_suggestion"]))
        )
    creator_leakage = {
        creator: sorted(families)
        for creator, families in creator_owners.items()
        if len(families) > 1
    }
    if creator_leakage:
        raise GateError(f"creator_group crosses experimental partitions: {creator_leakage}")


def assert_dhash_partition_isolation(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject perceptually near-identical images across train/val/print families."""

    for left_index, left in enumerate(rows):
        left_family = _role_family(str(left["experimental_split_suggestion"]))
        for right in rows[left_index + 1 :]:
            right_family = _role_family(str(right["experimental_split_suggestion"]))
            if left_family == right_family:
                continue
            distance = dhash_distance(str(left.get("dhash64", "")), str(right.get("dhash64", "")))
            if distance <= MAX_CROSS_PARTITION_DHASH_DISTANCE:
                raise GateError(
                    "dHash near-duplicate crosses experimental partitions: "
                    f"{left.get('asset')} ({left_family}) vs {right.get('asset')} "
                    f"({right_family}), distance={distance} <= {MAX_CROSS_PARTITION_DHASH_DISTANCE}"
                )


@dataclass(frozen=True)
class AuditedPack:
    workspace: Path
    root: Path
    rows: tuple[dict[str, Any], ...]
    train_rows: tuple[dict[str, Any], ...]
    validation_rows: tuple[dict[str, Any], ...]
    print_rows: tuple[dict[str, Any], ...]
    creator_holdout_rows: tuple[dict[str, Any], ...]
    status: str
    receipt: dict[str, Any]
    audit: dict[str, Any]
    immutable_snapshot: dict[str, str]


def _index_source_manifest(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        pageid = row.get("pageid")
        if isinstance(pageid, bool) or not isinstance(pageid, int):
            raise GateError(f"source manifest has invalid pageid: {pageid!r}")
        if pageid in result:
            raise GateError(f"source manifest has duplicate pageid {pageid}: {path}")
        result[pageid] = row
    return result


def verify_frozen_v1_reference(
    workspace: Path, receipt: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    frozen = receipt.get("frozen_v1")
    if frozen is None:
        return {}, {}
    if not isinstance(frozen, dict) or frozen.get("unchanged") is not True:
        raise GateError("receipt frozen_v1 reference is missing unchanged=true")
    root = safe_child(workspace, str(frozen.get("path", "")))
    expected_pairs = (
        ("tree_sha256_before", "tree_sha256_after"),
        ("manifest_sha256_before", "manifest_sha256_after"),
        ("sha256sums_sha256_before", "sha256sums_sha256_after"),
    )
    for before, after in expected_pairs:
        if frozen.get(before) != frozen.get(after):
            raise GateError(f"receipt frozen_v1 changed between {before} and {after}")
    current = {
        "tree_sha256": tree_sha256(root),
        "manifest_sha256": sha256_file(root / "manifest.jsonl"),
        "sha256sums_sha256": sha256_file(root / "SHA256SUMS"),
    }
    for key, actual in current.items():
        if frozen.get(f"{key}_after") != actual:
            raise GateError(f"current frozen_v1 {key} differs from receipt")
    return current, _index_source_manifest(root / "manifest.jsonl")


def verify_frozen_v2_reference(
    workspace: Path, receipt: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = receipt.get("frozen_v2")
    if frozen is None:
        return {}, {}
    if not isinstance(frozen, dict) or frozen.get("unchanged") is not True:
        raise GateError("receipt frozen_v2 reference is missing unchanged=true")
    root = safe_child(workspace, str(frozen.get("path", "")))
    if frozen.get("tree_sha256_before") != frozen.get("tree_sha256_after"):
        raise GateError("receipt frozen_v2 tree changed during v3 build")
    current = {
        "tree_sha256": tree_sha256(root),
        "manifest_sha256": sha256_file(root / "manifest.jsonl"),
        "receipt_sha256": sha256_file(root / "receipt.json"),
        "sha256sums_sha256": sha256_file(root / "SHA256SUMS"),
    }
    for key, actual in current.items():
        receipt_key = "tree_sha256_after" if key == "tree_sha256" else key
        if frozen.get(receipt_key) != actual:
            raise GateError(f"current frozen_v2 {key} differs from receipt")
    audit_path = frozen.get("independent_audit_path")
    if not isinstance(audit_path, str):
        raise GateError("receipt frozen_v2 lacks independent audit path")
    audit = safe_child(workspace, audit_path)
    if sha256_file(audit) != frozen.get("independent_audit_sha256"):
        raise GateError("current frozen_v2 independent audit differs from receipt")
    current["independent_audit_sha256"] = sha256_file(audit)
    return current, load_json(root / "receipt.json")


def verify_protected_inputs(
    workspace: Path,
    receipt: Mapping[str, Any],
    *,
    frozen_v2_receipt: Mapping[str, Any],
) -> tuple[dict[str, str], Path | None]:
    protected = receipt.get("protected_inputs")
    if protected is None:
        return {}, None
    if not isinstance(protected, dict) or protected.get("unchanged") is not True:
        raise GateError("receipt protected_inputs is not unchanged")
    before = protected.get("before")
    after = protected.get("after")
    if not isinstance(before, dict) or before != after:
        raise GateError("receipt protected_inputs before/after mismatch")
    formal = frozen_v2_receipt.get("formal_human_decisions")
    if not isinstance(formal, dict):
        raise GateError("frozen_v2 receipt lacks formal human-decision evidence")
    human_root = safe_child(workspace, str(formal.get("path", "")))
    journal = human_root / "decision_journal.jsonl"
    current: dict[str, str] = {
        "formal_human_decisions_tree_sha256": tree_sha256(human_root),
        "formal_decision_journal_sha256": sha256_file(journal),
    }
    for key in tuple(current):
        if before.get(key) != current[key]:
            raise GateError(f"receipt protected_inputs {key} differs from current evidence")
    protected_trees = before.get("protected_tree_sha256")
    if not isinstance(protected_trees, dict) or not protected_trees:
        raise GateError("receipt protected_inputs lacks protected tree hashes")
    for relative, expected in protected_trees.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise GateError("receipt protected_inputs tree entry is invalid")
        actual = tree_sha256(safe_child(workspace, relative))
        if actual != expected:
            raise GateError(f"protected input tree changed: {relative}")
        current[f"tree:{relative}"] = actual
    return current, human_root


def validate_source_decision_semantics(
    decisions: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_status: str,
    frozen_v1_rows: Mapping[int, Mapping[str, Any]] | None = None,
) -> None:
    if not decisions:
        raise GateError("source decision manifest is empty")
    manifest_by_pageid = {int(row["pageid"]): row for row in rows}
    selected_by_pageid: dict[int, Mapping[str, Any]] = {}
    seen_pageids: set[int] = set()
    for index, decision in enumerate(decisions):
        location = f"source_decision[{index}]"
        pageid = decision.get("pageid")
        if isinstance(pageid, bool) or not isinstance(pageid, int) or pageid in seen_pageids:
            raise GateError(f"{location} has invalid/duplicate pageid {pageid!r}")
        seen_pageids.add(pageid)
        if decision.get("status") != expected_status:
            raise GateError(f"{location} status differs from pack status")
        for field in (
            "data_locked",
            "human_reviewed",
            "print_eligible",
            "rights_approved",
            "training_eligible",
        ):
            if decision.get(field) is not False:
                raise GateError(f"{location}.{field} must be exactly false")
        if decision.get("machine_curated_only") is not True:
            raise GateError(f"{location}.machine_curated_only must be true")
        if decision.get("experimental_training_switch_required") is not True:
            raise GateError(f"{location}.experimental_training_switch_required must be true")
        if decision.get("split") != "UNASSIGNED_DO_NOT_TRAIN":
            raise GateError(f"{location}.split must remain UNASSIGNED_DO_NOT_TRAIN")
        require_false_authority(decision.get("authority"), location=location)
        selected = decision.get("selected")
        if not isinstance(selected, bool):
            raise GateError(f"{location}.selected must be boolean")
        disposition = decision.get("disposition")
        if not isinstance(disposition, str) or not disposition:
            raise GateError(f"{location}.disposition must be non-empty")
        selection_disposition = disposition.startswith("SELECTED_") or disposition.startswith(
            "INHERITED_"
        )
        if selected is not selection_disposition:
            raise GateError(f"{location} selected/disposition semantics disagree")
        if not selected:
            continue
        row = manifest_by_pageid.get(pageid)
        if row is None:
            raise GateError(f"{location} selects pageid absent from manifest")
        selected_by_pageid[pageid] = decision
        for field in ("source_group", "creator_group"):
            if decision.get(field) != row.get(field):
                raise GateError(f"{location} {field} differs from selected manifest row")
        if "source_dataset" in decision and decision.get("source_dataset") != row.get("source_dataset"):
            raise GateError(f"{location} source_dataset differs from selected manifest row")
        for class_field in ("candidate_class", "requested_class"):
            if class_field in decision and decision.get(class_field) != row.get("class_id"):
                raise GateError(f"{location} {class_field} differs from selected class_id")
        if "class_id" in decision and decision.get("class_id") != row.get("class_id"):
            raise GateError(f"{location} class_id differs from selected manifest row")
        for field in ("copied_image_sha256", "experimental_split_suggestion"):
            if field in decision and decision.get(field) != row.get(field):
                raise GateError(f"{location} {field} differs from selected manifest row")
        if "print_holdout_candidate" in decision:
            if decision.get("print_holdout_candidate") is not row.get("print_holdout_candidate"):
                raise GateError(f"{location} print-holdout semantics disagree")
        if "source_record_sha256" in decision:
            expected_source_record = row.get("source_record_sha256")
            if disposition == "INHERITED_FROZEN_V1":
                inherited = (frozen_v1_rows or {}).get(pageid)
                if inherited is None:
                    raise GateError(f"{location} lacks frozen_v1 source record")
                expected_source_record = sha256_bytes(canonical_json(inherited).encode("utf-8"))
            if decision.get("source_record_sha256") != expected_source_record:
                raise GateError(f"{location} source_record_sha256 semantics disagree")
    if set(selected_by_pageid) != set(manifest_by_pageid):
        mismatch = sorted(set(selected_by_pageid) ^ set(manifest_by_pageid))
        raise GateError(f"selected source decisions do not exactly cover manifest: {mismatch[:8]}")


def validate_machine_visual_review_evidence(
    receipt: Mapping[str, Any],
    pack_root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    required: bool,
) -> dict[str, Any]:
    if not required:
        return {
            "required_for_pack": False,
            "bound": False,
            "non_smoke_training_evidence_eligible": False,
        }
    relative = receipt.get("machine_visual_review_evidence_path")
    if relative != "machine_visual_review_evidence.json":
        raise GateError("receipt machine_visual_review_evidence_path must bind the canonical file")
    path = safe_child(pack_root, str(relative))
    digest = sha256_file(path)
    if receipt.get("machine_visual_review_evidence_sha256") != digest:
        raise GateError("receipt machine_visual_review_evidence_sha256 mismatch")
    evidence = load_json(path)
    if evidence.get("human_reviewed") is not False:
        raise GateError("machine visual review evidence must state human_reviewed=false")
    if evidence.get("dual_machine_review_completed") is not True:
        raise GateError("machine visual review evidence lacks dual-machine review completion")
    if evidence.get("dual_machine_review_scope") != "E4_SELECTED_ONLY":
        raise GateError("dual-machine review evidence must remain scoped to E4 selected records")
    if evidence.get("root_machine_adjudicated") is not True:
        raise GateError("machine visual review evidence lacks root machine adjudication")
    if evidence.get("root_machine_adjudication_scope") != "E4_SELECTED_ONLY":
        raise GateError("root machine adjudication evidence must remain scoped to E4 selected records")
    if evidence.get("all_selected_records_machine_screened") is not True:
        raise GateError("machine visual review evidence does not screen every selected record")
    if "authority" in evidence:
        require_false_authority(evidence.get("authority"), location="machine_visual_review_evidence")
    new_rows = {
        int(row["pageid"]): row for row in rows if row.get("source_dataset") in {"E3", "E4"}
    }
    selected_pageids = evidence.get("selected_pageids")
    if not isinstance(selected_pageids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in selected_pageids
    ):
        raise GateError("machine visual review evidence selected_pageids must be an integer list")
    if len(selected_pageids) != len(set(selected_pageids)) or set(selected_pageids) != set(new_rows):
        raise GateError("machine visual review evidence selected_pageids do not bind E3/E4 manifest rows")
    selected_records = evidence.get("selected_records")
    if not isinstance(selected_records, list) or len(selected_records) != len(new_rows):
        raise GateError("machine visual review evidence must bind every selected ID to its role")
    seen: set[int] = set()
    for index, record in enumerate(selected_records):
        if not isinstance(record, dict):
            raise GateError(f"machine visual review selected_records[{index}] must be an object")
        pageid = record.get("pageid")
        if isinstance(pageid, bool) or not isinstance(pageid, int) or pageid in seen:
            raise GateError(f"machine visual review selected_records[{index}] pageid is invalid")
        seen.add(pageid)
        row = new_rows.get(pageid)
        if row is None:
            raise GateError(f"machine visual review selected_records[{index}] is absent from manifest")
        for field in ("source_dataset", "class_id", "experimental_split_suggestion"):
            if record.get(field) != row.get(field):
                raise GateError(
                    f"machine visual review selected_records[{index}] {field} differs from manifest"
                )
        if row.get("source_dataset") == "E3":
            expected_review = (False, False, "E3_MACHINE_SCREEN_ONLY")
        else:
            expected_review = (
                True,
                True,
                "E4_DUAL_MACHINE_REVIEW_ROOT_ADJUDICATION",
            )
        actual_review = (
            record.get("dual_machine_reviewed"),
            record.get("root_machine_adjudicated"),
            record.get("review_scope"),
        )
        if actual_review != expected_review:
            raise GateError(
                f"machine visual review selected_records[{index}] review scope is overstated or invalid"
            )
    return {
        "required_for_pack": True,
        "bound": True,
        "path": str(relative),
        "sha256": digest,
        "selected_record_count": len(new_rows),
        "human_reviewed": False,
        "dual_machine_review_completed": True,
        "dual_machine_review_scope": "E4_SELECTED_ONLY",
        "root_machine_adjudicated": True,
        "root_machine_adjudication_scope": "E4_SELECTED_ONLY",
        "all_selected_records_machine_screened": True,
        "non_smoke_training_evidence_eligible": True,
    }


def audit_pack(workspace: Path, pack_root: Path) -> AuditedPack:
    workspace = workspace.resolve(strict=True)
    pack_root = pack_root.resolve(strict=True)
    try:
        pack_root.relative_to(workspace)
    except ValueError as error:
        raise GateError(f"pack must stay inside AdventureX workspace: {pack_root}") from error
    contract = PACK_CONTRACTS.get(pack_root.name)
    if contract is None:
        raise GateError(
            f"unexpected pack directory name {pack_root.name!r}; supported={sorted(PACK_CONTRACTS)}"
        )
    expected_status = str(contract["status"])
    allowed_source_keys = tuple(str(value) for value in contract["allowed_source_keys"])

    required_files = [
        "manifest.jsonl",
        "source_decision_manifest.jsonl",
        "experimental_split_suggestion.json",
        "receipt.json",
        "SHA256SUMS",
    ]
    if contract["requires_machine_visual_review_evidence"]:
        required_files.append("machine_visual_review_evidence.json")
    for relative in required_files:
        if not (pack_root / relative).is_file():
            raise GateError(f"missing required pack file: {relative}")

    receipt = load_json(pack_root / "receipt.json")
    if receipt.get("schema_version") != contract["receipt_schema"]:
        raise GateError("receipt schema does not match the selected pack version")
    if receipt.get("status") != expected_status:
        raise GateError("receipt status is not the frozen machine-curated experimental status")
    if receipt.get("formal_a1_dataset") is not False:
        raise GateError("receipt.formal_a1_dataset must be false")
    for field in ("data_locked", "human_reviewed", "print_eligible", "training_eligible"):
        if receipt.get(field) is not False:
            raise GateError(f"receipt.{field} must be exactly false")
    if "rights_approved" in receipt and receipt.get("rights_approved") is not False:
        raise GateError("receipt.rights_approved must be exactly false")
    if receipt.get("experimental_training_switch_required") is not True:
        raise GateError("receipt must require the experimental training switch")
    require_false_authority(receipt.get("authority"), location="receipt")

    sums = parse_sha256sums(pack_root / "SHA256SUMS")
    sums_required = {
        "manifest.jsonl",
        "receipt.json",
        "source_decision_manifest.jsonl",
        "experimental_split_suggestion.json",
    }
    if contract["requires_machine_visual_review_evidence"]:
        sums_required.add("machine_visual_review_evidence.json")
    for required in sorted(sums_required):
        if required not in sums:
            raise GateError(f"SHA256SUMS does not cover {required}")
    for relative, expected_digest in sums.items():
        file_path = safe_child(pack_root, relative)
        if not file_path.is_file():
            raise GateError(f"SHA256SUMS entry is not a file: {relative}")
        actual = sha256_file(file_path)
        if actual != expected_digest:
            raise GateError(f"pack SHA mismatch for {relative}: {actual} != {expected_digest}")

    manifest_path = pack_root / "manifest.jsonl"
    if receipt.get("manifest_sha256") != sha256_file(manifest_path):
        raise GateError("receipt manifest SHA does not match manifest.jsonl")
    decisions_path = pack_root / "source_decision_manifest.jsonl"
    if receipt.get("source_decision_manifest_sha256") != sha256_file(decisions_path):
        raise GateError("receipt source-decision SHA does not match")
    actual_payload_root = payload_root_sha256(pack_root)
    if receipt.get("payload_root_sha256_before_receipt") != actual_payload_root:
        raise GateError("receipt payload_root_sha256_before_receipt does not match pack payload")
    frozen_v1_snapshot, frozen_v1_rows = verify_frozen_v1_reference(workspace, receipt)
    frozen_v2_snapshot, frozen_v2_receipt = verify_frozen_v2_reference(workspace, receipt)
    protected_input_snapshot, protected_human_root = verify_protected_inputs(
        workspace,
        receipt,
        frozen_v2_receipt=frozen_v2_receipt,
    )

    split = load_json(pack_root / "experimental_split_suggestion.json")
    if split.get("status") != expected_status or split.get("formal_split_assignment") is not False:
        raise GateError("split suggestion attempts to claim formal assignment")
    if split.get("experimental_training_switch_required") is not True:
        raise GateError("split suggestion does not require experimental opt-in")
    require_false_authority(split.get("authority"), location="experimental_split_suggestion")
    roles = split.get("roles")
    role_items: dict[str, list[dict[str, Any]]]
    if isinstance(roles, dict):
        if set(roles) != ALLOWED_ROLES:
            raise GateError(f"split role set mismatch: {set(roles)}")
        role_items = {role: list(items) for role, items in roles.items()}
    else:
        flat_records = split.get("records")
        if not isinstance(flat_records, list):
            raise GateError("split suggestion must provide either roles or records")
        role_items = {role: [] for role in ALLOWED_ROLES}
        for item in flat_records:
            if not isinstance(item, dict) or item.get("role") not in ALLOWED_ROLES:
                raise GateError("v2 split suggestion contains an invalid role record")
            role_items[str(item["role"])].append(item)

    rows = load_jsonl(manifest_path)
    if not rows:
        raise GateError("machine-curated manifest is empty")
    split_assets: dict[str, str] = {}
    for role, items in role_items.items():
        if not isinstance(items, list):
            raise GateError(f"split role {role} must contain a list")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("asset"), str):
                raise GateError(f"invalid split item in role {role}")
            asset = item["asset"]
            if asset in split_assets:
                raise GateError(f"asset appears in multiple split roles: {asset}")
            split_assets[asset] = role

    source_indexes: dict[str, dict[int, dict[str, Any]]] = {}
    source_manifest_hashes: dict[str, str] = {}
    receipt_source_hashes = receipt.get("source_manifest_sha256")
    if receipt_source_hashes is not None and not isinstance(receipt_source_hashes, dict):
        raise GateError("receipt.source_manifest_sha256 must be an object when present")
    if isinstance(receipt_source_hashes, dict) and set(receipt_source_hashes) != set(
        allowed_source_keys
    ):
        raise GateError("receipt.source_manifest_sha256 keys differ from pack source contract")
    for source_key in allowed_source_keys:
        dataset_name = SOURCE_DATASET_NAMES[source_key]
        source_manifest = workspace / "datasets" / dataset_name / "manifest.jsonl"
        if not source_manifest.is_file():
            raise GateError(f"missing source manifest for {source_key}: {source_manifest}")
        digest = sha256_file(source_manifest)
        if isinstance(receipt_source_hashes, dict) and receipt_source_hashes.get(source_key) != digest:
            raise GateError(f"receipt source manifest SHA mismatch for {source_key}")
        if source_key == "E0" and receipt.get("source_e0_manifest_sha256") is not None:
            if receipt.get("source_e0_manifest_sha256") != digest:
                raise GateError("receipt source_e0_manifest_sha256 mismatch")
        source_manifest_hashes[source_key] = digest
        source_indexes[source_key] = _index_source_manifest(source_manifest)

    seen_assets: set[str] = set()
    seen_pageids: set[int] = set()
    seen_paths: set[str] = set()
    class_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        location = f"manifest[{index}]"
        validate_fail_closed_record(
            row,
            location=location,
            expected_status=expected_status,
            expected_schema=str(contract["asset_schema"]),
        )
        class_id = row.get("class_id")
        if class_id not in CLASS_TO_INDEX:
            raise GateError(f"{location} has unexpected class_id {class_id!r}")
        role = row.get("experimental_split_suggestion")
        if role not in ALLOWED_ROLES:
            raise GateError(f"{location} has unexpected experimental role {role!r}")
        asset = row.get("asset")
        if not isinstance(asset, str) or asset in seen_assets:
            raise GateError(f"{location} has missing/duplicate asset")
        seen_assets.add(asset)
        if split_assets.get(asset) != role:
            raise GateError(f"{location} role disagrees with split suggestion")
        pageid = row.get("pageid")
        if isinstance(pageid, bool) or not isinstance(pageid, int) or pageid in seen_pageids:
            raise GateError(f"{location} has invalid/duplicate pageid {pageid!r}")
        seen_pageids.add(pageid)
        filename = row.get("filename")
        if not isinstance(filename, str) or filename in seen_paths:
            raise GateError(f"{location} has invalid/duplicate filename")
        seen_paths.add(filename)
        image_path = safe_child(pack_root, filename)
        if not image_path.is_file():
            raise GateError(f"{location} image is not a file")
        image_digest = sha256_file(image_path)
        for field in ("copied_image_sha256", "source_image_sha256"):
            if row.get(field) != image_digest:
                raise GateError(f"{location}.{field} does not match image bytes")
        if not asset.endswith(f"@sha256:{image_digest}"):
            raise GateError(f"{location}.asset is not bound to copied image SHA")
        if sums.get(filename) != image_digest:
            raise GateError(f"SHA256SUMS does not bind manifest image {filename}")
        if row.get("print_holdout_candidate") is True and role != PRINT_ROLE:
            raise GateError(f"print holdout candidate {asset} is not isolated")
        if role == PRINT_ROLE and row.get("print_holdout_candidate") is not True:
            raise GateError(f"print role asset {asset} lacks print_holdout_candidate=true")

        source_key = row.get("source_dataset")
        if source_key not in allowed_source_keys:
            raise GateError(f"{location} has unknown source_dataset {source_key!r}")
        if row.get("source_dataset_name") != SOURCE_DATASET_NAMES[source_key]:
            raise GateError(f"{location} source_dataset_name mismatch")
        if row.get("source_manifest_sha256") != source_manifest_hashes[source_key]:
            raise GateError(f"{location} source_manifest_sha256 mismatch")
        source_row = source_indexes[source_key].get(pageid)
        if source_row is None:
            raise GateError(f"{location} pageid is absent from source manifest")
        source_record_digest = sha256_bytes(canonical_json(source_row).encode("utf-8"))
        if row.get("source_record_sha256") != source_record_digest:
            raise GateError(f"{location} source_record_sha256 mismatch")
        source_root = workspace / "datasets" / SOURCE_DATASET_NAMES[source_key]
        assert_source_record_binding(
            row,
            source_row,
            source_root=source_root,
            copied_image_path=image_path,
            location=location,
        )

        try:
            from PIL import Image

            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
                if width < 32 or height < 32:
                    raise GateError(f"{location} image is too small: {width}x{height}")
                dimension_counts[f"{width}x{height}"] += 1
        except GateError:
            raise
        except Exception as error:
            raise GateError(f"{location} image decode failed: {error}") from error
        class_counts[str(class_id)] += 1
        role_counts[str(role)] += 1

    if set(split_assets) != seen_assets:
        missing = sorted(set(split_assets) ^ seen_assets)
        raise GateError(f"manifest/split asset set mismatch: {missing[:8]}")
    assert_group_partition_isolation(rows)
    assert_dhash_partition_isolation(rows)
    decisions = load_jsonl(decisions_path)
    validate_source_decision_semantics(
        decisions,
        rows,
        expected_status=expected_status,
        frozen_v1_rows=frozen_v1_rows,
    )
    visual_review_evidence = validate_machine_visual_review_evidence(
        receipt,
        pack_root,
        rows,
        required=bool(contract["requires_machine_visual_review_evidence"]),
    )

    by_role = {
        role: tuple(row for row in rows if row["experimental_split_suggestion"] == role)
        for role in ALLOWED_ROLES
    }
    train_counts = Counter(str(row["class_id"]) for row in by_role[TRAIN_ROLE])
    missing_train = [name for name in CLASS_NAMES if train_counts[name] == 0]
    if missing_train:
        raise GateError(f"experimental train suggestion lacks classes: {missing_train}")
    if not by_role[VAL_ROLE]:
        raise GateError("natural validation suggestion is empty")
    if not by_role[PRINT_ROLE]:
        raise GateError("digital print-source holdout is empty")

    formal = receipt.get("formal_human_decisions")
    if isinstance(formal, dict):
        if formal.get("unchanged") is not True:
            raise GateError("receipt does not prove formal human decisions were unchanged")
        if formal.get("decision_journal_unchanged") not in (None, True):
            raise GateError("receipt says the formal human decision journal changed")
        if formal.get("tree_sha256_before") != formal.get("tree_sha256_after"):
            raise GateError("receipt formal human decision tree changed during pack build")
        if formal.get("decision_journal_sha256_before") != formal.get(
            "decision_journal_sha256_after"
        ):
            raise GateError("receipt formal human decision journal changed during pack build")
        human_root = safe_child(workspace, str(formal.get("path", "")))
        if tree_sha256(human_root) != formal.get("tree_sha256_after"):
            raise GateError("current formal human decision tree differs from pack receipt")
        journal = human_root / "decision_journal.jsonl"
        if sha256_file(journal) != formal.get("decision_journal_sha256_after"):
            raise GateError("current formal human decision journal differs from pack receipt")
    elif protected_human_root is not None:
        human_root = protected_human_root
        journal = human_root / "decision_journal.jsonl"
    else:
        raise GateError("receipt lacks formal human-decision preservation evidence")

    immutable_snapshot = {
        "pack_tree_sha256": tree_sha256(pack_root),
        "manifest_sha256": sha256_file(manifest_path),
        "split_suggestion_sha256": sha256_file(pack_root / "experimental_split_suggestion.json"),
        "receipt_sha256": sha256_file(pack_root / "receipt.json"),
        "payload_root_sha256_before_receipt": actual_payload_root,
        "formal_human_decisions_tree_sha256": tree_sha256(human_root),
        "formal_human_decision_journal_sha256": sha256_file(journal),
    }
    immutable_snapshot.update(
        {f"source_manifest_{key}_sha256": digest for key, digest in source_manifest_hashes.items()}
    )
    immutable_snapshot.update(
        {f"frozen_v1_{key}": value for key, value in frozen_v1_snapshot.items()}
    )
    immutable_snapshot.update(
        {f"frozen_v2_{key}": value for key, value in frozen_v2_snapshot.items()}
    )
    immutable_snapshot.update(
        {f"protected_input_{key}": value for key, value in protected_input_snapshot.items()}
    )
    val_counts = Counter(str(row["class_id"]) for row in by_role[VAL_ROLE])
    print_counts = Counter(str(row["class_id"]) for row in by_role[PRINT_ROLE])
    train_source_counts = {
        name: len(
            {str(row["source_group"]) for row in by_role[TRAIN_ROLE] if row["class_id"] == name}
        )
        for name in CLASS_NAMES
    }
    val_source_counts = {
        name: len(
            {str(row["source_group"]) for row in by_role[VAL_ROLE] if row["class_id"] == name}
        )
        for name in CLASS_NAMES
    }
    train_creator_counts = {
        name: len(
            {str(row["creator_group"]) for row in by_role[TRAIN_ROLE] if row["class_id"] == name}
        )
        for name in CLASS_NAMES
    }
    val_creator_counts = {
        name: len(
            {str(row["creator_group"]) for row in by_role[VAL_ROLE] if row["class_id"] == name}
        )
        for name in CLASS_NAMES
    }
    long_coverage = {
        name: {
            "train_count": train_counts[name],
            "train_minimum": LONG_TRAIN_MINIMUMS[name]["train"],
            "train_met": train_counts[name] >= LONG_TRAIN_MINIMUMS[name]["train"],
            "train_unique_source_count": train_source_counts[name],
            "train_unique_source_minimum": LONG_TRAIN_SOURCE_MINIMUMS[name]["train"],
            "train_unique_source_met": train_source_counts[name]
            >= LONG_TRAIN_SOURCE_MINIMUMS[name]["train"],
            "train_unique_creator_count": train_creator_counts[name],
            "train_unique_creator_minimum": LONG_TRAIN_CREATOR_MINIMUMS[name]["train"],
            "train_unique_creator_met": train_creator_counts[name]
            >= LONG_TRAIN_CREATOR_MINIMUMS[name]["train"],
            "validation_count": val_counts[name],
            "validation_minimum": LONG_TRAIN_MINIMUMS[name]["validation"],
            "validation_met": val_counts[name] >= LONG_TRAIN_MINIMUMS[name]["validation"],
            "validation_unique_source_count": val_source_counts[name],
            "validation_unique_source_minimum": LONG_TRAIN_SOURCE_MINIMUMS[name]["validation"],
            "validation_unique_source_met": val_source_counts[name]
            >= LONG_TRAIN_SOURCE_MINIMUMS[name]["validation"],
            "validation_unique_creator_count": val_creator_counts[name],
            "validation_unique_creator_minimum": LONG_TRAIN_CREATOR_MINIMUMS[name]["validation"],
            "validation_unique_creator_met": val_creator_counts[name]
            >= LONG_TRAIN_CREATOR_MINIMUMS[name]["validation"],
        }
        for name in CLASS_NAMES
    }
    long_coverage_passed = all(
        item["train_met"]
        and item["train_unique_source_met"]
        and item["train_unique_creator_met"]
        and item["validation_met"]
        and item["validation_unique_source_met"]
        and item["validation_unique_creator_met"]
        for item in long_coverage.values()
    )
    if "all_split_targets_met" in receipt and receipt.get("all_split_targets_met") is not long_coverage_passed:
        raise GateError("receipt all_split_targets_met disagrees with independently recomputed coverage")
    audit = {
        "schema_version": "rootscope.machine_curated_training_input_audit.v1",
        "status": expected_status,
        "formal_a1_dataset": False,
        "human_reviewed": False,
        "data_locked": False,
        "training_eligible": False,
        "class_order": list(CLASS_NAMES),
        "record_count": len(rows),
        "class_counts": {name: class_counts[name] for name in CLASS_NAMES},
        "role_counts": {role: role_counts[role] for role in sorted(ALLOWED_ROLES)},
        "train_class_counts": {name: train_counts[name] for name in CLASS_NAMES},
        "natural_validation_class_counts": {name: val_counts[name] for name in CLASS_NAMES},
        "digital_print_source_holdout_class_counts": {name: print_counts[name] for name in CLASS_NAMES},
        "natural_validation_all_classes_present": all(val_counts[name] > 0 for name in CLASS_NAMES),
        "digital_print_source_holdout_all_classes_present": all(
            print_counts[name] > 0 for name in CLASS_NAMES
        ),
        "long_training_coverage": long_coverage,
        "long_training_coverage_gate_passed": long_coverage_passed,
        "machine_visual_review_evidence": visual_review_evidence,
        "print_evaluation_domain": PRINT_EVAL_DOMAIN,
        "print_evaluation_is_uvc_recapture": False,
        "source_group_partition_leakage_count": 0,
        "creator_group_partition_leakage_count": 0,
        "content_sha_partition_leakage_count": 0,
        "cross_partition_dhash_distance_threshold": MAX_CROSS_PARTITION_DHASH_DISTANCE,
        "cross_partition_dhash_near_duplicate_count": 0,
        "image_decode_failures": 0,
        "unique_image_dimension_count": len(dimension_counts),
        "immutable_snapshot": immutable_snapshot,
        "warnings": [
            "This pack is machine-curated experimental evidence, not human-reviewed ground truth.",
            "The print holdout contains digital source files only; it is not a UVC recapture test.",
            "Missing classes in a read-only evaluation partition produce N/A per-class metrics and block qualification claims.",
        ],
    }
    return AuditedPack(
        workspace=workspace,
        root=pack_root,
        rows=tuple(rows),
        train_rows=by_role[TRAIN_ROLE],
        validation_rows=by_role[VAL_ROLE],
        print_rows=by_role[PRINT_ROLE],
        creator_holdout_rows=by_role[CREATOR_HOLDOUT_ROLE],
        status=expected_status,
        receipt=receipt,
        audit=audit,
        immutable_snapshot=immutable_snapshot,
    )


def set_determinism(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class RandomColorTemperature:
    """Small red/blue channel gain, simulating indoor/field white-balance drift."""

    def __init__(self, strength: float = 0.10, probability: float = 0.45) -> None:
        self.strength = float(strength)
        self.probability = float(probability)

    def __call__(self, image: Any) -> Any:
        import torch
        from PIL import Image

        if float(torch.rand(())) >= self.probability:
            return image
        delta = (float(torch.rand(())) * 2.0 - 1.0) * self.strength
        red, green, blue = image.convert("RGB").split()
        red = red.point(lambda value: max(0, min(255, round(value * (1.0 + delta)))))
        blue = blue.point(lambda value: max(0, min(255, round(value * (1.0 - delta)))))
        return Image.merge("RGB", (red, green, blue))


class RandomJpegRoundTrip:
    """In-memory JPEG artifact simulation; never writes augmented samples to disk."""

    def __init__(self, probability: float = 0.35, quality_min: int = 58, quality_max: int = 94) -> None:
        self.probability = float(probability)
        self.quality_min = int(quality_min)
        self.quality_max = int(quality_max)

    def __call__(self, image: Any) -> Any:
        import torch
        from PIL import Image

        if float(torch.rand(())) >= self.probability:
            return image
        quality = int(torch.randint(self.quality_min, self.quality_max + 1, (1,)).item())
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()


class RandomPaperBorderShadow:
    """Mild off-white border and soft cast shadow for printed-card robustness."""

    def __init__(self, probability: float = 0.50) -> None:
        self.probability = float(probability)

    def __call__(self, image: Any) -> Any:
        import torch
        from PIL import Image, ImageFilter

        if float(torch.rand(())) >= self.probability:
            return image
        source = image.convert("RGB")
        width, height = source.size
        border_ratio = 0.025 + 0.055 * float(torch.rand(()))
        border = max(2, round(min(width, height) * border_ratio))
        offset = max(1, round(border * (0.25 + 0.50 * float(torch.rand(())))))
        paper_value = int(torch.randint(238, 256, (1,)).item())
        canvas = Image.new("RGB", (width + 2 * border + offset, height + 2 * border + offset), (paper_value,) * 3)
        shadow_mask = Image.new("L", canvas.size, 0)
        shadow_rect = Image.new("L", (width, height), 105)
        shadow_mask.paste(shadow_rect, (border + offset, border + offset))
        shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, border * 0.45)))
        shadow_layer = Image.new("RGB", canvas.size, (110, 110, 110))
        canvas = Image.composite(shadow_layer, canvas, shadow_mask)
        canvas.paste(source, (border, border))
        return canvas


def build_transforms(*, train: bool) -> Any:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if not train:
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(INPUT_SIZE),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            RandomPaperBorderShadow(probability=0.50),
            transforms.RandomPerspective(
                distortion_scale=0.12,
                p=0.40,
                interpolation=InterpolationMode.BILINEAR,
                fill=245,
            ),
            transforms.RandomRotation(
                degrees=7,
                interpolation=InterpolationMode.BILINEAR,
                fill=245,
            ),
            transforms.RandomResizedCrop(
                INPUT_SIZE,
                scale=(0.72, 1.0),
                ratio=(0.86, 1.14),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.025),
            RandomColorTemperature(strength=0.10, probability=0.45),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.10, 1.10))], p=0.25
            ),
            RandomJpegRoundTrip(probability=0.35, quality_min=58, quality_max=94),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ManifestImageDataset:
    def __init__(self, pack_root: Path, rows: Sequence[Mapping[str, Any]], transform: Any) -> None:
        self.pack_root = pack_root
        self.rows = tuple(dict(row) for row in rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        from PIL import Image, ImageOps

        row = self.rows[index]
        path = safe_child(self.pack_root, str(row["filename"]))
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        tensor = self.transform(image)
        return tensor, CLASS_TO_INDEX[str(row["class_id"])], index


def build_model(*, pretrained: bool, workspace: Path) -> tuple[Any, dict[str, Any]]:
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    # Keep downloaded deployment inputs inside AdventureX instead of silently
    # populating a user-profile cache outside the competition workspace.
    os.environ["TORCH_HOME"] = str(workspace / "models" / "torch_cache")
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.avgpool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    provenance: dict[str, Any] = {
        "architecture": "torchvision.resnet18",
        "adaptive_pooling": False,
        "average_pool": {"kernel_size": [7, 7], "stride": [1, 1]},
        "input_shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
        "class_order": list(CLASS_NAMES),
        "pretrained": pretrained,
        "pretrained_weights_enum": "ResNet18_Weights.DEFAULT" if pretrained else None,
        "pretrained_weight_url": weights.url if weights is not None else None,
    }
    if weights is not None:
        cached = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints" / Path(weights.url).name
        if not cached.is_file():
            raise GateError(f"torchvision reported pretrained weights but cache file is absent: {cached}")
        provenance["pretrained_weight_path"] = str(cached.relative_to(workspace))
        provenance["pretrained_weight_sha256"] = sha256_file(cached)
    return model, provenance


def make_loaders(
    pack: AuditedPack,
    *,
    batch_size: int,
    seed: int,
    samples_per_class: int,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_dataset = ManifestImageDataset(pack.root, pack.train_rows, build_transforms(train=True))
    val_dataset = ManifestImageDataset(pack.root, pack.validation_rows, build_transforms(train=False))
    print_dataset = ManifestImageDataset(pack.root, pack.print_rows, build_transforms(train=False))
    counts = Counter(str(row["class_id"]) for row in pack.train_rows)
    sample_weights = [1.0 / counts[str(row["class_id"])] for row in pack.train_rows]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=samples_per_class * len(CLASS_NAMES),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print_loader = DataLoader(print_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    balancing = {
        "method": "inverse-frequency WeightedRandomSampler with replacement",
        "source_class_counts": {name: counts[name] for name in CLASS_NAMES},
        "class_balance_semantics": (
            "balanced_in_expectation_only; replacement sampling does not guarantee exact "
            "per-class realized draws in an epoch"
        ),
        "expected_draws_per_class": samples_per_class,
        "samples_per_epoch": samples_per_class * len(CLASS_NAMES),
        "loss_class_weights": None,
    }
    return train_loader, val_loader, print_loader, balancing


def build_onnx_consistency_probes(validation_dataset: ManifestImageDataset) -> dict[str, Any]:
    import torch

    probes: dict[str, Any] = {}
    seen_classes: set[str] = set()
    for index, row in enumerate(validation_dataset.rows):
        class_id = str(row["class_id"])
        if class_id in seen_classes:
            continue
        tensor, label, _source_index = validation_dataset[index]
        if int(label) != CLASS_TO_INDEX[class_id]:
            raise GateError(f"validation probe label mismatch for {class_id}")
        probes[f"natural_validation_first_{class_id}"] = tensor.unsqueeze(0).cpu()
        seen_classes.add(class_id)
    probes["synthetic_zero"] = torch.zeros(
        (1, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32
    )
    probes["synthetic_ramp"] = torch.linspace(
        -2.0,
        2.0,
        steps=3 * INPUT_SIZE * INPUT_SIZE,
        dtype=torch.float32,
    ).reshape(1, 3, INPUT_SIZE, INPUT_SIZE)
    return probes


def _metrics_from_logits(logits: Any, labels: Any, *, domain: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    if int(labels.numel()) == 0:
        raise GateError(f"cannot evaluate empty domain {domain}")
    probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[int(truth), int(prediction)] += 1
    per_class: dict[str, Any] = {}
    recalls: list[float] = []
    for class_index, name in enumerate(CLASS_NAMES):
        support = int((labels == class_index).sum().item())
        correct = int(confusion[class_index, class_index].item())
        recall = (correct / support) if support else None
        if recall is not None:
            recalls.append(recall)
        per_class[name] = {"support": support, "recall": recall}
    confidence, _ = probabilities.max(dim=1)
    correctness = predictions.eq(labels).float()
    ece = 0.0
    for low in torch.linspace(0.0, 0.9, 10):
        high = low + 0.1
        mask = (confidence >= low) & (confidence < high if float(high) < 1.0 else confidence <= 1.0)
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


def collect_logits(model: Any, loader: Any, device: Any, *, max_batches: int | None = None) -> tuple[Any, Any]:
    import torch

    model.eval()
    all_logits: list[Any] = []
    all_labels: list[Any] = []
    with torch.inference_mode():
        for batch_index, (images, labels, _indices) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            all_logits.append(model(images.to(device)).detach().cpu())
            all_labels.append(labels.detach().cpu())
    if not all_logits:
        raise GateError("evaluation loader produced no batches")
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: Any, labels: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    candidates = torch.logspace(math.log10(0.25), math.log10(4.0), steps=161)
    losses = torch.tensor([functional.cross_entropy(logits / value, labels) for value in candidates])
    best_index = int(losses.argmin().item())
    coarse = float(candidates[best_index].item())
    lower = max(0.05, coarse / 1.12)
    upper = min(10.0, coarse * 1.12)
    fine = torch.linspace(lower, upper, 121)
    fine_losses = torch.tensor([functional.cross_entropy(logits / value, labels) for value in fine])
    fine_index = int(fine_losses.argmin().item())
    temperature = float(fine[fine_index].item())
    return {
        "method": "deterministic_log_grid_then_local_grid",
        "temperature": temperature,
        "uncalibrated_nll": float(functional.cross_entropy(logits, labels).item()),
        "calibrated_nll": float(functional.cross_entropy(logits / temperature, labels).item()),
        "calibration_domain": NATURAL_VAL_DOMAIN,
    }


def wilson_lower_bound(correct: int, total: int, *, z: float = 1.96) -> float | None:
    if total <= 0:
        return None
    proportion = correct / total
    denominator = 1.0 + (z * z) / total
    centre = proportion + (z * z) / (2.0 * total)
    radius = z * math.sqrt(
        (proportion * (1.0 - proportion) + (z * z) / (4.0 * total)) / total
    )
    return max(0.0, (centre - radius) / denominator)


def calibrate_rejection(
    logits: Any,
    labels: Any,
    *,
    temperature: float,
    target_accepted_accuracy: float = 0.80,
    minimum_accepted: int = 2,
    per_predicted_class_minimum_accepted: int = 2,
    wilson_z: float = 1.96,
) -> dict[str, Any]:
    import torch

    probabilities = torch.softmax(logits / temperature, dim=1)
    top2 = probabilities.topk(k=2, dim=1)
    confidence = top2.values[:, 0]
    margins = top2.values[:, 0] - top2.values[:, 1]
    correct = top2.indices[:, 0].eq(labels)
    confidence_candidates = sorted({0.0, 1.0, *(float(value) for value in confidence.tolist())})
    margin_candidates = sorted({0.0, 1.0, *(float(value) for value in margins.tolist())})
    feasible: list[tuple[float, float, int, float, float]] = []
    for confidence_threshold in confidence_candidates:
        for margin_threshold in margin_candidates:
            accepted = (confidence >= confidence_threshold) & (margins >= margin_threshold)
            count = int(accepted.sum().item())
            if count < minimum_accepted:
                continue
            accuracy = float(correct[accepted].float().mean().item())
            if accuracy + 1e-12 >= target_accepted_accuracy:
                coverage = count / int(labels.numel())
                feasible.append((coverage, accuracy, count, confidence_threshold, margin_threshold))
    if feasible:
        global_coverage, global_accuracy, global_count, confidence_threshold, margin_threshold = max(
            feasible,
            key=lambda item: (item[0], item[1], -item[3], -item[4]),
        )
        global_mode = "VALIDATION_GRID_TARGET_MET"
    else:
        confidence_threshold = 1.0
        margin_threshold = 1.0
        global_count = 0
        global_coverage = 0.0
        global_accuracy = None
        global_mode = "FAIL_CLOSED_REJECT_ALL_TARGET_NOT_MET"
    globally_accepted = (confidence >= confidence_threshold) & (margins >= margin_threshold)
    predicted = top2.indices[:, 0]
    per_predicted_class: dict[str, Any] = {}
    enabled_by_index: list[bool] = []
    for class_index, name in enumerate(CLASS_NAMES):
        predicted_mask = predicted.eq(class_index)
        accepted_mask = globally_accepted & predicted_mask
        accepted_count = int(accepted_mask.sum().item())
        correct_count = int((correct & accepted_mask).sum().item())
        accepted_accuracy = (correct_count / accepted_count) if accepted_count else None
        lower_bound = wilson_lower_bound(correct_count, accepted_count, z=wilson_z)
        reasons: list[str] = []
        if global_mode != "VALIDATION_GRID_TARGET_MET":
            reasons.append("GLOBAL_TARGET_NOT_MET")
        if accepted_count < per_predicted_class_minimum_accepted:
            reasons.append("INSUFFICIENT_ACCEPTED_SUPPORT")
        if lower_bound is None or lower_bound + 1e-12 < target_accepted_accuracy:
            reasons.append("WILSON_LOWER_BOUND_BELOW_TARGET")
        enabled = not reasons
        enabled_by_index.append(enabled)
        per_predicted_class[name] = {
            "predicted_validation_support": int(predicted_mask.sum().item()),
            "global_threshold_accepted_count": accepted_count,
            "global_threshold_correct_count": correct_count,
            "global_threshold_accepted_accuracy": accepted_accuracy,
            "wilson_lower_bound": lower_bound,
            "minimum_accepted_required": per_predicted_class_minimum_accepted,
            "target_lower_bound": target_accepted_accuracy,
            "acceptance_enabled": enabled,
            "force_reject_reasons": reasons,
        }
    enabled_tensor = torch.tensor(enabled_by_index, dtype=torch.bool, device=predicted.device)
    accepted = globally_accepted & enabled_tensor[predicted]
    count = int(accepted.sum().item())
    coverage = count / int(labels.numel())
    accuracy = float(correct[accepted].float().mean().item()) if count else None
    if global_mode != "VALIDATION_GRID_TARGET_MET":
        mode = global_mode
    elif count == 0:
        mode = "VALIDATION_GRID_TARGET_MET_PER_CLASS_EVIDENCE_REJECTS_ALL"
    else:
        mode = "VALIDATION_GRID_TARGET_MET_WITH_PER_CLASS_EVIDENCE_GATE"
    return {
        "method": "joint_confidence_and_top1_top2_margin_validation_grid",
        "calibration_domain": NATURAL_VAL_DOMAIN,
        "temperature": temperature,
        "target_accepted_accuracy": target_accepted_accuracy,
        "minimum_accepted": minimum_accepted,
        "per_predicted_class_minimum_accepted": per_predicted_class_minimum_accepted,
        "wilson_z": wilson_z,
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "global_validation_accepted_count": global_count,
        "global_validation_coverage": global_coverage,
        "global_validation_accepted_accuracy": global_accuracy,
        "validation_accepted_count": count,
        "validation_coverage": coverage,
        "validation_accepted_accuracy": accuracy,
        "per_predicted_class_evidence": per_predicted_class,
        "mode": mode,
        "decision_rule": (
            "accept iff max_softmax >= confidence_threshold AND top1_minus_top2 >= "
            "margin_threshold AND the predicted class has acceptance_enabled=true from "
            "validation support plus Wilson-lower-bound evidence; else REJECT"
        ),
    }


def apply_rejection_metrics(
    logits: Any,
    labels: Any,
    *,
    domain: str,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    temperature = float(calibration["temperature"])
    probabilities = torch.softmax(logits / temperature, dim=1)
    top2 = probabilities.topk(k=2, dim=1)
    confidence = top2.values[:, 0]
    margin = top2.values[:, 0] - top2.values[:, 1]
    globally_accepted = (confidence >= float(calibration["confidence_threshold"])) & (
        margin >= float(calibration["margin_threshold"])
    )
    class_evidence = calibration.get("per_predicted_class_evidence")
    if not isinstance(class_evidence, dict) or set(class_evidence) != set(CLASS_NAMES):
        raise GateError("calibration lacks complete per-predicted-class rejection evidence")
    enabled_values: list[bool] = []
    for name in CLASS_NAMES:
        record = class_evidence[name]
        if not isinstance(record, dict) or not isinstance(record.get("acceptance_enabled"), bool):
            raise GateError(f"calibration per-class evidence is invalid for {name}")
        enabled_values.append(bool(record["acceptance_enabled"]))
    predicted = top2.indices[:, 0]
    enabled = torch.tensor(enabled_values, dtype=torch.bool, device=predicted.device)
    accepted = globally_accepted & enabled[predicted]
    count = int(accepted.sum().item())
    correct = predicted.eq(labels)
    per_predicted_class: dict[str, Any] = {}
    for class_index, name in enumerate(CLASS_NAMES):
        predicted_mask = predicted.eq(class_index)
        final_mask = accepted & predicted_mask
        final_count = int(final_mask.sum().item())
        per_predicted_class[name] = {
            "predicted_count": int(predicted_mask.sum().item()),
            "global_threshold_accepted_count": int((globally_accepted & predicted_mask).sum().item()),
            "accepted_count": final_count,
            "accepted_accuracy": (
                float(correct[final_mask].float().mean().item()) if final_count else None
            ),
            "acceptance_enabled_from_calibration": enabled_values[class_index],
        }
    return {
        "domain": domain,
        "sample_count": int(labels.numel()),
        "accepted_count": count,
        "rejected_count": int(labels.numel()) - count,
        "coverage": count / int(labels.numel()),
        "accepted_accuracy": float(correct[accepted].float().mean().item()) if count else None,
        "per_predicted_class": per_predicted_class,
        "per_predicted_class_gate_applied": True,
        "thresholds_locked_from": NATURAL_VAL_DOMAIN,
        "thresholds_optimized_on_this_domain": domain == NATURAL_VAL_DOMAIN,
    }


def train_one_seed(
    pack: AuditedPack,
    output: Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    samples_per_class: int,
    device_name: str,
    pretrained: bool,
    max_train_batches: int | None,
    artifact_status: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    set_determinism(seed)
    device = torch.device(device_name)
    model, model_provenance = build_model(pretrained=pretrained, workspace=pack.workspace)
    model.to(device)
    train_loader, val_loader, print_loader, balancing = make_loaders(
        pack,
        batch_size=batch_size,
        seed=seed,
        samples_per_class=samples_per_class,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")], "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 8e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_epoch = -1
    checkpoint_path = output / "best_checkpoint.pt"
    output.mkdir(parents=True, exist_ok=False)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for batch_index, (images, labels, _indices) in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            sample_count += int(labels.numel())
        if sample_count == 0:
            raise GateError("training produced zero samples")
        scheduler.step()
        val_logits, val_labels = collect_logits(model, val_loader, device)
        val_metrics = _metrics_from_logits(val_logits, val_labels, domain=NATURAL_VAL_DOMAIN)
        key = (
            float(val_metrics["balanced_accuracy_present_classes"]),
            -float(val_metrics["cross_entropy"]),
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_cross_entropy": total_loss / sample_count,
                "train_samples_seen": sample_count,
                "natural_validation": val_metrics,
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch + 1
            torch.save(
                {
                    "schema_version": "rootscope.resnet18_experimental_checkpoint.v1",
                    "status": artifact_status,
                    "seed": seed,
                    "epoch": best_epoch,
                    "class_order": list(CLASS_NAMES),
                    "input_shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
                    "architecture": "torchvision.resnet18_fixed_avgpool7x7",
                    "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "input_pack_manifest_sha256": pack.immutable_snapshot["manifest_sha256"],
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    val_logits, val_labels = collect_logits(model, val_loader, device)
    print_logits, print_labels = collect_logits(model, print_loader, device)
    temperature_record = fit_temperature(val_logits, val_labels)
    rejection = calibrate_rejection(
        val_logits,
        val_labels,
        temperature=float(temperature_record["temperature"]),
    )
    calibration = {**temperature_record, **rejection}
    calibration["status"] = (
        "SMOKE_PATH_EXERCISE_ONLY_NOT_VALID_CALIBRATION"
        if artifact_status == SMOKE_STATUS
        else "MACHINE_CURATED_EXPERIMENTAL_CALIBRATION_NOT_FORMALLY_QUALIFIED"
    )
    metrics = {
        "schema_version": "rootscope.machine_curated_experimental_metrics.v1",
        "status": artifact_status,
        "seed": seed,
        "best_epoch": best_epoch,
        "natural_validation_raw": _metrics_from_logits(
            val_logits / float(calibration["temperature"]), val_labels, domain=NATURAL_VAL_DOMAIN
        ),
        "natural_validation_rejection": apply_rejection_metrics(
            val_logits, val_labels, domain=NATURAL_VAL_DOMAIN, calibration=calibration
        ),
        "digital_print_source_holdout_raw": _metrics_from_logits(
            print_logits / float(calibration["temperature"]), print_labels, domain=PRINT_EVAL_DOMAIN
        ),
        "digital_print_source_holdout_rejection": apply_rejection_metrics(
            print_logits, print_labels, domain=PRINT_EVAL_DOMAIN, calibration=calibration
        ),
        "digital_print_source_holdout_is_uvc_recapture": False,
        "digital_print_source_holdout_claim": "DIGITAL_SOURCE_EVALUATION_ONLY_NOT_REAL_PRINT_DOMAIN_EVIDENCE",
        "history": history,
    }
    write_json(output / "calibration.json", calibration)
    write_json(output / "metrics.json", metrics)
    write_json(output / "model_provenance.json", model_provenance)
    onnx_path = output / "model_static_b1x3x224x224_opset11.onnx"
    export_onnx(model.to("cpu"), onnx_path)
    consistency = verify_onnx_consistency(
        model.to("cpu"),
        onnx_path,
        probe_tensors=build_onnx_consistency_probes(val_loader.dataset),
    )
    write_json(output / "onnx_consistency.json", consistency)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_key": list(best_key or (float("-inf"), float("-inf"))),
        "balancing": balancing,
        "model_provenance": model_provenance,
        "metrics": metrics,
        "calibration": calibration,
        "onnx_consistency": consistency,
        "artifacts": {
            "checkpoint": f"{output.name}/{checkpoint_path.name}",
            "onnx": f"{output.name}/{onnx_path.name}",
        },
    }


def export_onnx(model: Any, output_path: Path) -> None:
    import torch

    try:
        import onnx
    except ImportError as error:
        raise GateError("ONNX export requires the 'onnx' package; no export was claimed") from error
    model.eval()
    dummy = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    opsets = {entry.domain: entry.version for entry in graph.opset_import}
    if opsets.get("", opsets.get("ai.onnx")) != ONNX_OPSET:
        raise GateError(f"exported ONNX opset is not {ONNX_OPSET}: {opsets}")
    dimensions = [dimension.dim_value for dimension in graph.graph.input[0].type.tensor_type.shape.dim]
    if dimensions != [1, 3, INPUT_SIZE, INPUT_SIZE]:
        raise GateError(f"ONNX input is not static [1,3,224,224]: {dimensions}")
    node_types = [node.op_type for node in graph.graph.node]
    if "GlobalAveragePool" in node_types:
        raise GateError("ONNX contains GlobalAveragePool; fixed 7x7 AveragePool is required")
    fixed_pool = False
    for node in graph.graph.node:
        if node.op_type != "AveragePool":
            continue
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        if list(attributes.get("kernel_shape", [])) == [7, 7]:
            fixed_pool = True
    if not fixed_pool:
        raise GateError("ONNX does not contain the required fixed 7x7 AveragePool")


def verify_onnx_consistency(
    model: Any,
    onnx_path: Path,
    *,
    sample_tensor: Any | None = None,
    probe_tensors: Mapping[str, Any] | None = None,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    import numpy as np
    import torch

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise GateError("ONNX consistency requires 'onnxruntime'; no consistency claim was made") from error
    if probe_tensors is not None and sample_tensor is not None:
        raise GateError("provide either sample_tensor or probe_tensors, not both")
    if probe_tensors is None:
        if sample_tensor is None:
            raise GateError("ONNX consistency requires at least one probe")
        probe_tensors = {"legacy_sample": sample_tensor}
    if not probe_tensors:
        raise GateError("ONNX consistency probe set is empty")
    model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probe_records: list[dict[str, Any]] = []
    aggregate_absolute_error_sum = 0.0
    aggregate_element_count = 0
    max_absolute_error = 0.0
    for name, value in probe_tensors.items():
        if not isinstance(name, str) or not name:
            raise GateError("ONNX consistency probe names must be non-empty strings")
        tensor = value.detach().cpu().to(dtype=torch.float32)
        if list(tensor.shape) != [1, 3, INPUT_SIZE, INPUT_SIZE]:
            raise GateError(f"consistency input shape mismatch for {name}: {list(tensor.shape)}")
        array = tensor.numpy()
        if not np.isfinite(array).all():
            raise GateError(f"ONNX consistency probe contains non-finite values: {name}")
        with torch.inference_mode():
            torch_logits = model(tensor).detach().cpu().numpy()
        ort_logits = session.run(["logits"], {"image": array})[0]
        if list(torch_logits.shape) != [1, len(CLASS_NAMES)] or list(ort_logits.shape) != [
            1,
            len(CLASS_NAMES),
        ]:
            raise GateError(f"Torch/ONNX output shape mismatch for probe {name}")
        absolute_error = np.abs(torch_logits - ort_logits)
        probe_max = float(np.max(absolute_error))
        probe_mean = float(np.mean(absolute_error))
        if not np.isfinite(probe_max) or probe_max > tolerance:
            raise GateError(
                f"Torch/ONNX logits mismatch for {name}: "
                f"max_abs={probe_max:.8g} > {tolerance:.8g}"
            )
        max_absolute_error = max(max_absolute_error, probe_max)
        aggregate_absolute_error_sum += float(absolute_error.sum())
        aggregate_element_count += int(absolute_error.size)
        probe_records.append(
            {
                "name": name,
                "input_shape": list(tensor.shape),
                "max_absolute_error": probe_max,
                "mean_absolute_error": probe_mean,
                "passed": True,
            }
        )
    mean_absolute_error = aggregate_absolute_error_sum / aggregate_element_count
    validation_classes = sorted(
        name.removeprefix("natural_validation_first_")
        for name in probe_tensors
        if name.startswith("natural_validation_first_")
    )
    return {
        "schema_version": "rootscope.torch_onnx_consistency.v2",
        "passed": True,
        "input_shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
        "class_order": list(CLASS_NAMES),
        "opset": ONNX_OPSET,
        "probe_count": len(probe_records),
        "probes": probe_records,
        "natural_validation_classes_probed": validation_classes,
        "natural_validation_classes_missing": sorted(set(CLASS_NAMES) - set(validation_classes)),
        "synthetic_zero_probed": "synthetic_zero" in probe_tensors,
        "synthetic_ramp_probed": "synthetic_ramp" in probe_tensors,
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": mean_absolute_error,
        "tolerance": tolerance,
        "onnxruntime_providers": session.get_providers(),
    }


def verify_immutable_after(pack: AuditedPack) -> dict[str, Any]:
    receipt = load_json(pack.root / "receipt.json")
    frozen_v2_snapshot, frozen_v2_receipt = verify_frozen_v2_reference(pack.workspace, receipt)
    protected_snapshot, protected_human_root = verify_protected_inputs(
        pack.workspace,
        receipt,
        frozen_v2_receipt=frozen_v2_receipt,
    )
    formal = receipt.get("formal_human_decisions")
    if isinstance(formal, dict):
        formal_root = safe_child(pack.workspace, str(formal.get("path", "")))
    elif protected_human_root is not None:
        formal_root = protected_human_root
    else:
        raise GateError("receipt lacks formal human-decision preservation evidence")
    contract = PACK_CONTRACTS[pack.root.name]
    current = {
        "pack_tree_sha256": tree_sha256(pack.root),
        "manifest_sha256": sha256_file(pack.root / "manifest.jsonl"),
        "split_suggestion_sha256": sha256_file(pack.root / "experimental_split_suggestion.json"),
        "receipt_sha256": sha256_file(pack.root / "receipt.json"),
        "payload_root_sha256_before_receipt": payload_root_sha256(pack.root),
        "formal_human_decisions_tree_sha256": tree_sha256(formal_root),
        "formal_human_decision_journal_sha256": sha256_file(formal_root / "decision_journal.jsonl"),
    }
    for source_key in contract["allowed_source_keys"]:
        source_manifest = (
            pack.workspace
            / "datasets"
            / SOURCE_DATASET_NAMES[str(source_key)]
            / "manifest.jsonl"
        )
        current[f"source_manifest_{source_key}_sha256"] = sha256_file(source_manifest)
    frozen_snapshot, _rows = verify_frozen_v1_reference(pack.workspace, receipt)
    current.update({f"frozen_v1_{key}": value for key, value in frozen_snapshot.items()})
    current.update({f"frozen_v2_{key}": value for key, value in frozen_v2_snapshot.items()})
    current.update({f"protected_input_{key}": value for key, value in protected_snapshot.items()})
    if current != pack.immutable_snapshot:
        raise GateError(
            "input pack or formal human-decision evidence changed during training: "
            f"before={pack.immutable_snapshot}, after={current}"
        )
    return {"unchanged": True, "before": pack.immutable_snapshot, "after": current}


def artifact_hashes(root: Path, *, exclude: Iterable[str] = ()) -> dict[str, str]:
    excluded = set(exclude)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix())
        if path.relative_to(root).as_posix() not in excluded
    }


def runtime_versions() -> dict[str, str]:
    import numpy
    import onnx
    import onnxruntime
    import PIL
    import torch
    import torchvision

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
    }


def model_card_text(receipt: Mapping[str, Any]) -> str:
    selected = receipt["selected_seed"]
    return f"""# RootScope ResNet18 machine-curated experimental model

Status: `{receipt['status']}`

## Intended use

This is an isolated RootScope competition experiment for four-way fixed-ROI image classification.  The class order is fixed as `{', '.join(CLASS_NAMES)}` and the input is static `1x3x224x224` RGB.  The backbone is torchvision ResNet18 with standard Conv/BN/ReLU blocks and a **fixed 7x7 AvgPool**; the exported graph is static ONNX opset 11.

## Non-claims and data limits

- The input pack remains `{receipt['input_pack_status']}`.
- It is not formal A1 data, not human-reviewed truth, not rights-approved, not print-eligible, not data-locked, and not formally training-eligible.
- This model is not qualified for deployment or irrigation decisions.
- `natural_validation` is a small natural-web suggestion partition.
- `{PRINT_EVAL_DOMAIN}` evaluates the original digital files selected for future printing.  It is **not** evidence from a physical print, UVC camera recapture, glare, viewing-distance, or optical-loop test.
- Confidence and top-1/top-2 margin rejection thresholds were calibrated only on the natural validation suggestion.  The digital holdout never tuned weights, checkpoint selection, temperature, or rejection thresholds.

## Training design

The experimental training sampler uses inverse-frequency weights with replacement.  Its class draws are balanced **in expectation** over an epoch; an individual epoch is not guaranteed to contain an exact per-class count.  Augmentations target the likely demo-card optical domain: mild perspective, rotation and crop; brightness, contrast and color-temperature drift; Gaussian blur; JPEG artifacts; and an off-white paper border/cast-shadow simulation.  These synthetic effects do not constitute a real print-domain test.

Selected deterministic seed: `{selected['seed']}`.  Best epoch: `{selected['best_epoch']}`.

## Deployment boundary

The ONNX artifact is an experimental engineering artifact only; `model_candidate=false`.  CPU/ONNX agreement proves numerical export consistency, not BPU conversion, X5 runtime readiness, camera compatibility, or physical-domain accuracy.  BPU compilation and clean-X5 replay require separate evidence.
"""


def require_long_training_coverage(pack: AuditedPack) -> None:
    if pack.audit["long_training_coverage_gate_passed"]:
        return
    deficits = {
        name: value
        for name, value in pack.audit["long_training_coverage"].items()
        if not all(
            value[key]
            for key in (
                "train_met",
                "train_unique_source_met",
                "train_unique_creator_met",
                "validation_met",
                "validation_unique_source_met",
                "validation_unique_creator_met",
            )
        )
    }
    raise GateError(
        "refusing long training and formal calibration because class coverage is insufficient: "
        f"{deficits}"
    )


def validate_output_root(workspace: Path, candidate: Path) -> Path:
    """Validate the write boundary before creating any directory."""

    allowed_root = (workspace / "output").resolve(strict=False)
    output_root = candidate.resolve(strict=False)
    try:
        output_root.relative_to(allowed_root)
    except ValueError as error:
        raise GateError("all training outputs must stay inside AdventureX workspace/output") from error
    return output_root


def run(args: argparse.Namespace) -> Path:
    if not args.ack_machine_curated_experimental_only:
        raise GateError(
            "refusing: pass --ack-machine-curated-experimental-only to acknowledge that this is "
            "machine-curated experimental data with every formal authority bit false"
        )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)) or any(value < 0 for value in seeds):
        raise GateError("--seeds must be a non-empty comma-separated list of unique non-negative integers")
    if args.smoke:
        if len(seeds) != 1:
            raise GateError("smoke mode requires exactly one seed")
        if args.epochs != 1:
            raise GateError("smoke mode requires --epochs 1")
        if args.max_train_batches is None or args.max_train_batches < 1 or args.max_train_batches > 2:
            raise GateError("smoke mode requires --max-train-batches 1 or 2")
    elif args.max_train_batches is not None:
        raise GateError("--max-train-batches is smoke-only")
    if args.random_init and not args.smoke:
        raise GateError("random initialization is smoke-only; experimental training requires ImageNet pretraining")
    if args.epochs < 1 or args.batch_size < 1 or args.samples_per_class < 1:
        raise GateError("epochs, batch size, and samples per class must be positive")

    workspace = args.workspace.resolve(strict=True)
    pack_root = args.pack.resolve(strict=True)
    output_root = validate_output_root(workspace, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pack = audit_pack(workspace, pack_root)
    if not args.smoke:
        require_long_training_coverage(pack)
        if not pack.audit["machine_visual_review_evidence"].get(
            "non_smoke_training_evidence_eligible", False
        ):
            raise GateError(
                "refusing non-smoke training without receipt-bound machine_visual_review_evidence.json"
            )
    run_id = args.run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise GateError("run id may contain only letters, digits, '-' and '_'")
    final = output_root / run_id
    if final.exists():
        raise GateError(f"output run already exists: {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=str(output_root)))
    try:
        write_json(staging / "input_audit.json", pack.audit)
        device_name = args.device
        if device_name == "auto":
            import torch

            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        seed_results: list[dict[str, Any]] = []
        artifact_status = SMOKE_STATUS if args.smoke else MODEL_STATUS
        for seed in seeds:
            result = train_one_seed(
                pack,
                staging / f"seed_{seed:05d}",
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                samples_per_class=args.samples_per_class,
                device_name=device_name,
                pretrained=not args.random_init,
                max_train_batches=args.max_train_batches,
                artifact_status=artifact_status,
            )
            seed_results.append(result)
        selected = max(seed_results, key=lambda value: tuple(value["selection_key"]))
        immutable_after = verify_immutable_after(pack)
        receipt: dict[str, Any] = {
            "schema_version": "rootscope.machine_curated_experimental_training_receipt.v1",
            "status": artifact_status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "smoke_only": bool(args.smoke),
            "formal_a1_dataset": False,
            "human_reviewed": False,
            "rights_approved": False,
            "rights": False,
            "training_eligible": False,
            "print_eligible": False,
            "data_locked": False,
            "model_qualified": False,
            "model_candidate": False,
            "experimental_model_candidate": not args.smoke,
            "x5_ready": False,
            "bpu_compiled": False,
            "physical_print_tested": False,
            "uvc_recapture_evaluated": False,
            "execution_authority": False,
            "authority": {key: False for key in REQUIRED_FALSE_AUTHORITY},
            "ack_machine_curated_experimental_only": True,
            "legacy_cli_visual_audit_ack_non_authoritative": bool(
                args.ack_independent_visual_audit_complete
            ),
            "machine_visual_review_evidence": pack.audit[
                "machine_visual_review_evidence"
            ],
            "input_pack_status": pack.status,
            "input_pack": str(pack.root.relative_to(workspace)),
            "input_audit_sha256": sha256_file(staging / "input_audit.json"),
            "training_pipeline_sha256": sha256_file(Path(__file__).resolve()),
            "class_order": list(CLASS_NAMES),
            "architecture": "torchvision.resnet18_fixed_avgpool7x7",
            "input_shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
            "onnx_opset": ONNX_OPSET,
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "samples_per_class": args.samples_per_class,
            "device": device_name,
            "pretrained": not args.random_init,
            "natural_validation_domain": NATURAL_VAL_DOMAIN,
            "digital_print_holdout_domain": PRINT_EVAL_DOMAIN,
            "digital_print_holdout_is_uvc_recapture": False,
            "long_training_coverage_gate_passed": pack.audit[
                "long_training_coverage_gate_passed"
            ],
            "seed_results": seed_results,
            "selected_seed": selected,
            "input_and_formal_authority_unchanged": immutable_after,
            "python": sys.version,
            "platform": platform.platform(),
            "runtime_versions": runtime_versions(),
            "explicit_non_claims": [
                "FORMAL_A1_DATASET",
                "HUMAN_REVIEWED",
                "RIGHTS_APPROVED",
                "TRAIN_READY",
                "PRINT_ELIGIBLE",
                "DATA_LOCKED",
                "MODEL_QUALIFIED",
                "REAL_PRINT_DOMAIN_PASSED",
                "UVC_RECAPTURE_PASSED",
                "BPU_COMPILED",
                "X5_READY",
            ],
        }
        receipt["artifact_hashes_before_receipt"] = artifact_hashes(staging)
        write_json(staging / "run_receipt.json", receipt)
        write_text(staging / "MODEL_CARD.md", model_card_text(receipt))
        hashes = artifact_hashes(staging, exclude={"SHA256SUMS"})
        write_text(staging / "SHA256SUMS", "".join(f"{digest}  {path}\n" for path, digest in hashes.items()))
        staging.replace(final)
        return final
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--pack", type=Path, default=workspace / "datasets" / PACK_NAME)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=workspace / "output" / "rootscope_machine_curated_experimental_runs",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--ack-machine-curated-experimental-only", action="store_true")
    parser.add_argument(
        "--ack-independent-visual-audit-complete",
        action="store_true",
        help=(
            "legacy non-authoritative metadata only; non-smoke execution requires "
            "receipt-bound machine_visual_review_evidence.json"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--random-init", action="store_true", help="smoke-only; never allowed for a real experiment")
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=8,
        help="expected class-balanced draws per class; exact realized counts are not guaranteed",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-batches", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run(args)
    except (GateError, FileNotFoundError, ValueError) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2
    label = "PASS_SMOKE_ONLY" if args.smoke else "PASS_EXPERIMENTAL_ONLY"
    print(f"{label}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
