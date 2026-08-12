#!/usr/bin/env python3
"""Read-only independent audit for RootScope provisional v3.

This module intentionally does not import the v3 builder or any of its
validation helpers.  The frozen contract, source hashes, role allocation, and
protection roots below are independent audit inputs.  A report may be written
outside the audited pack; the pack itself is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


PACK_NAME = "rootscope_machine_curated_provisional_v3"
STATUS = "MACHINE_CURATED_EXPERIMENTAL_V3_ONLY_NOT_HUMAN_REVIEWED_NOT_A1_NOT_DATA_LOCKED"
ASSET_SCHEMA = "rootscope.machine_curated_provisional_asset.v3"
RECEIPT_SCHEMA = "rootscope.machine_curated_provisional_receipt.v3"
AUDIT_SCHEMA = "rootscope.machine_curated_provisional_independent_audit.v3"
DECISION_SCHEMA = "rootscope.machine_curated_source_decision.v3"
SPLIT_SCHEMA = "rootscope.machine_curated_v3_experimental_split_suggestion.v1"
EVIDENCE_SCHEMA = "rootscope.machine_visual_review_evidence.v3"
DHASH_ALGORITHM = "rootscope_rgb_center_sample_9x8_v1"

TRAIN_ROLE = "EXPERIMENTAL_TRAIN_SUGGESTION"
VAL_ROLE = "EXPERIMENTAL_VAL_SUGGESTION"
PRINT_ROLE = "PRINT_DEMO_HOLDOUT_NOT_TRAIN"
CREATOR_HOLDOUT_ROLE = "CREATOR_GROUP_HOLDOUT_NOT_TRAIN"
ROLES = (TRAIN_ROLE, VAL_ROLE, PRINT_ROLE, CREATOR_HOLDOUT_ROLE)
CLASSES = ("grass_clump", "low_shrub", "young_tree", "unknown")

SOURCE_DATASETS = {
    "E0": "desert_plants_wikimedia_staging_e0",
    "E1": "desert_plants_whole_plant_reacquisition_e1",
    "E2": "desert_plants_young_tree_reacquisition_e2",
    "E3": "desert_plants_young_tree_reacquisition_e3",
    "E4": "desert_plants_young_tree_category_reacquisition_e4",
}
EXPECTED_SOURCE_MANIFEST_SHA256 = {
    "E0": "e802f731588212d12c44c10a93156f019670fbf32e92ddc5a3dc15d1d80bffb1",
    "E1": "e4b0735d1e1c6ecbf09f508abda0eb28adf91493822073927d754bb9484682bb",
    "E2": "a065ab32efa18e6f1a45dfae659adcacbbc669d9f33838c90dbfbfc68f97296e",
    "E3": "bee046e6dd48a262c373937c915b4084ff1d2a42ac363c8cba9b7ce00ea38c78",
    "E4": "3c668e8fcef54660a59b84d4f1c10d7fef69a9bd371230ecf23714303d385428",
}

# Independent, pre-v3 roots.  These constants prevent a self-consistent but
# already-mutated receipt from redefining the audit baseline.
EXPECTED_PROTECTED_TREE_SHA256 = {
    "datasets/desert_plants_v1": "f9baa8d680852ec3252d837f9d51d7f83b64695741ad12ba0472e67d7ba1cb37",
    "datasets/desert_plants_wikimedia_staging_e0": "3b697b322e791dd4ab6193520f30473cca2bc8c0a4ee202a6b7cab3f2fc8bd3d",
    "datasets/desert_plants_whole_plant_reacquisition_e1": "d843485aef72e384856175227595aa04c5d48a81fdf90ee294b029c8b1760905",
    "datasets/desert_plants_young_tree_reacquisition_e2": "c852b34179231e985f0e4e59a1964adea9324978e57c52c2ad6ea9f6f72856b7",
    "datasets/desert_plants_young_tree_reacquisition_e3": "d545705fe03a03a26a5b635dfe7f856346beb79d377501a9da1f38901dc2d328",
    "datasets/desert_plants_young_tree_category_reacquisition_e4": "f0d7ca60a6a5022807f60be325ee7a4b8b7af1d1a4a0c91b9fa323ea6ba209f2",
    "datasets/rootscope_machine_curated_provisional_v1": "50d7ab00cddb6e23b7cc1afb90cbcdbdd311735ea79a57ac6a934eee0537284d",
    "datasets/rootscope_machine_curated_provisional_v2": "e8ca2ac35c905bfb7b24fd3ae269defc6c1a21454af5f902199bbe5014bf5f0d",
}
# ``desert_plants_v1`` is independently checked above as an extra ancestor,
# but the frozen v3 receipt contract protects these exact seven trees.
RECEIPT_PROTECTED_TREE_SHA256 = {
    key: value
    for key, value in EXPECTED_PROTECTED_TREE_SHA256.items()
    if key != "datasets/desert_plants_v1"
}
FORMAL_HUMAN_TREE_SHA256 = "be69131f4d47dbddcce51818134ef96a2bd100cb09d3c09b7e52b1ad7cf1a200"
FORMAL_JOURNAL_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
FROZEN_V1_MANIFEST_SHA256 = "c516350ee319b66f2505eed0ba608f670735628aee0861432a31dffae0fbcaa6"
FROZEN_V2_MANIFEST_SHA256 = "5ff2d90a8dd4d78d1079ba2d75675c3522e1b1a48a94f4c8c21151e276f4e48c"
FROZEN_V2_RECEIPT_SHA256 = "05d413fdd790c8a09f8bcb0e991bbb3bac75bef5e642a2affd6e4ce9d9a215b2"
FROZEN_V2_SUMS_SHA256 = "a8eec99a1890928b2c3f5d618f34128e48a777f6a59e23a8008f0f5af30a6c47"

REQUIRED_UPSTREAM_EVIDENCE = {
    "evidence/rootscope_machine_curated_provisional_v2_audit.json": (
        "a31aa37ca258644ccb6a4a3cff099ea7f827a29111182d056c182e8b53200be0"
    ),
    "datasets/desert_plants_young_tree_reacquisition_e3/review/"
    "machine_visual_screen_v1/decisions.jsonl": (
        "13a45cdaf8c04b67f9b2d3c38b09d1d8055e4b46970a4fc27f3a8e91726d0846"
    ),
    "datasets/desert_plants_young_tree_reacquisition_e3/review/"
    "machine_visual_screen_v1/receipt.json": (
        "a2502ce655e40ef6d60eb2e907f53d177cbadd5eec699f29ac359150f944c400"
    ),
    "datasets/desert_plants_young_tree_category_reacquisition_e4/review/"
    "machine_visual_screen_v1/adjudication_contract.json": (
        "aa0934ab0051eaef98db4b904f6398c0f47dbb70f36941ad05cbf9a166189bef"
    ),
    "datasets/desert_plants_young_tree_category_reacquisition_e4/review/"
    "machine_visual_screen_v1/manifest.jsonl": (
        "dde56cc3afdf62f9b4e16c7f10fc5cf579fb1c6f3e11590f4e7eb1b08115c96f"
    ),
    "datasets/desert_plants_young_tree_category_reacquisition_e4/review/"
    "machine_visual_screen_v1/receipt.json": (
        "62d9fac61bb0450311455f5ebb2c2f2558898ec9ed3e1a9fe6dd6e8337a7bd51"
    ),
}

EXPECTED_TOTAL = 78
EXPECTED_CLASS_COUNTS = {
    "grass_clump": 15,
    "low_shrub": 19,
    "young_tree": 13,
    "unknown": 31,
}
EXPECTED_ROLE_COUNTS = {
    TRAIN_ROLE: 55,
    VAL_ROLE: 9,
    PRINT_ROLE: 6,
    CREATOR_HOLDOUT_ROLE: 8,
}
EXPECTED_ROLE_CLASS_COUNTS = {
    TRAIN_ROLE: {"grass_clump": 8, "low_shrub": 13, "young_tree": 5, "unknown": 29},
    VAL_ROLE: {"grass_clump": 3, "low_shrub": 2, "young_tree": 2, "unknown": 2},
    PRINT_ROLE: {"grass_clump": 2, "low_shrub": 2, "young_tree": 2, "unknown": 0},
    CREATOR_HOLDOUT_ROLE: {"grass_clump": 2, "low_shrub": 2, "young_tree": 4, "unknown": 0},
}
EXPECTED_ROLE_CLASS_CREATOR_COUNTS = {
    TRAIN_ROLE: {"grass_clump": 6, "low_shrub": 8, "young_tree": 5, "unknown": 29},
    VAL_ROLE: {"grass_clump": 2, "low_shrub": 2, "young_tree": 2, "unknown": 2},
    PRINT_ROLE: {"grass_clump": 2, "low_shrub": 2, "young_tree": 2, "unknown": 0},
    CREATOR_HOLDOUT_ROLE: {"grass_clump": 2, "low_shrub": 1, "young_tree": 1, "unknown": 0},
}
EXPECTED_ROLE_CLASS_SOURCE_COUNTS = EXPECTED_ROLE_CLASS_COUNTS

EXPECTED_NEW_ROLE = {
    6191581: TRAIN_ROLE,
    92774234: TRAIN_ROLE,
    122973026: TRAIN_ROLE,
    180772202: VAL_ROLE,
    184915021: VAL_ROLE,
}
EXPECTED_NEW_SOURCE = {6191581: "E3", 92774234: "E4", 122973026: "E4", 180772202: "E4", 184915021: "E4"}
EXPECTED_ROLE_OVERRIDE = {28135991: (TRAIN_ROLE, VAL_ROLE)}
EXPECTED_PRINT_PAGEIDS = {38233728, 66745979, 74079996, 75760716, 94700516, 98911085}

AUTHORITY_KEYS = {
    "data_locked",
    "dataset_manifest_write",
    "human_review",
    "model_qualification",
    "print_eligibility",
    "rights_approval",
    "split_assignment",
    "training_eligibility",
    "visual_truth",
}
FALSE_FIELDS = ("data_locked", "human_reviewed", "print_eligible", "rights_approved", "training_eligible")

EXPECTED_RECEIPT_KEYS = {
    "schema_version", "status", "authority", "formal_a1_dataset", "formal_split_assigned",
    "human_reviewed", "data_locked", "rights_approved", "training_eligible", "print_eligible",
    "experimental_training_switch_required", "implementation_sha256", "manifest_sha256",
    "source_decision_manifest_sha256", "machine_visual_review_evidence_path",
    "machine_visual_review_evidence_sha256", "machine_visual_review_evidence",
    "payload_root_sha256_before_receipt", "audit", "all_split_targets_met",
    "source_manifest_sha256", "frozen_v2", "protected_inputs", "explicit_non_claims",
}
EXPECTED_EVIDENCE_KEYS = {
    "schema_version", "status", "authority", "human_reviewed", "human_label", "data_authority",
    "data_locked", "print_eligible", "rights_approved", "training_eligible",
    "all_selected_records_machine_screened", "dual_machine_review_completed",
    "dual_machine_review_scope", "independent_machine_review_count", "root_machine_adjudicated",
    "root_machine_adjudication_scope", "review_protocol", "v2_independent_audit",
    "upstream_machine_screens", "selected_pageids", "selected_records", "role_override",
    "explicit_non_claims",
}


class AuditError(RuntimeError):
    """The independent audit rejected the v3 pack."""


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
    rows: list[str] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in {"receipt.json", "SHA256SUMS"}:
            continue
        rows.append(f"{relative}\0{sha256_file(path)}\n")
    return sha256_bytes("".join(rows).encode("utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AuditError(f"JSONL row is not an object: {path}:{line_number}")
            result.append(value)
    return result


def safe_file(root: Path, relative_value: object) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise AuditError(f"invalid relative file path: {relative_value!r}")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise AuditError(f"unsafe relative file path: {relative_value!r}")
    resolved_root = root.resolve(strict=True)
    unresolved = resolved_root / relative
    if unresolved.is_symlink():
        raise AuditError(f"symbolic-link file is not allowed: {relative_value!r}")
    candidate = unresolved.resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise AuditError(f"path escapes root: {relative_value!r}") from error
    if not candidate.is_file():
        raise AuditError(f"path is not a regular file: {relative_value!r}")
    return candidate


def parse_sha256sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative:
            raise AuditError(f"invalid SHA256SUMS line {line_number}")
        try:
            int(digest, 16)
        except ValueError as error:
            raise AuditError(f"invalid SHA-256 at SHA256SUMS line {line_number}") from error
        if relative in result:
            raise AuditError(f"duplicate SHA256SUMS path: {relative}")
        result[relative] = digest.lower()
    if not result:
        raise AuditError("SHA256SUMS is empty")
    return result


def image_dhash64(path: Path) -> str:
    with Image.open(path) as opened:
        opened.load()
        rgb = opened.convert("RGB")
    if rgb.width < 1 or rgb.height < 1:
        raise AuditError(f"invalid image dimensions: {path}")
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
    if len(left) != 16 or len(right) != 16:
        raise AuditError("dHash values must be 16 hexadecimal characters")
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as error:
        raise AuditError("invalid hexadecimal dHash value") from error


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def require_authority_false(value: object, location: str) -> None:
    require(isinstance(value, dict), f"{location}.authority is not an object")
    assert isinstance(value, dict)
    require(set(value) == AUTHORITY_KEYS, f"{location}.authority key set differs from frozen contract")
    require(all(item is False for item in value.values()), f"{location}.authority contains non-false value")


def require_fail_closed(value: Mapping[str, Any], location: str, *, record: bool = False) -> None:
    for field in FALSE_FIELDS:
        require(value.get(field) is False, f"{location}.{field} must be exactly false")
    require_authority_false(value.get("authority"), location)
    if record:
        require(value.get("machine_curated_only") is True, f"{location}.machine_curated_only must be true")
        require(value.get("formal_a1_dataset") is False, f"{location}.formal_a1_dataset must be false")
        require(value.get("formal_split_assigned") is False, f"{location}.formal_split_assigned must be false")
        require(
            value.get("experimental_training_switch_required") is True,
            f"{location}.experimental_training_switch_required must be true",
        )
        require(value.get("split") == "UNASSIGNED_DO_NOT_TRAIN", f"{location}.split is not fail-closed")


def index_pageids(rows: Iterable[dict[str, Any]], location: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        pageid = row.get("pageid")
        require(isinstance(pageid, int) and not isinstance(pageid, bool), f"{location} invalid pageid")
        require(pageid not in result, f"{location} duplicate pageid {pageid}")
        result[pageid] = row
    return result


def role_family(role: str) -> str:
    if role == TRAIN_ROLE:
        return "train"
    if role == VAL_ROLE:
        return "validation"
    if role in {PRINT_ROLE, CREATOR_HOLDOUT_ROLE}:
        return "holdout"
    raise AuditError(f"unknown role: {role!r}")


def role_class_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        role: {class_id: sum(1 for row in rows if row["experimental_split_suggestion"] == role and row["class_id"] == class_id) for class_id in CLASSES}
        for role in ROLES
    }


def role_class_unique_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, dict[str, int]]:
    return {
        role: {
            class_id: len(
                {
                    str(row[field])
                    for row in rows
                    if row["experimental_split_suggestion"] == role and row["class_id"] == class_id
                }
            )
            for class_id in CLASSES
        }
        for role in ROLES
    }


def collect_pageid_roles(value: object) -> dict[int, str]:
    found: dict[int, str] = {}

    def visit(item: object) -> None:
        if isinstance(item, dict):
            pageid = item.get("pageid")
            role = item.get("role", item.get("experimental_split_suggestion"))
            if isinstance(pageid, int) and pageid in EXPECTED_NEW_ROLE and isinstance(role, str):
                if pageid in found and found[pageid] != role:
                    raise AuditError(f"machine evidence has conflicting roles for pageid {pageid}")
                found[pageid] = role
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return found


def expected_protected_receipt() -> dict[str, Any]:
    snapshot = {
        "formal_decision_journal_sha256": FORMAL_JOURNAL_SHA256,
        "formal_human_decisions_tree_sha256": FORMAL_HUMAN_TREE_SHA256,
        "protected_tree_sha256": dict(RECEIPT_PROTECTED_TREE_SHA256),
    }
    return {"before": snapshot, "after": snapshot, "unchanged": True}


def expected_frozen_v2_receipt() -> dict[str, Any]:
    return {
        "independent_audit_path": "evidence/rootscope_machine_curated_provisional_v2_audit.json",
        "independent_audit_sha256": REQUIRED_UPSTREAM_EVIDENCE[
            "evidence/rootscope_machine_curated_provisional_v2_audit.json"
        ],
        "manifest_sha256": FROZEN_V2_MANIFEST_SHA256,
        "path": "datasets/rootscope_machine_curated_provisional_v2",
        "receipt_sha256": FROZEN_V2_RECEIPT_SHA256,
        "sha256sums_sha256": FROZEN_V2_SUMS_SHA256,
        "tree_sha256_after": EXPECTED_PROTECTED_TREE_SHA256[
            "datasets/rootscope_machine_curated_provisional_v2"
        ],
        "tree_sha256_before": EXPECTED_PROTECTED_TREE_SHA256[
            "datasets/rootscope_machine_curated_provisional_v2"
        ],
        "unchanged": True,
    }


def validate_machine_evidence(
    evidence: Mapping[str, Any],
    *,
    screen_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    require(set(evidence) == EXPECTED_EVIDENCE_KEYS, "machine evidence top-level key set differs")
    require(evidence.get("schema_version") == EVIDENCE_SCHEMA, "machine evidence schema mismatch")
    require(evidence.get("status") == STATUS, "machine evidence status mismatch")
    require_fail_closed(evidence, "machine_visual_review_evidence")
    for key in ("human_reviewed", "human_label", "data_authority"):
        require(evidence.get(key) is False, f"machine evidence {key} must be false")
    require(evidence.get("all_selected_records_machine_screened") is True, "not all selected records machine screened")
    require(evidence.get("dual_machine_review_completed") is True, "E4 dual machine review not completed")
    require(evidence.get("dual_machine_review_scope") == "E4_SELECTED_ONLY", "dual review scope exceeds E4")
    require(evidence.get("independent_machine_review_count") == 2, "E4 independent review count is not two")
    require(evidence.get("root_machine_adjudicated") is True, "E4 root machine adjudication missing")
    require(
        evidence.get("root_machine_adjudication_scope") == "E4_SELECTED_ONLY",
        "root machine adjudication scope exceeds E4",
    )
    require(evidence.get("selected_pageids") == list(EXPECTED_NEW_ROLE), "machine evidence selected IDs/order mismatch")
    require(collect_pageid_roles(evidence) == EXPECTED_NEW_ROLE, "machine evidence new roles mismatch")
    override = evidence.get("role_override")
    require(isinstance(override, dict), "machine evidence role override missing")
    require(override.get("pageid") == 28135991, "machine evidence override pageid mismatch")
    require(override.get("from") == TRAIN_ROLE and override.get("to") == VAL_ROLE, "machine evidence override role mismatch")

    records = evidence.get("selected_records")
    require(isinstance(records, list) and len(records) == 5, "machine evidence must contain five selected records")
    by_id = index_pageids(records, "machine evidence selected records")
    require(set(by_id) == set(EXPECTED_NEW_ROLE), "machine evidence selected record set mismatch")
    for pageid, record in by_id.items():
        source_key = EXPECTED_NEW_SOURCE[pageid]
        require(record.get("class_id") == "young_tree", f"machine evidence {pageid} class mismatch")
        require(record.get("source_dataset") == source_key, f"machine evidence {pageid} source mismatch")
        require(record.get("experimental_split_suggestion") == EXPECTED_NEW_ROLE[pageid], f"machine evidence {pageid} role mismatch")
        require(record.get("machine_screened") is True, f"machine evidence {pageid} not screened")
        source_screen = screen_rows[pageid]
        require(source_screen.get("decision") == "SELECT", f"upstream screen did not SELECT {pageid}")
        require(
            record.get("screen_record_sha256")
            == sha256_bytes(canonical_json(source_screen).encode("utf-8")),
            f"machine evidence {pageid} screen-record SHA mismatch",
        )
        if source_key == "E3":
            require(record.get("review_scope") == "E3_MACHINE_SCREEN_ONLY", "E3 review scope inflated")
            require(record.get("dual_machine_reviewed") is False, "E3 falsely marked dual-machine-reviewed")
            require(record.get("dual_machine_review_completed") is False, "E3 falsely marked dual review completed")
            require(record.get("root_machine_adjudicated") is False, "E3 falsely marked root-adjudicated")
        else:
            require(
                record.get("review_scope") == "E4_DUAL_MACHINE_REVIEW_ROOT_ADJUDICATION",
                f"E4 review scope mismatch for {pageid}",
            )
            require(record.get("dual_machine_reviewed") is True, f"E4 {pageid} lacks dual review")
            require(record.get("dual_machine_review_completed") is True, f"E4 {pageid} dual review incomplete")
            require(record.get("root_machine_adjudicated") is True, f"E4 {pageid} lacks root adjudication")


def parse_split_roles(split: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    roles = split.get("roles")
    if isinstance(roles, dict):
        require(set(roles) == set(ROLES), "split roles differ from frozen role set")
        iterator = ((role, item) for role, items in roles.items() for item in items)
    else:
        records = split.get("records")
        require(isinstance(records, list), "split has neither roles nor records")
        iterator = ((str(item.get("role")), item) for item in records if isinstance(item, dict))
    for role, item in iterator:
        require(role in ROLES and isinstance(item, dict), "invalid split role item")
        asset = item.get("asset")
        require(isinstance(asset, str) and asset not in result, "split has missing/duplicate asset")
        result[asset] = role
    return result


def current_protected_snapshot(workspace: Path) -> dict[str, str]:
    current: dict[str, str] = {}
    for relative, expected in EXPECTED_PROTECTED_TREE_SHA256.items():
        actual = tree_sha256((workspace / relative).resolve(strict=True))
        require(actual == expected, f"protected input changed from independent baseline: {relative}")
        current[relative] = actual
    return current


def audit(workspace: Path, pack: Path) -> dict[str, Any]:
    workspace = workspace.resolve(strict=True)
    pack = pack.resolve(strict=True)
    expected_pack = (workspace / "datasets" / PACK_NAME).resolve(strict=True)
    require(pack == expected_pack and not pack.is_symlink(), f"audit target must be exactly {expected_pack}")
    pack_tree_before = tree_sha256(pack)
    checks: list[str] = []

    def checked(condition: bool, name: str) -> None:
        require(condition, name)
        checks.append(name)

    required = {
        "manifest.jsonl",
        "source_decision_manifest.jsonl",
        "experimental_split_suggestion.json",
        "machine_visual_review_evidence.json",
        "ATTRIBUTION.md",
        "README.md",
        "receipt.json",
        "SHA256SUMS",
    }
    checked(all((pack / name).is_file() for name in required), "REQUIRED_TOP_LEVEL_FILES_PRESENT")

    receipt = load_json(pack / "receipt.json")
    rows = load_jsonl(pack / "manifest.jsonl")
    decisions = load_jsonl(pack / "source_decision_manifest.jsonl")
    split = load_json(pack / "experimental_split_suggestion.json")
    evidence = load_json(pack / "machine_visual_review_evidence.json")
    sums = parse_sha256sums(pack / "SHA256SUMS")

    actual_non_sum_files = {
        path.relative_to(pack).as_posix()
        for path in pack.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    checked(set(sums) == actual_non_sum_files, "SHA256SUMS_EXACTLY_COVERS_ALL_NON_SUM_FILES")
    checked(
        all(sha256_file(safe_file(pack, relative)) == digest for relative, digest in sums.items()),
        "SHA256SUMS_ALL_DIGESTS_MATCH",
    )

    checked(set(receipt) == EXPECTED_RECEIPT_KEYS, "RECEIPT_TOP_LEVEL_KEY_SET_FROZEN")
    checked(receipt.get("schema_version") == RECEIPT_SCHEMA, "RECEIPT_SCHEMA_FROZEN_V3")
    checked(receipt.get("status") == STATUS, "RECEIPT_STATUS_MACHINE_ONLY_V3")
    require_fail_closed(receipt, "receipt")
    checked(receipt.get("formal_a1_dataset") is False, "RECEIPT_FORMAL_A1_FALSE")
    checked(receipt.get("experimental_training_switch_required") is True, "RECEIPT_EXPERIMENTAL_SWITCH_REQUIRED")
    checked(receipt.get("formal_split_assigned") is False, "RECEIPT_FORMAL_SPLIT_FALSE")
    builder_path = workspace / "tools/dataset/build_machine_curated_provisional_v3.py"
    checked(
        builder_path.is_file() and receipt.get("implementation_sha256") == sha256_file(builder_path),
        "RECEIPT_IMPLEMENTATION_SHA_MATCH",
    )
    checked(receipt.get("manifest_sha256") == sha256_file(pack / "manifest.jsonl"), "RECEIPT_MANIFEST_SHA_MATCH")
    checked(
        receipt.get("source_decision_manifest_sha256") == sha256_file(pack / "source_decision_manifest.jsonl"),
        "RECEIPT_SOURCE_DECISION_SHA_MATCH",
    )
    checked(
        receipt.get("machine_visual_review_evidence_sha256")
        == sha256_file(pack / "machine_visual_review_evidence.json"),
        "RECEIPT_MACHINE_VISUAL_EVIDENCE_SHA_MATCH",
    )
    checked(
        receipt.get("machine_visual_review_evidence_path") == "machine_visual_review_evidence.json",
        "RECEIPT_MACHINE_VISUAL_EVIDENCE_PATH_FIXED",
    )
    if "experimental_split_suggestion_sha256" in receipt:
        checked(
            receipt["experimental_split_suggestion_sha256"]
            == sha256_file(pack / "experimental_split_suggestion.json"),
            "RECEIPT_SPLIT_SUGGESTION_SHA_MATCH",
        )
    checked(
        receipt.get("payload_root_sha256_before_receipt") == payload_root_sha256(pack),
        "PAYLOAD_ROOT_REBUILDS_EXCLUDING_RECEIPT_AND_SUMS",
    )

    source_indexes: dict[str, dict[int, dict[str, Any]]] = {}
    for source_key, dataset_name in SOURCE_DATASETS.items():
        manifest = workspace / "datasets" / dataset_name / "manifest.jsonl"
        checked(
            sha256_file(manifest) == EXPECTED_SOURCE_MANIFEST_SHA256[source_key],
            f"SOURCE_{source_key}_MANIFEST_FROZEN",
        )
        source_indexes[source_key] = index_pageids(load_jsonl(manifest), f"source {source_key}")
    checked(
        receipt.get("source_manifest_sha256") == EXPECTED_SOURCE_MANIFEST_SHA256,
        "RECEIPT_SOURCE_MANIFEST_MAP_EXACT_E0_TO_E4",
    )
    e3_screen_path = (
        workspace
        / "datasets/desert_plants_young_tree_reacquisition_e3/review/"
        "machine_visual_screen_v1/decisions.jsonl"
    )
    e4_screen_path = (
        workspace
        / "datasets/desert_plants_young_tree_category_reacquisition_e4/review/"
        "machine_visual_screen_v1/manifest.jsonl"
    )
    screen_rows = {
        **index_pageids(load_jsonl(e3_screen_path), "E3 machine visual screen"),
        **index_pageids(load_jsonl(e4_screen_path), "E4 machine visual screen"),
    }

    protected = current_protected_snapshot(workspace)
    human_root = workspace / "datasets/desert_plants_wikimedia_staging_e0/review/human_decisions"
    journal = human_root / "decision_journal.jsonl"
    checked(tree_sha256(human_root) == FORMAL_HUMAN_TREE_SHA256, "FORMAL_HUMAN_TREE_FROZEN_EMPTY")
    checked(sha256_file(journal) == FORMAL_JOURNAL_SHA256, "FORMAL_JOURNAL_FROZEN_EMPTY")
    checked(
        sha256_file(workspace / "datasets/rootscope_machine_curated_provisional_v1/manifest.jsonl")
        == FROZEN_V1_MANIFEST_SHA256,
        "FROZEN_V1_MANIFEST_MATCH",
    )
    v2_root = workspace / "datasets/rootscope_machine_curated_provisional_v2"
    checked(sha256_file(v2_root / "manifest.jsonl") == FROZEN_V2_MANIFEST_SHA256, "FROZEN_V2_MANIFEST_MATCH")
    checked(sha256_file(v2_root / "receipt.json") == FROZEN_V2_RECEIPT_SHA256, "FROZEN_V2_RECEIPT_MATCH")
    checked(sha256_file(v2_root / "SHA256SUMS") == FROZEN_V2_SUMS_SHA256, "FROZEN_V2_SUMS_MATCH")
    checked(receipt.get("protected_inputs") == expected_protected_receipt(), "RECEIPT_PROTECTED_INPUTS_EXACT")
    checked(receipt.get("frozen_v2") == expected_frozen_v2_receipt(), "RECEIPT_FROZEN_V2_EXACT")

    checked(len(rows) == EXPECTED_TOTAL, "MANIFEST_EXACTLY_78_RECORDS")
    by_pageid = index_pageids(rows, "v3 manifest")
    checked(len(by_pageid) == EXPECTED_TOTAL, "MANIFEST_PAGEIDS_UNIQUE")
    v2_rows = index_pageids(load_jsonl(v2_root / "manifest.jsonl"), "frozen v2")
    checked(set(by_pageid) == set(v2_rows) | set(EXPECTED_NEW_ROLE), "V3_EQUALS_V2_PLUS_EXACT_FIVE_NEW_IDS")

    split_assets = parse_split_roles(split)
    checked(split.get("schema_version") == SPLIT_SCHEMA, "SPLIT_SCHEMA_FROZEN_V3")
    checked(split.get("status") == STATUS, "SPLIT_STATUS_FROZEN_V3")
    require_fail_closed(split, "split")
    checked(split.get("formal_split_assignment") is False, "SPLIT_IS_EXPERIMENTAL_NOT_FORMAL")
    checked(split.get("experimental_training_switch_required") is True, "SPLIT_EXPERIMENTAL_SWITCH_REQUIRED")

    seen_assets: set[str] = set()
    seen_filenames: set[str] = set()
    seen_source_groups: set[str] = set()
    seen_content_sha: set[str] = set()
    for index, row in enumerate(rows):
        location = f"manifest[{index}]"
        checked(row.get("schema_version") == ASSET_SCHEMA, f"{location}_ASSET_SCHEMA_V3")
        checked(row.get("status") == STATUS, f"{location}_STATUS_V3")
        require_fail_closed(row, location, record=True)
        class_id = row.get("class_id")
        role = row.get("experimental_split_suggestion")
        checked(class_id in CLASSES, f"{location}_CLASS_ALLOWED")
        checked(role in ROLES, f"{location}_ROLE_ALLOWED")
        asset = row.get("asset")
        filename = row.get("filename")
        checked(isinstance(asset, str) and asset not in seen_assets, f"{location}_ASSET_UNIQUE")
        checked(isinstance(filename, str) and filename not in seen_filenames, f"{location}_FILENAME_UNIQUE")
        seen_assets.add(str(asset)); seen_filenames.add(str(filename))
        checked(split_assets.get(str(asset)) == role, f"{location}_SPLIT_ROLE_BOUND")
        image_path = safe_file(pack, filename)
        image_sha = sha256_file(image_path)
        checked(row.get("copied_image_sha256") == image_sha, f"{location}_COPIED_IMAGE_SHA")
        checked(row.get("source_image_sha256") == image_sha, f"{location}_SOURCE_IMAGE_SHA")
        checked(str(asset).endswith(f"@sha256:{image_sha}"), f"{location}_ASSET_SHA_SUFFIX")
        checked(sums.get(str(filename)) == image_sha, f"{location}_IMAGE_IN_SHA256SUMS")
        checked(str(row.get("source_group")) not in seen_source_groups, f"{location}_SOURCE_GROUP_UNIQUE")
        checked(image_sha not in seen_content_sha, f"{location}_CONTENT_SHA_UNIQUE")
        seen_source_groups.add(str(row.get("source_group"))); seen_content_sha.add(image_sha)
        source_key = row.get("source_dataset")
        checked(source_key in SOURCE_DATASETS, f"{location}_SOURCE_KEY_ALLOWED")
        checked(row.get("source_dataset_name") == SOURCE_DATASETS[str(source_key)], f"{location}_SOURCE_NAME_BOUND")
        checked(
            row.get("source_manifest_sha256") == EXPECTED_SOURCE_MANIFEST_SHA256[str(source_key)],
            f"{location}_SOURCE_MANIFEST_SHA_BOUND",
        )
        source_row = source_indexes[str(source_key)].get(int(row["pageid"]))
        checked(source_row is not None, f"{location}_PAGEID_IN_SOURCE")
        assert source_row is not None
        checked(
            row.get("source_record_sha256") == sha256_bytes(canonical_json(source_row).encode("utf-8")),
            f"{location}_SOURCE_RECORD_SHA_BOUND",
        )
        for field in ("creator_group", "source_group"):
            checked(row.get(field) == source_row.get(field), f"{location}_{field.upper()}_BOUND")
        checked(row.get("source_image_path") == source_row.get("filename"), f"{location}_SOURCE_IMAGE_PATH_BOUND")
        source_path = safe_file(workspace / "datasets" / SOURCE_DATASETS[str(source_key)], source_row.get("filename"))
        source_sha = sha256_file(source_path)
        checked(source_row.get("download_sha256") == source_sha == image_sha, f"{location}_SOURCE_BYTES_BOUND")
        checked(source_row.get("dhash64_algorithm") == DHASH_ALGORITHM, f"{location}_SOURCE_DHASH_ALGORITHM")
        actual_dhash = image_dhash64(source_path)
        checked(source_row.get("dhash64") == actual_dhash, f"{location}_SOURCE_DHASH_RECOMPUTED")
        checked(row.get("dhash64") == actual_dhash, f"{location}_PACK_DHASH_BOUND")
        with Image.open(image_path) as opened:
            opened.verify()
        with Image.open(image_path) as opened:
            checked(opened.width >= 32 and opened.height >= 32, f"{location}_IMAGE_DIMENSIONS_VALID")
        if int(row["pageid"]) in v2_rows:
            old = v2_rows[int(row["pageid"])]
            for field in (
                "class_id", "creator_group", "source_group", "source_dataset", "source_dataset_name",
                "source_image_path", "source_image_sha256", "source_manifest_sha256", "source_record_sha256",
                "dhash64", "copied_image_sha256", "filename", "print_holdout_candidate",
            ):
                checked(row.get(field) == old.get(field), f"{location}_INHERITED_V2_{field.upper()}")
            expected_role = EXPECTED_ROLE_OVERRIDE.get(int(row["pageid"]), (old["experimental_split_suggestion"], old["experimental_split_suggestion"]))[1]
            checked(role == expected_role, f"{location}_INHERITED_OR_OVERRIDDEN_ROLE")
            checked(row.get("v3_origin") == "INHERITED_FROZEN_V2", f"{location}_V3_ORIGIN_V2")
            if "inherited_v2_record_sha256" in row:
                checked(
                    row["inherited_v2_record_sha256"] == sha256_bytes(canonical_json(old).encode("utf-8")),
                    f"{location}_INHERITED_V2_RECORD_SHA",
                )
        else:
            pageid = int(row["pageid"])
            checked(row.get("class_id") == "young_tree", f"{location}_NEW_CLASS_YOUNG_TREE")
            checked(role == EXPECTED_NEW_ROLE[pageid], f"{location}_NEW_ROLE_FROZEN")
            checked(source_key == EXPECTED_NEW_SOURCE[pageid], f"{location}_NEW_SOURCE_FROZEN")
            checked(
                row.get("v3_origin") == f"{source_key}_MACHINE_VISUAL_SELECTED",
                f"{location}_NEW_ORIGIN_FROZEN",
            )
            screen_record = screen_rows[pageid]
            checked(screen_record.get("decision") == "SELECT", f"{location}_UPSTREAM_SCREEN_SELECTED")
            checked(
                row.get("visual_screen_record_sha256")
                == sha256_bytes(canonical_json(screen_record).encode("utf-8")),
                f"{location}_VISUAL_SCREEN_RECORD_SHA_BOUND",
            )
            checked(row.get("biological_age_verified") is False, f"{location}_BIOLOGICAL_AGE_NOT_HUMAN_VERIFIED")

    checked(set(split_assets) == seen_assets, "SPLIT_ASSET_SET_EQUALS_MANIFEST")
    split_records = split.get("records")
    checked(isinstance(split_records, list) and len(split_records) == EXPECTED_TOTAL, "SPLIT_EXACTLY_78_RECORDS")
    split_by_asset = {str(item.get("asset")): item for item in split_records if isinstance(item, dict)}
    checked(len(split_by_asset) == EXPECTED_TOTAL, "SPLIT_ASSETS_UNIQUE")
    for row in rows:
        split_item = split_by_asset[str(row["asset"])]
        for field in ("pageid", "class_id", "creator_group", "source_group"):
            checked(split_item.get(field) == row.get(field), f"SPLIT_{row['pageid']}_{field.upper()}_BOUND")
        checked(split_item.get("role") == row.get("experimental_split_suggestion"), f"SPLIT_{row['pageid']}_ROLE_BOUND")
    image_files = {path.relative_to(pack).as_posix() for path in (pack / "images").rglob("*") if path.is_file()}
    checked(image_files == seen_filenames, "IMAGES_DIRECTORY_EXACTLY_MATCHES_MANIFEST")

    class_counts = Counter(str(row["class_id"]) for row in rows)
    role_counts = Counter(str(row["experimental_split_suggestion"]) for row in rows)
    checked(dict(class_counts) == EXPECTED_CLASS_COUNTS, "EXACT_CLASS_COUNTS")
    checked(dict(role_counts) == EXPECTED_ROLE_COUNTS, "EXACT_ROLE_COUNTS")
    checked(role_class_counts(rows) == EXPECTED_ROLE_CLASS_COUNTS, "EXACT_ROLE_CLASS_COUNTS")
    checked(
        role_class_unique_counts(rows, "creator_group") == EXPECTED_ROLE_CLASS_CREATOR_COUNTS,
        "EXACT_ROLE_CLASS_CREATOR_DIVERSITY",
    )
    checked(
        role_class_unique_counts(rows, "source_group") == EXPECTED_ROLE_CLASS_SOURCE_COUNTS,
        "EXACT_ROLE_CLASS_SOURCE_DIVERSITY",
    )
    checked(
        {int(row["pageid"]) for row in rows if row["experimental_split_suggestion"] == PRINT_ROLE}
        == EXPECTED_PRINT_PAGEIDS,
        "EXACT_FROZEN_PRINT_HOLDOUT_IDS",
    )

    for field in ("creator_group", "source_group", "copied_image_sha256"):
        owners: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            owners[str(row[field])].add(role_family(str(row["experimental_split_suggestion"])))
        checked(not {key: value for key, value in owners.items() if len(value) > 1}, f"NO_{field.upper()}_PARTITION_LEAKAGE")
    cross_distances = [
        dhash_distance(str(left["dhash64"]), str(right["dhash64"]))
        for left_index, left in enumerate(rows)
        for right in rows[left_index + 1 :]
        if role_family(str(left["experimental_split_suggestion"]))
        != role_family(str(right["experimental_split_suggestion"]))
    ]
    minimum_cross_dhash = min(cross_distances)
    checked(minimum_cross_dhash > 4, "NO_DHASH_DISTANCE_LE_4_ACROSS_PARTITIONS")

    checked(len(decisions) == EXPECTED_TOTAL, "SOURCE_DECISIONS_EXACTLY_78")
    decision_by_pageid = index_pageids(decisions, "v3 source decision")
    checked(set(decision_by_pageid) == set(by_pageid), "SOURCE_DECISIONS_EXACTLY_COVER_MANIFEST")
    for pageid, decision in decision_by_pageid.items():
        require_fail_closed(decision, f"decision[{pageid}]", record=True)
        checked(decision.get("schema_version") == DECISION_SCHEMA, f"DECISION_{pageid}_SCHEMA_V3")
        checked(decision.get("status") == STATUS, f"DECISION_{pageid}_STATUS_V3")
        checked(decision.get("selected") is True, f"DECISION_{pageid}_SELECTED_TRUE")
        selected = by_pageid[pageid]
        for field in (
            "class_id", "creator_group", "source_group", "source_dataset", "source_record_sha256",
            "copied_image_sha256", "experimental_split_suggestion",
        ):
            checked(decision.get(field) == selected.get(field), f"DECISION_{pageid}_{field.upper()}_BOUND")
        disposition = str(decision.get("disposition", ""))
        if pageid in v2_rows:
            checked(disposition == "INHERITED_FROZEN_V2", f"DECISION_{pageid}_INHERITED_DISPOSITION")
            checked(
                decision.get("inherited_v2_record_sha256")
                == sha256_bytes(canonical_json(v2_rows[pageid]).encode("utf-8")),
                f"DECISION_{pageid}_INHERITED_V2_RECORD_SHA",
            )
            checked("visual_screen_record_sha256" not in decision, f"DECISION_{pageid}_NO_SPURIOUS_VISUAL_SCREEN_SHA")
            if pageid == 28135991:
                override = decision.get("role_override")
                checked(
                    isinstance(override, dict)
                    and override.get("from") == TRAIN_ROLE
                    and override.get("to") == VAL_ROLE,
                    "DECISION_28135991_EXACT_ROLE_OVERRIDE",
                )
            else:
                checked("role_override" not in decision, f"DECISION_{pageid}_NO_ROLE_OVERRIDE")
        else:
            checked(disposition == "SELECTED_MACHINE_VISUAL_E3_OR_E4", f"DECISION_{pageid}_VISUAL_DISPOSITION")
            checked(decision.get("inherited_v2_record_sha256") is None, f"DECISION_{pageid}_NOT_INHERITED")
            checked(
                decision.get("visual_screen_record_sha256")
                == sha256_bytes(canonical_json(screen_rows[pageid]).encode("utf-8")),
                f"DECISION_{pageid}_VISUAL_SCREEN_RECORD_SHA",
            )

    validate_machine_evidence(evidence, screen_rows=screen_rows)
    evidence_text = canonical_json(evidence)
    checks.extend(
        [
            "MACHINE_EVIDENCE_FAIL_CLOSED",
            "MACHINE_EVIDENCE_BINDS_FIVE_NEW_IDS_AND_ROLES",
            "E3_SCOPE_MACHINE_SCREEN_ONLY_NOT_DUAL_OR_ROOT",
            "E4_SCOPE_DUAL_MACHINE_REVIEW_AND_ROOT_ADJUDICATION_ONLY",
        ]
    )
    for relative, digest in REQUIRED_UPSTREAM_EVIDENCE.items():
        checked(sha256_file(workspace / relative) == digest, f"UPSTREAM_EVIDENCE_FILE_FROZEN_{Path(relative).name}")
        checked(relative in evidence_text and digest in evidence_text, f"MACHINE_EVIDENCE_BINDS_{Path(relative).name}")
    checked("E4_two_independent_machine" in evidence_text, "MACHINE_EVIDENCE_RECORDS_E4_DUAL_MACHINE_REVIEW")
    checked("root_machine_adjudicat" in evidence_text, "MACHINE_EVIDENCE_RECORDS_E4_ROOT_ADJUDICATION")
    checked("NOT_HUMAN_REVIEWED" in evidence_text or "not_human_reviewed" in evidence_text, "MACHINE_EVIDENCE_EXPLICIT_NON_HUMAN_STATUS")

    evidence_summary = {
        "all_selected_records_machine_screened": True,
        "dual_machine_review_completed": True,
        "dual_machine_review_scope": "E4_SELECTED_ONLY",
        "human_reviewed": False,
        "root_machine_adjudicated": True,
        "root_machine_adjudication_scope": "E4_SELECTED_ONLY",
        "selected_pageids": list(EXPECTED_NEW_ROLE),
    }
    checked(
        receipt.get("machine_visual_review_evidence") == evidence_summary,
        "RECEIPT_MACHINE_EVIDENCE_SUMMARY_EXACT_AND_SCOPED_TO_E4",
    )
    checked(
        evidence.get("v2_independent_audit")
        == {
            "path": "evidence/rootscope_machine_curated_provisional_v2_audit.json",
            "selected_count": 73,
            "sha256": REQUIRED_UPSTREAM_EVIDENCE[
                "evidence/rootscope_machine_curated_provisional_v2_audit.json"
            ],
            "status": "PASS",
        },
        "MACHINE_EVIDENCE_V2_AUDIT_REFERENCE_EXACT",
    )

    expected_diversity = {
        role: {
            class_id: {
                "image_count": EXPECTED_ROLE_CLASS_COUNTS[role][class_id],
                "creator_count": EXPECTED_ROLE_CLASS_CREATOR_COUNTS[role][class_id],
                "source_count": EXPECTED_ROLE_CLASS_SOURCE_COUNTS[role][class_id],
            }
            for class_id in CLASSES
        }
        for role in (TRAIN_ROLE, VAL_ROLE)
    }
    checked(split.get("role_counts") == EXPECTED_ROLE_COUNTS, "SPLIT_ROLE_COUNTS_EXACT")
    checked(split.get("train_and_validation_diversity") == expected_diversity, "SPLIT_DIVERSITY_EXACT")
    receipt_audit = receipt.get("audit")
    checked(isinstance(receipt_audit, dict), "RECEIPT_AUDIT_PRESENT")
    assert isinstance(receipt_audit, dict)
    checked(receipt_audit.get("selected_count") == EXPECTED_TOTAL, "RECEIPT_AUDIT_SELECTED_COUNT")
    checked(receipt_audit.get("class_counts") == EXPECTED_CLASS_COUNTS, "RECEIPT_AUDIT_CLASS_COUNTS")
    checked(receipt_audit.get("experimental_role_counts") == EXPECTED_ROLE_COUNTS, "RECEIPT_AUDIT_ROLE_COUNTS")
    checked(receipt_audit.get("train_and_validation_diversity") == expected_diversity, "RECEIPT_AUDIT_DIVERSITY")
    checked(receipt_audit.get("creator_partition_leakage_count") == 0, "RECEIPT_AUDIT_CREATOR_LEAKAGE_ZERO")
    checked(receipt_audit.get("source_group_overlap_count") == 0, "RECEIPT_AUDIT_SOURCE_OVERLAP_ZERO")
    checked(receipt_audit.get("copied_sha256_overlap_count") == 0, "RECEIPT_AUDIT_CONTENT_OVERLAP_ZERO")
    checked(
        receipt_audit.get("cross_partition_minimum_dhash64_distance") == minimum_cross_dhash,
        "RECEIPT_AUDIT_CROSS_PARTITION_DHASH_REBUILDS",
    )

    checked(receipt.get("all_split_targets_met") is True, "ALL_SPLIT_TARGETS_MET_TRUE")
    pack_tree_after = tree_sha256(pack)
    checked(pack_tree_after == pack_tree_before, "AUDIT_DID_NOT_MUTATE_PACK")
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "PASS",
        "pack": pack.relative_to(workspace).as_posix(),
        "check_count": len(checks),
        "pass_count": len(checks),
        "failure_count": 0,
        "checks_passed": checks,
        "checks_failed": [],
        "record_count": len(rows),
        "class_counts": {name: class_counts[name] for name in CLASSES},
        "role_counts": {name: role_counts[name] for name in ROLES},
        "role_class_counts": role_class_counts(rows),
        "role_class_creator_counts": role_class_unique_counts(rows, "creator_group"),
        "role_class_source_counts": role_class_unique_counts(rows, "source_group"),
        "minimum_cross_partition_dhash64_distance": minimum_cross_dhash,
        "manifest_sha256": sha256_file(pack / "manifest.jsonl"),
        "receipt_sha256": sha256_file(pack / "receipt.json"),
        "sha256sums_sha256": sha256_file(pack / "SHA256SUMS"),
        "pack_tree_sha256": pack_tree_after,
        "protected_input_tree_sha256": protected,
        "formal_human_tree_sha256": FORMAL_HUMAN_TREE_SHA256,
        "formal_decision_journal_sha256": FORMAL_JOURNAL_SHA256,
        "formal_authority": False,
        "human_reviewed": False,
        "training_eligible": False,
        "dataset_mutated_by_audit": False,
    }


def failure_report(workspace: Path, pack: Path, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "FAIL",
        "pack": str(pack),
        "check_count": 1,
        "pass_count": 0,
        "failure_count": 1,
        "checks_passed": [],
        "checks_failed": [str(error)],
        "formal_authority": False,
        "human_reviewed": False,
        "training_eligible": False,
        "dataset_mutated_by_audit": False,
        "workspace": str(workspace),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--pack", type=Path, default=workspace / "datasets" / PACK_NAME)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit(args.workspace, args.pack)
    except (AuditError, FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
        report = failure_report(args.workspace, args.pack, error)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        output = args.output.resolve(strict=False)
        pack = args.pack.resolve(strict=False)
        try:
            output.relative_to(pack)
        except ValueError:
            pass
        else:
            raise AuditError("audit report must be written outside the audited pack")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")
    sys.stdout.write(text)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
