"""Build RootScope's fail-closed machine-curated provisional v3 pack.

V3 is an experimental derivative of frozen provisional v2.  It preserves all
73 v2 assets, moves only pageid 28135991 from the experimental train role to
the experimental validation role, and adds the five explicitly adjudicated E3
and E4 young-tree candidates.  No human, rights, split, training, print, model,
or data-lock authority is created by this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageOps


OUTPUT_NAME = "rootscope_machine_curated_provisional_v3"
V1_NAME = "rootscope_machine_curated_provisional_v1"
V2_NAME = "rootscope_machine_curated_provisional_v2"
STATUS = "MACHINE_CURATED_EXPERIMENTAL_V3_ONLY_NOT_HUMAN_REVIEWED_NOT_A1_NOT_DATA_LOCKED"
ASSET_SCHEMA = "rootscope.machine_curated_provisional_asset.v3"
RECEIPT_SCHEMA = "rootscope.machine_curated_provisional_receipt.v3"
DECISION_SCHEMA = "rootscope.machine_curated_source_decision.v3"
EVIDENCE_SCHEMA = "rootscope.machine_visual_review_evidence.v3"

SOURCE_DATASETS = {
    "E0": "desert_plants_wikimedia_staging_e0",
    "E1": "desert_plants_whole_plant_reacquisition_e1",
    "E2": "desert_plants_young_tree_reacquisition_e2",
    "E3": "desert_plants_young_tree_reacquisition_e3",
    "E4": "desert_plants_young_tree_category_reacquisition_e4",
}
SOURCE_MANIFEST_NAMES = {key: "manifest.jsonl" for key in SOURCE_DATASETS}

TRAIN_ROLE = "EXPERIMENTAL_TRAIN_SUGGESTION"
VAL_ROLE = "EXPERIMENTAL_VAL_SUGGESTION"
PRINT_ROLE = "PRINT_DEMO_HOLDOUT_NOT_TRAIN"
CREATOR_HOLDOUT_ROLE = "CREATOR_GROUP_HOLDOUT_NOT_TRAIN"
ALLOWED_ROLES = {TRAIN_ROLE, VAL_ROLE, PRINT_ROLE, CREATOR_HOLDOUT_ROLE}

ROLE_OVERRIDE_PAGEID = 28135991
NEW_ROLES = {
    6191581: ("E3", TRAIN_ROLE),
    92774234: ("E4", TRAIN_ROLE),
    122973026: ("E4", TRAIN_ROLE),
    180772202: ("E4", VAL_ROLE),
    184915021: ("E4", VAL_ROLE),
}
EXPECTED_ROLE_COUNTS = {
    TRAIN_ROLE: 55,
    VAL_ROLE: 9,
    PRINT_ROLE: 6,
    CREATOR_HOLDOUT_ROLE: 8,
}
EXPECTED_CLASS_COUNTS = {
    "grass_clump": 15,
    "low_shrub": 19,
    "young_tree": 13,
    "unknown": 31,
}
EXPECTED_DIVERSITY = {
    TRAIN_ROLE: {
        "grass_clump": (8, 6, 8),
        "low_shrub": (13, 8, 13),
        "young_tree": (5, 5, 5),
        "unknown": (29, 29, 29),
    },
    VAL_ROLE: {
        "grass_clump": (3, 2, 3),
        "low_shrub": (2, 2, 2),
        "young_tree": (2, 2, 2),
        "unknown": (2, 2, 2),
    },
}
AUTHORITY_KEYS = (
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


class V3BuildError(RuntimeError):
    """Fail-closed v3 construction error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pretty_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def write_json(path: Path, value: object) -> None:
    write_text(path, pretty_json(value))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise V3BuildError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise V3BuildError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise V3BuildError(f"non-object JSON at {path}:{line_number}")
            rows.append(value)
    return rows


def indexed_by_pageid(rows: Sequence[dict[str, Any]], *, label: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        pageid = row.get("pageid")
        if isinstance(pageid, bool) or not isinstance(pageid, int):
            raise V3BuildError(f"{label} has invalid pageid {pageid!r}")
        if pageid in result:
            raise V3BuildError(f"{label} has duplicate pageid {pageid}")
        result[pageid] = row
    return result


def tree_sha256(root: Path) -> str:
    rows: list[str] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        rows.append(f"{path.relative_to(root).as_posix()}\0{sha256_file(path)}\n")
    return sha256_bytes("".join(rows).encode("utf-8"))


def safe_child(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise V3BuildError(f"unsafe relative path {relative_value!r}")
    root = root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise V3BuildError(f"path escapes source root: {relative_value!r}") from error
    if not candidate.is_file():
        raise V3BuildError(f"expected source file: {candidate}")
    return candidate


def false_authority() -> dict[str, bool]:
    return {key: False for key in AUTHORITY_KEYS}


def status_fields() -> dict[str, Any]:
    return {
        "authority": false_authority(),
        "data_locked": False,
        "formal_a1_dataset": False,
        "formal_split_assigned": False,
        "human_reviewed": False,
        "machine_curated_only": True,
        "print_eligible": False,
        "rights_approved": False,
        "split": "UNASSIGNED_DO_NOT_TRAIN",
        "status": STATUS,
        "training_eligible": False,
        "experimental_training_switch_required": True,
    }


def source_record_digest(row: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(dict(row)).encode("utf-8"))


def image_dhash64(path: Path) -> str:
    with Image.open(path) as source:
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")
    bits = 0
    for row in range(8):
        source_y = min(image.height - 1, int(((row + 0.5) * image.height) // 8))
        for column in range(8):
            left_x = min(image.width - 1, int(((column + 0.5) * image.width) // 9))
            right_x = min(image.width - 1, int(((column + 1.5) * image.width) // 9))
            left = image.getpixel((left_x, source_y))
            right = image.getpixel((right_x, source_y))
            left_luma = 299 * left[0] + 587 * left[1] + 114 * left[2]
            right_luma = 299 * right[0] + 587 * right[1] + 114 * right[2]
            bits = (bits << 1) | int(left_luma > right_luma)
    return f"{bits:016x}"


def hamming64(left: str, right: str) -> int:
    if len(left) != 16 or len(right) != 16:
        raise V3BuildError("dhash64 must contain 16 hexadecimal digits")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def verify_sha256sums(root: Path) -> None:
    sums_path = root / "SHA256SUMS"
    expected: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(expected) != actual_files:
        raise V3BuildError("frozen v2 SHA256SUMS coverage mismatch")
    for relative, digest in expected.items():
        if sha256_file(safe_child(root, relative)) != digest:
            raise V3BuildError(f"frozen v2 SHA256SUMS mismatch: {relative}")


def protected_snapshot(workspace: Path) -> dict[str, Any]:
    datasets = workspace / "datasets"
    roots = {
        f"datasets/{V1_NAME}": datasets / V1_NAME,
        f"datasets/{V2_NAME}": datasets / V2_NAME,
        **{f"datasets/{name}": datasets / name for name in SOURCE_DATASETS.values()},
    }
    hashes = {name: tree_sha256(path.resolve(strict=True)) for name, path in sorted(roots.items())}
    human_root = (datasets / SOURCE_DATASETS["E0"] / "review" / "human_decisions").resolve(strict=True)
    journal = human_root / "decision_journal.jsonl"
    return {
        "protected_tree_sha256": hashes,
        "formal_human_decisions_tree_sha256": tree_sha256(human_root),
        "formal_decision_journal_sha256": sha256_file(journal),
    }


def validate_v2(workspace: Path) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    root = (workspace / "datasets" / V2_NAME).resolve(strict=True)
    verify_sha256sums(root)
    receipt = load_json(root / "receipt.json")
    rows = load_jsonl(root / "manifest.jsonl")
    audit_path = (workspace / "evidence" / "rootscope_machine_curated_provisional_v2_audit.json").resolve(strict=True)
    audit = load_json(audit_path)
    if receipt.get("schema_version") != "rootscope.machine_curated_provisional_receipt.v2":
        raise V3BuildError("unexpected frozen v2 receipt schema")
    if receipt.get("manifest_sha256") != sha256_file(root / "manifest.jsonl"):
        raise V3BuildError("frozen v2 receipt/manifest hash mismatch")
    if len(rows) != 73 or audit.get("status") != "PASS" or audit.get("failure_count") != 0:
        raise V3BuildError("frozen v2 or its independent audit is not passing")
    if audit.get("selected_count") != 73 or audit.get("check_count") != audit.get("pass_count"):
        raise V3BuildError("frozen v2 independent audit coverage mismatch")
    index = indexed_by_pageid(rows, label="frozen v2 manifest")
    if set(index) & set(NEW_ROLES):
        raise V3BuildError("new v3 pageid already exists in frozen v2")
    override = index.get(ROLE_OVERRIDE_PAGEID)
    if override is None or override.get("experimental_split_suggestion") != TRAIN_ROLE:
        raise V3BuildError("v3 role override source is not frozen v2 train")
    if Counter(str(row.get("experimental_split_suggestion")) for row in rows) != Counter(
        {TRAIN_ROLE: 53, VAL_ROLE: 6, PRINT_ROLE: 6, CREATOR_HOLDOUT_ROLE: 8}
    ):
        raise V3BuildError("frozen v2 role counts changed")
    return root, rows, {
        "audit_path": audit_path,
        "audit_sha256": sha256_file(audit_path),
        "manifest_sha256": sha256_file(root / "manifest.jsonl"),
        "receipt_sha256": sha256_file(root / "receipt.json"),
        "sha256sums_sha256": sha256_file(root / "SHA256SUMS"),
    }


def load_machine_screens(workspace: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    datasets = workspace / "datasets"
    e3_root = datasets / SOURCE_DATASETS["E3"]
    e4_root = datasets / SOURCE_DATASETS["E4"]
    e3_screen = e3_root / "review" / "machine_visual_screen_v1"
    e4_screen = e4_root / "review" / "machine_visual_screen_v1"
    e3_receipt = load_json(e3_screen / "receipt.json")
    e4_receipt = load_json(e4_screen / "receipt.json")
    e3_decisions_path = e3_screen / "decisions.jsonl"
    e4_manifest_path = e4_screen / "manifest.jsonl"
    e3_rows = load_jsonl(e3_decisions_path)
    e4_rows = load_jsonl(e4_manifest_path)
    if sha256_file(e3_decisions_path) != e3_receipt.get("decisions_sha256"):
        raise V3BuildError("E3 screen decision hash mismatch")
    if sha256_file(e4_manifest_path) != e4_receipt.get("screen_manifest_sha256"):
        raise V3BuildError("E4 screen manifest hash mismatch")
    if (e4_screen / "decisions.jsonl").read_bytes() != e4_manifest_path.read_bytes():
        raise V3BuildError("E4 decisions alias differs from screen manifest")
    e3_selected = {int(row["pageid"]) for row in e3_rows if row.get("decision") == "SELECT"}
    e4_selected = {int(row["pageid"]) for row in e4_rows if row.get("decision") == "SELECT"}
    if e3_selected != {6191581} or e4_selected != {92774234, 122973026, 180772202, 184915021}:
        raise V3BuildError("machine-screen SELECT set differs from frozen v3 contract")
    review = e4_receipt.get("review_pipeline", {})
    if not (
        review.get("independent_machine_reviews_completed") is True
        and review.get("independent_machine_review_count") == 2
        and review.get("root_machine_adjudicated") is True
        and review.get("human_review_authority") is False
    ):
        raise V3BuildError("E4 dual-machine/root-adjudication evidence is incomplete")
    decisions = indexed_by_pageid(e3_rows + e4_rows, label="E3/E4 screen")
    for pageid in NEW_ROLES:
        decision = decisions[pageid]
        if decision.get("decision") != "SELECT" or decision.get("human_reviewed") is not False:
            raise V3BuildError(f"pageid {pageid} is not a fail-closed machine SELECT")
        if decision.get("training_eligible") is not False or decision.get("rights_approved") is not False:
            raise V3BuildError(f"pageid {pageid} screen overclaims authority")
    evidence = {
        "E3": {
            "screen_receipt_path": e3_screen.joinpath("receipt.json").relative_to(workspace).as_posix(),
            "screen_receipt_sha256": sha256_file(e3_screen / "receipt.json"),
            "screen_manifest_path": e3_decisions_path.relative_to(workspace).as_posix(),
            "screen_manifest_sha256": sha256_file(e3_decisions_path),
            "selected_pageids": [6191581],
        },
        "E4": {
            "screen_receipt_path": e4_screen.joinpath("receipt.json").relative_to(workspace).as_posix(),
            "screen_receipt_sha256": sha256_file(e4_screen / "receipt.json"),
            "screen_manifest_path": e4_manifest_path.relative_to(workspace).as_posix(),
            "screen_manifest_sha256": sha256_file(e4_manifest_path),
            "adjudication_contract_path": e4_receipt["adjudication_contract_path"],
            "adjudication_contract_sha256": e4_receipt["adjudication_contract_sha256"],
            "selected_pageids": [92774234, 122973026, 180772202, 184915021],
        },
    }
    return decisions, evidence


def copy_v2_record(row: dict[str, Any], *, v2_root: Path, staging: Path, v2_manifest_sha256: str) -> dict[str, Any]:
    source = safe_child(v2_root, str(row["filename"]))
    digest = sha256_file(source)
    if digest != row.get("copied_image_sha256"):
        raise V3BuildError(f"frozen v2 image hash mismatch: {row.get('pageid')}")
    destination = staging / str(row["filename"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != digest:
        raise V3BuildError(f"v2 -> v3 copy mismatch: {row.get('pageid')}")
    result = dict(row)
    result.update(
        schema_version=ASSET_SCHEMA,
        asset=f"provisional_v3:v2:{row['pageid']}@sha256:{digest}",
        inherited_v2_asset=row.get("asset"),
        inherited_v2_manifest_sha256=v2_manifest_sha256,
        inherited_v2_record_sha256=source_record_digest(row),
        v3_origin="INHERITED_FROZEN_V2",
        **status_fields(),
    )
    if int(row["pageid"]) == ROLE_OVERRIDE_PAGEID:
        result["experimental_split_suggestion"] = VAL_ROLE
        result["v3_role_override"] = {
            "from": TRAIN_ROLE,
            "to": VAL_ROLE,
            "reason": "increase grass validation creator diversity while preserving creator isolation",
        }
    return result


def copy_new_record(
    *,
    pageid: int,
    generation: str,
    role: str,
    source_row: dict[str, Any],
    screen_row: dict[str, Any],
    source_root: Path,
    staging: Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    if source_record_digest(source_row) != screen_row.get("source_record_sha256"):
        raise V3BuildError(f"{generation} source record binding mismatch for {pageid}")
    source = safe_child(source_root, str(source_row["filename"]))
    digest = sha256_file(source)
    expected_digest = str(source_row.get("download_sha256"))
    if digest != expected_digest or digest != screen_row.get("image_sha256"):
        raise V3BuildError(f"{generation} source image hash mismatch for {pageid}")
    if image_dhash64(source) != source_row.get("dhash64"):
        raise V3BuildError(f"{generation} source dHash mismatch for {pageid}")
    for key in ("creator_group", "source_group"):
        if source_row.get(key) != screen_row.get(key):
            raise V3BuildError(f"{generation} screen/source {key} mismatch for {pageid}")
    destination = staging / str(source_row["filename"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != digest:
        raise V3BuildError(f"{generation} -> v3 copy mismatch for {pageid}")
    copied_fields = {
        key: source_row.get(key)
        for key in (
            "artist",
            "commons_sha1",
            "credit",
            "dhash64",
            "domain",
            "license",
            "license_binding_id",
            "license_canonical_id",
            "license_canonical_name",
            "license_canonical_url",
            "rights_review_status",
            "source_page",
            "source_provider",
            "title",
        )
    }
    return {
        "schema_version": ASSET_SCHEMA,
        "asset": f"provisional_v3:{generation.lower()}:{pageid}@sha256:{digest}",
        "pageid": pageid,
        "class_id": "young_tree",
        "filename": str(source_row["filename"]),
        "copied_image_sha256": digest,
        "creator_group": source_row["creator_group"],
        "source_group": source_row["source_group"],
        "source_dataset": generation,
        "source_dataset_name": SOURCE_DATASETS[generation],
        "source_image_path": str(source_row["filename"]),
        "source_image_sha256": digest,
        "source_manifest_sha256": source_manifest_sha256,
        "source_record_sha256": source_record_digest(source_row),
        "experimental_split_suggestion": role,
        "v3_origin": f"{generation}_MACHINE_VISUAL_SELECTED",
        "machine_decision": f"SELECTED_MACHINE_VISUAL_{generation}_FOR_PROVISIONAL_V3",
        "label_basis": (
            "dual_machine_visual_review_plus_root_machine_adjudication_not_human_truth"
            if generation == "E4"
            else "machine_visual_screen_only_not_human_truth"
        ),
        "visual_screen_record_sha256": source_record_digest(screen_row),
        "visual_adjudication_reason": screen_row.get("reason"),
        "biological_age_verified": False,
        "biological_age_status": "METADATA_YOUTH_GATED_MACHINE_VISUALLY_SELECTED_NOT_HUMAN_VERIFIED",
        **copied_fields,
        **status_fields(),
    }


def role_partition(role: str) -> str:
    if role == TRAIN_ROLE:
        return "train"
    if role == VAL_ROLE:
        return "val"
    if role in {PRINT_ROLE, CREATOR_HOLDOUT_ROLE}:
        return "holdout"
    raise V3BuildError(f"unknown v3 role {role!r}")


def validate_records(records: Sequence[dict[str, Any]], staging: Path) -> dict[str, Any]:
    if len(records) != 78:
        raise V3BuildError(f"v3 must contain exactly 78 records, got {len(records)}")
    pageids = [int(row["pageid"]) for row in records]
    if len(set(pageids)) != 78:
        raise V3BuildError("v3 pageids are not unique")
    for key in ("source_group", "copied_image_sha256"):
        values = [str(row[key]) for row in records]
        if len(set(values)) != 78:
            raise V3BuildError(f"v3 {key} values are not unique")
    for row in records:
        pageid = int(row["pageid"])
        if row.get("schema_version") != ASSET_SCHEMA:
            raise V3BuildError(f"v3 schema mismatch for {pageid}")
        for key, value in status_fields().items():
            if row.get(key) != value:
                raise V3BuildError(f"v3 fail-closed field {key} mismatch for {pageid}")
        if row.get("experimental_split_suggestion") not in ALLOWED_ROLES:
            raise V3BuildError(f"v3 role mismatch for {pageid}")
        if sha256_file(safe_child(staging, str(row["filename"]))) != row["copied_image_sha256"]:
            raise V3BuildError(f"v3 copied image mismatch for {pageid}")
    role_counts = Counter(str(row["experimental_split_suggestion"]) for row in records)
    if role_counts != Counter(EXPECTED_ROLE_COUNTS):
        raise V3BuildError(f"v3 role counts mismatch: {dict(role_counts)}")
    class_counts = Counter(str(row["class_id"]) for row in records)
    if class_counts != Counter(EXPECTED_CLASS_COUNTS):
        raise V3BuildError(f"v3 class counts mismatch: {dict(class_counts)}")
    diversity: dict[str, dict[str, dict[str, int]]] = {}
    for role, classes in EXPECTED_DIVERSITY.items():
        diversity[role] = {}
        for class_id, expected in classes.items():
            subset = [
                row for row in records
                if row["class_id"] == class_id and row["experimental_split_suggestion"] == role
            ]
            actual = (len(subset), len({row["creator_group"] for row in subset}), len({row["source_group"] for row in subset}))
            if actual != expected:
                raise V3BuildError(f"v3 diversity mismatch for {role}/{class_id}: {actual} != {expected}")
            diversity[role][class_id] = {"image_count": actual[0], "creator_count": actual[1], "source_count": actual[2]}
    creator_partitions: dict[str, set[str]] = defaultdict(set)
    for row in records:
        creator_partitions[str(row["creator_group"])].add(role_partition(str(row["experimental_split_suggestion"])))
    leakage = {creator: sorted(parts) for creator, parts in creator_partitions.items() if len(parts) > 1}
    if leakage:
        raise V3BuildError(f"v3 creator partition leakage: {leakage}")
    cross_distances = [
        hamming64(str(left["dhash64"]), str(right["dhash64"]))
        for index, left in enumerate(records)
        for right in records[index + 1 :]
        if role_partition(str(left["experimental_split_suggestion"]))
        != role_partition(str(right["experimental_split_suggestion"]))
    ]
    minimum_distance = min(cross_distances)
    if minimum_distance <= 4:
        raise V3BuildError(f"v3 cross-partition near-duplicate dHash distance {minimum_distance} <= 4")
    return {
        "selected_count": 78,
        "class_counts": dict(sorted(class_counts.items())),
        "experimental_role_counts": dict(sorted(role_counts.items())),
        "train_and_validation_diversity": diversity,
        "creator_partition_leakage_count": 0,
        "source_group_overlap_count": 0,
        "copied_sha256_overlap_count": 0,
        "cross_partition_minimum_dhash64_distance": minimum_distance,
    }


def build_pack(*, workspace: Path, output: Path) -> Path:
    workspace = workspace.resolve(strict=True)
    expected = (workspace / "datasets" / OUTPUT_NAME).resolve(strict=False)
    if output.resolve(strict=False) != expected:
        raise V3BuildError(f"v3 output must be exactly {expected}")
    if output.exists():
        raise FileExistsError(output)
    before = protected_snapshot(workspace)
    v2_root, v2_rows, v2_evidence = validate_v2(workspace)
    screen_rows, screen_evidence = load_machine_screens(workspace)
    source_roots = {key: (workspace / "datasets" / name).resolve(strict=True) for key, name in SOURCE_DATASETS.items()}
    source_rows = {
        key: indexed_by_pageid(load_jsonl(root / SOURCE_MANIFEST_NAMES[key]), label=f"{key} manifest")
        for key, root in source_roots.items()
    }
    source_manifest_hashes = {key: sha256_file(root / SOURCE_MANIFEST_NAMES[key]) for key, root in source_roots.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.tmp-", dir=str(output.parent))).resolve()
    try:
        records = [
            copy_v2_record(row, v2_root=v2_root, staging=staging, v2_manifest_sha256=v2_evidence["manifest_sha256"])
            for row in v2_rows
        ]
        for pageid, (generation, role) in sorted(NEW_ROLES.items()):
            source_row = source_rows[generation].get(pageid)
            if source_row is None:
                raise V3BuildError(f"missing {generation} source pageid {pageid}")
            records.append(
                copy_new_record(
                    pageid=pageid,
                    generation=generation,
                    role=role,
                    source_row=source_row,
                    screen_row=screen_rows[pageid],
                    source_root=source_roots[generation],
                    staging=staging,
                    source_manifest_sha256=source_manifest_hashes[generation],
                )
            )
        records.sort(key=lambda row: (str(row["class_id"]), int(row["pageid"])))
        audit = validate_records(records, staging)
        manifest_text = jsonl_text(records)
        write_text(staging / "manifest.jsonl", manifest_text)

        decisions: list[dict[str, Any]] = []
        for row in records:
            pageid = int(row["pageid"])
            is_new = pageid in NEW_ROLES
            decision = {
                "schema_version": DECISION_SCHEMA,
                "selected": True,
                "disposition": "SELECTED_MACHINE_VISUAL_E3_OR_E4" if is_new else "INHERITED_FROZEN_V2",
                "pageid": pageid,
                "class_id": row["class_id"],
                "source_dataset": row["source_dataset"],
                "source_group": row["source_group"],
                "creator_group": row["creator_group"],
                "source_record_sha256": row["source_record_sha256"],
                "copied_image_sha256": row["copied_image_sha256"],
                "experimental_split_suggestion": row["experimental_split_suggestion"],
                "inherited_v2_record_sha256": row.get("inherited_v2_record_sha256"),
                **status_fields(),
            }
            if pageid == ROLE_OVERRIDE_PAGEID:
                decision["role_override"] = row["v3_role_override"]
            if is_new:
                decision["visual_screen_record_sha256"] = row["visual_screen_record_sha256"]
            decisions.append(decision)
        decision_text = jsonl_text(decisions)
        write_text(staging / "source_decision_manifest.jsonl", decision_text)

        selected_records = [
            {
                "pageid": pageid,
                "class_id": "young_tree",
                "source_dataset": generation,
                "experimental_split_suggestion": role,
                "screen_record_sha256": source_record_digest(screen_rows[pageid]),
                "machine_screened": True,
                "dual_machine_reviewed": generation == "E4",
                "dual_machine_review_completed": generation == "E4",
                "root_machine_adjudicated": generation == "E4",
                "review_scope": (
                    "E4_DUAL_MACHINE_REVIEW_ROOT_ADJUDICATION"
                    if generation == "E4"
                    else "E3_MACHINE_SCREEN_ONLY"
                ),
            }
            for pageid, (generation, role) in sorted(NEW_ROLES.items())
        ]
        visual_evidence = {
            "schema_version": EVIDENCE_SCHEMA,
            "status": STATUS,
            "authority": false_authority(),
            "data_locked": False,
            "human_reviewed": False,
            "human_label": False,
            "data_authority": False,
            "print_eligible": False,
            "rights_approved": False,
            "training_eligible": False,
            "all_selected_records_machine_screened": True,
            "dual_machine_review_completed": True,
            "dual_machine_review_scope": "E4_SELECTED_ONLY",
            "independent_machine_review_count": 2,
            "root_machine_adjudicated": True,
            "root_machine_adjudication_scope": "E4_SELECTED_ONLY",
            "review_protocol": "E3_machine_screen_only; E4_two_independent_machine_pixel_reviews_plus_root_machine_adjudication",
            "v2_independent_audit": {
                "path": v2_evidence["audit_path"].relative_to(workspace).as_posix(),
                "sha256": v2_evidence["audit_sha256"],
                "status": "PASS",
                "selected_count": 73,
            },
            "upstream_machine_screens": screen_evidence,
            "selected_pageids": sorted(NEW_ROLES),
            "selected_records": selected_records,
            "role_override": {
                "pageid": ROLE_OVERRIDE_PAGEID,
                "from": TRAIN_ROLE,
                "to": VAL_ROLE,
                "reason": "increase grass validation creator diversity while preserving creator isolation",
            },
            "explicit_non_claims": [
                "HUMAN_REVIEWED",
                "HUMAN_LABEL",
                "VISUAL_GROUND_TRUTH",
                "RIGHTS_APPROVED",
                "FORMAL_SPLIT_ASSIGNED",
                "TRAIN_ELIGIBLE",
                "DATA_LOCKED",
            ],
        }
        write_json(staging / "machine_visual_review_evidence.json", visual_evidence)
        visual_evidence_sha = sha256_file(staging / "machine_visual_review_evidence.json")

        split_payload = {
            "schema_version": "rootscope.machine_curated_v3_experimental_split_suggestion.v1",
            "status": STATUS,
            "authority": false_authority(),
            "data_locked": False,
            "human_reviewed": False,
            "print_eligible": False,
            "rights_approved": False,
            "training_eligible": False,
            "formal_split_assignment": False,
            "experimental_training_switch_required": True,
            "policy": "fixed v3 role contract; creator groups are isolated across train, validation, and collapsed holdout partitions",
            "role_counts": EXPECTED_ROLE_COUNTS,
            "train_and_validation_diversity": audit["train_and_validation_diversity"],
            "records": [
                {
                    "asset": row["asset"],
                    "pageid": row["pageid"],
                    "class_id": row["class_id"],
                    "creator_group": row["creator_group"],
                    "source_group": row["source_group"],
                    "role": row["experimental_split_suggestion"],
                }
                for row in records
            ],
        }
        write_json(staging / "experimental_split_suggestion.json", split_payload)

        attribution = [
            "# RootScope machine-curated provisional v3 sources",
            "",
            f"> Status: `{STATUS}`",
            "> Machine-only experimental pack; human review, rights approval, formal split, training eligibility, print eligibility, and data lock are all false.",
            "",
        ]
        for row in records:
            license_name = row.get("license_canonical_name") or row.get("license") or "UNKNOWN"
            license_url = row.get("license_canonical_url")
            label = f"[{license_name}]({license_url})" if license_url else str(license_name)
            attribution.append(
                f"- `{row['filename']}` — {row.get('artist') or 'UNKNOWN'} — "
                f"[{row.get('title') or row['pageid']}]({row.get('source_page')}) — {label} — "
                f"origin `{row['v3_origin']}`"
            )
        write_text(staging / "ATTRIBUTION.md", "\n".join(attribution) + "\n")
        write_text(
            staging / "README.md",
            f"""# RootScope machine-curated provisional v3

Status: `{STATUS}`

V3 preserves 73 byte-verified v2 assets, adds exactly five machine-screened
E3/E4 young-tree candidates, and applies one fixed creator-safe role override.
It contains 78 records: train suggestion 55, validation suggestion 9, print
holdout 6, and creator holdout 8.  These are experimental roles, not formal
splits.  Every authority and eligibility field remains false.

`machine_visual_review_evidence.json` binds the passing v2 independent audit,
the E3/E4 screen artifacts, and the five added IDs and roles.  E3 is scoped to
machine-screen evidence only; the two independent machine reviews and root
machine adjudication apply only to the four E4 selections.  None of this is
human review or visual ground truth.
""",
        )

        after = protected_snapshot(workspace)
        if after != before:
            raise V3BuildError("protected v1/v2/E0-E4/formal inputs changed during v3 build")
        payload_files = sorted(
            (path for path in staging.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(staging).as_posix(),
        )
        payload_root = sha256_bytes(
            "".join(f"{path.relative_to(staging).as_posix()}\0{sha256_file(path)}\n" for path in payload_files).encode("utf-8")
        )
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
            "authority": false_authority(),
            "formal_a1_dataset": False,
            "formal_split_assigned": False,
            "human_reviewed": False,
            "data_locked": False,
            "rights_approved": False,
            "training_eligible": False,
            "print_eligible": False,
            "experimental_training_switch_required": True,
            "implementation_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "manifest_sha256": sha256_bytes(manifest_text.encode("utf-8")),
            "source_decision_manifest_sha256": sha256_bytes(decision_text.encode("utf-8")),
            "machine_visual_review_evidence_path": "machine_visual_review_evidence.json",
            "machine_visual_review_evidence_sha256": visual_evidence_sha,
            "machine_visual_review_evidence": {
                "human_reviewed": False,
                "all_selected_records_machine_screened": True,
                "dual_machine_review_completed": True,
                "dual_machine_review_scope": "E4_SELECTED_ONLY",
                "root_machine_adjudicated": True,
                "root_machine_adjudication_scope": "E4_SELECTED_ONLY",
                "selected_pageids": sorted(NEW_ROLES),
            },
            "payload_root_sha256_before_receipt": payload_root,
            "audit": audit,
            "all_split_targets_met": True,
            "source_manifest_sha256": source_manifest_hashes,
            "frozen_v2": {
                "path": f"datasets/{V2_NAME}",
                "tree_sha256_before": before["protected_tree_sha256"][f"datasets/{V2_NAME}"],
                "tree_sha256_after": after["protected_tree_sha256"][f"datasets/{V2_NAME}"],
                "manifest_sha256": v2_evidence["manifest_sha256"],
                "receipt_sha256": v2_evidence["receipt_sha256"],
                "sha256sums_sha256": v2_evidence["sha256sums_sha256"],
                "independent_audit_path": v2_evidence["audit_path"].relative_to(workspace).as_posix(),
                "independent_audit_sha256": v2_evidence["audit_sha256"],
                "unchanged": True,
            },
            "protected_inputs": {"before": before, "after": after, "unchanged": True},
            "explicit_non_claims": [
                "HUMAN_REVIEWED",
                "RIGHTS_APPROVED",
                "A1_DATASET",
                "TRAIN_READY",
                "FORMAL_SPLIT_ASSIGNED",
                "PRINT_ELIGIBLE",
                "DATA_LOCKED",
                "MODEL_QUALIFIED",
                "BIOLOGICAL_AGE_HUMAN_VERIFIED",
            ],
        }
        write_json(staging / "receipt.json", receipt)
        all_files = sorted(
            (path for path in staging.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(staging).as_posix(),
        )
        write_text(
            staging / "SHA256SUMS",
            "".join(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n" for path in all_files),
        )
        os.replace(staging, output)
        return output
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--output", type=Path, default=workspace / "datasets" / OUTPUT_NAME)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.resolve(strict=True)
    output = args.output.resolve(strict=False)
    expected = (workspace / "datasets" / OUTPUT_NAME).resolve(strict=False)
    if output != expected:
        raise V3BuildError(f"refusing non-standard v3 output: {output}")
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"v3 output exists; pass --replace: {output}")
        if output.is_symlink() or not output.is_dir() or output.parent != expected.parent or output.name != OUTPUT_NAME:
            raise V3BuildError(f"unsafe v3 replace target: {output}")
        shutil.rmtree(output)
    print(build_pack(workspace=workspace, output=output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
