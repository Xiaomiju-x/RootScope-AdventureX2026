#!/usr/bin/env python3
"""Build RootScope provisional v2 without mutating frozen provisional v1.

V2 inherits the 55 byte-verified v1 assets and may add a narrowly enumerated
E0 machine-visual supplement.  Every formal authority remains false.  The
``young_tree`` supplement is explicitly morphology-only: biological age is
unverified, and visibly mature large trees must be excluded by policy.
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
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

import build_machine_curated_provisional as v1


OUTPUT_NAME = "rootscope_machine_curated_provisional_v2"
V1_NAME = v1.OUTPUT_NAME
STATUS = "MACHINE_CURATED_EXPERIMENTAL_V2_ONLY_NOT_HUMAN_REVIEWED_NOT_A1_NOT_DATA_LOCKED"

TRAIN_MINIMUMS = {
    "grass_clump": 6,
    "low_shrub": 6,
    "young_tree": 5,
    "unknown": 15,
}
VAL_MINIMUMS = {
    "grass_clump": 2,
    "low_shrub": 2,
    "young_tree": 2,
    "unknown": 2,
}

SUPPLEMENT_REQUESTS: dict[str, list[int]] = {
    "grass_clump": [
        28135991,
        163498042,
        21981205,
        133270305,
        66707539,
        22752881,
        25133507,
        85374315,
        107123743,
        85376485,
    ],
    "low_shrub": [
        199434564,
        66248044,
        22749006,
        178094951,
        22749222,
        85376459,
        54947577,
        194934149,
        180194248,
        194934484,
    ],
    "young_tree": [
        108010572,
        135673279,
        135673278,
        38644781,
        35731466,
        6825129,
    ],
}

# Populated after reviewing the V2 supplement contact sheet.  The initial
# candidate build may be replaced before V2 is frozen; every exclusion remains
# in the source-decision manifest.
SUPPLEMENT_VISUAL_EXCLUSIONS: dict[int, str] = {
    21981205: (
        "requested as grass, but the source identifies Rhanterium epapposum and the pixels show "
        "a compact woody mound; conservative class-mismatch exclusion"
    ),
    6825129: "visibly mature large tree; must not be labeled as young_tree",
    35731466: "visibly mature large tree; must not be labeled as young_tree",
    108010572: "visibly mature large tree; must not be labeled as young_tree",
    38644781: "wind-thrown/leaning woody plant; not a conforming young-tree morphology sample",
    135673278: "at least a medium-sized tree with mixed background; biological youth is unsupported",
    135673279: "at least a medium-sized tree with mixed background; biological youth is unsupported",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return v1.sha256_file(path)


def canonical_json(value: object) -> str:
    return v1.canonical_json(value)


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return v1.jsonl_text(rows)


def write_text(path: Path, text: str) -> None:
    v1.write_text(path, text)


def write_json(path: Path, value: object) -> None:
    v1.write_json(path, value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return v1.load_jsonl(path)


def tree_sha256(root: Path) -> str:
    return v1.tree_sha256(root)


def authority_false() -> dict[str, bool]:
    return dict(v1.AUTHORITY_FALSE)


def status_fields() -> dict[str, Any]:
    return {
        "authority": authority_false(),
        "data_locked": False,
        "human_reviewed": False,
        "machine_curated_only": True,
        "print_eligible": False,
        "rights_approved": False,
        "split": "UNASSIGNED_DO_NOT_TRAIN",
        "status": STATUS,
        "training_eligible": False,
        "experimental_training_switch_required": True,
    }


def source_record_digest(row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(row).encode("utf-8"))


def stable_group_rank(value: str) -> tuple[str, str]:
    return sha256_bytes(value.encode("utf-8")), value


def plan_group_roles(
    records: Sequence[dict[str, Any]],
    *,
    train_minimums: dict[str, int],
    val_minimums: dict[str, int],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Assign creator groups while protecting print groups and train minima.

    All eligible groups begin in training.  Validation groups are then chosen
    deterministically, preferring the smallest move that meets a class deficit
    and refusing any move that would take another class below its train floor.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["creator_group"])].append(record)
    print_groups = {
        creator
        for creator, group_rows in grouped.items()
        if any(row.get("print_holdout_candidate") is True for row in group_rows)
    }
    group_roles = {
        creator: ("CREATOR_GROUP_HOLDOUT_NOT_TRAIN" if creator in print_groups else "EXPERIMENTAL_TRAIN_SUGGESTION")
        for creator in grouped
    }

    def counts(role: str) -> Counter[str]:
        return Counter(
            str(row["class_id"])
            for creator, group_rows in grouped.items()
            if group_roles[creator] == role
            for row in group_rows
        )

    class_ids = sorted(set(train_minimums) | set(val_minimums))
    for class_id in class_ids:
        while counts("EXPERIMENTAL_VAL_SUGGESTION")[class_id] < val_minimums.get(class_id, 0):
            train_counts = counts("EXPERIMENTAL_TRAIN_SUGGESTION")
            val_count = counts("EXPERIMENTAL_VAL_SUGGESTION")[class_id]
            deficit = val_minimums.get(class_id, 0) - val_count
            candidates: list[tuple[Any, ...]] = []
            for creator, group_rows in grouped.items():
                if group_roles[creator] != "EXPERIMENTAL_TRAIN_SUGGESTION":
                    continue
                moved = Counter(str(row["class_id"]) for row in group_rows)
                if moved[class_id] <= 0:
                    continue
                if any(
                    train_counts[moved_class] - moved_count < train_minimums.get(moved_class, 0)
                    for moved_class, moved_count in moved.items()
                ):
                    continue
                candidates.append(
                    (
                        abs(deficit - moved[class_id]),
                        sum(moved.values()),
                        len(moved),
                        stable_group_rank(creator),
                        creator,
                    )
                )
            if not candidates:
                break
            creator = min(candidates)[-1]
            group_roles[creator] = "EXPERIMENTAL_VAL_SUGGESTION"

    train_counts = counts("EXPERIMENTAL_TRAIN_SUGGESTION")
    val_counts = counts("EXPERIMENTAL_VAL_SUGGESTION")
    attainment: dict[str, dict[str, Any]] = {}
    for class_id in class_ids:
        train_count = int(train_counts[class_id])
        val_count = int(val_counts[class_id])
        train_required = int(train_minimums.get(class_id, 0))
        val_required = int(val_minimums.get(class_id, 0))
        attainment[class_id] = {
            "train_count": train_count,
            "train_minimum": train_required,
            "train_deficit": max(0, train_required - train_count),
            "train_met": train_count >= train_required,
            "val_count": val_count,
            "val_minimum": val_required,
            "val_deficit": max(0, val_required - val_count),
            "val_met": val_count >= val_required,
            "both_met": train_count >= train_required and val_count >= val_required,
        }
    return group_roles, attainment


def assign_roles(
    records: list[dict[str, Any]],
    *,
    train_minimums: dict[str, int],
    val_minimums: dict[str, int],
) -> dict[str, dict[str, Any]]:
    group_roles, attainment = plan_group_roles(
        records, train_minimums=train_minimums, val_minimums=val_minimums
    )
    for record in records:
        group_role = group_roles[str(record["creator_group"])]
        if record.get("print_holdout_candidate") is True:
            record["experimental_split_suggestion"] = "PRINT_DEMO_HOLDOUT_NOT_TRAIN"
        else:
            record["experimental_split_suggestion"] = group_role
    return attainment


def hamming64(left: str, right: str) -> int:
    if len(left) != 16 or len(right) != 16:
        raise ValueError("dhash64 must contain 16 hexadecimal digits")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def validate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("v2 manifest is empty")
    pageids: set[int] = set()
    sources: set[str] = set()
    hashes: set[str] = set()
    creator_roles: dict[str, set[str]] = defaultdict(set)
    for record in records:
        pageid = int(record["pageid"])
        if pageid in pageids:
            raise ValueError(f"duplicate v2 pageid {pageid}")
        pageids.add(pageid)
        source_group = str(record["source_group"])
        if source_group in sources:
            raise ValueError(f"duplicate v2 source_group {source_group}")
        sources.add(source_group)
        digest = str(record["copied_image_sha256"])
        if digest in hashes:
            raise ValueError(f"duplicate v2 image SHA {digest}")
        hashes.add(digest)
        for key, value in status_fields().items():
            if record.get(key) != value:
                raise ValueError(f"v2 record {pageid} violates fail-closed field {key}")
        creator_roles[str(record["creator_group"])].add(str(record["experimental_split_suggestion"]))
        if record.get("v2_origin") == "E0_MACHINE_VISUAL_SUPPLEMENT" and record["class_id"] == "young_tree":
            if record.get("biological_age_verified") is not False:
                raise ValueError(f"supplemental young-tree {pageid} claims biological-age verification")
            if record.get("biological_age_status") != "UNCERTAIN_VISIBLE_PLANT_MORPHOLOGY_ONLY":
                raise ValueError(f"supplemental young-tree {pageid} lacks age-uncertainty status")
    for creator, roles in creator_roles.items():
        normal = roles & {"EXPERIMENTAL_TRAIN_SUGGESTION", "EXPERIMENTAL_VAL_SUGGESTION"}
        held = roles & {"PRINT_DEMO_HOLDOUT_NOT_TRAIN", "CREATOR_GROUP_HOLDOUT_NOT_TRAIN"}
        if len(normal) > 1 or (normal and held):
            raise ValueError(f"creator-group leakage {creator}: {sorted(roles)}")
    print_rows = [row for row in records if row.get("print_holdout_candidate") is True]
    normal_rows = [
        row
        for row in records
        if row["experimental_split_suggestion"]
        in {"EXPERIMENTAL_TRAIN_SUGGESTION", "EXPERIMENTAL_VAL_SUGGESTION"}
    ]
    distances = [
        hamming64(str(print_row["dhash64"]), str(normal_row["dhash64"]))
        for print_row in print_rows
        for normal_row in normal_rows
    ]
    return {
        "selected_count": len(records),
        "class_counts": dict(sorted(Counter(str(row["class_id"]) for row in records).items())),
        "origin_counts": dict(sorted(Counter(str(row["v2_origin"]) for row in records).items())),
        "experimental_role_counts": dict(
            sorted(Counter(str(row["experimental_split_suggestion"]) for row in records).items())
        ),
        "class_role_counts": {
            class_id: dict(
                sorted(
                    Counter(
                        str(row["experimental_split_suggestion"])
                        for row in records
                        if row["class_id"] == class_id
                    ).items()
                )
            )
            for class_id in sorted({str(row["class_id"]) for row in records})
        },
        "creator_group_count": len(creator_roles),
        "source_group_overlap_count": 0,
        "copied_sha256_overlap_count": 0,
        "creator_role_leakage_count": 0,
        "print_holdout_count": len(print_rows),
        "print_to_train_or_val_minimum_dhash64_distance": min(distances) if distances else None,
    }


def copied_v1_record(
    row: dict[str, Any], *, v1_root: Path, output_root: Path, v1_manifest_sha256: str
) -> dict[str, Any]:
    source = v1.safe_source_image(v1_root, str(row["filename"]))
    source_sha = sha256_file(source)
    if source_sha != row.get("copied_image_sha256"):
        raise ValueError(f"frozen v1 asset SHA mismatch: {row.get('asset')}")
    destination = output_root / str(row["filename"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != source_sha:
        raise ValueError(f"v1 -> v2 copy mismatch: {row.get('asset')}")
    result = dict(row)
    result.update(
        schema_version="rootscope.machine_curated_provisional_asset.v2",
        asset=f"provisional_v2:v1:{row['pageid']}@sha256:{source_sha}",
        copied_image_sha256=source_sha,
        v2_origin="INHERITED_FROZEN_V1",
        inherited_v1_asset=row.get("asset"),
        inherited_v1_record_sha256=source_record_digest(row),
        inherited_v1_manifest_sha256=v1_manifest_sha256,
        **status_fields(),
    )
    result.pop("experimental_split_suggestion", None)
    return result


def supplemental_record(
    *,
    row: dict[str, Any],
    requested_class: str,
    e0_root: Path,
    output_root: Path,
    e0_manifest_sha256: str,
    strict_row: dict[str, Any],
) -> dict[str, Any]:
    source = v1.safe_source_image(e0_root, str(row["filename"]))
    source_sha = sha256_file(source)
    if source_sha != row.get("download_sha256"):
        raise ValueError(f"E0 source SHA mismatch for pageid {row['pageid']}")
    extension = source.suffix.lower() or ".img"
    relative = Path("images") / requested_class / f"{requested_class}_{row['pageid']}_{source_sha[:12]}{extension}"
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != source_sha:
        raise ValueError(f"E0 -> v2 copy mismatch for pageid {row['pageid']}")
    result = {
        "schema_version": "rootscope.machine_curated_provisional_asset.v2",
        "asset": f"provisional_v2:e0:{row['pageid']}@sha256:{source_sha}",
        "class_id": requested_class,
        "pageid": int(row["pageid"]),
        "filename": relative.as_posix(),
        "copied_image_sha256": source_sha,
        "source_image_sha256": source_sha,
        "source_dataset": "E0",
        "source_dataset_name": v1.SOURCE_DATASETS["E0"],
        "source_manifest_sha256": e0_manifest_sha256,
        "source_record_sha256": source_record_digest(row),
        "source_image_path": row["filename"],
        "source_group": row["source_group"],
        "creator_group": row["creator_group"],
        "commons_sha1": row.get("commons_sha1"),
        "dhash64": row.get("dhash64"),
        "domain": row.get("domain"),
        "source_provider": row.get("source_provider"),
        "source_page": row.get("source_page"),
        "title": row.get("title"),
        "artist": row.get("artist"),
        "credit": row.get("credit"),
        "license": row.get("license"),
        "license_canonical_id": row.get("license_canonical_id"),
        "license_canonical_name": row.get("license_canonical_name"),
        "license_canonical_url": row.get("license_canonical_url"),
        "license_binding_id": row.get("license_binding_id"),
        "rights_review_status": row.get("rights_review_status"),
        "source_acquisition_class_hint": row.get("class_id"),
        "source_strict_final_label": strict_row.get("final_label"),
        "source_strict_label_record_sha256": source_record_digest(strict_row),
        "v2_origin": "E0_MACHINE_VISUAL_SUPPLEMENT",
        "machine_decision": "SELECTED_MACHINE_VISUAL_SUPPLEMENT_V2",
        "label_basis": "machine_visual_recheck_not_human_truth",
        "visual_adjudication_note": (
            "independent machine visual morphology pass; formal human visual and rights review pending"
        ),
        "print_holdout_candidate": False,
        **status_fields(),
    }
    if requested_class == "young_tree":
        result.update(
            biological_age_verified=False,
            biological_age_status="UNCERTAIN_VISIBLE_PLANT_MORPHOLOGY_ONLY",
            label_basis="visible_plant_morphology_only_biological_age_uncertain_not_human_truth",
            explicit_non_claims=["BIOLOGICAL_AGE_VERIFIED", "SEEDLING_OR_SAPLING_CONFIRMED"],
        )
    return result


def resolve_font() -> ImageFont.ImageFont:
    return v1.resolve_font()


def render_selected_sheet(
    records: Sequence[dict[str, Any]], output_file: Path, *, title: str, columns: int = 5
) -> None:
    font = resolve_font()
    cell_w, image_h, label_h, header_h = 290, 190, 88, 58
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, header_h + rows * (image_h + label_h)), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width - 1, header_h - 1), fill="#182433")
    draw.text((12, 7), title, fill="white", font=font)
    draw.text((12, 31), "MACHINE ONLY | NOT HUMAN REVIEWED | NOT A1 | NOT DATA LOCKED", fill="#ffcf66", font=font)
    dataset_root = output_file.parent.parent
    for index, record in enumerate(records):
        grid_row, column = divmod(index, columns)
        x = column * cell_w
        y = header_h + grid_row * (image_h + label_h)
        with Image.open(dataset_root / str(record["filename"])) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            tile = ImageOps.contain(image, (cell_w - 8, image_h - 8), method=Image.Resampling.LANCZOS)
        backdrop = Image.new("RGB", (cell_w, image_h), "#d6d6d6")
        backdrop.paste(tile, ((cell_w - tile.width) // 2, (image_h - tile.height) // 2))
        sheet.paste(backdrop, (x, y))
        draw = ImageDraw.Draw(sheet)
        top = y + image_h
        draw.rectangle((x, top, x + cell_w - 1, top + label_h - 1), fill="white", outline="#aaaaaa")
        age = " age=UNCERTAIN" if record.get("biological_age_verified") is False else ""
        text = (
            f"class={record['class_id']} pageid={record['pageid']}{age}\n"
            f"origin={record['v2_origin']}\n"
            f"role={record['experimental_split_suggestion']}\n"
            f"creator={str(record['creator_group'])[-16:]} sha={str(record['copied_image_sha256'])[:10]}"
        )
        draw.multiline_text((x + 6, top + 4), text, fill="#111111", font=font, spacing=2)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_file, format="PNG", optimize=False, compress_level=9)


def render_supplement_candidate_sheet(
    *,
    requests: dict[str, list[int]],
    e0_index: dict[int, dict[str, Any]],
    decisions: dict[int, dict[str, Any]],
    e0_root: Path,
    output_file: Path,
    columns: int = 5,
) -> None:
    items = [
        (class_id, pageid, e0_index[pageid], decisions[pageid])
        for class_id, pageids in requests.items()
        for pageid in pageids
        if pageid in e0_index
    ]
    font = resolve_font()
    cell_w, image_h, label_h, header_h = 300, 190, 104, 58
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, header_h + rows * (image_h + label_h)), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width - 1, header_h - 1), fill="#182433")
    draw.text((12, 7), f"V2 E0 SUPPLEMENT CANDIDATES | {len(items)} existing", fill="white", font=font)
    draw.text((12, 31), "PIXEL QA | decisions are MACHINE-ONLY", fill="#ffcf66", font=font)
    for index, (requested_class, pageid, source_row, decision) in enumerate(items):
        grid_row, column = divmod(index, columns)
        x = column * cell_w
        y = header_h + grid_row * (image_h + label_h)
        source = v1.safe_source_image(e0_root, str(source_row["filename"]))
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            tile = ImageOps.contain(image, (cell_w - 8, image_h - 8), method=Image.Resampling.LANCZOS)
        backdrop = Image.new("RGB", (cell_w, image_h), "#d6d6d6")
        backdrop.paste(tile, ((cell_w - tile.width) // 2, (image_h - tile.height) // 2))
        sheet.paste(backdrop, (x, y))
        draw = ImageDraw.Draw(sheet)
        top = y + image_h
        draw.rectangle((x, top, x + cell_w - 1, top + label_h - 1), fill="white", outline="#aaaaaa")
        disposition = str(decision["disposition"])
        text = (
            f"requested={requested_class} pageid={pageid}\n"
            f"source_hint={source_row.get('class_id')}\n"
            f"decision={disposition}\n"
            f"age={'UNCERTAIN' if requested_class == 'young_tree' else 'n/a'}\n"
            f"creator={str(source_row.get('creator_group'))[-16:]}"
        )
        draw.multiline_text((x + 6, top + 4), text, fill="#111111", font=font, spacing=2)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_file, format="PNG", optimize=False, compress_level=9)


def decision_row(
    *, pageid: int, requested_class: str, source: dict[str, Any] | None, disposition: str, reason: str
) -> dict[str, Any]:
    row = {
        "schema_version": "rootscope.machine_curated_v2_source_decision.v1",
        "pageid": pageid,
        "requested_class": requested_class,
        "disposition": disposition,
        "reason": reason,
        "selected": disposition.startswith("SELECTED") or disposition.startswith("INHERITED"),
        **status_fields(),
    }
    if source is not None:
        row.update(
            source_group=source.get("source_group"),
            creator_group=source.get("creator_group"),
            source_record_sha256=source_record_digest(source),
            source_acquisition_class_hint=source.get("class_id"),
        )
    return row


def build_pack(*, workspace: Path, output: Path) -> Path:
    workspace = workspace.resolve(strict=True)
    datasets = (workspace / "datasets").resolve(strict=True)
    expected = (datasets / OUTPUT_NAME).resolve(strict=False)
    if output.resolve(strict=False) != expected:
        raise ValueError(f"v2 output must be exactly {expected}")
    if output.exists():
        raise FileExistsError(output)

    v1_root = (datasets / V1_NAME).resolve(strict=True)
    e0_root = (datasets / v1.SOURCE_DATASETS["E0"]).resolve(strict=True)
    human_root = e0_root / "review" / "human_decisions"
    strict_path = e0_root / "review" / "ai_final_labels_v1" / "strict_structure_labels.jsonl"
    v1_tree_before = tree_sha256(v1_root)
    v1_manifest_sha_before = sha256_file(v1_root / "manifest.jsonl")
    v1_sums_sha_before = sha256_file(v1_root / "SHA256SUMS")
    human_tree_before = tree_sha256(human_root)
    human_journal = human_root / "decision_journal.jsonl"
    human_journal_before = sha256_file(human_journal)

    v1_rows = load_jsonl(v1_root / "manifest.jsonl")
    e0_rows = load_jsonl(e0_root / "manifest.jsonl")
    e0_index = v1.indexed_by_pageid(e0_rows, label="E0 manifest")
    strict_rows = load_jsonl(strict_path)
    strict_index = v1.indexed_by_pageid(strict_rows, label="strict labels")
    e0_manifest_sha = sha256_file(e0_root / "manifest.jsonl")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.tmp-", dir=str(output.parent))).resolve()
    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    supplement_decisions_by_id: dict[int, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    try:
        for row in v1_rows:
            inherited = copied_v1_record(
                row,
                v1_root=v1_root,
                output_root=staging,
                v1_manifest_sha256=v1_manifest_sha_before,
            )
            selected.append(inherited)
            decisions.append(
                decision_row(
                    pageid=int(row["pageid"]),
                    requested_class=str(row["class_id"]),
                    source=row,
                    disposition="INHERITED_FROZEN_V1",
                    reason="byte-verified v1 mother-set asset; v1 itself was not modified",
                )
            )

        for requested_class, pageids in SUPPLEMENT_REQUESTS.items():
            for pageid in pageids:
                source = e0_index.get(pageid)
                if source is None:
                    decision = decision_row(
                        pageid=pageid,
                        requested_class=requested_class,
                        source=None,
                        disposition="UNRESOLVED_E0_PAGEID_NOT_FOUND",
                        reason="requested E0 pageid is absent; no silent substitution",
                    )
                    unresolved.append(
                        {
                            "pageid": pageid,
                            "requested_class": requested_class,
                            "status": "E0_PAGEID_NOT_FOUND_NO_SILENT_SUBSTITUTION",
                        }
                    )
                elif pageid in SUPPLEMENT_VISUAL_EXCLUSIONS:
                    decision = decision_row(
                        pageid=pageid,
                        requested_class=requested_class,
                        source=source,
                        disposition="EXCLUDE_CONSERVATIVE_MACHINE_VISUAL_GATE_V2",
                        reason=SUPPLEMENT_VISUAL_EXCLUSIONS[pageid],
                    )
                else:
                    strict_row = strict_index.get(pageid)
                    if strict_row is None:
                        raise ValueError(f"E0 supplement {pageid} lacks strict label provenance")
                    record = supplemental_record(
                        row=source,
                        requested_class=requested_class,
                        e0_root=e0_root,
                        output_root=staging,
                        e0_manifest_sha256=e0_manifest_sha,
                        strict_row=strict_row,
                    )
                    selected.append(record)
                    reason = "machine-visual supplement selected; not human truth"
                    if requested_class == "young_tree":
                        reason += "; biological age is uncertain and only visible morphology is asserted"
                    decision = decision_row(
                        pageid=pageid,
                        requested_class=requested_class,
                        source=source,
                        disposition="SELECTED_E0_MACHINE_VISUAL_SUPPLEMENT_V2",
                        reason=reason,
                    )
                    decision["biological_age_verified"] = False if requested_class == "young_tree" else None
                    decision["biological_age_status"] = (
                        "UNCERTAIN_VISIBLE_PLANT_MORPHOLOGY_ONLY" if requested_class == "young_tree" else "NOT_APPLICABLE"
                    )
                decisions.append(decision)
                supplement_decisions_by_id[pageid] = decision

        selected.sort(key=lambda row: (str(row["class_id"]), int(row["pageid"])))
        attainment = assign_roles(
            selected, train_minimums=TRAIN_MINIMUMS, val_minimums=VAL_MINIMUMS
        )
        audit = validate_records(selected)

        manifest_text = jsonl_text(selected)
        decision_text = jsonl_text(
            sorted(decisions, key=lambda row: (str(row["requested_class"]), int(row["pageid"]), str(row["disposition"])))
        )
        write_text(staging / "manifest.jsonl", manifest_text)
        write_text(staging / "source_decision_manifest.jsonl", decision_text)
        write_json(
            staging / "unresolved_requested_ids.json",
            {
                "schema_version": "rootscope.machine_curated_v2_unresolved_requests.v1",
                "status": STATUS,
                "authority": authority_false(),
                "no_silent_substitution": True,
                "records": unresolved,
            },
        )
        split_payload = {
            "schema_version": "rootscope.machine_curated_v2_experimental_split_suggestion.v1",
            "status": STATUS,
            "authority": authority_false(),
            "formal_split_assignment": False,
            "experimental_training_switch_required": True,
            "grouping_key": "creator_group; source_group/content SHA also enforced unique",
            "policy": (
                "all print creator groups held out; remaining creator groups start in train, then "
                "deterministically move to validation only when every affected class keeps its train minimum"
            ),
            "requested_minimums": {"train": TRAIN_MINIMUMS, "validation": VAL_MINIMUMS},
            "attainment": attainment,
            "records": [
                {
                    "asset": row["asset"],
                    "class_id": row["class_id"],
                    "creator_group": row["creator_group"],
                    "pageid": row["pageid"],
                    "role": row["experimental_split_suggestion"],
                    "source_group": row["source_group"],
                }
                for row in selected
            ],
        }
        write_json(staging / "experimental_split_suggestion.json", split_payload)

        attribution = [
            "# RootScope machine-curated provisional v2｜来源与署名",
            "",
            f"> Status: `{STATUS}`",
            "> 机器筛选实验包；不是人工视觉结论、权利批准、训练许可、年龄确认或打印许可。",
            "",
        ]
        for row in selected:
            license_name = row.get("license_canonical_name") or row.get("license") or "UNKNOWN"
            license_url = row.get("license_canonical_url")
            license_label = f"[{license_name}]({license_url})" if license_url else str(license_name)
            attribution.append(
                f"- `{row['filename']}` — {row.get('artist') or 'UNKNOWN'} — "
                f"[{row.get('title') or row['pageid']}]({row.get('source_page')}) — {license_label} — "
                f"origin `{row['v2_origin']}`"
            )
        write_text(staging / "ATTRIBUTION.md", "\n".join(attribution) + "\n")

        contact_dir = staging / "contact_sheets"
        for class_id in ("grass_clump", "low_shrub", "young_tree", "unknown"):
            class_rows = [row for row in selected if row["class_id"] == class_id]
            render_selected_sheet(
                class_rows,
                contact_dir / f"{class_id}.png",
                title=f"RootScope provisional v2 | {class_id} | {len(class_rows)} selected",
            )
        print_rows = [row for row in selected if row.get("print_holdout_candidate")]
        render_selected_sheet(
            print_rows,
            contact_dir / "print_demo_holdout_candidates.png",
            title=f"FROZEN V1 PRINT HOLDOUTS | {len(print_rows)} | print_eligible=false",
            columns=3,
        )
        supplement_rows = [row for row in selected if row["v2_origin"] == "E0_MACHINE_VISUAL_SUPPLEMENT"]
        render_selected_sheet(
            supplement_rows,
            contact_dir / "e0_supplement_selected.png",
            title=f"V2 E0 SUPPLEMENT SELECTED | {len(supplement_rows)} | MACHINE ONLY",
        )
        render_supplement_candidate_sheet(
            requests=SUPPLEMENT_REQUESTS,
            e0_index=e0_index,
            decisions=supplement_decisions_by_id,
            e0_root=e0_root,
            output_file=contact_dir / "e0_supplement_all_candidates_and_decisions.png",
        )

        readme = f"""# RootScope machine-curated provisional v2

Status: `{STATUS}`

V2 is independent from and downstream of frozen `{V1_NAME}`. It inherits v1
assets byte-for-byte, adds only the enumerated E0 machine-visual supplement,
and never writes v1 or E0 `review/human_decisions`.

Every record remains `training_eligible=false`, `print_eligible=false`,
`human_reviewed=false`, `data_locked=false`, and all authority bits are false.
The split is only an experimental suggestion and requires a later explicit
opt-in. Supplemental `young_tree` records assert visible plant morphology only;
biological age is explicitly unverified.

- `manifest.jsonl`: self-contained copied assets with source/copy SHA-256.
- `source_decision_manifest.jsonl`: inherited/selected/excluded/unresolved machine decisions.
- `experimental_split_suggestion.json`: creator-group-safe roles and minimum attainment.
- `contact_sheets/`: selected classes and all supplement candidate decisions.
- `receipt.json`, `SHA256SUMS`: integrity and immutability evidence.
"""
        write_text(staging / "README.md", readme)

        v1_tree_after = tree_sha256(v1_root)
        v1_manifest_sha_after = sha256_file(v1_root / "manifest.jsonl")
        v1_sums_sha_after = sha256_file(v1_root / "SHA256SUMS")
        human_tree_after = tree_sha256(human_root)
        human_journal_after = sha256_file(human_journal)
        if (
            v1_tree_before != v1_tree_after
            or v1_manifest_sha_before != v1_manifest_sha_after
            or v1_sums_sha_before != v1_sums_sha_after
        ):
            raise RuntimeError("frozen v1 changed during v2 build")
        if human_tree_before != human_tree_after or human_journal_before != human_journal_after:
            raise RuntimeError("formal E0 human_decisions changed during v2 build")

        target_failures = [
            {
                "class_id": class_id,
                "train_deficit": values["train_deficit"],
                "val_deficit": values["val_deficit"],
            }
            for class_id, values in sorted(attainment.items())
            if not values["both_met"]
        ]
        payload_files = sorted(
            (path for path in staging.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(staging).as_posix(),
        )
        payload_root = sha256_bytes(
            "".join(
                f"{path.relative_to(staging).as_posix()}\0{sha256_file(path)}\n" for path in payload_files
            ).encode("utf-8")
        )
        policy_payload = {
            "supplement_requests": SUPPLEMENT_REQUESTS,
            "supplement_visual_exclusions": SUPPLEMENT_VISUAL_EXCLUSIONS,
            "train_minimums": TRAIN_MINIMUMS,
            "val_minimums": VAL_MINIMUMS,
            "status": STATUS,
        }
        receipt = {
            "schema_version": "rootscope.machine_curated_provisional_receipt.v2",
            "status": STATUS,
            "authority": authority_false(),
            "formal_a1_dataset": False,
            "human_reviewed": False,
            "data_locked": False,
            "training_eligible": False,
            "print_eligible": False,
            "experimental_training_switch_required": True,
            "implementation_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "selection_policy_sha256": sha256_bytes(canonical_json(policy_payload).encode("utf-8")),
            "audit": audit,
            "split_target_attainment": attainment,
            "all_split_targets_met": not target_failures,
            "split_target_failures": target_failures,
            "manifest_sha256": sha256_bytes(manifest_text.encode("utf-8")),
            "source_decision_manifest_sha256": sha256_bytes(decision_text.encode("utf-8")),
            "payload_root_sha256_before_receipt": payload_root,
            "source_e0_manifest_sha256": e0_manifest_sha,
            "source_strict_labels_sha256": sha256_file(strict_path),
            "frozen_v1": {
                "path": v1_root.relative_to(workspace).as_posix(),
                "tree_sha256_before": v1_tree_before,
                "tree_sha256_after": v1_tree_after,
                "manifest_sha256_before": v1_manifest_sha_before,
                "manifest_sha256_after": v1_manifest_sha_after,
                "sha256sums_sha256_before": v1_sums_sha_before,
                "sha256sums_sha256_after": v1_sums_sha_after,
                "unchanged": True,
            },
            "formal_human_decisions": {
                "path": human_root.relative_to(workspace).as_posix(),
                "tree_sha256_before": human_tree_before,
                "tree_sha256_after": human_tree_after,
                "decision_journal_sha256_before": human_journal_before,
                "decision_journal_sha256_after": human_journal_after,
                "unchanged": True,
            },
            "young_tree_supplement_scope": {
                "selected_count": sum(
                    1
                    for row in selected
                    if row["v2_origin"] == "E0_MACHINE_VISUAL_SUPPLEMENT" and row["class_id"] == "young_tree"
                ),
                "biological_age_verified": False,
                "status": "UNCERTAIN_VISIBLE_PLANT_MORPHOLOGY_ONLY",
            },
            "visual_adjudication": {
                "source": "main_agent_and_subagent_machine_inspection",
                "human_review": False,
                "contact_sheet": "contact_sheets/e0_supplement_all_candidates_and_decisions.png",
                "selected_label_basis": "machine_visual_adjudication_not_human_truth",
                "excluded_mature_or_nonconforming_young_pageids": sorted(
                    pageid
                    for pageid in SUPPLEMENT_REQUESTS["young_tree"]
                    if pageid in SUPPLEMENT_VISUAL_EXCLUSIONS
                ),
            },
            "unresolved_request_count": len(unresolved),
            "explicit_non_claims": [
                "HUMAN_REVIEWED",
                "RIGHTS_APPROVED",
                "A1_DATASET",
                "TRAIN_READY",
                "FORMAL_SPLIT_ASSIGNED",
                "PRINT_ELIGIBLE",
                "DATA_LOCKED",
                "MODEL_QUALIFIED",
                "BIOLOGICAL_AGE_VERIFIED_FOR_SUPPLEMENTAL_YOUNG_TREE",
            ],
        }
        write_json(staging / "receipt.json", receipt)
        all_files = sorted(
            (path for path in staging.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(staging).as_posix(),
        )
        write_text(
            staging / "SHA256SUMS",
            "".join(
                f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n" for path in all_files
            ),
        )
        os.replace(staging, output)
        return output
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--output", type=Path, default=workspace / "datasets" / OUTPUT_NAME)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve(strict=True)
    output = args.output.resolve(strict=False)
    expected = (workspace / "datasets" / OUTPUT_NAME).resolve(strict=False)
    if output != expected:
        raise ValueError(f"refusing non-standard v2 output: {output}")
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"v2 output exists; pass --replace: {output}")
        if output.is_symlink() or not output.is_dir() or output.parent != expected.parent or output.name != OUTPUT_NAME:
            raise ValueError(f"unsafe v2 replace target: {output}")
        shutil.rmtree(output)
    print(build_pack(workspace=workspace, output=output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
