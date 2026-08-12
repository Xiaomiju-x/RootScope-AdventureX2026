#!/usr/bin/env python3
"""Build the isolated RootScope machine-curated experimental image pack.

This builder is intentionally downstream of the frozen E0 formal review area.
It copies source bytes into a separate dataset, emits machine-only decisions and
split *suggestions*, and keeps every formal authority bit false.  It never
writes E0 ``review/human_decisions`` or any formal data-lock artifact.
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


STATUS = "MACHINE_CURATED_EXPERIMENTAL_ONLY_NOT_HUMAN_REVIEWED_NOT_A1_NOT_DATA_LOCKED"
OUTPUT_NAME = "rootscope_machine_curated_provisional_v1"
UNKNOWN_LIMIT_MAX = 31

AUTHORITY_FALSE = {
    "data_locked": False,
    "dataset_manifest_write": False,
    "human_review": False,
    "model_qualification": False,
    "print_eligibility": False,
    "rights_approval": False,
    "split_assignment": False,
    "training_eligibility": False,
    "visual_truth": False,
}

SOURCE_DATASETS = {
    "E0": "desert_plants_wikimedia_staging_e0",
    "E1": "desert_plants_whole_plant_reacquisition_e1",
    "E2": "desert_plants_young_tree_reacquisition_e2",
}

# These lists are explicit machine-visual adjudications.  They are not human
# labels.  The requested pool is kept here so missing IDs cannot be silently
# ignored or substituted.
TARGET_REQUESTS: dict[str, dict[str, Any]] = {
    "grass_clump": {
        "source_dataset": "E1",
        "pageids": [
            114121472,
            38233728,
            38234300,
            74079989,
            74079996,
            81223396,
            95511816,
        ],
        "visual_exclusions": {
            38234300: "seedhead/detail crop; plant base and full clump are not visible",
        },
    },
    "low_shrub": {
        "source_dataset": "E1",
        "pageids": [
            142791312,
            54947569,
            60750960,
            66745979,
            68787114,
            68787139,
            76109130,
            94220593,
            94700516,
            94748691,
        ],
        "visual_exclusions": {},
    },
    "young_tree": {
        "source_dataset": "E2",
        "pageids": [
            105533532,
            105533534,
            105533535,
            105533543,
            133359583,
            137881651,
            18394775,
            22701613,
            25062664,
            59265209,
            70606244,
            75760716,
            88289656,
            98911085,
        ],
        "visual_exclusions": {
            133359583: "a hand is visibly holding the pot; hand-held presentation fails the conservative gate",
            137881651: "cotyledon-stage plug with insufficient young-tree structure for this provisional class",
            18394775: "small ground plant has insufficient tree/sapling structural evidence",
            22701613: "ambiguous clustered sprouts; isolated single young tree is not established",
            25062664: "two potted seedlings; not an isolated single plant",
            59265209: "hand-held/potted presentation is too ambiguous for the conservative whole-sapling gate",
        },
    },
}

PRINT_HOLDOUT_PAGEIDS = {
    "grass_clump": [38233728, 74079996],
    "low_shrub": [66745979, 94700516],
    "young_tree": [75760716, 98911085],
}

# These IDs appeared in an earlier request but do not exist in E2.  Corrected
# IDs were later requested explicitly; this file still records the historical
# misses so no implementation can claim that it silently fixed them.
HISTORICAL_UNRESOLVED_IDS = [137981651, 183947751, 227016131]

# Filled only after pixel-level inspection of the dedicated UNKNOWN contact
# sheet.  A strict UNKNOWN_CANDIDATE machine result is necessary but not
# sufficient; obvious target plants are excluded here.
UNKNOWN_VISUAL_EXCLUSIONS: dict[int, str] = {}


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


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def indexed_by_pageid(rows: Sequence[dict[str, Any]], *, label: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        pageid = row.get("pageid")
        if isinstance(pageid, bool) or not isinstance(pageid, int):
            raise ValueError(f"{label} contains invalid pageid {pageid!r}")
        if pageid in result:
            raise ValueError(f"{label} contains duplicate pageid {pageid}")
        result[pageid] = row
    return result


def safe_source_image(dataset_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source image path {value!r}")
    candidate = (dataset_root / relative).resolve(strict=True)
    try:
        candidate.relative_to(dataset_root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"source image escapes dataset root: {value!r}") from error
    if not candidate.is_file():
        raise ValueError(f"source image is not a file: {candidate}")
    return candidate


def tree_sha256(root: Path) -> str:
    rows: list[str] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        relative = path.relative_to(root).as_posix()
        rows.append(f"{relative}\0{sha256_file(path)}\n")
    return sha256_bytes("".join(rows).encode("utf-8"))


def false_authority_record() -> dict[str, bool]:
    return dict(AUTHORITY_FALSE)


def machine_status_fields() -> dict[str, Any]:
    return {
        "authority": false_authority_record(),
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


def decision_base(*, pageid: int, class_id: str, source_dataset: str) -> dict[str, Any]:
    return {
        "schema_version": "rootscope.machine_curated_source_decision.v1",
        "pageid": pageid,
        "candidate_class": class_id,
        "source_dataset": source_dataset,
        **machine_status_fields(),
    }


def source_record_digest(row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(row).encode("utf-8"))


def select_unknown_records(
    strict_rows: Sequence[dict[str, Any]],
    e0_index: dict[int, dict[str, Any]],
    *,
    limit: int,
    exclusions: dict[int, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 <= limit <= UNKNOWN_LIMIT_MAX:
        raise ValueError(f"unknown limit must be within [0, {UNKNOWN_LIMIT_MAX}]")
    exclusions = exclusions or {}
    candidates = [row for row in strict_rows if row.get("final_label") == "UNKNOWN_CANDIDATE"]
    candidates.sort(
        key=lambda row: (
            sha256_bytes(str(row.get("source_group") or f"commons:{row['pageid']}").encode("utf-8")),
            int(row["pageid"]),
        )
    )
    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    seen_creators: set[str] = set()
    for strict_row in candidates:
        pageid = int(strict_row["pageid"])
        source = e0_index.get(pageid)
        decision = decision_base(pageid=pageid, class_id="unknown", source_dataset="E0")
        decision["strict_final_label"] = strict_row.get("final_label")
        decision["strict_label_record_sha256"] = source_record_digest(strict_row)
        if source is None:
            decision.update(
                disposition="UNRESOLVED_SOURCE_RECORD_MISSING",
                selected=False,
                reason="strict UNKNOWN_CANDIDATE has no E0 manifest row",
            )
            decisions.append(decision)
            continue
        creator_group = str(source.get("creator_group") or f"MISSING_CREATOR:{pageid}")
        decision["creator_group"] = creator_group
        decision["source_group"] = source.get("source_group")
        if pageid in exclusions:
            decision.update(
                disposition="EXCLUDE_VISUAL_UNKNOWN_GATE",
                selected=False,
                reason=exclusions[pageid],
            )
        elif creator_group in seen_creators:
            decision.update(
                disposition="EXCLUDE_UNKNOWN_CREATOR_GROUP_CAP",
                selected=False,
                reason="creator_group cap is one selected UNKNOWN record",
            )
        elif len(selected) >= limit:
            decision.update(
                disposition="HOLD_UNKNOWN_LIMIT_REACHED",
                selected=False,
                reason=f"deterministic UNKNOWN cap reached ({limit})",
            )
        else:
            seen_creators.add(creator_group)
            selected.append(source)
            decision.update(
                disposition="SELECTED_MACHINE_CURATED_UNKNOWN",
                selected=True,
                reason="strict UNKNOWN_CANDIDATE plus machine visual screen; not human truth",
            )
        decisions.append(decision)
    return selected, decisions


def experimental_role(creator_group: str, print_creator_groups: set[str]) -> str:
    if creator_group in print_creator_groups:
        return "CREATOR_GROUP_HOLDOUT_NOT_TRAIN"
    bucket = int(sha256_bytes(creator_group.encode("utf-8"))[:8], 16) % 5
    return "EXPERIMENTAL_VAL_SUGGESTION" if bucket == 0 else "EXPERIMENTAL_TRAIN_SUGGESTION"


def assign_experimental_roles(records: list[dict[str, Any]]) -> None:
    print_creator_groups = {
        str(record["creator_group"])
        for record in records
        if record.get("print_holdout_candidate") is True
    }
    for record in records:
        creator = str(record["creator_group"])
        if record.get("print_holdout_candidate") is True:
            role = "PRINT_DEMO_HOLDOUT_NOT_TRAIN"
        else:
            role = experimental_role(creator, print_creator_groups)
        record["experimental_split_suggestion"] = role


def validate_selected_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("selected provisional manifest is empty")
    pageids: set[int] = set()
    source_groups: set[str] = set()
    content_hashes: set[str] = set()
    creator_roles: dict[str, set[str]] = defaultdict(set)
    unknown_creators: set[str] = set()
    print_pageids: set[int] = set()
    for record in records:
        pageid = int(record["pageid"])
        if pageid in pageids:
            raise ValueError(f"duplicate selected pageid {pageid}")
        pageids.add(pageid)
        source_group = str(record["source_group"])
        if source_group in source_groups:
            raise ValueError(f"duplicate source_group {source_group}")
        source_groups.add(source_group)
        digest = str(record["copied_image_sha256"])
        if digest in content_hashes:
            raise ValueError(f"duplicate copied content hash {digest}")
        content_hashes.add(digest)
        for key, expected in machine_status_fields().items():
            if record.get(key) != expected:
                raise ValueError(f"record {pageid} violates fail-closed field {key}")
        creator = str(record["creator_group"])
        role = str(record["experimental_split_suggestion"])
        creator_roles[creator].add(role)
        if record["class_id"] == "unknown":
            if creator in unknown_creators:
                raise ValueError(f"UNKNOWN creator_group cap violated: {creator}")
            unknown_creators.add(creator)
        if record.get("print_holdout_candidate"):
            print_pageids.add(pageid)
            if role != "PRINT_DEMO_HOLDOUT_NOT_TRAIN":
                raise ValueError(f"print holdout {pageid} received role {role}")
    for creator, roles in creator_roles.items():
        normal = roles & {"EXPERIMENTAL_TRAIN_SUGGESTION", "EXPERIMENTAL_VAL_SUGGESTION"}
        held = roles & {"PRINT_DEMO_HOLDOUT_NOT_TRAIN", "CREATOR_GROUP_HOLDOUT_NOT_TRAIN"}
        if len(normal) > 1 or (normal and held):
            raise ValueError(f"creator_group role leakage: {creator} -> {sorted(roles)}")
    return {
        "selected_count": len(records),
        "class_counts": dict(sorted(Counter(str(row["class_id"]) for row in records).items())),
        "experimental_role_counts": dict(
            sorted(Counter(str(row["experimental_split_suggestion"]) for row in records).items())
        ),
        "print_holdout_count": len(print_pageids),
        "creator_group_count": len(creator_roles),
        "unknown_creator_group_count": len(unknown_creators),
        "source_group_overlap_count": 0,
        "copied_sha256_overlap_count": 0,
        "creator_role_leakage_count": 0,
    }


def resolve_font() -> ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\consola.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=14)
    return ImageFont.load_default()


def render_contact_sheet(
    records: Sequence[dict[str, Any]], output: Path, *, title: str, columns: int = 5
) -> None:
    font = resolve_font()
    cell_w, image_h, label_h, header_h = 280, 190, 82, 58
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, header_h + rows * (image_h + label_h)), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width - 1, header_h - 1), fill="#182433")
    draw.text((12, 7), title, fill="white", font=font)
    draw.text((12, 31), "MACHINE ONLY | NOT HUMAN REVIEWED | NOT A1 | NOT DATA LOCKED", fill="#ffcf66", font=font)
    for index, record in enumerate(records):
        grid_row, column = divmod(index, columns)
        x = column * cell_w
        y = header_h + grid_row * (image_h + label_h)
        dataset_root = output.parent.parent
        with Image.open(dataset_root / str(record["filename"])) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            tile = ImageOps.contain(image, (cell_w - 8, image_h - 8), method=Image.Resampling.LANCZOS)
        backdrop = Image.new("RGB", (cell_w, image_h), "#d6d6d6")
        backdrop.paste(tile, ((cell_w - tile.width) // 2, (image_h - tile.height) // 2))
        sheet.paste(backdrop, (x, y))
        draw = ImageDraw.Draw(sheet)
        label_top = y + image_h
        draw.rectangle((x, label_top, x + cell_w - 1, label_top + label_h - 1), fill="white", outline="#aaaaaa")
        label = (
            f"class={record['class_id']} pageid={record['pageid']}\n"
            f"role={record['experimental_split_suggestion']}\n"
            f"creator={str(record['creator_group'])[-16:]}\n"
            f"sha={str(record['copied_image_sha256'])[:12]}"
        )
        draw.multiline_text((x + 6, label_top + 4), label, fill="#111111", font=font, spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=False, compress_level=9)


def selected_record(
    *,
    source: dict[str, Any],
    source_dataset: str,
    source_dataset_root: Path,
    source_manifest_sha256: str,
    class_id: str,
    output_root: Path,
    visual_note: str,
    print_holdout_candidate: bool,
) -> dict[str, Any]:
    pageid = int(source["pageid"])
    source_image = safe_source_image(source_dataset_root, str(source["filename"]))
    actual_source_sha = sha256_file(source_image)
    declared_sha = str(source.get("download_sha256") or "")
    if actual_source_sha != declared_sha:
        raise ValueError(
            f"source image SHA mismatch for {source_dataset}:{pageid}: "
            f"manifest={declared_sha} actual={actual_source_sha}"
        )
    extension = source_image.suffix.lower() or ".img"
    destination_relative = Path("images") / class_id / f"{class_id}_{pageid}_{actual_source_sha[:12]}{extension}"
    destination = output_root / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_image, destination)
    copied_sha = sha256_file(destination)
    if copied_sha != actual_source_sha:
        raise ValueError(f"copy SHA mismatch for {source_dataset}:{pageid}")
    record = {
        "schema_version": "rootscope.machine_curated_provisional_asset.v1",
        "asset": f"provisional:{source_dataset.lower()}:{pageid}@sha256:{copied_sha}",
        "class_id": class_id,
        "pageid": pageid,
        "filename": destination_relative.as_posix(),
        "copied_image_sha256": copied_sha,
        "source_image_sha256": actual_source_sha,
        "source_dataset": source_dataset,
        "source_dataset_name": SOURCE_DATASETS[source_dataset],
        "source_manifest_sha256": source_manifest_sha256,
        "source_record_sha256": source_record_digest(source),
        "source_image_path": str(source["filename"]),
        "source_group": source["source_group"],
        "creator_group": source["creator_group"],
        "commons_sha1": source.get("commons_sha1"),
        "dhash64": source.get("dhash64"),
        "domain": source.get("domain"),
        "source_provider": source.get("source_provider"),
        "source_page": source.get("source_page"),
        "title": source.get("title"),
        "artist": source.get("artist"),
        "credit": source.get("credit"),
        "license": source.get("license"),
        "license_canonical_id": source.get("license_canonical_id"),
        "license_canonical_name": source.get("license_canonical_name"),
        "license_canonical_url": source.get("license_canonical_url"),
        "license_binding_id": source.get("license_binding_id"),
        "rights_review_status": source.get("rights_review_status"),
        "machine_decision": "SELECTED_MACHINE_CURATED_EXPERIMENTAL",
        "label_basis": "machine_visual_adjudication_not_human_truth",
        "visual_adjudication_note": visual_note,
        "print_holdout_candidate": print_holdout_candidate,
        **machine_status_fields(),
    }
    return record


def build_pack(*, workspace: Path, output: Path, unknown_limit: int) -> Path:
    workspace = workspace.resolve(strict=True)
    datasets = (workspace / "datasets").resolve(strict=True)
    expected_output = (datasets / OUTPUT_NAME).resolve(strict=False)
    if output.resolve(strict=False) != expected_output:
        raise ValueError(f"production output must be exactly {expected_output}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}; use --replace at CLI")

    source_roots = {key: datasets / name for key, name in SOURCE_DATASETS.items()}
    for key, root in source_roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"missing {key} source dataset: {root}")
    human_root = source_roots["E0"] / "review" / "human_decisions"
    human_tree_before = tree_sha256(human_root)
    human_journal = human_root / "decision_journal.jsonl"
    human_journal_before = sha256_file(human_journal)

    source_rows: dict[str, list[dict[str, Any]]] = {
        key: load_jsonl(root / "manifest.jsonl") for key, root in source_roots.items()
    }
    source_indexes = {
        key: indexed_by_pageid(rows, label=f"{key} manifest") for key, rows in source_rows.items()
    }
    source_manifest_hashes = {
        key: sha256_file(root / "manifest.jsonl") for key, root in source_roots.items()
    }
    strict_path = source_roots["E0"] / "review" / "ai_final_labels_v1" / "strict_structure_labels.jsonl"
    strict_rows = load_jsonl(strict_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.tmp-", dir=str(output.parent))).resolve()
    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = [
        {
            "pageid": pageid,
            "status": "HISTORICAL_REQUESTED_ID_NOT_FOUND_NO_SILENT_SUBSTITUTION",
            "source_dataset": "E2",
        }
        for pageid in HISTORICAL_UNRESOLVED_IDS
    ]
    try:
        for class_id, request in TARGET_REQUESTS.items():
            source_key = str(request["source_dataset"])
            source_index = source_indexes[source_key]
            visual_exclusions = dict(request["visual_exclusions"])
            holdout_ids = set(PRINT_HOLDOUT_PAGEIDS[class_id])
            for pageid in request["pageids"]:
                decision = decision_base(pageid=pageid, class_id=class_id, source_dataset=source_key)
                source = source_index.get(pageid)
                if source is None:
                    decision.update(
                        disposition="UNRESOLVED_REQUESTED_ID_NOT_FOUND",
                        selected=False,
                        reason="requested pageid is absent from source manifest; no substitution made",
                    )
                    decisions.append(decision)
                    unresolved.append(
                        {
                            "pageid": pageid,
                            "status": "CURRENT_REQUESTED_ID_NOT_FOUND_NO_SILENT_SUBSTITUTION",
                            "source_dataset": source_key,
                        }
                    )
                    continue
                if source.get("class_id") != class_id:
                    raise ValueError(
                        f"requested {source_key}:{pageid} class mismatch: "
                        f"expected {class_id}, got {source.get('class_id')}"
                    )
                if pageid in visual_exclusions:
                    decision.update(
                        disposition="EXCLUDE_CONSERVATIVE_MACHINE_VISUAL_GATE",
                        selected=False,
                        reason=visual_exclusions[pageid],
                        source_group=source.get("source_group"),
                        creator_group=source.get("creator_group"),
                    )
                    decisions.append(decision)
                    continue
                visual_note = (
                    "machine-only conservative whole-plant structural pass; requires independent "
                    "human visual and rights review before any formal eligibility"
                )
                record = selected_record(
                    source=source,
                    source_dataset=source_key,
                    source_dataset_root=source_roots[source_key],
                    source_manifest_sha256=source_manifest_hashes[source_key],
                    class_id=class_id,
                    output_root=staging,
                    visual_note=visual_note,
                    print_holdout_candidate=pageid in holdout_ids,
                )
                selected.append(record)
                decision.update(
                    disposition="SELECTED_MACHINE_CURATED_TARGET",
                    selected=True,
                    reason=visual_note,
                    source_group=source.get("source_group"),
                    creator_group=source.get("creator_group"),
                    print_holdout_candidate=pageid in holdout_ids,
                )
                decisions.append(decision)

        unknown_sources, unknown_decisions = select_unknown_records(
            strict_rows,
            source_indexes["E0"],
            limit=unknown_limit,
            exclusions=UNKNOWN_VISUAL_EXCLUSIONS,
        )
        decisions.extend(unknown_decisions)
        selected_unknown_ids = {int(row["pageid"]) for row in unknown_sources}
        for source in unknown_sources:
            pageid = int(source["pageid"])
            strict_row = next(row for row in strict_rows if int(row["pageid"]) == pageid)
            record = selected_record(
                source=source,
                source_dataset="E0",
                source_dataset_root=source_roots["E0"],
                source_manifest_sha256=source_manifest_hashes["E0"],
                class_id="unknown",
                output_root=staging,
                visual_note=(
                    "strict UNKNOWN_CANDIDATE machine result plus machine visual non-target screen; "
                    "not human-reviewed and not formal ground truth"
                ),
                print_holdout_candidate=False,
            )
            record["strict_structure_label_sha256"] = source_record_digest(strict_row)
            selected.append(record)
        if len(selected_unknown_ids) != len(unknown_sources):
            raise ValueError("duplicate UNKNOWN selection")

        selected.sort(key=lambda row: (str(row["class_id"]), int(row["pageid"])))
        assign_experimental_roles(selected)
        audit = validate_selected_records(selected)

        manifest_text = jsonl_text(selected)
        decision_text = jsonl_text(
            sorted(decisions, key=lambda row: (str(row["candidate_class"]), int(row["pageid"])))
        )
        write_text(staging / "manifest.jsonl", manifest_text)
        write_text(staging / "source_decision_manifest.jsonl", decision_text)
        write_json(
            staging / "unresolved_requested_ids.json",
            {
                "schema_version": "rootscope.machine_curated_unresolved_requests.v1",
                "status": STATUS,
                "authority": false_authority_record(),
                "records": unresolved,
                "no_silent_substitution": True,
            },
        )

        roles: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in selected:
            roles[str(record["experimental_split_suggestion"])].append(
                {
                    "asset": record["asset"],
                    "class_id": record["class_id"],
                    "creator_group": record["creator_group"],
                    "pageid": record["pageid"],
                    "source_group": record["source_group"],
                }
            )
        split_suggestion = {
            "schema_version": "rootscope.machine_curated_experimental_split_suggestion.v1",
            "status": STATUS,
            "authority": false_authority_record(),
            "formal_split_assignment": False,
            "experimental_training_switch_required": True,
            "grouping_key": "creator_group; source_group and content SHA must also remain disjoint",
            "policy": (
                "explicit print pageids and every selected asset sharing their creator_group are held "
                "out; all other creator groups use deterministic sha256 modulo-5 (bucket 0 validation)"
            ),
            "roles": {key: value for key, value in sorted(roles.items())},
        }
        write_json(staging / "experimental_split_suggestion.json", split_suggestion)

        attribution = [
            "# RootScope machine-curated provisional v1｜来源与署名",
            "",
            f"> Status: `{STATUS}`",
            "> 机器筛选实验包；不是人工视觉结论、权利批准、训练许可或打印许可。",
            "",
        ]
        for record in selected:
            license_name = record.get("license_canonical_name") or record.get("license") or "UNKNOWN"
            license_url = record.get("license_canonical_url")
            license_label = f"[{license_name}]({license_url})" if license_url else str(license_name)
            attribution.append(
                f"- `{record['filename']}` — {record.get('artist') or 'UNKNOWN'} — "
                f"[{record.get('title') or record['pageid']}]({record.get('source_page')}) — "
                f"{license_label} — source `{record['source_dataset']}:{record['pageid']}`"
            )
        write_text(staging / "ATTRIBUTION.md", "\n".join(attribution) + "\n")

        contact_dir = staging / "contact_sheets"
        for class_id in ("grass_clump", "low_shrub", "young_tree", "unknown"):
            class_rows = [record for record in selected if record["class_id"] == class_id]
            render_contact_sheet(
                class_rows,
                contact_dir / f"{class_id}.png",
                title=f"RootScope provisional v1 | {class_id} | {len(class_rows)} selected",
            )
        print_rows = [record for record in selected if record.get("print_holdout_candidate")]
        render_contact_sheet(
            print_rows,
            contact_dir / "print_demo_holdout_candidates.png",
            title=f"PRINT HOLDOUT CANDIDATES | {len(print_rows)} | ALL print_eligible=false",
            columns=3,
        )

        readme = f"""# RootScope machine-curated provisional v1

Status: `{STATUS}`

This is an isolated, self-contained **experimental evidence pack**. It is not
the formal A1 dataset. Every record keeps `training_eligible=false`,
`print_eligible=false`, `human_reviewed=false`, `data_locked=false`, and every
authority bit false. The split file is a suggestion only; a later experimental
training command must require an explicit opt-in switch.

- `manifest.jsonl`: copied assets with source/copy SHA-256 and fail-closed flags.
- `source_decision_manifest.jsonl`: selected, excluded, held, and unresolved machine decisions.
- `experimental_split_suggestion.json`: creator-group-safe experimental roles.
- `unresolved_requested_ids.json`: explicit record of IDs that were never silently substituted.
- `ATTRIBUTION.md`: attribution aid; human file-page/non-copyright rights review remains pending.
- `contact_sheets/`: machine-only visual QA sheets.
- `receipt.json` and `SHA256SUMS`: reproducibility and integrity evidence.

Print-demo candidates are kept outside the experimental train/validation
suggestions. Every selected record sharing a print candidate's creator group is
also held out to reduce sequence/photographer leakage.
"""
        write_text(staging / "README.md", readme)

        human_tree_after = tree_sha256(human_root)
        human_journal_after = sha256_file(human_journal)
        if human_tree_before != human_tree_after or human_journal_before != human_journal_after:
            raise RuntimeError("formal E0 human_decisions changed while building; refusing receipt")

        payload_files = sorted(
            (path for path in staging.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(staging).as_posix(),
        )
        payload_rows = [
            f"{path.relative_to(staging).as_posix()}\0{sha256_file(path)}\n" for path in payload_files
        ]
        payload_root = sha256_bytes("".join(payload_rows).encode("utf-8"))
        receipt = {
            "schema_version": "rootscope.machine_curated_provisional_receipt.v1",
            "status": STATUS,
            "authority": false_authority_record(),
            "formal_a1_dataset": False,
            "human_reviewed": False,
            "data_locked": False,
            "training_eligible": False,
            "print_eligible": False,
            "experimental_training_switch_required": True,
            "audit": audit,
            "source_manifest_sha256": source_manifest_hashes,
            "strict_structure_labels_sha256": sha256_file(strict_path),
            "manifest_sha256": sha256_bytes(manifest_text.encode("utf-8")),
            "source_decision_manifest_sha256": sha256_bytes(decision_text.encode("utf-8")),
            "payload_root_sha256_before_receipt": payload_root,
            "formal_human_decisions": {
                "path": human_root.relative_to(workspace).as_posix(),
                "tree_sha256_before": human_tree_before,
                "tree_sha256_after": human_tree_after,
                "unchanged": human_tree_before == human_tree_after,
                "decision_journal_sha256_before": human_journal_before,
                "decision_journal_sha256_after": human_journal_after,
                "decision_journal_unchanged": human_journal_before == human_journal_after,
            },
            "unknown_selection": {
                "strict_unknown_candidate_count": sum(
                    1 for row in strict_rows if row.get("final_label") == "UNKNOWN_CANDIDATE"
                ),
                "selected_count": sum(1 for row in selected if row["class_id"] == "unknown"),
                "limit": unknown_limit,
                "creator_group_max_selected": 1,
            },
            "explicit_non_claims": [
                "HUMAN_REVIEWED",
                "RIGHTS_APPROVED",
                "A1_DATASET",
                "TRAIN_READY",
                "FORMAL_SPLIT_ASSIGNED",
                "PRINT_ELIGIBLE",
                "DATA_LOCKED",
                "MODEL_QUALIFIED",
            ],
        }
        write_json(staging / "receipt.json", receipt)

        all_files = sorted(
            (path for path in staging.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(staging).as_posix(),
        )
        sums = "".join(
            f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n" for path in all_files
        )
        write_text(staging / "SHA256SUMS", sums)

        os.replace(staging, output)
        return output
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    workspace = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--output", type=Path, default=workspace / "datasets" / OUTPUT_NAME)
    parser.add_argument("--unknown-limit", type=int, default=UNKNOWN_LIMIT_MAX)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve(strict=True)
    output = args.output.resolve(strict=False)
    expected = (workspace / "datasets" / OUTPUT_NAME).resolve(strict=False)
    if output != expected:
        raise ValueError(f"refusing non-standard output: {output}; expected {expected}")
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"output exists; pass --replace: {output}")
        if output.is_symlink() or not output.is_dir() or output.parent != expected.parent or output.name != OUTPUT_NAME:
            raise ValueError(f"unsafe replace target: {output}")
        shutil.rmtree(output)
    built = build_pack(workspace=workspace, output=output, unknown_limit=args.unknown_limit)
    print(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
