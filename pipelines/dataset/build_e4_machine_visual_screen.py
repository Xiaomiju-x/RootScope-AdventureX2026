#!/usr/bin/env python3
"""Build the isolated E4 double-machine visual-screen evidence package.

This package records two independent machine inspections followed by root-agent
machine adjudication.  It is not human review, not a human label, and grants no
training, printing, rights, split, or data-lock authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


STATUS = (
    "DOUBLE_INDEPENDENT_MACHINE_VISUAL_REVIEW_ROOT_MACHINE_ADJUDICATED_"
    "NOT_HUMAN_REVIEWED_NOT_TRAIN_ELIGIBLE"
)
DATASET_NAME = "desert_plants_young_tree_category_reacquisition_e4"
OUTPUT_NAME = "machine_visual_screen_v1"
DHASH_ALGORITHM = "rootscope_rgb_center_sample_9x8_v1"
DECISION_SCHEMA = "rootscope.e4_machine_visual_screen_decision.v1"
RECEIPT_SCHEMA = "rootscope.e4_machine_visual_screen_receipt.v1"
CONTRACT_SCHEMA = "rootscope.e4_machine_visual_adjudication_contract.v1"

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

# Root-agent machine adjudication after two independent machine pixel reviews.
# These are frozen machine dispositions, not human labels or formal truth.
DECISIONS: dict[int, tuple[str, str]] = {
    3722930: ("EXCLUDE", "nursery rows contain many cropped trunks and no single complete primary tree"),
    10746033: ("EXCLUDE", "wide hillside contains many guarded saplings and no isolated primary tree"),
    10746071: ("EXCLUDE", "multi-sapling landscape has no isolated primary tree"),
    17871222: ("EXCLUDE", "person and dense mixed nursery vegetation dominate the frame"),
    24898987: ("EXCLUDE", "mass field of seedlings has no single primary target"),
    27834036: ("EXCLUDE", "forest or nursery scene contains many saplings with no isolated target"),
    29077177: ("EXCLUDE", "dense pine seedling mass has overlapping bases and crowns"),
    30736432: ("EXCLUDE", "street scene contains multiple bare trees and buildings with no primary sapling"),
    32887473: ("EXCLUDE", "nursery infrastructure and many trays dominate with no isolated target"),
    33982433: ("EXCLUDE", "printed illustration is not a photograph of a real plant"),
    36522358: ("EXCLUDE", "many bagged nursery saplings and cropped foreground branches prevent a single target"),
    36522412: ("EXCLUDE", "nursery interior contains many bagged plants and surrounding trees"),
    37047880: ("EXCLUDE", "multiple potted flowering herbaceous plants are not a young tree"),
    39951109: ("EXCLUDE", "nursery field contains many young trees and a person with no isolated target"),
    39951119: ("EXCLUDE", "wide nursery field and mature foreground branch provide no single target"),
    39951195: ("EXCLUDE", "dense rows of grafted saplings provide no isolated primary tree"),
    40043047: ("EXCLUDE", "wide field contains many small conifers and no single target"),
    40043048: ("EXCLUDE", "wide field contains many conifers and no single target"),
    42271785: ("EXCLUDE", "many oil-palm seedlings in bags and rows provide no isolated plant"),
    42271824: ("EXCLUDE", "many oil-palm seedlings in bags and rows provide no isolated plant"),
    48741424: ("EXCLUDE", "large field of bagged seedlings has no single primary target"),
    48917417: ("EXCLUDE", "multiple potted leafless grafts lack a single complete crown"),
    49393026: ("EXCLUDE", "hands dominate and the tiny seedling is not visually distinct from a herbaceous seedling"),
    51109582: ("EXCLUDE", "mass nursery bed contains many seedlings and no single target"),
    59714825: ("EXCLUDE", "macro view contains many germinating seeds and no complete tree form"),
    68495029: ("EXCLUDE", "top-down cotyledon seedling lacks a discernible trunk and crown"),
    74067396: ("EXCLUDE", "tiny sprout lacks visually distinctive young-tree structure"),
    77961156: ("EXCLUDE", "multiple bagged seedlings make the target non-unique"),
    79322391: ("EXCLUDE", "greenhouse beds contain tulips and other herbaceous plants"),
    80535124: ("EXCLUDE", "landscape contains multiple mature conifers rather than one young tree"),
    86227079: ("EXCLUDE", "hands and at least two seedlings dominate the planting scene"),
    92774234: ("SELECT", "single upright newly planted young tree with visible base, trunk, and complete crown"),
    112711289: ("EXCLUDE", "many stakes and saplings in a landscape provide no isolated target"),
    112853574: ("EXCLUDE", "historical nursery catalogue page is not a plant photograph"),
    113549358: ("EXCLUDE", "field contains many tree tubes and plants with no single target"),
    115928770: ("EXCLUDE", "palm nursery and multiple established trees provide no isolated target"),
    115928771: ("EXCLUDE", "dense blocks of bagged seedlings provide no single target"),
    115928772: ("EXCLUDE", "multiple overlapping palm seedlings prevent whole-plant isolation"),
    122973026: ("SELECT", "single upright young tree remains structurally complete with visible base, trunk, and crown despite stressed leaves"),
    130133197: ("HOLD", "plausible complete sapling but dense tangled background makes its crown boundary ambiguous"),
    130133198: ("HOLD", "plausible complete sapling but dense flowering brush obscures a clean whole-tree boundary"),
    132723654: ("EXCLUDE", "large mature tree is outside the young-tree target"),
    132723682: ("EXCLUDE", "historical nursery overview contains mass containers and no single target"),
    133282008: ("EXCLUDE", "people planting multiple pine saplings dominate the scene"),
    139512279: ("EXCLUDE", "conifer crown or branch close-up omits the full trunk and base"),
    147998834: ("EXCLUDE", "multiple low ground shoots resemble herbaceous growth and are not isolated"),
    154154242: ("EXCLUDE", "sign dominates while the referenced tree is not visibly evaluable"),
    157332832: ("EXCLUDE", "nursery scene contains multiple protected saplings and mature trees"),
    162586688: ("EXCLUDE", "very large mature tree is outside the young-tree target"),
    162863718: ("EXCLUDE", "multiple pots contain tiny sprouts without complete tree form"),
    162863731: ("EXCLUDE", "person holds a tray of many seedlings so people and multiple targets dominate"),
    162886278: ("EXCLUDE", "two people and a seed bag dominate and no complete young tree is present"),
    163459362: ("EXCLUDE", "multi-tree landscape and many guards provide no isolated target"),
    163966360: ("EXCLUDE", "aerial mass-nursery view contains thousands of pots and people"),
    172566638: ("EXCLUDE", "top-down very young plant and a neighboring plant lack clear tree form"),
    172566639: ("EXCLUDE", "very young plant plus a neighboring plant cannot be distinguished reliably from herbaceous seedlings"),
    173706908: ("HOLD", "single complete seedling is visible but cotyledon-stage morphology remains herbaceous-like"),
    180772202: ("SELECT", "single complete leafless young deciduous tree has a clear base, trunk, and crown"),
    184914979: ("EXCLUDE", "prairie scene contains rows of distant saplings and no primary target"),
    184915021: ("SELECT", "single protected upright young tree has a clear base, trunk, and complete crown"),
    184915109: ("HOLD", "single root area has multiple slender shoots and a weak crown, making sapling versus shrub form ambiguous"),
    189494502: ("EXCLUDE", "rice-field landscape contains no young tree"),
}


class ScreenError(RuntimeError):
    """Fail-closed machine-screen construction error."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pretty_json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path, *, exclude_top_level: frozenset[str] = frozenset()) -> str:
    rows: list[str] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in exclude_top_level:
            continue
        rows.append(f"{relative.as_posix()}\0{sha256_file(path)}\n")
    return sha256_bytes("".join(rows).encode("utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ScreenError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ScreenError(f"non-object JSON at {path}:{line_number}")
            rows.append(value)
    return rows


def safe_child(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ScreenError(f"unsafe relative path {relative_value!r}")
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / relative).resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ScreenError(f"path escapes dataset root: {relative_value!r}") from error
    if not candidate.is_file():
        raise ScreenError(f"expected source file: {candidate}")
    return candidate


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def write_json(path: Path, value: object) -> None:
    write_text(path, pretty_json_text(value))


def jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def false_authority() -> dict[str, bool]:
    return dict(AUTHORITY_FALSE)


def machine_status_fields() -> dict[str, Any]:
    return {
        "status": STATUS,
        "machine_only": True,
        "human_reviewed": False,
        "human_label": False,
        "data_authority": False,
        "rights_approved": False,
        "training_eligible": False,
        "train_eligible": False,
        "print_eligible": False,
        "data_locked": False,
        "data_lock": False,
        "formal_a1_dataset": False,
        "formal_split_assigned": False,
        "split": "UNASSIGNED_DO_NOT_TRAIN",
        "authority": false_authority(),
    }


def review_process_fields() -> dict[str, Any]:
    return {
        "independent_machine_review_count": 2,
        "independent_machine_reviews_completed": True,
        "root_machine_adjudicated": True,
        "root_adjudication_grants_data_authority": False,
        "review_basis": "two_independent_machine_pixel_reviews_plus_root_machine_adjudication",
        "review_is_human_label": False,
    }


def adjudication_contract() -> dict[str, Any]:
    selected = sorted(pageid for pageid, (decision, _reason) in DECISIONS.items() if decision == "SELECT")
    held = sorted(pageid for pageid, (decision, _reason) in DECISIONS.items() if decision == "HOLD")
    excluded = sorted(pageid for pageid, (decision, _reason) in DECISIONS.items() if decision == "EXCLUDE")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": STATUS,
        "source_dataset": "E4",
        "source_dataset_name": DATASET_NAME,
        "pipeline": {
            **review_process_fields(),
            "all_review_actors_are_machine_agents": True,
            "machine_disposition_only": True,
            "human_review_authority": False,
        },
        "frozen_decisions": {
            "SELECT": selected,
            "HOLD": held,
            "EXCLUDE": excluded,
        },
        "select_semantics": "PROVISIONAL_MACHINE_CANDIDATE_ONLY_NOT_TRAIN_ELIGIBLE",
        "explicit_non_claims": [
            "HUMAN_REVIEWED",
            "HUMAN_LABEL",
            "VISUAL_GROUND_TRUTH",
            "DATA_AUTHORITY",
            "RIGHTS_APPROVED",
            "TRAIN_ELIGIBLE",
            "PRINT_ELIGIBLE",
            "DATA_LOCKED",
            "FORMAL_A1_DATASET",
            "FORMAL_SPLIT_ASSIGNED",
        ],
        **machine_status_fields(),
    }


def indexed_manifest(rows: Sequence[dict[str, Any]]) -> dict[int, tuple[int, dict[str, Any]]]:
    result: dict[int, tuple[int, dict[str, Any]]] = {}
    for line_number, row in enumerate(rows, start=1):
        pageid = row.get("pageid")
        if isinstance(pageid, bool) or not isinstance(pageid, int):
            raise ScreenError(f"invalid E4 pageid {pageid!r}")
        if pageid in result:
            raise ScreenError(f"duplicate E4 pageid {pageid}")
        result[pageid] = (line_number, row)
    if set(result) != set(DECISIONS):
        raise ScreenError(
            "frozen machine decisions do not exactly cover E4 manifest: "
            f"missing={sorted(set(result) - set(DECISIONS))}, "
            f"extra={sorted(set(DECISIONS) - set(result))}"
        )
    return result


def image_dhash64(path: Path) -> str:
    with Image.open(path) as opened:
        opened.load()
        rgb = opened.convert("RGB")
    if rgb.width < 1 or rgb.height < 1:
        raise ScreenError(f"invalid image dimensions for {path}")
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


def build_decision_rows(
    dataset_root: Path,
    manifest_rows: Sequence[dict[str, Any]],
    *,
    contract_sha256: str,
) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    collection_receipt_path = dataset_root / "collection_receipt.json"
    manifest_sha = sha256_file(manifest_path)
    collection_receipt_sha = sha256_file(collection_receipt_path)
    index = indexed_manifest(manifest_rows)
    output: list[dict[str, Any]] = []
    for pageid in sorted(index):
        line_number, source = index[pageid]
        decision, reason = DECISIONS[pageid]
        image_path = safe_child(dataset_root, str(source.get("filename", "")))
        image_sha = sha256_file(image_path)
        if source.get("download_sha256") != image_sha:
            raise ScreenError(f"E4 image SHA mismatch for pageid {pageid}")
        dhash = image_dhash64(image_path)
        if source.get("dhash64_algorithm") != DHASH_ALGORITHM:
            raise ScreenError(f"E4 dHash algorithm mismatch for pageid {pageid}")
        if source.get("dhash64") != dhash:
            raise ScreenError(f"E4 image dHash mismatch for pageid {pageid}")
        creator_group = source.get("creator_group")
        source_group = source.get("source_group")
        if not isinstance(creator_group, str) or not creator_group:
            raise ScreenError(f"missing creator_group for pageid {pageid}")
        if not isinstance(source_group, str) or not source_group:
            raise ScreenError(f"missing source_group for pageid {pageid}")
        output.append(
            {
                "schema_version": DECISION_SCHEMA,
                "pageid": pageid,
                "class_id": "young_tree",
                "decision": decision,
                "disposition": (
                    "SELECT_PROVISIONAL_CANDIDATE_ONLY"
                    if decision == "SELECT"
                    else f"{decision}_MACHINE_VISUAL_SCREEN"
                ),
                "reason": reason,
                "provisional_candidate_only": decision == "SELECT",
                "selection_grants_training_eligibility": False,
                "creator_group": creator_group,
                "source_group": source_group,
                "source_provider": source.get("source_provider"),
                "source_page": source.get("source_page"),
                "download_url": source.get("download_url"),
                "commons_sha1": source.get("commons_sha1"),
                "source_dataset": "E4",
                "source_dataset_name": DATASET_NAME,
                "source_manifest_path": f"datasets/{DATASET_NAME}/manifest.jsonl",
                "source_manifest_sha256": manifest_sha,
                "source_manifest_line_number": line_number,
                "source_record_sha256": sha256_bytes(canonical_json(source).encode("utf-8")),
                "source_collection_receipt_path": f"datasets/{DATASET_NAME}/collection_receipt.json",
                "source_collection_receipt_sha256": collection_receipt_sha,
                "image_path": str(source["filename"]).replace("\\", "/"),
                "image_sha256": image_sha,
                "dhash64_algorithm": DHASH_ALGORITHM,
                "dhash64": dhash,
                "dhash64_recomputed_from_original": True,
                "dhash64_matches_source_manifest": True,
                "adjudication_contract_path": (
                    f"datasets/{DATASET_NAME}/review/{OUTPUT_NAME}/adjudication_contract.json"
                ),
                "adjudication_contract_sha256": contract_sha256,
                "visual_basis": "independent_machine_inspection_of_original_pixels_not_human_truth",
                **review_process_fields(),
                **machine_status_fields(),
            }
        )
    return output


def resolve_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ("consolab.ttf", "arialbd.ttf") if bold else ("consola.ttf", "arial.ttf")
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    for path in (
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        ),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_contact_sheet(
    rows: Sequence[dict[str, Any]],
    dataset_root: Path,
    output: Path,
    *,
    columns: int = 4,
) -> None:
    cell_width, image_height, label_height = 390, 235, 150
    header_height = 142
    row_count = (len(rows) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * cell_width, header_height + row_count * (image_height + label_height)),
        "#171717",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = resolve_font(24, bold=True)
    body_font = resolve_font(16)
    small_font = resolve_font(13)
    draw.rectangle((0, 0, canvas.width, header_height), fill="#5c0014")
    draw.text((20, 12), "MACHINE-ONLY E4 VISUAL SCREEN", font=title_font, fill="white")
    draw.text(
        (20, 46),
        "2 INDEPENDENT MACHINE REVIEWS + ROOT MACHINE ADJUDICATION",
        font=body_font,
        fill="#fff2a8",
    )
    draw.text(
        (20, 75),
        "NOT HUMAN REVIEW / NOT HUMAN LABEL / NO DATA AUTHORITY",
        font=body_font,
        fill="white",
    )
    draw.text(
        (20, 104),
        "NOT TRAIN ELIGIBLE / NOT PRINT ELIGIBLE / NOT DATA LOCKED",
        font=body_font,
        fill="white",
    )
    colors = {"SELECT": "#146b3a", "HOLD": "#8a5a00", "EXCLUDE": "#741d1d"}
    for index, row in enumerate(rows):
        column = index % columns
        row_index = index // columns
        left = column * cell_width
        top = header_height + row_index * (image_height + label_height)
        image_path = safe_child(dataset_root, str(row["image_path"]))
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        fitted = ImageOps.contain(image, (cell_width - 12, image_height - 12), Image.Resampling.LANCZOS)
        image_area = Image.new("RGB", (cell_width, image_height), "#262626")
        image_area.paste(fitted, ((cell_width - fitted.width) // 2, (image_height - fitted.height) // 2))
        canvas.paste(image_area, (left, top))
        label_top = top + image_height
        draw.rectangle(
            (left, label_top, left + cell_width - 1, label_top + label_height - 1),
            fill=colors[str(row["decision"])],
            outline="#dddddd",
            width=1,
        )
        draw.text(
            (left + 8, label_top + 7),
            f"pageid={row['pageid']}  {row['decision']}",
            font=body_font,
            fill="white",
        )
        draw.text((left + 8, label_top + 33), str(row["creator_group"]), font=small_font, fill="#f2f2f2")
        draw.text(
            (left + 8, label_top + 53),
            f"dHash64={row['dhash64']}",
            font=small_font,
            fill="#f2f2f2",
        )
        reason_lines = textwrap.wrap(str(row["reason"]), width=46)[:3]
        for line_index, line in enumerate(reason_lines):
            draw.text(
                (left + 8, label_top + 76 + 20 * line_index),
                line,
                font=small_font,
                fill="white",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False, compress_level=9)


def validate_decision_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 62:
        raise ScreenError(f"expected 62 decisions, got {len(rows)}")
    counts = Counter(str(row.get("decision")) for row in rows)
    expected_counts = Counter({"EXCLUDE": 54, "HOLD": 4, "SELECT": 4})
    if counts != expected_counts:
        raise ScreenError(f"unexpected decision counts: {dict(counts)}")
    seen_pageids: set[int] = set()
    seen_source_lines: set[int] = set()
    for row in rows:
        pageid = int(row["pageid"])
        if pageid in seen_pageids:
            raise ScreenError(f"duplicate decision pageid {pageid}")
        seen_pageids.add(pageid)
        source_line = int(row["source_manifest_line_number"])
        if source_line in seen_source_lines:
            raise ScreenError(f"duplicate source manifest line {source_line}")
        seen_source_lines.add(source_line)
        for key, expected in machine_status_fields().items():
            if row.get(key) != expected:
                raise ScreenError(f"pageid {pageid} violates fail-closed field {key}")
        for key, expected in review_process_fields().items():
            if row.get(key) != expected:
                raise ScreenError(f"pageid {pageid} violates review-process field {key}")
        if row.get("dhash64_recomputed_from_original") is not True:
            raise ScreenError(f"pageid {pageid} lacks dHash recomputation evidence")
        if row.get("dhash64_matches_source_manifest") is not True:
            raise ScreenError(f"pageid {pageid} lacks dHash manifest binding")
        if row["decision"] == "SELECT":
            if row.get("provisional_candidate_only") is not True:
                raise ScreenError("SELECT must remain a provisional candidate only")
        elif row.get("provisional_candidate_only") is not False:
            raise ScreenError(f"non-SELECT pageid {pageid} is marked provisional candidate")
    return {
        "total": len(rows),
        "coverage": "62/62",
        "decision_counts": {key: counts[key] for key in ("SELECT", "HOLD", "EXCLUDE")},
        "selected_pageids": sorted(int(row["pageid"]) for row in rows if row["decision"] == "SELECT"),
        "held_pageids": sorted(int(row["pageid"]) for row in rows if row["decision"] == "HOLD"),
        "excluded_pageids": sorted(int(row["pageid"]) for row in rows if row["decision"] == "EXCLUDE"),
        "all_image_sha256_verified": True,
        "all_dhash64_recomputed_and_verified": True,
    }


def protected_input_snapshot(workspace: Path, dataset_root: Path) -> dict[str, Any]:
    protected_names = (
        "desert_plants_v1",
        "desert_plants_wikimedia_staging_e0",
        "desert_plants_whole_plant_reacquisition_e1",
        "desert_plants_young_tree_reacquisition_e2",
        "desert_plants_young_tree_reacquisition_e3",
        "rootscope_machine_curated_provisional_v1",
        "rootscope_machine_curated_provisional_v2",
    )
    roots: dict[str, str] = {}
    for name in protected_names:
        root = (workspace / "datasets" / name).resolve(strict=True)
        roots[f"datasets/{name}"] = tree_sha256(root)
    formal_journal = (
        workspace
        / "datasets"
        / "desert_plants_wikimedia_staging_e0"
        / "review"
        / "human_decisions"
        / "decision_journal.jsonl"
    ).resolve(strict=True)
    return {
        "protected_tree_sha256": roots,
        "formal_decision_journal_sha256": sha256_file(formal_journal),
        "e4_acquisition_tree_excluding_review_sha256": tree_sha256(
            dataset_root, exclude_top_level=frozenset({"review"})
        ),
    }


def artifact_sums(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix())
        if path.name != "SHA256SUMS"
    }


def build_screen(
    *,
    workspace: Path,
    output: Path,
    replace: bool = False,
    allow_nonproduction_output: bool = False,
) -> Path:
    workspace = workspace.resolve(strict=True)
    dataset_root = (workspace / "datasets" / DATASET_NAME).resolve(strict=True)
    expected_output = dataset_root / "review" / OUTPUT_NAME
    if not allow_nonproduction_output and output.resolve(strict=False) != expected_output.resolve(strict=False):
        raise ScreenError(f"production output must be exactly {expected_output}")
    manifest_path = dataset_root / "manifest.jsonl"
    collection_receipt_path = dataset_root / "collection_receipt.json"
    if not manifest_path.is_file() or not collection_receipt_path.is_file():
        raise ScreenError("missing E4 source manifest or collection receipt")

    protected_before = protected_input_snapshot(workspace, dataset_root)
    contract = adjudication_contract()
    contract_text = pretty_json_text(contract)
    contract_sha = sha256_bytes(contract_text.encode("utf-8"))
    manifest_rows = load_jsonl(manifest_path)
    decisions = build_decision_rows(dataset_root, manifest_rows, contract_sha256=contract_sha)
    stats = validate_decision_rows(decisions)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not replace:
        raise FileExistsError(f"output already exists: {output}; pass --replace to rebuild")
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.tmp-", dir=str(output.parent))).resolve()
    try:
        write_text(staging / "adjudication_contract.json", contract_text)
        rows_text = jsonl_text(decisions)
        write_text(staging / "manifest.jsonl", rows_text)
        write_text(staging / "decisions.jsonl", rows_text)
        render_contact_sheet(decisions, dataset_root, staging / "contact_sheet.png")

        if sha256_file(staging / "manifest.jsonl") != sha256_file(staging / "decisions.jsonl"):
            raise ScreenError("decisions compatibility alias differs from screen manifest")
        protected_after_render = protected_input_snapshot(workspace, dataset_root)
        if protected_after_render != protected_before:
            raise ScreenError("protected acquisition, prior datasets, journal, or v1/v2 changed during build")

        screen_manifest_sha = sha256_file(staging / "manifest.jsonl")
        decisions_sha = sha256_file(staging / "decisions.jsonl")
        contact_sha = sha256_file(staging / "contact_sheet.png")
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
            "source_dataset": "E4",
            "source_dataset_name": DATASET_NAME,
            "source_manifest_path": f"datasets/{DATASET_NAME}/manifest.jsonl",
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_collection_receipt_path": f"datasets/{DATASET_NAME}/collection_receipt.json",
            "source_collection_receipt_sha256": sha256_file(collection_receipt_path),
            "source_record_count": len(manifest_rows),
            "screen_manifest_path": f"datasets/{DATASET_NAME}/review/{OUTPUT_NAME}/manifest.jsonl",
            "screen_manifest_sha256": screen_manifest_sha,
            "decisions_path": f"datasets/{DATASET_NAME}/review/{OUTPUT_NAME}/decisions.jsonl",
            "decisions_sha256": decisions_sha,
            "decisions_alias_byte_identical_to_manifest": True,
            "contact_sheet_path": f"datasets/{DATASET_NAME}/review/{OUTPUT_NAME}/contact_sheet.png",
            "contact_sheet_sha256": contact_sha,
            "adjudication_contract_path": (
                f"datasets/{DATASET_NAME}/review/{OUTPUT_NAME}/adjudication_contract.json"
            ),
            "adjudication_contract_sha256": contract_sha,
            "implementation_path": "tools/dataset/build_e4_machine_visual_screen.py",
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
            "statistics": stats,
            "review_pipeline": {
                **review_process_fields(),
                "all_review_actors_are_machine_agents": True,
                "machine_disposition_only": True,
                "human_review_authority": False,
            },
            "binding_guarantees": {
                "creator_group_bound": True,
                "source_group_bound": True,
                "source_record_sha256_bound": True,
                "source_manifest_sha256_bound": True,
                "source_collection_receipt_sha256_bound": True,
                "image_sha256_recomputed_and_bound": True,
                "dhash64_recomputed_and_bound": True,
                "dhash64_algorithm": DHASH_ALGORITHM,
            },
            "protected_inputs": {
                "before": protected_before,
                "after": protected_after_render,
                "unchanged": True,
            },
            "select_semantics": "PROVISIONAL_MACHINE_CANDIDATE_ONLY_NOT_TRAIN_ELIGIBLE",
            "explicit_non_claims": contract["explicit_non_claims"],
            **machine_status_fields(),
        }
        receipt["run_id"] = "sha256:" + sha256_bytes(
            (
                receipt["source_manifest_sha256"]
                + screen_manifest_sha
                + contact_sha
                + contract_sha
            ).encode("utf-8")
        )
        write_json(staging / "receipt.json", receipt)
        sums = artifact_sums(staging)
        write_text(
            staging / "SHA256SUMS",
            "".join(f"{digest}  {relative}\n" for relative, digest in sums.items()),
        )

        protected_final = protected_input_snapshot(workspace, dataset_root)
        if protected_final != protected_before:
            raise ScreenError("protected inputs changed before machine-screen commit")
        if output.exists():
            if output.resolve(strict=True) != expected_output.resolve(strict=True) and not allow_nonproduction_output:
                raise ScreenError(f"refusing to replace unexpected output {output}")
            shutil.rmtree(output)
        staging.replace(output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    workspace = Path(__file__).resolve().parents[2]
    output = workspace / "datasets" / DATASET_NAME / "review" / OUTPUT_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--output", type=Path, default=output)
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = build_screen(workspace=args.workspace, output=args.output, replace=args.replace)
    except (ScreenError, FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2
    print(f"PASS_MACHINE_ONLY: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
