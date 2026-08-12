#!/usr/bin/env python3
"""Build the isolated, machine-only visual screen for E3 young-tree candidates.

The output is evidence about a conservative machine visual inspection.  It is
not a human decision, does not grant rights approval, does not assign a formal
split, and cannot make any record eligible for training or printing.
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


STATUS = "MACHINE_VISUAL_SCREEN_ONLY_NOT_HUMAN_REVIEWED_NOT_TRAIN_ELIGIBLE"
DATASET_NAME = "desert_plants_young_tree_reacquisition_e3"
OUTPUT_NAME = "machine_visual_screen_v1"
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

# Frozen independent machine inspection of the 15 E3 originals.  These are
# machine dispositions, not formal labels or human truth.
DECISIONS: dict[int, tuple[str, str]] = {
    6181451: (
        "EXCLUDE",
        "wide multi-sapling scene with a person; no isolated primary plant",
    ),
    6189155: (
        "HOLD",
        "foreground sapling is plausible but small and grass-obscured; base, trunk, and crown are insufficiently distinct",
    ),
    6191581: (
        "SELECT",
        "single complete medium-small sapling; base, trunk, and crown are visible without human or machinery dominance",
    ),
    6195484: (
        "EXCLUDE",
        "sapling is a tiny distant landscape element; whole-plant structure cannot be evaluated",
    ),
    68792230: (
        "EXCLUDE",
        "people and machinery dominate; the uprooted plant is hand-held",
    ),
    68792234: (
        "EXCLUDE",
        "people dominate and hold an uprooted plant rather than a naturally standing sapling",
    ),
    68792259: (
        "EXCLUDE",
        "people, tractor, and uprooting activity dominate; plant posture is not natural",
    ),
    68792268: (
        "EXCLUDE",
        "tractor and a large soil mass dominate while the sapling structure is obscured",
    ),
    70606247: (
        "EXCLUDE",
        "uprooted branch-like plant lies across grass with no clear base, upright trunk, or complete crown",
    ),
    81762022: (
        "EXCLUDE",
        "mature coarse trunk and foliage detail; not a complete young tree",
    ),
    81762023: (
        "EXCLUDE",
        "clearly a mature large tree",
    ),
    85555314: (
        "EXCLUDE",
        "parasitic seedling and host-branch close-up; no independent whole-tree form",
    ),
    85555315: (
        "EXCLUDE",
        "multiple host branch tips and parasitic growth with no base or isolated whole plant",
    ),
    105533544: (
        "HOLD",
        "single complete cotyledon-stage seedling, but the top-down view lacks tree-distinctive trunk and crown morphology",
    ),
    137881650: (
        "EXCLUDE",
        "multiple crowded cotyledon-stage sprouts; no single primary target or young-tree form",
    ),
}


class ScreenError(RuntimeError):
    """Fail-closed machine-screen construction error."""


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
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
        raise ScreenError(f"expected image file: {candidate}")
    return candidate


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def false_authority() -> dict[str, bool]:
    return dict(AUTHORITY_FALSE)


def machine_status_fields() -> dict[str, Any]:
    return {
        "status": STATUS,
        "machine_only": True,
        "human_reviewed": False,
        "rights_approved": False,
        "training_eligible": False,
        "print_eligible": False,
        "data_locked": False,
        "formal_a1_dataset": False,
        "formal_split_assigned": False,
        "split": "UNASSIGNED_DO_NOT_TRAIN",
        "authority": false_authority(),
    }


def indexed_manifest(rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        pageid = row.get("pageid")
        if isinstance(pageid, bool) or not isinstance(pageid, int):
            raise ScreenError(f"invalid E3 pageid {pageid!r}")
        if pageid in result:
            raise ScreenError(f"duplicate E3 pageid {pageid}")
        result[pageid] = row
    if set(result) != set(DECISIONS):
        raise ScreenError(
            "frozen machine decisions do not exactly cover E3 manifest: "
            f"missing={sorted(set(result) - set(DECISIONS))}, "
            f"extra={sorted(set(DECISIONS) - set(result))}"
        )
    return result


def build_decision_rows(dataset_root: Path, manifest_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_sha = sha256_file(manifest_path)
    index = indexed_manifest(manifest_rows)
    output: list[dict[str, Any]] = []
    for pageid in sorted(index):
        source = index[pageid]
        decision, reason = DECISIONS[pageid]
        image_path = safe_child(dataset_root, str(source.get("filename", "")))
        image_sha = sha256_file(image_path)
        if source.get("download_sha256") != image_sha:
            raise ScreenError(f"E3 image SHA mismatch for pageid {pageid}")
        creator_group = source.get("creator_group")
        source_group = source.get("source_group")
        if not isinstance(creator_group, str) or not creator_group:
            raise ScreenError(f"missing creator_group for pageid {pageid}")
        if not isinstance(source_group, str) or not source_group:
            raise ScreenError(f"missing source_group for pageid {pageid}")
        output.append(
            {
                "schema_version": "rootscope.e3_machine_visual_screen_decision.v1",
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
                "source_dataset": "E3",
                "source_dataset_name": DATASET_NAME,
                "source_manifest_path": f"datasets/{DATASET_NAME}/manifest.jsonl",
                "source_manifest_sha256": manifest_sha,
                "source_record_sha256": sha256_bytes(canonical_json(source).encode("utf-8")),
                "image_path": str(source["filename"]).replace("\\", "/"),
                "image_sha256": image_sha,
                "visual_basis": "independent_machine_inspection_of_original_pixels_not_human_truth",
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
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_contact_sheet(
    rows: Sequence[dict[str, Any]],
    dataset_root: Path,
    output: Path,
    *,
    columns: int = 3,
) -> None:
    cell_width, image_height, label_height = 420, 265, 132
    header_height = 112
    rows_count = (len(rows) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * cell_width, header_height + rows_count * (image_height + label_height)),
        "#171717",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = resolve_font(25, bold=True)
    body_font = resolve_font(17)
    small_font = resolve_font(14)
    draw.rectangle((0, 0, canvas.width, header_height), fill="#6f0012")
    draw.text((20, 14), "MACHINE ONLY / NOT HUMAN REVIEWED", font=title_font, fill="white")
    draw.text(
        (20, 50),
        "NOT TRAIN ELIGIBLE / NOT PRINT ELIGIBLE / NOT DATA LOCKED",
        font=body_font,
        fill="#fff2a8",
    )
    draw.text(
        (20, 78),
        "E3 young-tree candidate visual screen v1 - original image pixels",
        font=small_font,
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
            (left + 10, label_top + 8),
            f"pageid={row['pageid']}  {row['decision']}",
            font=body_font,
            fill="white",
        )
        creator = str(row["creator_group"])
        draw.text((left + 10, label_top + 35), creator, font=small_font, fill="#f2f2f2")
        reason_lines = textwrap.wrap(str(row["reason"]), width=49)[:3]
        for line_index, line in enumerate(reason_lines):
            draw.text(
                (left + 10, label_top + 58 + 20 * line_index),
                line,
                font=small_font,
                fill="white",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False, compress_level=9)


def validate_decision_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 15:
        raise ScreenError(f"expected 15 decisions, got {len(rows)}")
    counts = Counter(str(row.get("decision")) for row in rows)
    if counts != Counter({"EXCLUDE": 12, "HOLD": 2, "SELECT": 1}):
        raise ScreenError(f"unexpected decision counts: {dict(counts)}")
    seen_pageids: set[int] = set()
    for row in rows:
        pageid = int(row["pageid"])
        if pageid in seen_pageids:
            raise ScreenError(f"duplicate decision pageid {pageid}")
        seen_pageids.add(pageid)
        for key, expected in machine_status_fields().items():
            if row.get(key) != expected:
                raise ScreenError(f"pageid {pageid} violates fail-closed field {key}")
        if row["decision"] == "SELECT":
            if row.get("provisional_candidate_only") is not True:
                raise ScreenError("SELECT must remain a provisional candidate only")
        elif row.get("provisional_candidate_only") is not False:
            raise ScreenError(f"non-SELECT pageid {pageid} is marked provisional candidate")
    return {
        "total": len(rows),
        "decision_counts": {key: counts[key] for key in ("SELECT", "HOLD", "EXCLUDE")},
        "selected_pageids": sorted(int(row["pageid"]) for row in rows if row["decision"] == "SELECT"),
        "held_pageids": sorted(int(row["pageid"]) for row in rows if row["decision"] == "HOLD"),
        "excluded_pageids": sorted(int(row["pageid"]) for row in rows if row["decision"] == "EXCLUDE"),
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
    if not manifest_path.is_file():
        raise ScreenError(f"missing E3 manifest: {manifest_path}")
    formal_root = (
        workspace
        / "datasets"
        / "desert_plants_wikimedia_staging_e0"
        / "review"
        / "human_decisions"
    ).resolve(strict=True)
    journal = formal_root / "decision_journal.jsonl"
    if not journal.is_file():
        raise ScreenError(f"missing formal decision journal: {journal}")
    formal_before = {
        "tree_sha256": tree_sha256(formal_root),
        "decision_journal_sha256": sha256_file(journal),
    }
    manifest_rows = load_jsonl(manifest_path)
    decisions = build_decision_rows(dataset_root, manifest_rows)
    stats = validate_decision_rows(decisions)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not replace:
        raise FileExistsError(f"output already exists: {output}; pass --replace to rebuild")
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.tmp-", dir=str(output.parent))).resolve()
    try:
        write_text(staging / "decisions.jsonl", jsonl_text(decisions))
        render_contact_sheet(decisions, dataset_root, staging / "contact_sheet.png")
        formal_after_render = {
            "tree_sha256": tree_sha256(formal_root),
            "decision_journal_sha256": sha256_file(journal),
        }
        if formal_after_render != formal_before:
            raise ScreenError("formal human decisions changed while building machine-only screen")
        decision_sha = sha256_file(staging / "decisions.jsonl")
        contact_sha = sha256_file(staging / "contact_sheet.png")
        receipt = {
            "schema_version": "rootscope.e3_machine_visual_screen_receipt.v1",
            "status": STATUS,
            "source_dataset": "E3",
            "source_dataset_name": DATASET_NAME,
            "source_manifest_path": f"datasets/{DATASET_NAME}/manifest.jsonl",
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_record_count": len(manifest_rows),
            "decisions_sha256": decision_sha,
            "contact_sheet_sha256": contact_sha,
            "implementation_path": "tools/dataset/build_e3_machine_visual_screen.py",
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
            "statistics": stats,
            "formal_human_decisions": {
                "path": "datasets/desert_plants_wikimedia_staging_e0/review/human_decisions",
                "before": formal_before,
                "after": formal_after_render,
                "unchanged": True,
            },
            "human_reviewed": False,
            "rights_approved": False,
            "training_eligible": False,
            "print_eligible": False,
            "data_locked": False,
            "formal_a1_dataset": False,
            "formal_split_assigned": False,
            "authority": false_authority(),
            "select_semantics": "PROVISIONAL_MACHINE_CANDIDATE_ONLY_NOT_TRAIN_ELIGIBLE",
            "explicit_non_claims": [
                "HUMAN_REVIEWED",
                "VISUAL_GROUND_TRUTH",
                "RIGHTS_APPROVED",
                "TRAIN_ELIGIBLE",
                "PRINT_ELIGIBLE",
                "DATA_LOCKED",
                "FORMAL_A1_DATASET",
                "FORMAL_SPLIT_ASSIGNED",
            ],
        }
        receipt["run_id"] = "sha256:" + sha256_bytes(
            (receipt["source_manifest_sha256"] + decision_sha + contact_sha).encode("utf-8")
        )
        write_json(staging / "receipt.json", receipt)
        sums = artifact_sums(staging)
        write_text(
            staging / "SHA256SUMS",
            "".join(f"{digest}  {relative}\n" for relative, digest in sums.items()),
        )
        formal_final = {
            "tree_sha256": tree_sha256(formal_root),
            "decision_journal_sha256": sha256_file(journal),
        }
        if formal_final != formal_before:
            raise ScreenError("formal human decisions changed before machine-screen commit")
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
