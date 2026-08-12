#!/usr/bin/env python3
"""Render deterministic, AI-only contact sheets for SigLIP2 ensemble results.

This utility is deliberately isolated from the formal review workflow.  It reads
the immutable ensemble JSONL and source images, then writes only to an AI-named
sibling directory under ``review``.  It never writes a dataset manifest or any
file below ``human_decisions``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


SCHEMA_VERSION = "rootscope.ai_ensemble_visual_index.v1"
RECEIPT_SCHEMA_VERSION = "rootscope.ai_ensemble_visual_receipt.v1"
STATUS = "AI_ONLY_VISUALIZATION_NOT_HUMAN_REVIEWED_NOT_DATA_LOCKED"
AUTHORITY = {
    "data_locked": False,
    "dataset_manifest_write": False,
    "human_review": False,
    "print_eligibility": False,
    "split_assignment": False,
    "training_eligibility": False,
}
AI_OUTPUT_NAME = re.compile(r"^ai_[a-z0-9_]+_visuals$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def finite_probability(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric for asset {row.get('asset')!r}")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")
    return result


def load_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path.name}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"result row {line_number} is not an object")
            for field in (
                "asset",
                "pageid",
                "image_path",
                "decision",
                "acquisition_class_hint",
                "top1_class",
                "admissibility_probability",
                "top1_top2_margin",
            ):
                if field not in row:
                    raise ValueError(f"result row {line_number} is missing {field}")
            asset = row["asset"]
            if not isinstance(asset, str) or not asset:
                raise ValueError(f"result row {line_number} has an invalid asset")
            if asset in seen_assets:
                raise ValueError(f"duplicate asset in results: {asset}")
            seen_assets.add(asset)
            if isinstance(row["pageid"], bool) or not isinstance(row["pageid"], int):
                raise ValueError(f"pageid must be an integer for asset {asset}")
            for field in ("image_path", "decision", "acquisition_class_hint", "top1_class"):
                if not isinstance(row[field], str) or not row[field]:
                    raise ValueError(f"{field} must be a non-empty string for asset {asset}")
            finite_probability(row, "admissibility_probability")
            finite_probability(row, "top1_top2_margin")
            rows.append(row)
    if not rows:
        raise ValueError("ensemble results are empty")
    return rows


def ensure_source_image(dataset_root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe image_path: {relative_value!r}")
    target = (dataset_root / relative).resolve(strict=True)
    try:
        target.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"image_path escapes dataset root: {relative_value!r}") from error
    if not target.is_file():
        raise ValueError(f"image_path is not a file: {relative_value!r}")
    return target


def resolve_layout(results: Path, output: Path | None) -> tuple[Path, Path, Path, Path]:
    results = results.resolve(strict=True)
    source_dir = results.parent
    review_dir = source_dir.parent.resolve(strict=True)
    if review_dir.name != "review":
        raise ValueError("results must be inside a direct child of a review directory")
    dataset_root = review_dir.parent.resolve(strict=True)
    output_dir = (output or (review_dir / "ai_ensemble_v1_visuals")).resolve(strict=False)
    if output_dir.parent != review_dir:
        raise ValueError("output must be a direct sibling of the ensemble directory under review")
    if not AI_OUTPUT_NAME.fullmatch(output_dir.name):
        raise ValueError("output directory must match ai_[a-z0-9_]+_visuals")
    if output_dir == source_dir or "human_decisions" in output_dir.parts:
        raise ValueError("output may not overlap model results or human_decisions")
    return results, review_dir, dataset_root, output_dir


def resolve_font(font_arg: Path | None, size: int) -> tuple[ImageFont.ImageFont, dict[str, Any]]:
    candidates: list[Path] = []
    if font_arg is not None:
        candidates.append(font_arg)
    candidates.extend(
        [
            Path(r"C:\Windows\Fonts\consola.ttf"),
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve(strict=True)
            return ImageFont.truetype(str(resolved), size=size), {
                "font_file": resolved.name,
                "font_sha256": sha256_file(resolved),
                "font_size": size,
            }
    if font_arg is not None:
        raise FileNotFoundError(font_arg)
    return ImageFont.load_default(), {
        "font_file": "PIL_BUILTIN_DEFAULT",
        "font_sha256": None,
        "font_size": None,
    }


def clean_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return token or "none"


def predicted_class(row: dict[str, Any]) -> str:
    value = row.get("suggested_class") or row.get("top1_class")
    return str(value) if value else "none"


def asset_label(asset: str) -> tuple[str, str]:
    if "@sha256:" in asset:
        identity, digest = asset.split("@sha256:", 1)
        return identity, digest[:12]
    return asset[:30], "-"


def label_lines(row: dict[str, Any]) -> tuple[str, str, str, str]:
    identity, digest = asset_label(str(row["asset"]))
    return (
        f"asset={identity} sha={digest}",
        f"pageid={row['pageid']} hint={row['acquisition_class_hint']}",
        f"pred={predicted_class(row)} decision={row['decision']}",
        "adm={:.4f} margin={:.4f}".format(
            finite_probability(row, "admissibility_probability"),
            finite_probability(row, "top1_top2_margin"),
        ),
    )


def record_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["asset"]), int(row["pageid"])


def group_order(row: dict[str, Any]) -> tuple[float, float, str, int]:
    return (
        -finite_probability(row, "admissibility_probability"),
        -finite_probability(row, "top1_top2_margin"),
        str(row["asset"]),
        int(row["pageid"]),
    )


def chunked(values: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def render_sheet(
    *,
    rows: Sequence[dict[str, Any]],
    dataset_root: Path,
    target: Path,
    title: str,
    columns: int,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    cell_width = 320
    image_height = 190
    label_height = 92
    header_height = 58
    row_count = (len(rows) + columns - 1) // columns
    width = columns * cell_width
    height = header_height + row_count * (image_height + label_height)
    sheet = Image.new("RGB", (width, height), "#f4f4f4")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, width - 1, header_height - 1), fill="#182433")
    draw.text((12, 8), title, fill="white", font=font)
    draw.text(
        (12, 31),
        "AI-ONLY | NOT HUMAN REVIEWED | NOT DATA LOCKED",
        fill="#ffcf66",
        font=font,
    )
    decision_colors = {
        "AUTO_TARGET": "#2e8b57",
        "AUTO_UNKNOWN": "#b04a4a",
        "HOLD": "#c48a13",
    }
    for index, row in enumerate(rows):
        grid_row, column = divmod(index, columns)
        x = column * cell_width
        y = header_height + grid_row * (image_height + label_height)
        source = ensure_source_image(dataset_root, str(row["image_path"]))
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            tile = ImageOps.contain(
                image,
                (cell_width - 8, image_height - 8),
                method=Image.Resampling.LANCZOS,
            )
        tile_backdrop = Image.new("RGB", (cell_width, image_height), "#dadada")
        paste_x = (cell_width - tile.width) // 2
        paste_y = (image_height - tile.height) // 2
        tile_backdrop.paste(tile, (paste_x, paste_y))
        sheet.paste(tile_backdrop, (x, y))
        draw = ImageDraw.Draw(sheet)
        label_top = y + image_height
        draw.rectangle(
            (x, label_top, x + cell_width - 1, label_top + label_height - 1),
            fill="white",
            outline="#a7a7a7",
        )
        color = decision_colors.get(str(row["decision"]), "#607080")
        draw.rectangle((x, label_top, x + 4, label_top + label_height - 1), fill=color)
        draw.multiline_text(
            (x + 8, label_top + 5),
            "\n".join(label_lines(row)),
            fill="#111111",
            font=font,
            spacing=3,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # PNG parameters are explicit so re-running with the same Pillow build and
    # source inputs produces byte-identical sheets.
    sheet.save(target, format="PNG", optimize=False, compress_level=9)
    return width, height


def select_extreme(
    rows: Sequence[dict[str, Any]], field: str, count: int, *, reverse: bool
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -finite_probability(row, field) if reverse else finite_probability(row, field),
            *record_key(row),
        ),
    )[:count]


def select_boundary(
    rows: Sequence[dict[str, Any]], field: str, threshold: float, count: int
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            abs(finite_probability(row, field) - threshold),
            finite_probability(row, field),
            *record_key(row),
        ),
    )[:count]


def validate_threshold(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"stats threshold {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"stats threshold {name} must be within [0, 1]")
    return result


def build_visuals(
    *,
    results_path: Path,
    stats_path: Path,
    output_dir: Path | None,
    columns: int,
    page_size: int,
    sample_size: int,
    font_path: Path | None,
    replace: bool,
) -> Path:
    results_path, _review_dir, dataset_root, output_dir = resolve_layout(results_path, output_dir)
    stats_path = stats_path.resolve(strict=True)
    if stats_path.parent != results_path.parent:
        raise ValueError("stats must be in the same immutable ensemble directory as results")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(stats, dict) or not isinstance(stats.get("thresholds"), dict):
        raise ValueError("stats does not contain a thresholds object")
    rows = load_results(results_path)
    if stats.get("candidate_count") != len(rows):
        raise ValueError("stats candidate_count does not match results")
    thresholds = stats["thresholds"]
    boundary_values = {
        "admissibility_auto_unknown": validate_threshold(
            thresholds.get("auto_unknown_max_admissible"), "auto_unknown_max_admissible"
        ),
        "admissibility_auto_target": validate_threshold(
            thresholds.get("auto_target_min_admissible"), "auto_target_min_admissible"
        ),
        "margin_auto_target": validate_threshold(
            thresholds.get("auto_target_min_margin"), "auto_target_min_margin"
        ),
    }
    for row in rows:
        ensure_source_image(dataset_root, str(row["image_path"]))
    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError("existing output must be a real directory")
        if not replace:
            raise FileExistsError(f"output exists; pass --replace to rebuild: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=False)
    font, font_identity = resolve_font(font_path, 13)

    sheet_records: list[dict[str, Any]] = []

    def add_sheet(
        relative: Path,
        selected: Sequence[dict[str, Any]],
        title: str,
        kind: str,
        metadata: dict[str, Any],
    ) -> None:
        target = output_dir / relative
        width, height = render_sheet(
            rows=selected,
            dataset_root=dataset_root,
            target=target,
            title=title,
            columns=columns,
            font=font,
        )
        sheet_records.append(
            {
                "file": relative.as_posix(),
                "kind": kind,
                "title": title,
                "record_count": len(selected),
                "assets": [str(row["asset"]) for row in selected],
                "pixel_width": width,
                "pixel_height": height,
                "sha256": sha256_file(target),
                **metadata,
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["decision"]), predicted_class(row))].append(row)
    group_counts: dict[str, int] = {}
    for (decision, class_id), group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=group_order)
        key = f"{decision}/{class_id}"
        group_counts[key] = len(ordered)
        pages = list(chunked(ordered, page_size))
        for page_index, page_rows in enumerate(pages, start=1):
            relative = Path("by_decision_class") / (
                f"decision_{clean_token(decision)}__pred_{clean_token(class_id)}"
                f"__p{page_index:03d}.png"
            )
            add_sheet(
                relative,
                page_rows,
                (
                    f"decision={decision} | pred={class_id} | "
                    f"page {page_index}/{len(pages)} | group n={len(ordered)}"
                ),
                "DECISION_CLASS_PAGE",
                {
                    "decision": decision,
                    "predicted_class": class_id,
                    "page_index": page_index,
                    "page_count": len(pages),
                },
            )

    score_specs: list[tuple[str, str, list[dict[str, Any]], dict[str, Any]]] = [
        (
            "admissibility_high",
            "admissibility HIGH",
            select_extreme(rows, "admissibility_probability", sample_size, reverse=True),
            {"score_field": "admissibility_probability", "selection": "HIGH"},
        ),
        (
            "admissibility_low",
            "admissibility LOW",
            select_extreme(rows, "admissibility_probability", sample_size, reverse=False),
            {"score_field": "admissibility_probability", "selection": "LOW"},
        ),
        (
            "admissibility_boundary_auto_unknown",
            f"admissibility BOUNDARY @ {boundary_values['admissibility_auto_unknown']:.4f}",
            select_boundary(
                rows,
                "admissibility_probability",
                boundary_values["admissibility_auto_unknown"],
                sample_size,
            ),
            {
                "score_field": "admissibility_probability",
                "selection": "BOUNDARY",
                "threshold": boundary_values["admissibility_auto_unknown"],
            },
        ),
        (
            "admissibility_boundary_auto_target",
            f"admissibility BOUNDARY @ {boundary_values['admissibility_auto_target']:.4f}",
            select_boundary(
                rows,
                "admissibility_probability",
                boundary_values["admissibility_auto_target"],
                sample_size,
            ),
            {
                "score_field": "admissibility_probability",
                "selection": "BOUNDARY",
                "threshold": boundary_values["admissibility_auto_target"],
            },
        ),
        (
            "margin_high",
            "top1-top2 margin HIGH",
            select_extreme(rows, "top1_top2_margin", sample_size, reverse=True),
            {"score_field": "top1_top2_margin", "selection": "HIGH"},
        ),
        (
            "margin_low",
            "top1-top2 margin LOW",
            select_extreme(rows, "top1_top2_margin", sample_size, reverse=False),
            {"score_field": "top1_top2_margin", "selection": "LOW"},
        ),
        (
            "margin_boundary_auto_target",
            f"top1-top2 margin BOUNDARY @ {boundary_values['margin_auto_target']:.4f}",
            select_boundary(
                rows,
                "top1_top2_margin",
                boundary_values["margin_auto_target"],
                sample_size,
            ),
            {
                "score_field": "top1_top2_margin",
                "selection": "BOUNDARY",
                "threshold": boundary_values["margin_auto_target"],
            },
        ),
    ]
    for name, title, selected, metadata in score_specs:
        pages = list(chunked(selected, page_size))
        for page_index, page_rows in enumerate(pages, start=1):
            suffix = f"__p{page_index:03d}" if len(pages) > 1 else ""
            add_sheet(
                Path("score_samples") / f"{name}{suffix}.png",
                page_rows,
                f"{title} | sample n={len(selected)}",
                "SCORE_SAMPLE",
                {**metadata, "page_index": page_index, "page_count": len(pages)},
            )

    sheet_records.sort(key=lambda item: item["file"])
    index = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "authority": AUTHORITY,
        "explicit_non_claims": [
            "HUMAN_REVIEWED",
            "VISUAL_LABEL_APPROVED",
            "RIGHTS_APPROVED",
            "DATA_LOCKED",
            "TRAIN_READY",
            "PRINT_ELIGIBLE",
        ],
        "inputs": {
            "results_file": results_path.name,
            "results_sha256": sha256_file(results_path),
            "stats_file": stats_path.name,
            "stats_sha256": sha256_file(stats_path),
            "candidate_count": len(rows),
        },
        "rendering": {
            "columns": columns,
            "page_size": page_size,
            "sample_size": sample_size,
            "image_fit": "CONTAIN_NO_CROP",
            "format": "PNG",
            "compression_level": 9,
            **font_identity,
        },
        "thresholds": boundary_values,
        "decision_class_counts": group_counts,
        "sheet_count": len(sheet_records),
        "sheets": sheet_records,
    }
    index_path = output_dir / "visual_index.json"
    write_json(index_path, index)
    root_lines = [f"{sheet['file']}\0{sheet['sha256']}\n" for sheet in sheet_records]
    sheet_root = hashlib.sha256("".join(root_lines).encode("utf-8")).hexdigest()
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": STATUS,
        "authority": AUTHORITY,
        "candidate_count": len(rows),
        "sheet_count": len(sheet_records),
        "visual_index_sha256": sha256_file(index_path),
        "sheet_set_sha256": sheet_root,
    }
    write_json(output_dir / "visual_receipt.json", receipt)
    return output_dir


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_dataset = script.parents[2] / "datasets" / "desert_plants_wikimedia_staging_e0"
    default_ensemble = default_dataset / "review" / "ai_ensemble_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=default_ensemble / "ai_siglip2_ensemble_results.jsonl",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=default_ensemble / "ai_siglip2_ensemble_stats.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--page-size", type=int, default=40)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    for name in ("columns", "page_size", "sample_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> int:
    args = parse_args()
    output = build_visuals(
        results_path=args.results,
        stats_path=args.stats,
        output_dir=args.output_dir,
        columns=args.columns,
        page_size=args.page_size,
        sample_size=args.sample_size,
        font_path=args.font,
        replace=args.replace,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
