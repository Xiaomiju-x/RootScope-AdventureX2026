#!/usr/bin/env python3
"""Build two A4 print sheets with one image per half page.

The output is for RootScope event-demo camera recapture only.  It does not
register training data, change holdout membership, or qualify a model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


ADVENTUREX = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = (
    ADVENTUREX
    / "output"
    / "pdf"
    / "RootScope_A4_four_up_field_cards_20260723_manifest.json"
)
OUTPUT_ROOT = ADVENTUREX / "output" / "pdf"
TEMP_ROOT = ADVENTUREX / "tmp" / "pdfs" / "rootscope_half_a4_20260724"
OUTPUTS = (
    (
        "RootScope_A4_half_page_cards_grass_shrub_20260724.pdf",
        ("grass_clump", "low_shrub"),
    ),
    (
        "RootScope_A4_half_page_cards_tree_sand_20260724.pdf",
        ("young_tree", "unknown"),
    ),
)
FOCAL_POINTS = {
    "grass_clump": (0.50, 0.50),
    "low_shrub": (0.50, 0.52),
    # Keep the young-tree crown and trunk while removing the uninformative
    # lower foreground from the portrait source.
    "young_tree": (0.50, 0.28),
    "unknown": (0.50, 0.50),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cover_crop_box(
    width: int,
    height: int,
    target_ratio: float,
    focal_x: float,
    focal_y: float,
) -> tuple[int, int, int, int]:
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_width = int(round(height * target_ratio))
        left = int(round(focal_x * width - crop_width / 2))
        left = max(0, min(width - crop_width, left))
        return left, 0, left + crop_width, height
    crop_height = int(round(width / target_ratio))
    top = int(round(focal_y * height - crop_height / 2))
    top = max(0, min(height - crop_height, top))
    return 0, top, width, top + crop_height


def load_cards() -> dict[str, dict[str, Any]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    cards: dict[str, dict[str, Any]] = {}
    for card in manifest["cards"]:
        class_id = card["class_id"]
        source = ADVENTUREX / card["source_path_relative_to_adventurex"]
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"invalid source image: {source}")
        actual_sha = sha256_file(source)
        if actual_sha != card["sha256"]:
            raise RuntimeError(
                f"source hash mismatch for {class_id}: {actual_sha} != {card['sha256']}"
            )
        cards[class_id] = {**card, "source_path": source}
    expected = {"grass_clump", "low_shrub", "young_tree", "unknown"}
    if set(cards) != expected:
        raise RuntimeError(f"unexpected source-card set: {sorted(cards)}")
    return cards


def build_crop(
    card: dict[str, Any],
    *,
    page_width_pt: float,
    half_height_pt: float,
) -> tuple[Path, dict[str, Any]]:
    class_id = card["class_id"]
    source = Path(card["source_path"])
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        target_ratio = page_width_pt / half_height_pt
        focal_x, focal_y = FOCAL_POINTS[class_id]
        box = cover_crop_box(
            image.width,
            image.height,
            target_ratio,
            focal_x,
            focal_y,
        )
        cropped = image.crop(box)
        output = TEMP_ROOT / f"{class_id}_half_a4_crop.jpg"
        cropped.save(
            output,
            format="JPEG",
            quality=97,
            subsampling=0,
            optimize=True,
            progressive=False,
        )
    width_in = (page_width_pt / 72.0)
    height_in = (half_height_pt / 72.0)
    return output, {
        "source_width_px": card["source_width_px"],
        "source_height_px": card["source_height_px"],
        "crop_box_px": list(box),
        "crop_width_px": box[2] - box[0],
        "crop_height_px": box[3] - box[1],
        "focal_point_normalized": [focal_x, focal_y],
        "printed_width_mm": round(page_width_pt / mm, 3),
        "printed_height_mm": round(half_height_pt / mm, 3),
        "effective_dpi_x": round((box[2] - box[0]) / width_in, 2),
        "effective_dpi_y": round((box[3] - box[1]) / height_in, 2),
        "crop_sha256": sha256_file(output),
    }


def build_pdf(
    output_path: Path,
    class_ids: tuple[str, str],
    cards: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    page_width, page_height = A4
    half_height = page_height / 2
    pdf = canvas.Canvas(
        str(output_path),
        pagesize=A4,
        pageCompression=1,
        invariant=1,
    )
    pdf.setTitle(output_path.stem)
    pdf.setAuthor("RootScope AdventureX")
    pdf.setSubject("Half-A4 event-demo camera recapture cards")
    placements: list[dict[str, Any]] = []

    # First item is the top half; second is the bottom half.
    for position, class_id in zip(("TOP", "BOTTOM"), class_ids, strict=True):
        card = cards[class_id]
        crop_path, crop = build_crop(
            card,
            page_width_pt=page_width,
            half_height_pt=half_height,
        )
        y = half_height if position == "TOP" else 0
        pdf.drawImage(
            str(crop_path),
            0,
            y,
            width=page_width,
            height=half_height,
            preserveAspectRatio=False,
            anchor="c",
            mask="auto",
        )
        placements.append(
            {
                "position": position,
                "class_id": class_id,
                "display_name_zh": card["display_name_zh"],
                "role": card["role"],
                "source_path_relative_to_adventurex": card[
                    "source_path_relative_to_adventurex"
                ],
                "source_sha256": card["sha256"],
                "source_page": card["source_page"],
                "artist": card["artist"],
                "license": card["license"],
                **crop,
            }
        )

    # A single high-contrast center guide is the only overlay.  It lies exactly
    # on the A5 cut boundary and disappears when the two cards are separated.
    pdf.saveState()
    pdf.setStrokeColorRGB(1, 1, 1)
    pdf.setLineWidth(2.2)
    pdf.line(0, half_height, page_width, half_height)
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf.setLineWidth(0.55)
    pdf.setDash(5, 4)
    pdf.line(0, half_height, page_width, half_height)
    pdf.restoreState()
    pdf.showPage()
    pdf.save()

    return {
        "pdf_path_relative_to_adventurex": output_path.relative_to(
            ADVENTUREX
        ).as_posix(),
        "pdf_sha256": sha256_file(output_path),
        "pdf_bytes": output_path.stat().st_size,
        "page_count": 1,
        "paper": "A4",
        "orientation": "portrait",
        "page_width_mm": 210.0,
        "page_height_mm": 297.0,
        "cards_per_page": 2,
        "card_width_mm": 210.0,
        "card_height_mm": 148.5,
        "center_cut_y_mm": 148.5,
        "placements": placements,
    }


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    cards = load_cards()
    built = []
    for filename, class_ids in OUTPUTS:
        output = OUTPUT_ROOT / filename
        if output.exists():
            raise RuntimeError(f"refusing to overwrite existing PDF: {output}")
        built.append(build_pdf(output, class_ids, cards))

    manifest = {
        "schema": "rootscope.event-demo-half-a4-print-sheets.v1",
        "status": "BUILT_FOR_EVENT_DEMO_PRINT_AND_UVC_RECAPTURE_ONLY",
        "source_manifest_relative_to_adventurex": SOURCE_MANIFEST.relative_to(
            ADVENTUREX
        ).as_posix(),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "outputs": built,
        "print_settings": {
            "paper": "A4",
            "orientation": "portrait",
            "scale_percent": 100,
            "fit_to_page": False,
            "color": True,
            "quality": "high/photo",
            "duplex": False,
            "cut": "one horizontal cut on the dashed center line",
        },
        "truth_boundary": {
            "intended_use": "EVENT_DEMO_PRINT_AND_UVC_RECAPTURE_ONLY",
            "auto_training": False,
            "accuracy_evidence": False,
            "holdout_claimed": False,
            "physical_or_irrigation_authority": False,
        },
    }
    manifest_path = (
        OUTPUT_ROOT / "RootScope_A4_half_page_cards_20260724_manifest.json"
    )
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite existing manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "outputs": [
                    {
                        "path": item["pdf_path_relative_to_adventurex"],
                        "sha256": item["pdf_sha256"],
                        "bytes": item["pdf_bytes"],
                    }
                    for item in built
                ],
                "manifest": manifest_path.relative_to(ADVENTUREX).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
