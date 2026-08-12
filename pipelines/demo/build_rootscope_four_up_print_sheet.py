"""Build one landscape A4 sheet containing four cut-out RootScope demo cards.

The printable page deliberately contains no labels or overlays inside the four
cards.  Position-to-class mapping, source hashes, attribution and truth
boundaries are written to the adjacent JSON manifest instead.  This prevents
printed class text from becoming an accidental visual shortcut.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from reportlab.lib.colors import Color, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


ADVENTUREX = Path(__file__).resolve().parents[2]
DATASET = ADVENTUREX / "datasets" / "rootscope_machine_curated_provisional_v3"
OUTPUT_DIR = ADVENTUREX / "output" / "pdf"
OUTPUT_PDF = OUTPUT_DIR / "RootScope_A4_four_up_field_cards_20260723.pdf"
OUTPUT_MANIFEST = (
    OUTPUT_DIR / "RootScope_A4_four_up_field_cards_20260723_manifest.json"
)
OUTPUT_SHA256 = OUTPUT_DIR / "RootScope_A4_four_up_field_cards_20260723.sha256"


CARDS: tuple[dict[str, Any], ...] = (
    {
        "position": "TOP_LEFT",
        "class_id": "grass_clump",
        "display_name_zh": "草丛",
        "role": "REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT",
        "relative_path": (
            "images/grass_clump/"
            "grass_clump_163498042_b1f6262895c3.jpg"
        ),
        "sha256": (
            "b1f6262895c31e8e507be31cebba09140e2a2582aa4f266ab05261fe50751d23"
        ),
        "pageid": 163498042,
        "artist": "Krzysztof Ziarnek, Kenraiz",
        "license": "CC BY-SA 4.0",
        "source_page": (
            "https://commons.wikimedia.org/wiki/"
            "File:Stipagrostis_plumosa_kz06.jpg"
        ),
    },
    {
        "position": "TOP_RIGHT",
        "class_id": "low_shrub",
        "display_name_zh": "灌木",
        "role": "REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT",
        "relative_path": (
            "images/low_shrub/"
            "low_shrub_68787114_810c7649ac72.jpg"
        ),
        "sha256": (
            "810c7649ac729105367b3213bfafc467a036f4054244c424613da6c027c73610"
        ),
        "pageid": 68787114,
        "artist": "USDA NRCS Montana",
        "license": "Public domain",
        "source_page": (
            "https://commons.wikimedia.org/wiki/"
            "File:Plants22_(27104657009).jpg"
        ),
    },
    {
        "position": "BOTTOM_LEFT",
        "class_id": "young_tree",
        "display_name_zh": "幼树",
        "role": "REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT",
        "relative_path": (
            "images/young_tree/"
            "young_tree_92774234_0d994e838a2d.jpg"
        ),
        "sha256": (
            "0d994e838a2d7787ab3edfd8646e317390c790d92588c7ef9109778b843b40eb"
        ),
        "pageid": 92774234,
        "artist": "Wogatha Kanyi",
        "license": "CC BY-SA 4.0",
        "source_page": (
            "https://commons.wikimedia.org/wiki/"
            "File:A_newly_planted_tree.jpg"
        ),
    },
    {
        "position": "BOTTOM_RIGHT",
        "class_id": "unknown",
        "display_name_zh": "无目标/裸沙",
        "role": "UNREGISTERED_NEGATIVE_NOT_ACCURACY_HOLDOUT",
        "relative_path": (
            "images/unknown/"
            "unknown_157364276_04e7f49a1e66.jpg"
        ),
        "sha256": (
            "04e7f49a1e66186bda7a9a1102985560eac0e3a1bffcec892e6dc522868c985b"
        ),
        "pageid": 157364276,
        "artist": "Mostafameraji",
        "license": "CC BY-SA 4.0",
        "source_page": (
            "https://commons.wikimedia.org/wiki/"
            "File:A_sand_dune_in_the_Maranjab_Desert,"
            "_located_in_the_middle_of_the_Kavir_National_Park_IRAN_23.jpg"
        ),
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_sources() -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for card in CARDS:
        path = DATASET / card["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = sha256_file(path)
        if actual_sha256 != card["sha256"]:
            raise RuntimeError(
                f"source SHA mismatch for {path}: "
                f"{actual_sha256} != {card['sha256']}"
            )
        with Image.open(path) as image:
            width_px, height_px = image.size
            mode = image.mode
        verified.append(
            {
                **card,
                "source_path_relative_to_adventurex": path.relative_to(
                    ADVENTUREX
                ).as_posix(),
                "source_bytes": path.stat().st_size,
                "source_width_px": width_px,
                "source_height_px": height_px,
                "source_mode": mode,
            }
        )
    return verified


def _draw_crop_marks(
    pdf: canvas.Canvas,
    page_width: float,
    page_height: float,
    margin: float,
    gutter: float,
) -> None:
    center_x = page_width / 2
    center_y = page_height / 2
    mark = 4 * mm
    stroke = Color(0.42, 0.47, 0.49)
    pdf.setStrokeColor(stroke)
    pdf.setLineWidth(0.35)
    pdf.setDash(2, 2)
    pdf.line(center_x, margin, center_x, page_height - margin)
    pdf.line(margin, center_y, page_width - margin, center_y)
    pdf.setDash()
    pdf.setLineWidth(0.55)
    for x in (center_x - gutter / 2, center_x + gutter / 2):
        pdf.line(x, margin - mark, x, margin)
        pdf.line(x, page_height - margin, x, page_height - margin + mark)
    for y in (center_y - gutter / 2, center_y + gutter / 2):
        pdf.line(margin - mark, y, margin, y)
        pdf.line(page_width - margin, y, page_width - margin + mark, y)


def _draw_card(
    pdf: canvas.Canvas,
    card: dict[str, Any],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> dict[str, float]:
    path = DATASET / card["relative_path"]
    pad = 3 * mm
    pdf.setFillColor(white)
    pdf.rect(x, y, width, height, stroke=0, fill=1)
    pdf.setStrokeColor(Color(0.72, 0.75, 0.76))
    pdf.setLineWidth(0.35)
    pdf.rect(x, y, width, height, stroke=1, fill=0)

    with Image.open(path) as image:
        image_width, image_height = image.size
    available_width = width - 2 * pad
    available_height = height - 2 * pad
    scale = min(
        available_width / image_width,
        available_height / image_height,
    )
    draw_width = image_width * scale
    draw_height = image_height * scale
    draw_x = x + (width - draw_width) / 2
    draw_y = y + (height - draw_height) / 2
    pdf.drawImage(
        ImageReader(str(path)),
        draw_x,
        draw_y,
        width=draw_width,
        height=draw_height,
        preserveAspectRatio=True,
        anchor="c",
        mask="auto",
    )
    pdf.setStrokeColor(Color(0.15, 0.18, 0.19))
    pdf.setLineWidth(0.5)
    pdf.rect(draw_x, draw_y, draw_width, draw_height, stroke=1, fill=0)
    return {
        "card_x_mm": round(x / mm, 3),
        "card_y_mm": round(y / mm, 3),
        "card_width_mm": round(width / mm, 3),
        "card_height_mm": round(height / mm, 3),
        "image_x_mm": round(draw_x / mm, 3),
        "image_y_mm": round(draw_y / mm, 3),
        "image_width_mm": round(draw_width / mm, 3),
        "image_height_mm": round(draw_height / mm, 3),
    }


def build() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PDF.exists() or OUTPUT_MANIFEST.exists() or OUTPUT_SHA256.exists():
        raise FileExistsError(
            "refusing to overwrite an existing four-up print artifact"
        )
    cards = _verify_sources()
    page_width, page_height = landscape(A4)
    margin = 7 * mm
    gutter = 6 * mm
    card_width = (page_width - 2 * margin - gutter) / 2
    card_height = (page_height - 2 * margin - gutter) / 2

    pdf = canvas.Canvas(
        str(OUTPUT_PDF),
        pagesize=(page_width, page_height),
        pageCompression=1,
        invariant=1,
    )
    pdf.setTitle("RootScope A4 four-up field demo cards")
    pdf.setSubject(
        "Four hash-bound event demo cards; no text overlay; not accuracy evidence"
    )
    pdf.setAuthor("RootScope AdventureX team")
    pdf.setCreator("RootScope deterministic four-up print builder")
    pdf.setFillColor(white)
    pdf.rect(0, 0, page_width, page_height, stroke=0, fill=1)

    positions = {
        "TOP_LEFT": (
            margin,
            margin + card_height + gutter,
        ),
        "TOP_RIGHT": (
            margin + card_width + gutter,
            margin + card_height + gutter,
        ),
        "BOTTOM_LEFT": (
            margin,
            margin,
        ),
        "BOTTOM_RIGHT": (
            margin + card_width + gutter,
            margin,
        ),
    }
    geometry: dict[str, dict[str, float]] = {}
    for card in cards:
        x, y = positions[card["position"]]
        geometry[card["position"]] = _draw_card(
            pdf,
            card,
            x=x,
            y=y,
            width=card_width,
            height=card_height,
        )
    _draw_crop_marks(pdf, page_width, page_height, margin, gutter)
    pdf.showPage()
    pdf.save()

    pdf_sha256 = sha256_file(OUTPUT_PDF)
    manifest = {
        "schema": "rootscope.event-demo-four-up-print-sheet.v1",
        "status": "BUILT_FOR_EVENT_DEMO_PRINT_AND_UVC_RECAPTURE_ONLY",
        "page": {
            "paper": "A4",
            "orientation": "landscape",
            "page_width_mm": round(page_width / mm, 3),
            "page_height_mm": round(page_height / mm, 3),
            "margin_mm": round(margin / mm, 3),
            "center_gutter_mm": round(gutter / mm, 3),
            "cards_per_page": 4,
            "text_or_class_overlay_inside_cards": False,
            "crop_guides_present": True,
        },
        "position_map": {
            "TOP_LEFT": "grass_clump / 草丛",
            "TOP_RIGHT": "low_shrub / 灌木",
            "BOTTOM_LEFT": "young_tree / 幼树",
            "BOTTOM_RIGHT": "unknown / 无目标裸沙",
        },
        "cards": [
            {
                **card,
                "geometry": geometry[card["position"]],
                "holdout_claimed": False,
                "accuracy_evidence": False,
                "camera_recapture_evidence": False,
            }
            for card in cards
        ],
        "print_settings": {
            "color": True,
            "duplex": False,
            "scale_percent": 100,
            "fit_to_page": False,
            "paper_orientation": "landscape",
            "recommended_quality": "high/photo",
            "cut_order": "vertical center guide, then horizontal center guide",
        },
        "truth_boundary": {
            "user_requested_event_demo_print": True,
            "formal_dataset_print_eligible": False,
            "rights_approval_inferred": False,
            "human_reviewed": False,
            "model_qualified": False,
            "generalization_claimed": False,
            "physical_or_irrigation_authority": False,
            "intended_use": "EVENT_DEMO_PRINT_AND_UVC_RECAPTURE_ONLY",
            "note": (
                "The three positive images are registered demo references and "
                "are not holdout samples. The unknown image must remain "
                "unregistered. Printing does not create camera, accuracy, "
                "dataset, model, or physical-closure qualification."
            ),
        },
        "pdf": {
            "path_relative_to_adventurex": OUTPUT_PDF.relative_to(
                ADVENTUREX
            ).as_posix(),
            "bytes": OUTPUT_PDF.stat().st_size,
            "sha256": pdf_sha256,
            "page_count": 1,
        },
    }
    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_SHA256.write_text(
        f"{pdf_sha256}  {OUTPUT_PDF.name}\n"
        f"{sha256_file(OUTPUT_MANIFEST)}  {OUTPUT_MANIFEST.name}\n",
        encoding="ascii",
    )
    return manifest


def main() -> int:
    manifest = build()
    print(json.dumps(manifest["pdf"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
