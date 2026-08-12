"""Build print-ready RootScope experimental demo-reference cards.

The first three images are hash-frozen experimental demo references and are
explicitly not holdout/generalization evidence.  Generating this PDF does not
approve rights, qualify the model, qualify a camera/X5 or change the source
dataset.  The plant-free sand-dune card remains an unregistered negative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


WORKSPACE = Path(__file__).resolve().parents[2]
DATASET = WORKSPACE / "datasets" / "rootscope_machine_curated_provisional_v3"
MANIFEST = DATASET / "manifest.jsonl"
OUTPUT_DIR = WORKSPACE / "output" / "pdf"
OUTPUT_PDF = OUTPUT_DIR / "RootScope_demo_reference_candidate_cards_A4.pdf"
OUTPUT_MANIFEST = OUTPUT_DIR / "RootScope_demo_reference_candidate_cards_manifest.json"
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "bf14c7423aad965b8af736c7d77cef1ba134d78dd1f905c03cc14cff1192f3fe"
)
REGISTRY = WORKSPACE / "rootscope" / "app" / "vision" / "known_card_template_registry.frozen.experimental.json"
REGISTRATION_RECEIPT = WORKSPACE / "evidence" / "rootscope_demo_template_registry_receipt_20260717.json"
EXPECTED_REGISTRY_SHA256 = "f5328b66c6b385dd081c1a27577d92b5c63fc18dc00d815f7d3d2561bb68e29f"
EXPECTED_REGISTRATION_RECEIPT_SHA256 = "fc6a4ba195625fb01f4f7301edd5c6225b97342788a7b0ee7b2b22252b77ea32"

CARD_SPECS = (
    {
        "pageid": 163498042,
        "display": "GRASS CLUMP / 草丛",
        "class_id": "grass_clump",
        "kind": "FROZEN_EXPERIMENTAL_DEMO_REFERENCE",
        "accent": "#1E9E72",
    },
    {
        "pageid": 68787114,
        "display": "LOW SHRUB / 灌木",
        "class_id": "low_shrub",
        "kind": "FROZEN_EXPERIMENTAL_DEMO_REFERENCE",
        "accent": "#C69032",
    },
    {
        "pageid": 92774234,
        "display": "YOUNG TREE / 幼树",
        "class_id": "young_tree",
        "kind": "FROZEN_EXPERIMENTAL_DEMO_REFERENCE",
        "accent": "#2D78D2",
    },
    {
        "pageid": 157364276,
        "display": "NO TARGET / 无目标负例",
        "class_id": "unknown",
        "kind": "NEGATIVE_CARD_MUST_NOT_REGISTER",
        "accent": "#D64545",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records() -> list[dict[str, Any]]:
    if sha256_file(MANIFEST) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise RuntimeError("frozen v3 manifest SHA-256 changed")
    if sha256_file(REGISTRY) != EXPECTED_REGISTRY_SHA256:
        raise RuntimeError("frozen experimental template registry SHA-256 changed")
    if sha256_file(REGISTRATION_RECEIPT) != EXPECTED_REGISTRATION_RECEIPT_SHA256:
        raise RuntimeError("demo template registration receipt SHA-256 changed")
    records: dict[int, dict[str, Any]] = {}
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        pageid = int(row["pageid"])
        if pageid in {item["pageid"] for item in CARD_SPECS}:
            records[pageid] = row
    result: list[dict[str, Any]] = []
    for spec in CARD_SPECS:
        row = records.get(spec["pageid"])
        if row is None:
            raise RuntimeError(f"candidate pageid missing: {spec['pageid']}")
        if row["class_id"] != spec["class_id"]:
            raise RuntimeError(f"candidate class changed: {spec['pageid']}")
        if row["experimental_split_suggestion"] != "EXPERIMENTAL_TRAIN_SUGGESTION":
            raise RuntimeError(f"candidate role changed: {spec['pageid']}")
        image_path = (DATASET / row["filename"]).resolve(strict=True)
        if image_path.parent.parent.parent != DATASET.resolve():
            raise RuntimeError("candidate image escaped frozen dataset")
        if sha256_file(image_path) != row["copied_image_sha256"]:
            raise RuntimeError(f"candidate image SHA mismatch: {spec['pageid']}")
        result.append({**spec, "row": row, "image_path": image_path})
    return result


def register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\msyh.ttc")
    bold = Path(r"C:\Windows\Fonts\msyhbd.ttc")
    if regular.is_file():
        pdfmetrics.registerFont(TTFont("RootScopeCN", str(regular), subfontIndex=0))
        normal_name = "RootScopeCN"
    else:
        normal_name = "Helvetica"
    if bold.is_file():
        pdfmetrics.registerFont(TTFont("RootScopeCN-Bold", str(bold), subfontIndex=0))
        bold_name = "RootScopeCN-Bold"
    else:
        bold_name = "Helvetica-Bold"
    return normal_name, bold_name


def draw_wrapped_paragraph(
    pdf: canvas.Canvas,
    text: str,
    *,
    x: float,
    y_top: float,
    width: float,
    height: float,
    font_name: str,
    font_size: float,
    color: Color,
) -> None:
    style = ParagraphStyle(
        "card-footer",
        fontName=font_name,
        fontSize=font_size,
        leading=font_size * 1.28,
        textColor=color,
        alignment=TA_LEFT,
        splitLongWords=True,
        spaceAfter=0,
        spaceBefore=0,
    )
    paragraph = Paragraph(text, style)
    _width, used_height = paragraph.wrap(width, height)
    paragraph.drawOn(pdf, x, y_top - used_height)


def draw_fit_image(
    pdf: canvas.Canvas,
    image_path: Path,
    *,
    box_x: float,
    box_y: float,
    box_width: float,
    box_height: float,
) -> tuple[float, float, float, float]:
    with Image.open(image_path) as image:
        width, height = image.size
    scale = min(box_width / width, box_height / height)
    draw_width = width * scale
    draw_height = height * scale
    x = box_x + (box_width - draw_width) / 2
    y = box_y + (box_height - draw_height) / 2
    pdf.drawImage(
        ImageReader(str(image_path)),
        x,
        y,
        width=draw_width,
        height=draw_height,
        preserveAspectRatio=True,
        mask="auto",
    )
    return x, y, draw_width, draw_height


def page_size_for(image_path: Path) -> tuple[float, float]:
    with Image.open(image_path) as image:
        width, height = image.size
    return landscape(A4) if width >= height else portrait(A4)


def draw_card_page(
    pdf: canvas.Canvas,
    card: dict[str, Any],
    *,
    normal_font: str,
    bold_font: str,
    index: int,
) -> None:
    width, height = page_size_for(card["image_path"])
    pdf.setPageSize((width, height))
    accent = HexColor(card["accent"])
    pdf.setFillColor(HexColor("#0C1726"))
    pdf.rect(0, height - 58, width, 58, stroke=0, fill=1)
    pdf.setFillColor(accent)
    pdf.rect(0, height - 63, width, 5, stroke=0, fill=1)
    pdf.setFont(bold_font, 19)
    pdf.setFillColor(white)
    pdf.drawString(28, height - 36, f"RootScope  |  {card['display']}")
    pdf.setFont(normal_font, 8.5)
    badge = (
        "FROZEN EXPERIMENTAL DEMO REFERENCE"
        if card["kind"] == "FROZEN_EXPERIMENTAL_DEMO_REFERENCE"
        else "NEGATIVE CARD - MUST NOT REGISTER"
    )
    badge_width = pdf.stringWidth(badge, normal_font, 8.5) + 18
    pdf.setFillColor(accent)
    pdf.roundRect(width - badge_width - 24, height - 43, badge_width, 22, 6, stroke=0, fill=1)
    pdf.setFillColor(white)
    pdf.drawCentredString(width - badge_width / 2 - 24, height - 35.5, badge)

    footer_height = 84
    image_box_x = 24
    image_box_y = footer_height + 14
    image_box_width = width - 48
    image_box_height = height - 63 - image_box_y - 14
    pdf.setFillColor(HexColor("#F4F6F8"))
    pdf.roundRect(
        image_box_x,
        image_box_y,
        image_box_width,
        image_box_height,
        8,
        stroke=0,
        fill=1,
    )
    x, y, draw_width, draw_height = draw_fit_image(
        pdf,
        card["image_path"],
        box_x=image_box_x + 8,
        box_y=image_box_y + 8,
        box_width=image_box_width - 16,
        box_height=image_box_height - 16,
    )
    pdf.setStrokeColor(accent)
    pdf.setLineWidth(2.2)
    pdf.rect(x, y, draw_width, draw_height, stroke=1, fill=0)

    row = card["row"]
    license_url = row.get("license_canonical_url") or row["source_page"]
    attribution = (
        f"Card {index:02d} | PageID {row['pageid']} | {row['title']}<br/>"
        f"Creator: {row['artist']} | License: {row['license']} | "
        f"License/source: {license_url}<br/>"
        f"Source: {row['source_page']} | Source SHA-256: {row['copied_image_sha256']}"
    )
    draw_wrapped_paragraph(
        pdf,
        attribution,
        x=28,
        y_top=footer_height - 3,
        width=width - 56,
        height=footer_height - 8,
        font_name=normal_font,
        font_size=6.8,
        color=HexColor("#263548"),
    )
    pdf.showPage()


def draw_instruction_page(
    pdf: canvas.Canvas, *, normal_font: str, bold_font: str
) -> None:
    width, height = portrait(A4)
    pdf.setPageSize((width, height))
    pdf.setFillColor(HexColor("#0C1726"))
    pdf.rect(0, 0, width, height, stroke=0, fill=1)
    pdf.setFillColor(HexColor("#2DD4BF"))
    pdf.rect(0, height - 10, width, 10, stroke=0, fill=1)
    pdf.setFillColor(white)
    pdf.setFont(bold_font, 25)
    pdf.drawString(42, height - 70, "RootScope 打印与现场复拍规则")
    pdf.setFont(normal_font, 11)
    pdf.setFillColor(HexColor("#A9B8C9"))
    pdf.drawString(42, height - 94, "FROZEN EXPERIMENTAL REFERENCES - ZERO AUTHORITY - NO HOLDOUT CLAIM")

    rules = [
        "1. 以 100% 比例打印，优先使用哑光纸；不要自动缩放或裁掉照片边缘。",
        "2. 前三张已按原图 SHA 冻结为实验演示模板；它们不再属于留出集，不得作为泛化证据。",
        "3. 到场后必须用最终 UVC、灯光、距离和打印件复拍，冻结相机收据与阈值；若裁剪、改图或换打印源，必须重建注册表和哈希。",
        "4. 沙丘 NO TARGET 卡是拒答负例，必须保持未登记，用来验证 NO_REGISTERED_TEMPLATE_GEOMETRIC_PASS。",
        "5. 语义模型只给 DEMO_HYPOTHESIS；几何匹配只证明已知卡实例。二者一致仍不拥有泵、串口、状态机或灌溉权限。",
        "6. 正式演示前另采一组未参与调参的角度、距离和光照复拍，保留全部成功与失败帧。",
        "7. 当前标签仍是机器整理结论；人工审核、版权批准、打印/相机资格、模型资格和 X5 真机验证均为 false。",
    ]
    style = ParagraphStyle(
        "instructions",
        fontName=normal_font,
        fontSize=12,
        leading=21,
        textColor=white,
        spaceAfter=14,
    )
    y = height - 140
    for rule in rules:
        paragraph = Paragraph(rule, style)
        _w, used = paragraph.wrap(width - 84, 100)
        paragraph.drawOn(pdf, 42, y - used)
        y -= used + 18

    pdf.setFillColor(HexColor("#13263B"))
    pdf.roundRect(42, 58, width - 84, 92, 12, stroke=0, fill=1)
    footer_style = ParagraphStyle(
        "footer-note",
        fontName=normal_font,
        fontSize=9.5,
        leading=15,
        textColor=HexColor("#D6E2EF"),
    )
    note = Paragraph(
        "前三张模板登记由独立 receipt 绑定，本 PDF 本身不会改变冻结数据集、批准版权、证明相机复拍、启用 BPU，或授予任何物理执行权限。",
        footer_style,
    )
    _w, used = note.wrap(width - 116, 68)
    note.drawOn(pdf, 58, 104 - used / 2)
    pdf.showPage()


def build(output_pdf: Path, output_manifest: Path) -> dict[str, Any]:
    cards = load_records()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    normal_font, bold_font = register_fonts()
    pdf = canvas.Canvas(
        str(output_pdf), pagesize=A4, pageCompression=1, invariant=1
    )
    pdf.setTitle("RootScope experimental demo reference cards")
    pdf.setAuthor("RootScope AdventureX team")
    pdf.setSubject("Frozen experimental demo references; not holdout, rights-approved or qualified")
    for index, card in enumerate(cards, start=1):
        draw_card_page(
            pdf,
            card,
            normal_font=normal_font,
            bold_font=bold_font,
            index=index,
        )
    draw_instruction_page(pdf, normal_font=normal_font, bold_font=bold_font)
    pdf.save()

    payload = {
        "schema": "rootscope.demo-reference-cards.v2",
        "status": "POSITIVE_DEMO_REFERENCES_REGISTERED_EXPERIMENTAL_NOT_RIGHTS_OR_CAMERA_APPROVED",
        "pdf": {
            "path": output_pdf.relative_to(WORKSPACE).as_posix(),
            "bytes": output_pdf.stat().st_size,
            "sha256": sha256_file(output_pdf),
            "page_count": 5,
        },
        "frozen_dataset": {
            "path": DATASET.relative_to(WORKSPACE).as_posix(),
            "manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
            "mutated": False,
        },
        "frozen_experimental_registry": {
            "path": REGISTRY.relative_to(WORKSPACE).as_posix(),
            "sha256": EXPECTED_REGISTRY_SHA256,
            "registration_receipt_path": REGISTRATION_RECEIPT.relative_to(WORKSPACE).as_posix(),
            "registration_receipt_sha256": EXPECTED_REGISTRATION_RECEIPT_SHA256,
            "role": "DEMO_REFERENCE_NOT_HOLDOUT_ONCE_REGISTERED",
            "holdout_evidence": False,
            "generalization_evidence": False,
        },
        "cards": [
            {
                "pageid": card["pageid"],
                "class_id": card["class_id"],
                "display": card["display"],
                "kind": card["kind"],
                "source_role": card["row"]["experimental_split_suggestion"],
                "source_image_path": card["row"]["filename"],
                "source_image_sha256": card["row"]["copied_image_sha256"],
                "template_registered": card["class_id"] != "unknown",
                "holdout_evidence": False,
                "print_recapture_evidence": False,
            }
            for card in cards
        ],
        "registration_state": {
            "positive_templates_registered": True,
            "positive_template_count": 3,
            "unknown_negative_registered": False,
        },
        "authority": {
            "human_reviewed": False,
            "rights_approved": False,
            "print_eligible": False,
            "template_registry_write_authority": False,
            "model_qualified": False,
            "x5_validated": False,
            "execution_authority": False,
            "physical_authority": False,
        },
    }
    output_manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-pdf", type=Path, default=OUTPUT_PDF)
    parser.add_argument("--output-manifest", type=Path, default=OUTPUT_MANIFEST)
    args = parser.parse_args()
    result = build(args.output_pdf.resolve(), args.output_manifest.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
