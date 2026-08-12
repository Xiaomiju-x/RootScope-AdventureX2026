#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import qrcode
from PIL import Image, ImageDraw
from qrcode.constants import ERROR_CORRECT_H
from qrcode.image.svg import SvgPathImage

CANONICAL_URL = "https://xiaomiju.xyz/"
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output" / "poster"
PNG_PATH = OUTPUT_DIR / "RootScope_官网二维码_海报用_2048.png"
SVG_PATH = OUTPUT_DIR / "RootScope_官网二维码_海报用.svg"
RECEIPT_PATH = OUTPUT_DIR / "RootScope_官网二维码_验收回执.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_qr() -> qrcode.QRCode:
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_H,
        box_size=20,
        border=4,
    )
    qr.add_data(CANONICAL_URL)
    qr.make(fit=True)
    return qr


def write_png(qr: qrcode.QRCode) -> None:
    matrix = qr.get_matrix()
    module_count = len(matrix)
    canvas_size = 2048
    module_px = canvas_size // module_count
    rendered = module_px * module_count
    offset = (canvas_size - rendered) // 2
    image = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(image)
    for row, values in enumerate(matrix):
        for column, dark in enumerate(values):
            if dark:
                x0 = offset + column * module_px
                y0 = offset + row * module_px
                draw.rectangle(
                    (x0, y0, x0 + module_px - 1, y0 + module_px - 1),
                    fill=0,
                )
    image.save(PNG_PATH, "PNG", optimize=True)


def write_svg(qr: qrcode.QRCode) -> None:
    qr.make_image(
        image_factory=SvgPathImage,
        fill_color="#000000",
        back_color="#ffffff",
    ).save(SVG_PATH)


def decode_png(path: Path) -> str:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    value, points, _ = cv2.QRCodeDetector().detectAndDecode(image)
    if points is None or not value:
        raise RuntimeError("OpenCV QR decode failed")
    return value


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    qr = build_qr()
    write_png(qr)
    write_svg(qr)
    decoded = decode_png(PNG_PATH)
    if decoded != CANONICAL_URL:
        raise RuntimeError(f"decoded URL mismatch: {decoded!r}")
    receipt = {
        "status": "PASS",
        "canonical_url": CANONICAL_URL,
        "decoded_png": decoded,
        "error_correction": "H",
        "quiet_zone_modules": 4,
        "png_dimensions": [2048, 2048],
        "png_sha256": sha256(PNG_PATH),
        "svg_sha256": sha256(SVG_PATH),
        "print_recommendation": "海报印刷边长不小于 40 mm",
    }
    RECEIPT_PATH.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
