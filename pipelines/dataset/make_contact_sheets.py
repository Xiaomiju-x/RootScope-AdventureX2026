#!/usr/bin/env python3
"""Build review contact sheets from a RootScope dataset manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_dataset = script.parents[2] / "datasets" / "desert_plants_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument(
        "--review-label",
        default="pending manual review",
        help="Header label only; does not write or imply any review decision.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    records: dict[str, list[dict]] = defaultdict(list)
    for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            records[item["class_id"]].append(item)

    output = dataset / "contact_sheets"
    output.mkdir(exist_ok=True)
    font = ImageFont.load_default()
    cell_w, image_h, label_h = 256, 180, 48
    header_h = 42

    for class_id, items in sorted(records.items()):
        rows = (len(items) + args.columns - 1) // args.columns
        sheet = Image.new("RGB", (args.columns * cell_w, header_h + rows * (image_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 12),
            f"{class_id} | {args.review_label} | {len(items)} images",
            fill="black",
            font=font,
        )
        for index, item in enumerate(sorted(items, key=lambda value: value["filename"])):
            row, column = divmod(index, args.columns)
            x = column * cell_w
            y = header_h + row * (image_h + label_h)
            path = dataset / item["filename"]
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                tile = ImageOps.fit(image, (cell_w, image_h), method=Image.Resampling.LANCZOS)
            sheet.paste(tile, (x, y))
            # Recreate the drawing context after paste. Some Pillow builds keep
            # a stale ImagingDraw buffer when paste and draw calls are mixed.
            draw = ImageDraw.Draw(sheet)
            draw.rectangle((x, y + image_h, x + cell_w - 1, y + image_h + label_h - 1), outline="#cccccc")
            label = f"{item['pageid']} | {item['species_hint'][:27]}\n{item['license'][:38]}"
            draw.multiline_text((x + 5, y + image_h + 5), label, fill="black", font=font, spacing=3)
        target = output / f"{class_id}.jpg"
        sheet.save(target, quality=90, optimize=True)
        print(target)

    candidates = [
        item
        for class_items in records.values()
        for item in class_items
        if item.get("domain") == "print_demo_source"
    ]
    if candidates:
        columns = 3
        rows = (len(candidates) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * cell_w, header_h + rows * (image_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 12), "RootScope print-demo candidates | RESERVED FROM TRAINING", fill="black", font=font)
        for index, item in enumerate(sorted(candidates, key=lambda value: (value["class_id"], value["pageid"]))):
            row, column = divmod(index, columns)
            x = column * cell_w
            y = header_h + row * (image_h + label_h)
            with Image.open(dataset / item["filename"]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                tile = ImageOps.fit(image, (cell_w, image_h), method=Image.Resampling.LANCZOS)
            sheet.paste(tile, (x, y))
            draw = ImageDraw.Draw(sheet)
            draw.rectangle((x, y + image_h, x + cell_w - 1, y + image_h + label_h - 1), outline="#cccccc")
            label = f"{item['class_id']} | {item['pageid']}\n{item['license']} | UVC/print review pending"
            draw.multiline_text((x + 5, y + image_h + 5), label, fill="black", font=font, spacing=3)
        target = output / "print_demo_candidates.jpg"
        sheet.save(target, quality=92, optimize=True)
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
