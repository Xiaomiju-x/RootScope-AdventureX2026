"""Build deterministic *simulated* camera frames for dual-path smoke tests.

These files are software fixtures only.  They are not UVC recaptures, site
acceptance images, accuracy evidence, or physical-hardware evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


POSITIVE = (
    ("grass_clump", "grass_clump_163498042.jpg"),
    ("low_shrub", "low_shrub_68787114.jpg"),
    ("young_tree", "young_tree_92774234.jpg"),
)
NEGATIVE_RELATIVE = (
    "datasets/rootscope_machine_curated_provisional_v3/images/unknown/"
    "unknown_157364276_04e7f49a1e66.jpg"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_frame(source: Path, destination: Path, *, seed: int) -> dict[str, Any]:
    width, height = 1280, 960
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((height, width), dtype=np.int32)
    base = np.empty((height, width, 3), dtype=np.uint8)
    texture = rng.integers(-4, 5, size=(height, width), dtype=np.int16)
    base[:, :, 0] = np.clip(205 + ((xx + yy) % 13) + texture, 0, 255)
    base[:, :, 1] = np.clip(178 + ((2 * xx + yy) % 11) + texture, 0, 255)
    base[:, :, 2] = np.clip(132 + ((xx + 3 * yy) % 9) + texture, 0, 255)
    canvas = Image.fromarray(base, mode="RGB")
    with Image.open(source) as opened:
        card = opened.convert("RGB")
    card.thumbnail((820, 650), Image.Resampling.LANCZOS)
    x = (width - card.width) // 2
    y = (height - card.height) // 2
    border = Image.new("RGB", (card.width + 32, card.height + 32), (246, 244, 236))
    border.paste(card, (16, 16))
    canvas.paste(border, (x - 16, y - 16))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=False)
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(destination),
        "output_sha256": sha256_file(destination),
        "canvas": [height, width, 3],
        "embedded_image_xywh": [x, y, card.width, card.height],
        "hardware_touched": False,
        "uvc_recapture": False,
    }


def build(adventurex: Path) -> dict[str, Any]:
    adventurex = adventurex.resolve(strict=True)
    template_root = adventurex / "rootscope" / "app" / "vision" / "known_card_templates"
    output = adventurex / "output" / "rootscope_dual_path_pc_fixtures"
    items: list[dict[str, Any]] = []
    for index, (class_name, filename) in enumerate(POSITIVE, 1):
        item = make_frame(
            template_root / filename,
            output / f"simulated_{class_name}.png",
            seed=20260717 + index,
        )
        item.update({"expected_semantic_class": class_name, "expected_consensus": True})
        items.append(item)
    negative = make_frame(
        adventurex / NEGATIVE_RELATIVE,
        output / "simulated_unknown_negative.png",
        seed=20260721,
    )
    negative.update({"expected_semantic_class": "unknown", "expected_consensus": False})
    items.append(negative)
    receipt = {
        "schema": "rootscope.dual-path-pc-fixtures.v1",
        "status": "SIMULATED_FIXTURES_ONLY_NOT_CAMERA_EVIDENCE",
        "items": items,
        "authority": {
            "camera_qualified": False,
            "x5_validated": False,
            "model_qualified": False,
            "execution_authority": False,
            "physical_authority": False,
            "physical_completion": False,
        },
    }
    receipt_path = output / "fixture_manifest.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": "PASS_SIMULATED_ONLY",
        "output": str(output),
        "manifest": str(receipt_path),
        "manifest_sha256": sha256_file(receipt_path),
        "count": len(items),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adventurex", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build(args.adventurex), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
