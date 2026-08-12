from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "make_ai_ensemble_contact_sheets.py"
SPEC = importlib.util.spec_from_file_location("make_ai_ensemble_contact_sheets", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VisualFixture:
    def __init__(self, root: Path) -> None:
        self.dataset = root / "dataset"
        self.review = self.dataset / "review"
        self.ensemble = self.review / "ai_ensemble_v1"
        self.results = self.ensemble / "ai_siglip2_ensemble_results.jsonl"
        self.stats = self.ensemble / "ai_siglip2_ensemble_stats.json"
        self.output = self.review / "ai_ensemble_v1_visuals"
        self.human = self.review / "human_decisions"
        self.ensemble.mkdir(parents=True)
        self.human.mkdir()
        self.manifest = self.dataset / "manifest.jsonl"
        self.manifest.write_bytes(b"manifest-sentinel\n")
        (self.human / "journal.jsonl").write_bytes(b"human-sentinel\n")
        rows = []
        definitions = [
            (101, "grass_clump", "grass_clump", "AUTO_TARGET", 0.95, 0.80),
            (102, "low_shrub", "low_shrub", "HOLD", 0.61, 0.21),
            (103, "young_tree", "low_shrub", "HOLD", 0.88, 0.29),
            (104, "unknown", "young_tree", "AUTO_UNKNOWN", 0.08, 0.05),
            (105, "grass_clump", "grass_clump", "HOLD", 0.89, 0.31),
        ]
        for pageid, hint, pred, decision, admissibility, margin in definitions:
            relative = Path("images") / hint / f"{pageid}.png"
            path = self.dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (80 + pageid % 5, 64), (pageid % 255, 80, 120))
            image.putpixel((1, 1), (255, 255, 255))
            image.save(path)
            rows.append(
                {
                    "asset": f"wikimedia:{pageid}@sha256:{digest(path)}",
                    "pageid": pageid,
                    "image_path": relative.as_posix(),
                    "decision": decision,
                    "acquisition_class_hint": hint,
                    "suggested_class": pred,
                    "top1_class": pred,
                    "admissibility_probability": admissibility,
                    "top1_top2_margin": margin,
                }
            )
        self.results.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        self.stats.write_text(
            json.dumps(
                {
                    "candidate_count": len(rows),
                    "thresholds": {
                        "auto_unknown_max_admissible": 0.15,
                        "auto_target_min_admissible": 0.90,
                        "auto_target_min_margin": 0.30,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def build(self, *, replace: bool = False) -> Path:
        return MODULE.build_visuals(
            results_path=self.results,
            stats_path=self.stats,
            output_dir=self.output,
            columns=2,
            page_size=2,
            sample_size=3,
            font_path=None,
            replace=replace,
        )


class ContactSheetTests(unittest.TestCase):
    def test_builds_group_and_score_sheets_without_touching_authoritative_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = VisualFixture(Path(temporary))
            manifest_before = digest(fixture.manifest)
            human_before = digest(fixture.human / "journal.jsonl")
            output = fixture.build()
            index = json.loads((output / "visual_index.json").read_text(encoding="utf-8"))
            receipt = json.loads((output / "visual_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(index["status"], MODULE.STATUS)
            self.assertFalse(index["authority"]["human_review"])
            self.assertEqual(index["inputs"]["candidate_count"], 5)
            self.assertEqual(receipt["sheet_count"], index["sheet_count"])
            self.assertGreaterEqual(index["sheet_count"], 11)
            self.assertTrue((output / "by_decision_class").is_dir())
            self.assertTrue((output / "score_samples" / "admissibility_high__p001.png").is_file())
            self.assertEqual(digest(fixture.manifest), manifest_before)
            self.assertEqual(digest(fixture.human / "journal.jsonl"), human_before)

    def test_rebuild_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = VisualFixture(Path(temporary))
            output = fixture.build()
            first = {
                path.relative_to(output).as_posix(): digest(path)
                for path in output.rglob("*")
                if path.is_file()
            }
            fixture.build(replace=True)
            second = {
                path.relative_to(output).as_posix(): digest(path)
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first, second)

    def test_rejects_non_ai_sibling_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = VisualFixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "ai_"):
                MODULE.build_visuals(
                    results_path=fixture.results,
                    stats_path=fixture.stats,
                    output_dir=fixture.review / "human_decisions",
                    columns=2,
                    page_size=2,
                    sample_size=2,
                    font_path=None,
                    replace=False,
                )

    def test_rejects_image_path_escape_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = VisualFixture(Path(temporary))
            rows = [json.loads(line) for line in fixture.results.read_text(encoding="utf-8").splitlines()]
            rows[0]["image_path"] = "../outside.png"
            fixture.results.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsafe image_path"):
                fixture.build()
            self.assertFalse(fixture.output.exists())


if __name__ == "__main__":
    unittest.main()
