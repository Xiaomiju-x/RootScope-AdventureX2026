from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "collect_whole_plant_reacquisition.py"
SPEC = importlib.util.spec_from_file_location("collect_whole_plant_reacquisition", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WholePlantReacquisitionTests(unittest.TestCase):
    def test_every_acquisition_query_binds_the_structural_intent(self) -> None:
        self.assertTrue(MODULE.SOURCE_PLAN)
        for item in MODULE.SOURCE_PLAN:
            query = item.acquisition_query.lower()
            self.assertIn("whole", query)
            self.assertIn("base visible", query)
            self.assertIn("visible", query)
            self.assertIn("isolated", query)
            if item.class_id == "young_tree":
                self.assertRegex(query, r"sapling|seedling")

    def test_exact_license_pair_matching_has_reject_fallback(self) -> None:
        policy_path = SCRIPT.with_name("wikimedia_license_policy_v1.json")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        table = MODULE.license_table(policy)
        accepted = MODULE.resolve_license(
            table,
            "CC BY 4.0",
            "https://creativecommons.org/licenses/by/4.0/",
            "True",
        )
        self.assertEqual("CC_BY_4_0", accepted["canonical_id"])
        self.assertIsNone(
            MODULE.resolve_license(
                table,
                "CC BY 4.0 ",
                "https://creativecommons.org/licenses/by/4.0/",
                "True",
            )
        )
        self.assertIsNone(
            MODULE.resolve_license(table, "Public domain", "", "True")
        )

    def test_image_facts_are_deterministic(self) -> None:
        image = Image.new("RGB", (720, 720))
        for x in range(720):
            for y in range(720):
                image.putpixel((x, y), (255 - x % 256, y % 256, 0))
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        first = MODULE.image_facts(payload.getvalue())
        second = MODULE.image_facts(payload.getvalue())
        self.assertEqual(first, second)
        self.assertEqual((720, 720, "image/png"), first[:3])
        self.assertRegex(first[3], r"^[0-9a-f]{16}$")

    def test_output_is_explicitly_non_authoritative(self) -> None:
        sample = {
            "class_id": "grass_clump",
            "pageid": 1,
            "license_canonical_name": "CC0 1.0",
            "license_canonical_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "creator_group": "commons-creator:test",
            "artist": "Example",
            "filename": "images/grass_clump/example.jpg",
            "title": "File:Example.jpg",
            "source_page": "https://commons.wikimedia.org/?curid=1",
            "acquisition_query": "whole plant; base visible; crown visible; isolated",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            MODULE.save_outputs(root, [sample], "policy", "plan", 0)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["all_training_eligible"])
            self.assertFalse(summary["formal_human_review_authority"])
            plan = json.loads((root / "source_plan.json").read_text(encoding="utf-8"))
            self.assertFalse(plan["visual_ground_truth_authority"])
            self.assertEqual(
                "EXCLUDE_OR_HOLD_DO_NOT_FORCE_TO_TARGET_CLASS",
                plan["required_downstream_visual_gate"]["failure_action"],
            )


if __name__ == "__main__":
    unittest.main()
