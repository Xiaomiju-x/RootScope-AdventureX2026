from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


DATASET_TOOLS = Path(__file__).resolve().parents[1]
if str(DATASET_TOOLS) not in sys.path:
    sys.path.insert(0, str(DATASET_TOOLS))
SCRIPT = DATASET_TOOLS / "collect_young_tree_reacquisition.py"
SPEC = importlib.util.spec_from_file_location("collect_young_tree_reacquisition", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class YoungTreeReacquisitionTests(unittest.TestCase):
    def test_positive_youth_metadata(self) -> None:
        passed, youth, mature = MODULE.metadata_gate(
            "File:Acacia sapling.jpg",
            "A young Acacia sapling growing in dry sand",
        )
        self.assertTrue(passed)
        self.assertIn("young", youth)
        self.assertIn("sapling", youth)
        self.assertEqual([], mature)

    def test_missing_youth_text_is_rejected(self) -> None:
        passed, youth, mature = MODULE.metadata_gate(
            "File:Acacia tortilis.jpg", "An Acacia tree in a desert"
        )
        self.assertFalse(passed)
        self.assertEqual([], youth)
        self.assertEqual([], mature)

    def test_explicit_mature_language_overrides_youth_term(self) -> None:
        passed, youth, mature = MODULE.metadata_gate(
            "File:Young Acacia.jpg", "A mature old tree photographed in the desert"
        )
        self.assertFalse(passed)
        self.assertIn("young", youth)
        self.assertIn("mature", mature)
        self.assertIn("old_tree", mature)

    def test_detail_closeup_is_rejected(self) -> None:
        passed, youth, mature = MODULE.metadata_gate(
            "File:Acacia seedling close-up.jpg", "Leaf detail of a seedling"
        )
        self.assertFalse(passed)
        self.assertIn("seedling", youth)
        self.assertEqual([], mature)

    def test_every_query_is_young_tree_specific(self) -> None:
        self.assertGreaterEqual(len(MODULE.SOURCE_PLAN), 30)
        for item in MODULE.SOURCE_PLAN:
            self.assertEqual("young_tree", item.class_id)
            self.assertEqual("search", item.retrieval_mode)
            intent = item.acquisition_query.lower()
            self.assertIn("trunk base visible", intent)
            self.assertIn("entire crown visible", intent)
            self.assertIn("metadata must explicitly state", intent)
            self.assertIn("reject mature/ancient/old/large tree", intent)


if __name__ == "__main__":
    unittest.main()
