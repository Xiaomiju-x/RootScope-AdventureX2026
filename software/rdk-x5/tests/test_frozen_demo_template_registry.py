from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from app.vision.dual_path_demo import (
    AUTHORITY,
    REGISTERED_ROLE,
    REGISTRY_FROZEN_STATUS,
    load_template_registry,
)


ROOT = Path(__file__).resolve().parents[1]
ADVENTUREX = ROOT.parent
REGISTRY = ROOT / "app" / "vision" / "known_card_template_registry.frozen.experimental.json"
RECEIPT = ADVENTUREX / "evidence" / "rootscope_demo_template_registry_receipt_20260717.json"


class FrozenDemoTemplateRegistryTests(unittest.TestCase):
    def test_three_positive_templates_are_hash_bound_and_unknown_is_absent(self) -> None:
        registry = load_template_registry(REGISTRY)
        self.assertEqual(registry.status, REGISTRY_FROZEN_STATUS)
        self.assertEqual(len(registry.templates), 3)
        self.assertEqual(
            {item.class_name for item in registry.templates},
            {"grass_clump", "low_shrub", "young_tree"},
        )
        self.assertNotIn("unknown", {item.class_name for item in registry.templates})
        self.assertTrue(all(item.role == REGISTERED_ROLE for item in registry.templates))

    def test_receipt_is_explicitly_non_holdout_nonqualified_and_zero_authority(self) -> None:
        value = json.loads(RECEIPT.read_text(encoding="utf-8"))
        self.assertEqual(value["registry"]["sha256"], hashlib.sha256(REGISTRY.read_bytes()).hexdigest())
        self.assertTrue(value["registration_state"]["positive_templates_registered"])
        self.assertFalse(value["registration_state"]["unknown_negative_registered"])
        self.assertFalse(value["negative_card"]["registered"])
        self.assertTrue(all(not item["holdout_evidence"] for item in value["templates"]))
        self.assertFalse(value["authority"]["model_qualified"])
        self.assertFalse(value["authority"]["execution_authority"])
        self.assertFalse(value["authority"]["physical_authority"])
        self.assertFalse(value["authority"]["physical_completion"])
        self.assertTrue(all(flag is False for flag in value["authority"].values()))
        self.assertTrue(all(value["authority"][key] is False for key in AUTHORITY if key in value["authority"]))


if __name__ == "__main__":
    unittest.main()
