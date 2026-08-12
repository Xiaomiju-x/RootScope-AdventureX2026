from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from app.omega_vision.board_replay import (
    BoardReplayError,
    CLASS_ORDER,
    _AUTHORITY,
    preprocess,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "configs" / "omega" / "vision_board_replay_new_x5_20260723.json"
)


class OmegaVisionBoardReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_frozen_manifest_validates_without_touching_board_artifacts(self) -> None:
        result = validate_manifest(self.payload)
        self.assertEqual(result["calibration"].class_order, CLASS_ORDER)
        self.assertTrue(all(value is False for value in _AUTHORITY.values()))
        self.assertFalse(
            self.payload["calibration_provenance"][
                "formal_distribution_free_coverage_guarantee"
            ]
        )
        self.assertFalse(
            self.payload["truth_boundary"][
                "registered_demo_references_are_holdout"
            ]
        )

    def test_preprocess_is_finite_contiguous_static_contract(self) -> None:
        source = np.zeros((173, 311, 3), dtype=np.uint8)
        source[:, :, 0] = np.arange(311, dtype=np.uint8)[None, :]
        source[:, :, 1] = np.arange(173, dtype=np.uint8)[:, None]
        tensor = preprocess(Image.fromarray(source, mode="RGB"))
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertTrue(np.isfinite(tensor).all())

    def test_manifest_rejects_claim_upgrade_and_input_drift(self) -> None:
        upgraded = copy.deepcopy(self.payload)
        upgraded["truth_boundary"]["model_qualified"] = True
        with self.assertRaises(BoardReplayError):
            validate_manifest(upgraded)
        drifted = copy.deepcopy(self.payload)
        drifted["images"][0]["path"] = "/opt/rootscope/not-allowed.jpg"
        with self.assertRaises(BoardReplayError):
            validate_manifest(drifted)

    def test_pc_reference_has_three_demo_classes_and_one_abstention(self) -> None:
        references = self.payload["pc_reference"]
        self.assertEqual(
            [row["decision"] for row in references],
            ["CLASSIFY", "CLASSIFY", "CLASSIFY", "ABSTAIN"],
        )
        self.assertEqual(
            [row["raw_top1_class"] for row in references],
            list(CLASS_ORDER),
        )
        self.assertTrue(
            all(
                row["provenance_role"]
                == "REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT"
                for row in self.payload["images"][:3]
            )
        )


if __name__ == "__main__":
    unittest.main()
