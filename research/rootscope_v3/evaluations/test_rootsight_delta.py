from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from rootscope_v3.vision.domain_augmentation import augment_image, sample_recipe
from rootscope_v3.vision.group_split import assign_group_splits, audit_group_splits
from rootscope_v3.vision.rootsight_delta import (
    WettingDeltaConfig,
    evaluate_optical_ood,
    evaluate_wetting_delta,
    fuse_temporal_scores,
    register_translation,
)


def textured_fixture(height: int = 120, width: int = 180) -> np.ndarray:
    y, x = np.indices((height, width))
    base = 145 + 28 * np.sin(x / 4.0) + 22 * np.cos(y / 5.0)
    return np.stack((base + 8, base, base - 12), axis=2).clip(0, 255).astype(np.uint8)


class RootSightDeltaTests(unittest.TestCase):
    def test_augmentation_is_reproducible_and_changes_domain(self) -> None:
        source = Image.fromarray(textured_fixture())
        first = np.asarray(augment_image(source, sample_recipe(17)))
        second = np.asarray(augment_image(source, sample_recipe(17)))
        self.assertTrue(np.array_equal(first, second))
        self.assertGreater(float(np.mean(np.abs(first.astype(float) - np.asarray(source)))), 2.0)

    def test_capture_session_stays_in_one_split(self) -> None:
        rows = [
            {"session_id": "s1", "sha256": f"a{i:064x}", "class_id": "grass_clump"}
            for i in range(4)
        ] + [
            {"session_id": "s2", "sha256": f"b{i:064x}", "class_id": "unknown"}
            for i in range(3)
        ]
        assigned = assign_group_splits(rows)
        audit = audit_group_splits(assigned)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(len({row["split"] for row in assigned if row["session_id"] == "s1"}), 1)

    def test_ood_rejects_flat_low_confidence_frame(self) -> None:
        result = evaluate_optical_ood(np.full((80, 120, 3), 128, np.uint8), [0.1, 0.1, 0.1, 0.1])
        self.assertTrue(result.hold)
        self.assertIn("LOW_MODEL_CONFIDENCE", result.reasons)
        self.assertIn("BLUR_OR_FLAT_SCENE", result.reasons)

    def test_temporal_consensus_accepts_stable_and_holds_disagreement(self) -> None:
        stable = [{"grass": 0.82, "shrub": 0.18}, {"grass": 0.78, "shrub": 0.22}, {"grass": 0.80, "shrub": 0.20}]
        self.assertFalse(fuse_temporal_scores(stable).hold)
        unstable = [{"grass": 0.9, "shrub": 0.1}, {"grass": 0.1, "shrub": 0.9}, {"grass": 0.2, "shrub": 0.8}]
        self.assertTrue(fuse_temporal_scores(unstable, min_agreement=0.8).hold)

    def test_translation_registration_recovers_shift(self) -> None:
        before = textured_fixture()
        after = np.zeros_like(before)
        after[4:, 7:] = before[:-4, :-7]
        _, _, result = register_translation(before, after, max_shift_px=16)
        self.assertLessEqual(abs(result.dx + 7), 1)
        self.assertLessEqual(abs(result.dy + 4), 1)
        self.assertGreater(result.confidence, 4)

    def test_wetting_target_observed_with_mass_crosscheck(self) -> None:
        before = textured_fixture()
        after = before.copy()
        after[42:88, 64:116] = (after[42:88, 64:116].astype(float) * 0.62).astype(np.uint8)
        receipt = evaluate_wetting_delta(
            [before, before],
            [after, after],
            target_roi=(60, 38, 60, 54),
            neighbor_rois=((8, 38, 44, 54), (128, 38, 44, 54)),
            reference_roi=(4, 4, 30, 22),
            mass_delta_g=4.2,
            config=WettingDeltaConfig(min_registration_confidence=2.0),
        )
        self.assertTrue(receipt.passed, receipt.reasons)
        self.assertGreater(receipt.target_coverage, 0.5)
        self.assertLess(receipt.neighbor_spill, 0.05)
        self.assertEqual(receipt.mass_visual_consistency, "CONSISTENT")
        self.assertFalse(receipt.physical_authority)

    def test_neighbor_spill_and_mass_conflict_hold(self) -> None:
        before = textured_fixture()
        after = before.copy()
        after[35:95, 15:165] = (after[35:95, 15:165].astype(float) * 0.60).astype(np.uint8)
        receipt = evaluate_wetting_delta(
            [before],
            [after],
            target_roi=(60, 38, 60, 54),
            neighbor_rois=((8, 38, 44, 54), (128, 38, 44, 54)),
            reference_roi=(4, 4, 30, 22),
            mass_delta_g=0.0,
            config=WettingDeltaConfig(min_registration_confidence=2.0),
        )
        self.assertFalse(receipt.passed)
        self.assertIn("NEIGHBOR_SPILL_TOO_LARGE", receipt.reasons)
        self.assertIn("MASS_VISUAL_CONFLICT", receipt.reasons)


if __name__ == "__main__":
    unittest.main()
