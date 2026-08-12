from __future__ import annotations

import unittest

import numpy as np

from app.vision import PixelROI, evaluate_frame_quality, verify_wetting_change


class QualityGateTests(unittest.TestCase):
    def test_flat_frame_is_rejected(self) -> None:
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        result = evaluate_frame_quality(frame)
        self.assertFalse(result.passed)
        self.assertIn("LOW_CONTRAST", result.reasons)

    def test_black_frame_is_rejected(self) -> None:
        result = evaluate_frame_quality(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertFalse(result.passed)
        self.assertIn("UNDEREXPOSED", result.reasons)
        self.assertIn("DARK_CLIPPING", result.reasons)

    def test_textured_fixture_passes(self) -> None:
        y, x = np.indices((240, 320))
        checker = (((x // 12 + y // 12) % 2) * 150 + 50).astype(np.uint8)
        frame = np.repeat(checker[:, :, None], 3, axis=2)
        result = evaluate_frame_quality(frame)
        self.assertTrue(result.passed, result.reasons)


class WettingVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = np.full((240, 320, 3), 170, dtype=np.uint8)
        self.target = PixelROI("Z2", 100, 80, 100, 70)
        self.z1 = PixelROI("Z1", 100, 10, 100, 60)
        self.z3 = PixelROI("Z3", 100, 160, 100, 60)

    def test_target_only_change_passes(self) -> None:
        result_frame = self.baseline.copy()
        result_frame[80:150, 100:200] = 95
        result = verify_wetting_change(
            self.baseline, result_frame, self.target, (self.z1, self.z3)
        )
        self.assertTrue(result.passed, result.reasons)
        self.assertEqual(result.max_neighbor_changed_fraction, 0.0)

    def test_cross_layer_change_is_rejected(self) -> None:
        result_frame = self.baseline.copy()
        result_frame[80:150, 100:200] = 95
        result_frame[160:220, 100:200] = 95
        result = verify_wetting_change(
            self.baseline, result_frame, self.target, (self.z1, self.z3)
        )
        self.assertFalse(result.passed)
        self.assertIn("NON_TARGET_CHANGE_TOO_LARGE", result.reasons)

    def test_shape_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            verify_wetting_change(
                self.baseline,
                np.zeros((100, 100, 3), dtype=np.uint8),
                self.target,
            )


if __name__ == "__main__":
    unittest.main()
