from __future__ import annotations

import unittest

import numpy as np

from app.omega_vision.ood import (
    Calibration,
    QualityMetrics,
    VisionGateError,
    calibrate,
    decide,
    energy_score,
    evaluate_quality,
)


CLASSES = ("grass_clump", "low_shrub", "young_tree", "unknown")


def permissive_calibration() -> Calibration:
    return Calibration(
        class_order=CLASSES,
        alpha=0.20,
        temperature=1.0,
        energy_upper=0.0,
        maxprob_lower=0.50,
        brightness_lower=0.10,
        brightness_upper=0.90,
        contrast_lower=0.01,
        sharpness_lower=0.001,
        clipped_upper=0.80,
        conformal_nonconformity=(0.5,) * 9,
    )


class OmegaVisionOodTests(unittest.TestCase):
    def test_energy_is_lower_for_more_confident_logits(self) -> None:
        self.assertLess(energy_score([8.0, 0.0, 0.0, 0.0]), energy_score([0.0] * 4))

    def test_quality_is_deterministic_and_bounded(self) -> None:
        y, x = np.indices((32, 48))
        frame = np.stack(((x * 7) % 256, (y * 11) % 256, ((x + y) * 5) % 256), axis=-1).astype(
            np.uint8
        )
        first = evaluate_quality(frame)
        second = evaluate_quality(frame.copy())
        self.assertEqual(first, second)
        self.assertGreater(first.contrast, 0.0)
        self.assertGreater(first.sharpness, 0.0)

    def test_high_confidence_singleton_classifies(self) -> None:
        quality = QualityMetrics(0.5, 0.2, 0.1, 0.0)
        result = decide([8.0, 0.0, 0.0, 0.0], quality, permissive_calibration())
        self.assertEqual("CLASSIFY", result.decision)
        self.assertEqual("grass_clump", result.predicted_class)
        self.assertTrue(result.zero_authority)
        self.assertFalse(result.physical_authority)

    def test_unknown_and_bad_quality_fail_closed(self) -> None:
        unknown = decide(
            [0.0, 0.0, 0.0, 8.0],
            QualityMetrics(0.5, 0.2, 0.1, 0.0),
            permissive_calibration(),
        )
        self.assertEqual("ABSTAIN", unknown.decision)
        self.assertIn("UNKNOWN_CLASS", unknown.reasons)
        dark = decide(
            [8.0, 0.0, 0.0, 0.0],
            QualityMetrics(0.01, 0.0, 0.0, 1.0),
            permissive_calibration(),
        )
        self.assertEqual("ABSTAIN", dark.decision)
        self.assertIn("QUALITY_TOO_DARK", dark.reasons)
        self.assertIn("QUALITY_LOW_CONTRAST", dark.reasons)

    def test_calibration_uses_explicit_reference_and_validation(self) -> None:
        reference_logits = [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
        quality = [QualityMetrics(0.5, 0.2, 0.1, 0.0)] * 4
        result = calibrate(
            reference_logits=reference_logits,
            reference_quality=quality,
            validation_logits=reference_logits,
            validation_labels=[0, 1, 2, 3],
            class_order=CLASSES,
        )
        self.assertEqual(4, len(result.conformal_nonconformity))
        self.assertEqual("SKIPPED_NO_VALID_EMBEDDING_OUTPUT", result.mahalanobis_status)
        self.assertEqual(
            ("EXPERIMENTAL_TRAIN_SUGGESTION", "EXPERIMENTAL_VAL_SUGGESTION"),
            result.calibration_roles,
        )

    def test_nonfinite_and_wrong_shape_are_rejected(self) -> None:
        with self.assertRaises(VisionGateError):
            energy_score([0.0, float("nan")])
        with self.assertRaises(VisionGateError):
            evaluate_quality(np.zeros((4, 4), dtype=np.uint8))
        with self.assertRaises(VisionGateError):
            decide([1.0, 2.0], QualityMetrics(0.5, 0.2, 0.1, 0.0), permissive_calibration())


if __name__ == "__main__":
    unittest.main()
