from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.vision.card_geometric_matcher import (
    CLAIM_SCOPE,
    MatcherConfig,
    fuse_known_card_consensus,
    main,
    match_known_card,
)


def _synthetic_card() -> np.ndarray:
    """Deterministic print-like card with repeat-resistant local features."""

    image = np.full((360, 520, 3), 238, dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (511, 351), (18, 18, 18), 7)
    cv2.rectangle(image, (25, 25), (495, 335), (60, 110, 35), 3)
    cv2.putText(image, "ROOTSCOPE", (52, 82), cv2.FONT_HERSHEY_DUPLEX, 1.35, (20, 30, 20), 3)
    cv2.putText(image, "YOUNG TREE 01", (58, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (10, 10, 10), 2)
    cv2.line(image, (40, 105), (480, 105), (35, 80, 25), 4)
    cv2.line(image, (255, 115), (255, 287), (35, 80, 25), 3)
    for index in range(11):
        x = 55 + index * 40
        y = 135 + (index % 3) * 42
        radius = 8 + (index % 4) * 2
        color = (20 + index * 8, 45 + index * 5, 15 + index * 6)
        cv2.circle(image, (x, y), radius, color, -1)
        cv2.circle(image, (x, y), radius + 4, (15, 15, 15), 2)
    for index in range(8):
        x = 282 + (index % 4) * 48
        y = 140 + (index // 4) * 78
        cv2.rectangle(image, (x, y), (x + 25, y + 38), (35, 90 + index * 7, 30), -1)
        cv2.line(image, (x, y), (x + 25, y + 38), (240, 240, 240), 2)
        cv2.line(image, (x + 25, y), (x, y + 38), (15, 15, 15), 2)
    rng = np.random.default_rng(20260717)
    for _ in range(75):
        x = int(rng.integers(35, 490))
        y = int(rng.integers(115, 285))
        value = int(rng.integers(10, 210))
        cv2.drawMarker(
            image,
            (x, y),
            (value, 255 - value // 2, value // 3),
            markerType=cv2.MARKER_CROSS,
            markerSize=int(rng.integers(5, 11)),
            thickness=1,
        )
    return image


def _perspective_query(card: np.ndarray) -> np.ndarray:
    source = np.float32(
        [[0, 0], [card.shape[1] - 1, 0], [card.shape[1] - 1, card.shape[0] - 1], [0, card.shape[0] - 1]]
    )
    destination = np.float32([[104, 72], [566, 48], [588, 422], [77, 438]])
    homography = cv2.getPerspectiveTransform(source, destination)
    query = cv2.warpPerspective(card, homography, (680, 500), borderValue=(178, 163, 141))
    return cv2.convertScaleAbs(query, alpha=0.82, beta=22)


def _out_of_bounds_query(card: np.ndarray) -> np.ndarray:
    source = np.float32(
        [[0, 0], [card.shape[1] - 1, 0], [card.shape[1] - 1, card.shape[0] - 1], [0, card.shape[0] - 1]]
    )
    destination = np.float32([[-35, 42], [460, 58], [478, 408], [-48, 425]])
    homography = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(card, homography, (520, 470), borderValue=(185, 170, 145))


class MatcherConfigTests(unittest.TestCase):
    def test_unknown_config_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown matcher config"):
            MatcherConfig.from_mapping({"min_inliers": 10, "typo_gate": 1})

    def test_all_required_thresholds_are_explicit(self) -> None:
        payload = MatcherConfig().to_dict()
        required = {
            "min_template_keypoints",
            "min_query_keypoints",
            "min_mutual_good_matches",
            "min_inliers",
            "min_inlier_ratio",
            "max_median_reprojection_error_px",
            "min_projected_area_ratio",
            "max_projected_area_ratio",
            "projected_boundary_margin_px",
        }
        self.assertTrue(required.issubset(payload))


class CardGeometricMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.card = _synthetic_card()
        self.query = _perspective_query(self.card)

    def test_perspective_and_moderate_brightness_change_pass(self) -> None:
        result = match_known_card(
            self.card,
            self.query,
            template_id="synthetic-young-tree-card-v1",
            template_class="young_tree",
        )
        self.assertTrue(result.passed, result.reject_reasons)
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.claim_scope, CLAIM_SCOPE)
        self.assertFalse(result.irrigation_execution_authority)
        self.assertTrue(all(value is False for value in result.authority.values()))
        self.assertGreaterEqual(result.metrics["mutual_good_matches"], 18)
        self.assertGreaterEqual(result.metrics["inliers"], 14)
        self.assertTrue(all(gate["passed"] for gate in result.gates.values()))
        self.assertFalse(result.provenance["semantic_recognition_performed"])

    def test_template_file_sha_is_raw_file_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.png"
            self.assertTrue(cv2.imwrite(str(path), self.card))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            result = match_known_card(
                path,
                self.query,
                template_id="sha-binding",
                template_class="young_tree",
            )
        self.assertEqual(result.template_sha256, expected)
        self.assertEqual(result.provenance["template"]["sha256_scope"], "raw_file_bytes")

    def test_extreme_brightness_loss_is_rejected(self) -> None:
        flat_query = np.full_like(self.query, 252)
        result = match_known_card(
            self.card,
            flat_query,
            template_id="brightness-negative",
            template_class="young_tree",
        )
        self.assertFalse(result.passed)
        self.assertIn("QUERY_KEYPOINTS_BELOW_MIN", result.reject_reasons)
        self.assertFalse(result.irrigation_execution_authority)

    def test_heavy_occlusion_is_rejected(self) -> None:
        occluded = self.query.copy()
        cv2.rectangle(occluded, (55, 35), (625, 430), (127, 127, 127), -1)
        result = match_known_card(
            self.card,
            occluded,
            template_id="occlusion-negative",
            template_class="young_tree",
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.reject_reasons)
        self.assertFalse(result.irrigation_execution_authority)

    def test_unrelated_textured_image_is_rejected(self) -> None:
        rng = np.random.default_rng(91)
        unrelated = rng.integers(0, 256, size=self.query.shape, dtype=np.uint8)
        result = match_known_card(
            self.card,
            unrelated,
            template_id="unrelated-negative",
            template_class="young_tree",
        )
        self.assertFalse(result.passed)
        self.assertIn("HOMOGRAPHY_NOT_ESTIMATED", result.reject_reasons)

    def test_projected_quad_outside_query_bounds_is_rejected(self) -> None:
        result = match_known_card(
            self.card,
            _out_of_bounds_query(self.card),
            template_id="boundary-negative",
            template_class="young_tree",
        )
        self.assertFalse(result.passed)
        self.assertIsNotNone(result.metrics["homography_template_to_query"])
        self.assertIn(
            "PROJECTED_QUADRILATERAL_OUT_OF_BOUNDS_OR_UNAVAILABLE",
            result.reject_reasons,
        )

    def test_orb_is_used_only_as_recorded_fallback_when_akaze_cannot_run(self) -> None:
        with patch("app.vision.card_geometric_matcher.cv2.AKAZE_create", side_effect=RuntimeError("fixture")):
            result = match_known_card(
                self.card,
                self.query,
                template_id="orb-fallback",
                template_class="young_tree",
            )
        self.assertTrue(result.passed, result.reject_reasons)
        self.assertEqual(result.detector["selected"], "ORB")
        self.assertTrue(result.detector["fallback_used"])
        self.assertEqual(result.detector["detector_errors"][0]["detector"], "AKAZE")

    def test_cli_emits_json_and_reject_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path = root / "template.png"
            query_path = root / "query.png"
            output_path = root / "result.json"
            cv2.imwrite(str(template_path), self.card)
            cv2.imwrite(str(query_path), np.full_like(self.query, 252))
            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--template",
                        str(template_path),
                        "--query",
                        str(query_path),
                        "--template-id",
                        "cli-negative",
                        "--template-class",
                        "young_tree",
                        "--output-json",
                        str(output_path),
                    ]
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "REJECT")
        self.assertFalse(payload["irrigation_execution_authority"])


class ConsensusFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = match_known_card(
            _synthetic_card(),
            _perspective_query(_synthetic_card()),
            template_id="consensus-fixture",
            template_class="young_tree",
        )
        if not cls.geometry.passed:
            raise AssertionError(cls.geometry.reject_reasons)

    def test_matching_independent_gates_form_consensus_without_authority(self) -> None:
        fused = fuse_known_card_consensus(
            semantic_class="young_tree",
            semantic_gate_passed=True,
            template_class="young_tree",
            geometric_result=self.geometry,
        )
        self.assertEqual(fused["status"], "KNOWN_CARD_CONSENSUS")
        self.assertTrue(fused["passed"])
        self.assertFalse(fused["irrigation_execution_authority"])
        self.assertTrue(all(value is False for value in fused["authority"].values()))

    def test_class_disagreement_rejects(self) -> None:
        fused = fuse_known_card_consensus(
            semantic_class="shrub",
            semantic_gate_passed=True,
            template_class="young_tree",
            geometric_result=self.geometry,
        )
        self.assertEqual(fused["status"], "REJECT")
        self.assertIn("SEMANTIC_TEMPLATE_CLASS_DISAGREEMENT", fused["reject_reasons"])
        self.assertFalse(fused["irrigation_execution_authority"])

    def test_failed_semantic_gate_rejects_even_when_classes_match(self) -> None:
        fused = fuse_known_card_consensus(
            semantic_class="young_tree",
            semantic_gate_passed=False,
            template_class="young_tree",
            geometric_result=self.geometry,
        )
        self.assertEqual(fused["status"], "REJECT")
        self.assertIn("SEMANTIC_GATE_REJECTED", fused["reject_reasons"])

    def test_tampered_nested_authority_or_status_rejects(self) -> None:
        tampered = self.geometry.to_dict()
        tampered["authority"]["serial_write"] = True
        fused = fuse_known_card_consensus(
            semantic_class="young_tree",
            semantic_gate_passed=True,
            template_class="young_tree",
            geometric_result=tampered,
        )
        self.assertEqual(fused["status"], "REJECT")
        self.assertIn("GEOMETRIC_NESTED_AUTHORITY_VIOLATION", fused["reject_reasons"])

        tampered = self.geometry.to_dict()
        tampered["status"] = "REJECT"
        fused = fuse_known_card_consensus(
            semantic_class="young_tree",
            semantic_gate_passed=True,
            template_class="young_tree",
            geometric_result=tampered,
        )
        self.assertEqual(fused["status"], "REJECT")
        self.assertIn("GEOMETRIC_STATUS_MISMATCH", fused["reject_reasons"])


if __name__ == "__main__":
    unittest.main()
