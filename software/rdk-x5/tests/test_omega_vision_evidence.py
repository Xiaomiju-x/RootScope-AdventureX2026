from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from training.omega_vision.build_evidence import canonical_sha, sha256_file


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "omega_vision_v3_20260723" / "vision_consolidated.json"
ADDENDUM = (
    ROOT
    / "evidence"
    / "omega_vision_v3_20260723"
    / "vision_truth_boundary_addendum.json"
)


class OmegaVisionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        cls.addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))

    def test_composition_root_and_implementation_hashes_are_bound(self) -> None:
        payload = copy.deepcopy(self.report)
        expected = payload.pop("composition_root_sha256")
        self.assertEqual(expected, canonical_sha(payload))
        transition = self.addendum["source_hash_transition"]
        self.assertEqual(
            self.report["artifact_sha256"]["implementation_ood"],
            transition["app/omega_vision/ood.py_before_sha256"],
        )
        self.assertEqual(
            sha256_file(ROOT / "app" / "omega_vision" / "ood.py"),
            transition["app/omega_vision/ood.py_after_sha256"],
        )
        self.assertEqual(
            sha256_file(EVIDENCE),
            self.addendum["source_receipt"]["sha256"],
        )
        self.assertFalse(
            self.addendum["terminology_correction"][
                "formal_distribution_free_coverage_guarantee"
            ]
        )
        self.assertFalse(
            self.addendum["scope_clarification"][
                "holdout_reevaluated_for_this_addendum"
            ]
        )
        self.assertEqual(
            self.report["artifact_sha256"]["implementation_evidence_builder"],
            sha256_file(ROOT / "training" / "omega_vision" / "build_evidence.py"),
        )

    def test_holdouts_are_evaluation_only_and_truth_boundaries_remain_false(self) -> None:
        protocol = self.report["holdout_protocol"]
        self.assertEqual(1, protocol["creator_group_holdout_evaluation_count"])
        self.assertEqual(1, protocol["digital_print_source_holdout_evaluation_count"])
        self.assertFalse(protocol["used_for_weights_checkpoint_temperature_or_thresholds"])
        self.assertFalse(protocol["physical_print_domain_tested"])
        self.assertIsNone(self.report["model"]["selected_bin"])
        self.assertFalse(self.report["model"]["model_qualified"])
        self.assertEqual(
            "SKIPPED_NO_VALID_EMBEDDING_OUTPUT",
            self.report["mahalanobis"]["status"],
        )
        self.assertEqual(78, len(self.report["sample_records"]))

    def test_every_runtime_result_is_zero_authority(self) -> None:
        for record in self.report["sample_records"]:
            decision = record["decision"]
            self.assertTrue(decision["zero_authority"])
            self.assertFalse(decision["physical_authority"])
            self.assertFalse(decision["model_qualified"])
            self.assertIn(decision["decision"], ("CLASSIFY", "ABSTAIN"))


if __name__ == "__main__":
    unittest.main()
