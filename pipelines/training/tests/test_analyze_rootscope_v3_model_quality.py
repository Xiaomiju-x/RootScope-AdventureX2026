from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS / "analyze_rootscope_v3_model_quality.py"
SPEC = importlib.util.spec_from_file_location("analyze_rootscope_v3_model_quality", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RootScopeV3ModelQualityAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = TOOLS.parents[1]
        cls.run_root = cls.workspace / MODULE.RUN_RELATIVE
        cls.pack = cls.workspace / MODULE.PACK_RELATIVE

    def test_fixed_protocol_and_visual_selection_contract(self) -> None:
        self.assertEqual((17, 29, 43), MODULE.SEEDS)
        self.assertEqual(
            ("grass_clump", "low_shrub", "young_tree", "unknown"),
            MODULE.CLASS_NAMES,
        )
        self.assertEqual(224, MODULE.INPUT_SIZE)
        self.assertEqual(
            {
                "grass_clump": [163498042, 38233728],
                "low_shrub": [68787114, 66745979],
                "young_tree": [92774234],
                "unknown": [157364276],
            },
            MODULE.VISUALLY_INSPECTED_CARD_PAGEIDS,
        )

    def test_fixed_tta_has_exactly_five_crops_and_flips(self) -> None:
        from PIL import Image

        views = MODULE.build_tta_views(Image.new("RGB", (427, 311), (80, 120, 160)))
        self.assertEqual(10, len(views))
        self.assertTrue(all(tuple(view.shape) == (3, 224, 224) for view in views))

    def test_metric_confusion_contract(self) -> None:
        import torch

        labels = torch.tensor([0, 1, 2, 3])
        logits = torch.tensor(
            [
                [8.0, 0.0, 0.0, 0.0],
                [0.0, 8.0, 0.0, 0.0],
                [0.0, 8.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 8.0],
            ]
        )
        metrics = MODULE.metrics_from_logits(logits, labels, domain="fixture")
        self.assertEqual(0.75, metrics["accuracy"])
        self.assertEqual(0.75, metrics["balanced_accuracy_present_classes"])
        self.assertEqual(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            metrics["confusion_matrix_truth_rows_prediction_columns"],
        )

    def test_current_receipt_hash_envelope_and_false_authority_verify(self) -> None:
        receipt, verification = MODULE.verify_receipt(self.run_root, self.pack)
        self.assertEqual("PASS", verification["status"])
        self.assertTrue(verification["full_sha256_coverage"])
        self.assertTrue(all(value is False for value in receipt["authority"].values()))
        self.assertFalse(receipt["model_candidate"])
        self.assertFalse(receipt["x5_ready"])
        self.assertFalse(receipt["physical_print_tested"])

    def test_hash_envelope_fails_closed_on_uncovered_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.txt").write_text("a", encoding="utf-8")
            digest = MODULE.sha256_file(root / "a.txt")
            (root / "SHA256SUMS").write_text(f"{digest}  a.txt\n", encoding="utf-8")
            MODULE.verify_hash_envelope(root)
            (root / "uncovered.txt").write_text("not covered", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.AnalysisError, "coverage mismatch"):
                MODULE.verify_hash_envelope(root)

    def test_written_evidence_has_no_promoted_authority_or_hardware_claim(self) -> None:
        evidence_path = self.workspace / "evidence" / "rootscope_v3_model_quality_analysis.json"
        report = json.loads(evidence_path.read_text(encoding="utf-8"))
        self.assertTrue(all(value is False for value in report["authority"].values()))
        self.assertFalse(report["model_candidate"])
        self.assertFalse(report["model_qualified"])
        self.assertFalse(report["physical_print_tested"])
        self.assertFalse(report["uvc_recapture_evaluated"])
        self.assertFalse(report["x5_ready"])
        self.assertFalse(report["bpu_compiled"])
        self.assertFalse(report["project_hardware_touched_by_analysis"])
        self.assertFalse(report["network_touched_by_analysis"])


if __name__ == "__main__":
    unittest.main()
