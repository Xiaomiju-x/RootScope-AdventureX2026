import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "ai_vlm_second_pass.py"
SPEC = importlib.util.spec_from_file_location("ai_vlm_second_pass", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ParseAnswerTests(unittest.TestCase):
    def valid_payload(self):
        value = {field: False for field in MODULE.BOOL_FIELDS}
        value.update(
            {
                "is_photograph": True,
                "morphology_class": "other",
                "confidence": 0.8,
                "short_evidence": "pixels",
            }
        )
        return value

    def test_extracts_json_from_wrapping_text(self):
        value = self.valid_payload()
        parsed = MODULE.parse_answer("answer: " + json.dumps(value) + " end")
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.fields["morphology_class"], "other")

    def test_rejects_string_boolean(self):
        value = self.valid_payload()
        value["whole_plant_visible"] = "false"
        parsed = MODULE.parse_answer(json.dumps(value))
        self.assertFalse(parsed.valid)
        self.assertIn("whole_plant_visible", parsed.error)

    def test_rejects_missing_field(self):
        value = self.valid_payload()
        del value["base_visible"]
        parsed = MODULE.parse_answer(json.dumps(value))
        self.assertFalse(parsed.valid)
        self.assertIn("base_visible", parsed.error)

    def test_rejects_unknown_morphology(self):
        value = self.valid_payload()
        value["morphology_class"] = "tree"
        parsed = MODULE.parse_answer(json.dumps(value))
        self.assertFalse(parsed.valid)


class OutcomeTests(unittest.TestCase):
    def clean_fields(self):
        return {
            "is_photograph": True,
            "exactly_one_dominant_plant": True,
            "whole_plant_visible": True,
            "base_visible": True,
            "crown_visible": True,
            "closeup_or_part": False,
            "hand_or_person": False,
            "document_or_specimen": False,
            "multiple_or_landscape": False,
            "mature_tree": False,
            "morphology_class": "low_shrub",
            "confidence": 0.9,
            "short_evidence": "one whole shrub",
        }

    def test_clean_matching_sample_is_machine_strict_positive(self):
        outcome, reasons = MODULE.vlm_outcome(self.clean_fields(), "low_shrub")
        self.assertEqual(outcome, "VLM_STRICT_POSITIVE")
        self.assertEqual(reasons, ["ALL_CONSERVATIVE_STRUCTURE_GATES_PASS"])

    def test_morphology_disagreement_holds(self):
        outcome, reasons = MODULE.vlm_outcome(self.clean_fields(), "young_tree")
        self.assertEqual(outcome, "VLM_HOLD")
        self.assertIn("MORPHOLOGY_DISAGREES_WITH_ACQUISITION_HINT", reasons)

    def test_hand_is_hard_exclude(self):
        fields = self.clean_fields()
        fields["hand_or_person"] = True
        outcome, reasons = MODULE.vlm_outcome(fields, "low_shrub")
        self.assertEqual(outcome, "VLM_EXCLUDE")
        self.assertIn("HAND_OR_PERSON", reasons)

    def test_missing_base_without_hard_reject_holds(self):
        fields = self.clean_fields()
        fields["base_visible"] = False
        outcome, _ = MODULE.vlm_outcome(fields, "low_shrub")
        self.assertEqual(outcome, "VLM_HOLD")

    def test_cross_gate_never_upgrades_gpu_hold_to_consensus(self):
        self.assertEqual(
            MODULE.cross_gate_outcome("HOLD", "VLM_STRICT_POSITIVE"),
            "VLM_POSITIVE_GPU_HOLD_NO_CONSENSUS",
        )


class GoldenGateTests(unittest.TestCase):
    def test_perfect_golden_results_pass(self):
        results = []
        for case in MODULE.GOLDEN_CASES:
            fields = {field: False for field in MODULE.BOOL_FIELDS}
            fields.update(
                {
                    "morphology_class": "other",
                    "confidence": 0.9,
                    "short_evidence": "fixture",
                }
            )
            fields.update(case["expected"])
            results.append({"pageid": case["pageid"], "parse_valid": True, "vlm_fields": fields})
        report = MODULE.score_golden(results)
        self.assertTrue(report["qualified_for_this_machine_audit"])

    def test_unparseable_golden_results_fail_closed(self):
        results = [
            {"pageid": case["pageid"], "parse_valid": False, "vlm_fields": {}}
            for case in MODULE.GOLDEN_CASES
        ]
        report = MODULE.score_golden(results)
        self.assertFalse(report["qualified_for_this_machine_audit"])
        self.assertEqual(report["status"], "GOLDEN_SANITY_FAIL_STOP_FULL_RUN")


class HashTests(unittest.TestCase):
    def test_model_inventory_ignores_huggingface_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / ".cache" / "huggingface").mkdir(parents=True)
            (root / ".cache" / "huggingface" / "noise").write_text("x", encoding="utf-8")
            files, artifact_hash = MODULE.model_inventory(root)
            self.assertEqual([item["path"] for item in files], ["config.json"])
            self.assertEqual(len(artifact_hash), 64)


if __name__ == "__main__":
    unittest.main()
