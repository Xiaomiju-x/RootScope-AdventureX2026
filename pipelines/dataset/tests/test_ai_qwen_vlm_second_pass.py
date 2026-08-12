import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_DIR))
MODULE_PATH = DATASET_DIR / "ai_qwen_vlm_second_pass.py"
SPEC = importlib.util.spec_from_file_location("ai_qwen_vlm_second_pass", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QwenVLMContractTests(unittest.TestCase):
    def test_prompt_has_exact_required_fields(self):
        for field in MODULE.common.BOOL_FIELDS:
            self.assertIn(field, MODULE.USER_PROMPT)
        self.assertIn("morphology_class", MODULE.USER_PROMPT)
        self.assertIn("confidence", MODULE.USER_PROMPT)
        self.assertIn("short_evidence", MODULE.USER_PROMPT)

    def test_quantization_is_frozen_to_nf4(self):
        quant = MODULE.PROMPT_CONTRACT["quantization"]
        self.assertTrue(quant["load_in_4bit"])
        self.assertEqual(quant["bnb_4bit_quant_type"], "nf4")
        self.assertEqual(quant["bnb_4bit_compute_dtype"], "bfloat16")

    def test_model_commit_and_two_weight_hashes_are_pinned(self):
        self.assertEqual(len(MODULE.MODEL_COMMIT), 40)
        self.assertEqual(len(MODULE.EXPECTED_WEIGHT_SHA256), 2)
        for value in MODULE.EXPECTED_WEIGHT_SHA256.values():
            self.assertEqual(len(value), 64)

    def test_license_is_explicitly_noncommercial(self):
        self.assertIn("non-commercial", MODULE.MODEL_LICENSE.lower())

    def test_validate_model_rejects_missing_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "LICENSE").write_text(
                "Qwen RESEARCH LICENSE AGREEMENT NON-COMMERCIAL PURPOSES ONLY",
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                MODULE.validate_model(root)

    def test_build_result_never_grants_authority(self):
        source = {
            "pageid": 1,
            "candidate_sha256": "a" * 64,
            "local_path": "images/x.jpg",
            "acquisition_hint": "low_shrub",
            "outcome": "HOLD",
        }
        payload = {field: False for field in MODULE.common.BOOL_FIELDS}
        payload.update(
            {
                "is_photograph": True,
                "exactly_one_dominant_plant": True,
                "whole_plant_visible": True,
                "base_visible": True,
                "crown_visible": True,
                "morphology_class": "low_shrub",
                "confidence": 0.9,
                "short_evidence": "one whole shrub",
            }
        )
        result = MODULE.build_result(
            source,
            json.dumps(payload),
            1.0,
            10,
            {"cuda_peak_reserved_bytes": 1},
            {"x": "y"},
        )
        self.assertEqual(result["vlm_outcome"], "VLM_STRICT_POSITIVE")
        self.assertTrue(all(value is False for value in result["authority"].values()))


if __name__ == "__main__":
    unittest.main()
