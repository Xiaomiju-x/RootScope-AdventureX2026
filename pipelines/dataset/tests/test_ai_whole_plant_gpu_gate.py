from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "ai_whole_plant_gpu_gate.py"
POLICY_PATH = MODULE_PATH.with_name("ai_whole_plant_gpu_gate_policy_v1.json")
DATASET_ROOT = MODULE_PATH.parents[2] / "datasets" / "desert_plants_whole_plant_reacquisition_e1"
E2_ROOT = MODULE_PATH.parents[2] / "datasets" / "desert_plants_young_tree_reacquisition_e2"
E2_CONTRACT = MODULE_PATH.with_name("young_tree_reacquisition_e2_input_contract_v1.json")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("rootscope_ai_whole_plant_gpu_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def score_fixture(target: str = "grass_clump") -> dict[str, float]:
    policy = load_policy()
    scores = {prompt["id"]: -3.0 for prompt in MODULE._flatten_prompts(policy)}
    for prompt in policy["prompts"]["structural_anchors"]:
        scores[prompt["id"]] = 4.0
    for prompt in policy["prompts"]["class_prompts"][target]:
        scores[prompt["id"]] = 4.0
    for class_id in MODULE.TARGET_CLASSES:
        if class_id != target:
            for prompt in policy["prompts"]["class_prompts"][class_id]:
                scores[prompt["id"]] = 0.0
    return scores


class WholePlantGPUGateTests(unittest.TestCase):
    def test_policy_has_exact_non_authority(self) -> None:
        policy = load_policy()
        MODULE.validate_policy(policy)
        self.assertEqual(MODULE.AUTHORITY, policy["authority"])
        self.assertFalse(any(policy["authority"].values()))

    def test_clean_synthetic_whole_plant_can_only_be_machine_candidate(self) -> None:
        policy = load_policy()
        result = MODULE.decide_machine_outcome(
            score_fixture("grass_clump"), "grass_clump", policy
        )
        self.assertEqual("STRICT_POSITIVE_CANDIDATE_grass_clump", result["outcome"])
        self.assertEqual(
            "ALL_STRICT_STRUCTURE_AND_CLASS_GATES_PASSED", result["outcome_reasons"][0]
        )

    def test_one_individual_reject_is_not_diluted_by_other_rejects(self) -> None:
        policy = load_policy()
        scores = score_fixture("grass_clump")
        scores["anchor.whole_inside_frame"] = 0.0
        scores["reject.closeup.flower_or_seedhead"] = 3.0
        result = MODULE.decide_machine_outcome(scores, "grass_clump", policy)
        self.assertEqual("EXCLUDE", result["outcome"])
        self.assertEqual(
            "reject.closeup.flower_or_seedhead", result["max_individual_reject_prompt"]
        )
        self.assertIn("MAX_INDIVIDUAL_REJECT_DOMINATES_WHOLE_PLANT", result["outcome_reasons"])

    def test_hint_disagreement_fails_closed_to_hold(self) -> None:
        policy = load_policy()
        result = MODULE.decide_machine_outcome(
            score_fixture("low_shrub"), "grass_clump", policy
        )
        self.assertEqual("HOLD", result["outcome"])
        self.assertIn("ACQUISITION_HINT_AGREES", result["outcome_reasons"])

    def test_mature_tree_dominating_juvenile_is_excluded(self) -> None:
        policy = load_policy()
        scores = score_fixture("young_tree")
        scores["anchor.whole_inside_frame"] = 6.0
        scores["reject.tree.mature_large"] = 5.0
        scores["class.young_tree.juvenile_scale"] = 3.0
        result = MODULE.decide_machine_outcome(scores, "young_tree", policy)
        self.assertEqual("EXCLUDE", result["outcome"])
        self.assertIn("MATURE_TREE_REJECT_DOMINATES_JUVENILE", result["outcome_reasons"])

    def test_manifest_and_image_binding_are_frozen(self) -> None:
        policy = load_policy()
        raw = (DATASET_ROOT / "manifest.jsonl").read_bytes()
        self.assertEqual(
            policy["production_inputs"]["manifest_sha256"], hashlib.sha256(raw).hexdigest()
        )
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        binding = [
            {
                "class_id": row["class_id"],
                "pageid": row["pageid"],
                "filename": row["filename"],
                "download_sha256": row["download_sha256"],
            }
            for row in rows
        ]
        digest = hashlib.sha256(MODULE._canonical_bytes(binding)).hexdigest()
        self.assertEqual(policy["production_inputs"]["image_binding_sha256"], digest)
        self.assertEqual(90, len(rows))
        self.assertTrue(all(row["training_eligible"] is False for row in rows))
        self.assertTrue(all(row["print_eligible"] is False for row in rows))

    def test_hard_reject_contract_uses_every_individual_prompt(self) -> None:
        policy = load_policy()
        ids = MODULE._prompt_ids(policy)
        self.assertEqual(13, len(ids["rejects"]))
        self.assertEqual(13, len(set(ids["rejects"])))
        self.assertNotIn("family", " ".join(ids["rejects"]))

    def test_e2_contract_reuses_frozen_gate_and_binds_50_young_tree_images(self) -> None:
        gate = MODULE.WholePlantGPUGate(
            MODULE.GateConfig(
                dataset_root=E2_ROOT,
                manifest_path=E2_ROOT / "manifest.jsonl",
                summary_path=E2_ROOT / "summary.json",
                source_plan_path=E2_ROOT / "source_plan.json",
                output_dir=E2_ROOT / "review" / "ai_strict_gpu_gate_v1",
                input_contract_path=E2_CONTRACT,
            )
        )
        self.assertEqual(50, gate.production_inputs["expected_candidate_count"])
        self.assertEqual(
            {"grass_clump": 0, "low_shrub": 0, "young_tree": 50},
            gate.production_inputs["expected_class_counts"],
        )
        self.assertEqual(
            MODULE._sha256_bytes(MODULE._canonical_bytes(load_policy()["prompts"])),
            gate.prompt_set_sha256,
        )
        self.assertEqual(load_policy()["model"], gate.policy["model"])
        self.assertEqual(load_policy()["thresholds"], gate.policy["thresholds"])

        rows = [
            json.loads(line)
            for line in (E2_ROOT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        binding = [
            {
                "class_id": row["class_id"],
                "pageid": row["pageid"],
                "filename": row["filename"],
                "download_sha256": row["download_sha256"],
            }
            for row in rows
        ]
        self.assertEqual(
            gate.production_inputs["image_binding_sha256"],
            hashlib.sha256(MODULE._canonical_bytes(binding)).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
