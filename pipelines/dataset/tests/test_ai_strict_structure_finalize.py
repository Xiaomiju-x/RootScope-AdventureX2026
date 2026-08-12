from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "ai_strict_structure_finalize.py"
BASE_STRICT_POLICY = MODULE_PATH.with_name("ai_strict_structure_policy_v1.json")
BASE_ENSEMBLE_POLICY = MODULE_PATH.with_name("ai_siglip2_ensemble_policy_v1.json")
SPEC = importlib.util.spec_from_file_location("rootscope_ai_strict_structure_finalize", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.adventurex = root / "adventurex"
        self.review = self.adventurex / "dataset" / "review"
        self.ensemble_dir = self.review / "ai_ensemble_v1"
        self.metadata_dir = self.review / "ai_metadata_triage_v1"
        self.human = self.review / "human_decisions"
        self.output = self.review / "ai_final_labels_v1"
        self.results_path = self.ensemble_dir / "ai_siglip2_ensemble_results.jsonl"
        self.ensemble_receipt_path = self.ensemble_dir / "ai_siglip2_ensemble_receipt.json"
        self.metadata_path = self.metadata_dir / "metadata_risk_records.jsonl"
        self.metadata_receipt_path = self.metadata_dir / "receipt.json"
        self.ensemble_policy_path = self.adventurex / "tools" / "ai_siglip2_ensemble_policy_v1.json"
        self.strict_policy_path = self.adventurex / "tools" / "ai_strict_structure_policy_v1.json"
        self.model_provenance_path = self.adventurex / "models" / "SIGLIP2_MODEL_PROVENANCE.json"
        self.runtime_provenance_path = self.adventurex / "models" / "SIGLIP2_RUNTIME_PROVENANCE.json"
        self.weights_path = self.adventurex / "models" / "fixture_siglip2.npz"
        self.tokenizer_dir = self.adventurex / "models" / "fixture_tokenizer"
        self.human.mkdir(parents=True)
        (self.human / "decision_journal.jsonl").write_bytes(b"formal-human-sentinel\n")
        write_json(self.human / "human_review_receipt.json", {"sentinel": True})
        self.human_before = MODULE._artifact_root(self.human, "fixture human", allow_empty=True)["sha256"]
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.weights_path.write_bytes(b"fixture-big-vision-weights")
        self.tokenizer_dir.mkdir()
        (self.tokenizer_dir / "tokenizer.json").write_bytes(b'{"fixture":true}\n')
        (self.tokenizer_dir / "config.json").write_bytes(b'{"model_type":"siglip"}\n')

        self.ensemble_policy = json.loads(BASE_ENSEMBLE_POLICY.read_text(encoding="utf-8"))
        write_json(self.ensemble_policy_path, self.ensemble_policy)
        inference = self.ensemble_policy["inference"]
        self.prompt_spec = {
            "whole_quality_prompts": inference["whole_quality_prompts"],
            "class_prompts": inference["class_prompts"],
            "reject_family_prompts": inference["reject_family_prompts"],
        }
        self.prompt_set_sha = sha(canonical(self.prompt_spec))
        self.prompt_ids = [item["id"] for item in inference["whole_quality_prompts"]]
        for class_id in MODULE.TARGET_CLASSES:
            self.prompt_ids.extend(item["id"] for item in inference["class_prompts"][class_id])
        self.reject_families = list(inference["reject_family_prompts"])
        for records in inference["reject_family_prompts"].values():
            self.prompt_ids.extend(item["id"] for item in records)

        weights_artifact = MODULE._artifact_root(self.weights_path, "fixture weights")
        tokenizer_artifact = MODULE._artifact_root(self.tokenizer_dir, "fixture tokenizer")
        self.model_provenance = {
            "schema_version": "rootscope.local_model_provenance.v1",
            "weights": {
                "local_path": self.weights_path.relative_to(self.adventurex).as_posix(),
                "sha256": weights_artifact["entries"][0]["sha256"],
            },
            "tokenizer": {
                "local_path": self.tokenizer_dir.relative_to(self.adventurex).as_posix(),
            },
            "authority": {"human_review": False, "training_eligibility": False, "data_locked": False},
        }
        write_json(self.model_provenance_path, self.model_provenance)
        self.runtime_provenance = {
            "schema_version": "rootscope.siglip2_runtime_provenance.v1",
            "evidence_scope": "POST_HOC_LOCAL_REPLAY_ENVIRONMENT_BINDING_NOT_RETROACTIVE_INFERENCE_PROOF",
            "runtime": {
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "packages": {},
                "observed_platform": platform.platform(),
            },
            "bindings": {
                "model_provenance_sha256": sha(self.model_provenance_path.read_bytes()),
                "model_weights_raw_sha256": weights_artifact["entries"][0]["sha256"],
                "tokenizer_artifact_sha256": tokenizer_artifact["sha256"],
            },
            "authority": {
                "human_review": False,
                "visual_truth": False,
                "rights_approval": False,
                "training_eligibility": False,
                "data_locked": False,
            },
        }
        write_json(self.runtime_provenance_path, self.runtime_provenance)
        self.weights_artifact = weights_artifact
        self.tokenizer_artifact = tokenizer_artifact

        self.rows = [
            self.result(101, "grass_clump", "AUTO_TARGET", "grass_clump", "positive"),
            self.result(102, "unknown", "AUTO_UNKNOWN", "grass_clump", "unknown"),
            self.result(103, "low_shrub", "AUTO_TARGET", "low_shrub", "exclude"),
            self.result(104, "young_tree", "HOLD", "young_tree", "hold"),
        ]
        self.metadata = [
            self.metadata_row(1, self.rows[0]),
            self.metadata_row(2, self.rows[1]),
            self.metadata_row(
                3,
                self.rows[2],
                priority="HIGH_METADATA_RISK",
                flags=["detail_crop_signal"],
            ),
            self.metadata_row(4, self.rows[3]),
        ]
        self.strict_policy = json.loads(BASE_STRICT_POLICY.read_text(encoding="utf-8"))
        self.strict_policy["expected_candidate_count"] = len(self.rows)
        self.rebind()

    def score_map(self, profile: str, top1: str) -> dict[str, float]:
        scores = {prompt_id: -5.0 for prompt_id in self.prompt_ids}
        if profile == "positive":
            for prompt_id in self.strict_policy_prompt_ids("quality"):
                scores[prompt_id] = 5.0
            for prompt_id in self.ensemble_policy["inference"]["class_prompts"][top1]:
                scores[prompt_id["id"]] = 4.0
        elif profile == "unknown":
            for prompt_id in self.strict_policy_prompt_ids("quality"):
                scores[prompt_id] = -5.0
            scores["reject.no_target_object.empty_ground"] = 5.0
        elif profile == "exclude":
            for prompt_id in self.strict_policy_prompt_ids("quality"):
                scores[prompt_id] = 1.0
            for prompt in self.ensemble_policy["inference"]["class_prompts"][top1]:
                scores[prompt["id"]] = 2.0
            scores["reject.detail_crop.missing_base_or_crown"] = 3.0
            # The other detail prompt stays at -5: a family mean would dilute
            # the individual missing-base/crown blocker.
        elif profile == "hold":
            for prompt_id in self.strict_policy_prompt_ids("quality"):
                scores[prompt_id] = 3.0
            for prompt in self.ensemble_policy["inference"]["class_prompts"][top1]:
                scores[prompt["id"]] = 3.0
            scores["reject.detail_crop.missing_base_or_crown"] = 2.2
        return scores

    def strict_policy_prompt_ids(self, group: str) -> list[str]:
        if group != "quality":
            raise AssertionError(group)
        return [item["id"] for item in self.ensemble_policy["inference"]["whole_quality_prompts"]]

    def result(self, pageid: int, hint: str, decision: str, top1: str, profile: str) -> dict:
        scores = self.score_map(profile, top1)
        distributions = {
            "grass_clump": {
                "grass_clump": 0.99,
                "low_shrub": 0.006,
                "young_tree": 0.004,
            },
            "low_shrub": {
                "grass_clump": 0.006,
                "low_shrub": 0.99,
                "young_tree": 0.004,
            },
            "young_tree": {
                "grass_clump": 0.006,
                "low_shrub": 0.004,
                "young_tree": 0.99,
            },
        }
        probabilities = distributions[top1].copy()
        if profile == "unknown":
            probabilities = {"grass_clump": 0.40, "low_shrub": 0.35, "young_tree": 0.25}
        top_order = sorted(MODULE.TARGET_CLASSES, key=lambda key: (-probabilities[key], MODULE.TARGET_CLASSES.index(key)))
        reject_probability = {
            "positive": 0.01,
            "unknown": 0.99,
            "exclude": 0.10,
            "hold": 0.30,
        }[profile]
        dominant = "no_target_object" if profile == "unknown" else "detail_crop"
        reject_probabilities = {family: min(reject_probability, 0.005) for family in self.reject_families}
        reject_probabilities[dominant] = reject_probability
        reject_scores = {family: -5.0 for family in self.reject_families}
        reject_scores[dominant] = {
            "positive": -5.0,
            "unknown": 5.0,
            "exclude": -1.0,
            "hold": 0.0,
        }[profile]
        candidate_sha = f"{pageid:064x}"
        return {
            "schema_version": "rootscope.ai_siglip2_ensemble_result.v1",
            "mode": "FIXTURE",
            "asset": f"wikimedia:{pageid}@sha256:{candidate_sha}",
            "pageid": pageid,
            "candidate_sha256": candidate_sha,
            "image_path": f"images/{hint}/fixture_{pageid}.jpg",
            "acquisition_class_hint": hint,
            "model_id": "fixture/siglip2",
            "model_artifact_sha256": self.weights_artifact["sha256"],
            "prompt_set_sha256": self.prompt_set_sha,
            "calibration_sha256": "a" * 64,
            "calibration_status": "FIXTURE",
            "prompt_scores": scores,
            "quality_score": 0.0,
            "reject_family_scores": reject_scores,
            "dominant_reject_family": dominant,
            "dominant_reject_score": reject_scores[dominant],
            "admissibility_probability": 1.0 - reject_probability,
            "reject_family_probabilities": reject_probabilities,
            "reject_probability": reject_probability,
            "class_scores": {class_id: 0.0 for class_id in MODULE.TARGET_CLASSES},
            "class_probabilities": probabilities,
            "top1_class": top_order[0],
            "top1_probability": probabilities[top_order[0]],
            "top2_class": top_order[1],
            "top2_probability": probabilities[top_order[1]],
            "top1_top2_margin": probabilities[top_order[0]] - probabilities[top_order[1]],
            "acquisition_hint_agrees": hint == top_order[0],
            "decision": decision,
            "suggested_class": "unknown" if decision == "AUTO_UNKNOWN" else top_order[0],
            "decision_reasons": ["FIXTURE"],
            "authority": {
                "human_review": False,
                "dataset_manifest_write": False,
                "training_eligibility": False,
                "split_assignment": False,
                "print_eligibility": False,
                "data_locked": False,
            },
            "explicit_non_claims": ["HUMAN_REVIEWED", "TRAIN_READY", "DATA_LOCKED"],
        }

    def metadata_row(
        self,
        index: int,
        result: dict,
        priority: str = "NO_OBVIOUS_METADATA_RISK_SIGNAL",
        flags: list[str] | None = None,
    ) -> dict:
        return {
            "schema_version": "rootscope.ai_metadata_risk_triage.v1",
            "queue_index": index,
            "asset": result["asset"],
            "source_group": f"commons:{result['pageid']}",
            "local_path": result["image_path"],
            "acquisition_metadata": {
                "class_hint": result["acquisition_class_hint"],
                "title": f"Fixture {result['pageid']}",
            },
            "metadata_only": True,
            "visual_truth_established": False,
            "risk_priority": priority,
            "risk_flags": [
                {"id": flag, "severity": "HIGH", "evidence": {}, "rationale": "fixture"}
                for flag in (flags or [])
            ],
            "support_signals": [],
            "context_flags": [],
        }

    def rebind(self) -> None:
        write_jsonl(self.results_path, self.rows)
        write_jsonl(self.metadata_path, self.metadata)
        queue_sha = "f" * 64
        write_json(
            self.ensemble_receipt_path,
            {
                "schema_version": "rootscope.ai_siglip2_ensemble_receipt.v1",
                "candidate_count": len(self.rows),
                "human_review_files_touched": False,
                "dataset_manifest_written": False,
                "outputs": {self.results_path.name: sha(self.results_path.read_bytes())},
                "input_roots": {"candidate_review_queue_sha256": queue_sha},
                "model": {"artifact": MODULE._public_artifact(self.weights_artifact)},
                "prompt_set_sha256": self.prompt_set_sha,
                "authority": {
                    "human_review": False,
                    "dataset_manifest_write": False,
                    "training_eligibility": False,
                    "split_assignment": False,
                    "print_eligibility": False,
                    "data_locked": False,
                },
            },
        )
        write_json(
            self.metadata_receipt_path,
            {
                "schema_version": "rootscope.ai_metadata_risk_triage.v1",
                "rows": len(self.metadata),
                "queue_sha256": queue_sha,
                "artifacts_sha256": {self.metadata_path.name: sha(self.metadata_path.read_bytes())},
                "authority": {
                    "visual_truth": False,
                    "human_review": False,
                    "rights_approval": False,
                    "dataset_manifest_write": False,
                    "training_eligibility": False,
                    "print_eligibility": False,
                    "split_assignment": False,
                    "data_locked": False,
                },
            },
        )
        roots = {
            "ensemble_results_sha256": sha(self.results_path.read_bytes()),
            "ensemble_receipt_sha256": sha(self.ensemble_receipt_path.read_bytes()),
            "metadata_records_sha256": sha(self.metadata_path.read_bytes()),
            "metadata_receipt_sha256": sha(self.metadata_receipt_path.read_bytes()),
            "ensemble_policy_sha256": sha(self.ensemble_policy_path.read_bytes()),
            "model_provenance_sha256": sha(self.model_provenance_path.read_bytes()),
            "runtime_provenance_sha256": sha(self.runtime_provenance_path.read_bytes()),
            "tokenizer_artifact_sha256": self.tokenizer_artifact["sha256"],
            "model_weights_raw_sha256": self.weights_artifact["entries"][0]["sha256"],
        }
        self.strict_policy["production_input_roots"] = roots
        write_json(self.strict_policy_path, self.strict_policy)

    def config(self, **updates) -> MODULE.FinalizerConfig:
        values = {
            "ensemble_results": self.results_path,
            "ensemble_receipt": self.ensemble_receipt_path,
            "metadata_records": self.metadata_path,
            "metadata_receipt": self.metadata_receipt_path,
            "ensemble_policy": self.ensemble_policy_path,
            "policy": self.strict_policy_path,
            "model_provenance": self.model_provenance_path,
            "runtime_provenance": self.runtime_provenance_path,
            "tokenizer_dir": self.tokenizer_dir,
            "human_decisions": self.human,
            "output_dir": self.output,
            "adventurex_root": self.adventurex,
            "fixture_mode": True,
        }
        values.update(updates)
        return MODULE.FinalizerConfig(**values)


class StrictFinalizerTests(unittest.TestCase):
    def make_fixture(self) -> tuple[tempfile.TemporaryDirectory, Fixture]:
        temporary = tempfile.TemporaryDirectory()
        return temporary, Fixture(Path(temporary.name))

    def test_preflight_is_read_only_and_binds_artifacts(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        result = MODULE.StrictStructureFinalizer(fixture.config()).preflight()
        self.assertEqual(result["status"], "PASS_READ_ONLY_INPUTS_BOUND_OUTPUT_NOT_WRITTEN")
        self.assertFalse(fixture.output.exists())
        self.assertEqual(result["tokenizer_artifact"]["sha256"], fixture.tokenizer_artifact["sha256"])
        self.assertEqual(result["human_decisions_root_sha256"], fixture.human_before)

    def test_run_classifies_all_four_dispositions(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        receipt = MODULE.StrictStructureFinalizer(fixture.config()).run()["receipt"]
        labels = read_jsonl(fixture.output / "strict_structure_labels.jsonl")
        self.assertEqual(
            [row["final_label"] for row in labels],
            [
                "POSITIVE_CANDIDATE_GRASS_CLUMP",
                "UNKNOWN_CANDIDATE",
                "EXCLUDE_NONCONFORMING",
                "HOLD_INSUFFICIENT_EVIDENCE",
            ],
        )
        self.assertEqual(receipt["counts"]["positive_candidates"], 1)
        self.assertEqual(receipt["counts"]["unknown_candidates"], 1)
        self.assertFalse(receipt["original_ensemble_runtime_proven"])

    def test_individual_missing_base_score_is_not_diluted_by_family_mean(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        MODULE.StrictStructureFinalizer(fixture.config()).run()
        excluded = read_jsonl(fixture.output / "strict_structure_labels.jsonl")[2]
        self.assertEqual(excluded["final_label"], "EXCLUDE_NONCONFORMING")
        self.assertEqual(
            excluded["strict_metrics"]["max_individual_reject_prompt"],
            "reject.detail_crop.missing_base_or_crown",
        )
        self.assertLess(excluded["strict_metrics"]["complete_vs_max_individual_reject_margin"], 0)
        self.assertIn("MISSING_BASE_OR_CROWN_NOT_CLEARED", excluded["positive_gate_failures"])

    def test_all_output_authority_and_eligibility_fields_remain_false(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        MODULE.StrictStructureFinalizer(fixture.config()).run()
        for row in read_jsonl(fixture.output / "strict_structure_labels.jsonl"):
            self.assertTrue(all(value is False for value in row["authority"].values()))
            self.assertFalse(row["human_reviewed"])
            self.assertFalse(row["training_eligible"])
            self.assertFalse(row["print_eligible"])
            self.assertEqual(row["split"], "UNASSIGNED_DO_NOT_TRAIN")

    def test_human_decisions_guard_is_unchanged(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        receipt = MODULE.StrictStructureFinalizer(fixture.config()).run()["receipt"]
        after = MODULE._artifact_root(fixture.human, "fixture human", allow_empty=True)["sha256"]
        self.assertEqual(after, fixture.human_before)
        self.assertEqual(receipt["human_decisions_root_before_sha256"], after)
        self.assertEqual(receipt["human_decisions_root_after_sha256"], after)
        self.assertFalse(receipt["human_review_files_touched"])

    def test_output_is_immutable(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        MODULE.StrictStructureFinalizer(fixture.config()).run()
        with self.assertRaisesRegex(MODULE.FinalizerError, "already exists"):
            MODULE.StrictStructureFinalizer(fixture.config()).run()

    def test_result_payload_tamper_fails_closed(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        fixture.results_path.write_bytes(fixture.results_path.read_bytes() + b" ")
        with self.assertRaisesRegex(MODULE.FinalizerError, "strict UTF-8 JSON|frozen input root mismatch"):
            MODULE.StrictStructureFinalizer(fixture.config()).preflight()

    def test_tokenizer_tamper_fails_closed(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (fixture.tokenizer_dir / "tokenizer.json").write_bytes(b"tampered")
        with self.assertRaisesRegex(MODULE.FinalizerError, "tokenizer artifact root mismatch"):
            MODULE.StrictStructureFinalizer(fixture.config()).preflight()

    def test_model_weights_tamper_fails_closed(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        fixture.weights_path.write_bytes(b"tampered weights")
        with self.assertRaisesRegex(MODULE.FinalizerError, "model weights raw SHA-256 mismatch"):
            MODULE.StrictStructureFinalizer(fixture.config()).preflight()

    def test_runtime_manifest_mismatch_fails_closed(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        runtime = read_json(fixture.runtime_provenance_path)
        runtime["runtime"]["python_version"] = "0.0.0"
        write_json(fixture.runtime_provenance_path, runtime)
        fixture.strict_policy["production_input_roots"]["runtime_provenance_sha256"] = sha(
            fixture.runtime_provenance_path.read_bytes()
        )
        write_json(fixture.strict_policy_path, fixture.strict_policy)
        with self.assertRaisesRegex(MODULE.FinalizerError, "current environment differs"):
            MODULE.StrictStructureFinalizer(fixture.config()).preflight()

    def test_authority_escalation_in_policy_is_rejected(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        fixture.strict_policy["authority"]["training_eligibility"] = True
        write_json(fixture.strict_policy_path, fixture.strict_policy)
        with self.assertRaisesRegex(MODULE.FinalizerError, "authority"):
            MODULE.StrictStructureFinalizer(fixture.config()).preflight()

    def test_missing_reject_prompt_fails_strict_schema(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        del fixture.rows[0]["prompt_scores"]["reject.no_target_object.animal_or_human"]
        fixture.rebind()
        with self.assertRaisesRegex(MODULE.FinalizerError, "prompt scores"):
            MODULE.StrictStructureFinalizer(fixture.config()).preflight()

    def test_duplicate_asset_fails_closed(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        fixture.rows[1]["asset"] = fixture.rows[0]["asset"]
        fixture.metadata[1]["asset"] = fixture.metadata[0]["asset"]
        fixture.rebind()
        with self.assertRaisesRegex(MODULE.FinalizerError, "missing/duplicate"):
            MODULE.StrictStructureFinalizer(fixture.config()).preflight()

    def test_output_may_not_target_human_decisions(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(MODULE.FinalizerError, "human_decisions"):
            MODULE.StrictStructureFinalizer(
                fixture.config(output_dir=fixture.human / "ai_final_labels_v1")
            )

    def test_unknown_with_high_metadata_risk_is_not_promoted(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        fixture.metadata[1]["risk_priority"] = "HIGH_METADATA_RISK"
        fixture.metadata[1]["risk_flags"] = [
            {"id": "map_text_illustration_signal", "severity": "HIGH", "evidence": {}, "rationale": "fixture"}
        ]
        fixture.rebind()
        MODULE.StrictStructureFinalizer(fixture.config()).run()
        label = read_jsonl(fixture.output / "strict_structure_labels.jsonl")[1]
        self.assertNotEqual(label["final_label"], "UNKNOWN_CANDIDATE")
        self.assertEqual(label["final_label"], "EXCLUDE_NONCONFORMING")


if __name__ == "__main__":
    unittest.main()
