from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any


MODULE_PATH = Path(__file__).resolve().parents[1] / "audit_rootscope_experimental_run.py"
SPEC = importlib.util.spec_from_file_location("rootscope_independent_run_auditor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)
PIPELINE_PATH = MODULE_PATH.with_name("rootscope_machine_curated_pipeline.py")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 224, 224])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4])
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [3, 4],
        [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4, 0.05, 0.1, 0.15, 0.2],
    )
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [4], [0.0, 0.0, 0.0, 0.0])
    nodes = [
        helper.make_node(
            "AveragePool",
            ["image"],
            ["pooled"],
            name="fixed_avgpool_7x7",
            kernel_shape=[7, 7],
            strides=[1, 1],
        ),
        helper.make_node("ReduceMean", ["pooled"], ["features"], axes=[2, 3], keepdims=0),
        helper.make_node("Gemm", ["features", "weight", "bias"], ["logits"]),
    ]
    graph = helper.make_graph(nodes, "synthetic_fixed_pool_model", [image], [logits], [weight, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def make_consistency() -> dict[str, Any]:
    probes = [
        {
            "name": name,
            "input_shape": [1, 3, 224, 224],
            "max_absolute_error": 1e-6,
            "mean_absolute_error": 5e-7,
            "passed": True,
        }
        for name in sorted(AUDITOR.REQUIRED_PROBES)
    ]
    return {
        "schema_version": "rootscope.torch_onnx_consistency.v2",
        "passed": True,
        "input_shape": [1, 3, 224, 224],
        "class_order": list(AUDITOR.CLASS_NAMES),
        "opset": 11,
        "probe_count": len(probes),
        "probes": probes,
        "natural_validation_classes_probed": list(AUDITOR.CLASS_NAMES),
        "natural_validation_classes_missing": [],
        "synthetic_zero_probed": True,
        "synthetic_ramp_probed": True,
        "max_absolute_error": 1e-6,
        "mean_absolute_error": 5e-7,
        "tolerance": 1e-4,
        "onnxruntime_providers": ["CPUExecutionProvider"],
    }


def make_calibration() -> dict[str, Any]:
    evidence = {
        name: {
            "predicted_validation_support": 1,
            "global_threshold_accepted_count": 0,
            "global_threshold_correct_count": 0,
            "global_threshold_accepted_accuracy": None,
            "wilson_lower_bound": None,
            "minimum_accepted_required": 2,
            "target_lower_bound": 0.8,
            "acceptance_enabled": False,
            "force_reject_reasons": [
                "GLOBAL_TARGET_NOT_MET",
                "INSUFFICIENT_ACCEPTED_SUPPORT",
                "WILSON_LOWER_BOUND_BELOW_TARGET",
            ],
        }
        for name in AUDITOR.CLASS_NAMES
    }
    return {
        "method": "joint_confidence_and_top1_top2_margin_validation_grid",
        "calibration_domain": AUDITOR.NATURAL_VAL_DOMAIN,
        "temperature": 1.0,
        "target_accepted_accuracy": 0.8,
        "minimum_accepted": 2,
        "per_predicted_class_minimum_accepted": 2,
        "wilson_z": 1.96,
        "confidence_threshold": 1.0,
        "margin_threshold": 1.0,
        "global_validation_accepted_count": 0,
        "global_validation_coverage": 0.0,
        "global_validation_accepted_accuracy": None,
        "validation_accepted_count": 0,
        "validation_coverage": 0.0,
        "validation_accepted_accuracy": None,
        "per_predicted_class_evidence": evidence,
        "mode": "FAIL_CLOSED_REJECT_ALL_TARGET_NOT_MET",
        "decision_rule": AUDITOR.DECISION_RULE,
        "status": "MACHINE_CURATED_EXPERIMENTAL_CALIBRATION_NOT_FORMALLY_QUALIFIED",
    }


def make_metrics(seed: int, score: float) -> dict[str, Any]:
    rejection_base = {
        "sample_count": 4,
        "accepted_count": 0,
        "rejected_count": 4,
        "coverage": 0.0,
        "accepted_accuracy": None,
        "per_predicted_class_gate_applied": True,
    }
    return {
        "schema_version": "rootscope.machine_curated_experimental_metrics.v1",
        "status": AUDITOR.RUN_STATUS,
        "seed": seed,
        "best_epoch": 1,
        "natural_validation_rejection": {
            **rejection_base,
            "domain": AUDITOR.NATURAL_VAL_DOMAIN,
            "thresholds_locked_from": AUDITOR.NATURAL_VAL_DOMAIN,
            "thresholds_optimized_on_this_domain": True,
        },
        "digital_print_source_holdout_rejection": {
            **rejection_base,
            "domain": AUDITOR.PRINT_DOMAIN,
            "thresholds_locked_from": AUDITOR.NATURAL_VAL_DOMAIN,
            "thresholds_optimized_on_this_domain": False,
        },
        "digital_print_source_holdout_is_uvc_recapture": False,
        "digital_print_source_holdout_claim": "DIGITAL_SOURCE_EVALUATION_ONLY_NOT_REAL_PRINT_DOMAIN_EVIDENCE",
        "history": [
            {
                "epoch": 1,
                "train_cross_entropy": 1.2,
                "train_samples_seen": 4,
                "natural_validation": {
                    "domain": AUDITOR.NATURAL_VAL_DOMAIN,
                    "balanced_accuracy_present_classes": score,
                    "cross_entropy": 1.0,
                },
            }
        ],
    }


def required_model_card() -> str:
    return """# Synthetic RootScope experimental model

It is not formal A1 data, not human-reviewed truth, not rights-approved,
not print-eligible, not data-locked, and not formally training-eligible.
This model is not qualified for deployment or irrigation decisions.
The digital source holdout is not evidence from a physical print.
It never tuned weights, checkpoint selection, temperature, or rejection thresholds.
The artifact has `model_candidate=false`; numerical consistency is not BPU conversion,
X5 runtime readiness, or physical-domain accuracy.
"""


def finalize_run(run_root: Path) -> None:
    receipt_path = run_root / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    pre_receipt = {
        path.relative_to(run_root).as_posix(): AUDITOR.sha256_file(path)
        for path in sorted(run_root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
        and path.relative_to(run_root).as_posix()
        not in {"run_receipt.json", "MODEL_CARD.md", "SHA256SUMS"}
    }
    receipt["artifact_hashes_before_receipt"] = pre_receipt
    write_json(receipt_path, receipt)
    files = {
        path.relative_to(run_root).as_posix(): AUDITOR.sha256_file(path)
        for path in sorted(run_root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.relative_to(run_root).as_posix() != "SHA256SUMS"
    }
    (run_root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in files.items()),
        encoding="utf-8",
    )


class SyntheticRun:
    def __init__(self, root: Path) -> None:
        self.workspace = root
        self.run_root = root / "output" / "rootscope_machine_curated_experimental_runs" / "synthetic_v3"
        self.pack_root = root / Path(*Path(AUDITOR.PACK_PATH).parts)
        self.run_root.mkdir(parents=True)
        pipeline = root / "tools" / "training" / "rootscope_machine_curated_pipeline.py"
        pipeline.parent.mkdir(parents=True)
        shutil.copyfile(PIPELINE_PATH, pipeline)

        formal_root = root / "datasets" / "formal_human_decisions"
        formal_root.mkdir(parents=True)
        (formal_root / "decision_journal.jsonl").write_bytes(b"")
        formal_tree = AUDITOR.tree_sha256(formal_root)
        formal_journal = AUDITOR.sha256_file(formal_root / "decision_journal.jsonl")
        v2_root = root / "datasets" / "rootscope_machine_curated_provisional_v2"
        v2_root.mkdir(parents=True)
        write_json(
            v2_root / "receipt.json",
            {
                "formal_human_decisions": {
                    "path": "datasets/formal_human_decisions",
                    "tree_sha256_before": formal_tree,
                    "tree_sha256_after": formal_tree,
                    "decision_journal_sha256_before": formal_journal,
                    "decision_journal_sha256_after": formal_journal,
                    "unchanged": True,
                }
            },
        )

        manifest_rows: list[dict[str, Any]] = []
        for role_slug, role in (("train", AUDITOR.TRAIN_ROLE), ("val", AUDITOR.VAL_ROLE), ("print", AUDITOR.PRINT_ROLE)):
            for class_id in AUDITOR.CLASS_NAMES:
                relative = f"images/{class_id}/{role_slug}.jpg"
                image = self.pack_root / Path(*Path(relative).parts)
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"synthetic-{role_slug}-{class_id}".encode("ascii"))
                manifest_rows.append(
                    {
                        "class_id": class_id,
                        "experimental_split_suggestion": role,
                        "filename": relative,
                    }
                )
        write_jsonl(self.pack_root / "manifest.jsonl", manifest_rows)
        visual = {
            "required_for_pack": True,
            "bound": True,
            "human_reviewed": False,
            "non_smoke_training_evidence_eligible": True,
        }
        protected_snapshot = {
            "formal_human_decisions_tree_sha256": formal_tree,
            "formal_decision_journal_sha256": formal_journal,
        }
        write_json(
            self.pack_root / "receipt.json",
            {
                "schema_version": AUDITOR.PACK_SCHEMA,
                "status": AUDITOR.PACK_STATUS,
                "formal_a1_dataset": False,
                "human_reviewed": False,
                "rights_approved": False,
                "training_eligible": False,
                "print_eligible": False,
                "data_locked": False,
                "manifest_sha256": AUDITOR.sha256_file(self.pack_root / "manifest.jsonl"),
                "frozen_v2": {
                    "path": "datasets/rootscope_machine_curated_provisional_v2",
                    "receipt_sha256": AUDITOR.sha256_file(v2_root / "receipt.json"),
                    "unchanged": True,
                },
                "protected_inputs": {
                    "before": protected_snapshot,
                    "after": protected_snapshot,
                    "unchanged": True,
                },
            },
        )
        snapshot = {
            "pack_tree_sha256": AUDITOR.tree_sha256(self.pack_root),
            "manifest_sha256": AUDITOR.sha256_file(self.pack_root / "manifest.jsonl"),
            "receipt_sha256": AUDITOR.sha256_file(self.pack_root / "receipt.json"),
            "formal_human_decisions_tree_sha256": formal_tree,
            "formal_human_decision_journal_sha256": formal_journal,
        }
        input_audit = {
            "schema_version": "rootscope.machine_curated_training_input_audit.v1",
            "status": AUDITOR.PACK_STATUS,
            "class_order": list(AUDITOR.CLASS_NAMES),
            "formal_a1_dataset": False,
            "human_reviewed": False,
            "training_eligible": False,
            "data_locked": False,
            "long_training_coverage_gate_passed": True,
            "print_evaluation_domain": AUDITOR.PRINT_DOMAIN,
            "print_evaluation_is_uvc_recapture": False,
            "machine_visual_review_evidence": visual,
            "immutable_snapshot": snapshot,
        }
        write_json(self.run_root / "input_audit.json", input_audit)

        seed_results: list[dict[str, Any]] = []
        import torch

        for seed, score in ((17, 0.3), (29, 0.4), (43, 0.5)):
            seed_dir = self.run_root / f"seed_{seed:05d}"
            seed_dir.mkdir()
            calibration = make_calibration()
            metrics = make_metrics(seed, score)
            provenance = {
                "architecture": "torchvision.resnet18",
                "input_shape": [1, 3, 224, 224],
                "class_order": list(AUDITOR.CLASS_NAMES),
                "adaptive_pooling": False,
                "average_pool": {"kernel_size": [7, 7], "stride": [1, 1]},
            }
            consistency = make_consistency()
            write_json(seed_dir / "calibration.json", calibration)
            write_json(seed_dir / "metrics.json", metrics)
            write_json(seed_dir / "model_provenance.json", provenance)
            write_json(seed_dir / "onnx_consistency.json", consistency)
            make_onnx(seed_dir / AUDITOR.ONNX_NAME)
            torch.save(
                {
                    "schema_version": "rootscope.resnet18_experimental_checkpoint.v1",
                    "status": AUDITOR.RUN_STATUS,
                    "seed": seed,
                    "epoch": 1,
                    "class_order": list(AUDITOR.CLASS_NAMES),
                    "input_shape": [1, 3, 224, 224],
                    "architecture": "torchvision.resnet18_fixed_avgpool7x7",
                    "input_pack_manifest_sha256": snapshot["manifest_sha256"],
                    "model_state_dict": {"synthetic": torch.zeros(1)},
                },
                seed_dir / "best_checkpoint.pt",
            )
            seed_results.append(
                {
                    "seed": seed,
                    "best_epoch": 1,
                    "selection_key": [score, -1.0],
                    "calibration": calibration,
                    "metrics": metrics,
                    "model_provenance": provenance,
                    "onnx_consistency": consistency,
                    "artifacts": {
                        "checkpoint": f"seed_{seed:05d}/best_checkpoint.pt",
                        "onnx": f"seed_{seed:05d}/{AUDITOR.ONNX_NAME}",
                    },
                }
            )
        authority = {key: False for key in AUDITOR.REQUIRED_AUTHORITY_KEYS}
        receipt = {
            "schema_version": AUDITOR.RUN_SCHEMA,
            "status": AUDITOR.RUN_STATUS,
            "run_id": "synthetic_v3",
            "smoke_only": False,
            "formal_a1_dataset": False,
            "human_reviewed": False,
            "rights_approved": False,
            "rights": False,
            "training_eligible": False,
            "print_eligible": False,
            "data_locked": False,
            "model_qualified": False,
            "model_candidate": False,
            "experimental_model_candidate": True,
            "x5_ready": False,
            "bpu_compiled": False,
            "physical_print_tested": False,
            "uvc_recapture_evaluated": False,
            "execution_authority": False,
            "authority": authority,
            "ack_machine_curated_experimental_only": True,
            "long_training_coverage_gate_passed": True,
            "machine_visual_review_evidence": visual,
            "input_pack_status": AUDITOR.PACK_STATUS,
            "input_pack": AUDITOR.PACK_PATH,
            "input_audit_sha256": AUDITOR.sha256_file(self.run_root / "input_audit.json"),
            "training_pipeline_sha256": AUDITOR.sha256_file(pipeline),
            "class_order": list(AUDITOR.CLASS_NAMES),
            "architecture": "torchvision.resnet18_fixed_avgpool7x7",
            "input_shape": [1, 3, 224, 224],
            "onnx_opset": 11,
            "seeds": [17, 29, 43],
            "seed_results": seed_results,
            "selected_seed": copy.deepcopy(seed_results[-1]),
            "digital_print_holdout_domain": AUDITOR.PRINT_DOMAIN,
            "digital_print_holdout_is_uvc_recapture": False,
            "input_and_formal_authority_unchanged": {
                "unchanged": True,
                "before": snapshot,
                "after": snapshot,
            },
            "explicit_non_claims": sorted(AUDITOR.REQUIRED_NON_CLAIMS),
        }
        write_json(self.run_root / "run_receipt.json", receipt)
        (self.run_root / "MODEL_CARD.md").write_text(required_model_card(), encoding="utf-8")
        finalize_run(self.run_root)

    def receipt(self) -> dict[str, Any]:
        return json.loads((self.run_root / "run_receipt.json").read_text(encoding="utf-8"))

    def write_receipt(self, receipt: dict[str, Any]) -> None:
        write_json(self.run_root / "run_receipt.json", receipt)
        finalize_run(self.run_root)


def fake_replay(**kwargs: Any) -> dict[str, Any]:
    return {
        "passed": True,
        "provider_requested": "CPUExecutionProvider",
        "providers_actual": ["CPUExecutionProvider"],
        "probe_count": len(AUDITOR.REQUIRED_PROBES),
        "probe_names": sorted(AUDITOR.REQUIRED_PROBES),
        "max_absolute_error": 1e-6,
        "tolerance": 1e-4,
    }


class IndependentRunAuditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = SyntheticRun(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def audit(self) -> dict[str, Any]:
        return AUDITOR.audit_run(
            self.fixture.workspace,
            self.fixture.run_root,
            replay_runner=fake_replay,
        )

    def test_valid_synthetic_three_seed_run_passes(self) -> None:
        report = self.audit()
        self.assertEqual("PASS", report["status"])
        self.assertEqual(3, report["seed_count"])
        self.assertEqual(43, report["selected_seed"])
        self.assertTrue(report["full_sha256_coverage"])
        self.assertFalse(report["model_candidate"])
        self.assertTrue(report["experimental_model_candidate"])

    def test_auditor_does_not_import_training_pipeline(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotRegex(
            source,
            r"(?m)^\s*(?:from\s+rootscope_machine_curated_pipeline\s+import|import\s+rootscope_machine_curated_pipeline)",
        )

    def test_uncovered_extra_file_fails(self) -> None:
        (self.fixture.run_root / "uncovered.txt").write_text("not hashed", encoding="utf-8")
        with self.assertRaisesRegex(AUDITOR.AuditError, "not full coverage"):
            self.audit()

    def test_permission_flip_fails_even_when_rehashed(self) -> None:
        receipt = self.fixture.receipt()
        receipt["x5_ready"] = True
        self.fixture.write_receipt(receipt)
        with self.assertRaisesRegex(AUDITOR.AuditError, "x5_ready"):
            self.audit()

    def test_selected_seed_must_be_validation_winner(self) -> None:
        receipt = self.fixture.receipt()
        receipt["selected_seed"] = copy.deepcopy(receipt["seed_results"][0])
        self.fixture.write_receipt(receipt)
        with self.assertRaisesRegex(AUDITOR.AuditError, "selection-key winner"):
            self.audit()

    def test_missing_cpu_probe_fails_even_when_rehashed_and_rebound(self) -> None:
        receipt = self.fixture.receipt()
        seed_result = receipt["seed_results"][0]
        consistency = seed_result["onnx_consistency"]
        consistency["probes"] = consistency["probes"][:-1]
        consistency["probe_count"] -= 1
        path = self.fixture.run_root / "seed_00017" / "onnx_consistency.json"
        write_json(path, consistency)
        if receipt["selected_seed"]["seed"] == 17:
            receipt["selected_seed"]["onnx_consistency"] = copy.deepcopy(consistency)
        write_json(self.fixture.run_root / "run_receipt.json", receipt)
        finalize_run(self.fixture.run_root)
        with self.assertRaisesRegex(AUDITOR.AuditError, "exact four-class plus zero/ramp"):
            self.audit()

    def test_print_domain_cannot_enter_calibration(self) -> None:
        receipt = self.fixture.receipt()
        seed_result = receipt["seed_results"][0]
        calibration = seed_result["calibration"]
        calibration["calibration_domain"] = AUDITOR.PRINT_DOMAIN
        write_json(self.fixture.run_root / "seed_00017" / "calibration.json", calibration)
        write_json(self.fixture.run_root / "run_receipt.json", receipt)
        finalize_run(self.fixture.run_root)
        with self.assertRaisesRegex(AUDITOR.AuditError, "validation-only"):
            self.audit()

    def test_incomplete_per_class_calibration_fails(self) -> None:
        receipt = self.fixture.receipt()
        seed_result = receipt["seed_results"][1]
        calibration = seed_result["calibration"]
        del calibration["per_predicted_class_evidence"]["young_tree"]
        write_json(self.fixture.run_root / "seed_00029" / "calibration.json", calibration)
        write_json(self.fixture.run_root / "run_receipt.json", receipt)
        finalize_run(self.fixture.run_root)
        with self.assertRaisesRegex(AUDITOR.AuditError, "evidence is incomplete"):
            self.audit()

    def test_model_card_overclaim_fails(self) -> None:
        with (self.fixture.run_root / "MODEL_CARD.md").open("a", encoding="utf-8") as handle:
            handle.write("\nThis model is production-ready and x5_ready=true.\n")
        finalize_run(self.fixture.run_root)
        with self.assertRaisesRegex(AUDITOR.AuditError, "forbidden qualification"):
            self.audit()

    def test_current_v3_tree_mutation_fails(self) -> None:
        with (self.fixture.pack_root / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(AUDITOR.AuditError, "receipt does not bind its manifest"):
            self.audit()

    def test_formal_decision_journal_mutation_fails(self) -> None:
        journal = self.fixture.workspace / "datasets" / "formal_human_decisions" / "decision_journal.jsonl"
        journal.write_text('{"unauthorized":true}\n', encoding="utf-8")
        with self.assertRaisesRegex(AUDITOR.AuditError, "formal human-decision evidence changed"):
            self.audit()

    def test_dynamic_or_wrong_shape_onnx_fails(self) -> None:
        import onnx

        path = self.fixture.run_root / "seed_00017" / AUDITOR.ONNX_NAME
        model = onnx.load(path)
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
        model.graph.input[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
        onnx.save(model, path)
        finalize_run(self.fixture.run_root)
        with self.assertRaisesRegex(AUDITOR.AuditError, "ONNX input must be static"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
