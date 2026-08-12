from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


TRAINING_TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING_TOOLS / "rootscope_machine_curated_pipeline.py"
SPEC = importlib.util.spec_from_file_location("rootscope_machine_curated_pipeline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RootScopeMachineCuratedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = TRAINING_TOOLS.parents[1]
        cls.pack_v1 = cls.workspace / "datasets" / MODULE.PACK_NAME_V1
        cls.pack_v2 = cls.workspace / "datasets" / MODULE.PACK_NAME_V2
        cls.pack_v3 = cls.workspace / "datasets" / MODULE.PACK_NAME_V3

    def test_class_and_export_contract_is_frozen(self) -> None:
        self.assertEqual(
            ("grass_clump", "low_shrub", "young_tree", "unknown"),
            MODULE.CLASS_NAMES,
        )
        self.assertEqual(224, MODULE.INPUT_SIZE)
        self.assertEqual(11, MODULE.ONNX_OPSET)
        self.assertEqual(
            "DIGITAL_PRINT_SOURCE_HOLDOUT_NOT_UVC_RECAPTURE",
            MODULE.PRINT_EVAL_DOMAIN,
        )

    def test_ack_is_required_before_any_run(self) -> None:
        parser = MODULE.build_parser()
        args = parser.parse_args(
            [
                "--smoke",
                "--epochs",
                "1",
                "--seeds",
                "17",
                "--max-train-batches",
                "1",
            ]
        )
        with self.assertRaisesRegex(MODULE.GateError, "ack-machine-curated"):
            MODULE.run(args)

    def test_non_smoke_visual_evidence_is_receipt_bound_not_a_bare_cli_ack(self) -> None:
        audited = MODULE.audit_pack(self.workspace, self.pack_v3)
        evidence = audited.audit["machine_visual_review_evidence"]
        self.assertTrue(evidence["bound"])
        self.assertTrue(evidence["non_smoke_training_evidence_eligible"])
        self.assertEqual("E4_SELECTED_ONLY", evidence["dual_machine_review_scope"])
        receipt = dict(audited.receipt)
        receipt["machine_visual_review_evidence_sha256"] = "0" * 64
        with self.assertRaisesRegex(MODULE.GateError, "evidence_sha256 mismatch"):
            MODULE.validate_machine_visual_review_evidence(
                receipt,
                self.pack_v3,
                audited.rows,
                required=True,
            )

    def test_fail_closed_row_rejects_promoted_authority(self) -> None:
        row = {
            "schema_version": "rootscope.machine_curated_provisional_asset.v1",
            "status": MODULE.STATUS,
            "data_locked": False,
            "human_reviewed": False,
            "print_eligible": False,
            "rights_approved": False,
            "training_eligible": False,
            "machine_curated_only": True,
            "experimental_training_switch_required": True,
            "split": "UNASSIGNED_DO_NOT_TRAIN",
            "authority": {key: False for key in MODULE.REQUIRED_FALSE_AUTHORITY},
        }
        MODULE.validate_fail_closed_record(row, location="fixture")
        row["training_eligible"] = True
        with self.assertRaisesRegex(MODULE.GateError, "training_eligible"):
            MODULE.validate_fail_closed_record(row, location="fixture")

    def test_partition_gate_allows_print_family_but_rejects_train_leakage(self) -> None:
        base = {
            "copied_image_sha256": "a" * 64,
            "creator_group": "creator:shared",
            "source_group": "source:1",
            "experimental_split_suggestion": MODULE.PRINT_ROLE,
        }
        creator_hold = {
            **base,
            "copied_image_sha256": "b" * 64,
            "source_group": "source:2",
            "experimental_split_suggestion": MODULE.CREATOR_HOLDOUT_ROLE,
        }
        MODULE.assert_group_partition_isolation([base, creator_hold])
        leaked_train = {
            **base,
            "copied_image_sha256": "c" * 64,
            "source_group": "source:3",
            "experimental_split_suggestion": MODULE.TRAIN_ROLE,
        }
        with self.assertRaisesRegex(MODULE.GateError, "creator_group"):
            MODULE.assert_group_partition_isolation([base, leaked_train])

    def test_cross_partition_perceptual_near_duplicate_fails_closed(self) -> None:
        train = {
            "asset": "train",
            "dhash64": "0000000000000000",
            "experimental_split_suggestion": MODULE.TRAIN_ROLE,
        }
        validation = {
            "asset": "validation",
            "dhash64": "000000000000000f",
            "experimental_split_suggestion": MODULE.VAL_ROLE,
        }
        with self.assertRaisesRegex(MODULE.GateError, "dHash near-duplicate"):
            MODULE.assert_dhash_partition_isolation([train, validation])
        validation["dhash64"] = "000000000000001f"
        MODULE.assert_dhash_partition_isolation([train, validation])

    def test_source_binding_recomputes_creator_path_sha_and_dhash(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            copied_root = root / "copied"
            source_path = source_root / "images" / "fixture.png"
            copied_path = copied_root / "fixture.png"
            source_path.parent.mkdir(parents=True)
            copied_path.parent.mkdir(parents=True)
            Image.new("RGB", (64, 64), (30, 120, 210)).save(source_path)
            copied_path.write_bytes(source_path.read_bytes())
            digest = MODULE.sha256_file(source_path)
            dhash = MODULE.image_dhash64(source_path)
            source_row = {
                "filename": "images/fixture.png",
                "download_sha256": digest,
                "dhash64_algorithm": MODULE.DHASH_ALGORITHM,
                "dhash64": dhash,
                "source_group": "commons:1",
                "creator_group": "creator:1",
            }
            row = {
                "source_image_path": "images/fixture.png",
                "source_image_sha256": digest,
                "dhash64": dhash,
                "source_group": "commons:1",
                "creator_group": "creator:1",
            }
            self.assertEqual(
                dhash,
                MODULE.assert_source_record_binding(
                    row,
                    source_row,
                    source_root=source_root,
                    copied_image_path=copied_path,
                    location="fixture",
                ),
            )
            tampered = {**row, "creator_group": "creator:forged"}
            with self.assertRaisesRegex(MODULE.GateError, "creator_group"):
                MODULE.assert_source_record_binding(
                    tampered,
                    source_row,
                    source_root=source_root,
                    copied_image_path=copied_path,
                    location="fixture",
                )

    def test_source_decision_role_tamper_is_rejected(self) -> None:
        rows = MODULE.load_jsonl(self.pack_v3 / "manifest.jsonl")
        decisions = MODULE.load_jsonl(self.pack_v3 / "source_decision_manifest.jsonl")
        MODULE.validate_source_decision_semantics(
            decisions,
            rows,
            expected_status=MODULE.STATUS_V3,
        )
        tampered = [dict(value) for value in decisions]
        tampered[0]["experimental_split_suggestion"] = MODULE.VAL_ROLE
        with self.assertRaisesRegex(MODULE.GateError, "experimental_split_suggestion"):
            MODULE.validate_source_decision_semantics(
                tampered,
                rows,
                expected_status=MODULE.STATUS_V3,
            )

    def test_payload_root_excludes_only_receipt_and_sha256sums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "payload.txt").write_text("bound", encoding="utf-8")
            expected = MODULE.payload_root_sha256(root)
            (root / "receipt.json").write_text("{}", encoding="utf-8")
            (root / "SHA256SUMS").write_text("ignored", encoding="utf-8")
            self.assertEqual(expected, MODULE.payload_root_sha256(root))
            (root / "payload.txt").write_text("tampered", encoding="utf-8")
            self.assertNotEqual(expected, MODULE.payload_root_sha256(root))

    def test_actual_v1_pack_passes_full_byte_and_authority_audit(self) -> None:
        audited = MODULE.audit_pack(self.workspace, self.pack_v1)
        self.assertEqual(55, len(audited.rows))
        self.assertEqual(31, len(audited.train_rows))
        self.assertEqual(11, len(audited.validation_rows))
        self.assertEqual(6, len(audited.print_rows))
        self.assertEqual(7, len(audited.creator_holdout_rows))
        self.assertEqual(0, audited.audit["source_group_partition_leakage_count"])
        self.assertFalse(audited.audit["print_evaluation_is_uvc_recapture"])
        self.assertFalse(audited.audit["natural_validation_all_classes_present"])
        self.assertFalse(audited.audit["digital_print_source_holdout_all_classes_present"])
        self.assertEqual(
            {"grass_clump": 1, "low_shrub": 2, "young_tree": 2, "unknown": 26},
            audited.audit["train_class_counts"],
        )
        self.assertFalse(audited.audit["long_training_coverage_gate_passed"])

    def test_actual_v2_pack_audits_but_long_training_coverage_fails_closed(self) -> None:
        audited = MODULE.audit_pack(self.workspace, self.pack_v2)
        self.assertEqual(73, len(audited.rows))
        self.assertEqual(53, len(audited.train_rows))
        self.assertEqual(6, len(audited.validation_rows))
        self.assertEqual(6, len(audited.print_rows))
        self.assertEqual(8, len(audited.creator_holdout_rows))
        self.assertFalse(audited.audit["long_training_coverage_gate_passed"])
        young = audited.audit["long_training_coverage"]["young_tree"]
        self.assertEqual(2, young["train_count"])
        self.assertEqual(0, young["validation_count"])
        self.assertFalse(young["train_met"])
        self.assertFalse(young["validation_met"])
        self.assertIs(audited.receipt["all_split_targets_met"], False)
        with self.assertRaisesRegex(MODULE.GateError, "long training and formal calibration"):
            MODULE.require_long_training_coverage(audited)

    def test_actual_v3_pack_passes_image_source_creator_and_visual_evidence_gates(self) -> None:
        audited = MODULE.audit_pack(self.workspace, self.pack_v3)
        self.assertEqual(78, len(audited.rows))
        self.assertEqual(55, len(audited.train_rows))
        self.assertEqual(9, len(audited.validation_rows))
        self.assertEqual(6, len(audited.print_rows))
        self.assertEqual(8, len(audited.creator_holdout_rows))
        self.assertTrue(audited.audit["long_training_coverage_gate_passed"])
        young = audited.audit["long_training_coverage"]["young_tree"]
        self.assertEqual(5, young["train_count"])
        self.assertEqual(5, young["train_unique_source_count"])
        self.assertEqual(5, young["train_unique_creator_count"])
        self.assertEqual(2, young["validation_count"])
        self.assertEqual(2, young["validation_unique_source_count"])
        self.assertEqual(2, young["validation_unique_creator_count"])
        self.assertEqual(0, audited.audit["cross_partition_dhash_near_duplicate_count"])
        evidence = audited.audit["machine_visual_review_evidence"]
        self.assertEqual(5, evidence["selected_record_count"])
        self.assertEqual("E4_SELECTED_ONLY", evidence["root_machine_adjudication_scope"])

        dataset = MODULE.ManifestImageDataset(
            audited.root,
            audited.validation_rows,
            MODULE.build_transforms(train=False),
        )
        probes = MODULE.build_onnx_consistency_probes(dataset)
        self.assertEqual(
            {
                "natural_validation_first_grass_clump",
                "natural_validation_first_low_shrub",
                "natural_validation_first_young_tree",
                "natural_validation_first_unknown",
                "synthetic_zero",
                "synthetic_ramp",
            },
            set(probes),
        )
        self.assertTrue(all(list(value.shape) == [1, 3, 224, 224] for value in probes.values()))

    def test_resnet18_has_fixed_pool_and_exact_output_order(self) -> None:
        model, provenance = MODULE.build_model(pretrained=False, workspace=self.workspace)
        self.assertEqual((7, 7), model.avgpool.kernel_size)
        self.assertEqual((1, 1), model.avgpool.stride)
        self.assertEqual(4, model.fc.out_features)
        self.assertFalse(provenance["adaptive_pooling"])
        self.assertEqual(list(MODULE.CLASS_NAMES), provenance["class_order"])

    def test_rejection_calibration_fails_closed_when_target_is_impossible(self) -> None:
        import torch

        logits = torch.tensor(
            [
                [10.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0],
            ]
        )
        labels = torch.tensor([1, 2, 3])
        result = MODULE.calibrate_rejection(
            logits,
            labels,
            temperature=1.0,
            target_accepted_accuracy=1.0,
            minimum_accepted=2,
        )
        self.assertEqual("FAIL_CLOSED_REJECT_ALL_TARGET_NOT_MET", result["mode"])
        self.assertEqual(0, result["validation_accepted_count"])
        self.assertEqual(1.0, result["confidence_threshold"])
        self.assertEqual(1.0, result["margin_threshold"])

    def test_rejection_calibration_forces_off_classes_without_wilson_evidence(self) -> None:
        import torch

        logits = torch.tensor(
            [
                [8.0, 0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0, 0.0],
                [0.0, 8.0, 0.0, 0.0],
                [0.0, 8.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 8.0],
                [0.0, 0.0, 0.0, 8.0],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        result = MODULE.calibrate_rejection(logits, labels, temperature=1.0)
        self.assertEqual(
            "VALIDATION_GRID_TARGET_MET_PER_CLASS_EVIDENCE_REJECTS_ALL",
            result["mode"],
        )
        self.assertGreater(result["global_validation_accepted_count"], 0)
        self.assertEqual(0, result["validation_accepted_count"])
        self.assertTrue(
            all(
                not record["acceptance_enabled"]
                for record in result["per_predicted_class_evidence"].values()
            )
        )
        metrics = MODULE.apply_rejection_metrics(
            logits,
            labels,
            domain=MODULE.NATURAL_VAL_DOMAIN,
            calibration=result,
        )
        self.assertEqual(0, metrics["accepted_count"])
        self.assertTrue(metrics["per_predicted_class_gate_applied"])

    def test_output_path_is_required_to_stay_inside_workspace(self) -> None:
        args = MODULE.build_parser().parse_args(
            [
                "--ack-machine-curated-experimental-only",
                "--smoke",
                "--epochs",
                "1",
                "--seeds",
                "17",
                "--max-train-batches",
                "1",
                "--random-init",
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            args.output_root = Path(temporary) / "must_not_be_created"
            with self.assertRaisesRegex(MODULE.GateError, "workspace/output"):
                MODULE.run(args)
            self.assertFalse(args.output_root.exists())


if __name__ == "__main__":
    unittest.main()
