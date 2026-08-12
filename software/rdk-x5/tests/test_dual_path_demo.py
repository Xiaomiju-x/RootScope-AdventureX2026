from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.edge.capsule import CapsuleConfig, ROOTSCOPE_CLASS_ORDER
from app.vision.card_geometric_matcher import (
    CLAIM_SCOPE as GEOMETRIC_CLAIM_SCOPE,
    SCHEMA_VERSION as GEOMETRIC_SCHEMA_VERSION,
)
from app.vision.dual_path_demo import (
    AUTHORITY,
    CONSENSUS_STATUS,
    FORMAL_REJECTION_STATUS,
    MODEL_STATUS,
    REGISTERED_ROLE,
    REGISTRY_EMPTY_STATUS,
    REGISTRY_FROZEN_STATUS,
    REGISTRY_SCHEMA_VERSION,
    SEED17_MODEL_SHA256,
    DemoThresholds,
    DualPathContractError,
    evaluate_dual_path_demo,
    load_template_registry,
    run_seed17_semantic_hypothesis,
)


ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / "deploy" / "x5" / "capsule_config.seed17_cpu_experimental.json"


class _FakeSession:
    def __init__(self, logits: list[float]) -> None:
        self.logits = np.asarray([logits], dtype=np.float32)

    def run(self, output_names, feeds):
        if output_names != ["logits"] or list(feeds) != ["image"]:
            raise AssertionError("unexpected ONNX adapter call")
        if feeds["image"].shape != (1, 3, 224, 224):
            raise AssertionError("training-consistent preprocessing was not used")
        return [self.logits.copy()]


class _FakeSeed17Runner:
    def __init__(self, logits: list[float]) -> None:
        config = CapsuleConfig.from_json_file(CAPSULE).model
        self.model_sha256 = SEED17_MODEL_SHA256
        self.class_order = ROOTSCOPE_CLASS_ORDER
        self.expected_output_shape = (1, 4)
        self.providers = ["CPUExecutionProvider"]
        self.preprocess = config.preprocess
        self.input_name = "image"
        self.output_name = "logits"
        self._session = _FakeSession(logits)


def _write_image(path: Path, color: tuple[int, int, int]) -> str:
    y, x = np.indices((260, 340), dtype=np.uint16)
    image = np.empty((260, 340, 3), dtype=np.uint8)
    for channel, base in enumerate(color):
        image[:, :, channel] = (base + x * (channel + 2) + y * (channel + 3)) % 256
    Image.fromarray(image, mode="RGB").save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _template_entry(
    template_id: str,
    class_name: str,
    relative_path: str,
    sha256: str,
) -> dict:
    return {
        "template_id": template_id,
        "class_name": class_name,
        "relative_path": relative_path,
        "raw_sha256": sha256,
        "role": REGISTERED_ROLE,
        "dataset_record": {
            "record_id": f"record-{template_id}",
            "source_manifest": "datasets/frozen/manifest.jsonl",
            "source_url": f"https://example.invalid/source/{template_id}",
            "attribution": {
                "creator": "synthetic-test-only",
                "license": "test-fixture",
                "license_url": "https://example.invalid/license",
            },
        },
    }


def _write_registry(root: Path, entries: list[dict], *, status: str | None = None) -> Path:
    payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "status": status or (REGISTRY_FROZEN_STATUS if entries else REGISTRY_EMPTY_STATUS),
        "template_root": "templates",
        "templates": entries,
    }
    path = root / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _geometry_result(
    template_path,
    query_path,
    template_id,
    template_class,
    *,
    pass_ids,
    authority=False,
):
    passed = template_id in pass_ids
    return {
        "schema": GEOMETRIC_SCHEMA_VERSION,
        "status": "PASS" if passed else "REJECT",
        "passed": passed,
        "claim_scope": GEOMETRIC_CLAIM_SCOPE,
        "irrigation_execution_authority": authority,
        "template_sha256": hashlib.sha256(Path(template_path).read_bytes()).hexdigest(),
        "query_sha256": hashlib.sha256(Path(query_path).read_bytes()).hexdigest(),
        "template_id": template_id,
        "template_class": template_class,
        "authority": {
            "irrigation_execution": authority,
            "pump_command": False,
            "serial_write": False,
            "state_machine_write": False,
        },
        "provenance": {
            "semantic_recognition_performed": False,
            "physical_hardware_touched": False,
        },
    }


def _matcher_for(pass_ids: set[str], *, authority: bool = False):
    def matcher(template_path, query_path, *, template_id, template_class, config):
        del config
        return _geometry_result(
            template_path,
            query_path,
            template_id,
            template_class,
            pass_ids=pass_ids,
            authority=authority,
        )

    return matcher


class TemplateRegistryContractTests(unittest.TestCase):
    def test_empty_example_registers_no_real_template(self) -> None:
        example = ROOT / "app" / "vision" / "known_card_template_registry.empty.example.json"
        registry = load_template_registry(example)
        self.assertEqual(registry.status, REGISTRY_EMPTY_STATUS)
        self.assertEqual(registry.templates, ())

    def test_template_hash_is_checked_and_provenance_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "templates").mkdir()
            sha = _write_image(root / "templates" / "card.png", (17, 63, 111))
            registry = load_template_registry(
                _write_registry(
                    root,
                    [_template_entry("young-01", "young_tree", "card.png", sha)],
                )
            )
        self.assertEqual(registry.templates[0].raw_sha256, sha)
        self.assertEqual(registry.templates[0].role, REGISTERED_ROLE)
        self.assertEqual(registry.templates[0].dataset_record["record_id"], "record-young-01")

    def test_tampered_template_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "templates").mkdir()
            _write_image(root / "templates" / "card.png", (1, 2, 3))
            registry = _write_registry(
                root,
                [_template_entry("card-01", "young_tree", "card.png", "0" * 64)],
            )
            with self.assertRaisesRegex(DualPathContractError, "hash mismatch"):
                load_template_registry(registry)

    def test_path_escape_is_rejected_even_when_outside_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "templates").mkdir()
            sha = _write_image(root / "outside.png", (4, 5, 6))
            registry = _write_registry(
                root,
                [_template_entry("escape", "young_tree", "../outside.png", sha)],
            )
            with self.assertRaisesRegex(DualPathContractError, "normalized relative path"):
                load_template_registry(registry)

    def test_unknown_class_unknown_field_and_duplicate_identity_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "templates").mkdir()
            sha_a = _write_image(root / "templates" / "a.png", (7, 8, 9))
            sha_b = _write_image(root / "templates" / "b.png", (10, 11, 12))
            base = _template_entry("a", "young_tree", "a.png", sha_a)

            unknown_class = copy.deepcopy(base)
            unknown_class["class_name"] = "unknown"
            with self.assertRaisesRegex(DualPathContractError, "unknown cannot be registered"):
                load_template_registry(_write_registry(root, [unknown_class]))

            extra = copy.deepcopy(base)
            extra["unreviewed"] = True
            with self.assertRaisesRegex(DualPathContractError, "unknown=.*unreviewed"):
                load_template_registry(_write_registry(root, [extra]))

            duplicate_id = _template_entry("a", "low_shrub", "b.png", sha_b)
            with self.assertRaisesRegex(DualPathContractError, "duplicate template_id"):
                load_template_registry(_write_registry(root, [base, duplicate_id]))

            duplicate_hash = _template_entry("b", "low_shrub", "a.png", sha_a)
            with self.assertRaisesRegex(DualPathContractError, "duplicate template raw_sha256"):
                load_template_registry(_write_registry(root, [base, duplicate_hash]))


class DualPathEvidenceTests(unittest.TestCase):
    def _fixture(self, root: Path, classes: list[str]) -> tuple[Path, Path, list[dict]]:
        templates = root / "templates"
        templates.mkdir()
        entries = []
        for index, class_name in enumerate(classes):
            file_name = f"card-{index}.png"
            sha = _write_image(templates / file_name, (19 + index, 47, 91))
            entries.append(
                _template_entry(f"template-{index}", class_name, file_name, sha)
            )
        query = root / "query.png"
        _write_image(query, (81, 52, 23))
        return query, _write_registry(root, entries), entries

    def test_semantic_output_is_raw_hypothesis_and_formal_gate_stays_reject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            query = Path(directory) / "query.png"
            _write_image(query, (1, 50, 100))
            result = run_seed17_semantic_hypothesis(
                query,
                _FakeSeed17Runner([0.0, 0.0, 9.0, -1.0]),
                DemoThresholds(),
            )
        self.assertEqual(result["status"], "DEMO_HYPOTHESIS")
        self.assertEqual(result["model"]["selection"], "seed17")
        self.assertEqual(result["model"]["status"], MODEL_STATUS)
        self.assertEqual(result["raw_top1_class"], "young_tree")
        self.assertEqual(result["formal_rejection_gate"]["status"], FORMAL_REJECTION_STATUS)
        self.assertFalse(result["formal_rejection_gate"]["passed"])
        self.assertTrue(all(value is False for value in result["authority"].values()))

    def test_demo_thresholds_reject_boolean_and_unknown_fields(self) -> None:
        with self.assertRaisesRegex(DualPathContractError, "must be numeric"):
            DemoThresholds.from_mapping(
                {"min_top1_probability": True, "min_top1_margin": 0.2}
            )
        with self.assertRaisesRegex(DualPathContractError, "unknown=.*typo"):
            DemoThresholds.from_mapping(
                {
                    "min_top1_probability": 0.7,
                    "min_top1_margin": 0.2,
                    "typo": 1,
                }
            )

    def test_unique_geometry_and_matching_high_margin_semantics_form_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            query, registry, _ = self._fixture(Path(directory), ["young_tree"])
            result = evaluate_dual_path_demo(
                query_path=query,
                runner=_FakeSeed17Runner([0.0, -1.0, 9.0, -2.0]),
                registry_path=registry,
                matcher=_matcher_for({"template-0"}),
            )
        self.assertEqual(result["status"], CONSENSUS_STATUS)
        self.assertTrue(result["experimental_consensus_passed"])
        self.assertEqual(result["consensus"]["selected_template_class"], "young_tree")
        self.assertFalse(any(result["authority"].values()))
        self.assertFalse(any(result["consensus"]["authority"].values()))
        self.assertFalse(result["claims"]["irrigation_decision"])

    def test_multiple_geometric_passes_always_reject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            query, registry, _ = self._fixture(
                Path(directory), ["young_tree", "young_tree"]
            )
            result = evaluate_dual_path_demo(
                query_path=query,
                runner=_FakeSeed17Runner([0.0, -1.0, 9.0, -2.0]),
                registry_path=registry,
                matcher=_matcher_for({"template-0", "template-1"}),
            )
        self.assertFalse(result["experimental_consensus_passed"])
        self.assertIn(
            "MULTIPLE_REGISTERED_TEMPLATES_GEOMETRIC_PASS",
            result["consensus"]["reject_reasons"],
        )

    def test_semantic_template_class_disagreement_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            query, registry, _ = self._fixture(Path(directory), ["young_tree"])
            result = evaluate_dual_path_demo(
                query_path=query,
                runner=_FakeSeed17Runner([0.0, 9.0, -1.0, -2.0]),
                registry_path=registry,
                matcher=_matcher_for({"template-0"}),
            )
        self.assertIn(
            "SEMANTIC_TEMPLATE_CLASS_DISAGREEMENT",
            result["consensus"]["reject_reasons"],
        )
        self.assertFalse(result["experimental_consensus_passed"])

    def test_low_probability_or_margin_rejects_despite_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            query, registry, _ = self._fixture(Path(directory), ["young_tree"])
            result = evaluate_dual_path_demo(
                query_path=query,
                runner=_FakeSeed17Runner([0.0, 0.0, 0.05, 0.0]),
                registry_path=registry,
                thresholds=DemoThresholds(
                    min_top1_probability=0.26, min_top1_margin=0.10
                ),
                matcher=_matcher_for({"template-0"}),
            )
        self.assertIn(
            "EXPERIMENTAL_SEMANTIC_DEMO_THRESHOLD_NOT_MET",
            result["consensus"]["reject_reasons"],
        )
        self.assertFalse(result["semantic"]["experimental_demo_threshold_passed"])

    def test_geometry_authority_claim_is_excluded_from_pass_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            query, registry, _ = self._fixture(Path(directory), ["young_tree"])
            result = evaluate_dual_path_demo(
                query_path=query,
                runner=_FakeSeed17Runner([0.0, -1.0, 9.0, -2.0]),
                registry_path=registry,
                matcher=_matcher_for({"template-0"}, authority=True),
            )
        self.assertEqual(result["geometry"]["contract_valid_pass_count"], 0)
        self.assertIn(
            "GEOMETRIC_AUTHORITY_CONTRACT_VIOLATION",
            result["geometry"]["items"][0]["contract_reject_reasons"],
        )
        self.assertIn(
            "NO_REGISTERED_TEMPLATE_GEOMETRIC_PASS",
            result["consensus"]["reject_reasons"],
        )
        self.assertEqual(result["authority"], AUTHORITY)

    def test_missing_nested_geometry_authority_is_rejected(self) -> None:
        def matcher(template_path, query_path, *, template_id, template_class, config):
            del config
            result = _geometry_result(
                template_path,
                query_path,
                template_id,
                template_class,
                pass_ids={template_id},
            )
            result["authority"] = {}
            return result

        with tempfile.TemporaryDirectory() as directory:
            query, registry, _ = self._fixture(Path(directory), ["young_tree"])
            result = evaluate_dual_path_demo(
                query_path=query,
                runner=_FakeSeed17Runner([0.0, -1.0, 9.0, -2.0]),
                registry_path=registry,
                matcher=matcher,
            )
        self.assertEqual(result["geometry"]["contract_valid_pass_count"], 0)
        self.assertFalse(result["experimental_consensus_passed"])

    def test_non_seed17_runner_is_rejected_before_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            query = Path(directory) / "query.png"
            _write_image(query, (2, 3, 4))
            runner = _FakeSeed17Runner([0.0, 0.0, 1.0, 0.0])
            runner.model_sha256 = "0" * 64
            with self.assertRaisesRegex(DualPathContractError, "seed17"):
                run_seed17_semantic_hypothesis(query, runner, DemoThresholds())


if __name__ == "__main__":
    unittest.main()
