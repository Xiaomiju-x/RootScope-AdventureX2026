from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


DATASET_TOOLS = Path(__file__).resolve().parents[1]
if str(DATASET_TOOLS) not in sys.path:
    sys.path.insert(0, str(DATASET_TOOLS))
SCRIPT = DATASET_TOOLS / "collect_young_tree_category_reacquisition_e4.py"
SPEC = importlib.util.spec_from_file_location("collect_young_tree_category_reacquisition_e4", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class YoungTreeCategoryReacquisitionE4Tests(unittest.TestCase):
    def test_output_and_authority_are_fail_closed(self) -> None:
        self.assertEqual(
            "desert_plants_young_tree_category_reacquisition_e4",
            MODULE.OUTPUT_DATASET_NAME,
        )
        self.assertIn("NOT_TRAIN_READY", MODULE.STATUS)
        self.assertTrue(MODULE.FALSE_AUTHORITY)
        self.assertTrue(all(value is False for value in MODULE.FALSE_AUTHORITY.values()))

    def test_source_plan_contains_required_category_and_nursery_queries(self) -> None:
        queries = {source.retrieval_query for source in MODULE.E4_SOURCE_PLAN}
        for required in (
            "incategory:Seedlings tree",
            "incategory:Seedlings Prosopis",
            "incategory:Seedlings mesquite",
            "incategory:Plant_nurseries seedling",
            '"newly planted" tree sapling',
            '"tree sapling" whole plant',
        ):
            self.assertIn(required, queries)
        self.assertGreaterEqual(len(queries), 30)
        self.assertEqual(len(queries), len(MODULE.E4_SOURCE_PLAN))

    def test_every_query_retains_strict_structural_intent(self) -> None:
        for source in MODULE.E4_SOURCE_PLAN:
            self.assertEqual("young_tree", source.class_id)
            self.assertEqual("search", source.retrieval_mode)
            intent = source.acquisition_query.lower()
            self.assertIn("trunk base visible", intent)
            self.assertIn("entire crown visible", intent)
            self.assertIn("isolated single young plant", intent)
            self.assertIn("metadata must explicitly state", intent)
            self.assertIn("reject mature/ancient/old/large tree", intent)

    def test_metadata_gate_still_rejects_mature_and_detail_images(self) -> None:
        passed, youth, mature = MODULE.e2.metadata_gate(
            "File:Young Acacia sapling.jpg",
            "A mature old tree shown as a branch detail",
        )
        self.assertFalse(passed)
        self.assertIn("sapling", youth)
        self.assertIn("mature", mature)
        self.assertIn("old_tree", mature)

    def test_default_existing_always_contains_e0_through_e3(self) -> None:
        args = MODULE.parse_args([])
        names = {path.name for path in args.existing}
        self.assertTrue(set(MODULE.REQUIRED_EXISTING_DATASET_NAMES).issubset(names))
        self.assertEqual(80, args.target)
        self.assertEqual(3, args.max_per_creator)
        self.assertEqual(4, args.dhash_distance)

    def test_custom_existing_is_additive_not_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            extra = Path(temp) / "extra_dataset"
            args = MODULE.parse_args(["--existing", str(extra)])
            names = {path.name for path in args.existing}
            self.assertIn("extra_dataset", names)
            self.assertTrue(set(MODULE.REQUIRED_EXISTING_DATASET_NAMES).issubset(names))

    def test_missing_required_manifest_fails_before_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing = MODULE.required_existing_roots(root)
            backend_called = False

            def backend(_args: argparse.Namespace) -> list[dict]:
                nonlocal backend_called
                backend_called = True
                return []

            args = argparse.Namespace(
                output=root / "datasets" / MODULE.OUTPUT_DATASET_NAME,
                existing=existing,
                license_policy=DATASET_TOOLS / "wikimedia_license_policy_v1.json",
                target=1,
                api_batches=1,
                max_per_creator=1,
                dhash_distance=1,
                delay=0.0,
            )
            original_root = MODULE.adventurex_root
            MODULE.adventurex_root = lambda: root
            try:
                with self.assertRaises(FileNotFoundError):
                    MODULE.collect(args, backend=backend)
            finally:
                MODULE.adventurex_root = original_root
            self.assertFalse(backend_called)

    def test_fail_closed_record_mutation(self) -> None:
        row = {
            "split": "train",
            "training_eligible": True,
            "print_eligible": True,
        }
        MODULE.enforce_candidate_fail_closed(row)
        self.assertEqual("UNASSIGNED_DO_NOT_TRAIN", row["split"])
        self.assertIs(row["training_eligible"], False)
        self.assertIs(row["print_eligible"], False)
        self.assertIs(row["biological_age_verified"], False)
        self.assertIs(row["visual_whole_plant_verified"], False)
        self.assertTrue(all(value is False for value in row["authority"].values()))

    def test_concurrent_exclusion_manifest_change_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing = MODULE.required_existing_roots(root)
            for dataset in existing:
                dataset.mkdir(parents=True)
                (dataset / "manifest.jsonl").write_text("", encoding="utf-8")
            output = root / "datasets" / MODULE.OUTPUT_DATASET_NAME

            def backend(_args: argparse.Namespace) -> list[dict]:
                (existing[-1] / "manifest.jsonl").write_text('{"pageid":1}\n', encoding="utf-8")
                return []

            args = argparse.Namespace(
                output=output,
                existing=existing,
                license_policy=DATASET_TOOLS / "wikimedia_license_policy_v1.json",
                target=1,
                api_batches=1,
                max_per_creator=1,
                dhash_distance=1,
                delay=0.0,
            )
            original_root = MODULE.adventurex_root
            MODULE.adventurex_root = lambda: root
            try:
                with self.assertRaises(RuntimeError):
                    MODULE.collect(args, backend=backend)
            finally:
                MODULE.adventurex_root = original_root

    def test_output_may_not_be_nested_inside_exclusion_dataset(self) -> None:
        root = Path("C:/example")
        self.assertTrue(MODULE.paths_overlap(root / "E3" / "child", root / "E3"))
        self.assertFalse(MODULE.paths_overlap(root / "E4", root / "E3"))

    def test_input_contract_matches_code_and_confirms_not_run(self) -> None:
        contract = json.loads(
            (DATASET_TOOLS / "young_tree_category_reacquisition_e4_input_contract_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(MODULE.STATUS, contract["status"])
        self.assertEqual(MODULE.OUTPUT_DATASET_NAME, contract["default_output_dataset"])
        self.assertEqual(
            list(MODULE.REQUIRED_EXISTING_DATASET_NAMES),
            contract["required_exclusion_datasets"],
        )
        self.assertIs(contract["network_collection_executed_by_this_delivery"], False)
        self.assertEqual(MODULE.RUN_AUTHORIZATION_STATUS, contract["run_authorization_status"])


if __name__ == "__main__":
    unittest.main()
