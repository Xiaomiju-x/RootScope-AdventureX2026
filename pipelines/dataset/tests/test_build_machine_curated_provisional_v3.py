from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


DATASET_TOOLS = Path(__file__).resolve().parents[1]
WORKSPACE = DATASET_TOOLS.parents[1]
if str(DATASET_TOOLS) not in sys.path:
    sys.path.insert(0, str(DATASET_TOOLS))
SCRIPT = DATASET_TOOLS / "build_machine_curated_provisional_v3.py"
SPEC = importlib.util.spec_from_file_location("build_machine_curated_provisional_v3", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


class MachineCuratedProvisionalV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = WORKSPACE / "datasets" / MODULE.OUTPUT_NAME

    def test_fixed_contract(self) -> None:
        self.assertEqual("rootscope_machine_curated_provisional_v3", MODULE.OUTPUT_NAME)
        self.assertEqual("rootscope.machine_curated_provisional_asset.v3", MODULE.ASSET_SCHEMA)
        self.assertEqual("rootscope.machine_curated_provisional_receipt.v3", MODULE.RECEIPT_SCHEMA)
        self.assertEqual(
            {
                6191581: ("E3", MODULE.TRAIN_ROLE),
                92774234: ("E4", MODULE.TRAIN_ROLE),
                122973026: ("E4", MODULE.TRAIN_ROLE),
                180772202: ("E4", MODULE.VAL_ROLE),
                184915021: ("E4", MODULE.VAL_ROLE),
            },
            MODULE.NEW_ROLES,
        )
        self.assertEqual(28135991, MODULE.ROLE_OVERRIDE_PAGEID)

    def test_all_authority_is_fail_closed(self) -> None:
        self.assertTrue(all(value is False for value in MODULE.false_authority().values()))
        fields = MODULE.status_fields()
        for key in (
            "data_locked",
            "formal_a1_dataset",
            "formal_split_assigned",
            "human_reviewed",
            "print_eligible",
            "rights_approved",
            "training_eligible",
        ):
            self.assertIs(fields[key], False)
        self.assertIn("NOT_HUMAN_REVIEWED", MODULE.STATUS)
        self.assertIn("NOT_A1", MODULE.STATUS)
        self.assertIn("NOT_DATA_LOCKED", MODULE.STATUS)

    def test_role_partition_contract(self) -> None:
        self.assertEqual("train", MODULE.role_partition(MODULE.TRAIN_ROLE))
        self.assertEqual("val", MODULE.role_partition(MODULE.VAL_ROLE))
        self.assertEqual("holdout", MODULE.role_partition(MODULE.PRINT_ROLE))
        self.assertEqual("holdout", MODULE.role_partition(MODULE.CREATOR_HOLDOUT_ROLE))
        with self.assertRaises(MODULE.V3BuildError):
            MODULE.role_partition("FORMAL_TRAIN")

    def test_hamming_gate_helper(self) -> None:
        self.assertEqual(0, MODULE.hamming64("0000000000000000", "0000000000000000"))
        self.assertEqual(64, MODULE.hamming64("0000000000000000", "ffffffffffffffff"))
        with self.assertRaises(MODULE.V3BuildError):
            MODULE.hamming64("0", "0")

    def test_upstream_machine_screens_are_exact_and_fail_closed(self) -> None:
        screen, evidence = MODULE.load_machine_screens(WORKSPACE)
        self.assertEqual(set(MODULE.NEW_ROLES), {pageid for pageid in MODULE.NEW_ROLES if pageid in screen})
        self.assertEqual([6191581], evidence["E3"]["selected_pageids"])
        self.assertEqual(
            [92774234, 122973026, 180772202, 184915021],
            evidence["E4"]["selected_pageids"],
        )
        for pageid in MODULE.NEW_ROLES:
            self.assertEqual("SELECT", screen[pageid]["decision"])
            self.assertIs(screen[pageid]["human_reviewed"], False)
            self.assertIs(screen[pageid]["training_eligible"], False)

    def test_built_pack_exact_roles_diversity_and_creator_isolation(self) -> None:
        if not self.pack.exists():
            self.skipTest("v3 pack has not been built")
        manifest = rows(self.pack / "manifest.jsonl")
        self.assertEqual(78, len(manifest))
        self.assertEqual(Counter(MODULE.EXPECTED_ROLE_COUNTS), Counter(row["experimental_split_suggestion"] for row in manifest))
        by_id = {row["pageid"]: row for row in manifest}
        self.assertEqual(MODULE.VAL_ROLE, by_id[28135991]["experimental_split_suggestion"])
        for pageid, (generation, role) in MODULE.NEW_ROLES.items():
            self.assertEqual(generation, by_id[pageid]["source_dataset"])
            self.assertEqual(role, by_id[pageid]["experimental_split_suggestion"])
            self.assertEqual("young_tree", by_id[pageid]["class_id"])
        creator_parts: dict[str, set[str]] = {}
        for row in manifest:
            creator_parts.setdefault(row["creator_group"], set()).add(
                MODULE.role_partition(row["experimental_split_suggestion"])
            )
        self.assertFalse({creator: parts for creator, parts in creator_parts.items() if len(parts) > 1})
        for role, classes in MODULE.EXPECTED_DIVERSITY.items():
            for class_id, expected in classes.items():
                subset = [row for row in manifest if row["class_id"] == class_id and row["experimental_split_suggestion"] == role]
                actual = (len(subset), len({row["creator_group"] for row in subset}), len({row["source_group"] for row in subset}))
                self.assertEqual(expected, actual)

    def test_built_pack_decisions_and_visual_evidence_are_bound(self) -> None:
        if not self.pack.exists():
            self.skipTest("v3 pack has not been built")
        manifest = rows(self.pack / "manifest.jsonl")
        decisions = rows(self.pack / "source_decision_manifest.jsonl")
        receipt = json.loads((self.pack / "receipt.json").read_text(encoding="utf-8"))
        evidence = json.loads((self.pack / "machine_visual_review_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(78, len(decisions))
        self.assertEqual({row["pageid"] for row in manifest}, {row["pageid"] for row in decisions})
        self.assertTrue(all(row["selected"] is True for row in decisions))
        self.assertEqual("machine_visual_review_evidence.json", receipt["machine_visual_review_evidence_path"])
        self.assertEqual(
            MODULE.sha256_file(self.pack / "machine_visual_review_evidence.json"),
            receipt["machine_visual_review_evidence_sha256"],
        )
        self.assertIs(evidence["human_reviewed"], False)
        self.assertIs(evidence["data_locked"], False)
        self.assertIs(evidence["print_eligible"], False)
        self.assertIs(evidence["rights_approved"], False)
        self.assertIs(evidence["training_eligible"], False)
        self.assertIs(evidence["all_selected_records_machine_screened"], True)
        self.assertIs(evidence["dual_machine_review_completed"], True)
        self.assertEqual("E4_SELECTED_ONLY", evidence["dual_machine_review_scope"])
        self.assertIs(evidence["root_machine_adjudicated"], True)
        self.assertEqual("E4_SELECTED_ONLY", evidence["root_machine_adjudication_scope"])
        self.assertEqual(sorted(MODULE.NEW_ROLES), evidence["selected_pageids"])
        self.assertEqual(
            {pageid: role for pageid, (_generation, role) in MODULE.NEW_ROLES.items()},
            {row["pageid"]: row["experimental_split_suggestion"] for row in evidence["selected_records"]},
        )
        review_by_id = {row["pageid"]: row for row in evidence["selected_records"]}
        self.assertEqual("E3_MACHINE_SCREEN_ONLY", review_by_id[6191581]["review_scope"])
        self.assertIs(review_by_id[6191581]["dual_machine_reviewed"], False)
        self.assertIs(review_by_id[6191581]["dual_machine_review_completed"], False)
        self.assertIs(review_by_id[6191581]["root_machine_adjudicated"], False)
        for pageid in (92774234, 122973026, 180772202, 184915021):
            self.assertEqual("E4_DUAL_MACHINE_REVIEW_ROOT_ADJUDICATION", review_by_id[pageid]["review_scope"])
            self.assertIs(review_by_id[pageid]["dual_machine_reviewed"], True)
            self.assertIs(review_by_id[pageid]["dual_machine_review_completed"], True)
            self.assertIs(review_by_id[pageid]["root_machine_adjudicated"], True)

    def test_built_pack_hashes_and_protected_inputs(self) -> None:
        if not self.pack.exists():
            self.skipTest("v3 pack has not been built")
        MODULE.verify_sha256sums(self.pack)
        receipt = json.loads((self.pack / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(MODULE.sha256_file(self.pack / "manifest.jsonl"), receipt["manifest_sha256"])
        self.assertEqual(
            MODULE.sha256_file(self.pack / "source_decision_manifest.jsonl"),
            receipt["source_decision_manifest_sha256"],
        )
        self.assertTrue(receipt["protected_inputs"]["unchanged"])
        self.assertEqual(receipt["protected_inputs"]["before"], receipt["protected_inputs"]["after"])
        self.assertTrue(receipt["frozen_v2"]["unchanged"])
        self.assertEqual(0, receipt["audit"]["creator_partition_leakage_count"])
        self.assertGreater(receipt["audit"]["cross_partition_minimum_dhash64_distance"], 4)


if __name__ == "__main__":
    unittest.main()
