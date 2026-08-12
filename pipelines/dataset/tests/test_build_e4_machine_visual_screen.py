from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS / "build_e4_machine_visual_screen.py"
SPEC = importlib.util.spec_from_file_location("build_e4_machine_visual_screen", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class E4MachineVisualScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = TOOLS.parents[1]
        cls.dataset = cls.workspace / "datasets" / MODULE.DATASET_NAME
        cls.manifest_rows = MODULE.load_jsonl(cls.dataset / "manifest.jsonl")
        cls.contract_text = MODULE.pretty_json_text(MODULE.adjudication_contract())
        cls.contract_sha = MODULE.sha256_bytes(cls.contract_text.encode("utf-8"))

    def test_root_adjudication_has_exact_counts_and_pageids(self) -> None:
        counts = {}
        for decision, _reason in MODULE.DECISIONS.values():
            counts[decision] = counts.get(decision, 0) + 1
        self.assertEqual({"SELECT": 4, "HOLD": 4, "EXCLUDE": 54}, counts)
        self.assertEqual(
            [92774234, 122973026, 180772202, 184915021],
            sorted(pageid for pageid, value in MODULE.DECISIONS.items() if value[0] == "SELECT"),
        )
        self.assertEqual(
            [130133197, 130133198, 173706908, 184915109],
            sorted(pageid for pageid, value in MODULE.DECISIONS.items() if value[0] == "HOLD"),
        )

    def test_decisions_exactly_cover_e4_manifest(self) -> None:
        index = MODULE.indexed_manifest(self.manifest_rows)
        self.assertEqual(set(index), set(MODULE.DECISIONS))
        self.assertEqual(62, len(index))

    def test_rows_bind_manifest_source_creator_image_sha_and_recomputed_dhash(self) -> None:
        rows = MODULE.build_decision_rows(
            self.dataset,
            self.manifest_rows,
            contract_sha256=self.contract_sha,
        )
        stats = MODULE.validate_decision_rows(rows)
        manifest_sha = MODULE.sha256_file(self.dataset / "manifest.jsonl")
        collection_receipt_sha = MODULE.sha256_file(self.dataset / "collection_receipt.json")
        source_index = {row["pageid"]: row for row in self.manifest_rows}
        self.assertEqual({"SELECT": 4, "HOLD": 4, "EXCLUDE": 54}, stats["decision_counts"])
        self.assertEqual("62/62", stats["coverage"])
        for row in rows:
            source = source_index[row["pageid"]]
            image = self.dataset / source["filename"]
            self.assertEqual(manifest_sha, row["source_manifest_sha256"])
            self.assertEqual(collection_receipt_sha, row["source_collection_receipt_sha256"])
            self.assertEqual(
                MODULE.sha256_bytes(MODULE.canonical_json(source).encode("utf-8")),
                row["source_record_sha256"],
            )
            self.assertEqual(source["creator_group"], row["creator_group"])
            self.assertEqual(source["source_group"], row["source_group"])
            self.assertEqual(source["download_sha256"], row["image_sha256"])
            self.assertEqual(source["dhash64"], row["dhash64"])
            self.assertEqual(source["dhash64"], MODULE.image_dhash64(image))
            self.assertTrue(row["dhash64_recomputed_from_original"])
            self.assertTrue(row["dhash64_matches_source_manifest"])
            self.assertEqual(2, row["independent_machine_review_count"])
            self.assertTrue(row["root_machine_adjudicated"])
            self.assertFalse(row["review_is_human_label"])
            self.assertFalse(row["human_reviewed"])
            self.assertFalse(row["human_label"])
            self.assertFalse(row["data_authority"])
            self.assertFalse(row["train_eligible"])
            self.assertFalse(row["training_eligible"])
            self.assertFalse(row["print_eligible"])
            self.assertFalse(row["data_lock"])
            self.assertFalse(row["data_locked"])
            self.assertTrue(all(value is False for value in row["authority"].values()))

    def test_corrupt_source_sha_or_dhash_fails_closed(self) -> None:
        bad_sha = [dict(row) for row in self.manifest_rows]
        bad_sha[0]["download_sha256"] = "0" * 64
        with self.assertRaisesRegex(MODULE.ScreenError, "image SHA mismatch"):
            MODULE.build_decision_rows(self.dataset, bad_sha, contract_sha256=self.contract_sha)

        bad_dhash = [dict(row) for row in self.manifest_rows]
        bad_dhash[0]["dhash64"] = "0" * 16
        with self.assertRaisesRegex(MODULE.ScreenError, "image dHash mismatch"):
            MODULE.build_decision_rows(self.dataset, bad_dhash, contract_sha256=self.contract_sha)

    def test_select_is_only_provisional_and_never_screen_train_eligible(self) -> None:
        rows = MODULE.build_decision_rows(
            self.dataset,
            self.manifest_rows,
            contract_sha256=self.contract_sha,
        )
        selected = [row for row in rows if row["decision"] == "SELECT"]
        self.assertEqual(
            [92774234, 122973026, 180772202, 184915021],
            [row["pageid"] for row in selected],
        )
        for row in selected:
            self.assertEqual("SELECT_PROVISIONAL_CANDIDATE_ONLY", row["disposition"])
            self.assertTrue(row["provisional_candidate_only"])
            self.assertFalse(row["selection_grants_training_eligibility"])
            self.assertFalse(row["training_eligible"])

    def test_isolated_build_is_complete_hash_bound_and_preserves_sources(self) -> None:
        source_before = MODULE.tree_sha256(self.dataset, exclude_top_level=frozenset({"review"}))
        journal = (
            self.workspace
            / "datasets"
            / "desert_plants_wikimedia_staging_e0"
            / "review"
            / "human_decisions"
            / "decision_journal.jsonl"
        )
        journal_before = MODULE.sha256_file(journal)
        with tempfile.TemporaryDirectory(dir=self.workspace / "output") as temporary:
            output = Path(temporary) / "machine_visual_screen_v1"
            result = MODULE.build_screen(
                workspace=self.workspace,
                output=output,
                allow_nonproduction_output=True,
            )
            self.assertEqual(output, result)
            expected_files = {
                "adjudication_contract.json",
                "contact_sheet.png",
                "decisions.jsonl",
                "manifest.jsonl",
                "receipt.json",
                "SHA256SUMS",
            }
            self.assertEqual(expected_files, {path.name for path in result.iterdir() if path.is_file()})
            self.assertEqual(
                (result / "manifest.jsonl").read_bytes(),
                (result / "decisions.jsonl").read_bytes(),
            )
            receipt = json.loads((result / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(MODULE.RECEIPT_SCHEMA, receipt["schema_version"])
            self.assertEqual(MODULE.STATUS, receipt["status"])
            self.assertEqual(62, receipt["statistics"]["total"])
            self.assertEqual(4, receipt["statistics"]["decision_counts"]["SELECT"])
            self.assertEqual(2, receipt["review_pipeline"]["independent_machine_review_count"])
            self.assertTrue(receipt["review_pipeline"]["root_machine_adjudicated"])
            self.assertFalse(receipt["human_reviewed"])
            self.assertFalse(receipt["human_label"])
            self.assertFalse(receipt["data_authority"])
            self.assertFalse(receipt["training_eligible"])
            self.assertFalse(receipt["print_eligible"])
            self.assertFalse(receipt["data_locked"])
            self.assertTrue(receipt["protected_inputs"]["unchanged"])
            for line in (result / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                digest, relative = line.split("  ", 1)
                self.assertEqual(digest, MODULE.sha256_file(result / relative))
        self.assertEqual(source_before, MODULE.tree_sha256(self.dataset, exclude_top_level=frozenset({"review"})))
        self.assertEqual(journal_before, MODULE.sha256_file(journal))


if __name__ == "__main__":
    unittest.main()
