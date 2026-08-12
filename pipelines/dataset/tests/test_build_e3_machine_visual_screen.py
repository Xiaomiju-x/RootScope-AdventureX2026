from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS / "build_e3_machine_visual_screen.py"
SPEC = importlib.util.spec_from_file_location("build_e3_machine_visual_screen", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class E3MachineVisualScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = TOOLS.parents[1]
        cls.dataset = cls.workspace / "datasets" / MODULE.DATASET_NAME
        cls.manifest_rows = MODULE.load_jsonl(cls.dataset / "manifest.jsonl")

    def test_frozen_decisions_have_exact_counts_and_special_cases(self) -> None:
        counts = {}
        for decision, _reason in MODULE.DECISIONS.values():
            counts[decision] = counts.get(decision, 0) + 1
        self.assertEqual({"SELECT": 1, "HOLD": 2, "EXCLUDE": 12}, counts)
        self.assertEqual("SELECT", MODULE.DECISIONS[6191581][0])
        self.assertEqual("HOLD", MODULE.DECISIONS[105533544][0])
        self.assertEqual("EXCLUDE", MODULE.DECISIONS[137881650][0])

    def test_decisions_exactly_cover_e3_manifest(self) -> None:
        index = MODULE.indexed_manifest(self.manifest_rows)
        self.assertEqual(set(index), set(MODULE.DECISIONS))
        self.assertEqual(15, len(index))

    def test_rows_bind_manifest_source_record_image_and_creator(self) -> None:
        rows = MODULE.build_decision_rows(self.dataset, self.manifest_rows)
        stats = MODULE.validate_decision_rows(rows)
        manifest_sha = MODULE.sha256_file(self.dataset / "manifest.jsonl")
        source_index = {row["pageid"]: row for row in self.manifest_rows}
        self.assertEqual({"SELECT": 1, "HOLD": 2, "EXCLUDE": 12}, stats["decision_counts"])
        for row in rows:
            source = source_index[row["pageid"]]
            self.assertEqual(manifest_sha, row["source_manifest_sha256"])
            self.assertEqual(
                MODULE.sha256_bytes(MODULE.canonical_json(source).encode("utf-8")),
                row["source_record_sha256"],
            )
            self.assertEqual(source["download_sha256"], row["image_sha256"])
            self.assertEqual(source["creator_group"], row["creator_group"])
            self.assertFalse(row["training_eligible"])
            self.assertFalse(row["print_eligible"])
            self.assertFalse(row["data_locked"])
            self.assertFalse(row["human_reviewed"])
            self.assertTrue(all(value is False for value in row["authority"].values()))

    def test_isolated_fixture_build_is_complete_and_hash_bound(self) -> None:
        formal_root = (
            self.workspace
            / "datasets"
            / "desert_plants_wikimedia_staging_e0"
            / "review"
            / "human_decisions"
        )
        before = MODULE.tree_sha256(formal_root)
        with tempfile.TemporaryDirectory(dir=self.workspace / "output") as temporary:
            output = Path(temporary) / "machine_visual_screen_v1"
            result = MODULE.build_screen(
                workspace=self.workspace,
                output=output,
                allow_nonproduction_output=True,
            )
            self.assertEqual(output, result)
            self.assertTrue((result / "decisions.jsonl").is_file())
            self.assertTrue((result / "contact_sheet.png").is_file())
            self.assertTrue((result / "receipt.json").is_file())
            self.assertTrue((result / "SHA256SUMS").is_file())
            receipt = json.loads((result / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(MODULE.STATUS, receipt["status"])
            self.assertEqual(1, receipt["statistics"]["decision_counts"]["SELECT"])
            self.assertFalse(receipt["training_eligible"])
            self.assertTrue(receipt["formal_human_decisions"]["unchanged"])
            for line in (result / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                digest, relative = line.split("  ", 1)
                self.assertEqual(digest, MODULE.sha256_file(result / relative))
        self.assertEqual(before, MODULE.tree_sha256(formal_root))

    def test_select_is_never_train_eligible(self) -> None:
        rows = MODULE.build_decision_rows(self.dataset, self.manifest_rows)
        selected = [row for row in rows if row["decision"] == "SELECT"]
        self.assertEqual([6191581], [row["pageid"] for row in selected])
        self.assertTrue(selected[0]["provisional_candidate_only"])
        self.assertFalse(selected[0]["selection_grants_training_eligibility"])
        self.assertFalse(selected[0]["training_eligible"])


if __name__ == "__main__":
    unittest.main()
