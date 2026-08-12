from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


DATASET_TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = DATASET_TOOLS / "audit_machine_curated_provisional_v3.py"
SPEC = importlib.util.spec_from_file_location("audit_machine_curated_provisional_v3", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

WORKSPACE = DATASET_TOOLS.parents[1]
PACK = WORKSPACE / "datasets" / MODULE.PACK_NAME


class AuditMachineCuratedProvisionalV3Tests(unittest.TestCase):
    def test_auditor_does_not_import_v3_builder(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertFalse(
            any("build_machine_curated_provisional_v3" in name for name in imported),
            imported,
        )

    def test_frozen_v3_population_contract(self) -> None:
        self.assertEqual(78, MODULE.EXPECTED_TOTAL)
        self.assertEqual(
            {"grass_clump": 15, "low_shrub": 19, "young_tree": 13, "unknown": 31},
            MODULE.EXPECTED_CLASS_COUNTS,
        )
        self.assertEqual(55, MODULE.EXPECTED_ROLE_COUNTS[MODULE.TRAIN_ROLE])
        self.assertEqual(9, MODULE.EXPECTED_ROLE_COUNTS[MODULE.VAL_ROLE])
        self.assertEqual(6, MODULE.EXPECTED_ROLE_COUNTS[MODULE.PRINT_ROLE])
        self.assertEqual(8, MODULE.EXPECTED_ROLE_COUNTS[MODULE.CREATOR_HOLDOUT_ROLE])
        self.assertEqual(
            {
                6191581: MODULE.TRAIN_ROLE,
                92774234: MODULE.TRAIN_ROLE,
                122973026: MODULE.TRAIN_ROLE,
                180772202: MODULE.VAL_ROLE,
                184915021: MODULE.VAL_ROLE,
            },
            MODULE.EXPECTED_NEW_ROLE,
        )
        self.assertEqual(
            {28135991: (MODULE.TRAIN_ROLE, MODULE.VAL_ROLE)},
            MODULE.EXPECTED_ROLE_OVERRIDE,
        )

    def test_independent_frozen_source_hashes_match_workspace(self) -> None:
        for source_key, dataset_name in MODULE.SOURCE_DATASETS.items():
            path = WORKSPACE / "datasets" / dataset_name / "manifest.jsonl"
            self.assertEqual(MODULE.EXPECTED_SOURCE_MANIFEST_SHA256[source_key], MODULE.sha256_file(path))
        for relative, expected in MODULE.EXPECTED_PROTECTED_TREE_SHA256.items():
            self.assertEqual(expected, MODULE.tree_sha256(WORKSPACE / relative), relative)

    def test_payload_root_excludes_only_receipt_and_sums(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / "output") as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            (root / "nested" / "b.txt").write_text("b\n", encoding="utf-8")
            before = MODULE.payload_root_sha256(root)
            (root / "receipt.json").write_text("{}\n", encoding="utf-8")
            (root / "SHA256SUMS").write_text("placeholder\n", encoding="utf-8")
            self.assertEqual(before, MODULE.payload_root_sha256(root))
            (root / "nested" / "b.txt").write_text("tampered\n", encoding="utf-8")
            self.assertNotEqual(before, MODULE.payload_root_sha256(root))

    def test_sha256sums_duplicate_and_fail_closed_authority_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / "output") as temporary:
            sums = Path(temporary) / "SHA256SUMS"
            digest = "0" * 64
            sums.write_text(f"{digest}  x\n{digest}  x\n", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.AuditError, "duplicate"):
                MODULE.parse_sha256sums(sums)

        record = {
            "authority": {key: False for key in MODULE.AUTHORITY_KEYS},
            "data_locked": False,
            "human_reviewed": False,
            "print_eligible": False,
            "rights_approved": False,
            "training_eligible": False,
            "machine_curated_only": True,
            "formal_a1_dataset": False,
            "formal_split_assigned": False,
            "experimental_training_switch_required": True,
            "split": "UNASSIGNED_DO_NOT_TRAIN",
        }
        MODULE.require_fail_closed(record, "fixture", record=True)
        tampered = json.loads(json.dumps(record))
        tampered["authority"]["training_eligibility"] = True
        with self.assertRaisesRegex(MODULE.AuditError, "non-false"):
            MODULE.require_fail_closed(tampered, "fixture", record=True)

    def test_machine_evidence_role_collector_requires_exact_five(self) -> None:
        fixture = {
            "selected_records": [
                {"pageid": pageid, "role": role} for pageid, role in MODULE.EXPECTED_NEW_ROLE.items()
            ]
        }
        self.assertEqual(MODULE.EXPECTED_NEW_ROLE, MODULE.collect_pageid_roles(fixture))
        fixture["selected_records"][0]["role"] = MODULE.VAL_ROLE
        self.assertNotEqual(MODULE.EXPECTED_NEW_ROLE, MODULE.collect_pageid_roles(fixture))

    @unittest.skipUnless(PACK.is_dir(), "generated v3 pack is not present")
    def test_e3_dual_review_scope_inflation_is_rejected(self) -> None:
        evidence = MODULE.load_json(PACK / "machine_visual_review_evidence.json")
        e3 = MODULE.index_pageids(
            MODULE.load_jsonl(
                WORKSPACE
                / "datasets/desert_plants_young_tree_reacquisition_e3/review/"
                "machine_visual_screen_v1/decisions.jsonl"
            ),
            "E3 fixture",
        )
        e4 = MODULE.index_pageids(
            MODULE.load_jsonl(
                WORKSPACE
                / "datasets/desert_plants_young_tree_category_reacquisition_e4/review/"
                "machine_visual_screen_v1/manifest.jsonl"
            ),
            "E4 fixture",
        )
        MODULE.validate_machine_evidence(evidence, screen_rows={**e3, **e4})

        inflated = json.loads(json.dumps(evidence))
        e3_record = next(row for row in inflated["selected_records"] if row["pageid"] == 6191581)
        e3_record["dual_machine_reviewed"] = True
        with self.assertRaisesRegex(MODULE.AuditError, "E3 falsely marked dual-machine-reviewed"):
            MODULE.validate_machine_evidence(inflated, screen_rows={**e3, **e4})

        inflated = json.loads(json.dumps(evidence))
        inflated["dual_machine_review_scope"] = "E3_AND_E4"
        with self.assertRaisesRegex(MODULE.AuditError, "scope exceeds E4"):
            MODULE.validate_machine_evidence(inflated, screen_rows={**e3, **e4})

    def test_dhash_recomputes_from_pixels_and_detects_change(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE / "output") as temporary:
            first = Path(temporary) / "first.png"
            second = Path(temporary) / "second.png"
            image = Image.new("RGB", (9, 8))
            for y in range(8):
                for x in range(9):
                    image.putpixel((x, y), (x * 20, x * 20, x * 20))
            image.save(first)
            reverse = Image.new("RGB", (9, 8))
            for y in range(8):
                for x in range(9):
                    reverse.putpixel((x, y), ((8 - x) * 20, (8 - x) * 20, (8 - x) * 20))
            reverse.save(second)
            self.assertEqual("0000000000000000", MODULE.image_dhash64(first))
            self.assertEqual("ffffffffffffffff", MODULE.image_dhash64(second))
            self.assertEqual(64, MODULE.dhash_distance(MODULE.image_dhash64(first), MODULE.image_dhash64(second)))

    @unittest.skipUnless(PACK.is_dir(), "generated v3 pack is not present")
    def test_generated_pack_passes_independent_audit(self) -> None:
        report = MODULE.audit(WORKSPACE, PACK)
        self.assertEqual("PASS", report["status"])
        self.assertEqual(78, report["record_count"])
        self.assertEqual(0, report["failure_count"])
        self.assertGreater(report["check_count"], 1000)


if __name__ == "__main__":
    unittest.main()
