from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RootMindSftContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = Path(__file__).resolve().parents[1] / "llm" / "data" / "rootscope_sft_v1"
        cls.receipt = json.loads(
            (cls.data / "dataset_receipt.json").read_text(encoding="utf-8")
        )

    def test_raw_file_hashes_match_receipt(self) -> None:
        for name, expected in self.receipt["file_sha256"].items():
            self.assertEqual(
                hashlib.sha256((self.data / name).read_bytes()).hexdigest(),
                expected,
            )

    def test_records_are_retrieval_bound_and_zero_authority(self) -> None:
        rows = []
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            rows.extend(
                json.loads(line)
                for line in (self.data / name).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        self.assertEqual(len(rows), 1536)
        for row in rows:
            self.assertIs(row["output"]["authority"], False)
            self.assertEqual(
                row["input"]["retrieved_evidence_ids"],
                row["output"]["evidence_ids"],
            )
            unsigned = dict(row)
            observed = unsigned.pop("record_sha256")
            canonical = json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), observed)
            if row["input"]["adversarial_request"] is not None:
                self.assertIn(
                    "ADVERSARIAL_REQUEST_REJECTED",
                    row["output"]["reason_codes"],
                )
                self.assertTrue(
                    row["output"]["proposed_explanation"].startswith("拒绝")
                )

    def test_template_groups_do_not_cross_splits(self) -> None:
        groups = {}
        for name in ("train", "validation", "test"):
            groups[name] = {
                json.loads(line)["template_group"]
                for line in (self.data / f"{name}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            }
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])

    def test_adversarial_curriculum_is_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "curriculum.jsonl"
            manifest = root / "manifest.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parents[1]
                        / "llm"
                        / "build_adversarial_curriculum_v3.py"
                    ),
                    "--train",
                    str(self.data / "train.jsonl"),
                    "--validation",
                    str(self.data / "validation.jsonl"),
                    "--test",
                    str(self.data / "test.jsonl"),
                    "--dataset-receipt",
                    str(self.data / "dataset_receipt.json"),
                    "--output",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--adversarial-copies",
                    "4",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            value = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                value["status"],
                "PASS_TRAIN_ONLY_NO_HELD_OUT_RECORDS",
            )
            self.assertEqual(value["curriculum"]["held_out_record_count"], 0)
            self.assertEqual(
                value["curriculum"]["unique_record_ids"],
                value["source"]["train_unique_rows"],
            )

    def test_frozen_final_holdout_is_disjoint_from_prior_evaluations(self) -> None:
        holdout = self.data / "final_holdout_unseen_v3.jsonl"
        manifest_path = self.data / "final_holdout_unseen_v3.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(holdout.read_bytes()).hexdigest(),
            manifest["holdout"]["sha256"],
        )
        rows = [
            json.loads(line)
            for line in holdout.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 32)
        self.assertEqual(
            sum(row["input"]["adversarial_request"] is not None for row in rows),
            16,
        )
        prior_ids = set()
        adventurex = Path(__file__).resolve().parents[2]
        for relative, expected_hash in manifest["source"]["prior_details"].items():
            path = adventurex / relative
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_hash,
            )
            details = json.loads(path.read_text(encoding="utf-8"))
            prior_ids.update(row["record_id"] for row in details["results"])
        self.assertFalse({row["record_id"] for row in rows} & prior_ids)


if __name__ == "__main__":
    unittest.main()
