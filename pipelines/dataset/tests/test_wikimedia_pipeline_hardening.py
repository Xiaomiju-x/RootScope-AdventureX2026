from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from adventurex.tools.dataset.tests.test_build_wikimedia_review_queue import (
    POLICY_PATH,
    ReviewQueueFixture,
)


TOOLS_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = TOOLS_ROOT / "collect_wikimedia_candidates.ps1"
AUDITOR = TOOLS_ROOT / "audit_wikimedia_candidates.ps1"


class WikimediaPipelineHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.fixture = ReviewQueueFixture(Path(self.tempdir.name))
        self._write_legacy_summary()

    def _write_legacy_summary(self) -> None:
        summary = {
            "requested_targets": {
                "grass_clump": 1,
                "low_shrub": 1,
                "young_tree": 1,
                "unknown": 1,
            },
            "holdout_dhash_rejection_threshold": 6,
            "candidate_dhash_rejection_threshold": 6,
            "api_batches_per_query_limit": 4,
        }
        (self.fixture.staging_root / "summary.json").write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )

    def _collector(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(COLLECTOR),
                "-Output",
                str(self.fixture.staging_root),
                "-ExistingDataset",
                str(self.fixture.holdout_root),
                "-LicensePolicy",
                str(POLICY_PATH),
                *extra,
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )

    def _auditor(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(AUDITOR),
                "-Dataset",
                str(self.fixture.staging_root),
                "-ExistingDataset",
                str(self.fixture.holdout_root),
                "-LicensePolicy",
                str(POLICY_PATH),
                "-Out",
                str(self.fixture.staging_root / "auditor_result.json"),
                *extra,
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )

    def test_finalize_only_recovers_same_sha_pending_orphan_and_seals_receipts(self) -> None:
        orphan = self.fixture.make_candidate(900002, "low_shrub", b"pending-orphan")
        pending_dir = self.fixture.staging_root / "pending"
        pending_dir.mkdir()
        pending = {
            "schema_version": "rootscope.wikimedia_candidate_pending.v1",
            "run_id": "fixture-crashed-run",
            "created_at_utc": "2026-07-16T00:00:00Z",
            "temporary_path": "images/low_shrub/.900002.fixture.download",
            "final_path": orphan["filename"],
            "download_sha256": orphan["download_sha256"],
            "record": orphan,
        }
        (pending_dir / "900002_pending.json").write_text(
            json.dumps(pending, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )

        result = self._collector("-FinalizeOnly")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [
            json.loads(line)
            for line in self.fixture.staging_manifest.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual({item["pageid"] for item in records}, {900001, 900002})
        self.assertFalse(any(pending_dir.glob("*.json")))
        for record in records:
            self.assertEqual(record["license_policy_sha256"], hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest())
            self.assertEqual(record["dhash64_algorithm"], "rootscope_rgb_center_sample_9x8_v1")
            self.assertEqual(len(record["dhash64"]), 16)
            self.assertEqual(record["download_width"], 512)
            self.assertEqual(record["download_height"], 512)

        summary = json.loads((self.fixture.staging_root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            summary["manifest_sha256"],
            hashlib.sha256(self.fixture.staging_manifest.read_bytes()).hexdigest(),
        )
        self.assertEqual(summary["license_policy_sha256"], hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest())
        self.assertEqual(summary["collector_script_sha256"], hashlib.sha256(COLLECTOR.read_bytes()).hexdigest())
        self.assertEqual(summary["permanent_print_holdout_count"], 6)
        self.assertEqual(summary["existing_overlap_counts"], {"pageid": 0, "source_group": 0, "commons_sha1": 0, "download_sha256": 0})
        self.assertFalse(summary["unknown_hint_requirements"]["unknown_hint_minimums_met"])
        self.assertFalse(summary["acquisition_targets_met"])

        audit_process = self._auditor()
        self.assertEqual(audit_process.returncode, 1)
        audit = json.loads((self.fixture.staging_root / "auditor_result.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["result"], "FAIL")
        codes = {item["code"] for item in audit["failures"]}
        self.assertIn("unknown_hint_deficit", codes)
        self.assertNotIn("license_pair", codes)
        self.assertNotIn("license_canonical_binding", codes)
        self.assertNotIn("dhash_manifest", codes)
        self.assertNotIn("summary_manifest_sha256", codes)

    def test_collector_rejects_second_writer_while_os_lock_is_held(self) -> None:
        lock_path = self.fixture.staging_root / ".collector.lock"
        command = (
            f"$f=[IO.File]::Open('{lock_path}',[IO.FileMode]::OpenOrCreate,"
            "[IO.FileAccess]::ReadWrite,[IO.FileShare]::None);"
            "Write-Output 'READY';[Console]::In.ReadLine()|Out-Null;$f.Dispose()"
        )
        holder = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.addCleanup(lambda: holder.kill() if holder.poll() is None else None)
        assert holder.stdout is not None
        self.assertEqual(holder.stdout.readline().strip(), "READY")
        result = self._collector("-FinalizeOnly")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Another collector owns the staging lock", result.stderr)
        assert holder.stdin is not None
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(timeout=10)
        holder.stdin.close()
        holder.stdout.close()
        assert holder.stderr is not None
        holder.stderr.close()

    def test_auditor_rejects_threshold_argument_weaker_than_sealed_summary(self) -> None:
        sealed = self._collector("-FinalizeOnly")
        self.assertEqual(sealed.returncode, 0, sealed.stdout + sealed.stderr)
        result = self._auditor("-CandidateDHashDistance", "2")
        self.assertEqual(result.returncode, 1)
        audit = json.loads((self.fixture.staging_root / "auditor_result.json").read_text(encoding="utf-8"))
        codes = {item["code"] for item in audit["failures"]}
        self.assertIn("candidate_threshold_argument_mismatch", codes)

    def test_finalize_only_binds_quarantine_receipt_and_blocks_reappearance(self) -> None:
        record = json.loads(self.fixture.staging_manifest.read_text(encoding="utf-8"))
        source = self.fixture.staging_root / record["filename"]
        receipt_id = "fixture-quarantine-900001"
        destination_relative = f"quarantine/{receipt_id}/payload/{source.name}"
        destination = self.fixture.staging_root / destination_relative
        destination.parent.mkdir(parents=True)
        source.rename(destination)
        self.fixture.staging_manifest.write_text("", encoding="utf-8")
        receipt = {
            "schema_version": "rootscope.wikimedia_candidate_quarantine.v1",
            "status": "COMPLETE",
            "receipt_id": receipt_id,
            "pageid": record["pageid"],
            "reason": "fixture_policy_rejection",
            "destination": destination_relative,
            "file_sha256": record["download_sha256"],
            "record": record,
        }
        receipt_path = destination.parent.parent / "receipt.json"
        receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

        result = self._collector("-FinalizeOnly")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((self.fixture.staging_root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["quarantine_receipt_count"], 1)
        self.assertEqual(summary["quarantine_receipts"][0]["pageid"], record["pageid"])

        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, source)
        self.fixture.staging_manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
        rejected = self._collector("-FinalizeOnly")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("Quarantined pageid reappeared in manifest", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
