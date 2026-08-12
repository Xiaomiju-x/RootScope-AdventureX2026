from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "quarantine_wikimedia_candidate.py"
SPEC = importlib.util.spec_from_file_location("quarantine_wikimedia_candidate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QuarantineCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.dataset = Path(self.temporary.name)
        self.image = self.dataset / "images" / "unknown" / "candidate.jpg"
        self.image.parent.mkdir(parents=True)
        Image.new("RGB", (320, 120), (193, 171, 126)).save(self.image, "JPEG")
        self.record = {
            "class": "unknown",
            "download_sha256": sha256(self.image),
            "filename": "images/unknown/candidate.jpg",
            "pageid": 159472183,
            "source_page": "https://commons.wikimedia.org/?curid=159472183",
        }
        self.manifest = self.dataset / "manifest.jsonl"
        self.manifest.write_text(
            json.dumps(self.record, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        self.before_sha = sha256(self.manifest)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_quarantine(self, expected_sha: str | None = None) -> dict:
        return MODULE.quarantine_candidate(
            self.dataset,
            159472183,
            expected_sha or self.before_sha,
            "e0-min-side-159472183-test",
            "downloaded_short_side_below_448",
        )

    def test_wrong_manifest_hash_does_not_mutate_dataset(self) -> None:
        image_before = self.image.read_bytes()
        manifest_before = self.manifest.read_bytes()
        with self.assertRaises(MODULE.QuarantineError):
            self.run_quarantine("0" * 64)
        self.assertEqual(manifest_before, self.manifest.read_bytes())
        self.assertEqual(image_before, self.image.read_bytes())
        self.assertFalse((self.dataset / "quarantine").exists())

    def test_success_moves_payload_and_writes_complete_receipt(self) -> None:
        original_payload = self.image.read_bytes()
        receipt = self.run_quarantine()
        destination = self.dataset / receipt["destination"]
        self.assertEqual("COMPLETE", receipt["status"])
        self.assertTrue(receipt["reversible"])
        self.assertFalse(receipt["delete_performed"])
        self.assertTrue(receipt["manifest_row_removed"])
        self.assertFalse(self.image.exists())
        self.assertEqual(original_payload, destination.read_bytes())
        self.assertEqual(b"", self.manifest.read_bytes())
        receipt_path = self.dataset / "quarantine" / receipt["receipt_id"] / "receipt.json"
        self.assertTrue(receipt_path.is_file())
        self.assertEqual(receipt["manifest_after_sha256"], sha256(self.manifest))

    def test_tampered_payload_fails_without_manifest_mutation(self) -> None:
        manifest_before = self.manifest.read_bytes()
        self.image.write_bytes(b"tampered")
        with self.assertRaises(MODULE.QuarantineError):
            self.run_quarantine()
        self.assertEqual(manifest_before, self.manifest.read_bytes())
        self.assertEqual(b"tampered", self.image.read_bytes())
        self.assertFalse((self.dataset / "quarantine").exists())

    def test_immediate_repeat_is_idempotent(self) -> None:
        first = self.run_quarantine()
        second = self.run_quarantine()
        self.assertEqual(first, second)
        destination = self.dataset / first["destination"]
        self.assertTrue(destination.is_file())
        self.assertFalse(self.image.exists())

    def test_non_target_manifest_line_is_byte_preserved(self) -> None:
        target_line = self.manifest.read_bytes()
        untouched_line = b'{"pageid":2, "note" : "preserve spacing and CRLF"}\r\n'
        self.manifest.write_bytes(target_line + untouched_line)
        self.before_sha = sha256(self.manifest)
        self.run_quarantine()
        self.assertEqual(untouched_line, self.manifest.read_bytes())

    def test_recovers_when_crash_occurs_after_move_before_manifest_commit(self) -> None:
        real_atomic_write = MODULE._atomic_write

        def fail_manifest_commit(path: Path, payload: bytes) -> None:
            if Path(path).resolve() == self.manifest.resolve():
                raise OSError("injected crash before manifest commit")
            real_atomic_write(path, payload)

        with mock.patch.object(MODULE, "_atomic_write", side_effect=fail_manifest_commit):
            with self.assertRaises(OSError):
                self.run_quarantine()
        self.assertFalse(self.image.exists())
        self.assertEqual(self.before_sha, sha256(self.manifest))
        recovered = self.run_quarantine()
        self.assertEqual("COMPLETE", recovered["status"])
        self.assertEqual(recovered["manifest_after_sha256"], sha256(self.manifest))

    def test_recovers_when_crash_occurs_after_manifest_before_receipt(self) -> None:
        real_atomic_write = MODULE._atomic_write

        def fail_receipt(path: Path, payload: bytes) -> None:
            if Path(path).name == "receipt.json":
                raise OSError("injected crash before receipt")
            real_atomic_write(path, payload)

        with mock.patch.object(MODULE, "_atomic_write", side_effect=fail_receipt):
            with self.assertRaises(OSError):
                self.run_quarantine()
        self.assertFalse(self.image.exists())
        recovered = self.run_quarantine()
        self.assertEqual("COMPLETE", recovered["status"])

    def test_recovers_when_crash_occurs_after_intent_before_move(self) -> None:
        with mock.patch.object(Path, "rename", side_effect=OSError("injected crash before move")):
            with self.assertRaises(OSError):
                self.run_quarantine()
        self.assertTrue(self.image.exists())
        self.assertEqual(self.before_sha, sha256(self.manifest))
        recovered = self.run_quarantine()
        self.assertEqual("COMPLETE", recovered["status"])

    def test_manifest_cas_rejects_concurrent_change_without_overwrite(self) -> None:
        real_sha256_file = MODULE._sha256_file
        concurrent_line = b'{"pageid":999,"external":"concurrent"}\n'

        def change_before_cas(path: Path) -> str:
            path = Path(path)
            if path.resolve() == self.manifest.resolve():
                self.manifest.write_bytes(self.manifest.read_bytes() + concurrent_line)
            return real_sha256_file(path)

        with mock.patch.object(MODULE, "_sha256_file", side_effect=change_before_cas):
            with self.assertRaises(MODULE.QuarantineError):
                self.run_quarantine()
        self.assertTrue(self.image.exists())
        self.assertTrue(self.manifest.read_bytes().endswith(concurrent_line))

    def test_second_writer_is_rejected_while_dataset_lock_is_held(self) -> None:
        with MODULE._dataset_writer_lock(self.dataset):
            with self.assertRaises(MODULE.QuarantineError):
                self.run_quarantine()


if __name__ == "__main__":
    unittest.main()
