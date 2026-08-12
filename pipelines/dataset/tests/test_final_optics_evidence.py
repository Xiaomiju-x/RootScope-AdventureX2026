from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS_DATASET = Path(__file__).resolve().parents[1]
ADVENTUREX = TOOLS_DATASET.parents[1]
ROOTSCOPE_TRAINING = ADVENTUREX / "rootscope" / "training"
sys.path.insert(0, str(TOOLS_DATASET))

import final_optics_evidence as optics  # noqa: E402


class FinalOpticsEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_payload(self, parent: Path, *, reverse: bool = False) -> Path:
        payload = parent / "payload"
        payload.mkdir(parents=True)
        records = [
            (Path("a.txt"), b"alpha"),
            (Path("nested") / "b.bin", b"\x00\x01\xff"),
            (Path("unicode") / "沙地.txt", "根区".encode("utf-8")),
        ]
        if reverse:
            records.reverse()
        for relative, body in records:
            path = payload / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        return payload

    def _write_receipt(self, bundle: Path, roots: dict[str, str]) -> dict:
        approvals_dir = bundle / "approvals"
        approvals_dir.mkdir(parents=True, exist_ok=True)
        approval_hashes: dict[str, str] = {}
        for role in optics.ROLE_MEMBERS:
            body = f"fixture approval for {role}\n".encode("utf-8")
            (approvals_dir / f"{role}.approval").write_bytes(body)
            approval_hashes[role] = hashlib.sha256(body).hexdigest()
        receipt = {
            "schema_version": "1.0.0",
            "receipt_id": "fixture.final-optics.v1",
            "signed_roles": {
                role: {
                    "member": member,
                    "signed": True,
                    "signer": f"fixture-{role}",
                    "approval_evidence_sha256": approval_hashes[role],
                }
                for role, member in optics.ROLE_MEMBERS.items()
            },
            "evidence_roots": roots,
        }
        (bundle / "final_optics_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return receipt

    def _make_bundle(self) -> tuple[Path, dict[str, str], dict]:
        bundle = self.root / "final_optics_bundle"
        evidence_dir = bundle / "evidence"
        manifests_dir = bundle / "manifests"
        evidence_dir.mkdir(parents=True)
        manifests_dir.mkdir(parents=True)
        roots: dict[str, str] = {}
        for index, kind in enumerate(optics.EVIDENCE_KINDS):
            payload = evidence_dir / kind
            payload.mkdir()
            (payload / f"{kind}_fixture_{index}.txt").write_text(
                f"synthetic fixture bytes for {kind}\n",
                encoding="utf-8",
            )
            summary = optics.write_package_manifest(
                payload,
                kind,
                manifests_dir / f"{kind}.manifest.json",
            )
            roots[kind] = summary["evidence_root"]
        receipt = self._write_receipt(bundle, roots)
        return bundle, roots, receipt

    def _rewrite_receipt(self, bundle: Path, receipt: dict) -> None:
        (bundle / "final_optics_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def test_manifest_is_deterministic_across_locations_and_creation_order(self) -> None:
        first = self._make_payload(self.root / "first")
        second = self._make_payload(self.root / "second", reverse=True)
        first_manifest, first_root = optics.build_package_manifest(first, "uvc")
        second_manifest, second_root = optics.build_package_manifest(second, "uvc")
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_root, second_root)

    def test_manifest_bytes_are_exact_canonical_compact_json(self) -> None:
        payload = self.root / "payload"
        payload.mkdir()
        (payload / "a.txt").write_bytes(b"A")
        manifest_path = self.root / "manifests" / "uvc.manifest.json"
        result = optics.write_package_manifest(payload, "uvc", manifest_path)
        expected = (
            '{"evidence_kind":"uvc","files":[{"bytes":1,"path":"a.txt",'
            '"sha256":"559aead08264d5795d3909718cdd05abd49572e84fe55590eef31a88a08fdffd"}],'
            '"schema_version":"rootscope.final_optics.evidence_manifest.v1"}'
        ).encode("utf-8")
        self.assertEqual(expected, manifest_path.read_bytes())
        self.assertEqual(hashlib.sha256(expected).hexdigest(), result["evidence_root"])

    def test_empty_package_is_rejected(self) -> None:
        payload = self.root / "empty"
        payload.mkdir()
        with self.assertRaisesRegex(optics.EvidenceError, "empty"):
            optics.build_package_manifest(payload, "uvc")

    def test_manifest_inside_payload_is_rejected_without_writing(self) -> None:
        payload = self._make_payload(self.root / "inside")
        output = payload / "manifest.json"
        with self.assertRaisesRegex(optics.EvidenceError, "outside"):
            optics.write_package_manifest(payload, "uvc", output)
        self.assertFalse(output.exists())

    def test_existing_manifest_is_never_overwritten(self) -> None:
        payload = self._make_payload(self.root / "overwrite")
        output = self.root / "manifests" / "uvc.manifest.json"
        optics.write_package_manifest(payload, "uvc", output)
        original = output.read_bytes()
        with self.assertRaisesRegex(optics.EvidenceError, "overwrite"):
            optics.write_package_manifest(payload, "uvc", output)
        self.assertEqual(original, output.read_bytes())

    def test_file_tamper_and_added_file_are_detected(self) -> None:
        for mutation in ("tamper", "add"):
            with self.subTest(mutation=mutation):
                parent = self.root / mutation
                payload = self._make_payload(parent)
                output = parent / "uvc.manifest.json"
                optics.write_package_manifest(payload, "uvc", output)
                if mutation == "tamper":
                    (payload / "a.txt").write_bytes(b"changed")
                else:
                    (payload / "new.txt").write_bytes(b"new")
                with self.assertRaisesRegex(optics.EvidenceError, "does not match"):
                    optics.verify_package_manifest(payload, "uvc", output)

    def test_noncanonical_duplicate_and_extra_json_are_rejected(self) -> None:
        for mutation in ("pretty", "duplicate", "extra"):
            with self.subTest(mutation=mutation):
                parent = self.root / mutation
                payload = self._make_payload(parent)
                output = parent / "uvc.manifest.json"
                optics.write_package_manifest(payload, "uvc", output)
                value = json.loads(output.read_text(encoding="utf-8"))
                if mutation == "pretty":
                    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
                elif mutation == "duplicate":
                    raw = output.read_text(encoding="utf-8")
                    output.write_text(
                        raw[:-1] + ',"schema_version":"rootscope.final_optics.evidence_manifest.v1"}',
                        encoding="utf-8",
                    )
                else:
                    value["unexpected"] = True
                    output.write_bytes(optics._canonical_bytes(value))
                with self.assertRaises(optics.EvidenceError):
                    optics.verify_package_manifest(payload, "uvc", output)

    def test_manifest_kind_mismatch_is_rejected(self) -> None:
        payload = self._make_payload(self.root / "kind")
        output = self.root / "kind" / "uvc.manifest.json"
        optics.write_package_manifest(payload, "uvc", output)
        value = json.loads(output.read_text(encoding="utf-8"))
        value["evidence_kind"] = "paper"
        output.write_bytes(optics._canonical_bytes(value))
        with self.assertRaisesRegex(optics.EvidenceError, "kind"):
            optics.verify_package_manifest(payload, "uvc", output)

    def test_unsafe_windows_and_non_normalized_paths_are_rejected(self) -> None:
        unsafe = (
            "../escape.txt",
            "folder\\file.txt",
            "CON/photo.jpg",
            "photo.jpg:stream",
            "trail./file.txt",
            "e\u0301.txt",
            "a//b.txt",
        )
        for relative in unsafe:
            with self.subTest(relative=relative):
                with self.assertRaises(optics.EvidenceError):
                    optics._validate_relative_path(relative)

    def test_symlink_payload_entry_is_rejected_when_supported(self) -> None:
        payload = self.root / "symlink" / "payload"
        payload.mkdir(parents=True)
        target = self.root / "symlink" / "target.txt"
        target.write_bytes(b"target")
        link = payload / "link.txt"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            fallback = payload / "simulated_reparse.txt"
            fallback.write_bytes(b"fallback")
            original = optics._is_reparse

            def mark_regular_file_as_reparse(info: os.stat_result) -> bool:
                return stat.S_ISREG(info.st_mode) or original(info)

            with mock.patch.object(optics, "_is_reparse", side_effect=mark_regular_file_as_reparse):
                with self.assertRaisesRegex(optics.EvidenceError, "symlink|reparse"):
                    optics.build_package_manifest(payload, "uvc")
            return
        with self.assertRaisesRegex(optics.EvidenceError, "symlink|reparse"):
            optics.build_package_manifest(payload, "uvc")

    def test_post_hash_race_is_detected(self) -> None:
        payload = self.root / "race" / "payload"
        payload.mkdir(parents=True)
        artifact = payload / "artifact.bin"
        artifact.write_bytes(b"before")
        original = optics._hash_regular_file
        mutated = False

        def mutate_after_hash(path: Path, context: str = "evidence file"):
            nonlocal mutated
            result = original(path, context)
            if not mutated:
                path.write_bytes(b"after!")
                mutated = True
            return result

        with mock.patch.object(optics, "_hash_regular_file", side_effect=mutate_after_hash):
            with self.assertRaisesRegex(optics.EvidenceError, "changed after hashing"):
                optics.build_package_manifest(payload, "uvc")

    def test_complete_synthetic_bundle_passes_only_structural_preflight(self) -> None:
        bundle, _, _ = self._make_bundle()
        report = optics.preflight_bundle(bundle)
        self.assertEqual(
            "STRUCTURAL_PREFLIGHT_PASS_HUMAN_PHYSICAL_ATTESTATION_ONLY",
            report["status"],
        )
        self.assertTrue(report["byte_integrity_verified"])
        self.assertRegex(report["optical_domain_root"], r"^[0-9a-f]{64}$")
        self.assertTrue(all(value is False for value in report["claims"].values()))
        self.assertEqual(0, report["error_count"])

    def test_missing_bundle_fails_closed_without_creating_anything(self) -> None:
        bundle = self.root / "does-not-exist"
        report = optics.preflight_bundle(bundle)
        self.assertEqual("FINAL_OPTICS_NOT_READY", report["status"])
        self.assertFalse(report["byte_integrity_verified"])
        self.assertIsNone(report["optical_domain_root"])
        self.assertGreater(report["error_count"], 0)
        self.assertFalse(bundle.exists())

    def test_missing_one_evidence_package_fails_closed(self) -> None:
        bundle, _, _ = self._make_bundle()
        shutil.rmtree(bundle / "evidence" / "lighting")
        report = optics.preflight_bundle(bundle)
        self.assertEqual("FINAL_OPTICS_NOT_READY", report["status"])
        self.assertFalse(report["byte_integrity_verified"])
        self.assertIsNone(report["optical_domain_root"])
        self.assertTrue(any(item["code"] == "EVIDENCE_LAYOUT_INVALID" for item in report["errors"]))

    def test_receipt_root_mismatch_is_detected(self) -> None:
        bundle, _, receipt = self._make_bundle()
        receipt["evidence_roots"]["uvc"] = "f" * 64
        self._rewrite_receipt(bundle, receipt)
        report = optics.preflight_bundle(bundle)
        self.assertEqual("FINAL_OPTICS_NOT_READY", report["status"])
        self.assertTrue(
            any(item["code"] == "RECEIPT_EVIDENCE_ROOT_MISMATCH" for item in report["errors"])
        )

    def test_approval_file_tamper_is_detected(self) -> None:
        bundle, _, _ = self._make_bundle()
        (bundle / "approvals" / "hardware.approval").write_bytes(b"tampered approval")
        report = optics.preflight_bundle(bundle)
        self.assertEqual("FINAL_OPTICS_NOT_READY", report["status"])
        self.assertTrue(
            any(item["code"] == "RECEIPT_APPROVAL_HASH_MISMATCH" for item in report["errors"])
        )

    def test_duplicate_signers_signed_false_and_extra_fields_are_rejected(self) -> None:
        for mutation in ("duplicate_signers", "signed_false", "extra"):
            with self.subTest(mutation=mutation):
                local = self.root / mutation
                local.mkdir()
                prior_root = self.root
                self.root = local
                try:
                    bundle, _, receipt = self._make_bundle()
                finally:
                    self.root = prior_root
                if mutation == "duplicate_signers":
                    aliases = ("Same Person", " same  person ", "SAME PERSON")
                    for entry, signer in zip(receipt["signed_roles"].values(), aliases, strict=True):
                        entry["signer"] = signer
                elif mutation == "signed_false":
                    receipt["signed_roles"]["hardware"]["signed"] = False
                else:
                    receipt["unexpected"] = True
                self._rewrite_receipt(bundle, receipt)
                report = optics.preflight_bundle(bundle)
                self.assertEqual("FINAL_OPTICS_NOT_READY", report["status"])
                self.assertTrue(any(item["code"] == "RECEIPT_INVALID" for item in report["errors"]))

    def test_receipt_canonical_root_matches_dataset_auditor(self) -> None:
        bundle, _, receipt = self._make_bundle()
        sys.path.insert(0, str(ROOTSCOPE_TRAINING))
        try:
            from dataset_audit import _canonical_json_sha256
        finally:
            sys.path.pop(0)
        report = optics.preflight_bundle(bundle)
        self.assertEqual(_canonical_json_sha256(receipt), report["optical_domain_root"])

    def test_cli_missing_bundle_returns_two_and_writes_no_receipt(self) -> None:
        bundle = self.root / "missing-cli-bundle"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = optics.main(["preflight", "--bundle-dir", str(bundle)])
        self.assertEqual(2, exit_code)
        report = json.loads(stdout.getvalue())
        self.assertEqual("FINAL_OPTICS_NOT_READY", report["status"])
        self.assertFalse(bundle.exists())
        self.assertFalse((bundle / "final_optics_receipt.json").exists())

    def test_preflight_output_inside_evidence_is_rejected(self) -> None:
        bundle, _, _ = self._make_bundle()
        output = bundle / "evidence" / "preflight.json"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exit_code = optics.main(
                [
                    "preflight",
                    "--bundle-dir",
                    str(bundle),
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(2, exit_code)
        self.assertFalse(output.exists())

    def test_receipt_inside_evidence_payload_is_rejected(self) -> None:
        bundle, _, _ = self._make_bundle()
        receipt_inside = bundle / "evidence" / "uvc" / "receipt.json"
        receipt_inside.write_bytes((bundle / "final_optics_receipt.json").read_bytes())
        report = optics.preflight_bundle(bundle, receipt_inside)
        self.assertEqual("FINAL_OPTICS_NOT_READY", report["status"])
        self.assertIsNone(report["optical_domain_root"])
        self.assertTrue(any(item["code"] == "RECEIPT_LOCATION_INVALID" for item in report["errors"]))


if __name__ == "__main__":
    unittest.main()
