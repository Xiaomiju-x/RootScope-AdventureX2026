from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.release import audit_rootscope_event_vision_overlay_v1 as auditor
from tools.release import build_rootscope_event_vision_overlay_v1 as builder
from tools.release import verify_rootscope_event_vision_overlay_v1 as preflight


ROOT = Path(__file__).resolve().parents[3]


class EventVisionOverlayContractTests(unittest.TestCase):
    def test_formal_output_is_absent_or_builder_refuses_overwrite(self) -> None:
        output = ROOT / builder.OUTPUT_RELATIVE
        if output.exists():
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                builder.build_release(ROOT)
        else:
            self.assertFalse(output.exists())

    def test_builder_and_independent_auditor_exact_allowlists_match(self) -> None:
        builder_paths = {spec.package_relative for spec in builder.SOURCE_SPECS}
        self.assertEqual(builder_paths, set(auditor.EXPECTED_SOURCE_MAP))
        self.assertEqual(
            builder_paths | {"release_manifest.json", "SHA256SUMS"},
            auditor.EXPECTED_FILES,
        )
        self.assertEqual(auditor.EXPECTED_FILES, preflight.EXPECTED_FILES)
        self.assertEqual(len(builder_paths), 28)
        self.assertEqual(len(auditor.EXPECTED_FILES), 30)

    def test_source_import_and_asset_closure_passes_without_extra_archives(self) -> None:
        entries = builder.load_source_entries(ROOT)
        paths = {entry.path for entry in entries}
        self.assertEqual(paths, set(auditor.EXPECTED_SOURCE_MAP))
        self.assertFalse(any(path.endswith(".tar") for path in paths))
        self.assertFalse(any(path.endswith(".pdf") for path in paths))
        self.assertFalse(any("__pycache__" in path or path.endswith(".pyc") for path in paths))
        self.assertNotIn(
            "rootscope/deploy/x5/models/rootscope_seed17_cpu_experimental_opset11.onnx",
            paths,
        )

    def test_generated_manifest_has_exact_reference_only_boundaries(self) -> None:
        sources = builder.load_source_entries(ROOT)
        generated = builder.add_generated_entries(sources)
        manifest_entry = next(
            entry for entry in generated if entry.path == "release_manifest.json"
        )
        sums_entry = next(entry for entry in generated if entry.path == "SHA256SUMS")
        manifest = json.loads(manifest_entry.data.decode("utf-8"))
        self.assertEqual(
            manifest["immutable_references"]["x5_field_bundle_v2"]["sha256"],
            builder.V2_REFERENCE["sha256"],
        )
        self.assertEqual(
            manifest["immutable_references"]["omega_v3_delta_candidate"]["sha256"],
            builder.OMEGA_REFERENCE["sha256"],
        )
        for reference in manifest["immutable_references"].values():
            self.assertFalse(reference["bundled_in_overlay"])
            self.assertTrue(reference["immutable_reference_only"])
        self.assertIsNone(manifest["qualification"]["selected_bin"])
        self.assertTrue(all(value is False for value in manifest["authority"].values()))
        runtime_capsule = manifest["frozen_runtime_asset_contracts"][
            "runtime_capsule"
        ]
        self.assertEqual(
            runtime_capsule["sha256"],
            "1e839948e466895e0b416aababb39402380463f9ce125c8aeb59f4087191cb97",
        )
        self.assertEqual(runtime_capsule["bytes"], 2765)
        self.assertFalse(runtime_capsule["bundled_in_overlay"])
        self.assertTrue(
            runtime_capsule["reconstruction"][
                "runtime_value_must_not_be_self_promoted_to_expected"
            ]
        )
        sums = auditor.parse_sums(sums_entry.data)
        self.assertEqual(set(sums), auditor.EXPECTED_FILES - {"SHA256SUMS"})
        for entry in generated:
            if entry.path != "SHA256SUMS":
                self.assertEqual(sums[entry.path], entry.sha256)

    def test_camera_contract_and_print_manifest_are_frozen(self) -> None:
        camera = json.loads((ROOT / builder.CAMERA_CONTRACT_SOURCE).read_text("utf-8"))
        self.assertEqual(camera["camera"]["usb_vid_pid"], "32e6:9228")
        self.assertEqual(camera["camera"]["usb_serial"], "202604081837")
        self.assertEqual(
            camera["camera"]["stable_by_id_path"],
            (
                "/dev/v4l/by-id/"
                "usb-Web_Camera_Web_Camera_202604081837-video-index0"
            ),
        )
        modes = {
            (item["width"], item["height"], item["fourcc"], item["fps"])
            for item in camera["capture_modes"]
        }
        self.assertEqual(
            modes,
            {(1920, 1080, "MJPG", 30), (1280, 720, "MJPG", 30)},
        )
        print_path = ROOT / builder.PRINT_MANIFEST_SOURCE
        self.assertEqual(
            hashlib.sha256(print_path.read_bytes()).hexdigest(),
            builder.PRINT_MANIFEST_SHA256,
        )

    def test_tiny_ustar_writer_is_deterministic_and_refuses_overwrite(self) -> None:
        entries = [
            builder.PackageEntry(
                path="a/one.txt",
                data=b"one\n",
                mode=0o644,
                category="fixture",
                source_relative=None,
            ),
            builder.PackageEntry(
                path="tools/verify_rootscope_event_vision_overlay_v1.py",
                data=b"#!/usr/bin/env python3\n",
                mode=0o755,
                category="fixture",
                source_relative=None,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.tar"
            second = Path(directory) / "second.tar"
            builder.write_deterministic_ustar(first, entries)
            builder.write_deterministic_ustar(second, entries)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with tarfile.open(first, "r:") as archive:
                members = list(archive)
                self.assertTrue(all(member.mtime == 0 for member in members))
                self.assertTrue(all(member.uid == member.gid == 0 for member in members))
                order = [member.name for member in members]
                directories = {member.name for member in members if member.isdir()}
            rebuilt = Path(directory) / "rebuilt.tar"
            auditor.rebuild_archive(
                rebuilt,
                {entry.path: entry.data for entry in entries},
                order,
                directories,
            )
            self.assertEqual(first.read_bytes(), rebuilt.read_bytes())
            with self.assertRaises(FileExistsError):
                builder.write_deterministic_ustar(first, entries)

    def test_checksum_parser_rejects_duplicate_or_unsafe_paths(self) -> None:
        digest = "a" * 64
        with self.assertRaisesRegex(auditor.AuditError, "duplicate"):
            auditor.parse_sums(
                f"{digest}  a.txt\n{digest}  a.txt\n".encode("ascii")
            )
        with self.assertRaisesRegex(auditor.AuditError, "unsafe"):
            auditor.parse_sums(f"{digest}  ../escape.txt\n".encode("ascii"))

    def test_reference_constants_are_exact(self) -> None:
        self.assertEqual(builder.V2_REFERENCE["bytes"], 696_832_000)
        self.assertEqual(
            builder.V2_REFERENCE["sha256"],
            "e6627685170252004d118bf77a690a9f89ad3afa274910697554d0f5cc8c3ebb",
        )
        self.assertEqual(builder.OMEGA_REFERENCE["bytes"], 665_600)
        self.assertEqual(
            builder.OMEGA_REFERENCE["sha256"],
            "c910f4d2e002ccdbd5643fa47f300ade8e56af8ad1c1a2a04fa4e4a0a0fab881",
        )


if __name__ == "__main__":
    unittest.main()
