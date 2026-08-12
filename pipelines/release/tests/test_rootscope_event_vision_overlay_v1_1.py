from __future__ import annotations

import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.release import audit_rootscope_event_vision_overlay_v1_1 as auditor
from tools.release import build_rootscope_event_vision_overlay_v1_1 as builder
from tools.release import verify_rootscope_event_vision_overlay_v1_1 as preflight


ROOT = Path(__file__).resolve().parents[3]


class EventVisionOverlayV11ContractTests(unittest.TestCase):
    def test_identity_and_formal_output_is_immutable(self) -> None:
        self.assertEqual(
            builder.OVERLAY_ID,
            "rootscope_event_vision_overlay_v1_1",
        )
        self.assertEqual(builder.SCHEMA, "rootscope.event-vision-overlay.v1_1")
        output = ROOT / builder.OUTPUT_RELATIVE
        if output.exists():
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                builder.build_release(ROOT)

    def test_exact_allowlists_and_entrypoint_match(self) -> None:
        builder_paths = {spec.package_relative for spec in builder.SOURCE_SPECS}
        self.assertEqual(builder_paths, set(auditor.EXPECTED_SOURCE_MAP))
        self.assertEqual(
            builder_paths | {"release_manifest.json", "SHA256SUMS"},
            auditor.EXPECTED_FILES,
        )
        self.assertEqual(auditor.EXPECTED_FILES, preflight.EXPECTED_FILES)
        self.assertEqual(len(builder_paths), 29)
        self.assertEqual(len(auditor.EXPECTED_FILES), 31)
        modes = {
            spec.package_relative: spec.mode for spec in builder.SOURCE_SPECS
        }
        self.assertEqual(modes[builder.HELPER_PACKAGE], 0o755)
        self.assertEqual(modes[builder.LEGACY_HELPER_PACKAGE], 0o644)

    def test_source_closure_is_current_and_contains_no_large_payloads(self) -> None:
        entries = builder.load_source_entries(ROOT)
        paths = {entry.path for entry in entries}
        self.assertEqual(paths, set(auditor.EXPECTED_SOURCE_MAP))
        self.assertFalse(any(path.endswith((".tar", ".pdf", ".onnx")) for path in paths))
        self.assertFalse(any("unknown" in path.lower() for path in paths))
        self.assertFalse(any("__pycache__" in path or path.endswith(".pyc") for path in paths))
        local = {
            entry.path: entry.sha256 for entry in entries
        }
        self.assertEqual(
            local["rootscope/app/vision/uvc_card_capture.py"],
            builder._base.sha256_file(
                ROOT / "rootscope/app/vision/uvc_card_capture.py"
            ),
        )
        self.assertEqual(
            local["rootscope/app/omega_vision/uvc_card_frontend.py"],
            builder._base.sha256_file(
                ROOT / "rootscope/app/omega_vision/uvc_card_frontend.py"
            ),
        )

    def test_manifest_preserves_zero_authority_and_null_bpu_selection(self) -> None:
        entries = builder.add_generated_entries(
            builder.load_source_entries(ROOT)
        )
        manifest = json.loads(
            next(
                item.data
                for item in entries
                if item.path == "release_manifest.json"
            ).decode("utf-8")
        )
        self.assertEqual(manifest["overlay_id"], builder.OVERLAY_ID)
        self.assertEqual(manifest["schema"], builder.SCHEMA)
        self.assertEqual(
            manifest["entrypoints"]["read_only_preflight"],
            builder.HELPER_PACKAGE,
        )
        self.assertIsNone(manifest["qualification"]["selected_bin"])
        self.assertTrue(
            all(value is False for value in manifest["authority"].values())
        )
        self.assertFalse(
            manifest["dependency_boundary"]["model_binary_bundled"]
        )
        self.assertFalse(
            manifest["dependency_boundary"]["plant_bpu_binary_bundled"]
        )

    def test_runbook_paths_are_overlay_relative_and_v11_bound(self) -> None:
        text = (
            ROOT
            / "rootscope/deploy/x5/ROOTSCOPE_UVC_CARD_FRONTEND_RUNBOOK_ZH.md"
        ).read_text("utf-8")
        self.assertIn("rootscope_event_vision_overlay_v1_1", text)
        self.assertIn('APP_ROOT="$OVERLAY_ROOT/rootscope"', text)
        self.assertIn(
            'PRINT_MANIFEST="$OVERLAY_ROOT/output/pdf/'
            'RootScope_A4_four_up_field_cards_20260723_manifest.json"',
            text,
        )

    def test_deterministic_ustar_writer_refuses_overwrite(self) -> None:
        entries = [
            builder.PackageEntry(
                path="tools/verify_rootscope_event_vision_overlay_v1_1.py",
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
                self.assertTrue(
                    all(
                        member.mtime == 0
                        and member.uid == 0
                        and member.gid == 0
                        for member in archive
                    )
                )
            with self.assertRaises(FileExistsError):
                builder.write_deterministic_ustar(first, entries)


if __name__ == "__main__":
    unittest.main()
