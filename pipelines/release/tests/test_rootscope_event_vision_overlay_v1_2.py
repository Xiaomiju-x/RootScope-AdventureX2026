from __future__ import annotations

import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.release import audit_rootscope_event_vision_overlay_v1_2 as auditor
from tools.release import build_rootscope_event_vision_overlay_v1_2 as builder
from tools.release import verify_rootscope_event_vision_overlay_v1_2 as preflight


ROOT = Path(__file__).resolve().parents[3]


class EventVisionOverlayV12ContractTests(unittest.TestCase):
    def test_identity_and_formal_output_is_immutable(self) -> None:
        self.assertEqual(
            builder.OVERLAY_ID,
            "rootscope_event_vision_overlay_v1_2",
        )
        self.assertEqual(builder.SCHEMA, "rootscope.event-vision-overlay.v1_2")
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

    def test_source_closure_has_no_large_or_unregistered_payloads(self) -> None:
        entries = builder.load_source_entries(ROOT)
        paths = {entry.path for entry in entries}
        self.assertEqual(paths, set(auditor.EXPECTED_SOURCE_MAP))
        self.assertFalse(
            any(path.endswith((".tar", ".pdf", ".onnx")) for path in paths)
        )
        self.assertFalse(any("unknown" in path.lower() for path in paths))
        self.assertFalse(
            any("__pycache__" in path or path.endswith(".pyc") for path in paths)
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

    def test_runbook_semantic_deployment_paths_and_boundaries(self) -> None:
        entries = builder.load_source_entries(ROOT)
        builder.validate_runbook_semantics(entries)
        by_path = {item.path: item for item in entries}
        capture = by_path[builder.CAPTURE_RUNBOOK_PACKAGE].data.decode("utf-8")
        frontend = by_path[builder.FRONTEND_RUNBOOK_PACKAGE].data.decode("utf-8")
        for text in (capture, frontend):
            self.assertNotIn("/opt/rootscope/rootscope", text)
            self.assertNotIn("rootscope_inputs", text)
            self.assertNotIn("rootscope_event_vision_overlay_v1_1", text)
            self.assertIn("rootscope_event_vision_overlay_v1_2", text)
            self.assertIn('APP_ROOT="$OVERLAY_ROOT/rootscope"', text)
            self.assertIn(
                'PRINT_MANIFEST="$OVERLAY_ROOT/output/pdf/'
                'RootScope_A4_four_up_field_cards_20260723_manifest.json"',
                text,
            )
            self.assertIn("PYTHONDONTWRITEBYTECODE=1", text)
        self.assertIn("USB_VID='32e6'", frontend)
        self.assertIn("USB_PID='9228'", frontend)
        self.assertIn("USB_SERIAL='202604081837'", frontend)
        self.assertIn("以下命令只用于 PC 工作区开发验证", frontend)
        self.assertNotIn("在新的、单独审计的增量包形成前", frontend)
        self.assertIn("backend 构造成功后", capture)
        self.assertIn(
            "不会生成一份声称完成第三次身份核验的成功 manifest",
            capture,
        )
        self.assertIn(
            'PY="$HOME/.local/share/rootscope-field-v2/core_v1/venvs/'
            'rootscope_x5_offline_core_v1/bin/python3"',
            capture,
        )
        self.assertIn('OUT_ROOT="$HOME/rootscope_event_capture"', capture)

    def test_runbook_semantic_gate_rejects_obsolete_path(self) -> None:
        entries = list(builder.load_source_entries(ROOT))
        index = next(
            i
            for i, item in enumerate(entries)
            if item.path == builder.CAPTURE_RUNBOOK_PACKAGE
        )
        original = entries[index]
        entries[index] = builder.PackageEntry(
            path=original.path,
            data=original.data + b"\n/opt/rootscope/rootscope\n",
            mode=original.mode,
            category=original.category,
            source_relative=original.source_relative,
        )
        with self.assertRaisesRegex(ValueError, "obsolete deployment path"):
            builder.validate_runbook_semantics(entries)

    def test_deterministic_ustar_writer_refuses_overwrite(self) -> None:
        entries = [
            builder.PackageEntry(
                path="tools/verify_rootscope_event_vision_overlay_v1_2.py",
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
