from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOTSCOPE = Path(__file__).resolve().parents[1]
ADVENTUREX = ROOTSCOPE.parent
if str(ADVENTUREX) not in sys.path:
    sys.path.insert(0, str(ADVENTUREX))

from tools.release.build_rootscope_x5_field_bundle_v2 import (  # noqa: E402
    BPU_ARCHIVE,
    BPU_ID,
    PILLOW_SHA256,
    build_bpu_support,
    validate_selection_receipt,
)


def _load_installer():
    path = ROOTSCOPE / "deploy/x5/scripts/install_field_bundle_v2.py"
    spec = importlib.util.spec_from_file_location("field_bundle_v2_installer_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = _load_installer()


class BpuSupportOnlyTests(unittest.TestCase):
    def test_final_selection_is_null_and_hash_bound_to_both_searches(self) -> None:
        validated = validate_selection_receipt(
            ADVENTUREX / "evidence/rootscope_seed17_bpu_field_selection_20260717.json",
            ADVENTUREX,
        )
        payload = validated["payload"]
        self.assertIsNone(payload["selection"]["selected_variant"])
        self.assertIsNone(payload["selection"]["selected_bin"])
        self.assertFalse(payload["selection"]["all_predeclared_replay_gates_passed"])
        self.assertEqual(len(payload["source_results"]), 2)

    def test_bpu_component_is_deterministic_no_bin_and_pillow_only(self) -> None:
        selection = ADVENTUREX / "evidence/rootscope_seed17_bpu_field_selection_20260717.json"
        with tempfile.TemporaryDirectory(dir=ADVENTUREX / "tmp") as temporary:
            root = Path(temporary)
            first = build_bpu_support(ADVENTUREX, root / "first", selection)
            second = build_bpu_support(ADVENTUREX, root / "second", selection)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertFalse(first["bpu_binary_included"])
            with tarfile.open(root / "first" / BPU_ARCHIVE, mode="r:") as archive:
                members = [item.name for item in archive.getmembers() if item.isfile()]
                self.assertFalse(any(name.lower().endswith(".bin") for name in members))
                wheels = [name for name in members if name.lower().endswith(".whl")]
                self.assertEqual(len(wheels), 1)
                self.assertTrue(Path(wheels[0]).name.lower().startswith("pillow-"))
                manifest = json.loads(
                    archive.extractfile(f"{BPU_ID}/component_manifest.json")
                    .read()
                    .decode("utf-8")
                )
                self.assertEqual(manifest["pillow_wheel"]["sha256"], PILLOW_SHA256)
                self.assertFalse(manifest["dependency_policy"]["numpy_wheel_included"])
                self.assertFalse(manifest["dependency_policy"]["core_v1_venv_allowed"])
                self.assertEqual(
                    manifest["dependency_policy"]["local_wheel_install_allowlist"],
                    ["Pillow"],
                )


class FieldInstallerContractTests(unittest.TestCase):
    def test_mixed_fresh_board_stdout_extracts_only_expected_receipt(self) -> None:
        text = (
            "pip install progress {not-json}\n"
            + json.dumps(
                {
                    "log": {
                        "schema": "rootscope.x5-offline-user-install-receipt.v1",
                        "status": "MALICIOUS_NESTED_OBJECT",
                    }
                }
            )
            + "\n"
            + json.dumps({"schema": "unrelated.v1", "status": "PASS"})
            + "\nmore output\n"
            + json.dumps(
                {
                    "schema": "rootscope.x5-offline-user-install-receipt.v1",
                    "status": "PASS_LOCAL_AARCH64_CPU_SMOKE_NOT_X5_QUALIFIED",
                }
            )
            + "\n"
        )
        receipt = installer._extract_expected_receipt(
            text, "rootscope.x5-offline-user-install-receipt.v1"
        )
        self.assertEqual(
            receipt["status"], "PASS_LOCAL_AARCH64_CPU_SMOKE_NOT_X5_QUALIFIED"
        )
        with self.assertRaisesRegex(RuntimeError, "0 receipts"):
            installer._extract_expected_receipt(text, "missing.v1")
        duplicated = (
            json.dumps(
                {"schema": "fixture.duplicate.v1", "status": "FIRST"}, indent=2
            )
            + "\n"
            + json.dumps(
                {"schema": "fixture.duplicate.v1", "status": "SECOND"}, indent=2
            )
            + "\n"
        )
        with self.assertRaisesRegex(RuntimeError, "2 receipts"):
            installer._extract_expected_receipt(duplicated, "fixture.duplicate.v1")

    def test_fixed_disk_receipt_must_equal_stdout_receipt(self) -> None:
        expected = {"schema": "fixture.v1", "status": "PASS"}
        with tempfile.TemporaryDirectory(dir=ADVENTUREX / "tmp") as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(
                installer._require_disk_receipt_matches(expected, path, "fixture.v1"),
                expected,
            )
            with self.assertRaisesRegex(RuntimeError, "divergence"):
                installer._require_disk_receipt_matches(
                    {"schema": "fixture.v1", "status": "FAIL"}, path, "fixture.v1"
                )

    def test_installer_source_has_no_network_device_or_actuator_import(self) -> None:
        source = (
            ROOTSCOPE / "deploy/x5/scripts/install_field_bundle_v2.py"
        ).read_text(encoding="utf-8")
        for token in (
            "import requests",
            "import urllib",
            "import socket",
            "import serial",
            "import cv2",
            "import hobot_dnn",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, source)
        self.assertIn("--verify-only", source)
        self.assertIn("Linux/aarch64", source)
        self.assertIn("CPython 3.10", source)

    def test_current_non_target_host_fails_install_gate(self) -> None:
        if sys.platform.startswith("linux") and Path("/usr/bin/python3").exists():
            self.skipTest("host may be the target; exact target is covered statically")
        with self.assertRaisesRegex(RuntimeError, "RDK Linux/aarch64"):
            installer.assert_supported_install_host()


if __name__ == "__main__":
    unittest.main()
