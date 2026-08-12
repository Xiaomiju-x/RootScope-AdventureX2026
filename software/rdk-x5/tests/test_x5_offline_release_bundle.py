from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOTSCOPE = Path(__file__).resolve().parents[1]
ADVENTUREX = ROOTSCOPE.parent
RELEASE_TOOLS = ADVENTUREX / "tools/release"
sys.path.insert(0, str(ADVENTUREX))

from tools.release.build_rootscope_x5_offline_release import (  # noqa: E402
    CORE_STATUS,
    REGISTRY_RECEIPT_SHA256,
    REGISTRY_SHA256,
    build_deterministic_tar,
    collect_core_sources,
    sha256_file,
)


def _load_script(name: str):
    path = ROOTSCOPE / "deploy/x5/scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


offline_installer = _load_script("install_offline_core.py")
capture_once = _load_script("capture_and_dual_path_once.py")


class ReleaseBuilderTests(unittest.TestCase):
    def test_core_allowlist_contains_frozen_demo_and_no_dataset_or_bpu(self) -> None:
        entries = collect_core_sources(ADVENTUREX)
        paths = {entry.package_path: entry for entry in entries}
        self.assertIn(
            "rootscope/app/vision/known_card_template_registry.frozen.experimental.json",
            paths,
        )
        self.assertEqual(
            sha256_file(
                paths[
                    "rootscope/app/vision/known_card_template_registry.frozen.experimental.json"
                ].source
            ),
            REGISTRY_SHA256,
        )
        self.assertEqual(
            sha256_file(
                paths[
                    "rootscope/evidence/rootscope_demo_template_registry_receipt_20260717.json"
                ].source
            ),
            REGISTRY_RECEIPT_SHA256,
        )
        template_paths = {
            path
            for path in paths
            if path.startswith("rootscope/app/vision/known_card_templates/")
        }
        self.assertEqual(len(template_paths), 3)
        self.assertFalse(any("/datasets/" in path.lower() for path in paths))
        self.assertFalse(any(path.lower().endswith(".bin") for path in paths))
        self.assertNotIn(
            "rootscope/deploy/x5/systemd/rootscope-edge.service", paths
        )

    def test_deterministic_tar_primitive_is_byte_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"frozen-source\x00payload")
            kwargs = {
                "file_entries": [("release/payload/source.bin", source, 0o644)],
                "byte_entries": [("release/entry.sh", b"#!/bin/sh\n", 0o755)],
            }
            first = build_deterministic_tar(root / "first.tar", **kwargs)
            second = build_deterministic_tar(root / "second.tar", **kwargs)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual((root / "first.tar").read_bytes(), (root / "second.tar").read_bytes())

    def test_frozen_registry_has_three_positive_classes_and_no_unknown(self) -> None:
        registry = json.loads(
            (
                ROOTSCOPE
                / "app/vision/known_card_template_registry.frozen.experimental.json"
            ).read_text(encoding="utf-8")
        )
        classes = [item["class_name"] for item in registry["templates"]]
        self.assertEqual(set(classes), {"grass_clump", "low_shrub", "young_tree"})
        self.assertEqual(len(classes), 3)
        self.assertNotIn("unknown", classes)


class OfflineInstallerTests(unittest.TestCase):
    def test_target_gate_is_exact_and_fail_closed(self) -> None:
        offline_installer.assert_supported_host(
            system="Linux",
            machine="aarch64",
            version=(3, 10),
            implementation="CPython",
        )
        bad = (
            ("Windows", "AMD64", (3, 12), "CPython"),
            ("Linux", "x86_64", (3, 10), "CPython"),
            ("Linux", "aarch64", (3, 11), "CPython"),
            ("Linux", "aarch64", (3, 10), "PyPy"),
        )
        for system, machine, version, implementation in bad:
            with self.subTest(system=system, machine=machine, version=version):
                with self.assertRaisesRegex(RuntimeError, "requires Linux/aarch64/CPython 3.10"):
                    offline_installer.assert_supported_host(
                        system=system,
                        machine=machine,
                        version=version,
                        implementation=implementation,
                    )

    def test_runtime_config_renderer_preserves_all_false_boundaries(self) -> None:
        config = offline_installer.render_runtime_config(
            ROOTSCOPE / "deploy/x5/capsule_config.seed17_cpu_experimental.json",
            ROOTSCOPE,
            Path(sys.executable),
        )
        self.assertEqual(config["status"], offline_installer.CAPSULE_STATUS)
        self.assertFalse(any(config["authority"].values()))
        self.assertFalse(config["model"]["model_candidate"])
        self.assertFalse(config["model"]["model_qualified"])
        self.assertFalse(config["model"]["bpu_ready"])
        self.assertFalse(config["llm"]["enabled"])
        self.assertEqual(config["project_root"], str(ROOTSCOPE.resolve()))
        self.assertEqual(config["python_executable"], str(Path(sys.executable).resolve()))

    def test_package_verifier_detects_exact_coverage_and_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            payload = package / "rootscope/payload.txt"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"payload")
            record = {
                "path": "rootscope/payload.txt",
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "mode": "0644",
                "category": "fixture",
            }
            manifest = {
                "schema": offline_installer.RELEASE_SCHEMA,
                "release_id": offline_installer.RELEASE_ID,
                "status": CORE_STATUS,
                "formal_flags": {"model_qualified": False, "x5_validated": False},
                "authority": {"execution_authority": False, "physical_authority": False},
                "files": [record],
            }
            manifest_bytes = (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            (package / "release_manifest.json").write_bytes(manifest_bytes)
            (package / "SHA256SUMS").write_text(
                f"{record['sha256']}  rootscope/payload.txt\n"
                f"{hashlib.sha256(manifest_bytes).hexdigest()}  release_manifest.json\n",
                encoding="utf-8",
            )
            receipt = offline_installer.verify_package(package)
            self.assertEqual(receipt["status"], "PASS_PACKAGE_HASHES_NOT_X5_QUALIFIED")
            self.assertFalse(receipt["execution_authority"])
            payload.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size mismatch|hash mismatch"):
                offline_installer.verify_package(package)

    def test_installer_has_no_network_service_or_device_path(self) -> None:
        source = (
            ROOTSCOPE / "deploy/x5/scripts/install_offline_core.py"
        ).read_text(encoding="utf-8").lower()
        for forbidden in (
            "socket.socket(",
            "requests.",
            "urllib.",
            "http.client",
            'subprocess.run(["systemctl"',
            "videoCapture(".lower(),
            "serial.serial(",
            "hobot_dnn",
            "hbm_runtime",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertIn("--no-index", (ROOTSCOPE / "deploy/x5/scripts/install_cpu_venv_candidate.sh").read_text(encoding="utf-8"))


class ExplicitOneShotCaptureTests(unittest.TestCase):
    def test_capture_script_requires_explicit_device_and_has_no_enumeration(self) -> None:
        source = (
            ROOTSCOPE / "deploy/x5/scripts/capture_and_dual_path_once.py"
        ).read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--device", required=True', source)
        self.assertIn("cv2.CAP_V4L2", source)
        self.assertIn("cv2.VideoCapture(str(configured)", source)
        for forbidden in ("glob(", "iterdir(", "listdir(", "VideoCapture(0", "serial.", "systemctl"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_capture_device_validation_rejects_non_linux_host(self) -> None:
        if sys.platform.startswith("linux"):
            self.skipTest("host is Linux; static contract test covers explicit path")
        with self.assertRaisesRegex(ValueError, "requires Linux"):
            capture_once.validate_explicit_device("/dev/video0")


if __name__ == "__main__":
    unittest.main()
