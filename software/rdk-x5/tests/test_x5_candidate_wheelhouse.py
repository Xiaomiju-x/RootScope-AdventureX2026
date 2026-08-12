from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WHEELHOUSE = ROOT / "deploy" / "x5" / "wheelhouse"


class CandidateWheelhouseTests(unittest.TestCase):
    def test_real_candidate_wheelhouse_audit_passes_without_qualification(self) -> None:
        import importlib.util

        module_path = WHEELHOUSE / "audit_candidate_wheelhouse.py"
        spec = importlib.util.spec_from_file_location("candidate_wheel_audit", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        result = module.audit_manifest(WHEELHOUSE / "candidate_cp310_aarch64_manifest.json")
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "PASS_NOT_X5_QUALIFIED")
        self.assertFalse(result["x5_validated"])
        self.assertFalse(result["wheelhouse_qualified"])
        self.assertFalse(result["hardware_touched"])

    def test_manifest_has_no_foreign_binary_or_qualification_claim(self) -> None:
        payload = json.loads((WHEELHOUSE / "candidate_cp310_aarch64_manifest.json").read_text(encoding="utf-8"))
        filenames = [item["filename"].lower() for item in payload["wheels"]]
        self.assertEqual(len(filenames), 11)
        self.assertFalse(any("x86_64" in name or "win_amd64" in name or "macosx" in name for name in filenames))
        self.assertTrue(all(payload["claims"][name] is False for name in (
            "exact_twin_install_tested",
            "rdk_x5_import_tested",
            "golden_preprocess_replayed_on_x5",
            "onnx_replayed_on_x5",
            "wheelhouse_qualified",
            "x5_ready",
        )))

    def test_tampered_manifest_hash_fails_closed(self) -> None:
        import importlib.util

        module_path = WHEELHOUSE / "audit_candidate_wheelhouse.py"
        spec = importlib.util.spec_from_file_location("candidate_wheel_audit_tamper", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        payload = json.loads((WHEELHOUSE / "candidate_cp310_aarch64_manifest.json").read_text(encoding="utf-8"))
        payload = copy.deepcopy(payload)
        payload["wheels"][0]["sha256"] = "0" * 64
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False, dir=WHEELHOUSE) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)
        try:
            result = module.audit_manifest(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertFalse(result["passed"])

    def test_install_script_is_offline_hash_locked_and_arch_gated(self) -> None:
        text = (ROOT / "deploy" / "x5" / "scripts" / "install_cpu_venv_candidate.sh").read_text(encoding="utf-8")
        self.assertIn('"$(uname -m)" != "aarch64"', text)
        self.assertIn("--no-index", text)
        self.assertIn("--require-hashes", text)
        self.assertNotIn("curl ", text)
        self.assertNotIn("wget ", text)
        self.assertNotIn("apt ", text)
        self.assertNotIn("sudo ", text)


if __name__ == "__main__":
    unittest.main()
