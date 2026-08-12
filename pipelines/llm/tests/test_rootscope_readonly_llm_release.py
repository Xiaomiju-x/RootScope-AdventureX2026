from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from audit_rootscope_readonly_llm_release import (  # noqa: E402
    Audit,
    audit,
    audit_release_core,
    audit_service_contract,
)


ADVENTUREX_ROOT = Path(__file__).resolve().parents[3]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ActualReleaseAuditTests(unittest.TestCase):
    def test_actual_release_passes_independent_audit(self) -> None:
        report = audit(ADVENTUREX_ROOT)
        failed = [item for item in report["checks"] if not item["pass"]]
        self.assertEqual(failed, [])
        self.assertEqual(report["status"], "PASS")


class ReleaseCoreMutationTests(unittest.TestCase):
    def _fixture(self, root: Path, *, formal_flag: bool = False) -> tuple[Path, Path, str, int, str]:
        adventurex = root / "adventurex"
        source_dir = root / "models"
        release = adventurex / "output/rootscope_llm_readonly_release_v1"
        spec_dir = adventurex / "rootscope/deploy/x5"
        source_dir.mkdir(parents=True)
        release.mkdir(parents=True)
        spec_dir.mkdir(parents=True)
        name = "fixture.gguf"
        payload = b"GGUF-fixture-read-only"
        digest = _sha(payload)
        (source_dir / name).write_bytes(payload)
        (release / name).write_bytes(payload)
        flags = {"model_qualified": formal_flag, "x5_validated": False}
        runtime = {
            "host": "127.0.0.1",
            "port": 9080,
            "default_enabled": False,
            "manual_start_only": True,
            "external_network_allowed": False,
            "read_only": True,
            "tool_execution": False,
            "actuator_access": False,
        }
        dependency = {
            "bundled": False,
            "x5_binary_selected": False,
            "executable_sha256_required_before_start": True,
        }
        spec = {
            "schema": "rootscope.readonly_llm_release_spec.v1",
            "status": "STAGING_SPEC_READ_ONLY_LOCAL_LLM_NOT_X5_QUALIFIED",
            "release_id": "fixture",
            "selected_artifact": {
                "source_relative_to_adventurex": "../models/fixture.gguf",
                "output_relative_to_adventurex": "output/rootscope_llm_readonly_release_v1",
                "filename": name,
                "size_bytes": len(payload),
                "sha256": digest,
            },
            "runtime_contract": runtime,
            "llama_cpp_dependency": dependency,
            "formal_flags": flags,
        }
        spec_path = spec_dir / "readonly_llm_release_spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        manifest = {
            "schema": "rootscope.readonly_llm_release_manifest.v1",
            "status": "STAGED_READ_ONLY_LOCAL_LLM_NOT_X5_QUALIFIED",
            "release_id": "fixture",
            "artifact_staged": True,
            "artifact": {
                "filename": name,
                "size_bytes": len(payload),
                "sha256": digest,
                "destination_relative_to_adventurex": f"output/rootscope_llm_readonly_release_v1/{name}",
            },
            "staging_spec": {"sha256": _sha(spec_path.read_bytes())},
            "runtime_contract": runtime,
            "llama_cpp_dependency": dependency,
            "formal_flags": flags,
        }
        manifest_path = release / "release_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (release / "SHA256SUMS").write_text(
            f"{digest}  {name}\n{_sha(manifest_path.read_bytes())}  release_manifest.json\n",
            encoding="utf-8",
        )
        return adventurex, spec_path, digest, len(payload), name

    def _run(self, adventurex: Path, spec: Path, digest: str, size: int, name: str) -> Audit:
        state = Audit()
        audit_release_core(
            state,
            adventurex_root=adventurex,
            spec_path=spec,
            release_dir=adventurex / "output/rootscope_llm_readonly_release_v1",
            expected_sha=digest,
            expected_size=size,
            expected_name=name,
        )
        return state

    def test_tiny_exact_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = self._fixture(Path(temporary))
            self.assertTrue(self._run(*values).passed)

    def test_true_formal_flag_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = self._fixture(Path(temporary), formal_flag=True)
            state = self._run(*values)
            self.assertFalse(state.passed)
            self.assertFalse(next(item for item in state.checks if item["name"] == "spec_formal_flags_all_false")["pass"])

    def test_model_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = self._fixture(Path(temporary))
            model = values[0] / "output/rootscope_llm_readonly_release_v1" / values[4]
            model.write_bytes(b"tampered")
            state = self._run(*values)
            self.assertFalse(next(item for item in state.checks if item["name"] == "staged_sha_exact")["pass"])

    def test_extra_sha256sums_entry_fails_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = self._fixture(Path(temporary))
            sums = values[0] / "output/rootscope_llm_readonly_release_v1/SHA256SUMS"
            sums.write_text(sums.read_text(encoding="utf-8") + f"{'0' * 64}  extra.bin\n", encoding="utf-8")
            state = self._run(*values)
            self.assertFalse(next(item for item in state.checks if item["name"] == "sha256sums_exact_coverage")["pass"])


class ServiceContractMutationTests(unittest.TestCase):
    def test_hardened_user_service_passes(self) -> None:
        unit = (ADVENTUREX_ROOT / "rootscope/deploy/x5/systemd/rootscope-llm-readonly.service.disabled-template").read_text(encoding="utf-8")
        start = (ADVENTUREX_ROOT / "rootscope/deploy/x5/scripts/start_readonly_llm.sh").read_text(encoding="utf-8")
        state = Audit()
        audit_service_contract(state, unit, start)
        self.assertTrue(state.passed, state.checks)

    def test_hardcoded_account_and_remote_bind_fail(self) -> None:
        unit = "User=rootscope\nWantedBy=multi-user.target\n"
        start = "llama-server --host 0.0.0.0 --port 9080"
        state = Audit()
        audit_service_contract(state, unit, start)
        self.assertFalse(state.passed)


if __name__ == "__main__":
    unittest.main()
