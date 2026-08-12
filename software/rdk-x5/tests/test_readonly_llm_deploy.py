from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOTSCOPE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOTSCOPE_ROOT / "deploy/x5/scripts"
sys.path.insert(0, str(SCRIPTS))

from install_readonly_llm import install  # noqa: E402
from readonly_llm_preflight import (  # noqa: E402
    health_probe,
    preflight,
    validate_release_model,
)
from stage_readonly_llm import stage  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _false_flags() -> dict[str, bool]:
    return {
        "human_reviewed": False,
        "data_locked": False,
        "model_candidate": False,
        "model_qualified": False,
        "x5_validated": False,
        "latency_measured": False,
        "llama_cpp_bundled": False,
        "service_started": False,
        "hardware_touched": False,
        "network_touched": False,
        "external_network_allowed": False,
        "tool_execution": False,
        "actuator_access": False,
        "execution_authority": False,
        "physical_authority": False,
        "physical_completion": False,
    }


def _runtime() -> dict:
    return {
        "role": "CPU_LLAMA_CPP_READ_ONLY_EVIDENCE_EXPLANATION",
        "default_enabled": False,
        "manual_start_only": True,
        "host": "127.0.0.1",
        "port": 9080,
        "chat_path": "/v1/chat/completions",
        "health_path": "/health",
        "external_network_allowed": False,
        "read_only": True,
        "tool_execution": False,
        "actuator_access": False,
    }


def _dependency() -> dict:
    return {
        "bundled": False,
        "source": "EXTERNAL_EXACT_X5_BINARY_OR_EXISTING_XRD_DEPLOYMENT",
        "executable_sha256_required_before_start": True,
        "x5_binary_selected": False,
    }


def _make_staging_fixture(base: Path) -> tuple[Path, Path, bytes]:
    adventurex = base / "adventurex"
    source_dir = base / "models"
    spec_dir = adventurex / "rootscope/deploy/x5"
    (adventurex / "output").mkdir(parents=True)
    source_dir.mkdir(parents=True)
    spec_dir.mkdir(parents=True)
    payload = b"GGUF-read-only-staging-fixture"
    source = source_dir / "fixture.gguf"
    source.write_bytes(payload)
    spec = {
        "schema": "rootscope.readonly_llm_release_spec.v1",
        "status": "STAGING_SPEC_READ_ONLY_LOCAL_LLM_NOT_X5_QUALIFIED",
        "release_id": "fixture",
        "selected_artifact": {
            "source_relative_to_adventurex": "../models/fixture.gguf",
            "output_relative_to_adventurex": "output/rootscope_llm_readonly_release_v1",
            "filename": "fixture.gguf",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "copy_contract": "MECHANICAL_BYTE_COPY_HASH_AND_SIZE_EXACT",
        },
        "runtime_contract": _runtime(),
        "llama_cpp_dependency": _dependency(),
        "formal_flags": _false_flags(),
    }
    spec_path = spec_dir / "readonly_llm_release_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return adventurex, spec_path, payload


class StagingTests(unittest.TestCase):
    def test_stage_is_exact_and_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adventurex, spec, payload = _make_staging_fixture(Path(temporary))
            first = stage(adventurex, spec)
            second = stage(adventurex, spec)
            release = adventurex / "output/rootscope_llm_readonly_release_v1"
            self.assertEqual((release / "fixture.gguf").read_bytes(), payload)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertTrue(all(value is False for value in first["formal_flags"].values()))

    def test_stage_rejects_tampered_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adventurex, spec, _payload = _make_staging_fixture(Path(temporary))
            (adventurex.parent / "models/fixture.gguf").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size or SHA"):
                stage(adventurex, spec)

    def test_stage_rejects_output_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adventurex, spec, _payload = _make_staging_fixture(Path(temporary))
            value = json.loads(spec.read_text(encoding="utf-8"))
            value["selected_artifact"]["output_relative_to_adventurex"] = "../escaped"
            spec.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "below AdventureX/output"):
                stage(adventurex, spec)


class PreflightAndInstallTests(unittest.TestCase):
    def _release(self, base: Path) -> tuple[Path, Path, Path, str]:
        adventurex, spec, _payload = _make_staging_fixture(base)
        stage(adventurex, spec)
        release = adventurex / "output/rootscope_llm_readonly_release_v1"
        model = release / "fixture.gguf"
        llama = base / "llama-server"
        llama.write_bytes(b"external-exact-x5-binary-fixture")
        return release, model, llama, _sha(llama)

    def test_offline_preflight_checks_both_hashes_without_health(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release, model, llama, llama_sha = self._release(Path(temporary))
            receipt = preflight(
                manifest_path=release / "release_manifest.json",
                model_path=model,
                llama_server=llama,
                llama_server_sha256=llama_sha,
                host="127.0.0.1",
                port=9080,
                health=False,
            )
            self.assertFalse(receipt["health_checked"])
            self.assertFalse(receipt["service_started_by_preflight"])
            self.assertFalse(receipt["tool_execution"])

    def test_preflight_rejects_model_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release, model, _llama, _sha_value = self._release(Path(temporary))
            model.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_release_model(release / "release_manifest.json", model)

    def test_health_rejects_non_loopback_before_connect(self) -> None:
        with self.assertRaisesRegex(ValueError, "numeric IPv4 loopback"):
            health_probe("192.0.2.10", 9080)

    def test_health_does_not_follow_redirect(self) -> None:
        class FakeResponse:
            status = 302

            def read(self, _limit):
                return b"{}"

        class FakeConnection:
            def __init__(self, *_args, **_kwargs):
                pass

            def request(self, *_args, **_kwargs):
                pass

            def getresponse(self):
                return FakeResponse()

            def close(self):
                pass

        with patch("readonly_llm_preflight.http.client.HTTPConnection", FakeConnection):
            with self.assertRaisesRegex(ValueError, "must be 200"):
                health_probe("127.0.0.1", 9080)

    def test_user_install_remains_disabled_and_does_not_bundle_llama(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            release, _model, llama, llama_sha = self._release(base)
            project = base / "project"
            (project / "app/llm").mkdir(parents=True)
            (project / "deploy/x5/scripts").mkdir(parents=True)
            (project / "app/llm/read_only_explainer.py").write_text("# fixture\n", encoding="utf-8")
            (project / "deploy/x5/scripts/start_readonly_llm.sh").write_text("# fixture\n", encoding="utf-8")
            (project / "deploy/x5/scripts/readonly_llm_preflight.py").write_text("# fixture\n", encoding="utf-8")
            prefix = base / "install"
            config = base / "config"
            units = base / "user-units"
            receipt = install(
                release_dir=release,
                project_root=project,
                python_executable=Path(sys.executable),
                llama_server=llama,
                llama_server_sha256=llama_sha,
                prefix=prefix,
                config_dir=config,
                systemd_user_dir=units,
                template_path=ROOTSCOPE_ROOT
                / "deploy/x5/systemd/rootscope-llm-readonly.service.disabled-template",
            )
            self.assertFalse(receipt["activation_gate_created"])
            self.assertFalse(receipt["service_started"])
            self.assertFalse(receipt["systemctl_invoked"])
            self.assertFalse((config / "enable-readonly-llm").exists())
            unit = (units / "rootscope-llm-readonly.service").read_text(encoding="utf-8")
            self.assertNotIn("User=", unit)
            self.assertNotIn("[Install]", unit)
            env = (config / "rootscope-llm.env").read_text(encoding="utf-8")
            self.assertIn("ROOTSCOPE_LLM_MANUAL_ACK=NOT_ACKNOWLEDGED", env)
            self.assertEqual((prefix / "models/fixture.gguf").read_bytes(), (release / "fixture.gguf").read_bytes())


if __name__ == "__main__":
    unittest.main()
