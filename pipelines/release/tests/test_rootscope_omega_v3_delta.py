from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ADVENTUREX = Path(__file__).resolve().parents[3]
if str(ADVENTUREX) not in sys.path:
    sys.path.insert(0, str(ADVENTUREX))

from tools.release import audit_rootscope_omega_v3_delta as audit  # noqa: E402
from tools.release import build_rootscope_omega_v3_delta as build  # noqa: E402


REAL_HELPER = (
    ADVENTUREX / "tools/release/verify_run_rootscope_omega_v3_delta.py"
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


class DeltaFixture:
    def __init__(self, root: Path, *, recognized_x5: bool = True) -> None:
        self.root = root
        self.base = (
            root
            / "output/releases/rootscope_x5_field_bundle_v2"
            / "rootscope_x5_field_bundle_v2.tar"
        )
        self.base.parent.mkdir(parents=True, exist_ok=True)
        self.base.write_bytes(b"fixture-v2-base-is-referenced-not-packaged")
        self.base_sha = hashlib.sha256(self.base.read_bytes()).hexdigest()
        self.base_bytes = self.base.stat().st_size
        self._write_sources()
        self._write_helper()
        self.x5_receipt = root / "evidence/x5_summary.json"
        self.vision_receipt = root / "evidence/vision_truth_boundary_addendum.json"
        self.vision_source_receipt = root / "evidence/vision_consolidated.json"
        x5: dict[str, object] = {
            "schema": "rootscope.omega-v3-x5-deployment-summary.fixture.v1",
            "status": "FIXTURE",
        }
        if recognized_x5:
            x5.update(
                {
                    "actual_observations": {
                        "cpu_onnx_smoke": {"executed": True, "passed": True},
                        "readonly_llm_foreground_loopback_smoke": {
                            "executed": True,
                            "passed": True,
                            "process_stopped": True,
                            "port_closed_after_stop": True,
                        },
                    },
                    "authority": {
                        "execution_authority": False,
                        "physical_authority": False,
                        "physical_closure": False,
                    },
                }
            )
        _write(self.x5_receipt, json.dumps(x5, sort_keys=True) + "\n")
        _write(
            self.vision_source_receipt,
            json.dumps(
                {
                    "schema_version": "rootscope.omega-vision-consolidated.fixture.v1",
                    "status": "MACHINE_CURATED_EXPERIMENTAL_NOT_QUALIFIED",
                    "artifact_sha256": {
                        "implementation_ood": hashlib.sha256(
                            b"fixture-before-docstring-correction"
                        ).hexdigest(),
                        "implementation_evidence_builder": hashlib.sha256(
                            (
                                self.root
                                / "rootscope/training/omega_vision/build_evidence.py"
                            ).read_bytes()
                        ).hexdigest(),
                    },
                },
                sort_keys=True,
            )
            + "\n",
        )
        vision_source_sha = hashlib.sha256(
            self.vision_source_receipt.read_bytes()
        ).hexdigest()
        _write(
            self.vision_receipt,
            json.dumps(
                {
                    "schema": "rootscope.omega-vision-truth-boundary-addendum.v1",
                    "status": "BOUNDARY_CORRECTION_NO_REEVALUATION",
                    "source_receipt": {
                        "path": self.vision_source_receipt.name,
                        "sha256": vision_source_sha,
                    },
                    "terminology_correction": {
                        "implementation": "fixture pooled marginal conformal",
                        "formal_distribution_free_coverage_guarantee": False,
                    },
                    "scope_clarification": {
                        "holdout_reevaluated_for_this_addendum": False,
                        "inference_rerun_for_this_addendum": False,
                    },
                    "source_hash_transition": {
                        "app/omega_vision/ood.py_before_sha256": hashlib.sha256(
                            b"fixture-before-docstring-correction"
                        ).hexdigest(),
                        "app/omega_vision/ood.py_after_sha256": hashlib.sha256(
                            (
                                self.root
                                / "rootscope/app/omega_vision/ood.py"
                            ).read_bytes()
                        ).hexdigest(),
                        "training/omega_vision/build_evidence.py_sha256": hashlib.sha256(
                            (
                                self.root
                                / "rootscope/training/omega_vision/build_evidence.py"
                            ).read_bytes()
                        ).hexdigest(),
                    },
                    "qualification": {
                        "model_qualified": False,
                        "physical_print_domain_qualified": False,
                        "camera_qualified": False,
                        "bpu_plant_model_qualified": False,
                        "selected_bin": None,
                        "production_integration_allowed": False,
                    },
                    "authority": {
                        "training": False,
                        "holdout_evaluation": False,
                        "network_access": False,
                        "x5_access": False,
                        "bpu_access": False,
                        "camera_open": False,
                        "serial_open": False,
                        "gpio_access": False,
                        "pump_command": False,
                        "execution_authority": False,
                        "physical_authority": False,
                        "physical_closure": False,
                    },
                },
                sort_keys=True,
            )
            + "\n",
        )

    def _write_sources(self) -> None:
        _write(self.root / "rootscope/app/__init__.py", '"""fixture app."""\n')
        _write(
            self.root / "rootscope/app/omega/__init__.py",
            """
import hashlib
from enum import Enum
from types import SimpleNamespace

class EvidenceKind(Enum):
    SOURCE = "SOURCE"
class EvidenceMode(Enum):
    SEALED_REPLAY = "SEALED_REPLAY"
class EvidenceVerdict(Enum):
    PASS = "PASS"
class EvidenceNode:
    @classmethod
    def create(cls, **kwargs):
        return SimpleNamespace(
            content_sha256=hashlib.sha256(repr(sorted(kwargs.items())).encode()).hexdigest(),
            authority=SimpleNamespace(execution_authority=False),
        )
""".lstrip(),
        )
        _write(
            self.root / "rootscope/app/omega/README.md",
            "# Fixture Omega evidence core\n",
        )
        _write(
            self.root / "rootscope/app/omega_knowledge/__init__.py",
            '"""fixture knowledge."""\n',
        )
        _write(
            self.root / "rootscope/app/omega_runtime/__init__.py",
            '"""fixture runtime."""\n',
        )
        _write(
            self.root / "rootscope/app/omega_runtime/digital_twin.py",
            """
class TwinCaseInput:
    @classmethod
    def from_mapping(cls, payload):
        value = cls()
        value.payload = dict(payload)
        return value
""".lstrip(),
        )
        _write(
            self.root / "rootscope/app/omega_runtime/evidence_pipeline.py",
            """
import hashlib
from types import SimpleNamespace

def build_evidence_context(case_id, case):
    def digest(label):
        return hashlib.sha256((case_id + label).encode()).hexdigest()
    return SimpleNamespace(
        evidence_dag_root=digest("dag"),
        belief_state_hash=digest("belief"),
        failure_core_hash=digest("failure"),
        rb_voe_plan_hash=digest("voe"),
        belief=SimpleNamespace(
            authority=SimpleNamespace(execution_authority=False)
        ),
    )
""".lstrip(),
        )
        _write(
            self.root / "rootscope/app/omega_runtime/replay.py",
            """
def run_locked_replay(*, cases_path, profiles_path, corpus_path):
    return {
        "schema_version": "fixture",
        "execution_authority": False,
        "physical_closure": False,
    }
""".lstrip(),
        )
        omega_server_source = (
            ADVENTUREX / "rootscope/app/omega_runtime/omega_server.py"
        ).read_text(encoding="utf-8")
        _write(
            self.root / "rootscope/app/omega_runtime/omega_server.py",
            omega_server_source,
        )
        _write(
            self.root / "rootscope/app/omega_runtime/static/index.html",
            "<!doctype html><title>Fixture Truth Ribbon</title>\n",
        )
        _write(
            self.root / "rootscope/app/omega_bpu_aux/__init__.py",
            '"""support only; selected_bin remains null."""\n',
        )
        _write(
            self.root / "rootscope/app/omega_vision/__init__.py",
            '"""optional unqualified vision source."""\n',
        )
        _write(
            self.root / "rootscope/app/omega_vision/ood.py",
            '"""Pooled marginal conformal fixture; no formal coverage guarantee."""\n',
        )
        _write(
            self.root / "rootscope/app/omega_vision/board_replay.py",
            '"""Fixture explicit-image board replay; zero authority."""\n',
        )
        _write(
            self.root / "rootscope/training/omega_vision/calibrate.py",
            '"""training-source fixture; no artifact."""\n',
        )
        _write(
            self.root / "rootscope/training/omega_vision/build_evidence.py",
            '"""Fixture evidence builder; external dataset is not packaged."""\n',
        )
        _write(
            self.root / "rootscope/configs/omega/contracts.json",
            '{"execution_authority": false}\n',
        )
        _write(
            self.root
            / "rootscope/configs/omega/vision_board_replay_new_x5_20260723.json",
            (
                ADVENTUREX
                / "rootscope/configs/omega/"
                "vision_board_replay_new_x5_20260723.json"
            ).read_text(encoding="utf-8"),
        )
        for relative in (
            "app/web/__init__.py",
            "app/web/server.py",
            "app/web/state_store.py",
        ):
            _write(
                self.root / "rootscope" / relative,
                (ADVENTUREX / "rootscope" / relative).read_text(encoding="utf-8"),
            )
        _write(
            self.root / "rootscope/deploy/x5/omega_standalone_app_init.py",
            '"""candidate-only fixture package marker."""\n\n__all__ = []\n',
        )
        _write(
            self.root
            / "rootscope/deploy/x5/verify_omega_llm_role_cluster_foreground.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "echo 'fixture loopback foreground helper; no service install'\n",
        )
        _write(
            self.root
            / "rootscope/deploy/x5/bpu_aux_probe_new_x5_20260723.json",
            json.dumps(
                {
                    "schema_version": "rootscope.omega.bpu-aux-input-manifest.v1",
                    "run_id": "fixture-r1",
                    "model": {
                        "path": audit.BPU_VENDOR_MODEL_PATH,
                        "sha256": audit.BPU_VENDOR_MODEL_SHA256,
                        "output_semantics": "PROBABILITIES",
                    },
                    "top_k": 5,
                    "warmup_runs": 1,
                    "images": list(build.BPU_AUX_IMAGE_INPUTS),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        _write(self.root / "rootscope/tests/__init__.py", "")
        _write(
            self.root / "rootscope/tests/test_omega_fixture.py",
            "import unittest\n\nclass FixtureTest(unittest.TestCase):\n    pass\n",
        )
        _write(
            self.root / "rootscope/tests/test_omega_vision_dataset.py",
            "# local-only fixture requiring an external dataset\n",
        )
        _write(
            self.root / "rootscope/tests/test_omega_vision_evidence.py",
            "# local-only fixture requiring external frozen evidence\n",
        )
        _write(
            self.root / "rootscope/tests/test_omega_vision_board_replay.py",
            "import unittest\n\n"
            "class BoardReplayFixtureTest(unittest.TestCase):\n"
            "    pass\n",
        )
        _write(
            self.root / "rootscope/tests/omega_knowledge/__init__.py",
            "",
        )
        _write(
            self.root / "rootscope/tests/omega_knowledge/test_store.py",
            "import unittest\n\nclass StoreTest(unittest.TestCase):\n    pass\n",
        )
        for name in (
            "README.md",
            "PREEXISTING.md",
            "BUILT_DURING_EVENT.md",
            "OMEGA_V3_IMPLEMENTATION_STATUS.md",
            "OMEGA_V3_CANDIDATE_RELEASE_CHECKLIST.md",
        ):
            _write(
                self.root / "rootscope" / name,
                f"# {name}\n\nFixture truth boundary; physical closure is false.\n",
            )

    def _write_helper(self) -> None:
        helper = REAL_HELPER.read_text(encoding="utf-8")
        helper = helper.replace(build.BASE_SHA256, self.base_sha)
        helper = helper.replace(
            "BASE_BYTES = 696_832_000",
            f"BASE_BYTES = {self.base_bytes}",
        )
        _write(
            self.root
            / "tools/release/verify_run_rootscope_omega_v3_delta.py",
            helper,
        )

    def patches(self):
        return (
            mock.patch.multiple(
                build,
                BASE_SHA256=self.base_sha,
                BASE_BYTES=self.base_bytes,
            ),
            mock.patch.multiple(
                audit,
                BASE_SHA256=self.base_sha,
                BASE_BYTES=self.base_bytes,
            ),
        )


def _actual_source_mirror(root: Path) -> SimpleNamespace:
    root.mkdir(parents=True)
    base = (
        root
        / "output/releases/rootscope_x5_field_bundle_v2"
        / "rootscope_x5_field_bundle_v2.tar"
    )
    base.parent.mkdir(parents=True)
    base.write_bytes(b"fixture-base-for-actual-source-mirror")
    base_sha = hashlib.sha256(base.read_bytes()).hexdigest()
    base_bytes = base.stat().st_size
    for entry in build.collect_delta_sources(ADVENTUREX):
        relative = entry.source.relative_to(ADVENTUREX)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry.source, destination)
    helper = root / build.HELPER_RELATIVE_PATH
    helper_text = helper.read_text(encoding="utf-8")
    helper_text = helper_text.replace(build.BASE_SHA256, base_sha)
    helper_text = helper_text.replace(
        "BASE_BYTES = 696_832_000",
        f"BASE_BYTES = {base_bytes}",
    )
    _write(helper, helper_text)
    x5_relative = Path(
        "rootscope/evidence/new_x5_20260723/"
        "candidate_x5_observation_contract.json"
    )
    vision_relative = Path(
        "rootscope/evidence/omega_vision_v3_20260723/"
        "vision_truth_boundary_addendum.json"
    )
    vision_source_relative = vision_relative.with_name("vision_consolidated.json")
    for relative in (x5_relative, vision_relative, vision_source_relative):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ADVENTUREX / relative, destination)
    return SimpleNamespace(
        root=root,
        base=base,
        base_sha=base_sha,
        base_bytes=base_bytes,
        x5_receipt=root / x5_relative,
        vision_receipt=root / vision_relative,
    )


class OmegaV3DeltaReleaseTests(unittest.TestCase):
    def test_fixture_build_is_deterministic_and_independent_audit_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DeltaFixture(Path(temporary) / "adventurex")
            first = fixture.root / "output/candidate-a"
            second = fixture.root / "output/candidate-b"
            build_patch, audit_patch = fixture.patches()
            with build_patch, audit_patch:
                first_receipt = build.build_delta_candidate(
                    fixture.root,
                    first,
                    fixture.x5_receipt,
                    fixture.vision_receipt,
                )
                second_receipt = build.build_delta_candidate(
                    fixture.root,
                    second,
                    fixture.x5_receipt,
                    fixture.vision_receipt,
                )
                result = audit.audit_delta(
                    first / build.CANDIDATE_ARCHIVE,
                    fixture.root,
                )
            self.assertEqual(
                first_receipt["archive"]["sha256"],
                second_receipt["archive"]["sha256"],
            )
            self.assertEqual(
                (first / build.CANDIDATE_ARCHIVE).read_bytes(),
                (second / build.CANDIDATE_ARCHIVE).read_bytes(),
            )
            self.assertEqual(
                first_receipt["status"],
                "SAFE_CPU_PLUS_READONLY_LLM_CANDIDATE",
            )
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["checks_failed"], 0)

    def test_delta_does_not_duplicate_v2_and_sums_have_exact_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DeltaFixture(Path(temporary) / "adventurex")
            output = fixture.root / "output/candidate"
            build_patch, _audit_patch = fixture.patches()
            with build_patch:
                build.build_delta_candidate(
                    fixture.root,
                    output,
                    fixture.x5_receipt,
                    fixture.vision_receipt,
                )
            archive_path = output / build.CANDIDATE_ARCHIVE
            with tarfile.open(archive_path, mode="r:") as archive:
                names = [member.name for member in archive.getmembers()]
                self.assertFalse(any(name.endswith(".tar") for name in names))
                manifest_member = archive.extractfile(
                    f"{build.CANDIDATE_ID}/candidate_manifest.json"
                )
                sums_member = archive.extractfile(
                    f"{build.CANDIDATE_ID}/SHA256SUMS"
                )
                self.assertIsNotNone(manifest_member)
                self.assertIsNotNone(sums_member)
                manifest = json.load(manifest_member)
                sums_lines = sums_member.read().decode("ascii").splitlines()
            self.assertFalse(
                manifest["immutable_base_v2"]["bundled_in_delta"]
            )
            self.assertEqual(manifest["immutable_base_v2"]["sha256"], fixture.base_sha)
            self.assertEqual(
                {line[66:] for line in sums_lines},
                {record["path"] for record in manifest["files"]}
                | {"candidate_manifest.json"},
            )

    def test_unrecognized_x5_receipt_is_bound_without_claim_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DeltaFixture(
                Path(temporary) / "adventurex",
                recognized_x5=False,
            )
            output = fixture.root / "output/candidate"
            build_patch, _audit_patch = fixture.patches()
            with build_patch:
                receipt = build.build_delta_candidate(
                    fixture.root,
                    output,
                    fixture.x5_receipt,
                    fixture.vision_receipt,
                )
            self.assertEqual(
                receipt["status"],
                "SOURCE_DELTA_CANDIDATE_RECEIPTS_BOUND",
            )
            self.assertFalse(
                receipt["qualification"][
                    "x5_receipt_observation_contract_recognized"
                ]
            )
            self.assertFalse(
                receipt["qualification"]["cpu_onnx_smoke_observed_pass"]
            )
            self.assertFalse(
                receipt["qualification"][
                    "readonly_llm_foreground_loopback_smoke_observed_pass"
                ]
            )

    def test_refuses_overwrite_and_requires_distinct_in_scope_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DeltaFixture(Path(temporary) / "adventurex")
            output = fixture.root / "output/candidate"
            build_patch, _audit_patch = fixture.patches()
            with build_patch:
                build.build_delta_candidate(
                    fixture.root,
                    output,
                    fixture.x5_receipt,
                    fixture.vision_receipt,
                )
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    build.build_delta_candidate(
                        fixture.root,
                        output,
                        fixture.x5_receipt,
                        fixture.vision_receipt,
                    )
                with self.assertRaisesRegex(ValueError, "must be distinct"):
                    build.build_delta_candidate(
                        fixture.root,
                        fixture.root / "output/other",
                        fixture.x5_receipt,
                        fixture.x5_receipt,
                    )
                outside = Path(temporary) / "outside.json"
                _write(outside, '{"schema": "outside"}\n')
                with self.assertRaisesRegex(ValueError, "below AdventureX"):
                    build.build_delta_candidate(
                        fixture.root,
                        fixture.root / "output/outside",
                        outside,
                        fixture.vision_receipt,
                    )

    def test_allowlist_includes_optional_vision_source_and_excludes_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DeltaFixture(Path(temporary) / "adventurex")
            _write(
                fixture.root / "rootscope/app/omega/__pycache__/ignored.pyc",
                "not bytecode in fixture but excluded by path\n",
            )
            entries = build.collect_delta_sources(fixture.root)
            paths = {entry.package_path for entry in entries}
            self.assertIn(
                "rootscope/app/omega_vision/__init__.py",
                paths,
            )
            self.assertIn(
                "rootscope/training/omega_vision/calibrate.py",
                paths,
            )
            self.assertFalse(any("__pycache__" in path for path in paths))
            self.assertFalse(any(path.endswith(".pyc") for path in paths))
            self.assertNotIn(
                "rootscope/tests/test_omega_vision_dataset.py",
                paths,
            )
            self.assertNotIn(
                "rootscope/tests/test_omega_vision_evidence.py",
                paths,
            )

    def test_scanner_rejects_xrd_import_secret_key_path_and_absolute_temp(self) -> None:
        mutations = (
            (
                "xrd import",
                "rootscope/app/omega/unsafe.py",
                "import dashboard\n",
                "XRD/frozen runtime import",
            ),
            (
                "key path",
                "rootscope/app/omega/keys/unsafe.py",
                "VALUE = 1\n",
                "forbidden source path component",
            ),
            (
                "temp path",
                "rootscope/app/omega/temp_leak.py",
                'VALUE = "/tmp/rootscope-private-output.json"\n',
                "absolute temporary path",
            ),
            (
                "secret",
                "rootscope/app/omega/secret_leak.py",
                'api_key = "abcdefghijklmnopqrstuvwxyz123456"\n',
                "possible embedded secret",
            ),
        )
        for label, relative, content, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = DeltaFixture(Path(temporary) / "adventurex")
                _write(fixture.root / relative, content)
                with self.assertRaisesRegex(ValueError, message):
                    build.collect_delta_sources(fixture.root)

    def test_packaged_board_helper_verify_and_pure_cpu_smoke_are_zero_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DeltaFixture(Path(temporary) / "adventurex")
            output = fixture.root / "output/candidate"
            build_patch, _audit_patch = fixture.patches()
            with build_patch:
                build.build_delta_candidate(
                    fixture.root,
                    output,
                    fixture.x5_receipt,
                    fixture.vision_receipt,
                )
            extracted_parent = fixture.root / "output/extracted"
            extracted_parent.mkdir(parents=True)
            with tarfile.open(output / build.CANDIDATE_ARCHIVE, mode="r:") as archive:
                archive.extractall(extracted_parent, filter="data")
            candidate = extracted_parent / build.CANDIDATE_ID
            helper_path = candidate / build.HELPER_PACKAGE_PATH
            spec = importlib.util.spec_from_file_location(
                "fixture_packaged_omega_helper",
                helper_path,
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader if spec else None)
            module = importlib.util.module_from_spec(spec)
            previous_dont_write_bytecode = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                spec.loader.exec_module(module)
            finally:
                sys.dont_write_bytecode = previous_dont_write_bytecode
            try:
                verification = module.verify_extracted_delta(candidate)
                smoke = module.run_pure_cpu_smoke(candidate)
            finally:
                for name in list(sys.modules):
                    if name == "app" or name.startswith("app."):
                        del sys.modules[name]
            self.assertFalse(verification["pure_cpu_smoke_executed"])
            self.assertTrue(smoke["pure_cpu_smoke_executed"])
            self.assertEqual(
                smoke["status"],
                "PASS_PURE_CPU_SEALED_FIXTURE_ZERO_AUTHORITY",
            )
            self.assertTrue(
                all(value is False for value in smoke["authority"].values())
            )

    def test_actual_source_mirror_is_standalone_for_helper_and_omega_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _actual_source_mirror(
                Path(temporary) / "adventurex"
            )
            output = fixture.root / "output/candidate"
            with mock.patch.multiple(
                build,
                BASE_SHA256=fixture.base_sha,
                BASE_BYTES=fixture.base_bytes,
            ), mock.patch.multiple(
                audit,
                BASE_SHA256=fixture.base_sha,
                BASE_BYTES=fixture.base_bytes,
            ):
                build.build_delta_candidate(
                    fixture.root,
                    output,
                    fixture.x5_receipt,
                    fixture.vision_receipt,
                )
                audited = audit.audit_delta(
                    output / build.CANDIDATE_ARCHIVE,
                    fixture.root,
                )
            self.assertTrue(audited["passed"], audited)
            extracted_parent = fixture.root / "output/actual-source-extracted"
            extracted_parent.mkdir(parents=True)
            with tarfile.open(output / build.CANDIDATE_ARCHIVE, mode="r:") as archive:
                archive.extractall(extracted_parent, filter="data")
            candidate = extracted_parent / build.CANDIDATE_ID
            helper_path = candidate / build.HELPER_PACKAGE_PATH
            spec = importlib.util.spec_from_file_location(
                "actual_source_packaged_omega_helper",
                helper_path,
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader if spec else None)
            helper = importlib.util.module_from_spec(spec)
            previous_dont_write_bytecode = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                spec.loader.exec_module(helper)
                helper_verification = helper.verify_extracted_delta(candidate)
                helper_smoke = helper.run_pure_cpu_smoke(candidate)
            finally:
                sys.dont_write_bytecode = previous_dont_write_bytecode
            self.assertFalse(helper_verification["pure_cpu_smoke_executed"])
            self.assertTrue(helper_smoke["pure_cpu_smoke_executed"])
            for name in list(sys.modules):
                if name == "app" or name.startswith("app."):
                    del sys.modules[name]
            candidate_rootscope = candidate / "rootscope"
            sys.path.insert(0, str(candidate_rootscope))
            server = None
            try:
                from app.omega_runtime.omega_server import build_omega_server

                server = build_omega_server(
                    host="127.0.0.1",
                    port=0,
                    cases_path=candidate_rootscope
                    / "configs/omega/locked_replay_cases.v1.json",
                    profiles_path=candidate_rootscope
                    / "configs/omega/edge_profiles.v1.json",
                    corpus_path=candidate_rootscope
                    / "configs/omega/field_knowledge.v1.md",
                )
                self.assertEqual(server.actions, {})
                self.assertGreater(server.address[1], 0)
                loaded_app_files = [
                    Path(module.__file__).resolve()
                    for name, module in sys.modules.items()
                    if (name == "app" or name.startswith("app."))
                    and getattr(module, "__file__", None)
                ]
                self.assertTrue(loaded_app_files)
                self.assertTrue(
                    all(
                        candidate_rootscope.resolve() in path.parents
                        for path in loaded_app_files
                    ),
                    loaded_app_files,
                )
            finally:
                if server is not None:
                    server.httpd.server_close()
                try:
                    sys.path.remove(str(candidate_rootscope))
                except ValueError:
                    pass
                for name in list(sys.modules):
                    if name == "app" or name.startswith("app."):
                        del sys.modules[name]


if __name__ == "__main__":
    unittest.main()
