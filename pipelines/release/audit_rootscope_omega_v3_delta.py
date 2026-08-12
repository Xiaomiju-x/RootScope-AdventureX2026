#!/usr/bin/env python3
"""Independently audit a RootScope-Ω v3 deterministic delta candidate."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
from typing import Any


CANDIDATE_ID = "rootscope_omega_v3_delta_candidate_v1"
CANDIDATE_ARCHIVE = f"{CANDIDATE_ID}.tar"
CANDIDATE_SCHEMA = "rootscope.omega-v3-delta-candidate.v1"
BASE_RELATIVE_PATH = (
    "output/releases/rootscope_x5_field_bundle_v2/"
    "rootscope_x5_field_bundle_v2.tar"
)
BASE_SHA256 = "e6627685170252004d118bf77a690a9f89ad3afa274910697554d0f5cc8c3ebb"
BASE_BYTES = 696_832_000
HELPER_PACKAGE_PATH = "tools/verify_run_rootscope_omega_v3_delta.py"
BPU_AUX_CONFIG_PATH = "rootscope/deploy/x5/bpu_aux_probe_new_x5_20260723.json"
BPU_VENDOR_MODEL_PATH = "/opt/hobot/model/x5/basic/mobilenetv2_224x224_nv12.bin"
BPU_VENDOR_MODEL_SHA256 = "3e2b7c46fc3b3a6d07a5326c0b9632fe98fe5ca38835346ab2eedc22ed427158"
VISION_BOARD_REPLAY_SOURCE_PATH = "rootscope/app/omega_vision/board_replay.py"
VISION_BOARD_REPLAY_CONFIG_PATH = (
    "rootscope/configs/omega/vision_board_replay_new_x5_20260723.json"
)
VISION_BOARD_REPLAY_TEST_PATH = "rootscope/tests/test_omega_vision_board_replay.py"
VISION_BOARD_REPLAY_CONFIG_SHA256 = (
    "e82b196ab627a935fd571bba2b37aa637f040af5ba81fbab5183c1f15aa2e564"
)
BPU_AUX_IMAGE_INPUTS = (
    {
        "image_id": "demo-reference-grass-clump",
        "path": "/opt/rootscope/rootscope_omega_v3_inputs/bpu_aux/grass_clump_163498042_b1f6262895c3.jpg",
        "sha256": "b1f6262895c31e8e507be31cebba09140e2a2582aa4f266ab05261fe50751d23",
    },
    {
        "image_id": "demo-reference-low-shrub",
        "path": "/opt/rootscope/rootscope_omega_v3_inputs/bpu_aux/low_shrub_68787114_810c7649ac72.jpg",
        "sha256": "810c7649ac729105367b3213bfafc467a036f4054244c424613da6c027c73610",
    },
    {
        "image_id": "demo-reference-young-tree",
        "path": "/opt/rootscope/rootscope_omega_v3_inputs/bpu_aux/young_tree_92774234_0d994e838a2d.jpg",
        "sha256": "0d994e838a2d7787ab3edfd8646e317390c790d92588c7ef9109778b843b40eb",
    },
    {
        "image_id": "unregistered-negative-unknown",
        "path": "/opt/rootscope/rootscope_omega_v3_inputs/bpu_aux/unknown_157364276_04e7f49a1e66.jpg",
        "sha256": "04e7f49a1e66186bda7a9a1102985560eac0e3a1bffcec892e6dc522868c985b",
    },
)
LOCAL_ONLY_TEST_EXCLUSIONS = {
    "rootscope/tests/test_omega_vision_dataset.py": (
        "requires the external 78-image dataset and its manifests"
    ),
    "rootscope/tests/test_omega_vision_evidence.py": (
        "requires external frozen vision evidence receipts"
    ),
}

_FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".bin",
    ".onnx",
    ".pt",
    ".pth",
}
_FORBIDDEN_PATH_PARTS = {
    "__pycache__",
    ".git",
    ".ssh",
    "secrets",
    "credentials",
    "private_keys",
    "keys",
}
_XRD_IMPORT_ROOTS = {
    "dashboard",
    "predict_engine",
    "xrd_vision",
    "xrd_numerical",
    "spectrum_vision",
    "spectrum_numerical",
    "embodied_brain",
    "workstation",
    "rb_voe",
    "ros",
}
_TEMP_PATTERNS = (
    re.compile(r"(?i)\b[A-Z]:\\[^\r\n\"']*(?:\\Temp\\|\\AppData\\Local\\Temp\\)"),
    re.compile(r"(?<![A-Za-z0-9_])/(?:tmp|var/tmp)/[^\s\"']+"),
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(
        r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|private[_-]?key)"
        r"\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']"
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_compact(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _safe_relative(value: str, *, require_root: bool = False) -> PurePosixPath:
    if not value or "\\" in value:
        raise ValueError(f"unsafe archive path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe archive path: {value!r}")
    if require_root and (path.parts[0] != CANDIDATE_ID or len(path.parts) < 2):
        raise ValueError(f"archive member escapes candidate root: {value!r}")
    return path


def _strict_json(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _inside_regular(path: Path, root: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"symlink is forbidden: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"path escapes AdventureX: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"expected one regular file: {path}")
    return resolved


def _allowlisted_payload_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    if relative == HELPER_PACKAGE_PATH:
        return True
    if relative in {
        "rootscope/app/__init__.py",
        "rootscope/tests/__init__.py",
        "rootscope/pyproject.toml",
        "rootscope/README.md",
        "rootscope/PREEXISTING.md",
        "rootscope/BUILT_DURING_EVENT.md",
        "rootscope/OMEGA_V3_IMPLEMENTATION_STATUS.md",
        "rootscope/OMEGA_V3_CANDIDATE_RELEASE_CHECKLIST.md",
        "rootscope/app/web/__init__.py",
        "rootscope/app/web/server.py",
        "rootscope/app/web/state_store.py",
        "rootscope/deploy/x5/verify_omega_llm_role_cluster_foreground.sh",
        BPU_AUX_CONFIG_PATH,
    }:
        return True
    prefixes = (
        "rootscope/app/omega/",
        "rootscope/app/omega_knowledge/",
        "rootscope/app/omega_runtime/",
        "rootscope/app/omega_bpu_aux/",
        "rootscope/app/omega_vision/",
        "rootscope/training/omega_vision/",
        "rootscope/configs/omega/",
        "rootscope/tests/omega_knowledge/",
        "rootscope/tests/omega_vision/",
    )
    if any(relative.startswith(prefix) for prefix in prefixes):
        return True
    return (
        len(path.parts) == 3
        and path.parts[:2] == ("rootscope", "tests")
        and path.name.startswith("test_omega_")
        and path.suffix == ".py"
    )


def _scan_python_imports(relative: str, text: str) -> None:
    tree = ast.parse(text, filename=relative)
    for node in ast.walk(tree):
        roots: set[str] = set()
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
        forbidden = sorted(roots & _XRD_IMPORT_ROOTS)
        if forbidden:
            raise ValueError(f"XRD/frozen runtime import in {relative}: {forbidden}")


def _scan_payload(relative: str, path: Path) -> None:
    pure = PurePosixPath(relative)
    if {part.casefold() for part in pure.parts} & _FORBIDDEN_PATH_PARTS:
        raise ValueError(f"forbidden candidate path component: {relative}")
    if pure.suffix.lower() in _FORBIDDEN_SUFFIXES:
        raise ValueError(f"forbidden candidate artifact type: {relative}")
    text = path.read_text(encoding="utf-8")
    for pattern in _TEMP_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"absolute temporary path leaked into {relative}")
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"possible secret leaked into {relative}")
    if pure.suffix == ".py":
        _scan_python_imports(relative, text)


def _derive_observations(payload: Mapping[str, Any]) -> Mapping[str, bool]:
    authority = payload.get("authority")
    observations = payload.get("actual_observations")
    if (
        not isinstance(authority, Mapping)
        or not authority
        or any(value is not False for value in authority.values())
        or not isinstance(observations, Mapping)
    ):
        return {
            "recognized": False,
            "cpu": False,
            "llm": False,
        }
    cpu = observations.get("cpu_onnx_smoke")
    llm = observations.get("readonly_llm_foreground_loopback_smoke")
    return {
        "recognized": True,
        "cpu": (
            isinstance(cpu, Mapping)
            and cpu.get("executed") is True
            and cpu.get("passed") is True
        ),
        "llm": (
            isinstance(llm, Mapping)
            and llm.get("executed") is True
            and llm.get("passed") is True
            and llm.get("process_stopped") is True
            and llm.get("port_closed_after_stop") is True
        ),
    }


def _expected_status(observations: Mapping[str, bool]) -> str:
    if observations["cpu"] and observations["llm"]:
        return "SAFE_CPU_PLUS_READONLY_LLM_CANDIDATE"
    if observations["cpu"]:
        return "SAFE_CPU_QUALIFIED_CANDIDATE"
    return "SOURCE_DELTA_CANDIDATE_RECEIPTS_BOUND"


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def result(self, archive: Path, adventurex: Path) -> Mapping[str, Any]:
        failures = [check for check in self.checks if not check["passed"]]
        return {
            "schema": "rootscope.omega-v3-delta-independent-audit.v1",
            "status": "PASS" if not failures else "FAIL",
            "passed": not failures,
            "checks_passed": len(self.checks) - len(failures),
            "checks_failed": len(failures),
            "archive_relative_to_adventurex": archive.relative_to(adventurex).as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive),
            "checks": self.checks,
            "authority": {
                "hardware_touched": False,
                "network_touched": False,
                "camera_opened": False,
                "serial_opened": False,
                "gpio_opened": False,
                "pump_touched": False,
                "systemd_touched": False,
                "execution_authority": False,
                "physical_authority": False,
                "physical_closure": False,
            },
        }


def _extract_archive(archive_path: Path, destination: Path, audit: Audit) -> Path:
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        names: set[str] = set()
        metadata_ok = True
        for member in members:
            path = _safe_relative(member.name, require_root=True)
            if member.name in names:
                raise ValueError(f"duplicate archive member: {member.name}")
            names.add(member.name)
            if not member.isfile():
                raise ValueError(f"non-regular archive member is forbidden: {member.name}")
            metadata_ok = metadata_ok and (
                member.mtime == 0
                and member.uid == 0
                and member.gid == 0
                and member.uname == ""
                and member.gname == ""
                and not member.pax_headers
            )
            target = destination / Path(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            with target.open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            os.chmod(target, 0o755 if member.mode & 0o111 else 0o644)
    audit.check("archive_nonempty", bool(names), len(names))
    audit.check("archive_members_unique", len(names) == len(members), len(names))
    audit.check("archive_regular_files_only", all(item.isfile() for item in members), len(members))
    audit.check("archive_deterministic_metadata", metadata_ok, "mtime/uid/gid/names/pax")
    return destination / CANDIDATE_ID


def _parse_sums(path: Path) -> Mapping[str, str]:
    lines = path.read_text(encoding="ascii").splitlines()
    if lines != sorted(lines):
        raise ValueError("SHA256SUMS lines are not sorted")
    result: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("malformed SHA256SUMS line")
        digest, relative = line[:64], line[66:]
        _safe_relative(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("malformed SHA256SUMS digest")
        if relative in result:
            raise ValueError(f"duplicate SHA256SUMS path: {relative}")
        result[relative] = digest
    return result


def _load_helper(path: Path):
    spec = importlib.util.spec_from_file_location("rootscope_omega_v3_delta_board_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load packaged board helper")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def audit_delta(archive_path: Path, adventurex_root: Path) -> Mapping[str, Any]:
    adventurex = adventurex_root.resolve(strict=True)
    archive = _inside_regular(archive_path, adventurex)
    audit = Audit()
    sidecar = archive.with_name(archive.name + ".sha256")
    digest = sha256_file(archive)
    audit.check(
        "archive_sha256_sidecar",
        sidecar.is_file()
        and sidecar.read_text(encoding="ascii") == f"{digest}  {archive.name}\n",
        sidecar.relative_to(adventurex).as_posix() if sidecar.exists() else "missing",
    )

    temp_root = adventurex / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="omega_v3_delta_audit_", dir=temp_root) as temporary:
        extracted = _extract_archive(archive, Path(temporary), audit)
        manifest_path = extracted / "candidate_manifest.json"
        sums_path = extracted / "SHA256SUMS"
        manifest = _strict_json(manifest_path)
        audit.check("manifest_schema", manifest.get("schema") == CANDIDATE_SCHEMA, manifest.get("schema"))
        audit.check("manifest_candidate_id", manifest.get("candidate_id") == CANDIDATE_ID, manifest.get("candidate_id"))
        packaging = manifest.get("packaging", {})
        audit.check(
            "packaging_contract",
            isinstance(packaging, Mapping)
            and packaging.get("delta_only") is True
            and packaging.get("compression") == "none"
            and packaging.get("tar_format") == "USTAR"
            and packaging.get("deterministic_metadata") is True
            and packaging.get("no_overwrite") is True
            and packaging.get("sha256sums_exact_coverage") is True,
            packaging,
        )

        base = manifest.get("immutable_base_v2")
        expected_base = {
            "source_path_relative_to_adventurex": BASE_RELATIVE_PATH,
            "sha256": BASE_SHA256,
            "bytes": BASE_BYTES,
            "bundled_in_delta": False,
            "immutable_reference_only": True,
        }
        audit.check("immutable_v2_reference_exact", base == expected_base, base)
        base_source = adventurex / BASE_RELATIVE_PATH
        audit.check("immutable_v2_source_exists", base_source.is_file() and not base_source.is_symlink(), BASE_RELATIVE_PATH)
        audit.check(
            "immutable_v2_source_bytes",
            base_source.is_file() and base_source.stat().st_size == BASE_BYTES,
            base_source.stat().st_size if base_source.is_file() else None,
        )
        audit.check(
            "immutable_v2_source_sha256",
            base_source.is_file() and sha256_file(base_source) == BASE_SHA256,
            sha256_file(base_source) if base_source.is_file() else None,
        )

        bindings = manifest.get("receipt_bindings")
        audit.check(
            "two_explicit_receipt_bindings",
            isinstance(bindings, Mapping) and set(bindings) == {"x5", "vision"},
            sorted(bindings) if isinstance(bindings, Mapping) else bindings,
        )
        bound_payloads: dict[str, Mapping[str, Any]] = {}
        vision_transition_context: dict[str, Any] = {}
        if isinstance(bindings, Mapping):
            for role in ("x5", "vision"):
                record = bindings.get(role)
                valid_shape = (
                    isinstance(record, Mapping)
                    and record.get("role") == role
                    and record.get("copied_into_delta") is False
                    and record.get("claims_inferred") is False
                    and isinstance(record.get("source_path_relative_to_adventurex"), str)
                    and isinstance(record.get("sha256"), str)
                    and isinstance(record.get("bytes"), int)
                )
                audit.check(f"{role}_receipt_binding_shape", valid_shape, record)
                if not valid_shape:
                    continue
                relative = str(record["source_path_relative_to_adventurex"])
                _safe_relative(relative)
                source = _inside_regular(adventurex / Path(*PurePosixPath(relative).parts), adventurex)
                payload = _strict_json(source)
                bound_payloads[role] = payload
                audit.check(f"{role}_receipt_sha256", sha256_file(source) == record["sha256"], sha256_file(source))
                audit.check(f"{role}_receipt_bytes", source.stat().st_size == record["bytes"], source.stat().st_size)
                schema = payload.get("schema", payload.get("schema_version"))
                audit.check(f"{role}_receipt_schema_binding", schema == record.get("schema"), schema)
                if role == "vision":
                    source_relative = record.get(
                        "source_receipt_path_relative_to_adventurex"
                    )
                    source_sha = record.get("source_receipt_sha256")
                    source_ok = (
                        isinstance(source_relative, str)
                        and isinstance(source_sha, str)
                        and record.get("formal_coverage_guarantee") is False
                        and record.get("vision_qualification_inferred") is False
                    )
                    if source_ok:
                        _safe_relative(source_relative)
                        source_receipt_path = _inside_regular(
                            adventurex
                            / Path(*PurePosixPath(source_relative).parts),
                            adventurex,
                        )
                        source_ok = (
                            sha256_file(source_receipt_path) == source_sha
                            and isinstance(payload.get("source_receipt"), Mapping)
                            and payload["source_receipt"].get("sha256")
                            == source_sha
                        )
                        if source_ok:
                            vision_transition_context["source_payload"] = _strict_json(
                                source_receipt_path
                            )
                    qualification = payload.get("qualification")
                    terminology = payload.get("terminology_correction")
                    scope = payload.get("scope_clarification")
                    vision_boundary_ok = (
                        payload.get("schema")
                        == "rootscope.omega-vision-truth-boundary-addendum.v1"
                        and payload.get("status")
                        == "BOUNDARY_CORRECTION_NO_REEVALUATION"
                        and isinstance(terminology, Mapping)
                        and terminology.get(
                            "formal_distribution_free_coverage_guarantee"
                        )
                        is False
                        and isinstance(scope, Mapping)
                        and scope.get("holdout_reevaluated_for_this_addendum")
                        is False
                        and scope.get("inference_rerun_for_this_addendum") is False
                        and isinstance(qualification, Mapping)
                        and qualification.get("model_qualified") is False
                        and qualification.get("physical_print_domain_qualified")
                        is False
                        and qualification.get("camera_qualified") is False
                        and qualification.get("bpu_plant_model_qualified") is False
                        and qualification.get("selected_bin") is None
                        and qualification.get("production_integration_allowed")
                        is False
                        and isinstance(payload.get("authority"), Mapping)
                        and bool(payload["authority"])
                        and all(
                            value is False
                            for value in payload["authority"].values()
                        )
                    )
                    audit.check(
                        "vision_truth_addendum_and_source_bound",
                        source_ok and vision_boundary_ok,
                        {
                            "source_ok": source_ok,
                            "vision_boundary_ok": vision_boundary_ok,
                        },
                    )
                    vision_transition_context["addendum"] = payload
                    vision_transition_context["binding"] = record

        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise ValueError("manifest files array is missing")
        paths: set[str] = set()
        manifest_by_path: dict[str, Mapping[str, Any]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("manifest file record is not an object")
            relative = str(record.get("path", ""))
            _safe_relative(relative)
            if relative in paths:
                raise ValueError(f"duplicate manifest payload path: {relative}")
            paths.add(relative)
            manifest_by_path[relative] = record
            if not _allowlisted_payload_path(relative):
                raise ValueError(f"payload is outside the Ω allowlist: {relative}")
            source = extracted / Path(*PurePosixPath(relative).parts)
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"manifest payload missing or unsafe: {relative}")
            if source.stat().st_size != record.get("bytes"):
                raise ValueError(f"manifest payload size mismatch: {relative}")
            if sha256_file(source) != record.get("sha256"):
                raise ValueError(f"manifest payload hash mismatch: {relative}")
            expected_mode = (
                "0755"
                if relative
                in {
                    HELPER_PACKAGE_PATH,
                    "rootscope/deploy/x5/verify_omega_llm_role_cluster_foreground.sh",
                }
                else "0644"
            )
            if record.get("mode") != expected_mode:
                raise ValueError(f"manifest payload mode mismatch: {relative}")
            source_relative = record.get("source_path_relative_to_adventurex")
            if not isinstance(source_relative, str):
                raise ValueError(f"manifest source binding is missing: {relative}")
            _safe_relative(source_relative)
            _scan_payload(relative, source)
        audit.check("payload_paths_allowlisted", True, len(paths))
        audit.check("helper_included", HELPER_PACKAGE_PATH in paths, HELPER_PACKAGE_PATH)
        audit.check(
            "candidate_minimal_app_init_mapping",
            manifest_by_path.get("rootscope/app/__init__.py", {}).get(
                "source_path_relative_to_adventurex"
            )
            == "rootscope/deploy/x5/omega_standalone_app_init.py",
            manifest_by_path.get("rootscope/app/__init__.py", {}).get(
                "source_path_relative_to_adventurex"
            ),
        )
        app_init_text = (extracted / "rootscope/app/__init__.py").read_text(
            encoding="utf-8"
        )
        app_init_tree = ast.parse(app_init_text)
        audit.check(
            "candidate_minimal_app_init_has_no_imports",
            not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(app_init_tree)),
            "no eager imports",
        )
        audit.check(
            "omega_dashboard_dependencies_present",
            {
                "rootscope/app/web/__init__.py",
                "rootscope/app/web/server.py",
                "rootscope/app/web/state_store.py",
            }.issubset(paths),
            "app.web server/state store",
        )
        audit.check(
            "foreground_llm_role_helper_present",
            "rootscope/deploy/x5/verify_omega_llm_role_cluster_foreground.sh"
            in paths,
            "explicit foreground loopback only",
        )
        audit.check(
            "required_omega_components_present",
            all(
                any(path.startswith(prefix) for path in paths)
                for prefix in (
                    "rootscope/app/omega/",
                    "rootscope/app/omega_knowledge/",
                    "rootscope/app/omega_runtime/",
                    "rootscope/app/omega_bpu_aux/",
                    "rootscope/configs/omega/",
                    "rootscope/tests/test_omega_",
                )
            ),
            len(paths),
        )
        audit.check(
            "local_only_vision_tests_excluded",
            not (set(LOCAL_ONLY_TEST_EXCLUSIONS) & paths),
            sorted(set(LOCAL_ONLY_TEST_EXCLUSIONS) & paths),
        )
        if vision_transition_context:
            source_payload = vision_transition_context.get("source_payload")
            addendum = vision_transition_context.get("addendum")
            binding = vision_transition_context.get("binding")
            transition = (
                addendum.get("source_hash_transition")
                if isinstance(addendum, Mapping)
                else None
            )
            artifact_sha = (
                source_payload.get("artifact_sha256")
                if isinstance(source_payload, Mapping)
                else None
            )
            packaged_ood = manifest_by_path.get(
                "rootscope/app/omega_vision/ood.py"
            )
            packaged_builder = manifest_by_path.get(
                "rootscope/training/omega_vision/build_evidence.py"
            )
            transition_ok = (
                isinstance(transition, Mapping)
                and isinstance(artifact_sha, Mapping)
                and isinstance(binding, Mapping)
                and isinstance(packaged_ood, Mapping)
                and isinstance(packaged_builder, Mapping)
                and transition.get("app/omega_vision/ood.py_before_sha256")
                == artifact_sha.get("implementation_ood")
                and transition.get("app/omega_vision/ood.py_after_sha256")
                == packaged_ood.get("sha256")
                and transition.get(
                    "training/omega_vision/build_evidence.py_sha256"
                )
                == artifact_sha.get("implementation_evidence_builder")
                and transition.get(
                    "training/omega_vision/build_evidence.py_sha256"
                )
                == packaged_builder.get("sha256")
                and binding.get("packaged_ood_source_sha256")
                == packaged_ood.get("sha256")
                and binding.get("packaged_evidence_builder_sha256")
                == packaged_builder.get("sha256")
            )
            audit.check(
                "vision_hash_transition_matches_packaged_sources",
                transition_ok,
                {
                    "transition": transition,
                    "packaged_ood": packaged_ood.get("sha256")
                    if isinstance(packaged_ood, Mapping)
                    else None,
                    "packaged_builder": packaged_builder.get("sha256")
                    if isinstance(packaged_builder, Mapping)
                    else None,
                },
            )

        actual = {
            path.relative_to(extracted).as_posix()
            for path in extracted.rglob("*")
            if path.is_file()
        }
        expected = paths | {"candidate_manifest.json", "SHA256SUMS"}
        audit.check("archive_exact_file_coverage", actual == expected, {"extra": sorted(actual - expected), "missing": sorted(expected - actual)})
        sums = _parse_sums(sums_path)
        covered = expected - {"SHA256SUMS"}
        audit.check("sha256sums_exact_path_coverage", set(sums) == covered, {"covered": len(sums), "expected": len(covered)})
        sums_match = all(
            sha256_file(extracted / Path(*PurePosixPath(relative).parts)) == expected_sha
            for relative, expected_sha in sums.items()
        )
        audit.check("sha256sums_all_match", sums_match, len(sums))

        qualification = manifest.get("qualification")
        authority = manifest.get("authority")
        if not isinstance(qualification, Mapping) or not isinstance(authority, Mapping):
            raise ValueError("qualification or authority object is missing")
        audit.check("selected_bin_null", qualification.get("selected_bin") is None, qualification.get("selected_bin"))
        for field in (
            "readonly_llm_long_run_qualified",
            "bpu_plant_model_qualified",
            "plant_domain_accuracy_qualified",
            "provisional_dataset_qualified",
            "physical_closure",
            "production_integration_allowed",
        ):
            audit.check(f"qualification_{field}_false", qualification.get(field) is False, qualification.get(field))
        audit.check("authority_all_false", bool(authority) and all(value is False for value in authority.values()), authority)

        if "x5" in bound_payloads:
            observations = _derive_observations(bound_payloads["x5"])
            audit.check(
                "x5_observation_contract_matches",
                qualification.get("x5_receipt_observation_contract_recognized") is observations["recognized"]
                and qualification.get("cpu_onnx_smoke_observed_pass") is observations["cpu"]
                and qualification.get("readonly_llm_foreground_loopback_smoke_observed_pass") is observations["llm"],
                observations,
            )
            audit.check("candidate_status_not_upgraded", manifest.get("status") == _expected_status(observations), manifest.get("status"))

        composition = {
            "schema": "rootscope.omega-v3-delta-composition.v1",
            "candidate_id": CANDIDATE_ID,
            "immutable_base_v2": base,
            "receipt_bindings": bindings,
            "files": records,
            "qualification": qualification,
            "authority": authority,
        }
        expected_root = hashlib.sha256(_canonical_compact(composition)).hexdigest()
        audit.check("composition_root", manifest.get("composition_root_sha256") == expected_root, expected_root)

        contents = manifest.get("contents", {})
        audit.check(
            "content_exclusions_declared",
            isinstance(contents, Mapping)
            and contents.get("xrd_runtime_included") is False
            and contents.get("training_artifacts_included") is False
            and contents.get("secret_or_key_material_included") is False
            and contents.get("absolute_temporary_paths_included") is False
            and contents.get("v2_archive_duplicated") is False,
            contents,
        )
        audit.check(
            "portable_test_scope_declared",
            isinstance(contents, Mapping)
            and contents.get("portable_tests_only") is True
            and contents.get("local_only_tests_excluded")
            == [
                {"path": path, "reason": reason}
                for path, reason in sorted(LOCAL_ONLY_TEST_EXCLUSIONS.items())
            ],
            contents.get("local_only_tests_excluded")
            if isinstance(contents, Mapping)
            else None,
        )
        audit.check("v2_tar_not_packaged", not any(path.endswith(".tar") for path in paths), sorted(path for path in paths if path.endswith(".tar")))
        audit.check("bound_receipts_not_packaged", not any(path.startswith("evidence/") for path in paths), sorted(path for path in paths if path.startswith("evidence/")))
        probe = _strict_json(extracted / BPU_AUX_CONFIG_PATH)
        images = probe.get("images")
        probe_images_valid = (
            isinstance(images, list)
            and len(images) == 4
            and len(
                {
                    item.get("image_id")
                    for item in images
                    if isinstance(item, Mapping)
                }
            )
            == 4
        )
        if probe_images_valid:
            for item in images:
                if not isinstance(item, Mapping):
                    probe_images_valid = False
                    break
                image_path = item.get("path")
                image_sha = item.get("sha256")
                pure_path = (
                    PurePosixPath(image_path)
                    if isinstance(image_path, str)
                    else PurePosixPath(".")
                )
                if (
                    not isinstance(image_path, str)
                    or not pure_path.is_absolute()
                    or not image_path.startswith(
                        "/opt/rootscope/rootscope_omega_v3_inputs/bpu_aux/"
                    )
                    or "*" in image_path
                    or "?" in image_path
                    or not isinstance(image_sha, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", image_sha)
                ):
                    probe_images_valid = False
                    break
        model = probe.get("model")
        audit.check(
            "bpu_aux_probe_config_exact_boundary",
            probe.get("schema_version")
            == "rootscope.omega.bpu-aux-input-manifest.v1"
            and set(probe)
            == {
                "schema_version",
                "run_id",
                "model",
                "top_k",
                "warmup_runs",
                "images",
            }
            and isinstance(model, Mapping)
            and set(model) == {"path", "sha256", "output_semantics"}
            and model.get("path") == BPU_VENDOR_MODEL_PATH
            and PurePosixPath(str(model.get("path"))).is_absolute()
            and model.get("sha256") == BPU_VENDOR_MODEL_SHA256
            and model.get("output_semantics") == "PROBABILITIES"
            and probe.get("top_k") == 5
            and probe.get("warmup_runs") == 1
            and images == list(BPU_AUX_IMAGE_INPUTS)
            and probe_images_valid,
            {
                "model": model,
                "image_count": len(images) if isinstance(images, list) else None,
            },
        )
        audit.check(
            "bpu_aux_images_not_packaged",
            not any(PurePosixPath(path).suffix.lower() in {".jpg", ".jpeg", ".png"} for path in paths),
            sorted(path for path in paths if PurePosixPath(path).suffix.lower() in {".jpg", ".jpeg", ".png"}),
        )
        vision_board_required = {
            VISION_BOARD_REPLAY_SOURCE_PATH,
            VISION_BOARD_REPLAY_CONFIG_PATH,
            VISION_BOARD_REPLAY_TEST_PATH,
        }
        audit.check(
            "vision_board_replay_files_present",
            vision_board_required <= paths,
            sorted(vision_board_required - paths),
        )
        vision_board_config_path = extracted / VISION_BOARD_REPLAY_CONFIG_PATH
        vision_board_config = _strict_json(vision_board_config_path)
        vision_truth = vision_board_config.get("truth_boundary")
        vision_authority = vision_board_config.get("authority")
        vision_provenance = vision_board_config.get("calibration_provenance")
        audit.check(
            "vision_board_replay_complete_contract_frozen",
            sha256_file(vision_board_config_path)
            == VISION_BOARD_REPLAY_CONFIG_SHA256
            and vision_board_config.get("schema_version")
            == "rootscope.omega-vision-board-replay-manifest.v1"
            and isinstance(vision_truth, Mapping)
            and vision_truth.get("model_qualified") is False
            and vision_truth.get("plant_domain_accuracy_qualified") is False
            and vision_truth.get("camera_qualified") is False
            and vision_truth.get("bpu_used") is False
            and vision_truth.get("physical_completion") is False
            and vision_truth.get("registered_demo_references_are_holdout")
            is False
            and isinstance(vision_authority, Mapping)
            and bool(vision_authority)
            and all(value is False for value in vision_authority.values())
            and isinstance(vision_provenance, Mapping)
            and vision_provenance.get("holdout_reevaluated_for_board_replay")
            is False
            and vision_provenance.get(
                "formal_distribution_free_coverage_guarantee"
            )
            is False,
            {
                "sha256": sha256_file(vision_board_config_path),
                "truth_boundary": vision_truth,
                "authority": vision_authority,
                "calibration_provenance": vision_provenance,
            },
        )

        helper_path = extracted / HELPER_PACKAGE_PATH
        helper_text = helper_path.read_text(encoding="utf-8")
        helper_forbidden = (
            "import socket",
            "import requests",
            "import urllib",
            "import serial",
            "import cv2",
            "import gpiod",
            "import RPi",
            "from hobot_dnn",
            "subprocess.",
            "os.system(",
            "systemctl ",
            "VideoCapture(",
            "Serial(",
        )
        audit.check(
            "board_helper_no_external_or_device_actions",
            not any(token in helper_text for token in helper_forbidden),
            [token for token in helper_forbidden if token in helper_text],
        )
        helper = _load_helper(helper_path)
        helper_result = helper.verify_extracted_delta(extracted)
        audit.check(
            "board_helper_verify_only_pass",
            helper_result.get("status")
            == "PASS_HASHES_ZERO_AUTHORITY_NOT_PHYSICAL_QUALIFICATION"
            and helper_result.get("pure_cpu_smoke_executed") is False,
            helper_result.get("status"),
        )

    build_receipt_path = archive.parent / "release_build_receipt.json"
    build_receipt = _strict_json(build_receipt_path)
    audit.check(
        "build_receipt_schema",
        build_receipt.get("schema") == "rootscope.omega-v3-delta-build-receipt.v1",
        build_receipt.get("schema"),
    )
    audit.check("build_receipt_archive_sha", build_receipt.get("archive", {}).get("sha256") == digest, build_receipt.get("archive", {}).get("sha256"))
    audit.check("build_receipt_archive_bytes", build_receipt.get("archive", {}).get("bytes") == archive.stat().st_size, build_receipt.get("archive", {}).get("bytes"))
    return audit.result(archive, adventurex)


def _parser() -> argparse.ArgumentParser:
    adventurex = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adventurex-root", type=Path, default=adventurex)
    parser.add_argument(
        "--archive",
        type=Path,
        default=adventurex / "output/releases" / CANDIDATE_ID / CANDIDATE_ARCHIVE,
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_delta(args.archive, args.adventurex_root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json is not None:
        output = args.output_json.resolve()
        adventurex = args.adventurex_root.resolve(strict=True)
        try:
            output.relative_to(adventurex)
        except ValueError as exc:
            raise ValueError("audit output must stay below AdventureX") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
