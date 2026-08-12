#!/usr/bin/env python3
"""Build the deterministic, zero-authority RootScope-Ω v3 delta candidate.

The delta never duplicates the immutable v2 field bundle.  It verifies and
references that archive by exact byte count and SHA-256, then packages only an
explicit allowlist of standalone Ω source, configuration, documentation, static
assets, and tests.  X5 and vision receipts are mandatory external inputs; they
are hash-bound but are not copied into the delta and never upgrade BPU, long-run
LLM, vision-accuracy, or physical-closure claims.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from typing import Any


BUILD_DATE = "2026-07-23"
CANDIDATE_ID = "rootscope_omega_v3_delta_candidate_v1"
CANDIDATE_ARCHIVE = f"{CANDIDATE_ID}.tar"
CANDIDATE_SCHEMA = "rootscope.omega-v3-delta-candidate.v1"
BASE_RELATIVE_PATH = (
    "output/releases/rootscope_x5_field_bundle_v2/"
    "rootscope_x5_field_bundle_v2.tar"
)
BASE_SHA256 = "e6627685170252004d118bf77a690a9f89ad3afa274910697554d0f5cc8c3ebb"
BASE_BYTES = 696_832_000
HELPER_RELATIVE_PATH = "tools/release/verify_run_rootscope_omega_v3_delta.py"
HELPER_PACKAGE_PATH = "tools/verify_run_rootscope_omega_v3_delta.py"
BPU_VENDOR_MODEL_PATH = "/opt/hobot/model/x5/basic/mobilenetv2_224x224_nv12.bin"
BPU_VENDOR_MODEL_SHA256 = "3e2b7c46fc3b3a6d07a5326c0b9632fe98fe5ca38835346ab2eedc22ed427158"
VISION_BOARD_REPLAY_SOURCE_PATH = "rootscope/app/omega_vision/board_replay.py"
VISION_BOARD_REPLAY_CONFIG_PATH = (
    "rootscope/configs/omega/vision_board_replay_new_x5_20260723.json"
)
VISION_BOARD_REPLAY_TEST_PATH = (
    "rootscope/tests/test_omega_vision_board_replay.py"
)
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
    "test_omega_vision_dataset.py": (
        "requires the external 78-image dataset and its manifests"
    ),
    "test_omega_vision_evidence.py": (
        "requires external frozen vision evidence receipts"
    ),
}

_ALLOWED_SUFFIXES = frozenset(
    {
        ".py",
        ".json",
        ".md",
        ".html",
        ".css",
        ".js",
        ".svg",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".sh",
    }
)
_FORBIDDEN_PATH_PARTS = frozenset(
    {
        "__pycache__",
        ".git",
        ".ssh",
        "secrets",
        "credentials",
        "private_keys",
        "keys",
    }
)
_FORBIDDEN_FILE_SUFFIXES = frozenset(
    {".pyc", ".pyo", ".pem", ".key", ".p12", ".pfx", ".kdb", ".bin", ".onnx", ".pt", ".pth"}
)
_FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
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
)
_TEMP_ABSOLUTE_PATTERNS = (
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


@dataclass(frozen=True)
class SourceEntry:
    source: Path
    package_path: str
    category: str
    mode: int = 0o644


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_compact_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _safe_package_path(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError(f"unsafe package path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe package path: {value!r}")
    return value


def _inside_regular(path: Path, root: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} symlink is forbidden: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} must stay below AdventureX: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be one regular file: {path}")
    return resolved


def _strict_json_object(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"receipt contains non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"receipt contains duplicate JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("receipt must contain one JSON object")
    return payload


def validate_receipt(path: Path, adventurex_root: Path, *, role: str) -> Mapping[str, Any]:
    adventurex = adventurex_root.resolve(strict=True)
    source = _inside_regular(path, adventurex, label=f"{role} receipt")
    if source.suffix.lower() != ".json":
        raise ValueError(f"{role} receipt must be JSON")
    payload = _strict_json_object(source)
    schema = payload.get("schema", payload.get("schema_version"))
    if schema is not None and not isinstance(schema, str):
        raise ValueError(f"{role} receipt schema must be a string when present")
    return {
        "role": role,
        "source_path_relative_to_adventurex": source.relative_to(adventurex).as_posix(),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
        "schema": schema,
        "copied_into_delta": False,
        "claims_inferred": False,
        "_payload": payload,
    }


def validate_vision_truth_receipt(
    receipt: Mapping[str, Any], adventurex_root: Path
) -> Mapping[str, Any]:
    payload = receipt["_payload"]
    if (
        payload.get("schema") != "rootscope.omega-vision-truth-boundary-addendum.v1"
        or payload.get("status") != "BOUNDARY_CORRECTION_NO_REEVALUATION"
    ):
        raise ValueError(
            "vision receipt must be the truth-boundary addendum, not the superseded "
            "consolidated observation alone"
        )
    terminology = payload.get("terminology_correction")
    scope = payload.get("scope_clarification")
    qualification = payload.get("qualification")
    authority = payload.get("authority")
    if (
        not isinstance(terminology, Mapping)
        or terminology.get("formal_distribution_free_coverage_guarantee") is not False
        or not isinstance(scope, Mapping)
        or scope.get("holdout_reevaluated_for_this_addendum") is not False
        or scope.get("inference_rerun_for_this_addendum") is not False
        or not isinstance(qualification, Mapping)
        or qualification.get("model_qualified") is not False
        or qualification.get("physical_print_domain_qualified") is not False
        or qualification.get("camera_qualified") is not False
        or qualification.get("bpu_plant_model_qualified") is not False
        or qualification.get("selected_bin") is not None
        or qualification.get("production_integration_allowed") is not False
        or not _all_false_mapping(authority)
    ):
        raise ValueError("vision truth-boundary addendum cannot support qualification")
    source_receipt = payload.get("source_receipt")
    if not isinstance(source_receipt, Mapping):
        raise ValueError("vision addendum source receipt binding is missing")
    source_relative = source_receipt.get("path")
    source_sha = source_receipt.get("sha256")
    if not isinstance(source_relative, str) or not isinstance(source_sha, str):
        raise ValueError("vision addendum source receipt path/hash is missing")
    pure = PurePosixPath(source_relative)
    if (
        pure.is_absolute()
        or len(pure.parts) != 1
        or pure.name != source_relative
        or not re.fullmatch(r"[0-9a-f]{64}", source_sha)
    ):
        raise ValueError("vision addendum source receipt binding is unsafe")
    adventurex = adventurex_root.resolve(strict=True)
    addendum_path = adventurex / str(receipt["source_path_relative_to_adventurex"])
    source_path = _inside_regular(
        addendum_path.parent / source_relative,
        adventurex,
        label="vision source receipt",
    )
    if sha256_file(source_path) != source_sha:
        raise ValueError("vision addendum source receipt SHA-256 mismatch")
    source_payload = _strict_json_object(source_path)
    artifact_sha = source_payload.get("artifact_sha256")
    transition = payload.get("source_hash_transition")
    ood_source = _inside_regular(
        adventurex / "rootscope/app/omega_vision/ood.py",
        adventurex,
        label="Omega vision OOD source",
    )
    evidence_builder = _inside_regular(
        adventurex / "rootscope/training/omega_vision/build_evidence.py",
        adventurex,
        label="Omega vision evidence builder",
    )
    ood_after_sha = sha256_file(ood_source)
    evidence_builder_sha = sha256_file(evidence_builder)
    if (
        not isinstance(artifact_sha, Mapping)
        or not isinstance(transition, Mapping)
        or transition.get("app/omega_vision/ood.py_before_sha256")
        != artifact_sha.get("implementation_ood")
        or transition.get("app/omega_vision/ood.py_after_sha256")
        != ood_after_sha
        or transition.get("training/omega_vision/build_evidence.py_sha256")
        != artifact_sha.get("implementation_evidence_builder")
        or transition.get("training/omega_vision/build_evidence.py_sha256")
        != evidence_builder_sha
    ):
        raise ValueError(
            "vision addendum source-hash transition does not bind original receipt "
            "and current packaged sources"
        )
    return {
        **{key: value for key, value in receipt.items() if key != "_payload"},
        "_payload": payload,
        "source_receipt_path_relative_to_adventurex": source_path.relative_to(
            adventurex
        ).as_posix(),
        "source_receipt_sha256": source_sha,
        "formal_coverage_guarantee": False,
        "vision_qualification_inferred": False,
        "packaged_ood_source_sha256": ood_after_sha,
        "packaged_evidence_builder_sha256": evidence_builder_sha,
    }


def validate_bpu_aux_probe_config(path: Path) -> Mapping[str, Any]:
    payload = _strict_json_object(path)
    if set(payload) != {
        "schema_version",
        "run_id",
        "model",
        "top_k",
        "warmup_runs",
        "images",
    }:
        raise ValueError("BPU auxiliary probe config top-level contract changed")
    if payload.get("schema_version") != "rootscope.omega.bpu-aux-input-manifest.v1":
        raise ValueError("BPU auxiliary probe config schema changed")
    model = payload.get("model")
    if (
        not isinstance(model, Mapping)
        or set(model) != {"path", "sha256", "output_semantics"}
        or model.get("path") != BPU_VENDOR_MODEL_PATH
        or not PurePosixPath(str(model.get("path"))).is_absolute()
        or model.get("sha256") != BPU_VENDOR_MODEL_SHA256
        or model.get("output_semantics") != "PROBABILITIES"
    ):
        raise ValueError("BPU auxiliary vendor model contract changed")
    if payload.get("top_k") != 5 or payload.get("warmup_runs") != 1:
        raise ValueError("BPU auxiliary probe run parameters changed")
    images = payload.get("images")
    if images != list(BPU_AUX_IMAGE_INPUTS):
        raise ValueError("BPU auxiliary explicit image/hash allowlist changed")
    for item in images:
        path_value = str(item["path"])
        if (
            not PurePosixPath(path_value).is_absolute()
            or "*" in path_value
            or "?" in path_value
        ):
            raise ValueError("BPU auxiliary image paths must be explicit POSIX absolutes")
    return payload


def validate_vision_board_replay_config(path: Path) -> Mapping[str, Any]:
    """Bind the complete board replay manifest, including calibration evidence."""

    payload = _strict_json_object(path)
    if sha256_file(path) != VISION_BOARD_REPLAY_CONFIG_SHA256:
        raise ValueError("Omega vision board replay manifest SHA-256 changed")
    required = {
        "schema_version",
        "run_id",
        "board_identity",
        "model",
        "class_order",
        "preprocess",
        "calibration",
        "calibration_provenance",
        "pc_reference",
        "images",
        "truth_boundary",
        "authority",
    }
    truth = payload.get("truth_boundary")
    authority = payload.get("authority")
    provenance = payload.get("calibration_provenance")
    if (
        set(payload) != required
        or payload.get("schema_version")
        != "rootscope.omega-vision-board-replay-manifest.v1"
        or not isinstance(truth, Mapping)
        or truth.get("model_qualified") is not False
        or truth.get("plant_domain_accuracy_qualified") is not False
        or truth.get("camera_qualified") is not False
        or truth.get("bpu_used") is not False
        or truth.get("physical_completion") is not False
        or truth.get("registered_demo_references_are_holdout") is not False
        or not _all_false_mapping(authority)
        or not isinstance(provenance, Mapping)
        or provenance.get("holdout_reevaluated_for_board_replay") is not False
        or provenance.get("formal_distribution_free_coverage_guarantee") is not False
    ):
        raise ValueError("Omega vision board replay truth boundary changed")
    return payload


def _all_false_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        item is False for item in value.values()
    )


def derive_x5_smoke_observations(payload: Mapping[str, Any]) -> Mapping[str, bool]:
    """Recognize only a narrow, explicitly actual and zero-authority contract."""

    observations = payload.get("actual_observations")
    authority = payload.get("authority")
    if not isinstance(observations, Mapping) or not _all_false_mapping(authority):
        return {
            "receipt_observation_contract_recognized": False,
            "cpu_onnx_smoke_passed": False,
            "readonly_llm_foreground_loopback_smoke_passed": False,
        }
    cpu = observations.get("cpu_onnx_smoke")
    llm = observations.get("readonly_llm_foreground_loopback_smoke")
    cpu_passed = (
        isinstance(cpu, Mapping)
        and cpu.get("executed") is True
        and cpu.get("passed") is True
    )
    llm_passed = (
        isinstance(llm, Mapping)
        and llm.get("executed") is True
        and llm.get("passed") is True
        and llm.get("process_stopped") is True
        and llm.get("port_closed_after_stop") is True
    )
    return {
        "receipt_observation_contract_recognized": True,
        "cpu_onnx_smoke_passed": cpu_passed,
        "readonly_llm_foreground_loopback_smoke_passed": llm_passed,
    }


def _candidate_status(observations: Mapping[str, bool]) -> str:
    if observations["cpu_onnx_smoke_passed"] and observations[
        "readonly_llm_foreground_loopback_smoke_passed"
    ]:
        return "SAFE_CPU_PLUS_READONLY_LLM_CANDIDATE"
    if observations["cpu_onnx_smoke_passed"]:
        return "SAFE_CPU_QUALIFIED_CANDIDATE"
    return "SOURCE_DELTA_CANDIDATE_RECEIPTS_BOUND"


def _scan_python_imports(path: Path, text: str) -> None:
    try:
        tree = ast.parse(text, filename=path.as_posix())
    except SyntaxError as exc:
        raise ValueError(f"Python source does not parse: {path}: {exc}") from exc
    for node in ast.walk(tree):
        roots: list[str] = []
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.append(node.module.split(".", 1)[0])
        forbidden = sorted(set(roots) & _FORBIDDEN_IMPORT_ROOTS)
        if forbidden:
            raise ValueError(f"XRD/frozen runtime import is forbidden in {path}: {forbidden}")


def _scan_text_source(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"allowlisted source must be UTF-8 text: {path}") from exc
    for pattern in _TEMP_ABSOLUTE_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"absolute temporary path is forbidden in {path}")
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"possible embedded secret is forbidden in {path}")
    if path.suffix.lower() == ".py":
        _scan_python_imports(path, text)


def _iter_tree(root: Path) -> Iterable[Path]:
    if root.is_symlink():
        raise ValueError(f"source tree symlink is forbidden: {root}")
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"source symlink is forbidden: {path}")
        if path.is_file():
            yield path


def collect_delta_sources(adventurex_root: Path) -> list[SourceEntry]:
    """Collect the explicit text-only Ω delta allowlist."""

    adventurex = adventurex_root.resolve(strict=True)
    rootscope = adventurex / "rootscope"
    if not rootscope.is_dir() or rootscope.is_symlink():
        raise ValueError("AdventureX rootscope source tree is missing or unsafe")
    entries: dict[str, SourceEntry] = {}

    def add(source: Path, package_path: str, category: str, mode: int = 0o644) -> None:
        resolved = _inside_regular(source, adventurex, label="delta source")
        relative_parts = {part.casefold() for part in resolved.relative_to(adventurex).parts}
        if relative_parts & _FORBIDDEN_PATH_PARTS:
            raise ValueError(f"forbidden source path component: {source}")
        if resolved.suffix.lower() in _FORBIDDEN_FILE_SUFFIXES:
            raise ValueError(f"forbidden source file type: {source}")
        if resolved.suffix.lower() not in _ALLOWED_SUFFIXES:
            raise ValueError(f"source type is not allowlisted: {source}")
        safe_path = _safe_package_path(package_path)
        if safe_path in entries:
            raise ValueError(f"duplicate delta package path: {safe_path}")
        _scan_text_source(resolved)
        entries[safe_path] = SourceEntry(
            source=resolved,
            package_path=safe_path,
            category=category,
            mode=mode,
        )

    def add_tree(relative: str, category: str, *, required: bool) -> None:
        base = rootscope / relative
        if not base.exists():
            if required:
                raise ValueError(f"required Ω source tree is missing: rootscope/{relative}")
            return
        if not base.is_dir() or base.is_symlink():
            raise ValueError(f"Ω source tree is unsafe: rootscope/{relative}")
        count = 0
        for source in _iter_tree(base):
            if "__pycache__" in source.parts or source.suffix.lower() in {".pyc", ".pyo"}:
                continue
            add(
                source,
                f"rootscope/{source.relative_to(rootscope).as_posix()}",
                category,
            )
            count += 1
        if required and count == 0:
            raise ValueError(f"required Ω source tree is empty: rootscope/{relative}")

    add(
        rootscope / "deploy/x5/omega_standalone_app_init.py",
        "rootscope/app/__init__.py",
        "candidate_only_minimal_app_init",
    )
    add(
        rootscope / "tests/__init__.py",
        "rootscope/tests/__init__.py",
        "package_contract",
    )
    if (rootscope / "pyproject.toml").is_file():
        add(
            rootscope / "pyproject.toml",
            "rootscope/pyproject.toml",
            "package_contract",
        )

    for tree, category in (
        ("app/omega", "omega_evidence_core"),
        ("app/omega_knowledge", "omega_knowledge"),
        ("app/omega_runtime", "omega_runtime"),
        ("app/omega_bpu_aux", "omega_bpu_aux_support_only"),
        ("configs/omega", "omega_config"),
    ):
        add_tree(tree, category, required=True)
    add_tree("app/omega_vision", "omega_vision_optional_unqualified", required=False)
    add_tree("training/omega_vision", "omega_vision_training_source_optional", required=False)
    vision_board_config = (
        rootscope / "configs/omega/vision_board_replay_new_x5_20260723.json"
    )
    validate_vision_board_replay_config(vision_board_config)
    for relative in (
        "app/web/__init__.py",
        "app/web/server.py",
        "app/web/state_store.py",
    ):
        add(
            rootscope / relative,
            f"rootscope/{relative}",
            "omega_dashboard_dependency",
        )
    add(
        rootscope / "deploy/x5/verify_omega_llm_role_cluster_foreground.sh",
        "rootscope/deploy/x5/verify_omega_llm_role_cluster_foreground.sh",
        "explicit_foreground_loopback_llm_role_helper",
        mode=0o755,
    )
    bpu_aux_config = rootscope / "deploy/x5/bpu_aux_probe_new_x5_20260723.json"
    validate_bpu_aux_probe_config(bpu_aux_config)
    add(
        bpu_aux_config,
        "rootscope/deploy/x5/bpu_aux_probe_new_x5_20260723.json",
        "bpu_aux_explicit_hash_bound_input_config",
    )

    root_tests = sorted((rootscope / "tests").glob("test_omega_*.py"))
    if not root_tests:
        raise ValueError("no standalone Ω tests were found")
    for source in root_tests:
        if source.name in LOCAL_ONLY_TEST_EXCLUSIONS:
            continue
        add(
            source,
            f"rootscope/{source.relative_to(rootscope).as_posix()}",
            "omega_tests",
        )
    for optional_test_tree in ("tests/omega_knowledge", "tests/omega_vision"):
        base = rootscope / optional_test_tree
        if base.exists():
            for source in _iter_tree(base):
                if "__pycache__" in source.parts or source.suffix.lower() in {".pyc", ".pyo"}:
                    continue
                add(
                    source,
                    f"rootscope/{source.relative_to(rootscope).as_posix()}",
                    "omega_tests",
                )

    for relative in (
        "README.md",
        "PREEXISTING.md",
        "BUILT_DURING_EVENT.md",
        "OMEGA_V3_IMPLEMENTATION_STATUS.md",
        "OMEGA_V3_CANDIDATE_RELEASE_CHECKLIST.md",
    ):
        add(rootscope / relative, f"rootscope/{relative}", "truth_boundary_documentation")

    helper = adventurex / HELPER_RELATIVE_PATH
    add(helper, HELPER_PACKAGE_PATH, "board_zero_authority_helper", mode=0o755)
    required_board_replay_paths = {
        VISION_BOARD_REPLAY_SOURCE_PATH,
        VISION_BOARD_REPLAY_CONFIG_PATH,
        VISION_BOARD_REPLAY_TEST_PATH,
    }
    missing_board_replay = required_board_replay_paths - set(entries)
    if missing_board_replay:
        raise ValueError(
            f"required Omega vision board replay files missing: "
            f"{sorted(missing_board_replay)}"
        )
    return [entries[name] for name in sorted(entries)]


def _tar_add_bytes(archive: tarfile.TarFile, name: str, data: bytes, mode: int) -> None:
    info = tarfile.TarInfo(_safe_package_path(name))
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.type = tarfile.REGTYPE
    archive.addfile(info, BytesIO(data))


def _tar_add_file(
    archive: tarfile.TarFile, name: str, source: Path, mode: int
) -> None:
    info = tarfile.TarInfo(_safe_package_path(name))
    info.size = source.stat().st_size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.type = tarfile.REGTYPE
    with source.open("rb") as stream:
        archive.addfile(info, stream)


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _build_tar_exclusive(
    destination: Path,
    *,
    file_entries: Sequence[tuple[str, Path, int]],
    byte_entries: Sequence[tuple[str, bytes, int]],
) -> Mapping[str, Any]:
    names = [name for name, _source, _mode in file_entries] + [
        name for name, _data, _mode in byte_entries
    ]
    if len(names) != len(set(names)):
        raise ValueError("candidate tar member names must be unique")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite candidate archive: {destination}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"stale build temporary exists: {temporary}")
    file_map = {name: (source, mode) for name, source, mode in file_entries}
    byte_map = {name: (data, mode) for name, data, mode in byte_entries}
    try:
        with tarfile.open(temporary, mode="x", format=tarfile.USTAR_FORMAT) as archive:
            for name in sorted(names):
                if name in file_map:
                    source, mode = file_map[name]
                    _tar_add_file(archive, name, source, mode)
                else:
                    data, mode = byte_map[name]
                    _tar_add_bytes(archive, name, data, mode)
        with temporary.open("rb") as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "filename": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "compression": "none",
        "tar_format": "USTAR",
    }


def build_delta_candidate(
    adventurex_root: Path,
    output_dir: Path,
    x5_receipt_path: Path,
    vision_receipt_path: Path,
) -> Mapping[str, Any]:
    adventurex = adventurex_root.resolve(strict=True)
    output = output_dir.resolve()
    try:
        output.relative_to(adventurex)
    except ValueError as exc:
        raise ValueError("candidate output must stay below AdventureX") from exc
    if output.is_symlink():
        raise ValueError("candidate output symlink is forbidden")

    base_path = _inside_regular(
        adventurex / BASE_RELATIVE_PATH,
        adventurex,
        label="immutable v2 base",
    )
    if base_path.stat().st_size != BASE_BYTES or sha256_file(base_path) != BASE_SHA256:
        raise ValueError("immutable RootScope v2 base size/SHA-256 mismatch")

    x5_candidate_path = _inside_regular(
        x5_receipt_path,
        adventurex,
        label="x5 receipt",
    )
    vision_candidate_path = _inside_regular(
        vision_receipt_path,
        adventurex,
        label="vision receipt",
    )
    if x5_candidate_path == vision_candidate_path:
        raise ValueError("X5 and vision receipts must be distinct explicit files")
    x5 = validate_receipt(x5_receipt_path, adventurex, role="x5")
    vision = validate_vision_truth_receipt(
        validate_receipt(vision_receipt_path, adventurex, role="vision"),
        adventurex,
    )
    observations = derive_x5_smoke_observations(x5["_payload"])
    status = _candidate_status(observations)
    receipt_bindings = {
        role: {key: value for key, value in record.items() if key != "_payload"}
        for role, record in (("x5", x5), ("vision", vision))
    }

    sources = collect_delta_sources(adventurex)
    records = [
        {
            "path": entry.package_path,
            "source_path_relative_to_adventurex": entry.source.relative_to(
                adventurex
            ).as_posix(),
            "bytes": entry.source.stat().st_size,
            "sha256": sha256_file(entry.source),
            "mode": format(entry.mode, "04o"),
            "category": entry.category,
        }
        for entry in sources
    ]
    records.sort(key=lambda item: item["path"])
    base_reference = {
        "source_path_relative_to_adventurex": BASE_RELATIVE_PATH,
        "sha256": BASE_SHA256,
        "bytes": BASE_BYTES,
        "bundled_in_delta": False,
        "immutable_reference_only": True,
    }
    qualification = {
        "x5_receipt_bound": True,
        "vision_receipt_bound": True,
        "x5_receipt_observation_contract_recognized": observations[
            "receipt_observation_contract_recognized"
        ],
        "cpu_onnx_smoke_observed_pass": observations["cpu_onnx_smoke_passed"],
        "readonly_llm_foreground_loopback_smoke_observed_pass": observations[
            "readonly_llm_foreground_loopback_smoke_passed"
        ],
        "readonly_llm_long_run_qualified": False,
        "bpu_import_only_may_be_bound_but_is_not_model_qualification": True,
        "bpu_plant_model_qualified": False,
        "selected_bin": None,
        "plant_domain_accuracy_qualified": False,
        "provisional_dataset_qualified": False,
        "physical_closure": False,
        "production_integration_allowed": False,
    }
    authority = {
        "systemd_write": False,
        "network_configuration_write": False,
        "external_network_access": False,
        "camera_open": False,
        "serial_open": False,
        "gpio_access": False,
        "pump_command": False,
        "state_machine_write": False,
        "execution_authority": False,
        "physical_authority": False,
        "physical_closure": False,
    }
    composition_payload = {
        "schema": "rootscope.omega-v3-delta-composition.v1",
        "candidate_id": CANDIDATE_ID,
        "immutable_base_v2": base_reference,
        "receipt_bindings": receipt_bindings,
        "files": records,
        "qualification": qualification,
        "authority": authority,
    }
    composition_root = hashlib.sha256(
        canonical_compact_json(composition_payload)
    ).hexdigest()
    category_counts: dict[str, int] = {}
    for record in records:
        category = str(record["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
    manifest = {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "build_date": BUILD_DATE,
        "status": status,
        "packaging": {
            "delta_only": True,
            "compression": "none",
            "tar_format": "USTAR",
            "deterministic_metadata": True,
            "no_overwrite": True,
            "sha256sums_exact_coverage": True,
        },
        "immutable_base_v2": base_reference,
        "receipt_bindings": receipt_bindings,
        "contents": {
            "file_count": len(records),
            "payload_bytes": sum(int(record["bytes"]) for record in records),
            "category_counts": category_counts,
            "xrd_runtime_included": False,
            "training_artifacts_included": False,
            "secret_or_key_material_included": False,
            "absolute_temporary_paths_included": False,
            "v2_archive_duplicated": False,
            "foreground_loopback_llm_role_helper_included": True,
            "bpu_aux_input_images_included": False,
            "portable_tests_only": True,
            "local_only_tests_excluded": [
                {"path": f"rootscope/tests/{name}", "reason": reason}
                for name, reason in sorted(LOCAL_ONLY_TEST_EXCLUSIONS.items())
            ],
        },
        "qualification": qualification,
        "authority": authority,
        "composition_root_sha256": composition_root,
        "files": records,
    }
    manifest_bytes = canonical_json(manifest)
    sums_lines = [f"{record['sha256']}  {record['path']}" for record in records]
    sums_lines.append(
        f"{hashlib.sha256(manifest_bytes).hexdigest()}  candidate_manifest.json"
    )
    sums_bytes = ("\n".join(sorted(sums_lines)) + "\n").encode("ascii")

    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / CANDIDATE_ARCHIVE
    sidecar_path = output / f"{CANDIDATE_ARCHIVE}.sha256"
    receipt_path = output / "release_build_receipt.json"
    for path in (archive_path, sidecar_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing candidate output: {path}")
    archive = _build_tar_exclusive(
        archive_path,
        file_entries=[
            (f"{CANDIDATE_ID}/{entry.package_path}", entry.source, entry.mode)
            for entry in sources
        ],
        byte_entries=[
            (f"{CANDIDATE_ID}/candidate_manifest.json", manifest_bytes, 0o644),
            (f"{CANDIDATE_ID}/SHA256SUMS", sums_bytes, 0o644),
        ],
    )
    try:
        if (
            base_path.stat().st_size != BASE_BYTES
            or sha256_file(base_path) != BASE_SHA256
        ):
            raise RuntimeError("immutable v2 base changed during delta build")
        for role, record in (("x5", x5), ("vision", vision)):
            source = adventurex / str(record["source_path_relative_to_adventurex"])
            if (
                source.stat().st_size != record["bytes"]
                or sha256_file(source) != record["sha256"]
            ):
                raise RuntimeError(f"{role} receipt changed during delta build")
        vision_source_receipt = adventurex / str(
            vision["source_receipt_path_relative_to_adventurex"]
        )
        if (
            sha256_file(vision_source_receipt)
            != vision["source_receipt_sha256"]
        ):
            raise RuntimeError(
                "vision addendum source receipt changed during delta build"
            )
        for entry, record in zip(
            sources,
            sorted(records, key=lambda item: item["path"]),
        ):
            if entry.package_path != record["path"]:
                raise RuntimeError("internal source/manifest order changed")
            if (
                entry.source.stat().st_size != record["bytes"]
                or sha256_file(entry.source) != record["sha256"]
            ):
                raise RuntimeError(
                    f"allowlisted source changed during delta build: {entry.package_path}"
                )
        _write_exclusive(
            sidecar_path,
            f"{archive['sha256']}  {CANDIDATE_ARCHIVE}\n".encode("ascii"),
        )
        build_receipt = {
            "schema": "rootscope.omega-v3-delta-build-receipt.v1",
            "build_date": BUILD_DATE,
            "status": status,
            "output_relative_to_adventurex": output.relative_to(adventurex).as_posix(),
            "archive": archive,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "composition_root_sha256": composition_root,
            "immutable_base_v2": base_reference,
            "receipt_bindings": receipt_bindings,
            "qualification": qualification,
            "authority": authority,
        }
        _write_exclusive(receipt_path, canonical_json(build_receipt))
    except Exception:
        for path in (receipt_path, sidecar_path, archive_path):
            if path.exists():
                path.unlink()
        raise
    return build_receipt


def _parser() -> argparse.ArgumentParser:
    adventurex = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adventurex-root", type=Path, default=adventurex)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=adventurex / "output/releases" / CANDIDATE_ID,
    )
    parser.add_argument("--x5-receipt", type=Path, required=True)
    parser.add_argument("--vision-receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_delta_candidate(
        args.adventurex_root,
        args.output_dir,
        args.x5_receipt,
        args.vision_receipt,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
