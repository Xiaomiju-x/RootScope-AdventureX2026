#!/usr/bin/env python3
"""Build one deterministic, immutable, zero-authority event-vision overlay."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from typing import Any, Mapping, Sequence


BUILD_DATE = "2026-07-23"
OVERLAY_ID = "rootscope_event_vision_overlay_v1"
ARCHIVE_NAME = f"{OVERLAY_ID}.tar"
SCHEMA = "rootscope.event-vision-overlay.v1"
OUTPUT_RELATIVE = f"output/releases/{OVERLAY_ID}"
V2_REFERENCE = {
    "release_id": "rootscope_x5_field_bundle_v2",
    "release_dir_relative_to_adventurex": (
        "output/releases/rootscope_x5_field_bundle_v2"
    ),
    "archive_path_relative_to_adventurex": (
        "output/releases/rootscope_x5_field_bundle_v2/"
        "rootscope_x5_field_bundle_v2.tar"
    ),
    "sha256": "e6627685170252004d118bf77a690a9f89ad3afa274910697554d0f5cc8c3ebb",
    "bytes": 696_832_000,
    "bundled_in_overlay": False,
    "immutable_reference_only": True,
}
OMEGA_REFERENCE = {
    "release_id": "rootscope_omega_v3_delta_candidate_v1",
    "release_dir_relative_to_adventurex": (
        "output/releases/rootscope_omega_v3_delta_candidate_v1"
    ),
    "archive_path_relative_to_adventurex": (
        "output/releases/rootscope_omega_v3_delta_candidate_v1/"
        "rootscope_omega_v3_delta_candidate_v1.tar"
    ),
    "sha256": "c910f4d2e002ccdbd5643fa47f300ade8e56af8ad1c1a2a04fa4e4a0a0fab881",
    "bytes": 665_600,
    "bundled_in_overlay": False,
    "immutable_reference_only": True,
}
PRINT_MANIFEST_SOURCE = (
    "output/pdf/RootScope_A4_four_up_field_cards_20260723_manifest.json"
)
PRINT_MANIFEST_SHA256 = (
    "5e23e6133e9a59d8327bd751c4d6e5434d0c5a86402fd920c9d99e547613d827"
)
PRINT_PDF_REFERENCE = {
    "path_relative_to_adventurex": (
        "output/pdf/RootScope_A4_four_up_field_cards_20260723.pdf"
    ),
    "sha256": "113d2b2171e55f42df36d16b8772e02624d52d68a230cda541da66e56c19874e",
    "bundled_in_overlay": False,
}
CAMERA_CONTRACT_SOURCE = (
    "rootscope/configs/event_vision/camera_identity_x5_20260723.json"
)
CAMERA_CONTRACT_SHA256 = (
    "3fd12d7aa87936cc35c88ffdfab78829beef6fbc03a935451f227e4ee150cab4"
)
CAPSULE_TEMPLATE_PATH = (
    "rootscope/deploy/x5/capsule_config.seed17_cpu_experimental.json"
)
CAPSULE_TEMPLATE_SHA256 = (
    "1b7e9b96ccd4ec4e5ab534e1f305224c3c6330a3ab2efb8ca2e5d0fc52fcfcbb"
)
CAPSULE_TEMPLATE_BYTES = 2_330
RUNTIME_CAPSULE_PATH = (
    "/opt/rootscope/.local/share/rootscope-field-v2/core_v1/config/"
    "rootscope_x5_offline_core_v1.capsule.json"
)
RUNTIME_CAPSULE_SHA256 = (
    "1e839948e466895e0b416aababb39402380463f9ce125c8aeb59f4087191cb97"
)
RUNTIME_CAPSULE_BYTES = 2_765
RUNTIME_PROJECT_ROOT = (
    "/opt/rootscope/.local/share/rootscope-field-v2/core_v1/releases/"
    "rootscope_x5_offline_core_v1/rootscope"
)
RUNTIME_PYTHON = (
    "/opt/rootscope/.local/share/rootscope-field-v2/core_v1/venvs/"
    "rootscope_x5_offline_core_v1/bin/python3"
)
RUNTIME_MODEL_PATH = (
    f"{RUNTIME_PROJECT_ROOT}/deploy/x5/models/"
    "rootscope_seed17_cpu_experimental_opset11.onnx"
)
MODEL_SHA256 = "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad"
MODEL_BYTES = 44_704_833
CALIBRATION_SHA256 = (
    "e82b196ab627a935fd571bba2b37aa637f040af5ba81fbab5183c1f15aa2e564"
)
REGISTRY_SHA256 = (
    "f5328b66c6b385dd081c1a27577d92b5c63fc18dc00d815f7d3d2561bb68e29f"
)
THRESHOLDS_SHA256 = (
    "877205689ad903207e0bcb5ffabdcbc5f1472c00b8f82e72faeb7cdd7d140fcd"
)
MATCHER_SHA256 = (
    "9952864e50371675e7ea181cc57f2edd9eadd9189bfa0e9eda1e5cdd8f8ca61a"
)
HELPER_SOURCE = "tools/release/verify_rootscope_event_vision_overlay_v1.py"
HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1.py"
SHIM_ROOT = "tools/release/event_vision_overlay_shims"

FORBIDDEN_IMPORT_ROOTS = {
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
    "serial",
    "gpiozero",
    "RPi",
    "requests",
    "socket",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(
        r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|private[_-]?key)"
        r"\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']"
    ),
)
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:^|[^A-Za-z0-9_])[A-Z]:[\\/]")


@dataclass(frozen=True)
class SourceSpec:
    source_relative: str
    package_relative: str
    category: str
    mode: int = 0o644


@dataclass(frozen=True)
class PackageEntry:
    path: str
    data: bytes
    mode: int
    category: str
    source_relative: str | None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


SOURCE_SPECS = (
    SourceSpec(f"{SHIM_ROOT}/app/__init__.py", "rootscope/app/__init__.py", "shim"),
    SourceSpec(
        f"{SHIM_ROOT}/app/edge/__init__.py",
        "rootscope/app/edge/__init__.py",
        "shim",
    ),
    SourceSpec(
        "rootscope/app/edge/capsule.py",
        "rootscope/app/edge/capsule.py",
        "dependency_source",
    ),
    SourceSpec(
        "rootscope/app/edge/onnx_cpu.py",
        "rootscope/app/edge/onnx_cpu.py",
        "dependency_source",
    ),
    SourceSpec(
        f"{SHIM_ROOT}/app/vision/__init__.py",
        "rootscope/app/vision/__init__.py",
        "shim",
    ),
    SourceSpec(
        "rootscope/app/vision/quality_gate.py",
        "rootscope/app/vision/quality_gate.py",
        "dependency_source",
    ),
    SourceSpec(
        "rootscope/app/vision/uvc_card_capture.py",
        "rootscope/app/vision/uvc_card_capture.py",
        "event_capture_source",
    ),
    SourceSpec(
        "rootscope/app/vision/card_geometric_matcher.py",
        "rootscope/app/vision/card_geometric_matcher.py",
        "dependency_source",
    ),
    SourceSpec(
        "rootscope/app/vision/dual_path_demo.py",
        "rootscope/app/vision/dual_path_demo.py",
        "dependency_source",
    ),
    SourceSpec(
        f"{SHIM_ROOT}/app/omega_vision/__init__.py",
        "rootscope/app/omega_vision/__init__.py",
        "shim",
    ),
    SourceSpec(
        "rootscope/app/omega_vision/ood.py",
        "rootscope/app/omega_vision/ood.py",
        "dependency_source",
    ),
    SourceSpec(
        "rootscope/app/omega_vision/uvc_card_frontend.py",
        "rootscope/app/omega_vision/uvc_card_frontend.py",
        "event_frontend_source",
    ),
    SourceSpec(
        "rootscope/tests/__init__.py",
        "rootscope/tests/__init__.py",
        "test",
    ),
    SourceSpec(
        "rootscope/tests/test_uvc_card_capture.py",
        "rootscope/tests/test_uvc_card_capture.py",
        "test",
    ),
    SourceSpec(
        "rootscope/tests/test_omega_uvc_card_frontend.py",
        "rootscope/tests/test_omega_uvc_card_frontend.py",
        "test",
    ),
    SourceSpec(
        CAPSULE_TEMPLATE_PATH,
        CAPSULE_TEMPLATE_PATH,
        "runtime_contract",
    ),
    SourceSpec(
        "rootscope/configs/omega/vision_board_replay_new_x5_20260723.json",
        "rootscope/configs/omega/vision_board_replay_new_x5_20260723.json",
        "runtime_contract",
    ),
    SourceSpec(
        CAMERA_CONTRACT_SOURCE,
        CAMERA_CONTRACT_SOURCE,
        "camera_identity_contract",
    ),
    SourceSpec(
        "rootscope/app/vision/known_card_template_registry.frozen.experimental.json",
        "rootscope/app/vision/known_card_template_registry.frozen.experimental.json",
        "runtime_contract",
    ),
    SourceSpec(
        "rootscope/app/vision/known_card_templates/grass_clump_163498042.jpg",
        "rootscope/app/vision/known_card_templates/grass_clump_163498042.jpg",
        "registered_demo_asset",
    ),
    SourceSpec(
        "rootscope/app/vision/known_card_templates/low_shrub_68787114.jpg",
        "rootscope/app/vision/known_card_templates/low_shrub_68787114.jpg",
        "registered_demo_asset",
    ),
    SourceSpec(
        "rootscope/app/vision/known_card_templates/young_tree_92774234.jpg",
        "rootscope/app/vision/known_card_templates/young_tree_92774234.jpg",
        "registered_demo_asset",
    ),
    SourceSpec(
        "rootscope/app/vision/dual_path_demo.thresholds.example.json",
        "rootscope/app/vision/dual_path_demo.thresholds.example.json",
        "runtime_contract",
    ),
    SourceSpec(
        "rootscope/app/vision/card_geometric_matcher.config.example.json",
        "rootscope/app/vision/card_geometric_matcher.config.example.json",
        "runtime_contract",
    ),
    SourceSpec(
        PRINT_MANIFEST_SOURCE,
        PRINT_MANIFEST_SOURCE,
        "print_manifest",
    ),
    SourceSpec(
        "rootscope/app/vision/UVC_CARD_CAPTURE_RUNBOOK_ZH.md",
        "docs/UVC_CARD_CAPTURE_RUNBOOK_ZH.md",
        "runbook",
    ),
    SourceSpec(
        "rootscope/deploy/x5/ROOTSCOPE_UVC_CARD_FRONTEND_RUNBOOK_ZH.md",
        "docs/ROOTSCOPE_UVC_CARD_FRONTEND_RUNBOOK_ZH.md",
        "runbook",
    ),
    SourceSpec(
        HELPER_SOURCE,
        HELPER_PACKAGE,
        "zero_authority_preflight",
        mode=0o755,
    ),
)


def adventurex_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def compact_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_package_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe package path: {value!r}")


def _module_path(import_name: str) -> str:
    return f"rootscope/{import_name.replace('.', '/')}.py"


def _entry_module(entry_path: str) -> tuple[list[str], bool]:
    path = PurePosixPath(entry_path)
    if not entry_path.startswith("rootscope/") or path.suffix != ".py":
        return [], False
    parts = list(path.with_suffix("").parts[1:])
    is_package = bool(parts and parts[-1] == "__init__")
    if is_package:
        parts.pop()
    return parts, is_package


def audit_python_import_closure(entries: Sequence[PackageEntry]) -> None:
    available = {entry.path for entry in entries}
    for entry in entries:
        if not entry.path.endswith(".py"):
            continue
        tree = ast.parse(entry.data.decode("utf-8"), filename=entry.path)
        for node in ast.walk(tree):
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    module_parts, is_package = _entry_module(entry.path)
                    package_parts = module_parts if is_package else module_parts[:-1]
                    ascend = node.level - 1
                    if ascend > len(package_parts):
                        raise ValueError(f"relative import escapes package in {entry.path}")
                    base = package_parts[: len(package_parts) - ascend]
                    if node.module:
                        base.extend(node.module.split("."))
                        roots.append(".".join(base))
                    else:
                        roots.extend(".".join((*base, alias.name)) for alias in node.names)
                elif node.module:
                    roots.append(node.module)
            for imported in roots:
                root = imported.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    raise ValueError(f"forbidden import {imported!r} in {entry.path}")
                if root != "app":
                    continue
                parts = imported.split(".")
                candidates = {
                    _module_path(imported),
                    f"rootscope/{'/'.join(parts)}/__init__.py",
                }
                if not candidates & available:
                    raise ValueError(
                        f"local import closure missing for {imported!r} in {entry.path}"
                    )


def audit_text(entries: Sequence[PackageEntry]) -> None:
    for entry in entries:
        if Path(entry.path).suffix.lower() not in {".py", ".json", ".md", ".txt"}:
            continue
        text = entry.data.decode("utf-8")
        if WINDOWS_ABSOLUTE.search(text):
            raise ValueError(f"absolute Windows path found in {entry.path}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"secret-like text found in {entry.path}")


def load_source_entries(root: Path) -> list[PackageEntry]:
    entries: list[PackageEntry] = []
    seen: set[str] = set()
    for spec in SOURCE_SPECS:
        validate_package_path(spec.source_relative)
        validate_package_path(spec.package_relative)
        if spec.package_relative in seen:
            raise ValueError(f"duplicate package path: {spec.package_relative}")
        seen.add(spec.package_relative)
        source = root / Path(*PurePosixPath(spec.source_relative).parts)
        resolved = source.resolve(strict=True)
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError(f"source must be a regular non-symlink file: {source}")
        entries.append(
            PackageEntry(
                path=spec.package_relative,
                data=resolved.read_bytes(),
                mode=spec.mode,
                category=spec.category,
                source_relative=spec.source_relative,
            )
        )
    audit_python_import_closure(entries)
    audit_text(entries)
    return entries


def verify_reference(root: Path, record: Mapping[str, Any]) -> None:
    path = root / Path(
        *PurePosixPath(str(record["archive_path_relative_to_adventurex"])).parts
    )
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"immutable reference is not a regular file: {path}")
    if resolved.stat().st_size != record["bytes"]:
        raise ValueError(f"immutable reference byte mismatch: {path}")
    if sha256_file(resolved) != record["sha256"]:
        raise ValueError(f"immutable reference SHA-256 mismatch: {path}")


def content_root(entries: Sequence[PackageEntry]) -> str:
    records = [
        {
            "bytes": len(entry.data),
            "category": entry.category,
            "mode": entry.mode,
            "path": entry.path,
            "sha256": entry.sha256,
        }
        for entry in sorted(entries, key=lambda item: item.path)
    ]
    return hashlib.sha256(compact_json(records)).hexdigest()


def build_manifest(entries: Sequence[PackageEntry]) -> dict[str, Any]:
    source_records = [
        {
            "bytes": len(entry.data),
            "category": entry.category,
            "mode": f"{entry.mode:04o}",
            "package_path": entry.path,
            "sha256": entry.sha256,
            "source_path_relative_to_adventurex": entry.source_relative,
        }
        for entry in sorted(entries, key=lambda item: item.path)
    ]
    return {
        "schema": SCHEMA,
        "overlay_id": OVERLAY_ID,
        "status": "IMMUTABLE_EVENT_VISION_ZERO_AUTHORITY_OVERLAY",
        "build_date": BUILD_DATE,
        "purpose": (
            "Bounded UVC card capture plus CPU/OOD/registered-template display "
            "frontend; no physical execution authority"
        ),
        "immutable_references": {
            "x5_field_bundle_v2": dict(V2_REFERENCE),
            "omega_v3_delta_candidate": dict(OMEGA_REFERENCE),
        },
        "entrypoints": {
            "capture_cli": "app.vision.uvc_card_capture",
            "display_frontend_cli": "app.omega_vision.uvc_card_frontend",
            "read_only_preflight": HELPER_PACKAGE,
        },
        "camera_identity_contract": {
            "package_path": CAMERA_CONTRACT_SOURCE,
            "sha256": next(
                entry.sha256 for entry in entries if entry.path == CAMERA_CONTRACT_SOURCE
            ),
        },
        "frozen_runtime_asset_contracts": {
            "capsule_template": {
                "package_path": CAPSULE_TEMPLATE_PATH,
                "bytes": CAPSULE_TEMPLATE_BYTES,
                "sha256": CAPSULE_TEMPLATE_SHA256,
            },
            "runtime_capsule": {
                "bundled_in_overlay": False,
                "path_on_x5": RUNTIME_CAPSULE_PATH,
                "bytes": RUNTIME_CAPSULE_BYTES,
                "sha256": RUNTIME_CAPSULE_SHA256,
                "reconstruction": {
                    "template_sha256": CAPSULE_TEMPLATE_SHA256,
                    "project_root": RUNTIME_PROJECT_ROOT,
                    "python_executable": RUNTIME_PYTHON,
                    "model_path": RUNTIME_MODEL_PATH,
                    "encoding": (
                        "json.dumps(ensure_ascii=False,indent=2,sort_keys=True)"
                        "+newline_utf8"
                    ),
                    "runtime_value_must_not_be_self_promoted_to_expected": True,
                },
            },
            "seed17_cpu_onnx": {
                "bundled_in_overlay": False,
                "path_on_x5": RUNTIME_MODEL_PATH,
                "bytes": MODEL_BYTES,
                "sha256": MODEL_SHA256,
                "provider": "CPUExecutionProvider",
            },
            "omega_calibration_sha256": CALIBRATION_SHA256,
            "registered_template_registry_sha256": REGISTRY_SHA256,
            "dual_path_thresholds_sha256": THRESHOLDS_SHA256,
            "geometric_matcher_config_sha256": MATCHER_SHA256,
        },
        "print_assets": {
            "manifest_package_path": PRINT_MANIFEST_SOURCE,
            "manifest_sha256": PRINT_MANIFEST_SHA256,
            "large_pdf_reference_only": dict(PRINT_PDF_REFERENCE),
        },
        "content_composition_root_sha256": content_root(entries),
        "source_allowlist": source_records,
        "dependency_boundary": {
            "python_import_closure_exact": True,
            "external_runtime_packages": [
                "numpy",
                "Pillow",
                "opencv-python-headless",
                "onnxruntime",
            ],
            "model_binary_bundled": False,
            "plant_bpu_binary_bundled": False,
        },
        "qualification": {
            "camera_capture_observed_in_this_build": False,
            "generalization_claimed": False,
            "model_qualified": False,
            "plant_bpu_qualified": False,
            "production_integration_allowed": False,
            "selected_bin": None,
        },
        "authority": {
            "camera_opened_during_build": False,
            "execution_authority": False,
            "gpio_access": False,
            "network_configuration_write": False,
            "physical_authority": False,
            "pump_command": False,
            "serial_open": False,
            "service_or_gate_created": False,
            "state_machine_write": False,
            "systemd_write": False,
        },
    }


def add_generated_entries(source_entries: Sequence[PackageEntry]) -> list[PackageEntry]:
    entries = list(source_entries)
    manifest = build_manifest(source_entries)
    entries.append(
        PackageEntry(
            path="release_manifest.json",
            data=canonical_json(manifest),
            mode=0o644,
            category="release_manifest",
            source_relative=None,
        )
    )
    lines = [
        f"{entry.sha256}  {entry.path}\n"
        for entry in sorted(entries, key=lambda item: item.path)
    ]
    entries.append(
        PackageEntry(
            path="SHA256SUMS",
            data="".join(lines).encode("utf-8"),
            mode=0o644,
            category="exact_checksum_coverage_excluding_self",
            source_relative=None,
        )
    )
    return entries


def _tar_info(name: str, *, mode: int, size: int, directory: bool) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = mode
    info.size = 0 if directory else size
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    return info


def write_deterministic_ustar(path: Path, entries: Sequence[PackageEntry]) -> None:
    from io import BytesIO

    if path.exists():
        raise FileExistsError(f"archive overwrite refused: {path}")
    directories = {OVERLAY_ID}
    for entry in entries:
        validate_package_path(entry.path)
        current = PurePosixPath(OVERLAY_ID) / PurePosixPath(entry.path)
        for parent in current.parents:
            if parent.as_posix() not in {".", ""}:
                directories.add(parent.as_posix())
    with tarfile.open(path, mode="x", format=tarfile.USTAR_FORMAT) as archive:
        for directory in sorted(directories):
            archive.addfile(
                _tar_info(
                    f"{directory.rstrip('/')}/",
                    mode=0o755,
                    size=0,
                    directory=True,
                )
            )
        for entry in sorted(entries, key=lambda item: item.path):
            archive.addfile(
                _tar_info(
                    f"{OVERLAY_ID}/{entry.path}",
                    mode=entry.mode,
                    size=len(entry.data),
                    directory=False,
                ),
                BytesIO(entry.data),
            )


def _atomic_write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def build_release(root: Path) -> dict[str, Any]:
    output_dir = root / Path(*PurePosixPath(OUTPUT_RELATIVE).parts)
    if output_dir.exists():
        raise FileExistsError(f"immutable output directory already exists: {output_dir}")
    verify_reference(root, V2_REFERENCE)
    verify_reference(root, OMEGA_REFERENCE)
    source_entries = load_source_entries(root)
    print_entry = next(
        entry for entry in source_entries if entry.path == PRINT_MANIFEST_SOURCE
    )
    if print_entry.sha256 != PRINT_MANIFEST_SHA256:
        raise ValueError("frozen four-up print manifest SHA-256 mismatch")
    camera_entry = next(
        entry for entry in source_entries if entry.path == CAMERA_CONTRACT_SOURCE
    )
    frozen_source_hashes = {
        CAMERA_CONTRACT_SOURCE: CAMERA_CONTRACT_SHA256,
        CAPSULE_TEMPLATE_PATH: CAPSULE_TEMPLATE_SHA256,
        "rootscope/configs/omega/vision_board_replay_new_x5_20260723.json": (
            CALIBRATION_SHA256
        ),
        "rootscope/app/vision/known_card_template_registry.frozen.experimental.json": (
            REGISTRY_SHA256
        ),
        "rootscope/app/vision/dual_path_demo.thresholds.example.json": (
            THRESHOLDS_SHA256
        ),
        "rootscope/app/vision/card_geometric_matcher.config.example.json": (
            MATCHER_SHA256
        ),
    }
    source_by_path = {entry.path: entry for entry in source_entries}
    for package_path, expected_sha in frozen_source_hashes.items():
        if source_by_path[package_path].sha256 != expected_sha:
            raise ValueError(f"frozen runtime source SHA-256 mismatch: {package_path}")
    capsule_template = source_by_path[CAPSULE_TEMPLATE_PATH]
    if len(capsule_template.data) != CAPSULE_TEMPLATE_BYTES:
        raise ValueError("frozen capsule template byte count mismatch")
    camera_payload = json.loads(camera_entry.data.decode("utf-8"))
    if (
        camera_payload.get("camera", {}).get("usb_vid_pid") != "32e6:9228"
        or camera_payload.get("camera", {}).get("usb_serial") != "202604081837"
    ):
        raise ValueError("camera identity contract mismatch")

    entries = add_generated_entries(source_entries)
    release_parent = output_dir.parent
    if not release_parent.is_dir() or release_parent.is_symlink():
        raise ValueError("release parent must be an existing non-symlink directory")
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{OVERLAY_ID}.staging-", dir=release_parent)
    )
    published = False
    try:
        temp_archive = staging_dir / ARCHIVE_NAME
        write_deterministic_ustar(temp_archive, entries)
        archive_sha = sha256_file(temp_archive)
        archive_bytes = temp_archive.stat().st_size
        manifest_entry = next(
            entry for entry in entries if entry.path == "release_manifest.json"
        )
        sums_entry = next(entry for entry in entries if entry.path == "SHA256SUMS")
        receipt = {
            "schema": "rootscope.event-vision-overlay-build-receipt.v1",
            "status": "PASS_IMMUTABLE_ZERO_AUTHORITY_OVERLAY_BUILT_ONCE",
            "build_date": BUILD_DATE,
            "overlay_id": OVERLAY_ID,
            "output_relative_to_adventurex": OUTPUT_RELATIVE,
            "archive": {
                "filename": ARCHIVE_NAME,
                "bytes": archive_bytes,
                "sha256": archive_sha,
                "compression": "none",
                "tar_format": "USTAR",
            },
            "manifest_sha256": manifest_entry.sha256,
            "content_composition_root_sha256": json.loads(
                manifest_entry.data.decode("utf-8")
            )["content_composition_root_sha256"],
            "final_composition_root_sha256": sums_entry.sha256,
            "sha256sums_covered_file_count": len(entries) - 1,
            "immutable_references": {
                "x5_field_bundle_v2": dict(V2_REFERENCE),
                "omega_v3_delta_candidate": dict(OMEGA_REFERENCE),
            },
            "authority": {
                "camera_opened": False,
                "execution_authority": False,
                "gpio_access": False,
                "network_configuration_write": False,
                "physical_authority": False,
                "pump_command": False,
                "serial_open": False,
                "service_or_gate_created": False,
                "systemd_write": False,
            },
        }
        _atomic_write_new(
            staging_dir / "release_build_receipt.json",
            canonical_json(receipt),
        )
        _atomic_write_new(
            staging_dir / f"{ARCHIVE_NAME}.sha256",
            f"{archive_sha}  {ARCHIVE_NAME}\n".encode("ascii"),
        )
        if output_dir.exists():
            raise FileExistsError(
                f"immutable output directory appeared during build: {output_dir}"
            )
        os.rename(staging_dir, output_dir)
        published = True
    finally:
        if not published and staging_dir.exists():
            shutil.rmtree(staging_dir)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    try:
        receipt = build_release(adventurex_root())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"FAIL_CLOSED: {type(exc).__name__}: {exc}\n")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
