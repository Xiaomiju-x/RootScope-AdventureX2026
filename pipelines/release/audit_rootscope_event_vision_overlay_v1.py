#!/usr/bin/env python3
"""Independent audit for the immutable RootScope event-vision overlay v1."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
from typing import Any, Mapping, Sequence


OVERLAY_ID = "rootscope_event_vision_overlay_v1"
ARCHIVE_NAME = f"{OVERLAY_ID}.tar"
SCHEMA = "rootscope.event-vision-overlay.v1"
OUTPUT_RELATIVE = f"output/releases/{OVERLAY_ID}"
V2_PATH = (
    "output/releases/rootscope_x5_field_bundle_v2/"
    "rootscope_x5_field_bundle_v2.tar"
)
V2_DIR = "output/releases/rootscope_x5_field_bundle_v2"
V2_SHA = "e6627685170252004d118bf77a690a9f89ad3afa274910697554d0f5cc8c3ebb"
V2_BYTES = 696_832_000
OMEGA_PATH = (
    "output/releases/rootscope_omega_v3_delta_candidate_v1/"
    "rootscope_omega_v3_delta_candidate_v1.tar"
)
OMEGA_DIR = "output/releases/rootscope_omega_v3_delta_candidate_v1"
OMEGA_SHA = "c910f4d2e002ccdbd5643fa47f300ade8e56af8ad1c1a2a04fa4e4a0a0fab881"
OMEGA_BYTES = 665_600
PRINT_PATH = "output/pdf/RootScope_A4_four_up_field_cards_20260723_manifest.json"
PRINT_SHA = "5e23e6133e9a59d8327bd751c4d6e5434d0c5a86402fd920c9d99e547613d827"
CAMERA_PATH = "rootscope/configs/event_vision/camera_identity_x5_20260723.json"
HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1.py"
CAPSULE_TEMPLATE_SHA = (
    "1b7e9b96ccd4ec4e5ab534e1f305224c3c6330a3ab2efb8ca2e5d0fc52fcfcbb"
)
RUNTIME_CAPSULE_SHA = (
    "1e839948e466895e0b416aababb39402380463f9ce125c8aeb59f4087191cb97"
)
RUNTIME_CAPSULE_BYTES = 2_765
RUNTIME_CAPSULE_PATH = (
    "/opt/rootscope/.local/share/rootscope-field-v2/core_v1/config/"
    "rootscope_x5_offline_core_v1.capsule.json"
)
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

EXPECTED_SOURCE_MAP = {
    "rootscope/app/__init__.py": (
        "tools/release/event_vision_overlay_shims/app/__init__.py"
    ),
    "rootscope/app/edge/__init__.py": (
        "tools/release/event_vision_overlay_shims/app/edge/__init__.py"
    ),
    "rootscope/app/edge/capsule.py": "rootscope/app/edge/capsule.py",
    "rootscope/app/edge/onnx_cpu.py": "rootscope/app/edge/onnx_cpu.py",
    "rootscope/app/vision/__init__.py": (
        "tools/release/event_vision_overlay_shims/app/vision/__init__.py"
    ),
    "rootscope/app/vision/quality_gate.py": "rootscope/app/vision/quality_gate.py",
    "rootscope/app/vision/uvc_card_capture.py": (
        "rootscope/app/vision/uvc_card_capture.py"
    ),
    "rootscope/app/vision/card_geometric_matcher.py": (
        "rootscope/app/vision/card_geometric_matcher.py"
    ),
    "rootscope/app/vision/dual_path_demo.py": (
        "rootscope/app/vision/dual_path_demo.py"
    ),
    "rootscope/app/omega_vision/__init__.py": (
        "tools/release/event_vision_overlay_shims/app/omega_vision/__init__.py"
    ),
    "rootscope/app/omega_vision/ood.py": "rootscope/app/omega_vision/ood.py",
    "rootscope/app/omega_vision/uvc_card_frontend.py": (
        "rootscope/app/omega_vision/uvc_card_frontend.py"
    ),
    "rootscope/tests/__init__.py": "rootscope/tests/__init__.py",
    "rootscope/tests/test_uvc_card_capture.py": (
        "rootscope/tests/test_uvc_card_capture.py"
    ),
    "rootscope/tests/test_omega_uvc_card_frontend.py": (
        "rootscope/tests/test_omega_uvc_card_frontend.py"
    ),
    "rootscope/deploy/x5/capsule_config.seed17_cpu_experimental.json": (
        "rootscope/deploy/x5/capsule_config.seed17_cpu_experimental.json"
    ),
    "rootscope/configs/omega/vision_board_replay_new_x5_20260723.json": (
        "rootscope/configs/omega/vision_board_replay_new_x5_20260723.json"
    ),
    CAMERA_PATH: CAMERA_PATH,
    "rootscope/app/vision/known_card_template_registry.frozen.experimental.json": (
        "rootscope/app/vision/known_card_template_registry.frozen.experimental.json"
    ),
    "rootscope/app/vision/known_card_templates/grass_clump_163498042.jpg": (
        "rootscope/app/vision/known_card_templates/grass_clump_163498042.jpg"
    ),
    "rootscope/app/vision/known_card_templates/low_shrub_68787114.jpg": (
        "rootscope/app/vision/known_card_templates/low_shrub_68787114.jpg"
    ),
    "rootscope/app/vision/known_card_templates/young_tree_92774234.jpg": (
        "rootscope/app/vision/known_card_templates/young_tree_92774234.jpg"
    ),
    "rootscope/app/vision/dual_path_demo.thresholds.example.json": (
        "rootscope/app/vision/dual_path_demo.thresholds.example.json"
    ),
    "rootscope/app/vision/card_geometric_matcher.config.example.json": (
        "rootscope/app/vision/card_geometric_matcher.config.example.json"
    ),
    PRINT_PATH: PRINT_PATH,
    "docs/UVC_CARD_CAPTURE_RUNBOOK_ZH.md": (
        "rootscope/app/vision/UVC_CARD_CAPTURE_RUNBOOK_ZH.md"
    ),
    "docs/ROOTSCOPE_UVC_CARD_FRONTEND_RUNBOOK_ZH.md": (
        "rootscope/deploy/x5/ROOTSCOPE_UVC_CARD_FRONTEND_RUNBOOK_ZH.md"
    ),
    HELPER_PACKAGE: "tools/release/verify_rootscope_event_vision_overlay_v1.py",
}
EXPECTED_FILES = set(EXPECTED_SOURCE_MAP) | {"release_manifest.json", "SHA256SUMS"}
FORBIDDEN_IMPORTS = {
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
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:^|[^A-Za-z0-9_])[A-Z]:[\\/]")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
)


class AuditError(RuntimeError):
    """A release invariant failed."""


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_object_bytes(data: bytes, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON {label}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AuditError(f"JSON root must be an object: {label}")
    return payload


def safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or "\\" in name
        or path.as_posix() != name.rstrip("/")
        or any(part in {"", ".", ".."} for part in path.parts)
        or not path.parts
        or path.parts[0] != OVERLAY_ID
    ):
        raise AuditError(f"unsafe tar member: {name!r}")
    relative = PurePosixPath(*path.parts[1:]).as_posix()
    if not relative or relative == ".":
        return ""
    return relative.rstrip("/")


def parse_sums(data: bytes) -> dict[str, str]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise AuditError("SHA256SUMS is not UTF-8") from exc
    result: dict[str, str] = {}
    for index, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if match is None:
            raise AuditError(f"malformed SHA256SUMS line {index}")
        digest, relative = match.groups()
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or "\\" in relative
            or path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise AuditError(f"unsafe SHA256SUMS path: {relative}")
        if relative in result:
            raise AuditError(f"duplicate SHA256SUMS path: {relative}")
        result[relative] = digest
    return result


def read_archive(
    archive_path: Path,
) -> tuple[dict[str, bytes], list[str], set[str]]:
    files: dict[str, bytes] = {}
    order: list[str] = []
    directories: set[str] = set()
    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive:
            relative = safe_member_name(member.name)
            if member.name in seen:
                raise AuditError(f"duplicate tar member: {member.name}")
            seen.add(member.name)
            order.append(member.name)
            if member.uid != 0 or member.gid != 0 or member.mtime != 0:
                raise AuditError(f"non-deterministic tar metadata: {member.name}")
            if member.uname or member.gname:
                raise AuditError(f"tar owner names must be empty: {member.name}")
            if member.isdir():
                if member.mode != 0o755:
                    raise AuditError(f"invalid directory member: {member.name}")
                directories.add(member.name)
                continue
            if not member.isfile() or not relative:
                raise AuditError(f"non-regular tar member forbidden: {member.name}")
            expected_mode = 0o755 if relative == HELPER_PACKAGE else 0o644
            if member.mode != expected_mode:
                raise AuditError(f"unexpected file mode for {relative}")
            if member.size > 10_000_000:
                raise AuditError(f"unexpectedly large bundled file: {relative}")
            stream = archive.extractfile(member)
            if stream is None:
                raise AuditError(f"cannot read tar member: {relative}")
            files[relative] = stream.read()
    if set(files) != EXPECTED_FILES:
        raise AuditError(
            f"exact file allowlist mismatch: "
            f"missing={sorted(EXPECTED_FILES - set(files))} "
            f"extra={sorted(set(files) - EXPECTED_FILES)}"
        )
    expected_order = sorted(directories) + sorted(
        name for name in order if name not in directories
    )
    if order != expected_order:
        raise AuditError("tar members are not in deterministic directory/file order")
    return files, order, directories


def rebuild_archive(
    path: Path,
    files: Mapping[str, bytes],
    order: Sequence[str],
    directories: set[str],
) -> None:
    def info(name: str, directory: bool) -> tarfile.TarInfo:
        stored_name = f"{name.rstrip('/')}/" if directory else name
        result = tarfile.TarInfo(stored_name)
        result.mtime = 0
        result.uid = 0
        result.gid = 0
        result.uname = ""
        result.gname = ""
        result.mode = (
            0o755
            if directory or name.endswith(f"/{HELPER_PACKAGE}")
            else 0o644
        )
        result.size = 0 if directory else len(
            files[PurePosixPath(name).relative_to(OVERLAY_ID).as_posix()]
        )
        result.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
        return result

    with tarfile.open(path, mode="x", format=tarfile.USTAR_FORMAT) as archive:
        for name in order:
            if name in directories:
                archive.addfile(info(name, True))
            else:
                relative = PurePosixPath(name).relative_to(OVERLAY_ID).as_posix()
                archive.addfile(info(name, False), BytesIO(files[relative]))


def verify_external_reference(root: Path, relative: str, size: int, digest: str) -> None:
    path = root / Path(*PurePosixPath(relative).parts)
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise AuditError(f"immutable reference is not a regular file: {relative}")
    if resolved.stat().st_size != size or sha256_file(resolved) != digest:
        raise AuditError(f"immutable reference mismatch: {relative}")


def audit_release(root: Path, archive_path: Path) -> dict[str, Any]:
    checks: list[str] = []
    archive = archive_path.resolve(strict=True)
    release_dir = archive.parent
    receipt_path = release_dir / "release_build_receipt.json"
    sidecar_path = release_dir / f"{ARCHIVE_NAME}.sha256"
    receipt = load_object_bytes(receipt_path.read_bytes(), "release_build_receipt")
    archive_sha = sha256_file(archive)
    archive_bytes = archive.stat().st_size
    if (
        receipt.get("archive", {}).get("sha256") != archive_sha
        or receipt.get("archive", {}).get("bytes") != archive_bytes
        or receipt.get("archive", {}).get("tar_format") != "USTAR"
        or receipt.get("archive", {}).get("compression") != "none"
    ):
        raise AuditError("build receipt archive binding mismatch")
    if sidecar_path.read_text(encoding="ascii") != f"{archive_sha}  {ARCHIVE_NAME}\n":
        raise AuditError("archive sidecar mismatch")
    checks.append("ARCHIVE_RECEIPT_AND_SIDECAR")

    verify_external_reference(root, V2_PATH, V2_BYTES, V2_SHA)
    verify_external_reference(root, OMEGA_PATH, OMEGA_BYTES, OMEGA_SHA)
    checks.append("IMMUTABLE_REFERENCES_PRESENT_UNCHANGED")

    files, order, directories = read_archive(archive)
    checks.append("USTAR_SAFE_EXACT_ALLOWLIST")
    sums = parse_sums(files["SHA256SUMS"])
    if set(sums) != EXPECTED_FILES - {"SHA256SUMS"}:
        raise AuditError("SHA256SUMS exact coverage mismatch")
    for relative, expected in sums.items():
        if sha256_bytes(files[relative]) != expected:
            raise AuditError(f"SHA256SUMS digest mismatch: {relative}")
    checks.append("SHA256SUMS_EXACT_COVERAGE")

    manifest = load_object_bytes(files["release_manifest.json"], "release_manifest")
    if manifest.get("schema") != SCHEMA or manifest.get("overlay_id") != OVERLAY_ID:
        raise AuditError("release manifest identity mismatch")
    if manifest.get("status") != "IMMUTABLE_EVENT_VISION_ZERO_AUTHORITY_OVERLAY":
        raise AuditError("release manifest status mismatch")
    authority = manifest.get("authority")
    if not isinstance(authority, Mapping) or any(value is not False for value in authority.values()):
        raise AuditError("release authority must contain only false values")
    qualification = manifest.get("qualification")
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("selected_bin") is not None
        or qualification.get("production_integration_allowed") is not False
        or qualification.get("model_qualified") is not False
    ):
        raise AuditError("qualification boundary was upgraded")
    checks.append("ZERO_AUTHORITY_AND_NULL_SELECTED_BIN")

    assets = manifest.get("frozen_runtime_asset_contracts")
    if not isinstance(assets, Mapping):
        raise AuditError("frozen runtime asset contracts are missing")
    capsule_template = assets.get("capsule_template")
    runtime_capsule = assets.get("runtime_capsule")
    model = assets.get("seed17_cpu_onnx")
    if capsule_template != {
        "package_path": (
            "rootscope/deploy/x5/capsule_config.seed17_cpu_experimental.json"
        ),
        "bytes": 2_330,
        "sha256": CAPSULE_TEMPLATE_SHA,
    }:
        raise AuditError("capsule template prior mismatch")
    if (
        not isinstance(runtime_capsule, Mapping)
        or runtime_capsule.get("bundled_in_overlay") is not False
        or runtime_capsule.get("path_on_x5") != RUNTIME_CAPSULE_PATH
        or runtime_capsule.get("bytes") != RUNTIME_CAPSULE_BYTES
        or runtime_capsule.get("sha256") != RUNTIME_CAPSULE_SHA
        or runtime_capsule.get("reconstruction")
        != {
            "template_sha256": CAPSULE_TEMPLATE_SHA,
            "project_root": RUNTIME_PROJECT_ROOT,
            "python_executable": RUNTIME_PYTHON,
            "model_path": RUNTIME_MODEL_PATH,
            "encoding": (
                "json.dumps(ensure_ascii=False,indent=2,sort_keys=True)"
                "+newline_utf8"
            ),
            "runtime_value_must_not_be_self_promoted_to_expected": True,
        }
    ):
        raise AuditError("runtime capsule fixed-path reconstruction prior mismatch")
    if (
        not isinstance(model, Mapping)
        or model.get("bundled_in_overlay") is not False
        or model.get("path_on_x5") != RUNTIME_MODEL_PATH
        or model.get("bytes") != 44_704_833
        or model.get("sha256")
        != "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad"
        or model.get("provider") != "CPUExecutionProvider"
    ):
        raise AuditError("seed17 external CPU model prior mismatch")
    checks.append("FROZEN_RUNTIME_CAPSULE_AND_MODEL_PRIORS")

    references = manifest.get("immutable_references")
    expected_refs = {
        "x5_field_bundle_v2": {
            "release_id": "rootscope_x5_field_bundle_v2",
            "release_dir_relative_to_adventurex": V2_DIR,
            "archive_path_relative_to_adventurex": V2_PATH,
            "sha256": V2_SHA,
            "bytes": V2_BYTES,
            "bundled_in_overlay": False,
            "immutable_reference_only": True,
        },
        "omega_v3_delta_candidate": {
            "release_id": "rootscope_omega_v3_delta_candidate_v1",
            "release_dir_relative_to_adventurex": OMEGA_DIR,
            "archive_path_relative_to_adventurex": OMEGA_PATH,
            "sha256": OMEGA_SHA,
            "bytes": OMEGA_BYTES,
            "bundled_in_overlay": False,
            "immutable_reference_only": True,
        },
    }
    if references != expected_refs:
        raise AuditError("immutable reference records are not exact")
    checks.append("REFERENCE_ONLY_NO_OLD_ARCHIVES_COPIED")

    records = manifest.get("source_allowlist")
    if not isinstance(records, list) or len(records) != len(EXPECTED_SOURCE_MAP):
        raise AuditError("source allowlist cardinality mismatch")
    seen: set[str] = set()
    root_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise AuditError("source allowlist entry is not an object")
        package = record.get("package_path")
        if package not in EXPECTED_SOURCE_MAP or package in seen:
            raise AuditError(f"source allowlist package mismatch: {package}")
        seen.add(str(package))
        if (
            record.get("source_path_relative_to_adventurex")
            != EXPECTED_SOURCE_MAP[str(package)]
        ):
            raise AuditError(f"source provenance mismatch: {package}")
        source = root / Path(
            *PurePosixPath(EXPECTED_SOURCE_MAP[str(package)]).parts
        )
        source_bytes = source.resolve(strict=True).read_bytes()
        if source_bytes != files[str(package)]:
            raise AuditError(f"packaged source differs from local source: {package}")
        if (
            record.get("sha256") != sha256_bytes(source_bytes)
            or record.get("bytes") != len(source_bytes)
        ):
            raise AuditError(f"source record digest mismatch: {package}")
        root_records.append(
            {
                "bytes": record["bytes"],
                "category": record["category"],
                "mode": int(str(record["mode"]), 8),
                "path": package,
                "sha256": record["sha256"],
            }
        )
    if seen != set(EXPECTED_SOURCE_MAP):
        raise AuditError("source allowlist is incomplete")
    calculated_root = sha256_bytes(
        compact_json(sorted(root_records, key=lambda item: item["path"]))
    )
    if manifest.get("content_composition_root_sha256") != calculated_root:
        raise AuditError("content composition root mismatch")
    checks.append("SOURCE_PROVENANCE_AND_COMPOSITION_ROOT")

    if sha256_bytes(files[PRINT_PATH]) != PRINT_SHA:
        raise AuditError("four-up print manifest SHA-256 mismatch")
    print_record = manifest.get("print_assets")
    if (
        not isinstance(print_record, Mapping)
        or print_record.get("manifest_sha256") != PRINT_SHA
        or print_record.get("large_pdf_reference_only", {}).get(
            "bundled_in_overlay"
        )
        is not False
    ):
        raise AuditError("print asset reference boundary mismatch")
    camera = load_object_bytes(files[CAMERA_PATH], "camera identity contract")
    if (
        camera.get("camera", {}).get("usb_vid_pid") != "32e6:9228"
        or camera.get("camera", {}).get("usb_serial") != "202604081837"
        or camera.get("camera", {}).get("stable_by_id_path")
        != (
            "/dev/v4l/by-id/"
            "usb-Web_Camera_Web_Camera_202604081837-video-index0"
        )
    ):
        raise AuditError("camera identity contract mismatch")
    modes = camera.get("capture_modes")
    if not isinstance(modes, list) or {
        (item.get("width"), item.get("height"), item.get("fourcc"), item.get("fps"))
        for item in modes
        if isinstance(item, Mapping)
    } != {(1920, 1080, "MJPG", 30), (1280, 720, "MJPG", 30)}:
        raise AuditError("camera capture modes mismatch")
    checks.append("PRINT_AND_CAMERA_IDENTITY_CONTRACTS")

    for relative, data in files.items():
        if Path(relative).suffix.lower() not in {".py", ".json", ".md", ".txt"}:
            continue
        text = data.decode("utf-8")
        if WINDOWS_ABSOLUTE.search(text):
            raise AuditError(f"absolute PC path found: {relative}")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            raise AuditError(f"secret-like content found: {relative}")
        for forbidden in FORBIDDEN_IMPORTS:
            if re.search(
                rf"(?m)^\s*(?:from|import)\s+{re.escape(forbidden)}(?:\.|\s|$)",
                text,
            ):
                raise AuditError(f"forbidden import {forbidden}: {relative}")
    checks.append("NO_SECRETS_PC_PATHS_OR_FORBIDDEN_IMPORTS")

    with tempfile.TemporaryDirectory(prefix="event_vision_overlay_reaudit_") as tmp:
        rebuilt = Path(tmp) / ARCHIVE_NAME
        rebuild_archive(rebuilt, files, order, directories)
        rebuilt_sha = sha256_file(rebuilt)
        if rebuilt_sha != archive_sha or rebuilt.stat().st_size != archive_bytes:
            raise AuditError("archive is not reproducible deterministic USTAR")
    checks.append("DETERMINISTIC_USTAR_REBUILD_IDENTICAL")

    manifest_sha = sha256_bytes(files["release_manifest.json"])
    final_root = sha256_bytes(files["SHA256SUMS"])
    if (
        receipt.get("manifest_sha256") != manifest_sha
        or receipt.get("content_composition_root_sha256") != calculated_root
        or receipt.get("final_composition_root_sha256") != final_root
    ):
        raise AuditError("build receipt composition bindings mismatch")
    checks.append("BUILD_RECEIPT_COMPOSITION_BINDINGS")

    return {
        "schema": "rootscope.event-vision-overlay-independent-audit.v1",
        "status": "PASS_INDEPENDENT_IMMUTABLE_ZERO_AUTHORITY_AUDIT",
        "overlay_id": OVERLAY_ID,
        "archive": {
            "path_relative_to_adventurex": (
                f"{OUTPUT_RELATIVE}/{ARCHIVE_NAME}"
            ),
            "sha256": archive_sha,
            "bytes": archive_bytes,
            "compression": "none",
            "tar_format": "USTAR",
        },
        "manifest_sha256": manifest_sha,
        "content_composition_root_sha256": calculated_root,
        "final_composition_root_sha256": final_root,
        "covered_file_count": len(sums),
        "checks": checks,
        "check_count": len(checks),
        "selected_bin": None,
        "execution_authority": False,
        "physical_authority": False,
        "service_or_gate_created": False,
        "camera_opened": False,
    }


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=root / OUTPUT_RELATIVE / ARCHIVE_NAME,
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_release(root, args.archive)
        if args.output_json is not None:
            output = args.output_json
            if not output.is_absolute():
                output = root / output
            write_new(output, canonical_json(result))
    except (OSError, AuditError, json.JSONDecodeError, UnicodeError) as exc:
        parser.exit(2, f"FAIL_CLOSED: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
