#!/usr/bin/env python3
"""Read-only, zero-authority preflight for an extracted event-vision overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping, Sequence


OVERLAY_ID = "rootscope_event_vision_overlay_v1"
SCHEMA = "rootscope.event-vision-overlay.v1"
V2_SHA256 = "e6627685170252004d118bf77a690a9f89ad3afa274910697554d0f5cc8c3ebb"
V2_BYTES = 696_832_000
OMEGA_SHA256 = "c910f4d2e002ccdbd5643fa47f300ade8e56af8ad1c1a2a04fa4e4a0a0fab881"
OMEGA_BYTES = 665_600
PRINT_SHA256 = "5e23e6133e9a59d8327bd751c4d6e5434d0c5a86402fd920c9d99e547613d827"
CAPSULE_TEMPLATE_SHA256 = (
    "1b7e9b96ccd4ec4e5ab534e1f305224c3c6330a3ab2efb8ca2e5d0fc52fcfcbb"
)
RUNTIME_CAPSULE_SHA256 = (
    "1e839948e466895e0b416aababb39402380463f9ce125c8aeb59f4087191cb97"
)
RUNTIME_CAPSULE_BYTES = 2_765
RUNTIME_CAPSULE_PATH = (
    "/opt/rootscope/.local/share/rootscope-field-v2/core_v1/config/"
    "rootscope_x5_offline_core_v1.capsule.json"
)
CAMERA_CONTRACT_PATH = (
    "rootscope/configs/event_vision/camera_identity_x5_20260723.json"
)
PRINT_MANIFEST_PATH = (
    "output/pdf/RootScope_A4_four_up_field_cards_20260723_manifest.json"
)
EXPECTED_FILES = {
    "rootscope/app/__init__.py",
    "rootscope/app/edge/__init__.py",
    "rootscope/app/edge/capsule.py",
    "rootscope/app/edge/onnx_cpu.py",
    "rootscope/app/vision/__init__.py",
    "rootscope/app/vision/quality_gate.py",
    "rootscope/app/vision/uvc_card_capture.py",
    "rootscope/app/vision/card_geometric_matcher.py",
    "rootscope/app/vision/dual_path_demo.py",
    "rootscope/app/omega_vision/__init__.py",
    "rootscope/app/omega_vision/ood.py",
    "rootscope/app/omega_vision/uvc_card_frontend.py",
    "rootscope/tests/__init__.py",
    "rootscope/tests/test_uvc_card_capture.py",
    "rootscope/tests/test_omega_uvc_card_frontend.py",
    "rootscope/deploy/x5/capsule_config.seed17_cpu_experimental.json",
    "rootscope/configs/omega/vision_board_replay_new_x5_20260723.json",
    CAMERA_CONTRACT_PATH,
    "rootscope/app/vision/known_card_template_registry.frozen.experimental.json",
    "rootscope/app/vision/known_card_templates/grass_clump_163498042.jpg",
    "rootscope/app/vision/known_card_templates/low_shrub_68787114.jpg",
    "rootscope/app/vision/known_card_templates/young_tree_92774234.jpg",
    "rootscope/app/vision/dual_path_demo.thresholds.example.json",
    "rootscope/app/vision/card_geometric_matcher.config.example.json",
    PRINT_MANIFEST_PATH,
    "docs/UVC_CARD_CAPTURE_RUNBOOK_ZH.md",
    "docs/ROOTSCOPE_UVC_CARD_FRONTEND_RUNBOOK_ZH.md",
    "tools/verify_rootscope_event_vision_overlay_v1.py",
    "release_manifest.json",
    "SHA256SUMS",
}
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
}
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:^|[^A-Za-z0-9_])[A-Z]:[\\/]")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
)


class PreflightError(RuntimeError):
    """The extracted overlay or one immutable reference failed verification."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot parse JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PreflightError(f"JSON root must be an object: {path}")
    return payload


def safe_relative(text: str) -> PurePosixPath:
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        raise PreflightError(f"unsafe package path: {text!r}")
    return path


def parse_sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if match is None:
            raise PreflightError(f"malformed SHA256SUMS line {line_number}")
        digest, relative = match.groups()
        safe_relative(relative)
        if relative in entries:
            raise PreflightError(f"duplicate SHA256SUMS path: {relative}")
        entries[relative] = digest
    return entries


def verify_reference(path: Path, expected_sha: str, expected_bytes: int) -> None:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise PreflightError(f"immutable reference must be a regular file: {resolved}")
    if resolved.stat().st_size != expected_bytes:
        raise PreflightError(f"immutable reference byte mismatch: {resolved}")
    if sha256_file(resolved) != expected_sha:
        raise PreflightError(f"immutable reference SHA-256 mismatch: {resolved}")


def verify_overlay(
    overlay_root: Path,
    v2_archive: Path,
    omega_archive: Path,
    runtime_capsule: Path | None = None,
) -> dict[str, Any]:
    root = overlay_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise PreflightError("overlay root must be a regular non-symlink directory")

    manifest_path = root / "release_manifest.json"
    sums_path = root / "SHA256SUMS"
    manifest = load_object(manifest_path)
    if manifest.get("schema") != SCHEMA or manifest.get("overlay_id") != OVERLAY_ID:
        raise PreflightError("overlay manifest identity mismatch")
    authority = manifest.get("authority")
    if not isinstance(authority, Mapping) or any(value is not False for value in authority.values()):
        raise PreflightError("overlay authority must contain only false values")
    qualification = manifest.get("qualification")
    if not isinstance(qualification, Mapping):
        raise PreflightError("overlay qualification is missing")
    if qualification.get("selected_bin") is not None:
        raise PreflightError("plant BPU selected_bin must remain null")
    if qualification.get("production_integration_allowed") is not False:
        raise PreflightError("production integration must remain false")
    assets = manifest.get("frozen_runtime_asset_contracts")
    if not isinstance(assets, Mapping):
        raise PreflightError("frozen runtime asset contracts are missing")
    template = assets.get("capsule_template")
    runtime = assets.get("runtime_capsule")
    if (
        not isinstance(template, Mapping)
        or template.get("sha256") != CAPSULE_TEMPLATE_SHA256
        or template.get("bytes") != 2_330
        or not isinstance(runtime, Mapping)
        or runtime.get("sha256") != RUNTIME_CAPSULE_SHA256
        or runtime.get("bytes") != RUNTIME_CAPSULE_BYTES
        or runtime.get("path_on_x5") != RUNTIME_CAPSULE_PATH
        or runtime.get("bundled_in_overlay") is not False
        or runtime.get("reconstruction", {}).get(
            "runtime_value_must_not_be_self_promoted_to_expected"
        )
        is not True
    ):
        raise PreflightError("runtime capsule prior contract mismatch")

    sums = parse_sha256sums(sums_path)
    actual_files: set[str] = set()
    for item in root.rglob("*"):
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise PreflightError(f"symlink is forbidden in overlay: {relative}")
        if item.is_file() and relative != "SHA256SUMS":
            actual_files.add(relative)
    if set(sums) != actual_files:
        raise PreflightError(
            "SHA256SUMS coverage mismatch: "
            f"missing={sorted(actual_files - set(sums))} "
            f"extra={sorted(set(sums) - actual_files)}"
        )
    if actual_files | {"SHA256SUMS"} != EXPECTED_FILES:
        raise PreflightError(
            "fixed overlay allowlist mismatch: "
            f"missing={sorted(EXPECTED_FILES - (actual_files | {'SHA256SUMS'}))} "
            f"extra={sorted((actual_files | {'SHA256SUMS'}) - EXPECTED_FILES)}"
        )
    for relative, expected in sums.items():
        if sha256_file(root / Path(*PurePosixPath(relative).parts)) != expected:
            raise PreflightError(f"packaged file SHA-256 mismatch: {relative}")

    if sha256_file(root / PRINT_MANIFEST_PATH) != PRINT_SHA256:
        raise PreflightError("four-up print manifest SHA-256 mismatch")
    camera = load_object(root / CAMERA_CONTRACT_PATH)
    if camera.get("status") != "FROZEN_EVENT_CAMERA_IDENTITY_ZERO_AUTHORITY":
        raise PreflightError("camera identity contract status mismatch")
    camera_record = camera.get("camera")
    if not isinstance(camera_record, Mapping) or camera_record != {
        "stable_by_id_path": (
            "/dev/v4l/by-id/"
            "usb-Web_Camera_Web_Camera_202604081837-video-index0"
        ),
        "usb_serial": "202604081837",
        "usb_vid_pid": "32e6:9228",
    }:
        raise PreflightError("camera identity contract mismatch")

    references = manifest.get("immutable_references")
    if not isinstance(references, Mapping):
        raise PreflightError("immutable references are missing")
    expected_references = {
        "x5_field_bundle_v2": (V2_SHA256, V2_BYTES),
        "omega_v3_delta_candidate": (OMEGA_SHA256, OMEGA_BYTES),
    }
    for key, (expected_sha, expected_bytes) in expected_references.items():
        record = references.get(key)
        if not isinstance(record, Mapping):
            raise PreflightError(f"immutable reference record missing: {key}")
        if (
            record.get("sha256") != expected_sha
            or record.get("bytes") != expected_bytes
            or record.get("bundled_in_overlay") is not False
            or record.get("immutable_reference_only") is not True
        ):
            raise PreflightError(f"immutable reference manifest mismatch: {key}")
    verify_reference(v2_archive, V2_SHA256, V2_BYTES)
    verify_reference(omega_archive, OMEGA_SHA256, OMEGA_BYTES)
    runtime_capsule_verified = False
    if runtime_capsule is not None:
        verify_reference(
            runtime_capsule,
            RUNTIME_CAPSULE_SHA256,
            RUNTIME_CAPSULE_BYTES,
        )
        runtime_capsule_verified = True

    for relative in actual_files:
        path = root / Path(*PurePosixPath(relative).parts)
        if path.suffix.lower() not in {
            ".py",
            ".json",
            ".md",
            ".txt",
            ".jpg",
        }:
            raise PreflightError(f"unexpected overlay file suffix: {relative}")
        if path.suffix.lower() in {".py", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            if WINDOWS_ABSOLUTE.search(text):
                raise PreflightError(f"absolute Windows path found: {relative}")
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                raise PreflightError(f"secret-like material found: {relative}")
            for forbidden in FORBIDDEN_IMPORT_ROOTS:
                if re.search(
                    rf"(?m)^\s*(?:from|import)\s+{re.escape(forbidden)}(?:\.|\s|$)",
                    text,
                ):
                    raise PreflightError(
                        f"forbidden project import {forbidden}: {relative}"
                    )

    return {
        "schema": "rootscope.event-vision-overlay-preflight.v1",
        "status": "PASS_ZERO_AUTHORITY_READ_ONLY_PREFLIGHT",
        "overlay_id": OVERLAY_ID,
        "manifest_sha256": sha256_file(manifest_path),
        "sha256sums_sha256": sha256_file(sums_path),
        "covered_file_count": len(sums),
        "v2_reference_sha256": V2_SHA256,
        "omega_reference_sha256": OMEGA_SHA256,
        "runtime_capsule_expected_sha256": RUNTIME_CAPSULE_SHA256,
        "runtime_capsule_verified": runtime_capsule_verified,
        "camera_or_hardware_opened": False,
        "service_or_gate_created": False,
        "network_configuration_touched": False,
        "serial_gpio_pump_touched": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay-root", required=True, type=Path)
    parser.add_argument("--v2-archive", required=True, type=Path)
    parser.add_argument("--omega-archive", required=True, type=Path)
    parser.add_argument("--runtime-capsule", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify_overlay(
            args.overlay_root,
            args.v2_archive,
            args.omega_archive,
            args.runtime_capsule,
        )
    except (OSError, PreflightError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
