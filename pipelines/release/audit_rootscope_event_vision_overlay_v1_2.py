#!/usr/bin/env python3
"""Independent immutable-release audit for event-vision overlay v1.2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from tools.release import audit_rootscope_event_vision_overlay_v1 as _base


OVERLAY_ID = "rootscope_event_vision_overlay_v1_2"
ARCHIVE_NAME = f"{OVERLAY_ID}.tar"
SCHEMA = "rootscope.event-vision-overlay.v1_2"
OUTPUT_RELATIVE = f"output/releases/{OVERLAY_ID}"
HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1_2.py"
HELPER_SOURCE = "tools/release/verify_rootscope_event_vision_overlay_v1_2.py"
LEGACY_HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1.py"
LEGACY_HELPER_SOURCE = "tools/release/verify_rootscope_event_vision_overlay_v1.py"
CAPTURE_RUNBOOK = "docs/UVC_CARD_CAPTURE_RUNBOOK_ZH.md"
FRONTEND_RUNBOOK = "docs/ROOTSCOPE_UVC_CARD_FRONTEND_RUNBOOK_ZH.md"

EXPECTED_SOURCE_MAP = dict(_base.EXPECTED_SOURCE_MAP)
EXPECTED_SOURCE_MAP[LEGACY_HELPER_PACKAGE] = LEGACY_HELPER_SOURCE
EXPECTED_SOURCE_MAP[HELPER_PACKAGE] = HELPER_SOURCE
EXPECTED_FILES = set(EXPECTED_SOURCE_MAP) | {
    "release_manifest.json",
    "SHA256SUMS",
}

AuditError = _base.AuditError
canonical_json = _base.canonical_json
parse_sums = _base.parse_sums
rebuild_archive = _base.rebuild_archive
sha256_file = _base.sha256_file


def _activate() -> None:
    _base.OVERLAY_ID = OVERLAY_ID
    _base.ARCHIVE_NAME = ARCHIVE_NAME
    _base.SCHEMA = SCHEMA
    _base.OUTPUT_RELATIVE = OUTPUT_RELATIVE
    _base.HELPER_PACKAGE = HELPER_PACKAGE
    _base.EXPECTED_SOURCE_MAP = EXPECTED_SOURCE_MAP
    _base.EXPECTED_FILES = EXPECTED_FILES


def _audit_runbook_semantics(files: dict[str, bytes]) -> None:
    capture = files[CAPTURE_RUNBOOK].decode("utf-8")
    frontend = files[FRONTEND_RUNBOOK].decode("utf-8")
    for label, text in (("capture", capture), ("frontend", frontend)):
        if (
            "/opt/rootscope/rootscope" in text
            or "rootscope_inputs" in text
            or "rootscope_event_vision_overlay_v1_1" in text
        ):
            raise AuditError(f"{label} runbook contains obsolete path")
        for value in (
            "rootscope_event_vision_overlay_v1_2",
            'APP_ROOT="$OVERLAY_ROOT/rootscope"',
            'PRINT_MANIFEST="$OVERLAY_ROOT/output/pdf/'
            'RootScope_A4_four_up_field_cards_20260723_manifest.json"',
            "PYTHONDONTWRITEBYTECODE=1",
        ):
            if value not in text:
                raise AuditError(
                    f"{label} runbook missing deployment invariant: {value}"
                )
    for value in (
        "USB_VID='32e6'",
        "USB_PID='9228'",
        "USB_SERIAL='202604081837'",
        "以下命令只用于 PC 工作区开发验证",
    ):
        if value not in frontend:
            raise AuditError(
                f"frontend runbook missing semantic invariant: {value}"
            )
    if "在新的、单独审计的增量包形成前" in frontend:
        raise AuditError("frontend runbook has obsolete future-release wording")
    for value in (
        "backend 构造成功后",
        "不会生成一份声称完成第三次身份核验的成功 manifest",
        'PY="$HOME/.local/share/rootscope-field-v2/core_v1/venvs/'
        'rootscope_x5_offline_core_v1/bin/python3"',
        'OUT_ROOT="$HOME/rootscope_event_capture"',
    ):
        if value not in capture:
            raise AuditError(
                f"capture runbook missing failure boundary: {value}"
            )


def audit_release(root: Path, archive_path: Path) -> dict[str, Any]:
    _activate()
    result = dict(_base.audit_release(root, archive_path))
    files, _, _ = _base.read_archive(archive_path.resolve(strict=True))
    _audit_runbook_semantics(files)
    checks = list(result["checks"])
    checks.append("RUNBOOK_SEMANTIC_DEPLOYMENT_AND_FAILURE_BOUNDARIES")
    result["checks"] = checks
    result["check_count"] = len(checks)
    result["schema"] = (
        "rootscope.event-vision-overlay-independent-audit.v1_2"
    )
    result["covered_file_count"] = len(EXPECTED_FILES) - 1
    result["v1_2_preflight_entrypoint"] = HELPER_PACKAGE
    result["legacy_preflight_core_bundled_read_only"] = (
        LEGACY_HELPER_PACKAGE
    )
    return result


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
