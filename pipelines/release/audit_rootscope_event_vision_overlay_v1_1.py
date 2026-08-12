#!/usr/bin/env python3
"""Independent immutable-release audit for event-vision overlay v1.1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.release import audit_rootscope_event_vision_overlay_v1 as _base


OVERLAY_ID = "rootscope_event_vision_overlay_v1_1"
ARCHIVE_NAME = f"{OVERLAY_ID}.tar"
SCHEMA = "rootscope.event-vision-overlay.v1_1"
OUTPUT_RELATIVE = f"output/releases/{OVERLAY_ID}"
HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1_1.py"
HELPER_SOURCE = "tools/release/verify_rootscope_event_vision_overlay_v1_1.py"
LEGACY_HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1.py"
LEGACY_HELPER_SOURCE = "tools/release/verify_rootscope_event_vision_overlay_v1.py"

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


def audit_release(root: Path, archive_path: Path) -> dict[str, Any]:
    _activate()
    result = dict(_base.audit_release(root, archive_path))
    result["schema"] = (
        "rootscope.event-vision-overlay-independent-audit.v1_1"
    )
    result["covered_file_count"] = len(EXPECTED_FILES) - 1
    result["v1_1_preflight_entrypoint"] = HELPER_PACKAGE
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
