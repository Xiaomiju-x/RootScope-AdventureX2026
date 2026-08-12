#!/usr/bin/env python3
"""Build the immutable RootScope event-vision overlay v1.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tools.release import build_rootscope_event_vision_overlay_v1 as _base


BUILD_DATE = "2026-07-23"
OVERLAY_ID = "rootscope_event_vision_overlay_v1_2"
ARCHIVE_NAME = f"{OVERLAY_ID}.tar"
SCHEMA = "rootscope.event-vision-overlay.v1_2"
OUTPUT_RELATIVE = f"output/releases/{OVERLAY_ID}"
HELPER_SOURCE = "tools/release/verify_rootscope_event_vision_overlay_v1_2.py"
HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1_2.py"
LEGACY_HELPER_SOURCE = "tools/release/verify_rootscope_event_vision_overlay_v1.py"
LEGACY_HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1.py"
CAPTURE_RUNBOOK_PACKAGE = "docs/UVC_CARD_CAPTURE_RUNBOOK_ZH.md"
FRONTEND_RUNBOOK_PACKAGE = "docs/ROOTSCOPE_UVC_CARD_FRONTEND_RUNBOOK_ZH.md"

SourceSpec = _base.SourceSpec
PackageEntry = _base.PackageEntry
_CORE_LOAD_SOURCE_ENTRIES = _base.load_source_entries


def _v1_2_source_specs() -> tuple[SourceSpec, ...]:
    specs: list[SourceSpec] = []
    for spec in _base.SOURCE_SPECS:
        if spec.package_relative == LEGACY_HELPER_PACKAGE:
            specs.append(
                SourceSpec(
                    source_relative=LEGACY_HELPER_SOURCE,
                    package_relative=LEGACY_HELPER_PACKAGE,
                    category="zero_authority_preflight_core",
                    mode=0o644,
                )
            )
        else:
            specs.append(spec)
    specs.append(
        SourceSpec(
            source_relative=HELPER_SOURCE,
            package_relative=HELPER_PACKAGE,
            category="zero_authority_preflight",
            mode=0o755,
        )
    )
    return tuple(specs)


SOURCE_SPECS = _v1_2_source_specs()


def validate_runbook_semantics(entries: Sequence[PackageEntry]) -> None:
    by_path = {entry.path: entry for entry in entries}
    capture = by_path[CAPTURE_RUNBOOK_PACKAGE].data.decode("utf-8")
    frontend = by_path[FRONTEND_RUNBOOK_PACKAGE].data.decode("utf-8")
    for label, text in (("capture", capture), ("frontend", frontend)):
        if (
            "/opt/rootscope/rootscope" in text
            or "rootscope_inputs" in text
            or "rootscope_event_vision_overlay_v1_1" in text
        ):
            raise ValueError(f"{label} runbook contains obsolete deployment path")
        required = (
            "rootscope_event_vision_overlay_v1_2",
            'APP_ROOT="$OVERLAY_ROOT/rootscope"',
            'PRINT_MANIFEST="$OVERLAY_ROOT/output/pdf/'
            'RootScope_A4_four_up_field_cards_20260723_manifest.json"',
            "PYTHONDONTWRITEBYTECODE=1",
        )
        for value in required:
            if value not in text:
                raise ValueError(
                    f"{label} runbook missing deployment invariant: {value}"
                )
    for value in (
        "USB_VID='32e6'",
        "USB_PID='9228'",
        "USB_SERIAL='202604081837'",
        "以下命令只用于 PC 工作区开发验证",
    ):
        if value not in frontend:
            raise ValueError(
                f"frontend runbook missing semantic invariant: {value}"
            )
    if "在新的、单独审计的增量包形成前" in frontend:
        raise ValueError("frontend runbook contains obsolete future-release wording")
    for value in (
        "backend 构造成功后",
        "不会生成一份声称完成第三次身份核验的成功 manifest",
        'PY="$HOME/.local/share/rootscope-field-v2/core_v1/venvs/'
        'rootscope_x5_offline_core_v1/bin/python3"',
        'OUT_ROOT="$HOME/rootscope_event_capture"',
    ):
        if value not in capture:
            raise ValueError(
                f"capture runbook missing failure-boundary invariant: {value}"
            )


def _strict_load_source_entries(root: Path) -> list[PackageEntry]:
    entries = _CORE_LOAD_SOURCE_ENTRIES(root)
    validate_runbook_semantics(entries)
    return entries


def _activate() -> None:
    _base.BUILD_DATE = BUILD_DATE
    _base.OVERLAY_ID = OVERLAY_ID
    _base.ARCHIVE_NAME = ARCHIVE_NAME
    _base.SCHEMA = SCHEMA
    _base.OUTPUT_RELATIVE = OUTPUT_RELATIVE
    _base.HELPER_SOURCE = HELPER_SOURCE
    _base.HELPER_PACKAGE = HELPER_PACKAGE
    _base.SOURCE_SPECS = SOURCE_SPECS
    _base.load_source_entries = _strict_load_source_entries


def adventurex_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_source_entries(root: Path) -> list[PackageEntry]:
    _activate()
    return _strict_load_source_entries(root)


def add_generated_entries(
    source_entries: Sequence[PackageEntry],
) -> list[PackageEntry]:
    _activate()
    return _base.add_generated_entries(source_entries)


def build_manifest(entries: Sequence[PackageEntry]) -> dict[str, Any]:
    _activate()
    return _base.build_manifest(entries)


def write_deterministic_ustar(
    path: Path,
    entries: Sequence[PackageEntry],
) -> None:
    _activate()
    _base.write_deterministic_ustar(path, entries)


def build_release(root: Path) -> dict[str, Any]:
    _activate()
    return _base.build_release(root)


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
