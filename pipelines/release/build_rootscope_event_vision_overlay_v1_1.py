#!/usr/bin/env python3
"""Build the immutable RootScope event-vision overlay v1.1.

The proven v1 builder remains the implementation core.  This module binds a
new release identity and extends the exact source allowlist with the v1.1
preflight entrypoint.  The v1 output directory and archive are never touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tools.release import build_rootscope_event_vision_overlay_v1 as _base


BUILD_DATE = "2026-07-23"
OVERLAY_ID = "rootscope_event_vision_overlay_v1_1"
ARCHIVE_NAME = f"{OVERLAY_ID}.tar"
SCHEMA = "rootscope.event-vision-overlay.v1_1"
OUTPUT_RELATIVE = f"output/releases/{OVERLAY_ID}"
HELPER_SOURCE = "tools/release/verify_rootscope_event_vision_overlay_v1_1.py"
HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1_1.py"
LEGACY_HELPER_SOURCE = "tools/release/verify_rootscope_event_vision_overlay_v1.py"
LEGACY_HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1.py"

SourceSpec = _base.SourceSpec
PackageEntry = _base.PackageEntry


def _v1_1_source_specs() -> tuple[SourceSpec, ...]:
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


SOURCE_SPECS = _v1_1_source_specs()


def _activate() -> None:
    """Bind the v1 core to the v1.1 immutable identity."""

    _base.BUILD_DATE = BUILD_DATE
    _base.OVERLAY_ID = OVERLAY_ID
    _base.ARCHIVE_NAME = ARCHIVE_NAME
    _base.SCHEMA = SCHEMA
    _base.OUTPUT_RELATIVE = OUTPUT_RELATIVE
    _base.HELPER_SOURCE = HELPER_SOURCE
    _base.HELPER_PACKAGE = HELPER_PACKAGE
    _base.SOURCE_SPECS = SOURCE_SPECS


def adventurex_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_source_entries(root: Path) -> list[PackageEntry]:
    _activate()
    return _base.load_source_entries(root)


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
