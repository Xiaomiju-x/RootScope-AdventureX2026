#!/usr/bin/env python3
"""Read-only, zero-authority preflight for event-vision overlay v1.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


def _load_core() -> Any:
    try:
        from tools.release import (
            verify_rootscope_event_vision_overlay_v1 as core,
        )
        return core
    except ImportError:
        overlay_root = Path(__file__).resolve().parents[1]
        if str(overlay_root) not in sys.path:
            sys.path.insert(0, str(overlay_root))
        from tools import verify_rootscope_event_vision_overlay_v1 as core
        return core


_base = _load_core()
OVERLAY_ID = "rootscope_event_vision_overlay_v1_1"
SCHEMA = "rootscope.event-vision-overlay.v1_1"
HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1_1.py"
LEGACY_HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1.py"
EXPECTED_FILES = (
    set(_base.EXPECTED_FILES)
    | {HELPER_PACKAGE}
)

PreflightError = _base.PreflightError


def _activate() -> None:
    _base.OVERLAY_ID = OVERLAY_ID
    _base.SCHEMA = SCHEMA
    _base.EXPECTED_FILES = EXPECTED_FILES


def verify_overlay(
    overlay_root: Path,
    v2_archive: Path,
    omega_archive: Path,
    runtime_capsule: Path | None = None,
) -> Mapping[str, Any]:
    _activate()
    result = dict(
        _base.verify_overlay(
            overlay_root,
            v2_archive,
            omega_archive,
            runtime_capsule,
        )
    )
    result["schema"] = "rootscope.event-vision-overlay-preflight.v1_1"
    result["preflight_entrypoint"] = HELPER_PACKAGE
    result["legacy_core_bundled_read_only"] = LEGACY_HELPER_PACKAGE
    return result


def build_parser() -> argparse.ArgumentParser:
    return _base.build_parser()


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
