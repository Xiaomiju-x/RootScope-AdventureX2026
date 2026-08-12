#!/usr/bin/env python3
"""Read-only, zero-authority preflight for event-vision overlay v1.2."""

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
OVERLAY_ID = "rootscope_event_vision_overlay_v1_2"
SCHEMA = "rootscope.event-vision-overlay.v1_2"
HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1_2.py"
LEGACY_HELPER_PACKAGE = "tools/verify_rootscope_event_vision_overlay_v1.py"
CAPTURE_RUNBOOK = "docs/UVC_CARD_CAPTURE_RUNBOOK_ZH.md"
FRONTEND_RUNBOOK = "docs/ROOTSCOPE_UVC_CARD_FRONTEND_RUNBOOK_ZH.md"
EXPECTED_FILES = set(_base.EXPECTED_FILES) | {HELPER_PACKAGE}

PreflightError = _base.PreflightError


def _activate() -> None:
    _base.OVERLAY_ID = OVERLAY_ID
    _base.SCHEMA = SCHEMA
    _base.EXPECTED_FILES = EXPECTED_FILES


def _verify_runbook_semantics(overlay_root: Path) -> None:
    capture = (overlay_root / CAPTURE_RUNBOOK).read_text("utf-8")
    frontend = (overlay_root / FRONTEND_RUNBOOK).read_text("utf-8")
    for label, text in (("capture", capture), ("frontend", frontend)):
        if (
            "/opt/rootscope/rootscope" in text
            or "rootscope_inputs" in text
            or "rootscope_event_vision_overlay_v1_1" in text
        ):
            raise PreflightError(f"{label} runbook contains obsolete path")
        for value in (
            "rootscope_event_vision_overlay_v1_2",
            'APP_ROOT="$OVERLAY_ROOT/rootscope"',
            'PRINT_MANIFEST="$OVERLAY_ROOT/output/pdf/'
            'RootScope_A4_four_up_field_cards_20260723_manifest.json"',
            "PYTHONDONTWRITEBYTECODE=1",
        ):
            if value not in text:
                raise PreflightError(
                    f"{label} runbook missing deployment invariant: {value}"
                )
    for value in (
        "USB_VID='32e6'",
        "USB_PID='9228'",
        "USB_SERIAL='202604081837'",
        "以下命令只用于 PC 工作区开发验证",
    ):
        if value not in frontend:
            raise PreflightError(
                f"frontend runbook missing semantic invariant: {value}"
            )
    if "在新的、单独审计的增量包形成前" in frontend:
        raise PreflightError("frontend runbook has obsolete future-release wording")
    for value in (
        "backend 构造成功后",
        "不会生成一份声称完成第三次身份核验的成功 manifest",
        'PY="$HOME/.local/share/rootscope-field-v2/core_v1/venvs/'
        'rootscope_x5_offline_core_v1/bin/python3"',
        'OUT_ROOT="$HOME/rootscope_event_capture"',
    ):
        if value not in capture:
            raise PreflightError(
                f"capture runbook missing failure boundary: {value}"
            )


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
    _verify_runbook_semantics(overlay_root.resolve(strict=True))
    result["schema"] = "rootscope.event-vision-overlay-preflight.v1_2"
    result["preflight_entrypoint"] = HELPER_PACKAGE
    result["legacy_core_bundled_read_only"] = LEGACY_HELPER_PACKAGE
    result["runbook_semantic_gate"] = "PASS"
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
