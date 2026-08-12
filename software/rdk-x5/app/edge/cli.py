"""Command-line entry point for offline capsule preflight and self-test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .capsule import CapsuleConfig
from .preflight import run_preflight
from .selftest import run_simulated_selftest


def _emit(payload: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "selftest", "show-contract"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        config = CapsuleConfig.from_json_file(args.config)
        if args.command == "preflight":
            payload = run_preflight(config)
            _emit(payload, args.output)
            return 2 if payload["status"] == "FAIL" else 0
        if args.command == "selftest":
            payload = run_simulated_selftest(config)
            _emit(payload, args.output)
            return 0
        payload = config.to_dict()
        _emit(payload, args.output)
        return 0
    except Exception as exc:
        payload = {
            "schema_version": "rootscope.x5-capsule-cli-error.v1",
            "status": "FAIL_CLOSED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "hardware_touched": False,
            "network_touched": False,
            "ports_enumerated": False,
            "x5_validated": False,
            "bpu_ready": False,
            "bpu_used": False,
            "physical_authority": False,
            "execution_authority": False,
            "physical_completion": False,
        }
        _emit(payload, args.output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
