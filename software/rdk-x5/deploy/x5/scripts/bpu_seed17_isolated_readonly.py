#!/usr/bin/env python3
"""Manual, fail-closed RootScope seed17 BPU preflight and one-image replay.

With no input selector this command verifies platform, model hash, BPU load,
and the frozen input/output interface.  It does not open a camera and does not
run inference.  ``--image`` performs one hash-bound image replay.
``--camera-device`` is a separate, explicit one-frame V4L2 path and never
enumerates devices.  No mode imports the RootScope state machine, serial
layer, pump driver, or service launcher.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.edge.bpu_seed17 import (  # noqa: E402
    Seed17BpuRunner,
    capture_one_explicit_v4l2_bgr,
    load_hash_bound_image_bgr,
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser()
    if target.is_symlink():
        raise ValueError("--output-json must not be a symlink")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    partial.write_bytes(data)
    os.chmod(partial, 0o600)
    os.replace(partial, target)


def _emit(payload: Mapping[str, Any], output_json: Path | None) -> None:
    if output_json is not None:
        _atomic_write_json(output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _require_manual_x5_platform() -> None:
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Linux" or machine not in {"aarch64", "arm64"}:
        raise RuntimeError(
            f"manual BPU replay requires Linux/aarch64, got {system}/{machine}"
        )


def _require_bpu_runtime_venv() -> None:
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 10):
        raise RuntimeError("manual BPU replay requires CPython 3.10")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "manual BPU replay requires its independent --system-site-packages venv"
        )
    venv_root = Path(sys.prefix).resolve(strict=True)
    config_path = venv_root / "pyvenv.cfg"
    if not config_path.is_file():
        raise RuntimeError("BPU venv is missing pyvenv.cfg")
    config: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            config[key.strip().lower()] = value.strip().lower()
    if config.get("include-system-site-packages") != "true":
        raise RuntimeError(
            "BPU venv must set include-system-site-packages=true; core v1 venv is forbidden"
        )
    for module_name in ("numpy", "hobot_dnn"):
        spec = importlib.util.find_spec(module_name)
        origin = None if spec is None else spec.origin
        if not origin or origin in {"built-in", "frozen"}:
            raise RuntimeError(f"system {module_name} is unavailable")
        try:
            Path(origin).resolve(strict=True).relative_to(venv_root)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                f"{module_name} resolves inside the BPU venv; preserve the RDK system ABI"
            )


def _python_runtime_report() -> Mapping[str, Any]:
    modules: dict[str, Any] = {}
    for module_name in ("numpy", "hobot_dnn", "PIL"):
        spec = importlib.util.find_spec(module_name)
        modules[module_name] = {
            "available": spec is not None,
            "origin": None if spec is None else spec.origin,
        }
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "venv_prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "include_system_site_packages": True,
        "core_v1_venv_allowed": False,
        "modules": modules,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bin", required=True, type=Path)
    parser.add_argument("--expected-model-sha256", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--image",
        type=Path,
        help="one explicit golden/reference image; requires --expected-image-sha256",
    )
    source.add_argument(
        "--camera-device",
        help="one explicit /dev/... V4L2 path; reads exactly one frame",
    )
    parser.add_argument("--expected-image-sha256")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional provenance receipt; stdout is always emitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.image is not None and args.expected_image_sha256 is None:
        raise SystemExit("[fatal] --image requires --expected-image-sha256")
    if args.image is None and args.expected_image_sha256 is not None:
        raise SystemExit("[fatal] --expected-image-sha256 is valid only with --image")

    lifecycle = {
        "platform_check_passed": False,
        "runtime_environment_check_passed": False,
        "bpu_load_attempted": False,
        "bpu_forward_attempted": False,
        "camera_open_attempted": False,
    }
    try:
        _require_manual_x5_platform()
        lifecycle["platform_check_passed"] = True
        _require_bpu_runtime_venv()
        lifecycle["runtime_environment_check_passed"] = True
        lifecycle["bpu_load_attempted"] = True
        runner = Seed17BpuRunner(args.model_bin, args.expected_model_sha256)
        if args.image is not None:
            bgr, provenance = load_hash_bound_image_bgr(
                args.image, args.expected_image_sha256
            )
            lifecycle["bpu_forward_attempted"] = True
            report = dict(runner.run_bgr(bgr, source_provenance=provenance))
        elif args.camera_device is not None:
            lifecycle["camera_open_attempted"] = True
            bgr, provenance = capture_one_explicit_v4l2_bgr(args.camera_device)
            lifecycle["bpu_forward_attempted"] = True
            report = dict(runner.run_bgr(bgr, source_provenance=provenance))
        else:
            report = dict(runner.preflight_report())
        report["lifecycle"] = dict(lifecycle)
        report["python_runtime"] = dict(_python_runtime_report())
        _emit(report, args.output_json)
        return 0
    except Exception as exc:
        failure = {
            "schema": "rootscope.seed17-bpu-isolated-replay-error.v1",
            "status": "FAIL_CLOSED_NO_AUTHORITY",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "lifecycle": dict(lifecycle),
            "claims": {
                "x5_ready": False,
                "x5_validated": False,
                "camera_qualified": False,
                "model_candidate": False,
                "model_qualified": False,
                "production_integration_allowed": False,
                "production_authority_enabled": False,
                "irrigation_authority_enabled": False,
            },
            "authority": {
                # Conservative after an attempted load/open: an external
                # runtime may have allocated BPU/CMA or opened the device
                # before raising, so never claim that hardware was untouched.
                "hardware_touched": bool(
                    lifecycle["bpu_load_attempted"]
                    or lifecycle["camera_open_attempted"]
                ),
                "network_touched": False,
                "device_enumerated": False,
                "serial_write": False,
                "state_machine_write": False,
                "pump_command": False,
                "irrigation_execution": False,
                "execution_authority": False,
                "physical_authority": False,
                "physical_completion": False,
                "bpu_used": bool(lifecycle["bpu_forward_attempted"]),
            },
        }
        try:
            _emit(failure, args.output_json)
        except Exception:
            print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
