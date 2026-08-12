#!/usr/bin/env python3
"""Capture one frame from one explicit V4L2 device, then run dual-path evidence.

Nothing happens on import.  The caller must name an absolute ``/dev/...`` path
and every output path.  The script opens only that device, captures a bounded
number of warm-up frames plus one evidence frame, closes it, and invokes the
zero-authority image-file dual-path CLI.  It never enumerates devices and has
no pump, serial, state-machine, or service-start integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
from typing import Any, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_explicit_device(value: str) -> tuple[Path, Path]:
    if platform.system() != "Linux":
        raise ValueError("one-shot V4L2 capture requires Linux")
    configured = Path(value).expanduser()
    if not configured.is_absolute() or configured.as_posix() == "/dev":
        raise ValueError("--device must be an absolute path below /dev")
    if configured.parts[:2] != ("/", "dev"):
        raise ValueError("--device must be an absolute path below /dev")
    if not configured.exists():
        raise ValueError("explicit camera device does not exist")
    resolved = configured.resolve(strict=True)
    if resolved.parts[:2] != ("/", "dev"):
        raise ValueError("explicit camera alias must resolve below /dev")
    if not stat.S_ISCHR(resolved.stat().st_mode):
        raise ValueError("explicit camera device must resolve to a character device")
    return configured, resolved


def _atomic_write(path: Path, payload: bytes) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_bytes(payload)
    os.replace(partial, target)


def capture_once(
    *,
    device: str,
    output_png: Path,
    width: int | None,
    height: int | None,
    warmup_frames: int,
) -> dict[str, Any]:
    configured, resolved = validate_explicit_device(device)
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required") from exc

    capture = cv2.VideoCapture(str(configured), cv2.CAP_V4L2)
    opened = bool(capture.isOpened())
    if not opened:
        capture.release()
        raise RuntimeError("explicit UVC/V4L2 device could not be opened")
    try:
        if width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = None
        for _index in range(warmup_frames + 1):
            ok, candidate = capture.read()
            if not ok or candidate is None:
                raise RuntimeError("explicit UVC/V4L2 frame read failed")
            frame = candidate
    finally:
        capture.release()
    if frame is None or getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
        raise RuntimeError("captured frame is not BGR HxWx3")
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("captured frame PNG encoding failed")
    output = output_png.expanduser().resolve()
    _atomic_write(output, bytes(encoded))
    return {
        "schema": "rootscope.explicit-uvc-one-shot-receipt.v1",
        "status": "CAPTURED_ONE_EXPLICIT_DEVICE_FRAME_NOT_QUALIFICATION_EVIDENCE",
        "configured_device": str(configured),
        "resolved_device": str(resolved),
        "device_enumerated": False,
        "camera_opened": True,
        "frames_read": warmup_frames + 1,
        "warmup_frames": warmup_frames,
        "frame_width": int(frame.shape[1]),
        "frame_height": int(frame.shape[0]),
        "output_png": str(output),
        "output_png_bytes": output.stat().st_size,
        "output_png_sha256": sha256_file(output),
        "captured_monotonic": time.monotonic(),
        "hardware_touched": True,
        "network_touched": False,
        "pump_command": False,
        "serial_write": False,
        "state_machine_write": False,
        "service_started": False,
        "irrigation_execution": False,
        "execution_authority": False,
        "physical_authority": False,
        "physical_completion": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="one explicit /dev/... V4L2 path")
    parser.add_argument("--capture-png", required=True, type=Path)
    parser.add_argument("--capture-receipt", required=True, type=Path)
    parser.add_argument("--dual-path-output", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--capsule-config", required=True, type=Path)
    parser.add_argument("--thresholds-json", required=True, type=Path)
    parser.add_argument("--matcher-config-json", required=True, type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--warmup-frames", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in ("width", "height"):
        value = getattr(args, name)
        if value is not None and not 64 <= value <= 8192:
            raise SystemExit(f"[fatal] --{name} must be in [64,8192]")
    if not 0 <= args.warmup_frames <= 10:
        raise SystemExit("[fatal] --warmup-frames must be in [0,10]")
    camera_open_attempted = False
    try:
        camera_open_attempted = True
        capture_receipt = capture_once(
            device=args.device,
            output_png=args.capture_png,
            width=args.width,
            height=args.height,
            warmup_frames=args.warmup_frames,
        )
        _atomic_write(
            args.capture_receipt,
            (json.dumps(capture_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        command = [
            sys.executable,
            "-m",
            "app.vision.dual_path_demo",
            "--query",
            str(args.capture_png.resolve()),
            "--registry",
            str(args.registry.resolve()),
            "--capsule-config",
            str(args.capsule_config.resolve()),
            "--thresholds-json",
            str(args.thresholds_json.resolve()),
            "--matcher-config-json",
            str(args.matcher_config_json.resolve()),
            "--output-json",
            str(args.dual_path_output.resolve()),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode not in {0, 2}:
            return 1
        return result.returncode
    except Exception as exc:
        error = {
            "schema": "rootscope.explicit-uvc-one-shot-error.v1",
            "status": "FAIL_CLOSED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "device_enumerated": False,
            "camera_open_attempted": camera_open_attempted,
            "hardware_touched": camera_open_attempted,
            "network_touched": False,
            "pump_command": False,
            "serial_write": False,
            "state_machine_write": False,
            "service_started": False,
            "irrigation_execution": False,
            "execution_authority": False,
            "physical_authority": False,
            "physical_completion": False,
        }
        print(json.dumps(error, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
