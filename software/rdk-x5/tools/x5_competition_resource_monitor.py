#!/usr/bin/env python3
"""Observe RootScope competition resource watermarks without opening hardware.

The monitor reads procfs/sysfs and, when present, asks ``fuser`` who owns the
single frozen UVC character device.  It never opens the camera, serial, GPIO,
network configuration, a model, or an actuator.  The output path must be new.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Sequence


EXPECTED_IDENTITY = {
    "hostname": "rootscope-x5",
    "serial": "3281556110220e0c002bdeab0012004",
    "machine_id": "<redacted-device-boot-id>",
    "wlan_mac": "02:00:00:00:00:01",
}
CAMERA = Path(
    "/dev/v4l/by-id/"
    "usb-Web_Camera_Web_Camera_202604081837-video-index0"
)
ZERO_AUTHORITY = {
    "serial_open": False,
    "serial_write": False,
    "gpio_access": False,
    "pump_command": False,
    "state_machine_write": False,
    "execution_authority": False,
    "physical_authority": False,
    "irrigation_execution": False,
    "physical_completion": False,
    "network_configuration_write": False,
}
PROCESS_MARKERS = (
    "llama-server",
    "bpu_shadow_worker",
    "x5_competition_live_vision_v2.py",
    "x5_competition_static_cpu_bpu_replay.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_identity() -> dict[str, str]:
    serial_path = Path("/proc/device-tree/serial-number")
    if not serial_path.exists():
        serial_path = Path("/sys/firmware/devicetree/base/serial-number")
    return {
        "hostname": platform.node(),
        "serial": serial_path.read_bytes().replace(b"\x00", b"").decode("ascii"),
        "machine_id": Path("/etc/machine-id")
        .read_text(encoding="ascii")
        .strip(),
        "wlan_mac": Path("/sys/class/net/wlan0/address")
        .read_text(encoding="ascii")
        .strip(),
    }


def read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.replace(":", "").split()
        if len(fields) >= 2:
            try:
                values[fields[0]] = int(fields[1])
            except ValueError:
                continue
    return values


def temperature_millicelsius() -> int | None:
    values: list[int] = []
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            value = int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        if -40_000 <= value <= 150_000:
            values.append(value)
    return max(values) if values else None


def camera_owner() -> str:
    if not CAMERA.exists():
        return ""
    device = CAMERA.resolve(strict=True)
    result = subprocess.run(
        ["fuser", str(device)],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=3,
    )
    return " ".join((result.stdout + " " + result.stderr).split())


def process_rss_kib() -> dict[str, int]:
    totals = {marker: 0 for marker in PROCESS_MARKERS}
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            )
            status = read_key_values(entry / "status")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        for marker in PROCESS_MARKERS:
            if marker in command:
                totals[marker] += status.get("VmRSS", 0)
    return totals


def sample() -> dict[str, Any]:
    memory = read_key_values(Path("/proc/meminfo"))
    vmstat = read_key_values(Path("/proc/vmstat"))
    return {
        "timestamp_utc": utc_now(),
        "monotonic_seconds": time.monotonic(),
        "mem_available_kib": memory.get("MemAvailable"),
        "cma_free_kib": memory.get("CmaFree"),
        "swap_free_kib": memory.get("SwapFree"),
        "temperature_millicelsius": temperature_millicelsius(),
        "oom_kill_count": vmstat.get("oom_kill", 0),
        "camera_owner_observed": camera_owner(),
        "process_rss_kib": process_rss_kib(),
    }


def monitor(
    *,
    duration_seconds: float,
    interval_seconds: float,
) -> dict[str, Any]:
    identity = read_identity()
    if identity != EXPECTED_IDENTITY or platform.machine() != "aarch64":
        raise RuntimeError(
            f"X5 identity mismatch: {identity}, arch={platform.machine()}"
        )
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    while True:
        samples.append(sample())
        elapsed = time.monotonic() - started
        if elapsed >= duration_seconds:
            break
        time.sleep(min(interval_seconds, max(0.0, duration_seconds - elapsed)))
    oom_values = [int(item["oom_kill_count"]) for item in samples]
    mem_values = [int(item["mem_available_kib"]) for item in samples]
    cma_values = [int(item["cma_free_kib"]) for item in samples]
    temperatures = [
        int(item["temperature_millicelsius"])
        for item in samples
        if item["temperature_millicelsius"] is not None
    ]
    return {
        "schema": "rootscope.competition-resource-soak.v2",
        "status": "OBSERVED_ZERO_AUTHORITY",
        "started_at_utc": samples[0]["timestamp_utc"],
        "completed_at_utc": samples[-1]["timestamp_utc"],
        "duration_requested_seconds": duration_seconds,
        "interval_seconds": interval_seconds,
        "identity": identity,
        "architecture": platform.machine(),
        "boot_id": boot_id,
        "sample_count": len(samples),
        "samples": samples,
        "thresholds": {
            "mem_available_min_kib": 512 * 1024,
            "cma_free_min_kib": 128 * 1024,
            "oom_kill_delta_max": 0,
        },
        "observed": {
            "mem_available_min_kib": min(mem_values),
            "cma_free_min_kib": min(cma_values),
            "temperature_max_millicelsius": (
                max(temperatures) if temperatures else None
            ),
            "oom_kill_before": oom_values[0],
            "oom_kill_after": oom_values[-1],
            "oom_kill_delta": oom_values[-1] - oom_values[0],
        },
        "gates": {
            "mem_available_pass": min(mem_values) >= 512 * 1024,
            "cma_free_pass": min(cma_values) >= 128 * 1024,
            "no_new_oom_kill_pass": oom_values[-1] == oom_values[0],
            "same_boot_pass": (
                Path("/proc/sys/kernel/random/boot_id")
                .read_text(encoding="ascii")
                .strip()
                == boot_id
            ),
        },
        "camera_opened_by_monitor": False,
        "serial_gpio_pump_touched": False,
        "authority": dict(ZERO_AUTHORITY),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.duration_seconds <= 3600:
        raise ValueError("--duration-seconds must be within 0..3600")
    if not 0.5 <= args.interval_seconds <= 60:
        raise ValueError("--interval-seconds must be within 0.5..60")
    output = args.output.expanduser()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = monitor(
        duration_seconds=args.duration_seconds,
        interval_seconds=args.interval_seconds,
    )
    partial = output.with_name(output.name + f".{os.getpid()}.partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output)
    print(
        json.dumps(
            {
                "status": "PASS"
                if all(payload["gates"].values())
                else "FAIL_RESOURCE_GATE",
                "output": str(output.resolve()),
                "sha256": sha256_file(output),
                "sample_count": payload["sample_count"],
                "observed": payload["observed"],
                "gates": payload["gates"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if all(payload["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
