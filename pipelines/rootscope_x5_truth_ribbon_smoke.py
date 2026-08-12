#!/usr/bin/env python3
"""Run and normally close one candidate Truth Ribbon loopback server."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import urllib.request

from app.omega_runtime.omega_server import build_omega_server


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    project = args.project_root.resolve(strict=True)
    output = args.output_dir.resolve(strict=True)
    health_path = output / "04b_truth_ribbon.health.json"
    status_path = output / "04b_truth_ribbon.status.json"
    receipt_path = output / "04b_truth_ribbon.receipt.json"
    for path in (health_path, status_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"refusing overwrite: {path}")
    if args.port != 8765 or _port_open(args.port):
        raise RuntimeError("Truth Ribbon loopback port is invalid or already open")

    cases = project / "configs/omega/locked_replay_cases.v1.json"
    profiles = project / "configs/omega/edge_profiles.v1.json"
    corpus = project / "configs/omega/field_knowledge.v1.md"
    server = build_omega_server(
        host="127.0.0.1",
        port=args.port,
        cases_path=cases,
        profiles_path=profiles,
        corpus_path=corpus,
    )
    started = False
    try:
        server.start()
        started = True
        with urllib.request.urlopen(
            f"http://127.0.0.1:{args.port}/api/health", timeout=10
        ) as response:
            health = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(
            f"http://127.0.0.1:{args.port}/api/status", timeout=10
        ) as response:
            status = json.loads(response.read().decode("utf-8"))
        if health.get("ok") is not True:
            raise RuntimeError("health response did not pass")
        if not isinstance(status, dict) or status.get("schema_version") is None:
            raise RuntimeError("status response is not the locked replay snapshot")
        _write_exclusive(health_path, health)
        _write_exclusive(status_path, status)
    finally:
        if started:
            server.close()

    if _port_open(args.port):
        raise RuntimeError("Truth Ribbon port remained open after normal close")
    receipt = {
        "schema": "rootscope.omega-v3-x5-truth-ribbon-smoke.v1",
        "status": "PASS_LOOPBACK_HEALTH_STATUS_NORMAL_CLOSE",
        "bind_host": "127.0.0.1",
        "port": args.port,
        "health_sha256": _sha256(health_path),
        "status_sha256": _sha256(status_path),
        "normal_close_called": True,
        "port_closed_after_stop": True,
        "external_network_access": False,
        "service_started": False,
        "systemd_invoked": False,
        "camera_opened": False,
        "serial_opened": False,
        "gpio_access": False,
        "pump_command": False,
        "execution_authority": False,
        "physical_authority": False,
        "physical_closure": False,
    }
    _write_exclusive(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
