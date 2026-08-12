#!/usr/bin/env python3
"""Build a content-addressed X5 sidecar for the PB6 pump-only STM32 profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ADVENTUREX_ROOT = ROOT.parent
OUTPUT_ROOT = ADVENTUREX_ROOT / "output" / "releases"

FILES = (
    "app/hardware/__init__.py",
    "app/hardware/device_identity.py",
    "app/hardware/physical_serial.py",
    "app/hardware/serial_writer.py",
    "app/serial/__init__.py",
    "app/serial/fake_f407.py",
    "app/serial/frame.py",
    "app/serial/link.py",
    "tools/stm32_no_pump_dry_run.py",
    "tools/stm32_pb6_first_pulse.py",
    "deploy/x5/99-rootscope-stm32.rules",
    "deploy/x5/rootscope-stm32-no-pump-dry-run",
    "deploy/x5/rootscope-stm32-pb6-first-pulse",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    records: list[dict[str, Any]] = []
    root_digest = hashlib.sha256()
    for relative in FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = sha256(source)
        size = source.stat().st_size
        records.append({"path": relative, "sha256": digest, "bytes": size})
        root_digest.update(relative.encode("utf-8"))
        root_digest.update(b"\0")
        root_digest.update(digest.encode("ascii"))
        root_digest.update(b"\0")
        root_digest.update(str(size).encode("ascii"))
        root_digest.update(b"\n")

    content_root = root_digest.hexdigest()
    version = f"stm32-f103-pb6-v13-{content_root[:12]}"
    target = OUTPUT_ROOT / version
    if target.exists():
        raise FileExistsError(f"refusing to overwrite release: {target}")
    target.mkdir(parents=True)

    for record in records:
        relative = record["path"]
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    manifest = {
        "schema": "rootscope.stm32-sidecar-manifest.v2",
        "version": version,
        "content_root_sha256": content_root,
        "firmware_expectation": {
            "protocol_version": 1,
            "build_id": 2026072513,
            "hw_variant": 2,
            "build_tag": "rs-f103-pb6-v13",
            "required_capabilities_hex": "0x00000079",
            "pump_gpio": "PB6",
            "relay_profile": "ACTIVE_LOW_OPEN_DRAIN",
            "motion_profile": "EXTENSION_LOCKED",
        },
        "device_identity_sha256": (
            "4fd1776962e4dfbf50cac6c41406fb4d0101a08d1f70e09e376ef37c76be19de"
        ),
        "files": records,
        "authority": {
            "auto_start": False,
            "automatic_unlock": False,
            "unbounded_pump_on": False,
            "arm_timed_task_available_in_dry_run": False,
            "physical_action_requires_explicit_cli_confirmation": True,
            "physical_completion": False,
        },
    }
    manifest_path = target / "stm32_sidecar_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    archive = OUTPUT_ROOT / f"{version}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(target, arcname=version)
    archive_digest = sha256(archive)
    receipt = {
        "version": version,
        "release_directory": str(target),
        "manifest": str(manifest_path),
        "content_root_sha256": content_root,
        "archive": str(archive),
        "archive_sha256": archive_digest,
    }
    receipt_path = OUTPUT_ROOT / f"{version}.build_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
