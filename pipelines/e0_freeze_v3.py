#!/usr/bin/env python3
"""Freeze the pre-v3 RootScope asset boundary without touching XRD assets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ADVENTUREX = Path(__file__).resolve().parents[1]
ROOTSCOPE = ADVENTUREX / "rootscope"
OUTPUT = ROOTSCOPE / "evidence" / "e0_v3_20260723"
CAPTURED_AT = datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": path.relative_to(ADVENTUREX).as_posix(),
        "bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).astimezone().isoformat(),
        "sha256": sha256(path),
    }


def source_files() -> list[Path]:
    roots = [
        ROOTSCOPE / "app",
        ROOTSCOPE / "configs",
        ROOTSCOPE / "deploy",
        ROOTSCOPE / "tests",
        ROOTSCOPE / "training",
    ]
    fixed = [
        ROOTSCOPE / "README.md",
        ROOTSCOPE / "PREEXISTING.md",
        ROOTSCOPE / "BUILT_DURING_EVENT.md",
        ROOTSCOPE / "BLOCKERS.md",
        ROOTSCOPE / "H12_IMPLEMENTATION_STATUS.md",
        ROOTSCOPE / "pyproject.toml",
        ADVENTUREX / "AdventureX_RootScope_固定式根区灌溉舱_最终方案.md",
    ]
    selected: set[Path] = {path for path in fixed if path.is_file()}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            parts = set(path.parts)
            if "__pycache__" in parts or ".pytest_cache" in parts:
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            selected.add(path)
    return sorted(selected, key=lambda item: item.as_posix())


def release_files() -> list[Path]:
    release_root = ADVENTUREX / "output" / "releases"
    if not release_root.exists():
        return []
    selected: list[Path] = []
    for path in release_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".tar", ".json", ".sha256"} or path.name == "SHA256SUMS":
            selected.append(path)
    return sorted(selected, key=lambda item: item.as_posix())


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if (ROOTSCOPE / "app" / "omega").exists():
        raise SystemExit(
            "Refusing E0 freeze: rootscope/app/omega already exists; baseline is no longer clean."
        )

    OUTPUT.mkdir(parents=True, exist_ok=False)
    sources = [record(path) for path in source_files()]
    releases = [record(path) for path in release_files()]
    board_identity = {
        "captured_before_write": True,
        "capture_mode": "read_only_ssh_inspection",
        "target_ip_at_capture": "192.0.2.42",
        "username": "rootscope",
        "hostname": "ubuntu",
        "os": "Ubuntu 22.04.5 LTS",
        "architecture": "aarch64",
        "device_tree_model": "D-Robotics RDK X5 V1.0",
        "wlan_mac": "02:00:00:00:00:01",
        "device_tree_serial": "3281556110258c1902ab5d9b0012004",
        "machine_id": "<redacted-device-boot-id>",
        "ssh_ed25519_fingerprint": (
            "SHA256:K8oyEoddaZ4keNu39exAqMJLDEcncgoQGcRPQ898Bb4"
        ),
        "identity_warning": (
            "The factory SSH host key duplicates a frozen XRD board key; "
            "host key alone is not a sufficient identity proof."
        ),
    }
    write_json(
        OUTPUT / "pre_v3_source_manifest.json",
        {
            "schema_version": "rootscope.e0.source-manifest.v1",
            "captured_at": CAPTURED_AT,
            "scope": "adventurex-only",
            "xrd_runtime_imported": False,
            "file_count": len(sources),
            "files": sources,
        },
    )
    write_json(
        OUTPUT / "immutable_release_manifest.json",
        {
            "schema_version": "rootscope.e0.release-manifest.v1",
            "captured_at": CAPTURED_AT,
            "release_file_count": len(releases),
            "files": releases,
        },
    )
    write_json(OUTPUT / "new_x5_identity_baseline.json", board_identity)

    source_root = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    release_root = hashlib.sha256(
        json.dumps(releases, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    receipt = {
        "schema_version": "rootscope.e0.v3-receipt.v1",
        "captured_at": CAPTURED_AT,
        "status": "E0_COMPLETE_PRE_V3",
        "source_manifest_sha256": sha256(OUTPUT / "pre_v3_source_manifest.json"),
        "release_manifest_sha256": sha256(OUTPUT / "immutable_release_manifest.json"),
        "board_identity_sha256": sha256(OUTPUT / "new_x5_identity_baseline.json"),
        "source_composition_root": source_root,
        "release_composition_root": release_root,
        "constraints": {
            "xrd_project_read_only_reference": True,
            "v1_v2_releases_immutable": True,
            "physical_execution_authority": False,
            "camera_opened": False,
            "serial_opened": False,
            "pump_touched": False,
            "pc_network_touched": False,
        },
    }
    write_json(OUTPUT / "e0_v3_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
