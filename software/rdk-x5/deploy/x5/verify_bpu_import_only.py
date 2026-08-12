#!/usr/bin/env python3
"""Verify the X5 BPU Python import without loading or forwarding a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(partial, 0o600)
    os.replace(partial, path)


def distribution_version(module: Any) -> str | None:
    for name in ("hobot-dnn", "hobot_dnn", "hbm-runtime"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import hobot_dnn
    from hobot_dnn import pyeasy_dnn  # noqa: F401

    model = args.vendor_model
    exists = model.exists()
    readable = model.is_file() and os.access(model, os.R_OK)
    resolved = model.resolve(strict=True) if readable else None
    receipt: dict[str, Any] = {
        "schema": "rootscope.x5-bpu-import-only.v1",
        "status": "PASS_IMPORT_ONLY_NO_MODEL_LOAD_NO_FORWARD",
        "board_hostname": Path("/etc/hostname").read_text(encoding="utf-8").strip(),
        "machine_id": Path("/etc/machine-id").read_text(encoding="utf-8").strip(),
        "device_tree_serial": Path("/proc/device-tree/serial-number").read_bytes()
        .rstrip(b"\0").decode("ascii"),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "runtime": {
            "hobot_dnn_version": distribution_version(hobot_dnn),
            "hobot_dnn_origin": str(Path(hobot_dnn.__file__).resolve()),
            "pyeasy_dnn_imported": True,
        },
        "vendor_reference_model": {
            "requested_path": str(model),
            "exists": exists,
            "readable_regular_file": readable,
            "resolved_path": str(resolved) if resolved is not None else None,
            "size_bytes": resolved.stat().st_size if resolved is not None else None,
            "sha256": sha256_file(resolved) if resolved is not None else None,
            "role": "FUTURE_AUXILIARY_SEMANTIC_OOD_PROBE_ONLY_NOT_PLANT_CLASSIFIER",
            "qualified_for_rootscope": False,
        },
        "device_enumerated": False,
        "model_loaded": False,
        "bpu_forward_executed": False,
        "camera_opened": False,
        "service_started": False,
        "network_touched": False,
        "execution_authority": False,
        "physical_authority": False,
    }
    if not readable:
        receipt["status"] = "PASS_IMPORT_ONLY_VENDOR_REFERENCE_ABSENT"
    write_atomic(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
