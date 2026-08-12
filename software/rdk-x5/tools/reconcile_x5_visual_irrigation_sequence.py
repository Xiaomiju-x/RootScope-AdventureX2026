#!/usr/bin/env python3
"""Reconcile the auto-irrigation ledger to a separately audited read-only query.

This utility never opens a device.  It is intentionally narrow: it accepts the
known rejected local reservation 62769 and records the independently observed
accepted V15 QUERY_FIRMWARE sequence 30001.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile


EXPECTED_REJECTED_LOCAL_RESERVATION = 62769
AUDITED_ACCEPTED_FIRMWARE_SEQUENCE = 30001
EXPECTED_IDENTITY_SHA256 = (
    "16786c3f382dcbcac7ade731aa3c2ae7b25812efe3bd3fd9af1a07d62281c622"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def main() -> int:
    path = (
        Path.home()
        / ".local/state/rootscope-auto-irrigation/stm32_v15_sequence.json"
    )
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("device_identity_sha256") != EXPECTED_IDENTITY_SHA256:
        raise SystemExit("REFUSED: sequence ledger device identity mismatch")
    if (
        payload.get("last_reserved_sequence")
        != EXPECTED_REJECTED_LOCAL_RESERVATION
    ):
        raise SystemExit("REFUSED: sequence ledger is not at the audited rejection")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.before-readonly-reconcile-{stamp}")
    shutil.copy2(path, backup)
    os.chmod(backup, 0o600)

    payload.update(
        {
            "updated_at_utc": utc_now(),
            "firmware_boot_id_token": "boot-<redacted-device-boot-id>",
            "last_reserved_sequence": AUDITED_ACCEPTED_FIRMWARE_SEQUENCE,
            "reservation_semantics": (
                "Reconciled to independently audited read-only QUERY_FIRMWARE "
                "sequence 30001 after reservation 62769 was rejected at the "
                "V15 modular half-range boundary; no device was opened."
            ),
        }
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        json.dumps(
            {
                "schema": "rootscope.sequence_ledger_reconciliation.v1",
                "updated_at_utc": payload["updated_at_utc"],
                "path": str(path),
                "backup": str(backup),
                "before": EXPECTED_REJECTED_LOCAL_RESERVATION,
                "after": AUDITED_ACCEPTED_FIRMWARE_SEQUENCE,
                "firmware_boot_id_token": payload["firmware_boot_id_token"],
                "device_opened": False,
                "physical_command_sent": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
