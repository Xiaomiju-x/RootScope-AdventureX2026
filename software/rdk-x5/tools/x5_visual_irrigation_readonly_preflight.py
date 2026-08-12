#!/usr/bin/env python3
"""Read-only preflight using the exact auto-irrigation sequence ledger."""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.hardware.device_identity import UsbDeviceIdentity
from tools.x5_visual_irrigation_cycle import (
    EXPECTED_BUILD_ID,
    EXPECTED_CAPABILITIES,
    EXPECTED_VARIANT,
    EXPECTED_VERSION,
    SequenceLedger,
    SerialSession,
    query_firmware_with_safe_resync,
    utc_now,
    verify_safe_locked_state,
    verify_usb_identity,
)


def main() -> int:
    identity = UsbDeviceIdentity(
        alias="/dev/rootscope_stm32",
        vid="1a86",
        pid="7523",
        id_path=os.environ.get(
            "ROOTSCOPE_SERIAL_ID_PATH", "commission-with-udevadm"
        ),
        interface_number="00",
    )
    usb = verify_usb_identity(identity)
    ledger_path = (
        Path.home()
        / ".local/state/rootscope-auto-irrigation/stm32_v15_sequence.json"
    )
    ledger = SequenceLedger.load_or_create(ledger_path, identity.identity_sha256)
    with SerialSession(identity.alias) as session:
        version = session.query_ascii("VERSION", "VERSION,")
        status = session.query_ascii("STATUS", "STATUS,")
        io_status = session.query_ascii("IOSTATUS", "IOSTATUS,")
        verify_safe_locked_state(status, io_status)
        firmware, sequence = query_firmware_with_safe_resync(session, ledger)

    identity_ok = (
        EXPECTED_VERSION in version
        and firmware["build_id"] == EXPECTED_BUILD_ID
        and firmware["capabilities"] == EXPECTED_CAPABILITIES
        and firmware["hardware_variant"] == EXPECTED_VARIANT
    )
    result = {
        "schema": "rootscope.visual_irrigation.readonly_preflight.v1",
        "observed_at_utc": utc_now(),
        "commands": ["VERSION", "STATUS", "IOSTATUS", "QUERY_FIRMWARE"],
        "forbidden_commands_sent": False,
        "physical_action_authority": False,
        "usb_identity": usb,
        "sequence": sequence,
        "identity_ok": identity_ok,
        "safe_locked": True,
        "version": version,
        "status": status,
        "io_status": io_status,
        "firmware": firmware,
        "ledger_path": str(ledger_path),
        "ledger_last_reserved_sequence": ledger.value,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if identity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
