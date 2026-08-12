#!/usr/bin/env python3
"""Read-only V15 identity and fail-closed state probe.

The probe never sends heartbeat, clear, home, depth, pump, or stop commands.
It is safe to use only when the controller is expected to be boot-locked.
"""

from __future__ import annotations

import argparse
import json

from stm32_z3_level1_first_descent import (
    DEVICE_DEFAULT,
    SerialSession,
    firmware_query,
    parse_csv_fields,
    utc_now,
)


EXPECTED_VERSION = "2026-07-25-RS-F103-Z3-PB6-V15"
EXPECTED_BUILD_ID = 2026072515
EXPECTED_CAPABILITIES = 0x00000079
EXPECTED_VARIANT = 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE_DEFAULT)
    parser.add_argument("--sequence", type=int, required=True)
    args = parser.parse_args()

    if not 1 <= args.sequence <= 65535:
        raise SystemExit("REFUSED: --sequence must be in 1..65535")

    with SerialSession(args.device) as session:
        version = session.query_ascii("VERSION", "VERSION,")
        status = session.query_ascii("STATUS", "STATUS,")
        io_status = session.query_ascii("IOSTATUS", "IOSTATUS,")
        firmware = firmware_query(session, args.sequence)

    status_fields = parse_csv_fields(status)
    io_fields = parse_csv_fields(io_status)
    safe_locked = (
        status_fields.get("Z") == "0"
        and status_fields.get("P") == "0"
        and status_fields.get("LOCK") == "1"
        and status_fields.get("TASK") == "0"
        and io_fields.get("Z") == "0x0"
        and io_fields.get("PB6") == "1"
        and io_fields.get("PLOG") == "0"
    )
    identity_ok = (
        EXPECTED_VERSION in version
        and firmware["build_id"] == EXPECTED_BUILD_ID
        and firmware["capabilities"] == EXPECTED_CAPABILITIES
        and firmware["hardware_variant"] == EXPECTED_VARIANT
    )
    result = {
        "schema": "rootscope.stm32_z3_v15_readonly_probe.v1",
        "observed_at_utc": utc_now(),
        "device": args.device,
        "sequence": args.sequence,
        "commands": ["VERSION", "STATUS", "IOSTATUS", "QUERY_FIRMWARE"],
        "forbidden_commands_sent": False,
        "identity_ok": identity_ok,
        "safe_locked": safe_locked,
        "version": version,
        "status": status,
        "io_status": io_status,
        "firmware": firmware,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if identity_ok and safe_locked else 2


if __name__ == "__main__":
    raise SystemExit(main())
