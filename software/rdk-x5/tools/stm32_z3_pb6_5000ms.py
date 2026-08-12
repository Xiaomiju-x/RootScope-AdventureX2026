#!/usr/bin/env python3
"""Execute one V15-identity-bound, firmware-timed 5 s PB6 pump task.

This wrapper reuses the commissioned no-retry pump implementation while
selecting the flashed Z3+PB6 V15 identity and the currently commissioned USB
path.  It is not a free-form duration interface.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from app.serial.link import F103_Z3_PB6_IDENTITY_EXPECTATION
from tools import stm32_pb6_first_pulse as pulse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-state", type=Path, required=True)
    parser.add_argument("--task-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-alias", default="/dev/rootscope_stm32")
    parser.add_argument(
        "--id-path",
        default=os.environ.get(
            "ROOTSCOPE_SERIAL_ID_PATH", "commission-with-udevadm"
        ),
    )
    parser.add_argument("--confirm-physical-pb6-pulse", action="store_true")
    args = parser.parse_args()
    if not args.confirm_physical_pb6_pulse:
        raise SystemExit(
            "REFUSED: --confirm-physical-pb6-pulse is required; "
            "serial was not opened"
        )

    pulse.DURATION_MS = 5_000
    pulse.HARD_TIMEOUT_MS = 7_000
    pulse.DEVICE_ALIAS = args.device_alias
    pulse.DEVICE_ID_PATH = args.id_path
    pulse.IDENTITY_EXPECTATION = F103_Z3_PB6_IDENTITY_EXPECTATION
    sys.argv = [
        "stm32_z3_pb6_5000ms.py",
        "--sequence-state",
        str(args.sequence_state),
        "--task-state",
        str(args.task_state),
        "--output",
        str(args.output),
        "--confirm-physical-pb6-pulse",
    ]
    return pulse.main()


if __name__ == "__main__":
    raise SystemExit(main())
