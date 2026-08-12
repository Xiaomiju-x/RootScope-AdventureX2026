"""One-shot launcher for the operator-authorized 5 s PB6 diagnostic pulse.

This reuses the commissioned V13 pulse implementation, including firmware and
USB identity checks, sequence/task ledgers, initial/final E-stop, and the
no-retry rule.  It deliberately exposes no duration argument.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

from tools import stm32_pb6_first_pulse as pulse


def main() -> int:
    if (pulse.DURATION_MS, pulse.HARD_TIMEOUT_MS) != (100, 500):
        raise RuntimeError("commissioned pulse baseline changed; diagnostic refused")

    pulse.DURATION_MS = 5_000
    pulse.HARD_TIMEOUT_MS = 7_000

    state_root = Path.home() / ".local/state/rootscope"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        state_root
        / "evidence"
        / f"stm32_pb6_diagnostic_5000ms_{timestamp}.json"
    )
    sys.argv = [
        "stm32_pb6_diagnostic_5000ms_launcher.py",
        "--sequence-state",
        str(state_root / "stm32_sequence.json"),
        "--task-state",
        str(state_root / "stm32_task.json"),
        "--output",
        str(output),
        "--confirm-physical-pb6-pulse",
    ]
    return pulse.main()


if __name__ == "__main__":
    raise SystemExit(main())
