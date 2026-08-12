#!/usr/bin/env python3
"""Zero-device regression test for the auto-irrigation sequence ledger."""

from pathlib import Path
import tempfile

from tools.x5_visual_irrigation_cycle import SequenceLedger


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "ledger.json"
        ledger = SequenceLedger.load_or_create(path, "a" * 64)
        ledger.value = 30000
        ledger.persist("TEST_BASELINE")
        first = ledger.reserve_next()
        recovered = ledger.reserve_resync()
        assert (first, recovered) == (30001, 62768)
    print(
        "BOUNDARY_SAFE_RESYNC_PASS "
        "first=30001 recovered=62768 devices_opened=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
