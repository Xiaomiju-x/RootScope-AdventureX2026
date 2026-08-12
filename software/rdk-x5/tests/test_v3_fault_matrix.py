from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def test_v3_pc_fault_matrix() -> None:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "faults.json"
        result = subprocess.run(
            [
                sys.executable,
                str(root / "tools" / "run_v3_pc_fault_matrix.py"),
                "--output",
                str(output),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "PASS"
        assert report["count"] == 19
        assert report["passed"] == 19
        assert report["unsafe_accepts"] == 0
        assert report["hardware_touched"] is False
        assert report["serial_opened"] is False
        assert report["pump_touched"] is False
