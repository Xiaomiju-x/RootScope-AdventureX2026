from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.omega_runtime.fault_injection import run_fault_injection


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "configs" / "omega" / "locked_replay_cases.v1.json"


class OmegaFaultInjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = json.loads(CASES.read_text(encoding="utf-8"))
        cls.normal = payload["cases"][0]["inputs"]

    def test_fifteen_mutations_have_zero_unsafe_accepts(self) -> None:
        report = run_fault_injection(self.normal)
        self.assertEqual(report["fault_count"], 15)
        self.assertEqual(report["unsafe_accept_count"], 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["physical_command_count"], 0)
        self.assertFalse(report["hardware_touched"])
        self.assertEqual(len(report["report_sha256"]), 64)

    def test_report_is_deterministic_and_covers_contract_rejection(self) -> None:
        first = run_fault_injection(self.normal)
        second = run_fault_injection(self.normal)
        self.assertEqual(first["report_sha256"], second["report_sha256"])
        outcomes = {item["outcome"] for item in first["results"]}
        self.assertIn("HOLD", outcomes)
        self.assertIn("REJECT", outcomes)
        self.assertIn("CONTRACT_REJECTED", outcomes)


if __name__ == "__main__":
    unittest.main()
