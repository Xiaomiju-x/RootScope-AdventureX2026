from __future__ import annotations

import unittest
from pathlib import Path

from app.omega_runtime.evaluate_algorithms import evaluate_algorithms


ROOT = Path(__file__).resolve().parents[1]


class OmegaAlgorithmEvaluationTests(unittest.TestCase):
    def test_combined_evaluation_passes_without_physical_authority(self) -> None:
        report = evaluate_algorithms(
            dr_mpc_path=ROOT / "configs" / "omega" / "dr_mpc_scenarios.v1.json",
            locked_cases_path=ROOT
            / "configs"
            / "omega"
            / "locked_replay_cases.v1.json",
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["dr_mpc"]["matched_count"], 4)
        self.assertEqual(report["fault_injection"]["unsafe_accept_count"], 0)
        self.assertFalse(report["authority"]["execution_authority"])
        self.assertEqual(report["runtime_boundary"]["physical_command_count"], 0)
        self.assertEqual(len(report["report_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
