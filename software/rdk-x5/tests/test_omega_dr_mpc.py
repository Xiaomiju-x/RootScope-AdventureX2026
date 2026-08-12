from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.omega_runtime.dr_mpc import DrMpcScenario, solve_dr_mpc


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "configs" / "omega" / "dr_mpc_scenarios.v1.json"


class OmegaDrMpcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.items = json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"]

    def test_all_scenarios_match_locked_status_and_zero_authority(self) -> None:
        for item in self.items:
            expected = item["expected_status"]
            values = {key: value for key, value in item.items() if key != "expected_status"}
            proposal = solve_dr_mpc(DrMpcScenario.from_mapping(values))
            with self.subTest(scenario=item["scenario_id"]):
                self.assertEqual(proposal.status, expected)
                self.assertFalse(proposal.authority.execution_authority)
                self.assertFalse(proposal.to_dict()["physical_command_emitted"])
                self.assertEqual(len(proposal.proposal_sha256), 64)
                if proposal.status == "PROPOSAL":
                    self.assertGreater(proposal.total_dose_mg, 0)
                else:
                    self.assertEqual(proposal.total_dose_mg, 0)

    def test_solution_is_deterministic(self) -> None:
        values = {
            key: value
            for key, value in self.items[0].items()
            if key != "expected_status"
        }
        scenario = DrMpcScenario.from_mapping(values)
        self.assertEqual(solve_dr_mpc(scenario), solve_dr_mpc(scenario))

    def test_nonfinite_and_unknown_fields_fail_closed(self) -> None:
        values = {
            key: value
            for key, value in self.items[0].items()
            if key != "expected_status"
        }
        values["moisture_now"] = [float("nan"), 0.1, 0.1]
        with self.assertRaises(ValueError):
            DrMpcScenario.from_mapping(values)
        values = {
            key: value
            for key, value in self.items[0].items()
            if key != "expected_status"
        }
        values["serial_port"] = "/dev/ttyUSB0"
        with self.assertRaises(ValueError):
            DrMpcScenario.from_mapping(values)


if __name__ == "__main__":
    unittest.main()
