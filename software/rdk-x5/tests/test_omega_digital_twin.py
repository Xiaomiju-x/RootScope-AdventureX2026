from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.omega_runtime.digital_twin import TwinCaseInput, evaluate_case


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "configs" / "omega" / "locked_replay_cases.v1.json"


class OmegaDigitalTwinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(CASES.read_text(encoding="utf-8"))

    def test_all_five_locked_cases_match_frozen_projection(self) -> None:
        self.assertEqual(len(self.payload["cases"]), 5)
        for item in self.payload["cases"]:
            with self.subTest(case=item["case_id"]):
                case = TwinCaseInput.from_mapping(item["inputs"])
                actual = evaluate_case(case).projection.to_dict()
                expected = item["expected"]
                self.assertEqual(
                    actual["safety_decision"], expected["safety_decision"]
                )
                self.assertEqual(
                    actual["evidence_action"], expected["evidence_action"]
                )
                self.assertEqual(actual["terminal_state"], expected["terminal_state"])
                self.assertEqual(
                    actual["completion_claim"], expected["completion_claim"]
                )
                self.assertFalse(actual["physical_command_emitted"])
                self.assertTrue(actual["proposal_only"])

    def test_each_case_is_deterministic(self) -> None:
        for item in self.payload["cases"]:
            case = TwinCaseInput.from_mapping(item["inputs"])
            self.assertEqual(evaluate_case(case), evaluate_case(case))

    def test_unknown_input_field_fails_closed(self) -> None:
        values = dict(self.payload["cases"][0]["inputs"])
        values["serial_command"] = "PUMP_ON"
        with self.assertRaises(ValueError):
            TwinCaseInput.from_mapping(values)

    def test_mass_and_wetting_both_required(self) -> None:
        values = dict(self.payload["cases"][0]["inputs"])
        values["target_wetting_score"] = 0.1
        result = evaluate_case(TwinCaseInput.from_mapping(values))
        self.assertEqual(result.projection.safety_decision.value, "HOLD")
        self.assertEqual(result.projection.evidence_action.value, "WAIT")


if __name__ == "__main__":
    unittest.main()
