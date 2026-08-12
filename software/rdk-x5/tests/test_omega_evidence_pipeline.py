from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.omega_runtime.digital_twin import TwinCaseInput
from app.omega_runtime.evidence_pipeline import build_evidence_context


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "configs" / "omega" / "locked_replay_cases.v1.json"


class OmegaEvidencePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]

    def test_cases_stop_at_the_first_unsafe_stage(self) -> None:
        expected_nodes = {
            "CASE01_NORMAL_VERIFIED": 9,
            "CASE02_OOD_RECAPTURE": 5,
            "CASE03_ACK_WITHOUT_MASS_LOSS": 8,
            "CASE04_NEIGHBOR_SPILL": 9,
            "CASE05_STALE_TAMPER_ESTOP": 2,
        }
        for item in self.cases:
            with self.subTest(case=item["case_id"]):
                context = build_evidence_context(
                    item["case_id"], TwinCaseInput.from_mapping(item["inputs"])
                )
                self.assertEqual(len(context.dag), expected_nodes[item["case_id"]])
                self.assertEqual(
                    context.belief.revision, expected_nodes[item["case_id"]]
                )
                self.assertEqual(len(context.evidence_dag_root), 64)
                self.assertEqual(len(context.belief_state_hash), 64)
                self.assertEqual(len(context.failure_core_hash), 64)
                self.assertEqual(len(context.rb_voe_plan_hash), 64)
                self.assertEqual(context.rb_voe_plan.horizon, 2)
                self.assertFalse(context.belief.authority.execution_authority)

    def test_identical_case_has_identical_roots(self) -> None:
        item = self.cases[0]
        case = TwinCaseInput.from_mapping(item["inputs"])
        first = build_evidence_context(item["case_id"], case)
        second = build_evidence_context(item["case_id"], case)
        self.assertEqual(first.evidence_dag_root, second.evidence_dag_root)
        self.assertEqual(first.belief_state_hash, second.belief_state_hash)

    def test_h2_rb_voe_is_advisory_and_risk_bounded(self) -> None:
        expected = {
            "CASE01_NORMAL_VERIFIED": ("HOLD_CLEAR", "HOLD"),
            "CASE02_OOD_RECAPTURE": ("PLAN_H2", "REQUEST_OPERATOR_REVIEW"),
            "CASE03_ACK_WITHOUT_MASS_LOSS": ("PLAN_H2", "REWEIGH"),
            "CASE04_NEIGHBOR_SPILL": ("HOLD_BLOCKING", "HOLD"),
            "CASE05_STALE_TAMPER_ESTOP": ("HOLD_BLOCKING", "HOLD"),
        }
        for item in self.cases:
            context = build_evidence_context(
                item["case_id"], TwinCaseInput.from_mapping(item["inputs"])
            )
            self.assertEqual(
                (context.rb_voe_plan.status, context.rb_voe_plan.action),
                expected[item["case_id"]],
            )
            self.assertFalse(
                context.rb_voe_plan.authority.execution_authority
            )

    def test_tamper_changes_both_roots(self) -> None:
        item = self.cases[0]
        original = TwinCaseInput.from_mapping(item["inputs"])
        changed_payload = dict(item["inputs"])
        changed_payload["payload_hash_valid"] = False
        changed = TwinCaseInput.from_mapping(changed_payload)
        first = build_evidence_context(item["case_id"], original)
        second = build_evidence_context(item["case_id"], changed)
        self.assertNotEqual(first.evidence_dag_root, second.evidence_dag_root)
        self.assertNotEqual(first.belief_state_hash, second.belief_state_hash)


if __name__ == "__main__":
    unittest.main()
