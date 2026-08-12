from __future__ import annotations

import unittest
from pathlib import Path

from app.omega_runtime.replay import run_locked_replay


ROOT = Path(__file__).resolve().parents[1]


def run_report() -> dict:
    return dict(
        run_locked_replay(
            cases_path=ROOT / "configs" / "omega" / "locked_replay_cases.v1.json",
            profiles_path=ROOT / "configs" / "omega" / "edge_profiles.v1.json",
            corpus_path=ROOT / "configs" / "omega" / "field_knowledge.v1.md",
        )
    )


class OmegaLockedReplayTests(unittest.TestCase):
    def test_complete_chain_passes_all_five_cases(self) -> None:
        report = run_report()
        self.assertTrue(report["all_locked_cases_passed"])
        self.assertEqual(report["case_count"], 5)
        self.assertEqual(report["matched_case_count"], 5)
        self.assertEqual(report["selected_backend"]["profile"], "SAFE_CPU")
        self.assertIn(
            "BPU_MODEL_NOT_QUALIFIED",
            report["selected_backend"]["fallback_reasons"],
        )
        self.assertFalse(report["runtime_boundary"]["hardware_touched"])
        self.assertFalse(report["runtime_boundary"]["physical_completion_claim"])

    def test_every_receipt_binds_full_chain_and_truth_ribbon(self) -> None:
        report = run_report()
        for item in report["cases"]:
            receipt = item["decision_receipt"]
            for field_name in (
                "evidence_dag_root",
                "belief_state_hash",
                "failure_core_hash",
                "rb_voe_plan_hash",
                "claim_ledger_root",
                "receipt_sha256",
            ):
                self.assertEqual(len(receipt[field_name]), 64)
            self.assertFalse(receipt["authority"]["execution_authority"])
            self.assertFalse(
                receipt["projection"]["physical_command_emitted"]
            )
            ribbon = item["truth_ribbon"]
            self.assertEqual(ribbon["mode"], "SIMULATION")
            self.assertFalse(ribbon["physical_completion_claim"])
            self.assertFalse(ribbon["cloud_shadow_influenced_decision"])

    def test_composition_root_is_repeatable(self) -> None:
        first = run_report()
        second = run_report()
        self.assertEqual(
            first["case_receipts_root"],
            second["case_receipts_root"],
        )


if __name__ == "__main__":
    unittest.main()
