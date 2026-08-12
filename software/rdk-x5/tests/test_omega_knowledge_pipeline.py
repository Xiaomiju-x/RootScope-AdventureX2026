from __future__ import annotations

import unittest
from pathlib import Path

from app.omega_runtime.knowledge_pipeline import run_knowledge_roles


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "configs" / "omega" / "field_knowledge.v1.md"


class OmegaKnowledgePipelineTests(unittest.TestCase):
    def test_three_roles_fall_back_with_citations_and_zero_authority(self) -> None:
        context = run_knowledge_roles(
            case_id="CASE02_OOD_RECAPTURE",
            evidence_refs=("case02-ood", "case02-quality"),
            corpus_path=CORPUS,
        )
        self.assertEqual(len(context.responses), 3)
        self.assertEqual(len(context.claim_ledger_root), 64)
        self.assertTrue(context.integrity_report["passed"])
        self.assertEqual(context.claims, ())
        self.assertEqual(
            {item["role"] for item in context.responses},
            {"EVIDENCE_EXPLAINER", "SAFETY_AUDITOR", "DEFENSE_QA"},
        )
        for response in context.responses:
            self.assertEqual(response["status"], "READ_ONLY_FALLBACK")
            self.assertGreaterEqual(len(response["citations"]), 1)
            self.assertTrue(all(value is False for value in response["authority"].values()))
            self.assertFalse(response["provenance"]["model_output_accepted"])
            self.assertFalse(response["provenance"]["tool_interface_supplied"])

    def test_ledger_capsule_root_is_deterministic(self) -> None:
        values = {
            "case_id": "CASE03_ACK_WITHOUT_MASS_LOSS",
            "evidence_refs": ("case03-ack", "case03-mass"),
            "corpus_path": CORPUS,
        }
        first = run_knowledge_roles(**values)
        second = run_knowledge_roles(**values)
        self.assertEqual(first.claim_ledger_root, second.claim_ledger_root)


if __name__ == "__main__":
    unittest.main()
