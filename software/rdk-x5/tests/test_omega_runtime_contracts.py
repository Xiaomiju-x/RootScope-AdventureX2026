from __future__ import annotations

import unittest

from app.omega_runtime.contracts import (
    AuthorityFlags,
    BackendCapsule,
    DecisionProjection,
    DecisionReceipt,
    EvidenceAction,
    RuntimeMode,
    SafetyDecision,
    TruthRibbon,
    canonical_sha256,
)


H0 = "0" * 64
H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64


def receipt(*, remote_shadow: bool = False) -> DecisionReceipt:
    projection = DecisionProjection(
        SafetyDecision.HOLD,
        ("PERCEPTION_OOD",),
        EvidenceAction.RECAPTURE,
        "READY",
        "NO_COMPLETION",
    )
    backend = BackendCapsule(
        profile="DEEP_SHADOW" if remote_shadow else "SAFE_CPU",
        runtime_mode=RuntimeMode.SIMULATION,
        decision_backend_actual="deterministic_cpu",
        vision_backend_actual=(
            "qualified_bpu_probe_with_cpu_projection"
            if remote_shadow
            else "onnxruntime_cpu"
        ),
        retrieval_backend_actual="sqlite_fts5_bm25",
        explanation_backend_actual=(
            "remote_or_pc_deep_model_shadow_only"
            if remote_shadow
            else "deterministic_template"
        ),
        release_id="rootscope-omega-v3-alpha",
        bpu_model_qualified=remote_shadow,
        local_llm_active=False,
        remote_shadow_active=remote_shadow,
    )
    return DecisionReceipt(
        run_id="run-locked-case02",
        event_id="event-case02-evaluate",
        case_id="CASE02_OOD_RECAPTURE",
        evidence_dag_root=H0,
        belief_state_hash=H1,
        failure_core_hash=H2,
        rb_voe_plan_hash=H3,
        claim_ledger_root=H3,
        projection=projection,
        backend=backend,
        authority=AuthorityFlags(),
        generated_at_utc="2026-07-23T00:00:00Z",
    )


class OmegaRuntimeContractTests(unittest.TestCase):
    def test_canonical_hash_ignores_mapping_order(self) -> None:
        self.assertEqual(
            canonical_sha256({"b": 2, "a": 1}),
            canonical_sha256({"a": 1, "b": 2}),
        )

    def test_receipt_hash_is_timestamp_independent(self) -> None:
        first = receipt()
        second = DecisionReceipt(
            **{
                **first.__dict__,
                "generated_at_utc": "2099-01-01T00:00:00Z",
            }
        )
        self.assertEqual(first.receipt_sha256, second.receipt_sha256)

    def test_zero_authority_is_enforced(self) -> None:
        for field_name in AuthorityFlags.__dataclass_fields__:
            with self.subTest(field=field_name):
                with self.assertRaises(ValueError):
                    AuthorityFlags(**{field_name: True})

    def test_projection_cannot_emit_command(self) -> None:
        with self.assertRaises(ValueError):
            DecisionProjection(
                SafetyDecision.ACCEPT,
                ("BAD_TEST",),
                EvidenceAction.NONE,
                "READY",
                "NO_COMPLETION",
                physical_command_emitted=True,
            )

    def test_truth_ribbon_rejects_physical_claim(self) -> None:
        value = receipt()
        with self.assertRaises(ValueError):
            TruthRibbon(
                mode=RuntimeMode.SIMULATION,
                profile="SAFE_CPU",
                backend_actual="deterministic_cpu",
                evidence_state="FRESH",
                evidence_fresh=True,
                receipt_sha256=value.receipt_sha256,
                authority=AuthorityFlags(),
                physical_completion_claim=True,
                warnings=("SIMULATION_ONLY",),
            )

    def test_remote_shadow_changes_capsule_not_decision_projection(self) -> None:
        local = receipt(remote_shadow=False)
        shadow = receipt(remote_shadow=True)
        self.assertEqual(
            local.projection.projection_sha256,
            shadow.projection.projection_sha256,
        )
        self.assertNotEqual(local.receipt_sha256, shadow.receipt_sha256)


if __name__ == "__main__":
    unittest.main()
