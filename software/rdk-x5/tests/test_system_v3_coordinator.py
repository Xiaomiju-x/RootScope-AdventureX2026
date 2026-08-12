from __future__ import annotations

from app.runtime_v3.resource_broker import ResourceSnapshot
from app.system_v3.coordinator import PerceptionEvidence, RootScopeV3Coordinator


def _coordinator() -> RootScopeV3Coordinator:
    return RootScopeV3Coordinator(
        release_sha256="1" * 64,
        config_sha256="2" * 64,
    )


def test_low_cma_falls_back_without_losing_bounded_contract() -> None:
    result = _coordinator().pre_action_cycle(
        contract_id="cycle-1",
        sequence=1,
        boot_id="boot-1",
        evidence=PerceptionEvidence(
            plant_class="grass_clump",
            confidence=0.95,
            ood_hold=False,
            temporal_support=5,
            target_zone="zone-1",
            evidence_root_sha256="3" * 64,
            fresh=True,
        ),
        deterministic_volume_ml=15.0,
        interlocks_clear=True,
        requested_backend="BPU",
        resources=ResourceSnapshot(1200, 50, 50, 1),
        upstream_reason_codes=["RULE_ENGINE_PASS"],
    )
    assert result["truth_ribbon"]["primary_backend"] == "CPU_VISION"
    assert result["action_contract"]["proposed_volume_ml"] == 15.0
    assert result["authority"]["pump_command"] is False
    assert result["truth_ribbon"]["bpu_qualified"] is False


def test_temporal_or_ood_uncertainty_holds_zero_volume() -> None:
    for temporal_support, ood_hold in ((1, False), (5, True)):
        result = _coordinator().pre_action_cycle(
            contract_id=f"cycle-{temporal_support}-{ood_hold}",
            sequence=2,
            boot_id="boot-1",
            evidence=PerceptionEvidence(
                plant_class="young_tree",
                confidence=0.7,
                ood_hold=ood_hold,
                temporal_support=temporal_support,
                target_zone="zone-1",
                evidence_root_sha256="4" * 64,
                fresh=True,
            ),
            deterministic_volume_ml=20.0,
            interlocks_clear=True,
            requested_backend="CPU",
            resources=ResourceSnapshot(1200, 200, 50, 1),
            upstream_reason_codes=["RULE_ENGINE_PASS"],
        )
        assert result["action_contract"]["proposed_volume_ml"] == 0.0
        assert result["truth_ribbon"]["hold"] is True
