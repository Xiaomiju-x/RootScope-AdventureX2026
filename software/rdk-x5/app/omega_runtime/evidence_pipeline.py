"""Adapter from locked RootScope twin inputs into the strict Ω evidence core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from app.omega import (
    BoundedMeasurement,
    CounterfactualCoreResult,
    CounterfactualFailureCore,
    EvidenceDAG,
    EvidenceKind,
    EvidenceMode,
    EvidenceNode,
    EvidenceVerdict,
    FailureMode,
    HybridBeliefState,
    RbVoePlan,
    RbVoePlanner,
)

from .digital_twin import TwinCaseInput


@dataclass(frozen=True)
class EvidenceContext:
    dag: EvidenceDAG
    belief: HybridBeliefState
    failure_core: CounterfactualCoreResult
    rb_voe_plan: RbVoePlan

    @property
    def evidence_dag_root(self) -> str:
        return self.dag.root_sha256

    @property
    def belief_state_hash(self) -> str:
        return self.belief.belief_hash

    @property
    def failure_core_hash(self) -> str:
        return self.failure_core.failure_core_hash

    @property
    def rb_voe_plan_hash(self) -> str:
        return self.rb_voe_plan.plan_sha256


def _finalize_context(
    dag: EvidenceDAG, belief: HybridBeliefState
) -> EvidenceContext:
    dag.validate()
    failure_core = CounterfactualFailureCore().analyze(dag, belief)
    rb_voe_plan = RbVoePlanner(
        cost_weight=0.02,
        entropy_weight=0.60,
        minimum_value=1e-9,
    ).plan(
        dag=dag,
        belief=belief,
        failure_core=failure_core,
        horizon=2,
    )
    return EvidenceContext(
        dag=dag,
        belief=belief,
        failure_core=failure_core,
        rb_voe_plan=rb_voe_plan,
    )


def _likelihoods(
    *,
    passed: bool,
    failure_mode: FailureMode,
) -> Mapping[FailureMode, float]:
    values = {mode: 0.70 for mode in FailureMode}
    if passed:
        values[FailureMode.NORMAL] = 0.9
        values[failure_mode] = 0.20
    else:
        values = {mode: 0.20 for mode in FailureMode}
        values[FailureMode.NORMAL] = 0.05
        values[failure_mode] = 0.9
    return values


def _node(
    *,
    case_id: str,
    suffix: str,
    kind: EvidenceKind,
    passed: bool,
    observed_at_ms: int,
    payload: Mapping[str, Any],
    parents: tuple[str, ...] = (),
) -> EvidenceNode:
    return EvidenceNode.create(
        node_id=f"{case_id.lower()}-{suffix}",
        kind=kind,
        verdict=EvidenceVerdict.PASS if passed else EvidenceVerdict.FAIL,
        mode=EvidenceMode.SIMULATION,
        source_id="locked-case-fixture",
        observed_at_ms=observed_at_ms,
        payload=payload,
        parents=parents,
    )


def _measurement(
    name: str,
    value: float,
    *,
    lower: float,
    upper: float,
    hard_lower: float,
    hard_upper: float,
) -> BoundedMeasurement:
    span = max(upper - lower, 1e-6)
    return BoundedMeasurement(
        name=name,
        value=value,
        variance=max((span / 4.0) ** 2, 1e-6),
        lower=lower,
        upper=upper,
        hard_lower=hard_lower,
        hard_upper=hard_upper,
    )


def build_evidence_context(case_id: str, case: TwinCaseInput) -> EvidenceContext:
    dag = EvidenceDAG()
    belief = HybridBeliefState.create(belief_id=f"belief-{case_id.lower()}")

    source_pass = case.evidence_fresh and case.payload_hash_valid
    source = _node(
        case_id=case_id,
        suffix="source",
        kind=EvidenceKind.SOURCE,
        passed=source_pass,
        observed_at_ms=1000,
        payload={
            "evidence_fresh": case.evidence_fresh,
            "payload_hash_valid": case.payload_hash_valid,
        },
    )
    dag.add(source)
    belief = belief.update(
        dag=dag,
        evidence_node_id=source.node_id,
        likelihoods=_likelihoods(
            passed=source_pass, failure_mode=FailureMode.SENSOR_STALE
        ),
    )

    if not source_pass:
        safety_pass = case.firmware_connected and case.estop_clear
        safety = _node(
            case_id=case_id,
            suffix="safety",
            kind=EvidenceKind.SAFETY,
            passed=safety_pass,
            observed_at_ms=4000,
            payload={
                "firmware_connected": case.firmware_connected,
                "estop_clear": case.estop_clear,
            },
            parents=(source.node_id,),
        )
        dag.add(safety)
        belief = belief.update(
            dag=dag,
            evidence_node_id=safety.node_id,
            likelihoods=_likelihoods(
                passed=safety_pass,
                failure_mode=FailureMode.SAFETY_INTERLOCK,
            ),
        )
        return _finalize_context(dag, belief)

    quality = _node(
        case_id=case_id,
        suffix="quality",
        kind=EvidenceKind.QUALITY,
        passed=case.camera_quality_ok,
        observed_at_ms=2000,
        payload={"camera_quality_ok": case.camera_quality_ok},
        parents=(source.node_id,),
    )
    dag.add(quality)
    belief = belief.update(
        dag=dag,
        evidence_node_id=quality.node_id,
        likelihoods=_likelihoods(
            passed=case.camera_quality_ok,
            failure_mode=FailureMode.CAMERA_QUALITY,
        ),
    )

    perception_pass = case.camera_quality_ok and not case.ood_detected
    semantic = _node(
        case_id=case_id,
        suffix="semantic",
        kind=EvidenceKind.SEMANTIC,
        passed=perception_pass,
        observed_at_ms=2500,
        payload={
            "label": "locked-fixture-target",
            "path": "semantic-cpu-fixture",
        },
        parents=(quality.node_id,),
    )
    dag.add(semantic)
    belief = belief.update(
        dag=dag,
        evidence_node_id=semantic.node_id,
        likelihoods=_likelihoods(
            passed=perception_pass,
            failure_mode=FailureMode.PERCEPTION_ERROR,
        ),
    )

    geometry = _node(
        case_id=case_id,
        suffix="geometry",
        kind=EvidenceKind.GEOMETRY,
        passed=perception_pass,
        observed_at_ms=2600,
        payload={
            "label": "locked-fixture-target",
            "path": "geometry-template-fixture",
        },
        parents=(quality.node_id,),
    )
    dag.add(geometry)
    belief = belief.update(
        dag=dag,
        evidence_node_id=geometry.node_id,
        likelihoods=_likelihoods(
            passed=perception_pass,
            failure_mode=FailureMode.PERCEPTION_ERROR,
        ),
    )

    ood_pass = not case.ood_detected
    ood = _node(
        case_id=case_id,
        suffix="ood",
        kind=EvidenceKind.OOD,
        passed=ood_pass,
        observed_at_ms=3000,
        payload={"ood_detected": case.ood_detected},
        parents=tuple(sorted((geometry.node_id, semantic.node_id))),
    )
    dag.add(ood)
    belief = belief.update(
        dag=dag,
        evidence_node_id=ood.node_id,
        likelihoods=_likelihoods(
            passed=ood_pass, failure_mode=FailureMode.OOD_CONFLICT
        ),
    )
    if not perception_pass:
        return _finalize_context(dag, belief)

    safety_pass = case.firmware_connected and case.estop_clear
    safety = _node(
        case_id=case_id,
        suffix="safety",
        kind=EvidenceKind.SAFETY,
        passed=safety_pass,
        observed_at_ms=4000,
        payload={
            "firmware_connected": case.firmware_connected,
            "estop_clear": case.estop_clear,
        },
        parents=(source.node_id,),
    )
    dag.add(safety)
    belief = belief.update(
        dag=dag,
        evidence_node_id=safety.node_id,
        likelihoods=_likelihoods(
            passed=safety_pass, failure_mode=FailureMode.SAFETY_INTERLOCK
        ),
    )
    if not safety_pass:
        return _finalize_context(dag, belief)

    ack = _node(
        case_id=case_id,
        suffix="ack",
        kind=EvidenceKind.ACK,
        passed=case.ack_ok,
        observed_at_ms=5000,
        payload={"ack_ok": case.ack_ok},
        parents=(safety.node_id,),
    )
    dag.add(ack)
    belief = belief.update(
        dag=dag,
        evidence_node_id=ack.node_id,
        likelihoods=_likelihoods(
            passed=case.ack_ok, failure_mode=FailureMode.ACTUATOR_NO_ACK
        ),
    )
    if not case.ack_ok:
        return _finalize_context(dag, belief)

    mass_error = abs(case.measured_mass_loss_mg - case.target_mass_mg)
    mass_pass = case.ack_ok and mass_error <= case.tolerance_mg
    mass = _node(
        case_id=case_id,
        suffix="mass",
        kind=EvidenceKind.MASS,
        passed=mass_pass,
        observed_at_ms=6000,
        payload={
            "measured_mass_loss_mg": case.measured_mass_loss_mg,
            "target_mass_mg": case.target_mass_mg,
            "tolerance_mg": case.tolerance_mg,
            "mass_error_mg": mass_error,
        },
        parents=(ack.node_id,),
    )
    dag.add(mass)
    mass_measurement: Optional[BoundedMeasurement] = None
    mass_upper = float(max(case.target_mass_mg * 2, 1))
    mass_measurement = _measurement(
        "mass_loss_mg",
        float(case.measured_mass_loss_mg),
        lower=float(max(0, case.measured_mass_loss_mg - case.tolerance_mg)),
        upper=float(
            min(mass_upper, case.measured_mass_loss_mg + case.tolerance_mg)
        ),
        hard_lower=0.0,
        hard_upper=mass_upper + 1.0,
    )
    belief = belief.update(
        dag=dag,
        evidence_node_id=mass.node_id,
        likelihoods=_likelihoods(
            passed=mass_pass, failure_mode=FailureMode.MASS_ANOMALY
        ),
        measurements=(mass_measurement,),
    )
    if not mass_pass:
        return _finalize_context(dag, belief)

    spill = case.neighbor_wetting_score > case.neighbor_spill_threshold
    target_pass = case.target_wetting_score >= case.target_wetting_threshold
    wetting_pass = target_pass and not spill
    wetting = _node(
        case_id=case_id,
        suffix="wetting",
        kind=EvidenceKind.WETTING,
        passed=wetting_pass,
        observed_at_ms=7000,
        payload={
            "target_wetting_score": case.target_wetting_score,
            "target_wetting_threshold": case.target_wetting_threshold,
            "neighbor_wetting_score": case.neighbor_wetting_score,
            "neighbor_spill_threshold": case.neighbor_spill_threshold,
            "neighbor_spill": spill,
        },
        parents=tuple(sorted((mass.node_id, ood.node_id))),
    )
    dag.add(wetting)
    wetting_mode = (
        FailureMode.NEIGHBOR_SPILL if spill else FailureMode.WETTING_MISS
    )
    belief = belief.update(
        dag=dag,
        evidence_node_id=wetting.node_id,
        likelihoods=_likelihoods(
            passed=wetting_pass, failure_mode=wetting_mode
        ),
        measurements=(
            _measurement(
                "neighbor_wetting_score",
                case.neighbor_wetting_score,
                lower=max(0.0, case.neighbor_wetting_score - 0.02),
                upper=min(1.0, case.neighbor_wetting_score + 0.02),
                hard_lower=0.0,
                hard_upper=1.000001,
            ),
            _measurement(
                "target_wetting_score",
                case.target_wetting_score,
                lower=max(0.0, case.target_wetting_score - 0.02),
                upper=min(1.0, case.target_wetting_score + 0.02),
                hard_lower=0.0,
                hard_upper=1.000001,
            ),
        ),
    )
    return _finalize_context(dag, belief)
