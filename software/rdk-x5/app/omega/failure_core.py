"""Deterministic counterfactual failure-core extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .belief import HybridBeliefState
from .evidence_dag import EvidenceDAG
from .schemas import (
    AuthorityBoundary,
    CoreStatus,
    EvidenceActionType,
    EvidenceKind,
    EvidenceVerdict,
    FailureMode,
    OmegaContractError,
    canonical_sha256,
    enum_value,
    require_exact_keys,
    require_probability,
    require_safe_token,
    require_sha256,
)


FAILURE_SIGNAL_SCHEMA = "rootscope.omega.failure-signal.v1"
FAILURE_CORE_SCHEMA = "rootscope.omega.counterfactual-failure-core.v1"


_KIND_TO_FAILURE: Mapping[EvidenceKind, FailureMode] = {
    EvidenceKind.QUALITY: FailureMode.CAMERA_QUALITY,
    EvidenceKind.SEMANTIC: FailureMode.PERCEPTION_ERROR,
    EvidenceKind.GEOMETRY: FailureMode.PERCEPTION_ERROR,
    EvidenceKind.OOD: FailureMode.OOD_CONFLICT,
    EvidenceKind.SAFETY: FailureMode.SAFETY_INTERLOCK,
    EvidenceKind.ACK: FailureMode.ACTUATOR_NO_ACK,
    EvidenceKind.MASS: FailureMode.MASS_ANOMALY,
    EvidenceKind.WETTING: FailureMode.WETTING_MISS,
}

_KIND_TO_ACTION: Mapping[EvidenceKind, EvidenceActionType] = {
    EvidenceKind.QUALITY: EvidenceActionType.RECAPTURE_IMAGE,
    EvidenceKind.SEMANTIC: EvidenceActionType.RECAPTURE_IMAGE,
    EvidenceKind.GEOMETRY: EvidenceActionType.RECAPTURE_IMAGE,
    EvidenceKind.OOD: EvidenceActionType.REQUEST_OPERATOR_REVIEW,
    EvidenceKind.SAFETY: EvidenceActionType.VERIFY_SAFETY_INTERLOCK,
    EvidenceKind.ACK: EvidenceActionType.WAIT_FOR_FRESH_TELEMETRY,
    EvidenceKind.MASS: EvidenceActionType.REWEIGH,
    EvidenceKind.WETTING: EvidenceActionType.RECAPTURE_IMAGE,
}

DEFAULT_POSTERIOR_THRESHOLDS: Mapping[FailureMode, float] = {
    FailureMode.PERCEPTION_ERROR: 0.35,
    FailureMode.CAMERA_QUALITY: 0.30,
    FailureMode.SENSOR_STALE: 0.30,
    FailureMode.ACTUATOR_NO_ACK: 0.25,
    FailureMode.MASS_ANOMALY: 0.25,
    FailureMode.WETTING_MISS: 0.25,
    FailureMode.NEIGHBOR_SPILL: 0.15,
    FailureMode.SAFETY_INTERLOCK: 0.10,
    FailureMode.OOD_CONFLICT: 0.20,
}


@dataclass(frozen=True)
class FailureCorePolicy:
    required_kinds: tuple[EvidenceKind, ...] = (
        EvidenceKind.QUALITY,
        EvidenceKind.SEMANTIC,
        EvidenceKind.GEOMETRY,
        EvidenceKind.SAFETY,
    )
    posterior_thresholds: tuple[tuple[FailureMode, float], ...] = tuple(
        sorted(DEFAULT_POSTERIOR_THRESHOLDS.items(), key=lambda item: item[0].value)
    )

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.required_kinds), key=lambda kind: kind.value)) != tuple(
            sorted(self.required_kinds, key=lambda kind: kind.value)
        ):
            raise OmegaContractError("required_kinds must be unique")
        thresholds = dict(self.posterior_thresholds)
        if set(thresholds) != set(FailureMode) - {FailureMode.NORMAL}:
            raise OmegaContractError(
                "posterior thresholds must exactly cover non-normal modes"
            )
        for mode, value in thresholds.items():
            require_probability(value, f"posterior threshold {mode.value}")

    @property
    def threshold_map(self) -> dict[FailureMode, float]:
        return dict(self.posterior_thresholds)


@dataclass(frozen=True)
class FailureSignal:
    signal_id: str
    failure_mode: FailureMode
    predicate: str
    observed: EvidenceVerdict
    severity: int
    blocking: bool
    witness_node_ids: tuple[str, ...]
    counterfactual_action: EvidenceActionType
    rationale_code: str
    risk_probability: float

    def __post_init__(self) -> None:
        require_safe_token(self.signal_id, "signal_id")
        require_safe_token(self.predicate, "predicate")
        require_safe_token(self.rationale_code, "rationale_code")
        if not isinstance(self.failure_mode, FailureMode):
            raise OmegaContractError("failure_mode must be FailureMode")
        if not isinstance(self.observed, EvidenceVerdict):
            raise OmegaContractError("observed must be EvidenceVerdict")
        if not isinstance(self.counterfactual_action, EvidenceActionType):
            raise OmegaContractError("counterfactual_action must be EvidenceActionType")
        if (
            isinstance(self.severity, bool)
            or not isinstance(self.severity, int)
            or not 1 <= self.severity <= 5
        ):
            raise OmegaContractError("severity must be an integer within [1, 5]")
        if not isinstance(self.blocking, bool):
            raise OmegaContractError("blocking must be boolean")
        if tuple(sorted(set(self.witness_node_ids))) != self.witness_node_ids:
            raise OmegaContractError("witness_node_ids must be unique and sorted")
        require_probability(self.risk_probability, "risk_probability")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FAILURE_SIGNAL_SCHEMA,
            "signal_id": self.signal_id,
            "failure_mode": self.failure_mode.value,
            "predicate": self.predicate,
            "observed": self.observed.value,
            "severity": self.severity,
            "blocking": self.blocking,
            "witness_node_ids": list(self.witness_node_ids),
            "counterfactual_action": self.counterfactual_action.value,
            "rationale_code": self.rationale_code,
            "risk_probability": self.risk_probability,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureSignal":
        expected = {
            "schema_version",
            "signal_id",
            "failure_mode",
            "predicate",
            "observed",
            "severity",
            "blocking",
            "witness_node_ids",
            "counterfactual_action",
            "rationale_code",
            "risk_probability",
        }
        require_exact_keys(value, expected, "failure signal")
        if value["schema_version"] != FAILURE_SIGNAL_SCHEMA:
            raise OmegaContractError("failure signal schema mismatch")
        if not isinstance(value["witness_node_ids"], list):
            raise OmegaContractError("witness_node_ids must be an array")
        return cls(
            signal_id=value["signal_id"],
            failure_mode=enum_value(
                FailureMode, value["failure_mode"], "failure_mode"
            ),  # type: ignore[arg-type]
            predicate=value["predicate"],
            observed=enum_value(
                EvidenceVerdict, value["observed"], "observed"
            ),  # type: ignore[arg-type]
            severity=value["severity"],
            blocking=value["blocking"],
            witness_node_ids=tuple(value["witness_node_ids"]),
            counterfactual_action=enum_value(
                EvidenceActionType,
                value["counterfactual_action"],
                "counterfactual_action",
            ),  # type: ignore[arg-type]
            rationale_code=value["rationale_code"],
            risk_probability=value["risk_probability"],
        )


@dataclass(frozen=True)
class CounterfactualCoreResult:
    status: CoreStatus
    dag_root_sha256: str
    belief_sha256: str
    signals: tuple[FailureSignal, ...]
    minimal_witness_node_ids: tuple[str, ...]
    authority: AuthorityBoundary = field(default_factory=AuthorityBoundary)
    core_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, CoreStatus):
            raise OmegaContractError("status must be CoreStatus")
        require_sha256(self.dag_root_sha256, "dag_root_sha256")
        require_sha256(self.belief_sha256, "belief_sha256")
        if tuple(sorted(self.signals, key=lambda signal: signal.signal_id)) != self.signals:
            raise OmegaContractError("signals must be sorted by signal_id")
        ids = tuple(signal.signal_id for signal in self.signals)
        if len(ids) != len(set(ids)):
            raise OmegaContractError("failure signal ids must be unique")
        if (
            tuple(sorted(set(self.minimal_witness_node_ids)))
            != self.minimal_witness_node_ids
        ):
            raise OmegaContractError("minimal witness ids must be unique and sorted")
        union = {
            node_id for signal in self.signals for node_id in signal.witness_node_ids
        }
        if not set(self.minimal_witness_node_ids).issubset(union):
            raise OmegaContractError("minimal witness references an unknown signal node")
        expected_status = (
            CoreStatus.BLOCKING
            if any(signal.blocking for signal in self.signals)
            else CoreStatus.NEEDS_EVIDENCE
            if self.signals
            else CoreStatus.CLEAR
        )
        if self.status is not expected_status:
            raise OmegaContractError("failure core status does not match its signals")
        if not isinstance(self.authority, AuthorityBoundary):
            raise OmegaContractError("authority must be zero-authority capsule")
        expected = canonical_sha256(self.unsigned_dict())
        if not self.core_sha256:
            object.__setattr__(self, "core_sha256", expected)
        elif require_sha256(self.core_sha256, "core_sha256") != expected:
            raise OmegaContractError("failure core hash mismatch")

    @property
    def failure_core_hash(self) -> str:
        return self.core_sha256

    @property
    def requested_actions(self) -> tuple[EvidenceActionType, ...]:
        return tuple(
            sorted(
                {signal.counterfactual_action for signal in self.signals},
                key=lambda action: action.value,
            )
        )

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FAILURE_CORE_SCHEMA,
            "status": self.status.value,
            "dag_root_sha256": self.dag_root_sha256,
            "belief_sha256": self.belief_sha256,
            "signals": [signal.to_dict() for signal in self.signals],
            "minimal_witness_node_ids": list(self.minimal_witness_node_ids),
            "authority": self.authority.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "core_sha256": self.core_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CounterfactualCoreResult":
        expected = {
            "schema_version",
            "status",
            "dag_root_sha256",
            "belief_sha256",
            "signals",
            "minimal_witness_node_ids",
            "authority",
            "core_sha256",
        }
        require_exact_keys(value, expected, "counterfactual failure core")
        if value["schema_version"] != FAILURE_CORE_SCHEMA:
            raise OmegaContractError("failure core schema mismatch")
        if not isinstance(value["signals"], list) or not isinstance(
            value["minimal_witness_node_ids"], list
        ):
            raise OmegaContractError("failure core arrays have invalid types")
        return cls(
            status=enum_value(CoreStatus, value["status"], "status"),  # type: ignore[arg-type]
            dag_root_sha256=value["dag_root_sha256"],
            belief_sha256=value["belief_sha256"],
            signals=tuple(FailureSignal.from_dict(item) for item in value["signals"]),
            minimal_witness_node_ids=tuple(value["minimal_witness_node_ids"]),
            authority=AuthorityBoundary.from_dict(value["authority"]),
            core_sha256=value["core_sha256"],
        )


class CounterfactualFailureCore:
    def __init__(self, policy: FailureCorePolicy | None = None) -> None:
        self.policy = policy or FailureCorePolicy()

    @staticmethod
    def _signal(
        *,
        signal_id: str,
        mode: FailureMode,
        predicate: str,
        observed: EvidenceVerdict,
        severity: int,
        blocking: bool,
        witnesses: Iterable[str],
        action: EvidenceActionType,
        rationale: str,
        probability: float,
    ) -> FailureSignal:
        return FailureSignal(
            signal_id=signal_id,
            failure_mode=mode,
            predicate=predicate,
            observed=observed,
            severity=severity,
            blocking=blocking,
            witness_node_ids=tuple(sorted(set(witnesses))),
            counterfactual_action=action,
            rationale_code=rationale,
            risk_probability=probability,
        )

    def analyze(
        self, dag: EvidenceDAG, belief: HybridBeliefState
    ) -> CounterfactualCoreResult:
        if not isinstance(dag, EvidenceDAG):
            raise OmegaContractError("dag must be EvidenceDAG")
        if not isinstance(belief, HybridBeliefState):
            raise OmegaContractError("belief must be HybridBeliefState")
        snapshot = dag.validate()
        dag.require_all(belief.evidence_node_ids)
        probabilities = belief.probability_map
        signals: list[FailureSignal] = []

        for kind in self.policy.required_kinds:
            if dag.latest(kind) is None:
                mode = _KIND_TO_FAILURE[kind]
                signals.append(
                    self._signal(
                        signal_id=f"missing-{kind.value.lower()}",
                        mode=mode,
                        predicate=f"required-{kind.value.lower()}",
                        observed=EvidenceVerdict.UNKNOWN,
                        severity=2,
                        blocking=False,
                        witnesses=(),
                        action=_KIND_TO_ACTION[kind],
                        rationale="REQUIRED_EVIDENCE_MISSING",
                        probability=probabilities[mode],
                    )
                )

        for kind, mode in _KIND_TO_FAILURE.items():
            node = dag.latest(kind)
            if node is None or node.verdict is EvidenceVerdict.PASS:
                continue
            blocking = kind is EvidenceKind.SAFETY or (
                kind is EvidenceKind.WETTING
                and bool(node.payload.get("neighbor_spill", False))
            )
            effective_mode = (
                FailureMode.NEIGHBOR_SPILL
                if kind is EvidenceKind.WETTING
                and bool(node.payload.get("neighbor_spill", False))
                else mode
            )
            signals.append(
                self._signal(
                    signal_id=f"{kind.value.lower()}-{node.verdict.value.lower()}",
                    mode=effective_mode,
                    predicate=f"{kind.value.lower()}-verdict",
                    observed=node.verdict,
                    severity=5 if blocking else 4,
                    blocking=blocking,
                    witnesses=(node.node_id,),
                    action=_KIND_TO_ACTION[kind],
                    rationale=f"{kind.value}_EVIDENCE_{node.verdict.value}",
                    probability=probabilities[effective_mode],
                )
            )

        semantic = dag.latest(EvidenceKind.SEMANTIC)
        geometry = dag.latest(EvidenceKind.GEOMETRY)
        if semantic is not None and geometry is not None:
            semantic_label = semantic.payload.get("label")
            geometry_label = geometry.payload.get("label")
            if not isinstance(semantic_label, str) or not isinstance(
                geometry_label, str
            ):
                signals.append(
                    self._signal(
                        signal_id="perception-label-missing",
                        mode=FailureMode.PERCEPTION_ERROR,
                        predicate="semantic-geometry-label",
                        observed=EvidenceVerdict.UNKNOWN,
                        severity=3,
                        blocking=False,
                        witnesses=(semantic.node_id, geometry.node_id),
                        action=EvidenceActionType.RECAPTURE_IMAGE,
                        rationale="LABEL_FIELD_MISSING",
                        probability=probabilities[FailureMode.PERCEPTION_ERROR],
                    )
                )
            elif semantic_label != geometry_label:
                signals.append(
                    self._signal(
                        signal_id="perception-cross-path-conflict",
                        mode=FailureMode.OOD_CONFLICT,
                        predicate="semantic-geometry-consensus",
                        observed=EvidenceVerdict.CONFLICT,
                        severity=4,
                        blocking=False,
                        witnesses=(semantic.node_id, geometry.node_id),
                        action=EvidenceActionType.REQUEST_OPERATOR_REVIEW,
                        rationale="SEMANTIC_GEOMETRY_LABEL_CONFLICT",
                        probability=probabilities[FailureMode.OOD_CONFLICT],
                    )
                )

        ack = dag.latest(EvidenceKind.ACK)
        mass = dag.latest(EvidenceKind.MASS)
        wetting = dag.latest(EvidenceKind.WETTING)
        if ack is not None and ack.verdict is EvidenceVerdict.PASS and mass is None:
            signals.append(
                self._signal(
                    signal_id="mass-after-ack-missing",
                    mode=FailureMode.MASS_ANOMALY,
                    predicate="ack-requires-mass",
                    observed=EvidenceVerdict.UNKNOWN,
                    severity=4,
                    blocking=False,
                    witnesses=(ack.node_id,),
                    action=EvidenceActionType.REWEIGH,
                    rationale="ACK_WITHOUT_MASS_EVIDENCE",
                    probability=probabilities[FailureMode.MASS_ANOMALY],
                )
            )
        if mass is not None and mass.verdict is EvidenceVerdict.PASS and wetting is None:
            signals.append(
                self._signal(
                    signal_id="wetting-after-mass-missing",
                    mode=FailureMode.WETTING_MISS,
                    predicate="mass-requires-wetting",
                    observed=EvidenceVerdict.UNKNOWN,
                    severity=4,
                    blocking=False,
                    witnesses=(mass.node_id,),
                    action=EvidenceActionType.RECAPTURE_IMAGE,
                    rationale="MASS_WITHOUT_WETTING_EVIDENCE",
                    probability=probabilities[FailureMode.WETTING_MISS],
                )
            )

        thresholds = self.policy.threshold_map
        existing_modes = {signal.failure_mode for signal in signals}
        for mode, threshold in thresholds.items():
            probability = probabilities[mode]
            if probability < threshold or mode in existing_modes:
                continue
            action = {
                FailureMode.CAMERA_QUALITY: EvidenceActionType.RECAPTURE_IMAGE,
                FailureMode.PERCEPTION_ERROR: EvidenceActionType.RECAPTURE_IMAGE,
                FailureMode.SENSOR_STALE: EvidenceActionType.WAIT_FOR_FRESH_TELEMETRY,
                FailureMode.ACTUATOR_NO_ACK: EvidenceActionType.WAIT_FOR_FRESH_TELEMETRY,
                FailureMode.MASS_ANOMALY: EvidenceActionType.REWEIGH,
                FailureMode.WETTING_MISS: EvidenceActionType.RECAPTURE_IMAGE,
                FailureMode.NEIGHBOR_SPILL: EvidenceActionType.REQUEST_OPERATOR_REVIEW,
                FailureMode.SAFETY_INTERLOCK: EvidenceActionType.VERIFY_SAFETY_INTERLOCK,
                FailureMode.OOD_CONFLICT: EvidenceActionType.REQUEST_OPERATOR_REVIEW,
            }[mode]
            blocking = mode in {
                FailureMode.NEIGHBOR_SPILL,
                FailureMode.SAFETY_INTERLOCK,
            }
            signals.append(
                self._signal(
                    signal_id=f"posterior-{mode.value.lower()}",
                    mode=mode,
                    predicate="posterior-threshold",
                    observed=EvidenceVerdict.UNKNOWN,
                    severity=5 if blocking else 3,
                    blocking=blocking,
                    witnesses=belief.evidence_node_ids,
                    action=action,
                    rationale="POSTERIOR_EXCEEDS_FROZEN_THRESHOLD",
                    probability=probability,
                )
            )

        # One minimal witness per failure mode: blocking first, then severity,
        # then stable signal id.  This keeps the result small and reproducible.
        best_by_mode: dict[FailureMode, FailureSignal] = {}
        for signal in signals:
            previous = best_by_mode.get(signal.failure_mode)
            if previous is None or (
                int(signal.blocking),
                signal.severity,
                signal.signal_id,
            ) > (
                int(previous.blocking),
                previous.severity,
                previous.signal_id,
            ):
                best_by_mode[signal.failure_mode] = signal
        minimal_witness = tuple(
            sorted(
                {
                    node_id
                    for signal in best_by_mode.values()
                    for node_id in signal.witness_node_ids
                }
            )
        )
        ordered = tuple(sorted(signals, key=lambda signal: signal.signal_id))
        status = (
            CoreStatus.BLOCKING
            if any(signal.blocking for signal in ordered)
            else CoreStatus.NEEDS_EVIDENCE
            if ordered
            else CoreStatus.CLEAR
        )
        return CounterfactualCoreResult(
            status=status,
            dag_root_sha256=snapshot.root_sha256,
            belief_sha256=belief.state_sha256,
            signals=ordered,
            minimal_witness_node_ids=minimal_witness,
        )
