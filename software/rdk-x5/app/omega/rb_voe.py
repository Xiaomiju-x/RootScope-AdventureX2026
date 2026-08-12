"""H=1/H=2 risk-bounded Value-of-Evidence planner.

The planner recommends only observation actions.  It has no execution method
and every action/result carries an exact all-false authority capsule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .belief import HybridBeliefState
from .evidence_dag import EvidenceDAG
from .failure_core import CounterfactualCoreResult
from .schemas import (
    AuthorityBoundary,
    CoreStatus,
    EvidenceActionType,
    EvidenceKind,
    FailureMode,
    ObservationOutcome,
    OmegaContractError,
    canonical_sha256,
    enum_value,
    require_exact_keys,
    require_finite,
    require_probability,
    require_safe_token,
    require_sha256,
)


ACTION_SCHEMA = "rootscope.omega.evidence-action.v1"
PLAN_SCHEMA = "rootscope.omega.rb-voe-plan.v1"
EVALUATION_SCHEMA = "rootscope.omega.rb-voe-action-evaluation.v1"
BRANCH_SCHEMA = "rootscope.omega.rb-voe-branch.v1"
_PROBABILITY_TOLERANCE = 1e-9


DEFAULT_RISK_WEIGHTS: Mapping[FailureMode, float] = {
    FailureMode.NORMAL: 0.0,
    FailureMode.PERCEPTION_ERROR: 0.35,
    FailureMode.CAMERA_QUALITY: 0.30,
    FailureMode.SENSOR_STALE: 0.45,
    FailureMode.ACTUATOR_NO_ACK: 0.70,
    FailureMode.MASS_ANOMALY: 0.65,
    FailureMode.WETTING_MISS: 0.65,
    FailureMode.NEIGHBOR_SPILL: 0.95,
    FailureMode.SAFETY_INTERLOCK: 1.0,
    FailureMode.OOD_CONFLICT: 0.50,
}


def _validate_model(
    model: tuple[tuple[ObservationOutcome, tuple[tuple[FailureMode, float], ...]], ...]
) -> None:
    if tuple(outcome for outcome, _ in model) != tuple(ObservationOutcome):
        raise OmegaContractError("action model must use canonical outcome order")
    by_outcome = {outcome: dict(values) for outcome, values in model}
    for outcome, values in model:
        if tuple(mode for mode, _ in values) != tuple(
            sorted(FailureMode, key=lambda mode: mode.value)
        ):
            raise OmegaContractError(
                f"{outcome.value} likelihoods must use canonical mode order"
            )
        if set(dict(values)) != set(FailureMode):
            raise OmegaContractError("action outcome must cover every failure mode")
        for mode, value in values:
            require_probability(value, f"{outcome.value}.{mode.value}")
    for mode in FailureMode:
        total = math.fsum(by_outcome[outcome][mode] for outcome in ObservationOutcome)
        if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
            raise OmegaContractError(
                f"action outcome probabilities for {mode.value} sum to {total!r}"
            )


@dataclass(frozen=True)
class EvidenceAction:
    action_id: str
    action_type: EvidenceActionType
    evidence_kind: EvidenceKind
    cost: float
    expected_duration_ms: int
    observation_model: tuple[
        tuple[ObservationOutcome, tuple[tuple[FailureMode, float], ...]], ...
    ]
    authority: AuthorityBoundary = field(default_factory=AuthorityBoundary)

    def __post_init__(self) -> None:
        require_safe_token(self.action_id, "action_id")
        if not isinstance(self.action_type, EvidenceActionType):
            raise OmegaContractError("action_type must be EvidenceActionType")
        if self.action_type is EvidenceActionType.HOLD:
            raise OmegaContractError("HOLD is a planner result, not an evidence action")
        if not isinstance(self.evidence_kind, EvidenceKind):
            raise OmegaContractError("evidence_kind must be EvidenceKind")
        object.__setattr__(self, "cost", require_finite(self.cost, "cost"))
        if not 0.0 <= self.cost <= 1.0:
            raise OmegaContractError("cost must be within [0, 1]")
        if (
            isinstance(self.expected_duration_ms, bool)
            or not isinstance(self.expected_duration_ms, int)
            or self.expected_duration_ms <= 0
        ):
            raise OmegaContractError("expected_duration_ms must be positive integer")
        _validate_model(self.observation_model)
        if not isinstance(self.authority, AuthorityBoundary):
            raise OmegaContractError("authority must be zero-authority capsule")

    @property
    def model_map(
        self,
    ) -> dict[ObservationOutcome, dict[FailureMode, float]]:
        return {
            outcome: dict(probabilities)
            for outcome, probabilities in self.observation_model
        }

    def likelihoods(self, outcome: ObservationOutcome) -> dict[FailureMode, float]:
        return self.model_map[outcome]

    def outcome_probability(
        self, belief: HybridBeliefState, outcome: ObservationOutcome
    ) -> float:
        likelihoods = self.likelihoods(outcome)
        return math.fsum(
            probability * likelihoods[mode]
            for mode, probability in belief.probabilities
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_SCHEMA,
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "evidence_kind": self.evidence_kind.value,
            "cost": self.cost,
            "expected_duration_ms": self.expected_duration_ms,
            "observation_model": {
                outcome.value: {
                    mode.value: probability for mode, probability in probabilities
                }
                for outcome, probabilities in self.observation_model
            },
            "authority": self.authority.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceAction":
        expected = {
            "schema_version",
            "action_id",
            "action_type",
            "evidence_kind",
            "cost",
            "expected_duration_ms",
            "observation_model",
            "authority",
        }
        require_exact_keys(value, expected, "evidence action")
        if value["schema_version"] != ACTION_SCHEMA:
            raise OmegaContractError("evidence action schema mismatch")
        raw_model = value["observation_model"]
        if not isinstance(raw_model, Mapping):
            raise OmegaContractError("observation_model must be an object")
        expected_outcomes = {outcome.value for outcome in ObservationOutcome}
        require_exact_keys(raw_model, expected_outcomes, "observation_model")
        model: list[
            tuple[ObservationOutcome, tuple[tuple[FailureMode, float], ...]]
        ] = []
        expected_modes = {mode.value for mode in FailureMode}
        for outcome in ObservationOutcome:
            raw_probabilities = raw_model[outcome.value]
            if not isinstance(raw_probabilities, Mapping):
                raise OmegaContractError(
                    f"observation_model.{outcome.value} must be an object"
                )
            require_exact_keys(
                raw_probabilities,
                expected_modes,
                f"observation_model.{outcome.value}",
            )
            model.append(
                (
                    outcome,
                    tuple(
                        (
                            mode,
                            require_probability(
                                raw_probabilities[mode.value],
                                f"observation_model.{outcome.value}.{mode.value}",
                            ),
                        )
                        for mode in sorted(FailureMode, key=lambda item: item.value)
                    ),
                )
            )
        return cls(
            action_id=value["action_id"],
            action_type=enum_value(
                EvidenceActionType, value["action_type"], "action_type"
            ),  # type: ignore[arg-type]
            evidence_kind=enum_value(
                EvidenceKind, value["evidence_kind"], "evidence_kind"
            ),  # type: ignore[arg-type]
            cost=value["cost"],
            expected_duration_ms=value["expected_duration_ms"],
            observation_model=tuple(model),
            authority=AuthorityBoundary.from_dict(value["authority"]),
        )


def _observation_model(
    *,
    target_modes: set[FailureMode],
    sensitivity: float,
    specificity: float,
    unknown: float,
) -> tuple[tuple[ObservationOutcome, tuple[tuple[FailureMode, float], ...]], ...]:
    for name, value in (
        ("sensitivity", sensitivity),
        ("specificity", specificity),
        ("unknown", unknown),
    ):
        require_probability(value, name)
    result: list[
        tuple[ObservationOutcome, tuple[tuple[FailureMode, float], ...]]
    ] = []
    values: dict[ObservationOutcome, list[tuple[FailureMode, float]]] = {
        outcome: [] for outcome in ObservationOutcome
    }
    for mode in sorted(FailureMode, key=lambda item: item.value):
        if mode in target_modes:
            fail = sensitivity * (1.0 - unknown)
            passed = (1.0 - sensitivity) * (1.0 - unknown)
        else:
            passed = specificity * (1.0 - unknown)
            fail = (1.0 - specificity) * (1.0 - unknown)
        values[ObservationOutcome.PASS].append((mode, passed))
        values[ObservationOutcome.FAIL].append((mode, fail))
        values[ObservationOutcome.UNKNOWN].append((mode, unknown))
    for outcome in ObservationOutcome:
        result.append((outcome, tuple(values[outcome])))
    return tuple(result)


def default_evidence_actions() -> tuple[EvidenceAction, ...]:
    actions = (
        EvidenceAction(
            action_id="recapture-image",
            action_type=EvidenceActionType.RECAPTURE_IMAGE,
            evidence_kind=EvidenceKind.QUALITY,
            cost=0.07,
            expected_duration_ms=900,
            observation_model=_observation_model(
                target_modes={
                    FailureMode.PERCEPTION_ERROR,
                    FailureMode.CAMERA_QUALITY,
                    FailureMode.WETTING_MISS,
                    FailureMode.OOD_CONFLICT,
                },
                sensitivity=0.88,
                specificity=0.90,
                unknown=0.07,
            ),
        ),
        EvidenceAction(
            action_id="request-operator-review",
            action_type=EvidenceActionType.REQUEST_OPERATOR_REVIEW,
            evidence_kind=EvidenceKind.CLAIM,
            cost=0.16,
            expected_duration_ms=5000,
            observation_model=_observation_model(
                target_modes={
                    FailureMode.PERCEPTION_ERROR,
                    FailureMode.NEIGHBOR_SPILL,
                    FailureMode.SAFETY_INTERLOCK,
                    FailureMode.OOD_CONFLICT,
                },
                sensitivity=0.96,
                specificity=0.97,
                unknown=0.02,
            ),
        ),
        EvidenceAction(
            action_id="reweigh",
            action_type=EvidenceActionType.REWEIGH,
            evidence_kind=EvidenceKind.MASS,
            cost=0.05,
            expected_duration_ms=1200,
            observation_model=_observation_model(
                target_modes={
                    FailureMode.ACTUATOR_NO_ACK,
                    FailureMode.MASS_ANOMALY,
                },
                sensitivity=0.94,
                specificity=0.94,
                unknown=0.04,
            ),
        ),
        EvidenceAction(
            action_id="verify-safety-interlock",
            action_type=EvidenceActionType.VERIFY_SAFETY_INTERLOCK,
            evidence_kind=EvidenceKind.SAFETY,
            cost=0.04,
            expected_duration_ms=700,
            observation_model=_observation_model(
                target_modes={
                    FailureMode.NEIGHBOR_SPILL,
                    FailureMode.SAFETY_INTERLOCK,
                },
                sensitivity=0.99,
                specificity=0.99,
                unknown=0.005,
            ),
        ),
        EvidenceAction(
            action_id="wait-fresh-telemetry",
            action_type=EvidenceActionType.WAIT_FOR_FRESH_TELEMETRY,
            evidence_kind=EvidenceKind.ACK,
            cost=0.03,
            expected_duration_ms=600,
            observation_model=_observation_model(
                target_modes={
                    FailureMode.SENSOR_STALE,
                    FailureMode.ACTUATOR_NO_ACK,
                },
                sensitivity=0.90,
                specificity=0.92,
                unknown=0.06,
            ),
        ),
    )
    return tuple(sorted(actions, key=lambda action: action.action_id))


@dataclass(frozen=True)
class PlanBranch:
    outcome: ObservationOutcome
    probability: float
    posterior_belief_sha256: str
    posterior_risk: float
    second_action: EvidenceActionType
    terminal_expected_risk: float
    terminal_expected_loss: float

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ObservationOutcome):
            raise OmegaContractError("branch outcome must be ObservationOutcome")
        require_probability(self.probability, "branch probability")
        require_sha256(self.posterior_belief_sha256, "posterior_belief_sha256")
        for name in ("posterior_risk", "terminal_expected_risk", "terminal_expected_loss"):
            value = require_finite(getattr(self, name), name)
            if value < 0:
                raise OmegaContractError(f"{name} cannot be negative")
        if not isinstance(self.second_action, EvidenceActionType):
            raise OmegaContractError("second_action must be EvidenceActionType")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BRANCH_SCHEMA,
            "outcome": self.outcome.value,
            "probability": self.probability,
            "posterior_belief_sha256": self.posterior_belief_sha256,
            "posterior_risk": self.posterior_risk,
            "second_action": self.second_action.value,
            "terminal_expected_risk": self.terminal_expected_risk,
            "terminal_expected_loss": self.terminal_expected_loss,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanBranch":
        expected = {
            "schema_version",
            "outcome",
            "probability",
            "posterior_belief_sha256",
            "posterior_risk",
            "second_action",
            "terminal_expected_risk",
            "terminal_expected_loss",
        }
        require_exact_keys(value, expected, "RB-VoE plan branch")
        if value["schema_version"] != BRANCH_SCHEMA:
            raise OmegaContractError("RB-VoE branch schema mismatch")
        return cls(
            outcome=enum_value(
                ObservationOutcome, value["outcome"], "outcome"
            ),  # type: ignore[arg-type]
            probability=value["probability"],
            posterior_belief_sha256=value["posterior_belief_sha256"],
            posterior_risk=value["posterior_risk"],
            second_action=enum_value(
                EvidenceActionType, value["second_action"], "second_action"
            ),  # type: ignore[arg-type]
            terminal_expected_risk=value["terminal_expected_risk"],
            terminal_expected_loss=value["terminal_expected_loss"],
        )


@dataclass(frozen=True)
class ActionEvaluation:
    action_id: str
    action_type: EvidenceActionType
    horizon: int
    branches: tuple[PlanBranch, ...]
    expected_terminal_risk: float
    worst_case_terminal_risk: float
    expected_total_cost: float
    objective: float
    value_of_evidence: float

    def __post_init__(self) -> None:
        require_safe_token(self.action_id, "action evaluation id")
        if not isinstance(self.action_type, EvidenceActionType):
            raise OmegaContractError("action_type must be EvidenceActionType")
        if self.horizon not in {1, 2}:
            raise OmegaContractError("evaluation horizon must be 1 or 2")
        if tuple(branch.outcome for branch in self.branches) != tuple(
            ObservationOutcome
        ):
            raise OmegaContractError("branches must exactly cover canonical outcomes")
        if abs(math.fsum(branch.probability for branch in self.branches) - 1.0) > 1e-9:
            raise OmegaContractError("branch probabilities must sum to one")
        for name in (
            "expected_terminal_risk",
            "worst_case_terminal_risk",
            "expected_total_cost",
            "objective",
            "value_of_evidence",
        ):
            value = require_finite(getattr(self, name), name)
            if name != "value_of_evidence" and value < 0:
                raise OmegaContractError(f"{name} cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_SCHEMA,
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "horizon": self.horizon,
            "branches": [branch.to_dict() for branch in self.branches],
            "expected_terminal_risk": self.expected_terminal_risk,
            "worst_case_terminal_risk": self.worst_case_terminal_risk,
            "expected_total_cost": self.expected_total_cost,
            "objective": self.objective,
            "value_of_evidence": self.value_of_evidence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionEvaluation":
        expected = {
            "schema_version",
            "action_id",
            "action_type",
            "horizon",
            "branches",
            "expected_terminal_risk",
            "worst_case_terminal_risk",
            "expected_total_cost",
            "objective",
            "value_of_evidence",
        }
        require_exact_keys(value, expected, "RB-VoE action evaluation")
        if value["schema_version"] != EVALUATION_SCHEMA:
            raise OmegaContractError("RB-VoE evaluation schema mismatch")
        raw_branches = value["branches"]
        if not isinstance(raw_branches, list):
            raise OmegaContractError("RB-VoE evaluation branches must be an array")
        return cls(
            action_id=value["action_id"],
            action_type=enum_value(
                EvidenceActionType, value["action_type"], "action_type"
            ),  # type: ignore[arg-type]
            horizon=value["horizon"],
            branches=tuple(PlanBranch.from_dict(item) for item in raw_branches),
            expected_terminal_risk=value["expected_terminal_risk"],
            worst_case_terminal_risk=value["worst_case_terminal_risk"],
            expected_total_cost=value["expected_total_cost"],
            objective=value["objective"],
            value_of_evidence=value["value_of_evidence"],
        )


@dataclass(frozen=True)
class RbVoePlan:
    status: str
    horizon: int
    selected_action: EvidenceActionType
    selected_action_id: str | None
    dag_root_sha256: str
    belief_sha256: str
    failure_core_sha256: str
    initial_risk: float
    expected_terminal_risk: float
    worst_case_terminal_risk: float
    value_of_evidence: float
    reason_code: str
    evaluation: ActionEvaluation | None
    authority: AuthorityBoundary = field(default_factory=AuthorityBoundary)
    plan_sha256: str = ""

    def __post_init__(self) -> None:
        require_safe_token(self.status, "plan status")
        require_safe_token(self.reason_code, "reason_code")
        if self.horizon not in {1, 2}:
            raise OmegaContractError("plan horizon must be 1 or 2")
        if not isinstance(self.selected_action, EvidenceActionType):
            raise OmegaContractError("selected_action must be EvidenceActionType")
        for name in (
            "dag_root_sha256",
            "belief_sha256",
            "failure_core_sha256",
        ):
            require_sha256(getattr(self, name), name)
        for name in (
            "initial_risk",
            "expected_terminal_risk",
            "worst_case_terminal_risk",
            "value_of_evidence",
        ):
            value = require_finite(getattr(self, name), name)
            if name != "value_of_evidence" and value < 0:
                raise OmegaContractError(f"{name} cannot be negative")
        if self.selected_action is EvidenceActionType.HOLD:
            if self.selected_action_id is not None or self.evaluation is not None:
                raise OmegaContractError("HOLD cannot carry an action evaluation")
        else:
            if self.selected_action_id is None or self.evaluation is None:
                raise OmegaContractError("selected evidence action requires evaluation")
            require_safe_token(self.selected_action_id, "selected_action_id")
            if (
                self.evaluation.action_id != self.selected_action_id
                or self.evaluation.action_type is not self.selected_action
            ):
                raise OmegaContractError("selected action and evaluation diverged")
        if not isinstance(self.authority, AuthorityBoundary):
            raise OmegaContractError("authority must be zero-authority capsule")
        expected = canonical_sha256(self.unsigned_dict())
        if not self.plan_sha256:
            object.__setattr__(self, "plan_sha256", expected)
        elif require_sha256(self.plan_sha256, "plan_sha256") != expected:
            raise OmegaContractError("RB-VoE plan hash mismatch")

    @property
    def action(self) -> str:
        """Compact runtime-facing selected-action value."""

        return self.selected_action.value

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA,
            "status": self.status,
            "horizon": self.horizon,
            "selected_action": self.selected_action.value,
            "selected_action_id": self.selected_action_id,
            "dag_root_sha256": self.dag_root_sha256,
            "belief_sha256": self.belief_sha256,
            "failure_core_sha256": self.failure_core_sha256,
            "initial_risk": self.initial_risk,
            "expected_terminal_risk": self.expected_terminal_risk,
            "worst_case_terminal_risk": self.worst_case_terminal_risk,
            "value_of_evidence": self.value_of_evidence,
            "reason_code": self.reason_code,
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
            "authority": self.authority.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "plan_sha256": self.plan_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RbVoePlan":
        expected = {
            "schema_version",
            "status",
            "horizon",
            "selected_action",
            "selected_action_id",
            "dag_root_sha256",
            "belief_sha256",
            "failure_core_sha256",
            "initial_risk",
            "expected_terminal_risk",
            "worst_case_terminal_risk",
            "value_of_evidence",
            "reason_code",
            "evaluation",
            "authority",
            "plan_sha256",
        }
        require_exact_keys(value, expected, "RB-VoE plan")
        if value["schema_version"] != PLAN_SCHEMA:
            raise OmegaContractError("RB-VoE plan schema mismatch")
        raw_evaluation = value["evaluation"]
        if raw_evaluation is not None and not isinstance(raw_evaluation, Mapping):
            raise OmegaContractError("RB-VoE plan evaluation must be object or null")
        selected_action_id = value["selected_action_id"]
        if selected_action_id is not None and not isinstance(selected_action_id, str):
            raise OmegaContractError(
                "RB-VoE selected_action_id must be string or null"
            )
        return cls(
            status=value["status"],
            horizon=value["horizon"],
            selected_action=enum_value(
                EvidenceActionType, value["selected_action"], "selected_action"
            ),  # type: ignore[arg-type]
            selected_action_id=selected_action_id,
            dag_root_sha256=value["dag_root_sha256"],
            belief_sha256=value["belief_sha256"],
            failure_core_sha256=value["failure_core_sha256"],
            initial_risk=value["initial_risk"],
            expected_terminal_risk=value["expected_terminal_risk"],
            worst_case_terminal_risk=value["worst_case_terminal_risk"],
            value_of_evidence=value["value_of_evidence"],
            reason_code=value["reason_code"],
            evaluation=(
                None
                if raw_evaluation is None
                else ActionEvaluation.from_dict(raw_evaluation)
            ),
            authority=AuthorityBoundary.from_dict(value["authority"]),
            plan_sha256=value["plan_sha256"],
        )


class RbVoePlanner:
    def __init__(
        self,
        *,
        actions: Iterable[EvidenceAction] | None = None,
        risk_weights: Mapping[FailureMode, float] = DEFAULT_RISK_WEIGHTS,
        cost_weight: float = 0.20,
        entropy_weight: float = 0.20,
        max_expected_residual_risk: float = 1.0,
        minimum_value: float = 1e-6,
    ) -> None:
        self.actions = tuple(
            sorted(
                actions if actions is not None else default_evidence_actions(),
                key=lambda action: action.action_id,
            )
        )
        if not self.actions:
            raise OmegaContractError("RB-VoE action catalog cannot be empty")
        ids = [action.action_id for action in self.actions]
        types = [action.action_type for action in self.actions]
        if len(ids) != len(set(ids)) or len(types) != len(set(types)):
            raise OmegaContractError("action ids and types must be unique")
        if set(risk_weights) != set(FailureMode):
            raise OmegaContractError("risk_weights must exactly cover all modes")
        self.risk_weights = {
            mode: require_probability(value, f"risk weight {mode.value}")
            for mode, value in risk_weights.items()
        }
        self.cost_weight = require_finite(cost_weight, "cost_weight")
        self.entropy_weight = require_finite(entropy_weight, "entropy_weight")
        self.max_expected_residual_risk = require_probability(
            max_expected_residual_risk, "max_expected_residual_risk"
        )
        self.minimum_value = require_finite(minimum_value, "minimum_value")
        if min(
            self.cost_weight,
            self.entropy_weight,
            self.minimum_value,
        ) < 0:
            raise OmegaContractError("planner weights/minimum value cannot be negative")

    def _risk(self, belief: HybridBeliefState) -> float:
        return belief.risk(self.risk_weights)

    def _loss(self, belief: HybridBeliefState) -> float:
        normalizer = math.log(len(FailureMode))
        uncertainty = belief.entropy() / normalizer if normalizer > 0 else 0.0
        return self._risk(belief) + self.entropy_weight * uncertainty

    def _one_step(
        self, belief: HybridBeliefState, action: EvidenceAction
    ) -> tuple[float, float, float]:
        expected_risk = 0.0
        expected_loss = 0.0
        worst_risk = 0.0
        for outcome in ObservationOutcome:
            probability = action.outcome_probability(belief, outcome)
            if probability <= 0:
                continue
            posterior = belief.hypothetical_posterior(action.likelihoods(outcome))
            risk = self._risk(posterior)
            expected_risk += probability * risk
            expected_loss += probability * self._loss(posterior)
            worst_risk = max(worst_risk, risk)
        return expected_risk, expected_loss, worst_risk

    def _evaluate(
        self,
        belief: HybridBeliefState,
        first: EvidenceAction,
        candidates: tuple[EvidenceAction, ...],
        horizon: int,
    ) -> ActionEvaluation:
        initial_loss = self._loss(belief)
        expected_terminal_risk = 0.0
        expected_terminal_loss = 0.0
        expected_second_cost = 0.0
        worst_case_risk = 0.0
        branches: list[PlanBranch] = []
        for outcome in ObservationOutcome:
            branch_probability = first.outcome_probability(belief, outcome)
            posterior = belief.hypothetical_posterior(first.likelihoods(outcome))
            posterior_risk = self._risk(posterior)
            chosen_second_type = EvidenceActionType.HOLD
            branch_terminal_risk = posterior_risk
            branch_terminal_loss = self._loss(posterior)
            branch_second_cost = 0.0
            if horizon == 2:
                best_objective = branch_terminal_loss
                for second in candidates:
                    second_risk, second_loss, _ = self._one_step(posterior, second)
                    objective = second_loss + self.cost_weight * second.cost
                    candidate_key = (objective, second.action_id)
                    best_key = (
                        best_objective,
                        ""
                        if chosen_second_type is EvidenceActionType.HOLD
                        else chosen_second_type.value,
                    )
                    if candidate_key < best_key:
                        best_objective = objective
                        chosen_second_type = second.action_type
                        branch_terminal_risk = second_risk
                        branch_terminal_loss = second_loss
                        branch_second_cost = second.cost
            expected_terminal_risk += branch_probability * branch_terminal_risk
            expected_terminal_loss += branch_probability * branch_terminal_loss
            expected_second_cost += branch_probability * branch_second_cost
            worst_case_risk = max(worst_case_risk, branch_terminal_risk)
            branches.append(
                PlanBranch(
                    outcome=outcome,
                    probability=branch_probability,
                    posterior_belief_sha256=posterior.state_sha256,
                    posterior_risk=posterior_risk,
                    second_action=chosen_second_type,
                    terminal_expected_risk=branch_terminal_risk,
                    terminal_expected_loss=branch_terminal_loss,
                )
            )
        total_cost = first.cost + expected_second_cost
        objective = expected_terminal_loss + self.cost_weight * total_cost
        return ActionEvaluation(
            action_id=first.action_id,
            action_type=first.action_type,
            horizon=horizon,
            branches=tuple(branches),
            expected_terminal_risk=expected_terminal_risk,
            worst_case_terminal_risk=worst_case_risk,
            expected_total_cost=total_cost,
            objective=objective,
            value_of_evidence=initial_loss - objective,
        )

    def _hold(
        self,
        *,
        horizon: int,
        dag: EvidenceDAG,
        belief: HybridBeliefState,
        core: CounterfactualCoreResult,
        status: str,
        reason: str,
    ) -> RbVoePlan:
        risk = self._risk(belief)
        return RbVoePlan(
            status=status,
            horizon=horizon,
            selected_action=EvidenceActionType.HOLD,
            selected_action_id=None,
            dag_root_sha256=dag.root_sha256,
            belief_sha256=belief.state_sha256,
            failure_core_sha256=core.core_sha256,
            initial_risk=risk,
            expected_terminal_risk=risk,
            worst_case_terminal_risk=risk,
            value_of_evidence=0.0,
            reason_code=reason,
            evaluation=None,
        )

    def plan(
        self,
        *,
        dag: EvidenceDAG,
        belief: HybridBeliefState,
        failure_core: CounterfactualCoreResult,
        horizon: int = 2,
    ) -> RbVoePlan:
        if horizon not in {1, 2}:
            raise OmegaContractError("RB-VoE supports only exact H=1 or H=2")
        snapshot = dag.validate()
        if failure_core.dag_root_sha256 != snapshot.root_sha256:
            raise OmegaContractError("failure core is bound to another DAG root")
        if failure_core.belief_sha256 != belief.state_sha256:
            raise OmegaContractError("failure core is bound to another belief")
        if failure_core.status is CoreStatus.BLOCKING:
            return self._hold(
                horizon=horizon,
                dag=dag,
                belief=belief,
                core=failure_core,
                status="HOLD_BLOCKING",
                reason="BLOCKING_FAILURE_CORE_REQUIRES_MANUAL_SAFE_PATH",
            )
        if failure_core.status is CoreStatus.CLEAR:
            return self._hold(
                horizon=horizon,
                dag=dag,
                belief=belief,
                core=failure_core,
                status="HOLD_CLEAR",
                reason="NO_ADDITIONAL_EVIDENCE_REQUIRED",
            )

        requested = set(failure_core.requested_actions)
        candidates = tuple(
            action for action in self.actions if action.action_type in requested
        )
        if not candidates:
            return self._hold(
                horizon=horizon,
                dag=dag,
                belief=belief,
                core=failure_core,
                status="HOLD_NO_ACTION",
                reason="NO_CATALOG_ACTION_FOR_FAILURE_CORE",
            )
        evaluations = tuple(
            self._evaluate(belief, action, candidates, horizon)
            for action in candidates
        )
        best = min(evaluations, key=lambda item: (item.objective, item.action_id))
        if best.value_of_evidence <= self.minimum_value:
            return self._hold(
                horizon=horizon,
                dag=dag,
                belief=belief,
                core=failure_core,
                status="HOLD_NONPOSITIVE_VOE",
                reason="VALUE_OF_EVIDENCE_BELOW_FROZEN_MINIMUM",
            )
        if best.expected_terminal_risk > self.max_expected_residual_risk:
            return self._hold(
                horizon=horizon,
                dag=dag,
                belief=belief,
                core=failure_core,
                status="HOLD_RISK_BOUND",
                reason="EXPECTED_RESIDUAL_RISK_EXCEEDS_BOUND",
            )
        return RbVoePlan(
            status=f"PLAN_H{horizon}",
            horizon=horizon,
            selected_action=best.action_type,
            selected_action_id=best.action_id,
            dag_root_sha256=snapshot.root_sha256,
            belief_sha256=belief.state_sha256,
            failure_core_sha256=failure_core.core_sha256,
            initial_risk=self._risk(belief),
            expected_terminal_risk=best.expected_terminal_risk,
            worst_case_terminal_risk=best.worst_case_terminal_risk,
            value_of_evidence=best.value_of_evidence,
            reason_code="POSITIVE_RISK_BOUNDED_VALUE_OF_EVIDENCE",
            evaluation=best,
        )
