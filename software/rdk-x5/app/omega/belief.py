"""Hybrid discrete/continuous belief state for RootScope-Ω."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .evidence_dag import EvidenceDAG
from .schemas import (
    AuthorityBoundary,
    CalibrationLevel,
    FailureMode,
    OmegaContractError,
    canonical_sha256,
    enum_value,
    require_exact_keys,
    require_finite,
    require_probability,
    require_safe_token,
    require_sha256,
)


BELIEF_SCHEMA = "rootscope.omega.hybrid-belief-state.v1"
CONTINUOUS_SCHEMA = "rootscope.omega.continuous-estimate.v1"
MEASUREMENT_SCHEMA = "rootscope.omega.bounded-measurement.v1"
_NORMALIZATION_TOLERANCE = 1e-9


class BeliefUpdateError(OmegaContractError):
    pass


def _strict_probability_vector(
    probabilities: Mapping[FailureMode | str, Any],
    *,
    context: str,
    normalize: bool,
) -> tuple[tuple[FailureMode, float], ...]:
    if not isinstance(probabilities, Mapping):
        raise BeliefUpdateError(f"{context} must be an object")
    parsed: dict[FailureMode, float] = {}
    for raw_mode, raw_value in probabilities.items():
        mode = (
            raw_mode
            if isinstance(raw_mode, FailureMode)
            else enum_value(FailureMode, raw_mode, f"{context} key")
        )
        if mode in parsed:
            raise BeliefUpdateError(f"{context} contains duplicate mode {mode.value}")
        parsed[mode] = require_probability(raw_value, f"{context}.{mode.value}")
    expected = set(FailureMode)
    if set(parsed) != expected:
        raise BeliefUpdateError(
            f"{context} must exactly cover all modes; "
            f"missing={sorted(mode.value for mode in expected - set(parsed))}"
        )
    total = math.fsum(parsed.values())
    if total <= 0:
        raise BeliefUpdateError(f"{context} cannot have zero total mass")
    if normalize:
        parsed = {mode: value / total for mode, value in parsed.items()}
    elif abs(total - 1.0) > _NORMALIZATION_TOLERANCE:
        raise BeliefUpdateError(f"{context} must sum to one, got {total!r}")
    return tuple(sorted(parsed.items(), key=lambda pair: pair[0].value))


@dataclass(frozen=True)
class ContinuousEstimate:
    name: str
    mean: float
    variance: float
    lower: float
    upper: float
    hard_lower: float
    hard_upper: float

    def __post_init__(self) -> None:
        require_safe_token(self.name, "continuous estimate name")
        for field_name in (
            "mean",
            "variance",
            "lower",
            "upper",
            "hard_lower",
            "hard_upper",
        ):
            object.__setattr__(
                self, field_name, require_finite(getattr(self, field_name), field_name)
            )
        if self.variance <= 0:
            raise BeliefUpdateError("continuous variance must be positive")
        if not self.hard_lower < self.hard_upper:
            raise BeliefUpdateError("continuous hard bounds must be ordered")
        if not self.hard_lower <= self.lower <= self.mean <= self.upper <= self.hard_upper:
            raise BeliefUpdateError(
                "continuous interval/mean must stay inside ordered hard bounds"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTINUOUS_SCHEMA,
            "name": self.name,
            "mean": self.mean,
            "variance": self.variance,
            "lower": self.lower,
            "upper": self.upper,
            "hard_lower": self.hard_lower,
            "hard_upper": self.hard_upper,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContinuousEstimate":
        expected = {
            "schema_version",
            "name",
            "mean",
            "variance",
            "lower",
            "upper",
            "hard_lower",
            "hard_upper",
        }
        require_exact_keys(value, expected, "continuous estimate")
        if value["schema_version"] != CONTINUOUS_SCHEMA:
            raise BeliefUpdateError("continuous estimate schema mismatch")
        return cls(
            name=value["name"],
            mean=value["mean"],
            variance=value["variance"],
            lower=value["lower"],
            upper=value["upper"],
            hard_lower=value["hard_lower"],
            hard_upper=value["hard_upper"],
        )


@dataclass(frozen=True)
class BoundedMeasurement:
    name: str
    value: float
    variance: float
    lower: float
    upper: float
    hard_lower: float
    hard_upper: float

    def __post_init__(self) -> None:
        require_safe_token(self.name, "measurement name")
        for field_name in (
            "value",
            "variance",
            "lower",
            "upper",
            "hard_lower",
            "hard_upper",
        ):
            object.__setattr__(
                self, field_name, require_finite(getattr(self, field_name), field_name)
            )
        if self.variance <= 0:
            raise BeliefUpdateError("measurement variance must be positive")
        if not self.hard_lower < self.hard_upper:
            raise BeliefUpdateError("measurement hard bounds must be ordered")
        if not self.hard_lower <= self.lower <= self.value <= self.upper <= self.hard_upper:
            raise BeliefUpdateError("measurement interval/value violates hard bounds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MEASUREMENT_SCHEMA,
            "name": self.name,
            "value": self.value,
            "variance": self.variance,
            "lower": self.lower,
            "upper": self.upper,
            "hard_lower": self.hard_lower,
            "hard_upper": self.hard_upper,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BoundedMeasurement":
        expected = {
            "schema_version",
            "name",
            "value",
            "variance",
            "lower",
            "upper",
            "hard_lower",
            "hard_upper",
        }
        require_exact_keys(value, expected, "bounded measurement")
        if value["schema_version"] != MEASUREMENT_SCHEMA:
            raise BeliefUpdateError("bounded measurement schema mismatch")
        return cls(
            name=value["name"],
            value=value["value"],
            variance=value["variance"],
            lower=value["lower"],
            upper=value["upper"],
            hard_lower=value["hard_lower"],
            hard_upper=value["hard_upper"],
        )


DEFAULT_PRIOR: Mapping[FailureMode, float] = {
    FailureMode.NORMAL: 0.45,
    FailureMode.PERCEPTION_ERROR: 0.08,
    FailureMode.CAMERA_QUALITY: 0.06,
    FailureMode.SENSOR_STALE: 0.06,
    FailureMode.ACTUATOR_NO_ACK: 0.05,
    FailureMode.MASS_ANOMALY: 0.08,
    FailureMode.WETTING_MISS: 0.08,
    FailureMode.NEIGHBOR_SPILL: 0.06,
    FailureMode.SAFETY_INTERLOCK: 0.05,
    FailureMode.OOD_CONFLICT: 0.03,
}


@dataclass(frozen=True)
class HybridBeliefState:
    belief_id: str
    revision: int
    probabilities: tuple[tuple[FailureMode, float], ...]
    continuous: tuple[ContinuousEstimate, ...]
    evidence_node_ids: tuple[str, ...]
    calibration_level: CalibrationLevel
    authority: AuthorityBoundary = field(default_factory=AuthorityBoundary)
    state_sha256: str = ""

    def __post_init__(self) -> None:
        require_safe_token(self.belief_id, "belief_id")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise BeliefUpdateError("revision must be a non-negative integer")
        if not isinstance(self.calibration_level, CalibrationLevel):
            raise BeliefUpdateError("calibration_level must be CalibrationLevel")
        if not isinstance(self.probabilities, tuple):
            raise BeliefUpdateError("probabilities must be an immutable tuple")
        normalized = _strict_probability_vector(
            dict(self.probabilities), context="belief probabilities", normalize=False
        )
        if normalized != self.probabilities:
            raise BeliefUpdateError("belief probabilities must use canonical mode order")
        names = tuple(item.name for item in self.continuous)
        if tuple(sorted(set(names))) != names:
            raise BeliefUpdateError("continuous estimates must be unique and sorted")
        if tuple(sorted(set(self.evidence_node_ids))) != self.evidence_node_ids:
            raise BeliefUpdateError("evidence_node_ids must be unique and sorted")
        if not isinstance(self.authority, AuthorityBoundary):
            raise BeliefUpdateError("authority must be zero-authority capsule")
        expected = canonical_sha256(self.unsigned_dict())
        if not self.state_sha256:
            object.__setattr__(self, "state_sha256", expected)
        elif require_sha256(self.state_sha256, "state_sha256") != expected:
            raise BeliefUpdateError("belief state hash mismatch")

    @property
    def belief_hash(self) -> str:
        return self.state_sha256

    @property
    def probability_map(self) -> dict[FailureMode, float]:
        return dict(self.probabilities)

    @property
    def continuous_map(self) -> dict[str, ContinuousEstimate]:
        return {item.name: item for item in self.continuous}

    def probability(self, mode: FailureMode) -> float:
        return self.probability_map[mode]

    def entropy(self) -> float:
        return -math.fsum(
            probability * math.log(probability)
            for _, probability in self.probabilities
            if probability > 0
        )

    def risk(self, weights: Mapping[FailureMode, float]) -> float:
        if set(weights) != set(FailureMode):
            raise BeliefUpdateError("risk weights must exactly cover all failure modes")
        values = []
        for mode, probability in self.probabilities:
            weight = require_probability(weights[mode], f"risk weight {mode.value}")
            values.append(probability * weight)
        return math.fsum(values)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BELIEF_SCHEMA,
            "belief_id": self.belief_id,
            "revision": self.revision,
            "probabilities": {
                mode.value: probability for mode, probability in self.probabilities
            },
            "continuous": [item.to_dict() for item in self.continuous],
            "evidence_node_ids": list(self.evidence_node_ids),
            "calibration_level": self.calibration_level.value,
            "authority": self.authority.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "state_sha256": self.state_sha256}

    @classmethod
    def create(
        cls,
        *,
        belief_id: str = "rootscope-belief",
        probabilities: Mapping[FailureMode | str, Any] = DEFAULT_PRIOR,
        continuous: Iterable[ContinuousEstimate] = (),
        calibration_level: CalibrationLevel = CalibrationLevel.INTERVAL_ONLY,
    ) -> "HybridBeliefState":
        return cls(
            belief_id=belief_id,
            revision=0,
            probabilities=_strict_probability_vector(
                probabilities, context="initial probabilities", normalize=False
            ),
            continuous=tuple(sorted(continuous, key=lambda item: item.name)),
            evidence_node_ids=(),
            calibration_level=calibration_level,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HybridBeliefState":
        expected = {
            "schema_version",
            "belief_id",
            "revision",
            "probabilities",
            "continuous",
            "evidence_node_ids",
            "calibration_level",
            "authority",
            "state_sha256",
        }
        require_exact_keys(value, expected, "hybrid belief state")
        if value["schema_version"] != BELIEF_SCHEMA:
            raise BeliefUpdateError("belief schema mismatch")
        if not isinstance(value["continuous"], list) or not isinstance(
            value["evidence_node_ids"], list
        ):
            raise BeliefUpdateError("belief arrays have invalid types")
        return cls(
            belief_id=value["belief_id"],
            revision=value["revision"],
            probabilities=_strict_probability_vector(
                value["probabilities"],
                context="belief probabilities",
                normalize=False,
            ),
            continuous=tuple(
                ContinuousEstimate.from_dict(item) for item in value["continuous"]
            ),
            evidence_node_ids=tuple(value["evidence_node_ids"]),
            calibration_level=enum_value(
                CalibrationLevel, value["calibration_level"], "calibration_level"
            ),  # type: ignore[arg-type]
            authority=AuthorityBoundary.from_dict(value["authority"]),
            state_sha256=value["state_sha256"],
        )

    def posterior_from_likelihoods(
        self, likelihoods: Mapping[FailureMode | str, Any]
    ) -> tuple[tuple[FailureMode, float], ...]:
        parsed = _strict_probability_vector(
            likelihoods, context="observation likelihoods", normalize=True
        )
        # `_strict_probability_vector(normalize=True)` normalizes likelihoods as
        # a vector, which is harmless for Bayes because a common scale cancels.
        likelihood_map = dict(parsed)
        unnormalized = {
            mode: probability * likelihood_map[mode]
            for mode, probability in self.probabilities
        }
        total = math.fsum(unnormalized.values())
        if total <= 1e-15:
            raise BeliefUpdateError("observation has zero likelihood under prior")
        return tuple(
            sorted(
                ((mode, value / total) for mode, value in unnormalized.items()),
                key=lambda pair: pair[0].value,
            )
        )

    def update(
        self,
        *,
        dag: EvidenceDAG,
        evidence_node_id: str,
        likelihoods: Mapping[FailureMode | str, Any],
        measurements: Iterable[BoundedMeasurement] = (),
    ) -> "HybridBeliefState":
        if not isinstance(dag, EvidenceDAG):
            raise BeliefUpdateError("dag must be EvidenceDAG")
        dag.get(evidence_node_id)
        posterior = self.posterior_from_likelihoods(likelihoods)
        estimates = self.continuous_map
        seen_measurements: set[str] = set()
        for measurement in measurements:
            if not isinstance(measurement, BoundedMeasurement):
                raise BeliefUpdateError("measurements must be BoundedMeasurement")
            if measurement.name in seen_measurements:
                raise BeliefUpdateError("duplicate continuous measurement")
            seen_measurements.add(measurement.name)
            previous = estimates.get(measurement.name)
            if previous is None:
                estimates[measurement.name] = ContinuousEstimate(
                    name=measurement.name,
                    mean=measurement.value,
                    variance=measurement.variance,
                    lower=measurement.lower,
                    upper=measurement.upper,
                    hard_lower=measurement.hard_lower,
                    hard_upper=measurement.hard_upper,
                )
                continue
            if (
                previous.hard_lower != measurement.hard_lower
                or previous.hard_upper != measurement.hard_upper
            ):
                raise BeliefUpdateError("continuous hard bounds changed across update")
            lower = max(previous.lower, measurement.lower)
            upper = min(previous.upper, measurement.upper)
            if lower > upper:
                raise BeliefUpdateError(
                    "continuous evidence intervals are disjoint; refusing false precision"
                )
            precision_prior = 1.0 / previous.variance
            precision_measurement = 1.0 / measurement.variance
            variance = 1.0 / (precision_prior + precision_measurement)
            mean = variance * (
                previous.mean * precision_prior
                + measurement.value * precision_measurement
            )
            # Numerical rounding cannot be promoted beyond the interval.
            mean = min(max(mean, lower), upper)
            estimates[measurement.name] = ContinuousEstimate(
                name=measurement.name,
                mean=mean,
                variance=variance,
                lower=lower,
                upper=upper,
                hard_lower=previous.hard_lower,
                hard_upper=previous.hard_upper,
            )
        evidence_ids = tuple(sorted({*self.evidence_node_ids, evidence_node_id}))
        return HybridBeliefState(
            belief_id=self.belief_id,
            revision=self.revision + 1,
            probabilities=posterior,
            continuous=tuple(sorted(estimates.values(), key=lambda item: item.name)),
            evidence_node_ids=evidence_ids,
            calibration_level=self.calibration_level,
        )

    def hypothetical_posterior(
        self, likelihoods: Mapping[FailureMode | str, Any]
    ) -> "HybridBeliefState":
        """Posterior used inside a side-effect-free evidence planning branch."""

        return HybridBeliefState(
            belief_id=self.belief_id,
            revision=self.revision + 1,
            probabilities=self.posterior_from_likelihoods(likelihoods),
            continuous=self.continuous,
            evidence_node_ids=self.evidence_node_ids,
            calibration_level=self.calibration_level,
        )


# Compact runtime-facing name.
BeliefState = HybridBeliefState
