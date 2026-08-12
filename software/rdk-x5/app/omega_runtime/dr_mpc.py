"""Interval distributionally-robust MPC proposal kernel.

The uncertainty set is the frozen interval of target response and worst-case
neighbor crosstalk.  The enumerator minimizes a deterministic robust objective.
It emits a zero-authority DoseIntent proposal, never a hardware command.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

from .contracts import AuthorityFlags, canonical_sha256


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class DrMpcScenario:
    scenario_id: str
    target_zone: int
    moisture_now: Tuple[float, float, float]
    target_band: Tuple[float, float]
    target_gain_bounds_per_mg: Tuple[float, float]
    neighbor_crosstalk_upper_per_mg: Tuple[float, float, float]
    neighbor_spill_limit: float
    pulse_candidates_mg: Tuple[int, ...]
    horizon: int
    maximum_total_dose_mg: int

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("scenario_id is required")
        if self.target_zone not in {0, 1, 2}:
            raise ValueError("target_zone must be 0, 1 or 2")
        if len(self.moisture_now) != 3:
            raise ValueError("moisture_now must contain three zones")
        moisture = tuple(_finite(value, "moisture_now") for value in self.moisture_now)
        if any(not 0.0 <= value <= 1.0 for value in moisture):
            raise ValueError("moisture values must be within [0, 1]")
        lower, upper = (
            _finite(self.target_band[0], "target_band"),
            _finite(self.target_band[1], "target_band"),
        )
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("target_band must be ordered within [0, 1]")
        gain_low, gain_high = (
            _finite(self.target_gain_bounds_per_mg[0], "target_gain"),
            _finite(self.target_gain_bounds_per_mg[1], "target_gain"),
        )
        if not 0.0 < gain_low <= gain_high:
            raise ValueError("target gain bounds must be positive and ordered")
        crosstalk = tuple(
            _finite(value, "neighbor_crosstalk") for value in self.neighbor_crosstalk_upper_per_mg
        )
        if len(crosstalk) != 3 or any(value < 0 for value in crosstalk):
            raise ValueError("crosstalk upper bounds must be three non-negative values")
        if crosstalk[self.target_zone] != 0.0:
            raise ValueError("target-zone crosstalk coefficient must be zero")
        spill = _finite(self.neighbor_spill_limit, "neighbor_spill_limit")
        if not 0.0 < spill <= 1.0:
            raise ValueError("neighbor_spill_limit must be within (0, 1]")
        if self.horizon not in {1, 2, 3}:
            raise ValueError("horizon must be 1, 2 or 3")
        if (
            not self.pulse_candidates_mg
            or tuple(sorted(set(self.pulse_candidates_mg))) != self.pulse_candidates_mg
            or self.pulse_candidates_mg[0] != 0
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.pulse_candidates_mg
            )
        ):
            raise ValueError("pulse candidates must be sorted unique non-negative integers")
        if (
            isinstance(self.maximum_total_dose_mg, bool)
            or not isinstance(self.maximum_total_dose_mg, int)
            or self.maximum_total_dose_mg <= 0
        ):
            raise ValueError("maximum_total_dose_mg must be positive integer")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DrMpcScenario":
        expected = {
            "scenario_id",
            "target_zone",
            "moisture_now",
            "target_band",
            "target_gain_bounds_per_mg",
            "neighbor_crosstalk_upper_per_mg",
            "neighbor_spill_limit",
            "pulse_candidates_mg",
            "horizon",
            "maximum_total_dose_mg",
        }
        if set(payload) != expected:
            raise ValueError("DR-MPC scenario has unknown or missing fields")
        return cls(
            scenario_id=payload["scenario_id"],
            target_zone=payload["target_zone"],
            moisture_now=tuple(payload["moisture_now"]),
            target_band=tuple(payload["target_band"]),
            target_gain_bounds_per_mg=tuple(payload["target_gain_bounds_per_mg"]),
            neighbor_crosstalk_upper_per_mg=tuple(
                payload["neighbor_crosstalk_upper_per_mg"]
            ),
            neighbor_spill_limit=payload["neighbor_spill_limit"],
            pulse_candidates_mg=tuple(payload["pulse_candidates_mg"]),
            horizon=payload["horizon"],
            maximum_total_dose_mg=payload["maximum_total_dose_mg"],
        )


@dataclass(frozen=True)
class DrMpcProposal:
    scenario_id: str
    status: str
    reason_code: str
    target_zone: int
    pulse_sequence_mg: Tuple[int, ...]
    total_dose_mg: int
    robust_target_interval: Tuple[float, float]
    robust_neighbor_upper: Tuple[float, float, float]
    objective: float
    feasible_sequence_count: int
    authority: AuthorityFlags = field(default_factory=AuthorityFlags)
    proposal_sha256: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"PROPOSAL", "HOLD"}:
            raise ValueError("status must be PROPOSAL or HOLD")
        if self.status == "HOLD" and self.total_dose_mg != 0:
            raise ValueError("HOLD cannot carry a nonzero dose")
        if self.status == "PROPOSAL" and self.total_dose_mg <= 0:
            raise ValueError("PROPOSAL requires a positive dose")
        if not isinstance(self.authority, AuthorityFlags):
            raise ValueError("authority must be zero-authority flags")
        expected = canonical_sha256(self.unsigned_dict())
        if not self.proposal_sha256:
            object.__setattr__(self, "proposal_sha256", expected)
        elif self.proposal_sha256 != expected:
            raise ValueError("DR-MPC proposal hash mismatch")

    def unsigned_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": "rootscope.omega.dr-mpc-proposal.v1",
            "scenario_id": self.scenario_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "target_zone": self.target_zone,
            "pulse_sequence_mg": list(self.pulse_sequence_mg),
            "total_dose_mg": self.total_dose_mg,
            "robust_target_interval": list(self.robust_target_interval),
            "robust_neighbor_upper": list(self.robust_neighbor_upper),
            "objective": self.objective,
            "feasible_sequence_count": self.feasible_sequence_count,
            "authority": self.authority.to_dict(),
            "physical_command_emitted": False,
            "proposal_only": True,
        }

    def to_dict(self) -> Mapping[str, Any]:
        return {**self.unsigned_dict(), "proposal_sha256": self.proposal_sha256}


def solve_dr_mpc(scenario: DrMpcScenario) -> DrMpcProposal:
    feasible: list[
        tuple[
            float,
            Tuple[int, ...],
            Tuple[float, float],
            Tuple[float, float, float],
        ]
    ] = []
    target_low, target_high = scenario.target_band
    target_mid = (target_low + target_high) / 2.0
    gain_low, gain_high = scenario.target_gain_bounds_per_mg
    for pulses in itertools.product(
        scenario.pulse_candidates_mg, repeat=scenario.horizon
    ):
        total = sum(pulses)
        if total <= 0 or total > scenario.maximum_total_dose_mg:
            continue
        robust_target = (
            scenario.moisture_now[scenario.target_zone] + total * gain_low,
            scenario.moisture_now[scenario.target_zone] + total * gain_high,
        )
        neighbors = tuple(
            scenario.moisture_now[index]
            + total * scenario.neighbor_crosstalk_upper_per_mg[index]
            for index in range(3)
        )
        neighbor_ok = all(
            index == scenario.target_zone
            or neighbors[index] <= scenario.neighbor_spill_limit
            for index in range(3)
        )
        epsilon = 1e-12
        if (
            robust_target[0] < target_low - epsilon
            or robust_target[1] > target_high + epsilon
            or not neighbor_ok
        ):
            continue
        worst_target_error = max(
            abs(robust_target[0] - target_mid),
            abs(robust_target[1] - target_mid),
        )
        neighbor_pressure = max(
            (
                neighbors[index] / scenario.neighbor_spill_limit
                for index in range(3)
                if index != scenario.target_zone
            ),
            default=0.0,
        )
        effort = total / scenario.maximum_total_dose_mg
        objective = worst_target_error + 0.04 * effort + 0.02 * neighbor_pressure
        feasible.append(
            (objective, tuple(pulses), robust_target, tuple(neighbors))
        )
    if not feasible:
        return DrMpcProposal(
            scenario_id=scenario.scenario_id,
            status="HOLD",
            reason_code="NO_ROBUSTLY_FEASIBLE_DOSE",
            target_zone=scenario.target_zone,
            pulse_sequence_mg=tuple(0 for _ in range(scenario.horizon)),
            total_dose_mg=0,
            robust_target_interval=(
                scenario.moisture_now[scenario.target_zone],
                scenario.moisture_now[scenario.target_zone],
            ),
            robust_neighbor_upper=scenario.moisture_now,
            objective=0.0,
            feasible_sequence_count=0,
        )
    objective, pulses, target_interval, neighbors = min(
        feasible, key=lambda item: (item[0], sum(item[1]), item[1])
    )
    return DrMpcProposal(
        scenario_id=scenario.scenario_id,
        status="PROPOSAL",
        reason_code="ROBUST_INTERVAL_CONSTRAINTS_SATISFIED",
        target_zone=scenario.target_zone,
        pulse_sequence_mg=pulses,
        total_dose_mg=sum(pulses),
        robust_target_interval=target_interval,
        robust_neighbor_upper=neighbors,
        objective=objective,
        feasible_sequence_count=len(feasible),
    )
