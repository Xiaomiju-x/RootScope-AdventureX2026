"""Deterministic RootScope-Ω evidence twin.

This twin evaluates evidence consistency only.  It never emits a serial frame,
opens a device, or upgrades a replay result into a physical completion claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .contracts import DecisionProjection, EvidenceAction, SafetyDecision


def _finite_non_negative(value: float | int, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class TwinCaseInput:
    camera_quality_ok: bool
    ood_detected: bool
    evidence_fresh: bool
    payload_hash_valid: bool
    firmware_connected: bool
    estop_clear: bool
    ack_ok: bool
    target_mass_mg: int
    tolerance_mg: int
    measured_mass_loss_mg: int
    target_wetting_score: float
    target_wetting_threshold: float
    neighbor_wetting_score: float
    neighbor_spill_threshold: float

    def __post_init__(self) -> None:
        for name in (
            "camera_quality_ok",
            "ood_detected",
            "evidence_fresh",
            "payload_hash_valid",
            "firmware_connected",
            "estop_clear",
            "ack_ok",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("target_mass_mg", "tolerance_mg", "measured_mass_loss_mg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.target_mass_mg <= 0:
            raise ValueError("target_mass_mg must be positive")
        if self.tolerance_mg >= self.target_mass_mg:
            raise ValueError("tolerance_mg must be below target_mass_mg")
        for name in (
            "target_wetting_score",
            "target_wetting_threshold",
            "neighbor_wetting_score",
            "neighbor_spill_threshold",
        ):
            value = _finite_non_negative(getattr(self, name), name)
            if value > 1.0:
                raise ValueError(f"{name} must be within [0, 1]")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TwinCaseInput":
        required = {
            "camera_quality_ok",
            "ood_detected",
            "evidence_fresh",
            "payload_hash_valid",
            "firmware_connected",
            "estop_clear",
            "ack_ok",
            "target_mass_mg",
            "tolerance_mg",
            "measured_mass_loss_mg",
            "target_wetting_score",
            "target_wetting_threshold",
            "neighbor_wetting_score",
            "neighbor_spill_threshold",
        }
        if set(payload) != required:
            raise ValueError(
                f"invalid twin input keys; missing={sorted(required - set(payload))}, "
                f"unknown={sorted(set(payload) - required)}"
            )
        return cls(**dict(payload))

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "camera_quality_ok": self.camera_quality_ok,
            "ood_detected": self.ood_detected,
            "evidence_fresh": self.evidence_fresh,
            "payload_hash_valid": self.payload_hash_valid,
            "firmware_connected": self.firmware_connected,
            "estop_clear": self.estop_clear,
            "ack_ok": self.ack_ok,
            "target_mass_mg": self.target_mass_mg,
            "tolerance_mg": self.tolerance_mg,
            "measured_mass_loss_mg": self.measured_mass_loss_mg,
            "target_wetting_score": self.target_wetting_score,
            "target_wetting_threshold": self.target_wetting_threshold,
            "neighbor_wetting_score": self.neighbor_wetting_score,
            "neighbor_spill_threshold": self.neighbor_spill_threshold,
        }


@dataclass(frozen=True)
class TwinEvaluation:
    projection: DecisionProjection
    invariants: Tuple[str, ...]
    mass_error_mg: int

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "projection": self.projection.to_dict(),
            "invariants": list(self.invariants),
            "mass_error_mg": self.mass_error_mg,
        }


def evaluate_case(case: TwinCaseInput) -> TwinEvaluation:
    reasons: list[str] = []
    if not case.evidence_fresh:
        reasons.append("EVIDENCE_STALE_OR_REPLAYED")
    if not case.payload_hash_valid:
        reasons.append("PAYLOAD_HASH_INVALID")
    if not case.firmware_connected:
        reasons.append("FIRMWARE_DISCONNECTED")
    if not case.estop_clear:
        reasons.append("ESTOP_ACTIVE")
    if reasons:
        return TwinEvaluation(
            projection=DecisionProjection(
                SafetyDecision.REJECT,
                tuple(reasons),
                EvidenceAction.HOLD,
                "ABORTED_LOCKED",
                "NO_COMPLETION",
            ),
            invariants=(
                "FAIL_CLOSED_ON_IDENTITY_FRESHNESS_OR_ESTOP",
                "NO_AUTOMATIC_RECOVERY_AFTER_LOCK",
                "NO_PHYSICAL_COMMAND_EMITTED",
            ),
            mass_error_mg=abs(
                case.measured_mass_loss_mg - case.target_mass_mg
            ),
        )

    if case.ood_detected or not case.camera_quality_ok:
        quality_reasons = []
        if case.ood_detected:
            quality_reasons.append("PERCEPTION_OOD")
        if not case.camera_quality_ok:
            quality_reasons.append("CAMERA_QUALITY_INVALID")
        return TwinEvaluation(
            projection=DecisionProjection(
                SafetyDecision.HOLD,
                tuple(quality_reasons),
                EvidenceAction.RECAPTURE,
                "READY",
                "NO_COMPLETION",
            ),
            invariants=(
                "OOD_NEVER_AUTO_ACCEPTED",
                "RECAPTURE_PRECEDES_ANY_DOSE_PROPOSAL",
                "NO_PHYSICAL_COMMAND_EMITTED",
            ),
            mass_error_mg=abs(
                case.measured_mass_loss_mg - case.target_mass_mg
            ),
        )

    if not case.ack_ok:
        return TwinEvaluation(
            projection=DecisionProjection(
                SafetyDecision.REJECT,
                ("ACTUATOR_ACK_MISSING",),
                EvidenceAction.HOLD,
                "ABORTED_LOCKED",
                "NO_COMPLETION",
            ),
            invariants=(
                "ACK_REQUIRED_BUT_NOT_SUFFICIENT",
                "NO_AUTOMATIC_RECOVERY_AFTER_LOCK",
                "NO_PHYSICAL_COMMAND_EMITTED",
            ),
            mass_error_mg=abs(
                case.measured_mass_loss_mg - case.target_mass_mg
            ),
        )

    mass_error_mg = abs(case.measured_mass_loss_mg - case.target_mass_mg)
    if mass_error_mg > case.tolerance_mg:
        return TwinEvaluation(
            projection=DecisionProjection(
                SafetyDecision.REJECT,
                ("ACK_WITHOUT_QUALIFIED_MASS_LOSS",),
                EvidenceAction.REWEIGH,
                "ABORTED_LOCKED",
                "NO_COMPLETION",
            ),
            invariants=(
                "ACK_ALONE_NEVER_COMPLETES_TASK",
                "MASS_LOSS_MUST_MATCH_FROZEN_TOLERANCE",
                "NO_PHYSICAL_COMMAND_EMITTED",
            ),
            mass_error_mg=mass_error_mg,
        )

    if case.neighbor_wetting_score > case.neighbor_spill_threshold:
        return TwinEvaluation(
            projection=DecisionProjection(
                SafetyDecision.REJECT,
                ("NEIGHBOR_SPILL_OR_CROSSTALK",),
                EvidenceAction.HOLD,
                "ABORTED_LOCKED",
                "NO_COMPLETION",
            ),
            invariants=(
                "NEIGHBOR_SPILL_OVERRIDES_TARGET_WETTING",
                "NO_AUTOMATIC_RECOVERY_AFTER_LOCK",
                "NO_PHYSICAL_COMMAND_EMITTED",
            ),
            mass_error_mg=mass_error_mg,
        )

    if case.target_wetting_score < case.target_wetting_threshold:
        return TwinEvaluation(
            projection=DecisionProjection(
                SafetyDecision.HOLD,
                ("TARGET_WETTING_NOT_YET_VERIFIED",),
                EvidenceAction.WAIT,
                "VERIFYING",
                "NO_COMPLETION",
            ),
            invariants=(
                "MASS_LOSS_ALONE_NEVER_COMPLETES_TASK",
                "TARGET_WETTING_EVIDENCE_REQUIRED",
                "NO_PHYSICAL_COMMAND_EMITTED",
            ),
            mass_error_mg=mass_error_mg,
        )

    return TwinEvaluation(
        projection=DecisionProjection(
            SafetyDecision.ACCEPT,
            ("ALL_SIMULATED_EVIDENCE_GATES_PASSED",),
            EvidenceAction.NONE,
            "TARGET_WETTING_VERIFIED",
            "SIMULATED_EVIDENCE_COMPLETE",
        ),
        invariants=(
            "ACK_MASS_AND_WETTING_EVIDENCE_AGREE",
            "NEIGHBOR_SPILL_GATE_PASSED",
            "SIMULATION_IS_NOT_PHYSICAL_COMPLETION",
            "NO_PHYSICAL_COMMAND_EMITTED",
        ),
        mass_error_mg=mass_error_mg,
    )
