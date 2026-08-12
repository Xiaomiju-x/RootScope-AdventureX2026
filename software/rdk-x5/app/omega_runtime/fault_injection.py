"""Fifteen deterministic fail-closed mutations for the Ω safety compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .contracts import AuthorityFlags, canonical_sha256
from .digital_twin import TwinCaseInput, evaluate_case


@dataclass(frozen=True)
class FaultResult:
    fault_id: str
    outcome: str
    reason_codes: Tuple[str, ...]
    unsafe_accept: bool

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "fault_id": self.fault_id,
            "outcome": self.outcome,
            "reason_codes": list(self.reason_codes),
            "unsafe_accept": self.unsafe_accept,
            "authority": AuthorityFlags().to_dict(),
        }


_MUTATIONS: Tuple[Tuple[str, str, Any], ...] = (
    ("FI01_CAMERA_QUALITY", "camera_quality_ok", False),
    ("FI02_OOD", "ood_detected", True),
    ("FI03_STALE", "evidence_fresh", False),
    ("FI04_HASH_TAMPER", "payload_hash_valid", False),
    ("FI05_FIRMWARE_DISCONNECT", "firmware_connected", False),
    ("FI06_ESTOP", "estop_clear", False),
    ("FI07_ACK_MISSING", "ack_ok", False),
    ("FI08_MASS_UNDER", "measured_mass_loss_mg", 0),
    ("FI09_MASS_OVER", "measured_mass_loss_mg", 20000),
    ("FI10_WETTING_MISS", "target_wetting_score", 0.05),
    ("FI11_NEIGHBOR_SPILL", "neighbor_wetting_score", 0.50),
    ("FI12_NAN", "target_wetting_score", float("nan")),
    ("FI13_NEGATIVE_MASS", "measured_mass_loss_mg", -1),
    ("FI14_BAD_TOLERANCE", "tolerance_mg", 12000),
    ("FI15_UNKNOWN_FIELD", "serial_command", "PUMP_ON"),
)


def run_fault_injection(normal_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    # The control fixture itself must be a valid ACCEPT case.
    baseline = evaluate_case(TwinCaseInput.from_mapping(normal_payload))
    if baseline.projection.safety_decision.value != "ACCEPT":
        raise ValueError("fault injection baseline must be a valid ACCEPT fixture")
    results = []
    for fault_id, field_name, value in _MUTATIONS:
        mutated = dict(normal_payload)
        mutated[field_name] = value
        try:
            projection = evaluate_case(TwinCaseInput.from_mapping(mutated)).projection
            outcome = projection.safety_decision.value
            reasons = projection.reason_codes
        except (TypeError, ValueError) as exc:
            outcome = "CONTRACT_REJECTED"
            reasons = (type(exc).__name__.upper(),)
        results.append(
            FaultResult(
                fault_id=fault_id,
                outcome=outcome,
                reason_codes=tuple(reasons),
                unsafe_accept=outcome == "ACCEPT",
            )
        )
    report = {
        "schema_version": "rootscope.omega.fault-injection-report.v1",
        "fault_count": len(results),
        "unsafe_accept_count": sum(item.unsafe_accept for item in results),
        "passed": all(not item.unsafe_accept for item in results),
        "results": [item.to_dict() for item in results],
        "authority": AuthorityFlags().to_dict(),
        "hardware_touched": False,
        "physical_command_count": 0,
    }
    return {**report, "report_sha256": canonical_sha256(report)}
