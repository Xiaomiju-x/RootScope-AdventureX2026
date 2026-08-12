"""Strict Plant2Action contracts with a zero-authority software boundary.

An :class:`ActionContract` is a bounded proposal for a future, separately
qualified single-writer STM32 bridge.  Building it never emits bytes or opens a
device.  A :class:`PhysicalDecisionReceipt` records whether fresh ACK, mass and
wetting evidence jointly support completion; ACK alone is never enough.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
_KNOWN_CLASSES = frozenset({"grass_clump", "low_shrub", "young_tree", "non_target"})
ZERO_AUTHORITY = {
    "execution_authority": False,
    "serial_write": False,
    "gpio_write": False,
    "pump_command": False,
    "state_machine_write": False,
    "physical_completion": False,
}


class ActionContractError(ValueError):
    """Raised when evidence or bounded action fields fail closed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ActionContractError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ActionContractError(f"{name} has an invalid identifier")
    return value


def _finite(value: float, name: str, low: float, high: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ActionContractError(f"{name} must be within [{low}, {high}]")
    return number


@dataclass(frozen=True)
class ActionContract:
    contract_id: str
    sequence: int
    boot_id: str
    release_sha256: str
    config_sha256: str
    evidence_root_sha256: str
    plant_class: str
    plant_confidence: float
    ood_hold: bool
    target_zone: str
    proposed_volume_ml: float
    maximum_volume_ml: float
    maximum_duration_ms: int
    reason_codes: tuple[str, ...]
    proposal_only: bool = True

    def __post_init__(self) -> None:
        _identifier(self.contract_id, "contract_id")
        _identifier(self.boot_id, "boot_id")
        _identifier(self.target_zone, "target_zone")
        if not 0 <= int(self.sequence) <= 2**31 - 1:
            raise ActionContractError("sequence is out of range")
        _sha(self.release_sha256, "release_sha256")
        _sha(self.config_sha256, "config_sha256")
        _sha(self.evidence_root_sha256, "evidence_root_sha256")
        if self.plant_class not in _KNOWN_CLASSES:
            raise ActionContractError("plant_class is not in the frozen ontology")
        _finite(self.plant_confidence, "plant_confidence", 0.0, 1.0)
        proposed = _finite(self.proposed_volume_ml, "proposed_volume_ml", 0.0, 250.0)
        maximum = _finite(self.maximum_volume_ml, "maximum_volume_ml", 0.0, 250.0)
        if proposed > maximum:
            raise ActionContractError("proposed volume exceeds frozen maximum")
        if not 0 <= int(self.maximum_duration_ms) <= 30_000:
            raise ActionContractError("maximum_duration_ms is out of range")
        if not self.reason_codes or len(set(self.reason_codes)) != len(self.reason_codes):
            raise ActionContractError("reason_codes must be non-empty and unique")
        if not self.proposal_only:
            raise ActionContractError("software contract must remain proposal_only")
        if self.ood_hold and proposed != 0.0:
            raise ActionContractError("OOD HOLD must propose zero volume")
        if self.plant_class == "non_target" and proposed != 0.0:
            raise ActionContractError("non-target must propose zero volume")

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["reason_codes"] = list(self.reason_codes)
        return {
            "schema": "rootscope.plant2action.contract.v1",
            **value,
            "authority": dict(ZERO_AUTHORITY),
        }

    @property
    def sha256(self) -> str:
        return _hash(self.payload())


class ActionContractCompiler:
    """Compile a bounded proposal from already-validated evidence."""

    def __init__(
        self,
        *,
        release_sha256: str,
        config_sha256: str,
        maximum_volume_ml: float = 80.0,
        maximum_duration_ms: int = 10_000,
    ) -> None:
        self.release_sha256 = _sha(release_sha256, "release_sha256")
        self.config_sha256 = _sha(config_sha256, "config_sha256")
        self.maximum_volume_ml = _finite(
            maximum_volume_ml, "maximum_volume_ml", 0.0, 250.0
        )
        if not 0 <= maximum_duration_ms <= 30_000:
            raise ActionContractError("maximum_duration_ms is out of range")
        self.maximum_duration_ms = int(maximum_duration_ms)

    def compile(
        self,
        *,
        contract_id: str,
        sequence: int,
        boot_id: str,
        evidence_root_sha256: str,
        plant_class: str,
        plant_confidence: float,
        ood_hold: bool,
        target_zone: str,
        proposed_volume_ml: float,
        evidence_fresh: bool,
        interlocks_clear: bool,
        reason_codes: Sequence[str],
    ) -> ActionContract:
        reasons = set(str(item) for item in reason_codes)
        volume = float(proposed_volume_ml)
        if not evidence_fresh:
            reasons.add("EVIDENCE_STALE")
            volume = 0.0
        if not interlocks_clear:
            reasons.add("INTERLOCK_ACTIVE")
            volume = 0.0
        if ood_hold:
            reasons.add("VISION_OOD_HOLD")
            volume = 0.0
        if plant_class == "non_target":
            reasons.add("NON_TARGET_HOLD")
            volume = 0.0
        return ActionContract(
            contract_id=contract_id,
            sequence=int(sequence),
            boot_id=boot_id,
            release_sha256=self.release_sha256,
            config_sha256=self.config_sha256,
            evidence_root_sha256=_sha(
                evidence_root_sha256, "evidence_root_sha256"
            ),
            plant_class=plant_class,
            plant_confidence=float(plant_confidence),
            ood_hold=bool(ood_hold),
            target_zone=target_zone,
            proposed_volume_ml=volume,
            maximum_volume_ml=self.maximum_volume_ml,
            maximum_duration_ms=self.maximum_duration_ms,
            reason_codes=tuple(sorted(reasons)),
        )


@dataclass(frozen=True)
class PhysicalDecisionReceipt:
    receipt_id: str
    contract_sha256: str
    device_identity_sha256: str
    boot_id: str
    sequence: int
    ack_payload_sha256: str
    ack_fresh: bool
    expected_mass_loss_g: float
    observed_mass_loss_g: float
    target_wetting_coverage: float
    neighbor_spill_ratio: float
    completed: bool
    reason_codes: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["reason_codes"] = list(self.reason_codes)
        return {
            "schema": "rootscope.physical-decision-receipt.v1",
            **value,
            "authority": dict(ZERO_AUTHORITY),
            "claim_boundary": (
                "OBSERVATION_ONLY"
                if self.completed
                else "NO_PHYSICAL_COMPLETION_CLAIM"
            ),
        }

    @property
    def sha256(self) -> str:
        return _hash(self.payload())


class PhysicalReceiptCompiler:
    """Cross-check ACK, identity, mass and wetting evidence deterministically."""

    def __init__(
        self,
        *,
        mass_tolerance_g: float = 4.0,
        minimum_target_coverage: float = 0.18,
        maximum_neighbor_spill: float = 0.08,
    ) -> None:
        self.mass_tolerance_g = _finite(
            mass_tolerance_g, "mass_tolerance_g", 0.0, 100.0
        )
        self.minimum_target_coverage = _finite(
            minimum_target_coverage, "minimum_target_coverage", 0.0, 1.0
        )
        self.maximum_neighbor_spill = _finite(
            maximum_neighbor_spill, "maximum_neighbor_spill", 0.0, 1.0
        )

    def compile(
        self,
        *,
        receipt_id: str,
        contract: ActionContract,
        device_identity_sha256: str,
        ack_boot_id: str,
        ack_sequence: int,
        ack_payload_sha256: str,
        ack_fresh: bool,
        expected_mass_loss_g: float,
        observed_mass_loss_g: float,
        target_wetting_coverage: float,
        neighbor_spill_ratio: float,
    ) -> PhysicalDecisionReceipt:
        _identifier(receipt_id, "receipt_id")
        device_identity_sha256 = _sha(
            device_identity_sha256, "device_identity_sha256"
        )
        ack_payload_sha256 = _sha(ack_payload_sha256, "ack_payload_sha256")
        expected = _finite(expected_mass_loss_g, "expected_mass_loss_g", 0.0, 500.0)
        observed = _finite(observed_mass_loss_g, "observed_mass_loss_g", 0.0, 500.0)
        coverage = _finite(
            target_wetting_coverage, "target_wetting_coverage", 0.0, 1.0
        )
        spill = _finite(neighbor_spill_ratio, "neighbor_spill_ratio", 0.0, 1.0)
        reasons: list[str] = []
        if not ack_fresh:
            reasons.append("ACK_STALE_OR_MISSING")
        if ack_boot_id != contract.boot_id:
            reasons.append("ACK_BOOT_ID_MISMATCH")
        if int(ack_sequence) != contract.sequence:
            reasons.append("ACK_SEQUENCE_MISMATCH")
        if abs(observed - expected) > self.mass_tolerance_g:
            reasons.append("MASS_DELTA_OUT_OF_TOLERANCE")
        if coverage < self.minimum_target_coverage:
            reasons.append("TARGET_WETTING_INSUFFICIENT")
        if spill > self.maximum_neighbor_spill:
            reasons.append("NEIGHBOR_SPILL_EXCESSIVE")
        completed = not reasons and contract.proposed_volume_ml > 0
        if contract.proposed_volume_ml <= 0:
            reasons.append("ZERO_VOLUME_CONTRACT_NOT_A_COMPLETION")
        if completed:
            reasons.append("ACK_MASS_WETTING_CROSSCHECK_PASS")
        return PhysicalDecisionReceipt(
            receipt_id=receipt_id,
            contract_sha256=contract.sha256,
            device_identity_sha256=device_identity_sha256,
            boot_id=ack_boot_id,
            sequence=int(ack_sequence),
            ack_payload_sha256=ack_payload_sha256,
            ack_fresh=bool(ack_fresh),
            expected_mass_loss_g=expected,
            observed_mass_loss_g=observed,
            target_wetting_coverage=coverage,
            neighbor_spill_ratio=spill,
            completed=completed,
            reason_codes=tuple(sorted(set(reasons))),
        )


__all__ = [
    "ActionContract",
    "ActionContractCompiler",
    "ActionContractError",
    "PhysicalDecisionReceipt",
    "PhysicalReceiptCompiler",
]
