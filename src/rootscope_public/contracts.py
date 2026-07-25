"""Small public contracts that intentionally contain no device control details."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Decision(str, Enum):
    """Only two public outcomes are exposed."""

    HOLD = "HOLD"
    PROPOSE = "PROPOSE"


PUBLIC_CLASSES = ("grass_clump", "low_shrub", "young_tree", "non_target")


@dataclass(frozen=True)
class EvidenceBundle:
    """Synthetic/public evidence accepted by the safe reference gate."""

    semantic_label: str
    geometric_label: str | None
    semantic_confidence: float
    quality_passed: bool
    geometric_verified: bool
    ood_detected: bool
    evidence_fresh: bool
    device_safe: bool
    explanation: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EvidenceBundle":
        return cls(
            semantic_label=str(value["semantic_label"]),
            geometric_label=(
                None if value.get("geometric_label") is None
                else str(value["geometric_label"])
            ),
            semantic_confidence=float(value["semantic_confidence"]),
            quality_passed=bool(value["quality_passed"]),
            geometric_verified=bool(value["geometric_verified"]),
            ood_detected=bool(value["ood_detected"]),
            evidence_fresh=bool(value["evidence_fresh"]),
            device_safe=bool(value["device_safe"]),
            explanation=(
                None if value.get("explanation") is None
                else str(value["explanation"])
            ),
        )


@dataclass(frozen=True)
class ActionProposal:
    """An auditable abstract proposal. It can never be a hardware command."""

    decision: Decision
    action_tier: int
    plant_class: str
    reason_codes: tuple[str, ...]
    proposal_only: bool = True
    hardware_command: None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decision"] = self.decision.value
        value["reason_codes"] = list(self.reason_codes)
        return value

