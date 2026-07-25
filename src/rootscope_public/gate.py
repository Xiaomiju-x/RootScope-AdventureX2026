"""Fail-closed public reference gate.

This module demonstrates authority separation without disclosing production
thresholds, calibration, serial framing or actuator timing.
"""

from __future__ import annotations

from .contracts import ActionProposal, Decision, EvidenceBundle, PUBLIC_CLASSES


_ABSTRACT_TIER = {
    "non_target": 0,
    "grass_clump": 1,
    "low_shrub": 2,
    "young_tree": 3,
}


def _hold(evidence: EvidenceBundle, *reasons: str) -> ActionProposal:
    return ActionProposal(
        decision=Decision.HOLD,
        action_tier=0,
        plant_class=evidence.semantic_label,
        reason_codes=tuple(reasons),
    )


def evaluate_evidence(evidence: EvidenceBundle) -> ActionProposal:
    """Return a device-free abstract proposal.

    `explanation` is intentionally ignored. A language model may explain the
    evidence but cannot influence this deterministic gate.
    """

    if evidence.semantic_label not in PUBLIC_CLASSES:
        return _hold(evidence, "UNKNOWN_CLASS")
    if not evidence.quality_passed:
        return _hold(evidence, "IMAGE_QUALITY_REJECTED")
    if evidence.ood_detected:
        return _hold(evidence, "OOD_DETECTED")
    if not evidence.evidence_fresh:
        return _hold(evidence, "STALE_EVIDENCE")
    if not evidence.device_safe:
        return _hold(evidence, "DEVICE_NOT_SAFE")
    if not evidence.geometric_verified:
        return _hold(evidence, "GEOMETRY_NOT_VERIFIED")
    if evidence.geometric_label != evidence.semantic_label:
        return _hold(evidence, "EVIDENCE_CONFLICT")
    if evidence.semantic_label == "non_target":
        return _hold(evidence, "NON_TARGET")

    return ActionProposal(
        decision=Decision.PROPOSE,
        action_tier=_ABSTRACT_TIER[evidence.semantic_label],
        plant_class=evidence.semantic_label,
        reason_codes=("DUAL_EVIDENCE_AGREES", "ABSTRACT_TIER_ONLY"),
    )

