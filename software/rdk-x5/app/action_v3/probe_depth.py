"""Fail-closed mapping from RootSight classes to V15 probe depth presets.

This module is deliberately zero-authority.  It never opens a serial device
and never sends ``DEPTH`` or ``ZHOME``.  It compiles already-gated perception
evidence into a small, hashable plan that a separately qualified single-writer
bridge may consume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


CLASS_TO_DEPTH = {
    "non_target": (0, "hold", 0),
    "grass_clump": (1, "shallow", 1024),
    "low_shrub": (2, "medium", 1536),
    "young_tree": (3, "deep", 2048),
}

ZERO_AUTHORITY = {
    "execution_authority": False,
    "serial_write": False,
    "gpio_write": False,
    "probe_command": False,
    "pump_command": False,
    "physical_completion": False,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class ProbeDepthPlan:
    plant_class: str
    requested_level: int
    label: str
    configured_steps: int
    admitted: bool
    command_preview: str | None
    manual_return_required: bool
    calibration_state: str
    reason_codes: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["reason_codes"] = list(self.reason_codes)
        return {
            "schema": "rootscope.probe-depth-plan.v1",
            **value,
            "authority": dict(ZERO_AUTHORITY),
            "claim_boundary": (
                "STEP_PRESET_ONLY_NOT_A_CALIBRATED_LENGTH_OR_ROOT_DEPTH"
            ),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.payload())).hexdigest()


def compile_probe_depth_plan(
    *,
    plant_class: str,
    confidence: float,
    confidence_floor: float = 0.70,
    ood_hold: bool,
    evidence_fresh: bool,
    temporal_support: int,
    interlocks_clear: bool,
    manual_home_confirmed: bool,
    descent_available: bool,
) -> ProbeDepthPlan:
    """Compile one bounded depth plan without acquiring execution authority."""

    if plant_class not in CLASS_TO_DEPTH:
        raise ValueError("plant_class is outside the frozen RootSight ontology")
    if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be finite and within [0,1]")
    if not math.isfinite(float(confidence_floor)):
        raise ValueError("confidence_floor must be finite")
    if not 0.0 <= float(confidence_floor) <= 1.0:
        raise ValueError("confidence_floor must be within [0,1]")
    if not 0 <= int(temporal_support) <= 120:
        raise ValueError("temporal_support must be within [0,120]")

    level, label, steps = CLASS_TO_DEPTH[plant_class]
    reasons: set[str] = set()
    admitted = level > 0

    if level == 0:
        reasons.add("NON_TARGET_HOLD")
        admitted = False
    if ood_hold:
        reasons.add("VISION_OOD_HOLD")
        admitted = False
    if confidence < confidence_floor:
        reasons.add("VISION_CONFIDENCE_LOW")
        admitted = False
    if not evidence_fresh:
        reasons.add("EVIDENCE_STALE")
        admitted = False
    if temporal_support < 3:
        reasons.add("TEMPORAL_SUPPORT_LOW")
        admitted = False
    if not interlocks_clear:
        reasons.add("INTERLOCK_ACTIVE")
        admitted = False
    if level > 0 and not manual_home_confirmed:
        reasons.add("MANUAL_HOME_NOT_CONFIRMED")
        admitted = False
    if level > 0 and not descent_available:
        reasons.add("DESCENT_ALREADY_USED")
        admitted = False

    if admitted:
        reasons.add(f"CLASS_TO_Z_LEVEL_{level}")
        reasons.add("V15_SINGLE_DESCENT_PROPOSAL")
    command_preview = f"DEPTH,{level}" if admitted else None

    return ProbeDepthPlan(
        plant_class=plant_class,
        requested_level=level,
        label=label,
        configured_steps=steps,
        admitted=admitted,
        command_preview=command_preview,
        manual_return_required=level > 0,
        calibration_state="UNQUALIFIED_STEPS_ONLY",
        reason_codes=tuple(sorted(reasons)),
    )


__all__ = [
    "CLASS_TO_DEPTH",
    "ProbeDepthPlan",
    "compile_probe_depth_plan",
]
