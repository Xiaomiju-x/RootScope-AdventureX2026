"""Deterministic RootScope v3 resource-admission broker for the 4 GB X5.

The broker returns proposals only.  It does not start, stop, signal, or kill
processes.  Its purpose is to keep perception and the future safety state
machine ahead of optional LLM/VLM workloads under low-memory conditions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any


class RuntimePhase(str, Enum):
    IDLE = "IDLE"
    PERCEPTION = "PERCEPTION"
    IRRIGATION_CRITICAL = "IRRIGATION_CRITICAL"
    POST_ACTION_VERIFY = "POST_ACTION_VERIFY"


class Workload(str, Enum):
    CPU_VISION = "CPU_VISION"
    BPU_VISION = "BPU_VISION"
    FAST_LLM = "FAST_LLM"
    DEEP_LLM = "DEEP_LLM"
    VLM_AUDIT = "VLM_AUDIT"


@dataclass(frozen=True)
class ResourceSnapshot:
    mem_available_mib: float
    cma_free_mib: float
    temperature_c: float
    load_1m: float
    resident_llm_role: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "mem_available_mib",
            "cma_free_mib",
            "temperature_c",
            "load_1m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ResourceDecision:
    admitted: bool
    workload: str
    phase: str
    profile: str
    reason_codes: tuple[str, ...]
    required_actions: tuple[str, ...]
    snapshot_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "rootscope.runtime-v3.resource-decision.v1",
            **asdict(self),
            "reason_codes": list(self.reason_codes),
            "required_actions": list(self.required_actions),
            "authority": {
                "process_control": False,
                "service_control": False,
                "serial_write": False,
                "pump_control": False,
            },
        }


class ResourceBroker:
    """Apply frozen memory/CMA/thermal and phase exclusion rules."""

    def __init__(
        self,
        *,
        min_mem_available_mib: float = 512.0,
        min_cma_free_mib: float = 128.0,
        thermal_hold_c: float = 78.0,
        fast_llm_headroom_mib: float = 800.0,
        deep_llm_headroom_mib: float = 1600.0,
        vlm_headroom_mib: float = 1250.0,
    ) -> None:
        self.min_mem_available_mib = float(min_mem_available_mib)
        self.min_cma_free_mib = float(min_cma_free_mib)
        self.thermal_hold_c = float(thermal_hold_c)
        self.headroom = {
            Workload.FAST_LLM: float(fast_llm_headroom_mib),
            Workload.DEEP_LLM: float(deep_llm_headroom_mib),
            Workload.VLM_AUDIT: float(vlm_headroom_mib),
        }

    @staticmethod
    def _snapshot_sha256(snapshot: ResourceSnapshot) -> str:
        payload = json.dumps(
            asdict(snapshot),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def decide(
        self,
        workload: Workload,
        phase: RuntimePhase,
        snapshot: ResourceSnapshot,
    ) -> ResourceDecision:
        reasons: list[str] = []
        actions: list[str] = []
        if snapshot.temperature_c >= self.thermal_hold_c:
            reasons.append("THERMAL_HOLD")
        if snapshot.mem_available_mib < self.min_mem_available_mib:
            reasons.append("MEMORY_RESERVE_LOW")
        if workload == Workload.BPU_VISION and (
            snapshot.cma_free_mib < self.min_cma_free_mib
        ):
            reasons.append("CMA_RESERVE_LOW")

        optional = workload in {
            Workload.FAST_LLM,
            Workload.DEEP_LLM,
            Workload.VLM_AUDIT,
        }
        if optional:
            required = self.headroom[workload]
            if snapshot.mem_available_mib < required:
                reasons.append("WORKLOAD_HEADROOM_LOW")
            if phase == RuntimePhase.IRRIGATION_CRITICAL:
                reasons.append("OPTIONAL_MODEL_EXCLUDED_DURING_IRRIGATION")
            if snapshot.resident_llm_role not in (None, workload.value):
                actions.append("EVICT_EXISTING_LLM_BEFORE_ADMISSION")
                reasons.append("ONE_RESIDENT_MODEL_POLICY")

        # BPU proposals are allowed for perception and post-action verification,
        # but never introduced for the first time during the critical phase.
        if workload == Workload.BPU_VISION and phase == RuntimePhase.IRRIGATION_CRITICAL:
            reasons.append("BPU_SWAP_FORBIDDEN_DURING_IRRIGATION")

        hard_denials = {
            "THERMAL_HOLD",
            "MEMORY_RESERVE_LOW",
            "CMA_RESERVE_LOW",
            "WORKLOAD_HEADROOM_LOW",
            "OPTIONAL_MODEL_EXCLUDED_DURING_IRRIGATION",
            "BPU_SWAP_FORBIDDEN_DURING_IRRIGATION",
        }
        admitted = not any(item in hard_denials for item in reasons)
        if admitted and not reasons:
            reasons.append("ADMITTED_WITHIN_FROZEN_LIMITS")
        if not admitted:
            actions.append("USE_SAFE_CPU_OR_DETERMINISTIC_RAG_FALLBACK")
        profile = (
            "SAFE_CPU"
            if not admitted or workload == Workload.CPU_VISION
            else (
                "LOCAL_HYBRID"
                if workload in {Workload.BPU_VISION, Workload.FAST_LLM}
                else "DEEP_SHADOW"
            )
        )
        return ResourceDecision(
            admitted=admitted,
            workload=workload.value,
            phase=phase.value,
            profile=profile,
            reason_codes=tuple(sorted(set(reasons))),
            required_actions=tuple(sorted(set(actions))),
            snapshot_sha256=self._snapshot_sha256(snapshot),
        )


__all__ = [
    "ResourceBroker",
    "ResourceDecision",
    "ResourceSnapshot",
    "RuntimePhase",
    "Workload",
]
