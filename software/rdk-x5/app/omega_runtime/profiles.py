"""RootScope-Ω resource-aware EdgeOS profile selection."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .contracts import BackendCapsule, RuntimeMode


@dataclass(frozen=True)
class ResourceSnapshot:
    available_memory_mib: int
    cpu_temperature_c: float
    bpu_model_qualified: bool
    local_llm_available: bool
    remote_shadow_available: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.available_memory_mib, bool)
            or not isinstance(self.available_memory_mib, int)
            or self.available_memory_mib < 0
        ):
            raise ValueError("available_memory_mib must be non-negative integer")
        if (
            isinstance(self.cpu_temperature_c, bool)
            or not isinstance(self.cpu_temperature_c, (int, float))
            or not math.isfinite(float(self.cpu_temperature_c))
        ):
            raise ValueError("cpu_temperature_c must be finite")
        for name in (
            "bpu_model_qualified",
            "local_llm_available",
            "remote_shadow_available",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class EdgeProfile:
    profile_id: str
    decision_backend: str
    vision_backend: str
    retrieval_backend: str
    explanation_backend: str
    bpu_probe_allowed: bool
    local_llm_allowed: bool
    remote_shadow_allowed: bool
    fallback_profile: Optional[str]
    minimum_available_memory_mib: int
    maximum_cpu_temperature_c: float


class EdgeProfileRegistry:
    def __init__(self, profiles: Mapping[str, EdgeProfile]) -> None:
        self._profiles = dict(profiles)
        if "SAFE_CPU" not in self._profiles:
            raise ValueError("SAFE_CPU profile is mandatory")

    @classmethod
    def from_file(cls, path: Path) -> "EdgeProfileRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "rootscope.omega.edge-profiles.v1":
            raise ValueError("unsupported edge profile schema")
        expected_top = {"schema_version", "profiles", "global_authority"}
        if set(payload) != expected_top:
            raise ValueError("edge profile document has unknown or missing fields")
        authority = payload["global_authority"]
        if not isinstance(authority, dict) or any(authority.values()):
            raise ValueError("global Ω authority must remain all-false")
        profiles: dict[str, EdgeProfile] = {}
        expected_profile = {
            "decision_backend",
            "vision_backend",
            "retrieval_backend",
            "explanation_backend",
            "bpu_probe_allowed",
            "local_llm_allowed",
            "remote_shadow_allowed",
            "fallback_profile",
            "minimum_available_memory_mib",
            "maximum_cpu_temperature_c",
        }
        for profile_id, values in payload["profiles"].items():
            if not isinstance(values, dict) or set(values) != expected_profile:
                raise ValueError(f"invalid profile contract for {profile_id}")
            profiles[profile_id] = EdgeProfile(profile_id=profile_id, **values)
        return cls(profiles)

    def profile(self, profile_id: str) -> EdgeProfile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise ValueError(f"unknown edge profile {profile_id}") from exc

    def select(
        self,
        requested_profile: str,
        resources: ResourceSnapshot,
        *,
        runtime_mode: RuntimeMode,
        release_id: str,
    ) -> BackendCapsule:
        selected = self.profile(requested_profile)
        fallback_reasons: list[str] = []

        if resources.available_memory_mib < selected.minimum_available_memory_mib:
            fallback_reasons.append("MEMORY_RESERVE_GATE")
        if resources.cpu_temperature_c > selected.maximum_cpu_temperature_c:
            fallback_reasons.append("THERMAL_GATE")
        if selected.bpu_probe_allowed and not resources.bpu_model_qualified:
            fallback_reasons.append("BPU_MODEL_NOT_QUALIFIED")
        if selected.local_llm_allowed and not resources.local_llm_available:
            fallback_reasons.append("LOCAL_LLM_UNAVAILABLE")
        if selected.remote_shadow_allowed and not resources.remote_shadow_available:
            fallback_reasons.append("REMOTE_SHADOW_UNAVAILABLE")

        if fallback_reasons:
            if selected.fallback_profile is None:
                raise ValueError(
                    f"profile {selected.profile_id} failed without fallback: "
                    f"{fallback_reasons}"
                )
            selected = self.profile(selected.fallback_profile)

        safe = selected.profile_id == "SAFE_CPU"
        return BackendCapsule(
            profile=selected.profile_id,
            runtime_mode=runtime_mode,
            decision_backend_actual=selected.decision_backend,
            vision_backend_actual=(
                "onnxruntime_cpu"
                if safe
                else "qualified_bpu_probe_with_cpu_projection"
            ),
            retrieval_backend_actual=selected.retrieval_backend,
            explanation_backend_actual=selected.explanation_backend,
            release_id=release_id,
            bpu_model_qualified=(not safe and resources.bpu_model_qualified),
            local_llm_active=(selected.local_llm_allowed and resources.local_llm_available),
            remote_shadow_active=(
                selected.remote_shadow_allowed
                and resources.remote_shadow_available
            ),
            fallback_reasons=tuple(fallback_reasons),
        )
