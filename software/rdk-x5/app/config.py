"""Frozen RootScope configuration and its content hash.

There are deliberately no production dose or vision defaults in this module.
Those values must come from the final pump, scale, sand, camera, and lighting
commissioning run.  A configuration with ``commissioned=False`` can be loaded
for UI work but the state machine will never leave its lock with it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from .schemas import ExecutionMode, PerceptionSource, TaskRequest, Zone


CONFIG_SCHEMA_VERSION = "rootscope.config.v1"
MIN_TARGET_MASS_MG = 100
MAX_TARGET_MASS_MG = 200_000
MIN_HARD_TIMEOUT_MS = 500
MAX_HARD_TIMEOUT_MS = 120_000
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _token(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _TOKEN_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty safe token")


def _finite_unit_interval(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0, 1]")


@dataclass(frozen=True)
class ProfileConfig:
    profile_id: str
    channel: Zone
    morphology_label: str
    target_mass_mg: int
    tolerance_mg: int
    hard_timeout_ms: int
    settle_ms: int
    target_wetting_threshold: float
    neighbor_spill_threshold: float
    minimum_mass_samples: int
    max_final_mass_span_mg: int

    def __post_init__(self) -> None:
        _token(self.profile_id, "profile_id")
        _token(self.morphology_label, "morphology_label")
        if not isinstance(self.channel, Zone):
            raise ValueError("channel must be a Zone")
        for field_name in (
            "target_mass_mg",
            "tolerance_mg",
            "hard_timeout_ms",
            "settle_ms",
            "minimum_mass_samples",
            "max_final_mass_span_mg",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
        if not MIN_TARGET_MASS_MG <= self.target_mass_mg <= MAX_TARGET_MASS_MG:
            raise ValueError(
                f"target_mass_mg must be within protocol range "
                f"[{MIN_TARGET_MASS_MG}, {MAX_TARGET_MASS_MG}]"
            )
        if self.tolerance_mg < 0 or self.tolerance_mg >= self.target_mass_mg:
            raise ValueError("tolerance_mg must be non-negative and below target")
        if not MIN_HARD_TIMEOUT_MS <= self.hard_timeout_ms <= MAX_HARD_TIMEOUT_MS:
            raise ValueError(
                f"hard_timeout_ms must be within protocol range "
                f"[{MIN_HARD_TIMEOUT_MS}, {MAX_HARD_TIMEOUT_MS}]"
            )
        if not 0 <= self.settle_ms <= MAX_HARD_TIMEOUT_MS:
            raise ValueError(
                f"settle_ms must be within [0, {MAX_HARD_TIMEOUT_MS}]"
            )
        if self.minimum_mass_samples <= 0:
            raise ValueError("minimum_mass_samples must be positive")
        if self.max_final_mass_span_mg < 0:
            raise ValueError("max_final_mass_span_mg cannot be negative")
        _finite_unit_interval(
            self.target_wetting_threshold, "target_wetting_threshold"
        )
        _finite_unit_interval(
            self.neighbor_spill_threshold, "neighbor_spill_threshold"
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "profile_id": self.profile_id,
            "channel": self.channel.value,
            "morphology_label": self.morphology_label,
            "target_mass_mg": self.target_mass_mg,
            "tolerance_mg": self.tolerance_mg,
            "hard_timeout_ms": self.hard_timeout_ms,
            "settle_ms": self.settle_ms,
            "target_wetting_threshold": self.target_wetting_threshold,
            "neighbor_spill_threshold": self.neighbor_spill_threshold,
            "minimum_mass_samples": self.minimum_mass_samples,
            "max_final_mass_span_mg": self.max_final_mass_span_mg,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProfileConfig":
        expected = {
            "profile_id",
            "channel",
            "morphology_label",
            "target_mass_mg",
            "tolerance_mg",
            "hard_timeout_ms",
            "settle_ms",
            "target_wetting_threshold",
            "neighbor_spill_threshold",
            "minimum_mass_samples",
            "max_final_mass_span_mg",
        }
        unknown = set(data) - expected
        missing = expected - set(data)
        if unknown or missing:
            raise ValueError(
                f"invalid profile keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(
            profile_id=data["profile_id"],
            channel=Zone(data["channel"]),
            morphology_label=data["morphology_label"],
            target_mass_mg=data["target_mass_mg"],
            tolerance_mg=data["tolerance_mg"],
            hard_timeout_ms=data["hard_timeout_ms"],
            settle_ms=data["settle_ms"],
            target_wetting_threshold=data["target_wetting_threshold"],
            neighbor_spill_threshold=data["neighbor_spill_threshold"],
            minimum_mass_samples=data["minimum_mass_samples"],
            max_final_mass_span_mg=data["max_final_mass_span_mg"],
        )


@dataclass(frozen=True)
class RootScopeConfig:
    commissioning_id: str
    commissioned: bool
    execution_mode: ExecutionMode
    required_backend: str
    protocol_version: int
    expected_firmware_build_id: str
    required_firmware_capabilities: Tuple[str, ...]
    formal_perception_sources: Tuple[PerceptionSource, ...]
    profiles: Tuple[ProfileConfig, ...]
    schema_version: str = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported config schema: {self.schema_version}")
        _token(self.commissioning_id, "commissioning_id")
        if not isinstance(self.commissioned, bool):
            raise ValueError("commissioned must be boolean")
        if not isinstance(self.execution_mode, ExecutionMode):
            raise ValueError("execution_mode must be an ExecutionMode")
        _token(self.required_backend, "required_backend")
        if (
            self.execution_mode is ExecutionMode.SIMULATION_ONLY
            and self.required_backend != "FAKE_F407"
        ):
            raise ValueError("SIMULATION_ONLY requires backend FAKE_F407")
        if (
            self.execution_mode is ExecutionMode.PHYSICAL
            and self.required_backend == "FAKE_F407"
        ):
            raise ValueError("PHYSICAL configuration cannot require FAKE_F407")
        if isinstance(self.protocol_version, bool) or not isinstance(
            self.protocol_version, int
        ):
            raise ValueError("protocol_version must be an integer")
        if not 0 < self.protocol_version <= 0xFFFF:
            raise ValueError("protocol_version must be a positive uint16")
        _token(self.expected_firmware_build_id, "expected_firmware_build_id")
        if not self.required_firmware_capabilities:
            raise ValueError("required_firmware_capabilities cannot be empty")
        if len(set(self.required_firmware_capabilities)) != len(
            self.required_firmware_capabilities
        ):
            raise ValueError("required_firmware_capabilities contains duplicates")
        for capability in self.required_firmware_capabilities:
            _token(capability, "firmware capability")

        if not self.formal_perception_sources:
            raise ValueError("at least one formal perception source is required")
        if len(set(self.formal_perception_sources)) != len(
            self.formal_perception_sources
        ):
            raise ValueError("formal_perception_sources contains duplicates")
        physically_admissible = {PerceptionSource.TAG, PerceptionSource.BPU}
        if not set(self.formal_perception_sources).issubset(physically_admissible):
            raise ValueError(
                "only tag or qualified BPU may be formal physical perception sources"
            )

        if len(self.profiles) != 3:
            raise ValueError("RootScope P0 requires exactly three profiles")
        if {profile.channel for profile in self.profiles} != set(Zone):
            raise ValueError("profiles must map exactly once to Z1, Z2, and Z3")
        if len({profile.profile_id for profile in self.profiles}) != len(self.profiles):
            raise ValueError("profile_id values must be unique")
        if len({profile.morphology_label for profile in self.profiles}) != len(
            self.profiles
        ):
            raise ValueError("morphology labels must be unique")

    def profile_for(self, profile_id: str) -> ProfileConfig:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise KeyError(profile_id)

    def task_validation_error(self, request: TaskRequest) -> str:
        if request.config_hash != self.sha256:
            return "request config_hash does not match the loaded frozen config"
        try:
            profile = self.profile_for(request.profile_id)
        except KeyError:
            return "unknown profile_id"
        if request.channel is not profile.channel:
            return "request channel does not match the profile"
        if request.perception_label != profile.morphology_label:
            return "perception label does not match the profile"
        if request.target_mass_mg != profile.target_mass_mg:
            return "target_mass_mg does not match the frozen profile"
        if request.tolerance_mg != profile.tolerance_mg:
            return "tolerance_mg does not match the frozen profile"
        if request.hard_timeout_ms != profile.hard_timeout_ms:
            return "hard_timeout_ms does not match the frozen profile"
        if request.perception_source not in self.formal_perception_sources:
            return "perception source is not qualified for physical admission"
        return ""

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "commissioning_id": self.commissioning_id,
            "commissioned": self.commissioned,
            "execution_mode": self.execution_mode.value,
            "required_backend": self.required_backend,
            "protocol_version": self.protocol_version,
            "expected_firmware_build_id": self.expected_firmware_build_id,
            "required_firmware_capabilities": list(
                self.required_firmware_capabilities
            ),
            "formal_perception_sources": [
                source.value for source in self.formal_perception_sources
            ],
            "profiles": [profile.to_dict() for profile in self.profiles],
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RootScopeConfig":
        expected = {
            "schema_version",
            "commissioning_id",
            "commissioned",
            "execution_mode",
            "required_backend",
            "protocol_version",
            "expected_firmware_build_id",
            "required_firmware_capabilities",
            "formal_perception_sources",
            "profiles",
        }
        unknown = set(data) - expected
        missing = expected - set(data)
        if unknown or missing:
            raise ValueError(
                f"invalid config keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        capabilities = data["required_firmware_capabilities"]
        sources = data["formal_perception_sources"]
        profiles = data["profiles"]
        if not isinstance(capabilities, list):
            raise ValueError("required_firmware_capabilities must be a list")
        if not isinstance(sources, list):
            raise ValueError("formal_perception_sources must be a list")
        if not isinstance(profiles, list):
            raise ValueError("profiles must be a list")
        return cls(
            schema_version=data["schema_version"],
            commissioning_id=data["commissioning_id"],
            commissioned=data["commissioned"],
            execution_mode=ExecutionMode(data["execution_mode"]),
            required_backend=data["required_backend"],
            protocol_version=data["protocol_version"],
            expected_firmware_build_id=data["expected_firmware_build_id"],
            required_firmware_capabilities=tuple(capabilities),
            formal_perception_sources=tuple(PerceptionSource(item) for item in sources),
            profiles=tuple(ProfileConfig.from_dict(item) for item in profiles),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "RootScopeConfig":
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("configuration root must be a JSON object")
        return cls.from_dict(data)
