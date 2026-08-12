"""Typed, dependency-free contracts shared by the RootScope core.

Physical quantities use explicit integer units (milligrams and milliseconds)
so the X5 and F407 sides never need to guess float encoding or units.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_16_RE = re.compile(r"^[0-9a-f]{16}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class StringEnum(str, Enum):
    """A Python 3.10 compatible string enum."""

    def __str__(self) -> str:
        return self.value


class Zone(StringEnum):
    Z1 = "Z1"
    Z2 = "Z2"
    Z3 = "Z3"


class PerceptionSource(StringEnum):
    TAG = "tag"
    BPU = "bpu"
    MANUAL = "manual"
    FIXTURE = "fixture"
    REPLAY = "replay"


class ExecutionMode(StringEnum):
    SIMULATION_ONLY = "SIMULATION_ONLY"
    PHYSICAL = "PHYSICAL"


class MachineState(StringEnum):
    BOOT_LOCKED = "BOOT_LOCKED"
    SELF_CHECK = "SELF_CHECK"
    READY = "READY"
    TARGET_IDENTIFIED = "TARGET_IDENTIFIED"
    BASELINE_CAPTURED = "BASELINE_CAPTURED"
    DOSING_Z1 = "DOSING_Z1"
    DOSING_Z2 = "DOSING_Z2"
    DOSING_Z3 = "DOSING_Z3"
    SETTLING = "SETTLING"
    VERIFYING = "VERIFYING"
    TARGET_WETTING_VERIFIED = "TARGET_WETTING_VERIFIED"
    SAFE_STOP = "SAFE_STOP"
    ABORTED_LOCKED = "ABORTED_LOCKED"


class CompletionClass(StringEnum):
    SIMULATED_ONLY = "SIMULATED_ONLY"
    ACTUATOR_ACK = "ACTUATOR_ACK"
    MASS_LOSS_VERIFIED = "MASS_LOSS_VERIFIED"
    TARGET_WETTING_VERIFIED = "TARGET_WETTING_VERIFIED"
    ABORTED_LOCKED = "ABORTED_LOCKED"


class FaultCode(StringEnum):
    NONE = "NONE"
    NOT_READY = "NOT_READY"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    INVALID_TASK = "INVALID_TASK"
    STALE_TASK = "STALE_TASK"
    TASK_ID_CONFLICT = "TASK_ID_CONFLICT"
    TASK_BUSY = "TASK_BUSY"
    TASK_CONTEXT_MISMATCH = "TASK_CONTEXT_MISMATCH"
    CONFIG_MISMATCH = "CONFIG_MISMATCH"
    PERCEPTION_NOT_QUALIFIED = "PERCEPTION_NOT_QUALIFIED"
    SELF_CHECK_FAILED = "SELF_CHECK_FAILED"
    ESTOP_ACTIVE = "ESTOP_ACTIVE"
    LEAK_DETECTED = "LEAK_DETECTED"
    CARTRIDGE_MISSING = "CARTRIDGE_MISSING"
    GUARD_OPEN = "GUARD_OPEN"
    FIRMWARE_IDENTITY_INVALID = "FIRMWARE_IDENTITY_INVALID"
    FIRMWARE_REBOOTED = "FIRMWARE_REBOOTED"
    FIRMWARE_CAPABILITY_MISSING = "FIRMWARE_CAPABILITY_MISSING"
    F407_LOCK_LATCHED = "F407_LOCK_LATCHED"
    ACT_ENABLE_INVALID = "ACT_ENABLE_INVALID"
    HEARTBEAT_STALE = "HEARTBEAT_STALE"
    TELEMETRY_STALE = "TELEMETRY_STALE"
    SCALE_UNSTABLE = "SCALE_UNSTABLE"
    CAMERA_QUALITY_INVALID = "CAMERA_QUALITY_INVALID"
    PUMP_NOT_OFF = "PUMP_NOT_OFF"
    MULTIPLE_PUMPS_ACTIVE = "MULTIPLE_PUMPS_ACTIVE"
    WRONG_PUMP_ACTIVE = "WRONG_PUMP_ACTIVE"
    ACK_INVALID = "ACK_INVALID"
    CLEAR_ACK_INVALID = "CLEAR_ACK_INVALID"
    COMMAND_CONTEXT_INVALID = "COMMAND_CONTEXT_INVALID"
    MASS_OUT_OF_RANGE = "MASS_OUT_OF_RANGE"
    SETTLING_NOT_COMPLETE = "SETTLING_NOT_COMPLETE"
    WETTING_NOT_VERIFIED = "WETTING_NOT_VERIFIED"
    NEIGHBOR_SPILL = "NEIGHBOR_SPILL"
    OPERATOR_CONFIRMATION_REQUIRED = "OPERATOR_CONFIRMATION_REQUIRED"
    CARTRIDGE_CHANGE_REQUIRED = "CARTRIDGE_CHANGE_REQUIRED"
    USER_ABORT = "USER_ABORT"
    RESTART_RECOVERY = "RESTART_RECOVERY"
    EVIDENCE_WRITE_FAILED = "EVIDENCE_WRITE_FAILED"
    PHYSICAL_STOP_UNCONFIRMED = "PHYSICAL_STOP_UNCONFIRMED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AdmissionStatus(StringEnum):
    ACCEPTED = "ACCEPTED"
    IDEMPOTENT_REPLAY = "IDEMPOTENT_REPLAY"
    REJECTED = "REJECTED"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _require_token(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _TOKEN_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty safe token")


def _require_hash(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _HEX_64_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _require_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")


def _require_bool_fields(instance: Any, field_names: Tuple[str, ...]) -> None:
    for field_name in field_names:
        if not isinstance(getattr(instance, field_name), bool):
            raise ValueError(f"{field_name} must be a boolean")


def _require_u16_nonzero(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if not 1 <= value <= 0xFFFF:
        raise ValueError(f"{field_name} must be within nonzero uint16")


def _require_uint32_nonzero(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if not 1 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{field_name} must be within nonzero uint32")


@dataclass(frozen=True)
class TaskRequest:
    """A fully frozen, single-channel irrigation task request."""

    task_id: str
    task_seq: int
    profile_id: str
    channel: Zone
    target_mass_mg: int
    tolerance_mg: int
    hard_timeout_ms: int
    config_hash: str
    perception_source: PerceptionSource
    perception_label: str
    perception_score: Optional[float] = None
    created_at_utc: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError(
                "task_id must be 8-128 safe ASCII characters and start alphanumeric"
            )
        if isinstance(self.task_seq, bool) or not isinstance(self.task_seq, int):
            raise ValueError("task_seq must be an integer")
        if self.task_seq <= 0:
            raise ValueError("task_seq must be positive")
        if self.task_seq > 0xFFFFFFFF:
            raise ValueError("task_seq must fit the F407 uint32 wire task id")
        _require_token(self.profile_id, "profile_id")
        if not isinstance(self.channel, Zone):
            raise ValueError("channel must be a Zone")
        if not isinstance(self.perception_source, PerceptionSource):
            raise ValueError("perception_source must be a PerceptionSource")
        _require_token(self.perception_label, "perception_label")
        for field_name in ("target_mass_mg", "tolerance_mg", "hard_timeout_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
        if self.target_mass_mg <= 0:
            raise ValueError("target_mass_mg must be positive")
        if self.tolerance_mg < 0 or self.tolerance_mg >= self.target_mass_mg:
            raise ValueError("tolerance_mg must be non-negative and below target mass")
        if self.hard_timeout_ms <= 0:
            raise ValueError("hard_timeout_ms must be positive")
        _require_hash(self.config_hash, "config_hash")
        if self.perception_score is not None:
            _require_finite(self.perception_score, "perception_score")
            if not 0.0 <= float(self.perception_score) <= 1.0:
                raise ValueError("perception_score must be within [0, 1]")

    def identity_payload(self) -> Mapping[str, Any]:
        """Return physical/admission fields used for idempotency.

        ``created_at_utc`` is intentionally excluded: an HTTP retry may recreate
        a timestamp, but it must not change any physical execution parameter.
        """

        return {
            "task_id": self.task_id,
            "task_seq": self.task_seq,
            "wire_task_id": self.wire_task_id,
            "profile_id": self.profile_id,
            "channel": self.channel.value,
            "target_mass_mg": self.target_mass_mg,
            "tolerance_mg": self.tolerance_mg,
            "hard_timeout_ms": self.hard_timeout_ms,
            "config_hash": self.config_hash,
            "perception_source": self.perception_source.value,
            "perception_label": self.perception_label,
            "perception_score": self.perception_score,
        }

    @property
    def wire_task_id(self) -> int:
        """Persistent, unique uint32 identifier transmitted to the F407.

        The human-readable ``task_id`` remains in X5 evidence.  The monotonic
        sequence is the one-to-one wire mapping and is never reused; exhausting
        uint32 requires a new protocol/config generation rather than wraparound.
        """

        return self.task_seq

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(
            self.identity_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> Mapping[str, Any]:
        data = dict(self.identity_payload())
        data["created_at_utc"] = self.created_at_utc
        data["request_fingerprint"] = self.fingerprint
        return data


@dataclass(frozen=True)
class SafetySnapshot:
    """Already freshness-evaluated safety inputs for one decision instant."""

    estop_clear: bool
    leak_clear: bool
    cartridge_present: bool
    guard_closed: bool
    heartbeat_fresh: bool
    telemetry_fresh: bool
    scale_stable: bool
    camera_quality_ok: bool
    firmware_protocol_version: int
    firmware_build_id: str
    firmware_capabilities: Tuple[str, ...]
    execution_backend: str
    firmware_boot_id: str
    firmware_uptime_ms: int
    lock_latched: bool
    lock_reason: str
    act_enable: bool
    active_wire_task_id: Optional[int]
    pump_z1_on: bool = False
    pump_z2_on: bool = False
    pump_z3_on: bool = False
    observed_at_utc: str = ""

    def __post_init__(self) -> None:
        _require_bool_fields(
            self,
            (
                "estop_clear",
                "leak_clear",
                "cartridge_present",
                "guard_closed",
                "heartbeat_fresh",
                "telemetry_fresh",
                "scale_stable",
                "camera_quality_ok",
                "lock_latched",
                "act_enable",
                "pump_z1_on",
                "pump_z2_on",
                "pump_z3_on",
            ),
        )
        if isinstance(self.firmware_protocol_version, bool) or not isinstance(
            self.firmware_protocol_version, int
        ):
            raise ValueError("firmware_protocol_version must be an integer")
        _require_token(self.firmware_build_id, "firmware_build_id")
        _require_token(self.execution_backend, "execution_backend")
        _require_token(self.firmware_boot_id, "firmware_boot_id")
        _require_token(self.lock_reason, "lock_reason")
        if isinstance(self.firmware_uptime_ms, bool) or not isinstance(
            self.firmware_uptime_ms, int
        ):
            raise ValueError("firmware_uptime_ms must be an integer")
        if self.firmware_uptime_ms < 0:
            raise ValueError("firmware_uptime_ms cannot be negative")
        if self.active_wire_task_id is not None:
            if isinstance(self.active_wire_task_id, bool) or not isinstance(
                self.active_wire_task_id, int
            ):
                raise ValueError("active_wire_task_id must be integer or None")
            if not 0 < self.active_wire_task_id <= 0xFFFFFFFF:
                raise ValueError("active_wire_task_id must be a positive uint32")
        if self.lock_latched and self.lock_reason == "NONE":
            raise ValueError("a latched lock requires a non-NONE reason")
        if not self.lock_latched and self.lock_reason != "NONE":
            raise ValueError("an unlocked snapshot must use lock_reason=NONE")
        if not isinstance(self.firmware_capabilities, tuple):
            raise ValueError("firmware_capabilities must be a tuple")
        for capability in self.firmware_capabilities:
            _require_token(capability, "firmware capability")

    @property
    def active_pumps(self) -> Tuple[Zone, ...]:
        active = []
        if self.pump_z1_on:
            active.append(Zone.Z1)
        if self.pump_z2_on:
            active.append(Zone.Z2)
        if self.pump_z3_on:
            active.append(Zone.Z3)
        return tuple(active)

    @property
    def pumps_all_off(self) -> bool:
        return not self.active_pumps

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "estop_clear": self.estop_clear,
            "leak_clear": self.leak_clear,
            "cartridge_present": self.cartridge_present,
            "guard_closed": self.guard_closed,
            "heartbeat_fresh": self.heartbeat_fresh,
            "telemetry_fresh": self.telemetry_fresh,
            "scale_stable": self.scale_stable,
            "camera_quality_ok": self.camera_quality_ok,
            "firmware_protocol_version": self.firmware_protocol_version,
            "firmware_build_id": self.firmware_build_id,
            "firmware_capabilities": list(self.firmware_capabilities),
            "execution_backend": self.execution_backend,
            "firmware_boot_id": self.firmware_boot_id,
            "firmware_uptime_ms": self.firmware_uptime_ms,
            "lock_latched": self.lock_latched,
            "lock_reason": self.lock_reason,
            "act_enable": self.act_enable,
            "active_wire_task_id": self.active_wire_task_id,
            "active_pumps": [zone.value for zone in self.active_pumps],
            "observed_at_utc": self.observed_at_utc,
        }


@dataclass(frozen=True)
class ClearEstopCommandContext:
    frame_seq: int
    raw_frame_sha256: str
    transcript_id: str
    decoded_command: str
    execution_backend: str
    firmware_boot_id: str

    def __post_init__(self) -> None:
        _require_u16_nonzero(self.frame_seq, "frame_seq")
        _require_hash(self.raw_frame_sha256, "raw_frame_sha256")
        _require_token(self.transcript_id, "transcript_id")
        _require_token(self.decoded_command, "decoded_command")
        _require_token(self.execution_backend, "execution_backend")
        _require_token(self.firmware_boot_id, "firmware_boot_id")
        if self.decoded_command != "CLEAR_ESTOP":
            raise ValueError("decoded_command must be CLEAR_ESTOP")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "frame_seq": self.frame_seq,
            "raw_frame_sha256": self.raw_frame_sha256,
            "transcript_id": self.transcript_id,
            "decoded_command": self.decoded_command,
            "execution_backend": self.execution_backend,
            "firmware_boot_id": self.firmware_boot_id,
        }


@dataclass(frozen=True)
class ClearEstopAckEvidence:
    ack_for_type: str
    ack_for_seq: int
    ack_frame_sha256: str
    transcript_id: str
    acked: bool
    fresh: bool
    firmware_build_id: str
    firmware_boot_id: str
    execution_backend: str
    observed_at_utc: str = ""

    def __post_init__(self) -> None:
        _require_bool_fields(self, ("acked", "fresh"))
        _require_token(self.ack_for_type, "ack_for_type")
        _require_u16_nonzero(self.ack_for_seq, "ack_for_seq")
        _require_hash(self.ack_frame_sha256, "ack_frame_sha256")
        _require_token(self.transcript_id, "transcript_id")
        _require_token(self.firmware_build_id, "firmware_build_id")
        _require_token(self.firmware_boot_id, "firmware_boot_id")
        _require_token(self.execution_backend, "execution_backend")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "ack_for_type": self.ack_for_type,
            "ack_for_seq": self.ack_for_seq,
            "ack_frame_sha256": self.ack_frame_sha256,
            "transcript_id": self.transcript_id,
            "acked": self.acked,
            "fresh": self.fresh,
            "firmware_build_id": self.firmware_build_id,
            "firmware_boot_id": self.firmware_boot_id,
            "execution_backend": self.execution_backend,
            "observed_at_utc": self.observed_at_utc,
        }


@dataclass(frozen=True)
class ArmCommandContext:
    task_id: str
    wire_task_id: int
    frame_seq: int
    raw_frame_sha256: str
    transcript_id: str
    decoded_command: str
    decoded_channel: Zone
    decoded_target_mass_mg: int
    decoded_hard_timeout_ms: int
    decoded_config_hash_prefix: str
    execution_backend: str
    firmware_build_id: str
    firmware_boot_id: str

    def __post_init__(self) -> None:
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("invalid task_id")
        _require_uint32_nonzero(self.wire_task_id, "wire_task_id")
        _require_u16_nonzero(self.frame_seq, "frame_seq")
        _require_hash(self.raw_frame_sha256, "raw_frame_sha256")
        _require_token(self.transcript_id, "transcript_id")
        _require_token(self.decoded_command, "decoded_command")
        _require_token(self.execution_backend, "execution_backend")
        _require_token(self.firmware_build_id, "firmware_build_id")
        _require_token(self.firmware_boot_id, "firmware_boot_id")
        if self.decoded_command != "ARM_TASK":
            raise ValueError("decoded_command must be ARM_TASK")
        if not isinstance(self.decoded_channel, Zone):
            raise ValueError("decoded_channel must be a Zone")
        for field_name in ("decoded_target_mass_mg", "decoded_hard_timeout_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.decoded_config_hash_prefix, str) or not _HEX_16_RE.fullmatch(
            self.decoded_config_hash_prefix
        ):
            raise ValueError("decoded_config_hash_prefix must be 8-byte lowercase hex")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "task_id": self.task_id,
            "wire_task_id": self.wire_task_id,
            "frame_seq": self.frame_seq,
            "raw_frame_sha256": self.raw_frame_sha256,
            "transcript_id": self.transcript_id,
            "decoded_command": self.decoded_command,
            "decoded_channel": self.decoded_channel.value,
            "decoded_target_mass_mg": self.decoded_target_mass_mg,
            "decoded_hard_timeout_ms": self.decoded_hard_timeout_ms,
            "decoded_config_hash_prefix": self.decoded_config_hash_prefix,
            "execution_backend": self.execution_backend,
            "firmware_build_id": self.firmware_build_id,
            "firmware_boot_id": self.firmware_boot_id,
        }


@dataclass(frozen=True)
class StopCommandContext:
    task_id: Optional[str]
    wire_task_id: Optional[int]
    frame_seq: int
    raw_frame_sha256: str
    transcript_id: str
    decoded_command: str
    execution_backend: str
    firmware_build_id: str
    firmware_boot_id: str

    def __post_init__(self) -> None:
        if self.task_id is not None and not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("invalid task_id")
        if self.wire_task_id is not None:
            _require_uint32_nonzero(self.wire_task_id, "wire_task_id")
        _require_u16_nonzero(self.frame_seq, "frame_seq")
        _require_hash(self.raw_frame_sha256, "raw_frame_sha256")
        _require_token(self.transcript_id, "transcript_id")
        _require_token(self.decoded_command, "decoded_command")
        _require_token(self.execution_backend, "execution_backend")
        _require_token(self.firmware_build_id, "firmware_build_id")
        _require_token(self.firmware_boot_id, "firmware_boot_id")
        if self.decoded_command != "EMERGENCY_STOP":
            raise ValueError("decoded_command must be EMERGENCY_STOP")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "task_id": self.task_id,
            "wire_task_id": self.wire_task_id,
            "frame_seq": self.frame_seq,
            "raw_frame_sha256": self.raw_frame_sha256,
            "transcript_id": self.transcript_id,
            "decoded_command": self.decoded_command,
            "execution_backend": self.execution_backend,
            "firmware_build_id": self.firmware_build_id,
            "firmware_boot_id": self.firmware_boot_id,
        }


@dataclass(frozen=True)
class ActuatorAckEvidence:
    task_id: str
    wire_task_id: int
    ack_for_type: str
    ack_for_seq: int
    ack_frame_sha256: str
    transcript_id: str
    channel: Zone
    acked: bool
    fresh: bool
    all_other_pumps_off: bool
    firmware_build_id: str
    firmware_boot_id: str
    execution_backend: str
    observed_at_utc: str = ""

    def __post_init__(self) -> None:
        _require_bool_fields(self, ("acked", "fresh", "all_other_pumps_off"))
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("invalid task_id")
        if isinstance(self.wire_task_id, bool) or not isinstance(
            self.wire_task_id, int
        ):
            raise ValueError("wire_task_id must be an integer")
        if not 0 < self.wire_task_id <= 0xFFFFFFFF:
            raise ValueError("wire_task_id must be a positive uint32")
        _require_u16_nonzero(self.ack_for_seq, "ack_for_seq")
        _require_hash(self.ack_frame_sha256, "ack_frame_sha256")
        _require_token(self.transcript_id, "transcript_id")
        _require_token(self.ack_for_type, "ack_for_type")
        if not isinstance(self.channel, Zone):
            raise ValueError("channel must be a Zone")
        _require_token(self.firmware_build_id, "firmware_build_id")
        _require_token(self.firmware_boot_id, "firmware_boot_id")
        _require_token(self.execution_backend, "execution_backend")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "task_id": self.task_id,
            "wire_task_id": self.wire_task_id,
            "ack_for_type": self.ack_for_type,
            "ack_for_seq": self.ack_for_seq,
            "ack_frame_sha256": self.ack_frame_sha256,
            "transcript_id": self.transcript_id,
            "channel": self.channel.value,
            "acked": self.acked,
            "fresh": self.fresh,
            "all_other_pumps_off": self.all_other_pumps_off,
            "firmware_build_id": self.firmware_build_id,
            "firmware_boot_id": self.firmware_boot_id,
            "execution_backend": self.execution_backend,
            "observed_at_utc": self.observed_at_utc,
        }


@dataclass(frozen=True)
class BaselineEvidence:
    task_id: str
    wire_task_id: int
    baseline_id: str
    camera_frame_id: str
    camera_frame_sha256: str
    baseline_mass_mg: int
    mass_sample_count: int
    mass_last_sample_seq: int
    mass_sample_digest: str
    config_hash: str
    firmware_boot_id: str
    firmware_uptime_ms_at_capture: int
    stable: bool
    fresh: bool
    host_captured_monotonic_ms: int
    observed_at_utc: str = ""

    def __post_init__(self) -> None:
        _require_bool_fields(self, ("stable", "fresh"))
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("invalid task_id")
        _require_token(self.baseline_id, "baseline_id")
        _require_token(self.camera_frame_id, "camera_frame_id")
        _require_hash(self.camera_frame_sha256, "camera_frame_sha256")
        _require_hash(self.mass_sample_digest, "mass_sample_digest")
        _require_hash(self.config_hash, "config_hash")
        _require_token(self.firmware_boot_id, "firmware_boot_id")
        for field_name in (
            "wire_task_id",
            "baseline_mass_mg",
            "mass_sample_count",
            "mass_last_sample_seq",
            "firmware_uptime_ms_at_capture",
            "host_captured_monotonic_ms",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
        if not 0 < self.wire_task_id <= 0xFFFFFFFF:
            raise ValueError("wire_task_id must be a positive uint32")
        if self.baseline_mass_mg < 0:
            raise ValueError("baseline_mass_mg cannot be negative")
        if self.mass_sample_count <= 0:
            raise ValueError("mass_sample_count must be positive")
        if (
            self.mass_last_sample_seq < 0
            or self.firmware_uptime_ms_at_capture < 0
            or self.host_captured_monotonic_ms < 0
        ):
            raise ValueError("sample sequence/time cannot be negative")
        if self.firmware_uptime_ms_at_capture > 0xFFFFFFFF:
            raise ValueError("firmware_uptime_ms_at_capture must fit uint32")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "task_id": self.task_id,
            "wire_task_id": self.wire_task_id,
            "baseline_id": self.baseline_id,
            "camera_frame_id": self.camera_frame_id,
            "camera_frame_sha256": self.camera_frame_sha256,
            "baseline_mass_mg": self.baseline_mass_mg,
            "mass_sample_count": self.mass_sample_count,
            "mass_last_sample_seq": self.mass_last_sample_seq,
            "mass_sample_digest": self.mass_sample_digest,
            "config_hash": self.config_hash,
            "firmware_boot_id": self.firmware_boot_id,
            "firmware_uptime_ms_at_capture": self.firmware_uptime_ms_at_capture,
            "stable": self.stable,
            "fresh": self.fresh,
            "host_captured_monotonic_ms": self.host_captured_monotonic_ms,
            "observed_at_utc": self.observed_at_utc,
        }


@dataclass(frozen=True)
class MassEvidence:
    task_id: str
    wire_task_id: int
    result_type: str
    result_frame_seq: int
    result_frame_sha256: str
    terminal_reason: str
    firmware_build_id: str
    firmware_boot_id: str
    execution_backend: str
    baseline_id: str
    baseline_mass_mg: int
    baseline_sample_digest: str
    final_mass_mg: int
    final_mass_min_mg: int
    final_mass_max_mg: int
    first_result_sample_seq: int
    last_result_sample_seq: int
    sample_count: int
    post_stop_sample_count: int
    firmware_completed_uptime_ms: int
    host_result_received_monotonic_ms: int
    stable: bool
    task_result_scale_stable: bool
    pumps_all_off: bool
    fresh: bool
    observed_at_utc: str = ""

    def __post_init__(self) -> None:
        _require_bool_fields(
            self,
            ("stable", "task_result_scale_stable", "pumps_all_off", "fresh"),
        )
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("invalid task_id")
        _require_token(self.baseline_id, "baseline_id")
        _require_hash(self.baseline_sample_digest, "baseline_sample_digest")
        _require_token(self.result_type, "result_type")
        _require_hash(self.result_frame_sha256, "result_frame_sha256")
        _require_token(self.terminal_reason, "terminal_reason")
        _require_token(self.firmware_build_id, "firmware_build_id")
        _require_token(self.firmware_boot_id, "firmware_boot_id")
        _require_token(self.execution_backend, "execution_backend")
        for field_name in (
            "wire_task_id",
            "baseline_mass_mg",
            "final_mass_mg",
            "final_mass_min_mg",
            "final_mass_max_mg",
            "first_result_sample_seq",
            "last_result_sample_seq",
            "sample_count",
            "post_stop_sample_count",
            "firmware_completed_uptime_ms",
            "host_result_received_monotonic_ms",
            "result_frame_seq",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
        if (
            self.baseline_mass_mg < 0
            or self.final_mass_mg < 0
            or self.final_mass_min_mg < 0
            or self.final_mass_max_mg < 0
        ):
            raise ValueError("mass values cannot be negative")
        if not self.final_mass_min_mg <= self.final_mass_mg <= self.final_mass_max_mg:
            raise ValueError("final_mass_mg must lie inside the final mass window")
        if not 0 < self.wire_task_id <= 0xFFFFFFFF:
            raise ValueError("wire_task_id must be a positive uint32")
        if self.first_result_sample_seq < 0:
            raise ValueError("first_result_sample_seq cannot be negative")
        if self.last_result_sample_seq < self.first_result_sample_seq:
            raise ValueError("last_result_sample_seq precedes first sample")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.post_stop_sample_count <= 0:
            raise ValueError("post_stop_sample_count must be positive")
        if self.post_stop_sample_count > self.sample_count:
            raise ValueError("post_stop_sample_count cannot exceed sample_count")
        if self.sample_count > self.last_result_sample_seq - self.first_result_sample_seq + 1:
            raise ValueError("sample_count exceeds the declared sample sequence window")
        if self.firmware_completed_uptime_ms < 0:
            raise ValueError("firmware_completed_uptime_ms cannot be negative")
        if self.firmware_completed_uptime_ms > 0xFFFFFFFF:
            raise ValueError("firmware_completed_uptime_ms must fit uint32")
        if self.host_result_received_monotonic_ms < 0:
            raise ValueError("host_result_received_monotonic_ms cannot be negative")
        _require_u16_nonzero(self.result_frame_seq, "result_frame_seq")

    @property
    def mass_loss_mg(self) -> int:
        return self.baseline_mass_mg - self.final_mass_mg

    @property
    def final_mass_span_mg(self) -> int:
        return self.final_mass_max_mg - self.final_mass_min_mg

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "task_id": self.task_id,
            "wire_task_id": self.wire_task_id,
            "result_type": self.result_type,
            "result_frame_seq": self.result_frame_seq,
            "result_frame_sha256": self.result_frame_sha256,
            "terminal_reason": self.terminal_reason,
            "firmware_build_id": self.firmware_build_id,
            "firmware_boot_id": self.firmware_boot_id,
            "execution_backend": self.execution_backend,
            "baseline_id": self.baseline_id,
            "baseline_mass_mg": self.baseline_mass_mg,
            "baseline_sample_digest": self.baseline_sample_digest,
            "final_mass_mg": self.final_mass_mg,
            "final_mass_min_mg": self.final_mass_min_mg,
            "final_mass_max_mg": self.final_mass_max_mg,
            "final_mass_span_mg": self.final_mass_span_mg,
            "mass_loss_mg": self.mass_loss_mg,
            "first_result_sample_seq": self.first_result_sample_seq,
            "last_result_sample_seq": self.last_result_sample_seq,
            "sample_count": self.sample_count,
            "post_stop_sample_count": self.post_stop_sample_count,
            "firmware_completed_uptime_ms": self.firmware_completed_uptime_ms,
            "host_result_received_monotonic_ms": self.host_result_received_monotonic_ms,
            "stable": self.stable,
            "task_result_scale_stable": self.task_result_scale_stable,
            "pumps_all_off": self.pumps_all_off,
            "fresh": self.fresh,
            "observed_at_utc": self.observed_at_utc,
        }


@dataclass(frozen=True)
class WettingEvidence:
    task_id: str
    baseline_id: str
    baseline_frame_id: str
    baseline_frame_sha256: str
    result_frame_id: str
    result_frame_sha256: str
    target_score: float
    target_threshold: float
    neighbor_score: float
    spill_threshold: float
    captured_monotonic_ms: int
    camera_quality_ok: bool
    fresh: bool
    observed_at_utc: str = ""

    def __post_init__(self) -> None:
        _require_bool_fields(self, ("camera_quality_ok", "fresh"))
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("invalid task_id")
        _require_token(self.baseline_id, "baseline_id")
        _require_token(self.baseline_frame_id, "baseline_frame_id")
        _require_token(self.result_frame_id, "result_frame_id")
        _require_hash(self.baseline_frame_sha256, "baseline_frame_sha256")
        _require_hash(self.result_frame_sha256, "result_frame_sha256")
        if self.baseline_frame_id == self.result_frame_id:
            raise ValueError("baseline and result frame ids must differ")
        if isinstance(self.captured_monotonic_ms, bool) or not isinstance(
            self.captured_monotonic_ms, int
        ):
            raise ValueError("captured_monotonic_ms must be an integer")
        if self.captured_monotonic_ms < 0:
            raise ValueError("captured_monotonic_ms cannot be negative")
        for field_name in (
            "target_score",
            "target_threshold",
            "neighbor_score",
            "spill_threshold",
        ):
            value = getattr(self, field_name)
            _require_finite(value, field_name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{field_name} must be within [0, 1]")

    @property
    def target_passed(self) -> bool:
        return self.target_score >= self.target_threshold

    @property
    def spill_passed(self) -> bool:
        return self.neighbor_score <= self.spill_threshold

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "task_id": self.task_id,
            "baseline_id": self.baseline_id,
            "baseline_frame_id": self.baseline_frame_id,
            "baseline_frame_sha256": self.baseline_frame_sha256,
            "result_frame_id": self.result_frame_id,
            "result_frame_sha256": self.result_frame_sha256,
            "target_score": self.target_score,
            "target_threshold": self.target_threshold,
            "target_passed": self.target_passed,
            "neighbor_score": self.neighbor_score,
            "spill_threshold": self.spill_threshold,
            "spill_passed": self.spill_passed,
            "captured_monotonic_ms": self.captured_monotonic_ms,
            "camera_quality_ok": self.camera_quality_ok,
            "fresh": self.fresh,
            "observed_at_utc": self.observed_at_utc,
        }


@dataclass(frozen=True)
class PhysicalStopEvidence:
    task_id: Optional[str]
    wire_task_id: Optional[int]
    stop_frame_seq: int
    stop_raw_frame_sha256: str
    ack_frame_sha256: Optional[str]
    transcript_id: str
    decoded_command: str
    ack_for_type: str
    ack_for_seq: int
    acked: bool
    fresh: bool
    pumps_all_off: bool
    hard_power_cut_confirmed: bool
    firmware_build_id: str
    firmware_boot_id: str
    execution_backend: str
    observed_at_utc: str = ""

    def __post_init__(self) -> None:
        _require_bool_fields(
            self,
            ("acked", "fresh", "pumps_all_off", "hard_power_cut_confirmed"),
        )
        if self.task_id is not None and not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("invalid task_id")
        if self.wire_task_id is not None:
            if isinstance(self.wire_task_id, bool) or not isinstance(
                self.wire_task_id, int
            ):
                raise ValueError("wire_task_id must be integer or None")
            if not 0 < self.wire_task_id <= 0xFFFFFFFF:
                raise ValueError("wire_task_id must be a positive uint32")
        _require_u16_nonzero(self.stop_frame_seq, "stop_frame_seq")
        _require_u16_nonzero(self.ack_for_seq, "ack_for_seq")
        _require_hash(self.stop_raw_frame_sha256, "stop_raw_frame_sha256")
        if self.ack_frame_sha256 is not None:
            _require_hash(self.ack_frame_sha256, "ack_frame_sha256")
        _require_token(self.transcript_id, "transcript_id")
        _require_token(self.decoded_command, "decoded_command")
        if self.decoded_command != "EMERGENCY_STOP":
            raise ValueError("decoded_command must be EMERGENCY_STOP")
        _require_token(self.ack_for_type, "ack_for_type")
        _require_token(self.firmware_build_id, "firmware_build_id")
        _require_token(self.firmware_boot_id, "firmware_boot_id")
        _require_token(self.execution_backend, "execution_backend")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "task_id": self.task_id,
            "wire_task_id": self.wire_task_id,
            "stop_frame_seq": self.stop_frame_seq,
            "stop_raw_frame_sha256": self.stop_raw_frame_sha256,
            "ack_frame_sha256": self.ack_frame_sha256,
            "transcript_id": self.transcript_id,
            "decoded_command": self.decoded_command,
            "ack_for_type": self.ack_for_type,
            "ack_for_seq": self.ack_for_seq,
            "acked": self.acked,
            "fresh": self.fresh,
            "pumps_all_off": self.pumps_all_off,
            "hard_power_cut_confirmed": self.hard_power_cut_confirmed,
            "firmware_build_id": self.firmware_build_id,
            "firmware_boot_id": self.firmware_boot_id,
            "execution_backend": self.execution_backend,
            "observed_at_utc": self.observed_at_utc,
        }


@dataclass(frozen=True)
class TaskHistoryEntry:
    task_id: str
    task_seq: int
    request_fingerprint: str

    def __post_init__(self) -> None:
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("invalid task_id")
        if self.task_seq <= 0:
            raise ValueError("task_seq must be positive")
        if self.task_seq > 0xFFFFFFFF:
            raise ValueError("task_seq must fit uint32")
        _require_hash(self.request_fingerprint, "request_fingerprint")


@dataclass(frozen=True)
class AdmissionResult:
    status: AdmissionStatus
    state: MachineState
    task_id: Optional[str]
    fault_code: FaultCode = FaultCode.NONE
    detail: str = ""

    @property
    def may_create_physical_command(self) -> bool:
        """Only a new admission may continue toward a new physical command."""

        return self.status is AdmissionStatus.ACCEPTED


@dataclass(frozen=True)
class OperationResult:
    accepted: bool
    state: MachineState
    completion_class: CompletionClass
    fault_code: FaultCode = FaultCode.NONE
    detail: str = ""


@dataclass(frozen=True)
class StateSnapshot:
    state: MachineState
    completion_class: CompletionClass
    highest_verified_class: CompletionClass
    active_task: Optional[TaskRequest]
    last_fault: FaultCode
    fault_detail: str
    high_watermark_task_seq: int
    pending_arm_frame_seq: Optional[int]
    pending_clear_frame_seq: Optional[int]
    pending_stop_frame_seq: Optional[int]
    clear_estop_acknowledged: bool
    physical_stop_required: bool
    physical_stop_confirmed: bool
    boot_session_id: str

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "state": self.state.value,
            "completion_class": self.completion_class.value,
            "highest_verified_class": self.highest_verified_class.value,
            "active_task": self.active_task.to_dict() if self.active_task else None,
            "last_fault": self.last_fault.value,
            "fault_detail": self.fault_detail,
            "high_watermark_task_seq": self.high_watermark_task_seq,
            "pending_arm_frame_seq": self.pending_arm_frame_seq,
            "pending_clear_frame_seq": self.pending_clear_frame_seq,
            "pending_stop_frame_seq": self.pending_stop_frame_seq,
            "clear_estop_acknowledged": self.clear_estop_acknowledged,
            "physical_stop_required": self.physical_stop_required,
            "physical_stop_confirmed": self.physical_stop_confirmed,
            "boot_session_id": self.boot_session_id,
        }
