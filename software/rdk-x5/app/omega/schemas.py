"""Strict dependency-free contracts for the RootScope-Ω evidence core."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


EVIDENCE_NODE_SCHEMA = "rootscope.omega.evidence-node.v1"
AUTHORITY_SCHEMA = "rootscope.omega.zero-authority.v1"
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_RESERVED_AUTHORITY_KEYS = frozenset(
    {
        "execution_authority",
        "physical_authority",
        "serial_write",
        "pump_command",
        "state_machine_write",
        "tool_execution",
        "actuator_access",
        "irrigation_execution",
    }
)


class OmegaContractError(ValueError):
    """A payload violates a frozen RootScope-Ω contract."""


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class EvidenceKind(StringEnum):
    QUALITY = "QUALITY"
    SEMANTIC = "SEMANTIC"
    GEOMETRY = "GEOMETRY"
    OOD = "OOD"
    SAFETY = "SAFETY"
    ACK = "ACK"
    MASS = "MASS"
    WETTING = "WETTING"
    SOURCE = "SOURCE"
    CLAIM = "CLAIM"


class EvidenceVerdict(StringEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


class EvidenceMode(StringEnum):
    SIMULATION = "SIMULATION"
    SEALED_REPLAY = "SEALED_REPLAY"
    READ_ONLY = "READ_ONLY"


class FailureMode(StringEnum):
    NORMAL = "NORMAL"
    PERCEPTION_ERROR = "PERCEPTION_ERROR"
    CAMERA_QUALITY = "CAMERA_QUALITY"
    SENSOR_STALE = "SENSOR_STALE"
    ACTUATOR_NO_ACK = "ACTUATOR_NO_ACK"
    MASS_ANOMALY = "MASS_ANOMALY"
    WETTING_MISS = "WETTING_MISS"
    NEIGHBOR_SPILL = "NEIGHBOR_SPILL"
    SAFETY_INTERLOCK = "SAFETY_INTERLOCK"
    OOD_CONFLICT = "OOD_CONFLICT"


class CalibrationLevel(StringEnum):
    INTERVAL_ONLY = "INTERVAL_ONLY"
    EXPERIMENTAL_CALIBRATED = "EXPERIMENTAL_CALIBRATED"


class CoreStatus(StringEnum):
    CLEAR = "CLEAR"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
    BLOCKING = "BLOCKING"


class EvidenceActionType(StringEnum):
    HOLD = "HOLD"
    RECAPTURE_IMAGE = "RECAPTURE_IMAGE"
    REWEIGH = "REWEIGH"
    WAIT_FOR_FRESH_TELEMETRY = "WAIT_FOR_FRESH_TELEMETRY"
    VERIFY_SAFETY_INTERLOCK = "VERIFY_SAFETY_INTERLOCK"
    REQUEST_OPERATOR_REVIEW = "REQUEST_OPERATOR_REVIEW"


class ObservationOutcome(StringEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


def require_exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], context: str
) -> None:
    if not isinstance(value, Mapping):
        raise OmegaContractError(f"{context} must be an object")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise OmegaContractError(
            f"{context} key mismatch: missing={missing}, unknown={unknown}"
        )


def require_safe_token(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SAFE_TOKEN_RE.fullmatch(value):
        raise OmegaContractError(f"{field_name} must be one safe ASCII token")
    return value


def require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _HEX_64_RE.fullmatch(value):
        raise OmegaContractError(f"{field_name} must be lowercase SHA-256")
    return value


def require_finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OmegaContractError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OmegaContractError(f"{field_name} must be finite")
    return result


def require_probability(value: Any, field_name: str) -> float:
    result = require_finite(value, field_name)
    if not 0.0 <= result <= 1.0:
        raise OmegaContractError(f"{field_name} must be within [0, 1]")
    return result


def strict_json_value(value: Any, *, path: str = "payload", depth: int = 0) -> Any:
    """Return an immutable-canonicalisable JSON value or fail closed."""

    if depth > 8:
        raise OmegaContractError(f"{path} exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return require_finite(value, path)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not _PAYLOAD_KEY_RE.fullmatch(key):
                raise OmegaContractError(f"{path} contains an invalid key")
            if key.lower() in _RESERVED_AUTHORITY_KEYS:
                raise OmegaContractError(
                    f"{path}.{key} is reserved for the authority capsule"
                )
            result[key] = strict_json_value(
                item, path=f"{path}.{key}", depth=depth + 1
            )
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            raise OmegaContractError(f"{path} exceeds maximum array length")
        return [
            strict_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise OmegaContractError(f"{path} contains unsupported {type(value).__name__}")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OmegaContractError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def enum_value(enum_type: type[StringEnum], value: Any, field_name: str) -> StringEnum:
    if not isinstance(value, str):
        raise OmegaContractError(f"{field_name} must be a string enum")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise OmegaContractError(f"{field_name} has unsupported value {value!r}") from exc


@dataclass(frozen=True)
class AuthorityBoundary:
    execution_authority: bool = False
    physical_authority: bool = False
    serial_write: bool = False
    pump_command: bool = False
    state_machine_write: bool = False
    tool_execution: bool = False
    actuator_access: bool = False

    def __post_init__(self) -> None:
        for name, value in self.to_dict(include_schema=False).items():
            if not isinstance(value, bool) or value is not False:
                raise OmegaContractError(f"authority.{name} must be exactly false")

    def to_dict(self, *, include_schema: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "execution_authority": self.execution_authority,
            "physical_authority": self.physical_authority,
            "serial_write": self.serial_write,
            "pump_command": self.pump_command,
            "state_machine_write": self.state_machine_write,
            "tool_execution": self.tool_execution,
            "actuator_access": self.actuator_access,
        }
        if include_schema:
            payload = {"schema_version": AUTHORITY_SCHEMA, **payload}
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AuthorityBoundary":
        expected = {
            "schema_version",
            "execution_authority",
            "physical_authority",
            "serial_write",
            "pump_command",
            "state_machine_write",
            "tool_execution",
            "actuator_access",
        }
        require_exact_keys(value, expected, "authority")
        if value["schema_version"] != AUTHORITY_SCHEMA:
            raise OmegaContractError("authority schema mismatch")
        return cls(
            execution_authority=value["execution_authority"],
            physical_authority=value["physical_authority"],
            serial_write=value["serial_write"],
            pump_command=value["pump_command"],
            state_machine_write=value["state_machine_write"],
            tool_execution=value["tool_execution"],
            actuator_access=value["actuator_access"],
        )


ZERO_AUTHORITY = AuthorityBoundary()


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    kind: EvidenceKind
    verdict: EvidenceVerdict
    mode: EvidenceMode
    source_id: str
    observed_at_ms: int
    payload: Mapping[str, Any]
    parents: tuple[str, ...] = ()
    authority: AuthorityBoundary = field(default_factory=AuthorityBoundary)
    content_sha256: str = ""

    def __post_init__(self) -> None:
        require_safe_token(self.node_id, "node_id")
        require_safe_token(self.source_id, "source_id")
        if not isinstance(self.kind, EvidenceKind):
            raise OmegaContractError("kind must be EvidenceKind")
        if not isinstance(self.verdict, EvidenceVerdict):
            raise OmegaContractError("verdict must be EvidenceVerdict")
        if not isinstance(self.mode, EvidenceMode):
            raise OmegaContractError("mode must be EvidenceMode")
        if (
            isinstance(self.observed_at_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or self.observed_at_ms < 0
        ):
            raise OmegaContractError("observed_at_ms must be a non-negative integer")
        if not isinstance(self.parents, tuple):
            raise OmegaContractError("parents must be a tuple")
        for parent in self.parents:
            require_safe_token(parent, "parent")
        if tuple(sorted(set(self.parents))) != self.parents:
            raise OmegaContractError("parents must be unique and sorted")
        if self.node_id in self.parents:
            raise OmegaContractError("an evidence node cannot parent itself")
        if not isinstance(self.authority, AuthorityBoundary):
            raise OmegaContractError("authority must be AuthorityBoundary")
        canonical_payload = strict_json_value(self.payload)
        if not isinstance(canonical_payload, dict):
            raise OmegaContractError("payload must be an object")
        object.__setattr__(self, "payload", canonical_payload)
        expected = canonical_sha256(self.unsigned_dict())
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", expected)
        elif require_sha256(self.content_sha256, "content_sha256") != expected:
            raise OmegaContractError("evidence node content hash mismatch")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_NODE_SCHEMA,
            "node_id": self.node_id,
            "kind": self.kind.value,
            "verdict": self.verdict.value,
            "mode": self.mode.value,
            "source_id": self.source_id,
            "observed_at_ms": self.observed_at_ms,
            "payload": self.payload,
            "parents": list(self.parents),
            "authority": self.authority.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "content_sha256": self.content_sha256}

    @classmethod
    def create(
        cls,
        *,
        node_id: str,
        kind: EvidenceKind,
        verdict: EvidenceVerdict,
        mode: EvidenceMode,
        source_id: str,
        observed_at_ms: int,
        payload: Mapping[str, Any],
        parents: Sequence[str] = (),
    ) -> "EvidenceNode":
        return cls(
            node_id=node_id,
            kind=kind,
            verdict=verdict,
            mode=mode,
            source_id=source_id,
            observed_at_ms=observed_at_ms,
            payload=payload,
            parents=tuple(sorted(parents)),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceNode":
        expected = {
            "schema_version",
            "node_id",
            "kind",
            "verdict",
            "mode",
            "source_id",
            "observed_at_ms",
            "payload",
            "parents",
            "authority",
            "content_sha256",
        }
        require_exact_keys(value, expected, "evidence node")
        if value["schema_version"] != EVIDENCE_NODE_SCHEMA:
            raise OmegaContractError("evidence node schema mismatch")
        if not isinstance(value["parents"], list):
            raise OmegaContractError("evidence node parents must be an array")
        return cls(
            node_id=value["node_id"],
            kind=enum_value(EvidenceKind, value["kind"], "kind"),  # type: ignore[arg-type]
            verdict=enum_value(EvidenceVerdict, value["verdict"], "verdict"),  # type: ignore[arg-type]
            mode=enum_value(EvidenceMode, value["mode"], "mode"),  # type: ignore[arg-type]
            source_id=value["source_id"],
            observed_at_ms=value["observed_at_ms"],
            payload=value["payload"],
            parents=tuple(value["parents"]),
            authority=AuthorityBoundary.from_dict(value["authority"]),
            content_sha256=value["content_sha256"],
        )
