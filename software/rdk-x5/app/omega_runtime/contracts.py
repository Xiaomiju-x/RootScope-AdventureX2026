"""Strict, dependency-free RootScope-Ω runtime contracts.

The Ω layer is deliberately advisory.  These contracts make it impossible for
the laptop/replay implementation to silently claim serial, GPIO or pump
authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class RuntimeMode(StringEnum):
    LIVE = "LIVE"
    REPLAY = "REPLAY"
    SIMULATION = "SIMULATION"
    REMOTE = "REMOTE"


class SafetyDecision(StringEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    HOLD = "HOLD"


class EvidenceAction(StringEnum):
    NONE = "NONE"
    RECAPTURE = "RECAPTURE"
    REWEIGH = "REWEIGH"
    WAIT = "WAIT"
    REVIEW = "REVIEW"
    HOLD = "HOLD"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite values cannot enter a canonical receipt")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_token(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a safe non-empty token")


def _require_hash(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_tokens(values: Sequence[str], field_name: str) -> None:
    if not values:
        raise ValueError(f"{field_name} cannot be empty")
    for value in values:
        _require_token(value, field_name)


@dataclass(frozen=True)
class AuthorityFlags:
    """Authority facts for the Ω advisory/replay layer."""

    execution_authority: bool = False
    serial_write: bool = False
    gpio_write: bool = False
    pump_control: bool = False
    state_machine_write: bool = False
    physical_closure: bool = False

    def __post_init__(self) -> None:
        if any(
            (
                self.execution_authority,
                self.serial_write,
                self.gpio_write,
                self.pump_control,
                self.state_machine_write,
                self.physical_closure,
            )
        ):
            raise ValueError("RootScope-Ω v3 advisory runtime must have zero authority")

    def to_dict(self) -> Mapping[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class BackendCapsule:
    profile: str
    runtime_mode: RuntimeMode
    decision_backend_actual: str
    vision_backend_actual: str
    retrieval_backend_actual: str
    explanation_backend_actual: str
    release_id: str
    bpu_model_qualified: bool
    local_llm_active: bool
    remote_shadow_active: bool
    fallback_reasons: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "profile",
            "decision_backend_actual",
            "vision_backend_actual",
            "retrieval_backend_actual",
            "explanation_backend_actual",
            "release_id",
        ):
            _require_token(getattr(self, name), name)
        if not isinstance(self.runtime_mode, RuntimeMode):
            raise ValueError("runtime_mode must be RuntimeMode")
        for reason in self.fallback_reasons:
            _require_token(reason, "fallback_reasons")
        if self.profile == "SAFE_CPU" and (
            self.bpu_model_qualified
            or self.local_llm_active
            or self.remote_shadow_active
        ):
            raise ValueError("SAFE_CPU cannot silently claim BPU, LLM or remote shadow")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "profile": self.profile,
            "runtime_mode": self.runtime_mode.value,
            "decision_backend_actual": self.decision_backend_actual,
            "vision_backend_actual": self.vision_backend_actual,
            "retrieval_backend_actual": self.retrieval_backend_actual,
            "explanation_backend_actual": self.explanation_backend_actual,
            "release_id": self.release_id,
            "bpu_model_qualified": self.bpu_model_qualified,
            "local_llm_active": self.local_llm_active,
            "remote_shadow_active": self.remote_shadow_active,
            "fallback_reasons": list(self.fallback_reasons),
        }


@dataclass(frozen=True)
class DecisionProjection:
    safety_decision: SafetyDecision
    reason_codes: Tuple[str, ...]
    evidence_action: EvidenceAction
    terminal_state: str
    completion_claim: str
    proposal_only: bool = True
    physical_command_emitted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.safety_decision, SafetyDecision):
            raise ValueError("safety_decision must be SafetyDecision")
        if not isinstance(self.evidence_action, EvidenceAction):
            raise ValueError("evidence_action must be EvidenceAction")
        _require_tokens(self.reason_codes, "reason_codes")
        _require_token(self.terminal_state, "terminal_state")
        _require_token(self.completion_claim, "completion_claim")
        if not self.proposal_only or self.physical_command_emitted:
            raise ValueError("Ω replay output must remain proposal-only")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "safety_decision": self.safety_decision.value,
            "reason_codes": list(self.reason_codes),
            "evidence_action": self.evidence_action.value,
            "terminal_state": self.terminal_state,
            "completion_claim": self.completion_claim,
            "proposal_only": self.proposal_only,
            "physical_command_emitted": self.physical_command_emitted,
        }

    @property
    def projection_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class DecisionReceipt:
    run_id: str
    event_id: str
    case_id: str
    evidence_dag_root: str
    belief_state_hash: str
    failure_core_hash: str
    rb_voe_plan_hash: str
    claim_ledger_root: str
    projection: DecisionProjection
    backend: BackendCapsule
    authority: AuthorityFlags
    generated_at_utc: str
    schema_version: str = "rootscope.omega.decision-receipt.v1"

    def __post_init__(self) -> None:
        for name in ("run_id", "event_id", "case_id", "schema_version"):
            _require_token(getattr(self, name), name)
        for name in (
            "evidence_dag_root",
            "belief_state_hash",
            "failure_core_hash",
            "rb_voe_plan_hash",
            "claim_ledger_root",
        ):
            _require_hash(getattr(self, name), name)
        if not isinstance(self.projection, DecisionProjection):
            raise ValueError("projection must be DecisionProjection")
        if not isinstance(self.backend, BackendCapsule):
            raise ValueError("backend must be BackendCapsule")
        if not isinstance(self.authority, AuthorityFlags):
            raise ValueError("authority must be AuthorityFlags")
        if not isinstance(self.generated_at_utc, str) or not self.generated_at_utc:
            raise ValueError("generated_at_utc is required")

    def fingerprint_payload(self) -> Mapping[str, Any]:
        """Timestamp-free payload: identical evidence yields an identical receipt."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "event_id": self.event_id,
            "case_id": self.case_id,
            "evidence_dag_root": self.evidence_dag_root,
            "belief_state_hash": self.belief_state_hash,
            "failure_core_hash": self.failure_core_hash,
            "rb_voe_plan_hash": self.rb_voe_plan_hash,
            "claim_ledger_root": self.claim_ledger_root,
            "projection": self.projection.to_dict(),
            "backend": self.backend.to_dict(),
            "authority": self.authority.to_dict(),
        }

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.fingerprint_payload())

    def to_dict(self) -> Mapping[str, Any]:
        return {
            **self.fingerprint_payload(),
            "generated_at_utc": self.generated_at_utc,
            "decision_projection_sha256": self.projection.projection_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True)
class TruthRibbon:
    mode: RuntimeMode
    profile: str
    backend_actual: str
    evidence_state: str
    evidence_fresh: bool
    receipt_sha256: str
    authority: AuthorityFlags
    physical_completion_claim: bool
    warnings: Tuple[str, ...]
    cloud_shadow_influenced_decision: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RuntimeMode):
            raise ValueError("mode must be RuntimeMode")
        for name in ("profile", "backend_actual", "evidence_state"):
            _require_token(getattr(self, name), name)
        _require_hash(self.receipt_sha256, "receipt_sha256")
        if not isinstance(self.authority, AuthorityFlags):
            raise ValueError("authority must be AuthorityFlags")
        if self.physical_completion_claim:
            raise ValueError("simulation/replay ribbon cannot claim physical completion")
        if self.cloud_shadow_influenced_decision:
            raise ValueError("remote shadow cannot influence the authority projection")
        _require_tokens(self.warnings, "warnings")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "mode": self.mode.value,
            "profile": self.profile,
            "backend_actual": self.backend_actual,
            "evidence_state": self.evidence_state,
            "evidence_fresh": self.evidence_fresh,
            "receipt_sha256": self.receipt_sha256,
            "authority": self.authority.to_dict(),
            "physical_completion_claim": self.physical_completion_claim,
            "cloud_shadow_influenced_decision": (
                self.cloud_shadow_influenced_decision
            ),
            "warnings": list(self.warnings),
        }
