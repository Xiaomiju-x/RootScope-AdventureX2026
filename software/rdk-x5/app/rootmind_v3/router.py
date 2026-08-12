"""Fail-closed logical micro-cluster routing for RootMind.

The "cluster" is a set of logical roles, not simultaneously resident model
copies.  The router never starts a process and has no actuator interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Collection, Mapping


class ModelRole(str, Enum):
    TEMPLATE = "DETERMINISTIC_TEMPLATE"
    FAST = "ROOTMIND_FAST_05B"
    DEEP = "ROOTMIND_DEEP_17B"


@dataclass(frozen=True)
class RootMindRequest:
    intent: str
    evidence_ids: tuple[str, ...]
    question: str
    deadline_ms: int = 1800


@dataclass(frozen=True)
class RootMindRoute:
    selected: ModelRole
    reason_codes: tuple[str, ...]
    max_tokens: int
    timeout_ms: int
    one_resident_model: bool = True
    execution_authority: bool = False


@dataclass(frozen=True)
class SafetyCompileResult:
    decision: str
    transformation: str
    reason_codes: tuple[str, ...]
    raw_sha256: str
    final_sha256: str | None
    final_payload: Mapping[str, Any] | None
    execution_authority: bool = False


class RootMindRouter:
    """Choose one read-only explanation backend from a resource snapshot."""

    DEEP_INTENTS = frozenset({"DEFENSE_QA", "COUNTERFACTUAL_EXPLANATION"})

    def route(
        self,
        request: RootMindRequest,
        resources: Mapping[str, Any],
    ) -> RootMindRoute:
        if not request.evidence_ids:
            return self._template("CITATION_SET_EMPTY")
        if request.deadline_ms < 400:
            return self._template("DEADLINE_TOO_SHORT")
        if bool(resources.get("thermal_hold")):
            return self._template("THERMAL_HOLD")
        if int(resources.get("available_memory_mib", 0)) < 420:
            return self._template("MEMORY_RESERVE_GATE")
        if bool(resources.get("foreground_vision_busy")):
            return self._template("VISION_PRIORITY_GATE")
        if request.intent in self.DEEP_INTENTS:
            if (
                bool(resources.get("deep_model_qualified"))
                and int(resources.get("available_memory_mib", 0)) >= 1050
                and request.deadline_ms >= 1200
            ):
                return RootMindRoute(
                    ModelRole.DEEP,
                    ("DEEP_INTENT_AND_RESOURCE_GATE_PASS",),
                    max_tokens=220,
                    timeout_ms=min(request.deadline_ms, 5000),
                )
            if bool(resources.get("fast_model_qualified")):
                return RootMindRoute(
                    ModelRole.FAST,
                    ("DEEP_GATE_FAILED_FAST_FALLBACK",),
                    max_tokens=160,
                    timeout_ms=min(request.deadline_ms, 2600),
                )
            return self._template("NO_QUALIFIED_LOCAL_MODEL")
        if bool(resources.get("fast_model_qualified")):
            return RootMindRoute(
                ModelRole.FAST,
                ("FAST_INTENT_RESOURCE_GATE_PASS",),
                max_tokens=140,
                timeout_ms=min(request.deadline_ms, 2200),
            )
        return self._template("NO_QUALIFIED_LOCAL_MODEL")

    @staticmethod
    def _template(reason: str) -> RootMindRoute:
        return RootMindRoute(
            ModelRole.TEMPLATE,
            (reason,),
            max_tokens=0,
            timeout_ms=0,
        )


REQUIRED_KEYS = frozenset(
    {
        "authority",
        "evidence_ids",
        "observation_summary",
        "proposed_explanation",
        "reason_codes",
        "uncertainty",
    }
)
FORBIDDEN_TEXT = (
    "serial.write",
    "gpio.output",
    "pump_on",
    "open_pump",
    "tool_call",
    "<tool_call>",
    "打开水泵",
    "开启水泵",
    "启动水泵",
    "运行水泵",
    "立即灌溉",
    "执行灌溉",
    "发送串口",
    "串口发送",
    "串口写入",
    "向 stm32 发送",
    "给 stm32 发送",
    "gpio 写",
    "gpio置",
)


def validate_readonly_response(
    payload: str | Mapping[str, Any],
    citation_allowlist: Collection[str],
) -> dict[str, Any]:
    """Validate generated JSON and reject unsupported or action-bearing output."""

    value = json.loads(payload) if isinstance(payload, str) else dict(payload)
    if set(value) != REQUIRED_KEYS:
        raise ValueError("RootMind response keys do not exactly match the contract")
    if value["authority"] is not False:
        raise ValueError("RootMind authority must be false")
    evidence_ids = value["evidence_ids"]
    if not isinstance(evidence_ids, list) or not evidence_ids:
        raise ValueError("RootMind response requires at least one citation")
    allowed = set(citation_allowlist)
    if any(not isinstance(item, str) or item not in allowed for item in evidence_ids):
        raise ValueError("RootMind response contains a citation outside the allowlist")
    if not isinstance(value["reason_codes"], list) or not value["reason_codes"]:
        raise ValueError("RootMind response requires reason codes")
    serialized = json.dumps(value, ensure_ascii=False).casefold()
    if any(marker in serialized for marker in FORBIDDEN_TEXT):
        raise ValueError("RootMind response contains an action/tool marker")
    return value


def compile_readonly_response(
    payload: str | Mapping[str, Any],
    retrieved_evidence_ids: Collection[str],
    *,
    required_reason_codes: Collection[str] = (),
) -> SafetyCompileResult:
    """Accept a fully valid raw response or replace it with a fixed HOLD template.

    Rejected model prose is never copied into the deterministic replacement.
    Raw-model and compiled-system metrics can therefore remain separate.
    """

    raw_serialization_error = False
    try:
        raw = (
            payload
            if isinstance(payload, str)
            else json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError):
        raw_serialization_error = True
        raw = f"<UNSERIALIZABLE_ROOTMIND_PAYLOAD:{type(payload).__name__}>"
    raw_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    retrieved_contract_invalid = any(
        not isinstance(item, str) or not item
        for item in retrieved_evidence_ids
    )
    retrieved = tuple(
        dict.fromkeys(
            item
            for item in retrieved_evidence_ids
            if isinstance(item, str) and item
        )
    )
    required_contract_invalid = any(
        not isinstance(item, str) or not item for item in required_reason_codes
    )
    required = tuple(
        dict.fromkeys(
            item for item in required_reason_codes if isinstance(item, str) and item
        )
    )
    failure_reason = "MODEL_OUTPUT_REJECTED"
    try:
        if raw_serialization_error:
            raise ValueError("RootMind raw serialization invalid")
        if retrieved_contract_invalid:
            raise ValueError("RootMind retrieved evidence contract invalid")
        if required_contract_invalid:
            raise ValueError("RootMind required reason contract invalid")
        value = validate_readonly_response(payload, retrieved)
        observed_reasons = set(value["reason_codes"])
        missing = [item for item in required if item not in observed_reasons]
        if missing:
            failure_reason = "MODEL_REQUIRED_REASON_MISSING"
            raise ValueError("RootMind response is missing required reason codes")
        if (
            "ADVERSARIAL_REQUEST_REJECTED" in required
            and not value["proposed_explanation"].startswith("拒绝")
        ):
            failure_reason = "MODEL_ADVERSARIAL_SEMANTIC_REJECTION_MISSING"
            raise ValueError("RootMind adversarial explanation does not reject")
        final = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return SafetyCompileResult(
            decision="ACCEPT_RAW",
            transformation="NONE",
            reason_codes=("RAW_CONTRACT_PASS",),
            raw_sha256=raw_sha256,
            final_sha256=hashlib.sha256(final).hexdigest(),
            final_payload=value,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        detail = str(exc)
        if "JSON" in detail or isinstance(exc, json.JSONDecodeError):
            failure_reason = "MODEL_JSON_INVALID"
        elif "raw serialization" in detail:
            failure_reason = "MODEL_RAW_SERIALIZATION_INVALID"
        elif "retrieved evidence contract" in detail:
            failure_reason = "RETRIEVED_EVIDENCE_CONTRACT_INVALID"
            retrieved = ()
        elif "required reason contract" in detail:
            failure_reason = "REQUIRED_REASON_CONTRACT_INVALID"
        elif "keys" in detail:
            failure_reason = "MODEL_SCHEMA_INVALID"
        elif "authority" in detail:
            failure_reason = "MODEL_AUTHORITY_VIOLATION"
        elif "citation" in detail:
            failure_reason = "MODEL_CITATION_INVALID"
        elif "action/tool" in detail:
            failure_reason = "MODEL_ACTION_MARKER"
        elif "required reason" in detail:
            failure_reason = "MODEL_REQUIRED_REASON_MISSING"
        elif "adversarial explanation" in detail:
            failure_reason = "MODEL_ADVERSARIAL_SEMANTIC_REJECTION_MISSING"
    if not retrieved:
        return SafetyCompileResult(
            decision="REJECT_NO_VALID_CITATION",
            transformation="DROP_MODEL_OUTPUT",
            reason_codes=("MODEL_OUTPUT_REJECTED", failure_reason, "CITATION_SET_EMPTY"),
            raw_sha256=raw_sha256,
            final_sha256=None,
            final_payload=None,
        )
    reason_codes = tuple(
        dict.fromkeys(("MODEL_OUTPUT_REJECTED", failure_reason, *required))
    )
    fallback = {
        "authority": False,
        "evidence_ids": list(retrieved),
        "observation_summary": "模型输出未通过 RootMind 只读安全合同。",
        "proposed_explanation": "拒绝采用模型输出；保持 HOLD 并请求人工复核。",
        "reason_codes": list(reason_codes),
        "uncertainty": "原始模型正文已丢弃，未形成任何动作或完成结论。",
    }
    validate_readonly_response(fallback, retrieved)
    final = json.dumps(
        fallback,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return SafetyCompileResult(
        decision="REJECT_TO_DETERMINISTIC_TEMPLATE",
        transformation="DETERMINISTIC_REPLACEMENT",
        reason_codes=reason_codes,
        raw_sha256=raw_sha256,
        final_sha256=hashlib.sha256(final).hexdigest(),
        final_payload=fallback,
    )
