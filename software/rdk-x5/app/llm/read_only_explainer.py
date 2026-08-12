"""Fail-closed, read-only explanations from a loopback llama.cpp server.

The local language model is deliberately outside the RootScope control path.
It receives a bounded copy of already-structured evidence and may only return a
strict explanation object.  It has no tool interface, no serial handle, no
state-machine reference and no irrigation authority.  Invalid, unavailable or
overlong model output is replaced with a deterministic evidence summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import http.client
import ipaddress
import json
import math
from pathlib import Path
import re
import socket
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


SCHEMA_VERSION = "rootscope.read_only_explanation.v1"
PROMPT_VERSION = "rootscope-read-only-explainer/1.0.0"
MAX_EVIDENCE_BYTES = 65_536
MAX_RESPONSE_BYTES = 65_536
MAX_QUESTION_CHARS = 500
MAX_LIST_ITEMS = 12
MAX_ITEM_CHARS = 500
ALLOWED_STATUS = "EXPLANATION_ONLY"
AUTHORITY = {
    "irrigation_execution": False,
    "pump_command": False,
    "serial_write": False,
    "state_machine_write": False,
    "tool_execution": False,
    "actuator_access": False,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_DIRECTIVES = (
    "启动水泵",
    "打开水泵",
    "执行灌溉",
    "立即灌溉",
    "清除急停",
    "写入串口",
    "send serial",
    "clear estop",
    "/dev/tty",
)
_FORBIDDEN_CONTROL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:start|activate|turn\s*on|启动|打开|开启|运行|通电).{0,12}(?:pump|水泵|泵)",
        r"(?:pump|水泵|泵).{0,12}(?:start|activate|turn\s*on|启动|打开|开启|运行|通电)",
        r"(?:execute|start|立即|执行|开始).{0,12}(?:irrigation|watering|灌溉|浇水)",
        r"(?:irrigation|watering|灌溉|浇水).{0,12}(?:execute|start|立即|执行|开始)",
        r"(?:write|send|写入|写|发送).{0,12}(?:serial|uart|串口|/dev/tty)",
        r"(?:clear|reset|解除|清除|复位).{0,12}(?:estop|e-stop|急停)",
        r"(?:modify|write|transition|修改|写入|切换).{0,12}(?:state\s*machine|状态机)",
    )
)
_SAFE_EVIDENCE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bounded_mapping(value: Mapping[str, Any], *, name: str) -> tuple[dict[str, Any], bytes]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    copied = json.loads(_canonical_json_bytes(dict(value)).decode("utf-8"))
    encoded = _canonical_json_bytes(copied)
    if len(encoded) > MAX_EVIDENCE_BYTES:
        raise ValueError(f"{name} exceeds {MAX_EVIDENCE_BYTES} canonical UTF-8 bytes")
    return copied, encoded


def _is_loopback_host(host: str) -> bool:
    lowered = host.strip().lower()
    if lowered == "localhost":
        return True
    try:
        return ipaddress.ip_address(lowered).is_loopback
    except ValueError:
        return False


def _contains_forbidden_directive(value: str) -> bool:
    lowered = value.lower()
    return any(phrase in lowered for phrase in _FORBIDDEN_DIRECTIVES) or any(
        pattern.search(value) for pattern in _FORBIDDEN_CONTROL_PATTERNS
    )


def _safe_evidence_token(value: Any) -> str:
    """Return only inert identifier-like evidence, never arbitrary echoed prose."""

    if isinstance(value, str):
        cleaned = value.strip()
        if _SAFE_EVIDENCE_TOKEN_RE.fullmatch(cleaned) and not _contains_forbidden_directive(cleaned):
            return cleaned
    return "UNTRUSTED_VALUE_REDACTED"


def _resolve_numeric_loopback(host: str, port: int) -> str:
    """Resolve once, reject mixed answers, and connect to a numeric loopback IP.

    Connecting to the validated numeric result avoids both redirect following and
    a second hostname lookup between policy validation and socket creation.
    """

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses = {
            ipaddress.ip_address(answer[4][0].split("%", 1)[0]) for answer in answers
        }
        if not addresses or any(not address.is_loopback for address in addresses):
            raise ValueError("endpoint hostname did not resolve exclusively to loopback")
        # Prefer IPv4 for the default localhost deployment while remaining
        # deterministic when only IPv6 loopback is available.
        literal = sorted(addresses, key=lambda item: (item.version, str(item)))[0]
    if not literal.is_loopback:
        raise ValueError("endpoint did not resolve to loopback")
    return str(literal)


def _post_loopback_json(config: "ExplanationConfig", body: bytes) -> bytes:
    """POST directly to one numeric loopback address without redirect support."""

    parsed = urlsplit(config.endpoint)
    assert parsed.hostname is not None and parsed.port is not None
    connect_host = _resolve_numeric_loopback(parsed.hostname, parsed.port)
    connection = http.client.HTTPConnection(
        connect_host,
        parsed.port,
        timeout=float(config.timeout_seconds),
    )
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"llama.cpp HTTP status must be 200, got {response.status}")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    finally:
        connection.close()
    return raw


@dataclass(frozen=True)
class ExplanationConfig:
    """Configuration for one optional local llama.cpp endpoint."""

    enabled: bool = False
    endpoint: str = "http://127.0.0.1:9080"
    model_label: str = "qwen2-0.5b-q4km-rootscope-read-only"
    model_sha256: str | None = None
    timeout_seconds: float = 12.0
    max_tokens: int = 384
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        parsed = urlsplit(self.endpoint)
        if parsed.scheme != "http" or not parsed.hostname or not _is_loopback_host(parsed.hostname):
            raise ValueError("endpoint must be an HTTP loopback address")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("endpoint must not contain credentials, query or fragment")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("endpoint port is invalid") from exc
        if port is None or not 1024 <= port <= 65535:
            raise ValueError("endpoint must include an unprivileged port")
        if parsed.path not in {"", "/"}:
            raise ValueError("endpoint must not contain an API path")
        if not isinstance(self.model_label, str) or not self.model_label.strip():
            raise ValueError("model_label must be non-empty")
        if self.model_sha256 is not None and not _SHA256_RE.fullmatch(self.model_sha256):
            raise ValueError("model_sha256 must be null or lowercase SHA-256")
        if self.enabled and self.model_sha256 is None:
            raise ValueError("model_sha256 is required when the LLM endpoint is enabled")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be numeric")
        if not math.isfinite(float(self.timeout_seconds)) or not 0.1 <= float(self.timeout_seconds) <= 60.0:
            raise ValueError("timeout_seconds must be within 0.1..60")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or not 32 <= self.max_tokens <= 1024:
            raise ValueError("max_tokens must be an integer within 32..1024")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValueError("temperature must be numeric")
        if not math.isfinite(float(self.temperature)) or not 0.0 <= float(self.temperature) <= 0.3:
            raise ValueError("temperature must be within 0..0.3")

    @property
    def chat_completions_url(self) -> str:
        return self.endpoint.rstrip("/") + "/v1/chat/completions"


def build_explanation_messages(
    snapshot: Mapping[str, Any],
    *,
    question: str = "请解释当前 RootScope 证据、风险和不确定性。",
) -> tuple[list[dict[str, str]], Mapping[str, Any]]:
    """Build a bounded prompt and return immutable prompt provenance."""

    bounded, snapshot_bytes = _bounded_mapping(snapshot, name="snapshot")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question exceeds {MAX_QUESTION_CHARS} characters")
    if _contains_forbidden_directive(question):
        raise ValueError("question contains a control directive and is outside explanation scope")
    system = (
        "你是 RootScope 固定式根区灌溉舱的本地只读讲解器。"
        "你只能解释用户提供的结构化证据，不得补造传感器值、模型能力、比赛成绩或物理完成状态。"
        "你不是控制器：不得下达启动水泵、执行灌溉、写串口、清急停或修改状态机的命令。"
        "若证据不足，明确写入 uncertainty。只输出一个 JSON 对象，且键必须严格为："
        "status,summary,observations,uncertainty,suggested_checks,evidence_refs,authority。"
        "status 必须为 EXPLANATION_ONLY；六个 authority 值必须全部为 false。"
    )
    user_payload = {
        "question": question.strip(),
        "evidence": bounded,
        "required_authority": dict(AUTHORITY),
        "claim_boundary": "READ_ONLY_EXPLANATION_NOT_CONTROL_OR_PHYSICAL_EVIDENCE",
    }
    user = _canonical_json_bytes(user_payload).decode("utf-8")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt_bytes = _canonical_json_bytes(messages)
    provenance = {
        "prompt_version": PROMPT_VERSION,
        "snapshot_sha256": _sha256_bytes(snapshot_bytes),
        "prompt_sha256": _sha256_bytes(prompt_bytes),
        "snapshot_canonical_bytes": len(snapshot_bytes),
        "external_network_allowed": False,
        "tool_execution_allowed": False,
        "actuator_access_allowed": False,
    }
    return messages, provenance


def _extract_json_object(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if not isinstance(value, Mapping):
        raise ValueError("model response must be one JSON object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    cleaned = value.strip()
    if len(cleaned) > MAX_ITEM_CHARS:
        raise ValueError(f"{name} exceeds {MAX_ITEM_CHARS} characters")
    if _contains_forbidden_directive(cleaned):
        raise ValueError(f"{name} contains a forbidden control directive")
    return cleaned


def _text_list(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if len(value) > MAX_LIST_ITEMS or (not allow_empty and not value):
        raise ValueError(f"{name} must contain {'0' if allow_empty else '1'}..{MAX_LIST_ITEMS} items")
    return [_text(item, f"{name}[{index}]") for index, item in enumerate(value)]


def parse_explanation_response(text: str) -> dict[str, Any]:
    """Parse an untrusted LLM response against the no-authority contract."""

    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("model response is not text or exceeds the byte limit")
    value = _extract_json_object(text)
    expected = {
        "status",
        "summary",
        "observations",
        "uncertainty",
        "suggested_checks",
        "evidence_refs",
        "authority",
    }
    if set(value) != expected:
        raise ValueError(
            f"model response keys mismatch: missing={sorted(expected - set(value))} "
            f"unknown={sorted(set(value) - expected)}"
        )
    if value["status"] != ALLOWED_STATUS:
        raise ValueError("model response status must be EXPLANATION_ONLY")
    authority = value["authority"]
    if not isinstance(authority, Mapping) or set(authority) != set(AUTHORITY):
        raise ValueError("model response authority keys mismatch")
    if any(authority[name] is not False for name in AUTHORITY):
        raise ValueError("model response attempted to grant authority")
    return {
        "schema": SCHEMA_VERSION,
        "status": ALLOWED_STATUS,
        "summary": _text(value["summary"], "summary"),
        "observations": _text_list(value["observations"], "observations"),
        "uncertainty": _text_list(value["uncertainty"], "uncertainty"),
        "suggested_checks": _text_list(value["suggested_checks"], "suggested_checks"),
        "evidence_refs": _text_list(value["evidence_refs"], "evidence_refs", allow_empty=True),
        "authority": dict(AUTHORITY),
    }


def deterministic_explanation(
    snapshot: Mapping[str, Any],
    *,
    fallback_reason: str,
) -> dict[str, Any]:
    """Return a deterministic no-inference summary when the LLM cannot be used."""

    bounded, snapshot_bytes = _bounded_mapping(snapshot, name="snapshot")
    state = _safe_evidence_token(bounded.get("state", "UNKNOWN"))
    mode = _safe_evidence_token(bounded.get("mode", "UNKNOWN"))
    perception = bounded.get("perception") if isinstance(bounded.get("perception"), Mapping) else {}
    label = _safe_evidence_token(perception.get("class_id", "unknown"))
    confidence = perception.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
    ):
        confidence = "UNKNOWN"
    qualified = perception.get("qualified") is True
    observations = [
        f"系统模式={mode}，状态={state}。",
        f"感知类别={label}，置信度={confidence}，qualified={str(qualified).lower()}。",
    ]
    alerts = bounded.get("alerts")
    if isinstance(alerts, list) and alerts:
        safe_alerts = [_safe_evidence_token(item) for item in alerts[:6]]
        observations.append("当前告警=" + ", ".join(safe_alerts) + "。")
    uncertainty = ["该摘要只复述结构化字段，不构成植物泛化、BPU、真机或物理完成证据。"]
    if not qualified:
        uncertainty.append("感知 qualified=false，当前类别不得作为自动灌溉依据。")
    return {
        "schema": SCHEMA_VERSION,
        "status": ALLOWED_STATUS,
        "summary": "本地大模型未产生可接受输出，已使用确定性证据摘要。",
        "observations": observations,
        "uncertainty": uncertainty,
        "suggested_checks": ["核对仪表盘中的原始证据、时间戳与拒绝原因，由操作员决定下一步。"],
        "evidence_refs": [f"snapshot_sha256:{_sha256_bytes(snapshot_bytes)}"],
        "authority": dict(AUTHORITY),
        "provenance": {
            "backend": "deterministic_fallback",
            "fallback_reason": str(fallback_reason)[:240],
            "external_network_touched": False,
            "loopback_http_used": False,
            "model_output_accepted": False,
        },
    }


def _model_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("llama.cpp response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
        raise ValueError("llama.cpp response choice.message is missing")
    content = choice["message"].get("content")
    if not isinstance(content, str):
        raise ValueError("llama.cpp response message.content is not text")
    return content


def explain_snapshot(
    snapshot: Mapping[str, Any],
    config: ExplanationConfig,
    *,
    question: str = "请解释当前 RootScope 证据、风险和不确定性。",
) -> dict[str, Any]:
    """Query one loopback LLM, or fail closed to deterministic text."""

    messages, prompt_provenance = build_explanation_messages(snapshot, question=question)
    if not config.enabled:
        return deterministic_explanation(snapshot, fallback_reason="LLM_DISABLED")
    body = _canonical_json_bytes(
        {
            "model": config.model_label,
            "messages": messages,
            "temperature": float(config.temperature),
            "max_tokens": config.max_tokens,
            "stream": False,
        }
    )
    try:
        raw = _post_loopback_json(config, body)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("llama.cpp HTTP response exceeds the byte limit")
        envelope = json.loads(raw.decode("utf-8"))
        if not isinstance(envelope, Mapping):
            raise ValueError("llama.cpp HTTP response must be an object")
        parsed = parse_explanation_response(_model_content(envelope))
    except Exception as exc:  # fail closed at the optional explanation boundary
        return deterministic_explanation(
            snapshot,
            fallback_reason=f"{type(exc).__name__}:{str(exc)[:180]}",
        )
    parsed["provenance"] = {
        **dict(prompt_provenance),
        "backend": "loopback_llama_cpp",
        "endpoint_host": urlsplit(config.endpoint).hostname,
        "endpoint_port": urlsplit(config.endpoint).port,
        "model_label": config.model_label,
        "model_sha256_expected": config.model_sha256,
        "model_hash_verified_by_explainer": False,
        "transport_policy": "DIRECT_NUMERIC_LOOPBACK_NO_REDIRECT",
        "redirect_following_allowed": False,
        "external_network_touched": False,
        "loopback_http_used": True,
        "model_output_accepted": True,
    }
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-json", required=True)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9080")
    parser.add_argument("--model-label", default="qwen2-0.5b-q4km-rootscope-read-only")
    parser.add_argument("--model-sha256")
    parser.add_argument("--question", default="请解释当前 RootScope 证据、风险和不确定性。")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    snapshot = json.loads(Path(args.snapshot_json).read_text(encoding="utf-8"))
    result = explain_snapshot(
        snapshot,
        ExplanationConfig(
            enabled=args.enabled,
            endpoint=args.endpoint,
            model_label=args.model_label,
            model_sha256=args.model_sha256,
        ),
        question=args.question,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    return 0 if result["provenance"]["model_output_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
