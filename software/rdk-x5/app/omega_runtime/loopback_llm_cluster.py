"""Loopback-only llama.cpp adapter for the three RootScope-Ω knowledge roles.

One small local model may serve the three *logical* roles serially. This is not
presented as three resident models. The adapter exposes text generation only,
uses no tool interface, follows no redirects, and connects only to loopback.
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
import socket
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .contracts import canonical_sha256
from .knowledge_pipeline import run_knowledge_roles


_SHA256_CHARS = frozenset("0123456789abcdef")
_AUTHORITY = {
    "external_network": False,
    "tool_execution": False,
    "serial_write": False,
    "gpio_write": False,
    "state_machine_write": False,
    "actuator_access": False,
    "irrigation_execution": False,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _numeric_loopback(hostname: str, port: int) -> str:
    if hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("LLM endpoint hostname must be an explicit loopback host")
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        answers = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
        addresses = {
            ipaddress.ip_address(answer[4][0].split("%", 1)[0])
            for answer in answers
        }
        if not addresses or any(not address.is_loopback for address in addresses):
            raise ValueError("localhost did not resolve exclusively to loopback")
        literal = sorted(addresses, key=lambda item: (item.version, str(item)))[0]
    if not literal.is_loopback:
        raise ValueError("LLM endpoint did not resolve to loopback")
    return str(literal)


@dataclass(frozen=True)
class LoopbackLlamaConfig:
    endpoint: str
    model_id: str
    model_sha256: str
    timeout_seconds: float = 150.0
    max_response_bytes: int = 65_536

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("endpoint must be a credential-free HTTP loopback origin")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("endpoint port is invalid") from exc
        if port is None or not 1024 <= port <= 65_535:
            raise ValueError("endpoint must include an unprivileged port")
        _numeric_loopback(parsed.hostname, port)
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if (
            not isinstance(self.model_sha256, str)
            or len(self.model_sha256) != 64
            or any(character not in _SHA256_CHARS for character in self.model_sha256)
        ):
            raise ValueError("model_sha256 must be lowercase SHA-256")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or not 0.1 <= float(self.timeout_seconds) <= 180.0
        ):
            raise ValueError("timeout_seconds must be within 0.1..180")
        if (
            isinstance(self.max_response_bytes, bool)
            or not isinstance(self.max_response_bytes, int)
            or not 4_096 <= self.max_response_bytes <= 1_048_576
        ):
            raise ValueError("max_response_bytes must be within 4096..1048576")


class LoopbackLlamaModel:
    """Text-only OpenAI-compatible client with bounded transport receipts."""

    def __init__(self, config: LoopbackLlamaConfig) -> None:
        if not isinstance(config, LoopbackLlamaConfig):
            raise TypeError("config must be LoopbackLlamaConfig")
        self.config = config
        self.model_id = config.model_id
        self._attempts: list[dict[str, Any]] = []

    @property
    def transport_attempts(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(item) for item in self._attempts)

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_schema: Mapping[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or not 0.0 <= float(temperature) <= 0.3
        ):
            raise ValueError("temperature must be within 0..0.3")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 32 <= max_tokens <= 2_048
        ):
            raise ValueError("max_tokens must be within 32..2048")
        if not isinstance(response_schema, Mapping):
            raise TypeError("response_schema must be a mapping")
        bounded_messages: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if (
                not isinstance(message, Mapping)
                or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)
                or not message["content"]
            ):
                raise ValueError(f"messages[{index}] violates the text-only contract")
            bounded_messages.append(
                {"role": str(message["role"]), "content": message["content"]}
            )
        if not bounded_messages:
            raise ValueError("messages must not be empty")
        body = _canonical_bytes(
            {
                "model": self.config.model_id,
                "messages": bounded_messages,
                "temperature": float(temperature),
                "max_tokens": max_tokens,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rootscope_omega_readonly_response",
                        "strict": True,
                        "schema": dict(response_schema),
                    },
                },
            }
        )
        if len(body) > 131_072:
            raise ValueError("canonical LLM request exceeds 131072 bytes")
        parsed = urlsplit(self.config.endpoint)
        assert parsed.hostname is not None and parsed.port is not None
        address = _numeric_loopback(parsed.hostname, parsed.port)
        started = time.perf_counter()
        attempt: dict[str, Any] = {
            "schema_version": "rootscope.omega.loopback-llm-attempt.v1",
            "request_sha256": _sha256(body),
            "endpoint_host": parsed.hostname,
            "endpoint_port": parsed.port,
            "resolved_loopback": address,
            "model_id": self.config.model_id,
            "model_sha256_expected": self.config.model_sha256,
            "model_hash_verified_by_transport": False,
            "external_network_touched": False,
            "redirect_following_allowed": False,
            "tool_interface_supplied": False,
            "authority": dict(_AUTHORITY),
        }
        connection = http.client.HTTPConnection(
            address,
            parsed.port,
            timeout=float(self.config.timeout_seconds),
        )
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            response = connection.getresponse()
            raw = response.read(self.config.max_response_bytes + 1)
            if response.status != 200:
                raise ValueError(f"llama.cpp HTTP status must be 200, got {response.status}")
            if len(raw) > self.config.max_response_bytes:
                raise ValueError("llama.cpp response exceeds the byte limit")
            envelope = json.loads(raw.decode("utf-8"))
            if not isinstance(envelope, Mapping):
                raise ValueError("llama.cpp response must be an object")
            choices = envelope.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("llama.cpp response must contain exactly one choice")
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise ValueError("llama.cpp choice must be an object")
            message = choice.get("message")
            if not isinstance(message, Mapping) or not isinstance(
                message.get("content"), str
            ):
                raise ValueError("llama.cpp choice.message.content must be text")
            content = message["content"]
            if len(content.encode("utf-8")) > self.config.max_response_bytes:
                raise ValueError("llama.cpp content exceeds the byte limit")
            attempt.update(
                {
                    "transport_status": "HTTP_200_TEXT_RETURNED",
                    "response_sha256": _sha256(raw),
                    "response_bytes": len(raw),
                }
            )
            return content
        except Exception as exc:
            attempt.update(
                {
                    "transport_status": "FAILED_CLOSED",
                    "failure_type": type(exc).__name__,
                }
            )
            raise
        finally:
            connection.close()
            attempt["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1_000.0, 3
            )
            attempt["attempt_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in attempt.items()
                    if key not in {"elapsed_ms", "attempt_sha256"}
                }
            )
            self._attempts.append(attempt)


def run_loopback_role_cluster(
    *,
    case_id: str,
    evidence_refs: tuple[str, ...],
    corpus_path: Path,
    config: LoopbackLlamaConfig,
) -> Mapping[str, Any]:
    """Run three logical roles serially and return a zero-authority capsule."""

    model = LoopbackLlamaModel(config)
    context = run_knowledge_roles(
        case_id=case_id,
        evidence_refs=evidence_refs,
        corpus_path=corpus_path,
        model=model,
        compact_edge_prompt=True,
    )
    responses = [dict(item) for item in context.responses]
    accepted = sum(
        response["provenance"]["model_output_accepted"] is True
        for response in responses
    )
    report: dict[str, Any] = {
        "schema_version": "rootscope.omega.loopback-role-cluster-report.v1",
        "cluster_topology": {
            "logical_roles": [
                "EVIDENCE_EXPLAINER",
                "SAFETY_AUDITOR",
                "DEFENSE_QA",
            ],
            "resident_model_count": 1,
            "scheduling": "SERIAL_SHARED_ENDPOINT",
            "prompt_profile": {
                "name": "X5_COMPACT_CITED_V1",
                "max_hits_per_role": 1,
                "passage_char_limit": 320,
                "schema_repeated_in_user_payload": False,
                "max_model_tokens": 192,
            },
        },
        "case_id": case_id,
        "model": {
            "model_id": config.model_id,
            "model_sha256_expected": config.model_sha256,
            "endpoint": config.endpoint,
        },
        "role_statuses": [
            {"role": response["role"], "status": response["status"]}
            for response in responses
        ],
        "accepted_model_role_count": accepted,
        "deterministic_fallback_role_count": len(responses) - accepted,
        "claim_ledger_root": context.claim_ledger_root,
        "transport_attempts": [dict(item) for item in model.transport_attempts],
        "authority": dict(_AUTHORITY),
        "runtime_boundary": {
            "loopback_http_touched": bool(model.transport_attempts),
            "external_network_touched": False,
            "camera_opened": False,
            "serial_opened": False,
            "gpio_touched": False,
            "pump_touched": False,
            "physical_completion_claim": False,
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9080")
    parser.add_argument(
        "--model-id",
        default="qwen2-0.5b-q4km-rootscope-read-only",
    )
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=150.0)
    parser.add_argument("--case-id", default="CASE01_NORMAL_VERIFIED")
    parser.add_argument(
        "--evidence-ref",
        action="append",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=root / "configs/omega/field_knowledge.v1.md",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_loopback_role_cluster(
        case_id=args.case_id,
        evidence_refs=tuple(
            args.evidence_ref or ["manual-loopback-qualification"]
        ),
        corpus_path=args.corpus,
        config=LoopbackLlamaConfig(
            endpoint=args.endpoint,
            model_id=args.model_id,
            model_sha256=args.model_sha256,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    print(text, end="")
    return 0 if report["accepted_model_role_count"] == 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
