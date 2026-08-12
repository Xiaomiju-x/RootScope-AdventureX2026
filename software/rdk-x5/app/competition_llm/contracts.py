"""Strict contracts for the 4 GB RootScope competition LLM sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit


REPORT_SCHEMA = "rootscope.competition-llm.report.v1"
MODEL_SCHEMA = "rootscope.competition-llm.compact-three-role.v1"
MAX_MODEL_TOKENS = 64
MAX_PROMPT_BYTES = 4_096
MAX_MODEL_RESPONSE_BYTES = 4_096
MAX_CORPUS_BYTES = 2_000_000
MAX_CORPUS_CHUNKS = 256
MAX_CHUNK_CHARS = 4_000
MAX_QUERY_CHARS = 240
MAX_ROLE_TEXT_CHARS = 80

ROLE_KEYS: Mapping[str, str] = {
    "e": "EVIDENCE_EXPLAINER",
    "a": "SAFETY_AUDITOR",
    "q": "DEFENSE_QA",
}

AUTHORITY: Mapping[str, bool] = {
    "external_network_access": False,
    "tool_execution": False,
    "serial_write": False,
    "gpio_write": False,
    "state_machine_write": False,
    "actuator_access": False,
    "irrigation_execution": False,
    "physical_completion": False,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@#-]{0,95}$")
_CITATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@#-]{0,359}$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,159}$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class CompetitionLlmError(ValueError):
    """Raised when an input or model response violates the local contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def validate_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise CompetitionLlmError(f"{name} must be a safe identifier")
    return value


def validate_citation_id(value: Any, name: str = "citation_id") -> str:
    if not isinstance(value, str) or _CITATION_RE.fullmatch(value) is None:
        raise CompetitionLlmError(f"{name} must be a safe citation identifier")
    return value


def bounded_one_line(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise CompetitionLlmError(f"{name} must be text")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or "\r" in cleaned
        or "\n" in cleaned
        or _CONTROL_RE.search(cleaned)
    ):
        raise CompetitionLlmError(
            f"{name} must contain 1..{maximum} safe one-line characters"
        )
    return cleaned


@dataclass(frozen=True)
class CorpusChunk:
    citation_id: str
    title: str
    text: str
    locator: str
    source_sha256: str
    chunk_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "citation_id",
            validate_citation_id(self.citation_id),
        )
        object.__setattr__(
            self, "title", bounded_one_line(self.title, "title", 240)
        )
        if (
            not isinstance(self.text, str)
            or not self.text.strip()
            or len(self.text.strip()) > MAX_CHUNK_CHARS
            or _CONTROL_RE.search(self.text)
        ):
            raise CompetitionLlmError(
                f"chunk text must contain 1..{MAX_CHUNK_CHARS} safe characters"
            )
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(
            self, "locator", bounded_one_line(self.locator, "locator", 1_024)
        )
        for name in ("source_sha256", "chunk_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise CompetitionLlmError(f"{name} must be lowercase SHA-256")
        if sha256_bytes(self.text.encode("utf-8")) != self.chunk_sha256:
            raise CompetitionLlmError("chunk_sha256 does not match chunk text")

    def citation(self, *, excerpt_chars: int = 180) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "title": self.title,
            "locator": self.locator,
            "source_sha256": self.source_sha256,
            "chunk_sha256": self.chunk_sha256,
            "excerpt": self.text[:excerpt_chars],
        }


@dataclass(frozen=True)
class LoopbackConfig:
    endpoint: str
    model_id: str
    model_sha256: str
    timeout_seconds: float = 45.0
    max_response_bytes: int = MAX_MODEL_RESPONSE_BYTES
    api_mode: str = "chat"

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise CompetitionLlmError(
                "endpoint must be a credential-free http://127.0.0.1:<port> origin"
            )
        try:
            port = parsed.port
        except ValueError as exc:
            raise CompetitionLlmError("endpoint port is invalid") from exc
        if port is None or not 1_024 <= port <= 65_535:
            raise CompetitionLlmError("endpoint must use an unprivileged TCP port")
        if (
            not isinstance(self.model_id, str)
            or _MODEL_ID_RE.fullmatch(self.model_id) is None
        ):
            raise CompetitionLlmError("model_id has an invalid format")
        if (
            not isinstance(self.model_sha256, str)
            or _SHA256_RE.fullmatch(self.model_sha256) is None
        ):
            raise CompetitionLlmError("model_sha256 must be lowercase SHA-256")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or not 0.1 <= float(self.timeout_seconds) <= 180.0
        ):
            raise CompetitionLlmError("timeout_seconds must be within 0.1..180")
        if (
            isinstance(self.max_response_bytes, bool)
            or not isinstance(self.max_response_bytes, int)
            or not 512 <= self.max_response_bytes <= 65_536
        ):
            raise CompetitionLlmError("max_response_bytes must be within 512..65536")
        if self.api_mode not in {"chat", "completion"}:
            raise CompetitionLlmError("api_mode must be chat or completion")

    @property
    def port(self) -> int:
        parsed = urlsplit(self.endpoint)
        assert parsed.port is not None
        return parsed.port

    @property
    def request_path(self) -> str:
        return (
            "/v1/chat/completions"
            if self.api_mode == "chat"
            else "/completion"
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "model_id": self.model_id,
            "model_sha256_expected": self.model_sha256,
            "model_hash_verified_by_transport": False,
            "timeout_seconds": float(self.timeout_seconds),
            "api_mode": self.api_mode,
            "request_path": self.request_path,
        }


def require_corpus_path(path: Path) -> Path:
    resolved = Path(path)
    if resolved.suffix.lower() not in {".md", ".jsonl"}:
        raise CompetitionLlmError("corpus must be a .md or .jsonl file")
    return resolved
