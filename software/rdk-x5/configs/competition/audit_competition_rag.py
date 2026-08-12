#!/usr/bin/env python3
"""Dependency-free integrity and policy audit for the RootScope competition RAG."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable
from urllib.parse import urlparse


HERE = Path(__file__).resolve().parent
ROOTSCOPE_ROOT = HERE.parents[1]
REGISTRY_PATH = HERE / "rootscope_rag_sources.v1.json"
CORPUS_PATH = HERE / "rootscope_rag_corpus.v1.jsonl"
ALLOWLIST_PATH = HERE / "rootscope_rag_citation_allowlist.v1.json"
GOLD_PATH = HERE / "rootscope_rag_gold_qa.v1.jsonl"
FORBIDDEN_PATH = HERE / "rootscope_rag_forbidden_qa.v1.jsonl"

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*[\"'][^\"']{6,}"),
)
MATERIAL_POLLUTION = (
    "荧光粉",
    "钙钛矿",
    "晶胞参数",
    "Tanabe-Sugano",
    "CHGNet",
    "MatterSim",
    "MACE势",
    "XRD配方",
    "PL光谱",
)
ACTIONABLE_CONTROL_PATTERNS = (
    re.compile(r"(?i)\b(?:sudo|systemctl|ssh|curl)\s+[^\n]+"),
    re.compile(r"(?i)(?:serial|uart|gpio)\s*\.\s*(?:write|output)\s*\("),
    re.compile(r"(?i)\b(?:open|write)\s*\(\s*[\"']/dev/tty"),
    re.compile(r"(?:立即|现在|请)\s*(?:打开|启动|关闭|写入|发送).{0,12}(?:泵|阀|串口|GPIO)"),
)
SOURCE_BINDING_FIELDS = (
    "source_id",
    "publisher",
    "source_type",
    "title",
    "locator",
    "version",
    "license",
    "use_boundary",
    "public_safe",
    "source_sha256",
)
CORPUS_REQUIRED = {
    "schema",
    "id",
    "source",
    "title",
    "locator",
    "version",
    "license",
    "use_boundary",
    "paragraph",
    "text",
    "content_sha256",
    "citation_id",
    "public_safe",
}


class AuditFailure(RuntimeError):
    pass


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


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditFailure(f"{path.name}:{number}: invalid JSON: {exc}") from exc
        require(isinstance(value, dict), f"{path.name}:{number}: row must be an object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def source_fingerprint(source: dict[str, Any]) -> str:
    return sha256_bytes(
        canonical_bytes({field: source.get(field) for field in SOURCE_BINDING_FIELDS})
    )


def scan_no_secrets(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        require(not pattern.search(text), f"{path.name}: possible credential or private key")


def validate_safe_text(text: Any, context: str, *, corpus_policy: bool = False) -> str:
    require(isinstance(text, str) and text.strip(), f"{context}: expected non-empty text")
    require(not CONTROL_RE.search(text), f"{context}: contains a control character")
    for pattern in SECRET_PATTERNS:
        require(not pattern.search(text), f"{context}: possible credential")
    if corpus_policy:
        for term in MATERIAL_POLLUTION:
            require(term not in text, f"{context}: unrelated materials-domain term {term!r}")
        for pattern in ACTIONABLE_CONTROL_PATTERNS:
            require(
                not pattern.search(text),
                f"{context}: contains an actionable hardware/control instruction",
            )
    return text


def maybe_fix_hashes() -> None:
    registry = load_json(REGISTRY_PATH)
    for source in registry["sources"]:
        source["source_binding_sha256"] = source_fingerprint(source)
    write_json(REGISTRY_PATH, registry)

    rows = load_jsonl(CORPUS_PATH)
    for row in rows:
        row["content_sha256"] = sha256_text(row["text"].strip())
    write_jsonl(CORPUS_PATH, rows)


def audit_registry() -> dict[str, dict[str, Any]]:
    registry = load_json(REGISTRY_PATH)
    require(
        registry.get("schema") == "rootscope.competition.rag-source-registry.v1",
        "source registry schema mismatch",
    )
    allowed_domains = registry.get("allowed_web_domains")
    require(
        isinstance(allowed_domains, list)
        and allowed_domains
        and len(set(allowed_domains)) == len(allowed_domains),
        "web-domain allowlist must be a unique non-empty list",
    )
    require(
        set(allowed_domains) == {"www.fao.org", "developer.d-robotics.cc"},
        "web-domain allowlist must contain only the two reviewed official domains",
    )
    sources = registry.get("sources")
    require(isinstance(sources, list) and sources, "source registry must contain sources")
    by_id: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources):
        context = f"source[{index}]"
        require(isinstance(source, dict), f"{context}: must be an object")
        source_id = source.get("source_id")
        require(isinstance(source_id, str) and ID_RE.fullmatch(source_id), f"{context}: bad id")
        require(source_id not in by_id, f"duplicate source id: {source_id}")
        for field in SOURCE_BINDING_FIELDS:
            require(field in source, f"{source_id}: missing {field}")
        for field in ("publisher", "source_type", "title", "locator", "version", "license", "use_boundary"):
            validate_safe_text(source[field], f"{source_id}.{field}")
        require(source.get("public_safe") is True, f"{source_id}: source must be public-safe")
        locator = source["locator"]
        source_type = source["source_type"]
        if source_type == "OFFICIAL_WEB":
            parsed = urlparse(locator)
            require(parsed.scheme == "https", f"{source_id}: official source must use HTTPS")
            require(parsed.hostname in allowed_domains, f"{source_id}: source domain not allowed")
            require(source.get("source_sha256") is None, f"{source_id}: live web bytes are not vendored")
        else:
            require(
                source_type in {"LOCAL_EVIDENCE", "LOCAL_PLAN", "LOCAL_CODE"},
                f"{source_id}: unsupported local source type",
            )
            local_path = (ROOTSCOPE_ROOT / locator).resolve()
            require(
                local_path.is_relative_to(ROOTSCOPE_ROOT),
                f"{source_id}: local locator escapes RootScope",
            )
            require(local_path.is_file(), f"{source_id}: local source missing: {locator}")
            expected_sha = source.get("source_sha256")
            require(
                isinstance(expected_sha, str) and SHA_RE.fullmatch(expected_sha),
                f"{source_id}: invalid local source SHA",
            )
            require(
                sha256_bytes(local_path.read_bytes()) == expected_sha,
                f"{source_id}: local source SHA changed",
            )
        fingerprint = source.get("source_binding_sha256")
        require(
            isinstance(fingerprint, str) and SHA_RE.fullmatch(fingerprint),
            f"{source_id}: invalid binding SHA",
        )
        require(
            fingerprint == source_fingerprint(source),
            f"{source_id}: source binding SHA mismatch",
        )
        by_id[source_id] = source
    return by_id


def audit_corpus(
    sources: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    rows = load_jsonl(CORPUS_PATH)
    require(16 <= len(rows) <= 24, "corpus must contain 16..24 chunks")
    ids: set[str] = set()
    citations: set[str] = set()
    for index, row in enumerate(rows):
        context = f"corpus[{index}]"
        require(set(row) == CORPUS_REQUIRED, f"{context}: exact field contract mismatch")
        require(row["schema"] == "rootscope.competition.rag-chunk.v1", f"{context}: schema")
        chunk_id = row["id"]
        require(isinstance(chunk_id, str) and ID_RE.fullmatch(chunk_id), f"{context}: bad id")
        require(chunk_id not in ids, f"duplicate chunk id: {chunk_id}")
        ids.add(chunk_id)
        source_id = row["source"]
        require(source_id in sources, f"{chunk_id}: source not registered")
        source = sources[source_id]
        for field in ("title", "locator", "version", "license", "use_boundary", "public_safe"):
            require(row[field] == source[field], f"{chunk_id}: source metadata drift in {field}")
        paragraph = row["paragraph"]
        require(
            isinstance(paragraph, str) and ID_RE.fullmatch(paragraph),
            f"{chunk_id}: invalid paragraph id",
        )
        text = validate_safe_text(row["text"], f"{chunk_id}.text", corpus_policy=True).strip()
        digest = row["content_sha256"]
        require(isinstance(digest, str) and SHA_RE.fullmatch(digest), f"{chunk_id}: bad hash")
        require(digest == sha256_text(text), f"{chunk_id}: content SHA mismatch")
        citation = f"{source_id}#{paragraph}@{chunk_id}"
        require(row["citation_id"] == citation, f"{chunk_id}: citation id mismatch")
        require(citation not in citations, f"duplicate citation: {citation}")
        citations.add(citation)
    require(len({row["source"] for row in rows}) >= 10, "corpus source diversity too low")
    return rows, citations


def audit_allowlist(citations: set[str], sources: dict[str, dict[str, Any]]) -> None:
    value = load_json(ALLOWLIST_PATH)
    require(
        value.get("schema") == "rootscope.competition.rag-citation-allowlist.v1",
        "citation allowlist schema mismatch",
    )
    listed = value.get("citation_ids")
    require(
        isinstance(listed, list) and listed == sorted(citations),
        "citation allowlist must exactly equal sorted corpus citations",
    )
    source_ids = value.get("source_ids")
    require(
        isinstance(source_ids, list) and source_ids == sorted(sources),
        "source allowlist must exactly equal sorted registry sources",
    )


def audit_qa(path: Path, citations: set[str], *, forbidden: bool) -> None:
    rows = load_jsonl(path)
    require(len(rows) == 20, f"{path.name}: expected exactly 20 rows")
    ids: set[str] = set()
    expected_schema = (
        "rootscope.competition.rag-forbidden-qa.v1"
        if forbidden
        else "rootscope.competition.rag-gold-qa.v1"
    )
    answer_field = "expected_answer" if forbidden else "answer"
    required = (
        {"schema", "id", "question", answer_field, "citation_ids", "reason", "public_safe"}
        if forbidden
        else {"schema", "id", "question", answer_field, "citation_ids", "answer_boundary", "public_safe"}
    )
    for index, row in enumerate(rows):
        context = f"{path.name}[{index}]"
        require(set(row) == required, f"{context}: exact field contract mismatch")
        require(row["schema"] == expected_schema, f"{context}: schema mismatch")
        qa_id = row["id"]
        require(isinstance(qa_id, str) and ID_RE.fullmatch(qa_id), f"{context}: bad id")
        require(qa_id not in ids, f"{path.name}: duplicate id {qa_id}")
        ids.add(qa_id)
        validate_safe_text(row["question"], f"{qa_id}.question")
        validate_safe_text(row[answer_field], f"{qa_id}.{answer_field}")
        citation_ids = row["citation_ids"]
        require(
            isinstance(citation_ids, list)
            and citation_ids
            and len(citation_ids) == len(set(citation_ids)),
            f"{qa_id}: citations must be a unique non-empty list",
        )
        require(set(citation_ids) <= citations, f"{qa_id}: citation escaped allowlist")
        boundary_field = "reason" if forbidden else "answer_boundary"
        validate_safe_text(row[boundary_field], f"{qa_id}.{boundary_field}")
        require(row["public_safe"] is True, f"{qa_id}: QA row must be public-safe")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix-hashes",
        action="store_true",
        help="rewrite only deterministic source-binding and chunk-content hashes",
    )
    args = parser.parse_args()
    paths = (REGISTRY_PATH, CORPUS_PATH, ALLOWLIST_PATH, GOLD_PATH, FORBIDDEN_PATH)
    try:
        for path in paths:
            require(path.is_file(), f"missing required artifact: {path.name}")
        if args.fix_hashes:
            maybe_fix_hashes()
        for path in (*paths, Path(__file__).resolve()):
            scan_no_secrets(path)
        sources = audit_registry()
        corpus, citations = audit_corpus(sources)
        audit_allowlist(citations, sources)
        audit_qa(GOLD_PATH, citations, forbidden=False)
        audit_qa(FORBIDDEN_PATH, citations, forbidden=True)
    except (AuditFailure, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    report = {
        "status": "PASS",
        "source_count": len(sources),
        "chunk_count": len(corpus),
        "citation_count": len(citations),
        "gold_qa_count": 20,
        "forbidden_qa_count": 20,
        "official_domains": ["developer.d-robotics.cc", "www.fao.org"],
        "control_instruction_count": 0,
        "materials_pollution_count": 0,
        "secret_count": 0,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
