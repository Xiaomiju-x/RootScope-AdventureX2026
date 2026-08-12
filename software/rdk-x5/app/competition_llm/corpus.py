"""Small deterministic JSONL/Markdown corpus loader and lexical retriever."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable, Sequence

from .contracts import (
    MAX_CORPUS_BYTES,
    MAX_CORPUS_CHUNKS,
    CompetitionLlmError,
    CorpusChunk,
    require_corpus_path,
    sha256_bytes,
)


_MD_SECTION_RE = re.compile(
    r"^##\s+([A-Za-z0-9][A-Za-z0-9_.:@#-]{0,95})\s+(.+?)\s*$\n+"
    r"(.+?)(?=^##\s+[A-Za-z0-9][A-Za-z0-9_.:@#-]{0,95}\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_LATIN_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def _tokens(text: str) -> tuple[str, ...]:
    values: list[str] = [item.lower() for item in _LATIN_RE.findall(text)]
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            values.append(run)
        else:
            values.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tuple(dict.fromkeys(item for item in values if item))


def _make_chunks(
    rows: Iterable[tuple[str, str, str, str]], *, source_sha256: str
) -> tuple[CorpusChunk, ...]:
    chunks: list[CorpusChunk] = []
    seen: set[str] = set()
    for citation_id, title, text, locator in rows:
        if citation_id in seen:
            raise CompetitionLlmError(f"duplicate citation_id: {citation_id}")
        seen.add(citation_id)
        cleaned = text.strip()
        chunks.append(
            CorpusChunk(
                citation_id=citation_id,
                title=title.strip(),
                text=cleaned,
                locator=locator,
                source_sha256=source_sha256,
                chunk_sha256=sha256_bytes(cleaned.encode("utf-8")),
            )
        )
        if len(chunks) > MAX_CORPUS_CHUNKS:
            raise CompetitionLlmError(
                f"corpus exceeds {MAX_CORPUS_CHUNKS} chunks"
            )
    if not chunks:
        raise CompetitionLlmError("corpus contains no chunks")
    return tuple(chunks)


def load_corpus(path: Path) -> tuple[CorpusChunk, ...]:
    path = require_corpus_path(path)
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_CORPUS_BYTES:
        raise CompetitionLlmError(
            f"corpus must contain 1..{MAX_CORPUS_BYTES} bytes"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CompetitionLlmError("corpus must be UTF-8") from exc
    source_sha256 = sha256_bytes(raw)
    locator = path.as_posix()
    if path.suffix.lower() == ".md":
        rows = (
            (citation_id, title, body, locator)
            for citation_id, title, body in _MD_SECTION_RE.findall(text)
        )
        return _make_chunks(rows, source_sha256=source_sha256)

    allowed_keys = frozenset({"id", "title", "text", "locator"})
    parsed_rows: list[tuple[str, str, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompetitionLlmError(
                f"invalid JSONL object at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise CompetitionLlmError(
                f"JSONL line {line_number} must be an object"
            )
        if not {"id", "text"} <= set(value) or not set(value) <= allowed_keys:
            raise CompetitionLlmError(
                "JSONL rows require id/text and only permit id/title/text/locator"
            )
        citation_id = value["id"]
        title = value.get("title", citation_id)
        row_locator = value.get("locator", f"{locator}#L{line_number}")
        if not all(
            isinstance(item, str)
            for item in (citation_id, title, value["text"], row_locator)
        ):
            raise CompetitionLlmError(
                f"JSONL line {line_number} fields must be text"
            )
        parsed_rows.append((citation_id, title, value["text"], row_locator))
    return _make_chunks(parsed_rows, source_sha256=source_sha256)


def retrieve(
    chunks: Sequence[CorpusChunk],
    query: str,
    *,
    limit: int = 3,
) -> tuple[CorpusChunk, ...]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 3:
        raise CompetitionLlmError("retrieval limit must be within 1..3")
    query_tokens = _tokens(query)
    if not query_tokens:
        return ()
    ranked: list[tuple[int, str, CorpusChunk]] = []
    for chunk in chunks:
        title = chunk.title.lower()
        body = chunk.text.lower()
        score = sum(
            (4 * title.count(token)) + body.count(token)
            for token in query_tokens
        )
        if score > 0:
            ranked.append((-score, chunk.citation_id, chunk))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ranked[:limit])
