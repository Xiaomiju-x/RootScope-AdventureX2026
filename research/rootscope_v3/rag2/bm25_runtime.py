"""Standard-library-only RootScope RAG 2.0 BM25 runtime for RDK X5."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_PACK = HERE / "pack"
MAX_QUERY_CHARS = 256
_ALNUM = re.compile(r"[a-z0-9_+\-.]+", re.IGNORECASE)


class BM25RuntimeError(ValueError):
    pass


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def lexical_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    cjk = [char for char in normalized if _is_cjk(char)]
    candidates = (
        *cjk,
        *(cjk[index] + cjk[index + 1] for index in range(len(cjk) - 1)),
        *_ALNUM.findall(normalized),
    )
    output: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        if token and token not in seen:
            output.append(token)
            seen.add(token)
    return output[:192]


def match_expression(query: str) -> str:
    tokens = lexical_tokens(query)
    if not tokens:
        raise BM25RuntimeError("query has no searchable tokens")
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens[:96])


@dataclass(frozen=True)
class BM25Hit:
    rank: int
    citation_id: str
    chunk_id: str
    text: str
    score: float
    backend: str = "SQLITE_FTS5_BM25_V2"


class BM25Index:
    """Read-only FTS5 index with citation allowlist enforcement."""

    def __init__(self, pack_dir: Path = DEFAULT_PACK) -> None:
        self.pack_dir = Path(pack_dir)
        self.allowlist = frozenset(
            json.loads(
                (self.pack_dir / "rootscope_rag_citation_allowlist.v2.json").read_text(
                    encoding="utf-8"
                )
            )["citation_ids"]
        )
        database_uri = (
            (self.pack_dir / "rag2_index.sqlite3").resolve(strict=True).as_uri()
            + "?mode=ro&immutable=1"
        )
        self.connection = sqlite3.connect(
            database_uri,
            uri=True,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "BM25Index":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def search(self, query: str, *, limit: int = 5) -> list[BM25Hit]:
        if not isinstance(query, str) or not query.strip():
            raise BM25RuntimeError("query must be non-empty text")
        value = " ".join(query.strip().split())
        if len(value) > MAX_QUERY_CHARS:
            raise BM25RuntimeError("query exceeds maximum length")
        if isinstance(limit, bool) or not 1 <= int(limit) <= 20:
            raise BM25RuntimeError("limit must be in 1..20")
        rows = self.connection.execute(
            """
            SELECT c.citation_id, c.chunk_id, c.text,
                   bm25(chunks_fts, 1.0) AS score
            FROM chunks_fts
            JOIN chunks c ON c.rowid=chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC, c.chunk_id ASC
            LIMIT ?
            """,
            (match_expression(value), int(limit)),
        ).fetchall()
        return [
            BM25Hit(
                rank=index,
                citation_id=str(row["citation_id"]),
                chunk_id=str(row["chunk_id"]),
                text=str(row["text"]),
                score=float(row["score"]),
            )
            for index, row in enumerate(rows, start=1)
            if row["citation_id"] in self.allowlist
        ]


__all__ = ["BM25Hit", "BM25Index", "BM25RuntimeError", "lexical_tokens"]
