"""SQLite FTS5/BM25 corpus and immutable Claim Ledger for RootScope-Ω.

Only provenance-bound source chunks can enter the search index.  Claims are
linked to stable citation identifiers and their status is derived from those
links; callers cannot promote an unsupported claim to ``SUPPORTED``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence


SOURCE_SCHEMA = "rootscope.omega.source.v1"
CHUNK_SCHEMA = "rootscope.omega.chunk.v1"
CLAIM_SCHEMA = "rootscope.omega.claim-ledger.v1"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PARAGRAPH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LATIN_OR_NUMBER_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_SOURCE_TYPES = frozenset({"LOCAL_FILE", "OFFICIAL_WEB", "PAPER", "DATASET", "MANUAL"})
_ROLES = frozenset({"EVIDENCE_EXPLAINER", "SAFETY_AUDITOR", "DEFENSE_QA"})
_RELATIONS = frozenset({"SUPPORTS", "CONTRADICTS"})
_CLAIM_STATUSES = frozenset({"SUPPORTED", "CONFLICTING"})


class KnowledgeContractError(ValueError):
    """Raised when an immutable knowledge or citation contract is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes, rejecting NaN and non-JSON values."""

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


def _bounded_text(value: Any, name: str, *, maximum: int, allow_newline: bool = True) -> str:
    if not isinstance(value, str):
        raise KnowledgeContractError(f"{name} must be text")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or _CONTROL_RE.search(cleaned):
        raise KnowledgeContractError(f"{name} must contain 1..{maximum} safe characters")
    if not allow_newline and ("\n" in cleaned or "\r" in cleaned):
        raise KnowledgeContractError(f"{name} must be one line")
    return cleaned


def _identifier(value: Any, name: str) -> str:
    cleaned = _bounded_text(value, name, maximum=128, allow_newline=False)
    if not _ID_RE.fullmatch(cleaned):
        raise KnowledgeContractError(f"{name} has an invalid identifier format")
    return cleaned


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise KnowledgeContractError(f"{name} must be a lowercase SHA-256")
    return value


def _citation_ids(
    value: Sequence[str],
    name: str,
    *,
    maximum: int,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise KnowledgeContractError(f"{name} must be a citation sequence")
    if len(value) > maximum or (require_nonempty and not value):
        lower = 1 if require_nonempty else 0
        raise KnowledgeContractError(f"{name} must contain {lower}..{maximum} citations")
    normalized = tuple(
        _bounded_text(item, f"{name}[{index}]", maximum=360, allow_newline=False)
        for index, item in enumerate(value)
    )
    if len(set(normalized)) != len(normalized):
        raise KnowledgeContractError(f"{name} must contain unique citations")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    locator: str
    source_type: str
    version: str
    license: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(self, "title", _bounded_text(self.title, "title", maximum=300))
        object.__setattr__(
            self, "locator", _bounded_text(self.locator, "locator", maximum=2048)
        )
        if self.source_type not in _SOURCE_TYPES:
            raise KnowledgeContractError(f"unsupported source_type: {self.source_type!r}")
        object.__setattr__(
            self, "version", _bounded_text(self.version, "version", maximum=160)
        )
        object.__setattr__(
            self, "license", _bounded_text(self.license, "license", maximum=160)
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, "source sha256"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SOURCE_SCHEMA, **asdict(self)}


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source_id: str
    paragraph_id: str
    text: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _identifier(self.chunk_id, "chunk_id"))
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        paragraph = _bounded_text(
            self.paragraph_id, "paragraph_id", maximum=96, allow_newline=False
        )
        if not _PARAGRAPH_RE.fullmatch(paragraph):
            raise KnowledgeContractError("paragraph_id has an invalid format")
        object.__setattr__(self, "paragraph_id", paragraph)
        text = _bounded_text(self.text, "chunk text", maximum=20_000)
        object.__setattr__(self, "text", text)
        digest = _sha256(self.sha256, "chunk sha256")
        if digest != sha256_text(text):
            raise KnowledgeContractError("chunk sha256 does not match UTF-8 text")
        object.__setattr__(self, "sha256", digest)

    @classmethod
    def from_text(
        cls, *, chunk_id: str, source_id: str, paragraph_id: str, text: str
    ) -> "KnowledgeChunk":
        cleaned = _bounded_text(text, "chunk text", maximum=20_000)
        return cls(
            chunk_id=chunk_id,
            source_id=source_id,
            paragraph_id=paragraph_id,
            text=cleaned,
            sha256=sha256_text(cleaned),
        )

    @property
    def citation_id(self) -> str:
        return f"{self.source_id}#{self.paragraph_id}@{self.chunk_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHUNK_SCHEMA,
            **asdict(self),
            "citation_id": self.citation_id,
        }


@dataclass(frozen=True)
class SearchHit:
    citation_id: str
    source_id: str
    chunk_id: str
    paragraph_id: str
    title: str
    locator: str
    source_type: str
    source_version: str
    source_license: str
    source_sha256: str
    chunk_sha256: str
    text: str
    bm25_score: float

    def citation(self, *, include_excerpt: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "citation_id": self.citation_id,
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "paragraph_id": self.paragraph_id,
            "title": self.title,
            "locator": self.locator,
            "source_type": self.source_type,
            "source_version": self.source_version,
            "source_license": self.source_license,
            "source_sha256": self.source_sha256,
            "chunk_sha256": self.chunk_sha256,
        }
        if include_excerpt:
            payload["excerpt"] = self.text
        return payload


@dataclass(frozen=True)
class ClaimLedgerRecord:
    claim_id: str
    run_id: str
    role: str
    statement: str
    statement_sha256: str
    status: str
    safety_critical: bool
    support_citation_ids: tuple[str, ...]
    contradiction_citation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CLAIM_SCHEMA,
            "claim_id": self.claim_id,
            "run_id": self.run_id,
            "role": self.role,
            "statement": self.statement,
            "statement_sha256": self.statement_sha256,
            "status": self.status,
            "safety_critical": self.safety_critical,
            "support_citation_ids": list(self.support_citation_ids),
            "contradiction_citation_ids": list(self.contradiction_citation_ids),
            "authority": {
                "execution_authority": False,
                "physical_authority": False,
            },
        }


def _fts_tokens(text: str) -> list[str]:
    """Project English words and CJK bigrams into deterministic FTS tokens."""

    tokens: list[str] = [item.lower() for item in _LATIN_OR_NUMBER_RE.findall(text)]
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def _fts_projection(text: str) -> str:
    return " ".join(_fts_tokens(text))


def _safe_match_query(query: str) -> str:
    cleaned = _bounded_text(query, "query", maximum=1_000)
    tokens = _fts_tokens(cleaned)[:32]
    if not tokens:
        raise KnowledgeContractError("query has no searchable tokens")
    # Every token originates from strict lexical extraction and is quoted.
    # Operators, parentheses, column selectors and SQL text never survive.
    return " OR ".join(f'"{token}"' for token in tokens)


class KnowledgeStore:
    """Immutable SQLite-backed knowledge corpus and Claim Ledger."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _initialize(self) -> None:
        try:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    locator TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    license TEXT NOT NULL,
                    sha256 TEXT NOT NULL
                ) STRICT;

                CREATE TABLE IF NOT EXISTS chunks (
                    row_id INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE,
                    source_id TEXT NOT NULL REFERENCES sources(source_id),
                    paragraph_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    citation_id TEXT NOT NULL UNIQUE
                ) STRICT;

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    title,
                    body,
                    source_id UNINDEXED,
                    chunk_id UNINDEXED,
                    tokenize='unicode61 remove_diacritics 2'
                );

                CREATE TABLE IF NOT EXISTS claims (
                    claim_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    statement_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    safety_critical INTEGER NOT NULL CHECK(safety_critical IN (0, 1))
                ) STRICT;

                CREATE TABLE IF NOT EXISTS claim_links (
                    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
                    citation_id TEXT NOT NULL REFERENCES chunks(citation_id),
                    relation TEXT NOT NULL,
                    PRIMARY KEY (claim_id, citation_id, relation)
                ) STRICT;
                """
            )
        except sqlite3.OperationalError as exc:
            raise RuntimeError("SQLite FTS5 and STRICT table support are required") from exc

    @staticmethod
    def _same_row(row: sqlite3.Row, expected: Mapping[str, Any]) -> bool:
        return all(row[key] == value for key, value in expected.items())

    def add_source(self, source: SourceRecord) -> None:
        expected = {
            "source_id": source.source_id,
            "title": source.title,
            "locator": source.locator,
            "source_type": source.source_type,
            "version": source.version,
            "license": source.license,
            "sha256": source.sha256,
        }
        existing = self._connection.execute(
            "SELECT * FROM sources WHERE source_id=?", (source.source_id,)
        ).fetchone()
        if existing is not None:
            if not self._same_row(existing, expected):
                raise KnowledgeContractError("source_id is immutable and already bound")
            return
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO sources
                    (source_id, title, locator, source_type, version, license, sha256)
                VALUES (:source_id, :title, :locator, :source_type, :version, :license, :sha256)
                """,
                expected,
            )

    def add_chunk(self, chunk: KnowledgeChunk) -> None:
        source = self._connection.execute(
            "SELECT title FROM sources WHERE source_id=?", (chunk.source_id,)
        ).fetchone()
        if source is None:
            raise KnowledgeContractError("chunk source_id is not registered")
        expected = {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "paragraph_id": chunk.paragraph_id,
            "text": chunk.text,
            "sha256": chunk.sha256,
            "citation_id": chunk.citation_id,
        }
        existing = self._connection.execute(
            "SELECT * FROM chunks WHERE chunk_id=?", (chunk.chunk_id,)
        ).fetchone()
        if existing is not None:
            if not self._same_row(existing, expected):
                raise KnowledgeContractError("chunk_id is immutable and already bound")
            return
        with self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO chunks
                    (chunk_id, source_id, paragraph_id, text, sha256, citation_id)
                VALUES (:chunk_id, :source_id, :paragraph_id, :text, :sha256, :citation_id)
                """,
                expected,
            )
            row_id = int(cursor.lastrowid)
            self._connection.execute(
                """
                INSERT INTO chunks_fts(rowid, title, body, source_id, chunk_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    _fts_projection(str(source["title"])),
                    _fts_projection(chunk.text),
                    chunk.source_id,
                    chunk.chunk_id,
                ),
            )

    def add_documents(
        self, source: SourceRecord, chunks: Iterable[KnowledgeChunk]
    ) -> None:
        self.add_source(source)
        for chunk in chunks:
            if chunk.source_id != source.source_id:
                raise KnowledgeContractError("chunk/source binding mismatch")
            self.add_chunk(chunk)

    def search(self, query: str, *, limit: int = 6) -> list[SearchHit]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise KnowledgeContractError("limit must be an integer within 1..20")
        match = _safe_match_query(query)
        rows = self._connection.execute(
            """
            SELECT
                c.citation_id, c.source_id, c.chunk_id, c.paragraph_id,
                s.title, s.locator, s.source_type, s.version, s.license,
                s.sha256 AS source_sha256, c.sha256 AS chunk_sha256, c.text,
                bm25(chunks_fts, 4.0, 1.0, 0.0, 0.0) AS bm25_score
            FROM chunks_fts
            JOIN chunks AS c ON c.row_id = chunks_fts.rowid
            JOIN sources AS s ON s.source_id = c.source_id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score ASC, c.chunk_id ASC
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
        return [
            SearchHit(
                citation_id=str(row["citation_id"]),
                source_id=str(row["source_id"]),
                chunk_id=str(row["chunk_id"]),
                paragraph_id=str(row["paragraph_id"]),
                title=str(row["title"]),
                locator=str(row["locator"]),
                source_type=str(row["source_type"]),
                source_version=str(row["version"]),
                source_license=str(row["license"]),
                source_sha256=str(row["source_sha256"]),
                chunk_sha256=str(row["chunk_sha256"]),
                text=str(row["text"]),
                bm25_score=float(row["bm25_score"]),
            )
            for row in rows
        ]

    def resolve_citation(self, citation_id: str) -> SearchHit:
        citation = _bounded_text(
            citation_id, "citation_id", maximum=360, allow_newline=False
        )
        row = self._connection.execute(
            """
            SELECT
                c.citation_id, c.source_id, c.chunk_id, c.paragraph_id,
                s.title, s.locator, s.source_type, s.version, s.license,
                s.sha256 AS source_sha256, c.sha256 AS chunk_sha256, c.text
            FROM chunks AS c
            JOIN sources AS s ON s.source_id = c.source_id
            WHERE c.citation_id=?
            """,
            (citation,),
        ).fetchone()
        if row is None:
            raise KnowledgeContractError("citation_id is not registered")
        return SearchHit(
            citation_id=str(row["citation_id"]),
            source_id=str(row["source_id"]),
            chunk_id=str(row["chunk_id"]),
            paragraph_id=str(row["paragraph_id"]),
            title=str(row["title"]),
            locator=str(row["locator"]),
            source_type=str(row["source_type"]),
            source_version=str(row["version"]),
            source_license=str(row["license"]),
            source_sha256=str(row["source_sha256"]),
            chunk_sha256=str(row["chunk_sha256"]),
            text=str(row["text"]),
            bm25_score=0.0,
        )

    def record_claim(
        self,
        *,
        run_id: str,
        role: str,
        statement: str,
        safety_critical: bool,
        support_citation_ids: Sequence[str],
        contradiction_citation_ids: Sequence[str] = (),
        citation_allowlist: Sequence[str] | None = None,
    ) -> ClaimLedgerRecord:
        run = _identifier(run_id, "run_id")
        if role not in _ROLES:
            raise KnowledgeContractError(f"unsupported role: {role!r}")
        text = _bounded_text(statement, "statement", maximum=800)
        if not isinstance(safety_critical, bool):
            raise KnowledgeContractError("safety_critical must be boolean")
        if (
            isinstance(support_citation_ids, Sequence)
            and not isinstance(support_citation_ids, (str, bytes))
            and not support_citation_ids
        ):
            raise KnowledgeContractError("every recorded claim requires support citations")
        supports = _citation_ids(
            support_citation_ids,
            "support_citation_ids",
            maximum=8,
            require_nonempty=True,
        )
        contradictions = _citation_ids(
            contradiction_citation_ids,
            "contradiction_citation_ids",
            maximum=8,
        )
        if set(supports) & set(contradictions):
            raise KnowledgeContractError("one citation cannot both support and contradict")
        allowed = (
            set(
                _citation_ids(
                    citation_allowlist,
                    "citation_allowlist",
                    maximum=64,
                )
            )
            if citation_allowlist is not None
            else None
        )
        for citation_id in (*supports, *contradictions):
            if allowed is not None and citation_id not in allowed:
                raise KnowledgeContractError("claim cites evidence outside the retrieval allowlist")
            self.resolve_citation(citation_id)
        status = "CONFLICTING" if contradictions else "SUPPORTED"
        if status not in _CLAIM_STATUSES:
            raise AssertionError("unreachable claim status")
        identity = {
            "run_id": run,
            "role": role,
            "statement": text,
            "safety_critical": safety_critical,
            "support_citation_ids": list(supports),
            "contradiction_citation_ids": list(contradictions),
        }
        claim_id = f"claim-{sha256_bytes(canonical_json_bytes(identity))[:32]}"
        statement_sha = sha256_text(text)
        expected = {
            "claim_id": claim_id,
            "run_id": run,
            "role": role,
            "statement": text,
            "statement_sha256": statement_sha,
            "status": status,
            "safety_critical": int(safety_critical),
        }
        existing = self._connection.execute(
            "SELECT * FROM claims WHERE claim_id=?", (claim_id,)
        ).fetchone()
        if existing is not None:
            if not self._same_row(existing, expected):
                raise KnowledgeContractError("deterministic claim identity collision")
        else:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO claims
                        (claim_id, run_id, role, statement, statement_sha256,
                         status, safety_critical)
                    VALUES
                        (:claim_id, :run_id, :role, :statement, :statement_sha256,
                         :status, :safety_critical)
                    """,
                    expected,
                )
                for relation, citations in (
                    ("SUPPORTS", supports),
                    ("CONTRADICTS", contradictions),
                ):
                    if relation not in _RELATIONS:
                        raise AssertionError("unreachable relation")
                    self._connection.executemany(
                        """
                        INSERT INTO claim_links(claim_id, citation_id, relation)
                        VALUES (?, ?, ?)
                        """,
                        [(claim_id, citation, relation) for citation in citations],
                    )
        return ClaimLedgerRecord(
            claim_id=claim_id,
            run_id=run,
            role=role,
            statement=text,
            statement_sha256=statement_sha,
            status=status,
            safety_critical=safety_critical,
            support_citation_ids=supports,
            contradiction_citation_ids=contradictions,
        )

    def claims_for_run(self, run_id: str) -> list[ClaimLedgerRecord]:
        run = _identifier(run_id, "run_id")
        rows = self._connection.execute(
            "SELECT * FROM claims WHERE run_id=? ORDER BY claim_id", (run,)
        ).fetchall()
        records: list[ClaimLedgerRecord] = []
        for row in rows:
            links = self._connection.execute(
                """
                SELECT citation_id, relation FROM claim_links
                WHERE claim_id=? ORDER BY relation, citation_id
                """,
                (row["claim_id"],),
            ).fetchall()
            supports = tuple(
                str(link["citation_id"])
                for link in links
                if link["relation"] == "SUPPORTS"
            )
            contradictions = tuple(
                str(link["citation_id"])
                for link in links
                if link["relation"] == "CONTRADICTS"
            )
            records.append(
                ClaimLedgerRecord(
                    claim_id=str(row["claim_id"]),
                    run_id=str(row["run_id"]),
                    role=str(row["role"]),
                    statement=str(row["statement"]),
                    statement_sha256=str(row["statement_sha256"]),
                    status=str(row["status"]),
                    safety_critical=bool(row["safety_critical"]),
                    support_citation_ids=supports,
                    contradiction_citation_ids=contradictions,
                )
            )
        return records

    def integrity_report(self) -> dict[str, Any]:
        sqlite_integrity = str(
            self._connection.execute("PRAGMA integrity_check").fetchone()[0]
        )
        foreign_key_rows = self._connection.execute("PRAGMA foreign_key_check").fetchall()
        chunks = self._connection.execute(
            "SELECT row_id, text, sha256 FROM chunks ORDER BY row_id"
        ).fetchall()
        fts_rows = {
            int(row[0]) for row in self._connection.execute("SELECT rowid FROM chunks_fts")
        }
        chunk_rows = {int(row["row_id"]) for row in chunks}
        hash_mismatches = [
            int(row["row_id"])
            for row in chunks
            if sha256_text(str(row["text"])) != row["sha256"]
        ]
        claims = self._connection.execute(
            "SELECT * FROM claims ORDER BY claim_id"
        ).fetchall()
        claim_links = self._connection.execute(
            "SELECT claim_id, citation_id, relation FROM claim_links "
            "ORDER BY claim_id, relation, citation_id"
        ).fetchall()
        links_by_claim: dict[str, list[sqlite3.Row]] = {}
        invalid_relations: list[dict[str, str]] = []
        for link in claim_links:
            claim_id = str(link["claim_id"])
            links_by_claim.setdefault(claim_id, []).append(link)
            if link["relation"] not in _RELATIONS:
                invalid_relations.append(
                    {
                        "claim_id": claim_id,
                        "citation_id": str(link["citation_id"]),
                        "relation": str(link["relation"]),
                    }
                )
        claim_statement_hash_mismatches: list[str] = []
        claim_status_mismatches: list[str] = []
        claim_identity_mismatches: list[str] = []
        claims_without_support: list[str] = []
        overlapping_claim_links: list[str] = []
        for row in claims:
            claim_id = str(row["claim_id"])
            links = links_by_claim.get(claim_id, [])
            supports = tuple(
                sorted(
                    str(link["citation_id"])
                    for link in links
                    if link["relation"] == "SUPPORTS"
                )
            )
            contradictions = tuple(
                sorted(
                    str(link["citation_id"])
                    for link in links
                    if link["relation"] == "CONTRADICTS"
                )
            )
            statement = str(row["statement"])
            if sha256_text(statement) != row["statement_sha256"]:
                claim_statement_hash_mismatches.append(claim_id)
            if not supports:
                claims_without_support.append(claim_id)
            if set(supports) & set(contradictions):
                overlapping_claim_links.append(claim_id)
            expected_status = "CONFLICTING" if contradictions else "SUPPORTED"
            if row["status"] != expected_status:
                claim_status_mismatches.append(claim_id)
            identity = {
                "run_id": str(row["run_id"]),
                "role": str(row["role"]),
                "statement": statement,
                "safety_critical": bool(row["safety_critical"]),
                "support_citation_ids": list(supports),
                "contradiction_citation_ids": list(contradictions),
            }
            expected_claim_id = (
                f"claim-{sha256_bytes(canonical_json_bytes(identity))[:32]}"
            )
            if claim_id != expected_claim_id:
                claim_identity_mismatches.append(claim_id)
        passed = (
            sqlite_integrity == "ok"
            and not foreign_key_rows
            and fts_rows == chunk_rows
            and not hash_mismatches
            and not invalid_relations
            and not claim_statement_hash_mismatches
            and not claim_status_mismatches
            and not claim_identity_mismatches
            and not claims_without_support
            and not overlapping_claim_links
        )
        return {
            "schema_version": "rootscope.omega.knowledge-integrity.v1",
            "passed": passed,
            "sqlite_integrity": sqlite_integrity,
            "foreign_key_errors": len(foreign_key_rows),
            "chunk_count": len(chunk_rows),
            "fts_row_count": len(fts_rows),
            "chunk_hash_mismatches": hash_mismatches,
            "claim_count": len(claims),
            "claim_link_count": len(claim_links),
            "invalid_claim_relations": invalid_relations,
            "claim_statement_hash_mismatches": claim_statement_hash_mismatches,
            "claim_status_mismatches": claim_status_mismatches,
            "claim_identity_mismatches": claim_identity_mismatches,
            "claims_without_support": claims_without_support,
            "overlapping_claim_links": overlapping_claim_links,
            "authority": {
                "execution_authority": False,
                "physical_authority": False,
            },
        }
