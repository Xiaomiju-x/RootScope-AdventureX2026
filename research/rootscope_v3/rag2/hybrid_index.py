"""Low-memory RootScope RAG 2.0 retrieval runtime.

The lexical side is an embedded SQLite FTS5/BM25 index.  The optional semantic
side is a single 24 MB ONNX encoder plus a 42 x 512 float16 matrix.  Reciprocal
rank fusion (RRF) combines ranked lists without a learned online service.

This module is read-only and exposes no serial, GPIO, tool or physical action.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any, Iterable, Literal, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_PACK = HERE / "pack"
DEFAULT_MODEL = (
    HERE.parents[0]
    / "models"
    / "rag"
    / "bge-small-zh-v1.5-onnx-uint8"
)
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
MAX_QUERY_CHARS = 256
MAX_TOKENS = 128
RRF_K = 60


class Rag2Error(ValueError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def _is_punctuation(char: str) -> bool:
    category = unicodedata.category(char)
    return category.startswith("P") or char in "$+<=>^`|~"


def _strip_accents(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def _basic_tokens(text: str) -> list[str]:
    cleaned: list[str] = []
    for char in text:
        code = ord(char)
        if code == 0 or code == 0xFFFD or unicodedata.category(char) in {"Cc", "Cf"}:
            continue
        if char.isspace():
            cleaned.append(" ")
        elif _is_cjk(char) or _is_punctuation(char):
            cleaned.extend((" ", char, " "))
        else:
            cleaned.append(char)
    # The pinned tokenizer.json declares BertNormalizer(lowercase=false,
    # strip_accents=null).  Keeping case is therefore part of the model
    # contract; unknown English project names intentionally map to [UNK].
    return [token for token in "".join(cleaned).strip().split() if token]


class WordPieceTokenizer:
    """Minimal deterministic BERT WordPiece tokenizer for the pinned encoder."""

    def __init__(self, vocab_path: Path) -> None:
        if vocab_path.suffix == ".json":
            value = json.loads(vocab_path.read_text(encoding="utf-8"))
            vocab = value.get("model", {}).get("vocab")
            if not isinstance(vocab, dict):
                raise Rag2Error("tokenizer JSON does not contain a WordPiece vocab")
            self.vocab = {str(token): int(index) for token, index in vocab.items()}
        else:
            # tokenizer.json is authoritative for this model.  The text fallback
            # ignores blank lines so accidental line-ending artifacts do not
            # shift every downstream token id.
            vocab = [
                token
                for token in vocab_path.read_text(encoding="utf-8").splitlines()
                if token
            ]
            self.vocab = {token: index for index, token in enumerate(vocab)}
        for token in ("[PAD]", "[UNK]", "[CLS]", "[SEP]"):
            if token not in self.vocab:
                raise Rag2Error(f"missing tokenizer token: {token}")
        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]

    def _wordpiece(self, token: str) -> list[int]:
        if len(token) > 100:
            return [self.unk_id]
        start = 0
        pieces: list[int] = []
        while start < len(token):
            end = len(token)
            found: int | None = None
            while start < end:
                piece = token[start:end]
                if start:
                    piece = "##" + piece
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk_id]
            pieces.append(found)
            start = end
        return pieces

    def encode_batch(
        self, texts: Sequence[str], *, max_length: int = MAX_TOKENS
    ) -> dict[str, np.ndarray]:
        if not texts:
            raise Rag2Error("encode_batch requires text")
        rows: list[list[int]] = []
        for text in texts:
            ids = [self.cls_id]
            for token in _basic_tokens(text):
                ids.extend(self._wordpiece(token))
                if len(ids) >= max_length - 1:
                    break
            ids.append(self.sep_id)
            rows.append(ids[:max_length])
        width = max(len(row) for row in rows)
        input_ids = np.full((len(rows), width), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = row
            attention_mask[index, : len(row)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.zeros_like(input_ids),
        }


class DenseEncoder:
    def __init__(self, model_dir: Path = DEFAULT_MODEL) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise Rag2Error("onnxruntime is required for dense retrieval") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_dir / "bge-small-zh-v1.5.dynamic-uint8.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = WordPieceTokenizer(model_dir / "tokenizer.json")

    def encode(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
        batch_size: int = 8,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 512), dtype=np.float32)
        values: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            selected = list(texts[start : start + batch_size])
            if is_query:
                selected = [QUERY_INSTRUCTION + text for text in selected]
            feed = self.tokenizer.encode_batch(selected)
            last_hidden = self.session.run(None, feed)[0].astype(np.float32)
            mask = feed["attention_mask"].astype(np.float32)[..., None]
            pooled = (last_hidden * mask).sum(axis=1)
            pooled /= np.maximum(mask.sum(axis=1), 1e-9)
            pooled /= np.maximum(
                np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12
            )
            values.append(pooled)
        return np.concatenate(values, axis=0)


_ALNUM = re.compile(r"[a-z0-9_+\-.]+", re.IGNORECASE)


def lexical_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    cjk = [char for char in normalized if _is_cjk(char)]
    unigrams = cjk
    bigrams = [cjk[index] + cjk[index + 1] for index in range(len(cjk) - 1)]
    words = _ALNUM.findall(normalized)
    output: list[str] = []
    seen: set[str] = set()
    for token in (*unigrams, *bigrams, *words):
        if token and token not in seen:
            output.append(token)
            seen.add(token)
    return output[:192]


def lexical_projection(text: str) -> str:
    return " ".join(lexical_tokens(text))


def match_expression(query: str) -> str:
    tokens = lexical_tokens(query)
    if not tokens:
        raise Rag2Error("query has no searchable tokens")
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens[:96])


@dataclass(frozen=True)
class SearchHit:
    rank: int
    citation_id: str
    chunk_id: str
    text: str
    score: float
    backend: str


class HybridIndex:
    def __init__(
        self,
        *,
        pack_dir: Path = DEFAULT_PACK,
        dense_model_dir: Path = DEFAULT_MODEL,
        enable_dense: bool = True,
    ) -> None:
        self.pack_dir = Path(pack_dir)
        self.rows = _read_jsonl(self.pack_dir / "rootscope_rag_corpus.v2.jsonl")
        self.by_citation = {row["citation_id"]: row for row in self.rows}
        self.allowlist = frozenset(
            json.loads(
                (self.pack_dir / "rootscope_rag_citation_allowlist.v2.json").read_text(
                    encoding="utf-8"
                )
            )["citation_ids"]
        )
        self.connection = sqlite3.connect(
            str(self.pack_dir / "rag2_index.sqlite3"), check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only=ON")
        self.encoder: DenseEncoder | None = None
        self.embeddings: np.ndarray | None = None
        if enable_dense:
            self.encoder = DenseEncoder(dense_model_dir)
            self.embeddings = np.asarray(
                np.load(self.pack_dir / "corpus_embeddings.f16.npy"), dtype=np.float32
            )
            if self.embeddings.shape != (len(self.rows), 512):
                raise Rag2Error("dense matrix shape does not match corpus")
            self.embeddings /= np.maximum(
                np.linalg.norm(self.embeddings, axis=1, keepdims=True), 1e-12
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "HybridIndex":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @staticmethod
    def _query(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise Rag2Error("query must be non-empty text")
        value = " ".join(text.strip().split())
        if len(value) > MAX_QUERY_CHARS:
            raise Rag2Error("query exceeds maximum length")
        return value

    def bm25(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        value = self._query(query)
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
            SearchHit(
                rank=index,
                citation_id=str(row["citation_id"]),
                chunk_id=str(row["chunk_id"]),
                text=str(row["text"]),
                score=float(row["score"]),
                backend="SQLITE_FTS5_BM25",
            )
            for index, row in enumerate(rows, start=1)
            if row["citation_id"] in self.allowlist
        ]

    def dense(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        value = self._query(query)
        if self.encoder is None or self.embeddings is None:
            raise Rag2Error("dense retrieval is disabled")
        vector = self.encoder.encode([value], is_query=True)[0]
        scores = self.embeddings @ vector
        indices = np.argsort(-scores, kind="stable")[:limit]
        return [
            SearchHit(
                rank=rank,
                citation_id=str(self.rows[index]["citation_id"]),
                chunk_id=str(self.rows[index]["id"]),
                text=str(self.rows[index]["text"]),
                score=float(scores[index]),
                backend="BGE_SMALL_ZH_V1_5_ONNX_UINT8",
            )
            for rank, index in enumerate(indices, start=1)
            if self.rows[index]["citation_id"] in self.allowlist
        ]

    def rrf(
        self,
        query: str,
        *,
        limit: int = 10,
        candidate_limit: int = 20,
        dense_weight: float = 1.0,
    ) -> list[SearchHit]:
        lexical = self.bm25(query, limit=candidate_limit)
        semantic = self.dense(query, limit=candidate_limit)
        scores: dict[str, float] = {}
        for hit in lexical:
            scores[hit.citation_id] = scores.get(hit.citation_id, 0.0) + 1.0 / (
                RRF_K + hit.rank
            )
        for hit in semantic:
            scores[hit.citation_id] = scores.get(hit.citation_id, 0.0) + float(
                dense_weight
            ) / (RRF_K + hit.rank)
        ordered = sorted(
            scores,
            key=lambda citation: (
                -scores[citation],
                str(self.by_citation[citation]["id"]),
            ),
        )[:limit]
        return [
            SearchHit(
                rank=rank,
                citation_id=citation,
                chunk_id=str(self.by_citation[citation]["id"]),
                text=str(self.by_citation[citation]["text"]),
                score=float(scores[citation]),
                backend="RRF_BM25_BGE",
            )
            for rank, citation in enumerate(ordered, start=1)
        ]

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        backend: Literal["bm25", "dense", "rrf"] = "bm25",
    ) -> list[SearchHit]:
        if isinstance(limit, bool) or not 1 <= int(limit) <= 20:
            raise Rag2Error("limit must be in 1..20")
        if backend == "bm25":
            return self.bm25(query, limit=int(limit))
        if backend == "dense":
            return self.dense(query, limit=int(limit))
        if backend == "rrf":
            return self.rrf(query, limit=int(limit))
        raise Rag2Error("unsupported backend")


__all__ = [
    "DenseEncoder",
    "HybridIndex",
    "Rag2Error",
    "SearchHit",
    "WordPieceTokenizer",
    "lexical_projection",
]
