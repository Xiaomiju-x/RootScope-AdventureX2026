#!/usr/bin/env python3
"""Build the deterministic SQLite and dense artifacts for RAG 2.0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from hybrid_index import DEFAULT_MODEL, DEFAULT_PACK, DenseEncoder, lexical_projection


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build(pack_dir: Path, model_dir: Path) -> dict[str, object]:
    corpus = rows(pack_dir / "rootscope_rag_corpus.v2.jsonl")
    database = pack_dir / "rag2_index.sqlite3"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(str(database))
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            PRAGMA page_size=4096;
            CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                citation_id TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                content_sha256 TEXT NOT NULL
            ) STRICT;
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                body,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        for index, row in enumerate(corpus, start=1):
            connection.execute(
                """
                INSERT INTO chunks(rowid, chunk_id, citation_id, text, content_sha256)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    index,
                    row["id"],
                    row["citation_id"],
                    row["text"],
                    row["content_sha256"],
                ),
            )
            connection.execute(
                "INSERT INTO chunks_fts(rowid, body) VALUES (?, ?)",
                (index, lexical_projection(str(row["text"]))),
            )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()

    encoder = DenseEncoder(model_dir)
    embeddings = encoder.encode([str(row["text"]) for row in corpus], is_query=False)
    embedding_path = pack_dir / "corpus_embeddings.f16.npy"
    np.save(embedding_path, embeddings.astype(np.float16), allow_pickle=False)
    embedding_meta = {
        "schema": "rootscope.rag2.embedding-matrix.v1",
        "model_id": "BAAI/bge-small-zh-v1.5",
        "model_revision": "7999e1d3359715c523056ef9478215996d62a620",
        "model_file_sha256": sha256(
            model_dir / "bge-small-zh-v1.5.dynamic-uint8.onnx"
        ),
        "pooling": "attention-mask mean pooling plus L2 normalization",
        "document_instruction": None,
        "query_instruction": "为这个句子生成表示以用于检索相关文章：",
        "dtype": "float16",
        "shape": list(embeddings.shape),
        "row_order": "rootscope_rag_corpus.v2.jsonl",
        "matrix_sha256": sha256(embedding_path),
        "status": "PC_ONLY_X5_PENDING",
    }
    (pack_dir / "corpus_embeddings.v1.json").write_text(
        json.dumps(embedding_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt: dict[str, object] = {
        "schema": "rootscope.rag2.index-build-receipt.v1",
        "status": "PC_ONLY_X5_PENDING",
        "corpus_rows": len(corpus),
        "sqlite": {
            "path": database.name,
            "bytes": database.stat().st_size,
            "sha256": sha256(database),
        },
        "embeddings": embedding_meta,
        "authority": {
            "execution_authority": False,
            "physical_authority": False,
            "serial_write": False,
            "pump_command": False,
        },
    }
    (pack_dir / "index_build_receipt.v1.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.pack_dir.resolve(), args.model_dir.resolve()),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
