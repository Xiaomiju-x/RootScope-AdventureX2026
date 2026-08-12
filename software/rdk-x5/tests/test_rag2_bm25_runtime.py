from __future__ import annotations

from rag2.bm25_runtime import BM25Index


def test_bm25_runtime_has_no_dense_dependency_and_enforces_allowlist() -> None:
    with BM25Index() as index:
        hits = index.search("固定式根区灌溉舱需要导航吗", limit=5)
    assert hits
    assert all(hit.backend == "SQLITE_FTS5_BM25_V2" for hit in hits)
    assert all(hit.citation_id for hit in hits)
