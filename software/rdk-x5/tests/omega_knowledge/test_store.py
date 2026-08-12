from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.omega_knowledge.store import (
    KnowledgeChunk,
    KnowledgeContractError,
    KnowledgeStore,
    SourceRecord,
    sha256_text,
)


def _source(source_id: str = "official-rootscope") -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        title="RootScope 根区灌溉安全手册",
        locator="docs/rootscope_safety.md",
        source_type="MANUAL",
        version="event-v1",
        license="team-internal",
        sha256=sha256_text(f"immutable-source:{source_id}"),
    )


def _chunks(source_id: str = "official-rootscope") -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk.from_text(
            chunk_id=f"{source_id}-p1",
            source_id=source_id,
            paragraph_id="p1",
            text=(
                "RootScope uses cited evidence to explain root zone irrigation. "
                "Every completion statement requires mass and wetting evidence."
            ),
        ),
        KnowledgeChunk.from_text(
            chunk_id=f"{source_id}-p2",
            source_id=source_id,
            paragraph_id="p2",
            text="固定式根区灌溉舱必须区分现场、回放和仿真证据。",
        ),
    ]


class StoreContractTests(unittest.TestCase):
    def test_source_and_chunk_are_immutable_and_hash_bound(self) -> None:
        store = KnowledgeStore()
        source = _source()
        chunk = _chunks()[0]
        store.add_documents(source, [chunk])
        store.add_documents(source, [chunk])  # exact repeat is idempotent
        with self.assertRaisesRegex(KnowledgeContractError, "immutable"):
            store.add_source(
                SourceRecord(
                    source_id=source.source_id,
                    title="Changed title",
                    locator=source.locator,
                    source_type=source.source_type,
                    version=source.version,
                    license=source.license,
                    sha256=source.sha256,
                )
            )
        with self.assertRaisesRegex(KnowledgeContractError, "does not match"):
            KnowledgeChunk(
                chunk_id="bad-hash",
                source_id=source.source_id,
                paragraph_id="p9",
                text="content",
                sha256="0" * 64,
            )
        store.close()

    def test_missing_source_and_bad_identifier_reject(self) -> None:
        store = KnowledgeStore()
        with self.assertRaisesRegex(KnowledgeContractError, "not registered"):
            store.add_chunk(
                KnowledgeChunk.from_text(
                    chunk_id="orphan",
                    source_id="missing",
                    paragraph_id="p1",
                    text="orphan text",
                )
            )
        with self.assertRaises(KnowledgeContractError):
            _source("bad source id")
        store.close()


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KnowledgeStore()
        self.store.add_documents(_source(), _chunks())

    def tearDown(self) -> None:
        self.store.close()

    def test_bm25_retrieves_english_and_chinese(self) -> None:
        english = self.store.search("cited root zone evidence")
        chinese = self.store.search("根区灌溉 仿真证据")
        self.assertTrue(english)
        self.assertEqual(english[0].chunk_id, "official-rootscope-p1")
        self.assertTrue(chinese)
        self.assertEqual(chinese[0].chunk_id, "official-rootscope-p2")
        self.assertIn("#p2@", chinese[0].citation_id)

    def test_search_syntax_is_lexically_quoted(self) -> None:
        # FTS operators, column selectors, quotes and SQL text cannot escape the
        # lexical projection.  The ordinary word "root" may still match.
        hits = self.store.search('root" OR source_id:* -- DROP TABLE chunks')
        self.assertLessEqual(len(hits), 2)
        self.assertEqual(self.store.integrity_report()["chunk_count"], 2)

    def test_resolve_unknown_citation_rejects(self) -> None:
        with self.assertRaisesRegex(KnowledgeContractError, "not registered"):
            self.store.resolve_citation("made-up#p1@chunk")

    def test_limit_contract_is_strict(self) -> None:
        for limit in (True, 0, 21, 1.5):
            with self.subTest(limit=limit), self.assertRaises(KnowledgeContractError):
                self.store.search("root", limit=limit)  # type: ignore[arg-type]


class ClaimLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KnowledgeStore()
        self.store.add_documents(_source(), _chunks())
        self.hits = self.store.search("evidence irrigation")

    def tearDown(self) -> None:
        self.store.close()

    def test_supported_claim_is_deterministic_and_idempotent(self) -> None:
        citation = self.hits[0].citation_id
        first = self.store.record_claim(
            run_id="run-001",
            role="EVIDENCE_EXPLAINER",
            statement="The source binds completion to evidence.",
            safety_critical=False,
            support_citation_ids=[citation],
            citation_allowlist=[citation],
        )
        second = self.store.record_claim(
            run_id="run-001",
            role="EVIDENCE_EXPLAINER",
            statement="The source binds completion to evidence.",
            safety_critical=False,
            support_citation_ids=[citation],
            citation_allowlist=[citation],
        )
        self.assertEqual(first.claim_id, second.claim_id)
        self.assertEqual(first.status, "SUPPORTED")
        self.assertEqual(len(self.store.claims_for_run("run-001")), 1)

    def test_contradiction_derives_conflicting_status(self) -> None:
        all_hits = self.store.search("evidence 仿真")
        support = all_hits[0].citation_id
        contradiction = all_hits[-1].citation_id
        if contradiction == support:
            contradiction = self.store.search("固定式 仿真")[0].citation_id
        record = self.store.record_claim(
            run_id="run-002",
            role="SAFETY_AUDITOR",
            statement="The evidence set contains a qualification conflict.",
            safety_critical=True,
            support_citation_ids=[support],
            contradiction_citation_ids=[contradiction],
            citation_allowlist=[support, contradiction],
        )
        self.assertEqual(record.status, "CONFLICTING")
        self.assertTrue(record.safety_critical)

    def test_unsupported_and_out_of_allowlist_claims_reject(self) -> None:
        citation = self.hits[0].citation_id
        with self.assertRaisesRegex(KnowledgeContractError, "requires support"):
            self.store.record_claim(
                run_id="run-003",
                role="DEFENSE_QA",
                statement="Unsupported statement.",
                safety_critical=False,
                support_citation_ids=[],
            )
        with self.assertRaisesRegex(KnowledgeContractError, "outside"):
            self.store.record_claim(
                run_id="run-003",
                role="DEFENSE_QA",
                statement="Escaped statement.",
                safety_critical=False,
                support_citation_ids=[citation],
                citation_allowlist=[],
            )

    def test_citation_sequences_reject_strings_duplicates_and_non_text(self) -> None:
        citation = self.hits[0].citation_id
        invalid_sequences = (
            citation,
            [citation, citation],
            [citation, {"not": "a citation"}],
        )
        for citations in invalid_sequences:
            with self.subTest(citations=citations), self.assertRaises(
                KnowledgeContractError
            ):
                self.store.record_claim(
                    run_id="run-citation-contract",
                    role="DEFENSE_QA",
                    statement="Citation inputs stay structurally bounded.",
                    safety_critical=False,
                    support_citation_ids=citations,  # type: ignore[arg-type]
                )

    def test_integrity_report_covers_fts_and_claim_links(self) -> None:
        citation = self.hits[0].citation_id
        record = self.store.record_claim(
            run_id="run-004",
            role="DEFENSE_QA",
            statement="A traced answer.",
            safety_critical=False,
            support_citation_ids=[citation],
        )
        report = self.store.integrity_report()
        self.assertTrue(report["passed"])
        self.assertEqual(report["chunk_count"], report["fts_row_count"])
        self.assertEqual(report["claim_count"], 1)
        self.assertEqual(report["claim_link_count"], 1)
        self.assertFalse(report["authority"]["execution_authority"])
        # A direct database mutation is outside the public API, but the
        # release audit must still detect it rather than bless a corrupt
        # ledger as immutable.
        self.store._connection.execute(  # noqa: SLF001 - intentional tamper fixture
            "UPDATE claims SET statement=? WHERE claim_id=?",
            ("Tampered statement.", record.claim_id),
        )
        tampered = self.store.integrity_report()
        self.assertFalse(tampered["passed"])
        self.assertEqual(
            tampered["claim_statement_hash_mismatches"],
            [record.claim_id],
        )


class FileBackedStoreTests(unittest.TestCase):
    def test_sqlite_database_is_reopenable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "knowledge.sqlite3"
            with KnowledgeStore(path) as store:
                store.add_documents(_source(), _chunks())
            with KnowledgeStore(path) as reopened:
                # The source title is projected into the FTS row for every
                # chunk, so a title-only query correctly returns both chunks.
                hits = reopened.search("RootScope")
                self.assertEqual(
                    [hit.chunk_id for hit in hits],
                    ["official-rootscope-p1", "official-rootscope-p2"],
                )
                self.assertTrue(reopened.integrity_report()["passed"])


if __name__ == "__main__":
    unittest.main()
