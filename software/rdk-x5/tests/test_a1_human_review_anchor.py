from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from app.evidence import verify_live_ledger
from app.evidence.verifier import read_verified_records


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/a1_human_review_anchor.py"
SPEC = importlib.util.spec_from_file_location("a1_human_review_anchor", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def jsonl_bytes(*values: object) -> bytes:
    return b"".join(
        (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for value in values
    )


class AnchorFixture:
    def __init__(self, temporary: str) -> None:
        self.root = Path(temporary) / "adventurex"
        self.dataset_root = self.root / "datasets/staging"
        self.review_dir = self.dataset_root / "review/human_decisions"
        self.a1_dir = self.root / "rootscope/evidence/a1"
        self.queue = self.dataset_root / "review/candidate_review_queue.jsonl"
        self.policy = self.root / "tools/dataset/human_review_policy_v1.json"
        self.implementation = self.root / "tools/dataset/human_review_server.py"
        self.ui = self.root / "tools/dataset/human_review_app.html"
        self.production_inputs = {
            "review_queue_summary_sha256": self.dataset_root
            / "review/review_queue_summary.json",
            "integrity_audit_sha256": self.dataset_root / "integrity_audit.json",
            "dataset_manifest_schema_v2_sha256": self.root
            / "rootscope/training/dataset_manifest_schema_v2.json",
            "class_contract_sha256": self.root / "rootscope/configs/class_contract.json",
            "class_contract_lock_sha256": self.root
            / "rootscope/configs/class_contract.lock.json",
        }
        self.session_id = "703e9b45-d2e8-463f-a9c5-a4b3c22b3f5e"
        self.candidate = self.dataset_root / "images/grass.jpg"

        self.candidate.parent.mkdir(parents=True, exist_ok=True)
        self.candidate.write_bytes(b"fixture-image-payload")
        self.queue.parent.mkdir(parents=True, exist_ok=True)
        self.queue.write_bytes(
            jsonl_bytes(
                {
                    "asset": "commons-1",
                    "local_path": "images/grass.jpg",
                    "sha256": sha_bytes(self.candidate.read_bytes()),
                }
            )
        )
        for index, path in enumerate(self.production_inputs.values(), start=1):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(json_bytes({"fixture_root": index}))
        roots = {
            "candidate_review_queue_sha256": sha_bytes(self.queue.read_bytes()),
            **{
                key: sha_bytes(path.read_bytes())
                for key, path in self.production_inputs.items()
            },
        }
        self.policy.parent.mkdir(parents=True, exist_ok=True)
        self.policy.write_bytes(json_bytes({"production_input_roots": roots}))
        self.implementation.write_bytes(b"# frozen fixture implementation\n")
        self.ui.write_bytes(b"<!doctype html><title>fixture</title>\n")

        self.review_dir.mkdir(parents=True, exist_ok=True)
        (self.review_dir / ".human_review.lock").write_bytes(b"\0")
        (self.review_dir / "rights_evidence").mkdir()
        self.session_bytes = json_bytes(
            {"session_id": self.session_id, "mode": "PRODUCTION"}
        )
        (self.review_dir / "session.json").write_bytes(self.session_bytes)

        self.paths = MODULE.AnchorPaths(
            dataset_root=self.dataset_root,
            review_dir=self.review_dir,
            receipt=self.review_dir / "human_review_receipt.json",
            queue=self.queue,
            policy=self.policy,
            implementation=self.implementation,
            ui=self.ui,
            production_input_files=self.production_inputs,
            review_lock=self.review_dir / ".human_review.lock",
            ledger=self.a1_dir / "a1_gate_ledger.jsonl",
            ledger_lock=self.a1_dir / ".a1_gate_ledger.lock",
        )
        self.set_complete(False)

    def set_complete(self, complete: bool) -> dict:
        status = (
            "HUMAN_REVIEW_COMPLETE_NOT_DATA_LOCKED"
            if complete
            else "HUMAN_REVIEW_IN_PROGRESS_NOT_DATA_LOCKED"
        )
        if complete:
            rights_bytes = json_bytes(
                {
                    "schema_version": "fixture.rights.v1",
                    "session_id": self.session_id,
                    "source_permanent_url": "https://commons.wikimedia.org/?oldid=1",
                }
            )
            rights_sha = sha_bytes(rights_bytes)
            (self.review_dir / "rights_evidence" / f"{rights_sha}.json").write_bytes(
                rights_bytes
            )
            journal = jsonl_bytes({"event_sha256": "a" * 64})
            decisions = jsonl_bytes(
                {
                    "asset": "commons-1",
                    "decision": {"rights_evidence_sha256": rights_sha},
                }
            )
            families = jsonl_bytes(
                {"reviewed_source_group": "human:family-1", "issues": []}
            )
            counts = {
                "candidate": 1,
                "event": 1,
                "evented_asset": 1,
                "reviewed_asset": 1,
                "finalized_asset": 1,
                "needs_review_asset": 0,
                "reviewed_source_family": 1,
                "family_issue": 0,
                "referenced_rights_evidence": 1,
            }
            last_event = "a" * 64
        else:
            journal = b""
            decisions = b""
            families = b""
            counts = {
                "candidate": 1,
                "event": 0,
                "evented_asset": 0,
                "reviewed_asset": 0,
                "finalized_asset": 0,
                "needs_review_asset": 0,
                "reviewed_source_family": 0,
                "family_issue": 0,
                "referenced_rights_evidence": 0,
            }
            last_event = "0" * 64
        (self.review_dir / "decision_journal.jsonl").write_bytes(journal)
        (self.review_dir / "human_review_decisions.jsonl").write_bytes(decisions)
        (self.review_dir / "reviewed_source_families.jsonl").write_bytes(families)

        checkpoint_bytes = json_bytes(
            {
                "session_id": self.session_id,
                "session_sha256": sha_bytes(self.session_bytes),
                "event_count": counts["event"],
                "journal_sha256": sha_bytes(journal),
                "last_event_sha256": last_event,
            }
        )
        (self.review_dir / "journal_checkpoint.json").write_bytes(checkpoint_bytes)
        roots = json.loads(self.policy.read_text(encoding="utf-8"))[
            "production_input_roots"
        ]
        snapshot_bytes = json_bytes(
            {
                "schema_version": "fixture.snapshot.v1",
                "session_id": self.session_id,
                "status": status,
                "queue_sha256": sha_bytes(self.queue.read_bytes()),
                "policy_sha256": sha_bytes(self.policy.read_bytes()),
            }
        )
        (self.review_dir / "latest_decisions.json").write_bytes(snapshot_bytes)
        rights_references = []
        if complete:
            rights_references = [
                json.loads(decisions.decode("utf-8"))["decision"][
                    "rights_evidence_sha256"
                ]
            ]
        rights_list = "".join(f"{value}\n" for value in rights_references).encode(
            "ascii"
        )
        receipt = {
            "schema_version": "rootscope.wikimedia_human_review_receipt.v1",
            "mode": "PRODUCTION",
            "status": status,
            "session_id": self.session_id,
            "session_sha256": sha_bytes(self.session_bytes),
            "complete": complete,
            "production_binding_enforced": True,
            "verified_production_input_roots": roots,
            "queue_sha256": sha_bytes(self.queue.read_bytes()),
            "policy_sha256": sha_bytes(self.policy.read_bytes()),
            "implementation_sha256": sha_bytes(self.implementation.read_bytes()),
            "ui_sha256": sha_bytes(self.ui.read_bytes()),
            "journal_sha256": sha_bytes(journal),
            "journal_checkpoint_sha256": sha_bytes(checkpoint_bytes),
            "last_event_sha256": last_event,
            "human_review_decisions_sha256": sha_bytes(decisions),
            "reviewed_source_families_sha256": sha_bytes(families),
            "rights_evidence_reference_list_sha256": sha_bytes(rights_list),
            "snapshot_sha256": sha_bytes(snapshot_bytes),
            "rights_attestation_scope": "PERMALINK_BOUND_HUMAN_ATTESTATION_NO_PAGE_SNAPSHOT",
            "rights_source_page_payload_archived": False,
            "counts": counts,
            "review_status_counts": (
                {"CANONICAL_READY_FOR_FAMILY_AUDIT_NOT_DATA_LOCKED": 1}
                if complete
                else {"UNREVIEWED": 1}
            ),
            "visual_axis_counts": {"PASS": 1} if complete else {"UNREVIEWED": 1},
            "rights_axis_counts": {"PASS": 1} if complete else {"UNREVIEWED": 1},
            "visual_pass_target_class_counts": {"grass_clump": 1} if complete else {},
            "global_near_duplicate_issues": [],
            "validation_scope": {
                "runtime_inputs_revalidated": True,
                "journal_revalidated": True,
                "all_referenced_rights_attestations_revalidated": True,
                "all_candidate_payloads_revalidated": complete,
            },
            "authority": dict(MODULE.AUTHORITY),
            "explicit_non_claims": list(MODULE.EXPLICIT_NON_CLAIMS),
        }
        self.paths.receipt.write_bytes(json_bytes(receipt))
        return receipt

    def receipt(self) -> dict:
        return json.loads(self.paths.receipt.read_text(encoding="utf-8"))

    def write_receipt(self, receipt: dict) -> None:
        self.paths.receipt.write_bytes(json_bytes(receipt))


class A1HumanReviewAnchorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = AnchorFixture(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_zero_event_preflight_is_pure_read_and_not_ready(self) -> None:
        before_files = {
            path.relative_to(self.fixture.root).as_posix(): sha_bytes(path.read_bytes())
            for path in self.fixture.root.rglob("*")
            if path.is_file()
        }
        self.assertFalse(self.fixture.a1_dir.exists())

        result = MODULE.preflight(self.fixture.paths)

        after_files = {
            path.relative_to(self.fixture.root).as_posix(): sha_bytes(path.read_bytes())
            for path in self.fixture.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(result["status"], "WAITING_FOR_HUMAN_REVIEW")
        self.assertFalse(result["eligible_to_anchor"])
        self.assertFalse(result["authoritative_event_written"])
        self.assertEqual(result["human_review_event_count"], 0)
        self.assertEqual(before_files, after_files)
        self.assertFalse(self.fixture.a1_dir.exists())

    def test_preflight_rejects_forbidden_authority_without_creating_a1(self) -> None:
        receipt = self.fixture.receipt()
        receipt["authority"]["training_eligibility"] = True
        self.fixture.write_receipt(receipt)

        with self.assertRaisesRegex(MODULE.AnchorError, "forbidden authority"):
            MODULE.preflight(self.fixture.paths)
        self.assertFalse(self.fixture.a1_dir.exists())

    def test_anchor_does_not_implicitly_initialize_missing_ledger(self) -> None:
        self.fixture.set_complete(True)

        with self.assertRaisesRegex(MODULE.AnchorError, "lock file"):
            MODULE.anchor_complete(self.fixture.paths)
        self.assertFalse(self.fixture.a1_dir.exists())

    def test_incomplete_anchor_refuses_without_extending_initialized_ledger(self) -> None:
        MODULE.initialize_ledger(self.fixture.paths)
        before = verify_live_ledger(self.fixture.paths.ledger).require_valid()

        with self.assertRaises(MODULE.AnchorNotReady):
            MODULE.anchor_complete(self.fixture.paths)

        after = verify_live_ledger(self.fixture.paths.ledger).require_valid()
        self.assertEqual(before.record_count, 0)
        self.assertEqual(after.record_count, 0)
        self.assertEqual(before.terminal_hash, after.terminal_hash)

    def test_complete_receipt_appends_once_and_reuses_evidence_core(self) -> None:
        self.fixture.set_complete(True)
        MODULE.initialize_ledger(self.fixture.paths)
        expected_receipt_sha = sha_bytes(self.fixture.paths.receipt.read_bytes())

        result = MODULE.anchor_complete(self.fixture.paths)

        report = verify_live_ledger(self.fixture.paths.ledger).require_valid()
        records = read_verified_records(self.fixture.paths.ledger)
        self.assertEqual(report.record_count, 1)
        self.assertEqual(result["status"], "COMPLETE_RECEIPT_ANCHORED_NOT_DATA_LOCKED")
        self.assertEqual(result["record_hash"], report.terminal_hash)
        self.assertEqual(records[0]["event_type"], MODULE.ANCHOR_EVENT_TYPE)
        self.assertEqual(records[0]["payload"]["receipt_sha256"], expected_receipt_sha)
        self.assertEqual(records[0]["payload"]["authority"], MODULE.AUTHORITY)
        self.assertTrue(records[0]["payload"]["complete"])

    def test_exact_reanchor_is_idempotent(self) -> None:
        self.fixture.set_complete(True)
        MODULE.initialize_ledger(self.fixture.paths)
        first = MODULE.anchor_complete(self.fixture.paths)

        second = MODULE.anchor_complete(self.fixture.paths)

        self.assertEqual(second["status"], "ALREADY_ANCHORED_IDEMPOTENT")
        self.assertEqual(second["record_hash"], first["record_hash"])
        self.assertEqual(
            verify_live_ledger(self.fixture.paths.ledger).require_valid().record_count,
            1,
        )

    def test_changed_complete_receipt_requires_explicit_latest_supersedes(self) -> None:
        self.fixture.set_complete(True)
        MODULE.initialize_ledger(self.fixture.paths)
        first = MODULE.anchor_complete(self.fixture.paths)
        old_sha = first["receipt_sha256"]
        receipt = self.fixture.receipt()
        receipt["review_status_counts"]["ZERO_COUNT_DIAGNOSTIC"] = 0
        self.fixture.write_receipt(receipt)

        with self.assertRaisesRegex(MODULE.AnchorError, "requires the exact latest"):
            MODULE.anchor_complete(self.fixture.paths)
        second = MODULE.anchor_complete(
            self.fixture.paths, supersedes_receipt_sha256=old_sha
        )

        records = read_verified_records(self.fixture.paths.ledger)
        self.assertEqual(second["record_count"], 2)
        self.assertEqual(
            records[-1]["payload"]["supersedes_receipt_sha256"], old_sha
        )

    def test_verify_current_rejects_receipt_changed_after_anchor(self) -> None:
        self.fixture.set_complete(True)
        MODULE.initialize_ledger(self.fixture.paths)
        MODULE.anchor_complete(self.fixture.paths)
        receipt = self.fixture.receipt()
        receipt["review_status_counts"]["ZERO_COUNT_DIAGNOSTIC"] = 0
        self.fixture.write_receipt(receipt)

        with self.assertRaisesRegex(MODULE.AnchorError, "not the session's latest"):
            MODULE.verify_current(self.fixture.paths)

    def test_complete_preflight_rejects_tampered_candidate(self) -> None:
        self.fixture.set_complete(True)
        self.fixture.candidate.write_bytes(b"changed-candidate")

        with self.assertRaisesRegex(MODULE.AnchorError, "candidate row 1 SHA-256 mismatch"):
            MODULE.preflight(self.fixture.paths)
        self.assertFalse(self.fixture.a1_dir.exists())

    def test_complete_preflight_rejects_tampered_rights_evidence(self) -> None:
        self.fixture.set_complete(True)
        evidence_path = next((self.fixture.review_dir / "rights_evidence").glob("*.json"))
        evidence_path.write_bytes(evidence_path.read_bytes() + b"tampered")

        with self.assertRaisesRegex(MODULE.AnchorError, "rights evidence .* mismatch"):
            MODULE.preflight(self.fixture.paths)

    def test_receipt_duplicate_json_key_is_rejected(self) -> None:
        raw = self.fixture.paths.receipt.read_text(encoding="utf-8")
        raw = raw.replace("{\n", '{\n  "complete": false,\n', 1)
        self.fixture.paths.receipt.write_text(raw, encoding="utf-8", newline="\n")

        with self.assertRaisesRegex(MODULE.AnchorError, "duplicate JSON key"):
            MODULE.preflight(self.fixture.paths)

    def test_review_and_ledger_locks_are_enforced(self) -> None:
        with MODULE.ExclusiveFileLock(self.fixture.paths.review_lock, create=False):
            with self.assertRaisesRegex(MODULE.AnchorError, "locked by another process"):
                MODULE.preflight(self.fixture.paths)

        self.fixture.set_complete(True)
        MODULE.initialize_ledger(self.fixture.paths)
        with MODULE.ExclusiveFileLock(self.fixture.paths.ledger_lock, create=False):
            with self.assertRaisesRegex(MODULE.AnchorError, "locked by another process"):
                MODULE.anchor_complete(self.fixture.paths)

    def test_tampered_a1_chain_refuses_extension(self) -> None:
        self.fixture.set_complete(True)
        MODULE.initialize_ledger(self.fixture.paths)
        MODULE.anchor_complete(self.fixture.paths)
        raw = self.fixture.paths.ledger.read_text(encoding="utf-8")
        self.fixture.paths.ledger.write_text(
            raw.replace("NOT_SIGNATURE", "NOT-SIGNATURE", 1),
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaises(Exception):
            MODULE.anchor_complete(self.fixture.paths)
        self.assertFalse(verify_live_ledger(self.fixture.paths.ledger).valid)


if __name__ == "__main__":
    unittest.main()
