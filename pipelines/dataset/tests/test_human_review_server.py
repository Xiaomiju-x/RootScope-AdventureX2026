from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "human_review_server.py"
POLICY_PATH = MODULE_PATH.with_name("human_review_policy_v1.json")
SPEC = importlib.util.spec_from_file_location("rootscope_human_review_server", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class HumanReviewFixture:
    """Two-image UTF-8 queue with an independently bound policy copy."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.dataset_root = root / "staging"
        self.review_dir = self.dataset_root / "review"
        self.queue_path = self.review_dir / "candidate_review_queue.jsonl"
        self.policy_path = root / "human_review_policy_v1.json"
        self.output_dir = self.review_dir / "human_decisions"
        self.stores: list[MODULE.ReviewStore] = []
        self.review_dir.mkdir(parents=True)
        self.rows = [
            self._make_row(
                asset="commons-101",
                pageid=101,
                class_hint="grass_clump",
                relative_path="images/grass_clump/草丛一.png",
                color=(58, 126, 61),
            ),
            self._make_row(
                asset="commons-102",
                pageid=102,
                class_hint="low_shrub",
                relative_path="images/low_shrub/灌木二.png",
                color=(111, 94, 55),
            ),
        ]
        self.write_queue(rebind_policy=True)

    def _make_row(
        self,
        *,
        asset: str,
        pageid: int,
        class_hint: str,
        relative_path: str,
        color: tuple[int, int, int],
    ) -> dict:
        image_path = self.dataset_root.joinpath(*relative_path.split("/"))
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (64, 48), color)
        # A second color ensures the fixtures are not degenerate one-color payloads.
        image.putpixel((pageid % 64, pageid % 48), tuple(255 - value for value in color))
        image.save(image_path, format="PNG")
        payload = image_path.read_bytes()
        return {
            "schema_version": "rootscope.wikimedia_human_review_queue.v1",
            "asset": asset,
            "pageid": pageid,
            "title": f"File:沙漠植物候选 {pageid}.png",
            "local_path": relative_path,
            "source_url": f"https://commons.wikimedia.org/?curid={pageid}",
            "creator": f"创作者 {pageid}",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "class_hint": class_hint,
            "creator_group": f"commons-creator:{pageid}",
            "acquisition_mode": "fixture",
            "acquisition_query": f"fixture desert plant {pageid}",
            "species_hint": "仅采集提示，不是人工标签",
            "source_group": f"commons:{pageid}",
            "sha256": sha256_bytes(payload),
            "dhash64": f"{pageid:016x}",
            "download_width": 64,
            "download_height": 48,
            "download_mime": "image/png",
            "review_status": "UNREVIEWED",
            "visual_decision": "",
            "rights_decision": "",
            "target_class": "",
            "reviewed_source_group": "",
            "near_duplicate_family": "",
            "reviewer": "",
            "notes": "",
            "split": "UNASSIGNED_DO_NOT_TRAIN",
            "training_eligible": False,
            "print_eligible": False,
        }

    def write_queue(self, *, rebind_policy: bool) -> None:
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in self.rows
        ).encode("utf-8")
        self.queue_path.write_bytes(payload)
        if rebind_policy:
            policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
            policy["production_input_roots"][
                "candidate_review_queue_sha256"
            ] = sha256_bytes(payload)
            self.policy_path.write_text(
                json.dumps(policy, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )

    def store(self, output_dir: Path | None = None) -> MODULE.ReviewStore:
        store = MODULE.ReviewStore(
            self.queue_path,
            output_dir or self.output_dir,
            self.policy_path,
        )
        self.stores.append(store)
        return store

    def close_all(self) -> None:
        for store in reversed(self.stores):
            store.close()

    @staticmethod
    def empty_decision() -> dict:
        return {
            "visual_decision": "UNREVIEWED",
            "rights_decision": "UNREVIEWED",
            "target_class": "",
            "unknown_scenario": "",
            "reviewed_source_group": "",
            "family_role": "UNASSIGNED",
            "near_duplicate_family": "",
            "visual_reviewer": "",
            "rights_reviewer": "",
            "rights_source_page_checked": False,
            "rights_evidence_sha256": "",
            "source_page_revision_id": "",
            "visual_reason_codes": [],
            "rights_reason_codes": [],
            "notes": "",
        }

    def create_rights_evidence(
        self, store: MODULE.ReviewStore, index: int = 0
    ) -> dict:
        row = self.rows[index]
        return store.create_rights_evidence(
            {
                "asset": row["asset"],
                "candidate_sha256": row["sha256"],
                "reviewer": "reviewer.rights",
                "source_page_revision_id": str(800000 + row["pageid"]),
                "confirmed_source_page_checked": True,
                "confirmed_creator_license_attribution": True,
                "confirmed_non_copyright_rights_reviewed": True,
            }
        )

    def canonical_decision(self, store: MODULE.ReviewStore, index: int = 0) -> dict:
        row = self.rows[index]
        evidence = self.create_rights_evidence(store, index)
        return {
            **self.empty_decision(),
            "visual_decision": "PASS",
            "rights_decision": "PASS",
            "target_class": row["class_hint"],
            "reviewed_source_group": f"human:independent-family-{row['pageid']}",
            "family_role": "CANONICAL_REPRESENTATIVE",
            "visual_reviewer": "reviewer.visual",
            "rights_reviewer": "reviewer.rights",
            "rights_source_page_checked": True,
            "rights_evidence_sha256": evidence["rights_evidence_sha256"],
            "source_page_revision_id": str(800000 + row["pageid"]),
            "notes": "独立完成视觉与权利页复核。",
        }

    def request(self, index: int, decision: dict, expected_revision: int = 0) -> dict:
        row = self.rows[index]
        return {
            "asset": row["asset"],
            "candidate_sha256": row["sha256"],
            "expected_revision": expected_revision,
            "decision": decision,
        }


class HumanReviewServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = HumanReviewFixture(Path(self.temporary.name))
        self.addCleanup(self.fixture.close_all)

    def assert_no_authority(self, value: dict) -> None:
        self.assertEqual(
            set(value),
            {
                "dataset_manifest_write",
                "training_eligibility",
                "split_assignment",
                "print_eligibility",
                "data_locked",
            },
        )
        self.assertTrue(all(flag is False for flag in value.values()))

    def test_initialization_exports_empty_snapshot_without_authority(self) -> None:
        store = self.fixture.store()

        self.assertEqual(len(store.items), 2)
        self.assertEqual(store.event_count, 0)
        self.assertEqual(store.last_event_sha256, "0" * 64)
        self.assertTrue(store.session_path.is_file())
        self.assertEqual(store.journal_path.read_bytes(), b"")
        self.assertEqual(store.decisions_path.read_bytes(), b"")
        self.assertEqual(store.families_path.read_bytes(), b"")

        session = read_json(store.session_path)
        snapshot = read_json(store.snapshot_path)
        receipt = read_json(store.receipt_path)
        for document in (session, snapshot, receipt):
            self.assert_no_authority(document["authority"])
            self.assertIn("DATA_LOCKED", document["explicit_non_claims"])
            self.assertIn("TRAIN_READY", document["explicit_non_claims"])
        self.assertEqual(snapshot["records"], [])
        self.assertEqual(snapshot["candidate_count"], 2)
        self.assertEqual(snapshot["evented_asset_count"], 0)
        self.assertEqual(
            snapshot["status"], "FIXTURE_HUMAN_REVIEW_IN_PROGRESS_NOT_DATA_LOCKED"
        )
        self.assertFalse(receipt["complete"])
        self.assertEqual(receipt["queue_sha256"], sha256_bytes(self.fixture.queue_path.read_bytes()))
        self.assertEqual(receipt["human_review_decisions_sha256"], sha256_bytes(b""))
        self.assertEqual(receipt["reviewed_source_families_sha256"], sha256_bytes(b""))
        self.assertFalse(store.stats()["data_locked"])
        self.assertFalse(store.stats()["training_authority"])
        self.assertFalse(store.stats()["print_authority"])

    def test_valid_pass_pair_canonical_is_hash_chained_and_exported(self) -> None:
        store = self.fixture.store()
        decision = self.fixture.canonical_decision(store, 0)

        item = store.record_decision(self.fixture.request(0, decision))

        self.assertEqual(item["review_revision"], 1)
        self.assertEqual(
            item["review_status"],
            "CANONICAL_READY_FOR_FAMILY_AUDIT_NOT_DATA_LOCKED",
        )
        self.assertEqual(item["decision"]["visual_reviewer"], "reviewer.visual")
        self.assertEqual(item["decision"]["rights_reviewer"], "reviewer.rights")
        self.assertNotEqual(
            item["decision"]["visual_reviewer"], item["decision"]["rights_reviewer"]
        )
        self.assertTrue(item["decision"]["rights_source_page_checked"])
        evidence_sha = item["decision"]["rights_evidence_sha256"]
        self.assertRegex(evidence_sha, MODULE.HEX64)
        evidence_path = store.rights_evidence_dir / f"{evidence_sha}.json"
        self.assertTrue(evidence_path.is_file())
        self.assertEqual(sha256_bytes(evidence_path.read_bytes()), evidence_sha)
        evidence_document = read_json(evidence_path)
        self.assertTrue(evidence_document["confirmed_source_page_checked"])
        self.assertTrue(evidence_document["confirmed_creator_license_attribution"])
        self.assertTrue(evidence_document["confirmed_non_copyright_rights_reviewed"])
        self.assert_no_authority(evidence_document["authority"])
        self.assertRegex(item["decision"]["visual_reviewed_at_utc"], MODULE.UTC_TEXT)
        self.assertRegex(item["decision"]["rights_reviewed_at_utc"], MODULE.UTC_TEXT)

        events = read_jsonl(store.journal_path)
        exports = read_jsonl(store.decisions_path)
        families = read_jsonl(store.families_path)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(exports), 1)
        self.assertEqual(len(families), 1)
        event = events[0]
        event_body = {key: value for key, value in event.items() if key != "event_sha256"}
        self.assertEqual(event["previous_event_sha256"], "0" * 64)
        self.assertEqual(
            event["event_sha256"], sha256_bytes(MODULE._canonical_bytes(event_body))
        )
        self.assertEqual(exports[0]["event_sha256"], event["event_sha256"])
        self.assertEqual(families[0]["canonical_assets"], [self.fixture.rows[0]["asset"]])
        self.assertEqual(families[0]["issues"], [])
        for document in (event, exports[0], families[0], read_json(store.receipt_path)):
            self.assert_no_authority(document["authority"])

    def test_stale_revision_is_rejected_without_a_second_event(self) -> None:
        store = self.fixture.store()
        request = self.fixture.request(0, self.fixture.canonical_decision(store, 0))
        store.record_decision(request)

        with self.assertRaisesRegex(MODULE.ReviewError, "stale decision revision"):
            store.record_decision(request)

        self.assertEqual(store.event_count, 1)
        self.assertEqual(len(read_jsonl(store.journal_path)), 1)

    def test_visual_reject_requires_visual_reason(self) -> None:
        store = self.fixture.store()
        decision = self.fixture.empty_decision()
        decision.update(
            {
                "visual_decision": "REJECT",
                "visual_reviewer": "reviewer.visual",
            }
        )

        with self.assertRaisesRegex(MODULE.ReviewError, "visual REJECT requires"):
            store.record_decision(self.fixture.request(0, decision))

        self.assertEqual(store.event_count, 0)

    def test_rights_pass_requires_checked_page_and_evidence_sha(self) -> None:
        store = self.fixture.store()
        decision = self.fixture.empty_decision()
        decision.update(
            {
                "rights_decision": "PASS",
                "rights_reviewer": "reviewer.rights",
                "rights_source_page_checked": True,
                "rights_evidence_sha256": "",
            }
        )

        with self.assertRaisesRegex(MODULE.ReviewError, "rights PASS/REJECT requires"):
            store.record_decision(self.fixture.request(0, decision))

    def test_rights_pass_rejects_missing_content_addressed_evidence(self) -> None:
        store = self.fixture.store()
        decision = self.fixture.empty_decision()
        decision.update(
            {
                "rights_decision": "PASS",
                "rights_reviewer": "reviewer.rights",
                "rights_source_page_checked": True,
                "rights_evidence_sha256": "b" * 64,
                "source_page_revision_id": "800101",
            }
        )

        with self.assertRaisesRegex(MODULE.ReviewError, "evidence file is missing or changed"):
            store.record_decision(self.fixture.request(0, decision))

        self.assertEqual(store.event_count, 0)
        self.assertEqual(store.journal_path.read_bytes(), b"")

    def test_rights_pass_rejects_tampered_content_addressed_evidence(self) -> None:
        store = self.fixture.store()
        evidence = self.fixture.create_rights_evidence(store, 0)
        evidence_path = store.output_dir / evidence["evidence_path"]
        evidence_path.write_bytes(evidence_path.read_bytes() + b"tampered")
        decision = self.fixture.empty_decision()
        decision.update(
            {
                "rights_decision": "PASS",
                "rights_reviewer": "reviewer.rights",
                "rights_source_page_checked": True,
                "rights_evidence_sha256": evidence["rights_evidence_sha256"],
                "source_page_revision_id": "800101",
            }
        )

        with self.assertRaisesRegex(MODULE.ReviewError, "evidence file is missing or changed"):
            store.record_decision(self.fixture.request(0, decision))

        self.assertEqual(store.event_count, 0)
        self.assertEqual(store.journal_path.read_bytes(), b"")

    def test_unknown_scenario_is_forbidden_for_non_unknown_target_class(self) -> None:
        store = self.fixture.store()
        decision = self.fixture.empty_decision()
        decision.update(
            {
                "visual_decision": "PASS",
                "target_class": "grass_clump",
                "unknown_scenario": "bare_sand",
                "visual_reviewer": "reviewer.visual",
            }
        )

        with self.assertRaisesRegex(
            MODULE.ReviewError, "unknown_scenario is only allowed for target_class=unknown"
        ):
            store.record_decision(self.fixture.request(0, decision))

    def test_series_sibling_requires_near_duplicate_family(self) -> None:
        store = self.fixture.store()
        decision = self.fixture.canonical_decision(store, 0)
        decision["family_role"] = "SERIES_SIBLING_EXCLUDED"
        decision["near_duplicate_family"] = ""

        with self.assertRaisesRegex(MODULE.ReviewError, "series sibling requires"):
            store.record_decision(self.fixture.request(0, decision))

    def test_series_family_requires_matching_canonical_family_id(self) -> None:
        store = self.fixture.store()
        canonical = self.fixture.canonical_decision(store, 0)
        canonical["reviewed_source_group"] = "human:shared-series"
        store.record_decision(self.fixture.request(0, canonical))
        sibling = self.fixture.canonical_decision(store, 1)
        sibling["reviewed_source_group"] = "human:shared-series"
        sibling["family_role"] = "SERIES_SIBLING_EXCLUDED"
        sibling["near_duplicate_family"] = "family:shared-series"

        store.record_decision(self.fixture.request(1, sibling))

        receipt = read_json(store.receipt_path)
        families = read_jsonl(store.families_path)
        self.assertFalse(receipt["complete"])
        self.assertGreater(receipt["counts"]["family_issue"], 0)
        self.assertIn(
            "NEAR_DUPLICATE_FAMILY_REQUIRES_ONE_MATCHING_CANONICAL:family:shared-series",
            families[0]["issues"],
        )

    def test_image_tamper_is_rejected_before_event_write(self) -> None:
        store = self.fixture.store()
        image_path = self.fixture.dataset_root.joinpath(
            *self.fixture.rows[0]["local_path"].split("/")
        )
        image_path.write_bytes(image_path.read_bytes() + b"tampered")

        with self.assertRaisesRegex(MODULE.ReviewError, "candidate payload changed"):
            store.record_decision(
                self.fixture.request(0, self.fixture.canonical_decision(store, 0))
            )

        self.assertEqual(store.event_count, 0)
        self.assertEqual(store.journal_path.read_bytes(), b"")

    def test_restart_replays_multi_event_journal_and_preserves_chain(self) -> None:
        store = self.fixture.store()
        store.record_decision(
            self.fixture.request(0, self.fixture.canonical_decision(store, 0))
        )
        store.record_decision(
            self.fixture.request(1, self.fixture.canonical_decision(store, 1))
        )
        before_journal = store.journal_path.read_bytes()
        before_tail = store.last_event_sha256
        store.close()

        restarted = self.fixture.store()

        self.assertEqual(restarted.event_count, 2)
        self.assertEqual(restarted.last_event_sha256, before_tail)
        self.assertEqual(restarted.journal_path.read_bytes(), before_journal)
        self.assertEqual(restarted.public_item("commons-101")["review_revision"], 1)
        self.assertEqual(restarted.public_item("commons-102")["review_revision"], 1)
        receipt = read_json(restarted.receipt_path)
        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["status"], "FIXTURE_HUMAN_REVIEW_COMPLETE_NOT_DATA_LOCKED")
        self.assert_no_authority(receipt["authority"])

    def test_tampered_journal_event_fails_closed_on_restart(self) -> None:
        store = self.fixture.store()
        store.record_decision(
            self.fixture.request(0, self.fixture.canonical_decision(store, 0))
        )
        event = read_jsonl(store.journal_path)[0]
        store.close()
        event["decision"]["notes"] = "journal tampered after review"
        store.journal_path.write_text(
            json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaises(MODULE.ReviewError):
            self.fixture.store()

    def test_pending_event_already_in_journal_is_recovered_idempotently(self) -> None:
        store = self.fixture.store()
        store.record_decision(
            self.fixture.request(0, self.fixture.canonical_decision(store, 0))
        )
        before_journal = store.journal_path.read_bytes()
        event = read_jsonl(store.journal_path)[0]
        store.pending_path.write_text(
            json.dumps(event, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        store.close()

        restarted = self.fixture.store()

        self.assertFalse(restarted.pending_path.exists())
        self.assertEqual(restarted.event_count, 1)
        self.assertEqual(restarted.journal_path.read_bytes(), before_journal)
        self.assertEqual(restarted.last_event_sha256, event["event_sha256"])

    def test_partial_trailing_journal_is_preserved_and_replayed_from_wal(self) -> None:
        store = self.fixture.store()
        store.record_decision(
            self.fixture.request(0, self.fixture.canonical_decision(store, 0))
        )
        checkpoint_after_first_event = store.checkpoint_path.read_bytes()
        store.record_decision(
            self.fixture.request(1, self.fixture.canonical_decision(store, 1))
        )
        events = read_jsonl(store.journal_path)
        complete_journal = store.journal_path.read_bytes()
        lines = complete_journal.splitlines(keepends=True)
        self.assertEqual(len(lines), 2)
        store.pending_path.write_text(
            json.dumps(events[1], ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        store.checkpoint_path.write_bytes(checkpoint_after_first_event)
        store.journal_path.write_bytes(lines[0] + lines[1][: len(lines[1]) // 2])
        store.close()

        restarted = self.fixture.store()

        self.assertEqual(restarted.event_count, 2)
        self.assertEqual(restarted.journal_path.read_bytes(), complete_journal)
        self.assertFalse(restarted.pending_path.exists())
        self.assertEqual(
            len(list(restarted.recovery_dir.glob("partial_journal_*.bin"))), 1
        )
        self.assertEqual(
            len(list(restarted.recovery_dir.glob("journal_recovery_*.json"))), 1
        )

    def test_queue_change_is_rejected_by_policy_sha_binding(self) -> None:
        store = self.fixture.store()
        store.close()
        self.fixture.rows[0]["source_url"] += "&changed=1"
        self.fixture.write_queue(rebind_policy=False)

        with self.assertRaisesRegex(MODULE.ReviewError, "policy-bound queue SHA-256"):
            self.fixture.store()

    def test_policy_change_is_rejected_by_existing_session_binding(self) -> None:
        store = self.fixture.store()
        store.close()
        policy = read_json(self.fixture.policy_path)
        policy["notes_max_length"] += 1
        self.fixture.policy_path.write_text(
            json.dumps(policy, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaisesRegex(MODULE.ReviewError, "conflicts at policy_sha256"):
            self.fixture.store()

    def test_path_traversal_is_rejected_even_when_target_exists(self) -> None:
        outside = self.fixture.root / "outside.png"
        Image.new("RGB", (32, 32), (203, 179, 119)).save(outside, format="PNG")
        self.fixture.rows[0]["local_path"] = "../outside.png"
        self.fixture.rows[0]["sha256"] = sha256_bytes(outside.read_bytes())
        self.fixture.write_queue(rebind_policy=True)

        with self.assertRaisesRegex(MODULE.ReviewError, "queue image path is unsafe"):
            self.fixture.store()

        self.assertFalse(self.fixture.output_dir.exists())

    def test_loopback_host_helper_is_fail_closed(self) -> None:
        for host in ("127.0.0.1", "::1", "localhost"):
            with self.subTest(accepted=host):
                self.assertTrue(MODULE._is_loopback_host(host))
        for host in (
            "0.0.0.0",
            "192.0.2.42",
            "example.com",
            "localhost.example.com",
            "127.0.0.2",
            "LOCALHOST",
            "",
        ):
            with self.subTest(rejected=host):
                self.assertFalse(MODULE._is_loopback_host(host))

    def test_second_store_same_output_is_rejected_by_process_lock(self) -> None:
        first = self.fixture.store()

        with self.assertRaisesRegex(MODULE.ReviewError, "already locked"):
            self.fixture.store()

        first.close()
        restarted = self.fixture.store()
        self.assertEqual(restarted.event_count, 0)

    def test_deleted_historical_rights_attestation_blocks_completion(self) -> None:
        store = self.fixture.store()
        first_decision = self.fixture.canonical_decision(store, 0)
        store.record_decision(self.fixture.request(0, first_decision))
        second_decision = self.fixture.canonical_decision(store, 1)
        evidence_path = (
            store.rights_evidence_dir
            / f"{first_decision['rights_evidence_sha256']}.json"
        )
        evidence_path.unlink()

        with self.assertRaisesRegex(MODULE.ReviewError, "evidence file is missing or changed"):
            store.record_decision(self.fixture.request(1, second_decision))

        self.assertTrue(store.degraded)
        self.assertTrue(store.pending_path.is_file())
        self.assertFalse(read_json(store.receipt_path)["complete"])

    def test_tampered_historical_image_blocks_completion(self) -> None:
        store = self.fixture.store()
        store.record_decision(
            self.fixture.request(0, self.fixture.canonical_decision(store, 0))
        )
        second_decision = self.fixture.canonical_decision(store, 1)
        first_image = self.fixture.dataset_root.joinpath(
            *self.fixture.rows[0]["local_path"].split("/")
        )
        first_image.write_bytes(first_image.read_bytes() + b"tampered-after-review")

        with self.assertRaisesRegex(MODULE.ReviewError, "full completion validation"):
            store.record_decision(self.fixture.request(1, second_decision))

        self.assertTrue(store.degraded)
        self.assertTrue(store.pending_path.is_file())
        self.assertFalse(read_json(store.receipt_path)["complete"])

    def test_fixture_receipt_cannot_claim_verified_production_roots(self) -> None:
        store = self.fixture.store()
        receipt = read_json(store.receipt_path)

        self.assertFalse(store.production_mode)
        self.assertEqual(receipt["mode"], "FIXTURE")
        self.assertFalse(receipt["production_binding_enforced"])
        self.assertEqual(receipt["verified_production_input_roots"], {})
        self.assertTrue(receipt["status"].startswith("FIXTURE_"))

    def test_rights_attestation_requires_nonempty_permanent_revision(self) -> None:
        store = self.fixture.store()
        row = self.fixture.rows[0]

        with self.assertRaisesRegex(MODULE.ReviewError, "source-page revision is invalid"):
            store.create_rights_evidence(
                {
                    "asset": row["asset"],
                    "candidate_sha256": row["sha256"],
                    "reviewer": "reviewer.rights",
                    "source_page_revision_id": "",
                    "confirmed_source_page_checked": True,
                    "confirmed_creator_license_attribution": True,
                    "confirmed_non_copyright_rights_reviewed": True,
                }
            )

    def test_unreviewed_visual_axis_cannot_carry_target_label(self) -> None:
        store = self.fixture.store()
        decision = self.fixture.empty_decision()
        decision.update(
            {
                "rights_decision": "NEEDS_REVIEW",
                "rights_reviewer": "reviewer.rights",
                "target_class": "grass_clump",
            }
        )

        with self.assertRaisesRegex(MODULE.ReviewError, "require visual PASS"):
            store.record_decision(self.fixture.request(0, decision))

    def test_external_journal_mutation_is_rejected_before_append(self) -> None:
        store = self.fixture.store()
        decision = self.fixture.canonical_decision(store, 0)
        store.journal_path.write_bytes(b"\n")

        with self.assertRaisesRegex(MODULE.ReviewError, "changed outside"):
            store.record_decision(self.fixture.request(0, decision))

        self.assertEqual(store.event_count, 0)

    def test_clean_trailing_event_truncation_is_rejected_by_checkpoint(self) -> None:
        store = self.fixture.store()
        store.record_decision(
            self.fixture.request(0, self.fixture.canonical_decision(store, 0))
        )
        store.record_decision(
            self.fixture.request(1, self.fixture.canonical_decision(store, 1))
        )
        journal_lines = store.journal_path.read_bytes().splitlines(keepends=True)
        self.assertEqual(len(journal_lines), 2)
        self.assertFalse(store.pending_path.exists())
        store.close()

        store.journal_path.write_bytes(journal_lines[0])

        with self.assertRaises(MODULE.ReviewError):
            self.fixture.store()

    def test_full_event_without_terminal_newline_recovers_and_allows_future_append(
        self,
    ) -> None:
        store = self.fixture.store()
        client_decision = self.fixture.canonical_decision(store, 0)
        event_at_utc = MODULE._utc_now()
        decision, review_status = store._recordable_decision(
            client_decision, None, event_at_utc
        )
        row = self.fixture.rows[0]
        body = {
            "schema_version": store.policy["decision_schema_version"],
            "event_id": str(MODULE.uuid.uuid4()),
            "event_at_utc": event_at_utc,
            "session_id": store.session_id,
            "session_sha256": store.session_sha256,
            "queue_sha256": store.queue_sha256,
            "policy_sha256": store.policy_sha256,
            "implementation_sha256": store.tool_sha256,
            "ui_sha256": store.ui_sha256,
            "asset": row["asset"],
            "pageid": row["pageid"],
            "candidate_sha256": row["sha256"],
            "revision": 1,
            "previous_event_sha256": "0" * 64,
            "decision": decision,
            "review_status": review_status,
            "authority": MODULE.AUTHORITY,
        }
        event = {
            **body,
            "event_sha256": sha256_bytes(MODULE._canonical_bytes(body)),
        }
        store._validate_event(event, expected_previous="0" * 64)
        MODULE._atomic_write(store.pending_path, MODULE._json_bytes(event))
        store.journal_path.write_bytes(MODULE._canonical_bytes(event))
        store.close()

        restarted = self.fixture.store()
        expected_first_line = MODULE._canonical_bytes(event) + b"\n"
        self.assertEqual(restarted.journal_path.read_bytes(), expected_first_line)
        self.assertFalse(restarted.pending_path.exists())
        self.assertEqual(restarted.event_count, 1)

        restarted.record_decision(
            self.fixture.request(1, self.fixture.canonical_decision(restarted, 1))
        )
        self.assertEqual(restarted.event_count, 2)
        self.assertEqual(len(read_jsonl(restarted.journal_path)), 2)
        restarted.close()

        final_restart = self.fixture.store()
        self.assertEqual(final_restart.event_count, 2)
        self.assertEqual(len(read_jsonl(final_restart.journal_path)), 2)

    def test_cli_preflight_does_not_pollute_default_output_with_fixture_inputs(
        self,
    ) -> None:
        simulated_default_output = self.fixture.root / "official_human_decisions"
        with mock.patch.object(MODULE, "DEFAULT_OUTPUT", simulated_default_output):
            return_code = MODULE.main(
                [
                    "--queue",
                    str(self.fixture.queue_path),
                    "--policy",
                    str(self.fixture.policy_path),
                    "--output-dir",
                    str(simulated_default_output),
                    "--ui",
                    str(MODULE.DEFAULT_UI),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "18767",
                ]
            )

        self.assertEqual(return_code, 2)
        self.assertFalse(simulated_default_output.exists())

    def test_complete_readonly_state_rejects_deleted_rights_evidence(self) -> None:
        store = self.fixture.store()
        decisions = []
        for index in range(2):
            decision = self.fixture.canonical_decision(store, index)
            decisions.append(decision)
            store.record_decision(self.fixture.request(index, decision))
        self.assertIn("HUMAN_REVIEW_COMPLETE", store.current_status)

        evidence_path = (
            store.rights_evidence_dir
            / f"{decisions[0]['rights_evidence_sha256']}.json"
        )
        evidence_path.unlink()

        with self.assertRaises(MODULE.ReviewError):
            store.verify_readonly_state()

    def test_complete_readonly_state_rejects_changed_candidate_payload(self) -> None:
        store = self.fixture.store()
        for index in range(2):
            store.record_decision(
                self.fixture.request(
                    index, self.fixture.canonical_decision(store, index)
                )
            )
        self.assertIn("HUMAN_REVIEW_COMPLETE", store.current_status)

        image_path = self.fixture.dataset_root.joinpath(
            *self.fixture.rows[0]["local_path"].split("/")
        )
        image_path.write_bytes(image_path.read_bytes() + b"tampered-after-complete")

        with self.assertRaises(MODULE.ReviewError):
            store.verify_readonly_state()

    def test_policy_rejects_extra_unverified_production_input_root(self) -> None:
        policy = read_json(self.fixture.policy_path)
        policy["production_input_roots"]["unverified_extra_sha256"] = "a" * 64
        self.fixture.policy_path.write_text(
            json.dumps(policy, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaisesRegex(MODULE.ReviewError, "production_input_roots"):
            self.fixture.store()


if __name__ == "__main__":
    unittest.main()
