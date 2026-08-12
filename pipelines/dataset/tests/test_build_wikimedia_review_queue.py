from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_wikimedia_review_queue.py"
SPEC = importlib.util.spec_from_file_location("build_wikimedia_review_queue", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
review_queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_queue)


HOLDOUT_PAGEIDS = (133271396, 75559442, 2738023, 4424728, 5445424, 6021614)
POLICY_PATH = MODULE_PATH.parent / "wikimedia_license_policy_v1.json"
POLICY_SHA = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )


class ReviewQueueFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.staging_root = root / "staging"
        self.holdout_root = root / "holdouts"
        self.staging_manifest = self.staging_root / "manifest.jsonl"
        self.holdout_manifest = self.holdout_root / "manifest.jsonl"
        self.integrity_audit = self.staging_root / "integrity_audit.json"
        self.output_dir = self.staging_root / "review"
        self.candidates = [self.make_candidate(900001, "grass_clump", b"candidate-one")]
        self.holdouts = [self.make_holdout(pageid, index) for index, pageid in enumerate(HOLDOUT_PAGEIDS)]
        self.flush()

    def _asset_file(self, base: Path, relative: str, payload: bytes) -> dict:
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))
        tiny = Image.new("RGB", (9, 8))
        tiny.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(72)])
        image = tiny.resize((512, 512), Image.Resampling.NEAREST)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = buffer.getvalue()
        path.write_bytes(encoded)
        width, height, mime, dhash = review_queue._image_facts(path, "fixture")
        return {"sha256": sha256(encoded), "bytes": len(encoded), "width": width, "height": height, "mime": mime, "dhash": dhash}

    def make_candidate(self, pageid: int, class_id: str, payload: bytes) -> dict:
        filename = f"images/{class_id}/{pageid}.png"
        facts = self._asset_file(self.staging_root, filename, payload)
        artist = f"Creator {pageid}"
        return {
            "schema_version": "rootscope.wikimedia_candidate.v1",
            "class_id": class_id,
            "candidate_label_status": "query_or_category_derived_unverified",
            "domain": "natural_web_candidate",
            "split": "UNASSIGNED_DO_NOT_TRAIN",
            "review_status": "pending_human_visual_and_license_review",
            "training_eligible": False,
            "print_eligible": False,
            "source_provider": "Wikimedia Commons",
            "source_group": f"commons:{pageid}",
            "pageid": pageid,
            "title": f"File:Candidate {pageid}.jpg",
            "source_page": f"https://commons.wikimedia.org/wiki/File:Candidate_{pageid}.jpg",
            "original_url": f"https://upload.wikimedia.org/original/{pageid}.jpg",
            "download_url": f"https://upload.wikimedia.org/thumb/{pageid}.jpg",
            "commons_sha1": hashlib.sha1(payload).hexdigest(),
            "artist": f"Creator {pageid}",
            "creator_group": "commons-creator:" + hashlib.sha256(artist.encode()).hexdigest()[:16],
            "acquisition_mode": "search",
            "acquisition_query": f"fixture query {pageid}",
            "species_hint": "fixture acquisition hint",
            "species_hint_status": "acquisition_hint_not_a_reviewed_species_or_shape_label",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "license_raw_name": "CC BY-SA 4.0",
            "license_raw_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "license_canonical_id": "CC_BY_SA_4_0",
            "license_canonical_name": "CC BY-SA 4.0",
            "license_canonical_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "license_binding_id": "policy:CC_BY_SA_4_0:CC BY-SA 4.0|https://creativecommons.org/licenses/by-sa/4.0/",
            "license_policy_sha256": POLICY_SHA,
            "copyrighted": "True",
            "rights_review_status": "machine_allowlist_pass_human_file_page_and_non_copyright_rights_review_pending",
            "mime": "image/png",
            "original_width": 1024,
            "original_height": 1024,
            "filename": filename,
            "download_sha256": facts["sha256"],
            "download_bytes": facts["bytes"],
            "download_width": facts["width"],
            "download_height": facts["height"],
            "download_mime": facts["mime"],
            "dhash64_algorithm": "rootscope_rgb_center_sample_9x8_v1",
            "dhash64": facts["dhash"],
        }

    def make_holdout(self, pageid: int, index: int) -> dict:
        class_id = ("grass_clump", "low_shrub", "young_tree")[index % 3]
        filename = f"images/{class_id}/{pageid}.png"
        payload = f"holdout-{pageid}".encode("ascii")
        facts = self._asset_file(self.holdout_root, filename, payload)
        public_domain = index == 2
        legacy_http_cc = pageid == 75559442
        return {
            "class_id": class_id,
            "domain": "print_demo_source",
            "split": "print_demo",
            "source_provider": "Wikimedia Commons",
            "source_group": f"commons:{pageid}",
            "pageid": pageid,
            "title": f"File:Holdout {pageid}.jpg",
            "source_page": f"https://commons.wikimedia.org/wiki/File:Holdout_{pageid}.jpg",
            "commons_sha1": hashlib.sha1(payload).hexdigest(),
            "artist": f"Holdout Creator {pageid}",
            "license": (
                "Public domain"
                if public_domain
                else "CC BY-SA 3.0" if legacy_http_cc else "CC BY-SA 4.0"
            ),
            "license_url": (
                ""
                if public_domain
                else "http://creativecommons.org/licenses/by-sa/3.0/"
                if legacy_http_cc
                else "https://creativecommons.org/licenses/by-sa/4.0/"
            ),
            "copyrighted": "False" if public_domain else "True",
            "filename": filename,
            "download_sha256": facts["sha256"],
        }

    def flush(self) -> None:
        write_jsonl(self.staging_manifest, self.candidates)
        write_jsonl(self.holdout_manifest, self.holdouts)
        summary = {"schema_version": "fixture.summary.v1", "status": "NOT_TRAIN_READY"}
        (self.staging_root / "summary.json").write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        audit = {
            "schema_version": "rootscope.wikimedia_staging_integrity_audit.v2",
            "result": "PASS_STAGING_INTEGRITY_NOT_TRAIN_READY",
            "manifest_sha256": sha256(self.staging_manifest.read_bytes()),
            "summary_sha256": sha256((self.staging_root / "summary.json").read_bytes()),
            "license_policy_sha256": POLICY_SHA,
            "collector_script_sha256": sha256((MODULE_PATH.parent / "collect_wikimedia_candidates.ps1").read_bytes()),
            "failure_count": 0,
            "failures": [],
            "thresholds": {
                "holdout_dhash_reject_at_or_below": 1,
                "candidate_dhash_reject_at_or_below": 0,
            },
            "image_constraints": policy["image_constraints"],
        }
        self.integrity_audit.write_text(
            json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )


class BuildWikimediaReviewQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.fixture = ReviewQueueFixture(Path(self.tempdir.name))

    def build(self) -> dict:
        return review_queue.build_review_queue(
            self.fixture.staging_manifest,
            self.fixture.holdout_manifest,
            self.fixture.output_dir,
            self.fixture.integrity_audit,
            POLICY_PATH,
        )

    def assert_fails_without_output(self) -> None:
        with self.assertRaises(review_queue.ReviewQueueError):
            self.build()
        self.assertFalse(self.fixture.output_dir.exists())

    def test_builds_deterministic_unreviewed_queue_and_separate_holdouts(self) -> None:
        second = self.fixture.make_candidate(900002, "low_shrub", b"candidate-two")
        self.fixture.candidates.append(second)
        self.fixture.flush()

        summary = self.build()
        first_bytes = {
            path.name: path.read_bytes() for path in sorted(self.fixture.output_dir.iterdir())
        }
        summary_second = self.build()
        second_bytes = {
            path.name: path.read_bytes() for path in sorted(self.fixture.output_dir.iterdir())
        }

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(summary, summary_second)
        self.assertEqual(summary["status"], "UNREVIEWED_NOT_TRAIN_READY")
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["raw_commons_page_count"], 2)
        self.assertEqual(summary["approved_source_group_count"], 0)
        self.assertEqual(summary["permanent_print_holdout_count"], 6)
        self.assertEqual(
            summary["inputs"]["staging_manifest_sha256"],
            sha256(self.fixture.staging_manifest.read_bytes()),
        )

        candidates = [
            json.loads(line)
            for line in (self.fixture.output_dir / "candidate_review_queue.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        holdouts = [
            json.loads(line)
            for line in (self.fixture.output_dir / "permanent_print_holdouts.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len(holdouts), 6)
        for candidate in candidates:
            self.assertEqual(candidate["review_status"], "UNREVIEWED")
            self.assertEqual(candidate["split"], "UNASSIGNED_DO_NOT_TRAIN")
            self.assertFalse(candidate["training_eligible"])
            self.assertFalse(candidate["print_eligible"])
            for field in (
                "visual_decision",
                "rights_decision",
                "target_class",
                "reviewed_source_group",
                "near_duplicate_family",
                "reviewer",
                "notes",
            ):
                self.assertEqual(candidate[field], "")
        self.assertTrue(
            {candidate["pageid"] for candidate in candidates}.isdisjoint(
                {holdout["pageid"] for holdout in holdouts}
            )
        )
        self.assertTrue(all(not holdout["candidate_review_eligible"] for holdout in holdouts))
        public_domain_holdout = next(item for item in holdouts if item["pageid"] == HOLDOUT_PAGEIDS[2])
        self.assertEqual(
            public_domain_holdout["license_url_basis"],
            "public_domain_commons_file_page_fallback",
        )
        legacy_http_holdout = next(item for item in holdouts if item["pageid"] == 75559442)
        self.assertEqual(
            legacy_http_holdout["license_url"],
            "https://creativecommons.org/licenses/by-sa/3.0/",
        )
        self.assertEqual(
            legacy_http_holdout["license_url_basis"],
            "exception:legacy-holdout-75559442-http-by-sa-3.0",
        )

    def test_missing_required_provenance_fails_closed(self) -> None:
        del self.fixture.candidates[0]["artist"]
        self.fixture.flush()
        self.assert_fails_without_output()

    def test_duplicate_sha_fails_closed(self) -> None:
        duplicate = self.fixture.make_candidate(900002, "low_shrub", b"candidate-one")
        self.fixture.candidates.append(duplicate)
        self.fixture.flush()
        self.assert_fails_without_output()

    def test_duplicate_source_group_fails_closed(self) -> None:
        duplicate = self.fixture.make_candidate(900002, "low_shrub", b"candidate-two")
        duplicate["source_group"] = self.fixture.candidates[0]["source_group"]
        self.fixture.candidates.append(duplicate)
        self.fixture.flush()
        self.assert_fails_without_output()

    def test_missing_file_and_sha_mismatch_fail_closed(self) -> None:
        with self.subTest("missing file"):
            Path(self.fixture.staging_root / self.fixture.candidates[0]["filename"]).unlink()
            self.fixture.flush()
            self.assert_fails_without_output()

        self.fixture = ReviewQueueFixture(Path(self.tempdir.name) / "mismatch")
        with self.subTest("sha mismatch"):
            self.fixture.candidates[0]["download_sha256"] = "0" * 64
            self.fixture.flush()
            self.assert_fails_without_output()

    def test_candidate_overlapping_permanent_holdout_fails_closed(self) -> None:
        holdout_pageid = HOLDOUT_PAGEIDS[0]
        self.fixture.candidates[0]["pageid"] = holdout_pageid
        self.fixture.candidates[0]["source_group"] = f"commons:{holdout_pageid}"
        self.fixture.flush()
        self.assert_fails_without_output()

    def test_missing_one_of_six_permanent_holdouts_fails_closed(self) -> None:
        self.fixture.holdouts.pop()
        self.fixture.flush()
        self.assert_fails_without_output()

    def test_path_traversal_fails_closed_even_when_target_exists(self) -> None:
        outside = self.fixture.root / "outside.jpg"
        outside.write_bytes(b"outside")
        self.fixture.candidates[0]["filename"] = "../outside.jpg"
        self.fixture.candidates[0]["download_sha256"] = sha256(b"outside")
        self.fixture.flush()
        self.assert_fails_without_output()

    def test_wrong_license_host_path_and_cc0ish_name_fail_closed(self) -> None:
        corruptions = (
            ("wrong host", "CC BY-SA 4.0", "https://example.com/licenses/by-sa/4.0/"),
            ("wrong path", "CC BY-SA 4.0", "https://creativecommons.org/licenses/by/4.0/"),
            ("cc0-ish", "CC0-ish", "https://creativecommons.org/publicdomain/zero/1.0/"),
        )
        for index, (label, license_name, license_url) in enumerate(corruptions):
            with self.subTest(label):
                self.fixture = ReviewQueueFixture(Path(self.tempdir.name) / f"license-{index}")
                self.fixture.candidates[0]["license"] = license_name
                self.fixture.candidates[0]["license_url"] = license_url
                self.fixture.candidates[0]["license_canonical_url"] = license_url
                self.fixture.flush()
                self.assert_fails_without_output()

    def test_exact_cc0_and_by_sa_25_policy_pairs_are_accepted(self) -> None:
        cases = (
            (
                "cc0 commons raw",
                "CC0",
                "http://creativecommons.org/publicdomain/zero/1.0/deed.en",
                "CC0_1_0",
                "CC0 1.0",
                "https://creativecommons.org/publicdomain/zero/1.0/",
            ),
            (
                "by-sa-2.5",
                "CC BY-SA 2.5",
                "https://creativecommons.org/licenses/by-sa/2.5",
                "CC_BY_SA_2_5",
                "CC BY-SA 2.5",
                "https://creativecommons.org/licenses/by-sa/2.5/",
            ),
        )
        for index, (label, raw_name, raw_url, canonical_id, canonical_name, canonical_url) in enumerate(cases):
            with self.subTest(label):
                self.fixture = ReviewQueueFixture(Path(self.tempdir.name) / f"accepted-{index}")
                record = self.fixture.candidates[0]
                record.update(
                    {
                        "license": raw_name,
                        "license_url": raw_url,
                        "license_raw_name": raw_name,
                        "license_raw_url": raw_url,
                        "license_canonical_id": canonical_id,
                        "license_canonical_name": canonical_name,
                        "license_canonical_url": canonical_url,
                        "license_binding_id": f"policy:{canonical_id}:{raw_name}|{raw_url}",
                    }
                )
                self.fixture.flush()
                summary = self.build()
                self.assertEqual(summary["candidate_count"], 1)

    def test_correct_canonical_cannot_mask_wrong_raw_pair_or_unpinned_http(self) -> None:
        corruptions = (
            (
                "wrong raw host",
                "CC BY-SA 4.0",
                "https://example.com/licenses/by-sa/4.0/",
                "CC_BY_SA_4_0",
                "CC BY-SA 4.0",
                "https://creativecommons.org/licenses/by-sa/4.0/",
            ),
            (
                "unpinned legacy http",
                "CC BY-SA 3.0",
                "http://creativecommons.org/licenses/by-sa/3.0/",
                "CC_BY_SA_3_0",
                "CC BY-SA 3.0",
                "https://creativecommons.org/licenses/by-sa/3.0/",
            ),
        )
        for index, (label, raw_name, raw_url, canonical_id, canonical_name, canonical_url) in enumerate(corruptions):
            with self.subTest(label):
                self.fixture = ReviewQueueFixture(Path(self.tempdir.name) / f"raw-{index}")
                self.fixture.candidates[0].update(
                    {
                        "license": raw_name,
                        "license_url": raw_url,
                        "license_raw_name": raw_name,
                        "license_raw_url": raw_url,
                        "license_canonical_id": canonical_id,
                        "license_canonical_name": canonical_name,
                        "license_canonical_url": canonical_url,
                        "license_binding_id": f"policy:{canonical_id}:{raw_name}|{raw_url}",
                    }
                )
                self.fixture.flush()
                self.assert_fails_without_output()

    def test_stale_or_failed_integrity_audit_fails_closed(self) -> None:
        with self.subTest("failed result"):
            audit = json.loads(self.fixture.integrity_audit.read_text(encoding="utf-8"))
            audit["result"] = "FAIL"
            self.fixture.integrity_audit.write_text(json.dumps(audit) + "\n", encoding="utf-8")
            self.assert_fails_without_output()

        self.fixture = ReviewQueueFixture(Path(self.tempdir.name) / "stale")
        with self.subTest("stale manifest hash"):
            audit = json.loads(self.fixture.integrity_audit.read_text(encoding="utf-8"))
            audit["manifest_sha256"] = "0" * 64
            self.fixture.integrity_audit.write_text(json.dumps(audit) + "\n", encoding="utf-8")
            self.assert_fails_without_output()

    def test_recomputed_dhash_decode_and_creator_binding_fail_closed(self) -> None:
        with self.subTest("dhash mismatch"):
            self.fixture.candidates[0]["dhash64"] = "0" * 16
            self.fixture.flush()
            self.assert_fails_without_output()

        self.fixture = ReviewQueueFixture(Path(self.tempdir.name) / "decode")
        with self.subTest("non-image payload"):
            record = self.fixture.candidates[0]
            path = self.fixture.staging_root / record["filename"]
            path.write_bytes(b"not-an-image")
            record["download_sha256"] = sha256(b"not-an-image")
            record["download_bytes"] = len(b"not-an-image")
            self.fixture.flush()
            self.assert_fails_without_output()

        self.fixture = ReviewQueueFixture(Path(self.tempdir.name) / "creator")
        with self.subTest("creator group mismatch"):
            self.fixture.candidates[0]["creator_group"] = "commons-creator:" + "0" * 16
            self.fixture.flush()
            self.assert_fails_without_output()


if __name__ == "__main__":
    unittest.main()
