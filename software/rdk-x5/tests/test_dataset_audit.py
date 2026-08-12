from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin


ROOTSCOPE = Path(__file__).resolve().parents[1]
ADVENTUREX = ROOTSCOPE.parent
sys.path.insert(0, str(ROOTSCOPE / "training"))

from dataset_audit import REQUIRED_PARTITIONS, _canonical_json_sha256, audit_dataset  # noqa: E402


CLASS_CONTRACT = ROOTSCOPE / "configs" / "class_contract.json"
CLASSES = ("grass_clump", "low_shrub", "young_tree", "unknown")
TARGET_CLASSES = CLASSES[:3]


class DatasetAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dataset = Path(self.temp.name)
        self.optical_root: str | None = None
        self.rows = [self._row(index, class_id, split="unassigned", domain="natural_web") for index, class_id in enumerate(CLASSES, 1)]
        self.rows.append(self._row(20, "grass_clump", split="print_demo", domain="print_demo_source"))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_image(self, path: Path, seed: str, size: tuple[int, int] = (256, 256)) -> None:
        digest = hashlib.sha512(seed.encode("utf-8")).digest() + hashlib.sha256(seed.encode("utf-8")).digest()
        small = Image.new("L", (9, 8))
        small.putdata(list(digest[:72]))
        resampling = getattr(Image, "Resampling", Image)
        image = small.resize(size, resampling.NEAREST)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")

    def _row(
        self,
        pageid: int,
        class_id: str,
        *,
        split: str,
        domain: str,
        review_status: str = "pending",
        source_group: str | None = None,
        asset_role: str = "source",
        origin_pageid: int | str | None = None,
        origin_sha256: str | None = None,
        print_eligible: bool = False,
        ptq_calibration: bool = False,
        permanent_holdout: bool | None = None,
        sealed: bool | None = None,
        unknown_scenario: str | None = None,
        reviewed_by: str | None = None,
        optical_domain_root: str | None = None,
        capture_id: str | None = None,
        capture_quality_pass: bool | None = None,
        capture_operator: str | None = None,
        capture_condition_id: str | None = None,
        image_seed: str | None = None,
        image_size: tuple[int, int] = (256, 256),
    ) -> dict:
        relative = Path("images") / class_id / f"asset_{pageid}.png"
        absolute = self.dataset / relative
        self._write_image(absolute, image_seed or f"fixture-image-{pageid}", image_size)
        asset_sha256 = hashlib.sha256(absolute.read_bytes()).hexdigest()
        is_capture = asset_role in {"print_capture", "local_capture"}
        if origin_pageid is None:
            origin_pageid = pageid
        if origin_sha256 is None:
            origin_sha256 = asset_sha256
        if is_capture and optical_domain_root is None:
            optical_domain_root = self.optical_root or "a" * 64
        if is_capture and capture_id is None:
            capture_id = f"capture:{pageid}"
        if capture_quality_pass is None:
            capture_quality_pass = is_capture
        if is_capture and capture_operator is None:
            capture_operator = "fixture-capture-operator"
        if is_capture and capture_condition_id is None:
            capture_condition_id = f"condition:{pageid}"
        return {
            "record_schema_version": "2.0.0",
            "class_id": class_id,
            "domain": domain,
            "split": split,
            "review_status": review_status,
            "source_group": source_group or f"fixture:{pageid}",
            "asset_id": f"asset:{pageid}",
            "asset_sha256": asset_sha256,
            "origin_pageid": origin_pageid,
            "origin_sha256": origin_sha256,
            "asset_role": asset_role,
            "filename": relative.as_posix(),
            "source_provider_id": "wikimedia_commons",
            "source_provider": "Wikimedia Commons",
            "source_page": f"https://commons.wikimedia.org/wiki/File:fixture_{origin_pageid}.png",
            "download_url": f"https://upload.wikimedia.org/wikipedia/commons/fixture_{pageid}.png",
            "artist": "Fixture Author",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "print_eligible": print_eligible,
            "ptq_calibration": ptq_calibration,
            "permanent_holdout": split == "print_demo" if permanent_holdout is None else permanent_holdout,
            "sealed": split in {"natural_test", "printed_test", "print_demo", "site_acceptance"} if sealed is None else sealed,
            "unknown_scenario": unknown_scenario,
            "reviewed_by": reviewed_by,
            "optical_domain_root": optical_domain_root if is_capture else None,
            "capture_id": capture_id if is_capture else None,
            "capture_quality_pass": capture_quality_pass if is_capture else False,
            "capture_operator": capture_operator if is_capture else None,
            "capture_condition_id": capture_condition_id if is_capture else None,
        }

    def _approved(self, pageid: int, class_id: str, *, split: str, domain: str, **kwargs: object) -> dict:
        return self._row(
            pageid,
            class_id,
            split=split,
            domain=domain,
            review_status="approved",
            reviewed_by="fixture-reviewer",
            **kwargs,
        )

    def _source_and_captures(
        self,
        start_id: int,
        class_id: str,
        *,
        split: str,
        source_domain: str,
        capture_domain: str,
        capture_count: int,
        source_group: str,
        scenario: str | None = None,
        capture_role: str = "print_capture",
        print_eligible: bool = True,
    ) -> tuple[list[dict], int]:
        source = self._approved(
            start_id,
            class_id,
            split=split,
            domain=source_domain,
            source_group=source_group,
            print_eligible=print_eligible,
            unknown_scenario=scenario,
        )
        rows = [source]
        next_id = start_id + 1
        for _ in range(capture_count):
            capture = self._approved(
                next_id,
                class_id,
                split=split,
                domain=capture_domain,
                source_group=source_group,
                asset_role=capture_role,
                origin_pageid=source["origin_pageid"],
                origin_sha256=source["origin_sha256"],
                print_eligible=print_eligible,
                unknown_scenario=scenario,
                optical_domain_root=self.optical_root,
            )
            capture["download_url"] = source["download_url"]
            rows.append(capture)
            next_id += 1
        return rows, next_id

    def _write_optical_receipt(self, *, missing_role: str | None = None) -> tuple[Path, str]:
        receipt = {
            "schema_version": "1.0.0",
            "receipt_id": "fixture.final-optics.v1",
            "signed_roles": {
                "hardware": {
                    "member": "B",
                    "signed": True,
                    "signer": "fixture-hardware",
                    "approval_evidence_sha256": hashlib.sha256(b"approval-hardware").hexdigest(),
                },
                "mechanical": {
                    "member": "C",
                    "signed": True,
                    "signer": "fixture-mechanical",
                    "approval_evidence_sha256": hashlib.sha256(b"approval-mechanical").hexdigest(),
                },
                "operations": {
                    "member": "D",
                    "signed": True,
                    "signer": "fixture-operations",
                    "approval_evidence_sha256": hashlib.sha256(b"approval-operations").hexdigest(),
                },
            },
            "evidence_roots": {
                name: hashlib.sha256(f"fixture-{name}".encode()).hexdigest()
                for name in ("uvc", "lighting", "paper", "printer", "geometry")
            },
        }
        if missing_role is not None:
            receipt["signed_roles"].pop(missing_role)
        path = self.dataset / f"optical_receipt_{missing_role or 'complete'}.json"
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        root = _canonical_json_sha256(receipt)
        self.optical_root = root
        return path, root

    def _audit(
        self,
        rows: list[dict],
        contract_path: Path = CLASS_CONTRACT,
        receipt_path: Path | None = None,
        *,
        test_only_allow_unlocked_contract: bool = False,
        contract_lock_path: Path | None = None,
    ) -> dict:
        manifest = self.dataset / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return audit_dataset(
            self.dataset,
            manifest,
            contract_path,
            curation_path=None,
            generated_at="2026-07-16T00:00:00+00:00",
            optical_domain_receipt_path=receipt_path,
            contract_lock_path=contract_lock_path,
            test_only_allow_unlocked_contract=test_only_allow_unlocked_contract,
        )

    def _audit_tiny(self, rows: list[dict], receipt_path: Path | None) -> dict:
        return self._audit(
            rows,
            self._tiny_contract(),
            receipt_path,
            test_only_allow_unlocked_contract=True,
        )

    def _codes(self, result: dict) -> set[str]:
        return {item["code"] for item in result["errors"]}

    def _readiness_codes(self, result: dict) -> set[str]:
        return {item["code"] for item in result["training_readiness"]["reason_details"]}

    def _tiny_contract(self) -> Path:
        contract = json.loads(CLASS_CONTRACT.read_text(encoding="utf-8"))
        minimums = contract["dataset_contract"]["ready_minimums"]
        minimums["natural_unique_source_groups"] = {class_id: 5 for class_id in CLASSES}
        minimums["natural_source_group_minimums_by_split"] = {
            split: {class_id: 1 for class_id in CLASSES}
            for split in ("train", "validation", "decision_calibration", "conversion_golden", "natural_test")
        }
        minimums["conversion_golden_qualifying_assets_total"] = 4
        minimums["conversion_golden_qualifying_assets_per_class"] = 1
        minimums["unknown_scenario_coverage"] = 1
        minimums["local_negative_final_optics_scenario_coverage"] = 1
        minimums["printed_train_source_groups_per_target_class"] = 1
        minimums["printed_train_source_group_minimums_by_split"] = {
            "train": {class_id: 1 for class_id in TARGET_CLASSES},
            "validation": {class_id: 0 for class_id in TARGET_CLASSES},
            "decision_calibration": {class_id: 0 for class_id in TARGET_CLASSES},
            "conversion_golden": {class_id: 0 for class_id in TARGET_CLASSES},
        }
        minimums["printed_train_captures_per_source_group"] = 2
        minimums["printed_test_source_groups_per_target_class"] = 1
        minimums["printed_test_captures_per_source_group"] = 2
        minimums["print_demo_source_groups_per_target_class"] = 1
        minimums["print_demo_captures_per_source_group"] = 1
        minimums["site_acceptance_source_groups_per_target_class"] = 1
        minimums["site_acceptance_unknown_source_groups"] = 2
        minimums["site_acceptance_unknown_scenario_coverage"] = 2
        contract["dataset_contract"]["perceptual_duplicate_audit"]["distance_threshold"] = 2
        path = self.dataset / "tiny_contract.json"
        path.write_text(json.dumps(contract), encoding="utf-8")
        return path

    def _ready_rows(self) -> list[dict]:
        rows: list[dict] = []
        next_id = 100
        for split in ("train", "validation", "decision_calibration", "conversion_golden", "natural_test"):
            for class_id in CLASSES:
                if split == "train" and class_id == "unknown":
                    local_rows, next_id = self._source_and_captures(
                        next_id,
                        class_id,
                        split=split,
                        source_domain="local_negative",
                        capture_domain="local_negative",
                        capture_count=1,
                        source_group="local-negative:bare_sand",
                        scenario="bare_sand",
                        capture_role="local_capture",
                        print_eligible=False,
                    )
                    local_rows[0]["ptq_calibration"] = True
                    rows.extend(local_rows)
                else:
                    rows.append(
                        self._approved(
                            next_id,
                            class_id,
                            split=split,
                            domain="natural_web",
                            ptq_calibration=split == "train",
                            unknown_scenario="glare" if class_id == "unknown" else None,
                        )
                    )
                    next_id += 1

        for class_id in TARGET_CLASSES:
            group_rows, next_id = self._source_and_captures(
                next_id,
                class_id,
                split="train",
                source_domain="printed_train",
                capture_domain="printed_train",
                capture_count=2,
                source_group=f"printed-train:{class_id}",
            )
            rows.extend(group_rows)
        for class_id in TARGET_CLASSES:
            group_rows, next_id = self._source_and_captures(
                next_id,
                class_id,
                split="printed_test",
                source_domain="printed_test",
                capture_domain="printed_test",
                capture_count=2,
                source_group=f"printed-test:{class_id}",
            )
            rows.extend(group_rows)
        for class_id in TARGET_CLASSES:
            group_rows, next_id = self._source_and_captures(
                next_id,
                class_id,
                split="print_demo",
                source_domain="print_demo_source",
                capture_domain="printed_demo_capture",
                capture_count=1,
                source_group=f"print-demo:{class_id}",
            )
            rows.extend(group_rows)
        for class_id in TARGET_CLASSES:
            rows.append(
                self._approved(
                    next_id,
                    class_id,
                    split="site_acceptance",
                    domain="site_acceptance",
                    print_eligible=True,
                )
            )
            next_id += 1
        for scenario in ("empty_card", "hand"):
            rows.append(
                self._approved(
                    next_id,
                    "unknown",
                    split="site_acceptance",
                    domain="site_acceptance",
                    print_eligible=False,
                    unknown_scenario=scenario,
                )
            )
            next_id += 1
        return rows

    def test_contract_has_exact_eight_source_group_partitions(self) -> None:
        contract = json.loads(CLASS_CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(list(REQUIRED_PARTITIONS), contract["dataset_contract"]["source_group_partitions"])
        rules = contract["dataset_contract"]
        self.assertEqual(6, rules["perceptual_duplicate_audit"]["distance_threshold"])
        self.assertEqual(10, rules["ready_minimums"]["print_demo_captures_per_source_group"])
        self.assertEqual(100, rules["ready_minimums"]["conversion_golden_qualifying_assets_total"])

    def test_valid_integrity_can_be_not_train_ready(self) -> None:
        result = self._audit(self.rows)
        self.assertEqual("PASS", result["status"])
        self.assertEqual("NOT_TRAIN_READY", result["training_readiness"]["status"])
        self.assertTrue(result["inputs"]["class_contract_lock"]["production_bound"])
        self.assertNotIn("CLASS_CONTRACT_LOCK_INVALID", self._readiness_codes(result))
        self.assertIn("ROWS_UNASSIGNED", self._readiness_codes(result))
        self.assertIn("OPTICAL_RECEIPT_REQUIRED", self._readiness_codes(result))

    def test_stale_contract_lock_blocks_ready_without_failing_row_integrity(self) -> None:
        lock = self.dataset / "stale_contract.lock.json"
        lock.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "profile": "rootscope.dataset_contract.production.v2",
                    "contract_version": "2.0.0",
                    "class_contract_sha256": "0" * 64,
                }
            ),
            encoding="utf-8",
        )
        result = self._audit(self.rows, contract_lock_path=lock)
        self.assertEqual("PASS", result["integrity_status"])
        self.assertIn("CLASS_CONTRACT_LOCK_INVALID", self._readiness_codes(result))

    def test_tampered_contract_lock_profile_blocks_ready(self) -> None:
        lock = self.dataset / "tampered_contract.lock.json"
        lock.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "profile": "rootscope.dataset_contract.production.v1",
                    "contract_version": "2.0.0",
                    "class_contract_sha256": hashlib.sha256(CLASS_CONTRACT.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        result = self._audit(self.rows, contract_lock_path=lock)
        self.assertEqual("PASS", result["integrity_status"])
        self.assertIn("CLASS_CONTRACT_LOCK_INVALID", self._readiness_codes(result))

    def test_printed_domain_cannot_enter_wrong_partition(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[-1]["domain"] = "printed_test"
        rows[-1]["split"] = "train"
        rows[-1]["permanent_holdout"] = False
        rows[-1]["sealed"] = False
        result = self._audit(rows)
        self.assertIn("DOMAIN_SPLIT_MISMATCH", self._codes(result))

    def test_source_group_cannot_cross_partitions(self) -> None:
        rows = copy.deepcopy(self.rows)
        extra = self._row(30, "grass_clump", split="natural_test", domain="natural_web", source_group=rows[0]["source_group"])
        extra["asset_role"] = "crop"
        extra["origin_pageid"] = rows[0]["origin_pageid"]
        extra["origin_sha256"] = rows[0]["origin_sha256"]
        rows[0]["split"] = "train"
        rows.append(extra)
        result = self._audit(rows)
        self.assertIn("SOURCE_GROUP_PARTITION_LEAKAGE", self._codes(result))

    def test_source_group_cannot_cross_classes(self) -> None:
        rows = copy.deepcopy(self.rows)
        extra = self._row(31, "low_shrub", split="unassigned", domain="natural_web", source_group=rows[0]["source_group"])
        extra["asset_role"] = "crop"
        extra["origin_pageid"] = rows[0]["origin_pageid"]
        extra["origin_sha256"] = rows[0]["origin_sha256"]
        rows.append(extra)
        result = self._audit(rows)
        self.assertIn("SOURCE_GROUP_CLASS_LEAKAGE", self._codes(result))

    def test_source_plus_four_captures_share_lineage_legally(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = copy.deepcopy(self.rows)
        group_rows, _ = self._source_and_captures(
            40,
            "grass_clump",
            split="train",
            source_domain="printed_train",
            capture_domain="printed_train",
            capture_count=4,
            source_group="lineage:positive",
        )
        rows.extend(group_rows)
        result = self._audit(rows, receipt_path=receipt)
        self.assertEqual("PASS", result["integrity_status"])

    def test_origin_cannot_cross_source_groups(self) -> None:
        rows = copy.deepcopy(self.rows)
        extra = self._row(50, "grass_clump", split="unassigned", domain="natural_web")
        extra["origin_pageid"] = rows[0]["origin_pageid"]
        extra["origin_sha256"] = rows[0]["origin_sha256"]
        rows.append(extra)
        result = self._audit(rows)
        self.assertIn("ORIGIN_SOURCE_GROUP_LEAKAGE", self._codes(result))

    def test_duplicate_asset_identity_and_sha_are_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        duplicate = copy.deepcopy(rows[1])
        duplicate["source_group"] = "fixture:new-group"
        rows.append(duplicate)
        result = self._audit(rows)
        self.assertIn("DUPLICATE_ASSET_ID", self._codes(result))
        self.assertIn("DUPLICATE_ASSET_SHA256", self._codes(result))

    def test_rejected_visual_must_be_excluded_and_unprivileged(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["review_status"] = "rejected_visual"
        rows[0]["split"] = "train"
        rows[0]["print_eligible"] = True
        rows[0]["ptq_calibration"] = True
        result = self._audit(rows)
        self.assertIn("REJECTED_NOT_EXCLUDED", self._codes(result))
        self.assertIn("REJECTED_PRIVILEGED", self._codes(result))

    def test_asset_hash_mismatch_and_missing_license_url_fail(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[2]["asset_sha256"] = "0" * 64
        rows[2]["origin_sha256"] = "0" * 64
        rows[2]["license_url"] = ""
        result = self._audit(rows)
        self.assertIn("ASSET_SHA_MISMATCH", self._codes(result))
        self.assertIn("LICENSE_URL_MISSING", self._codes(result))

    def test_license_requires_exact_name_and_canonical_url(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["license"] = "CC0-ish-unverified"
        rows[1]["license_url"] = "https://example.test/not-a-license"
        rows[2]["license_url"] = "http://creativecommons.org/licenses/by-sa/4.0/"
        result = self._audit(rows)
        self.assertIn("LICENSE_NOT_ALLOWED", self._codes(result))
        self.assertIn("LICENSE_URL_NONCANONICAL", self._codes(result))

    def test_exact_cc_by_sa_25_with_canonical_url_is_allowed(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["license"] = "CC BY-SA 2.5"
        rows[0]["license_url"] = "https://creativecommons.org/licenses/by-sa/2.5/"
        result = self._audit(rows)
        self.assertEqual("PASS", result["integrity_status"])

    def test_public_domain_cannot_use_cc0_zero_url(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["license"] = "Public domain"
        rows[0]["license_url"] = "https://creativecommons.org/publicdomain/zero/1.0/"
        result = self._audit(rows)
        self.assertIn("LICENSE_URL_NONCANONICAL", self._codes(result))

    def test_filename_wrong_type_escape_and_missing_file_fail(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["filename"] = 123
        rows[1]["filename"] = "../escape.png"
        (self.dataset / rows[2]["filename"]).unlink()
        rows[3].pop("filename")
        result = self._audit(rows)
        self.assertIn("FIELD_TYPE_INVALID", self._codes(result))
        self.assertIn("FIELD_MISSING", self._codes(result))
        self.assertIn("FILENAME_INVALID", self._codes(result))
        self.assertIn("PATH_ESCAPE", self._codes(result))
        self.assertIn("FILE_MISSING", self._codes(result))

    def test_bad_decode_and_too_small_images_fail(self) -> None:
        rows = copy.deepcopy(self.rows)
        bad_path = self.dataset / rows[0]["filename"]
        bad_path.write_bytes(b"not-an-image")
        bad_sha = hashlib.sha256(bad_path.read_bytes()).hexdigest()
        rows[0]["asset_sha256"] = bad_sha
        rows[0]["origin_sha256"] = bad_sha
        small_path = self.dataset / rows[1]["filename"]
        self._write_image(small_path, "small", (32, 32))
        small_sha = hashlib.sha256(small_path.read_bytes()).hexdigest()
        rows[1]["asset_sha256"] = small_sha
        rows[1]["origin_sha256"] = small_sha
        result = self._audit(rows)
        self.assertIn("IMAGE_DECODE_FAILED", self._codes(result))
        self.assertIn("IMAGE_TOO_SMALL", self._codes(result))

    def test_required_boolean_and_enum_fields_are_not_coerced(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["sealed"] = "false"
        rows[1]["class_id"] = 123
        rows[2]["origin_pageid"] = True
        result = self._audit(rows)
        self.assertIn("FIELD_TYPE_INVALID", self._codes(result))

    def test_native_v2_rejects_unfrozen_extra_fields(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["silent_future_field"] = "must-not-pass"
        result = self._audit(rows)
        self.assertIn("NATIVE_V2_UNKNOWN_FIELDS", self._codes(result))

    def test_ptq_subset_is_train_only_and_approved(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["split"] = "validation"
        rows[0]["ptq_calibration"] = True
        result = self._audit(rows)
        self.assertIn("PTQ_OUTSIDE_TRAIN", self._codes(result))
        self.assertIn("PTQ_NOT_APPROVED", self._codes(result))

    def test_protected_splits_require_sealed_and_demo_requires_permanent(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[-1]["sealed"] = False
        rows[-1]["permanent_holdout"] = False
        result = self._audit(rows)
        self.assertIn("SEALED_REQUIRED", self._codes(result))
        self.assertIn("PRINT_DEMO_NOT_PERMANENT", self._codes(result))

    def test_natural_and_printed_test_are_sealed_but_not_permanent(self) -> None:
        rows = copy.deepcopy(self.rows)
        natural_test = self._row(55, "grass_clump", split="natural_test", domain="natural_web")
        printed_test = self._row(56, "grass_clump", split="printed_test", domain="printed_test")
        rows.extend((natural_test, printed_test))
        positive = self._audit(rows)
        self.assertEqual("PASS", positive["integrity_status"])
        rows[-2]["sealed"] = False
        rows[-1]["sealed"] = False
        negative = self._audit(rows)
        sealed_errors = [item for item in negative["errors"] if item["code"] == "SEALED_REQUIRED"]
        self.assertEqual(2, len(sealed_errors))

    def test_non_evaluation_split_cannot_claim_sealed(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["split"] = "train"
        rows[0]["sealed"] = True
        result = self._audit(rows)
        self.assertIn("SEALED_OUTSIDE_PROTECTED_SPLIT", self._codes(result))

    def test_near_duplicate_across_distinct_groups_fails(self) -> None:
        rows = copy.deepcopy(self.rows)
        first_path = self.dataset / rows[0]["filename"]
        second_path = self.dataset / rows[1]["filename"]
        with Image.open(first_path) as image:
            duplicate = image.copy()
        duplicate.putpixel((255, 255), (duplicate.getpixel((255, 255)) + 1) % 256)
        duplicate.save(second_path, format="PNG")
        second_sha = hashlib.sha256(second_path.read_bytes()).hexdigest()
        rows[1]["asset_sha256"] = second_sha
        rows[1]["origin_sha256"] = second_sha
        result = self._audit(rows)
        self.assertIn("NEAR_DUPLICATE_SOURCE_GROUP", self._codes(result))

    def test_origin_pageid_int_and_string_are_same_provider_identity(self) -> None:
        rows = copy.deepcopy(self.rows)
        extra = self._row(60, "grass_clump", split="natural_test", domain="natural_web", origin_pageid="1")
        rows.append(extra)
        result = self._audit(rows)
        self.assertIn("ORIGIN_PAGEID_BINDING_CONFLICT", self._codes(result))

    def test_provider_specific_origin_id_rejects_leading_zero_alias(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["origin_pageid"] = "01"
        result = self._audit(rows)
        self.assertIn("ORIGIN_ID_INVALID_FOR_PROVIDER", self._codes(result))

    def test_provider_case_alias_and_canonical_source_page_alias_fail(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["source_provider"] = "wikimedia commons"
        extra = self._row(61, "grass_clump", split="unassigned", domain="natural_web")
        extra["source_page"] = self.rows[0]["source_page"] + "/#same-page-fragment"
        rows.append(extra)
        result = self._audit(rows)
        self.assertIn("SOURCE_PROVIDER_NOT_CANONICAL", self._codes(result))
        self.assertIn("SOURCE_PAGE_BINDING_CONFLICT", self._codes(result))

    def test_local_provider_uses_canonical_rootscope_urn(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["source_provider_id"] = "rootscope_local_capture"
        rows[0]["source_provider"] = "RootScope Local Capture"
        rows[0]["origin_pageid"] = "capture-001"
        rows[0]["source_page"] = "urn:rootscope:local:capture-001"
        rows[0]["download_url"] = "urn:rootscope:local:asset-001"
        result = self._audit(rows)
        self.assertEqual("PASS", result["integrity_status"])

    def test_changed_legacy_manifest_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        for row in rows:
            for field in tuple(row):
                if field in {
                    "record_schema_version",
                    "asset_id",
                    "asset_sha256",
                    "origin_pageid",
                    "origin_sha256",
                    "asset_role",
                    "ptq_calibration",
                    "permanent_holdout",
                    "sealed",
                    "unknown_scenario",
                    "reviewed_by",
                    "optical_domain_root",
                    "capture_id",
                    "capture_quality_pass",
                    "capture_operator",
                    "capture_condition_id",
                }:
                    row.pop(field)
        result = self._audit(rows)
        self.assertIn("UNREGISTERED_LEGACY_MANIFEST", self._codes(result))

    def test_registered_33_row_v1_seed_projects_but_stays_not_ready(self) -> None:
        dataset = ADVENTUREX / "datasets" / "desert_plants_v1"
        result = audit_dataset(
            dataset,
            dataset / "manifest.jsonl",
            CLASS_CONTRACT,
            dataset / "curation_round1.json",
            generated_at="2026-07-16T00:00:00+00:00",
        )
        self.assertEqual("PASS", result["integrity_status"])
        self.assertTrue(result["migration"]["applied"])
        self.assertEqual(33, result["summary"]["row_count"])
        self.assertEqual("NOT_TRAIN_READY", result["training_readiness"]["status"])
        self.assertIn("NATIVE_V2_NOT_PERSISTED", self._readiness_codes(result))

        receipt_path = ROOTSCOPE / "training" / "migrations" / "20260715_v1_to_v2_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        lock_path = ROOTSCOPE / "configs" / "class_contract.lock.json"
        self.assertEqual(hashlib.sha256(CLASS_CONTRACT.read_bytes()).hexdigest(), receipt["target"]["class_contract_sha256"])
        self.assertEqual(hashlib.sha256(lock_path.read_bytes()).hexdigest(), receipt["target"]["class_contract_lock_sha256"])
        actual_defaults = {
            key.removeprefix("default:"): value
            for key, value in result["migration"]["transformation_counts"].items()
            if key.startswith("default:")
        }
        actual_derived = {
            key.removeprefix("derived:"): value
            for key, value in result["migration"]["transformation_counts"].items()
            if key.startswith("derived:")
        }
        self.assertEqual(receipt["transformations"]["field_defaults_applied_to_rows"], actual_defaults)
        self.assertEqual(receipt["transformations"]["derived_fields_applied_to_rows"], actual_derived)

    def test_complete_tiny_v2_fixture_reaches_ready(self) -> None:
        receipt, _ = self._write_optical_receipt()
        result = self._audit_tiny(self._ready_rows(), receipt)
        self.assertEqual("PASS", result["integrity_status"])
        self.assertEqual("READY", result["training_readiness"]["status"])
        self.assertEqual([], result["training_readiness"]["reasons"])
        self.assertEqual(2, result["training_readiness"]["metrics"]["site_acceptance_source_groups"]["unknown"])

    def test_unpinned_downgraded_contract_cannot_reach_production_ready(self) -> None:
        receipt, _ = self._write_optical_receipt()
        result = self._audit(self._ready_rows(), self._tiny_contract(), receipt)
        self.assertEqual("PASS", result["integrity_status"])
        self.assertEqual("NOT_TRAIN_READY", result["training_readiness"]["status"])
        self.assertIn("CLASS_CONTRACT_LOCK_INVALID", self._readiness_codes(result))

    def test_ready_shape_without_receipt_stays_not_ready_but_integrity_passes(self) -> None:
        self._write_optical_receipt()
        result = self._audit_tiny(self._ready_rows(), receipt_path=None)
        self.assertEqual("PASS", result["integrity_status"])
        self.assertEqual("NOT_TRAIN_READY", result["training_readiness"]["status"])
        self.assertIn("OPTICAL_RECEIPT_REQUIRED", self._readiness_codes(result))

    def test_tampered_receipt_root_is_rejected(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
        receipt_data["evidence_roots"]["uvc"] = "f" * 64
        receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
        result = self._audit_tiny(rows, receipt)
        self.assertIn("OPTICAL_RECEIPT_ROOT_MISMATCH", self._codes(result))

    def test_unknown_optical_receipt_schema_version_is_rejected(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
        receipt_data["schema_version"] = "banana-v999"
        receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
        result = self._audit_tiny(rows, receipt)
        self.assertIn("OPTICAL_RECEIPT_VERSION", self._codes(result))

    def test_multiple_final_optics_roots_are_rejected(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        capture = next(row for row in rows if row["asset_role"] == "print_capture")
        capture["optical_domain_root"] = "f" * 64
        result = self._audit_tiny(rows, receipt)
        self.assertIn("OPTICAL_ROOT_INCONSISTENT", self._codes(result))

    def test_receipt_missing_required_signed_role_is_rejected(self) -> None:
        receipt, _ = self._write_optical_receipt(missing_role="mechanical")
        rows = self._ready_rows()
        result = self._audit_tiny(rows, receipt)
        self.assertIn("OPTICAL_SIGNED_ROLE_INVALID", self._codes(result))

    def test_receipt_requires_three_distinct_signers(self) -> None:
        receipt, _ = self._write_optical_receipt()
        receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
        signer_aliases = ("Same Person", " same  person ", "SAME PERSON")
        for entry, signer in zip(receipt_data["signed_roles"].values(), signer_aliases, strict=True):
            entry["signer"] = signer
        receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
        rows = self._ready_rows()
        result = self._audit_tiny(rows, receipt)
        self.assertIn("OPTICAL_SIGNERS_NOT_DISTINCT", self._codes(result))

    def test_receipt_requires_distinct_approval_evidence(self) -> None:
        receipt, _ = self._write_optical_receipt()
        receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
        for entry in receipt_data["signed_roles"].values():
            entry["approval_evidence_sha256"] = "a" * 64
        receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
        result = self._audit_tiny(self._ready_rows(), receipt)
        self.assertIn("OPTICAL_APPROVAL_EVIDENCE_NOT_DISTINCT", self._codes(result))

    def test_receipt_rejects_control_character_signer(self) -> None:
        receipt, _ = self._write_optical_receipt()
        receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
        receipt_data["signed_roles"]["hardware"]["signer"] = "hardware\u0000alias"
        receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
        result = self._audit_tiny(self._ready_rows(), receipt)
        self.assertIn("OPTICAL_SIGNED_ROLE_INVALID", self._codes(result))

    def test_partition_requires_every_class(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        rows = [row for row in rows if not (row["split"] == "validation" and row["class_id"] == "unknown")]
        result = self._audit_tiny(rows, receipt)
        self.assertIn("PARTITION_CLASS_SOURCE_GROUP_MINIMUM", self._readiness_codes(result))

    def test_conversion_golden_requires_independent_source_assets(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        rows = [
            row
            for row in rows
            if not (
                row["split"] == "conversion_golden"
                and row["class_id"] == "young_tree"
                and row["asset_role"] == "source"
            )
        ]
        result = self._audit_tiny(rows, receipt)
        self.assertIn("CONVERSION_GOLDEN_TOTAL_MINIMUM", self._readiness_codes(result))
        self.assertIn("CONVERSION_GOLDEN_CLASS_MINIMUM", self._readiness_codes(result))

    def test_demo_source_without_quality_capture_cannot_satisfy_ready(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        rows = [row for row in rows if row["domain"] != "printed_demo_capture"]
        result = self._audit_tiny(rows, receipt)
        self.assertIn("PRINT_DEMO_MINIMUM", self._readiness_codes(result))

    def test_printed_source_must_itself_be_approved_and_print_eligible(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        for row in rows:
            if row["asset_role"] == "source" and row["domain"] in {"printed_train", "printed_test", "print_demo_source"}:
                row["print_eligible"] = False
        result = self._audit_tiny(rows, receipt)
        codes = self._readiness_codes(result)
        self.assertIn("PRINTED_TRAIN_MINIMUM", codes)
        self.assertIn("PRINTED_TEST_MINIMUM", codes)
        self.assertIn("PRINT_DEMO_MINIMUM", codes)

    def test_printed_train_split_minimum_cannot_be_reallocated(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        for row in rows:
            if row["source_group"] == "printed-train:grass_clump":
                row["split"] = "validation"
        result = self._audit_tiny(rows, receipt)
        self.assertIn("PRINTED_TRAIN_SPLIT_MINIMUM", self._readiness_codes(result))

    def test_natural_quota_ignores_crop_attached_to_printed_group(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        baseline = self._audit_tiny(rows, receipt)
        baseline_count = baseline["training_readiness"]["metrics"]["natural_unique_source_groups"]["grass_clump"]
        source = next(
            row
            for row in rows
            if row["source_group"] == "printed-train:grass_clump" and row["asset_role"] == "source"
        )
        crop = self._approved(
            990,
            "grass_clump",
            split="train",
            domain="natural_web",
            source_group=source["source_group"],
            asset_role="crop",
            origin_pageid=source["origin_pageid"],
            origin_sha256=source["origin_sha256"],
            print_eligible=True,
        )
        crop["download_url"] = source["download_url"]
        rows.append(crop)
        result = self._audit_tiny(rows, receipt)
        self.assertEqual("PASS", result["integrity_status"])
        self.assertEqual(
            baseline_count,
            result["training_readiness"]["metrics"]["natural_unique_source_groups"]["grass_clump"],
        )

    def test_capture_must_inherit_origin_attribution_tuple(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        capture = next(row for row in rows if row["asset_role"] == "print_capture")
        capture.update(
            {
                "source_provider": "Unrelated Provider",
                "source_page": "https://example.test/unrelated",
                "artist": "Unrelated Author",
                "license": "Public domain",
                "license_url": "",
            }
        )
        result = self._audit_tiny(rows, receipt)
        self.assertIn("SOURCE_GROUP_ATTRIBUTION_LEAKAGE", self._codes(result))

    def test_capture_cannot_replace_frozen_origin_download_locator(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        capture = next(row for row in rows if row["asset_role"] == "print_capture")
        capture["download_url"] = "https://upload.wikimedia.org/wikipedia/commons/unrelated-origin.png"
        result = self._audit_tiny(rows, receipt)
        self.assertIn("SOURCE_GROUP_ATTRIBUTION_LEAKAGE", self._codes(result))

    def test_failed_capture_does_not_count_toward_printed_minimum(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        capture = next(
            row
            for row in rows
            if row["domain"] == "printed_test" and row["asset_role"] == "print_capture" and row["class_id"] == "grass_clump"
        )
        capture["capture_quality_pass"] = False
        result = self._audit_tiny(rows, receipt)
        self.assertIn("PRINTED_TEST_MINIMUM", self._readiness_codes(result))

    def test_same_pixels_different_png_encoding_do_not_count_as_variations(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        captures = [
            row
            for row in rows
            if row["domain"] == "printed_test" and row["asset_role"] == "print_capture" and row["class_id"] == "grass_clump"
        ]
        first_path = self.dataset / captures[0]["filename"]
        second_path = self.dataset / captures[1]["filename"]
        with Image.open(first_path) as image:
            duplicate = image.copy()
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("encoding_variant", "different bytes, same pixels")
        duplicate.save(second_path, format="PNG", pnginfo=metadata, compress_level=9)
        captures[1]["asset_sha256"] = hashlib.sha256(second_path.read_bytes()).hexdigest()
        result = self._audit_tiny(rows, receipt)
        self.assertIn("PRINTED_TEST_MINIMUM", self._readiness_codes(result))

    def test_local_unknown_web_only_cannot_satisfy_final_optics_coverage(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        rows = [row for row in rows if row["asset_role"] != "local_capture"]
        for row in rows:
            if row["source_group"] == "local-negative:bare_sand":
                row["domain"] = "natural_web"
        result = self._audit_tiny(rows, receipt)
        self.assertIn("LOCAL_NEGATIVE_FINAL_OPTICS_COVERAGE", self._readiness_codes(result))

    def test_site_acceptance_unknown_requires_nonempty_scenario(self) -> None:
        receipt, _ = self._write_optical_receipt()
        rows = self._ready_rows()
        for row in rows:
            if row["split"] == "site_acceptance" and row["class_id"] == "unknown":
                row["unknown_scenario"] = None
        result = self._audit_tiny(rows, receipt)
        self.assertIn("SITE_UNKNOWN_SCENARIO_REQUIRED", self._codes(result))
        self.assertIn("SITE_ACCEPTANCE_UNKNOWN_SCENARIO_COVERAGE", self._readiness_codes(result))


if __name__ == "__main__":
    unittest.main()
