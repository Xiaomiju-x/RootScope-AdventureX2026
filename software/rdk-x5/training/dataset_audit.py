#!/usr/bin/env python3
"""Fail-closed RootScope dataset-contract v2 auditor.

The auditor validates provenance, licensing, file integrity, the eight-way
source-group partition, printed-domain isolation, permanent holdouts and the
train-only PTQ subset.  Integrity PASS is deliberately separate from
DATA_LOCKED readiness.  It never trains, transforms or infers with a model.

The original 33-row v1 seed manifest is supported through one deterministic,
SHA-allowlisted, in-memory migration.  Any changed manifest must be native v2;
this prevents a broad compatibility path from hiding newly missing fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from PIL import Image, UnidentifiedImageError


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTRACT_LOCK_SCHEMA_VERSION = "1.0.0"
CONTRACT_LOCK_PROFILE = "rootscope.dataset_contract.production.v2"
TARGET_CLASSES = ("grass_clump", "low_shrub", "young_tree")
REQUIRED_PARTITIONS = (
    "train",
    "validation",
    "decision_calibration",
    "conversion_golden",
    "natural_test",
    "printed_test",
    "print_demo",
    "site_acceptance",
)
REQUIRED_FIELDS = (
    "record_schema_version",
    "class_id",
    "domain",
    "split",
    "review_status",
    "source_group",
    "asset_id",
    "asset_sha256",
    "origin_pageid",
    "origin_sha256",
    "asset_role",
    "filename",
    "source_provider_id",
    "source_provider",
    "source_page",
    "download_url",
    "artist",
    "license",
    "license_url",
    "print_eligible",
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
)
NULLABLE_STRING_FIELDS = {
    "license_url",
    "unknown_scenario",
    "reviewed_by",
    "optical_domain_root",
    "capture_id",
    "capture_operator",
    "capture_condition_id",
}
STRING_FIELDS = {
    "record_schema_version",
    "class_id",
    "domain",
    "split",
    "review_status",
    "source_group",
    "asset_id",
    "asset_sha256",
    "origin_sha256",
    "asset_role",
    "filename",
    "source_provider_id",
    "source_provider",
    "source_page",
    "download_url",
    "artist",
    "license",
}
BOOLEAN_FIELDS = {"print_eligible", "ptq_calibration", "permanent_holdout", "sealed", "capture_quality_pass"}

CHECKS = {
    "CONTRACT": "Class and dataset contract is the frozen RootScope v2 shape.",
    "MANIFEST_PARSE": "Manifest is non-empty JSONL and every row is an object.",
    "MANIFEST_MIGRATION": "Legacy input is native v2 or the single SHA-allowlisted v1 seed manifest.",
    "CLASS_COVERAGE": "Manifest uses only the three visible classes plus unknown and covers all four.",
    "FIELDS": "Required v2 provenance, review, partition and holdout fields are present and typed.",
    "DOMAIN_PARTITION": "Every domain is allowed only in its contract-authorized partition.",
    "LINEAGE": "Every asset has unique identity and one source_group has one class, partition and origin.",
    "FILES": "Every string filename is contained, exists and is bound to asset_sha256.",
    "IMAGE_DECODE": "Every image decodes and meets the frozen minimum dimensions.",
    "FINAL_OPTICS": "Printed captures carry one consistent final-optics root and quality evidence.",
    "OPTICAL_RECEIPT": "Final-optics rows bind one canonical B/C/D-signed physical-domain receipt.",
    "NEAR_DUPLICATE": "Native-v2 dHash evidence has no cross-source-group near duplicate.",
    "LICENSE": "Licenses are allowlisted and attribution metadata is present.",
    "DUPLICATES": "asset_id, asset_sha256 and capture_id are unique.",
    "SOURCE_GROUP_PARTITION": "One source_group belongs to one class, one origin and one partition.",
    "PTQ_TRAIN_ONLY": "PTQ calibration rows are an approved subset of train and never a separate split.",
    "PERMANENT_HOLDOUT": "print_demo and all derivatives remain permanently held out.",
    "REJECTED_EXCLUDED": "Rejected samples are excluded from every usable partition.",
    "CURATION": "Curation reservations and rejections agree with the canonical manifest.",
    "READY_MINIMUM": "The frozen DATA_LOCKED minimums are met without holdout leakage.",
}


class Findings:
    def __init__(self) -> None:
        self.errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.warnings: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def error(self, check: str, code: str, message: str, **context: Any) -> None:
        self.errors[check].append({"code": code, "message": message, **context})

    def warning(self, check: str, code: str, message: str, **context: Any) -> None:
        self.warnings[check].append({"code": code, "message": message, **context})

    def check_results(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for check_id, description in CHECKS.items():
            errors = self.errors.get(check_id, [])
            warnings = self.warnings.get(check_id, [])
            status = "FAIL" if errors else ("WARN" if warnings else "PASS")
            results.append(
                {
                    "check_id": check_id,
                    "description": description,
                    "status": status,
                    "error_count": len(errors),
                    "warning_count": len(warnings),
                    "errors": errors,
                    "warnings": warnings,
                }
            )
        return results

    def flattened_errors(self) -> list[dict[str, Any]]:
        return [dict(check_id=check, **item) for check, items in self.errors.items() for item in items]

    def flattened_warnings(self) -> list[dict[str, Any]]:
        return [dict(check_id=check, **item) for check, items in self.warnings.items() for item in items]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json_file(path: Path) -> str | None:
    return _sha256_file(path) if path.is_file() else None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_contract_lock(
    contract_lock_path: Path,
    contract_sha256: str | None,
    contract: Any,
    *,
    test_only_allow_unlocked_contract: bool,
) -> dict[str, Any]:
    """Validate the external production root without changing integrity semantics.

    A missing or stale lock blocks DATA_LOCKED/READY, but does not make otherwise
    well-formed dataset rows corrupt.  Unit fixtures may bypass the production
    root only through the explicit, non-CLI test-only argument.
    """

    base = {
        "path": str(contract_lock_path),
        "present": contract_lock_path.is_file(),
        "valid": False,
        "production_bound": False,
        "mode": "production_lock",
        "profile": None,
        "contract_version": None,
        "expected_class_contract_sha256": None,
        "actual_class_contract_sha256": contract_sha256,
    }
    if test_only_allow_unlocked_contract:
        return {
            **base,
            "valid": True,
            "mode": "test_only_override",
            "reason": "Explicit test-only unpinned-contract override; never exposed by the production CLI.",
        }
    try:
        lock = _read_json(contract_lock_path)
    except FileNotFoundError:
        return {**base, "reason": "Production class-contract lock is missing."}
    except (OSError, json.JSONDecodeError) as exc:
        return {**base, "reason": f"Production class-contract lock is unreadable: {exc}."}
    required_keys = {
        "schema_version",
        "profile",
        "contract_version",
        "class_contract_sha256",
    }
    if not isinstance(lock, dict) or set(lock) != required_keys:
        return {**base, "reason": "Production class-contract lock has the wrong exact shape."}
    profile = lock.get("profile")
    version = lock.get("contract_version")
    expected_sha256 = lock.get("class_contract_sha256")
    bound = (
        lock.get("schema_version") == CONTRACT_LOCK_SCHEMA_VERSION
        and profile == CONTRACT_LOCK_PROFILE
        and version == "2.0.0"
        and isinstance(contract, dict)
        and contract.get("schema_version") == version
        and SHA256_RE.fullmatch(str(expected_sha256)) is not None
        and expected_sha256 == contract_sha256
    )
    return {
        **base,
        "valid": bound,
        "production_bound": bound,
        "profile": profile,
        "contract_version": version,
        "expected_class_contract_sha256": expected_sha256,
        "reason": None if bound else "Production class-contract lock does not bind this exact v2 contract.",
    }


def _read_manifest(path: Path, findings: Findings) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        findings.error("MANIFEST_PARSE", "MANIFEST_MISSING", "Manifest file does not exist.", path=str(path))
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                findings.error(
                    "MANIFEST_PARSE",
                    "INVALID_JSON",
                    f"Invalid JSON at line {line_number}: {exc.msg}.",
                    line=line_number,
                )
                continue
            if not isinstance(value, dict):
                findings.error(
                    "MANIFEST_PARSE",
                    "ROW_NOT_OBJECT",
                    f"Manifest line {line_number} is not a JSON object.",
                    line=line_number,
                )
                continue
            value = dict(value)
            value["_line"] = line_number
            rows.append(value)
    if not rows:
        findings.error("MANIFEST_PARSE", "MANIFEST_EMPTY", "Manifest has no valid rows.")
    return rows


def _validate_contract(contract: Any, findings: Findings) -> tuple[list[str], dict[str, Any]]:
    if not isinstance(contract, dict):
        findings.error("CONTRACT", "CONTRACT_NOT_OBJECT", "Class contract must be a JSON object.")
        return [], {}
    classes = contract.get("classes")
    class_order = contract.get("class_order")
    rules = contract.get("dataset_contract")
    if not isinstance(classes, list) or not isinstance(class_order, list) or not isinstance(rules, dict):
        findings.error("CONTRACT", "CONTRACT_SHAPE", "Contract must define classes, class_order and dataset_contract.")
        return [], rules if isinstance(rules, dict) else {}

    if contract.get("schema_version") != "2.0.0" or rules.get("record_schema_version") != "2.0.0":
        findings.error("CONTRACT", "CONTRACT_VERSION", "Contract and record schema versions must both be 2.0.0.")

    ids = [item.get("class_id") for item in classes if isinstance(item, dict)]
    indices = [item.get("index") for item in classes if isinstance(item, dict)]
    required_classes = {*TARGET_CLASSES, "unknown"}
    if len(classes) != 4 or set(ids) != required_classes or list(class_order) != ids:
        findings.error(
            "CONTRACT",
            "CLASS_SET_INVALID",
            "Contract must contain the frozen order grass_clump, low_shrub, young_tree, unknown.",
            actual_order=ids,
        )
    if indices != list(range(len(indices))) or len(indices) != len(set(indices)):
        findings.error("CONTRACT", "CLASS_INDEX_INVALID", "Class indices must be unique and contiguous from zero.")
    unknown = next((item for item in classes if isinstance(item, dict) and item.get("class_id") == "unknown"), {})
    if unknown.get("admission") != "reject" or unknown.get("demo_profile") is not None:
        findings.error("CONTRACT", "UNKNOWN_NOT_REJECT", "unknown must reject admission and have no demo profile.")

    partitions = rules.get("source_group_partitions")
    admin_splits = rules.get("administrative_splits")
    allowed_splits = rules.get("allowed_splits")
    if partitions != list(REQUIRED_PARTITIONS):
        findings.error(
            "CONTRACT",
            "PARTITIONS_INVALID",
            "The eight source-group partitions must use the frozen order and names.",
            actual=partitions,
        )
    if not isinstance(admin_splits, list) or set(admin_splits) != {"unassigned", "excluded"}:
        findings.error("CONTRACT", "ADMIN_SPLITS_INVALID", "Administrative splits must be unassigned and excluded.")
    if not isinstance(allowed_splits, list) or set(allowed_splits) != set(REQUIRED_PARTITIONS) | {
        "unassigned",
        "excluded",
    }:
        findings.error("CONTRACT", "ALLOWED_SPLITS_INVALID", "Allowed splits must equal eight partitions plus admin splits.")
    if rules.get("sealed_evaluation_splits") != [
        "natural_test",
        "printed_test",
        "print_demo",
        "site_acceptance",
    ]:
        findings.error(
            "CONTRACT",
            "SEALED_SPLITS_INVALID",
            "The four frozen evaluation partitions must be sealed in canonical order.",
        )
    if rules.get("permanent_holdout_split") != "print_demo":
        findings.error("CONTRACT", "PERMANENT_SPLIT_INVALID", "Only print_demo is the permanent holdout split.")

    roles = rules.get("domain_roles")
    matrix = rules.get("domain_split_matrix")
    allowed_domains = rules.get("allowed_domains")
    required_roles = {"natural", "printed_train", "printed_test", "print_demo", "site_acceptance"}
    if not isinstance(roles, dict) or set(roles) != required_roles:
        findings.error("CONTRACT", "DOMAIN_ROLES_INVALID", "Domain roles must define all five frozen roles.")
    role_domains = {
        str(domain)
        for values in roles.values()
        if isinstance(values, list)
        for domain in values
    } if isinstance(roles, dict) else set()
    if not isinstance(allowed_domains, list) or set(allowed_domains) != role_domains:
        findings.error("CONTRACT", "ALLOWED_DOMAINS_INVALID", "allowed_domains must exactly equal the domain-role union.")
    if not isinstance(matrix, dict) or set(matrix) != role_domains:
        findings.error("CONTRACT", "DOMAIN_MATRIX_INVALID", "Every allowed domain needs exactly one split-matrix entry.")
    elif any(not isinstance(value, list) or not set(value).issubset(set(allowed_splits or [])) for value in matrix.values()):
        findings.error("CONTRACT", "DOMAIN_MATRIX_SPLIT_INVALID", "Domain matrix contains an unknown split.")

    coverage = rules.get("partition_class_coverage")
    coverage_partitions = {"train", "validation", "decision_calibration", "conversion_golden", "natural_test"}
    if not isinstance(coverage, dict) or set(coverage) != coverage_partitions:
        findings.error(
            "CONTRACT",
            "PARTITION_CLASS_MATRIX_INVALID",
            "partition_class_coverage must cover the five natural/model partitions.",
        )
    elif any(value != list(class_order) for value in coverage.values()):
        findings.error(
            "CONTRACT",
            "PARTITION_CLASS_ORDER_INVALID",
            "Every applicable partition must require all four classes in frozen order.",
        )

    asset_roles = rules.get("asset_roles")
    role_matrix = rules.get("asset_role_domain_matrix")
    required_asset_roles = {"source", "crop", "augmentation", "print_capture", "local_capture"}
    if not isinstance(asset_roles, list) or set(asset_roles) != required_asset_roles:
        findings.error("CONTRACT", "ASSET_ROLES_INVALID", "The five frozen asset roles are required.")
    if not isinstance(role_matrix, dict) or set(role_matrix) != required_asset_roles:
        findings.error("CONTRACT", "ASSET_ROLE_MATRIX_INVALID", "Every asset role needs a domain matrix entry.")
    elif any(not isinstance(value, list) or not set(value).issubset(role_domains) for value in role_matrix.values()):
        findings.error("CONTRACT", "ASSET_ROLE_DOMAIN_INVALID", "Asset-role matrix contains an unknown domain.")

    if rules.get("lineage_contract") != {
        "source_provider_id_field": "source_provider_id",
        "origin_download_url_field": "download_url",
        "origin_attribution_fields": [
            "source_provider_id",
            "source_provider",
            "source_page",
            "download_url",
            "artist",
            "license",
            "license_url",
        ],
        "capture_operator_field": "capture_operator",
        "require_exact_group_inheritance": True,
    }:
        findings.error("CONTRACT", "LINEAGE_CONTRACT_INVALID", "Origin attribution inheritance contract is incomplete.")
    provider_registry = rules.get("source_provider_registry")
    if (
        not isinstance(provider_registry, dict)
        or not provider_registry
        or any(
            type(provider_id) is not str
            or not provider_id
            or not isinstance(provider_rule, dict)
            or set(provider_rule)
            != {
                "display_name",
                "source_page_kind",
                "allowed_source_hosts",
                "download_locator_kind",
                "allowed_download_hosts",
                "origin_id_kind",
            }
            or type(provider_rule.get("display_name")) is not str
            or not provider_rule.get("display_name", "").strip()
            or provider_rule.get("source_page_kind") not in {"https_url", "rootscope_local_urn"}
            or provider_rule.get("download_locator_kind") not in {"https_url", "rootscope_local_urn"}
            or provider_rule.get("origin_id_kind") not in {"positive_decimal", "rootscope_local_id"}
            or not isinstance(provider_rule.get("allowed_source_hosts"), list)
            or not isinstance(provider_rule.get("allowed_download_hosts"), list)
            or any(
                type(host) is not str or not host or host != host.casefold()
                for host in provider_rule.get("allowed_source_hosts", [])
                + provider_rule.get("allowed_download_hosts", [])
            )
            for provider_id, provider_rule in provider_registry.items()
        )
        or len(
            {
                _canonical_text_identity(value["display_name"])
                for value in provider_registry.values()
                if isinstance(value, dict) and type(value.get("display_name")) is str
            }
        )
        != len(provider_registry)
    ):
        findings.error(
            "CONTRACT",
            "SOURCE_PROVIDER_REGISTRY_INVALID",
            "A collision-free frozen source_provider_id registry is required.",
        )

    optics = rules.get("final_optics_evidence")
    if (
        not isinstance(optics, dict)
        or optics.get("capture_asset_roles")
        != {
            "print_capture": ["printed_train", "printed_test", "printed_demo_capture"],
            "local_capture": ["local_negative"],
        }
        or optics.get("capture_condition_id_pattern") != r"^[a-z0-9][a-z0-9_.:-]{2,63}$"
        or optics.get("optical_domain_root_pattern") != SHA256_RE.pattern
        or optics.get("require_one_consistent_root") is not True
    ):
        findings.error("CONTRACT", "FINAL_OPTICS_CONTRACT_INVALID", "Final-optics evidence contract is incomplete.")

    optical_receipt = rules.get("optical_domain_receipt")
    if (
        not isinstance(optical_receipt, dict)
        or optical_receipt.get("canonicalization") != "json_utf8_sort_keys_compact_sha256"
        or optical_receipt.get("receipt_schema_version") != "1.0.0"
        or optical_receipt.get("required_top_level_fields")
        != ["schema_version", "receipt_id", "signed_roles", "evidence_roots"]
        or optical_receipt.get("required_role_entry_fields")
        != ["member", "signed", "signer", "approval_evidence_sha256"]
        or optical_receipt.get("required_signed_roles")
        != {"hardware": "B", "mechanical": "C", "operations": "D"}
        or optical_receipt.get("require_distinct_signers") is not True
        or optical_receipt.get("require_distinct_approval_evidence") is not True
        or optical_receipt.get("approval_evidence_field") != "approval_evidence_sha256"
        or optical_receipt.get("required_evidence_roots")
        != ["uvc", "lighting", "paper", "printer", "geometry"]
    ):
        findings.error("CONTRACT", "OPTICAL_RECEIPT_CONTRACT_INVALID", "B/C/D optical-domain receipt contract is incomplete.")

    image_validation = rules.get("image_validation")
    if (
        not isinstance(image_validation, dict)
        or type(image_validation.get("minimum_width")) is not int
        or type(image_validation.get("minimum_height")) is not int
        or image_validation.get("minimum_width", 0) < 1
        or image_validation.get("minimum_height", 0) < 1
        or image_validation.get("require_decode") is not True
    ):
        findings.error("CONTRACT", "IMAGE_VALIDATION_INVALID", "Image decode and positive minimum dimensions are required.")

    perceptual = rules.get("perceptual_duplicate_audit")
    if (
        not isinstance(perceptual, dict)
        or perceptual.get("algorithm") != "dhash64"
        or type(perceptual.get("distance_threshold")) is not int
        or not 0 <= perceptual.get("distance_threshold", -1) <= 64
        or perceptual.get("native_v2_required") is not True
        or perceptual.get("compare_across_distinct_source_groups") is not True
    ):
        findings.error("CONTRACT", "PERCEPTUAL_AUDIT_INVALID", "Native-v2 dHash audit contract is required.")

    licenses = rules.get("allowed_licenses")
    if not isinstance(licenses, dict) or not licenses:
        findings.error("CONTRACT", "LICENSE_CONTRACT_INVALID", "Exact canonical license rules are required.")
    else:
        for license_name, license_rule in licenses.items():
            if (
                not isinstance(license_name, str)
                or not license_name
                or not isinstance(license_rule, dict)
                or type(license_rule.get("url_required")) is not bool
                or not isinstance(license_rule.get("canonical_url_pattern"), str)
            ):
                findings.error("CONTRACT", "LICENSE_RULE_INVALID", "Every exact license needs URL policy.", license=license_name)
                continue
            try:
                re.compile(license_rule["canonical_url_pattern"])
            except re.error:
                findings.error("CONTRACT", "LICENSE_URL_REGEX_INVALID", "License URL regex is invalid.", license=license_name)

    if rules.get("site_acceptance_unknown_policy") != {
        "requires_nonempty_scenario": True,
        "requires_independent_source_groups": True,
    }:
        findings.error(
            "CONTRACT",
            "SITE_UNKNOWN_POLICY_INVALID",
            "site_acceptance unknown rows must be independent groups with non-empty scenario labels.",
        )

    migration = rules.get("migration")
    if not isinstance(migration, dict):
        findings.error("CONTRACT", "MIGRATION_CONTRACT_MISSING", "The one-shot v1-to-v2 migration contract is required.")
    else:
        accepted = migration.get("accepted_manifest_sha256")
        if (
            migration.get("from_schema_version") != "1.0.0"
            or migration.get("to_schema_version") != "2.0.0"
            or migration.get("mode") != "allowlisted_manifest_sha256_projection"
            or not isinstance(accepted, list)
            or not accepted
            or any(not SHA256_RE.fullmatch(str(value)) for value in accepted)
        ):
            findings.error("CONTRACT", "MIGRATION_CONTRACT_INVALID", "Migration must be v1->v2 and SHA allowlisted.")

    minimums = rules.get("ready_minimums")
    if not isinstance(minimums, dict):
        findings.error("CONTRACT", "READY_MINIMUMS_MISSING", "ready_minimums must be an object.")
    else:
        natural_total = minimums.get("natural_unique_source_groups")
        natural_by_split = minimums.get("natural_source_group_minimums_by_split")
        coverage_keys = {"train", "validation", "decision_calibration", "conversion_golden", "natural_test"}
        class_keys = set(class_order)
        natural_shape_ok = (
            isinstance(natural_total, dict)
            and set(natural_total) == class_keys
            and all(type(value) is int and value >= 0 for value in natural_total.values())
            and isinstance(natural_by_split, dict)
            and set(natural_by_split) == coverage_keys
            and all(
                isinstance(per_class, dict)
                and set(per_class) == class_keys
                and all(type(value) is int and value >= 0 for value in per_class.values())
                for per_class in natural_by_split.values()
            )
        )
        if not natural_shape_ok:
            findings.error(
                "CONTRACT",
                "NATURAL_SPLIT_MINIMUMS_INVALID",
                "Natural/model minimums must define every split and class with non-negative integers.",
            )
        elif any(
            sum(natural_by_split[split][class_id] for split in coverage_keys) != natural_total[class_id]
            for class_id in class_keys
        ):
            findings.error(
                "CONTRACT",
                "NATURAL_MINIMUM_TOTAL_MISMATCH",
                "Per-split natural source-group minimums must sum to each frozen class total.",
            )

        printed_total = minimums.get("printed_train_source_groups_per_target_class")
        printed_by_split = minimums.get("printed_train_source_group_minimums_by_split")
        printed_split_keys = {"train", "validation", "decision_calibration", "conversion_golden"}
        target_keys = set(TARGET_CLASSES)
        printed_shape_ok = (
            type(printed_total) is int
            and printed_total >= 0
            and isinstance(printed_by_split, dict)
            and set(printed_by_split) == printed_split_keys
            and all(
                isinstance(per_class, dict)
                and set(per_class) == target_keys
                and all(type(value) is int and value >= 0 for value in per_class.values())
                for per_class in printed_by_split.values()
            )
        )
        if not printed_shape_ok:
            findings.error(
                "CONTRACT",
                "PRINTED_TRAIN_SPLIT_MINIMUMS_INVALID",
                "Printed-train minimums must define every model split and target class.",
            )
        elif any(
            sum(printed_by_split[split][class_id] for split in printed_split_keys) != printed_total
            for class_id in target_keys
        ):
            findings.error(
                "CONTRACT",
                "PRINTED_TRAIN_MINIMUM_TOTAL_MISMATCH",
                "Per-split printed-train source-group minimums must sum to the aggregate class minimum.",
            )

        golden_total = minimums.get("conversion_golden_qualifying_assets_total")
        golden_per_class = minimums.get("conversion_golden_qualifying_assets_per_class")
        if (
            type(golden_total) is not int
            or golden_total < 0
            or type(golden_per_class) is not int
            or golden_per_class < 0
            or golden_total < golden_per_class * len(class_order)
        ):
            findings.error(
                "CONTRACT",
                "CONVERSION_GOLDEN_MINIMUMS_INVALID",
                "conversion_golden needs a coherent total and per-class independent-source minimum.",
            )

    return [str(value) for value in class_order], rules


def _is_nonempty(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _field_type_valid(field: str, value: Any) -> bool:
    if field in STRING_FIELDS:
        return type(value) is str
    if field in NULLABLE_STRING_FIELDS:
        return value is None or type(value) is str
    if field in BOOLEAN_FIELDS:
        return type(value) is bool
    if field == "origin_pageid":
        return (type(value) is int and value >= 0) or (type(value) is str and bool(value.strip()))
    return False


def _dhash64(image: Image.Image) -> str:
    resampling = getattr(Image, "Resampling", Image)
    gray = image.convert("L").resize((9, 8), resampling.LANCZOS)
    flattened = getattr(gray, "get_flattened_data", None)
    pixels = list(flattened() if flattened is not None else gray.getdata())
    value = 0
    for y in range(8):
        offset = y * 9
        for x in range(8):
            value = (value << 1) | int(pixels[offset + x] > pixels[offset + x + 1])
    return f"{value:016x}"


def _hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_text_identity(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _canonical_signer_identity(value: Any) -> str | None:
    if type(value) is not str:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        return None
    canonical = " ".join(normalized.split()).casefold()
    return canonical or None


def _canonical_source_page(value: str, source_page_kind: str, allowed_hosts: set[str]) -> str | None:
    """Canonicalize a frozen external URL or RootScope local evidence URN."""

    normalized = unicodedata.normalize("NFKC", value).strip()
    if source_page_kind == "rootscope_local_urn":
        canonical_urn = normalized.casefold()
        return canonical_urn if re.fullmatch(r"urn:rootscope:local:[a-z0-9][a-z0-9_.:-]{2,127}", canonical_urn) else None
    if source_page_kind != "https_url":
        return None
    try:
        parts = urlsplit(normalized)
        scheme = parts.scheme.casefold()
        if scheme != "https" or not parts.hostname or parts.username or parts.password:
            return None
        host = parts.hostname.encode("idna").decode("ascii").casefold()
        if host not in allowed_hosts:
            return None
        port = parts.port
        if port is not None and not (scheme == "https" and port == 443):
            host = f"{host}:{port}"
        path = parts.path or "/"
        if path != "/":
            path = path.rstrip("/") or "/"
        return urlunsplit((scheme, host, path, parts.query, ""))
    except (UnicodeError, ValueError):
        return None


def _canonical_origin_id(value: Any, origin_id_kind: str) -> str | None:
    if origin_id_kind == "positive_decimal":
        if type(value) is int and value > 0:
            return str(value)
        if type(value) is str and re.fullmatch(r"[1-9][0-9]*", value.strip()):
            return str(int(value.strip()))
        return None
    if origin_id_kind == "rootscope_local_id" and type(value) is str:
        normalized = _canonical_text_identity(value)
        return normalized if re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{2,127}", normalized) else None
    return None


def _apply_manifest_migration(
    rows: list[dict[str, Any]],
    manifest_sha256: str | None,
    rules: dict[str, Any],
    findings: Findings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_version = str(rules.get("record_schema_version", "2.0.0"))
    legacy_rows = [row for row in rows if row.get("record_schema_version") != target_version]
    base = {
        "migration_id": rules.get("migration", {}).get("migration_id"),
        "mode": "native_v2",
        "applied": False,
        "source_manifest_sha256": manifest_sha256,
        "source_row_count": len(rows),
        "projected_row_count": len(rows),
        "persisted": True,
        "transformation_counts": {},
    }
    if not legacy_rows:
        return rows, base

    migration = rules.get("migration", {})
    accepted = set(migration.get("accepted_manifest_sha256", [])) if isinstance(migration, dict) else set()
    if manifest_sha256 not in accepted:
        findings.error(
            "MANIFEST_MIGRATION",
            "UNREGISTERED_LEGACY_MANIFEST",
            "Rows missing native v2 fields may only come from the one SHA-registered seed manifest.",
            manifest_sha256=manifest_sha256,
            legacy_row_count=len(legacy_rows),
        )
        return rows, {**base, "mode": "rejected_legacy", "persisted": False, "legacy_row_count": len(legacy_rows)}

    split_aliases = migration.get("split_aliases", {})
    domain_aliases = migration.get("domain_aliases", {})
    defaults = migration.get("field_defaults", {})
    if not all(isinstance(value, dict) for value in (split_aliases, domain_aliases, defaults)):
        findings.error("MANIFEST_MIGRATION", "MIGRATION_RULES_INVALID", "Migration aliases/defaults must be objects.")
        return rows, {**base, "mode": "failed_projection", "persisted": False}

    canonical: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    holdout_split = str(rules.get("permanent_holdout_split", "print_demo"))
    sealed_splits = set(rules.get("sealed_evaluation_splits", []))
    for row in rows:
        version = row.get("record_schema_version")
        if version not in (None, "", migration.get("from_schema_version"), target_version):
            findings.error(
                "MANIFEST_MIGRATION",
                "UNKNOWN_ROW_SCHEMA",
                "Registered legacy manifest contains an unsupported row schema.",
                line=row.get("_line"),
                actual=version,
            )
            canonical.append(dict(row))
            continue
        migrated = dict(row)
        old_split = migrated.get("split")
        new_split = split_aliases.get(old_split, old_split)
        if new_split != old_split:
            counts[f"split:{old_split}->{new_split}"] += 1
        migrated["split"] = new_split
        old_domain = migrated.get("domain")
        new_domain = domain_aliases.get(old_domain, old_domain)
        if new_domain != old_domain:
            counts[f"domain:{old_domain}->{new_domain}"] += 1
        migrated["domain"] = new_domain
        for field, value in defaults.items():
            if field not in migrated:
                migrated[field] = value
                counts[f"default:{field}"] += 1
        migrated["record_schema_version"] = target_version
        migrated["permanent_holdout"] = new_split == holdout_split
        migrated["sealed"] = new_split in sealed_splits
        migrated["asset_id"] = f"legacy:{migrated.get('pageid')}"
        migrated["asset_sha256"] = migrated.get("download_sha256")
        migrated["origin_pageid"] = migrated.get("pageid")
        migrated["origin_sha256"] = migrated.get("download_sha256")
        migrated["asset_role"] = "source"
        migrated["_migrated_from_v1"] = True
        for field in ("asset_id", "asset_sha256", "origin_pageid", "origin_sha256", "asset_role", "sealed"):
            counts[f"derived:{field}"] += 1
        canonical.append(migrated)

    return canonical, {
        **base,
        "mode": migration.get("mode"),
        "applied": True,
        "from_schema_version": migration.get("from_schema_version"),
        "to_schema_version": migration.get("to_schema_version"),
        "persisted": False,
        "legacy_row_count": len(legacy_rows),
        "transformation_counts": dict(sorted(counts.items())),
        "note": "Canonical v2 projection is audit-only; persist native v2 rows before DATA_LOCKED.",
    }


def _validate_rows(
    rows: list[dict[str, Any]],
    dataset_dir: Path,
    expected_classes: list[str],
    rules: dict[str, Any],
    findings: Findings,
) -> dict[str, Any]:
    dataset_root = dataset_dir.resolve()
    expected_set = set(expected_classes)
    expected_version = str(rules.get("record_schema_version", "2.0.0"))
    allowed_splits = set(rules.get("allowed_splits", []))
    excluded_split = str(rules.get("excluded_split", "excluded"))
    holdout_split = str(rules.get("permanent_holdout_split", "print_demo"))
    sealed_splits = set(rules.get("sealed_evaluation_splits", []))
    allowed_domains = set(rules.get("allowed_domains", []))
    domain_matrix = {str(key): set(value) for key, value in rules.get("domain_split_matrix", {}).items()}
    domain_roles = rules.get("domain_roles", {})
    print_demo_domains = set(domain_roles.get("print_demo", []))
    approved_statuses = set(rules.get("approved_review_statuses", []))
    rejected_statuses = set(rules.get("rejected_review_statuses", []))
    unknown_scenarios = set(rules.get("unknown_scenarios", []))
    asset_roles = set(rules.get("asset_roles", []))
    asset_role_matrix = {str(key): set(value) for key, value in rules.get("asset_role_domain_matrix", {}).items()}
    capture_role_domains = {
        str(key): set(value)
        for key, value in rules.get("final_optics_evidence", {}).get("capture_asset_roles", {}).items()
    }
    licenses = rules.get("allowed_licenses", {})
    provider_registry = rules.get("source_provider_registry", {})
    image_rule = rules.get("image_validation", {})
    minimum_width = int(image_rule.get("minimum_width", 1))
    minimum_height = int(image_rule.get("minimum_height", 1))
    perceptual_rule = rules.get("perceptual_duplicate_audit", {})
    dhash_threshold = int(perceptual_rule.get("distance_threshold", 0))
    ptq_split = str(rules.get("ptq_calibration", {}).get("allowed_split", "train"))

    class_counts: Counter[str] = Counter()
    active_class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    ptq_class_counts: Counter[str] = Counter()
    asset_ids: dict[str, list[int]] = defaultdict(list)
    asset_hashes: dict[str, list[int]] = defaultdict(list)
    capture_ids: dict[str, list[int]] = defaultdict(list)
    origin_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    origin_pageid_bindings: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    origin_sha_bindings: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    source_page_bindings: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    group_classes: dict[str, set[str]] = defaultdict(set)
    group_domains: dict[str, set[str]] = defaultdict(set)
    group_origins: dict[str, set[tuple[str, str]]] = defaultdict(set)
    group_unknown_scenarios: dict[str, set[str | None]] = defaultdict(set)
    group_attributions: dict[str, set[tuple[str, str, str, str, str, str, str | None]]] = defaultdict(set)
    group_capture_conditions: dict[str, list[str]] = defaultdict(list)
    group_source_count: Counter[str] = Counter()
    optical_roots: set[str] = set()
    dhash_records: list[dict[str, Any]] = []
    native_v2_row_count = 0
    native_v2_dhash_count = 0
    quality_capture_count = 0

    for row in rows:
        line = int(row.get("_line", 0))
        is_native_v2 = row.get("_migrated_from_v1") is not True
        if is_native_v2:
            native_v2_row_count += 1
            unknown_fields = sorted(set(row) - set(REQUIRED_FIELDS) - {"_line"})
            if unknown_fields:
                findings.error(
                    "FIELDS",
                    "NATIVE_V2_UNKNOWN_FIELDS",
                    "Native v2 rows may contain only the frozen schema fields.",
                    line=line,
                    fields=unknown_fields,
                )

        for field in REQUIRED_FIELDS:
            if field not in row:
                findings.error("FIELDS", "FIELD_MISSING", f"Missing required field {field}.", line=line, field=field)
                continue
            value = row.get(field)
            if not _field_type_valid(field, value):
                findings.error(
                    "FIELDS",
                    "FIELD_TYPE_INVALID",
                    f"Required field {field} has the wrong JSON type.",
                    line=line,
                    field=field,
                    actual_type=type(value).__name__,
                )
                continue
            if field in STRING_FIELDS and not value.strip():
                findings.error("FIELDS", "FIELD_EMPTY", f"Required string field {field} is empty.", line=line, field=field)

        class_value = row.get("class_id")
        split_value = row.get("split")
        domain_value = row.get("domain")
        review_value = row.get("review_status")
        group_value = row.get("source_group")
        role_value = row.get("asset_role")
        class_id = class_value if type(class_value) is str else "<invalid>"
        split = split_value if type(split_value) is str else "<invalid>"
        domain = domain_value if type(domain_value) is str else "<invalid>"
        review = review_value if type(review_value) is str else "<invalid>"
        source_group = group_value if type(group_value) is str and group_value.strip() else ""
        asset_role = role_value if type(role_value) is str else "<invalid>"

        class_counts[class_id] += 1
        split_counts[split] += 1
        domain_counts[domain] += 1
        review_counts[review] += 1
        if review not in rejected_statuses:
            active_class_counts[class_id] += 1

        if row.get("record_schema_version") != expected_version:
            findings.error(
                "FIELDS",
                "RECORD_SCHEMA_VERSION",
                "record_schema_version does not match the v2 contract.",
                line=line,
                actual=row.get("record_schema_version"),
            )
        if class_id not in expected_set:
            findings.error("CLASS_COVERAGE", "UNEXPECTED_CLASS", f"Unexpected class_id {class_id!r}.", line=line)
        if split not in allowed_splits:
            findings.error("FIELDS", "INVALID_SPLIT", f"Split {split!r} is not allowed.", line=line)
        if domain not in allowed_domains:
            findings.error("DOMAIN_PARTITION", "INVALID_DOMAIN", f"Domain {domain!r} is not allowed.", line=line)
        elif split not in domain_matrix.get(domain, set()):
            findings.error(
                "DOMAIN_PARTITION",
                "DOMAIN_SPLIT_MISMATCH",
                "Domain is not permitted in this source-group partition.",
                line=line,
                domain=domain,
                split=split,
            )
        if asset_role not in asset_roles:
            findings.error("LINEAGE", "ASSET_ROLE_INVALID", "asset_role is outside the frozen enum.", line=line, role=asset_role)
        elif domain not in asset_role_matrix.get(asset_role, set()):
            findings.error(
                "LINEAGE",
                "ASSET_ROLE_DOMAIN_MISMATCH",
                "asset_role is not allowed in this domain.",
                line=line,
                role=asset_role,
                domain=domain,
            )

        unknown_scenario = row.get("unknown_scenario")
        if class_id == "unknown" and _is_nonempty(unknown_scenario) and unknown_scenario not in unknown_scenarios:
            findings.error(
                "FIELDS",
                "UNKNOWN_SCENARIO_INVALID",
                "unknown_scenario is outside the frozen taxonomy.",
                line=line,
                value=unknown_scenario,
            )
        if class_id != "unknown" and _is_nonempty(unknown_scenario):
            findings.error("FIELDS", "UNKNOWN_SCENARIO_ON_TARGET", "Only unknown rows may define unknown_scenario.", line=line)
        if split == "site_acceptance" and class_id == "unknown" and (
            type(unknown_scenario) is not str or unknown_scenario not in unknown_scenarios
        ):
            findings.error(
                "FIELDS",
                "SITE_UNKNOWN_SCENARIO_REQUIRED",
                "Every site_acceptance unknown scene needs one non-empty frozen scenario label.",
                line=line,
            )

        asset_id_value = row.get("asset_id")
        asset_sha_value = row.get("asset_sha256")
        origin_pageid_value = row.get("origin_pageid")
        origin_sha_value = row.get("origin_sha256")
        asset_id = asset_id_value if type(asset_id_value) is str else ""
        asset_sha = asset_sha_value if type(asset_sha_value) is str else ""
        origin_sha = origin_sha_value if type(origin_sha_value) is str else ""
        if asset_id:
            asset_ids[asset_id].append(line)
        if asset_sha:
            asset_hashes[asset_sha].append(line)
        if not SHA256_RE.fullmatch(asset_sha):
            findings.error("LINEAGE", "ASSET_SHA_FORMAT", "asset_sha256 must be 64 lowercase hex characters.", line=line)
        if not SHA256_RE.fullmatch(origin_sha):
            findings.error("LINEAGE", "ORIGIN_SHA_FORMAT", "origin_sha256 must be 64 lowercase hex characters.", line=line)

        origin_key: tuple[str, str] | None = None
        source_provider_id_value = row.get("source_provider_id")
        source_provider_value = row.get("source_provider")
        source_page_value = row.get("source_page")
        provider_rule = (
            provider_registry.get(source_provider_id_value)
            if type(source_provider_id_value) is str and isinstance(provider_registry, dict)
            else None
        )
        if not isinstance(provider_rule, dict):
            findings.error(
                "LINEAGE",
                "SOURCE_PROVIDER_ID_INVALID",
                "source_provider_id is outside the frozen registry.",
                line=line,
                source_provider_id=source_provider_id_value,
            )
        elif (
            type(source_provider_value) is not str
            or source_provider_value != provider_rule.get("display_name")
            or _canonical_text_identity(source_provider_value)
            != _canonical_text_identity(str(provider_rule.get("display_name", "")))
        ):
            findings.error(
                "LINEAGE",
                "SOURCE_PROVIDER_NOT_CANONICAL",
                "source_provider must exactly match the display name bound to source_provider_id.",
                line=line,
                source_provider_id=source_provider_id_value,
                source_provider=source_provider_value,
            )
        canonical_source_page = (
            _canonical_source_page(
                source_page_value,
                str(provider_rule.get("source_page_kind")),
                set(provider_rule.get("allowed_source_hosts", [])),
            )
            if isinstance(provider_rule, dict) and type(source_page_value) is str
            else None
        )
        if canonical_source_page is None:
            findings.error(
                "LINEAGE",
                "SOURCE_PAGE_INVALID",
                "source_page must be a canonicalizable locator allowed by its frozen provider.",
                line=line,
                source_provider_id=source_provider_id_value,
                source_page=source_page_value,
            )
        download_url_value = row.get("download_url")
        canonical_download_locator = (
            _canonical_source_page(
                download_url_value,
                str(provider_rule.get("download_locator_kind")),
                set(provider_rule.get("allowed_download_hosts", [])),
            )
            if isinstance(provider_rule, dict) and type(download_url_value) is str
            else None
        )
        if canonical_download_locator is None:
            findings.error(
                "LINEAGE",
                "DOWNLOAD_LOCATOR_INVALID",
                "download_url must be a canonicalizable locator allowed by its frozen provider.",
                line=line,
                source_provider_id=source_provider_id_value,
                download_url=download_url_value,
            )
        canonical_origin_id = (
            _canonical_origin_id(origin_pageid_value, str(provider_rule.get("origin_id_kind")))
            if isinstance(provider_rule, dict)
            else None
        )
        if canonical_origin_id is None:
            findings.error(
                "LINEAGE",
                "ORIGIN_ID_INVALID_FOR_PROVIDER",
                "origin_pageid does not satisfy the provider-specific frozen identity grammar.",
                line=line,
                source_provider_id=source_provider_id_value,
                origin_pageid=origin_pageid_value,
            )
        if (
            canonical_origin_id is not None
            and SHA256_RE.fullmatch(origin_sha)
            and isinstance(provider_rule, dict)
            and type(source_provider_id_value) is str
        ):
            canonical_provider_pageid = (
                f"{_canonical_text_identity(source_provider_id_value)}|"
                f"{canonical_origin_id}"
            )
            origin_key = (canonical_provider_pageid, origin_sha)
        if source_group:
            group_splits[source_group].add(split)
            group_classes[source_group].add(class_id)
            group_domains[source_group].add(domain)
            if origin_key is not None:
                group_origins[source_group].add(origin_key)
                origin_groups[origin_key].add(source_group)
                origin_pageid_bindings[origin_key[0]].add((origin_sha, source_group, split))
                origin_sha_bindings[origin_sha].add((origin_key[0], source_group, split))
            if canonical_source_page is not None:
                source_page_bindings[canonical_source_page].add((origin_sha, source_group, split))
            if class_id == "unknown":
                group_unknown_scenarios[source_group].add(
                    unknown_scenario if type(unknown_scenario) is str and unknown_scenario else None
                )
            if asset_role == "source":
                group_source_count[source_group] += 1
            attribution_values = (
                row.get("source_provider_id"),
                row.get("source_provider"),
                row.get("source_page"),
                row.get("download_url"),
                row.get("artist"),
                row.get("license"),
                row.get("license_url"),
            )
            if (
                all(type(value) is str for value in attribution_values[:6])
                and (attribution_values[6] is None or type(attribution_values[6]) is str)
            ):
                group_attributions[source_group].add(attribution_values)
        if asset_role == "source" and SHA256_RE.fullmatch(asset_sha) and SHA256_RE.fullmatch(origin_sha) and asset_sha != origin_sha:
            findings.error(
                "LINEAGE",
                "SOURCE_ORIGIN_SHA_MISMATCH",
                "A source asset must bind its own bytes as origin_sha256.",
                line=line,
                asset_sha256=asset_sha,
                origin_sha256=origin_sha,
            )

        filename = row.get("filename")
        candidate: Path | None = None
        if type(filename) is not str or not filename.strip():
            findings.error("FILES", "FILENAME_INVALID", "filename must be a non-empty JSON string.", line=line)
        else:
            try:
                raw_path = Path(filename)
                if raw_path.is_absolute():
                    raise ValueError("absolute path")
                candidate = (dataset_root / raw_path).resolve()
                candidate.relative_to(dataset_root)
            except (OSError, ValueError):
                candidate = None
                findings.error("FILES", "PATH_ESCAPE", "Dataset filename is invalid or escapes dataset root.", line=line, filename=filename)

        if candidate is not None:
            if not candidate.is_file():
                findings.error("FILES", "FILE_MISSING", "Referenced image does not exist.", line=line, filename=filename)
            else:
                actual_sha = _sha256_file(candidate)
                if SHA256_RE.fullmatch(asset_sha) and actual_sha != asset_sha:
                    findings.error(
                        "FILES",
                        "ASSET_SHA_MISMATCH",
                        "Image bytes do not match asset_sha256.",
                        line=line,
                        expected=asset_sha,
                        actual=actual_sha,
                    )
                try:
                    with Image.open(candidate) as image:
                        image.load()
                        width, height = image.size
                        dhash = _dhash64(image)
                    if width < minimum_width or height < minimum_height:
                        findings.error(
                            "IMAGE_DECODE",
                            "IMAGE_TOO_SMALL",
                            "Decoded image is below the frozen minimum dimensions.",
                            line=line,
                            width=width,
                            height=height,
                            minimum_width=minimum_width,
                            minimum_height=minimum_height,
                        )
                    if source_group and asset_id:
                        dhash_records.append(
                            {
                                "line": line,
                                "asset_id": asset_id,
                                "source_group": source_group,
                                "dhash64": dhash,
                                "native_v2": is_native_v2,
                            }
                        )
                        if is_native_v2:
                            native_v2_dhash_count += 1
                        row["_dhash64"] = dhash
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
                    findings.error(
                        "IMAGE_DECODE",
                        "IMAGE_DECODE_FAILED",
                        "Image cannot be fully decoded.",
                        line=line,
                        filename=filename,
                        error=type(exc).__name__,
                    )

        license_name_value = row.get("license")
        license_url_value = row.get("license_url")
        if type(license_name_value) is str:
            license_rule = licenses.get(license_name_value)
            if not isinstance(license_rule, dict):
                findings.error(
                    "LICENSE",
                    "LICENSE_NOT_ALLOWED",
                    "License name is not an exact canonical allowlist entry.",
                    line=line,
                    license=license_name_value,
                )
            elif license_url_value is None or type(license_url_value) is str:
                normalized_url = "" if license_url_value is None else license_url_value.strip()
                canonical_candidate = (
                    "https://" + normalized_url[len("http://") :]
                    if not is_native_v2 and normalized_url.startswith("http://")
                    else normalized_url
                )
                if license_rule.get("url_required") is True and not normalized_url:
                    findings.error("LICENSE", "LICENSE_URL_MISSING", "A canonical license URL is required.", line=line)
                elif normalized_url and re.fullmatch(str(license_rule.get("canonical_url_pattern", r"(?!)")), canonical_candidate) is None:
                    findings.error(
                        "LICENSE",
                        "LICENSE_URL_NONCANONICAL",
                        "License URL does not match the exact canonical license/version.",
                        line=line,
                        license=license_name_value,
                        license_url=normalized_url,
                    )
        if type(row.get("artist")) is not str or not row.get("artist", "").strip() or type(row.get("source_page")) is not str or not row.get("source_page", "").strip():
            findings.error("LICENSE", "ATTRIBUTION_INCOMPLETE", "artist and source_page must be non-empty strings.", line=line)

        is_capture = asset_role in capture_role_domains
        optical_root = row.get("optical_domain_root")
        capture_id = row.get("capture_id")
        capture_operator = row.get("capture_operator")
        capture_condition_id = row.get("capture_condition_id")
        if is_capture:
            if domain not in capture_role_domains.get(asset_role, set()):
                findings.error(
                    "FINAL_OPTICS",
                    "CAPTURE_ROLE_DOMAIN_MISMATCH",
                    "Final-optics capture role is not allowed in this domain.",
                    line=line,
                    role=asset_role,
                    domain=domain,
                )
            if type(optical_root) is not str or SHA256_RE.fullmatch(optical_root) is None:
                findings.error(
                    "FINAL_OPTICS",
                    "OPTICAL_ROOT_INVALID",
                    "Every final-optics capture needs a 64-hex optical_domain_root.",
                    line=line,
                )
            else:
                optical_roots.add(optical_root)
            if type(capture_id) is not str or not capture_id.strip():
                findings.error("FINAL_OPTICS", "CAPTURE_ID_INVALID", "Every final-optics capture needs capture_id.", line=line)
            else:
                capture_ids[capture_id].append(line)
            if type(capture_operator) is not str or not capture_operator.strip():
                findings.error(
                    "FINAL_OPTICS",
                    "CAPTURE_OPERATOR_INVALID",
                    "Every final-optics capture needs a non-empty capture_operator distinct from origin attribution.",
                    line=line,
                )
            condition_pattern = str(
                rules.get("final_optics_evidence", {}).get("capture_condition_id_pattern", r"(?!)")
            )
            if type(capture_condition_id) is not str or re.fullmatch(condition_pattern, capture_condition_id) is None:
                findings.error(
                    "FINAL_OPTICS",
                    "CAPTURE_CONDITION_ID_INVALID",
                    "Every final-optics capture needs a valid capture_condition_id.",
                    line=line,
                )
            elif source_group:
                group_capture_conditions[source_group].append(capture_condition_id)
            if row.get("capture_quality_pass") is True:
                quality_capture_count += 1
        elif (
            optical_root is not None
            or capture_id is not None
            or capture_operator is not None
            or capture_condition_id is not None
            or row.get("capture_quality_pass") is not False
        ):
            findings.error(
                "FINAL_OPTICS",
                "OPTICAL_EVIDENCE_ON_NON_CAPTURE",
                "Non-capture assets must use null optical fields and capture_quality_pass=false.",
                line=line,
                role=asset_role,
            )

        if row.get("ptq_calibration") is True:
            ptq_class_counts[class_id] += 1
            if split != ptq_split:
                findings.error("PTQ_TRAIN_ONLY", "PTQ_OUTSIDE_TRAIN", "PTQ calibration rows must remain in train.", line=line, split=split)
            if review not in approved_statuses:
                findings.error(
                    "PTQ_TRAIN_ONLY",
                    "PTQ_NOT_APPROVED",
                    "PTQ calibration rows require approved visual/license review.",
                    line=line,
                    review_status=review,
                )
            if row.get("permanent_holdout") is True:
                findings.error("PTQ_TRAIN_ONLY", "PTQ_IS_HOLDOUT", "Permanent holdout rows cannot enter PTQ.", line=line)

        requires_seal = split in sealed_splits
        if requires_seal and row.get("sealed") is not True:
            findings.error(
                "PERMANENT_HOLDOUT",
                "SEALED_REQUIRED",
                "Every row in a frozen evaluation split must set sealed=true.",
                line=line,
                split=split,
            )
        if not requires_seal and row.get("sealed") is True:
            findings.error(
                "PERMANENT_HOLDOUT",
                "SEALED_OUTSIDE_PROTECTED_SPLIT",
                "Only contract-declared sealed evaluation splits may set sealed=true.",
                line=line,
                split=split,
            )
        if split == holdout_split:
            if row.get("permanent_holdout") is not True:
                findings.error("PERMANENT_HOLDOUT", "PRINT_DEMO_NOT_PERMANENT", "Every print_demo row must be permanent.", line=line)
            if domain not in print_demo_domains:
                findings.error("PERMANENT_HOLDOUT", "PRINT_DEMO_DOMAIN_INVALID", "print_demo row has the wrong domain.", line=line, domain=domain)
        elif row.get("permanent_holdout") is True:
            findings.error(
                "PERMANENT_HOLDOUT",
                "PERMANENT_FLAG_OUTSIDE_PRINT_DEMO",
                "Only print_demo may set permanent_holdout=true.",
                line=line,
                split=split,
            )

        if review in rejected_statuses:
            if split != excluded_split:
                findings.error("REJECTED_EXCLUDED", "REJECTED_NOT_EXCLUDED", "Rejected row must use split excluded.", line=line, split=split)
            if row.get("print_eligible") is not False:
                findings.error("REJECTED_EXCLUDED", "REJECTED_PRINT_ELIGIBLE", "Rejected row must not be printable.", line=line)
            if (
                row.get("ptq_calibration") is not False
                or row.get("permanent_holdout") is not False
                or row.get("sealed") is not False
                or row.get("capture_quality_pass") is not False
            ):
                findings.error("REJECTED_EXCLUDED", "REJECTED_PRIVILEGED", "Rejected row retains a privileged flag.", line=line)

    for class_id in expected_classes:
        if active_class_counts[class_id] == 0:
            findings.error(
                "CLASS_COVERAGE",
                "CLASS_WITHOUT_ACTIVE_SAMPLE",
                f"Class {class_id!r} has no non-rejected sample.",
                class_id=class_id,
            )

    for asset_id, lines in asset_ids.items():
        if len(lines) > 1:
            findings.error("DUPLICATES", "DUPLICATE_ASSET_ID", "asset_id must be unique per row.", asset_id=asset_id, lines=lines)
    for asset_sha, lines in asset_hashes.items():
        if len(lines) > 1:
            findings.error("DUPLICATES", "DUPLICATE_ASSET_SHA256", "asset_sha256 must be unique per row.", sha256=asset_sha, lines=lines)
    for capture_id, lines in capture_ids.items():
        if len(lines) > 1:
            findings.error("DUPLICATES", "DUPLICATE_CAPTURE_ID", "capture_id must be unique.", capture_id=capture_id, lines=lines)

    for source_group, splits in group_splits.items():
        if len(splits) != 1:
            findings.error(
                "SOURCE_GROUP_PARTITION",
                "SOURCE_GROUP_PARTITION_LEAKAGE",
                "source_group crosses partition boundaries.",
                source_group=source_group,
                splits=sorted(splits),
            )
        classes = group_classes[source_group]
        if len(classes) != 1:
            findings.error(
                "SOURCE_GROUP_PARTITION",
                "SOURCE_GROUP_CLASS_LEAKAGE",
                "source_group maps to more than one class.",
                source_group=source_group,
                classes=sorted(classes),
            )
        origins = group_origins[source_group]
        if len(origins) != 1:
            findings.error(
                "LINEAGE",
                "SOURCE_GROUP_ORIGIN_LEAKAGE",
                "source_group must bind exactly one origin_pageid/origin_sha256 pair.",
                source_group=source_group,
                origin_count=len(origins),
            )
        if group_source_count[source_group] != 1:
            findings.error(
                "LINEAGE",
                "SOURCE_ASSET_COUNT",
                "source_group must contain exactly one asset_role=source row.",
                source_group=source_group,
                actual=group_source_count[source_group],
            )
        scenarios = group_unknown_scenarios[source_group]
        if len(scenarios) > 1:
            findings.error(
                "LINEAGE",
                "SOURCE_GROUP_UNKNOWN_SCENARIO_LEAKAGE",
                "One unknown source_group cannot represent multiple scenario labels.",
                source_group=source_group,
                scenarios=sorted(scenarios, key=lambda value: "" if value is None else value),
            )
        attributions = group_attributions[source_group]
        if len(attributions) != 1:
            findings.error(
                "LINEAGE",
                "SOURCE_GROUP_ATTRIBUTION_LEAKAGE",
                "All derivatives and captures must inherit the source origin attribution tuple exactly.",
                source_group=source_group,
                attribution_count=len(attributions),
            )
        conditions = group_capture_conditions[source_group]
        if len(conditions) != len(set(conditions)):
            findings.error(
                "FINAL_OPTICS",
                "CAPTURE_CONDITION_DUPLICATE",
                "capture_condition_id must be unique within one source_group.",
                source_group=source_group,
                conditions=conditions,
            )

    for origin, source_groups in origin_groups.items():
        if len(source_groups) > 1:
            findings.error(
                "LINEAGE",
                "ORIGIN_SOURCE_GROUP_LEAKAGE",
                "One origin identity appears in multiple source_groups.",
                origin_pageid=origin[0],
                origin_sha256=origin[1],
                source_groups=sorted(source_groups),
            )

    for origin_pageid, bindings in origin_pageid_bindings.items():
        if len(bindings) > 1:
            findings.error(
                "LINEAGE",
                "ORIGIN_PAGEID_BINDING_CONFLICT",
                "origin_pageid must map to one origin_sha256, source_group and partition.",
                origin_pageid=origin_pageid,
                bindings=[
                    {"origin_sha256": sha, "source_group": group, "split": split}
                    for sha, group, split in sorted(bindings)
                ],
            )
    for origin_sha, bindings in origin_sha_bindings.items():
        if len(bindings) > 1:
            findings.error(
                "LINEAGE",
                "ORIGIN_SHA_BINDING_CONFLICT",
                "origin_sha256 must map to one origin_pageid, source_group and partition.",
                origin_sha256=origin_sha,
                bindings=[
                    {"origin_pageid": pageid, "source_group": group, "split": split}
                    for pageid, group, split in sorted(bindings)
                ],
            )

    for source_page, bindings in source_page_bindings.items():
        if len(bindings) > 1:
            findings.error(
                "LINEAGE",
                "SOURCE_PAGE_BINDING_CONFLICT",
                "A canonical source_page locator must map to one origin_sha256, source_group and partition.",
                source_page=source_page,
                bindings=[
                    {"origin_sha256": sha, "source_group": group, "split": split}
                    for sha, group, split in sorted(bindings)
                ],
            )

    if len(optical_roots) > 1:
        findings.error(
            "FINAL_OPTICS",
            "OPTICAL_ROOT_INCONSISTENT",
            "All final-optics captures must bind one optical_domain_root.",
            roots=sorted(optical_roots),
        )

    native_records = [record for record in dhash_records if record["native_v2"]]
    if native_v2_dhash_count != native_v2_row_count:
        findings.error(
            "NEAR_DUPLICATE",
            "PERCEPTUAL_EVIDENCE_INCOMPLETE",
            "Every native-v2 row must produce dHash evidence.",
            native_v2_rows=native_v2_row_count,
            computed=native_v2_dhash_count,
        )
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in native_records:
        by_group[record["source_group"]].append(record)
    near_duplicate_pairs = 0
    group_names = sorted(by_group)
    for left_index, left_group in enumerate(group_names):
        for right_group in group_names[left_index + 1 :]:
            closest: tuple[int, dict[str, Any], dict[str, Any]] | None = None
            for left_record in by_group[left_group]:
                for right_record in by_group[right_group]:
                    distance = _hamming_hex(left_record["dhash64"], right_record["dhash64"])
                    candidate_pair = (distance, left_record, right_record)
                    if closest is None or distance < closest[0]:
                        closest = candidate_pair
            if closest is not None and closest[0] <= dhash_threshold:
                near_duplicate_pairs += 1
                if near_duplicate_pairs <= 100:
                    findings.error(
                        "NEAR_DUPLICATE",
                        "NEAR_DUPLICATE_SOURCE_GROUP",
                        "dHash distance is at or below the cross-source-group threshold.",
                        left_source_group=left_group,
                        right_source_group=right_group,
                        left_asset_id=closest[1]["asset_id"],
                        right_asset_id=closest[2]["asset_id"],
                        distance=closest[0],
                        threshold=dhash_threshold,
                    )
    if near_duplicate_pairs > 100:
        findings.error(
            "NEAR_DUPLICATE",
            "NEAR_DUPLICATE_ERRORS_TRUNCATED",
            "More than 100 cross-source-group near-duplicate pairs were found.",
            total_pairs=near_duplicate_pairs,
        )

    perceptual_payload = [
        {"asset_id": item["asset_id"], "dhash64": item["dhash64"], "source_group": item["source_group"]}
        for item in sorted(native_records, key=lambda value: value["asset_id"])
    ]
    perceptual_root = _canonical_json_sha256(perceptual_payload) if perceptual_payload else None

    return {
        "row_count": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "active_class_counts": dict(sorted(active_class_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "review_status_counts": dict(sorted(review_counts.items())),
        "ptq_class_counts": dict(sorted(ptq_class_counts.items())),
        "unique_asset_ids": len(asset_ids),
        "unique_asset_sha256": len(asset_hashes),
        "unique_origins": len(origin_groups),
        "unique_capture_ids": len(capture_ids),
        "unique_source_groups": len(group_splits),
        "single_partition_source_groups": sum(1 for value in group_splits.values() if len(value) == 1),
        "optical_domain_roots": sorted(optical_roots),
        "quality_passed_capture_count": quality_capture_count,
        "perceptual_hash_evidence": {
            "algorithm": "dhash64",
            "distance_threshold": dhash_threshold,
            "native_v2_row_count": native_v2_row_count,
            "computed_native_v2_row_count": native_v2_dhash_count,
            "near_duplicate_source_group_pair_count": near_duplicate_pairs,
            "evidence_sha256": perceptual_root,
        },
    }


def _validate_curation(
    curation_path: Path | None,
    rows: list[dict[str, Any]],
    print_demo_domains: set[str],
    findings: Findings,
) -> dict[str, Any]:
    if curation_path is None or not curation_path.exists():
        findings.warning("CURATION", "CURATION_NOT_PRESENT", "No curation file was supplied; consistency check skipped.")
        return {"present": False, "reservation_count": 0, "rejection_count": 0}
    try:
        curation = _read_json(curation_path)
    except (OSError, json.JSONDecodeError) as exc:
        findings.error("CURATION", "CURATION_INVALID", f"Cannot read curation JSON: {exc}.")
        return {"present": True, "reservation_count": 0, "rejection_count": 0}
    if not isinstance(curation, dict):
        findings.error("CURATION", "CURATION_SHAPE", "Curation JSON must be an object.")
        return {"present": True, "reservation_count": 0, "rejection_count": 0}

    reservations = curation.get("reservations", {})
    rejections = curation.get("rejections", {})
    if not isinstance(reservations, dict) or not isinstance(rejections, dict):
        findings.error("CURATION", "CURATION_SHAPE", "reservations and rejections must be objects.")
        return {"present": True, "reservation_count": 0, "rejection_count": 0}
    overlap = set(reservations).intersection(rejections)
    if overlap:
        findings.error("CURATION", "CURATION_OVERLAP", "A pageid is both reserved and rejected.", pageids=sorted(overlap))

    by_pageid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("asset_role") == "source":
            by_pageid[str(row.get("origin_pageid", ""))].append(row)
    for pageid in reservations:
        matches = by_pageid.get(str(pageid), [])
        if len(matches) != 1:
            findings.error("CURATION", "RESERVATION_NOT_UNIQUE", "Reserved pageid must match exactly one row.", pageid=str(pageid))
            continue
        row = matches[0]
        if (
            row.get("split") != "print_demo"
            or row.get("domain") not in print_demo_domains
            or row.get("permanent_holdout") is not True
            or row.get("sealed") is not True
        ):
            findings.error(
                "CURATION",
                "RESERVATION_NOT_HELD_OUT",
                "Reserved pageid is not permanently isolated in print_demo.",
                pageid=str(pageid),
                line=row.get("_line"),
            )
    for pageid in rejections:
        matches = by_pageid.get(str(pageid), [])
        if len(matches) != 1:
            findings.error("CURATION", "REJECTION_NOT_UNIQUE", "Rejected pageid must match exactly one row.", pageid=str(pageid))
            continue
        row = matches[0]
        if row.get("review_status") != "rejected_visual" or row.get("split") != "excluded" or row.get("print_eligible") is not False:
            findings.error(
                "CURATION",
                "REJECTION_NOT_EXCLUDED",
                "Rejected pageid is not fully excluded.",
                pageid=str(pageid),
                line=row.get("_line"),
            )
    return {
        "present": True,
        "path": str(curation_path),
        "sha256": _sha256_json_file(curation_path),
        "version": curation.get("version"),
        "reservation_count": len(reservations),
        "rejection_count": len(rejections),
    }


def _validate_optical_receipt(
    receipt_path: Path | None,
    rules: dict[str, Any],
    optical_roots: list[str],
    findings: Findings,
) -> dict[str, Any]:
    if receipt_path is None:
        findings.warning(
            "OPTICAL_RECEIPT",
            "OPTICAL_RECEIPT_NOT_SUPPLIED",
            "No optical-domain receipt was supplied; integrity may pass but DATA_LOCKED cannot.",
        )
        return {"present": False, "valid": False, "canonical_sha256": None, "matches_capture_root": False}
    receipt_path = Path(receipt_path)
    if not receipt_path.is_file():
        findings.error(
            "OPTICAL_RECEIPT",
            "OPTICAL_RECEIPT_UNREADABLE",
            "The supplied optical-domain receipt does not exist.",
            path=str(receipt_path),
        )
        return {"present": False, "valid": False, "path": str(receipt_path), "canonical_sha256": None, "matches_capture_root": False}
    try:
        receipt = _read_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        findings.error(
            "OPTICAL_RECEIPT",
            "OPTICAL_RECEIPT_UNREADABLE",
            "The supplied optical-domain receipt is not readable JSON.",
            path=str(receipt_path),
            error=str(exc),
        )
        return {"present": True, "valid": False, "path": str(receipt_path), "canonical_sha256": None, "matches_capture_root": False}
    if not isinstance(receipt, dict):
        findings.error("OPTICAL_RECEIPT", "OPTICAL_RECEIPT_SHAPE", "Optical-domain receipt must be a JSON object.")
        return {"present": True, "valid": False, "path": str(receipt_path), "canonical_sha256": None, "matches_capture_root": False}

    contract = rules.get("optical_domain_receipt", {})
    required_roles = contract.get("required_signed_roles", {})
    required_roots = contract.get("required_evidence_roots", [])
    signed_roles = receipt.get("signed_roles")
    evidence_roots = receipt.get("evidence_roots")
    valid = True
    required_top_level = set(contract.get("required_top_level_fields", []))
    if set(receipt) != required_top_level:
        valid = False
        findings.error(
            "OPTICAL_RECEIPT",
            "OPTICAL_RECEIPT_FIELDS",
            "Receipt must contain exactly the frozen top-level fields.",
            actual_fields=sorted(receipt),
        )
    if receipt.get("schema_version") != contract.get("receipt_schema_version"):
        valid = False
        findings.error(
            "OPTICAL_RECEIPT",
            "OPTICAL_RECEIPT_VERSION",
            "Receipt schema_version must exactly match the frozen contract version.",
            expected=contract.get("receipt_schema_version"),
            actual=receipt.get("schema_version"),
        )
    if not isinstance(receipt.get("receipt_id"), str) or not receipt.get("receipt_id", "").strip():
        valid = False
        findings.error("OPTICAL_RECEIPT", "OPTICAL_RECEIPT_ID", "Receipt receipt_id must be a non-empty string.")
    if not isinstance(signed_roles, dict):
        valid = False
        findings.error("OPTICAL_RECEIPT", "OPTICAL_SIGNED_ROLES_MISSING", "Receipt signed_roles must be an object.")
    else:
        signers: list[str] = []
        approval_evidence: list[str] = []
        if set(signed_roles) != set(required_roles):
            valid = False
            findings.error(
                "OPTICAL_RECEIPT",
                "OPTICAL_SIGNED_ROLES_SHAPE",
                "Receipt signed_roles must contain exactly hardware, mechanical and operations.",
                actual_roles=sorted(signed_roles),
            )
        for role, member in required_roles.items():
            entry = signed_roles.get(role)
            signer_identity = _canonical_signer_identity(entry.get("signer")) if isinstance(entry, dict) else None
            if (
                not isinstance(entry, dict)
                or set(entry) != set(contract.get("required_role_entry_fields", []))
                or entry.get("member") != member
                or entry.get("signed") is not True
                or signer_identity is None
                or type(entry.get("approval_evidence_sha256")) is not str
                or SHA256_RE.fullmatch(entry.get("approval_evidence_sha256", "")) is None
            ):
                valid = False
                findings.error(
                    "OPTICAL_RECEIPT",
                    "OPTICAL_SIGNED_ROLE_INVALID",
                    "Required B/C/D role is absent or not explicitly signed.",
                    role=role,
                    expected_member=member,
                )
            else:
                signers.append(signer_identity)
                approval_evidence.append(entry["approval_evidence_sha256"])
        if len(signers) != len(required_roles) or len(set(signers)) != len(signers):
            valid = False
            findings.error(
                "OPTICAL_RECEIPT",
                "OPTICAL_SIGNERS_NOT_DISTINCT",
                "Hardware, mechanical and operations approvals require three distinct signer identities.",
                signers=signers,
            )
        if len(approval_evidence) != len(required_roles) or len(set(approval_evidence)) != len(approval_evidence):
            valid = False
            findings.error(
                "OPTICAL_RECEIPT",
                "OPTICAL_APPROVAL_EVIDENCE_NOT_DISTINCT",
                "B/C/D approvals must bind three distinct evidence artifacts.",
                approval_evidence_sha256=approval_evidence,
            )
    if not isinstance(evidence_roots, dict):
        valid = False
        findings.error("OPTICAL_RECEIPT", "OPTICAL_EVIDENCE_ROOTS_MISSING", "Receipt evidence_roots must be an object.")
    else:
        if set(evidence_roots) != set(required_roots):
            valid = False
            findings.error(
                "OPTICAL_RECEIPT",
                "OPTICAL_EVIDENCE_ROOTS_SHAPE",
                "Receipt evidence_roots must contain exactly the five frozen physical-domain roots.",
                actual_roots=sorted(evidence_roots),
            )
        for name in required_roots:
            value = evidence_roots.get(name)
            if type(value) is not str or SHA256_RE.fullmatch(value) is None:
                valid = False
                findings.error(
                    "OPTICAL_RECEIPT",
                    "OPTICAL_EVIDENCE_ROOT_INVALID",
                    "Required physical-domain evidence root is missing or malformed.",
                    evidence=name,
                )

    try:
        canonical_sha256 = _canonical_json_sha256(receipt)
    except (TypeError, ValueError):
        canonical_sha256 = None
        valid = False
        findings.error("OPTICAL_RECEIPT", "OPTICAL_RECEIPT_CANONICALIZATION", "Receipt cannot be canonically serialized.")
    matches = bool(canonical_sha256 and optical_roots and set(optical_roots) == {canonical_sha256})
    if optical_roots and canonical_sha256 and not matches:
        valid = False
        findings.error(
            "OPTICAL_RECEIPT",
            "OPTICAL_RECEIPT_ROOT_MISMATCH",
            "Final-optics rows do not bind the canonical receipt SHA-256.",
            canonical_sha256=canonical_sha256,
            row_roots=optical_roots,
        )
    return {
        "present": True,
        "valid": valid,
        "path": str(receipt_path),
        "raw_sha256": _sha256_file(receipt_path),
        "canonical_sha256": canonical_sha256,
        "matches_capture_root": matches,
        "signed_roles": sorted(signed_roles) if isinstance(signed_roles, dict) else [],
        "evidence_roots": sorted(evidence_roots) if isinstance(evidence_roots, dict) else [],
    }


def _evaluate_readiness(
    rows: list[dict[str, Any]],
    expected_classes: list[str],
    rules: dict[str, Any],
    migration: dict[str, Any],
    optical_receipt: dict[str, Any],
    contract_lock: dict[str, Any],
    findings: Findings,
) -> dict[str, Any]:
    minimums = rules.get("ready_minimums", {})
    partitions = set(rules.get("source_group_partitions", []))
    approved_statuses = set(rules.get("approved_review_statuses", []))
    roles = rules.get("domain_roles", {})
    natural_domains = set(roles.get("natural", []))
    printed_train_domains = set(roles.get("printed_train", []))
    printed_test_domains = set(roles.get("printed_test", []))
    site_domains = set(roles.get("site_acceptance", []))

    reason_details: list[dict[str, Any]] = []

    def add_reason(code: str, message: str, **context: Any) -> None:
        detail = {"code": code, "message": message, **context}
        reason_details.append(detail)
        findings.warning("READY_MINIMUM", code, message, **context)

    if not contract_lock.get("valid"):
        add_reason(
            "CLASS_CONTRACT_LOCK_INVALID",
            "The external production class-contract lock is missing, malformed, stale, or does not bind this exact contract.",
            contract_lock=contract_lock,
        )

    if migration.get("applied"):
        add_reason(
            "NATIVE_V2_NOT_PERSISTED",
            "The registered v1 seed manifest is only projected in memory; persist native v2 rows before DATA_LOCKED.",
        )

    if not optical_receipt.get("present"):
        add_reason(
            "OPTICAL_RECEIPT_REQUIRED",
            "A readable B/C/D-signed optical-domain receipt is required for DATA_LOCKED.",
        )
    elif not optical_receipt.get("valid") or not optical_receipt.get("matches_capture_root"):
        add_reason(
            "OPTICAL_RECEIPT_INVALID",
            "The optical-domain receipt is invalid or not bound by every final-optics capture.",
        )

    unassigned = [row for row in rows if row.get("split") == "unassigned"]
    if unassigned:
        add_reason("ROWS_UNASSIGNED", f"{len(unassigned)} rows remain unassigned.", count=len(unassigned))

    usable = [row for row in rows if row.get("split") in partitions]
    unapproved = [row for row in usable if row.get("review_status") not in approved_statuses]
    if unapproved:
        add_reason(
            "USABLE_ROWS_UNAPPROVED",
            f"{len(unapproved)} usable or sealed rows lack approved visual/license review.",
            count=len(unapproved),
        )
    if minimums.get("require_reviewer_for_usable_rows", True):
        reviewer_missing = [row for row in usable if row.get("review_status") in approved_statuses and not _is_nonempty(row.get("reviewed_by"))]
        if reviewer_missing:
            add_reason(
                "REVIEWER_MISSING",
                f"{len(reviewer_missing)} approved usable rows have no reviewed_by identity.",
                count=len(reviewer_missing),
            )

    approved = [row for row in usable if row.get("review_status") in approved_statuses and _is_nonempty(row.get("reviewed_by"))]
    approved_split_counts = Counter(str(row.get("split")) for row in approved)
    if minimums.get("require_all_source_group_partitions", True):
        missing = [name for name in REQUIRED_PARTITIONS if approved_split_counts[name] == 0]
        if missing:
            add_reason("PARTITIONS_MISSING", "Approved rows are missing from frozen partitions: " + ", ".join(missing), partitions=missing)

    partition_class_metrics: dict[str, dict[str, int]] = {}
    natural_split_minimums = minimums.get("natural_source_group_minimums_by_split", {})
    for partition, required_classes in rules.get("partition_class_coverage", {}).items():
        partition_class_metrics[partition] = {}
        for class_id in required_classes:
            groups = {
                str(row.get("source_group"))
                for row in approved
                if row.get("split") == partition
                and row.get("class_id") == class_id
                and row.get("asset_role") == "source"
                and row.get("domain") in natural_domains
            }
            partition_class_metrics[partition][class_id] = len(groups)
            required = int(natural_split_minimums.get(partition, {}).get(class_id, 0))
            if len(groups) < required:
                add_reason(
                    "PARTITION_CLASS_SOURCE_GROUP_MINIMUM",
                    f"{partition}/{class_id} has {len(groups)} approved independent source groups; {required} required.",
                    partition=partition,
                    class_id=class_id,
                    actual=len(groups),
                    required=required,
                )

    natural_required = minimums.get("natural_unique_source_groups", {})
    natural_counts: dict[str, int] = {}
    for class_id in expected_classes:
        groups = {
            str(row.get("source_group"))
            for row in approved
            if row.get("class_id") == class_id
            and row.get("domain") in natural_domains
            and row.get("asset_role") == "source"
        }
        natural_counts[class_id] = len(groups)
        required = int(natural_required.get(class_id, 0))
        if len(groups) < required:
            add_reason(
                "NATURAL_SOURCE_GROUP_MINIMUM",
                f"{class_id} has {len(groups)} approved natural source groups; {required} required.",
                class_id=class_id,
                actual=len(groups),
                required=required,
            )

    scenarios = {
        str(row.get("unknown_scenario"))
        for row in approved
        if row.get("class_id") == "unknown"
        and row.get("domain") in natural_domains
        and row.get("asset_role") == "source"
        and _is_nonempty(row.get("unknown_scenario"))
    }
    scenario_required = int(minimums.get("unknown_scenario_coverage", 0))
    if len(scenarios) < scenario_required:
        add_reason(
            "UNKNOWN_SCENARIO_COVERAGE",
            f"unknown covers {len(scenarios)} frozen scenarios; {scenario_required} required.",
            actual=len(scenarios),
            required=scenario_required,
            scenarios=sorted(scenarios),
        )

    golden_source_rows = [
        row
        for row in approved
        if row.get("split") == "conversion_golden" and row.get("asset_role") == "source"
    ]
    golden_groups_by_class = {
        class_id: {
            str(row.get("source_group"))
            for row in golden_source_rows
            if row.get("class_id") == class_id
        }
        for class_id in expected_classes
    }
    golden_total_groups = len(set().union(*golden_groups_by_class.values())) if golden_groups_by_class else 0
    golden_total_required = int(minimums.get("conversion_golden_qualifying_assets_total", 0))
    golden_per_class_required = int(minimums.get("conversion_golden_qualifying_assets_per_class", 0))
    if golden_total_groups < golden_total_required:
        add_reason(
            "CONVERSION_GOLDEN_TOTAL_MINIMUM",
            f"conversion_golden has {golden_total_groups} independent approved source assets; {golden_total_required} required.",
            actual=golden_total_groups,
            required=golden_total_required,
        )
    for class_id, groups in golden_groups_by_class.items():
        if len(groups) < golden_per_class_required:
            add_reason(
                "CONVERSION_GOLDEN_CLASS_MINIMUM",
                f"conversion_golden/{class_id} has {len(groups)} independent approved source assets; {golden_per_class_required} required.",
                class_id=class_id,
                actual=len(groups),
                required=golden_per_class_required,
            )

    eligible_local_sources: dict[str, str] = {
        str(row.get("source_group")): str(row.get("unknown_scenario"))
        for row in approved
        if row.get("class_id") == "unknown"
        and row.get("domain") == "local_negative"
        and row.get("asset_role") == "source"
        and type(row.get("unknown_scenario")) is str
        and row.get("unknown_scenario") in rules.get("unknown_scenarios", [])
    }
    local_scenario_groups: dict[str, set[str]] = defaultdict(set)
    for row in approved:
        if (
            row.get("class_id") == "unknown"
            and row.get("domain") == "local_negative"
            and row.get("asset_role") == "local_capture"
            and row.get("capture_quality_pass") is True
            and type(row.get("optical_domain_root")) is str
            and SHA256_RE.fullmatch(row["optical_domain_root"])
            and type(row.get("unknown_scenario")) is str
            and row.get("unknown_scenario") in rules.get("unknown_scenarios", [])
            and str(row.get("source_group")) in eligible_local_sources
            and eligible_local_sources[str(row.get("source_group"))] == row.get("unknown_scenario")
        ):
            local_scenario_groups[str(row["unknown_scenario"])].add(str(row.get("source_group")))
    local_scenario_required = int(minimums.get("local_negative_final_optics_scenario_coverage", 0))
    if len(local_scenario_groups) < local_scenario_required:
        add_reason(
            "LOCAL_NEGATIVE_FINAL_OPTICS_COVERAGE",
            f"Final-optics local_negative covers {len(local_scenario_groups)} scenarios; {local_scenario_required} required.",
            actual=len(local_scenario_groups),
            required=local_scenario_required,
            scenarios=sorted(local_scenario_groups),
        )

    variation_distance_threshold = int(rules.get("perceptual_duplicate_audit", {}).get("distance_threshold", 0))

    def effective_variation_counts(
        eligible_sources: set[str],
        candidates: list[dict[str, Any]],
    ) -> dict[str, int]:
        by_group: dict[str, list[dict[str, Any]]] = {group: [] for group in eligible_sources}
        for row in candidates:
            group = str(row.get("source_group"))
            if group in by_group:
                by_group[group].append(row)
        counts: dict[str, int] = {}
        for group, group_rows in by_group.items():
            unique_conditions: set[str] = set()
            cluster_representatives: list[str] = []
            for row in sorted(group_rows, key=lambda value: str(value.get("capture_condition_id"))):
                condition = row.get("capture_condition_id")
                dhash = row.get("_dhash64")
                if type(condition) is not str or not condition or type(dhash) is not str or condition in unique_conditions:
                    continue
                unique_conditions.add(condition)
                if all(_hamming_hex(dhash, representative) > variation_distance_threshold for representative in cluster_representatives):
                    cluster_representatives.append(dhash)
            counts[group] = min(len(unique_conditions), len(cluster_representatives))
        return counts

    def qualifying_printed_groups(
        domains: set[str],
        source_domain: str,
        class_id: str,
        capture_minimum: int,
    ) -> tuple[set[str], list[str], dict[str, str]]:
        eligible_source_splits = {
            str(row.get("source_group")): str(row.get("split"))
            for row in approved
            if row.get("class_id") == class_id
            and row.get("domain") == source_domain
            and row.get("asset_role") == "source"
            and row.get("print_eligible") is True
        }
        eligible_sources = set(eligible_source_splits)
        candidates = [
            row
            for row in approved
            if row.get("class_id") == class_id
            and row.get("domain") in domains
            and row.get("asset_role") == "print_capture"
            and row.get("capture_quality_pass") is True
            and row.get("print_eligible") is True
            and type(row.get("optical_domain_root")) is str
            and SHA256_RE.fullmatch(row["optical_domain_root"])
            and type(row.get("capture_id")) is str
            and bool(row.get("capture_id", "").strip())
            and str(row.get("source_group")) in eligible_sources
        ]
        counts = effective_variation_counts(eligible_sources, candidates)
        qualifying = {group for group, count in counts.items() if count >= capture_minimum}
        short = sorted(group for group, count in counts.items() if count < capture_minimum)
        return qualifying, short, eligible_source_splits

    printed_metrics: dict[str, Any] = {"train": {}, "test": {}}
    printed_train_split_minimums = minimums.get("printed_train_source_group_minimums_by_split", {})
    for role_name, domains, source_domain, group_key, capture_key, metric_key in (
        (
            "printed_train",
            printed_train_domains,
            "printed_train",
            "printed_train_source_groups_per_target_class",
            "printed_train_captures_per_source_group",
            "train",
        ),
        (
            "printed_test",
            printed_test_domains,
            "printed_test",
            "printed_test_source_groups_per_target_class",
            "printed_test_captures_per_source_group",
            "test",
        ),
    ):
        required_groups = int(minimums.get(group_key, 0))
        required_captures = int(minimums.get(capture_key, 0))
        for class_id in TARGET_CLASSES:
            qualifying_groups, short, source_splits = qualifying_printed_groups(
                domains, source_domain, class_id, required_captures
            )
            qualifying = len(qualifying_groups)
            printed_metrics[metric_key][class_id] = {
                "qualifying_source_groups": qualifying,
                "undercaptured_source_groups": short,
            }
            if qualifying < required_groups:
                add_reason(
                    f"{role_name.upper()}_MINIMUM",
                    f"{role_name}/{class_id} has {qualifying} qualifying source groups; {required_groups} required at {required_captures} captures each.",
                    class_id=class_id,
                    actual=qualifying,
                    required=required_groups,
                    captures_per_group=required_captures,
                    undercaptured_source_groups=short,
                )
            if role_name == "printed_train":
                by_split = {
                    split: sum(1 for group in qualifying_groups if source_splits.get(group) == split)
                    for split in printed_train_split_minimums
                }
                printed_metrics[metric_key][class_id]["qualifying_source_groups_by_split"] = by_split
                for split, per_class in printed_train_split_minimums.items():
                    split_required = int(per_class.get(class_id, 0))
                    actual = by_split.get(split, 0)
                    if actual < split_required:
                        add_reason(
                            "PRINTED_TRAIN_SPLIT_MINIMUM",
                            f"printed_train/{split}/{class_id} has {actual} qualifying source groups; {split_required} required.",
                            split=split,
                            class_id=class_id,
                            actual=actual,
                            required=split_required,
                            captures_per_group=required_captures,
                        )

    print_demo_required = int(minimums.get("print_demo_source_groups_per_target_class", 0))
    print_demo_capture_required = int(minimums.get("print_demo_captures_per_source_group", 0))
    print_demo_counts: dict[str, Any] = {}
    for class_id in TARGET_CLASSES:
        eligible_demo_sources = {
            str(row.get("source_group"))
            for row in approved
            if row.get("class_id") == class_id
            and row.get("domain") == "print_demo_source"
            and row.get("split") == "print_demo"
            and row.get("asset_role") == "source"
            and row.get("print_eligible") is True
            and row.get("permanent_holdout") is True
            and row.get("sealed") is True
        }
        demo_candidates = [
            row
            for row in approved
            if row.get("class_id") == class_id
            and row.get("domain") == "printed_demo_capture"
            and row.get("split") == "print_demo"
            and row.get("permanent_holdout") is True
            and row.get("sealed") is True
            and row.get("print_eligible") is True
            and row.get("asset_role") == "print_capture"
            and row.get("capture_quality_pass") is True
            and type(row.get("optical_domain_root")) is str
            and SHA256_RE.fullmatch(row["optical_domain_root"])
            and str(row.get("source_group")) in eligible_demo_sources
        ]
        capture_counts = effective_variation_counts(eligible_demo_sources, demo_candidates)
        qualifying_groups = sum(1 for count in capture_counts.values() if count >= print_demo_capture_required)
        undercaptured = sorted(group for group, count in capture_counts.items() if count < print_demo_capture_required)
        print_demo_counts[class_id] = {
            "qualifying_source_groups": qualifying_groups,
            "undercaptured_source_groups": undercaptured,
        }
        if qualifying_groups < print_demo_required:
            add_reason(
                "PRINT_DEMO_MINIMUM",
                f"print_demo/{class_id} has {qualifying_groups} source groups with quality-passed final-optics captures; {print_demo_required} required.",
                class_id=class_id,
                actual=qualifying_groups,
                required=print_demo_required,
                captures_per_group=print_demo_capture_required,
                undercaptured_source_groups=undercaptured,
            )

    site_target_required = int(minimums.get("site_acceptance_source_groups_per_target_class", 0))
    site_unknown_required = int(minimums.get("site_acceptance_unknown_source_groups", 0))
    site_counts: dict[str, int] = {}
    for class_id in expected_classes:
        groups = {
            str(row.get("source_group"))
            for row in approved
            if row.get("class_id") == class_id
            and row.get("domain") in site_domains
            and row.get("split") == "site_acceptance"
            and row.get("sealed") is True
            and row.get("asset_role") == "source"
            and (class_id == "unknown" or row.get("print_eligible") is True)
            and (
                class_id != "unknown"
                or (type(row.get("unknown_scenario")) is str and row.get("unknown_scenario") in rules.get("unknown_scenarios", []))
            )
        }
        site_counts[class_id] = len(groups)
        required = site_unknown_required if class_id == "unknown" else site_target_required
        if len(groups) < required:
            add_reason(
                "SITE_ACCEPTANCE_MINIMUM",
                f"site_acceptance/{class_id} has {len(groups)} sealed source groups; {required} required.",
                class_id=class_id,
                actual=len(groups),
                required=required,
            )

    site_unknown_scenario_groups: dict[str, set[str]] = defaultdict(set)
    for row in approved:
        if (
            row.get("split") == "site_acceptance"
            and row.get("domain") in site_domains
            and row.get("class_id") == "unknown"
            and row.get("sealed") is True
            and row.get("asset_role") == "source"
            and type(row.get("unknown_scenario")) is str
            and row.get("unknown_scenario") in rules.get("unknown_scenarios", [])
        ):
            site_unknown_scenario_groups[str(row["unknown_scenario"])].add(str(row.get("source_group")))
    site_scenario_required = int(minimums.get("site_acceptance_unknown_scenario_coverage", 0))
    if len(site_unknown_scenario_groups) < site_scenario_required:
        add_reason(
            "SITE_ACCEPTANCE_UNKNOWN_SCENARIO_COVERAGE",
            f"site_acceptance has {len(site_unknown_scenario_groups)} unknown scenario families; {site_scenario_required} required across independent sealed scenes.",
            actual=len(site_unknown_scenario_groups),
            required=site_scenario_required,
            scenarios=sorted(site_unknown_scenario_groups),
        )

    ptq_required = int(rules.get("ptq_calibration", {}).get("minimum_source_groups_per_class", 0))
    ptq_counts: dict[str, int] = {}
    for class_id in expected_classes:
        groups = {
            str(row.get("source_group"))
            for row in approved
            if row.get("class_id") == class_id
            and row.get("split") == "train"
            and row.get("asset_role") == "source"
            and row.get("ptq_calibration") is True
        }
        ptq_counts[class_id] = len(groups)
        if len(groups) < ptq_required:
            add_reason(
                "PTQ_SUBSET_MINIMUM",
                f"PTQ train subset/{class_id} has {len(groups)} source groups; {ptq_required} required.",
                class_id=class_id,
                actual=len(groups),
                required=ptq_required,
            )

    return {
        "status": "NOT_TRAIN_READY" if reason_details or findings.flattened_errors() else "READY",
        "reasons": [item["message"] for item in reason_details],
        "reason_details": reason_details,
        "metrics": {
            "approved_partition_row_counts": dict(sorted(approved_split_counts.items())),
            "partition_class_source_groups": partition_class_metrics,
            "natural_unique_source_groups": natural_counts,
            "conversion_golden_qualifying_source_assets": {
                "total": golden_total_groups,
                "per_class": {class_id: len(groups) for class_id, groups in golden_groups_by_class.items()},
            },
            "unknown_scenarios": sorted(scenarios),
            "local_negative_final_optics_scenarios": {
                scenario: len(groups) for scenario, groups in sorted(local_scenario_groups.items())
            },
            "printed": printed_metrics,
            "print_demo_source_groups": print_demo_counts,
            "site_acceptance_source_groups": site_counts,
            "site_acceptance_unknown_scenario_groups": {
                scenario: len(groups) for scenario, groups in sorted(site_unknown_scenario_groups.items())
            },
            "ptq_train_source_groups": ptq_counts,
            "optical_domain_receipt": {
                "present": optical_receipt.get("present", False),
                "valid": optical_receipt.get("valid", False),
                "canonical_sha256": optical_receipt.get("canonical_sha256"),
                "matches_capture_root": optical_receipt.get("matches_capture_root", False),
            },
            "class_contract_lock": contract_lock,
        },
    }


def audit_dataset(
    dataset_dir: Path,
    manifest_path: Path,
    contract_path: Path,
    curation_path: Path | None = None,
    generated_at: str | None = None,
    optical_domain_receipt_path: Path | None = None,
    contract_lock_path: Path | None = None,
    *,
    test_only_allow_unlocked_contract: bool = False,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    manifest_path = Path(manifest_path)
    contract_path = Path(contract_path)
    curation_path = Path(curation_path) if curation_path is not None else None
    optical_domain_receipt_path = Path(optical_domain_receipt_path) if optical_domain_receipt_path is not None else None
    contract_lock_path = (
        Path(contract_lock_path)
        if contract_lock_path is not None
        else contract_path.with_name("class_contract.lock.json")
    )
    findings = Findings()

    try:
        contract = _read_json(contract_path)
    except (OSError, json.JSONDecodeError) as exc:
        findings.error("CONTRACT", "CONTRACT_UNREADABLE", f"Cannot read class contract: {exc}.", path=str(contract_path))
        contract = {}
    contract_sha256 = _sha256_json_file(contract_path)
    contract_lock = _validate_contract_lock(
        contract_lock_path,
        contract_sha256,
        contract,
        test_only_allow_unlocked_contract=test_only_allow_unlocked_contract,
    )
    expected_classes, rules = _validate_contract(contract, findings)
    manifest_sha256 = _sha256_json_file(manifest_path)
    raw_rows = _read_manifest(manifest_path, findings)
    rows, migration = _apply_manifest_migration(raw_rows, manifest_sha256, rules, findings)
    summary = _validate_rows(rows, dataset_dir, expected_classes, rules, findings) if rows else {"row_count": 0}
    print_demo_domains = set(rules.get("domain_roles", {}).get("print_demo", []))
    curation = _validate_curation(curation_path, rows, print_demo_domains, findings)
    optical_receipt = _validate_optical_receipt(
        optical_domain_receipt_path,
        rules,
        list(summary.get("optical_domain_roots", [])),
        findings,
    )
    readiness = _evaluate_readiness(
        rows,
        expected_classes,
        rules,
        migration,
        optical_receipt,
        contract_lock,
        findings,
    ) if rows else {
        "status": "NOT_TRAIN_READY",
        "reasons": ["Manifest has no rows."],
        "reason_details": [{"code": "MANIFEST_EMPTY", "message": "Manifest has no rows."}],
        "metrics": {},
    }

    errors = findings.flattened_errors()
    warnings = findings.flattened_warnings()
    if errors and readiness["status"] == "READY":
        readiness["status"] = "NOT_TRAIN_READY"
        readiness["reasons"].append("Integrity checks failed.")
        readiness["reason_details"].append({"code": "INTEGRITY_FAILED", "message": "Integrity checks failed."})
    return {
        "schema_version": "2.0.0",
        "audit_id": "rootscope.dataset_contract.v2",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "status": "FAIL" if errors else "PASS",
        "integrity_status": "FAIL" if errors else "PASS",
        "training_readiness": readiness,
        "migration": migration,
        "inputs": {
            "dataset_dir": str(dataset_dir),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "class_contract": str(contract_path),
            "class_contract_sha256": contract_sha256,
            "class_contract_lock": contract_lock,
            "curation": curation,
            "optical_domain_receipt": optical_receipt,
        },
        "summary": summary,
        "checks": findings.check_results(),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def _default_paths() -> tuple[Path, Path, Path, Path, Path]:
    adventurex_root = Path(__file__).resolve().parents[2]
    dataset_dir = adventurex_root / "datasets" / "desert_plants_v1"
    return (
        dataset_dir,
        dataset_dir / "manifest.jsonl",
        adventurex_root / "rootscope" / "configs" / "class_contract.json",
        dataset_dir / "curation_round1.json",
        adventurex_root / "rootscope" / "evidence" / "local_h12" / "dataset_audit.json",
    )


def main(argv: list[str] | None = None) -> int:
    dataset_default, manifest_default, contract_default, curation_default, output_default = _default_paths()
    parser = argparse.ArgumentParser(description="Audit RootScope dataset provenance and v2 source-group isolation.")
    parser.add_argument("--dataset-dir", type=Path, default=dataset_default)
    parser.add_argument("--manifest", type=Path, default=manifest_default)
    parser.add_argument("--contract", type=Path, default=contract_default)
    parser.add_argument(
        "--contract-lock",
        type=Path,
        default=None,
        help="External frozen production lock; defaults beside --contract as class_contract.lock.json.",
    )
    parser.add_argument("--curation", type=Path, default=curation_default)
    parser.add_argument("--optical-domain-receipt", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=output_default)
    args = parser.parse_args(argv)

    result = audit_dataset(
        args.dataset_dir,
        args.manifest,
        args.contract,
        args.curation,
        optical_domain_receipt_path=args.optical_domain_receipt,
        contract_lock_path=args.contract_lock,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"RootScope dataset audit: {result['status']} | rows={result['summary'].get('row_count', 0)} "
        f"| errors={result['error_count']} | warnings={result['warning_count']} "
        f"| readiness={result['training_readiness']['status']} "
        f"| migration={result['migration']['mode']}"
    )
    print(f"Evidence: {args.output}")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
