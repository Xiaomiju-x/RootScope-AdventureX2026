#!/usr/bin/env python3
"""Build a deterministic, human-only Wikimedia candidate review queue.

This tool never labels an asset and never marks an asset as training- or
print-ready.  It validates the complete input before creating any output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError


SCHEMA_VERSION = "rootscope.wikimedia_human_review_queue.v1"
SUMMARY_SCHEMA_VERSION = "rootscope.wikimedia_human_review_queue_summary.v1"
HOLDOUT_SCHEMA_VERSION = "rootscope.permanent_print_holdout.v1"
GENERATOR_VERSION = "2.0.0"

EXPECTED_HOLDOUT_PAGEIDS = frozenset(
    {133271396, 75559442, 2738023, 4424728, 5445424, 6021614}
)
ALLOWED_CLASS_HINTS = frozenset(
    {"grass_clump", "low_shrub", "young_tree", "unknown"}
)
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
DHASH64_RE = re.compile(r"^[0-9a-f]{16}$")

ADVENTUREX_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LICENSE_POLICY = Path(__file__).resolve().parent / "wikimedia_license_policy_v1.json"
DEFAULT_STAGING_MANIFEST = (
    ADVENTUREX_ROOT
    / "datasets"
    / "desert_plants_wikimedia_staging_e0"
    / "manifest.jsonl"
)
DEFAULT_HOLDOUT_MANIFEST = (
    ADVENTUREX_ROOT / "datasets" / "desert_plants_v1" / "manifest.jsonl"
)
DEFAULT_INTEGRITY_AUDIT = DEFAULT_STAGING_MANIFEST.parent / "integrity_audit.json"


class ReviewQueueError(RuntimeError):
    """Raised when queue construction must fail closed."""


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ReviewQueueError(f"duplicate JSON key: {key!r}")
        obj[key] = value
    return obj


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path, description: str) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReviewQueueError(f"cannot read {description}: {path}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewQueueError(f"{description} is not UTF-8: {path}: {exc}") from exc

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ReviewQueueError(
                f"{description} contains a blank line at line {line_number}"
            )
        try:
            record = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, ReviewQueueError) as exc:
            raise ReviewQueueError(
                f"invalid {description} JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ReviewQueueError(
                f"{description} line {line_number} must be a JSON object"
            )
        records.append(record)
    if not records:
        raise ReviewQueueError(f"{description} is empty: {path}")
    return records, _sha256_bytes(raw)


def _load_json_object(path: Path, description: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ReviewQueueError) as exc:
        raise ReviewQueueError(f"cannot load {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewQueueError(f"{description} must be a JSON object")
    return value, _sha256_bytes(raw)


def _load_policy(path: Path) -> tuple[dict[str, Any], str]:
    policy, policy_sha = _load_json_object(path, "shared license policy")
    if policy.get("schema_version") != "rootscope.wikimedia_license_policy.v1":
        raise ReviewQueueError("unsupported shared license policy schema")
    matching = policy.get("matching")
    if not isinstance(matching, dict) or matching != {
        "comparison": "ordinal_case_sensitive",
        "trim_whitespace": False,
        "normalize_trailing_slash": False,
        "unknown_binding_action": "REJECT",
    }:
        raise ReviewQueueError("shared license policy is not exact-pair fail-closed")
    constraints = policy.get("image_constraints")
    if not isinstance(constraints, dict):
        raise ReviewQueueError("shared license policy has no image constraints")
    if constraints.get("dhash_algorithm") != "rootscope_rgb_center_sample_9x8_v1":
        raise ReviewQueueError("unsupported dHash algorithm")
    observed_ids: set[str] = set()
    observed_pairs: set[tuple[str, str]] = set()
    for license_item in policy.get("licenses", []):
        if not isinstance(license_item, dict):
            raise ReviewQueueError("invalid license policy entry")
        canonical_id = license_item.get("canonical_id")
        if not isinstance(canonical_id, str) or canonical_id in observed_ids:
            raise ReviewQueueError("duplicate or invalid canonical license id")
        observed_ids.add(canonical_id)
        for binding in license_item.get("raw_bindings", []):
            if not isinstance(binding, dict) or set(binding) != {"raw_name", "raw_url"}:
                raise ReviewQueueError("raw license binding must be an exact name/url pair")
            pair = (binding.get("raw_name"), binding.get("raw_url"))
            if not all(isinstance(item, str) for item in pair) or pair in observed_pairs:
                raise ReviewQueueError("duplicate or invalid raw license pair")
            observed_pairs.add(pair)
    return policy, policy_sha


def _require_string(record: Mapping[str, Any], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReviewQueueError(f"{context}: {key} must be a non-empty string")
    return value.strip()


def _require_exact(record: Mapping[str, Any], key: str, expected: Any, context: str) -> None:
    if key not in record or type(record[key]) is not type(expected) or record[key] != expected:
        raise ReviewQueueError(
            f"{context}: {key} must be exactly {expected!r}, got {record.get(key)!r}"
        )


def _require_pageid(record: Mapping[str, Any], context: str) -> int:
    pageid = record.get("pageid")
    if type(pageid) is not int or pageid <= 0:
        raise ReviewQueueError(f"{context}: pageid must be a positive integer")
    return pageid


def _require_https_url(value: str, field: str, context: str, host: str | None = None) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or not parsed.path:
        raise ReviewQueueError(f"{context}: {field} must be a complete HTTPS URL")
    if host is not None and parsed.hostname != host:
        raise ReviewQueueError(
            f"{context}: {field} must use host {host!r}, got {parsed.hostname!r}"
        )
    return value


def _resolve_bound_file(root: Path, filename: str, context: str) -> tuple[Path, str]:
    if "\\" in filename:
        raise ReviewQueueError(f"{context}: filename must use POSIX separators")
    pure = PurePosixPath(filename)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ReviewQueueError(f"{context}: unsafe relative filename: {filename!r}")
    if ":" in pure.parts[0]:
        raise ReviewQueueError(f"{context}: unsafe drive-qualified filename: {filename!r}")

    root_resolved = root.resolve(strict=True)
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ReviewQueueError(f"{context}: local file does not exist: {filename!r}") from exc
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ReviewQueueError(f"{context}: local file escapes manifest root: {filename!r}") from exc
    if not resolved.is_file():
        raise ReviewQueueError(f"{context}: local path is not a regular file: {filename!r}")
    return resolved, pure.as_posix()


def _validate_sha(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or HEX64_RE.fullmatch(value) is None:
        raise ReviewQueueError(f"{context}: {field} must be 64 lowercase hexadecimal characters")
    return value


def _validate_dhash(value: Any, context: str) -> str:
    if not isinstance(value, str) or DHASH64_RE.fullmatch(value) is None:
        raise ReviewQueueError(f"{context}: dhash64 must be 16 lowercase hexadecimal characters")
    return value


def _license_binding(
    record: Mapping[str, Any],
    source_page: str,
    context: str,
    policy: Mapping[str, Any],
    policy_sha: str,
    policy_context: str,
    require_canonical_fields: bool,
) -> tuple[str, str, str, str, str]:
    raw_name = record.get("license_raw_name", record.get("license"))
    raw_url = record.get("license_raw_url", record.get("license_url"))
    if not isinstance(raw_name, str) or not isinstance(raw_url, str):
        raise ReviewQueueError(f"{context}: raw license name/url must be strings")
    if "license_raw_name" in record and record.get("license") != raw_name:
        raise ReviewQueueError(f"{context}: license and license_raw_name disagree")
    if "license_raw_url" in record and record.get("license_url") != raw_url:
        raise ReviewQueueError(f"{context}: license_url and license_raw_url disagree")

    matched: tuple[Mapping[str, Any], str] | None = None
    for license_item in policy["licenses"]:
        for binding in license_item["raw_bindings"]:
            if binding["raw_name"] == raw_name and binding["raw_url"] == raw_url:
                allowed_false = license_item.get("copyrighted_exact_values", [])
                if allowed_false and record.get("copyrighted") not in allowed_false:
                    raise ReviewQueueError(f"{context}: copyrighted flag violates exact policy")
                matched = (license_item, f"policy:{license_item['canonical_id']}:{raw_name}|{raw_url}")
                break
        if matched:
            break
    if matched is None:
        for exception in policy.get("legacy_exceptions", []):
            if (
                exception.get("context") == policy_context
                and exception.get("source_provider") == record.get("source_provider")
                and exception.get("pageid") == record.get("pageid")
                and exception.get("source_group") == record.get("source_group")
                and exception.get("raw_name") == raw_name
                and exception.get("raw_url") == raw_url
            ):
                choices = [
                    item for item in policy["licenses"]
                    if item["canonical_id"] == exception.get("canonical_id")
                ]
                if len(choices) != 1:
                    raise ReviewQueueError(f"{context}: legacy exception references unknown license")
                matched = (choices[0], f"exception:{exception['exception_id']}")
                break
    if matched is None:
        raise ReviewQueueError(f"{context}: raw license name/url pair is not exactly allowlisted")

    license_item, binding_id = matched
    canonical_name = license_item["canonical_name"]
    canonical_url = license_item["canonical_url"]
    if require_canonical_fields:
        expected = {
            "license_canonical_id": license_item["canonical_id"],
            "license_canonical_name": canonical_name,
            "license_canonical_url": canonical_url,
            "license_binding_id": binding_id,
            "license_policy_sha256": policy_sha,
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise ReviewQueueError(f"{context}: {field} is not bound to the shared policy")
    evidence_url = source_page if not canonical_url else canonical_url
    basis = "public_domain_commons_file_page_fallback" if not canonical_url else binding_id
    return canonical_name, evidence_url, basis, raw_name, raw_url


def _image_facts(path: Path, context: str) -> tuple[int, int, str, str]:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            image_format = image.format
            mime = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(image_format or "", "unsupported")
            rgb = image.convert("RGB")
            pixels = rgb.load()
            bits: list[int] = []
            for y in range(8):
                source_y = min(height - 1, ((2 * y + 1) * height) // 16)
                for x in range(8):
                    left_x = min(width - 1, ((2 * x + 1) * width) // 18)
                    right_x = min(width - 1, ((2 * x + 3) * width) // 18)
                    left = pixels[left_x, source_y]
                    right = pixels[right_x, source_y]
                    left_luma = 299 * left[0] + 587 * left[1] + 114 * left[2]
                    right_luma = 299 * right[0] + 587 * right[1] + 114 * right[2]
                    bits.append(1 if left_luma > right_luma else 0)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ReviewQueueError(f"{context}: cannot decode image: {exc}") from exc
    value = sum(bit << (63 - index) for index, bit in enumerate(bits))
    return width, height, mime, f"{value:016x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _validate_holdouts(
    records: Sequence[Mapping[str, Any]],
    manifest_root: Path,
    policy: Mapping[str, Any],
    policy_sha: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    observed_pageids: set[int] = set()
    observed_hashes: set[str] = set()
    observed_groups: set[str] = set()

    for index, record in enumerate(records, start=1):
        context = f"holdout manifest line {index}"
        pageid = _require_pageid(record, context)
        if pageid not in EXPECTED_HOLDOUT_PAGEIDS:
            continue
        if pageid in observed_pageids:
            raise ReviewQueueError(f"{context}: duplicate permanent holdout pageid {pageid}")
        _require_exact(record, "source_provider", "Wikimedia Commons", context)
        _require_exact(record, "source_group", f"commons:{pageid}", context)
        _require_exact(record, "domain", "print_demo_source", context)
        _require_exact(record, "split", "print_demo", context)
        class_hint = _require_string(record, "class_id", context)
        if class_hint not in ALLOWED_CLASS_HINTS:
            raise ReviewQueueError(f"{context}: unsupported class_id {class_hint!r}")
        source_page = _require_https_url(
            _require_string(record, "source_page", context),
            "source_page",
            context,
            host="commons.wikimedia.org",
        )
        creator = _require_string(record, "artist", context)
        title = _require_string(record, "title", context)
        filename = _require_string(record, "filename", context)
        local_file, local_path = _resolve_bound_file(manifest_root, filename, context)
        expected_sha = _validate_sha(record.get("download_sha256"), "download_sha256", context)
        actual_sha = _sha256_file(local_file)
        if actual_sha != expected_sha:
            raise ReviewQueueError(
                f"{context}: SHA-256 mismatch for {local_path}: expected {expected_sha}, got {actual_sha}"
            )
        source_group = f"commons:{pageid}"
        if expected_sha in observed_hashes:
            raise ReviewQueueError(f"{context}: duplicate permanent holdout SHA-256 {expected_sha}")
        if source_group in observed_groups:
            raise ReviewQueueError(f"{context}: duplicate permanent holdout source_group {source_group}")
        width, height, decoded_mime, dhash64 = _image_facts(local_file, context)
        constraints = policy["image_constraints"]
        if decoded_mime not in constraints["allowed_mime"]:
            raise ReviewQueueError(f"{context}: decoded MIME is not allowlisted")
        if min(width, height) < constraints["minimum_downloaded_side"]:
            raise ReviewQueueError(f"{context}: downloaded holdout is below minimum dimensions")
        canonical_name, license_url, license_url_basis, raw_name, raw_url = _license_binding(
            record,
            source_page,
            context,
            policy,
            policy_sha,
            "legacy_holdout_manifest",
            require_canonical_fields=False,
        )

        selected.append(
            {
                "schema_version": HOLDOUT_SCHEMA_VERSION,
                "asset": f"wikimedia:{pageid}@sha256:{expected_sha}",
                "pageid": pageid,
                "title": title,
                "local_path": local_path,
                "source_url": source_page,
                "creator": creator,
                "license": canonical_name,
                "license_url": license_url,
                "license_url_basis": license_url_basis,
                "license_raw_name": raw_name,
                "license_raw_url": raw_url,
                "license_policy_sha256": policy_sha,
                "class_hint": class_hint,
                "source_group": source_group,
                "sha256": expected_sha,
                "download_width": width,
                "download_height": height,
                "download_mime": decoded_mime,
                "dhash64": dhash64,
                "queue_membership": "EXCLUDED_PERMANENT_PRINT_HOLDOUT",
                "candidate_review_eligible": False,
                "training_eligible": False,
            }
        )
        observed_pageids.add(pageid)
        observed_hashes.add(expected_sha)
        observed_groups.add(source_group)

    missing = EXPECTED_HOLDOUT_PAGEIDS - observed_pageids
    if missing or len(selected) != len(EXPECTED_HOLDOUT_PAGEIDS):
        raise ReviewQueueError(
            "holdout manifest must contain exactly the six permanent print holdouts; "
            f"missing={sorted(missing)}, selected={len(selected)}"
        )
    return sorted(selected, key=lambda item: (item["class_hint"], item["pageid"]))


def _validate_candidates(
    records: Sequence[Mapping[str, Any]],
    manifest_root: Path,
    holdouts: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    policy_sha: str,
    holdout_threshold: int,
    candidate_threshold: int,
) -> list[dict[str, Any]]:
    holdout_pageids = {item["pageid"] for item in holdouts}
    holdout_hashes = {item["sha256"] for item in holdouts}
    holdout_groups = {item["source_group"] for item in holdouts}
    observed_assets: set[str] = set()
    observed_pageids: set[int] = set()
    observed_hashes: set[str] = set()
    observed_groups: set[str] = set()
    observed_dhashes: list[str] = []
    holdout_dhashes = [item["dhash64"] for item in holdouts]
    candidates: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        context = f"staging manifest line {index}"
        _require_exact(record, "schema_version", "rootscope.wikimedia_candidate.v1", context)
        _require_exact(record, "source_provider", "Wikimedia Commons", context)
        _require_exact(record, "domain", "natural_web_candidate", context)
        _require_exact(record, "split", "UNASSIGNED_DO_NOT_TRAIN", context)
        _require_exact(
            record,
            "review_status",
            "pending_human_visual_and_license_review",
            context,
        )
        _require_exact(record, "training_eligible", False, context)
        _require_exact(record, "print_eligible", False, context)
        _require_exact(
            record,
            "candidate_label_status",
            "query_or_category_derived_unverified",
            context,
        )
        _require_exact(
            record,
            "rights_review_status",
            "machine_allowlist_pass_human_file_page_and_non_copyright_rights_review_pending",
            context,
        )
        _require_exact(
            record,
            "species_hint_status",
            "acquisition_hint_not_a_reviewed_species_or_shape_label",
            context,
        )
        pageid = _require_pageid(record, context)
        source_group = _require_string(record, "source_group", context)
        if source_group != f"commons:{pageid}":
            raise ReviewQueueError(
                f"{context}: source_group must be exactly 'commons:{pageid}'"
            )
        class_hint = _require_string(record, "class_id", context)
        if class_hint not in ALLOWED_CLASS_HINTS:
            raise ReviewQueueError(f"{context}: unsupported class_id {class_hint!r}")
        title = _require_string(record, "title", context)
        creator = _require_string(record, "artist", context)
        creator_group = _require_string(record, "creator_group", context)
        expected_creator_group = "commons-creator:" + hashlib.sha256(creator.encode("utf-8")).hexdigest()[:16]
        if creator_group != expected_creator_group:
            raise ReviewQueueError(f"{context}: creator_group does not match exact artist hash")
        acquisition_mode = _require_string(record, "acquisition_mode", context)
        if acquisition_mode not in {"category", "search"}:
            raise ReviewQueueError(f"{context}: unsupported acquisition_mode")
        acquisition_query = _require_string(record, "acquisition_query", context)
        species_hint = _require_string(record, "species_hint", context)
        source_page = _require_https_url(
            _require_string(record, "source_page", context),
            "source_page",
            context,
            host="commons.wikimedia.org",
        )
        _require_https_url(
            _require_string(record, "original_url", context), "original_url", context, host="upload.wikimedia.org"
        )
        _require_https_url(
            _require_string(record, "download_url", context), "download_url", context, host="upload.wikimedia.org"
        )
        filename = _require_string(record, "filename", context)
        local_file, local_path = _resolve_bound_file(manifest_root, filename, context)
        expected_sha = _validate_sha(record.get("download_sha256"), "download_sha256", context)
        actual_sha = _sha256_file(local_file)
        if actual_sha != expected_sha:
            raise ReviewQueueError(
                f"{context}: SHA-256 mismatch for {local_path}: expected {expected_sha}, got {actual_sha}"
            )
        width, height, decoded_mime, recomputed_dhash = _image_facts(local_file, context)
        constraints = policy["image_constraints"]
        if decoded_mime not in constraints["allowed_mime"]:
            raise ReviewQueueError(f"{context}: decoded MIME is not allowlisted")
        if min(width, height) < constraints["minimum_downloaded_side"]:
            raise ReviewQueueError(f"{context}: downloaded candidate is below minimum dimensions")
        if type(record.get("original_width")) is not int or type(record.get("original_height")) is not int:
            raise ReviewQueueError(f"{context}: original dimensions must be integers")
        if min(record["original_width"], record["original_height"]) < constraints["minimum_original_side"]:
            raise ReviewQueueError(f"{context}: original metadata dimensions are below policy")
        expected_image_fields = {
            "mime": decoded_mime,
            "download_mime": decoded_mime,
            "download_width": width,
            "download_height": height,
            "download_bytes": local_file.stat().st_size,
            "dhash64_algorithm": constraints["dhash_algorithm"],
            "dhash64": recomputed_dhash,
        }
        for field, value in expected_image_fields.items():
            if type(record.get(field)) is not type(value) or record.get(field) != value:
                raise ReviewQueueError(f"{context}: {field} is not bound to decoded payload")
        dhash64 = _validate_dhash(record.get("dhash64"), context)
        if min((_hamming(dhash64, item) for item in holdout_dhashes), default=64) <= holdout_threshold:
            raise ReviewQueueError(f"{context}: candidate is too close to permanent print holdout")
        if min((_hamming(dhash64, item) for item in observed_dhashes), default=64) <= candidate_threshold:
            raise ReviewQueueError(f"{context}: candidate is too close to an earlier candidate")
        canonical_name, license_url, license_url_basis, raw_name, raw_url = _license_binding(
            record,
            source_page,
            context,
            policy,
            policy_sha,
            "staging_manifest",
            require_canonical_fields=True,
        )
        asset = f"wikimedia:{pageid}@sha256:{expected_sha}"

        if pageid in holdout_pageids or expected_sha in holdout_hashes or source_group in holdout_groups:
            raise ReviewQueueError(
                f"{context}: candidate overlaps a permanent print holdout by pageid, SHA, or source_group"
            )
        duplicate_checks = (
            (asset, observed_assets, "asset"),
            (pageid, observed_pageids, "pageid"),
            (expected_sha, observed_hashes, "SHA-256"),
            (source_group, observed_groups, "source_group"),
        )
        for value, observed, label in duplicate_checks:
            if value in observed:
                raise ReviewQueueError(f"{context}: duplicate candidate {label}: {value}")

        candidates.append(
            {
                "schema_version": SCHEMA_VERSION,
                "asset": asset,
                "pageid": pageid,
                "title": title,
                "local_path": local_path,
                "source_url": source_page,
                "creator": creator,
                "license": canonical_name,
                "license_url": license_url,
                "license_url_basis": license_url_basis,
                "license_raw_name": raw_name,
                "license_raw_url": raw_url,
                "license_policy_sha256": policy_sha,
                "class_hint": class_hint,
                "class_hint_status": "ACQUISITION_HINT_ONLY_UNREVIEWED",
                "creator_group": creator_group,
                "acquisition_mode": acquisition_mode,
                "acquisition_query": acquisition_query,
                "species_hint": species_hint,
                "species_hint_status": "ACQUISITION_HINT_ONLY_UNREVIEWED",
                "source_group": source_group,
                "sha256": expected_sha,
                "dhash64": dhash64,
                "download_width": width,
                "download_height": height,
                "download_mime": decoded_mime,
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
        )
        observed_assets.add(asset)
        observed_pageids.add(pageid)
        observed_hashes.add(expected_sha)
        observed_groups.add(source_group)
        observed_dhashes.append(dhash64)

    if not candidates:
        raise ReviewQueueError("staging manifest contains no review candidates")
    return sorted(candidates, key=lambda item: (item["class_hint"], item["pageid"], item["sha256"]))


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        )
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _md(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _candidate_markdown(candidates: Sequence[Mapping[str, Any]], input_sha: str) -> bytes:
    lines = [
        "# Wikimedia 人工视觉与版权双审队列",
        "",
        "> 状态：**UNREVIEWED / NOT TRAIN READY**。本文件不构成标签、版权许可结论或训练授权。",
        "",
        f"- 输入 staging manifest SHA-256：`{input_sha}`",
        f"- 候选数量：{len(candidates)}",
        "- 人工填写字段：`visual_decision`、`rights_decision`、`target_class`、`reviewed_source_group`、`near_duplicate_family`、`reviewer`、`notes`",
        "- Commons pageid 只是原始文件页标识，不等于独立拍摄源；同作者连续序列/近重复图必须人工合并后，才能计入 approved source-group。",
        "- 六张永久打印保留图不在本队列中，见 `PERMANENT_PRINT_HOLDOUTS.md`。",
        "",
    ]
    for index, item in enumerate(candidates, start=1):
        image_path = "../" + str(item["local_path"])
        lines.extend(
            [
                f"## {index:04d} · {_md(item['asset'])}",
                "",
                f'<img src="{_md(image_path)}" width="320" alt="unreviewed candidate {index:04d}">',
                "",
                f"- 本地文件：`{_md(item['local_path'])}`",
                f"- 来源：[Wikimedia Commons]({_md(item['source_url'])})（pageid `{item['pageid']}`）",
                f"- 作者/创作者：{_md(item['creator'])}",
                f"- creator_group（仅合并线索，不是 reviewed_source_group）：`{_md(item['creator_group'])}`",
                f"- 许可证：{_md(item['license'])} · [证据链接]({_md(item['license_url'])})",
                f"- 采集类别提示（不是标签）：`{_md(item['class_hint'])}`",
                f"- 采集方式/查询（仅 acquisition hint）：`{_md(item['acquisition_mode'])}` / `{_md(item['acquisition_query'])}`",
                f"- species_hint（不是物种或类别标签）：`{_md(item['species_hint'])}`",
                f"- source_group：`{_md(item['source_group'])}`",
                f"- SHA-256：`{item['sha256']}` · dHash64：`{item['dhash64']}`",
                "- visual_decision：",
                "- rights_decision：",
                "- target_class：",
                "- reviewed_source_group：",
                "- near_duplicate_family：",
                "- reviewer：",
                "- notes：",
                "",
            ]
        )
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _holdout_markdown(holdouts: Sequence[Mapping[str, Any]], input_sha: str) -> bytes:
    lines = [
        "# 永久打印保留图（禁止进入候选审阅/训练）",
        "",
        "> 这六张图是永久 print holdouts。它们与候选队列物理分离，`candidate_review_eligible=false`、`training_eligible=false`。",
        "",
        f"- 输入 holdout manifest SHA-256：`{input_sha}`",
        f"- 固定数量：{len(holdouts)}",
        "",
        "| pageid | class hint | local path | source group | SHA-256 |",
        "|---:|---|---|---|---|",
    ]
    for item in holdouts:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["pageid"]),
                    _md(item["class_hint"]),
                    f"`{_md(item['local_path'])}`",
                    f"`{_md(item['source_group'])}`",
                    f"`{item['sha256']}`",
                ]
            )
            + " |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _validate_integrity_audit(
    audit_path: Path,
    staging_manifest: Path,
    staging_sha: str,
    policy: Mapping[str, Any],
    policy_sha: str,
) -> tuple[str, int, int]:
    audit, audit_sha = _load_json_object(audit_path, "staging integrity audit")
    if audit.get("schema_version") != "rootscope.wikimedia_staging_integrity_audit.v2":
        raise ReviewQueueError("integrity audit schema is not v2")
    if audit.get("result") != "PASS_STAGING_INTEGRITY_NOT_TRAIN_READY":
        raise ReviewQueueError("integrity audit is not PASS_STAGING_INTEGRITY_NOT_TRAIN_READY")
    if audit.get("failure_count") != 0 or audit.get("failures") != []:
        raise ReviewQueueError("integrity audit PASS has a non-empty failure ledger")
    if audit.get("manifest_sha256") != staging_sha:
        raise ReviewQueueError("integrity audit is stale for the staging manifest")
    if audit.get("license_policy_sha256") != policy_sha:
        raise ReviewQueueError("integrity audit is bound to a different license policy")
    summary_path = staging_manifest.parent / "summary.json"
    try:
        summary_sha = _sha256_file(summary_path)
    except OSError as exc:
        raise ReviewQueueError(f"cannot read staging summary: {exc}") from exc
    if audit.get("summary_sha256") != summary_sha:
        raise ReviewQueueError("integrity audit is stale for the staging summary")
    collector_path = Path(__file__).resolve().parent / "collect_wikimedia_candidates.ps1"
    if audit.get("collector_script_sha256") != _sha256_file(collector_path):
        raise ReviewQueueError("integrity audit is stale for the collector implementation")
    constraints = audit.get("image_constraints")
    if not isinstance(constraints, dict):
        raise ReviewQueueError("integrity audit has no image constraints")
    expected_constraints = policy["image_constraints"]
    for field in ("minimum_original_side", "minimum_downloaded_side", "dhash_algorithm"):
        if constraints.get(field) != expected_constraints.get(field):
            raise ReviewQueueError(f"integrity audit {field} does not match policy")
    if constraints.get("allowed_mime") != expected_constraints.get("allowed_mime"):
        raise ReviewQueueError("integrity audit MIME allowlist does not match policy")
    thresholds = audit.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ReviewQueueError("integrity audit has no dHash thresholds")
    holdout = thresholds.get("holdout_dhash_reject_at_or_below")
    candidate = thresholds.get("candidate_dhash_reject_at_or_below")
    if type(holdout) is not int or not 1 <= holdout <= 50:
        raise ReviewQueueError("invalid holdout dHash threshold in audit")
    if type(candidate) is not int or not 0 <= candidate <= 20:
        raise ReviewQueueError("invalid candidate dHash threshold in audit")
    return audit_sha, holdout, candidate


def build_review_queue(
    staging_manifest: Path | str,
    holdout_manifest: Path | str,
    output_dir: Path | str | None = None,
    integrity_audit: Path | str | None = None,
    license_policy: Path | str = DEFAULT_LICENSE_POLICY,
) -> dict[str, Any]:
    staging_manifest = Path(staging_manifest)
    holdout_manifest = Path(holdout_manifest)
    output_dir = Path(output_dir) if output_dir is not None else staging_manifest.parent / "review"
    integrity_audit = Path(integrity_audit) if integrity_audit is not None else staging_manifest.parent / "integrity_audit.json"
    license_policy = Path(license_policy)

    staging_records, staging_sha = _load_jsonl(staging_manifest, "staging manifest")
    holdout_records, holdout_sha = _load_jsonl(holdout_manifest, "holdout manifest")
    policy, policy_sha = _load_policy(license_policy)
    audit_sha, holdout_threshold, candidate_threshold = _validate_integrity_audit(
        integrity_audit, staging_manifest, staging_sha, policy, policy_sha
    )
    holdouts = _validate_holdouts(holdout_records, holdout_manifest.parent, policy, policy_sha)
    candidates = _validate_candidates(
        staging_records,
        staging_manifest.parent,
        holdouts,
        policy,
        policy_sha,
        holdout_threshold,
        candidate_threshold,
    )

    candidate_jsonl = _jsonl_bytes(candidates)
    candidate_md = _candidate_markdown(candidates, staging_sha)
    holdout_jsonl = _jsonl_bytes(holdouts)
    holdout_md = _holdout_markdown(holdouts, holdout_sha)
    class_counts = dict(sorted(Counter(item["class_hint"] for item in candidates).items()))
    license_counts = dict(sorted(Counter(item["license"] for item in candidates).items()))
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "status": "UNREVIEWED_NOT_TRAIN_READY",
        "candidate_count": len(candidates),
        "raw_commons_page_count": len(candidates),
        "approved_source_group_count": 0,
        "source_group_counting_rule": (
            "raw Commons page count MUST NOT be reported as approved source-group count; "
            "human review must merge same-shoot sequences and near-duplicate families first"
        ),
        "candidate_class_hint_counts": class_counts,
        "candidate_license_counts": license_counts,
        "candidate_creator_group_count": len({item["creator_group"] for item in candidates}),
        "permanent_print_holdout_count": len(holdouts),
        "permanent_print_holdout_pageids": sorted(EXPECTED_HOLDOUT_PAGEIDS),
        "dhash_thresholds": {
            "holdout_reject_at_or_below": holdout_threshold,
            "candidate_reject_at_or_below": candidate_threshold,
        },
        "inputs": {
            "staging_manifest_filename": staging_manifest.name,
            "staging_manifest_sha256": staging_sha,
            "holdout_manifest_filename": holdout_manifest.name,
            "holdout_manifest_sha256": holdout_sha,
            "integrity_audit_filename": integrity_audit.name,
            "integrity_audit_sha256": audit_sha,
            "license_policy_filename": license_policy.name,
            "license_policy_sha256": policy_sha,
        },
        "outputs": {
            "candidate_review_queue.jsonl": _sha256_bytes(candidate_jsonl),
            "CANDIDATE_REVIEW_QUEUE.md": _sha256_bytes(candidate_md),
            "permanent_print_holdouts.jsonl": _sha256_bytes(holdout_jsonl),
            "PERMANENT_PRINT_HOLDOUTS.md": _sha256_bytes(holdout_md),
        },
        "invariants": {
            "all_candidates_unreviewed": True,
            "all_candidate_decision_fields_empty": True,
            "raw_page_count_is_not_approved_source_group_count": True,
            "all_candidates_training_eligible_false": True,
            "all_candidates_print_eligible_false": True,
            "permanent_holdouts_excluded_from_candidate_queue": True,
        },
    }
    summary_json = _json_bytes(summary)

    # No output directory is touched until every input record and referenced file
    # has passed validation and every output has been rendered in memory.
    outputs = {
        "candidate_review_queue.jsonl": candidate_jsonl,
        "CANDIDATE_REVIEW_QUEUE.md": candidate_md,
        "permanent_print_holdouts.jsonl": holdout_jsonl,
        "PERMANENT_PRINT_HOLDOUTS.md": holdout_md,
        "review_queue_summary.json": summary_json,
    }
    for filename, payload in outputs.items():
        _atomic_write(output_dir / filename, payload)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-manifest", type=Path, default=DEFAULT_STAGING_MANIFEST)
    parser.add_argument("--holdout-manifest", type=Path, default=DEFAULT_HOLDOUT_MANIFEST)
    parser.add_argument("--integrity-audit", type=Path, default=DEFAULT_INTEGRITY_AUDIT)
    parser.add_argument("--license-policy", type=Path, default=DEFAULT_LICENSE_POLICY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: <staging manifest directory>/review",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = build_review_queue(
            args.staging_manifest,
            args.holdout_manifest,
            args.output_dir,
            args.integrity_audit,
            args.license_policy,
        )
    except ReviewQueueError as exc:
        print(f"FAIL_LOCKED: {exc}", file=sys.stderr)
        return 2
    print(
        "PASS: wrote deterministic UNREVIEWED queue "
        f"({summary['candidate_count']} candidates, "
        f"{summary['permanent_print_holdout_count']} permanent holdouts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
