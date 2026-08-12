#!/usr/bin/env python3
"""Fail-closed AI visual triage for the frozen RootScope candidate pool.

This tool is deliberately separate from the human-review journal.  It may emit
model suggestions and a low-confidence queue, but it never writes a dataset
manifest, assigns a split, grants training/print eligibility, or claims human
review / DATA_LOCKED authority.

Two inference paths are supported:

* ``external_scores`` ingests a strict, fully bound probability JSONL.  The
  receipt explicitly says that this is not proof that the named model ran.
* ``transformers_siglip`` loads a local Transformers/SigLIP-compatible model
  with ``local_files_only=True`` and performs zero-shot inference in-process.

Production inputs and the production output directory are frozen by the policy.
Unit fixtures require the explicit ``--fixture-mode`` boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError


SCRIPT_PATH = Path(__file__).resolve()
ADVENTUREX_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_STAGING = ADVENTUREX_ROOT / "datasets" / "desert_plants_wikimedia_staging_e0"
DEFAULT_MANIFEST = DEFAULT_STAGING / "manifest.jsonl"
DEFAULT_QUEUE = DEFAULT_STAGING / "review" / "candidate_review_queue.jsonl"
DEFAULT_QUEUE_SUMMARY = DEFAULT_STAGING / "review" / "review_queue_summary.json"
DEFAULT_INTEGRITY_AUDIT = DEFAULT_STAGING / "integrity_audit.json"
DEFAULT_CLASS_CONTRACT = ADVENTUREX_ROOT / "rootscope" / "configs" / "class_contract.json"
DEFAULT_POLICY = SCRIPT_PATH.with_name("ai_visual_triage_policy_v1.json")
DEFAULT_OUTPUT_DIR = DEFAULT_STAGING / "review" / "ai_triage_v1"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}

AUTHORITY = {
    "human_review": False,
    "dataset_manifest_write": False,
    "training_eligibility": False,
    "split_assignment": False,
    "print_eligibility": False,
    "data_locked": False,
}

POLICY_FIELDS = {
    "schema_version",
    "queue_schema_version",
    "manifest_schema_version",
    "model_output_schema_version",
    "result_schema_version",
    "low_confidence_schema_version",
    "stats_schema_version",
    "receipt_schema_version",
    "production_input_roots",
    "expected_candidate_count",
    "expected_acquisition_hint_counts",
    "target_classes",
    "image_contract",
    "inference",
    "thresholds",
    "output_contract",
    "authority",
    "explicit_non_claims",
}

QUEUE_FIELDS = {
    "acquisition_mode",
    "acquisition_query",
    "asset",
    "class_hint",
    "class_hint_status",
    "creator",
    "creator_group",
    "dhash64",
    "download_height",
    "download_mime",
    "download_width",
    "license",
    "license_policy_sha256",
    "license_raw_name",
    "license_raw_url",
    "license_url",
    "license_url_basis",
    "local_path",
    "near_duplicate_family",
    "notes",
    "pageid",
    "print_eligible",
    "review_status",
    "reviewed_source_group",
    "reviewer",
    "rights_decision",
    "schema_version",
    "sha256",
    "source_group",
    "source_url",
    "species_hint",
    "species_hint_status",
    "split",
    "target_class",
    "title",
    "training_eligible",
    "visual_decision",
}

MANIFEST_REQUIRED_FIELDS = {
    "schema_version",
    "class_id",
    "domain",
    "split",
    "review_status",
    "training_eligible",
    "print_eligible",
    "source_provider",
    "source_group",
    "pageid",
    "source_page",
    "download_url",
    "artist",
    "license_canonical_name",
    "license_canonical_url",
    "filename",
    "download_sha256",
    "download_width",
    "download_height",
    "download_mime",
}

MODEL_OUTPUT_FIELDS = {
    "schema_version",
    "asset",
    "candidate_sha256",
    "model_id",
    "model_artifact_sha256",
    "prompt_set_sha256",
    "class_probabilities",
}

RESULT_FIELDS = {
    "schema_version",
    "mode",
    "asset",
    "pageid",
    "candidate_sha256",
    "image_path",
    "model_id",
    "model_artifact_sha256",
    "prompt_set_sha256",
    "class_probabilities",
    "top1_class",
    "top1_probability",
    "top2_class",
    "top2_probability",
    "top1_top2_margin",
    "triage_decision",
    "low_confidence_reasons",
    "acquisition_class_hint",
    "acquisition_hint_agrees",
    "authority",
    "explicit_non_claims",
}

LOW_CONFIDENCE_FIELDS = {
    "schema_version",
    "mode",
    "asset",
    "pageid",
    "candidate_sha256",
    "suggested_class",
    "top1_probability",
    "top1_top2_margin",
    "low_confidence_reasons",
    "result_sha256",
    "authority",
    "explicit_non_claims",
}


class TriageError(RuntimeError):
    """A fail-closed input, inference or output-contract failure."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _parse_json(raw: bytes, context: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TriageError(f"{context} is not strict UTF-8 JSON: {exc}") from exc


def _read_json(path: Path, context: str) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TriageError(f"cannot read {context}: {exc}") from exc
    return _parse_json(raw, context), raw


def _read_jsonl(path: Path, context: str) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TriageError(f"cannot read {context}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TriageError(f"{context} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise TriageError(f"{context} line {line_number} is blank")
        value = _parse_json(line.encode("utf-8"), f"{context} line {line_number}")
        if not isinstance(value, dict):
            raise TriageError(f"{context} line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise TriageError(f"{context} is empty")
    return rows, raw


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_reparse(stat_result: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(stat_result, "st_file_attributes", 0) & flag)


def _fingerprint(stat_result: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        stat.S_IFMT(stat_result.st_mode),
        stat_result.st_size,
        getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)),
        getattr(stat_result, "st_ctime_ns", int(stat_result.st_ctime * 1_000_000_000)),
        stat_result.st_dev,
        stat_result.st_ino,
    )


def _open_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.S_IFMT(stat_result.st_mode),
        stat_result.st_size,
        getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)),
        stat_result.st_dev,
        stat_result.st_ino,
    )


def _regular_file_lstat(path: Path, context: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise TriageError(f"cannot inspect {context}: {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
        raise TriageError(f"{context} must not be a symlink, junction, or reparse point: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise TriageError(f"{context} is not a regular file: {path}")
    return info


def _sha256_file(path: Path) -> str:
    before = _regular_file_lstat(path, "hashed input")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TriageError(f"cannot open hashed input: {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if _open_identity(opened) != _open_identity(before):
            raise TriageError(f"hashed input changed while it was opened: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after_handle = os.fstat(handle.fileno())
            if _fingerprint(after_handle) != _fingerprint(opened):
                raise TriageError(f"hashed input changed while it was hashed: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after_path = _regular_file_lstat(path, "hashed input")
    if _fingerprint(after_path) != _fingerprint(before):
        raise TriageError(f"hashed input changed while it was hashed: {path}")
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _is_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _require_exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TriageError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        raise TriageError(
            f"{context} fields do not match strict schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _safe_relative_path(root: Path, value: Any, context: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TriageError(f"{context} must be a non-empty POSIX relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise TriageError(f"{context} is unsafe")
    for part in pure.parts:
        stem = part.split(".", 1)[0].upper()
        if ":" in part or part.endswith((" ", ".")) or stem in WINDOWS_RESERVED:
            raise TriageError(f"{context} is unsafe on Windows")
    try:
        candidate = root.joinpath(*pure.parts).resolve(strict=True)
        candidate.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise TriageError(f"{context} escapes the dataset root or is missing") from exc
    if not candidate.is_file():
        raise TriageError(f"{context} is not a regular file")
    return candidate, pure.as_posix()


def _artifact_root(path: Path) -> dict[str, Any]:
    try:
        source_info = os.lstat(path)
    except OSError as exc:
        raise TriageError(f"local model artifact does not exist: {path}") from exc
    if stat.S_ISLNK(source_info.st_mode) or _is_reparse(source_info):
        raise TriageError("local model artifact must not be a symlink, junction, or reparse point")
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise TriageError(f"local model artifact does not exist: {path}") from exc
    if root.is_file():
        size = root.stat().st_size
        if size <= 0:
            raise TriageError("local model artifact file is empty")
        sha = _sha256_file(root)
        return {
            "kind": "file",
            "sha256": sha,
            "file_count": 1,
            "byte_count": size,
            "entries_sha256": _sha256_bytes(
                _canonical_bytes([{"path": root.name, "bytes": size, "sha256": sha}])
            ),
        }
    if not root.is_dir():
        raise TriageError("local model artifact must be a regular file or directory")
    entries: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        try:
            candidate_info = os.lstat(candidate)
        except OSError as exc:
            raise TriageError(f"cannot inspect local model entry: {candidate}: {exc}") from exc
        if stat.S_ISLNK(candidate_info.st_mode) or _is_reparse(candidate_info):
            raise TriageError(
                "model directory links/reparse points are not allowed; copy a self-contained model directory"
            )
        if stat.S_ISDIR(candidate_info.st_mode):
            continue
        if not stat.S_ISREG(candidate_info.st_mode):
            raise TriageError("model directory contains a non-regular entry")
        relative = candidate.relative_to(root).as_posix()
        size = candidate_info.st_size
        entries.append({"path": relative, "bytes": size, "sha256": _sha256_file(candidate)})
    if not entries or sum(int(item["bytes"]) for item in entries) <= 0:
        raise TriageError("local model artifact directory has no non-empty file payload")
    entries_payload = _canonical_bytes(entries)
    return {
        "kind": "directory",
        "sha256": _sha256_bytes(entries_payload),
        "file_count": len(entries),
        "byte_count": sum(int(item["bytes"]) for item in entries),
        "entries_sha256": _sha256_bytes(entries_payload),
    }


@dataclass(frozen=True)
class TriageConfig:
    queue_path: Path = DEFAULT_QUEUE
    manifest_path: Path = DEFAULT_MANIFEST
    queue_summary_path: Path = DEFAULT_QUEUE_SUMMARY
    integrity_audit_path: Path = DEFAULT_INTEGRITY_AUDIT
    class_contract_path: Path = DEFAULT_CLASS_CONTRACT
    policy_path: Path = DEFAULT_POLICY
    output_dir: Path = DEFAULT_OUTPUT_DIR
    model_path: Path = Path()
    model_id: str = ""
    backend: str = "external_scores"
    model_output_path: Path | None = None
    fixture_mode: bool = False
    device: str = "auto"
    batch_size: int = 8


class AIVisualTriage:
    """Validate frozen inputs, ingest/infer probabilities, and render suggestions."""

    def __init__(self, config: TriageConfig) -> None:
        self.config = config
        self.queue_path = Path(config.queue_path)
        self.manifest_path = Path(config.manifest_path)
        self.queue_summary_path = Path(config.queue_summary_path)
        self.integrity_audit_path = Path(config.integrity_audit_path)
        self.class_contract_path = Path(config.class_contract_path)
        self.policy_path = Path(config.policy_path)
        self.output_dir = Path(config.output_dir)
        self.model_path = Path(config.model_path)
        self.model_output_path = Path(config.model_output_path) if config.model_output_path else None
        self.production_mode = self._is_production_configuration()
        self._enforce_mode_boundary()

        policy_value, policy_raw = _read_json(self.policy_path, "AI triage policy")
        self.policy = self._validate_policy(policy_value)
        self.policy_sha256 = _sha256_bytes(policy_raw)
        self.implementation_sha256 = _sha256_file(SCRIPT_PATH)
        self.classes = list(self.policy["target_classes"])
        self.prompt_set_sha256 = _sha256_bytes(
            _canonical_bytes(self.policy["inference"]["prompts"])
        )
        self.model_artifact = _artifact_root(self.model_path)
        self._validate_model_configuration()

        self.queue_rows: list[dict[str, Any]] = []
        self.manifest_rows: list[dict[str, Any]] = []
        self.image_records: list[dict[str, Any]] = []
        self.input_roots: dict[str, str] = {}
        self.image_payload_set_sha256 = ""

    def _is_production_configuration(self) -> bool:
        pairs = (
            (self.queue_path, DEFAULT_QUEUE),
            (self.manifest_path, DEFAULT_MANIFEST),
            (self.queue_summary_path, DEFAULT_QUEUE_SUMMARY),
            (self.integrity_audit_path, DEFAULT_INTEGRITY_AUDIT),
            (self.class_contract_path, DEFAULT_CLASS_CONTRACT),
            (self.policy_path, DEFAULT_POLICY),
            (self.output_dir, DEFAULT_OUTPUT_DIR),
        )
        return not self.config.fixture_mode and all(
            left.resolve(strict=False) == right.resolve(strict=False) for left, right in pairs
        )

    def _enforce_mode_boundary(self) -> None:
        if not self.config.fixture_mode and not self.production_mode:
            raise TriageError("custom inputs, policy, or output require explicit --fixture-mode")
        if self.config.fixture_mode and self.output_dir.resolve(strict=False) == DEFAULT_OUTPUT_DIR.resolve(strict=False):
            raise TriageError("fixture mode may not write the production ai_triage_v1 directory")
        human_dir = self.queue_path.parent / "human_decisions"
        output_resolved = self.output_dir.resolve(strict=False)
        human_resolved = human_dir.resolve(strict=False)
        if output_resolved == human_resolved or human_resolved in output_resolved.parents:
            raise TriageError("AI triage output may not be the human_decisions directory or its descendant")

    def _validate_policy(self, value: Any) -> dict[str, Any]:
        policy = _require_exact_keys(value, POLICY_FIELDS, "AI triage policy")
        if policy["schema_version"] != "rootscope.ai_visual_triage_policy.v1":
            raise TriageError("unsupported AI triage policy schema")
        versions = {
            "queue_schema_version": "rootscope.wikimedia_human_review_queue.v1",
            "manifest_schema_version": "rootscope.wikimedia_candidate.v1",
            "model_output_schema_version": "rootscope.ai_visual_model_output.v1",
            "result_schema_version": "rootscope.ai_visual_triage_result.v1",
            "low_confidence_schema_version": "rootscope.ai_visual_low_confidence.v1",
            "stats_schema_version": "rootscope.ai_visual_triage_stats.v1",
            "receipt_schema_version": "rootscope.ai_visual_triage_receipt.v1",
        }
        for field, expected in versions.items():
            if policy[field] != expected:
                raise TriageError(f"policy {field} is not the frozen version")
        roots = policy["production_input_roots"]
        expected_roots = {
            "candidate_review_queue_sha256",
            "staging_manifest_sha256",
            "review_queue_summary_sha256",
            "integrity_audit_sha256",
            "class_contract_sha256",
        }
        _require_exact_keys(roots, expected_roots, "policy production_input_roots")
        if any(type(value) is not str or HEX64.fullmatch(value) is None for value in roots.values()):
            raise TriageError("policy production input roots must be lowercase SHA-256")
        classes = policy["target_classes"]
        if classes != ["grass_clump", "low_shrub", "young_tree", "unknown"]:
            raise TriageError("policy target classes/order are not frozen")
        if type(policy["expected_candidate_count"]) is not int or policy["expected_candidate_count"] <= 0:
            raise TriageError("policy expected_candidate_count is invalid")
        hint_counts = policy["expected_acquisition_hint_counts"]
        if (
            not isinstance(hint_counts, dict)
            or set(hint_counts) != set(classes)
            or any(type(count) is not int or count < 0 for count in hint_counts.values())
            or sum(hint_counts.values()) != policy["expected_candidate_count"]
        ):
            raise TriageError("policy acquisition hint counts are invalid")
        image_contract = _require_exact_keys(
            policy["image_contract"],
            {"allowed_mime", "minimum_width", "minimum_height", "verify_decode"},
            "policy image_contract",
        )
        if (
            not isinstance(image_contract["allowed_mime"], list)
            or not image_contract["allowed_mime"]
            or any(type(value) is not str for value in image_contract["allowed_mime"])
            or type(image_contract["minimum_width"]) is not int
            or type(image_contract["minimum_height"]) is not int
            or image_contract["minimum_width"] < 1
            or image_contract["minimum_height"] < 1
            or image_contract["verify_decode"] is not True
        ):
            raise TriageError("policy image contract is invalid")
        inference = _require_exact_keys(
            policy["inference"],
            {"supported_backends", "model_id_pattern", "normalization", "prompts"},
            "policy inference",
        )
        if inference["supported_backends"] != ["external_scores", "transformers_siglip"]:
            raise TriageError("policy inference backends are not frozen")
        try:
            re.compile(inference["model_id_pattern"])
        except (TypeError, re.error) as exc:
            raise TriageError("policy model_id_pattern is invalid") from exc
        if inference["normalization"] != "softmax_over_frozen_prompts":
            raise TriageError("policy normalization is not frozen")
        if not isinstance(inference["prompts"], dict) or set(inference["prompts"]) != set(classes):
            raise TriageError("policy prompts must contain exactly the frozen classes")
        if any(type(prompt) is not str or not prompt.strip() for prompt in inference["prompts"].values()):
            raise TriageError("policy prompts must be non-empty strings")
        thresholds = _require_exact_keys(
            policy["thresholds"],
            {
                "minimum_top1_probability_by_class",
                "minimum_top1_top2_margin",
                "probability_sum_tolerance",
            },
            "policy thresholds",
        )
        per_class = thresholds["minimum_top1_probability_by_class"]
        if not isinstance(per_class, dict) or set(per_class) != set(classes):
            raise TriageError("policy per-class thresholds do not match the frozen classes")
        numeric_thresholds = list(per_class.values()) + [
            thresholds["minimum_top1_top2_margin"],
            thresholds["probability_sum_tolerance"],
        ]
        if any(not _is_number(item) or not 0 <= float(item) <= 1 for item in numeric_thresholds):
            raise TriageError("policy thresholds must be finite numbers in [0,1]")
        output = _require_exact_keys(
            policy["output_contract"],
            {
                "results_filename",
                "low_confidence_filename",
                "normalized_model_outputs_filename",
                "stats_filename",
                "receipt_filename",
                "status",
            },
            "policy output_contract",
        )
        expected_names = {
            "results_filename": "ai_visual_triage_results.jsonl",
            "low_confidence_filename": "low_confidence_queue.jsonl",
            "normalized_model_outputs_filename": "normalized_model_outputs.jsonl",
            "stats_filename": "ai_visual_triage_stats.json",
            "receipt_filename": "ai_visual_triage_receipt.json",
        }
        if any(output[key] != expected for key, expected in expected_names.items()):
            raise TriageError("policy output filenames are not frozen")
        if output["status"] != "AI_VISUAL_TRIAGE_COMPLETE_NOT_HUMAN_REVIEWED_NOT_DATA_LOCKED":
            raise TriageError("policy status is not the non-authoritative frozen status")
        if policy["authority"] != AUTHORITY:
            raise TriageError("policy must explicitly deny every authority")
        non_claims = policy["explicit_non_claims"]
        required_non_claims = {
            "HUMAN_REVIEWED",
            "VISUAL_LABEL_APPROVED",
            "RIGHTS_APPROVED",
            "DATA_LOCKED",
            "TRAIN_READY",
            "SPLIT_READY",
            "PRINT_ELIGIBLE",
            "MODEL_QUALIFIED",
        }
        if not isinstance(non_claims, list) or set(non_claims) != required_non_claims or len(non_claims) != len(required_non_claims):
            raise TriageError("policy explicit_non_claims are incomplete")
        return policy

    def _validate_model_configuration(self) -> None:
        inference = self.policy["inference"]
        if self.config.backend not in inference["supported_backends"]:
            raise TriageError("unsupported AI triage backend")
        if not isinstance(self.config.model_id, str) or re.fullmatch(
            inference["model_id_pattern"], self.config.model_id
        ) is None:
            raise TriageError("model_id does not match the frozen policy")
        if self.config.backend == "external_scores":
            if self.model_output_path is None:
                raise TriageError("external_scores requires --model-output-jsonl")
            if not self.model_output_path.is_file():
                raise TriageError("external model-output JSONL does not exist")
        elif self.model_output_path is not None:
            raise TriageError("transformers_siglip may not ingest an external score file")
        if type(self.config.batch_size) is not int or not 1 <= self.config.batch_size <= 64:
            raise TriageError("batch_size must be an integer in [1,64]")
        if self.config.device not in {"auto", "cpu", "cuda"}:
            raise TriageError("device must be auto, cpu, or cuda")
        if self.config.backend == "transformers_siglip":
            if self.model_artifact["kind"] != "directory":
                raise TriageError("transformers_siglip requires a self-contained local model directory")
            config_path = self.model_path / "config.json"
            if not config_path.is_file():
                raise TriageError("local Transformers model directory has no config.json")
            model_config, _ = _read_json(config_path, "local Transformers model config")
            if not isinstance(model_config, dict) or model_config.get("model_type") not in {
                "siglip",
                "siglip2",
            }:
                raise TriageError("local Transformers model config is not SigLIP/SigLIP2")
            single_weights = self.model_path / "model.safetensors"
            shard_index = self.model_path / "model.safetensors.index.json"
            shards = [
                path
                for path in self.model_path.glob("model-*-of-*.safetensors")
                if path.is_file() and path.stat().st_size > 0
            ]
            has_single_weights = single_weights.is_file() and single_weights.stat().st_size > 0
            has_sharded_weights = (
                shard_index.is_file() and shard_index.stat().st_size > 0 and bool(shards)
            )
            if not (has_single_weights or has_sharded_weights):
                raise TriageError(
                    "local SigLIP model has no complete root-level safetensors weights; "
                    "pickle weights are refused"
                )
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
            except ImportError as exc:
                raise TriageError("transformers_siglip requires local torch and transformers packages") from exc

    def _assert_model_artifact_unchanged(self) -> None:
        current = _artifact_root(self.model_path)
        if current != self.model_artifact:
            raise TriageError("local model artifact changed after its initial SHA-256 binding")

    def _validate_input_roots(self) -> None:
        roots = {
            "candidate_review_queue_sha256": _sha256_file(self.queue_path),
            "staging_manifest_sha256": _sha256_file(self.manifest_path),
            "review_queue_summary_sha256": _sha256_file(self.queue_summary_path),
            "integrity_audit_sha256": _sha256_file(self.integrity_audit_path),
            "class_contract_sha256": _sha256_file(self.class_contract_path),
        }
        expected = self.policy["production_input_roots"]
        if roots != expected:
            mismatches = {
                key: {"expected": expected[key], "actual": value}
                for key, value in roots.items()
                if expected.get(key) != value
            }
            raise TriageError(f"input roots do not match policy: {mismatches}")
        self.input_roots = roots

    def _validate_supporting_documents(self) -> None:
        summary, _ = _read_json(self.queue_summary_path, "review queue summary")
        if not isinstance(summary, dict) or summary.get("schema_version") != "rootscope.wikimedia_human_review_queue_summary.v1":
            raise TriageError("review queue summary schema is invalid")
        if summary.get("candidate_count") != self.policy["expected_candidate_count"]:
            raise TriageError("review queue summary candidate_count disagrees with policy")
        inputs = summary.get("inputs")
        outputs = summary.get("outputs")
        if not isinstance(inputs, dict) or inputs.get("staging_manifest_sha256") != self.input_roots["staging_manifest_sha256"]:
            raise TriageError("review queue summary does not bind the staging manifest")
        if not isinstance(outputs, dict) or outputs.get("candidate_review_queue.jsonl") != self.input_roots["candidate_review_queue_sha256"]:
            raise TriageError("review queue summary does not bind the candidate queue")

        audit, _ = _read_json(self.integrity_audit_path, "staging integrity audit")
        if (
            not isinstance(audit, dict)
            or audit.get("schema_version") != "rootscope.wikimedia_staging_integrity_audit.v2"
            or audit.get("result") != "PASS_STAGING_INTEGRITY_NOT_TRAIN_READY"
            or audit.get("failure_count") != 0
            or audit.get("failures") != []
            or audit.get("manifest_sha256") != self.input_roots["staging_manifest_sha256"]
        ):
            raise TriageError("staging integrity audit is not a clean manifest-bound PASS")
        constraints = audit.get("image_constraints")
        image_contract = self.policy["image_contract"]
        if (
            not isinstance(constraints, dict)
            or set(constraints.get("allowed_mime", [])) != set(image_contract["allowed_mime"])
            or constraints.get("minimum_downloaded_side") != min(
                image_contract["minimum_width"], image_contract["minimum_height"]
            )
        ):
            raise TriageError("staging integrity image constraints disagree with the triage policy")

        contract, _ = _read_json(self.class_contract_path, "class contract")
        if (
            not isinstance(contract, dict)
            or contract.get("schema_version") != "2.0.0"
            or contract.get("class_order") != self.classes
        ):
            raise TriageError("class contract does not bind the frozen class order")

    def _validate_manifest_and_queue(self) -> None:
        manifest_rows, _ = _read_jsonl(self.manifest_path, "staging manifest")
        queue_rows, _ = _read_jsonl(self.queue_path, "candidate review queue")
        if len(queue_rows) != self.policy["expected_candidate_count"]:
            raise TriageError("candidate queue count does not match policy")
        if len(manifest_rows) != len(queue_rows):
            raise TriageError("staging manifest and queue row counts differ")

        manifest_by_pageid: dict[int, dict[str, Any]] = {}
        for index, row in enumerate(manifest_rows, start=1):
            if not MANIFEST_REQUIRED_FIELDS.issubset(row):
                raise TriageError(
                    f"staging manifest line {index} is missing fields: "
                    f"{sorted(MANIFEST_REQUIRED_FIELDS - set(row))}"
                )
            if row.get("schema_version") != self.policy["manifest_schema_version"]:
                raise TriageError(f"staging manifest line {index} has unsupported schema")
            pageid = row.get("pageid")
            if type(pageid) is not int or pageid <= 0 or pageid in manifest_by_pageid:
                raise TriageError(f"staging manifest line {index} has invalid/duplicate pageid")
            manifest_by_pageid[pageid] = row

        dataset_root = self.manifest_path.parent.resolve(strict=True)
        seen_assets: set[str] = set()
        seen_shas: set[str] = set()
        seen_paths: set[str] = set()
        hint_counts: Counter[str] = Counter()
        image_records: list[dict[str, Any]] = []
        for index, row in enumerate(queue_rows, start=1):
            _require_exact_keys(row, QUEUE_FIELDS, f"candidate queue line {index}")
            if row.get("schema_version") != self.policy["queue_schema_version"]:
                raise TriageError(f"candidate queue line {index} has unsupported schema")
            if (
                row.get("review_status") != "UNREVIEWED"
                or row.get("visual_decision") != ""
                or row.get("rights_decision") != ""
                or row.get("target_class") != ""
                or row.get("reviewed_source_group") != ""
                or row.get("split") != "UNASSIGNED_DO_NOT_TRAIN"
                or row.get("training_eligible") is not False
                or row.get("print_eligible") is not False
            ):
                raise TriageError(f"candidate queue line {index} is not frozen unreviewed input")
            asset = row.get("asset")
            sha = row.get("sha256")
            pageid = row.get("pageid")
            hint = row.get("class_hint")
            if not isinstance(asset, str) or not asset or asset in seen_assets:
                raise TriageError(f"candidate queue line {index} has invalid/duplicate asset")
            if type(sha) is not str or HEX64.fullmatch(sha) is None or sha in seen_shas:
                raise TriageError(f"candidate queue line {index} has invalid/duplicate SHA-256")
            if type(pageid) is not int or pageid not in manifest_by_pageid:
                raise TriageError(f"candidate queue line {index} has no manifest pageid")
            if hint not in self.classes:
                raise TriageError(f"candidate queue line {index} has invalid acquisition class hint")
            manifest = manifest_by_pageid[pageid]
            expected_pairs = {
                "sha256": manifest.get("download_sha256"),
                "local_path": manifest.get("filename"),
                "source_group": manifest.get("source_group"),
                "source_url": manifest.get("source_page"),
                "creator": manifest.get("artist"),
                "license": manifest.get("license_canonical_name"),
                "license_url": manifest.get("license_canonical_url"),
                "class_hint": manifest.get("class_id"),
                "download_width": manifest.get("download_width"),
                "download_height": manifest.get("download_height"),
                "download_mime": manifest.get("download_mime"),
            }
            for field, expected in expected_pairs.items():
                if row.get(field) != expected:
                    raise TriageError(
                        f"candidate queue line {index} field {field} disagrees with staging manifest"
                    )
            image_path, normalized_path = _safe_relative_path(
                dataset_root, row.get("local_path"), f"candidate queue line {index} local_path"
            )
            if normalized_path in seen_paths:
                raise TriageError(f"candidate queue line {index} reuses an image path")
            actual_sha = _sha256_file(image_path)
            if actual_sha != sha:
                raise TriageError(f"candidate image bytes changed at queue line {index}")
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                    decoded_format = str(image.format or "").upper()
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                raise TriageError(f"candidate image decode failed at queue line {index}: {type(exc).__name__}") from exc
            mime = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(decoded_format)
            image_contract = self.policy["image_contract"]
            if (
                mime != row["download_mime"]
                or mime not in image_contract["allowed_mime"]
                or width != row["download_width"]
                or height != row["download_height"]
                or width < image_contract["minimum_width"]
                or height < image_contract["minimum_height"]
            ):
                raise TriageError(f"candidate image dimensions or MIME violate policy at queue line {index}")
            size = image_path.stat().st_size
            image_records.append(
                {
                    "asset": asset,
                    "pageid": pageid,
                    "candidate_sha256": sha,
                    "local_path": normalized_path,
                    "bytes": size,
                    "width": width,
                    "height": height,
                    "mime": mime,
                    "absolute_path": image_path,
                    "class_hint": hint,
                }
            )
            seen_assets.add(asset)
            seen_shas.add(sha)
            seen_paths.add(normalized_path)
            hint_counts[hint] += 1
        normalized_hint_counts = {class_id: hint_counts[class_id] for class_id in self.classes}
        if normalized_hint_counts != self.policy["expected_acquisition_hint_counts"]:
            raise TriageError("candidate acquisition hint counts do not match policy")
        payload = [
            {key: value for key, value in record.items() if key not in {"absolute_path", "class_hint"}}
            for record in image_records
        ]
        self.manifest_rows = manifest_rows
        self.queue_rows = queue_rows
        self.image_records = image_records
        self.image_payload_set_sha256 = _sha256_bytes(_canonical_bytes(payload))

    def preflight(self) -> dict[str, Any]:
        self._assert_model_artifact_unchanged()
        self._validate_input_roots()
        self._validate_supporting_documents()
        self._validate_manifest_and_queue()
        model_output_count: int | None = None
        external_output_sha256: str | None = None
        if self.config.backend == "external_scores":
            outputs, external_output_sha256 = self._load_external_model_outputs()
            model_output_count = len(outputs)
        status = "AI_VISUAL_TRIAGE_PREFLIGHT_PASS_NOT_HUMAN_REVIEWED_NOT_DATA_LOCKED"
        if not self.production_mode:
            status = "FIXTURE_" + status
        return {
            "schema_version": "rootscope.ai_visual_triage_preflight.v1",
            "mode": "PRODUCTION" if self.production_mode else "FIXTURE",
            "status": status,
            "candidate_count": len(self.queue_rows),
            "input_roots": self.input_roots,
            "image_payload_set_sha256": self.image_payload_set_sha256,
            "policy_sha256": self.policy_sha256,
            "implementation_sha256": self.implementation_sha256,
            "model": {
                "model_id": self.config.model_id,
                "backend": self.config.backend,
                "artifact": self.model_artifact,
                "prompt_set_sha256": self.prompt_set_sha256,
                "external_model_output_sha256": external_output_sha256,
                "model_output_count": model_output_count,
            },
            "output_would_be": str(self.output_dir),
            "writes_performed": False,
            "authority": AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }

    def _validate_probability_row(self, row: Any, index: int, image: Mapping[str, Any]) -> dict[str, Any]:
        row = _require_exact_keys(row, MODEL_OUTPUT_FIELDS, f"model output line {index}")
        if row["schema_version"] != self.policy["model_output_schema_version"]:
            raise TriageError(f"model output line {index} has unsupported schema")
        expected = {
            "asset": image["asset"],
            "candidate_sha256": image["candidate_sha256"],
            "model_id": self.config.model_id,
            "model_artifact_sha256": self.model_artifact["sha256"],
            "prompt_set_sha256": self.prompt_set_sha256,
        }
        for field, value in expected.items():
            if row[field] != value:
                raise TriageError(f"model output line {index} {field} binding mismatch")
        probabilities = row["class_probabilities"]
        if not isinstance(probabilities, dict) or set(probabilities) != set(self.classes):
            raise TriageError(f"model output line {index} probability keys are not the frozen classes")
        normalized: dict[str, float] = {}
        for class_id in self.classes:
            value = probabilities[class_id]
            if not _is_number(value) or not 0 <= float(value) <= 1:
                raise TriageError(f"model output line {index} has invalid probability")
            normalized[class_id] = float(value)
        tolerance = float(self.policy["thresholds"]["probability_sum_tolerance"])
        if abs(sum(normalized.values()) - 1.0) > tolerance:
            raise TriageError(f"model output line {index} probabilities do not sum to one")
        return {**row, "class_probabilities": normalized}

    def _load_external_model_outputs(self) -> tuple[list[dict[str, Any]], str]:
        assert self.model_output_path is not None
        rows, raw = _read_jsonl(self.model_output_path, "external model outputs")
        if len(rows) != len(self.image_records):
            raise TriageError("external model output count does not cover every candidate exactly once")
        normalized = [
            self._validate_probability_row(row, index, image)
            for index, (row, image) in enumerate(zip(rows, self.image_records, strict=True), start=1)
        ]
        return normalized, _sha256_bytes(raw)

    def _infer_transformers_siglip(self) -> list[dict[str, Any]]:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - guarded by preflight
            raise TriageError("transformers_siglip dependencies are unavailable") from exc
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            raise TriageError("CUDA was requested but is unavailable")
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        try:
            processor = AutoProcessor.from_pretrained(
                str(self.model_path), local_files_only=True, trust_remote_code=False
            )
            model = AutoModel.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            model.to(device)
            model.eval()
        except Exception as exc:  # transformers exposes many environment-specific exceptions
            raise TriageError(f"cannot load local Transformers/SigLIP model: {type(exc).__name__}: {exc}") from exc
        text_config = getattr(getattr(model, "config", None), "text_config", None)
        text_max_length = getattr(text_config, "max_position_embeddings", None)
        if type(text_max_length) is not int or not 1 <= text_max_length <= 512:
            raise TriageError("local SigLIP model has no safe fixed text context length")
        prompts = [self.policy["inference"]["prompts"][class_id] for class_id in self.classes]
        rows: list[dict[str, Any]] = []
        for start in range(0, len(self.image_records), self.config.batch_size):
            batch = self.image_records[start : start + self.config.batch_size]
            images: list[Image.Image] = []
            try:
                for item in batch:
                    with Image.open(item["absolute_path"]) as image:
                        image.load()
                        images.append(image.convert("RGB"))
                inputs = processor(
                    text=prompts,
                    images=images,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=text_max_length,
                )
                inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
                with torch.inference_mode():
                    outputs = model(**inputs)
                logits = getattr(outputs, "logits_per_image", None)
                if logits is None or tuple(logits.shape) != (len(batch), len(self.classes)):
                    raise TriageError("local model did not return logits_per_image with the frozen class shape")
                probabilities = torch.softmax(logits.float(), dim=-1).detach().cpu().tolist()
            except TriageError:
                raise
            except Exception as exc:
                raise TriageError(f"local SigLIP inference failed: {type(exc).__name__}: {exc}") from exc
            finally:
                for image in images:
                    image.close()
            for image_record, values in zip(batch, probabilities, strict=True):
                rounded = [round(float(value), 12) for value in values]
                correction = 1.0 - sum(rounded)
                # Apply the tiny decimal-rounding correction to the largest
                # probability.  Correcting class zero can make a genuinely
                # near-zero value negative even though the softmax was valid.
                correction_index = max(range(len(rounded)), key=rounded.__getitem__)
                rounded[correction_index] += correction
                row = {
                    "schema_version": self.policy["model_output_schema_version"],
                    "asset": image_record["asset"],
                    "candidate_sha256": image_record["candidate_sha256"],
                    "model_id": self.config.model_id,
                    "model_artifact_sha256": self.model_artifact["sha256"],
                    "prompt_set_sha256": self.prompt_set_sha256,
                    "class_probabilities": {
                        class_id: rounded[index] for index, class_id in enumerate(self.classes)
                    },
                }
                rows.append(self._validate_probability_row(row, len(rows) + 1, image_record))
        return rows

    def _model_outputs(self) -> tuple[list[dict[str, Any]], str | None, str]:
        if self.config.backend == "external_scores":
            rows, raw_sha = self._load_external_model_outputs()
            self._assert_model_artifact_unchanged()
            return rows, raw_sha, "EXTERNAL_SCORE_FILE_NOT_MODEL_EXECUTION_PROOF"
        rows = self._infer_transformers_siglip()
        self._assert_model_artifact_unchanged()
        return rows, None, "IN_PROCESS_LOCAL_TRANSFORMERS_SIGLIP"

    def _triage_result(
        self, image: Mapping[str, Any], model_output: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        probabilities = model_output["class_probabilities"]
        class_index = {class_id: index for index, class_id in enumerate(self.classes)}
        ranked = sorted(self.classes, key=lambda class_id: (-probabilities[class_id], class_index[class_id]))
        top1, top2 = ranked[:2]
        top1_probability = float(probabilities[top1])
        top2_probability = float(probabilities[top2])
        margin = top1_probability - top2_probability
        thresholds = self.policy["thresholds"]
        reasons: list[str] = []
        if top1_probability < float(thresholds["minimum_top1_probability_by_class"][top1]):
            reasons.append("TOP1_PROBABILITY_BELOW_CLASS_THRESHOLD")
        if margin < float(thresholds["minimum_top1_top2_margin"]):
            reasons.append("TOP1_TOP2_MARGIN_BELOW_THRESHOLD")
        decision = (
            "AI_HIGH_CONFIDENCE_SUGGESTION_NOT_HUMAN_APPROVED"
            if not reasons
            else "AI_LOW_CONFIDENCE_NEEDS_REVIEW"
        )
        mode = "PRODUCTION" if self.production_mode else "FIXTURE"
        result = {
            "schema_version": self.policy["result_schema_version"],
            "mode": mode,
            "asset": image["asset"],
            "pageid": image["pageid"],
            "candidate_sha256": image["candidate_sha256"],
            "image_path": image["local_path"],
            "model_id": self.config.model_id,
            "model_artifact_sha256": self.model_artifact["sha256"],
            "prompt_set_sha256": self.prompt_set_sha256,
            "class_probabilities": probabilities,
            "top1_class": top1,
            "top1_probability": top1_probability,
            "top2_class": top2,
            "top2_probability": top2_probability,
            "top1_top2_margin": margin,
            "triage_decision": decision,
            "low_confidence_reasons": reasons,
            "acquisition_class_hint": image["class_hint"],
            "acquisition_hint_agrees": image["class_hint"] == top1,
            "authority": AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }
        _require_exact_keys(result, RESULT_FIELDS, "rendered triage result")
        low: dict[str, Any] | None = None
        if reasons:
            low = {
                "schema_version": self.policy["low_confidence_schema_version"],
                "mode": mode,
                "asset": image["asset"],
                "pageid": image["pageid"],
                "candidate_sha256": image["candidate_sha256"],
                "suggested_class": top1,
                "top1_probability": top1_probability,
                "top1_top2_margin": margin,
                "low_confidence_reasons": reasons,
                "result_sha256": _sha256_bytes(_canonical_bytes(result)),
                "authority": AUTHORITY,
                "explicit_non_claims": self.policy["explicit_non_claims"],
            }
            _require_exact_keys(low, LOW_CONFIDENCE_FIELDS, "rendered low-confidence row")
        return result, low

    def run(self) -> dict[str, Any]:
        preflight = self.preflight()
        if self.output_dir.exists():
            raise TriageError("AI triage output directory already exists; immutable runs are never overwritten")
        model_outputs, external_output_sha, provenance = self._model_outputs()
        normalized_model_outputs_payload = _jsonl_bytes(model_outputs)
        results: list[dict[str, Any]] = []
        low_confidence: list[dict[str, Any]] = []
        for image, model_output in zip(self.image_records, model_outputs, strict=True):
            result, low = self._triage_result(image, model_output)
            results.append(result)
            if low is not None:
                low_confidence.append(low)
        results_payload = _jsonl_bytes(results)
        low_payload = _jsonl_bytes(low_confidence)

        predicted_counts = Counter(row["top1_class"] for row in results)
        decision_counts = Counter(row["triage_decision"] for row in results)
        high_by_class = Counter(
            row["top1_class"]
            for row in results
            if row["triage_decision"] == "AI_HIGH_CONFIDENCE_SUGGESTION_NOT_HUMAN_APPROVED"
        )
        low_by_class = Counter(row["top1_class"] for row in results if row["low_confidence_reasons"])
        agreement = sum(1 for row in results if row["acquisition_hint_agrees"])
        mode = "PRODUCTION" if self.production_mode else "FIXTURE"
        status = self.policy["output_contract"]["status"]
        if not self.production_mode:
            status = "FIXTURE_" + status
        stats = {
            "schema_version": self.policy["stats_schema_version"],
            "mode": mode,
            "status": status,
            "candidate_count": len(results),
            "high_confidence_suggestion_count": len(results) - len(low_confidence),
            "low_confidence_count": len(low_confidence),
            "predicted_class_counts": {class_id: predicted_counts[class_id] for class_id in self.classes},
            "triage_decision_counts": dict(sorted(decision_counts.items())),
            "high_confidence_by_class": {class_id: high_by_class[class_id] for class_id in self.classes},
            "low_confidence_by_class": {class_id: low_by_class[class_id] for class_id in self.classes},
            "acquisition_hint_agreement_count": agreement,
            "acquisition_hint_disagreement_count": len(results) - agreement,
            "thresholds": self.policy["thresholds"],
            "authority": AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }
        stats_payload = _json_bytes(stats)
        output_hashes = {
            self.policy["output_contract"]["normalized_model_outputs_filename"]: _sha256_bytes(
                normalized_model_outputs_payload
            ),
            self.policy["output_contract"]["results_filename"]: _sha256_bytes(results_payload),
            self.policy["output_contract"]["low_confidence_filename"]: _sha256_bytes(low_payload),
            self.policy["output_contract"]["stats_filename"]: _sha256_bytes(stats_payload),
        }
        run_binding = {
            "policy_sha256": self.policy_sha256,
            "implementation_sha256": self.implementation_sha256,
            "input_roots": self.input_roots,
            "image_payload_set_sha256": self.image_payload_set_sha256,
            "model_id": self.config.model_id,
            "model_artifact_sha256": self.model_artifact["sha256"],
            "prompt_set_sha256": self.prompt_set_sha256,
            "backend": self.config.backend,
            "normalized_model_outputs_sha256": output_hashes[
                self.policy["output_contract"]["normalized_model_outputs_filename"]
            ],
            "results_sha256": output_hashes[self.policy["output_contract"]["results_filename"]],
        }
        run_id = "sha256:" + _sha256_bytes(_canonical_bytes(run_binding))
        receipt = {
            "schema_version": self.policy["receipt_schema_version"],
            "mode": mode,
            "status": status,
            "run_id": run_id,
            "policy_sha256": self.policy_sha256,
            "implementation_sha256": self.implementation_sha256,
            "input_roots": self.input_roots,
            "image_payload_set_sha256": self.image_payload_set_sha256,
            "candidate_count": len(results),
            "model": {
                "model_id": self.config.model_id,
                "backend": self.config.backend,
                "artifact": self.model_artifact,
                "prompt_set_sha256": self.prompt_set_sha256,
                "output_provenance": provenance,
                "external_model_output_sha256": external_output_sha,
            },
            "counts": {
                "high_confidence_suggestion": len(results) - len(low_confidence),
                "low_confidence": len(low_confidence),
            },
            "outputs": output_hashes,
            "human_review_files_touched": False,
            "dataset_manifest_written": False,
            "authority": AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }
        receipt_payload = _json_bytes(receipt)
        output = self.policy["output_contract"]
        payloads = {
            output["normalized_model_outputs_filename"]: normalized_model_outputs_payload,
            output["results_filename"]: results_payload,
            output["low_confidence_filename"]: low_payload,
            output["stats_filename"]: stats_payload,
            output["receipt_filename"]: receipt_payload,
        }
        self._commit_output(payloads)
        return {"preflight": preflight, "receipt": receipt}

    def _commit_output(self, payloads: Mapping[str, bytes]) -> None:
        parent = self.output_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".ai_triage_v1.tmp-", dir=str(parent)))
        try:
            for name, payload in payloads.items():
                if Path(name).name != name:
                    raise TriageError("output filename is not a basename")
                path = temporary / name
                with path.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            if self.output_dir.exists():
                raise TriageError("AI triage output appeared during commit")
            os.replace(temporary, self.output_dir)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RootScope frozen-pool AI visual triage (never human review or DATA_LOCKED)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="read-only validation; writes nothing")
    mode.add_argument("--run", action="store_true", help="write one immutable AI-only triage run")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--queue-summary", type=Path, default=DEFAULT_QUEUE_SUMMARY)
    parser.add_argument("--integrity-audit", type=Path, default=DEFAULT_INTEGRITY_AUDIT)
    parser.add_argument("--class-contract", type=Path, default=DEFAULT_CLASS_CONTRACT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, required=True, help="existing local model file/directory")
    parser.add_argument("--model-id", required=True, help="stable model identifier bound into every row")
    parser.add_argument(
        "--backend",
        choices=("external_scores", "transformers_siglip"),
        default="external_scores",
    )
    parser.add_argument("--model-output-jsonl", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fixture-mode", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = TriageConfig(
        queue_path=args.queue,
        manifest_path=args.manifest,
        queue_summary_path=args.queue_summary,
        integrity_audit_path=args.integrity_audit,
        class_contract_path=args.class_contract,
        policy_path=args.policy,
        output_dir=args.output_dir,
        model_path=args.model_path,
        model_id=args.model_id,
        backend=args.backend,
        model_output_path=args.model_output_jsonl,
        fixture_mode=args.fixture_mode,
        device=args.device,
        batch_size=args.batch_size,
    )
    try:
        triage = AIVisualTriage(config)
        if args.preflight:
            report = triage.preflight()
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        else:
            result = triage.run()
            print(json.dumps(result["receipt"], ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except TriageError as exc:
        print(f"AI visual triage refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
