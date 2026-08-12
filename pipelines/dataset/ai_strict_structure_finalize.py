#!/usr/bin/env python3
"""Fail-closed strict structure finalizer for the frozen RootScope AI ensemble.

This program does not infer new pixels and does not turn AI suggestions into
ground truth.  It consumes the immutable SigLIP2 v1 prompt scores plus the
metadata-only risk records and emits a *new* immutable machine-curated
candidate directory.  In particular it never writes the v1 ensemble,
``human_decisions``, the dataset manifest, training splits, or eligibility
fields.

The positive gate deliberately uses the single ``quality.complete_plant``
prompt and the maximum *individual* reject prompt.  This avoids the v1 failure
mode where a strong missing-base/crown signal was diluted by a reject-family
mean.  Hand/person, detail crop, mature/dead tree, document/specimen, and
mixed/wide-scene prompts remain explicit auditable blockers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
ADVENTUREX_ROOT = SCRIPT_PATH.parents[2]
STAGING_ROOT = ADVENTUREX_ROOT / "datasets" / "desert_plants_wikimedia_staging_e0"
REVIEW_ROOT = STAGING_ROOT / "review"

DEFAULT_ENSEMBLE_RESULTS = REVIEW_ROOT / "ai_ensemble_v1" / "ai_siglip2_ensemble_results.jsonl"
DEFAULT_ENSEMBLE_RECEIPT = REVIEW_ROOT / "ai_ensemble_v1" / "ai_siglip2_ensemble_receipt.json"
DEFAULT_METADATA_RECORDS = REVIEW_ROOT / "ai_metadata_triage_v1" / "metadata_risk_records.jsonl"
DEFAULT_METADATA_RECEIPT = REVIEW_ROOT / "ai_metadata_triage_v1" / "receipt.json"
DEFAULT_ENSEMBLE_POLICY = SCRIPT_PATH.with_name("ai_siglip2_ensemble_policy_v1.json")
DEFAULT_POLICY = SCRIPT_PATH.with_name("ai_strict_structure_policy_v1.json")
DEFAULT_MODEL_PROVENANCE = ADVENTUREX_ROOT / "models" / "ai_triage" / "SIGLIP2_MODEL_PROVENANCE.json"
DEFAULT_RUNTIME_PROVENANCE = ADVENTUREX_ROOT / "models" / "ai_triage" / "SIGLIP2_RUNTIME_PROVENANCE.json"
DEFAULT_TOKENIZER = ADVENTUREX_ROOT / "models" / "ai_triage" / "siglip2_tokenizer_75de2d55"
DEFAULT_HUMAN_DECISIONS = REVIEW_ROOT / "human_decisions"
DEFAULT_OUTPUT_DIR = REVIEW_ROOT / "ai_final_labels_v1"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
TARGET_CLASSES = ("grass_clump", "low_shrub", "young_tree")
ALL_HINTS = (*TARGET_CLASSES, "unknown")
SOURCE_DECISIONS = ("AUTO_TARGET", "AUTO_UNKNOWN", "HOLD")
FINAL_LABELS = {
    "POSITIVE_CANDIDATE_GRASS_CLUMP",
    "POSITIVE_CANDIDATE_LOW_SHRUB",
    "POSITIVE_CANDIDATE_YOUNG_TREE",
    "UNKNOWN_CANDIDATE",
    "EXCLUDE_NONCONFORMING",
    "HOLD_INSUFFICIENT_EVIDENCE",
}
INPUT_AUTHORITY_KEYS = {
    "human_review",
    "dataset_manifest_write",
    "training_eligibility",
    "split_assignment",
    "print_eligibility",
    "data_locked",
}
OUTPUT_AUTHORITY = {
    "visual_truth": False,
    "human_review": False,
    "rights_approval": False,
    "dataset_manifest_write": False,
    "training_eligibility": False,
    "split_assignment": False,
    "print_eligibility": False,
    "model_qualification": False,
    "data_locked": False,
}

RESULT_FIELDS = {
    "schema_version",
    "mode",
    "asset",
    "pageid",
    "candidate_sha256",
    "image_path",
    "acquisition_class_hint",
    "model_id",
    "model_artifact_sha256",
    "prompt_set_sha256",
    "calibration_sha256",
    "calibration_status",
    "prompt_scores",
    "quality_score",
    "reject_family_scores",
    "dominant_reject_family",
    "dominant_reject_score",
    "admissibility_probability",
    "reject_family_probabilities",
    "reject_probability",
    "class_scores",
    "class_probabilities",
    "top1_class",
    "top1_probability",
    "top2_class",
    "top2_probability",
    "top1_top2_margin",
    "acquisition_hint_agrees",
    "decision",
    "suggested_class",
    "decision_reasons",
    "authority",
    "explicit_non_claims",
}

METADATA_FIELDS = {
    "schema_version",
    "queue_index",
    "asset",
    "source_group",
    "local_path",
    "acquisition_metadata",
    "metadata_only",
    "visual_truth_established",
    "risk_priority",
    "risk_flags",
    "support_signals",
    "context_flags",
}


class FinalizerError(RuntimeError):
    """A scope, schema, integrity, provenance, or decision error."""


@dataclass(frozen=True)
class FinalizerConfig:
    ensemble_results: Path = DEFAULT_ENSEMBLE_RESULTS
    ensemble_receipt: Path = DEFAULT_ENSEMBLE_RECEIPT
    metadata_records: Path = DEFAULT_METADATA_RECORDS
    metadata_receipt: Path = DEFAULT_METADATA_RECEIPT
    ensemble_policy: Path = DEFAULT_ENSEMBLE_POLICY
    policy: Path = DEFAULT_POLICY
    model_provenance: Path = DEFAULT_MODEL_PROVENANCE
    runtime_provenance: Path = DEFAULT_RUNTIME_PROVENANCE
    tokenizer_dir: Path = DEFAULT_TOKENIZER
    human_decisions: Path = DEFAULT_HUMAN_DECISIONS
    output_dir: Path = DEFAULT_OUTPUT_DIR
    adventurex_root: Path = ADVENTUREX_ROOT
    fixture_mode: bool = False


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
        raise FinalizerError(f"{context} is not strict UTF-8 JSON: {exc}") from exc


def _read_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(path, context)
    value = _parse_json(raw, context)
    if not isinstance(value, dict):
        raise FinalizerError(f"{context} must be a JSON object")
    return value, raw


def _read_jsonl(path: Path, context: str) -> tuple[list[dict[str, Any]], bytes]:
    raw = _read_regular_bytes(path, context)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FinalizerError(f"{context} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise FinalizerError(f"{context} line {line_number} is blank")
        value = _parse_json(line.encode("utf-8"), f"{context} line {line_number}")
        if not isinstance(value, dict):
            raise FinalizerError(f"{context} line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise FinalizerError(f"{context} is empty")
    return rows, raw


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
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_reparse(info: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & flag)


def _regular_lstat(path: Path, context: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise FinalizerError(f"cannot inspect {context}: {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
        raise FinalizerError(f"{context} must not be a link or reparse point: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise FinalizerError(f"{context} is not a regular file: {path}")
    return info


def _read_regular_bytes(path: Path, context: str) -> bytes:
    before = _regular_lstat(path, context)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise FinalizerError(f"cannot read {context}: {path}: {exc}") from exc
    after = _regular_lstat(path, context)
    before_key = (before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino)
    after_key = (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino)
    if before_key != after_key or len(payload) != before.st_size:
        raise FinalizerError(f"{context} changed while it was read: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    before = _regular_lstat(path, "hashed input")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FinalizerError(f"cannot open hashed input: {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            opened = os.fstat(handle.fileno())
            opened_key = (opened.st_size, opened.st_mtime_ns, opened.st_dev, opened.st_ino)
            before_key = (before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino)
            if opened_key != before_key:
                raise FinalizerError(f"hashed input changed while opened: {path}")
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = _regular_lstat(path, "hashed input")
    if (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino) != (
        before.st_size,
        before.st_mtime_ns,
        before.st_dev,
        before.st_ino,
    ):
        raise FinalizerError(f"hashed input changed while hashed: {path}")
    return digest.hexdigest()


def _artifact_root(path: Path, context: str, allow_empty: bool = False) -> dict[str, Any]:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise FinalizerError(f"cannot inspect {context}: {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
        raise FinalizerError(f"{context} must not be a link or reparse point")
    root = path.resolve(strict=True)
    if root.is_file():
        size = _regular_lstat(root, context).st_size
        if size <= 0 and not allow_empty:
            raise FinalizerError(f"{context} is empty")
        raw_sha = _sha256_file(root)
        entries = [{"path": root.name, "bytes": size, "sha256": raw_sha}]
        entries_sha = _sha256_bytes(_canonical_bytes(entries))
        return {
            "kind": "file",
            "sha256": entries_sha,
            "entries_sha256": entries_sha,
            "file_count": 1,
            "byte_count": size,
            "entries": entries,
        }
    if not root.is_dir():
        raise FinalizerError(f"{context} is neither a regular file nor a directory")
    entries: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        entry_info = os.lstat(candidate)
        if stat.S_ISLNK(entry_info.st_mode) or _is_reparse(entry_info):
            raise FinalizerError(f"{context} contains a link or reparse point")
        if stat.S_ISDIR(entry_info.st_mode):
            continue
        if not stat.S_ISREG(entry_info.st_mode):
            raise FinalizerError(f"{context} contains a non-regular entry")
        entries.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "bytes": entry_info.st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    if not entries and not allow_empty:
        raise FinalizerError(f"{context} contains no regular files")
    entries_sha = _sha256_bytes(_canonical_bytes(entries))
    return {
        "kind": "directory",
        "sha256": entries_sha,
        "entries_sha256": entries_sha,
        "file_count": len(entries),
        "byte_count": sum(int(entry["bytes"]) for entry in entries),
        "entries": entries,
    }


def _public_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("kind", "sha256", "entries_sha256", "file_count", "byte_count")}


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise FinalizerError(
            f"{context} strict fields differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_false_authority(value: Any, required: set[str], context: str) -> None:
    if not isinstance(value, dict) or not required.issubset(value):
        raise FinalizerError(f"{context} lacks the required authority fields")
    if any(value[key] is not False for key in value):
        raise FinalizerError(f"{context} grants forbidden authority")


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalizerError(f"{context} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise FinalizerError(f"{context} must be finite")
    return converted


def _probability(value: Any, context: str) -> float:
    converted = _finite_number(value, context)
    if not 0.0 <= converted <= 1.0:
        raise FinalizerError(f"{context} must be in [0,1]")
    return converted


def _round(value: float) -> float:
    if not math.isfinite(value):
        raise FinalizerError("attempted to serialize a non-finite metric")
    return round(float(value), 12)


def _safe_bound_path(root: Path, relative: Any, context: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise FinalizerError(f"{context} must be a non-empty POSIX relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise FinalizerError(f"{context} is unsafe")
    candidate = root.joinpath(*pure.parts)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise FinalizerError(f"{context} is missing or escapes AdventureX") from exc
    return resolved


def _current_runtime(runtime_manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime = runtime_manifest.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("packages"), dict):
        raise FinalizerError("runtime provenance has an invalid runtime section")
    packages: dict[str, str] = {}
    for name, expected in sorted(runtime["packages"].items()):
        if not isinstance(name, str) or not isinstance(expected, str):
            raise FinalizerError("runtime package bindings must be strings")
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise FinalizerError(f"required runtime package is absent: {name}") from exc
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": packages,
    }


class StrictStructureFinalizer:
    """Validate frozen inputs and create conservative machine-only candidates."""

    def __init__(self, config: FinalizerConfig) -> None:
        self.config = config
        self.adventurex_root = Path(config.adventurex_root)
        self.ensemble_results_path = Path(config.ensemble_results)
        self.ensemble_receipt_path = Path(config.ensemble_receipt)
        self.metadata_records_path = Path(config.metadata_records)
        self.metadata_receipt_path = Path(config.metadata_receipt)
        self.ensemble_policy_path = Path(config.ensemble_policy)
        self.policy_path = Path(config.policy)
        self.model_provenance_path = Path(config.model_provenance)
        self.runtime_provenance_path = Path(config.runtime_provenance)
        self.tokenizer_dir = Path(config.tokenizer_dir)
        self.human_decisions = Path(config.human_decisions)
        self.output_dir = Path(config.output_dir)
        self.fixture_mode = bool(config.fixture_mode)
        self._validate_scope()
        self._loaded = False

    def _validate_scope(self) -> None:
        if not self.fixture_mode:
            expected = {
                "ensemble_results": DEFAULT_ENSEMBLE_RESULTS,
                "ensemble_receipt": DEFAULT_ENSEMBLE_RECEIPT,
                "metadata_records": DEFAULT_METADATA_RECORDS,
                "metadata_receipt": DEFAULT_METADATA_RECEIPT,
                "ensemble_policy": DEFAULT_ENSEMBLE_POLICY,
                "policy": DEFAULT_POLICY,
                "model_provenance": DEFAULT_MODEL_PROVENANCE,
                "runtime_provenance": DEFAULT_RUNTIME_PROVENANCE,
                "tokenizer_dir": DEFAULT_TOKENIZER,
                "human_decisions": DEFAULT_HUMAN_DECISIONS,
                "output_dir": DEFAULT_OUTPUT_DIR,
                "adventurex_root": ADVENTUREX_ROOT,
            }
            actual = {
                "ensemble_results": self.ensemble_results_path,
                "ensemble_receipt": self.ensemble_receipt_path,
                "metadata_records": self.metadata_records_path,
                "metadata_receipt": self.metadata_receipt_path,
                "ensemble_policy": self.ensemble_policy_path,
                "policy": self.policy_path,
                "model_provenance": self.model_provenance_path,
                "runtime_provenance": self.runtime_provenance_path,
                "tokenizer_dir": self.tokenizer_dir,
                "human_decisions": self.human_decisions,
                "output_dir": self.output_dir,
                "adventurex_root": self.adventurex_root,
            }
            for name in expected:
                if actual[name].resolve(strict=False) != expected[name].resolve(strict=False):
                    raise FinalizerError(f"production {name} path is frozen")
        review = self.ensemble_results_path.parent.parent.resolve(strict=True)
        if self.ensemble_receipt_path.parent.resolve(strict=True) != self.ensemble_results_path.parent.resolve(strict=True):
            raise FinalizerError("ensemble result and receipt directories differ")
        if self.metadata_records_path.parent.resolve(strict=True) != self.metadata_receipt_path.parent.resolve(strict=True):
            raise FinalizerError("metadata result and receipt directories differ")
        if self.metadata_records_path.parent.parent.resolve(strict=True) != review:
            raise FinalizerError("metadata and ensemble inputs do not share one review root")
        if self.human_decisions.resolve(strict=True).parent != review:
            raise FinalizerError("human_decisions guard is outside the review root")
        if "human_decisions" in {part.casefold() for part in self.output_dir.parts}:
            raise FinalizerError("output may not target human_decisions")
        if self.output_dir.resolve(strict=False).parent != review:
            raise FinalizerError("output must be a direct child of the review root")

    def _load(self) -> None:
        if self._loaded:
            return
        self.policy, policy_raw = _read_json(self.policy_path, "strict structure policy")
        self.policy_sha256 = _sha256_bytes(policy_raw)
        self._validate_policy()
        self.ensemble_results, results_raw = _read_jsonl(
            self.ensemble_results_path, "ensemble results"
        )
        self.ensemble_receipt, ensemble_receipt_raw = _read_json(
            self.ensemble_receipt_path, "ensemble receipt"
        )
        self.metadata_records, metadata_raw = _read_jsonl(
            self.metadata_records_path, "metadata risk records"
        )
        self.metadata_receipt, metadata_receipt_raw = _read_json(
            self.metadata_receipt_path, "metadata receipt"
        )
        self.ensemble_policy, ensemble_policy_raw = _read_json(
            self.ensemble_policy_path, "ensemble policy"
        )
        self.model_provenance, model_provenance_raw = _read_json(
            self.model_provenance_path, "model provenance"
        )
        self.runtime_provenance, runtime_provenance_raw = _read_json(
            self.runtime_provenance_path, "runtime provenance"
        )
        self.input_roots = {
            "ensemble_results_sha256": _sha256_bytes(results_raw),
            "ensemble_receipt_sha256": _sha256_bytes(ensemble_receipt_raw),
            "metadata_records_sha256": _sha256_bytes(metadata_raw),
            "metadata_receipt_sha256": _sha256_bytes(metadata_receipt_raw),
            "ensemble_policy_sha256": _sha256_bytes(ensemble_policy_raw),
            "model_provenance_sha256": _sha256_bytes(model_provenance_raw),
            "runtime_provenance_sha256": _sha256_bytes(runtime_provenance_raw),
        }
        expected_roots = self.policy["production_input_roots"]
        for name, actual in self.input_roots.items():
            if expected_roots.get(name) != actual:
                raise FinalizerError(f"frozen input root mismatch: {name}")

        self._validate_source_receipts()
        self._validate_prompt_contract()
        self._validate_provenance()
        self._validate_rows()
        self.implementation_sha256 = _sha256_file(SCRIPT_PATH)
        self.human_root_before = _artifact_root(
            self.human_decisions, "formal human_decisions guard", allow_empty=True
        )
        self._input_path_hashes = self._current_path_hashes()
        self._loaded = True

    def _validate_policy(self) -> None:
        required = {
            "schema_version",
            "result_schema_version",
            "stats_schema_version",
            "receipt_schema_version",
            "expected_candidate_count",
            "target_classes",
            "allowed_source_decisions",
            "production_input_roots",
            "prompt_contract",
            "thresholds",
            "runtime_contract",
            "output_contract",
            "authority",
            "explicit_non_claims",
        }
        _require_exact_fields(self.policy, required, "strict structure policy")
        if self.policy["schema_version"] != "rootscope.ai_strict_structure_policy.v1":
            raise FinalizerError("unsupported strict structure policy schema")
        if self.policy["target_classes"] != list(TARGET_CLASSES):
            raise FinalizerError("strict target class order changed")
        if self.policy["allowed_source_decisions"] != list(SOURCE_DECISIONS):
            raise FinalizerError("allowed source decisions changed")
        if type(self.policy["expected_candidate_count"]) is not int or self.policy["expected_candidate_count"] <= 0:
            raise FinalizerError("expected candidate count must be positive")
        roots = self.policy["production_input_roots"]
        required_roots = {
            "ensemble_results_sha256",
            "ensemble_receipt_sha256",
            "metadata_records_sha256",
            "metadata_receipt_sha256",
            "ensemble_policy_sha256",
            "model_provenance_sha256",
            "runtime_provenance_sha256",
            "tokenizer_artifact_sha256",
            "model_weights_raw_sha256",
        }
        _require_exact_fields(roots, required_roots, "production input roots")
        if any(not isinstance(value, str) or not HEX64.fullmatch(value) for value in roots.values()):
            raise FinalizerError("every production input root must be lowercase SHA-256")
        if self.policy["authority"] != OUTPUT_AUTHORITY:
            raise FinalizerError("strict policy authority must be exactly all-false")
        output = self.policy["output_contract"]
        expected_output = {
            "directory_name",
            "labels_filename",
            "holds_filename",
            "stats_filename",
            "receipt_filename",
            "status",
        }
        _require_exact_fields(output, expected_output, "output contract")
        if output["directory_name"] != "ai_final_labels_v1":
            raise FinalizerError("output directory name is frozen")
        names = [output[key] for key in ("labels_filename", "holds_filename", "stats_filename", "receipt_filename")]
        if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
            raise FinalizerError("output filenames must be unique basenames")
        if self.output_dir.name != output["directory_name"]:
            raise FinalizerError("configured output directory disagrees with policy")

    def _validate_source_receipts(self) -> None:
        _require_false_authority(
            self.ensemble_receipt.get("authority"), INPUT_AUTHORITY_KEYS, "ensemble receipt authority"
        )
        if self.ensemble_receipt.get("human_review_files_touched") is not False:
            raise FinalizerError("ensemble receipt does not preserve human review")
        if self.ensemble_receipt.get("dataset_manifest_written") is not False:
            raise FinalizerError("ensemble receipt claims a manifest write")
        expected_count = self.policy["expected_candidate_count"]
        if self.ensemble_receipt.get("candidate_count") != expected_count:
            raise FinalizerError("ensemble receipt candidate count mismatch")
        result_name = self.ensemble_results_path.name
        if self.ensemble_receipt.get("outputs", {}).get(result_name) != self.input_roots["ensemble_results_sha256"]:
            raise FinalizerError("ensemble receipt does not bind the result payload")
        _require_false_authority(
            self.metadata_receipt.get("authority"), INPUT_AUTHORITY_KEYS, "metadata receipt authority"
        )
        if self.metadata_receipt.get("rows") != expected_count:
            raise FinalizerError("metadata receipt row count mismatch")
        if self.metadata_receipt.get("artifacts_sha256", {}).get(self.metadata_records_path.name) != self.input_roots["metadata_records_sha256"]:
            raise FinalizerError("metadata receipt does not bind metadata records")
        ensemble_queue = self.ensemble_receipt.get("input_roots", {}).get("candidate_review_queue_sha256")
        if ensemble_queue != self.metadata_receipt.get("queue_sha256"):
            raise FinalizerError("ensemble and metadata receipts bind different queues")

    def _validate_prompt_contract(self) -> None:
        inference = self.ensemble_policy.get("inference")
        if not isinstance(inference, dict):
            raise FinalizerError("ensemble policy inference block is missing")
        prompt_spec = {
            "whole_quality_prompts": inference.get("whole_quality_prompts"),
            "class_prompts": inference.get("class_prompts"),
            "reject_family_prompts": inference.get("reject_family_prompts"),
        }
        prompt_set_sha = _sha256_bytes(_canonical_bytes(prompt_spec))
        if prompt_set_sha != self.ensemble_receipt.get("prompt_set_sha256"):
            raise FinalizerError("ensemble prompt set no longer matches its receipt")
        prompt_ids: list[str] = []
        for record in prompt_spec["whole_quality_prompts"]:
            prompt_ids.append(record["id"])
        for class_id in TARGET_CLASSES:
            for record in prompt_spec["class_prompts"][class_id]:
                prompt_ids.append(record["id"])
        for records in prompt_spec["reject_family_prompts"].values():
            for record in records:
                prompt_ids.append(record["id"])
        if len(prompt_ids) != len(set(prompt_ids)):
            raise FinalizerError("ensemble prompt IDs are not unique")
        contract = self.policy["prompt_contract"]
        referenced = {
            contract["complete_plant_prompt"],
            *contract["quality_prompts"],
            *contract["hard_reject_prompts"],
        }
        for anchors in contract["class_structure_anchors"].values():
            referenced.update(anchors)
        for prompts in contract["named_hard_blocks"].values():
            referenced.update(prompts)
        if not referenced.issubset(prompt_ids):
            raise FinalizerError("strict policy references an unknown ensemble prompt")
        if set(contract["hard_reject_prompts"]) != {
            prompt_id for prompt_id in prompt_ids if prompt_id.startswith("reject.")
        }:
            raise FinalizerError("hard reject list must cover every individual reject prompt")
        self.prompt_ids = tuple(prompt_ids)
        self.reject_families = tuple(prompt_spec["reject_family_prompts"])
        self.prompt_set_sha256 = prompt_set_sha

    def _validate_provenance(self) -> None:
        roots = self.policy["production_input_roots"]
        if self.model_provenance.get("schema_version") != "rootscope.local_model_provenance.v1":
            raise FinalizerError("unsupported model provenance schema")
        weights = self.model_provenance.get("weights")
        tokenizer = self.model_provenance.get("tokenizer")
        if not isinstance(weights, dict) or not isinstance(tokenizer, dict):
            raise FinalizerError("model provenance lacks weights/tokenizer bindings")
        weights_path = _safe_bound_path(
            self.adventurex_root, weights.get("local_path"), "model weights path"
        )
        tokenizer_path = _safe_bound_path(
            self.adventurex_root, tokenizer.get("local_path"), "tokenizer path"
        )
        if tokenizer_path.resolve(strict=True) != self.tokenizer_dir.resolve(strict=True):
            raise FinalizerError("configured tokenizer differs from model provenance")
        self.model_weights_artifact = _artifact_root(weights_path, "model weights")
        if len(self.model_weights_artifact["entries"]) != 1:
            raise FinalizerError("the frozen Big Vision model must be one weights file")
        raw_weights_sha = self.model_weights_artifact["entries"][0]["sha256"]
        if weights.get("sha256") != raw_weights_sha or roots["model_weights_raw_sha256"] != raw_weights_sha:
            raise FinalizerError("model weights raw SHA-256 mismatch")
        receipt_artifact = self.ensemble_receipt.get("model", {}).get("artifact", {})
        if receipt_artifact.get("sha256") != self.model_weights_artifact["sha256"]:
            raise FinalizerError("ensemble receipt model artifact root mismatch")
        self.tokenizer_artifact = _artifact_root(self.tokenizer_dir, "local tokenizer")
        if self.tokenizer_artifact["sha256"] != roots["tokenizer_artifact_sha256"]:
            raise FinalizerError("tokenizer artifact root mismatch")
        if self.runtime_provenance.get("schema_version") != "rootscope.siglip2_runtime_provenance.v1":
            raise FinalizerError("unsupported runtime provenance schema")
        _require_false_authority(
            self.runtime_provenance.get("authority"),
            {"human_review", "visual_truth", "rights_approval", "training_eligibility", "data_locked"},
            "runtime provenance authority",
        )
        bindings = self.runtime_provenance.get("bindings")
        expected_bindings = {
            "model_provenance_sha256": self.input_roots["model_provenance_sha256"],
            "model_weights_raw_sha256": raw_weights_sha,
            "tokenizer_artifact_sha256": self.tokenizer_artifact["sha256"],
        }
        if bindings != expected_bindings:
            raise FinalizerError("runtime provenance artifact bindings mismatch")
        self.current_runtime = _current_runtime(self.runtime_provenance)
        expected_runtime = self.runtime_provenance["runtime"]
        expected_current = {
            "python_implementation": expected_runtime.get("python_implementation"),
            "python_version": expected_runtime.get("python_version"),
            "packages": expected_runtime.get("packages"),
        }
        if self.policy["runtime_contract"].get("require_current_environment_match") is not True:
            raise FinalizerError("current runtime matching must remain enabled")
        if self.current_runtime != expected_current:
            raise FinalizerError(
                "current environment differs from frozen replay runtime; run with adventurex/.ai_curation_venv"
            )

    def _validate_rows(self) -> None:
        expected_count = self.policy["expected_candidate_count"]
        if len(self.ensemble_results) != expected_count or len(self.metadata_records) != expected_count:
            raise FinalizerError("source row count differs from frozen candidate count")
        metadata_by_asset: dict[str, dict[str, Any]] = {}
        for index, metadata in enumerate(self.metadata_records, start=1):
            _require_exact_fields(metadata, METADATA_FIELDS, f"metadata row {index}")
            asset = metadata.get("asset")
            if not isinstance(asset, str) or asset in metadata_by_asset:
                raise FinalizerError(f"metadata row {index} has a missing/duplicate asset")
            if metadata.get("schema_version") != "rootscope.ai_metadata_risk_triage.v1":
                raise FinalizerError(f"metadata row {index} schema mismatch")
            if metadata.get("queue_index") != index:
                raise FinalizerError("metadata rows are not in frozen queue order")
            if metadata.get("metadata_only") is not True or metadata.get("visual_truth_established") is not False:
                raise FinalizerError("metadata record exceeds metadata-only authority")
            if metadata.get("risk_priority") not in {
                "HIGH_METADATA_RISK",
                "REVIEW_PRIORITY",
                "NO_OBVIOUS_METADATA_RISK_SIGNAL",
            }:
                raise FinalizerError("unsupported metadata risk priority")
            if not isinstance(metadata.get("risk_flags"), list):
                raise FinalizerError("metadata risk flags must be a list")
            metadata_by_asset[asset] = metadata

        seen: set[str] = set()
        tolerance = float(self.policy["thresholds"]["probability_sum_tolerance"])
        for index, result in enumerate(self.ensemble_results, start=1):
            _require_exact_fields(result, RESULT_FIELDS, f"ensemble result row {index}")
            asset = result.get("asset")
            if not isinstance(asset, str) or asset in seen or asset not in metadata_by_asset:
                raise FinalizerError(f"ensemble result row {index} has a missing/duplicate/unjoined asset")
            seen.add(asset)
            if self.metadata_records[index - 1]["asset"] != asset:
                raise FinalizerError("ensemble and metadata row order differ")
            if result.get("schema_version") != "rootscope.ai_siglip2_ensemble_result.v1":
                raise FinalizerError("unsupported ensemble result schema")
            if result.get("decision") not in SOURCE_DECISIONS:
                raise FinalizerError("unsupported ensemble decision")
            if result.get("acquisition_class_hint") not in ALL_HINTS:
                raise FinalizerError("unsupported acquisition class hint")
            if result.get("top1_class") not in TARGET_CLASSES or result.get("top2_class") not in TARGET_CLASSES:
                raise FinalizerError("invalid top class")
            if result.get("candidate_sha256") is None or not HEX64.fullmatch(result["candidate_sha256"]):
                raise FinalizerError("invalid candidate SHA-256")
            if not asset.endswith("@sha256:" + result["candidate_sha256"]):
                raise FinalizerError("asset does not bind candidate SHA-256")
            if result.get("image_path") != metadata_by_asset[asset].get("local_path"):
                raise FinalizerError("ensemble and metadata image paths differ")
            _require_false_authority(result.get("authority"), INPUT_AUTHORITY_KEYS, "ensemble row authority")
            if result.get("prompt_set_sha256") != self.prompt_set_sha256:
                raise FinalizerError("ensemble row prompt set mismatch")
            if result.get("model_artifact_sha256") != self.model_weights_artifact["sha256"]:
                raise FinalizerError("ensemble row model artifact mismatch")
            prompt_scores = result.get("prompt_scores")
            if not isinstance(prompt_scores, dict) or set(prompt_scores) != set(self.prompt_ids):
                raise FinalizerError("ensemble row prompt scores do not match frozen prompt IDs")
            for prompt_id, score in prompt_scores.items():
                _finite_number(score, f"{asset} prompt {prompt_id}")
            class_probabilities = result.get("class_probabilities")
            if not isinstance(class_probabilities, dict) or set(class_probabilities) != set(TARGET_CLASSES):
                raise FinalizerError("invalid class probability keys")
            class_scores = result.get("class_scores")
            if not isinstance(class_scores, dict) or set(class_scores) != set(TARGET_CLASSES):
                raise FinalizerError("invalid class score keys")
            for class_id in TARGET_CLASSES:
                _finite_number(class_scores[class_id], f"{asset} {class_id} class score")
            class_values = [_probability(class_probabilities[key], f"{asset} {key} probability") for key in TARGET_CLASSES]
            if abs(math.fsum(class_values) - 1.0) > tolerance:
                raise FinalizerError("class probabilities do not sum to one")
            top_order = sorted(TARGET_CLASSES, key=lambda key: (-class_probabilities[key], TARGET_CLASSES.index(key)))
            if result["top1_class"] != top_order[0] or result["top2_class"] != top_order[1]:
                raise FinalizerError("top class fields disagree with class probabilities")
            if abs(_probability(result["top1_probability"], "top1 probability") - class_probabilities[top_order[0]]) > tolerance:
                raise FinalizerError("top1 probability mismatch")
            if abs(_probability(result["top2_probability"], "top2 probability") - class_probabilities[top_order[1]]) > tolerance:
                raise FinalizerError("top2 probability mismatch")
            if abs(
                _finite_number(result["top1_top2_margin"], "top1/top2 margin")
                - (class_probabilities[top_order[0]] - class_probabilities[top_order[1]])
            ) > tolerance:
                raise FinalizerError("top1/top2 margin mismatch")
            hint_agrees = result["acquisition_class_hint"] == result["top1_class"]
            if result.get("acquisition_hint_agrees") is not hint_agrees:
                raise FinalizerError("acquisition hint agreement flag mismatch")
            if result.get("suggested_class") not in ALL_HINTS:
                raise FinalizerError("suggested class is outside the class contract")
            _probability(result["admissibility_probability"], "admissibility probability")
            _probability(result["reject_probability"], "reject probability")
            if abs(result["admissibility_probability"] + result["reject_probability"] - 1.0) > tolerance:
                raise FinalizerError("admissibility/reject probabilities are not complementary")
            reject_scores = result.get("reject_family_scores")
            reject_probabilities = result.get("reject_family_probabilities")
            if not isinstance(reject_scores, dict) or set(reject_scores) != set(self.reject_families):
                raise FinalizerError("invalid reject family score keys")
            if not isinstance(reject_probabilities, dict) or set(reject_probabilities) != set(self.reject_families):
                raise FinalizerError("invalid reject family probability keys")
            for family in self.reject_families:
                _finite_number(reject_scores[family], f"{asset} {family} reject score")
                _probability(reject_probabilities[family], f"{asset} {family} reject probability")
            dominant = max(
                self.reject_families,
                key=lambda family: (reject_probabilities[family], -self.reject_families.index(family)),
            )
            if result.get("dominant_reject_family") != dominant:
                raise FinalizerError("dominant reject family mismatch")
            if abs(result["reject_probability"] - reject_probabilities[dominant]) > tolerance:
                raise FinalizerError("reject probability does not equal dominant family probability")
            if abs(
                _finite_number(result["dominant_reject_score"], "dominant reject score")
                - reject_scores[dominant]
            ) > tolerance:
                raise FinalizerError("dominant reject score mismatch")
        if seen != set(metadata_by_asset):
            raise FinalizerError("metadata join contains unconsumed rows")
        self.metadata_by_asset = metadata_by_asset

    def _current_path_hashes(self) -> dict[str, str]:
        return {
            "ensemble_results_sha256": _sha256_file(self.ensemble_results_path),
            "ensemble_receipt_sha256": _sha256_file(self.ensemble_receipt_path),
            "metadata_records_sha256": _sha256_file(self.metadata_records_path),
            "metadata_receipt_sha256": _sha256_file(self.metadata_receipt_path),
            "ensemble_policy_sha256": _sha256_file(self.ensemble_policy_path),
            "strict_policy_sha256": _sha256_file(self.policy_path),
            "model_provenance_sha256": _sha256_file(self.model_provenance_path),
            "runtime_provenance_sha256": _sha256_file(self.runtime_provenance_path),
        }

    def preflight(self) -> dict[str, Any]:
        self._load()
        return {
            "schema_version": "rootscope.ai_strict_structure_preflight.v1",
            "mode": "FIXTURE" if self.fixture_mode else "PRODUCTION",
            "status": "PASS_READ_ONLY_INPUTS_BOUND_OUTPUT_NOT_WRITTEN",
            "candidate_count": len(self.ensemble_results),
            "input_roots": self.input_roots,
            "strict_policy_sha256": self.policy_sha256,
            "implementation_sha256": self.implementation_sha256,
            "prompt_set_sha256": self.prompt_set_sha256,
            "model_weights_artifact": _public_artifact(self.model_weights_artifact),
            "tokenizer_artifact": _public_artifact(self.tokenizer_artifact),
            "runtime_binding": self.current_runtime,
            "runtime_evidence_scope": self.runtime_provenance["evidence_scope"],
            "human_decisions_root_sha256": self.human_root_before["sha256"],
            "output_exists": self.output_dir.exists(),
            "authority": OUTPUT_AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }

    def _classify(self, result: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
        contract = self.policy["prompt_contract"]
        thresholds = self.policy["thresholds"]
        scores = result["prompt_scores"]
        complete = float(scores[contract["complete_plant_prompt"]])
        reject_ids = contract["hard_reject_prompts"]
        max_reject_id = max(reject_ids, key=lambda prompt_id: scores[prompt_id])
        max_reject = float(scores[max_reject_id])
        min_quality_id = min(contract["quality_prompts"], key=lambda prompt_id: scores[prompt_id])
        min_quality = float(scores[min_quality_id])
        top1 = result["top1_class"]
        anchors = contract["class_structure_anchors"][top1]
        min_anchor_id = min(anchors, key=lambda prompt_id: scores[prompt_id])
        min_anchor = float(scores[min_anchor_id])
        missing = float(scores["reject.detail_crop.missing_base_or_crown"])
        macro = float(scores["reject.detail_crop.macro_parts"])
        metadata_flags = sorted(
            {
                flag.get("id")
                for flag in metadata["risk_flags"]
                if isinstance(flag, dict) and isinstance(flag.get("id"), str)
            }
        )
        named_blocks = {
            name: [
                prompt_id
                for prompt_id in prompt_ids
                if complete - float(scores[prompt_id]) < thresholds["positive"]["minimum_complete_vs_max_individual_reject_margin"]
            ]
            for name, prompt_ids in contract["named_hard_blocks"].items()
        }
        named_blocks = {name: ids for name, ids in named_blocks.items() if ids}
        metrics = {
            "complete_plant_score": _round(complete),
            "max_individual_reject_prompt": max_reject_id,
            "max_individual_reject_score": _round(max_reject),
            "complete_vs_max_individual_reject_margin": _round(complete - max_reject),
            "complete_vs_missing_base_or_crown_margin": _round(complete - missing),
            "complete_vs_macro_parts_margin": _round(complete - macro),
            "minimum_quality_prompt": min_quality_id,
            "minimum_quality_score": _round(min_quality),
            "minimum_quality_vs_max_individual_reject_margin": _round(min_quality - max_reject),
            "minimum_class_structure_anchor": min_anchor_id,
            "minimum_class_structure_anchor_score": _round(min_anchor),
            "minimum_class_anchor_vs_max_individual_reject_margin": _round(min_anchor - max_reject),
            "named_hard_blocks": named_blocks,
        }

        positive = thresholds["positive"]
        positive_failures: list[str] = []
        checks = (
            (result["decision"] == positive["required_source_decision"], "SOURCE_DECISION_NOT_AUTO_TARGET"),
            (result["admissibility_probability"] >= positive["minimum_admissibility_probability"], "ADMISSIBILITY_BELOW_STRICT_MINIMUM"),
            (result["reject_probability"] <= positive["maximum_reject_probability"], "REJECT_PROBABILITY_ABOVE_STRICT_MAXIMUM"),
            (result["top1_probability"] >= positive["minimum_top1_probability"], "TOP1_PROBABILITY_BELOW_STRICT_MINIMUM"),
            (result["top1_top2_margin"] >= positive["minimum_top1_top2_margin"], "TOP1_MARGIN_BELOW_STRICT_MINIMUM"),
            (result["acquisition_hint_agrees"] is True, "ACQUISITION_HINT_DISAGREES"),
            (result["acquisition_class_hint"] == top1, "ACQUISITION_HINT_NOT_TOP1"),
            (complete - max_reject >= positive["minimum_complete_vs_max_individual_reject_margin"], "COMPLETE_PLANT_NOT_ABOVE_MAX_INDIVIDUAL_REJECT"),
            (complete - missing >= positive["minimum_complete_vs_missing_base_or_crown_margin"], "MISSING_BASE_OR_CROWN_NOT_CLEARED"),
            (complete - macro >= positive["minimum_complete_vs_macro_parts_margin"], "MACRO_OR_PART_CROP_NOT_CLEARED"),
            (min_quality - max_reject >= positive["minimum_min_quality_vs_max_individual_reject_margin"], "ONE_OR_MORE_QUALITY_PROMPTS_NOT_CLEARED"),
            (min_anchor - max_reject >= positive["minimum_class_anchor_vs_max_individual_reject_margin"], "CLASS_STRUCTURE_ANCHORS_NOT_CLEARED"),
            (metadata["risk_priority"] not in positive["blocked_metadata_risk_priorities"], "BLOCKED_METADATA_RISK_PRIORITY"),
            (not set(metadata_flags).intersection(positive["blocked_metadata_flag_ids"]), "BLOCKED_METADATA_FLAG"),
            (not named_blocks, "NAMED_STRUCTURAL_BLOCK_PRESENT"),
        )
        for passed, reason in checks:
            if not passed:
                positive_failures.append(reason)
        if not positive_failures:
            final_label = "POSITIVE_CANDIDATE_" + top1.upper()
            reasons = ["STRICT_POSITIVE_STRUCTURE_GATE_PASS"]
            candidate_class = top1
        else:
            unknown = thresholds["unknown"]
            unknown_failures: list[str] = []
            unknown_checks = (
                (result["acquisition_class_hint"] == unknown["required_acquisition_hint"], "UNKNOWN_HINT_REQUIRED"),
                (result["decision"] == unknown["required_source_decision"], "SOURCE_DECISION_NOT_AUTO_UNKNOWN"),
                (result["admissibility_probability"] <= unknown["maximum_admissibility_probability"], "ADMISSIBILITY_ABOVE_UNKNOWN_MAXIMUM"),
                (result["reject_probability"] >= unknown["minimum_reject_probability"], "REJECT_PROBABILITY_BELOW_UNKNOWN_MINIMUM"),
                (result["top1_probability"] <= unknown["maximum_top1_probability"], "TARGET_PROBABILITY_ABOVE_UNKNOWN_MAXIMUM"),
                (max_reject - complete >= unknown["minimum_max_individual_reject_vs_complete_margin"], "INDIVIDUAL_REJECT_NOT_ABOVE_COMPLETE_PLANT"),
                (metadata["risk_priority"] not in unknown["blocked_metadata_risk_priorities"], "BLOCKED_METADATA_RISK_PRIORITY"),
            )
            for passed, reason in unknown_checks:
                if not passed:
                    unknown_failures.append(reason)
            if not unknown_failures:
                final_label = "UNKNOWN_CANDIDATE"
                reasons = ["STRICT_UNKNOWN_NEGATIVE_GATE_PASS"]
                candidate_class = "unknown"
            else:
                exclusion = thresholds["exclusion"]
                exclusion_reasons: list[str] = []
                flagged = sorted(set(metadata_flags).intersection(exclusion["metadata_flag_ids"]))
                if flagged:
                    exclusion_reasons.append("EXCLUSION_METADATA_FLAG:" + ",".join(flagged))
                if complete - max_reject <= exclusion["maximum_complete_vs_max_individual_reject_margin"]:
                    exclusion_reasons.append("INDIVIDUAL_REJECT_EQUALS_OR_EXCEEDS_COMPLETE_PLANT")
                if exclusion["exclude_failed_source_auto_target"] and result["decision"] == "AUTO_TARGET":
                    exclusion_reasons.append("SOURCE_AUTO_TARGET_FAILED_STRICT_STRUCTURE_GATE")
                if (
                    exclusion["exclude_target_hint_source_auto_unknown"]
                    and result["acquisition_class_hint"] in TARGET_CLASSES
                    and result["decision"] == "AUTO_UNKNOWN"
                ):
                    exclusion_reasons.append("TARGET_HINT_REJECTED_BY_SOURCE_ENSEMBLE")
                if exclusion_reasons:
                    final_label = "EXCLUDE_NONCONFORMING"
                    reasons = exclusion_reasons
                else:
                    final_label = "HOLD_INSUFFICIENT_EVIDENCE"
                    reasons = ["STRICT_GATES_INCONCLUSIVE_REACQUIRE_OR_REVIEW"]
                candidate_class = ""
        if final_label not in FINAL_LABELS:
            raise FinalizerError("internal final label escaped the policy contract")
        return {
            "schema_version": self.policy["result_schema_version"],
            "mode": "FIXTURE" if self.fixture_mode else "PRODUCTION",
            "asset": result["asset"],
            "pageid": result["pageid"],
            "candidate_sha256": result["candidate_sha256"],
            "image_path": result["image_path"],
            "source_ensemble_decision": result["decision"],
            "source_ensemble_result_sha256": _sha256_bytes(_canonical_bytes(result)),
            "source_metadata_record_sha256": _sha256_bytes(_canonical_bytes(metadata)),
            "acquisition_class_hint": result["acquisition_class_hint"],
            "source_top1_class": top1,
            "source_top1_probability": result["top1_probability"],
            "source_top1_top2_margin": result["top1_top2_margin"],
            "source_admissibility_probability": result["admissibility_probability"],
            "source_reject_probability": result["reject_probability"],
            "metadata_risk_priority": metadata["risk_priority"],
            "metadata_risk_flag_ids": metadata_flags,
            "strict_metrics": metrics,
            "positive_gate_failures": positive_failures,
            "final_label": final_label,
            "candidate_class": candidate_class,
            "decision_reasons": reasons,
            "machine_curated_only": True,
            "human_reviewed": False,
            "training_eligible": False,
            "print_eligible": False,
            "split": "UNASSIGNED_DO_NOT_TRAIN",
            "authority": OUTPUT_AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }

    def run(self) -> dict[str, Any]:
        preflight = self.preflight()
        if self.output_dir.exists():
            raise FinalizerError("strict final-label output already exists; immutable runs are never overwritten")
        labels = [
            self._classify(result, self.metadata_by_asset[result["asset"]])
            for result in self.ensemble_results
        ]
        counts = Counter(label["final_label"] for label in labels)
        positive_counts = {
            class_id: counts["POSITIVE_CANDIDATE_" + class_id.upper()]
            for class_id in TARGET_CLASSES
        }
        output = self.policy["output_contract"]
        holds = [
            {
                "schema_version": "rootscope.ai_strict_structure_hold.v1",
                "asset": label["asset"],
                "pageid": label["pageid"],
                "candidate_sha256": label["candidate_sha256"],
                "image_path": label["image_path"],
                "final_label": label["final_label"],
                "decision_reasons": label["decision_reasons"],
                "label_record_sha256": _sha256_bytes(_canonical_bytes(label)),
                "authority": OUTPUT_AUTHORITY,
                "explicit_non_claims": self.policy["explicit_non_claims"],
            }
            for label in labels
            if label["final_label"] == "HOLD_INSUFFICIENT_EVIDENCE"
        ]
        status = output["status"]
        if self.fixture_mode:
            status = "FIXTURE_" + status
        stats = {
            "schema_version": self.policy["stats_schema_version"],
            "mode": "FIXTURE" if self.fixture_mode else "PRODUCTION",
            "status": status,
            "candidate_count": len(labels),
            "final_label_counts": {label: counts[label] for label in sorted(FINAL_LABELS)},
            "positive_candidate_counts_by_class": positive_counts,
            "machine_curated_only": True,
            "thresholds": self.policy["thresholds"],
            "authority": OUTPUT_AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }
        labels_payload = _jsonl_bytes(labels)
        holds_payload = _jsonl_bytes(holds)
        stats_payload = _json_bytes(stats)
        output_hashes = {
            output["labels_filename"]: _sha256_bytes(labels_payload),
            output["holds_filename"]: _sha256_bytes(holds_payload),
            output["stats_filename"]: _sha256_bytes(stats_payload),
        }
        if self._current_path_hashes() != self._input_path_hashes:
            raise FinalizerError("a frozen source changed during strict finalization")
        human_after = _artifact_root(
            self.human_decisions, "formal human_decisions guard", allow_empty=True
        )
        if human_after["sha256"] != self.human_root_before["sha256"]:
            raise FinalizerError("formal human_decisions changed during strict finalization")
        run_binding = {
            "strict_policy_sha256": self.policy_sha256,
            "implementation_sha256": self.implementation_sha256,
            "input_roots": self.input_roots,
            "model_weights_artifact_sha256": self.model_weights_artifact["sha256"],
            "tokenizer_artifact_sha256": self.tokenizer_artifact["sha256"],
            "runtime_provenance_sha256": self.input_roots["runtime_provenance_sha256"],
            "current_runtime": self.current_runtime,
            "prompt_set_sha256": self.prompt_set_sha256,
            "human_decisions_root_sha256": self.human_root_before["sha256"],
            "labels_sha256": output_hashes[output["labels_filename"]],
        }
        receipt = {
            "schema_version": self.policy["receipt_schema_version"],
            "mode": "FIXTURE" if self.fixture_mode else "PRODUCTION",
            "status": status,
            "run_id": "sha256:" + _sha256_bytes(_canonical_bytes(run_binding)),
            "strict_policy_sha256": self.policy_sha256,
            "implementation_sha256": self.implementation_sha256,
            "input_roots": self.input_roots,
            "candidate_count": len(labels),
            "counts": {
                "positive_candidates": sum(positive_counts.values()),
                "unknown_candidates": counts["UNKNOWN_CANDIDATE"],
                "excluded_nonconforming": counts["EXCLUDE_NONCONFORMING"],
                "hold_insufficient_evidence": counts["HOLD_INSUFFICIENT_EVIDENCE"],
            },
            "model_weights_artifact": _public_artifact(self.model_weights_artifact),
            "tokenizer_artifact": _public_artifact(self.tokenizer_artifact),
            "runtime_binding": self.current_runtime,
            "runtime_evidence_scope": self.runtime_provenance["evidence_scope"],
            "original_ensemble_runtime_proven": False,
            "prompt_set_sha256": self.prompt_set_sha256,
            "human_decisions_root_before_sha256": self.human_root_before["sha256"],
            "human_decisions_root_after_sha256": human_after["sha256"],
            "human_review_files_touched": False,
            "dataset_manifest_written": False,
            "outputs": output_hashes,
            "output_scope": "review/ai_final_labels_v1 only",
            "authority": OUTPUT_AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }
        payloads = {
            output["labels_filename"]: labels_payload,
            output["holds_filename"]: holds_payload,
            output["stats_filename"]: stats_payload,
            output["receipt_filename"]: _json_bytes(receipt),
        }
        self._commit(payloads)
        return {"preflight": preflight, "receipt": receipt}

    def _commit(self, payloads: Mapping[str, bytes]) -> None:
        parent = self.output_dir.parent.resolve(strict=True)
        if self.output_dir.name != self.policy["output_contract"]["directory_name"]:
            raise FinalizerError("output scope changed before commit")
        temporary = Path(tempfile.mkdtemp(prefix=".ai_final_labels_v1.tmp-", dir=str(parent)))
        try:
            if temporary.parent.resolve(strict=True) != parent:
                raise FinalizerError("temporary output escaped the review root")
            for name, payload in payloads.items():
                if Path(name).name != name:
                    raise FinalizerError("output filename is not a basename")
                with (temporary / name).open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            if self.output_dir.exists():
                raise FinalizerError("strict final-label output appeared during commit")
            os.replace(temporary, self.output_dir)
        except Exception:
            if temporary.exists() and temporary.parent.resolve(strict=True) == parent:
                shutil.rmtree(temporary)
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RootScope strict structure finalizer: machine-only candidates, never "
            "human review, training eligibility, print eligibility, or DATA_LOCKED."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="validate and hash inputs; write nothing")
    mode.add_argument("--run", action="store_true", help="write one new immutable ai_final_labels_v1 directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        finalizer = StrictStructureFinalizer(FinalizerConfig())
        result = finalizer.preflight() if args.preflight else finalizer.run()["receipt"]
    except FinalizerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
