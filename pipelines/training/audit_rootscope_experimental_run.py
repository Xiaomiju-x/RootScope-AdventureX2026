#!/usr/bin/env python3
"""Independent, fail-closed audit of a RootScope experimental training run.

This module deliberately does not import the training pipeline.  It treats the
run, its frozen v3 input pack, and the current pipeline source as mutually
untrusted evidence and rebuilds their bindings from bytes on disk.

The audit is read-only with respect to the run and datasets.  ``--evidence-out``
may be used to write a separate audit report after a real non-smoke run exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


CLASS_NAMES = ("grass_clump", "low_shrub", "young_tree", "unknown")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INPUT_SHAPE = [1, 3, 224, 224]
ONNX_OPSET = 11
EXPECTED_SEED_COUNT = 3
RUN_SCHEMA = "rootscope.machine_curated_experimental_training_receipt.v1"
RUN_STATUS = "MACHINE_CURATED_EXPERIMENTAL_MODEL_NOT_QUALIFIED"
PACK_PATH = "datasets/rootscope_machine_curated_provisional_v3"
PACK_PATH_NATIVE = str(Path("datasets") / "rootscope_machine_curated_provisional_v3")
PACK_SCHEMA = "rootscope.machine_curated_provisional_receipt.v3"
PACK_STATUS = (
    "MACHINE_CURATED_EXPERIMENTAL_V3_ONLY_NOT_HUMAN_REVIEWED_NOT_A1_NOT_DATA_LOCKED"
)
TRAIN_ROLE = "EXPERIMENTAL_TRAIN_SUGGESTION"
VAL_ROLE = "EXPERIMENTAL_VAL_SUGGESTION"
PRINT_ROLE = "PRINT_DEMO_HOLDOUT_NOT_TRAIN"
NATURAL_VAL_DOMAIN = "NATURAL_WEB_VALIDATION"
PRINT_DOMAIN = "DIGITAL_PRINT_SOURCE_HOLDOUT_NOT_UVC_RECAPTURE"
ONNX_NAME = "model_static_b1x3x224x224_opset11.onnx"
DECISION_RULE = (
    "accept iff max_softmax >= confidence_threshold AND top1_minus_top2 >= "
    "margin_threshold AND the predicted class has acceptance_enabled=true from "
    "validation support plus Wilson-lower-bound evidence; else REJECT"
)
REQUIRED_AUTHORITY_KEYS = {
    "data_locked",
    "dataset_manifest_write",
    "human_review",
    "model_qualification",
    "print_eligibility",
    "rights_approval",
    "split_assignment",
    "training_eligibility",
    "visual_truth",
}
REQUIRED_NON_CLAIMS = {
    "FORMAL_A1_DATASET",
    "HUMAN_REVIEWED",
    "RIGHTS_APPROVED",
    "TRAIN_READY",
    "PRINT_ELIGIBLE",
    "DATA_LOCKED",
    "MODEL_QUALIFIED",
    "REAL_PRINT_DOMAIN_PASSED",
    "UVC_RECAPTURE_PASSED",
    "BPU_COMPILED",
    "X5_READY",
}
REQUIRED_PROBES = {
    *(f"natural_validation_first_{name}" for name in CLASS_NAMES),
    "synthetic_zero",
    "synthetic_ramp",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
SUM_RE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")


class AuditError(RuntimeError):
    """A fail-closed audit gate rejected the run."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise AuditError(f"cannot read JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise AuditError(f"blank JSONL row at {path}:{line_number}")
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise AuditError(f"invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise AuditError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise AuditError(f"JSONL is empty: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise AuditError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    rows: list[str] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        if path.is_symlink():
            raise AuditError(f"symlink is not allowed in audited tree: {path}")
        rows.append(f"{path.relative_to(root).as_posix()}\0{sha256_file(path)}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _canonical_relative(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise AuditError(f"{location} must be a non-empty canonical POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise AuditError(f"unsafe or non-canonical path at {location}: {value!r}")
    return value


def safe_child(root: Path, relative: Any, *, location: str, must_exist: bool = True) -> Path:
    canonical = _canonical_relative(relative, location=location)
    candidate = (root / Path(*PurePosixPath(canonical).parts)).resolve(strict=must_exist)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise AuditError(f"path escapes audited root at {location}: {relative!r}") from error
    if candidate.is_symlink():
        raise AuditError(f"symlink is not allowed at {location}: {relative!r}")
    return candidate


def parse_sha256sums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise AuditError(f"cannot read SHA256SUMS: {error}") from error
    if not lines:
        raise AuditError("SHA256SUMS is empty")
    result: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        match = SUM_RE.fullmatch(line)
        if match is None:
            raise AuditError(f"invalid SHA256SUMS row {line_number}")
        digest, relative = match.groups()
        relative = _canonical_relative(relative, location=f"SHA256SUMS:{line_number}")
        if relative == "SHA256SUMS":
            raise AuditError("SHA256SUMS must not self-reference")
        if relative in result:
            raise AuditError(f"duplicate SHA256SUMS path: {relative}")
        result[relative] = digest
    return result


def audit_full_hash_coverage(run_root: Path) -> dict[str, str]:
    sums_path = run_root / "SHA256SUMS"
    if not sums_path.is_file() or sums_path.is_symlink():
        raise AuditError("missing regular SHA256SUMS")
    sums = parse_sha256sums(sums_path)
    actual_files: set[str] = set()
    for path in run_root.rglob("*"):
        if path.is_symlink():
            raise AuditError(f"symlink is not allowed in run: {path}")
        if path.is_file():
            relative = path.relative_to(run_root).as_posix()
            if relative != "SHA256SUMS":
                actual_files.add(relative)
    if set(sums) != actual_files:
        missing = sorted(actual_files - set(sums))
        stale = sorted(set(sums) - actual_files)
        raise AuditError(f"SHA256SUMS is not full coverage: uncovered={missing}, stale={stale}")
    for relative, expected in sums.items():
        path = safe_child(run_root, relative, location=f"SHA256SUMS[{relative}]")
        if not path.is_file():
            raise AuditError(f"SHA256SUMS target is not a file: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise AuditError(f"SHA256 mismatch for {relative}: {actual} != {expected}")
    return sums


def _require_bool(mapping: Mapping[str, Any], key: str, expected: bool, *, location: str) -> None:
    if mapping.get(key) is not expected:
        raise AuditError(f"{location}.{key} must be exactly {expected!r}")


def _require_sha(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise AuditError(f"{location} must be a lowercase SHA-256")
    return value


def _require_finite(value: Any, *, location: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise AuditError(f"{location} must be finite numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise AuditError(f"{location} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise AuditError(f"{location} must be <= {maximum}")
    return result


def _require_nonnegative_int(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditError(f"{location} must be a non-negative integer")
    return value


def _close(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def wilson_lower_bound(correct: int, total: int, *, z: float) -> float | None:
    if total <= 0:
        return None
    proportion = correct / total
    denominator = 1.0 + (z * z) / total
    centre = proportion + (z * z) / (2.0 * total)
    radius = z * math.sqrt(
        (proportion * (1.0 - proportion) + (z * z) / (4.0 * total)) / total
    )
    return max(0.0, (centre - radius) / denominator)


def audit_receipt_authority(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RUN_SCHEMA:
        raise AuditError("unexpected run receipt schema")
    if receipt.get("status") != RUN_STATUS:
        raise AuditError("run must have the non-qualified experimental model status")
    _require_bool(receipt, "smoke_only", False, location="run_receipt")
    for field in (
        "formal_a1_dataset",
        "human_reviewed",
        "rights_approved",
        "rights",
        "training_eligible",
        "print_eligible",
        "data_locked",
        "model_qualified",
        "model_candidate",
        "x5_ready",
        "bpu_compiled",
        "physical_print_tested",
        "uvc_recapture_evaluated",
        "execution_authority",
    ):
        _require_bool(receipt, field, False, location="run_receipt")
    _require_bool(receipt, "experimental_model_candidate", True, location="run_receipt")
    _require_bool(receipt, "ack_machine_curated_experimental_only", True, location="run_receipt")
    _require_bool(receipt, "long_training_coverage_gate_passed", True, location="run_receipt")
    authority = receipt.get("authority")
    if not isinstance(authority, dict) or set(authority) != REQUIRED_AUTHORITY_KEYS:
        raise AuditError("run receipt authority key set is incomplete or expanded")
    for key in sorted(REQUIRED_AUTHORITY_KEYS):
        _require_bool(authority, key, False, location="run_receipt.authority")
    non_claims = receipt.get("explicit_non_claims")
    if not isinstance(non_claims, list) or not REQUIRED_NON_CLAIMS.issubset(set(non_claims)):
        raise AuditError("run receipt explicit_non_claims is incomplete")
    visual = receipt.get("machine_visual_review_evidence")
    if not isinstance(visual, dict):
        raise AuditError("run receipt lacks machine visual review evidence")
    _require_bool(visual, "human_reviewed", False, location="machine_visual_review_evidence")
    _require_bool(
        visual,
        "non_smoke_training_evidence_eligible",
        True,
        location="machine_visual_review_evidence",
    )


def audit_pipeline_and_model_card(
    workspace: Path,
    run_root: Path,
    receipt: Mapping[str, Any],
) -> dict[str, str]:
    pipeline = workspace / "tools" / "training" / "rootscope_machine_curated_pipeline.py"
    if not pipeline.is_file():
        raise AuditError("current training pipeline source is missing")
    pipeline_sha = sha256_file(pipeline)
    if receipt.get("training_pipeline_sha256") != pipeline_sha:
        raise AuditError("receipt is not bound to the current training pipeline bytes")
    source = pipeline.read_text(encoding="utf-8")
    required_source_fragments = (
        'temperature_record = fit_temperature(val_logits, val_labels)',
        'selected = max(seed_results, key=lambda value: tuple(value["selection_key"]))',
        '"model_candidate": False',
        '"experimental_model_candidate": not args.smoke',
        '"x5_ready": False',
        '"bpu_compiled": False',
        '"physical_print_tested": False',
        '"digital_print_source_holdout_is_uvc_recapture": False',
    )
    missing = [fragment for fragment in required_source_fragments if fragment not in source]
    if missing:
        raise AuditError(f"current pipeline lacks required fail-closed/selection semantics: {missing}")
    if "fit_temperature(print_logits" in source or "calibrate_rejection(print_logits" in source:
        raise AuditError("pipeline attempts to calibrate on the digital print-source holdout")

    card_path = run_root / "MODEL_CARD.md"
    if not card_path.is_file():
        raise AuditError("MODEL_CARD.md is missing")
    card = card_path.read_text(encoding="utf-8")
    required_card_fragments = (
        "not formal A1 data",
        "not human-reviewed truth",
        "not rights-approved",
        "not print-eligible",
        "not data-locked",
        "not formally training-eligible",
        "not qualified for deployment or irrigation decisions",
        "not evidence from a physical print",
        "never tuned weights, checkpoint selection, temperature, or rejection thresholds",
        "`model_candidate=false`",
        "not BPU conversion",
        "X5 runtime readiness",
        "physical-domain accuracy",
    )
    card_plain = card.replace("*", "")
    missing_card = [fragment for fragment in required_card_fragments if fragment not in card_plain]
    if missing_card:
        raise AuditError(f"model card is missing required non-claims: {missing_card}")
    forbidden = (
        "model_candidate=true",
        "x5_ready=true",
        "bpu_compiled=true",
        "physical_print_tested=true",
        "production ready",
        "production-ready",
        "real print domain passed",
        "is qualified for deployment",
        "is qualified for irrigation",
    )
    lowered = card.lower()
    present = [phrase for phrase in forbidden if phrase in lowered]
    if present:
        raise AuditError(f"model card contains forbidden qualification claims: {present}")
    return {"pipeline_sha256": pipeline_sha, "model_card_sha256": sha256_file(card_path)}


def audit_input_bindings(
    workspace: Path,
    run_root: Path,
    receipt: Mapping[str, Any],
    sums: Mapping[str, str],
) -> tuple[Path, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if receipt.get("input_pack") not in {PACK_PATH, PACK_PATH_NATIVE} or receipt.get("input_pack_status") != PACK_STATUS:
        raise AuditError("run is not bound to the canonical frozen v3 pack")
    pack_root = safe_child(workspace, PACK_PATH, location="run_receipt.input_pack")
    if not pack_root.is_dir() or pack_root.name != "rootscope_machine_curated_provisional_v3":
        raise AuditError("canonical v3 input pack directory is missing")
    input_audit_path = run_root / "input_audit.json"
    if not input_audit_path.is_file() or "input_audit.json" not in sums:
        raise AuditError("input_audit.json is absent or not hash-covered")
    if receipt.get("input_audit_sha256") != sha256_file(input_audit_path):
        raise AuditError("receipt input_audit_sha256 mismatch")
    input_audit = load_json(input_audit_path)
    if input_audit.get("schema_version") != "rootscope.machine_curated_training_input_audit.v1":
        raise AuditError("unexpected input audit schema")
    if input_audit.get("status") != PACK_STATUS or input_audit.get("class_order") != list(CLASS_NAMES):
        raise AuditError("input audit does not describe the canonical v3 class contract")
    for field in ("formal_a1_dataset", "human_reviewed", "training_eligible", "data_locked"):
        _require_bool(input_audit, field, False, location="input_audit")
    _require_bool(input_audit, "long_training_coverage_gate_passed", True, location="input_audit")
    _require_bool(input_audit, "print_evaluation_is_uvc_recapture", False, location="input_audit")
    if input_audit.get("print_evaluation_domain") != PRINT_DOMAIN:
        raise AuditError("input audit print domain is overstated or changed")
    if receipt.get("machine_visual_review_evidence") != input_audit.get("machine_visual_review_evidence"):
        raise AuditError("receipt and input audit machine visual evidence differ")

    pack_receipt_path = pack_root / "receipt.json"
    pack_manifest_path = pack_root / "manifest.jsonl"
    pack_receipt = load_json(pack_receipt_path)
    if pack_receipt.get("schema_version") != PACK_SCHEMA or pack_receipt.get("status") != PACK_STATUS:
        raise AuditError("current input pack is not the frozen v3 contract")
    for field in (
        "formal_a1_dataset",
        "human_reviewed",
        "rights_approved",
        "training_eligible",
        "print_eligible",
        "data_locked",
    ):
        _require_bool(pack_receipt, field, False, location="pack_receipt")
    manifest_sha = sha256_file(pack_manifest_path)
    if pack_receipt.get("manifest_sha256") != manifest_sha:
        raise AuditError("current v3 receipt does not bind its manifest")
    snapshot = input_audit.get("immutable_snapshot")
    if not isinstance(snapshot, dict):
        raise AuditError("input audit immutable_snapshot is missing")
    if snapshot.get("manifest_sha256") != manifest_sha:
        raise AuditError("input audit manifest SHA differs from current v3 manifest")
    if snapshot.get("receipt_sha256") != sha256_file(pack_receipt_path):
        raise AuditError("input audit v3 receipt SHA differs from current bytes")
    current_pack_tree = tree_sha256(pack_root)
    if snapshot.get("pack_tree_sha256") != current_pack_tree:
        raise AuditError("current v3 tree changed after training input audit")

    immutable = receipt.get("input_and_formal_authority_unchanged")
    if not isinstance(immutable, dict) or immutable.get("unchanged") is not True:
        raise AuditError("run receipt lacks immutable input/formal evidence")
    if immutable.get("before") != snapshot or immutable.get("after") != snapshot:
        raise AuditError("run before/after immutable snapshots do not equal input audit")

    frozen_v2 = pack_receipt.get("frozen_v2")
    if not isinstance(frozen_v2, dict) or frozen_v2.get("unchanged") is not True:
        raise AuditError("v3 receipt lacks unchanged frozen-v2 binding")
    frozen_v2_root = safe_child(workspace, frozen_v2.get("path"), location="pack_receipt.frozen_v2.path")
    frozen_v2_receipt_path = frozen_v2_root / "receipt.json"
    if frozen_v2.get("receipt_sha256") != sha256_file(frozen_v2_receipt_path):
        raise AuditError("current frozen-v2 receipt changed")
    frozen_v2_receipt = load_json(frozen_v2_receipt_path)
    formal = frozen_v2_receipt.get("formal_human_decisions")
    if not isinstance(formal, dict) or formal.get("unchanged") is not True:
        raise AuditError("frozen-v2 receipt lacks formal human-decision binding")
    if formal.get("tree_sha256_before") != formal.get("tree_sha256_after"):
        raise AuditError("formal human-decision tree changed during source pack build")
    if formal.get("decision_journal_sha256_before") != formal.get("decision_journal_sha256_after"):
        raise AuditError("formal human-decision journal changed during source pack build")
    formal_root = safe_child(workspace, formal.get("path"), location="formal_human_decisions.path")
    formal_tree = tree_sha256(formal_root)
    formal_journal = sha256_file(formal_root / "decision_journal.jsonl")
    if formal_tree != formal.get("tree_sha256_after") or formal_journal != formal.get(
        "decision_journal_sha256_after"
    ):
        raise AuditError("current formal human-decision evidence changed")
    if snapshot.get("formal_human_decisions_tree_sha256") != formal_tree:
        raise AuditError("input audit formal tree SHA differs from current evidence")
    if snapshot.get("formal_human_decision_journal_sha256") != formal_journal:
        raise AuditError("input audit formal journal SHA differs from current evidence")
    protected = pack_receipt.get("protected_inputs")
    if not isinstance(protected, dict) or protected.get("unchanged") is not True:
        raise AuditError("v3 receipt lacks protected-input preservation evidence")
    if protected.get("before") != protected.get("after"):
        raise AuditError("v3 protected inputs changed during pack build")
    protected_after = protected.get("after")
    if not isinstance(protected_after, dict):
        raise AuditError("v3 protected-input snapshot is invalid")
    if protected_after.get("formal_human_decisions_tree_sha256") != formal_tree:
        raise AuditError("v3 protected formal tree SHA differs from current evidence")
    if protected_after.get("formal_decision_journal_sha256") != formal_journal:
        raise AuditError("v3 protected formal journal SHA differs from current evidence")

    manifest_rows = load_jsonl(pack_manifest_path)
    val_classes: set[str] = set()
    filenames: dict[str, set[str]] = {TRAIN_ROLE: set(), VAL_ROLE: set(), PRINT_ROLE: set()}
    for index, row in enumerate(manifest_rows):
        class_id = row.get("class_id")
        role = row.get("experimental_split_suggestion")
        if class_id not in CLASS_TO_INDEX:
            raise AuditError(f"invalid v3 class at manifest row {index}")
        filename = _canonical_relative(row.get("filename"), location=f"manifest[{index}].filename")
        image_path = safe_child(pack_root, filename, location=f"manifest[{index}].filename")
        if not image_path.is_file():
            raise AuditError(f"manifest image is missing: {filename}")
        if role in filenames:
            if filename in filenames[role]:
                raise AuditError(f"duplicate filename in role {role}: {filename}")
            filenames[role].add(filename)
        if role == VAL_ROLE:
            val_classes.add(str(class_id))
    if val_classes != set(CLASS_NAMES):
        raise AuditError("natural validation does not contain all four classes")
    if not filenames[TRAIN_ROLE] or not filenames[VAL_ROLE] or not filenames[PRINT_ROLE]:
        raise AuditError("train/validation/digital-print-source roles must all be non-empty")
    if any(filenames[left] & filenames[right] for left, right in ((TRAIN_ROLE, VAL_ROLE), (TRAIN_ROLE, PRINT_ROLE), (VAL_ROLE, PRINT_ROLE))):
        raise AuditError("digital print-source holdout overlaps train or validation")
    return pack_root, manifest_rows, input_audit, pack_receipt


def audit_artifact_hash_manifest(
    run_root: Path,
    receipt: Mapping[str, Any],
    sums: Mapping[str, str],
) -> None:
    declared = receipt.get("artifact_hashes_before_receipt")
    if not isinstance(declared, dict):
        raise AuditError("receipt artifact_hashes_before_receipt is missing")
    expected_paths = set(sums) - {"run_receipt.json", "MODEL_CARD.md"}
    if set(declared) != expected_paths:
        raise AuditError("artifact_hashes_before_receipt does not exactly cover pre-receipt artifacts")
    for relative, digest in declared.items():
        _canonical_relative(relative, location="artifact_hashes_before_receipt")
        _require_sha(digest, location=f"artifact_hashes_before_receipt[{relative}]")
        if sums.get(relative) != digest or sha256_file(safe_child(run_root, relative, location=relative)) != digest:
            raise AuditError(f"pre-receipt artifact hash mismatch: {relative}")


def validate_onnx_structure(path: Path) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as error:
        raise AuditError("the independent audit requires onnx") from error
    try:
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
    except Exception as error:
        raise AuditError(f"invalid ONNX graph {path}: {error}") from error
    opsets = {entry.domain: entry.version for entry in graph.opset_import}
    if opsets.get("", opsets.get("ai.onnx")) != ONNX_OPSET:
        raise AuditError(f"ONNX default opset is not {ONNX_OPSET}: {opsets}")
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise AuditError("ONNX must have exactly one input and one output")
    input_value = graph.graph.input[0]
    output_value = graph.graph.output[0]
    input_dims = [dimension.dim_value for dimension in input_value.type.tensor_type.shape.dim]
    output_dims = [dimension.dim_value for dimension in output_value.type.tensor_type.shape.dim]
    if input_value.name != "image" or input_dims != INPUT_SHAPE:
        raise AuditError(f"ONNX input must be static image:{INPUT_SHAPE}, got {input_value.name}:{input_dims}")
    if output_value.name != "logits" or output_dims != [1, len(CLASS_NAMES)]:
        raise AuditError("ONNX output must be static logits:[1,4]")
    node_types = [node.op_type for node in graph.graph.node]
    if "GlobalAveragePool" in node_types:
        raise AuditError("ONNX contains forbidden GlobalAveragePool")
    fixed_pools: list[dict[str, Any]] = []
    for node in graph.graph.node:
        if node.op_type != "AveragePool":
            continue
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        kernel = list(attributes.get("kernel_shape", []))
        stride = list(attributes.get("strides", [1, 1]))
        if kernel == [7, 7] and stride == [1, 1]:
            fixed_pools.append({"name": node.name, "kernel_shape": kernel, "strides": stride})
    if not fixed_pools:
        raise AuditError("ONNX lacks the required fixed 7x7 stride-1 AveragePool")
    return {
        "sha256": sha256_file(path),
        "input_shape": input_dims,
        "output_shape": output_dims,
        "opset": ONNX_OPSET,
        "fixed_average_pool_count": len(fixed_pools),
        "global_average_pool_present": False,
    }


def audit_recorded_consistency(record: Mapping[str, Any], *, seed: int) -> None:
    if record.get("schema_version") != "rootscope.torch_onnx_consistency.v2":
        raise AuditError(f"seed {seed} consistency schema is not v2 multi-probe")
    _require_bool(record, "passed", True, location=f"seed[{seed}].onnx_consistency")
    if record.get("input_shape") != INPUT_SHAPE or record.get("class_order") != list(CLASS_NAMES):
        raise AuditError(f"seed {seed} consistency shape/class order mismatch")
    if record.get("opset") != ONNX_OPSET:
        raise AuditError(f"seed {seed} consistency opset mismatch")
    tolerance = _require_finite(
        record.get("tolerance"), location=f"seed[{seed}].onnx_consistency.tolerance", minimum=0.0, maximum=1e-4
    )
    if tolerance <= 0.0:
        raise AuditError(f"seed {seed} consistency tolerance must be positive")
    probes = record.get("probes")
    if not isinstance(probes, list) or record.get("probe_count") != len(probes):
        raise AuditError(f"seed {seed} consistency probe_count mismatch")
    by_name: dict[str, Mapping[str, Any]] = {}
    probe_maxima: list[float] = []
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict):
            raise AuditError(f"seed {seed} consistency probe {index} is invalid")
        name = probe.get("name")
        if not isinstance(name, str) or name in by_name:
            raise AuditError(f"seed {seed} consistency probe name is missing or duplicated")
        by_name[name] = probe
        if probe.get("input_shape") != INPUT_SHAPE or probe.get("passed") is not True:
            raise AuditError(f"seed {seed} probe {name} did not pass static-shape consistency")
        probe_max = _require_finite(
            probe.get("max_absolute_error"), location=f"seed[{seed}].probe[{name}].max", minimum=0.0
        )
        probe_mean = _require_finite(
            probe.get("mean_absolute_error"), location=f"seed[{seed}].probe[{name}].mean", minimum=0.0
        )
        if probe_mean > probe_max + 1e-12 or probe_max > tolerance:
            raise AuditError(f"seed {seed} probe {name} exceeds its consistency tolerance")
        probe_maxima.append(probe_max)
    if set(by_name) != REQUIRED_PROBES or len(probes) != len(REQUIRED_PROBES):
        raise AuditError(f"seed {seed} does not contain the exact four-class plus zero/ramp probe set")
    if set(record.get("natural_validation_classes_probed", [])) != set(CLASS_NAMES):
        raise AuditError(f"seed {seed} did not probe every natural validation class")
    if record.get("natural_validation_classes_missing") != []:
        raise AuditError(f"seed {seed} reports missing natural validation classes")
    _require_bool(record, "synthetic_zero_probed", True, location=f"seed[{seed}].consistency")
    _require_bool(record, "synthetic_ramp_probed", True, location=f"seed[{seed}].consistency")
    providers = record.get("onnxruntime_providers")
    if not isinstance(providers, list) or "CPUExecutionProvider" not in providers:
        raise AuditError(f"seed {seed} consistency did not use the CPU execution provider")
    recorded_max = _require_finite(
        record.get("max_absolute_error"), location=f"seed[{seed}].consistency.max", minimum=0.0
    )
    if not _close(recorded_max, max(probe_maxima)) or recorded_max > tolerance:
        raise AuditError(f"seed {seed} aggregate consistency error is invalid")


def audit_calibration(calibration: Mapping[str, Any], metrics: Mapping[str, Any], *, seed: int) -> None:
    if calibration.get("status") != "MACHINE_CURATED_EXPERIMENTAL_CALIBRATION_NOT_FORMALLY_QUALIFIED":
        raise AuditError(f"seed {seed} calibration status is overstated or smoke-only")
    if calibration.get("calibration_domain") != NATURAL_VAL_DOMAIN:
        raise AuditError(f"seed {seed} calibration is not validation-only")
    if calibration.get("decision_rule") != DECISION_RULE:
        raise AuditError(f"seed {seed} calibration decision rule changed")
    serialized = json.dumps(calibration, sort_keys=True).lower()
    if any(token in serialized for token in ("print_source", "print_holdout", "uvc", "recapture")):
        raise AuditError(f"seed {seed} calibration contains print/UVC-domain evidence")
    temperature = _require_finite(calibration.get("temperature"), location=f"seed[{seed}].temperature", minimum=0.0)
    if temperature <= 0:
        raise AuditError(f"seed {seed} temperature must be positive")
    target = _require_finite(
        calibration.get("target_accepted_accuracy"), location=f"seed[{seed}].target_accuracy", minimum=0.0, maximum=1.0
    )
    minimum = _require_nonnegative_int(
        calibration.get("per_predicted_class_minimum_accepted"), location=f"seed[{seed}].class_minimum"
    )
    if minimum < 2:
        raise AuditError(f"seed {seed} per-class calibration support minimum is too weak")
    z = _require_finite(calibration.get("wilson_z"), location=f"seed[{seed}].wilson_z", minimum=0.0)
    if z <= 0:
        raise AuditError(f"seed {seed} Wilson z must be positive")
    global_mode_ok = calibration.get("mode") in {
        "VALIDATION_GRID_TARGET_MET_WITH_PER_CLASS_EVIDENCE_GATE",
        "VALIDATION_GRID_TARGET_MET_PER_CLASS_EVIDENCE_REJECTS_ALL",
    }
    evidence = calibration.get("per_predicted_class_evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(CLASS_NAMES):
        raise AuditError(f"seed {seed} per-predicted-class evidence is incomplete")
    final_accepted = 0
    for name in CLASS_NAMES:
        row = evidence[name]
        if not isinstance(row, dict):
            raise AuditError(f"seed {seed} calibration evidence for {name} is invalid")
        support = _require_nonnegative_int(row.get("predicted_validation_support"), location=f"seed[{seed}].{name}.support")
        accepted = _require_nonnegative_int(row.get("global_threshold_accepted_count"), location=f"seed[{seed}].{name}.accepted")
        correct = _require_nonnegative_int(row.get("global_threshold_correct_count"), location=f"seed[{seed}].{name}.correct")
        if correct > accepted or accepted > support:
            raise AuditError(f"seed {seed} calibration counts are inconsistent for {name}")
        if row.get("minimum_accepted_required") != minimum or not _close(float(row.get("target_lower_bound")), target):
            raise AuditError(f"seed {seed} calibration policy differs for {name}")
        expected_accuracy = (correct / accepted) if accepted else None
        actual_accuracy = row.get("global_threshold_accepted_accuracy")
        if expected_accuracy is None:
            if actual_accuracy is not None:
                raise AuditError(f"seed {seed} empty accepted class has an accuracy for {name}")
        elif not _close(_require_finite(actual_accuracy, location=f"seed[{seed}].{name}.accuracy"), expected_accuracy):
            raise AuditError(f"seed {seed} accepted accuracy mismatch for {name}")
        expected_wilson = wilson_lower_bound(correct, accepted, z=z)
        actual_wilson = row.get("wilson_lower_bound")
        if expected_wilson is None:
            if actual_wilson is not None:
                raise AuditError(f"seed {seed} empty class has a Wilson bound for {name}")
        elif not _close(_require_finite(actual_wilson, location=f"seed[{seed}].{name}.wilson"), expected_wilson):
            raise AuditError(f"seed {seed} Wilson lower bound mismatch for {name}")
        reasons: list[str] = []
        if not global_mode_ok:
            reasons.append("GLOBAL_TARGET_NOT_MET")
        if accepted < minimum:
            reasons.append("INSUFFICIENT_ACCEPTED_SUPPORT")
        if expected_wilson is None or expected_wilson + 1e-12 < target:
            reasons.append("WILSON_LOWER_BOUND_BELOW_TARGET")
        enabled = not reasons
        if row.get("acceptance_enabled") is not enabled or row.get("force_reject_reasons") != reasons:
            raise AuditError(f"seed {seed} fail-closed class decision mismatch for {name}")
        if enabled:
            final_accepted += accepted
    if calibration.get("validation_accepted_count") != final_accepted:
        raise AuditError(f"seed {seed} final accepted count ignores the per-class evidence gate")

    if metrics.get("schema_version") != "rootscope.machine_curated_experimental_metrics.v1":
        raise AuditError(f"seed {seed} metrics schema mismatch")
    if metrics.get("status") != RUN_STATUS or metrics.get("seed") != seed:
        raise AuditError(f"seed {seed} metrics status/identity mismatch")
    natural = metrics.get("natural_validation_rejection")
    printed = metrics.get("digital_print_source_holdout_rejection")
    if not isinstance(natural, dict) or not isinstance(printed, dict):
        raise AuditError(f"seed {seed} lacks natural/print rejection metrics")
    if natural.get("domain") != NATURAL_VAL_DOMAIN or natural.get("thresholds_locked_from") != NATURAL_VAL_DOMAIN:
        raise AuditError(f"seed {seed} natural validation threshold provenance mismatch")
    _require_bool(natural, "thresholds_optimized_on_this_domain", True, location=f"seed[{seed}].natural")
    _require_bool(natural, "per_predicted_class_gate_applied", True, location=f"seed[{seed}].natural")
    if printed.get("domain") != PRINT_DOMAIN or printed.get("thresholds_locked_from") != NATURAL_VAL_DOMAIN:
        raise AuditError(f"seed {seed} digital print holdout threshold provenance mismatch")
    _require_bool(printed, "thresholds_optimized_on_this_domain", False, location=f"seed[{seed}].print")
    _require_bool(printed, "per_predicted_class_gate_applied", True, location=f"seed[{seed}].print")
    _require_bool(metrics, "digital_print_source_holdout_is_uvc_recapture", False, location=f"seed[{seed}].metrics")
    if metrics.get("digital_print_source_holdout_claim") != "DIGITAL_SOURCE_EVALUATION_ONLY_NOT_REAL_PRINT_DOMAIN_EVIDENCE":
        raise AuditError(f"seed {seed} digital print-source claim is overstated")


def _validate_checkpoint_metadata(checkpoint: Any, *, seed: int, manifest_sha: str) -> Mapping[str, Any]:
    if not isinstance(checkpoint, dict):
        raise AuditError(f"seed {seed} checkpoint root is not a mapping")
    expected = {
        "schema_version": "rootscope.resnet18_experimental_checkpoint.v1",
        "status": RUN_STATUS,
        "seed": seed,
        "class_order": list(CLASS_NAMES),
        "input_shape": INPUT_SHAPE,
        "architecture": "torchvision.resnet18_fixed_avgpool7x7",
        "input_pack_manifest_sha256": manifest_sha,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise AuditError(f"seed {seed} checkpoint {key} mismatch")
    _require_nonnegative_int(checkpoint.get("epoch"), location=f"seed[{seed}].checkpoint.epoch")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise AuditError(f"seed {seed} checkpoint model_state_dict is empty")
    return checkpoint


def _load_checkpoint(path: Path, *, seed: int, manifest_sha: str) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise AuditError("the independent audit requires torch") from error
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise AuditError(f"cannot safely load checkpoint for seed {seed}: {error}") from error
    return _validate_checkpoint_metadata(checkpoint, seed=seed, manifest_sha=manifest_sha)


def replay_cpu_torch_onnx(
    *,
    checkpoint: Mapping[str, Any],
    onnx_path: Path,
    pack_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    recorded: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Independently rebuild ResNet18 and replay six CPU probes."""

    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        from PIL import Image, ImageOps
        from torchvision import models, transforms
        from torchvision.transforms import InterpolationMode
    except ImportError as error:
        raise AuditError("CPU replay requires numpy, onnxruntime, torch, torchvision, and Pillow") from error

    model = models.resnet18(weights=None)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as error:
        raise AuditError(f"seed {seed} checkpoint does not load into fixed-pool ResNet18: {error}") from error
    model.eval()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    probes: dict[str, Any] = {}
    for row in manifest_rows:
        if row.get("experimental_split_suggestion") != VAL_ROLE:
            continue
        class_id = str(row.get("class_id"))
        name = f"natural_validation_first_{class_id}"
        if name in probes:
            continue
        image_path = safe_child(pack_root, row.get("filename"), location=f"validation_probe[{class_id}]")
        with Image.open(image_path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
        probes[name] = transform(image).unsqueeze(0).cpu()
    probes["synthetic_zero"] = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    probes["synthetic_ramp"] = torch.linspace(
        -2.0, 2.0, steps=3 * 224 * 224, dtype=torch.float32
    ).reshape(INPUT_SHAPE)
    if set(probes) != REQUIRED_PROBES:
        raise AuditError(f"seed {seed} could not rebuild the exact required CPU probe set")
    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as error:
        raise AuditError(f"seed {seed} ONNX CPU session failed: {error}") from error
    tolerance = float(recorded["tolerance"])
    rows: list[dict[str, Any]] = []
    maximum = 0.0
    for name, tensor in probes.items():
        array = tensor.numpy().astype(np.float32, copy=False)
        with torch.inference_mode():
            torch_logits = model(tensor).detach().cpu().numpy()
        try:
            ort_logits = session.run(["logits"], {"image": array})[0]
        except Exception as error:
            raise AuditError(f"seed {seed} ONNX CPU replay failed for {name}: {error}") from error
        if list(torch_logits.shape) != [1, 4] or list(ort_logits.shape) != [1, 4]:
            raise AuditError(f"seed {seed} CPU replay output shape mismatch for {name}")
        absolute = np.abs(torch_logits - ort_logits)
        probe_max = float(absolute.max())
        probe_mean = float(absolute.mean())
        if not math.isfinite(probe_max) or probe_max > tolerance:
            raise AuditError(
                f"seed {seed} independent Torch/ONNX replay mismatch for {name}: {probe_max} > {tolerance}"
            )
        maximum = max(maximum, probe_max)
        rows.append({"name": name, "max_absolute_error": probe_max, "mean_absolute_error": probe_mean})
    return {
        "provider_requested": "CPUExecutionProvider",
        "providers_actual": session.get_providers(),
        "probe_count": len(rows),
        "probe_names": sorted(row["name"] for row in rows),
        "max_absolute_error": maximum,
        "tolerance": tolerance,
        "passed": True,
    }


ReplayRunner = Callable[..., dict[str, Any]]


def audit_seed(
    *,
    workspace: Path,
    run_root: Path,
    pack_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    input_audit: Mapping[str, Any],
    seed_result: Mapping[str, Any],
    sums: Mapping[str, str],
    replay_runner: ReplayRunner,
) -> dict[str, Any]:
    seed = _require_nonnegative_int(seed_result.get("seed"), location="seed_result.seed")
    seed_dir_name = f"seed_{seed:05d}"
    seed_root = run_root / seed_dir_name
    if not seed_root.is_dir():
        raise AuditError(f"missing canonical seed directory: {seed_dir_name}")
    canonical = {
        "checkpoint": f"{seed_dir_name}/best_checkpoint.pt",
        "onnx": f"{seed_dir_name}/{ONNX_NAME}",
        "calibration": f"{seed_dir_name}/calibration.json",
        "metrics": f"{seed_dir_name}/metrics.json",
        "provenance": f"{seed_dir_name}/model_provenance.json",
        "consistency": f"{seed_dir_name}/onnx_consistency.json",
    }
    artifacts = seed_result.get("artifacts")
    if not isinstance(artifacts, dict) or artifacts != {
        "checkpoint": canonical["checkpoint"],
        "onnx": canonical["onnx"],
    }:
        raise AuditError(f"seed {seed} artifact paths are not canonical")
    for label, relative in canonical.items():
        path = safe_child(run_root, relative, location=f"seed[{seed}].{label}")
        if not path.is_file() or relative not in sums:
            raise AuditError(f"seed {seed} {label} is absent or not SHA-covered")
    calibration = load_json(run_root / canonical["calibration"])
    metrics = load_json(run_root / canonical["metrics"])
    provenance = load_json(run_root / canonical["provenance"])
    consistency = load_json(run_root / canonical["consistency"])
    for key, value in (
        ("calibration", calibration),
        ("metrics", metrics),
        ("model_provenance", provenance),
        ("onnx_consistency", consistency),
    ):
        if seed_result.get(key) != value:
            raise AuditError(f"seed {seed} embedded {key} differs from its hash-covered file")

    if provenance.get("architecture") != "torchvision.resnet18":
        raise AuditError(f"seed {seed} model provenance architecture mismatch")
    if provenance.get("input_shape") != INPUT_SHAPE or provenance.get("class_order") != list(CLASS_NAMES):
        raise AuditError(f"seed {seed} model provenance shape/class mismatch")
    _require_bool(provenance, "adaptive_pooling", False, location=f"seed[{seed}].provenance")
    average_pool = provenance.get("average_pool")
    if not isinstance(average_pool, dict) or average_pool.get("kernel_size") != [7, 7] or average_pool.get("stride") != [1, 1]:
        raise AuditError(f"seed {seed} provenance does not bind fixed 7x7 stride-1 pooling")

    audit_calibration(calibration, metrics, seed=seed)
    audit_recorded_consistency(consistency, seed=seed)
    onnx_audit = validate_onnx_structure(run_root / canonical["onnx"])
    manifest_sha = str(input_audit["immutable_snapshot"]["manifest_sha256"])
    checkpoint = _load_checkpoint(run_root / canonical["checkpoint"], seed=seed, manifest_sha=manifest_sha)
    if checkpoint.get("epoch") != seed_result.get("best_epoch") or metrics.get("best_epoch") != seed_result.get("best_epoch"):
        raise AuditError(f"seed {seed} best epoch differs across checkpoint/result/metrics")

    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise AuditError(f"seed {seed} metrics history is empty")
    candidates: list[tuple[tuple[float, float], int]] = []
    for index, epoch_row in enumerate(history):
        if not isinstance(epoch_row, dict):
            raise AuditError(f"seed {seed} history row {index} is invalid")
        natural = epoch_row.get("natural_validation")
        if not isinstance(natural, dict) or natural.get("domain") != NATURAL_VAL_DOMAIN:
            raise AuditError(f"seed {seed} checkpoint history includes a non-validation selection domain")
        key = (
            _require_finite(natural.get("balanced_accuracy_present_classes"), location=f"seed[{seed}].history.balanced"),
            -_require_finite(natural.get("cross_entropy"), location=f"seed[{seed}].history.loss"),
        )
        candidates.append((key, _require_nonnegative_int(epoch_row.get("epoch"), location=f"seed[{seed}].history.epoch")))
    best_key, best_epoch = max(candidates, key=lambda item: item[0])
    selection_key = seed_result.get("selection_key")
    if not isinstance(selection_key, list) or len(selection_key) != 2:
        raise AuditError(f"seed {seed} selection key is invalid")
    if not all(_close(float(selection_key[index]), best_key[index]) for index in range(2)):
        raise AuditError(f"seed {seed} selection key is not derived only from natural validation")
    if seed_result.get("best_epoch") != best_epoch:
        raise AuditError(f"seed {seed} checkpoint was not selected only from natural validation")

    replay = replay_runner(
        checkpoint=checkpoint,
        onnx_path=run_root / canonical["onnx"],
        pack_root=pack_root,
        manifest_rows=manifest_rows,
        recorded=consistency,
        seed=seed,
    )
    if not isinstance(replay, dict) or replay.get("passed") is not True:
        raise AuditError(f"seed {seed} independent CPU multi-probe replay did not pass")
    if replay.get("probe_count") != len(REQUIRED_PROBES) or set(replay.get("probe_names", [])) != REQUIRED_PROBES:
        raise AuditError(f"seed {seed} independent CPU replay did not cover the exact probe contract")
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_key": list(best_key),
        "artifact_sha256": {label: sums[relative] for label, relative in canonical.items()},
        "onnx_structure": onnx_audit,
        "independent_cpu_replay": replay,
        "calibration_per_predicted_class_complete": True,
        "print_source_used_for_selection_or_calibration": False,
    }


def audit_run(
    workspace: Path,
    run_root: Path,
    *,
    replay_runner: ReplayRunner = replay_cpu_torch_onnx,
) -> dict[str, Any]:
    workspace = workspace.resolve(strict=True)
    run_root = run_root.resolve(strict=True)
    try:
        run_root.relative_to(workspace)
    except ValueError as error:
        raise AuditError("run must remain inside the AdventureX workspace") from error
    sums = audit_full_hash_coverage(run_root)
    if "run_receipt.json" not in sums or "MODEL_CARD.md" not in sums:
        raise AuditError("SHA256SUMS must cover run_receipt.json and MODEL_CARD.md")
    receipt = load_json(run_root / "run_receipt.json")
    audit_receipt_authority(receipt)
    audit_artifact_hash_manifest(run_root, receipt, sums)
    source_audit = audit_pipeline_and_model_card(workspace, run_root, receipt)
    pack_root, manifest_rows, input_audit, pack_receipt = audit_input_bindings(
        workspace, run_root, receipt, sums
    )

    seeds = receipt.get("seeds")
    seed_results = receipt.get("seed_results")
    if not isinstance(seeds, list) or len(seeds) != EXPECTED_SEED_COUNT:
        raise AuditError(f"non-smoke run must contain exactly {EXPECTED_SEED_COUNT} seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise AuditError("receipt seeds must be non-negative integers")
    if len(set(seeds)) != EXPECTED_SEED_COUNT:
        raise AuditError("receipt seeds are duplicated")
    if not isinstance(seed_results, list) or len(seed_results) != EXPECTED_SEED_COUNT:
        raise AuditError("receipt seed_results count mismatch")
    result_by_seed: dict[int, Mapping[str, Any]] = {}
    for result in seed_results:
        if not isinstance(result, dict):
            raise AuditError("seed_results entries must be objects")
        seed = result.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in result_by_seed:
            raise AuditError("seed_results seed identity is invalid or duplicated")
        result_by_seed[seed] = result
    if set(result_by_seed) != set(seeds):
        raise AuditError("receipt seeds and seed_results differ")
    actual_seed_dirs = {
        path.name for path in run_root.iterdir() if path.is_dir() and re.fullmatch(r"seed_[0-9]{5}", path.name)
    }
    expected_seed_dirs = {f"seed_{seed:05d}" for seed in seeds}
    if actual_seed_dirs != expected_seed_dirs:
        raise AuditError(f"seed directory set mismatch: actual={actual_seed_dirs}, expected={expected_seed_dirs}")

    seed_audits = [
        audit_seed(
            workspace=workspace,
            run_root=run_root,
            pack_root=pack_root,
            manifest_rows=manifest_rows,
            input_audit=input_audit,
            seed_result=result_by_seed[seed],
            sums=sums,
            replay_runner=replay_runner,
        )
        for seed in seeds
    ]
    selected = receipt.get("selected_seed")
    if not isinstance(selected, dict):
        raise AuditError("receipt selected_seed is missing")
    selected_seed = selected.get("seed")
    if selected_seed not in result_by_seed or selected != result_by_seed[selected_seed]:
        raise AuditError("selected_seed is not an exact member of seed_results")
    expected_selected = max(seed_results, key=lambda value: tuple(value["selection_key"]))
    if selected != expected_selected:
        raise AuditError("selected_seed is not the natural-validation selection-key winner")

    return {
        "schema_version": "rootscope.independent_experimental_run_audit.v1",
        "status": "PASS",
        "auditor_imports_training_pipeline": False,
        "run_relative_path": run_root.relative_to(workspace).as_posix(),
        "run_id": receipt.get("run_id"),
        "run_tree_sha256": tree_sha256(run_root),
        "sha256sums_sha256": sha256_file(run_root / "SHA256SUMS"),
        "run_receipt_sha256": sha256_file(run_root / "run_receipt.json"),
        "full_sha256_coverage": True,
        "hash_covered_file_count": len(sums),
        "input_pack": PACK_PATH,
        "input_pack_tree_sha256": tree_sha256(pack_root),
        "input_manifest_sha256": sha256_file(pack_root / "manifest.jsonl"),
        "input_pack_receipt_sha256": sha256_file(pack_root / "receipt.json"),
        "formal_human_decisions_tree_sha256": input_audit["immutable_snapshot"][
            "formal_human_decisions_tree_sha256"
        ],
        "formal_human_decision_journal_sha256": input_audit["immutable_snapshot"][
            "formal_human_decision_journal_sha256"
        ],
        "input_and_formal_hashes_current_unchanged": True,
        "seed_count": len(seed_audits),
        "seeds": seed_audits,
        "selected_seed": selected_seed,
        "selection_and_calibration_domain": NATURAL_VAL_DOMAIN,
        "digital_print_source_holdout_domain": PRINT_DOMAIN,
        "digital_print_source_used_for_selection_or_calibration": False,
        "model_candidate": False,
        "experimental_model_candidate": True,
        "model_qualified": False,
        "x5_ready": False,
        "bpu_compiled": False,
        "physical_print_tested": False,
        "uvc_recapture_evaluated": False,
        "execution_authority": False,
        "hardware_touched_by_audit": False,
        "network_touched_by_audit": False,
        "pipeline_and_model_card": source_audit,
        "input_pack_status": pack_receipt["status"],
    }


def _write_evidence(path: Path, report: Mapping[str, Any], *, workspace: Path, run_root: Path) -> None:
    path = path.resolve(strict=False)
    try:
        path.relative_to(workspace)
    except ValueError as error:
        raise AuditError("evidence output must remain inside the AdventureX workspace") from error
    try:
        path.relative_to(run_root)
    except ValueError:
        pass
    else:
        raise AuditError("evidence output must not mutate the audited run")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    default_workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=default_workspace)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        workspace = args.workspace.resolve(strict=True)
        run_root = args.run if args.run.is_absolute() else workspace / args.run
        report = audit_run(workspace, run_root)
        if args.evidence_out is not None:
            evidence = args.evidence_out if args.evidence_out.is_absolute() else workspace / args.evidence_out
            _write_evidence(evidence, report, workspace=workspace, run_root=run_root.resolve(strict=True))
    except (AuditError, OSError) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
