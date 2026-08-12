#!/usr/bin/env python3
"""Independent GPU SigLIP2 gate for the 90 whole-plant reacquisition candidates.

This tool is deliberately outside the frozen ``ai_ensemble_v1`` and formal
``human_decisions`` contracts.  It emits machine-only, fail-closed suggestions:

* ``STRICT_POSITIVE_CANDIDATE_<class>``
* ``EXCLUDE``
* ``HOLD``

No outcome grants visual ground truth, rights approval, training eligibility,
split assignment, print eligibility, or DATA_LOCKED authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from ai_siglip2_ensemble import (
    EnsembleError,
    LocalOpenCLIPBigVisionScorer,
    _artifact_root,
    _canonical_bytes,
    _json_bytes,
    _jsonl_bytes,
    _read_json,
    _read_jsonl,
    _safe_relative_path,
    _sha256_bytes,
    _sha256_file,
)


SCRIPT_PATH = Path(__file__).resolve()
ADVENTUREX_ROOT = SCRIPT_PATH.parents[2]
DATASET_ROOT = ADVENTUREX_ROOT / "datasets" / "desert_plants_whole_plant_reacquisition_e1"
DEFAULT_MANIFEST = DATASET_ROOT / "manifest.jsonl"
DEFAULT_SUMMARY = DATASET_ROOT / "summary.json"
DEFAULT_SOURCE_PLAN = DATASET_ROOT / "source_plan.json"
DEFAULT_POLICY = SCRIPT_PATH.with_name("ai_whole_plant_gpu_gate_policy_v1.json")
DEFAULT_OUTPUT = DATASET_ROOT / "review" / "ai_strict_gpu_gate_v1"
DEFAULT_MODEL = ADVENTUREX_ROOT / "models" / "ai_triage" / "siglip2_b16_224_big_vision.npz"
DEFAULT_TOKENIZER = ADVENTUREX_ROOT / "models" / "ai_triage" / "siglip2_tokenizer_75de2d55"
DEFAULT_RUNTIME_PROVENANCE = (
    ADVENTUREX_ROOT / "models" / "ai_triage" / "SIGLIP2_RUNTIME_PROVENANCE.json"
)
DEFAULT_FORMAL_HUMAN_DECISIONS = (
    ADVENTUREX_ROOT
    / "datasets"
    / "desert_plants_wikimedia_staging_e0"
    / "review"
    / "human_decisions"
)

TARGET_CLASSES = ("grass_clump", "low_shrub", "young_tree")
AUTHORITY = {
    "human_review": False,
    "visual_ground_truth": False,
    "rights_approval": False,
    "dataset_manifest_write": False,
    "training_eligibility": False,
    "split_assignment": False,
    "print_eligibility": False,
    "data_locked": False,
}


class GateError(RuntimeError):
    """Fail-closed input, inference, or output error."""


class PromptScorer(Protocol):
    def score(
        self, images: Sequence[Path], prompts: Sequence[Mapping[str, str]]
    ) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class GateConfig:
    dataset_root: Path = DATASET_ROOT
    manifest_path: Path = DEFAULT_MANIFEST
    summary_path: Path = DEFAULT_SUMMARY
    source_plan_path: Path = DEFAULT_SOURCE_PLAN
    policy_path: Path = DEFAULT_POLICY
    output_dir: Path = DEFAULT_OUTPUT
    model_path: Path = DEFAULT_MODEL
    tokenizer_path: Path = DEFAULT_TOKENIZER
    runtime_provenance_path: Path = DEFAULT_RUNTIME_PROVENANCE
    formal_human_decisions_path: Path = DEFAULT_FORMAL_HUMAN_DECISIONS
    input_contract_path: Path | None = None
    device: str = "cuda"
    batch_size: int = 8
    fixture_mode: bool = False


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise GateError("cannot average an empty score group")
    return math.fsum(values) / len(values)


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        raise GateError("cannot normalize an empty score group")
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = math.fsum(exps)
    if not math.isfinite(total) or total <= 0:
        raise GateError("non-finite class softmax denominator")
    return [value / total for value in exps]


def _round(value: float) -> float:
    if not math.isfinite(value):
        raise GateError("non-finite score cannot be serialized")
    return round(float(value), 10)


def _flatten_prompts(policy: Mapping[str, Any]) -> list[dict[str, str]]:
    prompts = policy["prompts"]
    flattened: list[dict[str, str]] = []
    flattened.extend(dict(item) for item in prompts["structural_anchors"])
    for class_id in TARGET_CLASSES:
        flattened.extend(dict(item) for item in prompts["class_prompts"][class_id])
    flattened.extend(dict(item) for item in prompts["hard_reject_prompts"])
    return flattened


def _prompt_ids(policy: Mapping[str, Any]) -> dict[str, Any]:
    prompts = policy["prompts"]
    return {
        "anchors": [item["id"] for item in prompts["structural_anchors"]],
        "classes": {
            class_id: [item["id"] for item in prompts["class_prompts"][class_id]]
            for class_id in TARGET_CLASSES
        },
        "rejects": [item["id"] for item in prompts["hard_reject_prompts"]],
    }


def validate_policy(policy: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "result_schema_version",
        "stats_schema_version",
        "receipt_schema_version",
        "production_inputs",
        "model",
        "prompts",
        "thresholds",
        "output_contract",
        "authority",
        "explicit_non_claims",
    }
    if not isinstance(policy, dict) or set(policy) != required:
        raise GateError("policy top-level schema mismatch")
    if policy["schema_version"] != "rootscope.ai_whole_plant_gpu_gate_policy.v1":
        raise GateError("unsupported policy schema")
    if policy["authority"] != AUTHORITY:
        raise GateError("policy attempts to grant forbidden authority")
    if policy["model"].get("require_cuda") is not True:
        raise GateError("production policy must require CUDA")
    if policy["model"].get("network_access_at_inference") is not False:
        raise GateError("production policy must remain offline")
    expected_classes = policy["production_inputs"].get("expected_class_counts")
    if not isinstance(expected_classes, dict) or tuple(expected_classes) != TARGET_CLASSES:
        raise GateError("policy target classes are not exact or ordered")
    prompts = _flatten_prompts(policy)
    if not prompts:
        raise GateError("prompt set is empty")
    ids: list[str] = []
    for item in prompts:
        if set(item) != {"id", "text"} or not all(
            isinstance(item[key], str) and item[key].strip() for key in ("id", "text")
        ):
            raise GateError("every prompt must contain only non-empty id/text strings")
        ids.append(item["id"])
    if len(ids) != len(set(ids)):
        raise GateError("prompt IDs are not unique")
    prompt_ids = _prompt_ids(policy)
    required_anchors = {
        "anchor.whole_inside_frame",
        "anchor.base_visible",
        "anchor.crown_visible",
        "anchor.single_dominant",
        "anchor.clear_natural_photo",
    }
    if set(prompt_ids["anchors"]) != required_anchors:
        raise GateError("structural anchors changed")
    required_rejects = {
        "reject.closeup.flower_or_seedhead",
        "reject.closeup.branch_or_leaves",
        "reject.closeup.bark_or_trunk",
        "reject.cropped.missing_base",
        "reject.cropped.missing_crown",
        "reject.human.hand_or_person",
        "reject.document.specimen_or_text",
        "reject.scene.plant_community",
        "reject.scene.multiple_plants",
        "reject.scene.wide_landscape",
        "reject.tree.mature_large",
        "reject.tree.dead_or_silhouette",
        "reject.scene.manmade_or_vehicle",
    }
    if set(prompt_ids["rejects"]) != required_rejects:
        raise GateError("hard reject list is incomplete")
    for section in policy["thresholds"].values():
        if not isinstance(section, dict):
            raise GateError("threshold section must be an object")
        for name, value in section.items():
            if name == "require_acquisition_hint_agreement":
                if value is not True:
                    raise GateError("strict positives must require acquisition-hint agreement")
            elif type(value) not in {int, float} or not math.isfinite(float(value)):
                raise GateError("threshold values must be finite numbers")
    output = policy["output_contract"]
    names = [
        output[key]
        for key in (
            "results_filename",
            "stats_filename",
            "runtime_filename",
            "contact_index_filename",
            "receipt_filename",
        )
    ]
    if any(not isinstance(name, str) or Path(name).name != name for name in names):
        raise GateError("output filenames must be safe basenames")
    if len(names) != len(set(names)):
        raise GateError("output filenames must be unique")


def decide_machine_outcome(
    scores: Mapping[str, float], acquisition_hint: str, policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the conservative gate; every hard reject remains individual."""

    validate_policy(policy)
    ids = _prompt_ids(policy)
    expected_ids = set(ids["anchors"] + ids["rejects"])
    for class_ids in ids["classes"].values():
        expected_ids.update(class_ids)
    if set(scores) != expected_ids:
        raise GateError("score vector does not match the full prompt set")
    if acquisition_hint not in TARGET_CLASSES:
        raise GateError("invalid acquisition hint")
    if any(type(value) not in {int, float} or not math.isfinite(float(value)) for value in scores.values()):
        raise GateError("score vector contains a non-finite value")

    class_raw = {
        class_id: _mean([float(scores[prompt_id]) for prompt_id in ids["classes"][class_id]])
        for class_id in TARGET_CLASSES
    }
    probabilities = _softmax([class_raw[class_id] for class_id in TARGET_CLASSES])
    ranked = sorted(
        range(len(TARGET_CLASSES)), key=lambda index: (-probabilities[index], index)
    )
    top1_index, top2_index = ranked[:2]
    top1, top2 = TARGET_CLASSES[top1_index], TARGET_CLASSES[top2_index]

    reject_scores = {prompt_id: float(scores[prompt_id]) for prompt_id in ids["rejects"]}
    max_reject_id = max(ids["rejects"], key=lambda prompt_id: reject_scores[prompt_id])
    max_reject = reject_scores[max_reject_id]
    whole = float(scores["anchor.whole_inside_frame"])
    base = float(scores["anchor.base_visible"])
    crown = float(scores["anchor.crown_visible"])
    single = float(scores["anchor.single_dominant"])
    natural = float(scores["anchor.clear_natural_photo"])
    scene_reject_ids = [
        "reject.scene.plant_community",
        "reject.scene.multiple_plants",
        "reject.scene.wide_landscape",
    ]
    document_hand_ids = [
        "reject.document.specimen_or_text",
        "reject.human.hand_or_person",
    ]
    scene_max_id = max(scene_reject_ids, key=lambda prompt_id: reject_scores[prompt_id])
    document_hand_max_id = max(
        document_hand_ids, key=lambda prompt_id: reject_scores[prompt_id]
    )
    juvenile = float(scores["class.young_tree.juvenile_scale"])
    mature = reject_scores["reject.tree.mature_large"]

    metrics = {
        "whole_vs_max_individual_reject_margin": whole - max_reject,
        "base_vs_missing_base_margin": base - reject_scores["reject.cropped.missing_base"],
        "crown_vs_missing_crown_margin": crown
        - reject_scores["reject.cropped.missing_crown"],
        "single_vs_scene_reject_margin": single - reject_scores[scene_max_id],
        "natural_photo_vs_document_hand_margin": natural
        - reject_scores[document_hand_max_id],
        "young_tree_juvenile_vs_mature_margin": juvenile - mature,
        "top1_probability": probabilities[top1_index],
        "top1_top2_margin": probabilities[top1_index] - probabilities[top2_index],
    }
    positive = policy["thresholds"]["strict_positive"]
    positive_gates = {
        "WHOLE_CLEARS_MAX_INDIVIDUAL_REJECT": metrics[
            "whole_vs_max_individual_reject_margin"
        ]
        >= float(positive["minimum_whole_vs_max_individual_reject_margin"]),
        "BASE_VISIBLE": metrics["base_vs_missing_base_margin"]
        >= float(positive["minimum_base_vs_missing_base_margin"]),
        "CROWN_VISIBLE": metrics["crown_vs_missing_crown_margin"]
        >= float(positive["minimum_crown_vs_missing_crown_margin"]),
        "SINGLE_DOMINANT_SUBJECT": metrics["single_vs_scene_reject_margin"]
        >= float(positive["minimum_single_vs_scene_reject_margin"]),
        "NATURAL_PHOTO": metrics["natural_photo_vs_document_hand_margin"]
        >= float(positive["minimum_natural_photo_vs_document_hand_margin"]),
        "CLASS_CONFIDENCE": metrics["top1_probability"]
        >= float(positive["minimum_top1_probability"]),
        "CLASS_MARGIN": metrics["top1_top2_margin"]
        >= float(positive["minimum_top1_top2_margin"]),
        "ACQUISITION_HINT_AGREES": top1 == acquisition_hint,
        "YOUNG_TREE_IS_JUVENILE_NOT_MATURE": acquisition_hint != "young_tree"
        or metrics["young_tree_juvenile_vs_mature_margin"]
        >= float(positive["young_tree_minimum_juvenile_vs_mature_margin"]),
    }

    exclusion = policy["thresholds"]["exclude"]
    specific = float(exclusion["minimum_specific_reject_vs_anchor_margin"])
    exclusion_blocks: list[str] = []
    if max_reject - whole >= float(
        exclusion["minimum_max_individual_reject_vs_whole_margin"]
    ):
        exclusion_blocks.append("MAX_INDIVIDUAL_REJECT_DOMINATES_WHOLE_PLANT")
    if reject_scores["reject.cropped.missing_base"] - base >= specific:
        exclusion_blocks.append("MISSING_BASE_REJECT_DOMINATES")
    if reject_scores["reject.cropped.missing_crown"] - crown >= specific:
        exclusion_blocks.append("MISSING_CROWN_REJECT_DOMINATES")
    if reject_scores[scene_max_id] - single >= specific:
        exclusion_blocks.append("MULTIPLE_OR_LANDSCAPE_REJECT_DOMINATES")
    if reject_scores[document_hand_max_id] - natural >= specific:
        exclusion_blocks.append("DOCUMENT_OR_HAND_REJECT_DOMINATES")
    if acquisition_hint == "young_tree" and mature - juvenile >= float(
        exclusion["young_tree_minimum_mature_vs_juvenile_margin"]
    ):
        exclusion_blocks.append("MATURE_TREE_REJECT_DOMINATES_JUVENILE")

    if exclusion_blocks:
        outcome = "EXCLUDE"
        reasons = exclusion_blocks
    elif all(positive_gates.values()):
        outcome = f"STRICT_POSITIVE_CANDIDATE_{top1}"
        reasons = ["ALL_STRICT_STRUCTURE_AND_CLASS_GATES_PASSED"]
    else:
        outcome = "HOLD"
        reasons = [name for name, passed in positive_gates.items() if not passed]

    return {
        "outcome": outcome,
        "outcome_reasons": reasons,
        "positive_gates": positive_gates,
        "top1_class": top1,
        "top2_class": top2,
        "class_raw_scores": {class_id: _round(class_raw[class_id]) for class_id in TARGET_CLASSES},
        "class_probabilities": {
            class_id: _round(probabilities[index])
            for index, class_id in enumerate(TARGET_CLASSES)
        },
        "metrics": {name: _round(value) for name, value in metrics.items()},
        "max_individual_reject_prompt": max_reject_id,
        "max_individual_reject_score": _round(max_reject),
        "scene_max_individual_reject_prompt": scene_max_id,
        "document_hand_max_individual_reject_prompt": document_hand_max_id,
        "individual_reject_ranking": [
            {"prompt_id": prompt_id, "score": _round(reject_scores[prompt_id])}
            for prompt_id in sorted(ids["rejects"], key=lambda item: (-reject_scores[item], item))
        ],
    }


class WholePlantGPUGate:
    def __init__(self, config: GateConfig, scorer: PromptScorer | None = None) -> None:
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.manifest_path = Path(config.manifest_path)
        self.summary_path = Path(config.summary_path)
        self.source_plan_path = Path(config.source_plan_path)
        self.policy_path = Path(config.policy_path)
        self.output_dir = Path(config.output_dir)
        self.model_path = Path(config.model_path)
        self.tokenizer_path = Path(config.tokenizer_path)
        self.runtime_provenance_path = Path(config.runtime_provenance_path)
        self.formal_human_decisions_path = Path(config.formal_human_decisions_path)
        self.input_contract_path = (
            Path(config.input_contract_path) if config.input_contract_path is not None else None
        )
        self.scorer = scorer

        try:
            policy, raw = _read_json(self.policy_path, "whole-plant GPU gate policy")
        except EnsembleError as exc:
            raise GateError(str(exc)) from exc
        validate_policy(policy)
        self.policy = policy
        self.policy_sha256 = _sha256_bytes(raw)
        self.production_inputs = dict(policy["production_inputs"])
        self.input_contract_source_sha256 = self.policy_sha256
        self.input_contract_source = "EMBEDDED_IN_FROZEN_GATE_POLICY"
        if self.input_contract_path is not None:
            try:
                contract, contract_raw = _read_json(
                    self.input_contract_path, "GPU gate dataset input contract"
                )
            except EnsembleError as exc:
                raise GateError(str(exc)) from exc
            if not isinstance(contract, dict) or set(contract) != {
                "schema_version",
                "dataset_name",
                "production_inputs",
            }:
                raise GateError("dataset input contract schema mismatch")
            if contract["schema_version"] != "rootscope.ai_gpu_gate_input_contract.v1":
                raise GateError("unsupported dataset input contract schema")
            if contract["dataset_name"] != self.dataset_root.name:
                raise GateError("dataset input contract names a different dataset")
            production = contract["production_inputs"]
            required_production = {
                "manifest_sha256",
                "summary_sha256",
                "source_plan_sha256",
                "image_binding_sha256",
                "expected_candidate_count",
                "expected_class_counts",
                "expected_summary_status",
            }
            if not isinstance(production, dict) or set(production) != required_production:
                raise GateError("dataset input contract production_inputs mismatch")
            class_counts = production["expected_class_counts"]
            if not isinstance(class_counts, dict) or tuple(class_counts) != TARGET_CLASSES:
                raise GateError("dataset input contract must bind all three class count keys")
            if any(type(class_counts[name]) is not int or class_counts[name] < 0 for name in TARGET_CLASSES):
                raise GateError("dataset input contract class counts are invalid")
            if sum(class_counts.values()) != production["expected_candidate_count"]:
                raise GateError("dataset input contract count total mismatch")
            hash_fields = (
                "manifest_sha256",
                "summary_sha256",
                "source_plan_sha256",
                "image_binding_sha256",
            )
            if any(
                not isinstance(production[name], str)
                or len(production[name]) != 64
                or any(character not in "0123456789abcdef" for character in production[name])
                for name in hash_fields
            ):
                raise GateError("dataset input contract contains an invalid SHA-256")
            if not isinstance(production["expected_summary_status"], str) or not production[
                "expected_summary_status"
            ]:
                raise GateError("dataset input contract expected status is invalid")
            self.production_inputs = dict(production)
            self.input_contract_source_sha256 = _sha256_bytes(contract_raw)
            self.input_contract_source = self.input_contract_path.name
        self.effective_input_contract_sha256 = _sha256_bytes(
            _canonical_bytes(self.production_inputs)
        )
        self.prompt_records = _flatten_prompts(policy)
        self.prompt_set_sha256 = _sha256_bytes(_canonical_bytes(policy["prompts"]))
        self.implementation_sha256 = _sha256_file(SCRIPT_PATH)
        self.scorer_implementation_sha256 = _sha256_file(
            SCRIPT_PATH.with_name("ai_siglip2_ensemble.py")
        )

    def _runtime_binding(self) -> dict[str, Any]:
        try:
            provenance, provenance_raw = _read_json(
                self.runtime_provenance_path, "SigLIP2 runtime provenance"
            )
        except EnsembleError as exc:
            raise GateError(str(exc)) from exc
        expected_hash = self.policy["model"]["runtime_provenance_sha256"]
        if _sha256_bytes(provenance_raw) != expected_hash:
            raise GateError("runtime provenance SHA-256 does not match policy")
        expected = provenance.get("runtime")
        if not isinstance(expected, dict) or not isinstance(expected.get("packages"), dict):
            raise GateError("runtime provenance is malformed")
        current_packages: dict[str, str] = {}
        for package_name, expected_version in expected["packages"].items():
            try:
                current_packages[package_name] = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise GateError(f"required runtime package is missing: {package_name}") from exc
            if current_packages[package_name] != expected_version:
                raise GateError(
                    f"runtime package mismatch for {package_name}: "
                    f"expected={expected_version}, current={current_packages[package_name]}"
                )
        if platform.python_implementation() != expected.get("python_implementation"):
            raise GateError("Python implementation differs from runtime provenance")
        if platform.python_version() != expected.get("python_version"):
            raise GateError("Python version differs from runtime provenance")
        try:
            import torch
        except ImportError as exc:
            raise GateError("torch is required") from exc
        if self.config.device != "cuda":
            raise GateError("production whole-plant gate is CUDA-only")
        if not torch.cuda.is_available():
            raise GateError("CUDA was required but is unavailable")
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        return {
            "schema_version": "rootscope.ai_whole_plant_gpu_gate_runtime.v1",
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": current_packages,
            "torch_cuda_version": torch.version.cuda,
            "torch_cudnn_version": torch.backends.cudnn.version(),
            "device": "cuda",
            "cuda_device_index": device_index,
            "cuda_device_name": properties.name,
            "cuda_compute_capability": [properties.major, properties.minor],
            "cuda_total_memory_bytes": properties.total_memory,
            "offline_inference": True,
            "deterministic_seed": 0,
            "runtime_provenance_sha256": expected_hash,
        }

    def _load_and_bind_inputs(self) -> None:
        roots = {
            "manifest_sha256": _sha256_file(self.manifest_path),
            "summary_sha256": _sha256_file(self.summary_path),
            "source_plan_sha256": _sha256_file(self.source_plan_path),
        }
        production = self.production_inputs
        if roots != {key: production[key] for key in roots}:
            raise GateError("dataset input roots do not match the policy")
        try:
            rows, _ = _read_jsonl(self.manifest_path, "reacquisition manifest")
            summary, _ = _read_json(self.summary_path, "reacquisition summary")
        except EnsembleError as exc:
            raise GateError(str(exc)) from exc
        if len(rows) != production["expected_candidate_count"]:
            raise GateError("candidate count does not match policy")
        if not isinstance(summary, dict) or summary.get("manifest_sha256") != roots["manifest_sha256"]:
            raise GateError("summary does not bind the input manifest")
        expected_status = production.get(
            "expected_summary_status",
            "MACHINE_ACQUIRED_WHOLE_PLANT_CANDIDATES_NOT_TRAIN_READY",
        )
        if summary.get("status") != expected_status:
            raise GateError("unexpected reacquisition summary status")

        images: list[dict[str, Any]] = []
        class_counts: Counter[str] = Counter()
        seen_pageids: set[int] = set()
        seen_paths: set[str] = set()
        seen_hashes: set[str] = set()
        for index, row in enumerate(rows, start=1):
            required = {
                "schema_version",
                "class_id",
                "pageid",
                "source_group",
                "filename",
                "download_sha256",
                "download_width",
                "download_height",
                "download_mime",
                "training_eligible",
                "print_eligible",
                "split",
                "source_page",
                "title",
                "artist",
                "license_canonical_name",
            }
            if not isinstance(row, dict) or not required.issubset(row):
                raise GateError(f"manifest row {index} is missing required fields")
            if row["schema_version"] != "rootscope.wikimedia_candidate.v1":
                raise GateError(f"manifest row {index} schema mismatch")
            class_id = row["class_id"]
            if class_id not in TARGET_CLASSES:
                raise GateError(f"manifest row {index} has an invalid class hint")
            if row["training_eligible"] is not False or row["print_eligible"] is not False:
                raise GateError(f"manifest row {index} improperly grants eligibility")
            if row["split"] != "UNASSIGNED_DO_NOT_TRAIN":
                raise GateError(f"manifest row {index} improperly assigns a split")
            try:
                image_path, safe_path = _safe_relative_path(
                    self.dataset_root, row["filename"], f"manifest row {index} image"
                )
            except EnsembleError as exc:
                raise GateError(str(exc)) from exc
            actual_sha = _sha256_file(image_path)
            if actual_sha != row["download_sha256"]:
                raise GateError(f"image SHA-256 mismatch: {safe_path}")
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                    image_format = image.format
            except (OSError, UnidentifiedImageError) as exc:
                raise GateError(f"cannot decode image: {safe_path}: {exc}") from exc
            if (width, height) != (row["download_width"], row["download_height"]):
                raise GateError(f"image dimensions mismatch: {safe_path}")
            mime = Image.MIME.get(image_format, "")
            if mime != row["download_mime"] or mime not in {"image/jpeg", "image/png"}:
                raise GateError(f"image MIME mismatch: {safe_path}")
            if width < 448 or height < 448 or width * height > 100_000_000:
                raise GateError(f"image dimension policy failure: {safe_path}")
            pageid = row["pageid"]
            if type(pageid) is not int or pageid <= 0 or pageid in seen_pageids:
                raise GateError(f"duplicate or invalid pageid at row {index}")
            if safe_path in seen_paths or actual_sha in seen_hashes:
                raise GateError(f"duplicate image payload/path at row {index}")
            seen_pageids.add(pageid)
            seen_paths.add(safe_path)
            seen_hashes.add(actual_sha)
            class_counts[class_id] += 1
            images.append(
                {
                    "row": row,
                    "path": image_path,
                    "local_path": safe_path,
                    "sha256": actual_sha,
                }
            )
        actual_class_counts = {class_id: class_counts[class_id] for class_id in TARGET_CLASSES}
        if actual_class_counts != production["expected_class_counts"]:
            raise GateError("actual class-hint counts do not match policy")
        image_binding = [
            {
                "class_id": item["row"]["class_id"],
                "pageid": item["row"]["pageid"],
                "filename": item["local_path"],
                "download_sha256": item["sha256"],
            }
            for item in images
        ]
        image_binding_sha = _sha256_bytes(_canonical_bytes(image_binding))
        if image_binding_sha != production["image_binding_sha256"]:
            raise GateError("image payload binding does not match policy")
        self.inputs = roots
        self.images = images
        self.image_binding_sha256 = image_binding_sha

    def _bind_artifacts(self) -> None:
        if not self.model_path.is_file() or self.model_path.suffix.lower() != ".npz":
            raise GateError("model weights must be one local NPZ file")
        weights_raw_sha = _sha256_file(self.model_path)
        if weights_raw_sha != self.policy["model"]["weights_raw_sha256"]:
            raise GateError("model weights raw SHA-256 does not match policy")
        try:
            tokenizer_artifact = _artifact_root(self.tokenizer_path)
        except EnsembleError as exc:
            raise GateError(str(exc)) from exc
        if tokenizer_artifact["sha256"] != self.policy["model"]["tokenizer_artifact_sha256"]:
            raise GateError("tokenizer artifact root does not match policy")
        self.weights_raw_sha256 = weights_raw_sha
        self.weights_artifact = _artifact_root(self.model_path)
        self.tokenizer_artifact = tokenizer_artifact

    def preflight(self) -> dict[str, Any]:
        self._load_and_bind_inputs()
        self._bind_artifacts()
        runtime = self._runtime_binding()
        formal_root = _artifact_root(self.formal_human_decisions_path)
        return {
            "schema_version": "rootscope.ai_whole_plant_gpu_gate_preflight.v1",
            "status": "GPU_GATE_PREFLIGHT_PASS_WRITES_PERFORMED_FALSE",
            "candidate_count": len(self.images),
            "input_roots": self.inputs,
            "image_binding_sha256": self.image_binding_sha256,
            "policy_sha256": self.policy_sha256,
            "effective_input_contract_sha256": self.effective_input_contract_sha256,
            "input_contract_source_sha256": self.input_contract_source_sha256,
            "prompt_set_sha256": self.prompt_set_sha256,
            "prompt_count": len(self.prompt_records),
            "weights_raw_sha256": self.weights_raw_sha256,
            "tokenizer_artifact_sha256": self.tokenizer_artifact["sha256"],
            "runtime_binding_sha256": _sha256_bytes(_canonical_bytes(runtime)),
            "cuda_device_name": runtime["cuda_device_name"],
            "formal_human_decisions_guard_sha256": formal_root["sha256"],
            "output_exists": self.output_dir.exists(),
            "writes_performed": False,
            "authority": AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }

    def _score(self) -> tuple[list[list[float]], dict[str, Any]]:
        runtime = self._runtime_binding()
        if self.scorer is None:
            scorer: PromptScorer = LocalOpenCLIPBigVisionScorer(
                self.model_path,
                self.tokenizer_path,
                self.policy["model"]["openclip_model_name"],
                int(self.policy["model"]["context_length"]),
                self.config.device,
                self.config.batch_size,
            )
        else:
            scorer = self.scorer
        previous = {
            name: os.environ.get(name)
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
        }
        os.environ.update(
            {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        )
        try:
            matrix_raw = scorer.score(
                [item["path"] for item in self.images], self.prompt_records
            )
        except (GateError, EnsembleError):
            raise
        except Exception as exc:
            raise GateError(f"GPU prompt scoring failed: {type(exc).__name__}: {exc}") from exc
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        matrix = [list(row) for row in matrix_raw]
        if len(matrix) != len(self.images):
            raise GateError("scorer returned the wrong image count")
        for index, row in enumerate(matrix, start=1):
            if len(row) != len(self.prompt_records):
                raise GateError(f"scorer row {index} has the wrong prompt count")
            if any(type(value) not in {int, float} or not math.isfinite(float(value)) for value in row):
                raise GateError(f"scorer row {index} contains a non-finite value")
        return matrix, runtime

    def _render_results(
        self, matrix: Sequence[Sequence[float]], runtime_sha256: str
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        bindings = {
            "manifest_sha256": self.inputs["manifest_sha256"],
            "image_binding_sha256": self.image_binding_sha256,
            "model_weights_raw_sha256": self.weights_raw_sha256,
            "tokenizer_artifact_sha256": self.tokenizer_artifact["sha256"],
            "policy_sha256": self.policy_sha256,
            "prompt_set_sha256": self.prompt_set_sha256,
            "runtime_binding_sha256": runtime_sha256,
            "effective_input_contract_sha256": self.effective_input_contract_sha256,
            "input_contract_source_sha256": self.input_contract_source_sha256,
        }
        for image_record, values in zip(self.images, matrix, strict=True):
            raw_scores = {
                prompt["id"]: float(values[index])
                for index, prompt in enumerate(self.prompt_records)
            }
            decision = decide_machine_outcome(
                raw_scores, image_record["row"]["class_id"], self.policy
            )
            row = image_record["row"]
            results.append(
                {
                    "schema_version": self.policy["result_schema_version"],
                    "mode": "FIXTURE" if self.config.fixture_mode else "PRODUCTION_GPU",
                    "pageid": row["pageid"],
                    "source_group": row["source_group"],
                    "source_page": row["source_page"],
                    "title": row["title"],
                    "artist": row["artist"],
                    "license_canonical_name": row["license_canonical_name"],
                    "local_path": image_record["local_path"],
                    "candidate_sha256": image_record["sha256"],
                    "acquisition_hint": row["class_id"],
                    "acquisition_hint_is_ground_truth": False,
                    **decision,
                    "raw_prompt_scores": {
                        prompt_id: _round(raw_scores[prompt_id])
                        for prompt_id in sorted(raw_scores)
                    },
                    "bindings": bindings,
                    "authority": AUTHORITY,
                    "explicit_non_claims": self.policy["explicit_non_claims"],
                }
            )
        return results

    def preview(self) -> dict[str, Any]:
        preflight = self.preflight()
        matrix, runtime = self._score()
        runtime_sha = _sha256_bytes(_canonical_bytes(runtime))
        results = self._render_results(matrix, runtime_sha)
        counts = Counter(row["outcome"] for row in results)
        metric_names = tuple(results[0]["metrics"])
        quantiles: dict[str, dict[str, float]] = {}
        for name in metric_names:
            ordered = sorted(float(row["metrics"][name]) for row in results)
            quantiles[name] = {
                label: _round(ordered[round((len(ordered) - 1) * fraction)])
                for label, fraction in (
                    ("min", 0.0),
                    ("p10", 0.10),
                    ("p25", 0.25),
                    ("p50", 0.50),
                    ("p75", 0.75),
                    ("p90", 0.90),
                    ("max", 1.0),
                )
            }
        reason_counts = Counter(
            reason for row in results for reason in row["outcome_reasons"]
        )
        candidates = sorted(
            results,
            key=lambda row: (
                -row["metrics"]["whole_vs_max_individual_reject_margin"],
                row["pageid"],
            ),
        )[:15]
        return {
            "schema_version": "rootscope.ai_whole_plant_gpu_gate_preview.v1",
            "status": "GPU_GATE_PREVIEW_COMPLETE_WRITES_PERFORMED_FALSE",
            "preflight": preflight,
            "outcome_counts": dict(sorted(counts.items())),
            "outcome_reason_counts": dict(sorted(reason_counts.items())),
            "metric_quantiles": quantiles,
            "top_structural_candidates": [
                {
                    "pageid": row["pageid"],
                    "hint": row["acquisition_hint"],
                    "top1": row["top1_class"],
                    "outcome": row["outcome"],
                    "metrics": row["metrics"],
                    "max_reject": row["max_individual_reject_prompt"],
                }
                for row in candidates
            ],
            "writes_performed": False,
            "authority": AUTHORITY,
        }

    def _make_contact_sheets(
        self, root: Path, results: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        output = root / "contact_sheets"
        output.mkdir()
        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in results:
            groups[(row["acquisition_hint"], row["outcome"])].append(row)
        font = ImageFont.load_default()
        index: list[dict[str, Any]] = []
        columns, rows_per_page = 4, 4
        cell_width, cell_height = 300, 245
        for (class_id, outcome), records in sorted(groups.items()):
            for page_number, start in enumerate(
                range(0, len(records), columns * rows_per_page), start=1
            ):
                page_records = records[start : start + columns * rows_per_page]
                sheet = Image.new(
                    "RGB", (columns * cell_width, rows_per_page * cell_height + 34), "white"
                )
                draw = ImageDraw.Draw(sheet)
                draw.text(
                    (8, 8),
                    f"hint={class_id} | outcome={outcome} | page={page_number}",
                    fill="black",
                    font=font,
                )
                for slot, row in enumerate(page_records):
                    col, line = slot % columns, slot // columns
                    x, y = col * cell_width, 34 + line * cell_height
                    image_path = self.dataset_root.joinpath(*Path(row["local_path"]).parts)
                    with Image.open(image_path) as source:
                        source.load()
                        picture = source.convert("RGB")
                        picture.thumbnail((cell_width - 12, 170), Image.Resampling.LANCZOS)
                    px = x + (cell_width - picture.width) // 2
                    py = y + 4 + (170 - picture.height) // 2
                    sheet.paste(picture, (px, py))
                    draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#777777")
                    metric = row["metrics"]["whole_vs_max_individual_reject_margin"]
                    lines = [
                        f"pageid={row['pageid']} top1={row['top1_class']}",
                        f"whole-maxReject={metric:.3f}",
                        f"max={row['max_individual_reject_prompt'].removeprefix('reject.')}",
                    ]
                    for text_line, text_value in enumerate(lines):
                        draw.text((x + 5, y + 180 + text_line * 17), text_value, fill="black", font=font)
                safe_outcome = outcome.lower()
                filename = f"{class_id}__{safe_outcome}__p{page_number:02d}.jpg"
                path = output / filename
                sheet.save(path, format="JPEG", quality=92, optimize=True)
                index.append(
                    {
                        "class_hint": class_id,
                        "outcome": outcome,
                        "page": page_number,
                        "record_count": len(page_records),
                        "pageids": [row["pageid"] for row in page_records],
                        "path": f"contact_sheets/{filename}",
                        "sha256": _sha256_file(path),
                    }
                )
        return index

    def run(self) -> dict[str, Any]:
        preflight = self.preflight()
        if self.output_dir.exists():
            raise GateError("immutable GPU gate output already exists")
        formal_before = _artifact_root(self.formal_human_decisions_path)
        matrix, runtime = self._score()
        runtime_sha = _sha256_bytes(_canonical_bytes(runtime))
        results = self._render_results(matrix, runtime_sha)

        # Re-bind every immutable input after GPU execution.
        self._load_and_bind_inputs()
        self._bind_artifacts()
        formal_mid = _artifact_root(self.formal_human_decisions_path)
        if formal_mid["sha256"] != formal_before["sha256"]:
            raise GateError("formal human_decisions changed during GPU inference")

        counts = Counter(row["outcome"] for row in results)
        hint_outcome: dict[str, Counter[str]] = {
            class_id: Counter(
                row["outcome"] for row in results if row["acquisition_hint"] == class_id
            )
            for class_id in TARGET_CLASSES
        }
        reject_counts = Counter(row["max_individual_reject_prompt"] for row in results)
        stats = {
            "schema_version": self.policy["stats_schema_version"],
            "status": self.policy["output_contract"]["status"],
            "candidate_count": len(results),
            "outcome_counts": dict(sorted(counts.items())),
            "counts_by_acquisition_hint": {
                class_id: dict(sorted(hint_outcome[class_id].items()))
                for class_id in TARGET_CLASSES
            },
            "strict_positive_counts_by_class": {
                class_id: counts[f"STRICT_POSITIVE_CANDIDATE_{class_id}"]
                for class_id in TARGET_CLASSES
            },
            "max_individual_reject_prompt_counts": dict(sorted(reject_counts.items())),
            "hard_reject_aggregation": "MAX_INDIVIDUAL_PROMPT_NEVER_FAMILY_MEAN",
            "thresholds": self.policy["thresholds"],
            "authority": AUTHORITY,
            "explicit_non_claims": self.policy["explicit_non_claims"],
        }

        expected_parent = self.dataset_root / "review"
        expected_parent.mkdir(exist_ok=True)
        if self.output_dir != expected_parent / self.policy["output_contract"]["directory_name"]:
            raise GateError("production output escaped its dedicated review directory")
        temporary = Path(tempfile.mkdtemp(prefix=".ai_strict_gpu_gate_v1.tmp-", dir=expected_parent))
        try:
            output = self.policy["output_contract"]
            (temporary / output["results_filename"]).write_bytes(_jsonl_bytes(results))
            (temporary / output["stats_filename"]).write_bytes(_json_bytes(stats))
            (temporary / output["runtime_filename"]).write_bytes(_json_bytes(runtime))
            contact_index = self._make_contact_sheets(temporary, results)
            (temporary / output["contact_index_filename"]).write_bytes(
                _json_bytes(
                    {
                        "schema_version": "rootscope.ai_whole_plant_gpu_gate_contact_index.v1",
                        "sheet_count": len(contact_index),
                        "sheets": contact_index,
                        "authority": AUTHORITY,
                    }
                )
            )
            payload_hashes: dict[str, str] = {}
            for candidate in sorted(temporary.rglob("*"), key=lambda path: path.as_posix()):
                if candidate.is_file():
                    payload_hashes[candidate.relative_to(temporary).as_posix()] = _sha256_file(candidate)
            run_binding = {
                "implementation_sha256": self.implementation_sha256,
                "scorer_implementation_sha256": self.scorer_implementation_sha256,
                "policy_sha256": self.policy_sha256,
                "effective_input_contract_sha256": self.effective_input_contract_sha256,
                "input_contract_source_sha256": self.input_contract_source_sha256,
                "prompt_set_sha256": self.prompt_set_sha256,
                "input_roots": self.inputs,
                "image_binding_sha256": self.image_binding_sha256,
                "weights_raw_sha256": self.weights_raw_sha256,
                "tokenizer_artifact_sha256": self.tokenizer_artifact["sha256"],
                "runtime_binding_sha256": runtime_sha,
                "formal_human_decisions_guard_sha256": formal_before["sha256"],
                "payload_hashes": payload_hashes,
            }
            receipt = {
                "schema_version": self.policy["receipt_schema_version"],
                "status": output["status"],
                "run_id": "sha256:" + _sha256_bytes(_canonical_bytes(run_binding)),
                **run_binding,
                "model_id": self.policy["model"]["model_id"],
                "backend": self.policy["model"]["backend"],
                "candidate_count": len(results),
                "outcome_counts": dict(sorted(counts.items())),
                "contact_sheet_count": len(contact_index),
                "formal_human_decisions_root_before_sha256": formal_before["sha256"],
                "formal_human_decisions_root_after_sha256": formal_mid["sha256"],
                "formal_human_review_files_touched": False,
                "frozen_ai_ensemble_v1_touched": False,
                "dataset_manifest_written": False,
                "output_scope": (
                    f"{self.dataset_root.name}/review/"
                    f"{self.policy['output_contract']['directory_name']} only"
                ),
                "authority": AUTHORITY,
                "explicit_non_claims": self.policy["explicit_non_claims"],
            }
            (temporary / output["receipt_filename"]).write_bytes(_json_bytes(receipt))
            formal_after = _artifact_root(self.formal_human_decisions_path)
            if formal_after["sha256"] != formal_before["sha256"]:
                raise GateError("formal human_decisions changed before output commit")
            if self.output_dir.exists():
                raise GateError("immutable output appeared before commit")
            os.replace(temporary, self.output_dir)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return {"preflight": preflight, "receipt": receipt}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent CUDA SigLIP2 whole-plant structure/class gate; machine-only."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--input-contract",
        type=Path,
        help="optional dataset-specific input binding; prompts/model/thresholds remain frozen",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size < 1 or args.batch_size > 64:
        print("GPU gate refused: batch size must be in [1,64]", file=sys.stderr)
        return 2
    try:
        dataset_root = Path(args.dataset_root)
        gate = WholePlantGPUGate(
            GateConfig(
                dataset_root=dataset_root,
                manifest_path=dataset_root / "manifest.jsonl",
                summary_path=dataset_root / "summary.json",
                source_plan_path=dataset_root / "source_plan.json",
                policy_path=Path(args.policy),
                output_dir=dataset_root / "review" / "ai_strict_gpu_gate_v1",
                input_contract_path=args.input_contract,
                batch_size=args.batch_size,
            )
        )
        if args.preflight:
            result = gate.preflight()
        elif args.preview:
            result = gate.preview()
        else:
            result = gate.run()["receipt"]
    except (GateError, EnsembleError) as exc:
        print(f"GPU gate refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
