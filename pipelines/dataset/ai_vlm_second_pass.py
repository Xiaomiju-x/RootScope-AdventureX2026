#!/usr/bin/env python3
"""Independent, fail-closed VLM structure audit for RootScope candidates.

This tool never writes the acquisition manifest or any human-review record.  Its
outputs are machine-only evidence and never grant training, print, rights, split,
or visual-ground-truth authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


SCHEMA_VERSION = "rootscope.ai_vlm_second_pass.v1"
MODEL_REPO = "HuggingFaceTB/SmolVLM-500M-Instruct"
MODEL_COMMIT = "a7da5b986cb59b408707209984f360a5f4ad7e47"
MODEL_LICENSE = "Apache-2.0"
MODEL_CARD_URL = "https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct"
FLORENCE_MODEL_CARD_URL = "https://huggingface.co/microsoft/Florence-2-base-ft"

BOOL_FIELDS = (
    "is_photograph",
    "exactly_one_dominant_plant",
    "whole_plant_visible",
    "base_visible",
    "crown_visible",
    "closeup_or_part",
    "hand_or_person",
    "document_or_specimen",
    "multiple_or_landscape",
    "mature_tree",
)
MORPHOLOGY_CLASSES = {
    "grass_clump",
    "low_shrub",
    "young_tree",
    "mature_tree",
    "other",
    "uncertain",
}
FALSE_AUTHORITY = {
    "human_review": False,
    "visual_ground_truth": False,
    "rights_approval": False,
    "training_eligibility": False,
    "split_assignment": False,
    "print_eligibility": False,
    "data_locked": False,
    "dataset_manifest_write": False,
}
EXPLICIT_NON_CLAIMS = [
    "HUMAN_REVIEWED",
    "VISUAL_GROUND_TRUTH",
    "RIGHTS_APPROVED",
    "TRAIN_READY",
    "SPLIT_READY",
    "PRINT_ELIGIBLE",
    "DATA_LOCKED",
    "MODEL_QUALIFIED_FOR_GROUND_TRUTH",
]

PROMPT_VERSION = "rootscope.smolvlm_forced_choice_structure_prompt.v2"
BOOL_QUESTIONS: dict[str, str] = {
    "is_photograph": "Is this image a real-world photograph rather than a drawing, diagram, document, or specimen sheet?",
    "exactly_one_dominant_plant": "Does the image have exactly one clearly dominant plant as its subject, rather than several competing plants?",
    "whole_plant_visible": "Is the entire dominant plant visible inside the frame, from its ground-level base to its complete top or crown?",
    "base_visible": "Is the ground-level base or trunk base of the dominant plant clearly visible inside the frame?",
    "crown_visible": "Is the complete top or crown of the dominant plant clearly visible inside the frame?",
    "closeup_or_part": "Is the image mainly a close-up or plant part such as a flower, seedhead, leaf, branch, bark, or trunk, rather than a whole plant?",
    "hand_or_person": "Is any human hand or person visibly present in the image?",
    "document_or_specimen": "Is the subject a document, drawing, text panel, or collected specimen rather than a living plant in a natural photograph?",
    "multiple_or_landscape": "Is the subject a wide landscape, plant community, or scene with several competing plants rather than one dominant plant?",
    "mature_tree": "Is the dominant plant a developed adult tree rather than a small juvenile sapling or seedling?",
}
MORPHOLOGY_OPTIONS: dict[str, str] = {
    "A": "grass_clump",
    "B": "low_shrub",
    "C": "young_tree",
    "D": "mature_tree",
    "E": "other",
    "F": "uncertain",
}
PROMPT_CONTRACT = {
    "method": "NEXT_TOKEN_CLOSED_SET_LOGIT_SCORING",
    "boolean_instruction": "Inspect only the image pixels. {question} Answer 1 for yes or 0 for no. Answer only one digit: 1 or 0.",
    "boolean_questions": BOOL_QUESTIONS,
    "morphology_instruction": "Inspect only the image pixels. Classify the dominant visible plant morphology. A=grass clump or tussock; B=low woody shrub or bush; C=young tree sapling or seedling; D=mature adult tree; E=other/non-plant; F=uncertain. Answer only one letter: A, B, C, D, E, or F.",
    "morphology_options": MORPHOLOGY_OPTIONS,
    "confidence": "normalized probability within the compared closed option set; not calibrated",
}


# The golden set deliberately contains obvious whole subjects and obvious failure
# modes.  Only visually unambiguous fields are scored for each image.
GOLDEN_CASES: tuple[dict[str, Any], ...] = (
    {
        "pageid": 38234303,
        "role": "clear_whole_grass",
        "expected": {
            "is_photograph": True,
            "exactly_one_dominant_plant": True,
            "whole_plant_visible": True,
            "base_visible": True,
            "crown_visible": True,
            "closeup_or_part": False,
            "hand_or_person": False,
            "document_or_specimen": False,
            "multiple_or_landscape": False,
            "mature_tree": False,
            "morphology_class": "grass_clump",
        },
    },
    {
        "pageid": 68787114,
        "role": "clear_whole_shrub",
        "expected": {
            "is_photograph": True,
            "exactly_one_dominant_plant": True,
            "whole_plant_visible": True,
            "base_visible": True,
            "crown_visible": True,
            "closeup_or_part": False,
            "hand_or_person": False,
            "document_or_specimen": False,
            "multiple_or_landscape": False,
            "mature_tree": False,
            "morphology_class": "low_shrub",
        },
    },
    {
        "pageid": 38152262,
        "role": "whole_but_mature_tree",
        "expected": {
            "is_photograph": True,
            "exactly_one_dominant_plant": True,
            "whole_plant_visible": True,
            "base_visible": True,
            "crown_visible": True,
            "closeup_or_part": False,
            "hand_or_person": False,
            "document_or_specimen": False,
            "mature_tree": True,
            "morphology_class": "mature_tree",
        },
    },
    {
        "pageid": 38230023,
        "role": "grass_part_closeup",
        "expected": {
            "is_photograph": True,
            "whole_plant_visible": False,
            "base_visible": False,
            "closeup_or_part": True,
            "hand_or_person": False,
            "document_or_specimen": False,
            "morphology_class": "grass_clump",
        },
    },
    {
        "pageid": 52843619,
        "role": "hand_held_seed_part",
        "expected": {
            "is_photograph": True,
            "whole_plant_visible": False,
            "base_visible": False,
            "closeup_or_part": True,
            "hand_or_person": True,
            "document_or_specimen": False,
            "morphology_class": "grass_clump",
        },
    },
    {
        "pageid": 65890968,
        "role": "machinery_landscape",
        "expected": {
            "is_photograph": True,
            "exactly_one_dominant_plant": False,
            "whole_plant_visible": False,
            "base_visible": False,
            "crown_visible": False,
            "multiple_or_landscape": True,
            "morphology_class": "other",
        },
    },
    {
        "pageid": 112352491,
        "role": "trunk_closeup",
        "expected": {
            "is_photograph": True,
            "whole_plant_visible": False,
            "base_visible": False,
            "crown_visible": False,
            "closeup_or_part": True,
            "hand_or_person": False,
            "document_or_specimen": False,
        },
    },
    {
        "pageid": 93016094,
        "role": "branch_flower_closeup",
        "expected": {
            "is_photograph": True,
            "whole_plant_visible": False,
            "base_visible": False,
            "crown_visible": False,
            "closeup_or_part": True,
            "hand_or_person": False,
            "document_or_specimen": False,
        },
    },
    {
        "pageid": 44212101,
        "role": "garden_multiple_plants",
        "expected": {
            "is_photograph": True,
            "exactly_one_dominant_plant": False,
            "whole_plant_visible": False,
            "multiple_or_landscape": True,
        },
    },
    {
        "pageid": 45161359,
        "role": "tree_landscape",
        "expected": {
            "is_photograph": True,
            "exactly_one_dominant_plant": False,
            "whole_plant_visible": False,
            "multiple_or_landscape": True,
        },
    },
)


@dataclass(frozen=True)
class ParsedAnswer:
    valid: bool
    fields: dict[str, Any]
    error: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def model_inventory(model_dir: Path) -> tuple[list[dict[str, Any]], str]:
    files: list[dict[str, Any]] = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = path.relative_to(model_dir).as_posix()
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return files, sha256_json(files)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def extract_first_json_object(raw: str) -> str | None:
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    return None


def parse_answer(raw: str) -> ParsedAnswer:
    payload = extract_first_json_object(raw)
    if payload is None:
        return ParsedAnswer(False, {}, "NO_COMPLETE_JSON_OBJECT")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        return ParsedAnswer(False, {}, f"INVALID_JSON:{error.msg}")
    if not isinstance(value, dict):
        return ParsedAnswer(False, {}, "JSON_NOT_OBJECT")
    normalized: dict[str, Any] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for field in BOOL_FIELDS:
        if field not in value:
            missing.append(field)
        elif isinstance(value[field], bool):
            normalized[field] = value[field]
        else:
            invalid.append(field)
    morphology = value.get("morphology_class")
    if morphology not in MORPHOLOGY_CLASSES:
        invalid.append("morphology_class")
    else:
        normalized["morphology_class"] = morphology
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        invalid.append("confidence")
    else:
        normalized["confidence"] = min(1.0, max(0.0, float(confidence)))
    evidence = value.get("short_evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        invalid.append("short_evidence")
    else:
        normalized["short_evidence"] = re.sub(r"\s+", " ", evidence.strip())[:500]
    if missing or invalid:
        return ParsedAnswer(
            False,
            normalized,
            "SCHEMA_ERROR:" + canonical_json({"missing": sorted(missing), "invalid": sorted(set(invalid))}),
        )
    return ParsedAnswer(True, normalized, None)


def vlm_outcome(fields: Mapping[str, Any], acquisition_hint: str) -> tuple[str, list[str]]:
    if not fields:
        return "VLM_HOLD", ["UNPARSEABLE_ANSWER"]
    rejection_reasons: list[str] = []
    if not fields.get("is_photograph", False):
        rejection_reasons.append("NOT_PHOTOGRAPH")
    if fields.get("closeup_or_part", True):
        rejection_reasons.append("CLOSEUP_OR_PART")
    if fields.get("hand_or_person", True):
        rejection_reasons.append("HAND_OR_PERSON")
    if fields.get("document_or_specimen", True):
        rejection_reasons.append("DOCUMENT_OR_SPECIMEN")
    if fields.get("multiple_or_landscape", True):
        rejection_reasons.append("MULTIPLE_OR_LANDSCAPE")
    if fields.get("mature_tree", True):
        rejection_reasons.append("MATURE_TREE")
    if not fields.get("whole_plant_visible", False):
        rejection_reasons.append("WHOLE_PLANT_NOT_VISIBLE")
    if not fields.get("base_visible", False):
        rejection_reasons.append("BASE_NOT_VISIBLE")
    if not fields.get("crown_visible", False):
        rejection_reasons.append("CROWN_NOT_VISIBLE")
    if not fields.get("exactly_one_dominant_plant", False):
        rejection_reasons.append("NOT_EXACTLY_ONE_DOMINANT")
    if rejection_reasons:
        hard = {
            "NOT_PHOTOGRAPH",
            "CLOSEUP_OR_PART",
            "HAND_OR_PERSON",
            "DOCUMENT_OR_SPECIMEN",
            "MULTIPLE_OR_LANDSCAPE",
            "MATURE_TREE",
        }
        outcome = "VLM_EXCLUDE" if hard.intersection(rejection_reasons) else "VLM_HOLD"
        return outcome, rejection_reasons
    if float(fields.get("confidence", 0.0)) < 0.55:
        return "VLM_HOLD", ["CLOSED_SET_MIN_CONFIDENCE_BELOW_0_55"]
    if fields.get("morphology_class") != acquisition_hint:
        return "VLM_HOLD", ["MORPHOLOGY_DISAGREES_WITH_ACQUISITION_HINT"]
    return "VLM_STRICT_POSITIVE", ["ALL_CONSERVATIVE_STRUCTURE_GATES_PASS"]


def cross_gate_outcome(gpu_outcome: str, outcome: str) -> str:
    if gpu_outcome == "STRICT_POSITIVE" and outcome == "VLM_STRICT_POSITIVE":
        return "CONSENSUS_STRICT_POSITIVE_MACHINE_ONLY"
    if gpu_outcome == "EXCLUDE" and outcome == "VLM_EXCLUDE":
        return "CONSENSUS_EXCLUDE"
    if gpu_outcome == "HOLD" and outcome == "VLM_EXCLUDE":
        return "VLM_EXCLUDE_GPU_HOLD"
    if gpu_outcome == "EXCLUDE" and outcome == "VLM_HOLD":
        return "VLM_HOLD_GPU_EXCLUDE"
    if gpu_outcome == "HOLD" and outcome == "VLM_STRICT_POSITIVE":
        return "VLM_POSITIVE_GPU_HOLD_NO_CONSENSUS"
    if gpu_outcome == "EXCLUDE" and outcome == "VLM_STRICT_POSITIVE":
        return "GATE_DISAGREEMENT_NO_CONSENSUS"
    return f"GPU_{gpu_outcome}__{outcome}"


def runtime_binding(torch_module: Any) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("torch", "transformers", "Pillow", "tokenizers", "safetensors"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    binding = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": bool(torch_module.cuda.is_available()),
        "device": "cuda",
        "offline_inference": True,
        "deterministic_seed": 0,
    }
    if torch_module.cuda.is_available():
        properties = torch_module.cuda.get_device_properties(0)
        binding.update(
            {
                "cuda_device_index": 0,
                "cuda_device_name": torch_module.cuda.get_device_name(0),
                "cuda_compute_capability": list(torch_module.cuda.get_device_capability(0)),
                "cuda_total_memory_bytes": properties.total_memory,
                "torch_cuda_version": torch_module.version.cuda,
                "torch_cudnn_version": torch_module.backends.cudnn.version(),
            }
        )
    binding["runtime_provenance_sha256"] = sha256_json(binding)
    return binding


class SmolVLMScorer:
    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUIRED_FOR_PRODUCTION_VLM_SECOND_PASS")
        self.torch = torch
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval().to("cuda")
        # The checkpoint's generation_config is authoritative; setting these
        # explicitly avoids a stale tokenizer default in newer Transformers.
        self.model.generation_config.pad_token_id = 2
        self.model.generation_config.eos_token_id = 49279
        self.model.generation_config.bos_token_id = 0

        self.boolean_token_ids = {
            "0": self.processor.tokenizer.encode("0", add_special_tokens=False)[0],
            "1": self.processor.tokenizer.encode("1", add_special_tokens=False)[0],
        }
        self.morphology_token_ids = {
            label: self.processor.tokenizer.encode(label, add_special_tokens=False)[0]
            for label in MORPHOLOGY_OPTIONS
        }

    def _closed_set_score(
        self,
        image: Image.Image,
        question: str,
        option_token_ids: Mapping[str, int],
    ) -> tuple[str, dict[str, float], float]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in inputs.items()}
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model(**inputs, use_cache=False, return_dict=True)
        latency_ms = (time.perf_counter() - started) * 1000.0
        logits = output.logits[0, -1]
        labels = list(option_token_ids)
        selected_logits = self.torch.stack([logits[option_token_ids[label]] for label in labels])
        probabilities_tensor = self.torch.softmax(selected_logits.float(), dim=0).cpu()
        probabilities = {
            label: round(float(probabilities_tensor[index]), 8)
            for index, label in enumerate(labels)
        }
        choice = labels[int(probabilities_tensor.argmax().item())]
        return choice, probabilities, latency_ms

    def infer(self, image_path: Path) -> tuple[str, float, int]:
        image = Image.open(image_path).convert("RGB")
        fields: dict[str, Any] = {}
        raw_choices: dict[str, str] = {}
        option_probabilities: dict[str, dict[str, float]] = {}
        chosen_confidences: list[float] = []
        latency_ms = 0.0
        bool_template = PROMPT_CONTRACT["boolean_instruction"]
        for field in BOOL_FIELDS:
            question = bool_template.format(question=BOOL_QUESTIONS[field])
            choice, probabilities, elapsed = self._closed_set_score(
                image, question, self.boolean_token_ids
            )
            latency_ms += elapsed
            raw_choices[field] = choice
            option_probabilities[field] = probabilities
            fields[field] = choice == "1"
            chosen_confidences.append(probabilities[choice])
        morphology_choice, morphology_probabilities, elapsed = self._closed_set_score(
            image,
            str(PROMPT_CONTRACT["morphology_instruction"]),
            self.morphology_token_ids,
        )
        latency_ms += elapsed
        raw_choices["morphology_class"] = morphology_choice
        option_probabilities["morphology_class"] = morphology_probabilities
        fields["morphology_class"] = MORPHOLOGY_OPTIONS[morphology_choice]
        chosen_confidences.append(morphology_probabilities[morphology_choice])
        fields["confidence"] = round(min(chosen_confidences), 8)
        fields["short_evidence"] = "closed-set raw token choices and option probabilities"
        fields["raw_forced_choice_answers"] = raw_choices
        fields["raw_option_probabilities"] = option_probabilities
        raw = canonical_json(fields)
        return raw, latency_ms, len(raw_choices)


def score_golden(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_pageid = {int(row["pageid"]): row for row in results}
    comparisons: list[dict[str, Any]] = []
    field_total = 0
    field_correct = 0
    critical_total = 0
    critical_correct = 0
    morphology_total = 0
    morphology_correct = 0
    case_passes = 0
    parse_valid = 0
    critical_fields = {
        "whole_plant_visible",
        "base_visible",
        "crown_visible",
        "closeup_or_part",
        "hand_or_person",
        "multiple_or_landscape",
    }
    for case in GOLDEN_CASES:
        row = by_pageid.get(case["pageid"])
        fields = (row or {}).get("vlm_fields", {})
        valid = bool((row or {}).get("parse_valid", False))
        parse_valid += int(valid)
        mismatches: list[dict[str, Any]] = []
        for field, expected in case["expected"].items():
            actual = fields.get(field, "__MISSING__")
            correct = actual == expected
            field_total += 1
            field_correct += int(correct)
            if field in critical_fields:
                critical_total += 1
                critical_correct += int(correct)
            if field == "morphology_class":
                morphology_total += 1
                morphology_correct += int(correct)
            if not correct:
                mismatches.append({"field": field, "expected": expected, "actual": actual})
        case_pass = valid and not mismatches
        case_passes += int(case_pass)
        comparisons.append(
            {
                "pageid": case["pageid"],
                "role": case["role"],
                "parse_valid": valid,
                "case_pass": case_pass,
                "mismatches": mismatches,
            }
        )
    field_accuracy = field_correct / field_total if field_total else 0.0
    critical_accuracy = critical_correct / critical_total if critical_total else 0.0
    morphology_accuracy = morphology_correct / morphology_total if morphology_total else 0.0
    qualified = (
        parse_valid == len(GOLDEN_CASES)
        and field_accuracy >= 0.80
        and critical_accuracy >= 0.80
        and morphology_accuracy >= 0.60
        and case_passes >= 5
    )
    return {
        "schema_version": "rootscope.ai_vlm_second_pass_golden_report.v1",
        "status": "GOLDEN_SANITY_PASS" if qualified else "GOLDEN_SANITY_FAIL_STOP_FULL_RUN",
        "qualified_for_this_machine_audit": qualified,
        "not_ground_truth_model_qualification": True,
        "thresholds": {
            "parse_valid_count": len(GOLDEN_CASES),
            "minimum_field_accuracy": 0.80,
            "minimum_critical_field_accuracy": 0.80,
            "minimum_morphology_accuracy": 0.60,
            "minimum_exact_case_pass_count": 5,
        },
        "metrics": {
            "case_count": len(GOLDEN_CASES),
            "parse_valid_count": parse_valid,
            "field_correct": field_correct,
            "field_total": field_total,
            "field_accuracy": round(field_accuracy, 6),
            "critical_field_correct": critical_correct,
            "critical_field_total": critical_total,
            "critical_field_accuracy": round(critical_accuracy, 6),
            "morphology_correct": morphology_correct,
            "morphology_total": morphology_total,
            "morphology_accuracy": round(morphology_accuracy, 6),
            "exact_case_pass_count": case_passes,
        },
        "comparisons": comparisons,
        "authority": dict(FALSE_AUTHORITY),
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
    }


def make_contact_sheet(
    dataset_root: Path,
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    title: str,
    columns: int = 4,
) -> None:
    cell_width, cell_height = 400, 330
    header_height = 42
    rows_count = max(1, math.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * cell_width, header_height + rows_count * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 12), title, fill="black", font=font)
    for index, row in enumerate(rows):
        left = (index % columns) * cell_width
        top = header_height + (index // columns) * cell_height
        image = Image.open(dataset_root / row["local_path"]).convert("RGB")
        fitted = ImageOps.contain(image, (cell_width - 16, 230))
        image_left = left + (cell_width - fitted.width) // 2
        canvas.paste(fitted, (image_left, top + 5))
        fields = row.get("vlm_fields", {})
        lines = [
            f"pageid={row['pageid']} hint={row['acquisition_hint']}",
            f"gpu={row['gpu_gate_outcome']} vlm={row['vlm_outcome']}",
            f"whole/base/crown={fields.get('whole_plant_visible')}/{fields.get('base_visible')}/{fields.get('crown_visible')}",
            f"part/hand/multi/mature={fields.get('closeup_or_part')}/{fields.get('hand_or_person')}/{fields.get('multiple_or_landscape')}/{fields.get('mature_tree')}",
            f"morph={fields.get('morphology_class')} conf={fields.get('confidence')}",
        ]
        text_top = top + 240
        for line_index, line in enumerate(lines):
            draw.text((left + 8, text_top + line_index * 16), line[:68], fill="black", font=font)
        draw.rectangle((left, top, left + cell_width - 1, top + cell_height - 1), outline="#666666")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90, optimize=True)


def build_result(
    source: Mapping[str, Any],
    raw: str,
    parsed: ParsedAnswer,
    latency_ms: float,
    output_tokens: int,
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    outcome, reasons = vlm_outcome(parsed.fields, str(source["acquisition_hint"]))
    raw_payload: dict[str, Any] = {}
    raw_object = extract_first_json_object(raw)
    if raw_object is not None:
        try:
            loaded = json.loads(raw_object)
            if isinstance(loaded, dict):
                raw_payload = loaded
        except json.JSONDecodeError:
            pass
    return {
        "schema_version": SCHEMA_VERSION,
        "pageid": int(source["pageid"]),
        "candidate_sha256": source["candidate_sha256"],
        "local_path": source["local_path"],
        "source_page": source.get("source_page"),
        "title": source.get("title"),
        "acquisition_hint": source["acquisition_hint"],
        "acquisition_hint_is_ground_truth": False,
        "gpu_gate_outcome": source["outcome"],
        "parse_valid": parsed.valid,
        "parse_error": parsed.error,
        "vlm_fields": parsed.fields,
        "closed_set_min_confidence_not_calibrated": parsed.fields.get("confidence"),
        "raw_forced_choice_answers": raw_payload.get("raw_forced_choice_answers"),
        "raw_option_probabilities_not_calibrated": raw_payload.get("raw_option_probabilities"),
        "raw_answer": raw,
        "raw_answer_sha256": sha256_bytes(raw.encode("utf-8")),
        "latency_ms": round(latency_ms, 3),
        "output_token_count": output_tokens,
        "vlm_outcome": outcome,
        "vlm_outcome_reasons": reasons,
        "cross_gate_outcome": cross_gate_outcome(str(source["outcome"]), outcome),
        "bindings": dict(bindings),
        "authority": dict(FALSE_AUTHORITY),
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
    }


def validate_inputs(dataset_root: Path, gate_path: Path, model_dir: Path) -> list[dict[str, Any]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    manifest = dataset_root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not gate_path.is_file():
        raise FileNotFoundError(gate_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    rows = load_jsonl(gate_path)
    if len(rows) != 90:
        raise ValueError(f"expected exactly 90 gate rows, got {len(rows)}")
    pageids = [int(row["pageid"]) for row in rows]
    if len(set(pageids)) != len(pageids):
        raise ValueError("duplicate pageid in gate results")
    for row in rows:
        image_path = dataset_root / row["local_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if sha256_file(image_path) != row["candidate_sha256"]:
            raise ValueError(f"candidate hash mismatch: {image_path}")
    missing_golden = sorted({case["pageid"] for case in GOLDEN_CASES} - set(pageids))
    if missing_golden:
        raise ValueError(f"missing golden pageids: {missing_golden}")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gate-results", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("golden", "full", "all"), default="all")
    args = parser.parse_args(argv)

    dataset_root = args.dataset_root.resolve()
    gate_path = args.gate_results.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    rows = validate_inputs(dataset_root, gate_path, model_dir)

    model_files, model_artifact_sha = model_inventory(model_dir)
    if not any(item["path"] == "model.safetensors" for item in model_files):
        raise RuntimeError("model.safetensors is absent")
    prompt_sha = sha256_json(PROMPT_CONTRACT)
    golden_spec_sha = sha256_json(GOLDEN_CASES)
    manifest_path = dataset_root / "manifest.jsonl"
    input_bindings = {
        "manifest_sha256": sha256_file(manifest_path),
        "gpu_gate_results_sha256": sha256_file(gate_path),
        "model_artifact_sha256": model_artifact_sha,
        "prompt_sha256": prompt_sha,
        "golden_spec_sha256": golden_spec_sha,
    }

    # Imports and model load happen only after all immutable input checks pass.
    import torch

    runtime = runtime_binding(torch)
    if not runtime["cuda_available"]:
        raise RuntimeError("CUDA_REQUIRED_FOR_PRODUCTION_VLM_SECOND_PASS")
    run_bindings = dict(input_bindings)
    run_bindings["runtime_provenance_sha256"] = runtime["runtime_provenance_sha256"]
    run_id = "sha256:" + sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "model_repo": MODEL_REPO,
            "model_commit": MODEL_COMMIT,
            "bindings": run_bindings,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "runtime_binding.json", runtime)
    write_json(
        output_dir / "model_provenance.json",
        {
            "schema_version": "rootscope.vlm_model_provenance.v1",
            "repository": MODEL_REPO,
            "commit": MODEL_COMMIT,
            "license": MODEL_LICENSE,
            "model_card_url": MODEL_CARD_URL,
            "selected_reason": "Instruction-following image QA is required for a fixed multi-field JSON audit; Florence-2 task prompts provide caption/detection outputs but are not an instruction-following structured QA interface.",
            "florence_considered_url": FLORENCE_MODEL_CARD_URL,
            "model_files": model_files,
            "model_artifact_sha256": model_artifact_sha,
        },
    )
    write_json(
        output_dir / "prompt_contract.json",
        {
            "schema_version": PROMPT_VERSION,
            "prompt_contract": PROMPT_CONTRACT,
            "prompt_sha256": prompt_sha,
            "golden_cases": list(GOLDEN_CASES),
            "golden_spec_sha256": golden_spec_sha,
        },
    )

    scorer = SmolVLMScorer(model_dir)
    source_by_pageid = {int(row["pageid"]): row for row in rows}
    golden_dir = output_dir / "golden_sanity"
    golden_results_path = golden_dir / "results.jsonl"
    golden_report_path = golden_dir / "report.json"

    if args.phase in {"golden", "all"}:
        golden_results: list[dict[str, Any]] = []
        for index, case in enumerate(GOLDEN_CASES, start=1):
            source = source_by_pageid[case["pageid"]]
            raw, latency_ms, output_tokens = scorer.infer(dataset_root / source["local_path"])
            parsed = parse_answer(raw)
            result = build_result(source, raw, parsed, latency_ms, output_tokens, run_bindings)
            result["golden_role"] = case["role"]
            golden_results.append(result)
            print(f"golden {index}/{len(GOLDEN_CASES)} pageid={case['pageid']} parse={parsed.valid}", flush=True)
        write_jsonl(golden_results_path, golden_results)
        golden_report = score_golden(golden_results)
        golden_report["run_id"] = run_id
        golden_report["bindings"] = run_bindings
        write_json(golden_report_path, golden_report)
        make_contact_sheet(dataset_root, golden_results, golden_dir / "contact_sheet.jpg", "RootScope VLM golden sanity")
        if not golden_report["qualified_for_this_machine_audit"]:
            failure_receipt = {
                "schema_version": "rootscope.ai_vlm_second_pass_failure_receipt.v1",
                "generated_at_utc": utc_now(),
                "status": "STOPPED_AFTER_GOLDEN_SANITY_FAILURE",
                "reason": "VLM made too many obvious structural errors; no 90-image pseudo-precise label set was generated.",
                "run_id": run_id,
                "bindings": run_bindings,
                "outputs": {
                    "golden_results_sha256": sha256_file(golden_results_path),
                    "golden_report_sha256": sha256_file(golden_report_path),
                    "golden_contact_sheet_sha256": sha256_file(golden_dir / "contact_sheet.jpg"),
                    "model_provenance_sha256": sha256_file(output_dir / "model_provenance.json"),
                    "prompt_contract_sha256": sha256_file(output_dir / "prompt_contract.json"),
                    "runtime_binding_sha256": sha256_file(output_dir / "runtime_binding.json"),
                },
                "manifest_unchanged": sha256_file(manifest_path) == input_bindings["manifest_sha256"],
                "gpu_gate_results_unchanged": sha256_file(gate_path) == input_bindings["gpu_gate_results_sha256"],
                "full_90_image_run_executed": False,
                "machine_outcomes_generated": False,
                "authority": dict(FALSE_AUTHORITY),
                "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
            }
            failure_receipt_path = output_dir / "failure_receipt.json"
            write_json(failure_receipt_path, failure_receipt)
            write_json(
                output_dir / "STOPPED.json",
                {
                    "status": "STOPPED_AFTER_GOLDEN_SANITY_FAILURE",
                    "reason": "VLM made too many obvious structural errors; no 90-image pseudo-precise label set was generated.",
                    "run_id": run_id,
                    "failure_receipt_sha256": sha256_file(failure_receipt_path),
                    "authority": dict(FALSE_AUTHORITY),
                    "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
                },
            )
            print(canonical_json(golden_report["metrics"]), flush=True)
            return 3
        if args.phase == "golden":
            print(canonical_json(golden_report["metrics"]), flush=True)
            return 0
    else:
        if not golden_report_path.is_file() or not golden_results_path.is_file():
            raise RuntimeError("full phase requires existing golden results and report")
        golden_report = json.loads(golden_report_path.read_text(encoding="utf-8"))
        if not golden_report.get("qualified_for_this_machine_audit"):
            raise RuntimeError("golden sanity did not pass; refusing full run")
        if golden_report.get("run_id") != run_id or golden_report.get("bindings") != run_bindings:
            raise RuntimeError("golden evidence bindings do not match this run")
        golden_results = load_jsonl(golden_results_path)

    golden_by_pageid = {int(row["pageid"]): row for row in golden_results}
    all_results: list[dict[str, Any]] = []
    for index, source in enumerate(rows, start=1):
        pageid = int(source["pageid"])
        if pageid in golden_by_pageid:
            result = golden_by_pageid[pageid]
        else:
            raw, latency_ms, output_tokens = scorer.infer(dataset_root / source["local_path"])
            parsed = parse_answer(raw)
            result = build_result(source, raw, parsed, latency_ms, output_tokens, run_bindings)
        all_results.append(result)
        print(f"full {index}/{len(rows)} pageid={pageid} outcome={result['vlm_outcome']} parse={result['parse_valid']}", flush=True)

    results_path = output_dir / "machine_outcomes.jsonl"
    write_jsonl(results_path, all_results)
    counts = Counter(str(row["vlm_outcome"]) for row in all_results)
    cross_counts = Counter(str(row["cross_gate_outcome"]) for row in all_results)
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in all_results:
        class_counts[str(row["acquisition_hint"])][str(row["vlm_outcome"])] += 1
    parse_valid_count = sum(bool(row["parse_valid"]) for row in all_results)
    latencies = sorted(float(row["latency_ms"]) for row in all_results)
    stats = {
        "schema_version": "rootscope.ai_vlm_second_pass_stats.v1",
        "status": "MACHINE_VLM_SECOND_PASS_COMPLETE_NOT_HUMAN_REVIEWED_NOT_TRAIN_READY",
        "candidate_count": len(all_results),
        "parse_valid_count": parse_valid_count,
        "parse_failure_count": len(all_results) - parse_valid_count,
        "vlm_outcome_counts": dict(sorted(counts.items())),
        "cross_gate_counts": dict(sorted(cross_counts.items())),
        "counts_by_acquisition_hint": {
            hint: dict(sorted(values.items())) for hint, values in sorted(class_counts.items())
        },
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3),
            "p50": round(latencies[len(latencies) // 2], 3),
            "p95": round(latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)], 3),
            "max": round(max(latencies), 3),
        },
        "golden_sanity": golden_report,
        "authority": dict(FALSE_AUTHORITY),
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
    }
    write_json(output_dir / "stats.json", stats)

    contact_index: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_results:
        grouped[str(row["vlm_outcome"])].append(row)
    for outcome, outcome_rows in sorted(grouped.items()):
        for page_index in range(0, len(outcome_rows), 16):
            page_rows = outcome_rows[page_index : page_index + 16]
            page_number = page_index // 16 + 1
            relative = Path("contact_sheets") / f"{outcome.lower()}__p{page_number:02d}.jpg"
            target = output_dir / relative
            make_contact_sheet(dataset_root, page_rows, target, f"RootScope VLM {outcome} page {page_number}")
            contact_index.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(target),
                    "vlm_outcome": outcome,
                    "page": page_number,
                    "record_count": len(page_rows),
                    "pageids": [int(row["pageid"]) for row in page_rows],
                }
            )
    write_json(
        output_dir / "contact_sheet_index.json",
        {
            "schema_version": "rootscope.ai_vlm_second_pass_contact_index.v1",
            "sheet_count": len(contact_index),
            "sheets": contact_index,
            "authority": dict(FALSE_AUTHORITY),
        },
    )
    receipt = {
        "schema_version": "rootscope.ai_vlm_second_pass_receipt.v1",
        "generated_at_utc": utc_now(),
        "run_id": run_id,
        "model": {
            "repository": MODEL_REPO,
            "commit": MODEL_COMMIT,
            "license": MODEL_LICENSE,
            "model_card_url": MODEL_CARD_URL,
        },
        "bindings": run_bindings,
        "outputs": {
            "machine_outcomes_sha256": sha256_file(results_path),
            "stats_sha256": sha256_file(output_dir / "stats.json"),
            "golden_report_sha256": sha256_file(golden_report_path),
            "contact_sheet_index_sha256": sha256_file(output_dir / "contact_sheet_index.json"),
        },
        "manifest_unchanged": sha256_file(manifest_path) == input_bindings["manifest_sha256"],
        "gpu_gate_results_unchanged": sha256_file(gate_path) == input_bindings["gpu_gate_results_sha256"],
        "authority": dict(FALSE_AUTHORITY),
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
    }
    write_json(output_dir / "receipt.json", receipt)
    print(canonical_json(stats), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
