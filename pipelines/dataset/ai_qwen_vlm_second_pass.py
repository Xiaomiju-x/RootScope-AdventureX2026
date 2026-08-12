#!/usr/bin/env python3
"""Qwen2.5-VL-3B fail-closed second-pass audit for RootScope imagery.

The 10-image golden gate is mandatory.  Full E1/E2 inference is refused unless
that frozen gate passes.  No output produced by this tool has human-review,
rights, training, split, print, manifest-write, or ground-truth authority.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

import ai_vlm_second_pass as common


SCHEMA_VERSION = "rootscope.ai_qwen_vlm_second_pass.v2"
MODEL_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_COMMIT = "66285546d2b821cf421d4f5eb2576359d3770cd3"
MODEL_LICENSE = "Qwen Research License Agreement; non-commercial research/evaluation only"
MODEL_CARD_URL = "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct"
EXPECTED_WEIGHT_SHA256 = {
    "model-00001-of-00002.safetensors": "41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8",
    "model-00002-of-00002.safetensors": "365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4",
}
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28

SYSTEM_PROMPT = """You are a conservative botanical image-dataset auditor. Judge only visible pixels in the supplied image. Do not infer from filenames, metadata, species names, or the user's desired class. When evidence is ambiguous, reject positive whole-plant conditions rather than guessing. Return exactly one JSON object and no markdown."""

USER_PROMPT = """Audit the image for a strict whole-plant dataset.

Definitions:
- is_photograph: a real-world photograph, not a drawing, diagram, text panel, or specimen sheet.
- exactly_one_dominant_plant: one plant clearly dominates; several competing plants or a community is false.
- whole_plant_visible: the same dominant plant is visibly contained from ground/trunk base through its complete top/crown.
- base_visible: the plant-to-ground or trunk-to-ground base is visible, not inferred behind foliage or outside the frame.
- crown_visible: the complete plant top/crown is inside the frame, not cropped.
- closeup_or_part: the subject is a flower, seedhead, leaf, branch, bark, trunk, or other plant part rather than the entire plant.
- hand_or_person: any visible human hand or person.
- document_or_specimen: document, illustration, sign, text panel, or collected/pressed specimen.
- multiple_or_landscape: wide landscape, plant community, row, grove, or several competing plants.
- mature_tree: a developed adult tree, not a small juvenile sapling/seedling.
- morphology_class: exactly one of grass_clump, low_shrub, young_tree, mature_tree, other, uncertain.

Consistency rules:
- closeup_or_part=true implies whole_plant_visible=false.
- whole_plant_visible=true requires base_visible=true and crown_visible=true.
- multiple_or_landscape=true normally implies exactly_one_dominant_plant=false.
- mature_tree=true requires morphology_class=mature_tree.

Return exactly this JSON schema using real JSON booleans:
{"is_photograph":true,"exactly_one_dominant_plant":false,"whole_plant_visible":false,"base_visible":false,"crown_visible":false,"closeup_or_part":true,"hand_or_person":false,"document_or_specimen":false,"multiple_or_landscape":false,"mature_tree":false,"morphology_class":"other","confidence":0.80,"short_evidence":"brief concrete pixel evidence for base, top, crop and scene"}

confidence is a conservative 0-to-1 self-assessment and is not calibrated."""

PROMPT_CONTRACT = {
    "schema_version": "rootscope.qwen25_vl_structure_prompt.v2",
    "system": SYSTEM_PROMPT,
    "user": USER_PROMPT,
    "min_pixels": MIN_PIXELS,
    "max_pixels": MAX_PIXELS,
    "generation": {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 320,
        "use_cache": True,
    },
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    common.write_json(path, value)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    common.write_jsonl(path, rows)


def validate_dataset(dataset_root: Path, gate_path: Path, expected_count: int) -> list[dict[str, Any]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.is_file() or not gate_path.is_file():
        raise FileNotFoundError(manifest_path if not manifest_path.is_file() else gate_path)
    rows = common.load_jsonl(gate_path)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} gate rows, got {len(rows)}")
    pageids = [int(row["pageid"]) for row in rows]
    if len(pageids) != len(set(pageids)):
        raise ValueError("duplicate pageid")
    for row in rows:
        path = dataset_root / row["local_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if common.sha256_file(path) != row["candidate_sha256"]:
            raise ValueError(f"candidate hash mismatch: {path}")
    return rows


def validate_model(model_dir: Path) -> tuple[list[dict[str, Any]], str, str]:
    files, artifact_sha = common.model_inventory(model_dir)
    by_name = {item["path"]: item for item in files}
    for name, expected_sha in EXPECTED_WEIGHT_SHA256.items():
        if name not in by_name:
            raise FileNotFoundError(model_dir / name)
        if by_name[name]["sha256"] != expected_sha:
            raise ValueError(f"published weight hash mismatch: {name}")
    license_path = model_dir / "LICENSE"
    license_text = license_path.read_text(encoding="utf-8")
    if "Qwen RESEARCH LICENSE AGREEMENT" not in license_text or "NON-COMMERCIAL PURPOSES ONLY" not in license_text:
        raise ValueError("unexpected or unbound model license")
    return files, artifact_sha, common.sha256_file(license_path)


def runtime_binding(torch_module: Any, scorer: "QwenVLMScorer") -> dict[str, Any]:
    package_versions: dict[str, str] = {}
    for name in (
        "torch",
        "transformers",
        "bitsandbytes",
        "accelerate",
        "psutil",
        "Pillow",
        "safetensors",
        "tokenizers",
    ):
        try:
            package_versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            package_versions[name] = "NOT_INSTALLED"
    properties = torch_module.cuda.get_device_properties(0)
    value = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions,
        "device": "cuda:0",
        "cuda_device_name": torch_module.cuda.get_device_name(0),
        "cuda_compute_capability": list(torch_module.cuda.get_device_capability(0)),
        "cuda_total_memory_bytes": properties.total_memory,
        "torch_cuda_version": torch_module.version.cuda,
        "torch_cudnn_version": torch_module.backends.cudnn.version(),
        "offline_inference": True,
        "deterministic_seed": 0,
        "quantization": PROMPT_CONTRACT["quantization"],
        "model_memory_footprint_bytes": scorer.model_memory_footprint_bytes,
        "post_load_cuda_allocated_bytes": torch_module.cuda.memory_allocated(),
        "post_load_cuda_reserved_bytes": torch_module.cuda.memory_reserved(),
        "post_load_cuda_peak_reserved_bytes": torch_module.cuda.max_memory_reserved(),
        "hf_device_map": scorer.device_map,
    }
    value["runtime_provenance_sha256"] = common.sha256_json(value)
    return value


class QwenVLMScorer:
    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUIRED")
        self.torch = torch
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.processor = AutoProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            local_files_only=True,
            quantization_config=quantization,
            device_map={"": 0},
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        self.device_map = {
            str(key): str(value) for key, value in getattr(self.model, "hf_device_map", {"": 0}).items()
        }
        self.model_memory_footprint_bytes = int(self.model.get_memory_footprint())

    def infer(self, image_path: Path) -> tuple[str, float, int, dict[str, int]]:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=320,
                use_cache=True,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        input_count = int(inputs["input_ids"].shape[-1])
        trimmed = generated[:, input_count:]
        raw = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        memory = {
            "cuda_allocated_bytes": int(self.torch.cuda.memory_allocated()),
            "cuda_reserved_bytes": int(self.torch.cuda.memory_reserved()),
            "cuda_peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved()),
        }
        return raw, latency_ms, int(trimmed.shape[-1]), memory


def build_result(
    source: Mapping[str, Any],
    raw: str,
    latency_ms: float,
    token_count: int,
    memory: Mapping[str, int],
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    parsed = common.parse_answer(raw)
    outcome, reasons = common.vlm_outcome(parsed.fields, str(source["acquisition_hint"]))
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
        "model_self_reported_confidence_not_calibrated": parsed.fields.get("confidence"),
        "raw_answer": raw,
        "raw_answer_sha256": common.sha256_bytes(raw.encode("utf-8")),
        "latency_ms": round(latency_ms, 3),
        "output_token_count": token_count,
        "inference_memory": dict(memory),
        "vlm_outcome": outcome,
        "vlm_outcome_reasons": reasons,
        "cross_gate_outcome": common.cross_gate_outcome(str(source["outcome"]), outcome),
        "bindings": dict(bindings),
        "authority": dict(common.FALSE_AUTHORITY),
        "explicit_non_claims": list(common.EXPLICIT_NON_CLAIMS),
    }


def write_dataset_outputs(
    dataset_root: Path,
    gate_path: Path,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    run_id: str,
    bindings: Mapping[str, str],
    golden_report: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    results_path = output_dir / "machine_outcomes.jsonl"
    write_jsonl(results_path, results)
    counts = Counter(str(row["vlm_outcome"]) for row in results)
    cross_counts = Counter(str(row["cross_gate_outcome"]) for row in results)
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        class_counts[str(row["acquisition_hint"])][str(row["vlm_outcome"])] += 1
    latencies = sorted(float(row["latency_ms"]) for row in results)
    peak_reserved = max(int(row["inference_memory"]["cuda_peak_reserved_bytes"]) for row in results)
    stats = {
        "schema_version": "rootscope.ai_qwen_vlm_second_pass_stats.v2",
        "status": "MACHINE_VLM_SECOND_PASS_COMPLETE_NOT_HUMAN_REVIEWED_NOT_TRAIN_READY",
        "candidate_count": len(results),
        "parse_valid_count": sum(bool(row["parse_valid"]) for row in results),
        "parse_failure_count": sum(not bool(row["parse_valid"]) for row in results),
        "vlm_outcome_counts": dict(sorted(counts.items())),
        "cross_gate_counts": dict(sorted(cross_counts.items())),
        "counts_by_acquisition_hint": {
            key: dict(sorted(value.items())) for key, value in sorted(class_counts.items())
        },
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3),
            "p50": round(latencies[len(latencies) // 2], 3),
            "p95": round(latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)], 3),
            "max": round(max(latencies), 3),
        },
        "max_cuda_peak_reserved_bytes": peak_reserved,
        "golden_gate_passed": True,
        "golden_report_sha256": common.sha256_json(golden_report),
        "authority": dict(common.FALSE_AUTHORITY),
        "explicit_non_claims": list(common.EXPLICIT_NON_CLAIMS),
    }
    write_json(output_dir / "stats.json", stats)
    contact_records: list[dict[str, Any]] = []
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        groups[str(row["vlm_outcome"])].append(row)
    for outcome, group in sorted(groups.items()):
        for offset in range(0, len(group), 16):
            page_rows = group[offset : offset + 16]
            page = offset // 16 + 1
            relative = Path("contact_sheets") / f"{outcome.lower()}__p{page:02d}.jpg"
            target = output_dir / relative
            common.make_contact_sheet(dataset_root, page_rows, target, f"Qwen2.5-VL {outcome} page {page}")
            contact_records.append(
                {
                    "path": relative.as_posix(),
                    "sha256": common.sha256_file(target),
                    "vlm_outcome": outcome,
                    "page": page,
                    "pageids": [int(row["pageid"]) for row in page_rows],
                }
            )
    write_json(
        output_dir / "contact_sheet_index.json",
        {
            "schema_version": "rootscope.ai_qwen_vlm_contact_index.v2",
            "sheet_count": len(contact_records),
            "sheets": contact_records,
            "authority": dict(common.FALSE_AUTHORITY),
        },
    )
    manifest_path = dataset_root / "manifest.jsonl"
    receipt = {
        "schema_version": "rootscope.ai_qwen_vlm_second_pass_receipt.v2",
        "generated_at_utc": utc_now(),
        "run_id": run_id,
        "bindings": dict(bindings),
        "outputs": {
            "machine_outcomes_sha256": common.sha256_file(results_path),
            "stats_sha256": common.sha256_file(output_dir / "stats.json"),
            "contact_sheet_index_sha256": common.sha256_file(output_dir / "contact_sheet_index.json"),
        },
        "manifest_unchanged": common.sha256_file(manifest_path) == bindings["manifest_sha256"],
        "gpu_gate_results_unchanged": common.sha256_file(gate_path) == bindings["gpu_gate_results_sha256"],
        "authority": dict(common.FALSE_AUTHORITY),
        "explicit_non_claims": list(common.EXPLICIT_NON_CLAIMS),
    }
    write_json(output_dir / "receipt.json", receipt)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-root", type=Path, required=True)
    parser.add_argument("--e1-gate", type=Path, required=True)
    parser.add_argument("--e2-root", type=Path, required=True)
    parser.add_argument("--e2-gate", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--e1-output", type=Path, required=True)
    parser.add_argument("--e2-output", type=Path, required=True)
    parser.add_argument("--phase", choices=("golden", "full", "all"), default="all")
    args = parser.parse_args(argv)

    e1_root = args.e1_root.resolve()
    e2_root = args.e2_root.resolve()
    e1_gate = args.e1_gate.resolve()
    e2_gate = args.e2_gate.resolve()
    model_dir = args.model_dir.resolve()
    e1_output = args.e1_output.resolve()
    e2_output = args.e2_output.resolve()
    if e1_output.exists() or e2_output.exists():
        raise FileExistsError("versioned output is immutable and must not already exist")
    allowed_output_names = {"ai_vlm_second_pass_v2", "ai_vlm_second_pass_v3"}
    if e1_output.name != e2_output.name or e1_output.name not in allowed_output_names:
        raise ValueError(
            "matching output directory names must be ai_vlm_second_pass_v2 or "
            "ai_vlm_second_pass_v3"
        )
    e1_rows = validate_dataset(e1_root, e1_gate, 90)
    e2_rows = validate_dataset(e2_root, e2_gate, 50)
    golden_ids = {case["pageid"] for case in common.GOLDEN_CASES}
    if not golden_ids.issubset({int(row["pageid"]) for row in e1_rows}):
        raise ValueError("E1 no longer contains frozen golden cases")
    model_files, model_artifact_sha, license_sha = validate_model(model_dir)

    scorer = QwenVLMScorer(model_dir)
    runtime = runtime_binding(scorer.torch, scorer)
    prompt_sha = common.sha256_json(PROMPT_CONTRACT)
    golden_spec_sha = common.sha256_json(common.GOLDEN_CASES)
    shared_bindings = {
        "model_artifact_sha256": model_artifact_sha,
        "model_license_sha256": license_sha,
        "prompt_sha256": prompt_sha,
        "golden_spec_sha256": golden_spec_sha,
        "runtime_provenance_sha256": runtime["runtime_provenance_sha256"],
    }
    run_id = "sha256:" + common.sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "model_repo": MODEL_REPO,
            "model_commit": MODEL_COMMIT,
            "bindings": shared_bindings,
            "e1_manifest": common.sha256_file(e1_root / "manifest.jsonl"),
            "e2_manifest": common.sha256_file(e2_root / "manifest.jsonl"),
            "output_artifact_name": e1_output.name,
        }
    )

    # Golden evidence lives in a staging directory until qualification.  A
    # failed golden never creates the immutable full-output directory.
    golden_stage = e1_output.parent / f"{e1_output.name}_golden"
    if args.phase in {"golden", "all"}:
        if golden_stage.exists():
            raise FileExistsError(golden_stage)
        golden_stage.mkdir(parents=True)
        write_json(golden_stage / "runtime_binding.json", runtime)
        write_json(
            golden_stage / "model_provenance.json",
            {
                "schema_version": "rootscope.qwen25_vl_model_provenance.v2",
                "repository": MODEL_REPO,
                "commit": MODEL_COMMIT,
                "license": MODEL_LICENSE,
                "license_sha256": license_sha,
                "model_card_url": MODEL_CARD_URL,
                "expected_weight_sha256": EXPECTED_WEIGHT_SHA256,
                "model_files": model_files,
                "model_artifact_sha256": model_artifact_sha,
            },
        )
        write_json(
            golden_stage / "prompt_contract.json",
            {"prompt_contract": PROMPT_CONTRACT, "prompt_sha256": prompt_sha},
        )
        e1_source = {int(row["pageid"]): row for row in e1_rows}
        golden_results: list[dict[str, Any]] = []
        golden_bindings = dict(shared_bindings)
        golden_bindings.update(
            {
                "manifest_sha256": common.sha256_file(e1_root / "manifest.jsonl"),
                "gpu_gate_results_sha256": common.sha256_file(e1_gate),
            }
        )
        for index, case in enumerate(common.GOLDEN_CASES, start=1):
            source = e1_source[case["pageid"]]
            raw, latency_ms, tokens, memory = scorer.infer(e1_root / source["local_path"])
            result = build_result(source, raw, latency_ms, tokens, memory, golden_bindings)
            result["golden_role"] = case["role"]
            golden_results.append(result)
            print(
                f"golden {index}/{len(common.GOLDEN_CASES)} pageid={case['pageid']} parse={result['parse_valid']} latency_ms={latency_ms:.1f}",
                flush=True,
            )
        write_jsonl(golden_stage / "results.jsonl", golden_results)
        golden_report = common.score_golden(golden_results)
        golden_report["run_id"] = run_id
        golden_report["bindings"] = golden_bindings
        golden_report["model"] = {"repository": MODEL_REPO, "commit": MODEL_COMMIT, "quantized_4bit": True}
        write_json(golden_stage / "report.json", golden_report)
        common.make_contact_sheet(e1_root, golden_results, golden_stage / "contact_sheet.jpg", "Qwen2.5-VL-3B golden sanity")
        if not golden_report["qualified_for_this_machine_audit"]:
            failure = {
                "schema_version": "rootscope.ai_qwen_vlm_failure_receipt.v2",
                "status": "STOPPED_AFTER_GOLDEN_SANITY_FAILURE",
                "run_id": run_id,
                "full_e1_executed": False,
                "full_e2_executed": False,
                "bindings": golden_bindings,
                "outputs": {
                    "golden_results_sha256": common.sha256_file(golden_stage / "results.jsonl"),
                    "golden_report_sha256": common.sha256_file(golden_stage / "report.json"),
                    "golden_contact_sheet_sha256": common.sha256_file(golden_stage / "contact_sheet.jpg"),
                },
                "authority": dict(common.FALSE_AUTHORITY),
                "explicit_non_claims": list(common.EXPLICIT_NON_CLAIMS),
            }
            write_json(golden_stage / "failure_receipt.json", failure)
            print(common.canonical_json(golden_report["metrics"]), flush=True)
            return 3
        if args.phase == "golden":
            print(common.canonical_json(golden_report["metrics"]), flush=True)
            return 0
    else:
        report_path = golden_stage / "report.json"
        result_path = golden_stage / "results.jsonl"
        if not report_path.is_file() or not result_path.is_file():
            raise FileNotFoundError("qualified golden evidence is absent")
        golden_report = json.loads(report_path.read_text(encoding="utf-8"))
        if not golden_report.get("qualified_for_this_machine_audit") or golden_report.get("run_id") != run_id:
            raise RuntimeError("golden evidence is not qualified or does not bind this run")
        golden_results = common.load_jsonl(result_path)

    golden_by_id = {int(row["pageid"]): row for row in golden_results}

    def run_dataset(
        dataset_root: Path,
        gate_path: Path,
        rows: Sequence[Mapping[str, Any]],
        reuse: Mapping[int, Mapping[str, Any]],
        label: str,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        bindings = dict(shared_bindings)
        bindings.update(
            {
                "manifest_sha256": common.sha256_file(dataset_root / "manifest.jsonl"),
                "gpu_gate_results_sha256": common.sha256_file(gate_path),
            }
        )
        results: list[dict[str, Any]] = []
        for index, source in enumerate(rows, start=1):
            pageid = int(source["pageid"])
            if pageid in reuse:
                result = dict(reuse[pageid])
                result["bindings"] = bindings
            else:
                raw, latency_ms, tokens, memory = scorer.infer(dataset_root / source["local_path"])
                result = build_result(source, raw, latency_ms, tokens, memory, bindings)
            results.append(result)
            print(
                f"{label} {index}/{len(rows)} pageid={pageid} parse={result['parse_valid']} outcome={result['vlm_outcome']}",
                flush=True,
            )
        return results, bindings

    e1_results, e1_bindings = run_dataset(e1_root, e1_gate, e1_rows, golden_by_id, "E1")
    e2_results, e2_bindings = run_dataset(e2_root, e2_gate, e2_rows, {}, "E2")
    write_dataset_outputs(e1_root, e1_gate, e1_output, e1_rows, e1_results, run_id, e1_bindings, golden_report)
    write_dataset_outputs(e2_root, e2_gate, e2_output, e2_rows, e2_results, run_id, e2_bindings, golden_report)
    print(common.canonical_json({"e1": len(e1_results), "e2": len(e2_results), "run_id": run_id}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
