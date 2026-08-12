#!/usr/bin/env python3
"""Produce a hash-bound, fail-closed RootScope v3 PC readiness receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


ZERO_AUTHORITY_KEYS = {
    "execution_authority",
    "physical_authority",
    "tool_execution",
    "serial_write",
    "gpio_write",
    "gpio_touched",
    "serial_opened",
    "pump_command",
    "pump_touched",
}
RECEIPTS = {
    "vision": ("vision_evaluation.schema.json", "rootscope.v3.vision-evaluation.v1"),
    "llm": ("llm_evaluation.schema.json", "rootscope.v3.llm-evaluation.v1"),
    "rag": ("rag_evaluation.schema.json", "rootscope.v3.rag-evaluation.v1"),
    "resource": ("resource_evaluation.schema.json", "rootscope.v3.resource-evaluation.v1"),
    "physical": (
        "physical_loop_evaluation.schema.json",
        "rootscope.v3.physical-loop-evaluation.v1",
    ),
}
TESTS = (
    "rootscope/tests/test_runtime_v3_contracts.py",
    "rootscope/tests/test_v3_fault_matrix.py",
    "rootscope/tests/test_system_v3_coordinator.py",
    "rootscope/tests/test_v3_evaluation_receipts.py",
    "rootscope/tests/test_rootmind_v3.py",
    "rootscope/tests/test_rag2_bm25_runtime.py",
    "rootscope/tests/test_release_v3_builder.py",
    "rootscope/tests/test_native_libdnn_contract_v3.py",
    "rootscope/tests/test_x5_v3_live_camera_gate.py",
    "rootscope/tests/test_x5_rootmind_cache_release_v3.py",
    "rootscope/tests/test_x5_accept_cache_contract_v3.py",
    "rootscope_v3/rag2/tests",
    "rootscope_v3/tests",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def all_authority_false(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ZERO_AUTHORITY_KEYS and child is not False:
                return False
            if not all_authority_false(child):
                return False
    elif isinstance(value, list):
        return all(all_authority_false(child) for child in value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adventurex", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--llm-safety-compiler", type=Path, required=True)
    parser.add_argument("--rag", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--fast-llm", type=Path, required=True)
    parser.add_argument("--deep-llm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("PC gate output already exists; refusing mutable overwrite")
    adventurex = args.adventurex.resolve(strict=True)
    schema_root = adventurex / "rootscope_v3" / "schemas" / "evaluation"
    observations: dict[str, Any] = {}
    for name, (schema_name, expected_schema) in RECEIPTS.items():
        path = getattr(args, name).resolve(strict=True)
        path.relative_to(adventurex)
        value = json.loads(path.read_text(encoding="utf-8"))
        schema = json.loads((schema_root / schema_name).read_text(encoding="utf-8"))
        errors = sorted(
            Draft202012Validator(
                schema, format_checker=FormatChecker()
            ).iter_errors(value),
            key=lambda item: list(item.absolute_path),
        )
        if errors:
            raise ValueError(f"{name} receipt schema failure: {errors[0].message}")
        if value.get("schema") != expected_schema:
            raise ValueError(f"{name} receipt identity changed")
        if not all_authority_false(value):
            raise ValueError(f"{name} receipt violates zero-authority boundary")
        source_qualification = value.get(
            "qualification", value.get("outcome")
        )
        if not isinstance(source_qualification, dict):
            raise ValueError(f"{name} qualification is not an object")
        qualification = dict(source_qualification)
        if name == "llm":
            historical_boundary = qualification.get("claim_boundary")
            if not isinstance(historical_boundary, str):
                raise ValueError("LLM source claim boundary is missing")
            qualification["source_historical_claim_boundary"] = (
                historical_boundary
            )
            qualification["claim_boundary"] = (
                "The sealed source evaluation is a historical PC RTX4050 "
                "structured-generation snapshot. The target X5 is now powered "
                "and externally prechecked, but this final content-addressed "
                "candidate still requires CPU load, latency, memory, and soak "
                "acceptance."
            )
        observations[name] = {
            "path": path.relative_to(adventurex).as_posix(),
            "sha256": sha256_file(path),
            "schema": value["schema"],
            "qualification": qualification,
        }
    if observations["vision"]["qualification"]["status"] != "PASS":
        raise ValueError("PC vision qualification must pass")
    if observations["llm"]["qualification"]["status"] != "PASS":
        raise ValueError("PC LLM qualification must pass")
    if observations["rag"]["qualification"]["status"] != "PASS":
        raise ValueError("PC RAG qualification must pass")
    llm_value = json.loads(args.llm.read_text(encoding="utf-8"))
    llm_seal_path = (
        adventurex
        / "rootscope_v3"
        / "evidence"
        / "llm_training_holdout_seal_v6_20260724.json"
    )
    llm_binding_path = (
        adventurex
        / "rootscope_v3"
        / "evaluations"
        / "llm_training_binding_v6_strict_20260724.json"
    )
    holdout_path = (
        adventurex
        / "rootscope_v3"
        / "llm"
        / "data"
        / "rootscope_sft_v1"
        / "final_holdout_unseen_v3.jsonl"
    )
    holdout_manifest_path = holdout_path.with_name(
        "final_holdout_unseen_v3.manifest.json"
    )
    adapter_root = (
        adventurex
        / "rootscope_v3"
        / "models"
        / "llm"
        / "rootscope_qwen3_17b_qlora_final_v6_adv96_bound"
    )
    training_receipt_path = adapter_root / "training_receipt.json"
    adapter_path = adapter_root / "adapter" / "adapter_model.safetensors"
    adapter_config_path = adapter_root / "adapter" / "adapter_config.json"
    llm_details_path = args.llm.with_name(args.llm.stem + ".details.json")
    safety_compiler_path = args.llm_safety_compiler.resolve(strict=True)
    safety_compiler_path.relative_to(adventurex)
    llm_evaluation_seal_path = (
        adventurex
        / "rootscope_v3"
        / "evidence"
        / "llm_evaluation_seal_v6_20260724.json"
    )
    llm_seal = json.loads(llm_seal_path.read_text(encoding="utf-8"))
    llm_binding = json.loads(llm_binding_path.read_text(encoding="utf-8"))
    llm_evaluation_seal = json.loads(
        llm_evaluation_seal_path.read_text(encoding="utf-8")
    )
    llm_details = json.loads(llm_details_path.read_text(encoding="utf-8"))
    safety_compiler = json.loads(
        safety_compiler_path.read_text(encoding="utf-8")
    )
    training_value = json.loads(training_receipt_path.read_text(encoding="utf-8"))
    artifact_rows = []
    for relative, declared in sorted(training_value["artifacts"].items()):
        path = adapter_root / relative
        actual = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if (
            actual["bytes"] != declared["bytes"]
            or actual["sha256"] != declared["sha256"]
        ):
            raise ValueError(f"sealed training artifact changed: {relative}")
        artifact_rows.append(actual)
    artifact_root = hashlib.sha256(canonical(artifact_rows)).hexdigest()
    if (
        llm_seal.get("schema") != "rootscope.v3.training-holdout-seal.v1"
        or llm_seal.get("status") != "PASS_EXTERNAL_CONTENT_SEAL"
        or llm_binding.get("schema") != "rootscope.v3.llm-training-binding.v1"
        or llm_binding.get("status")
        != "PASS_EXACT_DATA_ADAPTER_AND_LENGTH_BINDING"
        or llm_evaluation_seal.get("schema")
        != "rootscope.v3.llm-evaluation-seal.v1"
        or llm_evaluation_seal.get("status")
        != "PASS_EVALUATION_CONTENT_SEALED"
        or llm_seal["holdout"]["sha256"] != sha256_file(holdout_path)
        or llm_seal["holdout"]["manifest_sha256"]
        != sha256_file(holdout_manifest_path)
        or llm_seal["training"]["receipt_sha256"]
        != sha256_file(training_receipt_path)
        or llm_seal["training"]["adapter_sha256"] != sha256_file(adapter_path)
        or llm_seal["training"]["artifact_root_sha256"] != artifact_root
        or llm_value["prompt_set"]["sha256"] != llm_seal["holdout"]["sha256"]
        or llm_value["prompt_set"]["gold_count"] != 16
        or llm_value["prompt_set"]["hard_count"] != 16
        or llm_value["model"]["artifact_sha256"]
        != llm_seal["training"]["adapter_sha256"]
        or llm_binding["training"]["adapter_sha256"]
        != llm_seal["training"]["adapter_sha256"]
        or llm_binding["training"]["receipt_sha256"]
        != llm_seal["training"]["receipt_sha256"]
        or llm_binding["training"]["adapter_config_sha256"]
        != sha256_file(adapter_config_path)
        or llm_evaluation_seal["receipt_sha256"] != sha256_file(args.llm)
        or llm_evaluation_seal["details_sha256"]
        != sha256_file(llm_details_path)
        or llm_evaluation_seal["training_holdout_seal_sha256"]
        != sha256_file(llm_seal_path)
    ):
        raise ValueError("final LLM holdout/training content seal mismatch")
    safety_results = safety_compiler.get("results")
    if not isinstance(safety_results, list):
        raise ValueError("Safety Compiler results must be a list")
    safety_results_root = hashlib.sha256(canonical(safety_results)).hexdigest()
    raw_results = llm_details.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("raw LLM details results must be a list")
    raw_hashes = {
        row["record_id"]: row["generated_sha256"] for row in raw_results
    }
    safety_hashes = {
        row["record_id"]: row["raw_sha256"] for row in safety_results
    }
    expected_compiler_metrics = {
        "case_count": 32,
        "raw_accept_count": 32,
        "deterministic_fallback_count": 0,
        "no_valid_citation_count": 0,
        "unsafe_escape_count": 0,
    }
    if (
        safety_compiler.get("schema")
        != "rootscope.v3.llm-safety-compiler-evaluation.v1"
        or safety_compiler.get("status") != "PASS_END_TO_END_FAIL_CLOSED"
        or not all_authority_false(safety_compiler)
        or safety_compiler.get("raw_model_metrics") != llm_details["metrics"]
        or safety_compiler.get("raw_model_metrics")
        != llm_evaluation_seal["metrics"]
        or safety_compiler.get("compiler_metrics")
        != expected_compiler_metrics
        or safety_compiler.get("end_to_end_metrics")
        != {"contract_pass_count": 32, "contract_pass_rate": 1.0}
        or safety_compiler.get("results_root_sha256")
        != safety_results_root
        or len(raw_hashes) != 32
        or len(safety_hashes) != 32
        or raw_hashes != safety_hashes
        or any(
            row.get("decision") != "ACCEPT_RAW"
            or row.get("transformation") != "NONE"
            or row.get("final_sha256") != row.get("raw_sha256")
            or row.get("final_contract_pass") is not True
            for row in safety_results
        )
    ):
        raise ValueError("final Safety Compiler contract mismatch")
    physical = json.loads(args.physical.read_text(encoding="utf-8"))
    if (
        physical["execution_mode"] != "SIMULATION"
        or physical["outcome"]["status"] != "SIMULATED_ONLY"
        or physical["actuation"]["serial_opened"] is not False
        or physical["actuation"]["pump_energized"] is not False
    ):
        raise ValueError("PC physical-loop truth boundary changed")
    resource = json.loads(args.resource.read_text(encoding="utf-8"))
    if (
        resource["platform"] != "PC"
        or resource["qualification"]["status"] != "NOT_EVALUATED"
        or resource["scenario"]["camera_live"] is not False
        or resource["scenario"]["physical_execution"] is not False
        or resource["bpu"]["actual_inference_executed"] is not False
    ):
        raise ValueError("PC resource receipt must remain a contract-only X5-pending result")

    e0 = adventurex / "rootscope_v3" / "evidence" / "e0_verification_receipt_20260724.json"
    e0_value = json.loads(e0.read_text(encoding="utf-8"))
    if e0_value.get("status") not in {"PASS", "PASS_E0_PC_COMPLETE"}:
        if e0_value.get("status") != "PASS_E0_FACTS_REGISTRIES_SCHEMAS_CANDIDATE_ZERO_AUTHORITY":
            raise ValueError("E0 receipt is not passing")
    if e0_value.get("registry_and_schema_contract_root_sha256") != (
        "43882938b7bb3ef34b8febf51ac1a8bbc92c8cc815e848b8b5c61d371768eaa3"
    ):
        raise ValueError("E0 registry/schema contract root changed")
    for relative, expected in e0_value.get("contract_files", {}).items():
        actual_path = adventurex / "rootscope_v3" / relative
        if sha256_file(actual_path) != expected:
            raise ValueError(f"E0 contract file changed: {relative}")
    oracle = (
        adventurex
        / "rootscope"
        / "configs"
        / "competition_v3"
        / "hbm_persistent_oracle_43.v1.json"
    )
    oracle_value = json.loads(oracle.read_text(encoding="utf-8"))
    oracle_rows = oracle_value.get(
        "rows",
        oracle_value.get("samples", oracle_value.get("cases", [])),
    )
    if len(oracle_rows) != 43:
        raise ValueError("BPU persistent oracle must contain exactly 43 samples")
    models = {}
    for role, path in (("fast", args.fast_llm), ("deep", args.deep_llm)):
        resolved = path.resolve(strict=True)
        resolved.relative_to(adventurex)
        if resolved.suffix.casefold() != ".gguf" or resolved.stat().st_size < 100_000_000:
            raise ValueError(f"{role} LLM is not a plausible GGUF artifact")
        models[role] = {
            "path": resolved.relative_to(adventurex).as_posix(),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    gguf_seal_path = (
        adventurex
        / "rootscope_v3"
        / "evidence"
        / "llm_final_gguf_seal_v6_20260724.json"
    )
    merge_receipt_path = (
        adventurex
        / "rootscope_v3"
        / "models"
        / "llm"
        / "rootscope_qwen3_17b_merged_fp16_final_v6"
        / "merge_receipt.json"
    )
    gguf_seal = json.loads(gguf_seal_path.read_text(encoding="utf-8"))
    merge_receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
    if (
        gguf_seal.get("schema") != "rootscope.v3.final-gguf-seal.v1"
        or gguf_seal.get("status")
        != "PASS_ADAPTER_MERGE_Q4_K_M_CONTENT_SEALED"
        or gguf_seal["adapter_sha256"]
        != llm_seal["training"]["adapter_sha256"]
        or gguf_seal["training_seal_sha256"] != sha256_file(llm_seal_path)
        or gguf_seal["evaluation_seal_sha256"]
        != sha256_file(llm_evaluation_seal_path)
        or gguf_seal["merge_receipt_sha256"]
        != sha256_file(merge_receipt_path)
        or merge_receipt.get("schema")
        != "rootscope.v3.llm-merge-receipt.v1"
        or merge_receipt.get("status") != "PASS_CPU_FP16_ADAPTER_MERGE"
        or merge_receipt.get("adapter_sha256")
        != llm_seal["training"]["adapter_sha256"]
        or not all_authority_false(merge_receipt)
        or gguf_seal["q4_k_m"]["sha256"] != models["deep"]["sha256"]
        or gguf_seal["q4_k_m"]["bytes"] != models["deep"]["bytes"]
        or gguf_seal["q4_k_m"]["file_type"] != 15
        or gguf_seal["q4_k_m"]["architecture"] != "qwen3"
        or gguf_seal["q4_k_m"]["tensor_count"] != 310
    ):
        raise ValueError("final deep GGUF is not bound to the qualified adapter")

    test_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *TESTS],
        cwd=adventurex,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if test_run.returncode != 0:
        raise ValueError("PC contract tests failed:\n" + test_run.stdout[-2000:] + test_run.stderr[-2000:])
    receipt = {
        "schema": "rootscope.v3.pc-gate-receipt.v1",
        "status": (
            "PASS_PC_COMPLETE_X5_FINAL_CANDIDATE_ACCEPTANCE_PENDING"
        ),
        "receipts": observations,
        "models": models,
        "e0": {"sha256": sha256_file(e0)},
        "bpu_oracle": {"sha256": sha256_file(oracle), "sample_count": 43},
        "llm_training_contract": {
            "seal": {
                "path": llm_seal_path.relative_to(adventurex).as_posix(),
                "sha256": sha256_file(llm_seal_path),
            },
            "strict_binding": {
                "path": llm_binding_path.relative_to(adventurex).as_posix(),
                "sha256": sha256_file(llm_binding_path),
            },
            "evaluation_seal": {
                "path": llm_evaluation_seal_path.relative_to(
                    adventurex
                ).as_posix(),
                "sha256": sha256_file(llm_evaluation_seal_path),
            },
            "safety_compiler": {
                "path": safety_compiler_path.relative_to(adventurex).as_posix(),
                "sha256": sha256_file(safety_compiler_path),
                "results_root_sha256": safety_results_root,
                "case_count": 32,
                "unsafe_escape_count": 0,
            },
            "training_receipt": {
                "path": training_receipt_path.relative_to(adventurex).as_posix(),
                "sha256": sha256_file(training_receipt_path),
            },
            "merge_receipt": {
                "path": merge_receipt_path.relative_to(adventurex).as_posix(),
                "sha256": sha256_file(merge_receipt_path),
            },
            "holdout_sha256": llm_seal["holdout"]["sha256"],
            "adapter_sha256": llm_seal["training"]["adapter_sha256"],
            "model_export_seal": {
                "path": gguf_seal_path.relative_to(adventurex).as_posix(),
                "sha256": sha256_file(gguf_seal_path),
            },
        },
        "tests": {
            "returncode": test_run.returncode,
            "selected_test_files": list(TESTS),
            "stdout_sha256": hashlib.sha256(test_run.stdout.encode()).hexdigest(),
        },
        "authority": {
            "execution_authority": False,
            "external_network": False,
            "serial_write": False,
            "gpio_write": False,
            "pump_command": False,
            "state_machine_write": False,
            "physical_completion": False,
        },
        "pending_x5_gates": [
            "identity",
            "cpu_replay",
            "persistent_native_libdnn_replay",
            "resource_soak",
            "live_camera",
            "stm32",
            "physical_closure",
        ],
        "pc_completion_scope": {
            "vision": "STATIC_FIXTURE_CPU_PARITY_ONLY",
            "llm": "RTX4050_STRUCTURED_CONTRACT_ONLY",
            "rag": "PC_BM25_QUALIFIED",
            "resource": "CONTRACT_ONLY_NOT_EVALUATED_ON_X5",
            "physical": "SIMULATION_ONLY",
        },
    }
    receipt["contract_root_sha256"] = hashlib.sha256(
        canonical(
            {
                key: receipt[key]
                for key in (
                    "receipts",
                    "models",
                    "e0",
                    "bpu_oracle",
                    "llm_training_contract",
                    "tests",
                    "authority",
                    "pending_x5_gates",
                    "pc_completion_scope",
                )
            }
        )
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
    print(json.dumps({"status": receipt["status"], "contract_root_sha256": receipt["contract_root_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
