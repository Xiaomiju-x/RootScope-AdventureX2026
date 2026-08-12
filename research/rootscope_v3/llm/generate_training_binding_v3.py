#!/usr/bin/env python3
"""Bind the final RootMind adapter to its exact data and token-length contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_count(value: Any) -> int:
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    elif hasattr(value, "ids"):
        value = value.ids
    if value and isinstance(value[0], list):
        value = value[0]
    return len(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curriculum-manifest", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("binding output already exists; refusing mutable overwrite")
    receipt = json.loads(
        (args.data / "dataset_receipt.json").read_text(encoding="utf-8")
    )
    rows = []
    dataset_files = {}
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        path = args.data / name
        observed = sha256_file(path)
        if observed != receipt["file_sha256"][name]:
            raise SystemExit(f"dataset receipt mismatch: {name}")
        dataset_files[name] = {"bytes": path.stat().st_size, "sha256": observed}
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    full_lengths = []
    prompt_lengths = []
    for row in rows:
        prompt = [
            {"role": "system", "content": row["instruction"]},
            {
                "role": "user",
                "content": json.dumps(
                    row["input"], ensure_ascii=False, separators=(",", ":")
                ),
            },
        ]
        full = [
            *prompt,
            {
                "role": "assistant",
                "content": json.dumps(
                    row["output"], ensure_ascii=False, separators=(",", ":")
                ),
            },
        ]
        prompt_lengths.append(
            token_count(
                tokenizer.apply_chat_template(
                    prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        )
        full_lengths.append(
            token_count(
                tokenizer.apply_chat_template(
                    full,
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            )
        )
    if max(full_lengths) > args.max_length:
        raise SystemExit("one or more SFT sequences exceed the training max length")
    training_receipt = args.adapter_root / "training_receipt.json"
    adapter = args.adapter_root / "adapter" / "adapter_model.safetensors"
    training_value = json.loads(training_receipt.read_text(encoding="utf-8"))
    if (
        training_value.get("schema")
        != "rootscope.v3.qlora-training-receipt.v1"
        or training_value.get("status") != "PASS_REAL_RTX4050_QLORA_ADAPTER"
    ):
        raise SystemExit("training receipt schema/status is not passing")
    if args.max_length != training_value["method"]["max_length"]:
        raise SystemExit("requested max length differs from training receipt")
    adapter_config = args.adapter_root / "adapter" / "adapter_config.json"
    for relative, path in (
        ("adapter/adapter_model.safetensors", adapter),
        ("adapter/adapter_config.json", adapter_config),
    ):
        declared = training_value["artifacts"].get(relative)
        if (
            not isinstance(declared, dict)
            or declared.get("sha256") != sha256_file(path)
            or declared.get("bytes") != path.stat().st_size
        ):
            raise SystemExit(f"training artifact mismatch: {relative}")
    training_inputs = training_value.get("training_inputs")
    refinement = (
        training_value["method"]["name"]
        == "NF4_QLORA_TRAIN_ONLY_ADVERSARIAL_REFINEMENT"
    )
    if refinement and args.curriculum_manifest is None:
        raise SystemExit("refinement training requires curriculum-manifest")
    if refinement and not isinstance(training_inputs, dict):
        raise SystemExit("refinement training inputs are missing")
    if isinstance(training_inputs, dict):
        if (
            training_inputs.get("validation_sha256")
            != dataset_files["validation.jsonl"]["sha256"]
            or training_inputs.get("validation_rows")
            != receipt["splits"]["validation"]
        ):
            raise SystemExit("training validation input differs from canonical split")
        if not refinement and (
            training_inputs.get("train_sha256")
            != dataset_files["train.jsonl"]["sha256"]
            or training_inputs.get("train_rows") != receipt["splits"]["train"]
        ):
            raise SystemExit("training input differs from canonical train split")
    curriculum_binding = None
    if args.curriculum_manifest is not None:
        manifest_path = args.curriculum_manifest.resolve(strict=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS_TRAIN_ONLY_NO_HELD_OUT_RECORDS":
            raise SystemExit("curriculum manifest is not passing")
        if manifest["curriculum"]["held_out_record_count"] != 0:
            raise SystemExit("curriculum contains held-out records")
        for name in ("train", "validation", "test"):
            if (
                manifest["source"][f"{name}_sha256"]
                != dataset_files[f"{name}.jsonl"]["sha256"]
            ):
                raise SystemExit(f"curriculum source mismatch: {name}")
        if (
            manifest["curriculum"]["sha256"]
            != training_value["training_inputs"]["train_sha256"]
        ):
            raise SystemExit("training receipt did not use the bound curriculum")
        if (
            manifest["curriculum"]["rows"]
            != training_value["training_inputs"]["train_rows"]
        ):
            raise SystemExit("training row count did not use the bound curriculum")
        curriculum_binding = {
            "manifest_sha256": sha256_file(manifest_path),
            "curriculum_sha256": manifest["curriculum"]["sha256"],
            "rows": manifest["curriculum"]["rows"],
            "unique_record_ids": manifest["curriculum"]["unique_record_ids"],
            "held_out_record_count": 0,
        }
    value = {
        "schema": "rootscope.v3.llm-training-binding.v1",
        "status": "PASS_EXACT_DATA_ADAPTER_AND_LENGTH_BINDING",
        "dataset": {
            "receipt_sha256": sha256_file(args.data / "dataset_receipt.json"),
            "files": dataset_files,
            "rows": len(rows),
            "retrieval_bound_mismatch_count": sum(
                row["input"]["retrieved_evidence_ids"]
                != row["output"]["evidence_ids"]
                for row in rows
            ),
            "authority_violation_count": sum(
                row["output"]["authority"] is not False for row in rows
            ),
        },
        "training": {
            "receipt_sha256": sha256_file(training_receipt),
            "adapter_sha256": sha256_file(adapter),
            "max_length": args.max_length,
            "full_sequence_max_tokens": max(full_lengths),
            "prompt_max_tokens": max(prompt_lengths),
            "truncated_sequence_count": 0,
            "curriculum": curriculum_binding,
            "adapter_config_sha256": sha256_file(adapter_config),
        },
        "qualification_boundary": (
            "Deterministic structured-contract supervision with template-group "
            "split isolation; knowledge-source contents overlap across splits, so "
            "this is not an unseen-knowledge generalization claim."
        ),
        "authority": {
            "execution_authority": False,
            "serial_write": False,
            "gpio_write": False,
            "pump_command": False,
            "physical_completion": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
    print(json.dumps({"status": value["status"], **value["training"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
