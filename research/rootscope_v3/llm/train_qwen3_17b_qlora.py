#!/usr/bin/env python3
"""RTX4050 4-bit QLoRA for RootScope structured, zero-authority answers."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,"
        "utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=10, check=False
    )
    return {
        "returncode": completed.returncode,
        "line": completed.stdout.strip(),
    }


def encode(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    def token_ids(value: Any) -> list[int]:
        if isinstance(value, dict) or hasattr(value, "keys"):
            value = value["input_ids"]
        elif hasattr(value, "ids"):
            value = value.ids
        if value and isinstance(value[0], list):
            value = value[0]
        return [int(item) for item in value]

    system = row["instruction"]
    user = json.dumps(row["input"], ensure_ascii=False, separators=(",", ":"))
    answer = json.dumps(row["output"], ensure_ascii=False, separators=(",", ":"))
    prompt_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": answer},
    ]
    prompt_encoded = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_encoded = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    # Transformers 5 may return tokenizers.Encoding while older releases
    # return a plain list.  Normalize both without re-tokenizing the template.
    prompt_ids = token_ids(prompt_encoded)
    full_ids = token_ids(full_encoded)
    removed = max(0, len(full_ids) - max_length)
    full_ids = full_ids[-max_length:]
    prompt_length = max(0, len(prompt_ids) - removed)
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, : min(prompt_length, labels.shape[1])] = -100
    return input_ids, labels


@torch.no_grad()
def evaluate(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    limit: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for row in rows[:limit]:
        input_ids, labels = encode(tokenizer, row, max_length)
        input_ids = input_ids.to("cuda")
        labels = labels.to("cuda")
        output = model(input_ids=input_ids, labels=labels, use_cache=False)
        value = float(output.loss.detach().float().cpu())
        if not math.isfinite(value):
            raise RuntimeError("non-finite evaluation loss")
        losses.append(value)
    model.train()
    return sum(losses) / len(losses)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument(
        "--init-adapter",
        type=Path,
        help="Optional previously trained adapter to refine on train-only data.",
    )
    parser.add_argument("--curriculum-manifest", type=Path)
    parser.add_argument("--dataset-receipt", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; CPU fallback is forbidden for this run")
    device_name = torch.cuda.get_device_name(0)
    if "RTX 4050" not in device_name:
        raise SystemExit(f"expected RTX 4050, got {device_name}")
    if not 1 <= args.max_steps <= 200:
        raise SystemExit("max-steps out of frozen range")
    random.seed(20260724)
    torch.manual_seed(20260724)
    torch.cuda.manual_seed_all(20260724)
    torch.backends.cuda.matmul.allow_tf32 = True
    model_path = args.model.resolve(strict=True)
    train_rows = load_jsonl(args.train)
    validation_rows = load_jsonl(args.validation)
    if args.output.exists():
        raise SystemExit("output directory already exists; refusing mutable reuse")
    if len(train_rows) < args.max_steps * args.grad_accum:
        raise SystemExit("insufficient train rows for requested updates")
    if len(validation_rows) < 8:
        raise SystemExit("validation split too small")
    rng = random.Random(20260724)
    rng.shuffle(train_rows)
    selected_rows = train_rows[: args.max_steps * args.grad_accum]
    selected_ids = [row["record_id"] for row in selected_rows]
    selected_root = hashlib.sha256(
        json.dumps(
            selected_ids,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    refinement_contract = None
    if args.init_adapter is not None:
        if args.curriculum_manifest is None or args.dataset_receipt is None:
            raise SystemExit(
                "init-adapter requires curriculum-manifest and dataset-receipt"
            )
        manifest_path = args.curriculum_manifest.resolve(strict=True)
        receipt_path = args.dataset_receipt.resolve(strict=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS_TRAIN_ONLY_NO_HELD_OUT_RECORDS":
            raise SystemExit("curriculum manifest is not passing")
        if manifest["source"]["dataset_receipt_sha256"] != sha256_file(receipt_path):
            raise SystemExit("curriculum dataset receipt mismatch")
        if manifest["curriculum"]["sha256"] != sha256_file(args.train):
            raise SystemExit("curriculum file hash mismatch")
        if manifest["curriculum"]["rows"] != len(train_rows):
            raise SystemExit("curriculum row count mismatch")
        if (
            manifest["curriculum"]["held_out_record_count"] != 0
            or manifest["curriculum"]["held_out_template_group_count"] != 0
        ):
            raise SystemExit("curriculum contains held-out content")
        if (
            manifest["source"]["validation_sha256"] != sha256_file(args.validation)
            or dataset_receipt["file_sha256"]["validation.jsonl"]
            != sha256_file(args.validation)
        ):
            raise SystemExit("validation binding mismatch")
        canonical_train = manifest_path.parent / "train.jsonl"
        canonical_test = manifest_path.parent / "test.jsonl"
        if (
            sha256_file(canonical_train)
            != dataset_receipt["file_sha256"]["train.jsonl"]
            or sha256_file(canonical_test)
            != dataset_receipt["file_sha256"]["test.jsonl"]
            or manifest["source"]["train_sha256"] != sha256_file(canonical_train)
            or manifest["source"]["test_sha256"] != sha256_file(canonical_test)
        ):
            raise SystemExit("canonical split binding mismatch")
        validation_ids = {row["record_id"] for row in validation_rows}
        validation_groups = {row["template_group"] for row in validation_rows}
        test_rows = load_jsonl(canonical_test)
        held_out_ids = validation_ids | {row["record_id"] for row in test_rows}
        held_out_groups = validation_groups | {
            row["template_group"] for row in test_rows
        }
        if any(row.get("split") != "train" for row in train_rows):
            raise SystemExit("curriculum contains a non-train split row")
        if (
            {row["record_id"] for row in train_rows} & held_out_ids
            or {row["template_group"] for row in train_rows} & held_out_groups
        ):
            raise SystemExit("curriculum overlaps validation/test")
        for row in train_rows:
            if (
                row["output"]["authority"] is not False
                or row["input"]["retrieved_evidence_ids"]
                != row["output"]["evidence_ids"]
            ):
                raise SystemExit("curriculum violates authority/citation binding")
            if row["input"].get("adversarial_request") is not None and (
                "ADVERSARIAL_REQUEST_REJECTED"
                not in row["output"]["reason_codes"]
                or not row["output"]["proposed_explanation"].startswith("拒绝")
            ):
                raise SystemExit("curriculum adversarial rejection is incomplete")
        refinement_contract = {
            "curriculum_manifest_sha256": sha256_file(manifest_path),
            "dataset_receipt_sha256": sha256_file(receipt_path),
            "held_out_record_count": 0,
            "held_out_template_group_count": 0,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    gpu_before = gpu_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    parent_adapter = None
    parent_binding = None
    effective_lora_rank = 8
    effective_lora_alpha = 16
    if args.init_adapter is not None:
        parent_adapter = args.init_adapter.resolve(strict=True)
        if not (parent_adapter / "adapter_model.safetensors").is_file():
            raise SystemExit("init-adapter does not contain adapter_model.safetensors")
        parent_config_path = parent_adapter / "adapter_config.json"
        parent_receipt_path = parent_adapter.parent / "training_receipt.json"
        parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
        parent_receipt = json.loads(parent_receipt_path.read_text(encoding="utf-8"))
        if parent_receipt.get("status") != "PASS_REAL_RTX4050_QLORA_ADAPTER":
            raise SystemExit("parent training receipt is not passing")
        if (
            parent_receipt["base_model"]["config_sha256"]
            != sha256_file(model_path / "config.json")
            or parent_receipt["base_model"]["index_sha256"]
            != sha256_file(model_path / "model.safetensors.index.json")
        ):
            raise SystemExit("parent adapter base-model identity mismatch")
        for relative, path in (
            ("adapter/adapter_model.safetensors", parent_adapter / "adapter_model.safetensors"),
            ("adapter/adapter_config.json", parent_config_path),
        ):
            declared = parent_receipt["artifacts"].get(relative)
            if (
                not isinstance(declared, dict)
                or declared.get("sha256") != sha256_file(path)
                or declared.get("bytes") != path.stat().st_size
            ):
                raise SystemExit(f"parent training artifact mismatch: {relative}")
        effective_lora_rank = int(parent_config["r"])
        effective_lora_alpha = int(parent_config["lora_alpha"])
        parent_binding = {
            "adapter_model_sha256": sha256_file(
                parent_adapter / "adapter_model.safetensors"
            ),
            "adapter_config_sha256": sha256_file(parent_config_path),
            "training_receipt_sha256": sha256_file(parent_receipt_path),
            "base_config_sha256": parent_receipt["base_model"]["config_sha256"],
            "base_index_sha256": parent_receipt["base_model"]["index_sha256"],
            "lora_rank": effective_lora_rank,
            "lora_alpha": effective_lora_alpha,
        }
        model = PeftModel.from_pretrained(
            model,
            parent_adapter,
            is_trainable=True,
            local_files_only=True,
        )
    else:
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    torch.cuda.reset_peak_memory_stats()
    baseline_loss = evaluate(
        model, tokenizer, validation_rows, args.max_length, limit=8
    )
    log_path = args.output / "training_steps.jsonl"
    optimizer.zero_grad(set_to_none=True)
    step_losses: list[float] = []
    started_perf = time.perf_counter()
    sample_index = 0
    with log_path.open("w", encoding="utf-8") as log:
        for update in range(args.max_steps):
            accumulated = 0.0
            for micro in range(args.grad_accum):
                row = train_rows[sample_index]
                sample_index += 1
                input_ids, labels = encode(tokenizer, row, args.max_length)
                input_ids = input_ids.to("cuda", non_blocking=True)
                labels = labels.to("cuda", non_blocking=True)
                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    use_cache=False,
                )
                loss = output.loss
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                (loss / args.grad_accum).backward()
                accumulated += float(loss.detach().float().cpu())
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            mean_loss = accumulated / args.grad_accum
            step_losses.append(mean_loss)
            record = {
                "optimizer_step": update + 1,
                "mean_micro_loss": mean_loss,
                "samples_seen": sample_index,
                "cuda_memory_allocated_mib": (
                    torch.cuda.memory_allocated() / (1024 * 1024)
                ),
                "cuda_memory_reserved_mib": (
                    torch.cuda.memory_reserved() / (1024 * 1024)
                ),
            }
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
    final_loss = evaluate(
        model, tokenizer, validation_rows, args.max_length, limit=8
    )
    adapter_root = args.output / "adapter"
    model.save_pretrained(adapter_root, safe_serialization=True)
    tokenizer.save_pretrained(adapter_root)
    files = {
        path.relative_to(args.output).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in args.output.rglob("*")
        if (
            path.is_file()
            and path.name != "training_receipt.json"
            and path.suffix != ".log"
            and path.name != "pid.txt"
        )
    }
    receipt = {
        "schema": "rootscope.v3.qlora-training-receipt.v1",
        "status": "PASS_REAL_RTX4050_QLORA_ADAPTER",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "base_model": {
            "upstream_id": "Qwen/Qwen3-1.7B",
            "upstream_revision": "main_snapshot_20260724",
            "config_sha256": sha256_file(model_path / "config.json"),
            "index_sha256": sha256_file(model_path / "model.safetensors.index.json"),
        },
        "gpu": {
            "device_name": device_name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "before": gpu_before,
            "after": gpu_snapshot(),
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved() / (1024 * 1024)
            ),
        },
        "method": {
            "name": (
                "NF4_QLORA_TRAIN_ONLY_ADVERSARIAL_REFINEMENT"
                if parent_adapter is not None
                else "NF4_QLORA_SFT"
            ),
            "lora_rank": effective_lora_rank,
            "lora_alpha": effective_lora_alpha,
            "max_length": args.max_length,
            "optimizer_steps": args.max_steps,
            "gradient_accumulation": args.grad_accum,
            "samples_seen": sample_index,
            "learning_rate": args.learning_rate,
            "teacher_logits_used": False,
            "selected_record_sequence_sha256": selected_root,
            "selected_adversarial_rows": sum(
                row["input"].get("adversarial_request") is not None
                for row in selected_rows
            ),
            "selected_regular_rows": sum(
                row["input"].get("adversarial_request") is None
                for row in selected_rows
            ),
            "selected_unique_record_ids": len(set(selected_ids)),
            "selected_duplicate_rows": len(selected_ids) - len(set(selected_ids)),
        },
        "training_inputs": {
            "train_path": args.train.resolve().as_posix(),
            "train_sha256": sha256_file(args.train),
            "train_rows": len(train_rows),
            "validation_path": args.validation.resolve().as_posix(),
            "validation_sha256": sha256_file(args.validation),
            "validation_rows": len(validation_rows),
            "parent_adapter_sha256": (
                sha256_file(parent_adapter / "adapter_model.safetensors")
                if parent_adapter is not None
                else None
            ),
            "parent_binding": parent_binding,
            "refinement_contract": refinement_contract,
        },
        "metrics": {
            "validation_loss_before": baseline_loss,
            "validation_loss_after": final_loss,
            "training_loss_first": step_losses[0],
            "training_loss_last": step_losses[-1],
            "elapsed_seconds": time.perf_counter() - started_perf,
        },
        "artifacts": files,
        "authority": {
            "execution_authority": False,
            "serial_write": False,
            "gpio_write": False,
            "pump_command": False,
            "physical_completion": False,
        },
        "truth_boundary": (
            "Adapter training on deterministic RootScope structured supervision; "
            "not X5-qualified, not cloud-logit distillation, not physical closure."
        ),
    }
    (args.output / "training_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
