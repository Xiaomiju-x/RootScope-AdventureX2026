#!/usr/bin/env python3
"""Merge the frozen RootScope LoRA adapter into Qwen3-1.7B on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("merge output must not already exist")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        device_map={"": "cpu"},
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(
        model,
        args.adapter,
        is_trainable=False,
    ).merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True)
    merged.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.save_pretrained(args.output)
    artifacts = {
        path.relative_to(args.output).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(args.output.rglob("*"))
        if path.is_file()
    }
    receipt = {
        "schema": "rootscope.v3.llm-merge-receipt.v1",
        "status": "PASS_CPU_FP16_ADAPTER_MERGE",
        "upstream_model": "Qwen/Qwen3-1.7B",
        "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "artifacts": artifacts,
        "authority": {
            "execution_authority": False,
            "serial_write": False,
            "gpio_write": False,
            "pump_command": False,
            "physical_completion": False,
        },
    }
    (args.output / "merge_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "files": len(artifacts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
