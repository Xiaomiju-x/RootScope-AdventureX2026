#!/usr/bin/env python3
"""Bind the final RootMind adapter through merge and GGUF quantization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def metadata(path: Path) -> dict[str, Any]:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))

    def raw(name: str) -> list[int]:
        field = reader.fields[name]
        return field.parts[field.data[0]].tolist()

    return {
        "architecture": bytes(raw("general.architecture")).decode("ascii"),
        "file_type": int(raw("general.file_type")[0]),
        "tensor_count": len(reader.tensors),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--training-seal", type=Path, required=True)
    parser.add_argument("--evaluation-seal", type=Path, required=True)
    parser.add_argument("--merge-root", type=Path, required=True)
    parser.add_argument("--f16", type=Path, required=True)
    parser.add_argument("--q4-k-m", type=Path, required=True)
    parser.add_argument("--quantizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("GGUF seal output already exists")
    sys.path.insert(0, str(args.gguf_py.resolve(strict=True)))
    training = json.loads(args.training_seal.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation_seal.read_text(encoding="utf-8"))
    merge_receipt_path = args.merge_root / "merge_receipt.json"
    merge_receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
    if (
        training.get("status") != "PASS_EXTERNAL_CONTENT_SEAL"
        or evaluation.get("status") != "PASS_EVALUATION_CONTENT_SEALED"
        or merge_receipt.get("status") != "PASS_CPU_FP16_ADAPTER_MERGE"
        or merge_receipt["adapter_sha256"]
        != training["training"]["adapter_sha256"]
        or evaluation["model_adapter_sha256"]
        != training["training"]["adapter_sha256"]
    ):
        raise SystemExit("adapter/evaluation/merge chain mismatch")
    merge_artifacts = []
    for relative, declared in sorted(merge_receipt["artifacts"].items()):
        path = args.merge_root / relative
        actual = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if (
            actual["bytes"] != declared["bytes"]
            or actual["sha256"] != declared["sha256"]
        ):
            raise SystemExit(f"merged artifact mismatch: {relative}")
        merge_artifacts.append(actual)
    f16 = metadata(args.f16)
    q4 = metadata(args.q4_k_m)
    if (
        f16["architecture"] != "qwen3"
        or f16["file_type"] != 1
        or f16["tensor_count"] != 310
        or q4["architecture"] != "qwen3"
        or q4["file_type"] != 15
        or q4["tensor_count"] != 310
        or q4["bytes"] < 100_000_000
    ):
        raise SystemExit("GGUF metadata contract mismatch")
    seal = {
        "schema": "rootscope.v3.final-gguf-seal.v1",
        "status": "PASS_ADAPTER_MERGE_Q4_K_M_CONTENT_SEALED",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "adapter_sha256": training["training"]["adapter_sha256"],
        "training_seal_sha256": sha256_file(args.training_seal),
        "evaluation_seal_sha256": sha256_file(args.evaluation_seal),
        "merge_receipt_sha256": sha256_file(merge_receipt_path),
        "merged_artifact_root_sha256": hashlib.sha256(
            canonical(merge_artifacts)
        ).hexdigest(),
        "f16": f16,
        "q4_k_m": q4,
        "quantizer": {
            "sha256": sha256_file(args.quantizer),
            "tool": "llama-quantize b9637",
        },
        "claim_boundary": (
            "PC merge, GGUF conversion, quantization and metadata validation; "
            "X5 load, latency, memory and soak remain pending board power."
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
    with args.output.open("xb") as handle:
        handle.write(canonical(seal))
    print(
        json.dumps(
            {
                "status": seal["status"],
                "q4_k_m": seal["q4_k_m"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
