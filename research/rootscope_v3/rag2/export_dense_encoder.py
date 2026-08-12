#!/usr/bin/env python3
"""Export and quantize the pinned BAAI bge-small-zh-v1.5 encoder.

The output is a retrieval challenger, not an X5 qualification claim.  It uses
only the local pinned Hugging Face snapshot and records exact file hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from transformers import AutoModel, AutoTokenizer


HERE = Path(__file__).resolve().parent
ADVENTUREX = HERE.parents[1]
DEFAULT_SOURCE = (
    ADVENTUREX / "rootscope_v3" / "models" / "rag" / "bge-small-zh-v1.5-hf"
)
DEFAULT_OUT = (
    ADVENTUREX / "rootscope_v3" / "models" / "rag" / "bge-small-zh-v1.5-onnx-uint8"
)
REVISION = "7999e1d3359715c523056ef9478215996d62a620"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Encoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )[0]


def pooled(last_hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    expanded = mask.astype(np.float32)[..., None]
    value = (last_hidden.astype(np.float32) * expanded).sum(axis=1)
    value /= np.maximum(expanded.sum(axis=1), 1e-9)
    value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
    return value


def run(source: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModel.from_pretrained(source, local_files_only=True)
    model.eval()
    wrapper = Encoder(model).eval()

    sample = tokenizer(
        ["RootScope 是固定式根区灌溉舱。", "只凭植物类别不能决定泵量。"],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]
    token_type_ids = sample.get("token_type_ids", torch.zeros_like(input_ids))

    fp32 = output / "bge-small-zh-v1.5.fp32.onnx"
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, token_type_ids),
        str(fp32),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    int8 = output / "bge-small-zh-v1.5.dynamic-uint8.onnx"
    quantize_dynamic(
        str(fp32),
        str(int8),
        weight_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=False,
        extra_options={"MatMulConstBOnly": True},
    )

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "config.json",
    ):
        shutil.copy2(source / name, output / name)

    feed = {
        "input_ids": input_ids.cpu().numpy().astype(np.int64),
        "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
        "token_type_ids": token_type_ids.cpu().numpy().astype(np.int64),
    }
    fp_session = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    q_session = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    fp_embedding = pooled(
        fp_session.run(None, feed)[0], feed["attention_mask"]
    )
    q_embedding = pooled(
        q_session.run(None, feed)[0], feed["attention_mask"]
    )
    cosine = np.sum(fp_embedding * q_embedding, axis=1)

    latencies_ms: list[float] = []
    for _ in range(3):
        q_session.run(None, feed)
    for _ in range(20):
        start = time.perf_counter()
        q_session.run(None, feed)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    files = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "model_manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest: dict[str, object] = {
        "schema": "rootscope.rag2.dense-model-manifest.v1",
        "model_id": "BAAI/bge-small-zh-v1.5",
        "revision": REVISION,
        "locator": (
            "https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/" + REVISION
        ),
        "license": "MIT",
        "export": {
            "framework": "torch.onnx.export",
            "opset": 17,
            "quantization": "ONNX Runtime dynamic per-channel unsigned INT8 weights",
            "pooling": "attention-mask mean pooling followed by L2 normalization",
            "max_length_runtime": 128,
        },
        "pc_validation": {
            "providers": q_session.get_providers(),
            "fp32_int8_cosine_min": float(cosine.min()),
            "fp32_int8_cosine_mean": float(cosine.mean()),
            "batch2_latency_ms_p50": float(np.percentile(latencies_ms, 50)),
            "batch2_latency_ms_p95": float(np.percentile(latencies_ms, 95)),
            "runs": len(latencies_ms),
        },
        "files": files,
        "status": "PC_ONLY_X5_PENDING",
        "authority": {
            "execution_authority": False,
            "physical_authority": False,
        },
    }
    (output / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
