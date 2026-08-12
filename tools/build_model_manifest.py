#!/usr/bin/env python3
"""Rebuild the public model asset manifest from the approved allow-list."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    ("model-assets/vision/rootscope_answer_cards_resnet18_opset11.onnx", "rootsight-fixed-card-resnet18-onnx", "Apache-2.0", "team-trained; torchvision ResNet-18 initialization", "FOUR_FIXED_PRINTED_ANSWER_CARDS_ONLY"),
    ("model-assets/bpu/rootscope_seed17_resnet18_static_b1x3x224x224_opset11.onnx", "rootsight-seed17-bpu-source-onnx", "Apache-2.0", "team-trained seed-17 ResNet-18 source graph for the published Bayes-e binary", "FOUR_FIXED_PRINTED_ANSWER_CARDS_BPU_SOURCE_ONLY"),
    ("model-assets/bpu/rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes.bin", "rootsight-fixed-card-bayes-e-r7", "Apache-2.0", "compiled from the adjacent seed-17 ONNX with the D-Robotics toolchain", "BPU_REPLAY_AUXILIARY_NO_ACTUATOR_AUTHORITY"),
    ("model-assets/rootmind-adapter/adapter_model.safetensors", "rootmind-qwen3-1.7b-qlora-adapter-v6", "Apache-2.0", "Qwen/Qwen3-1.7B (not redistributed)", "READ_ONLY_STRUCTURED_EXPLANATION_NO_CONTROL_AUTHORITY"),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    rows = []
    for relative, model_id, license_id, upstream, boundary in ASSETS:
        path = ROOT / relative
        rows.append(
            {
                "model_id": model_id,
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest(path),
                "storage": "git-lfs",
                "license": license_id,
                "upstream": upstream,
                "scope": boundary,
                "physical_authority": False,
            }
        )
    output = {
        "schema": "rootscope.public-model-assets.v1",
        "license_note": "See THIRD_PARTY_NOTICES.md and per-directory model cards.",
        "assets": rows,
    }
    target = ROOT / "model-assets" / "MANIFEST.json"
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {target} with {len(rows)} assets")


if __name__ == "__main__":
    main()
