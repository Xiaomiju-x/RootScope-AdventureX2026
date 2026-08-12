#!/usr/bin/env python3
"""Validate the immutable four-card RootScope bundle on an RDK X5.

This script is vision-only. It never opens serial, GPIO, pump, or probe devices.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


CLASSES = ("grass_clump", "low_shrub", "young_tree", "non_target")


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("rootscope_answer_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    runtime_module = load_runtime(bundle / "runtime" / "x5_answer_card_live.py")
    runtime = runtime_module.AnswerCardRuntime(bundle)
    output_dir = bundle / "evidence_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for truth in CLASSES:
        image_path = bundle / "evidence_inputs" / f"{truth}_holdout.jpg"
        image = runtime_module.cv2.imread(
            str(image_path), runtime_module.cv2.IMREAD_COLOR
        )
        if image is None:
            raise RuntimeError(f"cannot decode: {image_path}")
        result = runtime.infer(image)
        result["expected_class"] = truth
        result["x5_replay_passed"] = (
            result["decision"] == truth
            and result["state"] == "CONFIRMED_DUAL_EVIDENCE"
        )
        runtime_module.write_result(output_dir / f"{truth}.json", result)
        annotated = runtime_module.annotate(image, result, 0.0)
        if not runtime_module.cv2.imwrite(
            str(output_dir / f"{truth}_annotated.jpg"), annotated
        ):
            raise RuntimeError(f"cannot write annotation for {truth}")
        rows.append(
            {
                "truth": truth,
                "decision": result["decision"],
                "state": result["state"],
                "confidence": result["cnn"]["confidence"],
                "margin": result["cnn"]["margin"],
                "template": result["template"]["prediction"],
                "template_inliers": result["template"]["homography_inliers"],
                "latency_ms": result["latency_ms"],
                "passed": result["x5_replay_passed"],
            }
        )
    summary = {
        "schema": "rootscope.answer_cards.x5_replay.v1",
        "scope": "FOUR_FIXED_PRINTED_ANSWER_CARDS_ONLY",
        "physical_action_authority": False,
        "passed": all(row["passed"] for row in rows),
        "passed_count": sum(row["passed"] for row in rows),
        "total": len(rows),
        "rows": rows,
    }
    runtime_module.write_result(output_dir / "x5_replay_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
