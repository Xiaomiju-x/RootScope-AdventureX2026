#!/usr/bin/env python3
"""Generate deterministic, citation-bound RootScope structured SFT records.

These records are rule-validated supervision, not cloud-teacher output.  Cloud
teacher samples, when available, must live in a separate provenance stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any


PLANTS = ("grass_clump", "low_shrub", "young_tree", "non_target")
INTENTS = (
    "FIELD_EXPLANATION",
    "SENSOR_CONFLICT",
    "OOD_REJECTION",
    "FAILURE_ANALYSIS",
    "DEFENSE_QA",
    "COUNTERFACTUAL",
)
FAULTS = (
    "NONE",
    "ACK_WITHOUT_MASS_LOSS",
    "MASS_LOSS_WITH_NEIGHBOR_SPILL",
    "STALE_SENSOR",
    "VISION_OOD",
    "SERIAL_DISCONNECTED",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1536)
    args = parser.parse_args()
    if args.count < 1200 or args.count > 2400:
        raise SystemExit("count must stay within the frozen 1200-2400 range")
    gold = load_jsonl(args.gold)
    forbidden = load_jsonl(args.forbidden)
    if len(gold) < 60 or len(forbidden) < 30:
        raise SystemExit("RAG v2 must provide at least 60 gold and 30 forbidden rows")
    rng = random.Random(20260724)
    rows: list[dict[str, Any]] = []
    for index in range(args.count):
        intent = INTENTS[index % len(INTENTS)]
        plant = PLANTS[(index // len(INTENTS)) % len(PLANTS)]
        fault = FAULTS[(index // (len(INTENTS) * len(PLANTS))) % len(FAULTS)]
        qa = gold[index % len(gold)]
        forbidden_row = forbidden[index % len(forbidden)]
        confidence = round(0.42 + 0.07 * (index % 8), 2)
        ood = intent == "OOD_REJECTION" or fault == "VISION_OOD"
        ack = fault not in {"SERIAL_DISCONNECTED", "STALE_SENSOR"}
        mass_loss = 0.0 if fault == "ACK_WITHOUT_MASS_LOSS" else float(8 + index % 23)
        spill = 0.22 if fault == "MASS_LOSS_WITH_NEIGHBOR_SPILL" else 0.03
        adversarial = index % 5 == 0
        citations = (
            list(forbidden_row["citation_ids"])[:4]
            if adversarial
            else list(qa["citation_ids"])[:4]
        )
        if not citations:
            raise SystemExit(f"gold row has no citations: {qa['id']}")
        hold = (
            plant == "non_target"
            or ood
            or fault != "NONE"
            or confidence < 0.70
            or adversarial
        )
        reason_codes = []
        if plant == "non_target":
            reason_codes.append("NON_TARGET_HOLD")
        if ood:
            reason_codes.append("VISION_OOD_HOLD")
        if confidence < 0.70:
            reason_codes.append("VISION_CONFIDENCE_LOW")
        if fault == "ACK_WITHOUT_MASS_LOSS":
            reason_codes.append("ACK_MASS_CONFLICT")
        if fault == "MASS_LOSS_WITH_NEIGHBOR_SPILL":
            reason_codes.append("NEIGHBOR_SPILL_EXCESSIVE")
        if fault == "STALE_SENSOR":
            reason_codes.append("SENSOR_STALE")
        if fault == "SERIAL_DISCONNECTED":
            reason_codes.append("SERIAL_DISCONNECTED")
        if adversarial:
            reason_codes.append("ADVERSARIAL_REQUEST_REJECTED")
        if not reason_codes:
            reason_codes.append("EVIDENCE_CONSISTENT")
        observation = {
            "plant_class": plant,
            "plant_confidence": confidence,
            "vision_ood": ood,
            "fault": fault,
            "ack_received": ack,
            "mass_loss_g": mass_loss,
            "neighbor_spill_ratio": spill,
            "question": qa["question"],
            "retrieved_evidence_ids": citations,
            "adversarial_request": (
                forbidden_row["question"] if adversarial else None
            ),
        }
        response = {
            "observation_summary": (
                f"{plant}；视觉置信度 {confidence:.2f}；故障状态 {fault}。"
            ),
            "evidence_ids": citations,
            "reason_codes": sorted(reason_codes),
            "uncertainty": (
                "证据不足或冲突，保持 HOLD。"
                if hold
                else "当前证据一致，但仍仅生成只读建议。"
            ),
            "proposed_explanation": (
                forbidden_row.get("expected_answer", forbidden_row.get("safe_answer"))
                if adversarial
                else "拒绝形成灌溉动作；请补采或人工复核。"
                if hold
                else qa["answer"]
            ),
            "authority": False,
        }
        template_group = (
            f"{intent}:{plant}:{fault}:{index % len(gold)}:"
            f"{index % len(forbidden)}"
        )
        group_hash = int(hashlib.sha256(template_group.encode()).hexdigest()[:8], 16)
        split_bucket = group_hash % 100
        split = "train" if split_bucket < 82 else ("validation" if split_bucket < 91 else "test")
        row = {
            "schema": "rootscope.v3.sft-record.v1",
            "record_id": f"rootscope-sft-{index:05d}",
            "split": split,
            "template_group": template_group,
            "intent": intent,
            "instruction": (
                "你是 RootScope 固定式根区灌溉舱的只读证据解释器。"
                "仅根据给定观测和 citation id 输出严格 JSON；"
                "authority 必须为 false，不得生成串口、GPIO 或水泵命令。"
            ),
            "input": observation,
            "output": response,
            "provenance": {
                "generator": "deterministic_rootscope_contract_v3",
                "cloud_teacher_called": False,
                "teacher_logits_available": False,
                "gold_qa_id": qa["id"],
                "forbidden_qa_id": forbidden_row["id"],
            },
        }
        row["record_sha256"] = hashlib.sha256(canonical(row).encode()).hexdigest()
        rows.append(row)
    splits = {name: [row for row in rows if row["split"] == name] for name in (
        "train", "validation", "test"
    )}
    groups = {
        name: {row["template_group"] for row in values}
        for name, values in splits.items()
    }
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise SystemExit("template group leakage detected")
    adversarial_rows = [
        row for row in rows if row["input"]["adversarial_request"] is not None
    ]
    adversarial_rejected = [
        row
        for row in adversarial_rows
        if "ADVERSARIAL_REQUEST_REJECTED" in row["output"]["reason_codes"]
        and row["output"]["authority"] is False
        and row["output"]["proposed_explanation"].startswith("拒绝")
    ]
    if len(adversarial_rejected) != len(adversarial_rows):
        raise SystemExit("adversarial request rejection contract failed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, values in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        rendered = "".join(canonical(row) + "\n" for row in values)
        rendered_bytes = rendered.encode("utf-8")
        path.write_bytes(rendered_bytes)
        hashes[path.name] = hashlib.sha256(rendered_bytes).hexdigest()
    receipt = {
        "schema": "rootscope.v3.sft-dataset-receipt.v1",
        "status": "PASS_DETERMINISTIC_RULE_VALIDATED_NOT_CLOUD_TEACHER",
        "total": len(rows),
        "splits": {name: len(values) for name, values in splits.items()},
        "unique_template_groups": {
            name: len(values) for name, values in groups.items()
        },
        "file_sha256": hashes,
        "authority_true_count": sum(
            int(row["output"]["authority"] is not False) for row in rows
        ),
        "citation_empty_count": sum(
            int(not row["output"]["evidence_ids"]) for row in rows
        ),
        "cloud_teacher_calls": 0,
        "secret_values_read": False,
        "adversarial_request_count": len(adversarial_rows),
        "adversarial_rejected_count": len(adversarial_rejected),
    }
    (args.output_dir / "dataset_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
