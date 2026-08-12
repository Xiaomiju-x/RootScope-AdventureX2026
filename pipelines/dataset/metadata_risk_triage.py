#!/usr/bin/env python3
"""Deterministic metadata-only risk triage for the frozen RootScope queue.

This utility deliberately does *not* open image files.  It records lexical and
provenance-concentration signals that can help prioritize later visual review.
It cannot establish image contents, class truth, rights approval, eligibility,
split assignment, or a dataset lock.

Production output is constrained to ``review/ai_metadata_triage_v1`` beside the
input queue.  In particular, this tool never writes ``human_decisions`` or a
dataset manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
ADVENTUREX_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_QUEUE = (
    ADVENTUREX_ROOT
    / "datasets"
    / "desert_plants_wikimedia_staging_e0"
    / "review"
    / "candidate_review_queue.jsonl"
)
OUTPUT_DIR_NAME = "ai_metadata_triage_v1"
DEFAULT_OUTPUT_DIR = DEFAULT_QUEUE.parent / OUTPUT_DIR_NAME

SCHEMA_VERSION = "rootscope.ai_metadata_risk_triage.v1"
REQUIRED_FIELDS = (
    "asset",
    "source_group",
    "local_path",
    "class_hint",
    "acquisition_query",
    "title",
    "species_hint",
    "creator_group",
)
ALLOWED_CLASSES = {"grass_clump", "low_shrub", "young_tree", "unknown"}

AUTHORITY = {
    "visual_truth": False,
    "human_review": False,
    "rights_approval": False,
    "dataset_manifest_write": False,
    "training_eligibility": False,
    "print_eligibility": False,
    "split_assignment": False,
    "data_locked": False,
}

# Terms are intentionally conservative and auditable.  A hit is a review-risk
# signal, not a finding that the image actually has the named property.
RISK_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "detail_crop_signal",
        "severity": "HIGH",
        "fields": ("title",),
        "terms": (
            "close",
            "closeup",
            "close up",
            "macro",
            "detail",
            "details",
            "flower",
            "flowers",
            "flowerhead",
            "fruit",
            "fruits",
            "seed",
            "seeds",
            "seed head",
            "inflorescence",
            "spikelet",
            "panicle",
            "awn",
            "awns",
            "leaf",
            "leaves",
            "foliage",
            "branch",
            "branches",
            "bark",
            "trunk",
            "root",
            "roots",
            "pod",
            "pods",
            "thorn",
            "thorns",
            "spine",
            "spines",
            "bud",
            "buds",
            "blossom",
            "hoja",
            "folha",
            "fruto",
            "tronco",
            "rama",
            "detalle",
            "blatt",
            "blute",
            "frucht",
            "stamm",
            "zweig",
        ),
        "rationale": "The title suggests a detail, organ, or close crop rather than a whole-plant view.",
    },
    {
        "id": "herbarium_specimen_signal",
        "severity": "HIGH",
        "fields": ("title",),
        "terms": (
            "herbarium",
            "herbier",
            "specimen",
            "pressed plant",
            "pressed specimen",
            "exsiccat",
            "holotype",
            "isotype",
            "lectotype",
            "syntype",
            "type sheet",
            "museum sheet",
        ),
        "rationale": "The title suggests preserved/specimen material rather than a field photograph.",
    },
    {
        "id": "map_text_illustration_signal",
        "severity": "HIGH",
        "fields": ("title",),
        "terms": (
            "map",
            "dmap",
            "distribution map",
            "range map",
            "illustration",
            "botanical plate",
            "plate",
            "drawing",
            "diagram",
            "engraving",
            "lithograph",
            "watercolor",
            "watercolour",
            "screenshot",
            "screen shot",
            "index card",
            "file card",
            "report of",
            "page ",
            "issue ",
            "poster",
            "logo",
            "icon",
            "scan",
        ),
        "rationale": "The title suggests a document, map, text page, or illustration rather than a natural scene.",
    },
    {
        "id": "landscape_many_subjects_signal",
        "severity": "MEDIUM",
        "fields": ("title",),
        "terms": (
            "landscape",
            "panorama",
            "panoramio",
            "habitat",
            "forest",
            "woodland",
            "grove",
            "plantation",
            "field",
            "meadow",
            "prairie",
            "garden",
            "park",
            "reserve",
            "oasis",
            "savanna",
            "steppe",
            "vegetation",
            "bushland",
            "thicket",
            "mountain",
            "mountains",
            "valley",
            "dune",
            "dunes",
            "desert scene",
            "many plants",
            "many trees",
            "cactus forest",
        ),
        "rationale": "The title suggests a broad scene or multiple subjects, which may weaken single-subject morphology.",
    },
)

MATURE_DEAD_TERMS = (
    "mature",
    "old tree",
    "old poplar",
    "old tamarix",
    "ancient",
    "dead tree",
    "deadwood",
    "dead wood",
    "fallen tree",
    "tree stump",
    "stump",
    "snag",
    "burnt tree",
    "burned tree",
    "tree skeleton",
    "deadvlei",
    "dead vlei",
)
YOUNG_TREE_TERMS = ("young", "sapling", "seedling", "juvenile", "baby tree", "baby saguaro")

HIGH_RISK_IDS = {
    "detail_crop_signal",
    "herbarium_specimen_signal",
    "map_text_illustration_signal",
    "mature_dead_tree_signal",
}


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[_\-]+", " ", value)


def _term_pattern(term: str) -> re.Pattern[str]:
    normalized = _normalized(term).strip()
    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![\w]){escaped}(?![\w])")


COMPILED_RISK_RULES = tuple(
    (
        rule,
        tuple((term, _term_pattern(term)) for term in rule["terms"]),
    )
    for rule in RISK_RULES
)
COMPILED_MATURE_DEAD = tuple((term, _term_pattern(term)) for term in MATURE_DEAD_TERMS)
COMPILED_YOUNG_TREE = tuple((term, _term_pattern(term)) for term in YOUNG_TREE_TERMS)


def _matches(value: str, compiled: Iterable[tuple[str, re.Pattern[str]]]) -> list[str]:
    normalized = _normalized(value)
    return [term for term, pattern in compiled if pattern.search(normalized)]


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _validate_output_boundary(queue: Path, output_dir: Path) -> None:
    queue_parent = queue.resolve(strict=True).parent
    output_resolved = output_dir.resolve(strict=False)
    expected = queue_parent / OUTPUT_DIR_NAME
    if output_resolved != expected:
        raise ValueError(f"output must be exactly {expected}")
    if any(part.casefold() == "human_decisions" for part in output_resolved.parts):
        raise ValueError("human_decisions is outside this tool's authority")
    if output_dir.exists() and output_dir.is_symlink():
        raise ValueError("output directory must not be a symlink")


def load_queue(queue: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    assets: set[str] = set()
    with queue.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on queue line {line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"queue line {line_number} is not an object")
            for field in REQUIRED_FIELDS:
                if not isinstance(raw.get(field), str) or not raw[field]:
                    raise ValueError(f"queue line {line_number} has invalid {field}")
            if raw["class_hint"] not in ALLOWED_CLASSES:
                raise ValueError(f"queue line {line_number} has unsupported class_hint")
            if raw["asset"] in assets:
                raise ValueError(f"duplicate asset on queue line {line_number}")
            assets.add(raw["asset"])
            rows.append({field: raw[field] for field in REQUIRED_FIELDS})
    if not rows:
        raise ValueError("queue is empty")
    return rows


def _flag(
    identifier: str,
    severity: str,
    rationale: str,
    evidence: Mapping[str, Sequence[str] | str],
) -> dict[str, Any]:
    normalized_evidence: dict[str, Any] = {}
    for key in sorted(evidence):
        value = evidence[key]
        normalized_evidence[key] = list(value) if isinstance(value, (tuple, list)) else value
    return {
        "id": identifier,
        "severity": severity,
        "rationale": rationale,
        "evidence": normalized_evidence,
    }


def analyze_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    creator_total = Counter(row["creator_group"] for row in rows)
    creator_query = Counter(
        (row["class_hint"], row["acquisition_query"], row["creator_group"])
        for row in rows
    )
    results: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        risk_flags: list[dict[str, Any]] = []
        context_flags: list[dict[str, Any]] = []
        support_signals: list[dict[str, Any]] = []

        for rule, compiled_terms in COMPILED_RISK_RULES:
            evidence: dict[str, list[str]] = {}
            for field in rule["fields"]:
                hits = _matches(row[field], compiled_terms)
                if hits:
                    evidence[field] = hits
            if evidence:
                risk_flags.append(
                    _flag(rule["id"], rule["severity"], rule["rationale"], evidence)
                )

        if row["class_hint"] == "young_tree":
            mature_hits = _matches(row["title"], COMPILED_MATURE_DEAD)
            young_hits = _matches(row["title"], COMPILED_YOUNG_TREE)
            if mature_hits:
                risk_flags.append(
                    _flag(
                        "mature_dead_tree_signal",
                        "HIGH",
                        "The title explicitly suggests a mature, old, dead, fallen, or damaged tree.",
                        {"title": mature_hits},
                    )
                )
            if young_hits:
                support_signals.append(
                    _flag(
                        "young_tree_title_support",
                        "CONTEXT",
                        "The title contains an explicit young-age term; image content remains unverified.",
                        {"title": young_hits},
                    )
                )
            else:
                risk_flags.append(
                    _flag(
                        "young_tree_age_unverified",
                        "MEDIUM",
                        "The species acquisition query does not establish that the depicted tree is young.",
                        {"acquisition_query": row["acquisition_query"]},
                    )
                )

        if row["class_hint"] == "unknown":
            negative_hint = row["species_hint"] if row["species_hint"].casefold().startswith("negative:") else ""
            context_flags.append(
                _flag(
                    "unknown_acquisition_bucket",
                    "CONTEXT",
                    "This row belongs to an intentionally heterogeneous unknown/negative acquisition bucket.",
                    {
                        "acquisition_query": row["acquisition_query"],
                        "species_hint": negative_hint or row["species_hint"],
                    },
                )
            )

        same_query_count = creator_query[
            (row["class_hint"], row["acquisition_query"], row["creator_group"])
        ]
        total_creator_count = creator_total[row["creator_group"]]
        if same_query_count >= 8 or total_creator_count >= 20:
            risk_flags.append(
                _flag(
                    "creator_series_concentration",
                    "MEDIUM",
                    "Repeated creator provenance may represent a correlated capture series and reduce source diversity.",
                    {
                        "creator_group": row["creator_group"],
                        "same_class_query_count": str(same_query_count),
                        "queue_total_count": str(total_creator_count),
                    },
                )
            )

        risk_ids = {flag["id"] for flag in risk_flags}
        if risk_ids & HIGH_RISK_IDS:
            priority = "HIGH_METADATA_RISK"
        elif risk_flags:
            priority = "REVIEW_PRIORITY"
        else:
            priority = "NO_OBVIOUS_METADATA_RISK_SIGNAL"

        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "queue_index": index,
                "asset": row["asset"],
                "source_group": row["source_group"],
                "local_path": row["local_path"],
                "acquisition_metadata": {
                    "class_hint": row["class_hint"],
                    "acquisition_query": row["acquisition_query"],
                    "title": row["title"],
                    "species_hint": row["species_hint"],
                    "creator_group": row["creator_group"],
                },
                "risk_priority": priority,
                "risk_flags": risk_flags,
                "context_flags": context_flags,
                "support_signals": support_signals,
                "metadata_only": True,
                "visual_truth_established": False,
            }
        )
    return results


def _aggregate(records: Sequence[Mapping[str, Any]], key_fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record["acquisition_metadata"]
        key = tuple(metadata[field] for field in key_fields)
        groups[key].append(record)

    summaries: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        risk_counts = Counter(flag["id"] for row in group for flag in row["risk_flags"])
        context_counts = Counter(flag["id"] for row in group for flag in row["context_flags"])
        priority_counts = Counter(row["risk_priority"] for row in group)
        creators = Counter(row["acquisition_metadata"]["creator_group"] for row in group)
        summaries.append(
            {
                **dict(zip(key_fields, key)),
                "rows": len(group),
                "priority_counts": dict(sorted(priority_counts.items())),
                "risk_flag_counts": dict(sorted(risk_counts.items())),
                "context_flag_counts": dict(sorted(context_counts.items())),
                "creator_group_count": len(creators),
                "largest_creator_series": max(creators.values()),
            }
        )
    return summaries


def build_summary(queue: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    risk_counts = Counter(flag["id"] for row in records for flag in row["risk_flags"])
    context_counts = Counter(flag["id"] for row in records for flag in row["context_flags"])
    priority_counts = Counter(row["risk_priority"] for row in records)
    creator_counts = Counter(
        row["acquisition_metadata"]["creator_group"] for row in records
    )
    concentrated = [
        {"creator_group": creator, "rows": count}
        for creator, count in sorted(creator_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 8
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": "METADATA_ONLY_RISK_SIGNALS",
        "queue_sha256": _sha256_file(queue),
        "rows": len(records),
        "fields_used_for_risk_inference": [
            "acquisition_query",
            "title",
            "species_hint",
            "creator_group",
        ],
        "priority_counts": dict(sorted(priority_counts.items())),
        "risk_flag_counts": dict(sorted(risk_counts.items())),
        "context_flag_counts": dict(sorted(context_counts.items())),
        "by_class": _aggregate(records, ("class_hint",)),
        "by_class_query": _aggregate(records, ("class_hint", "acquisition_query")),
        "concentrated_creator_groups_min_8": concentrated,
        "authority": AUTHORITY,
        "limitations": [
            "No image bytes were opened or analyzed.",
            "Keyword hits may be false positives and misses may be false negatives.",
            "Creator-group concentration indicates correlation risk, not duplication or bad content.",
            "Acquisition hints are not reviewed target labels.",
        ],
    }


def build_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# RootScope 冻结候选队列：AI 元数据风险初筛 v1",
        "",
        "> 本报告只分析 acquisition_query、title、species_hint、creator_group。没有读取图像像素，不能证明视觉内容、类别真值、授权结论或训练资格。",
        "",
        f"- 队列行数：{summary['rows']}",
        f"- 队列 SHA-256：`{summary['queue_sha256']}`",
    ]
    for name, count in summary["priority_counts"].items():
        lines.append(f"- {name}：{count}")
    lines.extend(["", "## 风险信号计数", ""])
    if summary["risk_flag_counts"]:
        lines.extend(
            f"- `{name}`：{count}" for name, count in summary["risk_flag_counts"].items()
        )
    else:
        lines.append("- 无")

    lines.extend(["", "## 按类别", "", "| acquisition hint | rows | high | review | no obvious signal |", "|---|---:|---:|---:|---:|"])
    for row in summary["by_class"]:
        counts = row["priority_counts"]
        lines.append(
            "| {class_hint} | {rows} | {high} | {review} | {none} |".format(
                class_hint=row["class_hint"],
                rows=row["rows"],
                high=counts.get("HIGH_METADATA_RISK", 0),
                review=counts.get("REVIEW_PRIORITY", 0),
                none=counts.get("NO_OBVIOUS_METADATA_RISK_SIGNAL", 0),
            )
        )

    lines.extend(["", "## 按类别 / acquisition query", "", "| class | query | rows | high | review | no obvious | top signals |", "|---|---|---:|---:|---:|---:|---|"])
    for row in summary["by_class_query"]:
        counts = row["priority_counts"]
        signals = ", ".join(f"{key}:{value}" for key, value in row["risk_flag_counts"].items()) or "-"
        lines.append(
            "| {class_hint} | {query} | {rows} | {high} | {review} | {none} | {signals} |".format(
                class_hint=row["class_hint"].replace("|", "\\|"),
                query=row["acquisition_query"].replace("|", "\\|"),
                rows=row["rows"],
                high=counts.get("HIGH_METADATA_RISK", 0),
                review=counts.get("REVIEW_PRIORITY", 0),
                none=counts.get("NO_OBVIOUS_METADATA_RISK_SIGNAL", 0),
                signals=signals.replace("|", "\\|"),
            )
        )

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- HIGH/REVIEW 表示应优先检查，不表示图像已经被判为错类。",
            "- unknown 是既定负类采集桶；`unknown_acquisition_bucket` 只是上下文标记。",
            "- young_tree 若标题没有明确 young/sapling/seedling 等年龄词，会标为年龄未证实；这不是成熟树判决。",
            "- creator_group 集中只说明来源相关性风险，不能单独证明重复图或低质量。",
            "",
        ]
    )
    return "\n".join(lines)


def run(queue: Path, output_dir: Path) -> dict[str, Any]:
    queue = queue.resolve(strict=True)
    _validate_output_boundary(queue, output_dir)
    rows = load_queue(queue)
    records = analyze_rows(rows)
    summary = build_summary(queue, records)

    output_dir.mkdir(parents=False, exist_ok=True)
    records_path = output_dir / "metadata_risk_records.jsonl"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "REPORT.md"
    receipt_path = output_dir / "receipt.json"

    records_payload = b"".join(_canonical_json(record) for record in records)
    summary_payload = _canonical_json(summary)
    report_payload = build_report(summary).encode("utf-8")
    _write_atomic(records_path, records_payload)
    _write_atomic(summary_path, summary_payload)
    _write_atomic(report_path, report_payload)

    artifacts = {
        records_path.name: _sha256_bytes(records_payload),
        summary_path.name: _sha256_bytes(summary_payload),
        report_path.name: _sha256_bytes(report_payload),
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "AI_METADATA_TRIAGE_COMPLETE",
        "analysis_type": "METADATA_ONLY_RISK_SIGNALS",
        "queue_sha256": summary["queue_sha256"],
        "rows": len(records),
        "artifacts_sha256": artifacts,
        "authority": AUTHORITY,
    }
    _write_atomic(receipt_path, _canonical_json(receipt))
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run(args.queue, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
