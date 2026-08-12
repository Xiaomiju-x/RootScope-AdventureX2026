#!/usr/bin/env python3
"""Prepare an independent, category-focused E4 young-tree candidate collector.

E4 reuses the hardened E2 Commons API, canonical-license allowlist, decoded
image checks, page/SHA/dHash overlap rejection, metadata youth gate, mature and
detail rejection, and creator cap.  Its source plan prioritizes Commons search
category operators and nursery/newly-planted language that E3 did not target
as directly.

This script performs network requests only when explicitly executed without
``--help``.  Merely importing it, compiling it, or running its offline tests
does not contact Commons.  Every eventual output remains a machine-only,
unassigned candidate and is not A1, train-ready, print-eligible, or data-locked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import collect_young_tree_reacquisition as e2


STATUS = "MACHINE_ACQUIRED_YOUNG_TREE_E4_CATEGORY_CANDIDATES_NOT_TRAIN_READY"
OUTPUT_DATASET_NAME = "desert_plants_young_tree_category_reacquisition_e4"
COLLECTION_GENERATION = "E4"
RUN_AUTHORIZATION_STATUS = "NOT_RUN_PENDING_POST_E3_DECISION"

FALSE_AUTHORITY = {
    "data_locked": False,
    "dataset_manifest_write": False,
    "human_review": False,
    "model_qualification": False,
    "print_eligibility": False,
    "rights_approval": False,
    "split_assignment": False,
    "training_eligibility": False,
    "visual_truth": False,
}

REQUIRED_EXISTING_DATASET_NAMES = (
    "desert_plants_v1",
    "desert_plants_wikimedia_staging_e0",
    "desert_plants_whole_plant_reacquisition_e1",
    "desert_plants_young_tree_reacquisition_e2",
    "desert_plants_young_tree_reacquisition_e3",
)


def category_source(query: str, hint: str) -> e2.base.Source:
    return e2.young_source(query, hint)


# Keep category-operator queries literal and first.  Metadata youth evidence is
# still required from the Commons title/ImageDescription by e2.metadata_gate;
# a query match never becomes a label by itself.
E4_SOURCE_PLAN = tuple(
    category_source(query, hint)
    for query, hint in (
        ("incategory:Seedlings tree", "tree seedling from Commons Seedlings category"),
        ("incategory:Seedlings sapling", "tree sapling from Commons Seedlings category"),
        ("incategory:Seedlings Acacia", "Acacia spp. seedling"),
        ("incategory:Seedlings Vachellia", "Vachellia spp. seedling"),
        ("incategory:Seedlings Senegalia", "Senegalia spp. seedling"),
        ("incategory:Seedlings Prosopis", "Prosopis spp. seedling"),
        ("incategory:Seedlings mesquite", "mesquite seedling"),
        ("incategory:Seedlings Tamarix", "Tamarix spp. seedling"),
        ("incategory:Plant_nurseries seedling", "plant-nursery tree seedling"),
        ("incategory:Plant_nurseries tree seedling", "plant-nursery tree seedling"),
        ("incategory:Plant_nurseries sapling", "plant-nursery tree sapling"),
        ("incategory:Plant_nurseries Acacia seedling", "nursery Acacia seedling"),
        ("incategory:Plant_nurseries Vachellia seedling", "nursery Vachellia seedling"),
        ("incategory:Plant_nurseries Senegalia seedling", "nursery Senegalia seedling"),
        ("incategory:Plant_nurseries Prosopis seedling", "nursery Prosopis seedling"),
        ("incategory:Plant_nurseries mesquite seedling", "nursery mesquite seedling"),
        ("incategory:Plant_nurseries Tamarix seedling", "nursery Tamarix seedling"),
        ('"newly planted" tree sapling', "newly planted tree sapling"),
        ('"newly planted sapling"', "newly planted tree sapling"),
        ('"newly planted" Acacia sapling', "newly planted Acacia sapling"),
        ('"newly planted" Prosopis sapling', "newly planted Prosopis sapling"),
        ('"tree sapling" whole plant', "whole tree sapling"),
        ('"tree seedling" whole plant', "whole tree seedling"),
        ('"tree sapling" nursery', "nursery tree sapling"),
        ('"tree seedling" nursery', "nursery tree seedling"),
        ('"single sapling" tree', "isolated single tree sapling"),
        ('"single seedling" tree', "isolated single tree seedling"),
        ('"young tree" newly planted', "newly planted young tree"),
        ('"desert nursery" tree seedling', "desert nursery tree seedling"),
        ('"dryland nursery" tree sapling', "dryland nursery tree sapling"),
        ('"arid nursery" tree seedling', "arid nursery tree seedling"),
        ('"mesquite nursery" seedling', "nursery mesquite seedling"),
        ('"Prosopis nursery" sapling', "nursery Prosopis sapling"),
        ('"Acacia nursery" sapling', "nursery Acacia sapling"),
        ('"Tamarix nursery" sapling', "nursery Tamarix sapling"),
    )
)


def adventurex_root() -> Path:
    return Path(__file__).resolve().parents[2]


def required_existing_roots(root: Path | None = None) -> list[Path]:
    base = (root or adventurex_root()).resolve()
    return [base / "datasets" / name for name in REQUIRED_EXISTING_DATASET_NAMES]


def merge_required_existing(extras: list[Path] | None, root: Path | None = None) -> list[Path]:
    ordered = required_existing_roots(root) + list(extras or [])
    result: list[Path] = []
    seen: set[str] = set()
    for path in ordered:
        resolved = path.resolve(strict=False)
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def validate_required_existing(existing: list[Path], root: Path | None = None) -> None:
    normalized = {str(path.resolve(strict=False)).casefold(): path for path in existing}
    missing_bindings = [
        path
        for path in required_existing_roots(root)
        if str(path.resolve(strict=False)).casefold() not in normalized
    ]
    if missing_bindings:
        raise ValueError(
            "E4 must bind every E0/E1/E2/E3 and seed exclusion root: "
            + ", ".join(str(path) for path in missing_bindings)
        )
    missing_manifests = [path / "manifest.jsonl" for path in required_existing_roots(root) if not (path / "manifest.jsonl").is_file()]
    if missing_manifests:
        raise FileNotFoundError(
            "E4 is fail-closed until every required exclusion manifest exists: "
            + ", ".join(str(path) for path in missing_manifests)
        )


def existing_manifest_snapshot(existing: list[Path]) -> list[dict[str, str]]:
    return [
        {
            "dataset_name": path.name,
            "dataset_root": str(path.resolve(strict=False)),
            "manifest_sha256": e2.base.sha256_file(path / "manifest.jsonl"),
        }
        for path in existing
    ]


def paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve(strict=False)
    right = right.resolve(strict=False)
    return left == right or left in right.parents or right in left.parents


def _write_json(path: Path, value: object) -> None:
    e2.base.atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def enforce_candidate_fail_closed(record: dict[str, Any]) -> None:
    record.update(
        {
            "collection_generation": COLLECTION_GENERATION,
            "collection_status": STATUS,
            "machine_curated_only": True,
            "human_reviewed": False,
            "data_locked": False,
            "formal_a1_dataset": False,
            "rights_approved": False,
            "visual_whole_plant_verified": False,
            "biological_age_verified": False,
            "biological_age_status": "METADATA_YOUTH_GATED_VISUALLY_UNVERIFIED",
            "authority": dict(FALSE_AUTHORITY),
            "split": "UNASSIGNED_DO_NOT_TRAIN",
            "training_eligible": False,
            "print_eligible": False,
        }
    )


def _e4_save_outputs(
    original_save: Callable[[Path, list[dict[str, Any]], str, int], None],
    output: Path,
    records: list[dict[str, Any]],
    policy_sha: str,
    existing_count: int,
    run_args: argparse.Namespace,
) -> None:
    for record in records:
        enforce_candidate_fail_closed(record)
    original_save(output, records, policy_sha, existing_count)

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "schema_version": "rootscope.young_tree_category_reacquisition_summary.e4.v1",
            "status": STATUS,
            "collection_generation": COLLECTION_GENERATION,
            "collection_strategy": "COMMONS_CATEGORY_AND_NURSERY_SEARCH_PRIORITY",
            "authority": dict(FALSE_AUTHORITY),
            "human_reviewed": False,
            "data_locked": False,
            "formal_a1_dataset": False,
            "biological_age_verified": False,
            "visual_whole_plant_verified": False,
            "required_exclusion_datasets": list(REQUIRED_EXISTING_DATASET_NAMES),
            "overlap_rejection_axes": [
                "pageid",
                "commons_sha1",
                "download_sha256",
                "dhash64_hamming_distance",
            ],
            "max_per_creator": run_args.max_per_creator,
            "dhash_reject_distance_inclusive": run_args.dhash_distance,
            "metadata_youth_gate_required": True,
            "mature_and_detail_reject_required": True,
        }
    )
    _write_json(summary_path, summary)

    plan_path = output / "source_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.update(
        {
            "schema_version": "rootscope.young_tree_category_reacquisition_source_plan.e4.v1",
            "status": STATUS,
            "collection_generation": COLLECTION_GENERATION,
            "collection_strategy": "COMMONS_CATEGORY_AND_NURSERY_SEARCH_PRIORITY",
            "authority": dict(FALSE_AUTHORITY),
            "required_exclusion_datasets": list(REQUIRED_EXISTING_DATASET_NAMES),
            "metadata_youth_gate_required": True,
            "mature_and_detail_reject_required": True,
            "candidate_visual_truth": False,
        }
    )
    _write_json(plan_path, plan)

    receipt = {
        "schema_version": "rootscope.young_tree_category_reacquisition_receipt.e4.v1",
        "status": STATUS,
        "collection_generation": COLLECTION_GENERATION,
        "authority": dict(FALSE_AUTHORITY),
        "candidate_count": len(records),
        "formal_a1_dataset": False,
        "human_reviewed": False,
        "data_locked": False,
        "training_eligible": False,
        "print_eligible": False,
        "biological_age_verified": False,
        "visual_whole_plant_verified": False,
        "required_exclusion_datasets": list(REQUIRED_EXISTING_DATASET_NAMES),
        "existing_manifest_snapshot_start": run_args.existing_manifest_snapshot_start,
        "existing_manifests_must_remain_unchanged_through_run": True,
        "manifest_sha256": e2.base.sha256_file(output / "manifest.jsonl"),
        "source_plan_sha256": e2.base.sha256_file(output / "source_plan.json"),
        "license_policy_sha256": policy_sha,
        "overlap_rejection_axes": [
            "pageid",
            "commons_sha1",
            "download_sha256",
            "dhash64_hamming_distance",
        ],
        "explicit_non_claims": [
            "HUMAN_REVIEWED",
            "VISUAL_WHOLE_PLANT_VERIFIED",
            "BIOLOGICAL_AGE_VERIFIED",
            "RIGHTS_APPROVED",
            "TRAIN_READY",
            "PRINT_ELIGIBLE",
            "DATA_LOCKED",
            "A1_DATASET",
        ],
    }
    _write_json(output / "collection_receipt.json", receipt)

    readme = f"""# RootScope category-focused young-tree reacquisition E4

Status: `{STATUS}`

E4 prioritizes Wikimedia Commons category-operator searches for seedlings,
plant nurseries, newly planted saplings, and species-specific Acacia,
Vachellia, Senegalia, Prosopis/mesquite and Tamarix seedlings. It reuses the
hardened E2 collector for API access, canonical-license allowlisting, decoded
image validation, creator caps and page/SHA/dHash overlap rejection.

The seed set plus E0, E1, E2 and E3 are mandatory exclusion manifests. The
collector fails closed if any is absent. A youth term in Commons title or image
description remains mandatory; mature/detail terms remain hard rejects.

Every row remains `UNASSIGNED_DO_NOT_TRAIN`, `training_eligible=false`,
`print_eligible=false`, `human_reviewed=false`, `data_locked=false`, and all
authority values are false. Metadata is not visual or biological-age truth.
Strict whole-single-plant visual triage is required after acquisition.
"""
    e2.base.atomic_text(output / "README.md", readme)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script = Path(__file__).resolve()
    root = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "datasets" / OUTPUT_DATASET_NAME,
    )
    parser.add_argument(
        "--existing",
        type=Path,
        action="append",
        default=None,
        help=(
            "additional dataset root to exclude; repeatable. Mandatory seed/E0/E1/E2/E3 "
            "roots are always included and cannot be removed"
        ),
    )
    parser.add_argument(
        "--license-policy",
        type=Path,
        default=script.with_name("wikimedia_license_policy_v1.json"),
    )
    parser.add_argument("--target", type=int, default=80)
    parser.add_argument("--api-batches", type=int, default=20)
    parser.add_argument("--max-per-creator", type=int, default=3)
    parser.add_argument("--dhash-distance", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args(argv)
    args.existing = merge_required_existing(args.existing, root)
    if not 1 <= args.target <= 300:
        parser.error("--target must be between 1 and 300")
    if not 1 <= args.api_batches <= 20:
        parser.error("--api-batches must be between 1 and 20")
    if not 1 <= args.max_per_creator <= 50:
        parser.error("--max-per-creator must be between 1 and 50")
    if not 0 <= args.dhash_distance <= 16:
        parser.error("--dhash-distance must be between 0 and 16")
    if args.delay < 0:
        parser.error("--delay must be non-negative")
    return args


def collect(
    args: argparse.Namespace,
    *,
    backend: Callable[[argparse.Namespace], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    root = adventurex_root()
    validate_required_existing(args.existing, root)
    output = args.output.resolve(strict=False)
    overlapping = [path for path in args.existing if paths_overlap(output, path)]
    if overlapping:
        raise ValueError(
            "E4 output must be independent and non-overlapping with every exclusion dataset: "
            + ", ".join(str(path) for path in overlapping)
        )
    args.existing_manifest_snapshot_start = existing_manifest_snapshot(args.existing)

    original_status = e2.STATUS
    original_plan = e2.SOURCE_PLAN
    original_save = e2.save_outputs
    runner = backend or e2.collect
    try:
        e2.STATUS = STATUS
        e2.SOURCE_PLAN = E4_SOURCE_PLAN
        e2.save_outputs = lambda output_path, records, policy_sha, existing_count: _e4_save_outputs(
            original_save,
            output_path,
            records,
            policy_sha,
            existing_count,
            args,
        )
        records = runner(args)
        snapshot_after = existing_manifest_snapshot(args.existing)
        if snapshot_after != args.existing_manifest_snapshot_start:
            raise RuntimeError(
                "one or more exclusion manifests changed during E4 collection; "
                "the candidate run is not valid and must be restarted after E3 is frozen"
            )
        return records
    finally:
        e2.STATUS = original_status
        e2.SOURCE_PLAN = original_plan
        e2.save_outputs = original_save


def main() -> int:
    args = parse_args()
    records = collect(args)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": STATUS,
                "total": len(records),
                "target": args.target,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if len(records) >= args.target else 2


if __name__ == "__main__":
    raise SystemExit(main())
