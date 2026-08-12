#!/usr/bin/env python3
"""Acquire an independent E3 young-tree candidate pool.

E3 deliberately reuses the hardened Wikimedia, licensing, duplicate and image
checks from ``collect_young_tree_reacquisition`` while searching deeper result
pages and excluding E2 as well as every earlier RootScope dataset.  Everything
written by this collector remains machine-only, unassigned and ineligible for
formal training, printing or data lock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import collect_young_tree_reacquisition as e2


STATUS = "MACHINE_ACQUIRED_YOUNG_TREE_E3_CANDIDATES_NOT_TRAIN_READY"
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

# Put the most literal seedling/sapling searches first.  The remaining E2 plan
# is retained as a long-tail fallback, with exact page/SHA/dHash exclusions
# preventing any reuse of earlier assets.
E3_PREFIX = tuple(
    e2.young_source(query, hint)
    for query, hint in (
        ('"Acacia seedling" whole plant', "Acacia spp. seedling"),
        ('"Acacia sapling" whole plant', "Acacia spp. sapling"),
        ('"Vachellia seedling" whole plant', "Vachellia spp. seedling"),
        ('"Vachellia sapling" whole plant', "Vachellia spp. sapling"),
        ('"Senegalia seedling" whole plant', "Senegalia spp. seedling"),
        ('"Senegalia sapling" whole plant', "Senegalia spp. sapling"),
        ('"Prosopis seedling" whole plant', "Prosopis spp. seedling"),
        ('"Prosopis sapling" whole plant', "Prosopis spp. sapling"),
        ('"Tamarix seedling" whole plant', "Tamarix spp. seedling"),
        ('"Tamarix sapling" whole plant', "Tamarix spp. sapling"),
        ('"tree seedling" nursery pot', "nursery tree seedling"),
        ('"tree sapling" nursery pot', "nursery tree sapling"),
        ('"desert tree seedling"', "desert tree seedling"),
        ('"dryland tree seedling"', "dryland tree seedling"),
        ('"arid tree seedling"', "arid tree seedling"),
    )
)
E3_SOURCE_PLAN = E3_PREFIX + tuple(
    source for source in e2.SOURCE_PLAN if source.retrieval_query not in {item.retrieval_query for item in E3_PREFIX}
)


def _write_json(path: Path, value: object) -> None:
    e2.base.atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _e3_save_outputs(
    original_save,
    output: Path,
    records: list[dict],
    policy_sha: str,
    existing_count: int,
) -> None:
    for record in records:
        record["collection_generation"] = "E3"
        record["human_reviewed"] = False
        record["data_locked"] = False
        record["authority"] = dict(FALSE_AUTHORITY)
    original_save(output, records, policy_sha, existing_count)

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "schema_version": "rootscope.young_tree_reacquisition_summary.e3.v1",
            "status": STATUS,
            "collection_generation": "E3",
            "authority": dict(FALSE_AUTHORITY),
            "human_reviewed": False,
            "data_locked": False,
            "formal_a1_dataset": False,
        }
    )
    _write_json(summary_path, summary)

    plan_path = output / "source_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.update(
        {
            "schema_version": "rootscope.young_tree_reacquisition_source_plan.e3.v1",
            "status": STATUS,
            "collection_generation": "E3",
            "authority": dict(FALSE_AUTHORITY),
        }
    )
    _write_json(plan_path, plan)

    readme = f"""# RootScope young-tree-only reacquisition E3

Status: `{STATUS}`

This is an independent, metadata-gated Wikimedia Commons candidate pool.  It
excludes the seed set, E0, E1 and E2 by page id, Commons SHA-1, downloaded
SHA-256 and perceptual dHash.  A youth term in the Commons title or image
description is required, but that metadata is not visual ground truth.

Every row remains `UNASSIGNED_DO_NOT_TRAIN`, `training_eligible=false`,
`print_eligible=false`, `human_reviewed=false`, and `data_locked=false`.
Strict pixel-level machine triage followed by a separate formal review would
be required before any A1 or data-lock claim.
"""
    e2.base.atomic_text(output / "README.md", readme)


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    adventurex = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=adventurex / "datasets" / "desert_plants_young_tree_reacquisition_e3",
    )
    parser.add_argument("--existing", type=Path, action="append", default=None)
    parser.add_argument(
        "--license-policy",
        type=Path,
        default=script.with_name("wikimedia_license_policy_v1.json"),
    )
    parser.add_argument("--target", type=int, default=80)
    parser.add_argument("--api-batches", type=int, default=20)
    parser.add_argument("--max-per-creator", type=int, default=4)
    parser.add_argument("--dhash-distance", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args()
    if args.existing is None:
        args.existing = [
            adventurex / "datasets" / "desert_plants_v1",
            adventurex / "datasets" / "desert_plants_wikimedia_staging_e0",
            adventurex / "datasets" / "desert_plants_whole_plant_reacquisition_e1",
            adventurex / "datasets" / "desert_plants_young_tree_reacquisition_e2",
        ]
    if not 1 <= args.target <= 300:
        parser.error("--target must be between 1 and 300")
    if not 1 <= args.api_batches <= 20:
        parser.error("--api-batches must be between 1 and 20")
    if not 1 <= args.max_per_creator <= 50:
        parser.error("--max-per-creator must be between 1 and 50")
    if not 0 <= args.dhash_distance <= 16:
        parser.error("--dhash-distance must be between 0 and 16")
    return args


def main() -> int:
    args = parse_args()
    original_save = e2.save_outputs
    e2.STATUS = STATUS
    e2.SOURCE_PLAN = E3_SOURCE_PLAN
    e2.save_outputs = lambda output, records, policy_sha, existing_count: _e3_save_outputs(
        original_save, output, records, policy_sha, existing_count
    )
    records = e2.collect(args)
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
