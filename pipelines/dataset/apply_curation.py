#!/usr/bin/env python3
"""Apply an explicit RootScope visual-curation decision file to a manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_dataset = script.parents[2] / "datasets" / "desert_plants_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--curation", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    curation_path = args.curation.resolve() if args.curation else dataset / "curation_round1.json"
    curation = json.loads(curation_path.read_text(encoding="utf-8"))
    reservations = curation.get("reservations", {})
    rejections = curation.get("rejections", {})

    records = []
    seen = set()
    for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = str(item["pageid"])
        if key in reservations:
            item.update(
                {
                    "domain": "print_demo_source",
                    "split": "print_demo",
                    "review_status": "visual_pass_license_pending",
                    "print_eligible": False,
                    "curation_note": reservations[key],
                    "curation_version": curation["version"],
                }
            )
            seen.add(key)
        elif key in rejections:
            item.update(
                {
                    "split": "excluded",
                    "review_status": "rejected_visual",
                    "print_eligible": False,
                    "curation_note": rejections[key],
                    "curation_version": curation["version"],
                }
            )
            seen.add(key)
        records.append(item)

    expected = set(reservations) | set(rejections)
    missing = sorted(expected - seen)
    if missing:
        raise SystemExit(f"curation page IDs missing from manifest: {missing}")

    records.sort(key=lambda item: (item["class_id"], item["pageid"]))
    with (dataset / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    summary_path = dataset / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "curation_version": curation["version"],
            "review_status_counts": dict(sorted(Counter(item["review_status"] for item in records).items())),
            "domain_counts": dict(sorted(Counter(item["domain"] for item in records).items())),
            "split_counts": dict(sorted(Counter(item["split"] for item in records).items())),
            "print_eligible_count": sum(bool(item["print_eligible"]) for item in records),
            "all_review_status": "mixed_see_review_status_counts",
            "all_splits": "mixed_see_split_counts",
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "curated": len(seen), "missing": missing}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
