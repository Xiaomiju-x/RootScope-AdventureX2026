"""Capture-session/source-group split assignment with leakage auditing."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


def derive_group(record: Mapping[str, Any]) -> str:
    for key in ("capture_session_id", "session_id", "source_group", "creator_group"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return f"{key}:{value.strip()}"
    raise ValueError("record has no capture/source group identity")


def assign_group_splits(
    records: Iterable[Mapping[str, Any]],
    *,
    salt: str = "rootscope-v3-e0",
    train_fraction: float = 0.72,
    validation_fraction: float = 0.14,
) -> list[dict[str, Any]]:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation must be less than one")
    output: list[dict[str, Any]] = []
    assignments: dict[str, str] = {}
    for item in records:
        row = dict(item)
        group = derive_group(row)
        if group not in assignments:
            unit = int(hashlib.sha256(f"{salt}|{group}".encode()).hexdigest()[:16], 16) / 2**64
            assignments[group] = (
                "train" if unit < train_fraction
                else "validation" if unit < train_fraction + validation_fraction
                else "test"
            )
        row["split_group"] = group
        row["split"] = assignments[group]
        output.append(row)
    return output


def audit_group_splits(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for row in records:
        group, split = derive_group(row), str(row.get("split", ""))
        groups[group].add(split)
        split_counts[split] += 1
        sha = row.get("sha256") or row.get("asset_sha256") or row.get("copied_image_sha256")
        if isinstance(sha, str) and sha:
            hashes[sha].add(split)
    group_leaks = sorted(group for group, splits in groups.items() if len(splits) > 1)
    hash_leaks = sorted(value for value, splits in hashes.items() if len(splits) > 1)
    return {
        "schema_version": "rootscope.capture-group-split-audit.v1",
        "status": "PASS" if not group_leaks and not hash_leaks else "FAIL",
        "record_count": sum(split_counts.values()),
        "group_count": len(groups),
        "split_counts": dict(sorted(split_counts.items())),
        "group_leak_count": len(group_leaks),
        "hash_leak_count": len(hash_leaks),
        "group_leaks": group_leaks,
        "hash_leaks": hash_leaks,
        "truth_boundary": "SPLIT_INTEGRITY_ONLY_NOT_MODEL_ACCURACY",
    }
