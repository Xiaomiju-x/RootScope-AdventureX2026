#!/usr/bin/env python3
"""Independently audit RootScope machine-curated provisional v2.

The audit is read-only with respect to the dataset.  An optional report may be
written outside the pack (normally under ``adventurex/evidence``).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import build_machine_curated_provisional as v1
import build_machine_curated_provisional_v2 as v2


EXPECTED_PRINT_PAGEIDS = {38233728, 74079996, 66745979, 94700516, 75760716, 98911085}
EXPECTED_EXCLUDED_YOUNG = set(v2.SUPPLEMENT_REQUESTS["young_tree"])


def check(condition: bool, name: str, failures: list[str], checks: list[str]) -> None:
    if condition:
        checks.append(name)
    else:
        failures.append(name)


def read_sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative:
            raise ValueError(f"invalid SHA256SUMS line {line_number}")
        if relative in result:
            raise ValueError(f"duplicate SHA256SUMS path {relative}")
        result[relative] = digest
    return result


def audit(workspace: Path, pack: Path) -> dict[str, Any]:
    workspace = workspace.resolve(strict=True)
    pack = pack.resolve(strict=True)
    expected_pack = (workspace / "datasets" / v2.OUTPUT_NAME).resolve(strict=True)
    if pack != expected_pack:
        raise ValueError(f"audit target must be exactly {expected_pack}")

    checks: list[str] = []
    failures: list[str] = []
    receipt = json.loads((pack / "receipt.json").read_text(encoding="utf-8"))
    rows = v1.load_jsonl(pack / "manifest.jsonl")
    decisions = v1.load_jsonl(pack / "source_decision_manifest.jsonl")
    sums = read_sums(pack / "SHA256SUMS")

    actual_files = {
        path.relative_to(pack).as_posix()
        for path in pack.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    check(set(sums) == actual_files, "SHA256SUMS_COVERS_EVERY_NON_SUM_FILE", failures, checks)
    sum_mismatches = [
        relative
        for relative, expected in sums.items()
        if not (pack / relative).is_file() or v1.sha256_file(pack / relative) != expected
    ]
    check(not sum_mismatches, "SHA256SUMS_ALL_MATCH", failures, checks)

    manifest_text = (pack / "manifest.jsonl").read_bytes()
    decision_text = (pack / "source_decision_manifest.jsonl").read_bytes()
    check(
        v2.sha256_bytes(manifest_text) == receipt.get("manifest_sha256"),
        "RECEIPT_MANIFEST_SHA_MATCH",
        failures,
        checks,
    )
    check(
        v2.sha256_bytes(decision_text) == receipt.get("source_decision_manifest_sha256"),
        "RECEIPT_DECISION_SHA_MATCH",
        failures,
        checks,
    )
    check(
        v1.sha256_file(Path(v2.__file__).resolve(strict=True)) == receipt.get("implementation_sha256"),
        "RECEIPT_IMPLEMENTATION_SHA_MATCH",
        failures,
        checks,
    )

    fail_closed = True
    for container in [receipt, *rows, *decisions]:
        authority = container.get("authority")
        if not isinstance(authority, dict) or not authority or any(value is not False for value in authority.values()):
            fail_closed = False
            break
        for key in ("data_locked", "human_reviewed", "print_eligible", "training_eligible"):
            if container.get(key) is not False:
                fail_closed = False
                break
        if not fail_closed:
            break
    check(fail_closed, "ALL_RECORD_AND_RECEIPT_AUTHORITY_FAIL_CLOSED", failures, checks)
    check(
        receipt.get("status") == v2.STATUS and all(row.get("status") == v2.STATUS for row in rows),
        "STATUS_IS_MACHINE_ONLY_NOT_A1_NOT_LOCKED",
        failures,
        checks,
    )

    pageids = [int(row["pageid"]) for row in rows]
    source_groups = [str(row["source_group"]) for row in rows]
    content_hashes = [str(row["copied_image_sha256"]) for row in rows]
    check(len(pageids) == len(set(pageids)), "PAGEIDS_UNIQUE", failures, checks)
    check(len(source_groups) == len(set(source_groups)), "SOURCE_GROUPS_UNIQUE", failures, checks)
    check(len(content_hashes) == len(set(content_hashes)), "CONTENT_SHA256_UNIQUE", failures, checks)

    copied_mismatches: list[int] = []
    for row in rows:
        path = pack / str(row["filename"])
        if not path.is_file() or v1.sha256_file(path) != row["copied_image_sha256"]:
            copied_mismatches.append(int(row["pageid"]))
    check(not copied_mismatches, "EVERY_COPIED_IMAGE_SHA_MATCH", failures, checks)
    image_files = [path for path in (pack / "images").rglob("*") if path.is_file()]
    check(len(image_files) == len(rows), "ONE_IMAGE_FILE_PER_MANIFEST_ROW", failures, checks)

    v1_root = workspace / "datasets" / v1.OUTPUT_NAME
    e0_root = workspace / "datasets" / v1.SOURCE_DATASETS["E0"]
    v1_rows = v1.indexed_by_pageid(v1.load_jsonl(v1_root / "manifest.jsonl"), label="frozen v1")
    e0_rows = v1.indexed_by_pageid(v1.load_jsonl(e0_root / "manifest.jsonl"), label="E0")
    strict_rows = v1.indexed_by_pageid(
        v1.load_jsonl(e0_root / "review" / "ai_final_labels_v1" / "strict_structure_labels.jsonl"),
        label="strict labels",
    )
    provenance_ok = True
    for row in rows:
        pageid = int(row["pageid"])
        if row["v2_origin"] == "INHERITED_FROZEN_V1":
            source_row = v1_rows.get(pageid)
            source_path = v1_root / str(row["filename"])
            if (
                source_row is None
                or v1.source_record_digest(source_row) != row.get("inherited_v1_record_sha256")
                or not source_path.is_file()
                or v1.sha256_file(source_path) != row["copied_image_sha256"]
            ):
                provenance_ok = False
                break
        elif row["v2_origin"] == "E0_MACHINE_VISUAL_SUPPLEMENT":
            source_row = e0_rows.get(pageid)
            strict_row = strict_rows.get(pageid)
            if (
                source_row is None
                or strict_row is None
                or v1.source_record_digest(source_row) != row.get("source_record_sha256")
                or v1.source_record_digest(strict_row) != row.get("source_strict_label_record_sha256")
                or v1.sha256_file(e0_root / str(source_row["filename"])) != row["copied_image_sha256"]
            ):
                provenance_ok = False
                break
        else:
            provenance_ok = False
            break
    check(provenance_ok, "EVERY_ASSET_PROVENANCE_CHAIN_REBUILDS", failures, checks)

    frozen = receipt["frozen_v1"]
    v1_unchanged = (
        v1.tree_sha256(v1_root) == frozen["tree_sha256_after"] == frozen["tree_sha256_before"]
        and v1.sha256_file(v1_root / "manifest.jsonl")
        == frozen["manifest_sha256_after"]
        == frozen["manifest_sha256_before"]
        and v1.sha256_file(v1_root / "SHA256SUMS")
        == frozen["sha256sums_sha256_after"]
        == frozen["sha256sums_sha256_before"]
    )
    check(v1_unchanged, "FROZEN_V1_UNCHANGED", failures, checks)

    human = receipt["formal_human_decisions"]
    human_root = workspace / human["path"]
    journal = human_root / "decision_journal.jsonl"
    human_unchanged = (
        v1.tree_sha256(human_root) == human["tree_sha256_after"] == human["tree_sha256_before"]
        and v1.sha256_file(journal)
        == human["decision_journal_sha256_after"]
        == human["decision_journal_sha256_before"]
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    check(human_unchanged, "FORMAL_E0_HUMAN_DECISIONS_UNCHANGED_AND_EMPTY", failures, checks)

    creators: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        creators[str(row["creator_group"])].add(str(row["experimental_split_suggestion"]))
    creator_leakage = False
    for roles in creators.values():
        normal = roles & {"EXPERIMENTAL_TRAIN_SUGGESTION", "EXPERIMENTAL_VAL_SUGGESTION"}
        held = roles & {"PRINT_DEMO_HOLDOUT_NOT_TRAIN", "CREATOR_GROUP_HOLDOUT_NOT_TRAIN"}
        if len(normal) > 1 or (normal and held):
            creator_leakage = True
            break
    check(not creator_leakage, "NO_CREATOR_GROUP_SPLIT_OR_PRINT_LEAKAGE", failures, checks)
    print_rows = [row for row in rows if row.get("print_holdout_candidate") is True]
    check(
        {int(row["pageid"]) for row in print_rows} == EXPECTED_PRINT_PAGEIDS,
        "EXACT_SIX_FROZEN_PRINT_HOLDOUTS",
        failures,
        checks,
    )
    check(
        all(row["experimental_split_suggestion"] == "PRINT_DEMO_HOLDOUT_NOT_TRAIN" for row in print_rows),
        "PRINT_HOLDOUTS_NEVER_TRAIN_OR_VAL",
        failures,
        checks,
    )
    normal_rows = [
        row
        for row in rows
        if row["experimental_split_suggestion"]
        in {"EXPERIMENTAL_TRAIN_SUGGESTION", "EXPERIMENTAL_VAL_SUGGESTION"}
    ]
    minimum_dhash = min(
        v2.hamming64(str(left["dhash64"]), str(right["dhash64"]))
        for left in print_rows
        for right in normal_rows
    )
    check(
        minimum_dhash == receipt["audit"]["print_to_train_or_val_minimum_dhash64_distance"],
        "PRINT_TO_NORMAL_DHASH_MINIMUM_REBUILDS",
        failures,
        checks,
    )

    role_counts = Counter(str(row["experimental_split_suggestion"]) for row in rows)
    class_counts = Counter(str(row["class_id"]) for row in rows)
    check(
        dict(sorted(role_counts.items())) == receipt["audit"]["experimental_role_counts"],
        "ROLE_COUNTS_REBUILD",
        failures,
        checks,
    )
    check(
        dict(sorted(class_counts.items())) == receipt["audit"]["class_counts"],
        "CLASS_COUNTS_REBUILD",
        failures,
        checks,
    )
    attainment_ok = True
    for class_id, values in receipt["split_target_attainment"].items():
        train_count = sum(
            1
            for row in rows
            if row["class_id"] == class_id
            and row["experimental_split_suggestion"] == "EXPERIMENTAL_TRAIN_SUGGESTION"
        )
        val_count = sum(
            1
            for row in rows
            if row["class_id"] == class_id
            and row["experimental_split_suggestion"] == "EXPERIMENTAL_VAL_SUGGESTION"
        )
        if train_count != values["train_count"] or val_count != values["val_count"]:
            attainment_ok = False
    check(attainment_ok, "SPLIT_ATTAINMENT_REBUILDS", failures, checks)
    check(
        receipt.get("all_split_targets_met") is False
        and receipt["split_target_attainment"]["young_tree"]["train_deficit"] == 3
        and receipt["split_target_attainment"]["young_tree"]["val_deficit"] == 2,
        "YOUNG_TREE_SHORTFALL_EXPLICIT_NOT_HIDDEN",
        failures,
        checks,
    )

    selected_supplemental_young = [
        row
        for row in rows
        if row["v2_origin"] == "E0_MACHINE_VISUAL_SUPPLEMENT" and row["class_id"] == "young_tree"
    ]
    excluded_young = {
        int(row["pageid"])
        for row in decisions
        if row["requested_class"] == "young_tree"
        and row["disposition"] == "EXCLUDE_CONSERVATIVE_MACHINE_VISUAL_GATE_V2"
    }
    check(not selected_supplemental_young, "NO_MATURE_E0_TREE_FORCED_INTO_YOUNG_CLASS", failures, checks)
    check(excluded_young == EXPECTED_EXCLUDED_YOUNG, "ALL_SIX_E0_YOUNG_CANDIDATES_EXCLUDED", failures, checks)
    unresolved = json.loads((pack / "unresolved_requested_ids.json").read_text(encoding="utf-8"))
    check(
        unresolved.get("no_silent_substitution") is True
        and [row["pageid"] for row in unresolved["records"]] == [199434564],
        "MISSING_SUPPLEMENT_ID_EXPLICIT_NO_SUBSTITUTION",
        failures,
        checks,
    )
    contact_files = sorted((pack / "contact_sheets").glob("*.png"))
    check(len(contact_files) == 7 and all(path.stat().st_size > 0 for path in contact_files), "SEVEN_CONTACT_SHEETS_PRESENT", failures, checks)

    return {
        "schema_version": "rootscope.machine_curated_provisional_independent_audit.v2",
        "status": "PASS" if not failures else "FAIL",
        "pack": pack.relative_to(workspace).as_posix(),
        "check_count": len(checks) + len(failures),
        "pass_count": len(checks),
        "failure_count": len(failures),
        "checks_passed": checks,
        "checks_failed": failures,
        "selected_count": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "minimum_print_to_train_or_val_dhash64_distance": minimum_dhash,
        "formal_authority": False,
        "dataset_mutated_by_audit": False,
    }


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--pack", type=Path, default=workspace / "datasets" / v2.OUTPUT_NAME)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit(args.workspace, args.pack)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        output = args.output.resolve(strict=False)
        pack = args.pack.resolve(strict=True)
        try:
            output.relative_to(pack)
        except ValueError:
            pass
        else:
            raise ValueError("audit report must be written outside the audited pack")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
