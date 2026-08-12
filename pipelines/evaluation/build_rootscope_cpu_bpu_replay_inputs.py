#!/usr/bin/env python3
"""Build the explicit 43-image RootScope CPU/BPU replay input set.

The set combines the frozen 23-image r7 conversion replay scope and the 20
operator-labelled laptop/card captures.  It is an integration replay set, not
a new accuracy holdout, and the emitted manifest preserves that boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


SET_ID = "rootscope_cpu_bpu_replay_inputs_43_20260723"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def build(adventurex: Path, output: Path) -> Mapping[str, Any]:
    root = adventurex.resolve(strict=True)
    destination = output.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must stay below AdventureX") from exc
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite input set: {destination}")
    destination.mkdir(parents=True)

    replay_path = (
        root
        / "evidence/rootscope_seed17_bpu_compile_20260717/"
        "r7_horizon_x86_replay.json"
    )
    replay = strict_json(replay_path)
    dataset = root / "datasets/rootscope_machine_curated_provisional_v3"
    replay_rows = replay.get("rows")
    if not isinstance(replay_rows, list) or len(replay_rows) != 23:
        raise ValueError("r7 replay must contain exactly 23 rows")

    capture_root = (
        root
        / "rootscope/evidence/physical_laptop_batch_20260723T131242Z/input/"
        "laptop_card_session_20260723_205230"
    )
    capture_manifest = capture_root / "captures.jsonl"
    capture_rows = [
        json.loads(line)
        for line in capture_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(capture_rows) != 20:
        raise ValueError("physical capture manifest must contain exactly 20 rows")

    records: list[dict[str, Any]] = []

    def copy_bound(
        source: Path,
        target_relative: str,
        expected_sha: str,
        record: dict[str, Any],
    ) -> None:
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"input must be one regular file: {source}")
        actual = sha256_file(source)
        if actual != expected_sha:
            raise ValueError(f"input hash mismatch: {source}")
        target = destination / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"duplicate replay target: {target}")
        shutil.copyfile(source, target)
        if sha256_file(target) != actual:
            raise RuntimeError(f"copied input hash mismatch: {target}")
        records.append(
            {
                "schema": "rootscope.cpu-bpu-replay-input.v1",
                "relative_path": target_relative,
                "sha256": actual,
                "bytes": target.stat().st_size,
                **record,
                "authority": {
                    "execution_authority": False,
                    "physical_authority": False,
                    "serial_write": False,
                    "gpio_access": False,
                    "pump_command": False,
                    "irrigation_execution": False,
                    "physical_completion": False,
                },
            }
        )

    for index, row in enumerate(replay_rows):
        if not isinstance(row, Mapping):
            raise ValueError("r7 replay row must be an object")
        filename = row.get("filename")
        expected = row.get("image_sha256")
        if not isinstance(filename, str) or not isinstance(expected, str):
            raise ValueError("r7 replay row omits filename/hash")
        source = (dataset / filename).resolve(strict=True)
        if dataset.resolve() not in source.parents:
            raise ValueError("r7 replay image leaves dataset root")
        suffix = source.suffix.lower()
        target = f"images/r7_scope/{index:02d}_{source.stem}{suffix}"
        copy_bound(
            source,
            target,
            expected,
            {
                "class_id": row.get("class_id"),
                "role": row.get("role"),
                "source_scope": "R7_23_NON_CALIBRATION_CONVERSION_REPLAY",
                "truth_boundary": (
                    "FROZEN_R7_CONVERSION_REPLAY_NOT_NEW_HOLDOUT_"
                    "NOT_OPEN_WORLD_ACCURACY"
                ),
                "source_top1": row.get("source_top1"),
                "x86_quantized_top1": row.get("quantized_top1"),
            },
        )

    for index, row in enumerate(capture_rows):
        if not isinstance(row, Mapping):
            raise ValueError("capture row must be an object")
        relative = row.get("relative_path")
        expected = row.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("capture row omits path/hash")
        source = (capture_root / relative).resolve(strict=True)
        if capture_root.resolve() not in source.parents:
            raise ValueError("capture image leaves capture root")
        class_id = row.get("class_id")
        normalized = "unknown" if class_id == "non_target" else class_id
        target = f"images/physical20/{index:02d}_{source.name}"
        copy_bound(
            source,
            target,
            expected,
            {
                "class_id": normalized,
                "operator_class_id": class_id,
                "role": "OPERATOR_LABELLED_LAPTOP_CAPTURE_NOT_HOLDOUT",
                "source_scope": "PHYSICAL_PRINT_CARD_DEVELOPMENT_INTEGRATION_20",
                "truth_boundary": "OPERATOR_LABELLED_LAPTOP_CAPTURE_NOT_HOLDOUT",
            },
        )

    manifest_lines = [
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in records
    ]
    manifest_path = destination / "manifest.jsonl"
    write_new(manifest_path, ("\n".join(manifest_lines) + "\n").encode("utf-8"))
    counts: dict[str, int] = {}
    for item in records:
        label = str(item["class_id"])
        counts[label] = counts.get(label, 0) + 1
    receipt = {
        "schema": "rootscope.cpu-bpu-replay-input-set.v1",
        "set_id": SET_ID,
        "status": "PASS_EXPLICIT_HASH_BOUND_INPUT_SET_NOT_ACCURACY_HOLDOUT",
        "record_count": len(records),
        "scope_counts": {
            "r7_conversion_replay": 23,
            "operator_labelled_physical_prints": 20,
        },
        "class_counts": dict(sorted(counts.items())),
        "manifest": {
            "path": "manifest.jsonl",
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "source_bindings": {
            "r7_replay_json_sha256": sha256_file(replay_path),
            "capture_manifest_sha256": sha256_file(capture_manifest),
        },
        "claims": {
            "formal_holdout": False,
            "open_world_accuracy": False,
            "camera_generalization": False,
            "bpu_model_qualification": False,
            "physical_completion": False,
        },
        "authority": {
            "execution_authority": False,
            "physical_authority": False,
            "serial_write": False,
            "gpio_access": False,
            "pump_command": False,
            "irrigation_execution": False,
            "physical_completion": False,
        },
    }
    receipt_path = destination / "input_set_receipt.json"
    write_new(
        receipt_path,
        (
            json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    here = Path(__file__).resolve()
    adventurex = here.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adventurex-root", type=Path, default=adventurex)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            adventurex
            / "rootscope/evidence"
            / SET_ID
        ),
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build(args.adventurex_root, args.output),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
