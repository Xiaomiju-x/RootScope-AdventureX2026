#!/usr/bin/env python3
"""Static same-image CPU/BPU replay for the RootScope competition runtime.

This tool opens only explicitly listed image files.  It uses the core CPU ONNX
runner as audit/fallback and the local AF_UNIX r7 worker as an optional BPU
shadow proposal.  It never opens a camera, network socket, serial port, GPIO,
pump, or state machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.competition_runtime.bpu_shadow_client import BpuShadowClient
from app.competition_runtime.bpu_shadow_protocol import (
    MAX_BATCH,
    R7_REFERENCE_SHA256,
    ZERO_AUTHORITY,
)
from app.competition_runtime.plant_cpu_bpu_replay import PlantCpuBpuReplay
from app.vision.dual_path_demo import build_seed17_runner_from_capsule

CPU_CAPSULE_SHA256 = (
    "1e839948e466895e0b416aababb39402380463f9ce125c8aeb59f4087191cb97"
)
CPU_MODEL_SHA256 = (
    "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_file(path: Path, expected_sha256: str, label: str) -> Path:
    configured = path.expanduser()
    if configured.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = configured.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: actual={actual} expected={expected_sha256}"
        )
    return resolved


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load canonical JSONL or a bounded JSON manifest representation."""

    text = path.read_text(encoding="utf-8")
    candidates: list[Any]
    if path.suffix.lower() == ".jsonl":
        candidates = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"manifest line {line_number} is invalid JSON"
                ) from exc
    else:
        try:
            root = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("JSON manifest is invalid") from exc
        if isinstance(root, list):
            candidates = root
        elif isinstance(root, dict):
            selected = root.get("selected_representatives")
            if isinstance(selected, dict):
                candidates = []
                for label, item in selected.items():
                    if not isinstance(item, dict):
                        raise ValueError(
                            "selected_representatives values must be objects"
                        )
                    normalized = dict(item)
                    normalized.setdefault("class_id", label)
                    normalized.setdefault(
                        "truth_boundary", root.get("truth_boundary")
                    )
                    candidates.append(normalized)
            else:
                candidates = []
                for key in ("records", "items", "images", "captures"):
                    value = root.get(key)
                    if isinstance(value, list):
                        candidates = value
                        break
                if not candidates:
                    raise ValueError(
                        "JSON manifest must be an array, contain records/items/"
                        "images/captures, or contain selected_representatives"
                    )
        else:
            raise ValueError("JSON manifest root must be an object or array")

    records: list[dict[str, Any]] = []
    for index, value in enumerate(candidates, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"manifest record {index} must be an object")
        relative_path = value.get("relative_path")
        expected_sha = value.get("sha256")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"manifest record {index} omits relative_path")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
        ):
            raise ValueError(f"manifest record {index} has invalid sha256")
        records.append(value)
    if not records:
        raise ValueError("manifest has no image records")
    return records


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    manifest_group = parser.add_mutually_exclusive_group(required=True)
    manifest_group.add_argument(
        "--manifest",
        type=Path,
        help="explicit .jsonl or .json manifest",
    )
    manifest_group.add_argument(
        "--manifest-jsonl",
        type=Path,
        help="compatibility alias for an explicit JSONL manifest",
    )
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--cpu-model", required=True, type=Path)
    parser.add_argument("--bpu-socket", required=True, type=Path)
    parser.add_argument(
        "--expected-bpu-model-sha256",
        default=R7_REFERENCE_SHA256,
    )
    parser.add_argument("--bpu-timeout-s", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.batch_size <= MAX_BATCH:
        raise ValueError(f"--batch-size must be within 1-{MAX_BATCH}")
    input_root = args.input_root.expanduser().resolve(strict=True)
    manifest_argument = args.manifest or args.manifest_jsonl
    manifest = manifest_argument.expanduser().resolve(strict=True)
    capsule = bind_file(args.capsule, CPU_CAPSULE_SHA256, "CPU capsule")
    cpu_model = bind_file(args.cpu_model, CPU_MODEL_SHA256, "CPU model")
    records = load_manifest(manifest)
    runner = build_seed17_runner_from_capsule(capsule, model_path=cpu_model)
    if list(runner.providers) != ["CPUExecutionProvider"]:
        raise RuntimeError(f"CPU provider contract changed: {runner.providers}")
    client = BpuShadowClient(
        args.bpu_socket,
        expected_model_sha256=args.expected_bpu_model_sha256,
        timeout_s=args.bpu_timeout_s,
    )
    replay = PlantCpuBpuReplay(runner, client)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists() or args.summary_json.exists():
        raise FileExistsError("output paths must be new")
    output_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    started = time.perf_counter()
    with args.output_jsonl.open("x", encoding="utf-8", newline="\n") as output:
        for batch_start in range(0, len(records), args.batch_size):
            source_batch = records[batch_start : batch_start + args.batch_size]
            images: list[np.ndarray] = []
            bound_sources: list[dict[str, Any]] = []
            for local_index, source in enumerate(source_batch):
                relative = Path(source["relative_path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("manifest image path must stay below input root")
                image_path = (input_root / relative).resolve(strict=True)
                if input_root != image_path and input_root not in image_path.parents:
                    raise ValueError("manifest image path escapes input root")
                actual_sha = sha256_file(image_path)
                if actual_sha != source["sha256"]:
                    raise ValueError(
                        f"image SHA-256 mismatch for {source['relative_path']}"
                    )
                with Image.open(image_path) as opened:
                    rgb = np.asarray(
                        ImageOps.exif_transpose(opened).convert("RGB"),
                        dtype=np.uint8,
                    )
                images.append(rgb)
                bound_sources.append(
                    {
                        "global_index": batch_start + local_index,
                        "relative_path": source["relative_path"],
                        "file_sha256": actual_sha,
                        "file_bytes": image_path.stat().st_size,
                        "width": int(rgb.shape[1]),
                        "height": int(rgb.shape[0]),
                        "label": source.get("class_id"),
                        "truth_boundary": source.get("truth_boundary"),
                    }
                )

            receipt = replay.infer_rgb_batch(images)
            counters[f"batch_{receipt['bpu_client_status']}"] += 1
            for source, replay_row in zip(
                bound_sources, receipt["rows"], strict=True
            ):
                row = {
                    "schema": "rootscope.static-cpu-bpu-replay-row.v1",
                    "timestamp_utc": utc_now(),
                    **source,
                    **replay_row,
                    "bpu_client_status": receipt["bpu_client_status"],
                    "bpu_backend_actual": receipt["bpu_backend_actual"],
                    "bpu_backend_metadata": receipt["bpu_backend_metadata"],
                    "bpu_qualification": "SHADOW_CANDIDATE_NOT_DEFAULT",
                    "selected_bin_changed": False,
                    "zero_authority": True,
                    "authority": dict(ZERO_AUTHORITY),
                }
                output.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                output.flush()
                output_rows.append(row)
                counters["images"] += 1
                if row["bpu_proposal"]["available"]:
                    counters["bpu_shadow_proposals"] += 1
                    if row["cpu_bpu_top1_agreement"] is True:
                        counters["cpu_bpu_top1_agree"] += 1
                    else:
                        counters["cpu_bpu_top1_disagree"] += 1
                else:
                    counters["cpu_fallback_rows"] += 1
        os.fsync(output.fileno())

    summary = {
        "schema": "rootscope.static-cpu-bpu-replay-summary.v1",
        "completed_at_utc": utc_now(),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "inputs": {
            "input_root": str(input_root),
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "record_count": len(records),
        },
        "cpu": {
            "model_sha256": CPU_MODEL_SHA256,
            "provider": "CPUExecutionProvider",
            "role": "AUDIT_AND_FALLBACK_PRIMARY",
        },
        "bpu": {
            "expected_model_sha256": args.expected_bpu_model_sha256,
            "transport": "AF_UNIX",
            "role": "SHADOW_PROPOSAL_ONLY",
            "qualification": "SHADOW_CANDIDATE_NOT_DEFAULT",
            "selected_bin_changed": False,
        },
        "counters": dict(counters),
        "output_jsonl": {
            "path": str(args.output_jsonl.resolve()),
            "sha256": sha256_file(args.output_jsonl),
            "rows": len(output_rows),
        },
        "shadow_blocks_primary_display": False,
        "zero_authority": True,
        "authority": dict(ZERO_AUTHORITY),
    }
    _atomic_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
