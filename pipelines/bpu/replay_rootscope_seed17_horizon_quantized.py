#!/usr/bin/env python3
"""Replay the 23 non-calibration RootScope images in Horizon x86 runtime.

This program is intended to run inside the locked OpenExplorer X5 container.
It compares the frozen FP32 ONNX with the mapper float model and the Horizon
quantized ONNX.  It never executes the Bayes-e ``.bin`` and never touches X5,
camera, SSH, network configuration, or irrigation hardware.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps
from horizon_tc_ui import HB_ONNXRuntime


CLASS_ORDER = ("grass_clump", "low_shrub", "young_tree", "unknown")
REPLAY_ROLES = (
    "EXPERIMENTAL_VAL_SUGGESTION",
    "PRINT_DEMO_HOLDOUT_NOT_TRAIN",
    "CREATOR_GROUP_HOLDOUT_NOT_TRAIN",
)
EXPECTED_ROLE_COUNTS = {
    "EXPERIMENTAL_VAL_SUGGESTION": 9,
    "PRINT_DEMO_HOLDOUT_NOT_TRAIN": 6,
    "CREATOR_GROUP_HOLDOUT_NOT_TRAIN": 8,
}
EXPECTED_MODEL_SHA256 = (
    "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "bf14c7423aad965b8af736c7d77cef1ba134d78dd1f905c03cc14cff1192f3fe"
)
EXPECTED_STAGING_RECEIPT_SHA256 = (
    "d61561291dfca23bcc658ca23255b132e910a041c8a9a887d46ccbc02ff05cac"
)
EXPECTED_STAGING_SUMS_SHA256 = (
    "a2d160cd26c48edf4e750af24e51231668661bc24770939566c41c352364de22"
)
STAGING_REL = Path("output/rootscope_bpu_seed17_staging_20260717_r3")
DATASET_REL = Path("datasets/rootscope_machine_curated_provisional_v3")
MODEL_PREFIX = "rootscope_seed17_resnet18_224x224_rgb_ddr"
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
GATES = {
    "top1_agreement_min_count": 23,
    "softmax_l1_mean_max": 0.03,
    "softmax_l1_p99_max": 0.10,
    "softmax_l1_max": 0.15,
    "centered_logit_abs_mean_max": 0.15,
    "centered_logit_abs_p99_max": 0.50,
    "centered_logit_abs_max": 1.00,
    "mapper_float_vs_frozen_max_abs_max": 0.0001,
}


class ReplayError(RuntimeError):
    """A replay contract or gate failed closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayError(f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ReplayError(f"blank JSONL row: {path}:{index}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ReplayError(f"JSONL row is not an object: {path}:{index}")
        result.append(value)
    return result


def safe_child(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ReplayError(f"unsafe relative path: {relative!r}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ReplayError(f"unsafe relative path: {relative!r}")
    resolved = (root / Path(*pure.parts)).resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ReplayError(f"path escapes root: {relative!r}") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise ReplayError(f"path is not a regular file: {relative!r}")
    return resolved


def preprocess_rgb(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with Image.open(path) as opened:
        opened.load()
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    source_short = min(width, height)
    source_long = max(width, height)
    resized_long = int(256 * source_long / source_short)
    if width <= height:
        resized_width, resized_height = 256, resized_long
    else:
        resized_width, resized_height = resized_long, 256
    resized = image.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    crop_top = int(round((resized_height - 224) / 2.0))
    crop_left = int(round((resized_width - 224) / 2.0))
    cropped = resized.crop((crop_left, crop_top, crop_left + 224, crop_top + 224))
    rgb_hwc_u8 = np.asarray(cropped, dtype=np.uint8)
    rgb_nchw_u8 = np.ascontiguousarray(rgb_hwc_u8.transpose(2, 0, 1)[None, ...])
    if rgb_nchw_u8.shape != (1, 3, 224, 224) or rgb_nchw_u8.dtype != np.uint8:
        raise ReplayError("host RGB tensor contract differs")
    # Horizon quantized ONNX exposes INT8 with first preprocess op RGB_128.
    # The real .bin interface remains RGB uint8; this explicit adapter is only
    # for HB_ONNXRuntime x86 replay of the quantized ONNX.
    rgb128_nchw_i8 = np.ascontiguousarray(
        (rgb_nchw_u8.astype(np.int16) - 128).astype(np.int8)
    )
    geometry = {
        "source_size_wh": [width, height],
        "resized_size_wh": [resized_width, resized_height],
        "center_crop_box_ltrb": [crop_left, crop_top, crop_left + 224, crop_top + 224],
    }
    return rgb_nchw_u8, rgb128_nchw_i8, geometry


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)


def higher_quantile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ReplayError("metric vector is empty or non-finite")
    return float(np.quantile(array, quantile, method="higher"))


def metric_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ReplayError("metric vector is empty or non-finite")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p99_higher": higher_quantile(array, 0.99),
        "max": float(np.max(array)),
    }


def session_contract(session: HB_ONNXRuntime) -> dict[str, Any]:
    return {
        "input_names": list(session.input_names),
        "output_names": list(session.output_names),
        "input_shapes": [list(shape) for shape in session.input_shapes],
        "output_shapes": [list(shape) for shape in session.output_shapes],
        "input_types": list(session.input_types),
        "layout": list(session.layout),
    }


def run_replay(
    workspace: Path, quantized_model: Path | None = None
) -> dict[str, Any]:
    workspace = workspace.resolve(strict=True)
    staging = (workspace / STAGING_REL).resolve(strict=True)
    dataset = (workspace / DATASET_REL).resolve(strict=True)
    manifest_path = dataset / "manifest.jsonl"
    receipt_path = staging / "staging_receipt.json"
    sums_path = staging / "STAGING_SHA256SUMS"
    if sha256_file(manifest_path) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ReplayError("frozen dataset manifest hash differs")
    if sha256_file(receipt_path) != EXPECTED_STAGING_RECEIPT_SHA256:
        raise ReplayError("r3 staging receipt hash differs")
    if sha256_file(sums_path) != EXPECTED_STAGING_SUMS_SHA256:
        raise ReplayError("r3 staging input sums hash differs")
    receipt = load_json(receipt_path)
    if receipt.get("model", {}).get("sha256") != EXPECTED_MODEL_SHA256:
        raise ReplayError("staging receipt model hash differs")
    if receipt.get("model", {}).get("class_order") != list(CLASS_ORDER):
        raise ReplayError("class order differs")
    for key in ("bpu_compiled", "x5_ready", "model_qualified"):
        if receipt.get("formal_flags", {}).get(key) is not False:
            raise ReplayError(f"staging receipt must remain false for {key}")

    rows = load_jsonl(manifest_path)
    replay_rows = [row for row in rows if row.get("experimental_split_suggestion") in REPLAY_ROLES]
    role_counts = Counter(row.get("experimental_split_suggestion") for row in replay_rows)
    if dict(role_counts) != EXPECTED_ROLE_COUNTS or len(replay_rows) != 23:
        raise ReplayError(f"unexpected replay role counts: {dict(role_counts)}")
    calibration_rows = load_jsonl(staging / "calibration_manifest.jsonl")
    calibration_source_hashes = {row.get("source_sha256") for row in calibration_rows}
    replay_hashes = {row.get("copied_image_sha256") for row in replay_rows}
    if None in replay_hashes or replay_hashes & calibration_source_hashes:
        raise ReplayError("replay set overlaps calibration source hashes")

    frozen_path = staging / "model/rootscope_seed17_resnet18_static_b1x3x224x224_opset11.onnx"
    mapper_float_path = staging / f"model_output/{MODEL_PREFIX}_original_float_model.onnx"
    if quantized_model is None:
        quantized_path = staging / f"model_output/{MODEL_PREFIX}_quantized_model.onnx"
    else:
        candidate = quantized_model
        if not candidate.is_absolute():
            candidate = workspace / candidate
        quantized_path = candidate.resolve(strict=True)
        try:
            quantized_path.relative_to(workspace)
        except ValueError as error:
            raise ReplayError("quantized model must remain inside AdventureX") from error
    for path in (frozen_path, mapper_float_path, quantized_path):
        if not path.is_file():
            raise ReplayError(f"missing replay model: {path}")
    if sha256_file(frozen_path) != EXPECTED_MODEL_SHA256:
        raise ReplayError("frozen source ONNX changed")

    frozen_session = HB_ONNXRuntime(model_file=str(frozen_path))
    mapper_float_session = HB_ONNXRuntime(model_file=str(mapper_float_path))
    quantized_session = HB_ONNXRuntime(model_file=str(quantized_path))
    contracts = {
        "frozen_source": session_contract(frozen_session),
        "mapper_float": session_contract(mapper_float_session),
        "horizon_quantized": session_contract(quantized_session),
    }
    if contracts["frozen_source"] != {
        "input_names": ["image"],
        "output_names": ["logits"],
        # HB_ONNXRuntime exposes a replay-only dynamic batch even though the
        # frozen ONNX protobuf itself remains static batch 1.
        "input_shapes": [["?", 3, 224, 224]],
        "output_shapes": [[1, 4]],
        "input_types": [1],
        "layout": ["NCHW"],
    }:
        raise ReplayError(f"frozen source runtime contract differs: {contracts['frozen_source']}")
    expected_quantized = {
        "input_names": ["image"],
        "output_names": ["logits"],
        "input_shapes": [["?", 3, 224, 224]],
        "output_shapes": [[1, 4]],
        "input_types": [3],
        "layout": ["NCHW"],
    }
    if contracts["horizon_quantized"] != expected_quantized:
        raise ReplayError(f"quantized runtime contract differs: {contracts['horizon_quantized']}")

    result_rows: list[dict[str, Any]] = []
    softmax_l1_values: list[float] = []
    centered_logit_abs_values: list[float] = []
    mapper_float_abs_values: list[float] = []
    top1_agreement = 0
    source_correct = 0
    quantized_correct = 0
    for row in replay_rows:
        image_path = safe_child(dataset, row.get("filename"))
        copied_sha = sha256_file(image_path)
        if copied_sha != row.get("copied_image_sha256"):
            raise ReplayError(f"image hash differs: {row.get('filename')}")
        rgb_u8, rgb128_i8, geometry = preprocess_rgb(image_path)
        source_input = (
            rgb_u8.astype(np.float32) / np.float32(255.0) - MEAN
        ) / STD
        mapper_float_input = rgb_u8.astype(np.float32)
        source_logits = np.asarray(
            frozen_session.run(["logits"], {"image": source_input})[0], dtype=np.float64
        ).reshape(-1)
        mapper_float_logits = np.asarray(
            mapper_float_session.run(["logits"], {"image": mapper_float_input})[0],
            dtype=np.float64,
        ).reshape(-1)
        quantized_logits = np.asarray(
            quantized_session.run(["logits"], {"image": rgb128_i8})[0], dtype=np.float64
        ).reshape(-1)
        if source_logits.shape != (4,) or mapper_float_logits.shape != (4,) or quantized_logits.shape != (4,):
            raise ReplayError("logit shape differs from class order")
        if not all(np.all(np.isfinite(values)) for values in (source_logits, mapper_float_logits, quantized_logits)):
            raise ReplayError("non-finite logits")
        source_prob = softmax(source_logits)
        quantized_prob = softmax(quantized_logits)
        softmax_l1 = float(np.sum(np.abs(source_prob - quantized_prob)))
        centered_source = source_logits - float(np.mean(source_logits))
        centered_quantized = quantized_logits - float(np.mean(quantized_logits))
        centered_abs = np.abs(centered_source - centered_quantized)
        mapper_float_abs = np.abs(source_logits - mapper_float_logits)
        softmax_l1_values.append(softmax_l1)
        centered_logit_abs_values.extend(float(value) for value in centered_abs)
        mapper_float_abs_values.extend(float(value) for value in mapper_float_abs)
        source_top1 = int(np.argmax(source_logits))
        quantized_top1 = int(np.argmax(quantized_logits))
        agreed = source_top1 == quantized_top1
        top1_agreement += int(agreed)
        truth_index = CLASS_ORDER.index(str(row.get("class_id")))
        source_correct += int(source_top1 == truth_index)
        quantized_correct += int(quantized_top1 == truth_index)
        result_rows.append(
            {
                "class_id": row.get("class_id"),
                "role": row.get("experimental_split_suggestion"),
                "filename": row.get("filename"),
                "image_sha256": copied_sha,
                "geometry": geometry,
                "runtime_rgb_uint8_nchw_sha256": sha256_array(rgb_u8),
                "x86_rgb128_int8_nchw_sha256": sha256_array(rgb128_i8),
                "source_logits": [float(value) for value in source_logits],
                "mapper_float_logits": [float(value) for value in mapper_float_logits],
                "quantized_logits": [float(value) for value in quantized_logits],
                "source_top1": CLASS_ORDER[source_top1],
                "quantized_top1": CLASS_ORDER[quantized_top1],
                "top1_agreement": agreed,
                "source_truth_correct": source_top1 == truth_index,
                "quantized_truth_correct": quantized_top1 == truth_index,
                "softmax_l1": softmax_l1,
                "centered_logit_abs_max": float(np.max(centered_abs)),
                "mapper_float_vs_frozen_max_abs": float(np.max(mapper_float_abs)),
            }
        )

    softmax_summary = metric_summary(softmax_l1_values)
    centered_summary = metric_summary(centered_logit_abs_values)
    mapper_float_summary = metric_summary(mapper_float_abs_values)
    gate_results = {
        "top1_agreement_23_of_23": top1_agreement >= GATES["top1_agreement_min_count"],
        "softmax_l1_mean": softmax_summary["mean"] <= GATES["softmax_l1_mean_max"],
        "softmax_l1_p99": softmax_summary["p99_higher"] <= GATES["softmax_l1_p99_max"],
        "softmax_l1_max": softmax_summary["max"] <= GATES["softmax_l1_max"],
        "centered_logit_abs_mean": centered_summary["mean"] <= GATES["centered_logit_abs_mean_max"],
        "centered_logit_abs_p99": centered_summary["p99_higher"] <= GATES["centered_logit_abs_p99_max"],
        "centered_logit_abs_max": centered_summary["max"] <= GATES["centered_logit_abs_max"],
        "mapper_float_vs_frozen_max_abs": mapper_float_summary["max"]
        <= GATES["mapper_float_vs_frozen_max_abs_max"],
    }
    all_gates_passed = all(gate_results.values())
    return {
        "schema_version": "rootscope.seed17.bayes_e.horizon_x86_replay.v1",
        "status": (
            "PASS_HORIZON_X86_QUANTIZED_ONNX_REPLAY_NOT_X5"
            if all_gates_passed
            else "FAIL_HORIZON_X86_QUANTIZED_ONNX_REPLAY_GATES_NOT_X5"
        ),
        "created_date": "2026-07-17",
        "scope": "23_NON_CALIBRATION_IMAGES_VAL9_PRINT6_CREATOR8",
        "class_order": list(CLASS_ORDER),
        "role_counts": dict(role_counts),
        "artifact_bindings": {
            "dataset_manifest": {"path": (DATASET_REL / "manifest.jsonl").as_posix(), "sha256": sha256_file(manifest_path)},
            "staging_receipt": {"path": (STAGING_REL / "staging_receipt.json").as_posix(), "sha256": sha256_file(receipt_path)},
            "staging_input_sums": {"path": (STAGING_REL / "STAGING_SHA256SUMS").as_posix(), "sha256": sha256_file(sums_path)},
            "frozen_source_onnx": {"path": frozen_path.relative_to(workspace).as_posix(), "sha256": sha256_file(frozen_path)},
            "mapper_float_onnx": {"path": mapper_float_path.relative_to(workspace).as_posix(), "sha256": sha256_file(mapper_float_path)},
            "horizon_quantized_onnx": {"path": quantized_path.relative_to(workspace).as_posix(), "sha256": sha256_file(quantized_path)},
        },
        "runtime_contract": {
            "bin_interface": "CONTIGUOUS_UINT8_RGB_NCHW_1x3x224x224_DDR",
            "host_preprocess": ["short_side_resize_256_pil_bilinear", "center_crop_224", "BGR_OR_FILE_TO_RGB", "HWC_TO_NCHW_WITH_BATCH"],
            "host_normalization": False,
            "mean_scale_owner": "HORIZON_MODEL_PREPROCESS",
            "x86_quantized_onnx_adapter": "uint8_rgb_minus_128_to_int8_RGB_128",
            "sessions": contracts,
        },
        "metrics": {
            "top1_agreement_count": top1_agreement,
            "top1_agreement_total": len(result_rows),
            "source_truth_correct": source_correct,
            "quantized_truth_correct": quantized_correct,
            "softmax_l1_per_image": softmax_summary,
            "centered_logit_abs_per_class_value": centered_summary,
            "mapper_float_vs_frozen_abs_per_class_value": mapper_float_summary,
        },
        "gates": GATES,
        "gate_results": gate_results,
        "rows": result_rows,
        "formal_flags": {
            "x86_horizon_quantized_replay_passed": all_gates_passed,
            "bpu_bin_executed": False,
            "x5_replay_executed": False,
            "x5_ready": False,
            "camera_replay_executed": False,
            "model_candidate": False,
            "model_qualified": False,
            "human_reviewed": False,
            "rights_approved": False,
            "data_locked": False,
            "execution_authority": False,
            "irrigation_authority": False,
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/adventurex"))
    parser.add_argument(
        "--quantized-model",
        type=Path,
        default=None,
        help="Optional candidate quantized ONNX, absolute or relative to --workspace.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    output = args.out.resolve(strict=False)
    try:
        output.relative_to(workspace)
    except ValueError as error:
        raise ReplayError("output must remain inside AdventureX") from error
    report = run_replay(workspace, args.quantized_model)
    write_json(output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "output_sha256": sha256_file(output),
        "metrics": report["metrics"],
        "gate_results": report["gate_results"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["formal_flags"]["x86_horizon_quantized_replay_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
