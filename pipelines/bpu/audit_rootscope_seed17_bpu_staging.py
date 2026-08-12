#!/usr/bin/env python3
"""Independent, fail-closed audit of the RootScope seed-17 BPU staging pack.

The auditor deliberately does not import the staging builder.  It independently
rebuilds dataset partition membership, deterministic calibration tensors,
mapper normalization bindings, ONNX metadata, config semantics, and complete
file hashes.  It never invokes Docker, WSL, hb_mapper, SSH, or a device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import onnx
import yaml
from PIL import Image, ImageEnhance, ImageOps


WORKSPACE = Path(__file__).resolve().parents[2]
XRD_ROOT = WORKSPACE.parent
DEFAULT_STAGING = WORKSPACE / "output/rootscope_bpu_seed17_staging_20260717_r3"
PACK_REL = Path("datasets/rootscope_machine_curated_provisional_v3")
RUN_REL = Path(
    "output/rootscope_machine_curated_experimental_runs/"
    "v3_rtx4050_multiseed_20260717_r1"
)
SOURCE_ONNX_REL = Path("seed_00017/model_static_b1x3x224x224_opset11.onnx")
GENERATOR_REL = Path("tools/bpu/prepare_rootscope_seed17_bpu_staging.py")
EXPECTED_PACK_MANIFEST_SHA256 = (
    "bf14c7423aad965b8af736c7d77cef1ba134d78dd1f905c03cc14cff1192f3fe"
)
EXPECTED_RUN_RECEIPT_SHA256 = (
    "6eb4c07f175e97a4c3941b43847bc739f97a2a5788420d1fea8f494334d7d526"
)
EXPECTED_ONNX_SHA256 = (
    "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad"
)
TRAIN_ROLE = "EXPERIMENTAL_TRAIN_SUGGESTION"
CLASS_ORDER = ("grass_clump", "low_shrub", "young_tree", "unknown")
EXPECTED_TRAIN_CLASS_COUNTS = {
    "grass_clump": 8,
    "low_shrub": 13,
    "young_tree": 5,
    "unknown": 29,
}
EXPECTED_TRAIN_SOURCE_COUNT = sum(EXPECTED_TRAIN_CLASS_COUNTS.values())
SAMPLES_PER_CLASS = 64
EXPECTED_SAMPLE_COUNT = SAMPLES_PER_CLASS * len(CLASS_ORDER)
INPUT_SHAPE = [1, 3, 224, 224]
STORED_SHAPE = [3, 224, 224]
CALIBRATION_FILE_BYTES = int(np.prod(STORED_SHAPE)) * np.dtype(np.float32).itemsize
RAW_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
MAPPER_SCALE = np.asarray(
    [0.01712475, 0.017507, 0.01742919], dtype=np.float32
).reshape(3, 1, 1)
UNIT_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
UNIT_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
STATUS = "BPU_OFFLINE_STAGING_ONLY_NOT_COMPILED_NOT_X5_READY"
SCHEMA = "rootscope.seed17.bayes_e.offline_staging_receipt.v1"
TOOLCHAIN_IMAGE = "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310"
SUM_RE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")

VARIANTS: tuple[dict[str, Any], ...] = (
    {"id": "identity", "horizontal_flip": False, "brightness": 1.0, "contrast": 1.0, "saturation": 1.0},
    {"id": "hflip", "horizontal_flip": True, "brightness": 1.0, "contrast": 1.0, "saturation": 1.0},
    {"id": "brightness_090", "horizontal_flip": False, "brightness": 0.90, "contrast": 1.0, "saturation": 1.0},
    {"id": "brightness_110", "horizontal_flip": False, "brightness": 1.10, "contrast": 1.0, "saturation": 1.0},
    {"id": "contrast_090", "horizontal_flip": False, "brightness": 1.0, "contrast": 0.90, "saturation": 1.0},
    {"id": "contrast_110", "horizontal_flip": False, "brightness": 1.0, "contrast": 1.10, "saturation": 1.0},
    {"id": "saturation_090", "horizontal_flip": False, "brightness": 1.0, "contrast": 1.0, "saturation": 0.90},
    {"id": "saturation_110", "horizontal_flip": False, "brightness": 1.0, "contrast": 1.0, "saturation": 1.10},
    {"id": "hflip_brightness_095", "horizontal_flip": True, "brightness": 0.95, "contrast": 1.0, "saturation": 1.0},
    {"id": "hflip_brightness_105", "horizontal_flip": True, "brightness": 1.05, "contrast": 1.0, "saturation": 1.0},
    {"id": "brightness_095_contrast_105", "horizontal_flip": False, "brightness": 0.95, "contrast": 1.05, "saturation": 1.0},
    {"id": "brightness_105_contrast_095", "horizontal_flip": False, "brightness": 1.05, "contrast": 0.95, "saturation": 1.0},
    {"id": "hflip_saturation_095", "horizontal_flip": True, "brightness": 1.0, "contrast": 1.0, "saturation": 0.95},
)

XRD_REFERENCE_PATHS = {
    "local_d_robotics_classification_guide": Path(
        "tools/bpu_transformer/rdk_model_zoo/samples/vision/classification/"
        "Model quantization deployment.md"
    ),
    "local_d_robotics_resnext_bayes_e_yaml": Path(
        "tools/bpu_transformer/rdk_model_zoo/samples/vision/classification/"
        "ResNeXt/yaml/ResNeXt50_32x4d_config.yaml"
    ),
    "xrd_successful_bayes_e_compiler_convention": Path(
        "deploy/ai_brain_x5/bpu_ipop_future_compact_eq_r3/config_bpu.yaml"
    ),
}


class AuditError(RuntimeError):
    """The independent staging audit failed closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise AuditError(f"cannot read JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise AuditError(f"blank JSONL row: {path}:{line_number}")
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise AuditError(f"invalid JSONL row: {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise AuditError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise AuditError(f"empty JSONL: {path}")
    return rows


def canonical_relative(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise AuditError(f"{location} must be a non-empty canonical POSIX path")
    posix = PurePosixPath(value)
    if posix.is_absolute() or value != posix.as_posix() or any(
        part in {"", ".", ".."} for part in posix.parts
    ):
        raise AuditError(f"unsafe path at {location}: {value!r}")
    return value


def safe_child(root: Path, relative: Any, *, location: str) -> Path:
    value = canonical_relative(relative, location=location)
    candidate = (root / Path(*PurePosixPath(value).parts)).resolve(strict=True)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise AuditError(f"path escapes root at {location}: {value}") from error
    if candidate.is_symlink() or not candidate.is_file():
        raise AuditError(f"path is not a regular file at {location}: {value}")
    return candidate


def parse_sums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = SUM_RE.fullmatch(line)
        if match is None:
            raise AuditError(f"invalid STAGING_SHA256SUMS row {line_number}")
        digest, relative = match.groups()
        relative = canonical_relative(relative, location=f"STAGING_SHA256SUMS:{line_number}")
        if relative == "STAGING_SHA256SUMS" or relative in rows:
            raise AuditError(f"invalid or duplicate sums path: {relative}")
        rows[relative] = digest
    if not rows:
        raise AuditError("STAGING_SHA256SUMS is empty")
    return rows


def audit_full_hash_coverage(staging: Path) -> dict[str, str]:
    sums_path = staging / "STAGING_SHA256SUMS"
    if not sums_path.is_file() or sums_path.is_symlink():
        raise AuditError("missing regular STAGING_SHA256SUMS")
    sums = parse_sums(sums_path)
    actual: set[str] = set()
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise AuditError(f"symlink is not permitted in staging pack: {path}")
        if path.is_file() and path.name != "STAGING_SHA256SUMS":
            actual.add(path.relative_to(staging).as_posix())
    if set(sums) != actual:
        raise AuditError(
            f"STAGING_SHA256SUMS is not full coverage: "
            f"uncovered={sorted(actual-set(sums))}, stale={sorted(set(sums)-actual)}"
        )
    for relative, expected in sums.items():
        target = safe_child(staging, relative, location=f"sums[{relative}]")
        if sha256_file(target) != expected:
            raise AuditError(f"SHA mismatch: {relative}")
    return sums


def apply_variant(image: Image.Image, spec: Mapping[str, Any]) -> Image.Image:
    result = image.convert("RGB")
    if spec["horizontal_flip"]:
        result = ImageOps.mirror(result)
    if spec["brightness"] != 1.0:
        result = ImageEnhance.Brightness(result).enhance(float(spec["brightness"]))
    if spec["contrast"] != 1.0:
        result = ImageEnhance.Contrast(result).enhance(float(spec["contrast"]))
    if spec["saturation"] != 1.0:
        result = ImageEnhance.Color(result).enhance(float(spec["saturation"]))
    return result


def preprocess_independent(image: Image.Image) -> tuple[np.ndarray, dict[str, Any]]:
    width, height = image.size
    if width <= height:
        new_width = 256
        new_height = int(256 * height / width)
    else:
        new_height = 256
        new_width = int(256 * width / height)
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    left = int(round((new_width - 224) / 2.0))
    top = int(round((new_height - 224) / 2.0))
    box = [left, top, left + 224, top + 224]
    tensor = np.asarray(resized.crop(tuple(box)).convert("RGB"), dtype=np.float32)
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1))
    return tensor, {
        "source_size_wh": [width, height],
        "resized_size_wh": [new_width, new_height],
        "center_crop_box_ltrb": box,
    }


def normalized_bindings_independent(raw: np.ndarray) -> dict[str, Any]:
    mapper = np.asarray((raw - RAW_MEAN) * MAPPER_SCALE, dtype=np.float32)
    training = np.asarray((raw / np.float32(255.0) - UNIT_MEAN) / UNIT_STD, dtype=np.float32)
    return {
        "mapper_formula_sha256": sha256_bytes(mapper.tobytes(order="C")),
        "training_formula_sha256": sha256_bytes(training.tobytes(order="C")),
        "mapper_vs_training_max_abs_delta": float(np.max(np.abs(mapper - training))),
        "normalized_min": float(mapper.min()),
        "normalized_max": float(mapper.max()),
    }


def require_false(mapping: Mapping[str, Any], keys: tuple[str, ...], *, location: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise AuditError(f"{location}.{key} must be exactly false")


def inspect_onnx(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)

    def contract(value: Any) -> dict[str, Any]:
        tensor = value.type.tensor_type
        dims: list[Any] = []
        for dim in tensor.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(str(dim.dim_param))
            else:
                dims.append(None)
        return {"name": value.name, "shape": dims, "elem_type": int(tensor.elem_type)}

    inputs = [contract(item) for item in model.graph.input]
    outputs = [contract(item) for item in model.graph.output]
    opsets = [{"domain": item.domain, "version": int(item.version)} for item in model.opset_import]
    operators = dict(sorted(Counter(node.op_type for node in model.graph.node).items()))
    if inputs != [{"name": "image", "shape": INPUT_SHAPE, "elem_type": 1}]:
        raise AuditError(f"ONNX input contract differs: {inputs}")
    if outputs != [{"name": "logits", "shape": [1, 4], "elem_type": 1}]:
        raise AuditError(f"ONNX output contract differs: {outputs}")
    if opsets != [{"domain": "", "version": 11}]:
        raise AuditError(f"ONNX opset differs: {opsets}")
    if set(operators) != {"Add", "AveragePool", "Conv", "Flatten", "Gemm", "MaxPool", "Relu"}:
        raise AuditError(f"unexpected ONNX operator set: {sorted(operators)}")
    return {
        "sha256": sha256_file(path),
        "ir_version": int(model.ir_version),
        "inputs": inputs,
        "outputs": outputs,
        "opsets": opsets,
        "node_count": len(model.graph.node),
        "operator_counts": operators,
    }


def audit_config(staging: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    config_path = safe_child(staging, receipt["mapper"]["config_path"], location="mapper.config_path")
    if sha256_file(config_path) != receipt["mapper"]["config_sha256"]:
        raise AuditError("mapper config hash does not match receipt")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise AuditError("mapper YAML root is not an object")
    model = config.get("model_parameters", {})
    inputs = config.get("input_parameters", {})
    calibration = config.get("calibration_parameters", {})
    compiler = config.get("compiler_parameters", {})
    expected_model = {
        "onnx_model": "../model/rootscope_seed17_resnet18_static_b1x3x224x224_opset11.onnx",
        "march": "bayes-e",
        "working_dir": "../model_output",
        "output_model_file_prefix": "rootscope_seed17_resnet18_224x224_rgb_ddr",
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise AuditError(f"mapper YAML model_parameters.{key} differs")
    expected_inputs = {
        "input_name": "image",
        "input_type_rt": "rgb",
        "input_layout_rt": "NCHW",
        "input_type_train": "rgb",
        "input_layout_train": "NCHW",
        "input_shape": "1x3x224x224",
        "norm_type": "data_mean_and_scale",
        "mean_value": "123.675 116.28 103.53",
        "scale_value": "0.01712475 0.017507 0.01742919",
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise AuditError(f"mapper YAML input_parameters.{key} differs")
    expected_calibration = {
        "cal_data_dir": "../calibration_data_rgb_f32",
        "cal_data_type": "float32",
        "calibration_type": "default",
    }
    for key, expected in expected_calibration.items():
        if calibration.get(key) != expected:
            raise AuditError(f"mapper YAML calibration_parameters.{key} differs")
    expected_compiler = {
        "compile_mode": "latency",
        "debug": False,
        "optimize_level": "O3",
        "core_num": 1,
        "input_source": {"image": "ddr"},
    }
    for key, expected in expected_compiler.items():
        if compiler.get(key) != expected:
            raise AuditError(f"mapper YAML compiler_parameters.{key} differs")
    return {
        "march": model["march"],
        "runtime_input_type": inputs["input_type_rt"],
        "runtime_layout": inputs["input_layout_rt"],
        "train_input_type": inputs["input_type_train"],
        "train_layout": inputs["input_layout_train"],
        "calibration_type": calibration["calibration_type"],
        "compile_mode": compiler["compile_mode"],
        "optimize_level": compiler["optimize_level"],
        "core_num": compiler["core_num"],
        "input_source": compiler["input_source"],
    }


def audit_calibration(staging: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    pack = (WORKSPACE / PACK_REL).resolve(strict=True)
    pack_manifest = pack / "manifest.jsonl"
    pack_receipt_path = pack / "receipt.json"
    if sha256_file(pack_manifest) != EXPECTED_PACK_MANIFEST_SHA256:
        raise AuditError("current frozen v3 manifest hash differs")
    if receipt["source_dataset"]["manifest_sha256"] != EXPECTED_PACK_MANIFEST_SHA256:
        raise AuditError("staging receipt does not bind frozen v3 manifest")
    if receipt["source_dataset"]["receipt_sha256"] != sha256_file(pack_receipt_path):
        raise AuditError("staging receipt does not bind frozen v3 receipt")
    pack_receipt = load_json(pack_receipt_path)
    require_false(
        pack_receipt,
        ("human_reviewed", "rights_approved", "data_locked", "training_eligible"),
        location="frozen_v3_receipt",
    )

    rows = load_jsonl(pack_manifest)
    train = [row for row in rows if row.get("experimental_split_suggestion") == TRAIN_ROLE]
    other = [row for row in rows if row.get("experimental_split_suggestion") != TRAIN_ROLE]
    if len(train) != EXPECTED_TRAIN_SOURCE_COUNT or Counter(row.get("class_id") for row in train) != Counter(
        EXPECTED_TRAIN_CLASS_COUNTS
    ):
        raise AuditError("frozen v3 experimental-train population differs")
    overlap_counts: dict[str, int] = {}
    for key in ("copied_image_sha256", "source_group", "creator_group"):
        train_values = {row.get(key) for row in train}
        other_values = {row.get(key) for row in other}
        if None in train_values or "" in train_values:
            raise AuditError(f"frozen v3 train row lacks {key}")
        overlap = train_values & other_values
        overlap_counts[key] = len(overlap)
        if overlap:
            raise AuditError(f"frozen v3 has cross-partition {key} overlap")

    train_by_asset = {str(row["asset"]): row for row in train}
    if len(train_by_asset) != len(train):
        raise AuditError("duplicate train asset identity")
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        by_class[str(row["class_id"])].append(row)
    for class_id in CLASS_ORDER:
        by_class[class_id].sort(key=lambda row: (str(row["asset"]), str(row["filename"])))

    manifest_path = safe_child(
        staging, receipt["calibration"]["manifest_path"], location="calibration.manifest_path"
    )
    if sha256_file(manifest_path) != receipt["calibration"]["manifest_sha256"]:
        raise AuditError("calibration manifest hash differs from receipt")
    samples = load_jsonl(manifest_path)
    if len(samples) != EXPECTED_SAMPLE_COUNT:
        raise AuditError(f"expected {EXPECTED_SAMPLE_COUNT} calibration samples")
    if receipt["calibration"]["sample_count"] != EXPECTED_SAMPLE_COUNT:
        raise AuditError("receipt calibration sample_count differs")

    seen_paths: set[str] = set()
    seen_assets: set[str] = set()
    class_counts: Counter[str] = Counter()
    max_delta = 0.0
    for index, sample in enumerate(samples):
        if sample.get("calibration_index") != index:
            raise AuditError(f"non-sequential calibration index at row {index}")
        class_segment = index // SAMPLES_PER_CLASS
        class_position = index % SAMPLES_PER_CLASS
        expected_class = CLASS_ORDER[class_segment]
        sources = by_class[expected_class]
        expected_source = sources[class_position % len(sources)]
        occurrence = class_position // len(sources)
        if occurrence >= len(VARIANTS):
            raise AuditError("variant table cannot reproduce calibration selection")
        expected_variant = VARIANTS[occurrence]
        if sample.get("class_id") != expected_class:
            raise AuditError(f"calibration class ordering differs at row {index}")
        if sample.get("source_asset") != expected_source["asset"]:
            raise AuditError(f"deterministic source selection differs at row {index}")
        if sample.get("variant") != expected_variant:
            raise AuditError(f"deterministic transform differs at row {index}")
        if sample.get("experimental_role") != TRAIN_ROLE:
            raise AuditError(f"non-train role in calibration row {index}")
        if sample.get("formal_split_assigned") is not False:
            raise AuditError(f"formal split is overstated in calibration row {index}")
        source_row = train_by_asset.get(str(sample.get("source_asset")))
        if source_row is None:
            raise AuditError(f"non-train source asset in calibration row {index}")
        for field, source_key in (
            ("source_filename", "filename"),
            ("source_sha256", "copied_image_sha256"),
            ("source_group", "source_group"),
            ("creator_group", "creator_group"),
            ("source_dataset", "source_dataset"),
            ("source_pageid", "pageid"),
        ):
            if sample.get(field) != source_row.get(source_key):
                raise AuditError(f"source binding {field} differs at row {index}")
        source_path = safe_child(pack, source_row["filename"], location=f"source[{index}]")
        if sha256_file(source_path) != source_row["copied_image_sha256"]:
            raise AuditError(f"source file hash differs at row {index}")

        with Image.open(source_path) as opened:
            decoded = ImageOps.exif_transpose(opened).convert("RGB")
        transformed = apply_variant(decoded, expected_variant)
        raw, geometry = preprocess_independent(transformed)
        expected_payload = raw.astype(np.float32, copy=False).tobytes(order="C")
        relative = canonical_relative(
            sample.get("calibration_path"), location=f"calibration[{index}].path"
        )
        if relative in seen_paths:
            raise AuditError(f"duplicate calibration path: {relative}")
        seen_paths.add(relative)
        target = safe_child(staging, relative, location=f"calibration[{index}].path")
        if target.stat().st_size != CALIBRATION_FILE_BYTES:
            raise AuditError(f"calibration byte length differs at row {index}")
        actual_digest = sha256_file(target)
        expected_digest = sha256_bytes(expected_payload)
        if actual_digest != expected_digest or sample.get("calibration_sha256") != actual_digest:
            raise AuditError(f"calibration payload reproduction failed at row {index}")
        if sample.get("calibration_bytes") != CALIBRATION_FILE_BYTES:
            raise AuditError(f"manifest calibration_bytes differs at row {index}")
        if sample.get("stored_shape") != STORED_SHAPE or sample.get("logical_model_shape") != INPUT_SHAPE:
            raise AuditError(f"calibration shape metadata differs at row {index}")
        if sample.get("dtype") != "float32" or sample.get("terminal_color_order") != "RGB":
            raise AuditError(f"calibration dtype/color metadata differs at row {index}")
        if sample.get("terminal_layout") != "NCHW_WITH_BATCH_OMITTED_PER_SAMPLE_FILE":
            raise AuditError(f"calibration layout metadata differs at row {index}")
        expected_range = [float(raw.min()), float(raw.max())]
        if sample.get("value_range") != expected_range or sample.get("geometry") != geometry:
            raise AuditError(f"calibration geometry/range metadata differs at row {index}")
        bindings = normalized_bindings_independent(raw)
        actual_bindings = sample.get("normalized_bindings")
        if not isinstance(actual_bindings, dict):
            raise AuditError(f"missing normalization bindings at row {index}")
        for key in ("mapper_formula_sha256", "training_formula_sha256"):
            if actual_bindings.get(key) != bindings[key]:
                raise AuditError(f"normalization hash {key} differs at row {index}")
        for key in (
            "mapper_vs_training_max_abs_delta",
            "normalized_min",
            "normalized_max",
        ):
            if abs(float(actual_bindings.get(key)) - float(bindings[key])) > 1e-7:
                raise AuditError(f"normalization numeric binding {key} differs at row {index}")
        max_delta = max(max_delta, float(bindings["mapper_vs_training_max_abs_delta"]))
        seen_assets.add(str(sample["source_asset"]))
        class_counts[expected_class] += 1

    if class_counts != Counter({name: SAMPLES_PER_CLASS for name in CLASS_ORDER}):
        raise AuditError("calibration class balance differs")
    if seen_assets != set(train_by_asset):
        raise AuditError("calibration does not cover exactly all expected train assets")
    return {
        "sample_count": len(samples),
        "samples_per_class": dict(sorted(class_counts.items())),
        "unique_train_sources_covered": len(seen_assets),
        "non_train_sources": 0,
        "cross_partition_overlap_counts": overlap_counts,
        "replayed_payloads": len(samples),
        "bytes_per_sample": CALIBRATION_FILE_BYTES,
        "max_mapper_vs_training_formula_abs_delta": max_delta,
    }


def audit_staging(staging: Path) -> dict[str, Any]:
    workspace = WORKSPACE.resolve(strict=True)
    staging = staging.resolve(strict=True)
    try:
        staging.relative_to(workspace)
    except ValueError as error:
        raise AuditError("staging pack is outside the AdventureX workspace") from error
    if staging == workspace or not staging.is_dir() or staging.is_symlink():
        raise AuditError("invalid staging directory")

    sums = audit_full_hash_coverage(staging)
    receipt_path = staging / "staging_receipt.json"
    receipt = load_json(receipt_path)
    if receipt.get("schema_version") != SCHEMA or receipt.get("status") != STATUS:
        raise AuditError("staging receipt schema/status differs")
    if receipt.get("workspace_scope") != "ADVENTUREX_ONLY":
        raise AuditError("workspace scope is not AdventureX-only")
    formal = receipt.get("formal_flags")
    execution = receipt.get("execution")
    if not isinstance(formal, dict) or not isinstance(execution, dict):
        raise AuditError("receipt flags are missing")
    require_false(
        formal,
        (
            "human_reviewed",
            "rights_approved",
            "data_locked",
            "formal_a1_dataset",
            "formal_split_assigned",
            "model_candidate",
            "model_qualified",
            "bpu_compiled",
            "x5_ready",
            "execution_authority",
            "irrigation_authority",
        ),
        location="formal_flags",
    )
    require_false(
        execution,
        (
            "docker_daemon_started_by_this_stage",
            "wsl_started_by_this_stage",
            "docker_or_wsl_queried_by_builder",
            "hb_mapper_checker_executed",
            "hb_mapper_makertbin_executed",
            "bpu_binary_present",
            "x5_or_other_device_touched",
            "ssh_used",
            "network_configuration_touched",
            "x5_replay_executed",
        ),
        location="execution",
    )
    if list(staging.rglob("*.bin")):
        raise AuditError("unexpected .bin artifact in uncompiled staging pack")
    if (staging / "model_output").exists() or (staging / "logs").exists():
        raise AuditError("mapper output/log directory exists despite unexecuted status")

    run_receipt_path = WORKSPACE / RUN_REL / "run_receipt.json"
    if sha256_file(run_receipt_path) != EXPECTED_RUN_RECEIPT_SHA256:
        raise AuditError("source run receipt changed")
    if receipt["source_training_run"]["run_receipt_sha256"] != EXPECTED_RUN_RECEIPT_SHA256:
        raise AuditError("staging receipt does not bind source run receipt")
    run_receipt = load_json(run_receipt_path)
    require_false(
        run_receipt,
        ("model_candidate", "model_qualified", "x5_ready", "bpu_compiled"),
        location="source_run",
    )
    selected = run_receipt.get("selected_seed")
    if not isinstance(selected, dict) or selected.get("seed") != 17:
        raise AuditError("source run selected seed is not 17")
    if selected.get("artifacts", {}).get("onnx") != SOURCE_ONNX_REL.as_posix():
        raise AuditError("source run selected ONNX path differs")

    source_model = WORKSPACE / RUN_REL / SOURCE_ONNX_REL
    staged_model = safe_child(staging, receipt["model"]["staged_path"], location="model.staged_path")
    for path, name in ((source_model, "source"), (staged_model, "staged")):
        if sha256_file(path) != EXPECTED_ONNX_SHA256:
            raise AuditError(f"{name} ONNX hash differs")
    if receipt["model"]["sha256"] != EXPECTED_ONNX_SHA256:
        raise AuditError("receipt ONNX hash differs")
    onnx_contract = inspect_onnx(staged_model)
    if receipt["model"]["onnx_contract"] != onnx_contract:
        raise AuditError("receipt ONNX contract differs from independent inspection")
    if receipt["model"]["class_order"] != list(CLASS_ORDER):
        raise AuditError("receipt class order differs")

    provenance_path = WORKSPACE / RUN_REL / "seed_00017/model_provenance.json"
    if receipt["model"]["model_provenance_sha256"] != sha256_file(provenance_path):
        raise AuditError("model provenance hash differs")
    provenance = load_json(provenance_path)
    if provenance.get("class_order") != list(CLASS_ORDER) or provenance.get("input_shape") != INPUT_SHAPE:
        raise AuditError("model provenance contract differs")

    generator = WORKSPACE / GENERATOR_REL
    if receipt["implementation"]["sha256"] != sha256_file(generator):
        raise AuditError("staging builder source changed after package creation")
    preprocess_path = safe_child(
        staging, receipt["preprocess_contract"]["path"], location="preprocess_contract.path"
    )
    if sha256_file(preprocess_path) != receipt["preprocess_contract"]["sha256"]:
        raise AuditError("preprocess contract hash differs")
    contract = load_json(preprocess_path)
    terminal = contract.get("horizon_calibration_terminal_contract", {})
    if terminal.get("stored_shape") != STORED_SHAPE or terminal.get("color_order") != "RGB":
        raise AuditError("preprocess terminal shape/color differs")
    if terminal.get("layout") != "NCHW_WITH_BATCH_OMITTED_PER_SAMPLE_FILE":
        raise AuditError("preprocess terminal layout differs")
    target_runtime = contract.get("target_runtime", {})
    expected_runtime = {
        "mapper_input_type_rt": "rgb",
        "mapper_input_source": "ddr",
        "host_geometry_before_rgb_ddr": ["short-side resize 256", "center crop 224"],
        "camera_source_color": "BGR",
        "host_color_conversion": "BGR_TO_RGB",
        "host_layout_conversion": "HWC_TO_NCHW_WITH_BATCH",
    }
    for key, expected in expected_runtime.items():
        if target_runtime.get(key) != expected:
            raise AuditError(f"preprocess target_runtime.{key} differs")
    if target_runtime.get("host_tensor_contract") != {
        "dtype": "uint8",
        "layout": "NCHW",
        "shape": [1, 3, 224, 224],
        "contiguous": True,
        "normalization_on_host": False,
    }:
        raise AuditError("preprocess target runtime tensor contract differs")

    reuse_path = safe_child(
        staging, receipt["xrd_readonly_reuse"]["path"], location="xrd_reuse.path"
    )
    if sha256_file(reuse_path) != receipt["xrd_readonly_reuse"]["sha256"]:
        raise AuditError("XRD reuse inventory hash differs")
    reuse = load_json(reuse_path)
    if reuse.get("reuse_boundary") != "READ_ONLY_REFERENCE_ONLY_NO_XRD_ARTIFACT_MODIFIED":
        raise AuditError("XRD reuse boundary differs")
    for role, relative in XRD_REFERENCE_PATHS.items():
        record = reuse.get("sources", {}).get(role)
        if not isinstance(record, dict) or record.get("path_from_xrd_root") != relative.as_posix():
            raise AuditError(f"XRD reuse record differs: {role}")
        if record.get("sha256") != sha256_file(XRD_ROOT / relative):
            raise AuditError(f"XRD reuse source changed: {role}")
        if record.get("reuse_mode") != "READ_ONLY_CONTRACT_REFERENCE_NOT_COPIED":
            raise AuditError(f"XRD reuse mode differs: {role}")

    mapper = receipt.get("mapper", {})
    if mapper.get("toolchain_image_expected") != TOOLCHAIN_IMAGE:
        raise AuditError("toolchain image binding differs")
    mapper_script = safe_child(
        staging, mapper.get("inside_container_script_path"), location="mapper.script"
    )
    if sha256_file(mapper_script) != mapper.get("inside_container_script_sha256"):
        raise AuditError("inside-container mapper script hash differs")
    script_text = mapper_script.read_text(encoding="utf-8")
    for required in (
        "sha256sum -c STAGING_SHA256SUMS",
        "hb_mapper checker",
        "--march bayes-e",
        "hb_mapper makertbin",
        "--model-type onnx",
    ):
        if required not in script_text:
            raise AuditError(f"inside-container script lacks: {required}")
    for forbidden in (r"(?m)^\s*docker\b", r"(?m)^\s*wsl\b", r"(?m)^\s*ssh\b"):
        if re.search(forbidden, script_text):
            raise AuditError("inside-container script attempts forbidden host/device action")
    mapper_summary = audit_config(staging, receipt)
    calibration_summary = audit_calibration(staging, receipt)

    explicit_non_claims = set(receipt.get("explicit_non_claims", []))
    required_non_claims = {
        "BPU_COMPILED",
        "CHECKER_PASSED",
        "MAKERTBIN_PASSED",
        "BPU_BINARY_PRESENT",
        "X5_READY",
        "X5_REPLAY_PASSED",
        "MODEL_QUALIFIED",
        "HUMAN_REVIEWED",
        "RIGHTS_APPROVED",
        "DATA_LOCKED",
        "FORMAL_A1_DATASET",
        "IRRIGATION_AUTHORITY",
    }
    if not required_non_claims.issubset(explicit_non_claims):
        raise AuditError("explicit non-claims are incomplete")

    return {
        "schema_version": "rootscope.seed17.bayes_e.independent_staging_audit.v1",
        "status": "PASS",
        "staging_status": STATUS,
        "staging_path": str(staging),
        "full_sha256_coverage": True,
        "hashed_file_count": len(sums),
        "staging_sha256sums_sha256": sha256_file(staging / "STAGING_SHA256SUMS"),
        "staging_receipt_sha256": sha256_file(receipt_path),
        "model": onnx_contract,
        "calibration": calibration_summary,
        "mapper_config": mapper_summary,
        "formal_flags": dict(formal),
        "execution": dict(execution),
        "read_only_audit": True,
        "docker_wsl_hb_mapper_device_invocations": 0,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument(
        "--evidence-out",
        type=Path,
        help="optional JSON report path under AdventureX; staging remains read-only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_staging(args.staging)
    if args.evidence_out is not None:
        evidence = args.evidence_out.resolve(strict=False)
        try:
            evidence.relative_to(WORKSPACE.resolve(strict=True))
        except ValueError as error:
            raise AuditError("evidence output must stay inside AdventureX") from error
        write_json(evidence, report)
        report = {**report, "evidence_out": str(evidence), "evidence_sha256": sha256_file(evidence)}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
