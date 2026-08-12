#!/usr/bin/env python3
"""Build the RootScope seed-17 Bayes-e *offline staging* package.

This program never invokes Docker, WSL, hb_mapper, SSH, or a device.  It only
reads the frozen AdventureX inputs and writes a deterministic package below the
AdventureX workspace.  The package remains explicitly uncompiled and has no
execution or irrigation authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np
import onnx
from PIL import Image, ImageEnhance, ImageOps


WORKSPACE = Path(__file__).resolve().parents[2]
XRD_ROOT = WORKSPACE.parent
PACK_REL = Path("datasets/rootscope_machine_curated_provisional_v3")
RUN_REL = Path(
    "output/rootscope_machine_curated_experimental_runs/"
    "v3_rtx4050_multiseed_20260717_r1"
)
SOURCE_ONNX_REL = Path("seed_00017/model_static_b1x3x224x224_opset11.onnx")
DEFAULT_OUTPUT_REL = Path("output/rootscope_bpu_seed17_staging_20260717_r3")

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
RAW_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
MAPPER_SCALE = np.asarray(
    [0.01712475, 0.017507, 0.01742919], dtype=np.float32
).reshape(3, 1, 1)
UNIT_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
UNIT_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
CALIBRATION_FILE_BYTES = int(np.prod(STORED_SHAPE)) * np.dtype(np.float32).itemsize
STATUS = "BPU_OFFLINE_STAGING_ONLY_NOT_COMPILED_NOT_X5_READY"
SCHEMA = "rootscope.seed17.bayes_e.offline_staging_receipt.v1"
TOOLCHAIN_IMAGE = "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310"


# Thirteen fixed, non-random variants are enough to cycle the five young-tree
# train sources up to the per-class quota.  Every class and every one of the 55
# train sources is covered; no validation/print/creator-holdout image is used.
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


class StagingError(RuntimeError):
    """A fail-closed staging gate rejected the inputs or destination."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StagingError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise StagingError(f"blank JSONL row: {path}:{line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise StagingError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def safe_pack_file(pack_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise StagingError(f"unsafe pack path: {relative!r}")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise StagingError(f"unsafe pack path: {relative!r}")
    candidate = (pack_root / Path(*posix.parts)).resolve(strict=True)
    try:
        candidate.relative_to(pack_root.resolve(strict=True))
    except ValueError as error:
        raise StagingError(f"pack path escapes root: {relative!r}") from error
    if candidate.is_symlink() or not candidate.is_file():
        raise StagingError(f"pack path is not a regular file: {relative!r}")
    return candidate


def ensure_under_workspace(path: Path, *, must_not_exist: bool) -> Path:
    workspace = WORKSPACE.resolve(strict=True)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise StagingError(f"output must remain under AdventureX workspace: {resolved}") from error
    if resolved == workspace:
        raise StagingError("output cannot be the AdventureX workspace root")
    if must_not_exist and resolved.exists():
        raise StagingError(f"refusing to overwrite existing staging directory: {resolved}")
    return resolved


def inspect_onnx(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)

    def value_info(value: Any) -> dict[str, Any]:
        tensor = value.type.tensor_type
        shape: list[Any] = []
        for dim in tensor.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                shape.append(str(dim.dim_param))
            else:
                shape.append(None)
        return {"name": value.name, "shape": shape, "elem_type": int(tensor.elem_type)}

    inputs = [value_info(value) for value in model.graph.input]
    outputs = [value_info(value) for value in model.graph.output]
    opsets = [{"domain": item.domain, "version": int(item.version)} for item in model.opset_import]
    operators = Counter(node.op_type for node in model.graph.node)
    result = {
        "sha256": sha256_file(path),
        "ir_version": int(model.ir_version),
        "inputs": inputs,
        "outputs": outputs,
        "opsets": opsets,
        "node_count": len(model.graph.node),
        "operator_counts": dict(sorted(operators.items())),
    }
    if inputs != [{"name": "image", "shape": INPUT_SHAPE, "elem_type": 1}]:
        raise StagingError(f"unexpected ONNX input contract: {inputs}")
    if outputs != [{"name": "logits", "shape": [1, 4], "elem_type": 1}]:
        raise StagingError(f"unexpected ONNX output contract: {outputs}")
    if opsets != [{"domain": "", "version": 11}]:
        raise StagingError(f"unexpected ONNX opset contract: {opsets}")
    return result


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


def short_side_resize_center_crop(image: Image.Image) -> tuple[np.ndarray, dict[str, Any]]:
    """Match torchvision Resize(256, bilinear)+CenterCrop(224), before ToTensor."""
    width, height = image.size
    short, long = (width, height) if width <= height else (height, width)
    new_short = 256
    new_long = int(new_short * long / short)
    if width <= height:
        new_width, new_height = new_short, new_long
    else:
        new_width, new_height = new_long, new_short
    resized = image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    top = int(round((new_height - 224) / 2.0))
    left = int(round((new_width - 224) / 2.0))
    crop_box = [left, top, left + 224, top + 224]
    cropped = resized.crop(tuple(crop_box))
    raw = np.asarray(cropped, dtype=np.float32).transpose(2, 0, 1).copy()
    if list(raw.shape) != STORED_SHAPE:
        raise StagingError(f"unexpected calibration tensor shape: {raw.shape}")
    return raw, {
        "source_size_wh": [width, height],
        "resized_size_wh": [new_width, new_height],
        "center_crop_box_ltrb": crop_box,
    }


def normalized_bindings(raw_rgb_chw: np.ndarray) -> dict[str, Any]:
    mapper = ((raw_rgb_chw - RAW_MEAN) * MAPPER_SCALE).astype(np.float32, copy=False)
    training = ((raw_rgb_chw / np.float32(255.0) - UNIT_MEAN) / UNIT_STD).astype(
        np.float32, copy=False
    )
    return {
        "mapper_formula_sha256": sha256_bytes(mapper.tobytes(order="C")),
        "training_formula_sha256": sha256_bytes(training.tobytes(order="C")),
        "mapper_vs_training_max_abs_delta": float(np.max(np.abs(mapper - training))),
        "normalized_min": float(mapper.min()),
        "normalized_max": float(mapper.max()),
    }


def validate_sources() -> dict[str, Any]:
    pack_root = (WORKSPACE / PACK_REL).resolve(strict=True)
    run_root = (WORKSPACE / RUN_REL).resolve(strict=True)
    manifest_path = pack_root / "manifest.jsonl"
    pack_receipt_path = pack_root / "receipt.json"
    run_receipt_path = run_root / "run_receipt.json"
    source_onnx = run_root / SOURCE_ONNX_REL
    model_provenance_path = run_root / "seed_00017/model_provenance.json"

    if sha256_file(manifest_path) != EXPECTED_PACK_MANIFEST_SHA256:
        raise StagingError("frozen v3 manifest hash differs from the selected seed-17 input")
    pack_receipt = load_json(pack_receipt_path)
    if pack_receipt.get("manifest_sha256") != EXPECTED_PACK_MANIFEST_SHA256:
        raise StagingError("frozen v3 receipt does not bind the expected manifest")
    for field in ("human_reviewed", "rights_approved", "data_locked", "training_eligible"):
        if pack_receipt.get(field) is not False:
            raise StagingError(f"frozen v3 receipt.{field} must remain false")

    if sha256_file(run_receipt_path) != EXPECTED_RUN_RECEIPT_SHA256:
        raise StagingError("training run receipt hash differs from the selected seed-17 run")
    run_receipt = load_json(run_receipt_path)
    if run_receipt.get("status") != "MACHINE_CURATED_EXPERIMENTAL_MODEL_NOT_QUALIFIED":
        raise StagingError("source run status is not the frozen experimental status")
    for field in ("model_candidate", "model_qualified", "x5_ready", "bpu_compiled"):
        if run_receipt.get(field) is not False:
            raise StagingError(f"source run {field} must remain false")
    selected = run_receipt.get("selected_seed")
    if not isinstance(selected, dict) or selected.get("seed") != 17:
        raise StagingError("source run selected seed is not 17")
    expected_artifact = SOURCE_ONNX_REL.as_posix()
    if selected.get("artifacts", {}).get("onnx") != expected_artifact:
        raise StagingError("source run selected ONNX path differs from the frozen seed-17 artifact")
    if sha256_file(source_onnx) != EXPECTED_ONNX_SHA256:
        raise StagingError("seed-17 ONNX hash differs from the frozen artifact")
    onnx_contract = inspect_onnx(source_onnx)

    provenance = load_json(model_provenance_path)
    if provenance.get("architecture") != "torchvision.resnet18":
        raise StagingError("model provenance architecture is not torchvision.resnet18")
    if provenance.get("class_order") != list(CLASS_ORDER):
        raise StagingError("model provenance class order differs")
    if provenance.get("input_shape") != INPUT_SHAPE:
        raise StagingError("model provenance input shape differs")

    all_rows = load_jsonl(manifest_path)
    train_rows = [row for row in all_rows if row.get("experimental_split_suggestion") == TRAIN_ROLE]
    non_train_rows = [row for row in all_rows if row.get("experimental_split_suggestion") != TRAIN_ROLE]
    if len(train_rows) != EXPECTED_TRAIN_SOURCE_COUNT:
        raise StagingError(
            f"expected {EXPECTED_TRAIN_SOURCE_COUNT} experimental-train rows, found {len(train_rows)}"
        )
    if Counter(row.get("class_id") for row in train_rows) != Counter(EXPECTED_TRAIN_CLASS_COUNTS):
        raise StagingError("experimental-train class counts differ from the frozen v3 contract")

    for key in ("copied_image_sha256", "source_group", "creator_group"):
        train_values = {row.get(key) for row in train_rows}
        other_values = {row.get(key) for row in non_train_rows}
        if None in train_values or "" in train_values:
            raise StagingError(f"train row lacks {key}")
        overlap = train_values & other_values
        if overlap:
            raise StagingError(f"cross-partition {key} overlap: {sorted(overlap)}")

    for row in train_rows:
        image = safe_pack_file(pack_root, row.get("filename"))
        actual = sha256_file(image)
        if actual != row.get("copied_image_sha256") or actual != row.get("source_image_sha256"):
            raise StagingError(f"train image hash binding failed: {row.get('filename')}")

    xrd_reuse: dict[str, Any] = {}
    for role, relative in XRD_REFERENCE_PATHS.items():
        path = (XRD_ROOT / relative).resolve(strict=True)
        if not path.is_file() or path.is_symlink():
            raise StagingError(f"XRD reuse source is not a regular file: {relative.as_posix()}")
        xrd_reuse[role] = {
            "path_from_xrd_root": relative.as_posix(),
            "sha256": sha256_file(path),
            "reuse_mode": "READ_ONLY_CONTRACT_REFERENCE_NOT_COPIED",
        }

    return {
        "pack_root": pack_root,
        "run_root": run_root,
        "source_onnx": source_onnx,
        "model_provenance_path": model_provenance_path,
        "manifest_path": manifest_path,
        "pack_receipt_path": pack_receipt_path,
        "run_receipt_path": run_receipt_path,
        "all_rows": all_rows,
        "train_rows": train_rows,
        "onnx_contract": onnx_contract,
        "xrd_reuse": xrd_reuse,
    }


def canonical_mapper_yaml() -> str:
    return """# RootScope seed-17 ResNet18: Bayes-e OFFLINE STAGING ONLY.
# This file has not been executed by hb_mapper.  It does not prove BPU support.
model_parameters:
  # hb_mapper resolves paths relative to the YAML file, not the caller's cwd.
  onnx_model: '../model/rootscope_seed17_resnet18_static_b1x3x224x224_opset11.onnx'
  march: 'bayes-e'
  layer_out_dump: False
  working_dir: '../model_output'
  output_model_file_prefix: 'rootscope_seed17_resnet18_224x224_rgb_ddr'
  log_level: 'debug'

input_parameters:
  input_name: 'image'
  input_type_rt: 'rgb'
  input_layout_rt: 'NCHW'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  input_shape: '1x3x224x224'
  norm_type: 'data_mean_and_scale'
  mean_value: '123.675 116.28 103.53'
  scale_value: '0.01712475 0.017507 0.01742919'

calibration_parameters:
  cal_data_dir: '../calibration_data_rgb_f32'
  cal_data_type: 'float32'
  calibration_type: 'default'

compiler_parameters:
  compile_mode: 'latency'
  debug: False
  optimize_level: 'O3'
  core_num: 1
  input_source: {'image': 'ddr'}
"""


def inside_mapper_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# This script is for an already-running OpenExplorer container only.  The
# staging builder and host preflight never start Docker/WSL or invoke hb_mapper.
sha256sum -c STAGING_SHA256SUMS
mkdir -p logs

hb_mapper --version 2>&1 | tee logs/hb_mapper_version.log
hb_mapper checker \\
  --model-type onnx \\
  --model ./model/rootscope_seed17_resnet18_static_b1x3x224x224_opset11.onnx \\
  --march bayes-e 2>&1 | tee logs/hb_mapper_checker.log

hb_mapper makertbin \\
  --config ./config/rootscope_seed17_bayes_e.yaml \\
  --model-type onnx 2>&1 | tee logs/hb_mapper_makertbin.log

echo "Mapper commands completed, but BPU/X5 readiness still requires an independent post-build audit and X5 replay."
"""


def readme_text() -> str:
    return f"""# RootScope seed-17 RDK X5 Bayes-e 离线准备包

状态：`{STATUS}`。

本阶段**没有启动 Docker daemon 或 WSL**，没有运行 `hb_mapper checker` / `makertbin`，
没有生成 `.bin`，没有连接或上板 RDK X5，也没有取得任何灌溉执行权限。本目录只是可复核的
离线输入包；`bpu_compiled=false`、`x5_ready=false`、`model_qualified=false`。

## 已准备内容

- 冻结 seed-17 ResNet18 ONNX：静态 `1x3x224x224`、opset 11、输出 `1x4`；
- 256 个 **train-only** 校准张量，每类 64 个，覆盖冻结 v3 的全部 55 个
  `EXPERIMENTAL_TRAIN_SUGGESTION` 来源；
- Bayes-e mapper YAML、容器内 checker/makertbin 脚本、完整 SHA-256 清单；
- 独立的外层只读 preflight/audit，二者都不会查询或启动 Docker/WSL。

## 预处理合同

模型评估合同是：RGB → 短边按双线性缩放到 256 → 中心裁剪 224 →
`ToTensor` → ImageNet Normalize。校准文件保存 mapper 官方分类路径的终端输入：
`float32 RGB NCHW [0,255]`（每个文件 `{CALIBRATION_FILE_BYTES}` bytes）；mapper YAML 再执行
`(x - [123.675,116.28,103.53]) * [0.01712475,0.017507,0.01742919]`。
运行时目标输入为连续 `uint8 RGB NCHW [1,3,224,224]`，输入源显式固定为 DDR。
摄像头/OpenCV 的 BGR 图像只在主机完成同一短边缩放、中心裁剪、BGR→RGB 和 HWC→NCHW；mean/scale 由模型内预处理执行，
主机禁止再次归一化。USB/UVC 或文件帧不走 VIO pyramid 隐式路径。

固定亮度/对比度/饱和度/水平翻转仅用于从 55 个 train 来源构造量化校准覆盖；每个变换、
源图路径、类别、来源 SHA、输出 SHA 和归一化参考 SHA 都记录在
`calibration_manifest.jsonl`。validation、print-demo、creator-holdout 均未进入校准集。

## 只读预检（不会触碰 Docker/WSL/设备）

从 AdventureX 根目录运行：

```powershell
.ai_curation_venv\\Scripts\\python.exe tools\\bpu\\preflight_rootscope_seed17_bpu_staging.py
```

## 后续显式编译（本阶段未执行）

仅在用户允许启动 Docker 且网络/设备边界确认后，人工把本目录挂载到
`/workspace`，镜像固定为 `{TOOLCHAIN_IMAGE}`，再在容器内运行：

```bash
bash scripts/inside_toolchain_mapper.sh
```

即使 checker/makertbin 成功，也只能把 `bpu_compiled` 状态交给新的独立 post-build
证据包更新；在 X5 真机隔离回放、CPU/BPU 漂移和摄像头实拍验证前，仍不得标记
`x5_ready` 或 `model_qualified`，更不得直接驱动灌溉。
"""


def preprocess_contract() -> dict[str, Any]:
    return {
        "schema_version": "rootscope.seed17.mapper_preprocess_contract.v1",
        "model_input": {"name": "image", "shape": INPUT_SHAPE, "dtype": "float32"},
        "training_and_cpu_evaluation": {
            "decode": "PIL Image.open + EXIF transpose + RGB",
            "geometry": [
                "Resize(short_side=256, interpolation=BILINEAR)",
                "CenterCrop(224x224)",
            ],
            "tensor": "RGB NCHW float32 [0,1]",
            "normalization": {
                "formula": "(x/255 - mean) / std",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "horizon_calibration_terminal_contract": {
            "stored_shape": STORED_SHAPE,
            "logical_shape": INPUT_SHAPE,
            "dtype": "float32",
            "value_range": [0.0, 255.0],
            "color_order": "RGB",
            "layout": "NCHW_WITH_BATCH_OMITTED_PER_SAMPLE_FILE",
            "equivalent_official_transform_sequence": [
                "ShortSideResizeTransformer(256)",
                "CenterCropTransformer(224)",
                "HWC2CHWTransformer()",
                "BGR2RGBTransformer()",
            ],
            "mapper_normalization": {
                "norm_type": "data_mean_and_scale",
                "mean_value": [123.675, 116.28, 103.53],
                "scale_value": [0.01712475, 0.017507, 0.01742919],
            },
        },
        "target_runtime": {
            "mapper_input_type_rt": "rgb",
            "mapper_input_source": "ddr",
            "host_geometry_before_rgb_ddr": ["short-side resize 256", "center crop 224"],
            "camera_source_color": "BGR",
            "host_color_conversion": "BGR_TO_RGB",
            "host_layout_conversion": "HWC_TO_NCHW_WITH_BATCH",
            "host_tensor_contract": {
                "dtype": "uint8",
                "layout": "NCHW",
                "shape": [1, 3, 224, 224],
                "contiguous": True,
                "normalization_on_host": False,
            },
            "authority": "INFERENCE_EVIDENCE_ONLY_NO_IRRIGATION_AUTHORITY",
        },
    }


def build_staging(output: Path) -> dict[str, Any]:
    output = ensure_under_workspace(output, must_not_exist=True)
    source = validate_sources()
    output.mkdir(parents=True, exist_ok=False)
    (output / "model").mkdir()
    (output / "config").mkdir()
    (output / "scripts").mkdir()
    calibration_dir = output / "calibration_data_rgb_f32"
    calibration_dir.mkdir()

    staged_model_rel = Path(
        "model/rootscope_seed17_resnet18_static_b1x3x224x224_opset11.onnx"
    )
    staged_model = output / staged_model_rel
    shutil.copyfile(source["source_onnx"], staged_model)
    if sha256_file(staged_model) != EXPECTED_ONNX_SHA256:
        raise StagingError("mechanical ONNX copy changed bytes")

    (output / "config/rootscope_seed17_bayes_e.yaml").write_text(
        canonical_mapper_yaml(), encoding="utf-8", newline="\n"
    )
    (output / "scripts/inside_toolchain_mapper.sh").write_text(
        inside_mapper_script(), encoding="utf-8", newline="\n"
    )
    (output / "README.md").write_text(readme_text(), encoding="utf-8", newline="\n")
    write_json(output / "preprocess_contract.json", preprocess_contract())
    write_json(
        output / "xrd_readonly_reuse_inventory.json",
        {
            "schema_version": "rootscope.xrd_bpu_reuse_inventory.v1",
            "reuse_boundary": "READ_ONLY_REFERENCE_ONLY_NO_XRD_ARTIFACT_MODIFIED",
            "sources": source["xrd_reuse"],
        },
    )

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source["train_rows"]:
        by_class[str(row["class_id"])].append(row)
    for class_id in CLASS_ORDER:
        by_class[class_id].sort(key=lambda row: (str(row["asset"]), str(row["filename"])))

    calibration_rows: list[dict[str, Any]] = []
    source_usage: Counter[str] = Counter()
    variant_usage: Counter[str] = Counter()
    max_formula_delta = 0.0
    index = 0
    for class_id in CLASS_ORDER:
        class_sources = by_class[class_id]
        if not class_sources:
            raise StagingError(f"no train source for class {class_id}")
        for class_position in range(SAMPLES_PER_CLASS):
            row = class_sources[class_position % len(class_sources)]
            occurrence = class_position // len(class_sources)
            if occurrence >= len(VARIANTS):
                raise StagingError(f"variant table is too small for class {class_id}")
            variant = VARIANTS[occurrence]
            source_image = safe_pack_file(source["pack_root"], row["filename"])
            with Image.open(source_image) as opened:
                decoded = ImageOps.exif_transpose(opened).convert("RGB")
            transformed = apply_variant(decoded, variant)
            raw, geometry = short_side_resize_center_crop(transformed)
            payload = raw.astype(np.float32, copy=False).tobytes(order="C")
            if len(payload) != CALIBRATION_FILE_BYTES:
                raise StagingError("calibration payload byte count differs")
            filename = f"calib_{index:04d}_{class_id}.rgb"
            relative = Path("calibration_data_rgb_f32") / filename
            (output / relative).write_bytes(payload)
            normalized = normalized_bindings(raw)
            max_formula_delta = max(
                max_formula_delta, float(normalized["mapper_vs_training_max_abs_delta"])
            )
            source_usage[str(row["asset"])] += 1
            variant_usage[str(variant["id"])] += 1
            calibration_rows.append(
                {
                    "schema_version": "rootscope.seed17.bayes_e.calibration_sample.v1",
                    "calibration_index": index,
                    "calibration_path": relative.as_posix(),
                    "calibration_sha256": sha256_bytes(payload),
                    "calibration_bytes": len(payload),
                    "stored_shape": STORED_SHAPE,
                    "logical_model_shape": INPUT_SHAPE,
                    "dtype": "float32",
                    "value_range": [float(raw.min()), float(raw.max())],
                    "terminal_layout": "NCHW_WITH_BATCH_OMITTED_PER_SAMPLE_FILE",
                    "terminal_color_order": "RGB",
                    "class_id": class_id,
                    "experimental_role": TRAIN_ROLE,
                    "formal_split_assigned": False,
                    "source_asset": row["asset"],
                    "source_filename": row["filename"],
                    "source_sha256": row["copied_image_sha256"],
                    "source_group": row["source_group"],
                    "creator_group": row["creator_group"],
                    "source_dataset": row["source_dataset"],
                    "source_pageid": row["pageid"],
                    "variant": dict(variant),
                    "geometry": geometry,
                    "normalized_bindings": normalized,
                }
            )
            index += 1

    if index != EXPECTED_SAMPLE_COUNT:
        raise StagingError(f"expected {EXPECTED_SAMPLE_COUNT} calibration tensors, generated {index}")
    expected_assets = {str(row["asset"]) for row in source["train_rows"]}
    if set(source_usage) != expected_assets:
        raise StagingError("calibration selection does not cover every train source exactly as a set")
    write_jsonl(output / "calibration_manifest.jsonl", calibration_rows)

    class_counts = Counter(row["class_id"] for row in calibration_rows)
    summary = {
        "schema_version": "rootscope.seed17.bayes_e.calibration_summary.v1",
        "sample_count": len(calibration_rows),
        "samples_per_class": dict(sorted(class_counts.items())),
        "unique_train_sources_covered": len(source_usage),
        "expected_train_sources": len(source["train_rows"]),
        "source_usage_min": min(source_usage.values()),
        "source_usage_max": max(source_usage.values()),
        "variant_usage": dict(sorted(variant_usage.items())),
        "max_mapper_vs_training_formula_abs_delta": max_formula_delta,
        "non_train_source_count": 0,
        "validation_source_count": 0,
        "print_source_count": 0,
        "creator_holdout_source_count": 0,
    }
    write_json(output / "calibration_summary.json", summary)

    receipt = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "created_date": "2026-07-17",
        "workspace_scope": "ADVENTUREX_ONLY",
        "formal_flags": {
            "human_reviewed": False,
            "rights_approved": False,
            "data_locked": False,
            "formal_a1_dataset": False,
            "formal_split_assigned": False,
            "model_candidate": False,
            "model_qualified": False,
            "bpu_compiled": False,
            "x5_ready": False,
            "execution_authority": False,
            "irrigation_authority": False,
        },
        "execution": {
            "docker_daemon_started_by_this_stage": False,
            "wsl_started_by_this_stage": False,
            "docker_or_wsl_queried_by_builder": False,
            "hb_mapper_checker_executed": False,
            "hb_mapper_makertbin_executed": False,
            "bpu_binary_present": False,
            "x5_or_other_device_touched": False,
            "ssh_used": False,
            "network_configuration_touched": False,
            "x5_replay_executed": False,
        },
        "source_dataset": {
            "path": PACK_REL.as_posix(),
            "manifest_sha256": sha256_file(source["manifest_path"]),
            "receipt_sha256": sha256_file(source["pack_receipt_path"]),
            "experimental_train_role": TRAIN_ROLE,
            "train_row_count": len(source["train_rows"]),
            "train_class_counts": EXPECTED_TRAIN_CLASS_COUNTS,
            "cross_partition_sha_overlap_count": 0,
            "cross_partition_source_group_overlap_count": 0,
            "cross_partition_creator_group_overlap_count": 0,
        },
        "source_training_run": {
            "path": RUN_REL.as_posix(),
            "run_receipt_sha256": sha256_file(source["run_receipt_path"]),
            "selected_seed": 17,
            "status": "MACHINE_CURATED_EXPERIMENTAL_MODEL_NOT_QUALIFIED",
        },
        "model": {
            "source_path": (RUN_REL / SOURCE_ONNX_REL).as_posix(),
            "staged_path": staged_model_rel.as_posix(),
            "sha256": sha256_file(staged_model),
            "source_and_staged_bytes_identical": True,
            "class_order": list(CLASS_ORDER),
            "onnx_contract": source["onnx_contract"],
            "model_provenance_path": (
                RUN_REL / "seed_00017/model_provenance.json"
            ).as_posix(),
            "model_provenance_sha256": sha256_file(source["model_provenance_path"]),
        },
        "calibration": {
            "manifest_path": "calibration_manifest.jsonl",
            "manifest_sha256": sha256_file(output / "calibration_manifest.jsonl"),
            "summary_path": "calibration_summary.json",
            "summary_sha256": sha256_file(output / "calibration_summary.json"),
            "sample_count": len(calibration_rows),
            "stored_shape": STORED_SHAPE,
            "logical_shape": INPUT_SHAPE,
            "bytes_per_sample": CALIBRATION_FILE_BYTES,
            "dtype": "float32",
            "color_order": "RGB",
            "layout": "NCHW_WITH_BATCH_OMITTED_PER_SAMPLE_FILE",
            "train_only": True,
            "all_expected_train_sources_covered": True,
        },
        "mapper": {
            "toolchain_image_expected": TOOLCHAIN_IMAGE,
            "hb_mapper_version_expected_from_xrd_history_not_reverified_here": "1.24.3",
            "march": "bayes-e",
            "config_path": "config/rootscope_seed17_bayes_e.yaml",
            "config_sha256": sha256_file(output / "config/rootscope_seed17_bayes_e.yaml"),
            "inside_container_script_path": "scripts/inside_toolchain_mapper.sh",
            "inside_container_script_sha256": sha256_file(
                output / "scripts/inside_toolchain_mapper.sh"
            ),
            "runtime_input_type": "rgb",
            "runtime_layout": "NCHW",
            "input_source": {"image": "ddr"},
            "training_input_type": "rgb",
            "training_layout": "NCHW",
            "calibration_type": "default",
            "compile_mode": "latency",
            "optimize_level": "O3",
            "core_num": 1,
        },
        "preprocess_contract": {
            "path": "preprocess_contract.json",
            "sha256": sha256_file(output / "preprocess_contract.json"),
        },
        "xrd_readonly_reuse": {
            "path": "xrd_readonly_reuse_inventory.json",
            "sha256": sha256_file(output / "xrd_readonly_reuse_inventory.json"),
            "xrd_files_modified": False,
        },
        "implementation": {
            "path": "tools/bpu/prepare_rootscope_seed17_bpu_staging.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "explicit_non_claims": [
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
        ],
    }
    write_json(output / "staging_receipt.json", receipt)

    files = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.name != "STAGING_SHA256SUMS"
    }
    (output / "STAGING_SHA256SUMS").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in files.items()),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": STATUS,
        "output": str(output),
        "calibration_count": len(calibration_rows),
        "unique_train_sources": len(source_usage),
        "model_sha256": EXPECTED_ONNX_SHA256,
        "receipt_sha256": sha256_file(output / "staging_receipt.json"),
        "sha256sums_sha256": sha256_file(output / "STAGING_SHA256SUMS"),
        "bpu_compiled": False,
        "x5_ready": False,
        "model_qualified": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / DEFAULT_OUTPUT_REL,
        help="new output directory under AdventureX; existing directories are rejected",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_staging(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
