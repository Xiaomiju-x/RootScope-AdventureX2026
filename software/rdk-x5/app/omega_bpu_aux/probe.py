"""Manifest-only generic MobileNetV2 BPU semantic/OOD descriptor probe.

This module is an AdventureX-only, zero-authority evidence utility.  It runs
the vendor generic ImageNet-1000 MobileNetV2 binary against explicitly named,
hash-bound image files.  It does not discover cameras or files, and its output
cannot enter the RootScope Safety Compiler.

``hobot_dnn`` is imported lazily only after the model and every input artifact
have passed their SHA-256 checks.  A caller-injected DNN module is accepted
only for unit tests; every such receipt is permanently marked as fake-backend
test evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


SCHEMA_VERSION = "rootscope.omega.bpu-aux-input-manifest.v1"
RECEIPT_SCHEMA_VERSION = "rootscope.omega.bpu-aux-receipt.v1"
VENDOR_MODEL_PATH = Path(
    "/opt/hobot/model/x5/basic/mobilenetv2_224x224_nv12.bin"
)
VENDOR_LABELS_PATH = Path(
    "/app/pydev_demo/01_basic_sample/imagenet1000_clsidx_to_labels.txt"
)
RUNTIME_BACKEND = "hobot_dnn.pyeasy_dnn"
MODEL_TASK = "generic ImageNet-1000 auxiliary semantics"
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
CLASS_COUNT = 1000
MAX_IMAGES = 64
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class BpuAuxProbeError(RuntimeError):
    """The explicit artifact, runtime interface, or output failed closed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    field: str,
) -> None:
    required_set = set(required)
    optional_set = set(optional)
    actual = set(value)
    missing = sorted(required_set - actual)
    unknown = sorted(actual - required_set - optional_set)
    if missing or unknown:
        raise BpuAuxProbeError(
            f"{field} keys invalid: missing={missing}, unknown={unknown}"
        )


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BpuAuxProbeError(f"{field} must be one lowercase SHA-256 digest")
    return value


def _require_safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise BpuAuxProbeError(f"{field} must be one safe non-empty identifier")
    return value


def _explicit_regular_file(
    configured: Any,
    *,
    field: str,
    expected_sha256: str,
    allowed_suffixes: frozenset[str] | None = None,
    maximum_bytes: int | None = None,
) -> tuple[Path, int]:
    if not isinstance(configured, str) or not configured:
        raise BpuAuxProbeError(f"{field} must be one explicit absolute path")
    if any(token in configured for token in ("*", "?", "[", "]")):
        raise BpuAuxProbeError(f"{field} must not contain wildcard syntax")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        raise BpuAuxProbeError(f"{field} must be one explicit absolute path")
    if path.is_symlink():
        raise BpuAuxProbeError(f"{field} must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BpuAuxProbeError(f"{field} does not exist: {path}") from exc
    if not resolved.is_file():
        raise BpuAuxProbeError(f"{field} must be one regular file")
    if resolved.parts[:2] == ("/", "dev"):
        raise BpuAuxProbeError(f"{field} must not resolve below /dev")
    if allowed_suffixes is not None and resolved.suffix.lower() not in allowed_suffixes:
        raise BpuAuxProbeError(
            f"{field} suffix {resolved.suffix!r} is not in the explicit allowlist"
        )
    size = resolved.stat().st_size
    if size <= 0:
        raise BpuAuxProbeError(f"{field} must not be empty")
    if maximum_bytes is not None and size > maximum_bytes:
        raise BpuAuxProbeError(f"{field} exceeds {maximum_bytes} bytes")
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise BpuAuxProbeError(
            f"{field} SHA-256 mismatch: actual={actual_sha256} "
            f"expected={expected_sha256}"
        )
    return resolved, size


def _load_json_mapping(path: Path, *, field: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BpuAuxProbeError(f"{field} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise BpuAuxProbeError(f"{field} root must be an object")
    return payload


def _validate_manifest(
    payload: Mapping[str, Any],
    *,
    allow_injected_model_path: bool,
) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=(
            "schema_version",
            "run_id",
            "model",
            "top_k",
            "warmup_runs",
            "images",
        ),
        optional=("labels",),
        field="manifest",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise BpuAuxProbeError(
            f"schema_version must equal {SCHEMA_VERSION!r}"
        )
    run_id = _require_safe_id(payload["run_id"], "run_id")

    model = payload["model"]
    if not isinstance(model, Mapping):
        raise BpuAuxProbeError("model must be an object")
    _require_exact_keys(
        model,
        required=("path", "sha256", "output_semantics"),
        field="model",
    )
    model_path = model["path"]
    if not isinstance(model_path, str):
        raise BpuAuxProbeError("model.path must be a string")
    if not allow_injected_model_path and Path(model_path) != VENDOR_MODEL_PATH:
        raise BpuAuxProbeError(
            f"model.path must equal the frozen vendor path {VENDOR_MODEL_PATH}"
        )
    model_sha256 = _require_sha256(model["sha256"], "model.sha256")
    output_semantics = model["output_semantics"]
    if output_semantics not in ("LOGITS", "PROBABILITIES"):
        raise BpuAuxProbeError(
            "model.output_semantics must be LOGITS or PROBABILITIES"
        )

    top_k = payload["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise BpuAuxProbeError("top_k must be an integer in [1, 20]")
    warmup_runs = payload["warmup_runs"]
    if (
        isinstance(warmup_runs, bool)
        or not isinstance(warmup_runs, int)
        or not 0 <= warmup_runs <= 3
    ):
        raise BpuAuxProbeError("warmup_runs must be an integer in [0, 3]")

    images = payload["images"]
    if (
        not isinstance(images, list)
        or isinstance(images, (str, bytes))
        or not 1 <= len(images) <= MAX_IMAGES
    ):
        raise BpuAuxProbeError(f"images must contain between 1 and {MAX_IMAGES} rows")
    normalized_images: list[Mapping[str, str]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, row in enumerate(images):
        if not isinstance(row, Mapping):
            raise BpuAuxProbeError(f"images[{index}] must be an object")
        _require_exact_keys(
            row,
            required=("image_id", "path", "sha256"),
            field=f"images[{index}]",
        )
        image_id = _require_safe_id(row["image_id"], f"images[{index}].image_id")
        path_value = row["path"]
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise BpuAuxProbeError(
                f"images[{index}].path must be one explicit absolute path"
            )
        if image_id in seen_ids:
            raise BpuAuxProbeError(f"duplicate image_id: {image_id}")
        if path_value in seen_paths:
            raise BpuAuxProbeError(f"duplicate explicit image path: {path_value}")
        seen_ids.add(image_id)
        seen_paths.add(path_value)
        normalized_images.append(
            {
                "image_id": image_id,
                "path": path_value,
                "sha256": _require_sha256(
                    row["sha256"], f"images[{index}].sha256"
                ),
            }
        )

    labels_payload = payload.get("labels")
    labels: Mapping[str, str] | None
    if labels_payload is None:
        labels = None
    else:
        if not isinstance(labels_payload, Mapping):
            raise BpuAuxProbeError("labels must be an object")
        _require_exact_keys(
            labels_payload,
            required=("path", "sha256", "format"),
            field="labels",
        )
        if labels_payload["format"] != "PYTHON_LITERAL_DICT_INT_TO_STRING":
            raise BpuAuxProbeError(
                "labels.format must be PYTHON_LITERAL_DICT_INT_TO_STRING"
            )
        label_path = labels_payload["path"]
        if not isinstance(label_path, str):
            raise BpuAuxProbeError("labels.path must be a string")
        if (
            not allow_injected_model_path
            and Path(label_path) != VENDOR_LABELS_PATH
        ):
            raise BpuAuxProbeError(
                f"labels.path must equal the vendor path {VENDOR_LABELS_PATH}"
            )
        labels = {
            "path": label_path,
            "sha256": _require_sha256(labels_payload["sha256"], "labels.sha256"),
            "format": labels_payload["format"],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model": {
            "path": model_path,
            "sha256": model_sha256,
            "output_semantics": output_semantics,
        },
        "labels": labels,
        "top_k": top_k,
        "warmup_runs": warmup_runs,
        "images": normalized_images,
    }


def _load_labels(specification: Mapping[str, str] | None) -> tuple[dict[int, str], Any]:
    if specification is None:
        return {}, None
    path, size = _explicit_regular_file(
        specification["path"],
        field="labels.path",
        expected_sha256=specification["sha256"],
        maximum_bytes=2 * 1024 * 1024,
    )
    try:
        parsed = ast.literal_eval(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
        raise BpuAuxProbeError(f"labels artifact is invalid: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise BpuAuxProbeError("labels artifact must evaluate to a mapping")
    labels: dict[int, str] = {}
    for key, value in parsed.items():
        if isinstance(key, bool):
            raise BpuAuxProbeError("label class IDs must be integers")
        try:
            class_id = int(key)
        except (TypeError, ValueError) as exc:
            raise BpuAuxProbeError("label class IDs must be integers") from exc
        if not 0 <= class_id < CLASS_COUNT:
            raise BpuAuxProbeError(f"label class ID out of range: {class_id}")
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 160
            or "\n" in value
            or "\r" in value
        ):
            raise BpuAuxProbeError(f"invalid label text for class {class_id}")
        labels[class_id] = value
    if set(labels) != set(range(CLASS_COUNT)):
        raise BpuAuxProbeError(
            f"labels artifact must contain exactly class IDs 0..{CLASS_COUNT - 1}"
        )
    return labels, {
        "path": str(path),
        "sha256": specification["sha256"],
        "size_bytes": size,
        "format": specification["format"],
        "class_count": len(labels),
    }


def _load_explicit_rgb(
    row: Mapping[str, str],
) -> tuple[np.ndarray, Mapping[str, Any]]:
    path, size = _explicit_regular_file(
        row["path"],
        field=f"image[{row['image_id']}].path",
        expected_sha256=row["sha256"],
        allowed_suffixes=ALLOWED_IMAGE_SUFFIXES,
        maximum_bytes=MAX_IMAGE_BYTES,
    )
    try:
        with Image.open(path) as raw:
            if int(getattr(raw, "n_frames", 1)) != 1:
                raise BpuAuxProbeError("animated or multi-frame images are forbidden")
            width, height = raw.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise BpuAuxProbeError("decoded image dimensions exceed the safety limit")
            rgb_image = ImageOps.exif_transpose(raw).convert("RGB")
            rgb = np.asarray(rgb_image, dtype=np.uint8)
    except BpuAuxProbeError:
        raise
    except Exception as exc:
        raise BpuAuxProbeError(
            f"failed to decode explicit image {row['image_id']}: {exc}"
        ) from exc
    rgb = np.ascontiguousarray(rgb)
    return rgb, {
        "image_id": row["image_id"],
        "source_kind": "EXPLICIT_HASH_BOUND_IMAGE_FILE",
        "configured_path": row["path"],
        "resolved_path": str(path),
        "source_file_sha256": row["sha256"],
        "source_file_bytes": size,
        "decoded_rgb_shape": list(rgb.shape),
        "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
        "camera_opened": False,
        "camera_frames_read": 0,
        "device_enumerated": False,
    }


def rgb_to_nv12(rgb: np.ndarray) -> np.ndarray:
    """Convert an even-sized RGB uint8 image to BT.601 limited-range NV12.

    The result is one C-contiguous flat ``uint8`` buffer containing the full Y
    plane followed by interleaved 2x2-subsampled U/V chroma samples.
    """

    image = np.asarray(rgb)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or image.shape[0] <= 0
        or image.shape[1] <= 0
        or image.shape[0] % 2
        or image.shape[1] % 2
    ):
        raise BpuAuxProbeError(
            "RGB input must be positive even-sized HxWx3 uint8"
        )
    values = image.astype(np.int32)
    red = values[:, :, 0]
    green = values[:, :, 1]
    blue = values[:, :, 2]
    y_plane = ((66 * red + 129 * green + 25 * blue + 128) >> 8) + 16
    u_plane = ((-38 * red - 74 * green + 112 * blue + 128) >> 8) + 128
    v_plane = ((112 * red - 94 * green - 18 * blue + 128) >> 8) + 128
    y_plane = np.clip(y_plane, 0, 255).astype(np.uint8)
    u_plane = np.clip(u_plane, 0, 255).astype(np.int32)
    v_plane = np.clip(v_plane, 0, 255).astype(np.int32)
    u_subsampled = (
        u_plane[0::2, 0::2]
        + u_plane[0::2, 1::2]
        + u_plane[1::2, 0::2]
        + u_plane[1::2, 1::2]
        + 2
    ) // 4
    v_subsampled = (
        v_plane[0::2, 0::2]
        + v_plane[0::2, 1::2]
        + v_plane[1::2, 0::2]
        + v_plane[1::2, 1::2]
        + 2
    ) // 4
    uv_plane = np.empty(
        (image.shape[0] // 2, image.shape[1]), dtype=np.uint8
    )
    uv_plane[:, 0::2] = u_subsampled.astype(np.uint8)
    uv_plane[:, 1::2] = v_subsampled.astype(np.uint8)
    return np.ascontiguousarray(
        np.concatenate((y_plane.reshape(-1), uv_plane.reshape(-1)))
    )


def preprocess_rgb_to_nv12(rgb: np.ndarray) -> tuple[np.ndarray, Mapping[str, Any]]:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    resized = image.resize(
        (INPUT_WIDTH, INPUT_HEIGHT), resample=Image.Resampling.BILINEAR
    )
    resized_rgb = np.asarray(resized, dtype=np.uint8)
    nv12 = rgb_to_nv12(resized_rgb)
    expected_bytes = INPUT_WIDTH * INPUT_HEIGHT * 3 // 2
    if (
        nv12.shape != (expected_bytes,)
        or nv12.dtype != np.uint8
        or not nv12.flags.c_contiguous
    ):
        raise BpuAuxProbeError("preprocessor did not produce the frozen NV12 buffer")
    return nv12, {
        "resize": [INPUT_WIDTH, INPUT_HEIGHT],
        "resize_policy": "DIRECT_RESIZE",
        "resize_interpolation": "PIL_BILINEAR",
        "source_color_order": "RGB",
        "runtime_color_format": "NV12",
        "yuv_conversion": "BT.601_LIMITED_RANGE_INTEGER",
        "chroma_subsampling": "2x2_MEAN",
        "layout": "Y_THEN_INTERLEAVED_UV_FLAT",
        "dtype": "uint8",
        "shape": [expected_bytes],
        "bytes": expected_bytes,
        "nv12_sha256": hashlib.sha256(nv12.tobytes(order="C")).hexdigest(),
    }


def _property_value(tensor: Any, name: str) -> Any:
    properties = getattr(tensor, "properties", None)
    if properties is not None and hasattr(properties, name):
        return getattr(properties, name)
    return getattr(tensor, name, None)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _tensor_metadata(tensor: Any) -> Mapping[str, Any]:
    return {
        name: _json_value(_property_value(tensor, name))
        for name in ("name", "shape", "validShape", "alignedShape", "layout", "dtype")
    }


def _vector_sha256(values: np.ndarray) -> str:
    vector = np.asarray(values, dtype="<f4")
    return hashlib.sha256(np.ascontiguousarray(vector).tobytes(order="C")).hexdigest()


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def describe_output(
    raw_output: np.ndarray,
    *,
    output_semantics: str,
    top_k: int,
    labels: Mapping[int, str],
) -> Mapping[str, Any]:
    raw = np.asarray(raw_output)
    if not np.issubdtype(raw.dtype, np.number):
        raise BpuAuxProbeError("runtime output must be numeric")
    vector = raw.astype(np.float64, copy=False).reshape(-1)
    if vector.shape != (CLASS_COUNT,):
        raise BpuAuxProbeError(
            f"runtime output must contain exactly {CLASS_COUNT} values, "
            f"got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise BpuAuxProbeError("runtime output must contain only finite values")

    if output_semantics == "LOGITS":
        canonical_logits = vector
        shifted = canonical_logits - float(np.max(canonical_logits))
        probabilities = np.exp(shifted)
        probabilities /= float(probabilities.sum())
        energy_basis = "RAW_MODEL_LOGITS_TEMPERATURE_1"
        energy_raw_logit_comparable = True
    elif output_semantics == "PROBABILITIES":
        if float(np.min(vector)) < 0.0 or float(np.max(vector)) > 1.0:
            raise BpuAuxProbeError("probability output values must be in [0, 1]")
        total = float(vector.sum())
        if not math.isfinite(total) or total <= 0.0 or abs(total - 1.0) > 0.01:
            raise BpuAuxProbeError(
                "probability output must sum to 1 within absolute tolerance 0.01"
            )
        probabilities = vector / total
        canonical_logits = np.log(np.clip(probabilities, 1e-30, 1.0))
        energy_basis = "LOG_PROBABILITY_CANONICALIZATION"
        energy_raw_logit_comparable = False
    else:
        raise BpuAuxProbeError(f"unsupported output semantics: {output_semantics}")

    if not np.isfinite(canonical_logits).all() or not np.isfinite(probabilities).all():
        raise BpuAuxProbeError("postprocessed logits/probabilities must remain finite")
    probability_sum = float(probabilities.sum())
    if abs(probability_sum - 1.0) > 1e-6:
        raise BpuAuxProbeError("softmax probabilities failed the sum-to-one check")

    entropy = float(
        -np.sum(
            probabilities
            * np.log(np.clip(probabilities, np.finfo(np.float64).tiny, 1.0))
        )
    )
    normalized_entropy = entropy / math.log(CLASS_COUNT)
    energy = -_logsumexp(canonical_logits)
    order = np.argsort(-probabilities, kind="stable")[:top_k]
    top_rows = []
    for index in order:
        class_id = int(index)
        top_rows.append(
            {
                "generic_imagenet_class_id": class_id,
                "vendor_label": labels.get(class_id),
                "raw_model_value": float(vector[class_id]),
                "canonical_logit": float(canonical_logits[class_id]),
                "probability": float(probabilities[class_id]),
            }
        )

    return {
        "raw_output_semantics_declared": output_semantics,
        "class_count": CLASS_COUNT,
        "all_raw_values_finite": True,
        "all_canonical_logits_finite": True,
        "all_probabilities_finite": True,
        "probabilities_sum": probability_sum,
        "raw_values_min": float(np.min(vector)),
        "raw_values_max": float(np.max(vector)),
        "canonical_logits_min": float(np.min(canonical_logits)),
        "canonical_logits_max": float(np.max(canonical_logits)),
        "raw_values_float32_sha256": _vector_sha256(vector),
        "canonical_logits_float32_sha256": _vector_sha256(canonical_logits),
        "probabilities_float32_sha256": _vector_sha256(probabilities),
        "vectors": {
            "raw_model_values": [float(value) for value in vector],
            "canonical_logits": [float(value) for value in canonical_logits],
            "probabilities": [float(value) for value in probabilities],
        },
        "top_k": top_rows,
        "generic_descriptors": {
            "maximum_softmax_probability": float(np.max(probabilities)),
            "predictive_entropy_nats": entropy,
            "normalized_predictive_entropy": normalized_entropy,
            "energy_score_temperature_1": energy,
            "energy_basis": energy_basis,
            "energy_raw_logit_comparable": energy_raw_logit_comparable,
        },
        "ood_interpretation": {
            "status": "UNCALIBRATED_DESCRIPTORS_ONLY",
            "threshold_source": None,
            "threshold": None,
            "ood_decision": None,
            "plant_domain_ood_claim": False,
        },
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing_summary(values: Sequence[float]) -> Mapping[str, Any]:
    if not values or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise BpuAuxProbeError("timing values must be non-empty, finite and non-negative")
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


class GenericBpuAuxRunner:
    """Hash-bound one-model runner with no RootScope decision interface."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        expected_sha256: str,
        dnn_module: Any | None = None,
    ) -> None:
        expected = _require_sha256(expected_sha256, "model.sha256")
        resolved, size = _explicit_regular_file(
            str(model_path),
            field="model.path",
            expected_sha256=expected,
            allowed_suffixes=frozenset({".bin"}),
        )
        self.injected_test_backend = dnn_module is not None
        if dnn_module is None:
            try:
                from hobot_dnn import pyeasy_dnn as dnn_module  # type: ignore
            except ImportError as exc:
                raise BpuAuxProbeError(
                    "hobot_dnn.pyeasy_dnn is unavailable; no CPU/fake fallback exists"
                ) from exc
        started = time.perf_counter_ns()
        try:
            models = dnn_module.load(str(resolved))
        except Exception as exc:
            raise BpuAuxProbeError(f"pyeasy_dnn model load failed: {exc}") from exc
        self.model_load_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if not isinstance(models, (list, tuple)) or len(models) != 1:
            count = len(models) if isinstance(models, (list, tuple)) else "non-sequence"
            raise BpuAuxProbeError(
                f"vendor .bin must expose exactly one model, got {count}"
            )
        model = models[0]
        inputs = list(getattr(model, "inputs", ()))
        outputs = list(getattr(model, "outputs", ()))
        if len(inputs) != 1 or len(outputs) != 1:
            raise BpuAuxProbeError(
                f"model must expose exactly one input and one output, "
                f"got {len(inputs)}/{len(outputs)}"
            )
        self.model = model
        self.model_path = resolved
        self.model_sha256 = expected
        self.model_size_bytes = size
        self.input_metadata = _tensor_metadata(inputs[0])
        self.output_metadata = _tensor_metadata(outputs[0])
        self.model_name = str(getattr(model, "name", "mobilenetv2_224x224_nv12"))

    def forward(self, nv12: np.ndarray) -> np.ndarray:
        expected_bytes = INPUT_WIDTH * INPUT_HEIGHT * 3 // 2
        tensor = np.asarray(nv12)
        if (
            tensor.shape != (expected_bytes,)
            or tensor.dtype != np.uint8
            or not tensor.flags.c_contiguous
        ):
            raise BpuAuxProbeError("BPU input violates the frozen flat NV12 contract")
        try:
            outputs = self.model.forward(tensor)
        except Exception as exc:
            raise BpuAuxProbeError(f"pyeasy_dnn forward failed: {exc}") from exc
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            count = len(outputs) if isinstance(outputs, (list, tuple)) else "non-sequence"
            raise BpuAuxProbeError(
                f"BPU forward must return exactly one output, got {count}"
            )
        return np.asarray(getattr(outputs[0], "buffer", outputs[0]))


def _run_one(
    runner: GenericBpuAuxRunner,
    rgb: np.ndarray,
    *,
    provenance: Mapping[str, Any],
    output_semantics: str,
    top_k: int,
    labels: Mapping[int, str],
) -> Mapping[str, Any]:
    started_total = time.perf_counter_ns()
    started = time.perf_counter_ns()
    nv12, preprocess_receipt = preprocess_rgb_to_nv12(rgb)
    preprocess_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    started = time.perf_counter_ns()
    raw_output = runner.forward(nv12)
    forward_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    started = time.perf_counter_ns()
    descriptors = describe_output(
        raw_output,
        output_semantics=output_semantics,
        top_k=top_k,
        labels=labels,
    )
    postprocess_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    total_ms = (time.perf_counter_ns() - started_total) / 1_000_000.0
    return {
        "input_provenance": dict(provenance),
        "preprocess": preprocess_receipt,
        "output": descriptors,
        "timing": {
            "preprocess_ms": preprocess_ms,
            "bpu_forward_ms": forward_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_ms,
            "clock": "time.perf_counter_ns",
        },
    }


def run_manifest_probe(
    manifest_path: str | Path,
    *,
    dnn_module: Any | None = None,
    injected_model_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Execute one explicit manifest and return an in-memory evidence receipt.

    ``injected_model_path`` is rejected unless ``dnn_module`` is also injected.
    This keeps the production entry point pinned to the vendor model while
    allowing deterministic unit tests to use a temporary fake artifact.
    """

    configured_manifest = Path(manifest_path).expanduser()
    if not configured_manifest.is_absolute():
        raise BpuAuxProbeError("manifest path must be explicit and absolute")
    if configured_manifest.is_symlink():
        raise BpuAuxProbeError("manifest must not be a symbolic link")
    try:
        resolved_manifest = configured_manifest.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BpuAuxProbeError(f"manifest does not exist: {configured_manifest}") from exc
    if not resolved_manifest.is_file():
        raise BpuAuxProbeError("manifest must be one regular file")
    manifest_sha256 = sha256_file(resolved_manifest)
    manifest_size = resolved_manifest.stat().st_size
    if manifest_size <= 0 or manifest_size > 2 * 1024 * 1024:
        raise BpuAuxProbeError("manifest size is outside the allowed range")
    injected = dnn_module is not None
    if injected_model_path is not None and not injected:
        raise BpuAuxProbeError(
            "injected_model_path is allowed only with an injected test backend"
        )
    raw_manifest = _load_json_mapping(resolved_manifest, field="manifest")
    manifest = _validate_manifest(
        raw_manifest,
        allow_injected_model_path=injected,
    )
    effective_model_path: str | Path = (
        injected_model_path
        if injected_model_path is not None
        else manifest["model"]["path"]
    )
    if injected and Path(effective_model_path) != Path(manifest["model"]["path"]):
        raise BpuAuxProbeError(
            "injected test model path must still equal the hash-bound manifest path"
        )

    # Hash and decode every explicit non-device artifact before hobot_dnn import/load.
    labels, labels_receipt = _load_labels(manifest["labels"])
    decoded_images: list[tuple[np.ndarray, Mapping[str, Any]]] = []
    for image_row in manifest["images"]:
        decoded_images.append(_load_explicit_rgb(image_row))

    runner = GenericBpuAuxRunner(
        model_path=effective_model_path,
        expected_sha256=manifest["model"]["sha256"],
        dnn_module=dnn_module,
    )

    warmup_forward_ms: list[float] = []
    if manifest["warmup_runs"]:
        warmup_rgb = decoded_images[0][0]
        warmup_nv12, _ = preprocess_rgb_to_nv12(warmup_rgb)
        for _ in range(manifest["warmup_runs"]):
            started = time.perf_counter_ns()
            warmup_output = runner.forward(warmup_nv12)
            forward_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            # A warm-up cannot hide a malformed/non-finite output.
            describe_output(
                warmup_output,
                output_semantics=manifest["model"]["output_semantics"],
                top_k=manifest["top_k"],
                labels=labels,
            )
            warmup_forward_ms.append(forward_ms)

    image_receipts = []
    for rgb, provenance in decoded_images:
        row = _run_one(
            runner,
            rgb,
            provenance=provenance,
            output_semantics=manifest["model"]["output_semantics"],
            top_k=manifest["top_k"],
            labels=labels,
        )
        image_receipts.append(row)

    forward_timings = [
        float(row["timing"]["bpu_forward_ms"]) for row in image_receipts
    ]
    total_timings = [float(row["timing"]["total_ms"]) for row in image_receipts]
    real_bpu = not runner.injected_test_backend
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": (
            "FAKE_BACKEND_TEST_PASS_NOT_BPU_EVIDENCE"
            if runner.injected_test_backend
            else "GENERIC_BPU_AUXILIARY_EVIDENCE_ONLY"
        ),
        "generated_at_utc": generated_at,
        "input_manifest": {
            "configured_path": str(configured_manifest),
            "resolved_path": str(resolved_manifest),
            "sha256": manifest_sha256,
            "size_bytes": manifest_size,
            "schema_version": manifest["schema_version"],
            "run_id": manifest["run_id"],
            "explicit_image_count": len(manifest["images"]),
            "directory_or_camera_discovery": False,
        },
        "runtime": {
            "backend": RUNTIME_BACKEND,
            "backend_actual": (
                "CALLER_INJECTED_FAKE_DNN_FOR_UNIT_TEST"
                if runner.injected_test_backend
                else RUNTIME_BACKEND
            ),
            "injected_test_backend": runner.injected_test_backend,
            "model_loaded": True,
            "bpu_forward_executed": real_bpu,
            "fake_forward_executed": runner.injected_test_backend,
            "model_load_ms": runner.model_load_ms,
            "measured_forward_count": len(image_receipts),
            "warmup_forward_count": len(warmup_forward_ms),
        },
        "model": {
            "name": runner.model_name,
            "path": str(runner.model_path),
            "sha256": runner.model_sha256,
            "size_bytes": runner.model_size_bytes,
            "vendor_platform": "Bayes-e RDK X5 BPU",
            "task": MODEL_TASK,
            "class_count": CLASS_COUNT,
            "output_semantics_declared": manifest["model"]["output_semantics"],
            "input_metadata": runner.input_metadata,
            "output_metadata": runner.output_metadata,
            "label_artifact": labels_receipt,
        },
        "frozen_preprocess_contract": {
            "input_source": "EXPLICIT_HASH_BOUND_IMAGE_FILES_ONLY",
            "decode": "PIL_EXIF_TRANSPOSE_RGB",
            "resize": [INPUT_WIDTH, INPUT_HEIGHT],
            "resize_policy": "DIRECT_RESIZE",
            "resize_interpolation": "PIL_BILINEAR",
            "runtime_color_format": "NV12",
            "yuv_conversion": "BT.601_LIMITED_RANGE_INTEGER",
            "runtime_dtype": "uint8",
            "runtime_shape": [INPUT_WIDTH * INPUT_HEIGHT * 3 // 2],
        },
        "warmup": {
            "source_image_id": (
                manifest["images"][0]["image_id"] if warmup_forward_ms else None
            ),
            "forward_ms": warmup_forward_ms,
            "excluded_from_measured_summary": True,
        },
        "images": image_receipts,
        "timing_summary": {
            "bpu_forward": _timing_summary(forward_timings),
            "end_to_end": _timing_summary(total_timings),
            "scope": (
                "FAKE_UNIT_TEST_TIMING_NOT_PERFORMANCE_EVIDENCE"
                if runner.injected_test_backend
                else "THIS_MANIFEST_ON_THIS_RUNTIME_ONLY"
            ),
        },
        "claims": {
            "generic_imagenet_class_indices_emitted": True,
            "plant_classification": False,
            "plant_species_identification": False,
            "plant_domain_ood_model_qualified": False,
            "ood_threshold_calibrated": False,
            "rootscope_classifier_model_qualified": False,
            "rootscope_classifier_selected_bin": None,
            "rootscope_classifier_selected_bin_remains_null": True,
            "production_integration_allowed": False,
            "physical_completion": False,
        },
        "integration_boundary": {
            "standalone_evidence_probe": True,
            "safety_compiler_imported": False,
            "safety_compiler_influence": False,
            "state_machine_imported": False,
            "decision_or_actuation_output": False,
            "semantic_role": "GENERIC_IMAGENET1000_AUXILIARY_DESCRIPTOR_ONLY",
            "ood_role": "UNCALIBRATED_NUMERIC_DESCRIPTORS_ONLY",
        },
        "effects_and_authority": {
            "bpu_hardware_touched": real_bpu,
            "camera_opened": False,
            "camera_frames_read": 0,
            "device_enumerated": False,
            "network_touched": False,
            "serial_write": False,
            "gpio_write": False,
            "pump_command": False,
            "tool_call": False,
            "state_machine_write": False,
            "execution_authority": False,
            "physical_authority": False,
        },
        "claim_boundary": (
            "This receipt proves only hash-bound generic ImageNet-1000 "
            "MobileNetV2 numeric output and, on a non-injected runtime, the "
            "listed BPU forwards. Entropy, energy and maximum probability are "
            "uncalibrated descriptors, not plant-domain OOD decisions. Generic "
            "class IDs or labels are not RootScope plant classifications. The "
            "probe cannot influence the Safety Compiler or any physical action; "
            "the RootScope classifier selected_bin remains null."
        ),
        "provenance": {
            "xrd_source_copied_or_executed": False,
            "reference_lineage": (
                "AdventureX clean-room implementation; frozen XRD "
                "workstation/dual_arm/overhead_bpu_aux_probe_x5.py was read-only "
                "architectural reference only"
            ),
        },
    }
    fingerprint = dict(receipt)
    fingerprint.pop("generated_at_utc")
    receipt["receipt_sha256"] = _canonical_sha256(fingerprint)
    return receipt


def write_receipt_exclusive(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                receipt,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
    except FileExistsError as exc:
        raise BpuAuxProbeError(f"refusing to overwrite existing receipt: {output}") from exc
    return output.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the zero-authority RootScope generic MobileNetV2 BPU evidence "
            "probe against one explicit hash-bound manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        receipt = run_manifest_probe(args.manifest)
        output = write_receipt_exclusive(args.out, receipt)
    except BpuAuxProbeError as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR_FAIL_CLOSED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "receipt_written": False,
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_path": str(output),
                "receipt_sha256": receipt["receipt_sha256"],
                "bpu_forward_executed": receipt["runtime"]["bpu_forward_executed"],
                "claim_boundary": receipt["claim_boundary"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
