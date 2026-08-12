"""Fail-closed, zero-authority RootScope seed17 BPU inference adapter.

This module deliberately does not import ``hobot_dnn`` at import time.  A
real backend is imported only while constructing :class:`Seed17BpuRunner`,
after the caller-supplied ``.bin`` has passed its SHA-256 check.  Supplying a
backend object is supported solely so unit tests can exercise the contract
without pretending that a BPU was used.

The Bayes-e runtime input contract is one contiguous ``uint8`` RGB tensor in
NCHW layout with shape ``[1, 3, 224, 224]``.  Geometry matches the frozen
evaluation transform: resize the shorter side to 256 with PIL bilinear
resampling and then take a 224x224 centre crop.  ImageNet mean/scale are
compiled into the BPU model and therefore must not be applied on the host.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

CLASS_ORDER: tuple[str, ...] = (
    "grass_clump",
    "low_shrub",
    "young_tree",
    "unknown",
)
INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_SHAPE = (1, 4)
SHORT_SIDE = 256
CROP_SIZE = (224, 224)
RUNTIME_BACKEND = "hobot_dnn.pyeasy_dnn"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Seed17BpuContractError(RuntimeError):
    """The artifact, runtime interface, input, or output violates the contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise Seed17BpuContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def preprocess_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Convert one BGR frame to the frozen RGB/NCHW/DDR runtime tensor.

    Host-side normalization is intentionally absent.  The returned array is
    always C-contiguous ``uint8`` with shape ``[1, 3, 224, 224]``.
    """

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise Seed17BpuContractError("input frame must have shape HxWx3")
    if array.dtype != np.uint8:
        raise Seed17BpuContractError("input frame must be uint8 BGR")
    source_height, source_width = array.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise Seed17BpuContractError("input frame dimensions must be positive")

    # OpenCV supplies BGR.  Make a positive-stride RGB copy before PIL owns it.
    rgb = np.ascontiguousarray(array[:, :, ::-1])
    source_short = min(source_height, source_width)
    source_long = max(source_height, source_width)
    resized_long = int(SHORT_SIDE * source_long / source_short)
    if source_width <= source_height:
        resized_width, resized_height = SHORT_SIDE, resized_long
    else:
        resized_width, resized_height = resized_long, SHORT_SIDE

    pil = Image.fromarray(rgb, mode="RGB")
    resized = pil.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    crop_height, crop_width = CROP_SIZE
    crop_top = int(round((resized_height - crop_height) / 2.0))
    crop_left = int(round((resized_width - crop_width) / 2.0))
    cropped = resized.crop(
        (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
    )
    hwc = np.asarray(cropped, dtype=np.uint8)
    tensor = np.ascontiguousarray(np.transpose(hwc, (2, 0, 1))[None, ...])
    if tensor.shape != INPUT_SHAPE or tensor.dtype != np.uint8 or not tensor.flags.c_contiguous:
        raise Seed17BpuContractError("preprocessor did not produce contiguous uint8 NCHW")
    return tensor


def load_hash_bound_image_bgr(
    path: str | Path, expected_sha256: str
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Read one explicitly named, hash-bound image file as BGR uint8."""

    expected = _require_sha256(expected_sha256, "expected image SHA-256")
    configured = Path(path).expanduser()
    if configured.is_symlink():
        raise Seed17BpuContractError("golden image must not be a symlink")
    resolved = configured.resolve(strict=True)
    if not resolved.is_file():
        raise Seed17BpuContractError("golden image is not a regular file")
    actual = sha256_file(resolved)
    if actual != expected:
        raise Seed17BpuContractError(
            f"golden image SHA-256 mismatch: actual={actual} expected={expected}"
        )
    with Image.open(resolved) as raw:
        rgb = np.asarray(ImageOps.exif_transpose(raw).convert("RGB"), dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return bgr, {
        "source_kind": "HASH_BOUND_IMAGE_FILE",
        "source_path": str(resolved),
        "source_file_sha256": actual,
        "source_file_bytes": resolved.stat().st_size,
        "camera_opened": False,
        "camera_frames_read": 0,
    }


def capture_one_explicit_v4l2_bgr(
    device: str | Path, *, cv2_module: Any | None = None
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Open exactly one explicit ``/dev/...`` alias and read exactly one frame.

    The function never scans or enumerates devices.  It is Linux-only and
    rejects aliases that resolve outside ``/dev`` or do not identify a
    character device.  OpenCV is imported only after those checks pass.
    """

    import platform
    import stat

    if platform.system() != "Linux":
        raise Seed17BpuContractError("explicit V4L2 capture requires Linux")
    configured = Path(device).expanduser()
    if not configured.is_absolute() or configured.parts[:2] != ("/", "dev"):
        raise Seed17BpuContractError("camera device must be one absolute path below /dev")
    resolved = configured.resolve(strict=True)
    if resolved.parts[:2] != ("/", "dev"):
        raise Seed17BpuContractError("camera alias must resolve below /dev")
    if not stat.S_ISCHR(resolved.stat().st_mode):
        raise Seed17BpuContractError("camera alias must resolve to a character device")
    if cv2_module is None:
        try:
            import cv2 as cv2_module  # type: ignore
        except ImportError as exc:
            raise Seed17BpuContractError("OpenCV is required for explicit V4L2 capture") from exc

    capture = cv2_module.VideoCapture(str(configured), cv2_module.CAP_V4L2)
    if not bool(capture.isOpened()):
        capture.release()
        raise Seed17BpuContractError("explicit V4L2 camera could not be opened")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise Seed17BpuContractError("explicit V4L2 camera did not return one frame")
    bgr = np.asarray(frame)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise Seed17BpuContractError("captured frame must be uint8 BGR HxWx3")
    bgr = np.ascontiguousarray(bgr)
    return bgr, {
        "source_kind": "EXPLICIT_V4L2_ONE_FRAME",
        "configured_device": str(configured),
        "resolved_device": str(resolved),
        "device_enumerated": False,
        "camera_opened": True,
        "camera_frames_read": 1,
        "source_bgr_sha256": hashlib.sha256(bgr.tobytes(order="C")).hexdigest(),
    }


def _property_value(tensor: Any, name: str) -> Any:
    properties = getattr(tensor, "properties", None)
    if properties is not None and hasattr(properties, name):
        return getattr(properties, name)
    if hasattr(tensor, name):
        return getattr(tensor, name)
    return None


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    for attribute in ("dimensionSize", "dims"):
        if hasattr(value, attribute):
            value = getattr(value, attribute)
            break
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        tokens = re.findall(r"\d+", value)
        if not tokens:
            return None
        value = tokens
    try:
        items = list(value)
    except TypeError:
        return None
    result: list[int] = []
    for item in items:
        if isinstance(item, bool):
            return None
        try:
            number = int(item)
        except (TypeError, ValueError):
            return None
        if number <= 0:
            return None
        result.append(number)
    return tuple(result) if result else None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _tensor_metadata(tensor: Any) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "name",
        "dtype",
        "layout",
        "shape",
        "validShape",
        "alignedShape",
        "tensor_type",
    ):
        result[name] = _json_value(_property_value(tensor, name))
    buffer = getattr(tensor, "buffer", None)
    if buffer is not None:
        array = np.asarray(buffer)
        result["buffer_shape"] = list(array.shape)
        result["buffer_dtype"] = str(array.dtype)
        result["buffer_c_contiguous"] = bool(array.flags.c_contiguous)
    else:
        result["buffer_shape"] = None
        result["buffer_dtype"] = None
        result["buffer_c_contiguous"] = None
    return result


def _effective_shape(tensor: Any) -> tuple[int, ...] | None:
    for field in ("validShape", "shape"):
        shape = _shape_tuple(_property_value(tensor, field))
        if shape is not None:
            return shape
    buffer = getattr(tensor, "buffer", None)
    return tuple(np.asarray(buffer).shape) if buffer is not None else None


def _input_is_uint8_rgb(tensor: Any, metadata: Mapping[str, Any]) -> bool:
    buffer_dtype = metadata.get("buffer_dtype")
    if buffer_dtype is not None:
        return buffer_dtype == "uint8"
    dtype = str(_property_value(tensor, "dtype") or "").upper()
    tensor_type = str(_property_value(tensor, "tensor_type") or "").upper()
    combined = f"{dtype} {tensor_type}"
    return "UINT8" in combined or "RGB" in combined


def _zero_authority(*, hardware_touched: bool, bpu_used: bool) -> Mapping[str, bool]:
    return {
        "hardware_touched": hardware_touched,
        "network_touched": False,
        "device_enumerated": False,
        "serial_write": False,
        "state_machine_write": False,
        "pump_command": False,
        "irrigation_execution": False,
        "execution_authority": False,
        "physical_authority": False,
        "physical_completion": False,
        "bpu_used": bpu_used,
    }


def _frozen_claims() -> Mapping[str, bool]:
    return {
        "x5_ready": False,
        "x5_validated": False,
        "camera_qualified": False,
        "model_candidate": False,
        "model_qualified": False,
        "production_integration_allowed": False,
        "production_authority_enabled": False,
        "irrigation_authority_enabled": False,
    }


class Seed17BpuRunner:
    """Hash-bound one-model pyeasy_dnn runner with a frozen interface."""

    def __init__(
        self,
        model_path: str | Path,
        expected_sha256: str,
        *,
        class_order: Sequence[str] = CLASS_ORDER,
        dnn_module: Any | None = None,
    ) -> None:
        expected = _require_sha256(expected_sha256, "expected model SHA-256")
        configured = Path(model_path).expanduser()
        if configured.is_symlink():
            raise Seed17BpuContractError("BPU model must not be a symlink")
        try:
            resolved = configured.resolve(strict=True)
        except FileNotFoundError as exc:
            raise Seed17BpuContractError(f"BPU model is missing: {configured}") from exc
        if not resolved.is_file() or resolved.suffix.lower() != ".bin":
            raise Seed17BpuContractError("BPU model must be one regular .bin file")
        actual = sha256_file(resolved)
        if actual != expected:
            raise Seed17BpuContractError(
                f"BPU model SHA-256 mismatch: actual={actual} expected={expected}"
            )
        if tuple(class_order) != CLASS_ORDER:
            raise Seed17BpuContractError("class order must match the frozen seed17 order")

        self.runtime_injected = dnn_module is not None
        if dnn_module is None:
            try:
                from hobot_dnn import pyeasy_dnn as dnn_module  # type: ignore
            except ImportError as exc:
                raise Seed17BpuContractError(
                    "hobot_dnn.pyeasy_dnn is unavailable; no fake fallback is allowed"
                ) from exc
        try:
            models = dnn_module.load(str(resolved))
        except Exception as exc:
            raise Seed17BpuContractError(f"pyeasy_dnn model load failed: {exc}") from exc
        if not isinstance(models, (list, tuple)) or len(models) != 1:
            count = len(models) if isinstance(models, (list, tuple)) else "non-sequence"
            raise Seed17BpuContractError(f"BPU bin must expose exactly one model, got {count}")
        model = models[0]
        inputs = list(getattr(model, "inputs", ()))
        outputs = list(getattr(model, "outputs", ()))
        if len(inputs) != 1 or len(outputs) != 1:
            raise Seed17BpuContractError(
                f"BPU model must expose one input and one output, got {len(inputs)}/{len(outputs)}"
            )

        input_metadata = _tensor_metadata(inputs[0])
        output_metadata = _tensor_metadata(outputs[0])
        input_shape = _effective_shape(inputs[0])
        output_shape = _effective_shape(outputs[0])
        if input_shape != INPUT_SHAPE:
            raise Seed17BpuContractError(
                f"BPU input shape must be {INPUT_SHAPE}, got {input_shape}"
            )
        if output_shape != OUTPUT_SHAPE:
            raise Seed17BpuContractError(
                f"BPU output shape must be {OUTPUT_SHAPE}, got {output_shape}"
            )
        layout = str(_property_value(inputs[0], "layout") or "").upper()
        if "NCHW" not in layout:
            raise Seed17BpuContractError(f"BPU input layout must be NCHW, got {layout!r}")
        if not _input_is_uint8_rgb(inputs[0], input_metadata):
            raise Seed17BpuContractError(
                "BPU input metadata/buffer does not identify a uint8 RGB runtime input"
            )

        self.model_path = resolved
        self.model_sha256 = actual
        self.model_size_bytes = resolved.stat().st_size
        self.model = model
        self.class_order = CLASS_ORDER
        self.input_metadata = input_metadata
        self.output_metadata = output_metadata

    def _base_report(self, *, forward_executed: bool, camera_opened: bool) -> dict[str, Any]:
        real_runtime = not self.runtime_injected
        return {
            "schema": "rootscope.seed17-bpu-isolated-replay.v1",
            "runtime": {
                "backend": RUNTIME_BACKEND,
                "injected_test_backend": self.runtime_injected,
                "model_loaded": True,
                "forward_executed": forward_executed,
                "camera_opened": camera_opened,
                "evidence_scope": (
                    "FAKE_DNN_UNIT_TEST_ONLY"
                    if self.runtime_injected
                    else "MANUAL_ISOLATED_X5_RUNTIME_ONLY"
                ),
            },
            "model": {
                "path": str(self.model_path),
                "size_bytes": self.model_size_bytes,
                "sha256": self.model_sha256,
                "class_order": list(self.class_order),
            },
            "interface": {
                "input_shape": list(INPUT_SHAPE),
                "input_dtype": "uint8",
                "input_color_order": "RGB",
                "input_layout": "NCHW",
                "input_source": "DDR",
                "output_shape": list(OUTPUT_SHAPE),
                "input_metadata": dict(self.input_metadata),
                "output_metadata": dict(self.output_metadata),
            },
            "preprocess": {
                "source_color_order": "BGR",
                "short_side": SHORT_SIDE,
                "center_crop": list(CROP_SIZE),
                "resize_interpolation": "PIL_BILINEAR",
                "host_color_conversion": "BGR_TO_RGB",
                "host_normalization": False,
                "compiled_model_mean_scale": True,
                "output_contiguous": True,
            },
            "claims": dict(_frozen_claims()),
            "authority": dict(
                _zero_authority(
                    hardware_touched=real_runtime or camera_opened,
                    bpu_used=real_runtime and forward_executed,
                )
            ),
        }

    def preflight_report(self) -> Mapping[str, Any]:
        report = self._base_report(forward_executed=False, camera_opened=False)
        report["status"] = (
            "FAKE_DNN_INTERFACE_PASS_NOT_BPU_EVIDENCE"
            if self.runtime_injected
            else "HASH_AND_INTERFACE_PREFLIGHT_PASS_NOT_X5_OR_MODEL_QUALIFICATION"
        )
        report["inference"] = None
        return report

    def run_bgr(
        self,
        image: np.ndarray,
        *,
        source_provenance: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        tensor = preprocess_bgr_uint8(image)
        try:
            values = self.model.forward(tensor)
        except Exception as exc:
            raise Seed17BpuContractError(f"pyeasy_dnn forward failed: {exc}") from exc
        if not isinstance(values, (list, tuple)) or len(values) != 1:
            count = len(values) if isinstance(values, (list, tuple)) else "non-sequence"
            raise Seed17BpuContractError(f"BPU forward must return one output, got {count}")
        buffer = getattr(values[0], "buffer", values[0])
        logits_raw = np.asarray(buffer)
        if logits_raw.shape != OUTPUT_SHAPE:
            raise Seed17BpuContractError(
                f"BPU runtime output shape must be {OUTPUT_SHAPE}, got {logits_raw.shape}"
            )
        if not np.issubdtype(logits_raw.dtype, np.number):
            raise Seed17BpuContractError("BPU runtime output must be numeric logits")
        logits = logits_raw.astype(np.float32, copy=False)
        if not np.isfinite(logits).all():
            raise Seed17BpuContractError("BPU runtime logits must all be finite")

        shifted = logits[0] - np.max(logits[0])
        probabilities = np.exp(shifted)
        probabilities = probabilities / probabilities.sum()
        prediction_index = int(np.argmax(logits[0]))
        provenance = dict(source_provenance or {"source_kind": "CALLER_PROVIDED_BGR"})
        camera_opened = provenance.get("camera_opened") is True
        report = self._base_report(forward_executed=True, camera_opened=camera_opened)
        report["status"] = (
            "FAKE_DNN_REPLAY_PASS_NOT_BPU_EVIDENCE"
            if self.runtime_injected
            else (
                "ISOLATED_ONE_FRAME_BPU_REPLAY_PASS_NOT_CAMERA_QUALIFICATION"
                if camera_opened
                else "ISOLATED_HASH_BOUND_IMAGE_BPU_REPLAY_PASS_NOT_MODEL_QUALIFICATION"
            )
        )
        provenance.update(
            {
                "source_bgr_shape": list(np.asarray(image).shape),
                "source_bgr_dtype": str(np.asarray(image).dtype),
                "source_bgr_sha256": hashlib.sha256(
                    np.ascontiguousarray(image).tobytes(order="C")
                ).hexdigest(),
            }
        )
        report["input_provenance"] = provenance
        report["inference"] = {
            "input_tensor_shape": list(tensor.shape),
            "input_tensor_dtype": str(tensor.dtype),
            "input_tensor_c_contiguous": bool(tensor.flags.c_contiguous),
            "input_tensor_sha256": hashlib.sha256(tensor.tobytes(order="C")).hexdigest(),
            "output_shape": list(logits.shape),
            "output_dtype_observed": str(logits_raw.dtype),
            "output_finite": True,
            "logits": [float(item) for item in logits[0]],
            "probabilities": [float(item) for item in probabilities],
            "raw_top1_index": prediction_index,
            "raw_top1_class": self.class_order[prediction_index],
            "semantic_scope": "RAW_TOP1_HYPOTHESIS_NOT_OPEN_WORLD_ACCURACY_EVIDENCE",
        }
        return report


__all__ = [
    "CLASS_ORDER",
    "CROP_SIZE",
    "INPUT_SHAPE",
    "OUTPUT_SHAPE",
    "SHORT_SIDE",
    "Seed17BpuContractError",
    "Seed17BpuRunner",
    "capture_one_explicit_v4l2_bgr",
    "load_hash_bound_image_bgr",
    "preprocess_bgr_uint8",
    "sha256_file",
]
