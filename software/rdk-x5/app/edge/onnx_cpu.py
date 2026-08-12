"""Model-agnostic static image ONNX runner using CPU only.

No fallback provider is accepted.  A caller must provide an expected model
SHA-256 and a complete preprocessing contract.  The deterministic simulated
input is useful for clean-board replay but is not an accuracy test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .capsule import CPU_PROVIDER, PreprocessConfig


class OnnxCpuContractError(RuntimeError):
    """The model/runtime does not match the frozen CPU-only contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_simulated_rgb(height: int, width: int) -> np.ndarray:
    """Return a deterministic RGB uint8 frame without camera access."""

    if isinstance(height, bool) or isinstance(width, bool) or height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")
    y, x = np.indices((height, width), dtype=np.uint32)
    red = (x * 17 + y * 3 + 19) % 256
    green = (x * 5 + y * 11 + ((x // 8 + y // 8) % 2) * 53) % 256
    blue = (x * 7 + y * 13 + 101) % 256
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def preprocess_rgb(image: np.ndarray, contract: PreprocessConfig) -> np.ndarray:
    """Match training ``build_transforms(train=False)`` without torchvision.

    Semantics are ``Resize(shorter_side, PIL bilinear) -> CenterCrop ->
    RGB float tensor -> normalize``.  Integer output sizing and Python's
    ``round`` intentionally match torchvision's PIL functional path.
    """

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise OnnxCpuContractError("input image must have shape HxWx3")
    if array.dtype != np.uint8:
        raise OnnxCpuContractError("input image must be uint8 RGB")
    source_height, source_width = array.shape[:2]
    source_short = min(source_height, source_width)
    source_long = max(source_height, source_width)
    resized_long = int(contract.short_side * source_long / source_short)
    if source_width <= source_height:
        resized_width, resized_height = contract.short_side, resized_long
    else:
        resized_width, resized_height = resized_long, contract.short_side
    pil = Image.fromarray(array, mode="RGB")
    resized = pil.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    crop_height, crop_width = contract.center_crop
    crop_top = int(round((resized_height - crop_height) / 2.0))
    crop_left = int(round((resized_width - crop_width) / 2.0))
    cropped = resized.crop(
        (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
    )
    tensor = np.asarray(cropped, dtype=np.float32) * np.float32(contract.scale)
    mean = np.asarray(contract.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(contract.std, dtype=np.float32).reshape(1, 1, 3)
    tensor = (tensor - mean) / std
    return np.transpose(tensor, (2, 0, 1))[None, ...].astype(np.float32, copy=False)


class CpuOnnxRunner:
    """One-input image ONNX runner with an explicit CPU-only session."""

    def __init__(
        self,
        model_path: str | Path,
        expected_sha256: str,
        preprocess: PreprocessConfig,
        *,
        input_name: str | None = None,
        output_name: str | None = None,
        expected_output_shape: tuple[int, int],
        class_order: tuple[str, ...],
        ort_module: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.preprocess = preprocess
        if not self.model_path.is_file():
            raise OnnxCpuContractError(f"model file is missing: {self.model_path}")
        actual_hash = sha256_file(self.model_path)
        if actual_hash != expected_sha256:
            raise OnnxCpuContractError(
                f"model SHA-256 mismatch: actual={actual_hash} expected={expected_sha256}"
            )
        if ort_module is None:
            try:
                import onnxruntime as ort_module  # type: ignore
            except ImportError as exc:
                raise OnnxCpuContractError("onnxruntime is not installed") from exc
        available = list(ort_module.get_available_providers())
        if CPU_PROVIDER not in available:
            raise OnnxCpuContractError("CPUExecutionProvider is unavailable")
        self._session = ort_module.InferenceSession(
            str(self.model_path), providers=[CPU_PROVIDER]
        )
        actual_providers = list(self._session.get_providers())
        if actual_providers != [CPU_PROVIDER]:
            raise OnnxCpuContractError(
                f"session providers must be CPU-only, got {actual_providers}"
            )
        inputs = list(self._session.get_inputs())
        if input_name is None:
            if len(inputs) != 1:
                raise OnnxCpuContractError(
                    "input_name is required when the model has multiple inputs"
                )
            selected_input = inputs[0]
        else:
            matches = [item for item in inputs if item.name == input_name]
            if len(matches) != 1:
                raise OnnxCpuContractError(f"configured input_name not found: {input_name}")
            selected_input = matches[0]
        if list(selected_input.shape) != list(preprocess.input_shape):
            raise OnnxCpuContractError(
                f"model input shape {selected_input.shape} does not match "
                f"contract {preprocess.input_shape}"
            )
        if getattr(selected_input, "type", "tensor(float)") != "tensor(float)":
            raise OnnxCpuContractError("model input must be tensor(float)")
        outputs = list(self._session.get_outputs())
        if not outputs:
            raise OnnxCpuContractError("model has no outputs")
        if output_name is None:
            if len(outputs) != 1:
                raise OnnxCpuContractError(
                    "output_name is required when the model has multiple outputs"
                )
            selected_output = outputs[0]
        else:
            matches = [item for item in outputs if item.name == output_name]
            if len(matches) != 1:
                raise OnnxCpuContractError(
                    f"configured output_name not found: {output_name}"
                )
            selected_output = matches[0]
        if list(selected_output.shape) != list(expected_output_shape):
            raise OnnxCpuContractError(
                f"model output shape {selected_output.shape} does not match "
                f"contract {expected_output_shape}"
            )
        if getattr(selected_output, "type", "tensor(float)") != "tensor(float)":
            raise OnnxCpuContractError("model output must be tensor(float)")
        if expected_output_shape != (1, len(class_order)):
            raise OnnxCpuContractError(
                "class_order length must match static output dimension"
            )
        self.input_name = selected_input.name
        self.output_name = selected_output.name
        self.expected_output_shape = expected_output_shape
        self.class_order = class_order
        self.model_sha256 = actual_hash
        self.providers = actual_providers

    def run_rgb(self, image: np.ndarray) -> Mapping[str, Any]:
        tensor = preprocess_rgb(image, self.preprocess)
        values = self._session.run([self.output_name], {self.input_name: tensor})
        if not values:
            raise OnnxCpuContractError("ONNX session returned no values")
        output = np.asarray(values[0])
        if output.size == 0 or not np.isfinite(output).all():
            raise OnnxCpuContractError("ONNX output is empty or non-finite")
        if tuple(output.shape) != self.expected_output_shape:
            raise OnnxCpuContractError(
                f"runtime output shape {output.shape} does not match "
                f"contract {self.expected_output_shape}"
            )
        prediction_index = int(output.argmax(axis=1)[0])
        return {
            "schema_version": "rootscope.onnx-cpu-selftest.v1",
            "status": "SIMULATED_INPUT_CPU_ONNX_PASS_NOT_ACCURACY_EVIDENCE",
            "model_sha256": self.model_sha256,
            "provider_requested": CPU_PROVIDER,
            "providers_actual": list(self.providers),
            "input_name": self.input_name,
            "input_shape": list(tensor.shape),
            "input_tensor_sha256": hashlib.sha256(tensor.tobytes(order="C")).hexdigest(),
            "output_name": self.output_name,
            "output_shape": list(output.shape),
            "class_order": list(self.class_order),
            "simulated_prediction_index": prediction_index,
            "simulated_prediction_class": self.class_order[prediction_index],
            "output_tensor_sha256": hashlib.sha256(output.tobytes(order="C")).hexdigest(),
            "output_finite": True,
            "hardware_touched": False,
            "network_touched": False,
            "x5_validated": False,
            "bpu_ready": False,
            "bpu_used": False,
            "model_candidate": False,
            "model_qualified": False,
            "execution_authority": False,
            "physical_authority": False,
            "physical_completion": False,
        }
