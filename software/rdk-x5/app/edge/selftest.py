"""Deterministic simulated-input self-test for the X5 capsule."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from .capsule import CapsuleConfig
from .onnx_cpu import (
    CpuOnnxRunner,
    OnnxCpuContractError,
    make_simulated_rgb,
    preprocess_rgb,
)


def run_simulated_selftest(
    config: CapsuleConfig, *, ort_module: Any | None = None
) -> Mapping[str, Any]:
    contract = config.model.preprocess
    height, width, channels = contract.golden_source_shape
    if channels != 3:
        raise OnnxCpuContractError("golden source must be RGB")
    image = make_simulated_rgb(height, width)
    tensor = preprocess_rgb(image, contract)
    tensor_sha256 = hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
    if tensor_sha256 != contract.golden_tensor_sha256:
        raise OnnxCpuContractError(
            "preprocess golden mismatch: "
            f"actual={tensor_sha256} expected={contract.golden_tensor_sha256}"
        )
    base = {
        "schema_version": "rootscope.x5-simulated-selftest.v1",
        "simulated_rgb_sha256": hashlib.sha256(image.tobytes(order="C")).hexdigest(),
        "preprocess_mode": contract.mode,
        "preprocessed_tensor_sha256": tensor_sha256,
        "preprocess_golden_matched": True,
        "preprocessed_shape": list(tensor.shape),
        "preprocessed_finite": bool(np.isfinite(tensor).all()),
        "hardware_touched": False,
        "network_touched": False,
        "ports_enumerated": False,
        "x5_validated": False,
        "bpu_ready": False,
        "bpu_used": False,
        "model_candidate": False,
        "model_qualified": False,
        "physical_authority": False,
        "execution_authority": False,
        "physical_completion": False,
    }
    if not config.model.enabled:
        return {
            **base,
            "status": "PASS_PREPROCESS_ONLY_MODEL_DISABLED_SIMULATED_ONLY",
            "onnx_executed": False,
            "accuracy_evidence": False,
        }
    runner = CpuOnnxRunner(
        config.model.path,
        config.model.sha256 or "",
        contract,
        input_name=config.model.input_name,
        output_name=config.model.output_name,
        expected_output_shape=config.model.output_shape,
        class_order=config.model.class_order,
        ort_module=ort_module,
    )
    inference = runner.run_rgb(image)
    return {
        **base,
        "status": "PASS_CPU_ONNX_SIMULATED_INPUT_NOT_ACCURACY_EVIDENCE",
        "onnx_executed": True,
        "accuracy_evidence": False,
        "inference": inference,
    }
