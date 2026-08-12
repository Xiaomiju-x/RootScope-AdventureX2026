"""Shared same-tensor CPU audit and r7 BPU shadow proposal core.

The CPU path remains the displayed/audited result.  The unqualified r7 BPU
path is a proposal-only shadow; a missing or invalid AF_UNIX response falls
back to the same CPU logits.  This module has no camera, serial, GPIO, pump,
state-machine, or network interface.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from app.edge.bpu_seed17 import preprocess_bgr_uint8

from .bpu_shadow_protocol import (
    MAX_BATCH,
    OUTPUT_SHAPE,
    TENSOR_SHAPE,
    ZERO_AUTHORITY,
    tensor_sha256,
    validate_logits,
)

CLASS_ORDER = ("grass_clump", "low_shrub", "young_tree", "unknown")


class ShadowClient(Protocol):
    def infer_tensors(
        self,
        tensors: Sequence[np.ndarray],
        *,
        cpu_fallback: Any,
    ) -> Mapping[str, Any]:
        """Return a BPU shadow or CPU fallback client receipt."""


def rgb_to_bpu_tensor(rgb: np.ndarray) -> np.ndarray:
    """Create the compiled uint8 RGB NCHW tensor from one RGB image."""

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("image must be uint8 RGB HxWx3")
    # The frozen BPU helper receives OpenCV BGR and performs BGR->RGB before
    # resize/crop.  Reversing the caller RGB here therefore preserves RGB.
    bgr = np.ascontiguousarray(array[:, :, ::-1])
    tensor = preprocess_bgr_uint8(bgr)
    if tuple(tensor.shape) != TENSOR_SHAPE:
        raise RuntimeError("BPU preprocessor contract changed")
    return tensor


class CpuTensorAudit:
    """Run the frozen CPU ONNX model on the exact BPU uint8 tensor geometry."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        if list(getattr(runner, "providers", ())) != ["CPUExecutionProvider"]:
            raise ValueError("CPU audit runner must expose CPUExecutionProvider only")
        self.class_order = tuple(getattr(runner, "class_order", ()))
        if self.class_order != CLASS_ORDER:
            raise ValueError("CPU audit runner class order changed")
        preprocess = runner.preprocess
        self.scale = np.float32(preprocess.scale)
        self.mean = np.asarray(preprocess.mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(preprocess.std, dtype=np.float32).reshape(1, 3, 1, 1)

    def normalize(self, tensor: np.ndarray) -> np.ndarray:
        if tuple(tensor.shape) != TENSOR_SHAPE or tensor.dtype != np.uint8:
            raise ValueError("CPU audit input must be uint8 RGB NCHW [1,3,224,224]")
        normalized = (tensor.astype(np.float32) * self.scale - self.mean) / self.std
        return np.ascontiguousarray(normalized, dtype=np.float32)

    def run_one(self, tensor: np.ndarray) -> Mapping[str, Any]:
        normalized = self.normalize(tensor)
        started = time.perf_counter_ns()
        values = self.runner._session.run(
            [self.runner.output_name],
            {self.runner.input_name: normalized},
        )
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if not isinstance(values, (list, tuple)) or len(values) != 1:
            raise RuntimeError("CPU ONNX audit must return exactly one output")
        logits = validate_logits(np.asarray(values[0]))
        top1_index = int(np.argmax(np.asarray(logits)))
        return {
            "backend": "CPUExecutionProvider",
            "role": "AUDIT_AND_FALLBACK_PRIMARY",
            "normalized_tensor_sha256": hashlib.sha256(
                normalized.tobytes(order="C")
            ).hexdigest(),
            "logits": logits,
            "top1_index": top1_index,
            "top1_class": CLASS_ORDER[top1_index],
            "latency_ms": latency_ms,
        }


class PlantCpuBpuReplay:
    """Generate immutable-friendly CPU/BPU receipts for batches of 1-4 images."""

    def __init__(self, runner: Any, bpu_client: ShadowClient) -> None:
        self.cpu = CpuTensorAudit(runner)
        self.bpu_client = bpu_client

    def infer_rgb_batch(self, images: Sequence[np.ndarray]) -> Mapping[str, Any]:
        if not isinstance(images, Sequence) or isinstance(
            images, (str, bytes, bytearray)
        ):
            raise ValueError("images must be a sequence")
        if not 1 <= len(images) <= MAX_BATCH:
            raise ValueError(f"image batch must contain 1-{MAX_BATCH} items")

        arrays = [np.asarray(image) for image in images]
        image_hashes = [
            hashlib.sha256(np.ascontiguousarray(image).tobytes(order="C")).hexdigest()
            for image in arrays
        ]
        tensors = [rgb_to_bpu_tensor(image) for image in arrays]
        tensor_hashes = [tensor_sha256(tensor) for tensor in tensors]
        cpu_audits = [dict(self.cpu.run_one(tensor)) for tensor in tensors]

        def _precomputed_cpu_fallback(
            fallback_tensors: Sequence[np.ndarray],
        ) -> list[list[float]]:
            fallback_hashes = [tensor_sha256(tensor) for tensor in fallback_tensors]
            if fallback_hashes != tensor_hashes:
                raise RuntimeError("CPU fallback tensor order/hash changed")
            return [list(item["logits"]) for item in cpu_audits]

        bpu_receipt = dict(
            self.bpu_client.infer_tensors(
                tensors,
                cpu_fallback=_precomputed_cpu_fallback,
            )
        )
        bpu_ok = bpu_receipt.get("status") == "BPU_SHADOW_OK"
        bpu_logits = bpu_receipt.get("logits") if bpu_ok else None
        bpu_results = bpu_receipt.get("bpu_results") if bpu_ok else None
        if bpu_ok:
            if not isinstance(bpu_logits, list) or len(bpu_logits) != len(images):
                raise RuntimeError("BPU shadow receipt result count changed")
            if not isinstance(bpu_results, list) or len(bpu_results) != len(images):
                raise RuntimeError("BPU shadow receipt latency/provenance is incomplete")

        rows: list[dict[str, Any]] = []
        for index, cpu_item in enumerate(cpu_audits):
            proposal: dict[str, Any]
            if bpu_ok:
                logits = validate_logits(bpu_logits[index])
                top1_index = int(np.argmax(np.asarray(logits)))
                worker_item = bpu_results[index]
                proposal = {
                    "available": True,
                    "backend": bpu_receipt.get("backend_actual"),
                    "role": "SHADOW_PROPOSAL_ONLY",
                    "qualification": "SHADOW_CANDIDATE_NOT_DEFAULT",
                    "logits": logits,
                    "top1_index": top1_index,
                    "top1_class": CLASS_ORDER[top1_index],
                    "latency_ms": worker_item.get("latency_ms"),
                    "input_sha256": worker_item.get("input_sha256"),
                }
                agreement: bool | None = top1_index == cpu_item["top1_index"]
            else:
                proposal = {
                    "available": False,
                    "backend": None,
                    "role": "SHADOW_PROPOSAL_ONLY",
                    "qualification": "SHADOW_CANDIDATE_NOT_DEFAULT",
                    "logits": None,
                    "top1_index": None,
                    "top1_class": None,
                    "latency_ms": None,
                    "input_sha256": tensor_hashes[index],
                    "fallback_status": bpu_receipt.get("status"),
                    "fallback": bpu_receipt.get("fallback"),
                }
                agreement = None
            rows.append(
                {
                    "index": index,
                    "decoded_rgb_sha256": image_hashes[index],
                    "input_tensor_sha256": tensor_hashes[index],
                    "input_tensor_shape": list(TENSOR_SHAPE),
                    "input_tensor_dtype": "uint8",
                    "cpu_audit": cpu_item,
                    "bpu_proposal": proposal,
                    "cpu_bpu_top1_agreement": agreement,
                    "display_source": "CPU_AUDIT",
                    "shadow_blocks_primary_display": False,
                }
            )

        return {
            "schema": "rootscope.plant-cpu-bpu-replay.v1",
            "status": (
                "CPU_AUDIT_WITH_BPU_SHADOW_PROPOSAL"
                if bpu_ok
                else "CPU_PRIMARY_BPU_SHADOW_FALLBACK"
            ),
            "batch_count": len(rows),
            "class_order": list(CLASS_ORDER),
            "primary_backend": "CPUExecutionProvider",
            "bpu_role": "SHADOW_PROPOSAL_ONLY",
            "bpu_qualification": "SHADOW_CANDIDATE_NOT_DEFAULT",
            "selected_bin_changed": False,
            "bpu_client_status": bpu_receipt.get("status"),
            "bpu_backend_actual": (
                bpu_receipt.get("backend_actual") if bpu_ok else None
            ),
            "bpu_backend_metadata": (
                bpu_receipt.get("backend") if bpu_ok else None
            ),
            "bpu_batch": bpu_receipt.get("bpu_batch") if bpu_ok else None,
            "rows": rows,
            "shadow_blocks_primary_display": False,
            "zero_authority": True,
            "authority": dict(ZERO_AUTHORITY),
        }


__all__ = [
    "CLASS_ORDER",
    "CpuTensorAudit",
    "PlantCpuBpuReplay",
    "ShadowClient",
    "rgb_to_bpu_tensor",
]
