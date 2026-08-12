from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from app.competition_runtime.bpu_shadow_protocol import (
    TENSOR_SHAPE,
    ZERO_AUTHORITY,
    tensor_sha256,
)
from app.competition_runtime.plant_cpu_bpu_replay import (
    CLASS_ORDER,
    CpuTensorAudit,
    PlantCpuBpuReplay,
    rgb_to_bpu_tensor,
)
from app.edge.onnx_cpu import preprocess_rgb


class _Session:
    def __init__(self, logits: list[float]) -> None:
        self.logits = np.asarray([logits], dtype=np.float32)
        self.calls: list[np.ndarray] = []

    def run(self, output_names, inputs):
        assert output_names == ["logits"]
        tensor = inputs["image"]
        self.calls.append(tensor.copy())
        return [self.logits.copy()]


class _Runner:
    def __init__(self, logits: list[float]) -> None:
        self.providers = ["CPUExecutionProvider"]
        self.class_order = CLASS_ORDER
        self.preprocess = SimpleNamespace(
            scale=1.0 / 255.0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            short_side=256,
            center_crop=(224, 224),
            input_shape=(1, 3, 224, 224),
        )
        self.input_name = "image"
        self.output_name = "logits"
        self._session = _Session(logits)


class _ShadowClient:
    def __init__(self, *, status: str, bpu_logits: list[float] | None = None) -> None:
        self.status = status
        self.bpu_logits = bpu_logits
        self.calls = 0

    def infer_tensors(self, tensors, *, cpu_fallback):
        self.calls += 1
        if self.status != "BPU_SHADOW_OK":
            fallback_logits = cpu_fallback(tensors)
            return {
                "status": "CPU_FALLBACK_OK",
                "backend_actual": "CPU_FALLBACK",
                "logits": fallback_logits,
                "bpu_batch": None,
                "bpu_results": None,
                "fallback": {"bpu_error": "fixture unavailable"},
            }
        assert self.bpu_logits is not None
        return {
            "status": "BPU_SHADOW_OK",
            "backend_actual": "FAKE_MODEL_UNIT_TEST_ONLY",
            "logits": [list(self.bpu_logits) for _ in tensors],
            "bpu_batch": {"latency_ms": 1.25},
            "bpu_results": [
                {
                    "index": index,
                    "input_sha256": tensor_sha256(tensor),
                    "logits": list(self.bpu_logits),
                    "latency_ms": 0.2 + index,
                }
                for index, tensor in enumerate(tensors)
            ],
            "fallback": None,
        }


def _image(seed: int) -> np.ndarray:
    y, x = np.indices((240, 320), dtype=np.uint16)
    return np.stack(
        (
            (x + seed) % 256,
            (y * 3 + seed) % 256,
            (x + y + seed * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def test_same_bpu_tensor_normalizes_identically_to_frozen_cpu_preprocess():
    runner = _Runner([0.1, 0.2, 0.3, 0.4])
    rgb = _image(7)
    bpu_tensor = rgb_to_bpu_tensor(rgb)
    assert bpu_tensor.shape == TENSOR_SHAPE
    assert bpu_tensor.dtype == np.uint8
    from_same_tensor = CpuTensorAudit(runner).normalize(bpu_tensor)
    cpu_reference = preprocess_rgb(rgb, runner.preprocess)
    np.testing.assert_allclose(from_same_tensor, cpu_reference, rtol=0.0, atol=1.0e-6)


def test_replay_records_image_tensor_logits_top1_latency_backend_and_agreement():
    runner = _Runner([0.0, 2.0, 1.0, -1.0])
    client = _ShadowClient(
        status="BPU_SHADOW_OK",
        bpu_logits=[0.0, 3.0, 1.0, -2.0],
    )
    images = [_image(1), _image(2), _image(3), _image(4)]
    receipt = PlantCpuBpuReplay(runner, client).infer_rgb_batch(images)
    assert receipt["status"] == "CPU_AUDIT_WITH_BPU_SHADOW_PROPOSAL"
    assert receipt["batch_count"] == 4
    assert receipt["bpu_backend_actual"] == "FAKE_MODEL_UNIT_TEST_ONLY"
    assert receipt["bpu_qualification"] == "SHADOW_CANDIDATE_NOT_DEFAULT"
    assert receipt["selected_bin_changed"] is False
    assert receipt["authority"] == ZERO_AUTHORITY
    for image, row in zip(images, receipt["rows"], strict=True):
        assert row["decoded_rgb_sha256"] == hashlib.sha256(
            image.tobytes(order="C")
        ).hexdigest()
        assert len(row["input_tensor_sha256"]) == 64
        assert row["cpu_audit"]["top1_class"] == "low_shrub"
        assert row["bpu_proposal"]["top1_class"] == "low_shrub"
        assert row["cpu_bpu_top1_agreement"] is True
        assert row["cpu_audit"]["latency_ms"] >= 0.0
        assert row["bpu_proposal"]["latency_ms"] is not None
        assert row["display_source"] == "CPU_AUDIT"
        assert row["shadow_blocks_primary_display"] is False


def test_bpu_disagreement_never_replaces_cpu_display():
    runner = _Runner([4.0, 0.0, 0.0, 0.0])
    client = _ShadowClient(
        status="BPU_SHADOW_OK",
        bpu_logits=[0.0, 0.0, 5.0, 0.0],
    )
    row = PlantCpuBpuReplay(runner, client).infer_rgb_batch([_image(8)])["rows"][0]
    assert row["cpu_audit"]["top1_class"] == "grass_clump"
    assert row["bpu_proposal"]["top1_class"] == "young_tree"
    assert row["cpu_bpu_top1_agreement"] is False
    assert row["display_source"] == "CPU_AUDIT"


def test_bpu_failure_uses_precomputed_cpu_fallback_and_preserves_zero_authority():
    runner = _Runner([0.0, 0.0, 0.0, 6.0])
    client = _ShadowClient(status="CPU_FALLBACK_OK")
    receipt = PlantCpuBpuReplay(runner, client).infer_rgb_batch([_image(9)])
    row = receipt["rows"][0]
    assert receipt["status"] == "CPU_PRIMARY_BPU_SHADOW_FALLBACK"
    assert receipt["bpu_client_status"] == "CPU_FALLBACK_OK"
    assert receipt["bpu_backend_actual"] is None
    assert row["cpu_audit"]["top1_class"] == "unknown"
    assert row["bpu_proposal"]["available"] is False
    assert row["cpu_bpu_top1_agreement"] is None
    assert row["display_source"] == "CPU_AUDIT"
    assert receipt["authority"] == ZERO_AUTHORITY
