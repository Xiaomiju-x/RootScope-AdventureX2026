from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pytest


ROOTSCOPE = Path(__file__).resolve().parents[1]
if str(ROOTSCOPE) not in sys.path:
    sys.path.insert(0, str(ROOTSCOPE))

from tools import x5_v3_live_camera_gate as gate


def test_frozen_runtime_hashes_match_packaged_assets() -> None:
    expected_capsule_sha256 = (
        "1b7e9b96ccd4ec4e5ab534e1f305224c3c6330a3ab2efb8ca2e5d0fc52fcfcbb"
    )
    expected_model_sha256 = (
        "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad"
    )
    capsule = (
        ROOTSCOPE
        / "deploy"
        / "x5"
        / "capsule_config.seed17_cpu_experimental.json"
    )
    model = (
        ROOTSCOPE
        / "deploy"
        / "x5"
        / "models"
        / "rootscope_seed17_cpu_experimental_opset11.onnx"
    )
    assert gate.FROZEN_CAPSULE_SHA256 == expected_capsule_sha256
    assert gate.FROZEN_MODEL_SHA256 == expected_model_sha256
    assert gate.sha256_file(capsule) == expected_capsule_sha256
    assert gate.sha256_file(model) == expected_model_sha256


def test_owner_probe_uses_x5_compatible_fuser_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    receipt = gate.probe_camera_owner("/dev/video0")
    assert observed["argv"] == ["fuser", "/dev/video0"]
    assert observed["kwargs"]["timeout"] == 3.0
    assert receipt == {
        "state": "NO_OWNER",
        "no_owner": True,
        "returncode": 1,
        "stdout": "",
        "stderr": "",
    }


class FakeSource:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.closed = False
        self.read_count = 0
        self.fail_at = fail_at

    def read_rgb(self) -> np.ndarray:
        if self.fail_at is not None and self.read_count == self.fail_at:
            raise RuntimeError("synthetic capture failure")
        self.read_count += 1
        y, x = np.indices((240, 320), dtype=np.uint16)
        return np.stack(
            (
                (x + self.read_count * 3) % 256,
                (y * 2 + 40) % 256,
                (x // 2 + y // 2 + 90) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)

    def negotiated_settings(self) -> Mapping[str, Any]:
        return {
            "backend": "fake_opencv_v4l2",
            "configured_device": gate.FROZEN_DEVICE,
            "resolved_device": "/dev/video0",
            "fourcc": "MJPG",
            "width": 320,
            "height": 240,
            "fps": 30.0,
        }

    def close(self) -> Mapping[str, Any]:
        self.closed = True
        return {
            "release_called": True,
            "opened_after_release": False,
            "release_completed": True,
            "identity_match_after_close": True,
        }


class FakeInferencer:
    model_sha256 = gate.FROZEN_MODEL_SHA256
    providers = ("CPUExecutionProvider",)
    preprocess_contract_sha256 = "a" * 64

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.calls = 0
        self.fail_at = fail_at

    def infer_view(self, image: np.ndarray) -> Mapping[str, Any]:
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("synthetic ONNX failure")
        self.calls += 1
        tensor_hash = gate.hashlib.sha256(
            np.ascontiguousarray(image).tobytes(order="C")
        ).hexdigest()
        probabilities = [0.7, 0.1, 0.1, 0.1]
        return {
            "provider_actual": "CPUExecutionProvider",
            "input_tensor_sha256": tensor_hash,
            "output_tensor_sha256": str(self.calls).zfill(64),
            "top1_index": 0,
            "top1_class": "grass_clump",
            "top1_probability": 0.7,
            "probabilities": probabilities,
            "latency_ms": 1.0,
        }


def board_identity() -> Mapping[str, str]:
    return dict(gate.EXPECTED_BOARD)


def camera_identity(
    device: str, expected: gate.ExpectedCameraIdentity
) -> Mapping[str, Any]:
    assert device == gate.FROZEN_DEVICE
    assert expected == gate.EXPECTED_CAMERA
    return {
        "configured_device": gate.FROZEN_DEVICE,
        "resolved_device": "/dev/video0",
        "kernel_video_node": "video0",
        "usb_vid": "32e6",
        "usb_pid": "9228",
        "usb_serial": "202604081837",
        "identity_match": True,
        "device_discovery_used": False,
    }


class OwnerSequence:
    def __init__(self, values: list[bool]) -> None:
        self.values = list(values)
        self.calls = 0

    def __call__(self, device: str) -> Mapping[str, Any]:
        assert device == "/dev/video0"
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return {
            "state": "NO_OWNER" if value else "OWNER_PRESENT",
            "no_owner": value,
            "returncode": 1 if value else 0,
            "stdout": "" if value else "123",
            "stderr": "",
        }


def test_fake_capture_and_fake_cpu_runner_pass_bounded_zero_authority() -> None:
    source = FakeSource()
    inferencer = FakeInferencer()
    owners = OwnerSequence([True, True])
    receipt = gate.qualify(
        frames=5,
        warmup_frames=1,
        source_factory=lambda: source,
        inferencer=inferencer,
        identity_reader=board_identity,
        camera_identity_reader=camera_identity,
        owner_probe=owners,
    )
    assert receipt["status"] == (
        "PASS_X5_BOUNDED_LIVE_CAMERA_CPU_ONNX_ZERO_AUTHORITY"
    )
    assert source.closed is True
    assert source.read_count == 6
    assert inferencer.calls == 10
    assert owners.calls == 2
    assert receipt["camera"]["frames_captured"] == 5
    assert receipt["cpu_onnx"]["inference_count"] == 10
    assert receipt["cpu_onnx"]["ordered_preprocess_tensor_root_sha256"]
    assert receipt["claims"]["bounded_live_session_qualified"] is True
    assert receipt["runtime_boundary"]["camera_read_only_touched"] is True
    assert receipt["runtime_boundary"]["bpu_used"] is False
    assert all(value is False for value in receipt["authority"].values())
    assert all(receipt["gates"].values())


def test_owner_before_open_fails_without_opening_camera() -> None:
    opened = False

    def source_factory() -> FakeSource:
        nonlocal opened
        opened = True
        return FakeSource()

    receipt = gate.qualify(
        frames=5,
        warmup_frames=0,
        source_factory=source_factory,
        inferencer=FakeInferencer(),
        identity_reader=board_identity,
        camera_identity_reader=camera_identity,
        owner_probe=OwnerSequence([False, False]),
    )
    assert receipt["status"] == "FAIL_CLOSED"
    assert opened is False
    assert receipt["camera"]["opened"] is False
    assert receipt["gates"]["no_owner_before_open_pass"] is False


def test_cpu_failure_still_releases_and_rechecks_no_owner() -> None:
    source = FakeSource()
    owners = OwnerSequence([True, True])
    receipt = gate.qualify(
        frames=5,
        warmup_frames=0,
        source_factory=lambda: source,
        inferencer=FakeInferencer(fail_at=3),
        identity_reader=board_identity,
        camera_identity_reader=camera_identity,
        owner_probe=owners,
    )
    assert receipt["status"] == "FAIL_CLOSED"
    assert source.closed is True
    assert owners.calls == 2
    assert receipt["camera"]["close_receipt"]["release_completed"] is True
    assert receipt["camera"]["owner_after_close"]["no_owner"] is True
    assert receipt["error"]["message"] == "synthetic ONNX failure"


def test_exact_board_mismatch_fails_before_camera_identity_or_open() -> None:
    touched = False

    def forbidden(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        nonlocal touched
        touched = True
        raise AssertionError("camera path must not be touched")

    wrong = dict(gate.EXPECTED_BOARD)
    wrong["hostname"] = "wrong-x5"
    receipt = gate.qualify(
        frames=5,
        warmup_frames=0,
        source_factory=lambda: FakeSource(),
        inferencer=FakeInferencer(),
        identity_reader=lambda: wrong,
        camera_identity_reader=forbidden,
        owner_probe=forbidden,
    )
    assert receipt["status"] == "FAIL_CLOSED"
    assert touched is False
    assert receipt["gates"]["exact_x5_identity_pass"] is False


def test_publish_json_exclusive_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path.resolve() / "receipt.json"
    gate.publish_json_exclusive(output, {"schema": "test", "passed": True})
    assert output.is_file()
    with pytest.raises(FileExistsError):
        gate.publish_json_exclusive(output, {"schema": "test", "passed": False})
