#!/usr/bin/env python3
"""Bounded RootScope v3 live-camera + CPU ONNX qualification gate.

The gate opens exactly one frozen UVC by-id node, runs a finite number of
read-only CPU ONNX inferences on raw and Gray-World views, releases the camera
in ``finally``, and proves that no process owns the device afterwards.  It has
no BPU, serial, GPIO, pump, network, service, or physical execution interface.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from app.edge.onnx_cpu import preprocess_rgb
from app.omega_vision.uvc_card_frontend import (
    ExpectedCameraIdentity,
    FrontendRequest,
    LiveUvcFrameSource,
    read_explicit_usb_identity,
)
from app.vision.dual_path_demo import build_seed17_runner_from_capsule


SCHEMA = "rootscope.v3.x5-live-camera-cpu-qualification.v1"
FROZEN_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-Web_Camera_Web_Camera_202604081837-video-index0"
)
EXPECTED_CAMERA = ExpectedCameraIdentity(
    usb_vid="32e6",
    usb_pid="9228",
    usb_serial="202604081837",
)
EXPECTED_BOARD = {
    "hostname": "rootscope-x5",
    "machine_id": "<redacted-device-boot-id>",
    "serial": "3281556110220e0c002bdeab0012004",
    "wlan_mac": "02:00:00:00:00:01",
    "architecture": "aarch64",
}
FROZEN_MODEL_SHA256 = (
    "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad"
)
FROZEN_CAPSULE_SHA256 = (
    "1b7e9b96ccd4ec4e5ab534e1f305224c3c6330a3ab2efb8ca2e5d0fc52fcfcbb"
)
CLASS_ORDER = ("grass_clump", "low_shrub", "young_tree", "unknown")
ZERO_AUTHORITY = {
    "execution_authority": False,
    "physical_authority": False,
    "serial_open": False,
    "serial_write": False,
    "gpio_access": False,
    "pump_command": False,
    "state_machine_write": False,
    "irrigation_execution": False,
    "physical_completion": False,
    "network_access": False,
    "network_configuration_write": False,
    "service_configuration_write": False,
}


class GateError(RuntimeError):
    """A fail-closed qualification contract error."""


class FrameSource(Protocol):
    def read_rgb(self) -> np.ndarray: ...

    def negotiated_settings(self) -> Mapping[str, Any]: ...

    def close(self) -> Mapping[str, Any]: ...


class ViewInferencer(Protocol):
    model_sha256: str
    providers: Sequence[str]
    preprocess_contract_sha256: str

    def infer_view(self, image: np.ndarray) -> Mapping[str, Any]: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_x5_identity() -> dict[str, str]:
    serial_path = Path("/proc/device-tree/serial-number")
    if not serial_path.exists():
        serial_path = Path("/sys/firmware/devicetree/base/serial-number")
    return {
        "hostname": platform.node(),
        "machine_id": Path("/etc/machine-id")
        .read_text(encoding="ascii")
        .strip(),
        "serial": serial_path.read_bytes().replace(b"\x00", b"").decode("ascii"),
        "wlan_mac": Path("/sys/class/net/wlan0/address")
        .read_text(encoding="ascii")
        .strip()
        .lower(),
        "architecture": platform.machine(),
    }


def probe_camera_owner(device: str) -> dict[str, Any]:
    """Return a fail-closed fuser result for one already-resolved device."""

    try:
        result = subprocess.run(
            ["fuser", device],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {
            "state": "UNKNOWN",
            "no_owner": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    stdout = " ".join(result.stdout.split())
    stderr = " ".join(result.stderr.split())
    if result.returncode == 1 and not stdout and not stderr:
        state = "NO_OWNER"
    elif result.returncode == 0:
        state = "OWNER_PRESENT"
    else:
        state = "UNKNOWN"
    return {
        "state": state,
        "no_owner": state == "NO_OWNER",
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def gray_world(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    floating = rgb.astype(np.float32)
    means = floating.reshape(-1, 3).mean(axis=0)
    neutral = float(np.mean(means))
    scales = neutral / np.maximum(means, 1.0)
    corrected = np.clip(
        floating * scales.reshape(1, 1, 3), 0, 255
    ).astype(np.uint8)
    return corrected, float(means[0] / max(float(means[2]), 1.0))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    values = np.exp(shifted)
    denominator = float(values.sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise GateError("softmax denominator is invalid")
    return values / denominator


def frame_statistics(rgb: np.ndarray) -> dict[str, Any]:
    array = np.asarray(rgb)
    format_valid = (
        array.dtype == np.uint8
        and array.ndim == 3
        and array.shape[2] == 3
        and array.shape[0] >= 240
        and array.shape[1] >= 320
        and array.size > 0
    )
    if not format_valid:
        raise GateError(
            f"invalid live frame: dtype={array.dtype} shape={array.shape}"
        )
    floating = array.astype(np.float32)
    channel_means = floating.reshape(-1, 3).mean(axis=0)
    luma = (
        floating[:, :, 0] * np.float32(0.2126)
        + floating[:, :, 1] * np.float32(0.7152)
        + floating[:, :, 2] * np.float32(0.0722)
    )
    p01, p99 = np.percentile(luma, (1.0, 99.0))
    black_fraction = float(np.mean(luma <= 5.0))
    white_fraction = float(np.mean(luma >= 250.0))
    exposure_usable = bool(
        1.0 < float(luma.mean()) < 254.0
        and float(p99 - p01) >= 2.0
        and black_fraction < 0.98
        and white_fraction < 0.98
    )
    return {
        "format_valid": True,
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "rgb_sha256": hashlib.sha256(
            np.ascontiguousarray(array).tobytes(order="C")
        ).hexdigest(),
        "channel_mean_rgb": [round(float(value), 6) for value in channel_means],
        "luma_mean": round(float(luma.mean()), 6),
        "luma_std": round(float(luma.std()), 6),
        "luma_p01": round(float(p01), 6),
        "luma_p99": round(float(p99), 6),
        "black_fraction": round(black_fraction, 8),
        "white_fraction": round(white_fraction, 8),
        "warmth_red_over_blue": round(
            float(channel_means[0] / max(float(channel_means[2]), 1.0)), 6
        ),
        "exposure_usable": exposure_usable,
    }


class CpuOnnxViewInferencer:
    """Expose truthful live-view inference over the frozen CPU runner."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner
        self.model_sha256 = str(runner.model_sha256)
        self.providers = tuple(str(value) for value in runner.providers)
        if self.model_sha256 != FROZEN_MODEL_SHA256:
            raise GateError("live CPU model SHA-256 is not the frozen seed17 model")
        if self.providers != ("CPUExecutionProvider",):
            raise GateError(f"CPU-only provider contract failed: {self.providers}")
        self.preprocess_contract_sha256 = canonical_sha256(
            asdict(runner.preprocess)
        )

    def infer_view(self, image: np.ndarray) -> Mapping[str, Any]:
        started = time.perf_counter()
        tensor = preprocess_rgb(image, self._runner.preprocess)
        values = self._runner._session.run(
            [self._runner.output_name],
            {self._runner.input_name: tensor},
        )
        if len(values) != 1:
            raise GateError("CPU ONNX returned an unexpected output count")
        logits = np.asarray(values[0], dtype=np.float32)
        if logits.shape != (1, len(CLASS_ORDER)) or not np.isfinite(logits).all():
            raise GateError("CPU ONNX logits are not finite [1,4]")
        probabilities = softmax(logits[0].astype(np.float64))
        top1 = int(np.argmax(probabilities))
        return {
            "provider_actual": "CPUExecutionProvider",
            "input_tensor_sha256": hashlib.sha256(
                np.ascontiguousarray(tensor).tobytes(order="C")
            ).hexdigest(),
            "output_tensor_sha256": hashlib.sha256(
                np.ascontiguousarray(logits).tobytes(order="C")
            ).hexdigest(),
            "top1_index": top1,
            "top1_class": CLASS_ORDER[top1],
            "top1_probability": round(float(probabilities[top1]), 8),
            "probabilities": [round(float(value), 8) for value in probabilities],
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 4),
        }


def _base_receipt(frames: int, warmup_frames: int) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "FAIL_CLOSED",
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "request": {
            "device": FROZEN_DEVICE,
            "frames": frames,
            "warmup_frames": warmup_frames,
            "max_frames": 120,
            "views_per_frame": ["RAW_RGB", "GRAY_WORLD_RGB"],
        },
        "board": None,
        "camera": {
            "identity_before_open": None,
            "owner_before_open": None,
            "opened": False,
            "negotiated_settings": None,
            "warmup_frames_captured": 0,
            "frames_captured": 0,
            "close_receipt": None,
            "owner_after_close": None,
        },
        "cpu_onnx": {
            "model_sha256": None,
            "providers_actual": None,
            "preprocess_contract_sha256": None,
            "inference_count": 0,
            "raw_and_gray_world_tta": True,
            "frames": [],
            "ordered_preprocess_tensor_root_sha256": None,
        },
        "gates": {},
        "claims": {
            "bounded_live_session_qualified": False,
            "general_field_accuracy_claim": False,
            "plant_model_qualified_by_this_gate": False,
            "bpu_qualified_by_this_gate": False,
            "irrigation_decision_claim": False,
            "physical_completion_claim": False,
        },
        "runtime_boundary": {
            "camera_read_only_touched": False,
            "bpu_used": False,
            "serial_opened": False,
            "gpio_touched": False,
            "pump_touched": False,
            "network_touched": False,
            "service_configuration_touched": False,
        },
        "authority": dict(ZERO_AUTHORITY),
        "error": None,
    }


def qualify(
    *,
    frames: int,
    warmup_frames: int,
    source_factory: Callable[[], FrameSource],
    inferencer: ViewInferencer,
    identity_reader: Callable[[], Mapping[str, str]] = read_x5_identity,
    camera_identity_reader: Callable[
        [str, ExpectedCameraIdentity], Mapping[str, Any]
    ] = read_explicit_usb_identity,
    owner_probe: Callable[[str], Mapping[str, Any]] = probe_camera_owner,
) -> dict[str, Any]:
    """Run one bounded session.  Dependencies are injectable for pure tests."""

    if not 5 <= frames <= 120:
        raise ValueError("frames must be within 5..120")
    if not 0 <= warmup_frames <= 30:
        raise ValueError("warmup_frames must be within 0..30")
    receipt = _base_receipt(frames, warmup_frames)
    source: FrameSource | None = None
    resolved_device: str | None = None
    tensor_hashes: list[str] = []
    try:
        board = dict(identity_reader())
        receipt["board"] = board
        if board != EXPECTED_BOARD:
            raise GateError(f"exact X5 identity mismatch: {board}")

        camera_identity = dict(camera_identity_reader(FROZEN_DEVICE, EXPECTED_CAMERA))
        receipt["camera"]["identity_before_open"] = camera_identity
        if camera_identity.get("configured_device") != FROZEN_DEVICE:
            raise GateError("camera identity reader did not bind the frozen by-id path")
        if camera_identity.get("identity_match") is not True:
            raise GateError("camera VID/PID/serial identity did not match")
        resolved_device = str(camera_identity.get("resolved_device", ""))
        if not resolved_device.startswith("/dev/video"):
            raise GateError("frozen by-id alias did not resolve to a video character node")

        owner_before = dict(owner_probe(resolved_device))
        receipt["camera"]["owner_before_open"] = owner_before
        if owner_before.get("no_owner") is not True:
            raise GateError(f"camera owner before open is not empty: {owner_before}")

        source = source_factory()
        receipt["camera"]["opened"] = True
        receipt["runtime_boundary"]["camera_read_only_touched"] = True
        receipt["camera"]["negotiated_settings"] = dict(
            source.negotiated_settings()
        )
        for _ in range(warmup_frames):
            frame_statistics(source.read_rgb())
            receipt["camera"]["warmup_frames_captured"] += 1

        for index in range(frames):
            frame = source.read_rgb()
            statistics = frame_statistics(frame)
            corrected, warmth = gray_world(frame)
            raw = dict(inferencer.infer_view(frame))
            corrected_result = dict(inferencer.infer_view(corrected))
            tensor_hashes.extend(
                [
                    str(raw["input_tensor_sha256"]),
                    str(corrected_result["input_tensor_sha256"]),
                ]
            )
            raw_prob = np.asarray(raw["probabilities"], dtype=np.float64)
            corrected_prob = np.asarray(
                corrected_result["probabilities"], dtype=np.float64
            )
            if raw_prob.shape != (4,) or corrected_prob.shape != (4,):
                raise GateError("view probability vector is not length four")
            ensemble = (raw_prob + corrected_prob) / 2.0
            top1 = int(np.argmax(ensemble))
            receipt["cpu_onnx"]["frames"].append(
                {
                    "frame_index": index,
                    "frame": statistics,
                    "gray_world": {
                        "warmth_red_over_blue": round(warmth, 6),
                        "rgb_sha256": hashlib.sha256(
                            np.ascontiguousarray(corrected).tobytes(order="C")
                        ).hexdigest(),
                    },
                    "raw_cpu_onnx": raw,
                    "gray_world_cpu_onnx": corrected_result,
                    "tta_ensemble": {
                        "top1_index": top1,
                        "top1_class": CLASS_ORDER[top1],
                        "top1_probability": round(float(ensemble[top1]), 8),
                    },
                }
            )
            receipt["camera"]["frames_captured"] += 1
            receipt["cpu_onnx"]["inference_count"] += 2
    except BaseException as exc:
        receipt["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        if source is not None:
            try:
                receipt["camera"]["close_receipt"] = dict(source.close())
            except BaseException as exc:
                receipt["camera"]["close_receipt"] = {
                    "release_completed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if receipt["error"] is None:
                    receipt["error"] = {
                        "type": type(exc).__name__,
                        "message": f"camera close failed: {exc}",
                    }
        if resolved_device is not None:
            try:
                receipt["camera"]["owner_after_close"] = dict(
                    owner_probe(resolved_device)
                )
            except BaseException as exc:
                receipt["camera"]["owner_after_close"] = {
                    "state": "UNKNOWN",
                    "no_owner": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    receipt["cpu_onnx"]["model_sha256"] = inferencer.model_sha256
    receipt["cpu_onnx"]["providers_actual"] = list(inferencer.providers)
    receipt["cpu_onnx"][
        "preprocess_contract_sha256"
    ] = inferencer.preprocess_contract_sha256
    if tensor_hashes:
        receipt["cpu_onnx"][
            "ordered_preprocess_tensor_root_sha256"
        ] = canonical_sha256(tensor_hashes)
    records = receipt["cpu_onnx"]["frames"]
    usable = sum(
        1 for item in records if item["frame"]["exposure_usable"] is True
    )
    close = receipt["camera"]["close_receipt"] or {}
    owner_after = receipt["camera"]["owner_after_close"] or {}
    gates = {
        "exact_x5_identity_pass": receipt["board"] == EXPECTED_BOARD,
        "frozen_camera_identity_pass": bool(
            receipt["camera"]["identity_before_open"]
            and receipt["camera"]["identity_before_open"].get("identity_match")
            is True
            and receipt["camera"]["identity_before_open"].get("configured_device")
            == FROZEN_DEVICE
        ),
        "no_owner_before_open_pass": bool(
            receipt["camera"]["owner_before_open"]
            and receipt["camera"]["owner_before_open"].get("no_owner") is True
        ),
        "bounded_frame_count_pass": receipt["camera"]["frames_captured"] == frames,
        "all_frame_formats_valid_pass": len(records) == frames
        and all(item["frame"]["format_valid"] is True for item in records),
        "exposure_usable_fraction_pass": len(records) == frames
        and usable >= max(1, int(np.ceil(frames * 0.8))),
        "cpu_onnx_real_inference_count_pass": (
            receipt["cpu_onnx"]["inference_count"] == frames * 2
            and receipt["cpu_onnx"]["inference_count"] >= 10
        ),
        "cpu_only_provider_pass": tuple(inferencer.providers)
        == ("CPUExecutionProvider",),
        "frozen_model_pass": inferencer.model_sha256 == FROZEN_MODEL_SHA256,
        "camera_release_pass": close.get("release_completed") is True,
        "no_owner_after_close_pass": owner_after.get("no_owner") is True,
        "zero_authority_pass": all(value is False for value in ZERO_AUTHORITY.values()),
        "no_bpu_serial_gpio_pump_network_pass": all(
            receipt["runtime_boundary"][key] is False
            for key in (
                "bpu_used",
                "serial_opened",
                "gpio_touched",
                "pump_touched",
                "network_touched",
                "service_configuration_touched",
            )
        ),
    }
    receipt["gates"] = gates
    if receipt["error"] is None and all(gates.values()):
        receipt["status"] = "PASS_X5_BOUNDED_LIVE_CAMERA_CPU_ONNX_ZERO_AUTHORITY"
        receipt["claims"]["bounded_live_session_qualified"] = True
    receipt["completed_at_utc"] = utc_now()
    return receipt


def _regular_bound_file(path: Path, expected_sha256: str, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise GateError(f"{label} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if sha256_file(resolved) != expected_sha256:
        raise GateError(f"{label} SHA-256 mismatch")
    return resolved


def _canonical_directory(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise GateError(f"{label} must be absolute")
    if expanded.is_symlink() or not expanded.is_dir():
        raise GateError(f"{label} must be a non-symlink directory")
    resolved = expanded.resolve(strict=True)
    if resolved != expanded:
        raise GateError(f"{label} must be canonical")
    return resolved


def publish_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish canonical JSON without replacing any destination."""

    output = path.expanduser()
    if not output.is_absolute():
        raise GateError("--output must be absolute")
    parent = _canonical_directory(output.parent, "output parent")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    partial = parent / f".{output.name}.{os.getpid()}.partial"
    descriptor = os.open(
        partial,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, output)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not 5 <= args.frames <= 120:
        parser.error("--frames must be within 5..120")
    if not 0 <= args.warmup_frames <= 30:
        parser.error("--warmup-frames must be within 0..30")
    if not 320 <= args.width <= 4096 or not 240 <= args.height <= 2160:
        parser.error("camera dimensions are outside the bounded range")
    if not 1.0 <= args.fps <= 60.0:
        parser.error("--fps must be within 1..60")

    # Output safety is established before any camera access.
    output_parent = _canonical_directory(
        args.output.expanduser().parent, "output parent"
    )
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    receipt: dict[str, Any]
    try:
        release_root = _canonical_directory(args.release_root, "release root")
        capsule = _regular_bound_file(
            release_root
            / "rootscope"
            / "deploy"
            / "x5"
            / "capsule_config.seed17_cpu_experimental.json",
            FROZEN_CAPSULE_SHA256,
            "CPU capsule",
        )
        model = _regular_bound_file(
            release_root / "models" / "rootscope_seed17_cpu.onnx",
            FROZEN_MODEL_SHA256,
            "CPU model",
        )
        inferencer = CpuOnnxViewInferencer(
            build_seed17_runner_from_capsule(capsule, model_path=model)
        )
        request = FrontendRequest(
            device=FROZEN_DEVICE,
            expected_camera=EXPECTED_CAMERA,
            print_manifest=capsule,
            mode="bounded",
            frames=args.frames,
            warmup_frames=args.warmup_frames,
            interval_seconds=0.0,
            width=args.width,
            height=args.height,
            fps=args.fps,
            output_root=output_parent,
            jsonl_path=output_parent / f".{args.output.name}.unused.jsonl",
        )
        receipt = qualify(
            frames=args.frames,
            warmup_frames=args.warmup_frames,
            source_factory=lambda: LiveUvcFrameSource(request),
            inferencer=inferencer,
        )
    except BaseException as exc:
        receipt = _base_receipt(args.frames, args.warmup_frames)
        receipt["completed_at_utc"] = utc_now()
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}

    publish_json_exclusive(args.output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
                "gates": receipt["gates"],
                "authority": receipt["authority"],
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if receipt["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
