"""Fail-closed, event-only UVC capture for the printed RootScope demo cards.

This module deliberately does not discover cameras, mutate the template
registry, start training, or acquire any physical-control authority.  The live
OpenCV dependency is imported only when ``LiveUVCBackend`` is constructed, so
the capture contract can be tested entirely with fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from PIL import Image

from .quality_gate import QualityThresholds, evaluate_frame_quality


SCHEMA = "rootscope.event_optical_card_capture.v1"
DISPOSITION = "EVENT_OPTICAL_CAPTURE_NOT_AUTO_TRAIN"
REGISTERED_ROLE = "REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT"
UNKNOWN_ROLE = "UNREGISTERED_NEGATIVE_NOT_ACCURACY_HOLDOUT"
KNOWN_CARD_IDS = ("grass_clump", "low_shrub", "young_tree", "unknown")
ALLOWED_RESOLUTIONS = ((1920, 1080), (1280, 720))
EXPECTED_PRINT_STATUS = "BUILT_FOR_EVENT_DEMO_PRINT_AND_UVC_RECAPTURE_ONLY"
EXPECTED_PDF_RELATIVE_PATH = (
    "output/pdf/RootScope_A4_four_up_field_cards_20260723.pdf"
)
EXPECTED_CARD_LAYOUT = {
    "grass_clump": ("TOP_LEFT", REGISTERED_ROLE),
    "low_shrub": ("TOP_RIGHT", REGISTERED_ROLE),
    "young_tree": ("BOTTOM_LEFT", REGISTERED_ROLE),
    "unknown": ("BOTTOM_RIGHT", UNKNOWN_ROLE),
}
EXIT_ACCEPTED = 0
EXIT_QUALITY_REJECTED = 20
EXIT_CAPTURE_ERROR = 30
EXIT_USAGE = 64
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
VID_PID = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{4}$")
SAFE_SERIAL = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
VIDEO_NODE = re.compile(r"^/dev/video[0-9]+$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_role(card_id: str) -> str:
    if card_id not in KNOWN_CARD_IDS:
        raise ValueError(f"unsupported card_id: {card_id}")
    return UNKNOWN_ROLE if card_id == "unknown" else REGISTERED_ROLE


@dataclass(frozen=True)
class CaptureRequest:
    device_path: str
    expected_vid_pid: str
    expected_serial: str
    print_manifest: Path
    expected_print_manifest_sha256: str
    card_id: str
    class_role: str
    output_root: Path
    output_dir: Path
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    warmup_frames: int = 30
    frame_count: int = 5
    interval_seconds: float = 0.35
    exposure_mode: str = "keep"
    exposure_value: float | None = None
    white_balance_mode: str = "keep"
    white_balance_temperature: float | None = None


@dataclass(frozen=True)
class CaptureOutcome:
    status: str
    manifest_path: Path
    exit_code: int


class CaptureBackend(Protocol):
    def negotiated_settings(self) -> dict[str, Any]: ...

    def snapshot_controls(self) -> dict[str, float]: ...

    def apply_controls(self, request: CaptureRequest) -> dict[str, Any]: ...

    def restore_controls(self) -> dict[str, Any]: ...

    def read_rgb(self) -> np.ndarray: ...

    def close(self) -> dict[str, Any]: ...


DeviceVerifier = Callable[[CaptureRequest], dict[str, Any]]


def _validate_device_path_syntax(device_path: str) -> PurePosixPath:
    if "\x00" in device_path:
        raise ValueError("--device contains a NUL byte")
    raw_components = device_path.split("/")
    if "." in raw_components or ".." in raw_components:
        raise ValueError("--device must not contain '.' or '..' path components")
    parsed = PurePosixPath(device_path)
    if not parsed.is_absolute():
        raise ValueError("--device must be absolute")
    if parsed.parent != PurePosixPath("/dev/v4l/by-id") or not parsed.name:
        raise ValueError(
            "--device must be a direct-child symlink under /dev/v4l/by-id; "
            "numeric /dev/video* aliases and nested paths are refused"
        )
    if str(parsed) != device_path:
        raise ValueError("--device must already be in canonical textual form")
    return parsed


def _stat_receipt(result: os.stat_result, *, include_rdev: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": int(result.st_mode),
        "file_type": stat.S_IFMT(result.st_mode),
        "device": int(result.st_dev),
        "inode": int(result.st_ino),
        "uid": int(result.st_uid),
        "gid": int(result.st_gid),
        "size": int(result.st_size),
    }
    if include_rdev:
        payload.update(
            {
                "rdev": int(result.st_rdev),
                "major": int(os.major(result.st_rdev)),
                "minor": int(os.minor(result.st_rdev)),
            }
        )
    return payload


def _read_small_ascii(path: Path) -> str:
    raw = path.read_bytes()
    if len(raw) > 512:
        raise ValueError(f"sysfs identity value is unexpectedly large: {path}")
    try:
        return raw.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"sysfs identity value is not ASCII: {path}") from exc


def _read_usb_identity_from_sysfs(video_node: Path) -> dict[str, Any]:
    class_device = Path("/sys/class/video4linux") / video_node.name / "device"
    resolved = class_device.resolve(strict=True)
    candidates = (resolved, *resolved.parents)
    usb_parent: Path | None = None
    for candidate in candidates:
        if (candidate / "idVendor").is_file() and (candidate / "idProduct").is_file():
            usb_parent = candidate
            break
    if usb_parent is None:
        raise ValueError(f"USB identity not found in read-only sysfs for {video_node}")

    vendor = _read_small_ascii(usb_parent / "idVendor").lower()
    product = _read_small_ascii(usb_parent / "idProduct").lower()
    serial = _read_small_ascii(usb_parent / "serial")
    if not re.fullmatch(r"[0-9a-f]{4}", vendor) or not re.fullmatch(
        r"[0-9a-f]{4}", product
    ):
        raise ValueError("sysfs USB VID/PID values are malformed")
    if not SAFE_SERIAL.fullmatch(serial):
        raise ValueError("sysfs USB serial is empty or malformed")
    return {
        "source": "READ_ONLY_SYSFS",
        "usb_parent": str(usb_parent),
        "vid_pid": f"{vendor}:{product}",
        "serial": serial,
    }


def verify_uvc_device_identity(request: CaptureRequest) -> dict[str, Any]:
    """Verify the explicit by-id symlink without device discovery or writes."""

    parsed = _validate_device_path_syntax(request.device_path)
    by_id = Path(str(parsed))
    link_stat = os.lstat(by_id)
    if not stat.S_ISLNK(link_stat.st_mode):
        raise ValueError("--device must be a symlink, not a regular path")

    resolved = by_id.resolve(strict=True)
    resolved_text = resolved.as_posix()
    if not VIDEO_NODE.fullmatch(resolved_text):
        raise ValueError(
            f"--device must resolve directly to /dev/videoN, got {resolved_text}"
        )
    target_stat = os.stat(resolved)
    if not stat.S_ISCHR(target_stat.st_mode):
        raise ValueError(f"resolved device is not a character device: {resolved_text}")

    usb = _read_usb_identity_from_sysfs(resolved)
    expected_vid_pid = request.expected_vid_pid.lower()
    if usb["vid_pid"] != expected_vid_pid:
        raise ValueError(
            f"USB VID:PID mismatch: expected {expected_vid_pid}, got {usb['vid_pid']}"
        )
    if usb["serial"] != request.expected_serial:
        raise ValueError(
            f"USB serial mismatch: expected {request.expected_serial}, got {usb['serial']}"
        )
    return {
        "verification_method": "DIRECT_BY_ID_SYMLINK_PLUS_READ_ONLY_SYSFS",
        "device_path": request.device_path,
        "by_id_lstat": _stat_receipt(link_stat, include_rdev=False),
        "resolved_device": resolved_text,
        "target_stat": _stat_receipt(target_stat, include_rdev=True),
        "usb": usb,
        "expected": {
            "vid_pid": expected_vid_pid,
            "serial": request.expected_serial,
        },
    }


class LiveUVCBackend:
    """A single explicitly named V4L2/UVC device; there is no device scan."""

    _CONTROL_PROPERTIES = {
        "auto_exposure": "CAP_PROP_AUTO_EXPOSURE",
        "exposure": "CAP_PROP_EXPOSURE",
        "auto_white_balance": "CAP_PROP_AUTO_WB",
        "white_balance_temperature": "CAP_PROP_WB_TEMPERATURE",
    }

    def __init__(self, request: CaptureRequest, cv2_module: Any | None = None) -> None:
        if cv2_module is None:
            try:
                import cv2 as cv2_module  # type: ignore
            except ImportError as exc:  # pragma: no cover - exercised on the X5
                raise RuntimeError("OpenCV is required for live UVC capture") from exc

        self._cv2 = cv2_module
        self._capture = cv2_module.VideoCapture(
            request.device_path, cv2_module.CAP_V4L2
        )
        self._closed = False
        self._close_receipt: dict[str, Any] | None = None
        self._before_controls: dict[str, float] | None = None
        self._touched_controls: list[str] = []
        try:
            if not self._capture.isOpened():
                raise RuntimeError(
                    f"unable to open explicit UVC device: {request.device_path}"
                )
            requested_fourcc = cv2_module.VideoWriter_fourcc(*"MJPG")
            self._capture.set(cv2_module.CAP_PROP_FOURCC, requested_fourcc)
            self._capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, request.width)
            self._capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, request.height)
            self._capture.set(cv2_module.CAP_PROP_FPS, request.fps)

            negotiated = self.negotiated_settings()
            if (
                negotiated["fourcc"] != "MJPG"
                or negotiated["width"] != request.width
                or negotiated["height"] != request.height
                or abs(float(negotiated["fps"]) - request.fps) > 1.0
            ):
                raise RuntimeError(
                    "camera negotiation mismatch: "
                    f"requested=MJPG/{request.width}x{request.height}@{request.fps}, "
                    f"negotiated={negotiated}"
                )
        except BaseException:
            try:
                self._capture.release()
            finally:
                self._closed = True
            raise

    def _property_id(self, name: str) -> int:
        return int(getattr(self._cv2, self._CONTROL_PROPERTIES[name]))

    def negotiated_settings(self) -> dict[str, Any]:
        value = int(round(self._capture.get(self._cv2.CAP_PROP_FOURCC)))
        fourcc = "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))
        return {
            "backend": "opencv_v4l2",
            "fourcc": fourcc,
            "width": int(round(self._capture.get(self._cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(self._capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": round(float(self._capture.get(self._cv2.CAP_PROP_FPS)), 4),
        }

    def snapshot_controls(self) -> dict[str, float]:
        return {
            name: round(float(self._capture.get(self._property_id(name))), 6)
            for name in self._CONTROL_PROPERTIES
        }

    @staticmethod
    def _control_readback(
        name: str, requested: float, effective: float
    ) -> dict[str, Any]:
        if not math.isfinite(effective):
            return {
                "matches": False,
                "comparison": "FINITE_VALUE_REQUIRED",
                "absolute_tolerance": None,
            }
        if name == "auto_exposure":
            if math.isclose(requested, 0.25, abs_tol=0.01) or math.isclose(
                requested, 1.0, abs_tol=0.01
            ):
                accepted = (0.25, 1.0)
            elif math.isclose(requested, 0.75, abs_tol=0.01) or math.isclose(
                requested, 3.0, abs_tol=0.01
            ):
                accepted = (0.75, 3.0)
            else:
                accepted = (requested,)
            matches = any(
                math.isclose(effective, candidate, abs_tol=0.06)
                for candidate in accepted
            )
            return {
                "matches": matches,
                "comparison": "V4L2_AUTO_EXPOSURE_EQUIVALENCE",
                "accepted_effective_values": list(accepted),
                "absolute_tolerance": 0.06,
            }
        tolerance = {
            "auto_white_balance": 0.05,
            "exposure": max(1.0, abs(requested) * 0.01),
            "white_balance_temperature": 5.0,
        }[name]
        return {
            "matches": math.isclose(effective, requested, abs_tol=tolerance),
            "comparison": "ABSOLUTE_TOLERANCE",
            "absolute_tolerance": tolerance,
        }

    def _set_control(self, name: str, value: float) -> dict[str, Any]:
        self._touched_controls.append(name)
        acknowledged = bool(self._capture.set(self._property_id(name), float(value)))
        effective = round(float(self._capture.get(self._property_id(name))), 6)
        readback = self._control_readback(name, float(value), effective)
        return {
            "name": name,
            "requested": float(value),
            "set_acknowledged": acknowledged,
            "effective_after_set": effective,
            "readback": readback,
            "confirmed": acknowledged and bool(readback["matches"]),
        }

    def apply_controls(self, request: CaptureRequest) -> dict[str, Any]:
        self._before_controls = self.snapshot_controls()
        operations: list[dict[str, Any]] = []

        if request.exposure_mode == "auto":
            operations.append(self._set_control("auto_exposure", 0.75))
        elif request.exposure_mode == "manual":
            operations.append(self._set_control("auto_exposure", 0.25))
            operations.append(self._set_control("exposure", float(request.exposure_value)))

        if request.white_balance_mode == "auto":
            operations.append(self._set_control("auto_white_balance", 1.0))
        elif request.white_balance_mode == "manual":
            operations.append(self._set_control("auto_white_balance", 0.0))
            operations.append(
                self._set_control(
                    "white_balance_temperature",
                    float(request.white_balance_temperature),
                )
            )

        all_set_acknowledged = all(
            bool(operation["set_acknowledged"]) for operation in operations
        )
        all_set_confirmed = all(
            bool(operation["confirmed"]) for operation in operations
        )
        return {
            "policy": "EXPLICIT_REQUEST_ONLY_RESTORE_IN_FINALLY",
            "before": self._before_controls,
            "operations": operations,
            "effective": self.snapshot_controls(),
            "all_set_acknowledged": all_set_acknowledged,
            "all_set_confirmed": all_set_confirmed,
            "persistence_requested": False,
        }

    def restore_controls(self) -> dict[str, Any]:
        if not self._touched_controls:
            return {
                "required": False,
                "attempted": False,
                "all_restore_acknowledged": True,
                "all_restore_confirmed": True,
                "operations": [],
            }
        if self._before_controls is None:
            return {
                "required": True,
                "attempted": False,
                "all_restore_acknowledged": False,
                "all_restore_confirmed": False,
                "operations": [],
                "error": "missing pre-change control snapshot",
            }

        operations: list[dict[str, Any]] = []
        restored: set[str] = set()
        for name in reversed(self._touched_controls):
            if name in restored:
                continue
            restored.add(name)
            value = self._before_controls[name]
            acknowledged = bool(self._capture.set(self._property_id(name), value))
            effective = round(float(self._capture.get(self._property_id(name))), 6)
            readback = self._control_readback(name, value, effective)
            operations.append(
                {
                    "name": name,
                    "restore_requested": value,
                    "set_acknowledged": acknowledged,
                    "effective_after_restore": effective,
                    "readback": readback,
                    "confirmed": acknowledged and bool(readback["matches"]),
                }
            )
        all_restore_acknowledged = all(
            bool(operation["set_acknowledged"]) for operation in operations
        )
        all_restore_confirmed = all(
            bool(operation["confirmed"]) for operation in operations
        )
        return {
            "required": True,
            "attempted": True,
            "all_restore_acknowledged": all_restore_acknowledged,
            "all_restore_confirmed": all_restore_confirmed,
            "operations": operations,
            "after": self.snapshot_controls(),
        }

    def read_rgb(self) -> np.ndarray:
        ok, bgr = self._capture.read()
        if not ok or bgr is None:
            raise RuntimeError("UVC frame read failed")
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> dict[str, Any]:
        """Release the handle and prove that OpenCV no longer reports it open."""

        if self._close_receipt is not None:
            return dict(self._close_receipt)

        release_called = False
        release_error: str | None = None
        opened_after_release: bool | None = None
        if not self._closed:
            try:
                release_called = True
                self._capture.release()
            except BaseException as exc:
                release_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._closed = True

        try:
            opened_after_release = bool(self._capture.isOpened())
        except BaseException as exc:
            readback_error = (
                f"release readback failed: {type(exc).__name__}: {exc}"
            )
            release_error = (
                f"{release_error}; {readback_error}"
                if release_error
                else readback_error
            )

        self._close_receipt = {
            "release_called": release_called,
            "opened_after_release": opened_after_release,
            "release_error": release_error,
            "release_completed": (
                release_called
                and release_error is None
                and opened_after_release is False
            ),
        }
        return dict(self._close_receipt)


def _validate_request(request: CaptureRequest) -> None:
    _validate_device_path_syntax(request.device_path)
    if not VID_PID.fullmatch(request.expected_vid_pid):
        raise ValueError("--expected-vid-pid must have the form 1234:abcd")
    if not SAFE_SERIAL.fullmatch(request.expected_serial):
        raise ValueError("--expected-serial is empty or contains unsafe characters")
    if not HEX_SHA256.fullmatch(request.expected_print_manifest_sha256.lower()):
        raise ValueError("--expected-print-manifest-sha256 must be 64 hexadecimal digits")
    if request.card_id not in KNOWN_CARD_IDS:
        raise ValueError(f"--card-id must be one of: {', '.join(KNOWN_CARD_IDS)}")
    required_role = expected_role(request.card_id)
    if request.class_role != required_role:
        raise ValueError(
            f"role mismatch for {request.card_id}: expected {required_role}, "
            f"got {request.class_role}"
        )
    if (request.width, request.height) not in ALLOWED_RESOLUTIONS:
        raise ValueError("resolution must be 1920x1080 or 1280x720")
    if not math.isclose(request.fps, 30.0):
        raise ValueError("capture fps is frozen at MJPG 30")
    if not 1 <= request.warmup_frames <= 300:
        raise ValueError("warmup_frames must be in [1, 300]")
    if not 1 <= request.frame_count <= 30:
        raise ValueError("frame_count must be in [1, 30]")
    if not 0.05 <= request.interval_seconds <= 10.0:
        raise ValueError("interval_seconds must be in [0.05, 10.0]")
    if request.exposure_mode not in {"keep", "auto", "manual"}:
        raise ValueError("exposure_mode must be keep, auto, or manual")
    if request.exposure_mode == "manual" and request.exposure_value is None:
        raise ValueError("manual exposure requires --exposure-value")
    if request.exposure_mode != "manual" and request.exposure_value is not None:
        raise ValueError("--exposure-value is only valid with manual exposure")
    if request.white_balance_mode not in {"keep", "auto", "manual"}:
        raise ValueError("white_balance_mode must be keep, auto, or manual")
    if (
        request.white_balance_mode == "manual"
        and request.white_balance_temperature is None
    ):
        raise ValueError(
            "manual white balance requires --white-balance-temperature"
        )
    if (
        request.white_balance_mode != "manual"
        and request.white_balance_temperature is not None
    ):
        raise ValueError(
            "--white-balance-temperature is only valid with manual white balance"
        )

    if not request.print_manifest.is_absolute():
        raise ValueError("--print-manifest must be an explicit absolute path")
    if not request.output_root.is_absolute():
        raise ValueError("--output-root must be an explicit absolute path")
    if not request.output_dir.is_absolute():
        raise ValueError("--output-dir must be an explicit absolute path")
    if "." in request.output_root.parts or ".." in request.output_root.parts:
        raise ValueError("--output-root must not contain '.' or '..'")
    if "." in request.output_dir.parts or ".." in request.output_dir.parts:
        raise ValueError("--output-dir must not contain '.' or '..'")

    print_manifest = request.print_manifest.resolve(strict=True)
    output_root = request.output_root
    output_root_stat = os.lstat(output_root)
    if stat.S_ISLNK(output_root_stat.st_mode) or not stat.S_ISDIR(
        output_root_stat.st_mode
    ):
        raise ValueError("--output-root must be an existing non-symlink directory")
    resolved_output_root = output_root.resolve(strict=True)
    if resolved_output_root != output_root:
        raise ValueError("--output-root must already be canonical and contain no symlinks")
    output_dir = request.output_dir
    if output_dir.parent != output_root:
        raise ValueError("--output-dir must be a direct child of --output-root")
    if not output_dir.name or output_dir.name in {".", ".."}:
        raise ValueError("--output-dir must have one non-empty direct-child name")
    if not print_manifest.is_file():
        raise ValueError(f"explicit print manifest does not exist: {print_manifest}")
    if output_dir.exists():
        raise ValueError(f"output directory already exists; overwrite refused: {output_dir}")


def _load_print_binding(request: CaptureRequest) -> dict[str, Any]:
    path = request.print_manifest.resolve(strict=True)
    raw = path.read_bytes()
    observed_sha256 = _sha256_bytes(raw)
    expected_sha256 = request.expected_print_manifest_sha256.lower()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "four-up print manifest SHA-256 mismatch: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("four-up print manifest is not strict UTF-8") from exc
    payload = json.loads(text)
    if payload.get("schema") != "rootscope.event-demo-four-up-print-sheet.v1":
        raise ValueError("unexpected four-up print manifest schema")
    if payload.get("status") != EXPECTED_PRINT_STATUS:
        raise ValueError("unexpected four-up print manifest status")

    pdf = payload.get("pdf")
    if not isinstance(pdf, dict):
        raise ValueError("print manifest pdf record is missing")
    if pdf.get("path_relative_to_adventurex") != EXPECTED_PDF_RELATIVE_PATH:
        raise ValueError("print manifest PDF path is not the frozen event sheet")
    pdf_sha256 = pdf.get("sha256")
    if not isinstance(pdf_sha256, str) or not HEX_SHA256.fullmatch(
        pdf_sha256.lower()
    ):
        raise ValueError("print manifest PDF SHA-256 is malformed")

    cards = payload.get("cards")
    if not isinstance(cards, list) or len(cards) != 4:
        raise ValueError("print manifest must contain exactly four cards")
    observed_layout: dict[str, tuple[str, str]] = {}
    for card in cards:
        if not isinstance(card, dict):
            raise ValueError("print manifest card entries must be objects")
        class_id = card.get("class_id")
        position = card.get("position")
        role = card.get("role")
        if class_id not in EXPECTED_CARD_LAYOUT:
            raise ValueError(f"unexpected print card class: {class_id}")
        if class_id in observed_layout:
            raise ValueError(f"duplicate print card class: {class_id}")
        if (position, role) != EXPECTED_CARD_LAYOUT[class_id]:
            raise ValueError(
                f"print card layout/role mismatch for {class_id}: "
                f"expected {EXPECTED_CARD_LAYOUT[class_id]}, got {(position, role)}"
            )
        source_sha256 = card.get("sha256")
        if not isinstance(source_sha256, str) or not HEX_SHA256.fullmatch(
            source_sha256.lower()
        ):
            raise ValueError(f"print card source SHA-256 is malformed for {class_id}")
        if card.get("holdout_claimed") is not False:
            raise ValueError(f"holdout_claimed must be false for {class_id}")
        if card.get("accuracy_evidence") is not False:
            raise ValueError(f"accuracy_evidence must be false for {class_id}")
        observed_layout[class_id] = (position, role)
    if observed_layout != EXPECTED_CARD_LAYOUT:
        raise ValueError("print manifest card layout is incomplete or unexpected")

    matches = [card for card in cards if card.get("class_id") == request.card_id]
    if len(matches) != 1:
        raise ValueError(
            f"print manifest must contain exactly one {request.card_id} card"
        )
    card = matches[0]
    if card.get("role") != request.class_role:
        raise ValueError("print manifest role does not match the explicit class role")
    if request.card_id == "unknown" and card.get("role") != UNKNOWN_ROLE:
        raise ValueError("unknown card registration is forbidden")
    return {
        "manifest_path": str(path),
        "manifest_sha256": observed_sha256,
        "expected_manifest_sha256": expected_sha256,
        "schema": payload["schema"],
        "print_status": payload.get("status"),
        "pdf_path_relative_to_adventurex": pdf["path_relative_to_adventurex"],
        "pdf_sha256": pdf_sha256.lower(),
        "position": card.get("position"),
        "source_image_sha256": card.get("sha256").lower(),
        "card_role": card.get("role"),
        "holdout_claimed": bool(card.get("holdout_claimed", False)),
        "accuracy_evidence": bool(card.get("accuracy_evidence", False)),
    }


def _verify_created_output_directory(request: CaptureRequest) -> dict[str, Any]:
    root_stat = os.lstat(request.output_root)
    directory_stat = os.lstat(request.output_dir)
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError("output root changed into a symlink or non-directory")
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(
        directory_stat.st_mode
    ):
        raise RuntimeError("output directory changed into a symlink or non-directory")
    if request.output_root.resolve(strict=True) != request.output_root:
        raise RuntimeError("output root canonical identity changed")
    if request.output_dir.resolve(strict=True) != request.output_dir:
        raise RuntimeError("output directory canonical identity changed")
    if request.output_dir.parent != request.output_root:
        raise RuntimeError("output directory is no longer a direct child")
    return {
        "policy": "EXCLUSIVE_DIRECT_CHILD_NON_SYMLINK",
        "exclusive_mkdir_completed": True,
        "direct_child": True,
        "root_lstat": _stat_receipt(root_stat, include_rdev=False),
        "directory_lstat": _stat_receipt(directory_stat, include_rdev=False),
    }


def _save_rgb_jpeg(
    path: Path,
    frame: np.ndarray,
    *,
    directory_fd: int | None = None,
) -> None:
    array = np.asarray(frame)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("captured frame must be an HxWx3 uint8 RGB array")
    image = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0, optimize=False)
    _atomic_write_bytes(path, buffer.getvalue(), directory_fd=directory_fd)


def _atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    directory_fd: int | None = None,
) -> None:
    if path.exists():
        raise FileExistsError(f"atomic output target already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    use_dir_fd = directory_fd is not None and os.open in os.supports_dir_fd
    descriptor = (
        os.open(temporary.name, flags, 0o640, dir_fd=directory_fd)
        if use_dir_fd
        else os.open(temporary, flags, 0o640)
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if directory_fd is not None and os.link in os.supports_dir_fd:
            os.link(
                temporary.name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.unlink(temporary.name, dir_fd=directory_fd)
        else:
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
        try:
            if directory_fd is not None:
                os.fsync(directory_fd)
            else:
                parent_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
        except OSError:
            # Windows does not fsync directory handles; file fsync + atomic
            # replace remains the portable guarantee used by fixture tests.
            if os.name != "nt":
                raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            if directory_fd is not None and os.unlink in os.supports_dir_fd:
                os.unlink(temporary.name, dir_fd=directory_fd)
            else:
                temporary.unlink()
        except FileNotFoundError:
            pass


def _write_manifest(
    path: Path,
    payload: dict[str, Any],
    *,
    directory_fd: int | None = None,
) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded, directory_fd=directory_fd)


def capture_card(
    request: CaptureRequest,
    *,
    backend_factory: Callable[[CaptureRequest], CaptureBackend] = LiveUVCBackend,
    device_verifier: DeviceVerifier = verify_uvc_device_identity,
    thresholds: QualityThresholds | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    utc_now_fn: Callable[[], str] = _utc_now,
) -> CaptureOutcome:
    """Capture a new event-only directory and return a fail-closed outcome."""

    _validate_request(request)
    print_binding = _load_print_binding(request)
    quality_thresholds = thresholds or QualityThresholds(
        min_width=request.width,
        min_height=request.height,
    )

    output_dir = request.output_dir
    os.mkdir(output_dir, mode=0o750)
    output_path_evidence = _verify_created_output_directory(request)
    output_directory_fd: int | None = None
    if os.name != "nt":
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        output_directory_fd = os.open(output_dir, directory_flags)
        opened_directory = os.fstat(output_directory_fd)
        expected_directory = output_path_evidence["directory_lstat"]
        if (
            int(opened_directory.st_dev) != int(expected_directory["device"])
            or int(opened_directory.st_ino) != int(expected_directory["inode"])
        ):
            os.close(output_directory_fd)
            raise RuntimeError("output directory changed while pinning its descriptor")
    manifest_path = output_dir / "capture_manifest.json"
    backend: CaptureBackend | None = None
    frame_receipts: list[dict[str, Any]] = []
    capture_error: str | None = None
    device_identity: dict[str, Any] = {
        "verification_policy": (
            "VERIFY_BEFORE_OPEN_AFTER_OPEN_AND_AFTER_CLOSE_EXACT_MATCH"
        ),
        "before_open": None,
        "after_open": None,
        "after_close": None,
        "after_close_verification_attempted": False,
        "after_close_verification_error": None,
        "identity_unchanged_before_and_after_open": False,
        "identity_unchanged_after_close": False,
        "identity_unchanged_across_lifecycle": False,
        "identity_unchanged": False,
    }
    control_record: dict[str, Any] = {
        "policy": "EXPLICIT_REQUEST_ONLY_RESTORE_IN_FINALLY",
        "apply": {"status": "NOT_REACHED"},
        "restore": {"status": "NOT_REACHED"},
        "close": {
            "status": "NOT_REACHED",
            "attempted": False,
            "release_completed": False,
            "contract_satisfied": False,
        },
    }
    negotiated: dict[str, Any] | None = None
    started_utc = utc_now_fn()
    before_identity: dict[str, Any] | None = None
    after_identity: dict[str, Any] | None = None

    def record_capture_error(message: str) -> None:
        nonlocal capture_error
        capture_error = f"{capture_error}; {message}" if capture_error else message

    try:
        before_identity = device_verifier(request)
        device_identity["before_open"] = before_identity
        backend = backend_factory(request)
        after_identity = device_verifier(request)
        device_identity["after_open"] = after_identity
        open_identity_match = before_identity == after_identity
        device_identity["identity_unchanged_before_and_after_open"] = (
            open_identity_match
        )
        if before_identity != after_identity:
            raise RuntimeError("UVC identity changed between pre-open and post-open checks")
        negotiated = backend.negotiated_settings()
        applied = backend.apply_controls(request)
        control_record["apply"] = applied
        if not bool(applied.get("all_set_confirmed", False)):
            raise RuntimeError(
                "one or more explicitly requested controls were not confirmed by readback"
            )

        for _ in range(request.warmup_frames):
            backend.read_rgb()

        first_target = monotonic_fn()
        for index in range(request.frame_count):
            target = first_target + index * request.interval_seconds
            delay = target - monotonic_fn()
            if delay > 0:
                sleep_fn(delay)
            frame = backend.read_rgb()
            if frame.shape[:2] != (request.height, request.width):
                raise RuntimeError(
                    "captured frame dimensions changed: "
                    f"expected={request.width}x{request.height}, "
                    f"got={frame.shape[1]}x{frame.shape[0]}"
                )
            quality = evaluate_frame_quality(frame, quality_thresholds)
            filename = f"{request.card_id}_{index + 1:03d}.jpg"
            frame_path = output_dir / filename
            _verify_created_output_directory(request)
            _save_rgb_jpeg(
                frame_path,
                frame,
                directory_fd=output_directory_fd,
            )
            frame_receipts.append(
                {
                    "capture_index": index + 1,
                    "file": filename,
                    "sha256": _sha256_file(frame_path),
                    "bytes": frame_path.stat().st_size,
                    "quality": quality.to_dict(),
                    "semantic_label_source": "EXPLICIT_OPERATOR_CARD_ID",
                    "training_eligible": False,
                    "accuracy_evaluation_eligible": False,
                }
            )
    except Exception as exc:  # capture errors are retained as evidence
        capture_error = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None:
            try:
                control_record["restore"] = backend.restore_controls()
            except Exception as exc:
                control_record["restore"] = {
                    "required": True,
                    "attempted": True,
                    "all_restore_acknowledged": False,
                    "all_restore_confirmed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                try:
                    close_receipt = backend.close()
                    if not isinstance(close_receipt, dict):
                        raise RuntimeError(
                            "backend close did not return a structured release receipt"
                        )
                    release_contract_satisfied = (
                        close_receipt.get("release_called") is True
                        and close_receipt.get("opened_after_release") is False
                        and close_receipt.get("release_error") is None
                        and close_receipt.get("release_completed") is True
                    )
                    control_record["close"] = {
                        **close_receipt,
                        "status": (
                            "RELEASE_CONFIRMED"
                            if release_contract_satisfied
                            else "RELEASE_NOT_CONFIRMED"
                        ),
                        "attempted": True,
                        "contract_satisfied": release_contract_satisfied,
                    }
                    if not release_contract_satisfied:
                        record_capture_error(
                            "CAMERA_RELEASE_NOT_CONFIRMED_ISOPENED_FALSE"
                        )
                except Exception as exc:
                    close_error = f"{type(exc).__name__}: {exc}"
                    control_record["close"] = {
                        "status": "RELEASE_ERROR",
                        "attempted": True,
                        "release_called": None,
                        "opened_after_release": None,
                        "release_error": close_error,
                        "release_completed": False,
                        "contract_satisfied": False,
                        "error": close_error,
                    }
                    record_capture_error(f"camera close failed: {close_error}")

            device_identity["after_close_verification_attempted"] = True
            try:
                close_identity = device_verifier(request)
                device_identity["after_close"] = close_identity
                close_identity_match = (
                    before_identity is not None
                    and after_identity is not None
                    and before_identity == after_identity == close_identity
                )
                device_identity["identity_unchanged_after_close"] = (
                    close_identity_match
                )
                device_identity["identity_unchanged_across_lifecycle"] = (
                    close_identity_match
                )
                device_identity["identity_unchanged"] = close_identity_match
                if not close_identity_match:
                    record_capture_error(
                        "UVC identity changed after camera close"
                    )
            except Exception as exc:
                identity_error = f"{type(exc).__name__}: {exc}"
                device_identity["after_close_verification_error"] = identity_error
                record_capture_error(
                    f"post-close UVC identity verification failed: {identity_error}"
                )

    all_frames_present = len(frame_receipts) == request.frame_count
    all_frames_passed = all(
        bool(frame["quality"]["passed"]) for frame in frame_receipts
    )
    restore_ok = bool(control_record["restore"].get("all_restore_confirmed", False))
    close_ok = bool(control_record["close"].get("contract_satisfied", False))
    lifecycle_identity_ok = bool(
        device_identity["identity_unchanged_across_lifecycle"]
    )
    if not restore_ok and capture_error is None:
        capture_error = "CONTROL_RESTORE_NOT_CONFIRMED_BY_READBACK"
    if not close_ok and capture_error is None:
        capture_error = "CAMERA_RELEASE_NOT_CONFIRMED_ISOPENED_FALSE"
    if not lifecycle_identity_ok and capture_error is None:
        capture_error = "UVC_IDENTITY_NOT_CONFIRMED_ACROSS_FULL_LIFECYCLE"

    if (
        capture_error is not None
        or not all_frames_present
        or not restore_ok
        or not close_ok
        or not lifecycle_identity_ok
    ):
        status = "EVENT_OPTICAL_CAPTURE_ERROR_NOT_AUTO_TRAIN"
        exit_code = EXIT_CAPTURE_ERROR
    elif not all_frames_passed:
        status = "EVENT_OPTICAL_CAPTURE_REJECTED_NOT_AUTO_TRAIN"
        exit_code = EXIT_QUALITY_REJECTED
    else:
        status = "EVENT_OPTICAL_CAPTURE_ACCEPTED_NOT_AUTO_TRAIN"
        exit_code = EXIT_ACCEPTED

    manifest = {
        "schema": SCHEMA,
        "status": status,
        "disposition": DISPOSITION,
        "created_utc": started_utc,
        "completed_utc": utc_now_fn(),
        "request": {
            "device_path": request.device_path,
            "device_discovery_used": False,
            "expected_vid_pid": request.expected_vid_pid.lower(),
            "expected_serial": request.expected_serial,
            "card_id": request.card_id,
            "class_role": request.class_role,
            "output_root": str(request.output_root),
            "output_dir": str(output_dir),
            "fourcc": "MJPG",
            "width": request.width,
            "height": request.height,
            "fps": request.fps,
            "warmup_frames": request.warmup_frames,
            "frame_count": request.frame_count,
            "fixed_interval_seconds": request.interval_seconds,
            "exposure_mode": request.exposure_mode,
            "exposure_value": request.exposure_value,
            "white_balance_mode": request.white_balance_mode,
            "white_balance_temperature": request.white_balance_temperature,
        },
        "print_sheet_binding": print_binding,
        "device_identity": device_identity,
        "output_path_evidence": output_path_evidence,
        "negotiated_capture": negotiated,
        "controls": control_record,
        "quality_gate": {
            "policy": "ALL_REQUESTED_FRAMES_MUST_PASS",
            "thresholds": asdict(quality_thresholds),
            "requested_frame_count": request.frame_count,
            "captured_frame_count": len(frame_receipts),
            "all_frames_present": all_frames_present,
            "all_frames_passed": all_frames_passed,
            "accepted": status == "EVENT_OPTICAL_CAPTURE_ACCEPTED_NOT_AUTO_TRAIN",
        },
        "frames": frame_receipts,
        "error": capture_error,
        "truth_boundary": {
            "event_optical_capture_only": True,
            "auto_train": False,
            "training_eligible": False,
            "accuracy_claim_allowed": False,
            "model_qualified": False,
            "registry_mutated": False,
            "registry_write_code_path_present": False,
            "unknown_registration_allowed": False,
            "unknown_is_unregistered_negative": request.card_id == "unknown",
            "serial_gpio_pump_or_systemd_touched": False,
            "physical_or_irrigation_authority": False,
        },
    }
    try:
        _verify_created_output_directory(request)
        _write_manifest(
            manifest_path,
            manifest,
            directory_fd=output_directory_fd,
        )
    finally:
        if output_directory_fd is not None:
            os.close(output_directory_fd)
    return CaptureOutcome(status=status, manifest_path=manifest_path, exit_code=exit_code)


def _parse_resolution(value: str) -> tuple[int, int]:
    choices = {"1920x1080": (1920, 1080), "1280x720": (1280, 720)}
    if value not in choices:
        raise argparse.ArgumentTypeError("resolution must be 1920x1080 or 1280x720")
    return choices[value]


class StructuredArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        payload = {
            "status": "EVENT_OPTICAL_CAPTURE_USAGE_ERROR",
            "exit_code": EXIT_USAGE,
            "error": message,
        }
        self.exit(EXIT_USAGE, json.dumps(payload, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = StructuredArgumentParser(
        description=(
            "Capture one explicitly identified printed RootScope card from one "
            "explicit UVC device. This never trains or mutates the registry."
        )
    )
    parser.add_argument(
        "--device",
        required=True,
        help="explicit stable /dev/v4l/by-id path (numeric /dev/video* is refused)",
    )
    parser.add_argument("--expected-vid-pid", required=True)
    parser.add_argument("--expected-serial", required=True)
    parser.add_argument("--print-manifest", required=True, type=Path)
    parser.add_argument("--expected-print-manifest-sha256", required=True)
    parser.add_argument("--card-id", required=True, choices=KNOWN_CARD_IDS)
    parser.add_argument(
        "--class-role",
        required=True,
        choices=(REGISTERED_ROLE, UNKNOWN_ROLE),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=(1920, 1080),
        metavar="{1920x1080,1280x720}",
    )
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--interval-ms", type=int, default=350)
    parser.add_argument(
        "--exposure-mode",
        choices=("keep", "auto", "manual"),
        default="keep",
    )
    parser.add_argument("--exposure-value", type=float)
    parser.add_argument(
        "--white-balance-mode",
        choices=("keep", "auto", "manual"),
        default="keep",
    )
    parser.add_argument("--white-balance-temperature", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    width, height = args.resolution
    request = CaptureRequest(
        device_path=args.device,
        expected_vid_pid=args.expected_vid_pid,
        expected_serial=args.expected_serial,
        print_manifest=args.print_manifest,
        expected_print_manifest_sha256=args.expected_print_manifest_sha256,
        card_id=args.card_id,
        class_role=args.class_role,
        output_root=args.output_root,
        output_dir=args.output_dir,
        width=width,
        height=height,
        warmup_frames=args.warmup_frames,
        frame_count=args.frames,
        interval_seconds=args.interval_ms / 1000.0,
        exposure_mode=args.exposure_mode,
        exposure_value=args.exposure_value,
        white_balance_mode=args.white_balance_mode,
        white_balance_temperature=args.white_balance_temperature,
    )
    try:
        outcome = capture_card(request)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "EVENT_OPTICAL_CAPTURE_USAGE_ERROR",
                    "exit_code": EXIT_USAGE,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_USAGE
    except OSError as exc:
        print(
            json.dumps(
                {
                    "status": "EVENT_OPTICAL_CAPTURE_IO_ERROR",
                    "exit_code": EXIT_CAPTURE_ERROR,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_CAPTURE_ERROR
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "EVENT_OPTICAL_CAPTURE_INTERNAL_ERROR",
                    "exit_code": EXIT_CAPTURE_ERROR,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_CAPTURE_ERROR
    print(
        json.dumps(
            {
                "status": outcome.status,
                "manifest": str(outcome.manifest_path),
                "exit_code": outcome.exit_code,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
