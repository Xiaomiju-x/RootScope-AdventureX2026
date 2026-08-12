"""Bounded, fail-closed UVC frontend for the printed RootScope demo cards.

The frontend has exactly four visible evidence layers:

1. seed17 CPU ONNX semantic hypothesis;
2. Omega quality/OOD abstention;
3. registered-template geometric evidence;
4. a display-only final consensus.

It never discovers cameras, never runs a service, never touches serial/GPIO/a
pump, and never grants physical authority.  A caller must provide one stable
``/dev/v4l/by-id/...`` camera path and a finite frame count.  The plant BPU
selection remains explicitly null.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image, ImageDraw

from app.edge.capsule import CPU_PROVIDER, ROOTSCOPE_CLASS_ORDER
from app.omega_vision.ood import Calibration, decide, evaluate_quality
from app.vision.card_geometric_matcher import MatcherConfig
from app.vision.dual_path_demo import (
    DemoThresholds,
    SEED17_MODEL_SHA256,
    build_seed17_runner_from_capsule,
    evaluate_dual_path_demo,
    load_template_registry,
)


SCHEMA = "rootscope.uvc-card-frontend.v1"
JSONL_SCHEMA = "rootscope.uvc-card-frontend-jsonl.v1"
CALIBRATION_MANIFEST_SCHEMA = (
    "rootscope.omega-vision-board-replay-manifest.v1"
)
FINAL_ACCEPT = "DISPLAY_ONLY_REGISTERED_CARD_CONSENSUS"
FINAL_REJECT = "SAFE_REJECT_NO_PHYSICAL_AUTHORITY"
MAX_FRAMES = 30
PRINT_MANIFEST_SCHEMA = "rootscope.event-demo-four-up-print-sheet.v1"
PRINT_MANIFEST_STATUS = "BUILT_FOR_EVENT_DEMO_PRINT_AND_UVC_RECAPTURE_ONLY"
FROZEN_PRINT_MANIFEST_SHA256 = (
    "5e23e6133e9a59d8327bd751c4d6e5434d0c5a86402fd920c9d99e547613d827"
)
FROZEN_PRINT_PDF_SHA256 = (
    "113d2b2171e55f42df36d16b8772e02624d52d68a230cda541da66e56c19874e"
)
FROZEN_THRESHOLDS_SHA256 = (
    "877205689ad903207e0bcb5ffabdcbc5f1472c00b8f82e72faeb7cdd7d140fcd"
)
FROZEN_MATCHER_CONFIG_SHA256 = (
    "9952864e50371675e7ea181cc57f2edd9eadd9189bfa0e9eda1e5cdd8f8ca61a"
)
FROZEN_REGISTRY_SHA256 = (
    "f5328b66c6b385dd081c1a27577d92b5c63fc18dc00d815f7d3d2561bb68e29f"
)
FROZEN_CALIBRATION_SHA256 = (
    "e82b196ab627a935fd571bba2b37aa637f040af5ba81fbab5183c1f15aa2e564"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_PRINT_CARDS = {
    "grass_clump": {
        "position": "TOP_LEFT",
        "role": "REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT",
        "sha256": "b1f6262895c31e8e507be31cebba09140e2a2582aa4f266ab05261fe50751d23",
    },
    "low_shrub": {
        "position": "TOP_RIGHT",
        "role": "REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT",
        "sha256": "810c7649ac729105367b3213bfafc467a036f4054244c424613da6c027c73610",
    },
    "young_tree": {
        "position": "BOTTOM_LEFT",
        "role": "REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT",
        "sha256": "0d994e838a2d7787ab3edfd8646e317390c790d92588c7ef9109778b843b40eb",
    },
    "unknown": {
        "position": "BOTTOM_RIGHT",
        "role": "UNREGISTERED_NEGATIVE_NOT_ACCURACY_HOLDOUT",
        "sha256": "04e7f49a1e66186bda7a9a1102985560eac0e3a1bffcec892e6dc522868c985b",
    },
}
_USB_ID_RE = re.compile(r"^[0-9a-f]{4}$")

ZERO_AUTHORITY = {
    "irrigation_execution": False,
    "pump_command": False,
    "serial_write": False,
    "gpio_access": False,
    "state_machine_write": False,
    "hardware_control": False,
    "execution_authority": False,
    "physical_authority": False,
    "physical_completion": False,
}


class UvcFrontendError(RuntimeError):
    """A runtime asset or frame violated the fail-closed frontend contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenFileBinding:
    path: Path
    sha256: str
    byte_count: int
    raw: bytes


def _bind_frozen_file(
    path: str | Path,
    *,
    expected_sha256: str,
    frozen_sha256: str | None,
    label: str,
) -> FrozenFileBinding:
    """Bind one non-symlink file from one byte snapshot before it is consumed."""

    expected = expected_sha256.lower()
    if not _SHA256_RE.fullmatch(expected):
        raise UvcFrontendError(f"{label} expected SHA-256 is malformed")
    if frozen_sha256 is not None and expected != frozen_sha256:
        raise UvcFrontendError(
            f"{label} expected SHA-256 does not equal the frozen contract"
        )
    candidate = Path(path).expanduser()
    try:
        candidate_stat = os.lstat(candidate)
    except OSError as exc:
        raise UvcFrontendError(f"{label} is unavailable: {candidate}") from exc
    if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISREG(
        candidate_stat.st_mode
    ):
        raise UvcFrontendError(f"{label} must be a regular non-symlink file")
    resolved = candidate.resolve(strict=True)
    raw = resolved.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise UvcFrontendError(
            f"{label} SHA-256 mismatch: expected={expected} observed={observed}"
        )
    return FrozenFileBinding(
        path=resolved,
        sha256=observed,
        byte_count=len(raw),
        raw=raw,
    )


def _decode_json_object(binding: FrozenFileBinding, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            binding.raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_json_without_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UvcFrontendError(f"cannot decode frozen {label}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise UvcFrontendError(f"frozen {label} must contain an object")
    return payload


def _sha256_rgb(rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(rgb.shape)).encode("ascii"))
    digest.update(b"|uint8|RGB|")
    digest.update(np.ascontiguousarray(rgb).tobytes(order="C"))
    return digest.hexdigest()


def _json_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UvcFrontendError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_object(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise UvcFrontendError(f"runtime JSON must be a regular non-symlink file: {resolved}")
    try:
        payload = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_json_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UvcFrontendError(f"cannot load runtime JSON {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise UvcFrontendError(f"runtime JSON must contain an object: {resolved}")
    return payload


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, field: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise UvcFrontendError(
            f"{field} keys mismatch: "
            f"missing={sorted(expected - actual)} unknown={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class PrintManifestBinding:
    path: Path
    sha256: str
    byte_count: int
    pdf_sha256: str
    cards: Mapping[str, Mapping[str, Any]]


def load_print_manifest_binding(
    path: str | Path,
    *,
    expected_sha256: str = FROZEN_PRINT_MANIFEST_SHA256,
) -> PrintManifestBinding:
    """Read/hash/decode the four-up manifest from one immutable byte snapshot."""

    binding = _bind_frozen_file(
        path,
        expected_sha256=expected_sha256,
        frozen_sha256=FROZEN_PRINT_MANIFEST_SHA256,
        label="four-up print manifest",
    )
    resolved = binding.path
    raw = binding.raw
    payload = _decode_json_object(binding, label="four-up print manifest")
    _require_exact_keys(
        payload,
        {
            "cards",
            "page",
            "pdf",
            "position_map",
            "print_settings",
            "schema",
            "status",
            "truth_boundary",
        },
        field="print manifest",
    )
    if payload["schema"] != PRINT_MANIFEST_SCHEMA:
        raise UvcFrontendError("unexpected print manifest schema")
    if payload["status"] != PRINT_MANIFEST_STATUS:
        raise UvcFrontendError("unexpected print manifest status")
    cards = payload["cards"]
    if not isinstance(cards, list) or len(cards) != 4:
        raise UvcFrontendError("print manifest must contain exactly four cards")
    card_keys = {
        "accuracy_evidence",
        "artist",
        "camera_recapture_evidence",
        "class_id",
        "display_name_zh",
        "geometry",
        "holdout_claimed",
        "license",
        "pageid",
        "position",
        "relative_path",
        "role",
        "sha256",
        "source_bytes",
        "source_height_px",
        "source_mode",
        "source_page",
        "source_path_relative_to_adventurex",
        "source_width_px",
    }
    bound_cards: dict[str, Mapping[str, Any]] = {}
    seen_positions: set[str] = set()
    for index, item in enumerate(cards):
        if not isinstance(item, Mapping):
            raise UvcFrontendError(f"print manifest card[{index}] must be an object")
        _require_exact_keys(item, card_keys, field=f"print manifest card[{index}]")
        class_id = item["class_id"]
        if class_id not in EXPECTED_PRINT_CARDS or class_id in bound_cards:
            raise UvcFrontendError("print manifest card classes are missing or duplicated")
        expected = EXPECTED_PRINT_CARDS[class_id]
        for field in ("position", "role", "sha256"):
            if item[field] != expected[field]:
                raise UvcFrontendError(
                    f"print manifest {class_id}.{field} does not match the frozen card"
                )
        if item["position"] in seen_positions:
            raise UvcFrontendError("print manifest card positions are duplicated")
        seen_positions.add(str(item["position"]))
        if (
            item["accuracy_evidence"] is not False
            or item["camera_recapture_evidence"] is not False
            or item["holdout_claimed"] is not False
        ):
            raise UvcFrontendError("print card would upgrade its evidence role")
        geometry = item["geometry"]
        if not isinstance(geometry, Mapping):
            raise UvcFrontendError("print card geometry must be an object")
        _require_exact_keys(
            geometry,
            {
                "card_height_mm",
                "card_width_mm",
                "card_x_mm",
                "card_y_mm",
                "image_height_mm",
                "image_width_mm",
                "image_x_mm",
                "image_y_mm",
            },
            field=f"print manifest {class_id}.geometry",
        )
        bound_cards[str(class_id)] = dict(item)
    if set(bound_cards) != set(EXPECTED_PRINT_CARDS):
        raise UvcFrontendError("print manifest does not contain the frozen four classes")

    pdf = payload["pdf"]
    if not isinstance(pdf, Mapping):
        raise UvcFrontendError("print manifest.pdf must be an object")
    _require_exact_keys(
        pdf, {"bytes", "page_count", "path_relative_to_adventurex", "sha256"},
        field="print manifest.pdf",
    )
    pdf_sha = pdf["sha256"]
    if (
        pdf_sha != FROZEN_PRINT_PDF_SHA256
        or pdf["page_count"] != 1
        or pdf["path_relative_to_adventurex"]
        != "output/pdf/RootScope_A4_four_up_field_cards_20260723.pdf"
        or pdf["bytes"] != 2320982
    ):
        raise UvcFrontendError("print manifest PDF binding is invalid")
    truth = payload["truth_boundary"]
    if not isinstance(truth, Mapping):
        raise UvcFrontendError("print manifest truth boundary is missing")
    if (
        truth.get("model_qualified") is not False
        or truth.get("generalization_claimed") is not False
        or truth.get("physical_or_irrigation_authority") is not False
    ):
        raise UvcFrontendError("print manifest truth boundary was upgraded")
    return PrintManifestBinding(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
        pdf_sha256=pdf_sha,
        cards=bound_cards,
    )


def bind_print_manifest_to_registry(
    print_binding: PrintManifestBinding, registry_path: str | Path
) -> None:
    """Require the three positive print sources and forbid unknown registration."""

    registry = load_template_registry(registry_path)
    actual = {
        template.class_name: template.raw_sha256 for template in registry.templates
    }
    expected = {
        class_id: record["sha256"]
        for class_id, record in EXPECTED_PRINT_CARDS.items()
        if class_id != "unknown"
    }
    if actual != expected:
        raise UvcFrontendError(
            "registered template registry does not exactly bind the three print positives"
        )
    unknown = print_binding.cards["unknown"]
    if unknown["role"] != "UNREGISTERED_NEGATIVE_NOT_ACCURACY_HOLDOUT":
        raise UvcFrontendError("unknown print card must remain unregistered")


@dataclass(frozen=True)
class CalibrationBinding:
    calibration: Calibration
    manifest_path: Path
    manifest_sha256: str
    provenance: Mapping[str, Any]


def load_calibration_binding(
    path: str | Path,
    *,
    expected_sha256: str = FROZEN_CALIBRATION_SHA256,
) -> CalibrationBinding:
    """Load the existing Omega calibration without upgrading its claims."""

    binding = _bind_frozen_file(
        path,
        expected_sha256=expected_sha256,
        frozen_sha256=FROZEN_CALIBRATION_SHA256,
        label="Omega calibration manifest",
    )
    resolved = binding.path
    payload = _decode_json_object(binding, label="Omega calibration manifest")
    if payload.get("schema_version") != CALIBRATION_MANIFEST_SCHEMA:
        raise UvcFrontendError("unexpected Omega calibration manifest schema")
    calibration_payload = payload.get("calibration")
    provenance = payload.get("calibration_provenance")
    if not isinstance(calibration_payload, Mapping) or not isinstance(
        provenance, Mapping
    ):
        raise UvcFrontendError("calibration or calibration provenance is missing")
    if provenance.get("formal_distribution_free_coverage_guarantee") is not False:
        raise UvcFrontendError("calibration cannot claim formal coverage")
    if provenance.get("holdout_reevaluated_for_board_replay") is not False:
        raise UvcFrontendError("calibration provenance changed its holdout boundary")

    normalized = dict(calibration_payload)
    for key in ("class_order", "conformal_nonconformity", "calibration_roles"):
        value = normalized.get(key)
        if not isinstance(value, list):
            raise UvcFrontendError(f"calibration.{key} must be an array")
        normalized[key] = tuple(value)
    try:
        calibration = Calibration(**normalized)
    except (TypeError, ValueError) as exc:
        raise UvcFrontendError(f"invalid Omega calibration: {exc}") from exc
    if calibration.class_order != tuple(ROOTSCOPE_CLASS_ORDER):
        raise UvcFrontendError("Omega calibration class order is not frozen RootScope order")
    if calibration.status != "MACHINE_CURATED_EXPERIMENTAL_NOT_QUALIFIED":
        raise UvcFrontendError("Omega calibration status would upgrade the model claim")
    return CalibrationBinding(
        calibration=calibration,
        manifest_path=resolved,
        manifest_sha256=binding.sha256,
        provenance=dict(provenance),
    )


def validate_stable_device_syntax(value: str) -> str:
    """Accept one explicit stable V4L2 alias; never enumerate or guess."""

    if not isinstance(value, str) or not value:
        raise UvcFrontendError("--device must be a non-empty path")
    if "\\" in value:
        raise UvcFrontendError("--device must use POSIX path separators")
    path = PurePosixPath(value)
    prefix = PurePosixPath("/dev/v4l/by-id")
    if not path.is_absolute() or path.parent != prefix or path.name in {"", ".", ".."}:
        raise UvcFrontendError(
            "--device must be one explicit /dev/v4l/by-id/<camera>-video-indexN path"
        )
    if value != path.as_posix():
        raise UvcFrontendError("--device must be a normalized stable path")
    return value


@dataclass(frozen=True)
class ExpectedCameraIdentity:
    usb_vid: str
    usb_pid: str
    usb_serial: str

    def __post_init__(self) -> None:
        vid = self.usb_vid.lower()
        pid = self.usb_pid.lower()
        if not _USB_ID_RE.fullmatch(vid) or not _USB_ID_RE.fullmatch(pid):
            raise UvcFrontendError("USB VID/PID must each be exactly four hex digits")
        if (
            not isinstance(self.usb_serial, str)
            or not self.usb_serial.strip()
            or len(self.usb_serial) > 256
            or any(character in self.usb_serial for character in "\r\n\0")
        ):
            raise UvcFrontendError("USB serial must be one explicit non-empty value")
        object.__setattr__(self, "usb_vid", vid)
        object.__setattr__(self, "usb_pid", pid)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def resolve_live_device(value: str) -> tuple[str, str]:
    """Verify the explicit alias on Linux without looking at any sibling device."""

    configured = validate_stable_device_syntax(value)
    if platform.system() != "Linux":
        raise UvcFrontendError("live UVC frontend requires Linux")
    path = Path(configured)
    if not path.exists():
        raise UvcFrontendError("the explicit stable camera path does not exist")
    resolved = path.resolve(strict=True)
    if resolved.parts[:2] != ("/", "dev"):
        raise UvcFrontendError("stable camera alias must resolve below /dev")
    if not stat.S_ISCHR(resolved.stat().st_mode):
        raise UvcFrontendError("stable camera alias must resolve to a character device")
    return configured, str(resolved)


def read_explicit_usb_identity(
    configured: str,
    expected: ExpectedCameraIdentity,
) -> Mapping[str, Any]:
    """Read only the resolved target's bounded sysfs ancestor chain."""

    configured, resolved = resolve_live_device(configured)
    kernel_name = Path(resolved).name
    if not kernel_name or "/" in kernel_name:
        raise UvcFrontendError("resolved camera node name is invalid")
    sysfs_link = Path("/sys/class/video4linux") / kernel_name / "device"
    try:
        sysfs_device = sysfs_link.resolve(strict=True)
    except OSError as exc:
        raise UvcFrontendError(
            "explicit camera sysfs binding is unavailable"
        ) from exc
    identity: dict[str, str] | None = None
    candidate = sysfs_device
    for _ in range(10):
        vendor_path = candidate / "idVendor"
        product_path = candidate / "idProduct"
        serial_path = candidate / "serial"
        if vendor_path.is_file() and product_path.is_file() and serial_path.is_file():
            identity = {
                "usb_vid": vendor_path.read_text(encoding="ascii").strip().lower(),
                "usb_pid": product_path.read_text(encoding="ascii").strip().lower(),
                "usb_serial": serial_path.read_text(encoding="utf-8").strip(),
                "usb_device_sysfs": str(candidate),
            }
            break
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    if identity is None:
        raise UvcFrontendError(
            "VID/PID/serial were not found on the explicit camera sysfs chain"
        )
    actual_tuple = (
        identity["usb_vid"],
        identity["usb_pid"],
        identity["usb_serial"],
    )
    expected_tuple = (
        expected.usb_vid,
        expected.usb_pid,
        expected.usb_serial,
    )
    if actual_tuple != expected_tuple:
        raise UvcFrontendError(
            "explicit camera identity mismatch: "
            f"actual={actual_tuple!r} expected={expected_tuple!r}"
        )
    return {
        "configured_device": configured,
        "resolved_device": resolved,
        "kernel_video_node": kernel_name,
        **identity,
        "identity_match": True,
        "device_discovery_used": False,
    }


@dataclass(frozen=True)
class FrontendRequest:
    device: str
    expected_camera: ExpectedCameraIdentity
    print_manifest: Path
    mode: str
    frames: int
    warmup_frames: int
    interval_seconds: float
    width: int
    height: int
    fps: float
    output_root: Path
    jsonl_path: Path
    annotated_dir: Path | None = None


def validate_request(request: FrontendRequest) -> None:
    validate_stable_device_syntax(request.device)
    if not isinstance(request.expected_camera, ExpectedCameraIdentity):
        raise UvcFrontendError("expected_camera identity binding is required")
    if request.mode not in {"one-shot", "bounded"}:
        raise UvcFrontendError("mode must be one-shot or bounded")
    if request.mode == "one-shot" and request.frames != 1:
        raise UvcFrontendError("one-shot mode requires exactly one frame")
    if not 1 <= request.frames <= MAX_FRAMES:
        raise UvcFrontendError(f"frames must be in [1,{MAX_FRAMES}]")
    if not 0 <= request.warmup_frames <= 120:
        raise UvcFrontendError("warmup_frames must be in [0,120]")
    if not 0.0 <= request.interval_seconds <= 10.0:
        raise UvcFrontendError("interval_seconds must be in [0,10]")
    if not 320 <= request.width <= 4096 or not 240 <= request.height <= 2160:
        raise UvcFrontendError("camera width/height are outside the bounded range")
    if not math.isfinite(request.fps) or not 1.0 <= request.fps <= 60.0:
        raise UvcFrontendError("fps must be finite and in [1,60]")
    print_manifest = request.print_manifest.expanduser()
    if not print_manifest.is_absolute():
        raise UvcFrontendError("print manifest must be one explicit absolute path")
    if (
        not print_manifest.exists()
        or not print_manifest.is_file()
        or print_manifest.is_symlink()
    ):
        raise UvcFrontendError(
            "print manifest must be an existing regular non-symlink file"
        )
    output_root = request.output_root.expanduser()
    if not output_root.is_absolute():
        raise UvcFrontendError("output root must be one explicit absolute path")
    if not output_root.exists() or not output_root.is_dir() or output_root.is_symlink():
        raise UvcFrontendError("output root must be an existing non-symlink directory")
    root = output_root.resolve(strict=True)
    if root != output_root:
        raise UvcFrontendError(
            "output root must be canonical and contain no symlink ancestors"
        )
    output_input = request.jsonl_path.expanduser()
    if not output_input.is_absolute():
        raise UvcFrontendError("JSONL output must be one explicit absolute path")
    output = output_input.resolve()
    if output.parent != root:
        raise UvcFrontendError("JSONL output must be a direct child of output root")
    if output.exists() or output.is_symlink():
        raise UvcFrontendError(f"JSONL output exists; overwrite refused: {output}")
    final_manifest = output.with_suffix(".final.json")
    if final_manifest.exists() or final_manifest.is_symlink():
        raise UvcFrontendError("final manifest path exists; overwrite refused")
    if request.annotated_dir is not None:
        annotated_input = request.annotated_dir.expanduser()
        if not annotated_input.is_absolute():
            raise UvcFrontendError(
                "annotated output directory must be one explicit absolute path"
            )
        annotated = annotated_input.resolve()
        if annotated.parent != root:
            raise UvcFrontendError(
                "annotated output directory must be a direct child of output root"
            )
        if annotated.exists() or annotated.is_symlink():
            raise UvcFrontendError(
                f"annotated output directory exists; overwrite refused: {annotated}"
            )


class FrameSource(Protocol):
    configured_device: str
    resolved_device: str

    def read_rgb(self) -> np.ndarray: ...

    def negotiated_settings(self) -> Mapping[str, Any]: ...

    def close(self) -> Mapping[str, Any]: ...


class LiveUvcFrameSource:
    """Open one verified camera path and expose RGB frames until explicitly closed."""

    def __init__(self, request: FrontendRequest) -> None:
        before = read_explicit_usb_identity(
            request.device, request.expected_camera
        )
        configured = str(before["configured_device"])
        resolved = str(before["resolved_device"])
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - X5-only path
            raise UvcFrontendError("OpenCV is required for live UVC") from exc
        self.configured_device = configured
        self.resolved_device = resolved
        self._cv2 = cv2
        self._closed = True
        self._expected_camera = request.expected_camera
        self._identity_before_open = dict(before)
        self._identity_after_open: Mapping[str, Any] | None = None
        self._close_receipt: Mapping[str, Any] | None = None
        capture: Any | None = None
        try:
            capture = cv2.VideoCapture(configured, cv2.CAP_V4L2)
            self._capture = capture
            self._closed = False
            if not capture.isOpened():
                raise UvcFrontendError("the explicit camera could not be opened")
            self._capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
            )
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, request.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, request.height)
            self._capture.set(cv2.CAP_PROP_FPS, request.fps)
            actual = self.negotiated_settings()
            if (
                actual["fourcc"] != "MJPG"
                or actual["width"] != request.width
                or actual["height"] != request.height
                or abs(float(actual["fps"]) - request.fps) > 1.0
            ):
                raise UvcFrontendError(
                    "camera mode negotiation mismatch: "
                    f"requested=MJPG/{request.width}x{request.height}@{request.fps} "
                    f"actual={actual['fourcc']}/{actual['width']}x"
                    f"{actual['height']}@{actual['fps']}"
                )
            after = read_explicit_usb_identity(
                request.device, request.expected_camera
            )
            identity_fields = (
                "configured_device",
                "resolved_device",
                "kernel_video_node",
                "usb_vid",
                "usb_pid",
                "usb_serial",
                "usb_device_sysfs",
            )
            if any(before[field] != after[field] for field in identity_fields):
                raise UvcFrontendError(
                    "explicit camera identity changed across the open boundary"
                )
            self._identity_after_open = dict(after)
        except BaseException:
            if capture is not None:
                try:
                    capture.release()
                finally:
                    self._closed = True
            raise

    def negotiated_settings(self) -> Mapping[str, Any]:
        cv2 = self._cv2
        fourcc_value = int(round(self._capture.get(cv2.CAP_PROP_FOURCC)))
        fourcc = "".join(
            chr((fourcc_value >> (8 * offset)) & 0xFF) for offset in range(4)
        )
        return {
            "backend": "opencv_v4l2",
            "configured_device": self.configured_device,
            "resolved_device": self.resolved_device,
            "fourcc": fourcc,
            "width": int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": round(float(self._capture.get(cv2.CAP_PROP_FPS)), 4),
            "expected_identity": {
                "usb_vid": self._identity_before_open["usb_vid"],
                "usb_pid": self._identity_before_open["usb_pid"],
                "usb_serial": self._identity_before_open["usb_serial"],
            },
            "identity_before_open": dict(self._identity_before_open),
            "identity_after_open": (
                dict(self._identity_after_open)
                if self._identity_after_open is not None
                else None
            ),
            "identity_match_before_and_after_open": (
                self._identity_after_open is not None
            ),
            "device_discovery_used": False,
        }

    def read_rgb(self) -> np.ndarray:
        ok, bgr = self._capture.read()
        if not ok or bgr is None:
            raise UvcFrontendError("UVC frame read failed")
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def close(self) -> Mapping[str, Any]:
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
            release_error = (
                f"{release_error}; release readback failed: {type(exc).__name__}: {exc}"
                if release_error
                else f"release readback failed: {type(exc).__name__}: {exc}"
            )
        identity_after_close: Mapping[str, Any] | None = None
        identity_error: str | None = None
        try:
            identity_after_close = read_explicit_usb_identity(
                self.configured_device, self._expected_camera
            )
        except Exception as exc:
            identity_error = f"{type(exc).__name__}: {exc}"
        identity_fields = (
            "configured_device",
            "resolved_device",
            "kernel_video_node",
            "usb_vid",
            "usb_pid",
            "usb_serial",
            "usb_device_sysfs",
        )
        identity_match = (
            identity_after_close is not None
            and all(
                self._identity_before_open[field] == identity_after_close[field]
                for field in identity_fields
            )
        )
        self._close_receipt = {
            "release_called": release_called,
            "opened_after_release": opened_after_release,
            "release_error": release_error,
            "release_completed": (
                release_error is None and opened_after_release is False
            ),
            "identity_after_close": (
                dict(identity_after_close)
                if identity_after_close is not None
                else None
            ),
            "identity_match_after_close": identity_match,
            "identity_error": identity_error,
            "device_discovery_used": False,
        }
        return dict(self._close_receipt)


class SafeOutputDirectory:
    """Pinned direct-child output directory with exclusive, no-replace writes."""

    def __init__(
        self,
        path: Path,
        *,
        expected_identity: tuple[int, int] | None = None,
    ) -> None:
        candidate = path.expanduser()
        if not candidate.is_absolute():
            raise UvcFrontendError("safe output directory must be absolute")
        before = os.lstat(candidate)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise UvcFrontendError("safe output directory must be a non-symlink directory")
        resolved = candidate.resolve(strict=True)
        if resolved != candidate:
            raise UvcFrontendError(
                "safe output directory must be canonical and contain no symlink ancestors"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        directory_fd: int | None = None
        if os.name != "nt":
            directory_fd = os.open(candidate, flags)
            opened = os.fstat(directory_fd)
            if not stat.S_ISDIR(opened.st_mode):
                os.close(directory_fd)
                raise UvcFrontendError("pinned output descriptor is not a directory")
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                os.close(directory_fd)
                raise UvcFrontendError("output directory changed while it was pinned")
        self.path = candidate
        self._fd = directory_fd
        self._identity = (int(before.st_dev), int(before.st_ino))
        if expected_identity is not None and self._identity != expected_identity:
            self.close()
            raise UvcFrontendError("created child output directory identity changed")

    @staticmethod
    def _name(value: str) -> str:
        if (
            not isinstance(value, str)
            or not value
            or value in {".", ".."}
            or Path(value).name != value
            or "/" in value
            or "\\" in value
        ):
            raise UvcFrontendError("output name must be one plain direct-child name")
        return value

    def _revalidate(self) -> None:
        current = os.lstat(self.path)
        if (
            stat.S_ISLNK(current.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (int(current.st_dev), int(current.st_ino)) != self._identity
        ):
            raise UvcFrontendError("pinned output directory path identity changed")

    def _open_exclusive(self, name: str, *, mode: int = 0o640) -> int:
        child = self._name(name)
        self._revalidate()
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        if self._fd is not None and os.open in os.supports_dir_fd:
            descriptor = os.open(child, flags, mode, dir_fd=self._fd)
        else:
            descriptor = os.open(self.path / child, flags, mode)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            os.close(descriptor)
            raise UvcFrontendError("exclusive output child is not a regular file")
        return descriptor

    def verify_child(self, name: str, identity: tuple[int, int]) -> None:
        child = self._name(name)
        self._revalidate()
        if self._fd is not None and os.stat in os.supports_dir_fd:
            observed = os.stat(child, dir_fd=self._fd, follow_symlinks=False)
        else:
            observed = os.lstat(self.path / child)
        if (
            not stat.S_ISREG(observed.st_mode)
            or (int(observed.st_dev), int(observed.st_ino)) != identity
        ):
            raise UvcFrontendError("output child identity changed during the session")

    def create_child_directory(self, name: str) -> "SafeOutputDirectory":
        child = self._name(name)
        self._revalidate()
        if self._fd is not None and os.mkdir in os.supports_dir_fd:
            os.mkdir(child, mode=0o750, dir_fd=self._fd)
            created = os.stat(child, dir_fd=self._fd, follow_symlinks=False)
        else:
            os.mkdir(self.path / child, mode=0o750)
            created = os.lstat(self.path / child)
        if stat.S_ISLNK(created.st_mode) or not stat.S_ISDIR(created.st_mode):
            raise UvcFrontendError("created output child is not a safe directory")
        return SafeOutputDirectory(
            self.path / child,
            expected_identity=(int(created.st_dev), int(created.st_ino)),
        )

    def publish_bytes(self, name: str, payload: bytes) -> tuple[str, int]:
        """Publish bytes with an atomic hard-link that can never replace a target."""

        child = self._name(name)
        temporary = self._name(f".{child}.partial")
        descriptor = self._open_exclusive(temporary)
        temporary_created = True
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._revalidate()
            if (
                self._fd is not None
                and os.link in os.supports_dir_fd
                and os.unlink in os.supports_dir_fd
            ):
                os.link(
                    temporary,
                    child,
                    src_dir_fd=self._fd,
                    dst_dir_fd=self._fd,
                    follow_symlinks=False,
                )
                os.unlink(temporary, dir_fd=self._fd)
            else:
                os.link(
                    self.path / temporary,
                    self.path / child,
                    follow_symlinks=False,
                )
                os.unlink(self.path / temporary)
            temporary_created = False
            if self._fd is not None:
                os.fsync(self._fd)
            return hashlib.sha256(payload).hexdigest(), len(payload)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_created:
                try:
                    if self._fd is not None and os.unlink in os.supports_dir_fd:
                        os.unlink(temporary, dir_fd=self._fd)
                    else:
                        os.unlink(self.path / temporary)
                except FileNotFoundError:
                    pass

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class JsonlLedger:
    """Small exclusive-create SHA-256 chain for one bounded foreground run."""

    def __init__(
        self,
        path: Path,
        *,
        directory: SafeOutputDirectory | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self._directory = directory or SafeOutputDirectory(self.path.parent)
        self._owns_directory = directory is None
        descriptor = self._directory._open_exclusive(self.path.name)
        opened = os.fstat(descriptor)
        self._identity = (int(opened.st_dev), int(opened.st_ino))
        self._stream = os.fdopen(descriptor, "wb", closefd=True)
        self._sequence = 0
        self._previous_sha256: str | None = None
        self._file_digest = hashlib.sha256()
        self._byte_count = 0

    def append(self, event: str, payload: Mapping[str, Any]) -> str:
        record: dict[str, Any] = {
            "schema": JSONL_SCHEMA,
            "sequence": self._sequence,
            "previous_record_sha256": self._previous_sha256,
            "event": event,
            "payload": dict(payload),
        }
        record_sha256 = hashlib.sha256(_canonical_bytes(record)).hexdigest()
        record["record_sha256"] = record_sha256
        encoded = (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self._stream.write(encoded)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._file_digest.update(encoded)
        self._byte_count += len(encoded)
        self._sequence += 1
        self._previous_sha256 = record_sha256
        return record_sha256

    @property
    def record_count(self) -> int:
        return self._sequence

    @property
    def root_sha256(self) -> str | None:
        return self._previous_sha256

    @property
    def file_sha256(self) -> str:
        return self._file_digest.hexdigest()

    @property
    def byte_count(self) -> int:
        return self._byte_count

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()
        self._directory.verify_child(self.path.name, self._identity)
        if self._owns_directory:
            self._directory.close()


def _validate_negotiated_camera_binding(
    negotiated: Mapping[str, Any], request: FrontendRequest
) -> None:
    if negotiated.get("configured_device") != request.device:
        raise UvcFrontendError("camera backend changed the explicit configured path")
    if negotiated.get("device_discovery_used") is not False:
        raise UvcFrontendError("camera backend did not preserve the no-discovery contract")
    if negotiated.get("identity_match_before_and_after_open") is not True:
        raise UvcFrontendError("camera identity was not verified before and after open")
    expected = request.expected_camera.to_dict()
    if negotiated.get("expected_identity") != expected:
        raise UvcFrontendError("camera backend expected identity binding changed")
    before = negotiated.get("identity_before_open")
    after = negotiated.get("identity_after_open")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise UvcFrontendError("camera identity evidence is incomplete")
    for field, value in expected.items():
        if before.get(field) != value or after.get(field) != value:
            raise UvcFrontendError(f"camera {field} did not match before and after open")
    for field in ("resolved_device", "kernel_video_node", "usb_device_sysfs"):
        if (
            not isinstance(before.get(field), str)
            or before.get(field) != after.get(field)
        ):
            raise UvcFrontendError(f"camera {field} changed across open")


def _all_false_authority(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(item is False for item in value.values())
    )


def _validate_frame_result_contract(result: Mapping[str, Any]) -> None:
    if result.get("schema") != SCHEMA:
        raise UvcFrontendError("frame processor schema is invalid")
    final = result.get("final_consensus")
    if not isinstance(final, Mapping) or final.get("passed") not in {True, False}:
        raise UvcFrontendError("frame processor omitted final consensus")
    if not _all_false_authority(result.get("authority")):
        raise UvcFrontendError("frame result top-level authority contract is invalid")
    if not _all_false_authority(final.get("authority")):
        raise UvcFrontendError("frame result consensus authority contract is invalid")
    compute = result.get("compute_boundary")
    if (
        not isinstance(compute, Mapping)
        or compute.get("seed17_provider") != CPU_PROVIDER
        or compute.get("plant_bpu_selected_bin") is not None
        or compute.get("plant_bpu_used") is not False
    ):
        raise UvcFrontendError("frame result CPU/BPU boundary is invalid")
    if final["passed"] is True:
        if not all(
            isinstance(result.get(field), Mapping)
            for field in (
                "semantic_hypothesis",
                "omega_ood_abstention",
                "registered_card_geometry",
            )
        ):
            raise UvcFrontendError("accepted frame is missing one or more evidence layers")
        if result["omega_ood_abstention"].get("decision") != "CLASSIFY":
            raise UvcFrontendError("accepted frame did not pass Omega abstention")
        if (
            result["registered_card_geometry"].get("contract_valid_pass_count")
            != 1
        ):
            raise UvcFrontendError("accepted frame lacks exactly one geometry pass")
        if final.get("status") != FINAL_ACCEPT or not final.get("display_class"):
            raise UvcFrontendError("accepted frame final binding is incomplete")


def evaluate_rgb_frame(
    *,
    rgb: np.ndarray,
    frame_index: int,
    runner: Any,
    calibration_binding: CalibrationBinding,
    registry_path: Path,
    thresholds: DemoThresholds,
    matcher_config: MatcherConfig,
    asset_binding: Mapping[str, Any],
    dual_evaluator: Callable[..., Mapping[str, Any]] = evaluate_dual_path_demo,
    matcher: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run all four evidence layers for one already captured RGB fixture/frame."""

    array = np.asarray(rgb)
    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[2] != 3
        or array.shape[0] < 2
        or array.shape[1] < 2
    ):
        raise UvcFrontendError("frame must be uint8 RGB HxWx3 and at least 2x2")
    array = np.ascontiguousarray(array)
    seed_asset = asset_binding.get("seed17_onnx")
    registry_asset = asset_binding.get("registered_template_registry")
    print_asset = asset_binding.get("four_up_print_manifest")
    plant_bpu = asset_binding.get("plant_bpu")
    if (
        not isinstance(seed_asset, Mapping)
        or seed_asset.get("sha256") != SEED17_MODEL_SHA256
        or seed_asset.get("provider") != CPU_PROVIDER
    ):
        raise UvcFrontendError("runtime asset binding omitted the frozen seed17 CPU model")
    if (
        not isinstance(plant_bpu, Mapping)
        or plant_bpu.get("selected_bin") is not None
        or plant_bpu.get("used") is not False
    ):
        raise UvcFrontendError("runtime asset binding promoted the plant BPU")
    registry_resolved = registry_path.expanduser().resolve(strict=True)
    if (
        not isinstance(registry_asset, Mapping)
        or registry_asset.get("path") != str(registry_resolved)
        or registry_asset.get("sha256") != _sha256_file(registry_resolved)
    ):
        raise UvcFrontendError("registered template registry changed after preflight")
    if (
        not isinstance(print_asset, Mapping)
        or print_asset.get("card_count") != 4
        or not isinstance(print_asset.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", print_asset["sha256"])
    ):
        raise UvcFrontendError("four-up print manifest binding is incomplete")

    with tempfile.TemporaryDirectory(prefix="rootscope-uvc-frame-") as temporary:
        query_path = Path(temporary) / "frame.png"
        Image.fromarray(array, mode="RGB").save(query_path, format="PNG")
        call: dict[str, Any] = {
            "query_path": query_path,
            "runner": runner,
            "registry_path": registry_resolved,
            "thresholds": thresholds,
            "matcher_config": matcher_config,
        }
        if matcher is not None:
            call["matcher"] = matcher
        dual = dict(dual_evaluator(**call))

    semantic = dual.get("semantic")
    geometry = dual.get("geometry")
    dual_consensus = dual.get("consensus")
    if not isinstance(semantic, Mapping):
        raise UvcFrontendError("dual path omitted semantic hypothesis")
    if not isinstance(geometry, Mapping):
        raise UvcFrontendError("dual path omitted geometry evidence")
    if not isinstance(dual_consensus, Mapping):
        raise UvcFrontendError("dual path omitted consensus evidence")
    logits = semantic.get("raw_logits")
    if not isinstance(logits, list) or len(logits) != len(ROOTSCOPE_CLASS_ORDER):
        raise UvcFrontendError("semantic hypothesis omitted frozen four logits")

    omega = decide(
        logits,
        evaluate_quality(array),
        calibration_binding.calibration,
    )
    omega_dict = omega.to_dict()
    reasons: list[str] = []
    if semantic.get("status") != "DEMO_HYPOTHESIS":
        reasons.append("SEMANTIC_HYPOTHESIS_STATUS_INVALID")
    if semantic.get("raw_top1_class") not in ROOTSCOPE_CLASS_ORDER:
        reasons.append("SEMANTIC_CLASS_INVALID")
    if not _all_false_authority(semantic.get("authority")):
        reasons.append("SEMANTIC_AUTHORITY_CONTRACT_INVALID")
    if omega.decision != "CLASSIFY":
        reasons.append("OMEGA_OOD_ABSTAIN")
        reasons.extend(f"OMEGA_{reason}" for reason in omega.reasons)
    if omega.raw_top1_class == "unknown":
        reasons.append("UNKNOWN_CLASS_FAIL_CLOSED")
    if geometry.get("contract_valid_pass_count") != 1:
        reasons.append("GEOMETRY_NOT_EXACTLY_ONE_REGISTERED_PASS")
    if dual.get("experimental_consensus_passed") is not True:
        reasons.append("DUAL_PATH_CONSENSUS_REJECTED")
    if dual_consensus.get("passed") is not True:
        reasons.append("DUAL_PATH_FINAL_BINDING_REJECTED")
    selected_class = dual_consensus.get("selected_template_class")
    if omega.predicted_class != selected_class:
        reasons.append("OMEGA_GEOMETRY_CLASS_DISAGREEMENT")
    if not _all_false_authority(dual_consensus.get("authority")):
        reasons.append("DUAL_PATH_AUTHORITY_CONTRACT_INVALID")
    if not _all_false_authority(dual.get("authority")):
        reasons.append("DUAL_PATH_TOP_LEVEL_AUTHORITY_INVALID")

    reasons = list(dict.fromkeys(reasons))
    passed = not reasons
    semantic_public = copy.deepcopy(dict(semantic))
    query = semantic_public.get("query")
    if isinstance(query, dict):
        query["path"] = "EPHEMERAL_CAPTURE_FRAME_NOT_RETAINED"
        query["path_retained"] = False

    return {
        "schema": SCHEMA,
        "event": "frame_result",
        "frame_index": frame_index,
        "captured_rgb": {
            "width": int(array.shape[1]),
            "height": int(array.shape[0]),
            "channels": 3,
            "dtype": "uint8",
            "color_order": "RGB",
            "rgb_sha256": _sha256_rgb(array),
            "raw_frame_retained": False,
        },
        "runtime_assets": dict(asset_binding),
        "semantic_hypothesis": semantic_public,
        "omega_ood_abstention": {
            **omega_dict,
            "calibration_manifest_sha256": calibration_binding.manifest_sha256,
            "calibration_status": calibration_binding.calibration.status,
            "formal_distribution_free_coverage_guarantee": False,
        },
        "registered_card_geometry": copy.deepcopy(dict(geometry)),
        "final_consensus": {
            "status": FINAL_ACCEPT if passed else FINAL_REJECT,
            "passed": passed,
            "display_class": selected_class if passed else None,
            "reject_reasons": reasons,
            "claim_scope": (
                "REGISTERED_PRINTED_CARD_DISPLAY_ONLY_NOT_GENERAL_PLANT_"
                "RECOGNITION_NOT_IRRIGATION_DECISION"
            ),
            "authority": dict(ZERO_AUTHORITY),
        },
        "compute_boundary": {
            "seed17_provider": CPU_PROVIDER,
            "seed17_cpu_executed": True,
            "plant_bpu_selected_bin": None,
            "plant_bpu_used": False,
            "generic_bpu_aux_used": False,
        },
        "truth_boundary": {
            "camera_frame_processed": True,
            "camera_qualified": False,
            "general_plant_recognition": False,
            "model_candidate": False,
            "model_qualified": False,
            "accuracy_evidence": False,
            "registered_demo_references_are_holdout": False,
            "physical_or_irrigation_completion": False,
        },
        "authority": dict(ZERO_AUTHORITY),
    }


def _frame_error(index: int, exc: Exception) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "event": "frame_result",
        "frame_index": index,
        "captured_rgb": None,
        "runtime_assets": None,
        "semantic_hypothesis": None,
        "omega_ood_abstention": None,
        "registered_card_geometry": None,
        "final_consensus": {
            "status": FINAL_REJECT,
            "passed": False,
            "display_class": None,
            "reject_reasons": [
                f"FRAME_PROCESSING_ERROR:{type(exc).__name__}:{exc}"
            ],
            "claim_scope": "FAIL_CLOSED_MISSING_EVIDENCE",
            "authority": dict(ZERO_AUTHORITY),
        },
        "compute_boundary": {
            "seed17_provider": CPU_PROVIDER,
            "seed17_cpu_executed": False,
            "plant_bpu_selected_bin": None,
            "plant_bpu_used": False,
            "generic_bpu_aux_used": False,
        },
        "truth_boundary": {
            "camera_frame_processed": False,
            "camera_qualified": False,
            "general_plant_recognition": False,
            "model_candidate": False,
            "model_qualified": False,
            "accuracy_evidence": False,
            "physical_or_irrigation_completion": False,
        },
        "authority": dict(ZERO_AUTHORITY),
    }


def _annotate_frame(
    rgb: np.ndarray,
    result: Mapping[str, Any],
    output: Path,
    *,
    directory: SafeOutputDirectory,
) -> tuple[str, int]:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    final = result["final_consensus"]
    semantic = result.get("semantic_hypothesis") or {}
    omega = result.get("omega_ood_abstention") or {}
    geometry = result.get("registered_card_geometry") or {}
    color = (20, 160, 75) if final["passed"] else (220, 55, 55)
    lines = [
        f"SEMANTIC: {semantic.get('raw_top1_class', 'UNAVAILABLE')}",
        f"OMEGA OOD: {omega.get('decision', 'UNAVAILABLE')}",
        f"GEOMETRY PASSES: {geometry.get('contract_valid_pass_count', 0)}",
        f"FINAL: {final['status']}",
    ]
    line_height = 18
    bar_height = 12 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, bar_height), fill=(10, 15, 24))
    for line_index, line in enumerate(lines):
        draw.text((10, 6 + line_index * line_height), line, fill=(245, 245, 245))
    draw.rectangle((1, 1, image.width - 2, image.height - 2), outline=color, width=5)
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92, subsampling=0)
    return directory.publish_bytes(output.name, buffer.getvalue())


def _write_final_manifest_exclusive(
    path: Path,
    payload: Mapping[str, Any],
    *,
    directory: SafeOutputDirectory,
) -> tuple[str, int]:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return directory.publish_bytes(path.name, encoded)


@dataclass(frozen=True)
class SessionOutcome:
    status: str
    exit_code: int
    jsonl_path: Path
    final_manifest_path: Path
    captured_frames: int
    accepted_frames: int
    rejected_frames: int
    processing_errors: int
    camera_released: bool


def run_bounded_frontend(
    request: FrontendRequest,
    *,
    print_binding: PrintManifestBinding,
    frame_processor: Callable[[np.ndarray, int], Mapping[str, Any]],
    source_factory: Callable[[FrontendRequest], FrameSource] = LiveUvcFrameSource,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> SessionOutcome:
    """Run one finite foreground session and release the source in ``finally``."""

    validate_request(request)
    if not isinstance(print_binding, PrintManifestBinding):
        raise UvcFrontendError("validated print manifest binding is required")
    if (
        print_binding.path
        != request.print_manifest.expanduser().resolve(strict=True)
        or set(print_binding.cards) != set(EXPECTED_PRINT_CARDS)
    ):
        raise UvcFrontendError("print manifest binding does not match the request")
    output = request.jsonl_path.expanduser().resolve()
    annotated = (
        request.annotated_dir.expanduser().resolve()
        if request.annotated_dir is not None
        else None
    )
    output_directory = SafeOutputDirectory(request.output_root.expanduser().resolve())
    annotated_directory = (
        output_directory.create_child_directory(annotated.name)
        if annotated is not None
        else None
    )
    ledger = JsonlLedger(output, directory=output_directory)
    source: FrameSource | None = None
    camera_released = False
    captured = 0
    accepted = 0
    rejected = 0
    processing_errors = 0
    fatal_error: str | None = None
    negotiated: Mapping[str, Any] | None = None
    close_receipt: Mapping[str, Any] | None = None
    ledger.append(
        "session_start",
        {
            "started_utc": _utc_now(),
            "request": {
                "device": request.device,
                "expected_camera": request.expected_camera.to_dict(),
                "print_manifest": str(request.print_manifest.expanduser().resolve()),
                "print_manifest_sha256": print_binding.sha256,
                "print_manifest_bytes": print_binding.byte_count,
                "print_pdf_sha256": print_binding.pdf_sha256,
                "mode": request.mode,
                "frames": request.frames,
                "warmup_frames": request.warmup_frames,
                "interval_seconds": request.interval_seconds,
                "width": request.width,
                "height": request.height,
                "fps": request.fps,
                "output_root": str(request.output_root.expanduser().resolve()),
                "jsonl_path": str(output),
                "annotated_dir": str(annotated) if annotated is not None else None,
                "device_discovery_used": False,
            },
            "compute_policy": {
                "seed17_provider": CPU_PROVIDER,
                "plant_bpu_selected_bin": None,
                "plant_bpu_used": False,
            },
            "authority": dict(ZERO_AUTHORITY),
        },
    )
    try:
        source = source_factory(request)
        negotiated = dict(source.negotiated_settings())
        _validate_negotiated_camera_binding(negotiated, request)
        ledger.append("camera_opened", negotiated)
        for _ in range(request.warmup_frames):
            source.read_rgb()
        target = monotonic_fn()
        for frame_index in range(1, request.frames + 1):
            delay = target - monotonic_fn()
            if delay > 0:
                sleep_fn(delay)
            rgb = source.read_rgb()
            captured += 1
            try:
                result = dict(frame_processor(rgb, frame_index))
                _validate_frame_result_contract(result)
            except Exception as exc:
                result = _frame_error(frame_index, exc)
                processing_errors += 1
            frame_was_accepted = bool(result["final_consensus"]["passed"])
            if frame_was_accepted:
                accepted += 1
            else:
                rejected += 1
            if annotated is not None:
                annotated_path = annotated / f"frame_{frame_index:03d}.jpg"
                try:
                    if annotated_directory is None:
                        raise UvcFrontendError("annotated output guard is missing")
                    annotated_sha256, annotated_bytes = _annotate_frame(
                        rgb,
                        result,
                        annotated_path,
                        directory=annotated_directory,
                    )
                    result["annotated_frame"] = {
                        "path": str(annotated_path),
                        "sha256": annotated_sha256,
                        "bytes": annotated_bytes,
                        "training_eligible": False,
                        "accuracy_evidence": False,
                    }
                except Exception as exc:
                    result["final_consensus"]["passed"] = False
                    result["final_consensus"]["status"] = FINAL_REJECT
                    result["final_consensus"]["display_class"] = None
                    result["final_consensus"]["reject_reasons"].append(
                        f"ANNOTATION_WRITE_ERROR:{type(exc).__name__}:{exc}"
                    )
                    processing_errors += 1
                    if frame_was_accepted:
                        accepted -= 1
                        rejected += 1
            ledger.append("frame_result", result)
            print(
                json.dumps(
                    {
                        "event": "frame_result",
                        "frame_index": frame_index,
                        "semantic": (
                            result.get("semantic_hypothesis") or {}
                        ).get("raw_top1_class"),
                        "omega_ood": (
                            result.get("omega_ood_abstention") or {}
                        ).get("decision"),
                        "geometry_pass_count": (
                            result.get("registered_card_geometry") or {}
                        ).get("contract_valid_pass_count"),
                        "final_status": result["final_consensus"]["status"],
                        "display_class": result["final_consensus"]["display_class"],
                        "reject_reasons": result["final_consensus"][
                            "reject_reasons"
                        ],
                        "plant_bpu_selected_bin": None,
                        "authority": ZERO_AUTHORITY,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            target += request.interval_seconds
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        ledger.append(
            "session_error",
            {
                "error": fatal_error,
                "fail_closed": True,
                "authority": dict(ZERO_AUTHORITY),
            },
        )
    finally:
        if source is not None:
            try:
                close_receipt = dict(source.close())
                camera_released = (
                    close_receipt.get("release_completed") is True
                    and close_receipt.get("identity_match_after_close") is True
                    and close_receipt.get("device_discovery_used") is False
                )
                ledger.append("camera_closed", close_receipt)
                if not camera_released:
                    close_error = "camera release/identity-after-close contract failed"
                    fatal_error = (
                        f"{fatal_error}; {close_error}"
                        if fatal_error
                        else close_error
                    )
            except Exception as exc:
                fatal_error = (
                    f"{fatal_error}; camera release failed: {type(exc).__name__}: {exc}"
                    if fatal_error
                    else f"camera release failed: {type(exc).__name__}: {exc}"
                )
                ledger.append(
                    "camera_release_error",
                    {
                        "error": fatal_error,
                        "fail_closed": True,
                        "authority": dict(ZERO_AUTHORITY),
                    },
                )

    complete = captured == request.frames
    if fatal_error is not None or not complete or not camera_released:
        status, exit_code = "ERROR_FAIL_CLOSED_CAMERA_RELEASED_OR_ATTEMPTED", 3
    elif processing_errors:
        status, exit_code = "COMPLETE_WITH_FRAME_ERRORS_SAFE_REJECT", 3
    elif rejected:
        status, exit_code = "COMPLETE_WITH_SAFE_REJECTIONS", 2
    else:
        status, exit_code = "COMPLETE_ALL_FRAMES_DISPLAY_CONSENSUS", 0
    summary = {
        "completed_utc": _utc_now(),
        "status": status,
        "exit_code": exit_code,
        "requested_frames": request.frames,
        "captured_frames": captured,
        "accepted_frames": accepted,
        "rejected_frames": rejected,
        "processing_errors": processing_errors,
        "complete": complete,
        "camera_opened": source is not None,
        "camera_released": camera_released,
        "camera_close_receipt": (
            dict(close_receipt) if close_receipt is not None else None
        ),
        "device_discovery_used": False,
        "negotiated_capture": dict(negotiated) if negotiated is not None else None,
        "fatal_error": fatal_error,
        "persistent_service_started": False,
        "serial_gpio_pump_touched": False,
        "plant_bpu_selected_bin": None,
        "authority": dict(ZERO_AUTHORITY),
    }
    ledger.append("session_end", summary)
    record_count = ledger.record_count
    chain_root = ledger.root_sha256
    jsonl_sha256 = ledger.file_sha256
    jsonl_bytes = ledger.byte_count
    ledger.close()
    if annotated_directory is not None:
        annotated_directory.close()
    final_manifest_path = output.with_suffix(".final.json")
    try:
        _write_final_manifest_exclusive(
            final_manifest_path,
            {
                "schema": "rootscope.uvc-card-frontend-final-manifest.v1",
                "status": status,
                "exit_code": exit_code,
                "jsonl": {
                    "path": str(output),
                    "sha256": jsonl_sha256,
                    "bytes": jsonl_bytes,
                    "record_count": record_count,
                    "chain_root_sha256": chain_root,
                },
                "summary": summary,
                "final_manifest_atomic_write": True,
                "final_manifest_publish_policy": "HARDLINK_NOREPLACE",
                "persistent_service_started": False,
                "plant_bpu_selected_bin": None,
                "authority": dict(ZERO_AUTHORITY),
            },
            directory=output_directory,
        )
    finally:
        output_directory.close()
    return SessionOutcome(
        status=status,
        exit_code=exit_code,
        jsonl_path=output,
        final_manifest_path=final_manifest_path,
        captured_frames=captured,
        accepted_frames=accepted,
        rejected_frames=rejected,
        processing_errors=processing_errors,
        camera_released=camera_released,
    )


def _write_preflight_failure(path: Path, exc: Exception) -> None:
    output_directory = SafeOutputDirectory(path.parent)
    ledger = JsonlLedger(path, directory=output_directory)
    try:
        ledger.append(
            "preflight_error",
            {
                "status": "ERROR_FAIL_CLOSED_BEFORE_CAMERA_OPEN",
                "error": f"{type(exc).__name__}: {exc}",
                "camera_open_attempted": False,
                "device_discovery_used": False,
                "persistent_service_started": False,
                "plant_bpu_selected_bin": None,
                "authority": dict(ZERO_AUTHORITY),
            },
        )
        record_count = ledger.record_count
        chain_root = ledger.root_sha256
        jsonl_sha256 = ledger.file_sha256
        jsonl_bytes = ledger.byte_count
    finally:
        ledger.close()
    final_manifest = path.with_suffix(".final.json")
    try:
        _write_final_manifest_exclusive(
            final_manifest,
            {
                "schema": "rootscope.uvc-card-frontend-final-manifest.v1",
                "status": "ERROR_FAIL_CLOSED_BEFORE_CAMERA_OPEN",
                "exit_code": 3,
                "jsonl": {
                    "path": str(path),
                    "sha256": jsonl_sha256,
                    "bytes": jsonl_bytes,
                    "record_count": record_count,
                    "chain_root_sha256": chain_root,
                },
                "camera_open_attempted": False,
                "final_manifest_atomic_write": True,
                "final_manifest_publish_policy": "HARDLINK_NOREPLACE",
                "persistent_service_started": False,
                "plant_bpu_selected_bin": None,
                "authority": dict(ZERO_AUTHORITY),
            },
            directory=output_directory,
        )
    finally:
        output_directory.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        required=True,
        help="one explicit /dev/v4l/by-id/...-video-indexN path; no scan",
    )
    parser.add_argument("--expected-usb-vid", required=True)
    parser.add_argument("--expected-usb-pid", required=True)
    parser.add_argument("--expected-usb-serial", required=True)
    parser.add_argument("--print-manifest", required=True, type=Path)
    parser.add_argument("--expected-print-manifest-sha256", required=True)
    parser.add_argument("--mode", choices=("one-shot", "bounded"), default="one-shot")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--annotated-dir", type=Path)
    parser.add_argument("--capsule-config", required=True, type=Path)
    parser.add_argument("--expected-capsule-sha256", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--omega-calibration-manifest", required=True, type=Path)
    parser.add_argument("--expected-omega-calibration-sha256", required=True)
    parser.add_argument("--thresholds-json", required=True, type=Path)
    parser.add_argument("--expected-thresholds-sha256", required=True)
    parser.add_argument("--matcher-config-json", required=True, type=Path)
    parser.add_argument("--expected-matcher-config-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.jsonl.expanduser().resolve()
    session_invoked = False
    try:
        request = FrontendRequest(
            device=args.device,
            expected_camera=ExpectedCameraIdentity(
                usb_vid=args.expected_usb_vid,
                usb_pid=args.expected_usb_pid,
                usb_serial=args.expected_usb_serial,
            ),
            print_manifest=args.print_manifest,
            mode=args.mode,
            frames=args.frames,
            warmup_frames=args.warmup_frames,
            interval_seconds=args.interval_ms / 1000.0,
            width=args.width,
            height=args.height,
            fps=args.fps,
            output_root=args.output_root,
            jsonl_path=args.jsonl,
            annotated_dir=args.annotated_dir,
        )
        validate_request(request)
        capsule_asset = _bind_frozen_file(
            args.capsule_config,
            expected_sha256=args.expected_capsule_sha256,
            frozen_sha256=None,
            label="runtime capsule",
        )
        model_asset = _bind_frozen_file(
            args.model_path,
            expected_sha256=args.expected_model_sha256,
            frozen_sha256=SEED17_MODEL_SHA256,
            label="seed17 CPU ONNX",
        )
        registry_asset = _bind_frozen_file(
            args.registry,
            expected_sha256=args.expected_registry_sha256,
            frozen_sha256=FROZEN_REGISTRY_SHA256,
            label="registered template registry",
        )
        thresholds_asset = _bind_frozen_file(
            args.thresholds_json,
            expected_sha256=args.expected_thresholds_sha256,
            frozen_sha256=FROZEN_THRESHOLDS_SHA256,
            label="demo thresholds",
        )
        matcher_asset = _bind_frozen_file(
            args.matcher_config_json,
            expected_sha256=args.expected_matcher_config_sha256,
            frozen_sha256=FROZEN_MATCHER_CONFIG_SHA256,
            label="geometric matcher config",
        )
        calibration_binding = load_calibration_binding(
            args.omega_calibration_manifest,
            expected_sha256=args.expected_omega_calibration_sha256,
        )
        print_binding = load_print_manifest_binding(
            args.print_manifest,
            expected_sha256=args.expected_print_manifest_sha256,
        )
        thresholds = DemoThresholds.from_mapping(
            _decode_json_object(thresholds_asset, label="demo thresholds")
        )
        matcher_config = MatcherConfig.from_mapping(
            _decode_json_object(matcher_asset, label="geometric matcher config")
        )
        runner = build_seed17_runner_from_capsule(
            capsule_asset.path,
            model_path=model_asset.path,
        )
        if (
            _sha256_file(capsule_asset.path) != capsule_asset.sha256
            or _sha256_file(model_asset.path) != model_asset.sha256
        ):
            raise UvcFrontendError("capsule or CPU model changed while the runner loaded")
        registry = registry_asset.path
        bind_print_manifest_to_registry(print_binding, registry)
        if _sha256_file(registry) != registry_asset.sha256:
            raise UvcFrontendError("template registry changed during preflight")
        asset_binding = {
            "capsule_config": {
                "path": str(capsule_asset.path),
                "sha256": capsule_asset.sha256,
                "expected_sha256": args.expected_capsule_sha256.lower(),
                "bytes": capsule_asset.byte_count,
            },
            "seed17_onnx": {
                "path": str(model_asset.path),
                "sha256": model_asset.sha256,
                "expected_sha256": SEED17_MODEL_SHA256,
                "bytes": model_asset.byte_count,
                "provider": CPU_PROVIDER,
            },
            "omega_calibration_manifest": {
                "path": str(calibration_binding.manifest_path),
                "sha256": calibration_binding.manifest_sha256,
            },
            "registered_template_registry": {
                "path": str(registry),
                "sha256": registry_asset.sha256,
                "expected_sha256": FROZEN_REGISTRY_SHA256,
                "bytes": registry_asset.byte_count,
            },
            "four_up_print_manifest": {
                "path": str(print_binding.path),
                "sha256": print_binding.sha256,
                "bytes": print_binding.byte_count,
                "pdf_sha256": print_binding.pdf_sha256,
                "card_count": len(print_binding.cards),
                "card_roles": {
                    class_id: card["role"]
                    for class_id, card in sorted(print_binding.cards.items())
                },
            },
            "thresholds_json": {
                "path": str(thresholds_asset.path),
                "sha256": thresholds_asset.sha256,
                "expected_sha256": FROZEN_THRESHOLDS_SHA256,
                "bytes": thresholds_asset.byte_count,
            },
            "matcher_config_json": {
                "path": str(matcher_asset.path),
                "sha256": matcher_asset.sha256,
                "expected_sha256": FROZEN_MATCHER_CONFIG_SHA256,
                "bytes": matcher_asset.byte_count,
            },
            "plant_bpu": {
                "selected_bin": None,
                "used": False,
                "qualification": False,
            },
        }

        def processor(rgb: np.ndarray, index: int) -> Mapping[str, Any]:
            return evaluate_rgb_frame(
                rgb=rgb,
                frame_index=index,
                runner=runner,
                calibration_binding=calibration_binding,
                registry_path=registry,
                thresholds=thresholds,
                matcher_config=matcher_config,
                asset_binding=asset_binding,
            )

        session_invoked = True
        outcome = run_bounded_frontend(
            request,
            print_binding=print_binding,
            frame_processor=processor,
        )
    except Exception as exc:
        root_input = args.output_root.expanduser()
        safe_preflight_output = False
        if (
            root_input.is_absolute()
            and root_input.exists()
            and root_input.is_dir()
            and not root_input.is_symlink()
        ):
            root = root_input.resolve(strict=True)
            safe_preflight_output = (
                args.jsonl.expanduser().is_absolute()
                and output.parent == root
                and not output.exists()
                and not output.is_symlink()
                and not output.with_suffix(".final.json").exists()
                and not output.with_suffix(".final.json").is_symlink()
            )
        evidence_error: str | None = None
        if safe_preflight_output and not session_invoked:
            try:
                _write_preflight_failure(output, exc)
            except Exception as receipt_exc:
                evidence_error = (
                    f"{type(receipt_exc).__name__}: {receipt_exc}"
                )
        print(
            json.dumps(
                {
                    "status": "ERROR_FAIL_CLOSED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "camera_open_attempted": (
                        None if session_invoked else False
                    ),
                    "device_discovery_used": False,
                    "preflight_failure_receipt_error": evidence_error,
                    "plant_bpu_selected_bin": None,
                    "authority": ZERO_AUTHORITY,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 3
    print(
        json.dumps(
            {
                "status": outcome.status,
                "exit_code": outcome.exit_code,
                "jsonl": str(outcome.jsonl_path),
                "final_manifest": str(outcome.final_manifest_path),
                "captured_frames": outcome.captured_frames,
                "accepted_frames": outcome.accepted_frames,
                "rejected_frames": outcome.rejected_frames,
                "processing_errors": outcome.processing_errors,
                "camera_released": outcome.camera_released,
                "plant_bpu_selected_bin": None,
                "authority": ZERO_AUTHORITY,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
