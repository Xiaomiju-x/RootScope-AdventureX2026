"""Explicit-image CPU ONNX + Omega abstention replay for the new RDK X5.

The replay accepts one frozen manifest, one experimental four-class ONNX file,
and four explicitly named/hash-bound images. It never discovers a directory or
device, never opens a camera, and cannot emit a physical command. Embedded PC
observations are a portability check only, not accuracy qualification.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps

from .ood import Calibration, decide, evaluate_quality


SCHEMA_VERSION = "rootscope.omega-vision-board-replay-manifest.v1"
RECEIPT_SCHEMA_VERSION = "rootscope.omega-vision-board-replay-receipt.v1"
RUN_ID = "new-x5-cpu-omega-vision-explicit-four-20260723-r1"
MODEL_PATH = (
    "/opt/rootscope/.local/share/rootscope-field-v2/core_v1/releases/"
    "rootscope_x5_offline_core_v1/rootscope/deploy/x5/models/"
    "rootscope_seed17_cpu_experimental_opset11.onnx"
)
MODEL_SHA256 = "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad"
MODEL_BYTES = 44_704_833
CLASS_ORDER = ("grass_clump", "low_shrub", "young_tree", "unknown")
CALIBRATION_SHA256 = "c5e4c442586a2ca1726ef8f99d626f54f79e15227cd0754c8240f2de410eb663"
CALIBRATION_PROVENANCE_SHA256 = (
    "076f8b413a3dca56b84ee82aa313eec36bd21d66b12144aa294c463f682abfff"
)
PC_REFERENCE_SHA256 = (
    "d71588bf4dd9f45335e2fd65ec5f344ff217a4d153dfc37161e481129893d61d"
)
IMAGE_ROWS = (
    (
        "demo-reference-grass-clump",
        "/opt/rootscope/rootscope_omega_v3_inputs/bpu_aux/"
        "grass_clump_163498042_b1f6262895c3.jpg",
        "b1f6262895c31e8e507be31cebba09140e2a2582aa4f266ab05261fe50751d23",
        "REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT",
    ),
    (
        "demo-reference-low-shrub",
        "/opt/rootscope/rootscope_omega_v3_inputs/bpu_aux/"
        "low_shrub_68787114_810c7649ac72.jpg",
        "810c7649ac729105367b3213bfafc467a036f4054244c424613da6c027c73610",
        "REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT",
    ),
    (
        "demo-reference-young-tree",
        "/opt/rootscope/rootscope_omega_v3_inputs/bpu_aux/"
        "young_tree_92774234_0d994e838a2d.jpg",
        "0d994e838a2d7787ab3edfd8646e317390c790d92588c7ef9109778b843b40eb",
        "REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT",
    ),
    (
        "unregistered-negative-unknown",
        "/opt/rootscope/rootscope_omega_v3_inputs/bpu_aux/"
        "unknown_157364276_04e7f49a1e66.jpg",
        "04e7f49a1e66186bda7a9a1102985560eac0e3a1bffcec892e6dc522868c985b",
        "UNREGISTERED_NEGATIVE_NOT_ACCURACY_HOLDOUT",
    ),
)
EXPECTED_BOARD_IDENTITY = {
    "hostname": "rootscope-x5",
    "machine_id": "<redacted-device-boot-id>",
    "device_tree_serial": "3281556110258c1902ab5d9b0012004",
    "architecture": "aarch64",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AUTHORITY = {
    "camera_open": False,
    "directory_discovery": False,
    "external_network_access": False,
    "serial_open": False,
    "gpio_access": False,
    "pump_command": False,
    "state_machine_write": False,
    "execution_authority": False,
    "physical_authority": False,
    "physical_closure": False,
}


class BoardReplayError(RuntimeError):
    """A frozen artifact, runtime contract, or observation failed closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise BoardReplayError(
            f"{field} keys differ: actual={actual}, expected={sorted(expected)}"
        )
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise BoardReplayError(f"non-finite JSON constant: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except BoardReplayError:
        raise
    except Exception as exc:
        raise BoardReplayError(f"invalid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise BoardReplayError("manifest root must be an object")
    return payload


def _explicit_file(
    value: Any,
    *,
    expected_path: str,
    expected_sha256: str,
    expected_bytes: int | None,
    field: str,
) -> Path:
    if not isinstance(value, str) or value != expected_path:
        raise BoardReplayError(f"{field} path differs from the frozen manifest")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise BoardReplayError(f"{field} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BoardReplayError(f"{field} does not exist") from exc
    if not resolved.is_file() or resolved.parts[:2] == ("/", "dev"):
        raise BoardReplayError(f"{field} must be a regular non-device file")
    if expected_bytes is not None and resolved.stat().st_size != expected_bytes:
        raise BoardReplayError(f"{field} byte count mismatch")
    if sha256_file(resolved) != expected_sha256:
        raise BoardReplayError(f"{field} SHA-256 mismatch")
    return resolved


def validate_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the frozen contract without touching configured artifacts."""

    top = _require_exact_keys(
        payload,
        {
            "schema_version",
            "run_id",
            "board_identity",
            "model",
            "class_order",
            "preprocess",
            "calibration",
            "calibration_provenance",
            "pc_reference",
            "images",
            "truth_boundary",
            "authority",
        },
        field="manifest",
    )
    if top["schema_version"] != SCHEMA_VERSION:
        raise BoardReplayError("unsupported manifest schema")
    if top["run_id"] != RUN_ID:
        raise BoardReplayError("run_id changed")
    if top["board_identity"] != EXPECTED_BOARD_IDENTITY:
        raise BoardReplayError("board identity contract changed")
    model = _require_exact_keys(
        top["model"],
        {"path", "sha256", "bytes", "provider", "input", "output"},
        field="model",
    )
    if (
        model["path"] != MODEL_PATH
        or model["sha256"] != MODEL_SHA256
        or model["bytes"] != MODEL_BYTES
        or model["provider"] != "CPUExecutionProvider"
        or model["input"]
        != {"name": "image", "shape": [1, 3, 224, 224], "dtype": "tensor(float)"}
        or model["output"]
        != {"name": "logits", "shape": [1, 4], "dtype": "tensor(float)"}
    ):
        raise BoardReplayError("model contract changed")
    if tuple(top["class_order"]) != CLASS_ORDER:
        raise BoardReplayError("class order changed")
    if top["preprocess"] != {
        "mode": "torchvision_resize_short_side_center_crop_rgb_imagenet_v1",
        "resize_short_side": 256,
        "crop": [224, 224],
        "interpolation": "PIL_BILINEAR",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }:
        raise BoardReplayError("preprocess contract changed")

    calibration = _require_exact_keys(
        top["calibration"],
        {field.name for field in fields(Calibration)},
        field="calibration",
    )
    if _canonical_sha256(calibration) != CALIBRATION_SHA256:
        raise BoardReplayError("calibration values or labels changed")
    normalized_calibration = dict(calibration)
    normalized_calibration["class_order"] = tuple(calibration["class_order"])
    normalized_calibration["conformal_nonconformity"] = tuple(
        calibration["conformal_nonconformity"]
    )
    normalized_calibration["calibration_roles"] = tuple(
        calibration["calibration_roles"]
    )
    try:
        calibration_object = Calibration(**normalized_calibration)
    except Exception as exc:
        raise BoardReplayError(f"calibration contract invalid: {exc}") from exc
    if calibration_object.class_order != CLASS_ORDER:
        raise BoardReplayError("calibration class order changed")

    provenance = _require_exact_keys(
        top["calibration_provenance"],
        {
            "vision_receipt_sha256",
            "truth_boundary_addendum_sha256",
            "ood_source_sha256",
            "holdout_reevaluated_for_board_replay",
            "formal_distribution_free_coverage_guarantee",
        },
        field="calibration_provenance",
    )
    if _canonical_sha256(provenance) != CALIBRATION_PROVENANCE_SHA256:
        raise BoardReplayError("calibration provenance changed")
    for name in (
        "vision_receipt_sha256",
        "truth_boundary_addendum_sha256",
        "ood_source_sha256",
    ):
        if not isinstance(provenance[name], str) or not _SHA256.fullmatch(
            provenance[name]
        ):
            raise BoardReplayError(f"{name} must be SHA-256")
    if (
        provenance["holdout_reevaluated_for_board_replay"] is not False
        or provenance["formal_distribution_free_coverage_guarantee"] is not False
    ):
        raise BoardReplayError("board replay cannot upgrade calibration claims")

    references = top["pc_reference"]
    images = top["images"]
    if (
        not isinstance(references, list)
        or len(references) != len(IMAGE_ROWS)
        or not isinstance(images, list)
        or len(images) != len(IMAGE_ROWS)
    ):
        raise BoardReplayError("images and pc_reference must contain four rows")
    if _canonical_sha256(references) != PC_REFERENCE_SHA256:
        raise BoardReplayError("PC reference observations changed")
    for index, frozen in enumerate(IMAGE_ROWS):
        image_id, path, sha256, role = frozen
        row = _require_exact_keys(
            images[index],
            {"image_id", "path", "sha256", "provenance_role"},
            field=f"images[{index}]",
        )
        if row != {
            "image_id": image_id,
            "path": path,
            "sha256": sha256,
            "provenance_role": role,
        }:
            raise BoardReplayError(f"images[{index}] changed")
        reference = _require_exact_keys(
            references[index],
            {
                "image_id",
                "logits",
                "decision",
                "raw_top1_class",
                "absolute_tolerance",
            },
            field=f"pc_reference[{index}]",
        )
        logits = np.asarray(reference["logits"], dtype=np.float64)
        if (
            reference["image_id"] != image_id
            or logits.shape != (len(CLASS_ORDER),)
            or not np.isfinite(logits).all()
            or reference["decision"] not in {"CLASSIFY", "ABSTAIN"}
            or reference["raw_top1_class"] not in CLASS_ORDER
            or isinstance(reference["absolute_tolerance"], bool)
            or not isinstance(reference["absolute_tolerance"], (int, float))
            or not math.isfinite(float(reference["absolute_tolerance"]))
            or not 0.0 < float(reference["absolute_tolerance"]) <= 1e-4
        ):
            raise BoardReplayError(f"pc_reference[{index}] invalid")
    truth = _require_exact_keys(
        top["truth_boundary"],
        {
            "experimental_model",
            "model_qualified",
            "plant_domain_accuracy_qualified",
            "camera_qualified",
            "bpu_used",
            "physical_completion",
            "registered_demo_references_are_holdout",
        },
        field="truth_boundary",
    )
    if truth != {
        "experimental_model": True,
        "model_qualified": False,
        "plant_domain_accuracy_qualified": False,
        "camera_qualified": False,
        "bpu_used": False,
        "physical_completion": False,
        "registered_demo_references_are_holdout": False,
    }:
        raise BoardReplayError("truth boundary changed")
    if top["authority"] != _AUTHORITY:
        raise BoardReplayError("authority must remain the frozen all-false contract")
    return {"payload": top, "calibration": calibration_object}


def preprocess(image: Image.Image) -> np.ndarray:
    """Match the frozen seed17 evaluation transform without torchvision."""

    rgb = ImageOps.exif_transpose(image).convert("RGB")
    width, height = rgb.size
    if width <= 0 or height <= 0 or width * height > 40_000_000:
        raise BoardReplayError("decoded image dimensions are outside the safety limit")
    short = min(width, height)
    long = max(width, height)
    resized_long = int(256 * long / short)
    resized_size = (256, resized_long) if width <= height else (resized_long, 256)
    resized = rgb.resize(resized_size, resample=Image.Resampling.BILINEAR)
    left = int(round((resized.size[0] - 224) / 2.0))
    top = int(round((resized.size[1] - 224) / 2.0))
    cropped = resized.crop((left, top, left + 224, top + 224))
    array = np.asarray(cropped, dtype=np.float32) / np.float32(255.0)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    tensor = np.transpose((array - mean) / std, (2, 0, 1))[None].astype(
        np.float32
    )
    if tensor.shape != (1, 3, 224, 224) or not np.isfinite(tensor).all():
        raise BoardReplayError("preprocessor output contract failed")
    return np.ascontiguousarray(tensor)


def current_board_identity() -> Mapping[str, str]:
    try:
        serial = (
            Path("/proc/device-tree/serial-number")
            .read_bytes()
            .rstrip(b"\0")
            .decode("ascii")
        )
        return {
            "hostname": Path("/etc/hostname").read_text(encoding="utf-8").strip(),
            "machine_id": Path("/etc/machine-id")
            .read_text(encoding="utf-8")
            .strip(),
            "device_tree_serial": serial,
            "architecture": platform.machine(),
        }
    except Exception as exc:
        raise BoardReplayError(f"could not read board identity: {exc}") from exc


def _float32_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f4")
    return hashlib.sha256(
        np.ascontiguousarray(canonical).tobytes(order="C")
    ).hexdigest()


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
    except FileExistsError as exc:
        raise BoardReplayError(f"refusing to overwrite receipt: {path}") from exc


def run_board_replay(manifest_path: Path) -> Mapping[str, Any]:
    if not manifest_path.is_absolute() or manifest_path.is_symlink():
        raise BoardReplayError("manifest must be one explicit absolute non-symlink path")
    manifest_resolved = manifest_path.resolve(strict=True)
    validated = validate_manifest(_load_json(manifest_resolved))
    payload = validated["payload"]
    calibration: Calibration = validated["calibration"]
    identity = current_board_identity()
    if identity != EXPECTED_BOARD_IDENTITY:
        raise BoardReplayError(f"board identity mismatch: {identity}")

    model = _explicit_file(
        payload["model"]["path"],
        expected_path=MODEL_PATH,
        expected_sha256=MODEL_SHA256,
        expected_bytes=MODEL_BYTES,
        field="model",
    )
    decoded: list[tuple[np.ndarray, Image.Image, Mapping[str, Any]]] = []
    for index, frozen in enumerate(IMAGE_ROWS):
        image_id, path, sha256, _role = frozen
        resolved = _explicit_file(
            payload["images"][index]["path"],
            expected_path=path,
            expected_sha256=sha256,
            expected_bytes=None,
            field=f"image[{image_id}]",
        )
        try:
            with Image.open(resolved) as source:
                if int(getattr(source, "n_frames", 1)) != 1:
                    raise BoardReplayError("multi-frame images are forbidden")
                oriented = ImageOps.exif_transpose(source).convert("RGB")
                rgb = np.asarray(oriented, dtype=np.uint8)
                tensor_source = oriented.copy()
        except BoardReplayError:
            raise
        except Exception as exc:
            raise BoardReplayError(f"image decode failed: {image_id}: {exc}") from exc
        decoded.append(
            (
                rgb,
                tensor_source,
                {
                    "image_id": image_id,
                    "path": str(resolved),
                    "sha256": sha256,
                    "bytes": resolved.stat().st_size,
                    "provenance_role": payload["images"][index]["provenance_role"],
                    "camera_opened": False,
                    "device_enumerated": False,
                },
            )
        )

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise BoardReplayError("onnxruntime is unavailable") from exc
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise BoardReplayError(
            f"runtime provider is not CPU-only: {session.get_providers()}"
        )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != "image"
        or list(inputs[0].shape) != [1, 3, 224, 224]
        or len(outputs) != 1
        or outputs[0].name != "logits"
        or list(outputs[0].shape) != [1, 4]
    ):
        raise BoardReplayError("ONNX input/output contract mismatch")

    rows: list[Mapping[str, Any]] = []
    parity_passed = True
    for index, (rgb, image, provenance) in enumerate(decoded):
        tensor = preprocess(image)
        started = time.perf_counter_ns()
        output = np.asarray(
            session.run(["logits"], {"image": tensor})[0],
            dtype=np.float32,
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if output.shape != (1, 4) or not np.isfinite(output).all():
            raise BoardReplayError("ONNX output must be finite [1,4] logits")
        logits = output[0]
        decision = decide(logits, evaluate_quality(rgb), calibration)
        reference = payload["pc_reference"][index]
        reference_logits = np.asarray(reference["logits"], dtype=np.float64)
        max_abs = float(np.max(np.abs(logits.astype(np.float64) - reference_logits)))
        row_parity = (
            max_abs <= float(reference["absolute_tolerance"])
            and decision.decision == reference["decision"]
            and decision.raw_top1_class == reference["raw_top1_class"]
        )
        parity_passed = parity_passed and row_parity
        rows.append(
            {
                "input_provenance": provenance,
                "tensor": {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": _float32_sha256(tensor),
                },
                "inference": {
                    "logits": [float(value) for value in logits],
                    "logits_float32_sha256": _float32_sha256(logits),
                    "cpu_inference_ms": elapsed_ms,
                },
                "omega_decision": decision.to_dict(),
                "pc_reference_parity": {
                    "passed": row_parity,
                    "max_absolute_logit_error": max_abs,
                    "absolute_tolerance": float(reference["absolute_tolerance"]),
                    "decision_match": decision.decision == reference["decision"],
                    "raw_top1_match": (
                        decision.raw_top1_class == reference["raw_top1_class"]
                    ),
                },
            }
        )
    if not parity_passed:
        raise BoardReplayError("board/PC numerical or decision parity failed")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "PASS_X5_CPU_EXPLICIT_IMAGE_REPLAY_ZERO_AUTHORITY",
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "manifest": {
            "path": str(manifest_resolved),
            "sha256": sha256_file(manifest_resolved),
            "run_id": payload["run_id"],
        },
        "board_identity": identity,
        "runtime": {
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "provider_requested": "CPUExecutionProvider",
            "providers_actual": session.get_providers(),
            "model_path": str(model),
            "model_sha256": MODEL_SHA256,
            "model_bytes": MODEL_BYTES,
            "board_cpu_inference_executed": True,
            "bpu_inference_executed": False,
        },
        "calibration_provenance": dict(payload["calibration_provenance"]),
        "images": rows,
        "summary": {
            "explicit_image_count": len(rows),
            "pc_reference_parity_passed_count": sum(
                bool(row["pc_reference_parity"]["passed"]) for row in rows
            ),
            "all_pc_reference_parity_passed": parity_passed,
            "classify_count": sum(
                row["omega_decision"]["decision"] == "CLASSIFY" for row in rows
            ),
            "abstain_count": sum(
                row["omega_decision"]["decision"] == "ABSTAIN" for row in rows
            ),
            "holdout_reevaluated": False,
            "accuracy_computed": False,
        },
        "truth_boundary": dict(payload["truth_boundary"]),
        "effects_and_authority": {
            **_AUTHORITY,
            "board_cpu_compute_touched": True,
            "external_device_touched": False,
        },
        "claim_boundary": (
            "This receipt proves only hash-bound four-image CPUExecutionProvider "
            "ONNX portability plus the experimental Omega abstention projection "
            "on the named RDK X5. Three positive inputs are registered demo "
            "references, not holdout samples. No accuracy, camera, BPU plant "
            "model, production, irrigation, or physical-closure qualification "
            "is granted."
        ),
    }
    fingerprint = dict(receipt)
    fingerprint.pop("generated_at_utc")
    receipt["receipt_sha256"] = _canonical_sha256(fingerprint)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = run_board_replay(args.manifest)
        _write_exclusive(args.out, receipt)
    except (BoardReplayError, FileNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR_FAIL_CLOSED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "receipt_written": False,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_path": str(args.out.resolve()),
                "receipt_sha256": receipt["receipt_sha256"],
                "explicit_image_count": receipt["summary"]["explicit_image_count"],
                "all_pc_reference_parity_passed": receipt["summary"][
                    "all_pc_reference_parity_passed"
                ],
                "claim_boundary": receipt["claim_boundary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
