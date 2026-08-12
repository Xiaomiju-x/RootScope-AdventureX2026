"""Strict AF_UNIX protocol for the RootScope r7 BPU shadow worker.

The wire format is a four-byte unsigned big-endian JSON length followed by
exactly that many UTF-8 bytes.  Tensor bytes are base64 encoded inside JSON so
the protocol remains inspectable while retaining an exact SHA-256 binding.
Only uint8 RGB NCHW tensors with the compiled model shape are accepted.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import socket
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PROTOCOL_VERSION = "rootscope.bpu-shadow-unix.v1"
REQUEST_SCHEMA = "rootscope.bpu-shadow-request.v1"
RESPONSE_SCHEMA = "rootscope.bpu-shadow-response.v1"
CLIENT_RESULT_SCHEMA = "rootscope.bpu-shadow-client-result.v1"
R7_RELEASE_ID = "rootscope-seed17-r7-default-int16-all-nodes"

TENSOR_SHAPE = (1, 3, 224, 224)
OUTPUT_SHAPE = (1, 4)
TENSOR_DTYPE = "uint8"
TENSOR_LAYOUT = "NCHW"
TENSOR_COLOR_ORDER = "RGB"
TENSOR_NBYTES = int(np.prod(TENSOR_SHAPE))
MIN_BATCH = 1
MAX_BATCH = 4
MAX_FRAME_BYTES = 900_000
HEADER_BYTES = 4

R7_REFERENCE_SHA256 = (
    "4dc4bc24741332bb6bc6da184dd1d6f954ae539ba740330490339d76eb200285"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

ZERO_AUTHORITY: dict[str, bool] = {
    "external_network": False,
    "serial_write": False,
    "gpio_write": False,
    "state_machine_write": False,
    "pump_command": False,
    "irrigation_execution": False,
    "execution_authority": False,
    "physical_authority": False,
    "physical_completion": False,
}


class BpuShadowProtocolError(RuntimeError):
    """A framed message or strict schema contract was violated."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BpuShadowProtocolError(
            "INVALID_SHA256", f"{field} must be one lowercase SHA-256 digest"
        )
    return value


def require_request_id(value: Any) -> str:
    if not isinstance(value, str) or _REQUEST_ID_RE.fullmatch(value) is None:
        raise BpuShadowProtocolError(
            "INVALID_REQUEST_ID",
            "request_id must contain 1-64 safe ASCII identifier characters",
        )
    return value


def _strict_keys(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    actual = set(payload)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing or unknown:
        raise BpuShadowProtocolError(
            "INVALID_SCHEMA_FIELDS",
            f"{context} fields differ: missing={missing} unknown={unknown}",
        )


def validate_tensor(tensor: Any) -> np.ndarray:
    array = np.asarray(tensor)
    if tuple(array.shape) != TENSOR_SHAPE:
        raise BpuShadowProtocolError(
            "INVALID_TENSOR_SHAPE",
            f"tensor shape must be {TENSOR_SHAPE}, got {tuple(array.shape)}",
        )
    if array.dtype != np.uint8:
        raise BpuShadowProtocolError(
            "INVALID_TENSOR_DTYPE",
            f"tensor dtype must be uint8, got {array.dtype}",
        )
    if not array.flags.c_contiguous:
        raise BpuShadowProtocolError(
            "NON_CONTIGUOUS_TENSOR", "tensor must be C-contiguous"
        )
    return array


def tensor_sha256(tensor: np.ndarray) -> str:
    array = validate_tensor(tensor)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def encode_tensor(tensor: Any) -> Mapping[str, Any]:
    array = validate_tensor(tensor)
    raw = array.tobytes(order="C")
    return {
        "shape": list(TENSOR_SHAPE),
        "dtype": TENSOR_DTYPE,
        "layout": TENSOR_LAYOUT,
        "color_order": TENSOR_COLOR_ORDER,
        "encoding": "base64",
        "nbytes": TENSOR_NBYTES,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


def decode_tensor(payload: Any) -> tuple[np.ndarray, str]:
    if not isinstance(payload, Mapping):
        raise BpuShadowProtocolError(
            "INVALID_TENSOR_OBJECT", "each tensor must be a JSON object"
        )
    _strict_keys(
        payload,
        required={
            "shape",
            "dtype",
            "layout",
            "color_order",
            "encoding",
            "nbytes",
            "sha256",
            "data_b64",
        },
        context="tensor",
    )
    if payload["shape"] != list(TENSOR_SHAPE):
        raise BpuShadowProtocolError(
            "INVALID_TENSOR_SHAPE",
            f"tensor shape must be {list(TENSOR_SHAPE)}",
        )
    expected_fields = {
        "dtype": TENSOR_DTYPE,
        "layout": TENSOR_LAYOUT,
        "color_order": TENSOR_COLOR_ORDER,
        "encoding": "base64",
        "nbytes": TENSOR_NBYTES,
    }
    for field, expected in expected_fields.items():
        if payload[field] != expected:
            raise BpuShadowProtocolError(
                f"INVALID_TENSOR_{field.upper()}",
                f"tensor {field} must be {expected!r}",
            )
    expected_sha = require_sha256(payload["sha256"], "tensor sha256")
    encoded = payload["data_b64"]
    if not isinstance(encoded, str):
        raise BpuShadowProtocolError(
            "INVALID_TENSOR_BASE64", "tensor data_b64 must be a string"
        )
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise BpuShadowProtocolError(
            "INVALID_TENSOR_BASE64", "tensor data_b64 is not canonical base64"
        ) from exc
    if len(raw) != TENSOR_NBYTES:
        raise BpuShadowProtocolError(
            "INVALID_TENSOR_LENGTH",
            f"decoded tensor must contain {TENSOR_NBYTES} bytes, got {len(raw)}",
        )
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise BpuShadowProtocolError(
            "TENSOR_SHA256_MISMATCH",
            f"tensor sha256 mismatch: actual={actual_sha} expected={expected_sha}",
        )
    array = np.frombuffer(raw, dtype=np.uint8).reshape(TENSOR_SHAPE).copy()
    return validate_tensor(array), actual_sha


def make_request(request_id: str, tensors: Sequence[Any]) -> Mapping[str, Any]:
    identity = require_request_id(request_id)
    if not isinstance(tensors, Sequence) or isinstance(
        tensors, (str, bytes, bytearray)
    ):
        raise BpuShadowProtocolError(
            "INVALID_BATCH", "tensors must be a sequence"
        )
    if not MIN_BATCH <= len(tensors) <= MAX_BATCH:
        raise BpuShadowProtocolError(
            "INVALID_BATCH_SIZE",
            f"request must contain {MIN_BATCH}-{MAX_BATCH} tensors",
        )
    return {
        "schema": REQUEST_SCHEMA,
        "protocol": PROTOCOL_VERSION,
        "request_id": identity,
        "operation": "infer_shadow",
        "shadow_only": True,
        "zero_authority_required": True,
        "tensors": [encode_tensor(tensor) for tensor in tensors],
    }


def decode_request(
    payload: Any,
) -> tuple[str, list[np.ndarray], list[str]]:
    if not isinstance(payload, Mapping):
        raise BpuShadowProtocolError(
            "INVALID_REQUEST_OBJECT", "request must be a JSON object"
        )
    _strict_keys(
        payload,
        required={
            "schema",
            "protocol",
            "request_id",
            "operation",
            "shadow_only",
            "zero_authority_required",
            "tensors",
        },
        context="request",
    )
    if payload["schema"] != REQUEST_SCHEMA or payload["protocol"] != PROTOCOL_VERSION:
        raise BpuShadowProtocolError(
            "PROTOCOL_VERSION_MISMATCH", "request schema/protocol is not supported"
        )
    request_id = require_request_id(payload["request_id"])
    if (
        payload["operation"] != "infer_shadow"
        or payload["shadow_only"] is not True
        or payload["zero_authority_required"] is not True
    ):
        raise BpuShadowProtocolError(
            "NON_SHADOW_OPERATION",
            "only infer_shadow with mandatory zero authority is accepted",
        )
    tensors_payload = payload["tensors"]
    if not isinstance(tensors_payload, list):
        raise BpuShadowProtocolError(
            "INVALID_BATCH", "request tensors must be a JSON array"
        )
    if not MIN_BATCH <= len(tensors_payload) <= MAX_BATCH:
        raise BpuShadowProtocolError(
            "INVALID_BATCH_SIZE",
            f"request must contain {MIN_BATCH}-{MAX_BATCH} tensors",
        )
    tensors: list[np.ndarray] = []
    hashes: list[str] = []
    for tensor_payload in tensors_payload:
        tensor, digest = decode_tensor(tensor_payload)
        tensors.append(tensor)
        hashes.append(digest)
    return request_id, tensors, hashes


def encode_json(payload: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BpuShadowProtocolError(
            "JSON_ENCODE_ERROR", f"message is not strict JSON: {exc}"
        ) from exc
    if not encoded or len(encoded) > MAX_FRAME_BYTES:
        raise BpuShadowProtocolError(
            "FRAME_TOO_LARGE",
            f"JSON frame must contain 1-{MAX_FRAME_BYTES} bytes",
        )
    return encoded


def decode_json(payload: bytes) -> Mapping[str, Any]:
    try:
        decoded = payload.decode("utf-8")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BpuShadowProtocolError(
            "INVALID_JSON", "frame body must be one UTF-8 JSON document"
        ) from exc
    if not isinstance(value, Mapping):
        raise BpuShadowProtocolError(
            "INVALID_JSON_OBJECT", "frame body must decode to one JSON object"
        )
    return value


def pack_frame(payload: Mapping[str, Any]) -> bytes:
    encoded = encode_json(payload)
    return struct.pack(">I", len(encoded)) + encoded


def send_frame(sock: socket.socket, payload: Mapping[str, Any]) -> None:
    sock.sendall(pack_frame(payload))


def _recv_exact(sock: socket.socket, size: int, *, context: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            received = size - remaining
            raise BpuShadowProtocolError(
                "TRUNCATED_FRAME",
                f"{context} truncated after {received}/{size} bytes",
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> Mapping[str, Any]:
    header = _recv_exact(sock, HEADER_BYTES, context="frame header")
    (size,) = struct.unpack(">I", header)
    if size < 1 or size > MAX_FRAME_BYTES:
        raise BpuShadowProtocolError(
            "INVALID_FRAME_LENGTH",
            f"frame length must be 1-{MAX_FRAME_BYTES}, got {size}",
        )
    return decode_json(_recv_exact(sock, size, context="frame body"))


def validate_logits(value: Any) -> list[float]:
    array = np.asarray(value)
    if tuple(array.shape) == OUTPUT_SHAPE:
        array = array[0]
    if tuple(array.shape) != (OUTPUT_SHAPE[1],):
        raise BpuShadowProtocolError(
            "INVALID_LOGITS_SHAPE",
            f"logits must have shape {OUTPUT_SHAPE} or {(OUTPUT_SHAPE[1],)}",
        )
    if not np.issubdtype(array.dtype, np.number):
        raise BpuShadowProtocolError(
            "INVALID_LOGITS_DTYPE", "logits must be numeric"
        )
    result = [float(item) for item in array]
    if not all(math.isfinite(item) for item in result):
        raise BpuShadowProtocolError(
            "NONFINITE_LOGITS", "all logits must be finite"
        )
    return result


def zero_authority_is_valid(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if set(payload) != set(ZERO_AUTHORITY):
        return False
    return all(payload[key] is False for key in ZERO_AUTHORITY)


__all__ = [
    "BpuShadowProtocolError",
    "CLIENT_RESULT_SCHEMA",
    "MAX_BATCH",
    "MAX_FRAME_BYTES",
    "OUTPUT_SHAPE",
    "PROTOCOL_VERSION",
    "R7_REFERENCE_SHA256",
    "R7_RELEASE_ID",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "TENSOR_NBYTES",
    "TENSOR_SHAPE",
    "ZERO_AUTHORITY",
    "decode_json",
    "decode_request",
    "decode_tensor",
    "encode_json",
    "encode_tensor",
    "make_request",
    "pack_frame",
    "recv_frame",
    "require_request_id",
    "require_sha256",
    "send_frame",
    "tensor_sha256",
    "validate_logits",
    "validate_tensor",
    "zero_authority_is_valid",
]
