"""Fail-closed core-venv client for the local r7 BPU shadow worker.

Every failure invokes a caller-supplied CPU fallback.  This module can create
only ``AF_UNIX`` sockets and has no network, camera, serial, GPIO, pump, or
state-machine dependency.  The default remains a short live-path timeout;
callers doing the canonical cold-load ``hrt_model_exec`` audit may explicitly
allow a longer timeout without changing the CPU-primary display contract.
"""

from __future__ import annotations

import math
import os
import socket
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .bpu_shadow_protocol import (
    BpuShadowProtocolError,
    CLIENT_RESULT_SCHEMA,
    OUTPUT_SHAPE,
    PROTOCOL_VERSION,
    R7_REFERENCE_SHA256,
    RESPONSE_SCHEMA,
    ZERO_AUTHORITY,
    make_request,
    recv_frame,
    require_sha256,
    send_frame,
    tensor_sha256,
    validate_logits,
    validate_tensor,
    zero_authority_is_valid,
)

CpuFallback = Callable[[Sequence[np.ndarray]], Sequence[Any]]
SocketConnector = Callable[[float], socket.socket]


class BpuShadowClient:
    """Fail closed to CPU when the optional local BPU shadow is unavailable."""

    def __init__(
        self,
        socket_path: str | Path,
        *,
        expected_model_sha256: str = R7_REFERENCE_SHA256,
        timeout_s: float = 0.18,
        connector: SocketConnector | None = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.expected_model_sha256 = require_sha256(
            expected_model_sha256, "expected model sha256"
        )
        if not 0.01 <= float(timeout_s) <= 10.0:
            raise ValueError("timeout_s must be within [0.01, 10.0]")
        self.timeout_s = float(timeout_s)
        self._connector = connector

    def _connect_unix(self, timeout_s: float) -> socket.socket:
        path = self.socket_path.expanduser()
        if not path.is_absolute():
            raise BpuShadowProtocolError(
                "INVALID_SOCKET_PATH", "AF_UNIX socket path must be absolute"
            )
        try:
            metadata = os.lstat(path)
        except FileNotFoundError as exc:
            raise BpuShadowProtocolError(
                "SOCKET_UNAVAILABLE", "BPU shadow Unix socket does not exist"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISSOCK(metadata.st_mode):
            raise BpuShadowProtocolError(
                "INVALID_SOCKET_PATH", "configured path is not a real Unix socket"
            )
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(timeout_s)
        try:
            connection.connect(str(path))
        except Exception:
            connection.close()
            raise
        return connection

    def _connect(self) -> socket.socket:
        connector = self._connector or self._connect_unix
        connection = connector(self.timeout_s)
        if not isinstance(connection, socket.socket):
            raise BpuShadowProtocolError(
                "INVALID_CONNECTOR", "connector did not return a socket"
            )
        connection.settimeout(self.timeout_s)
        return connection

    def _validate_response(
        self,
        response: Any,
        *,
        request_id: str,
        tensor_hashes: Sequence[str],
    ) -> tuple[list[list[float]], Mapping[str, Any], Mapping[str, Any]]:
        if not isinstance(response, Mapping):
            raise BpuShadowProtocolError(
                "INVALID_RESPONSE", "worker response must be a JSON object"
            )
        if (
            response.get("schema") != RESPONSE_SCHEMA
            or response.get("protocol") != PROTOCOL_VERSION
        ):
            raise BpuShadowProtocolError(
                "PROTOCOL_VERSION_MISMATCH", "worker response protocol is unsupported"
            )
        if response.get("request_id") != request_id:
            raise BpuShadowProtocolError(
                "REQUEST_ID_MISMATCH", "worker response request_id differs"
            )
        if response.get("status") != "OK_SHADOW_ONLY":
            error = response.get("error")
            raise BpuShadowProtocolError(
                "WORKER_REJECTED_REQUEST", f"worker failed closed: {error!r}"
            )
        if (
            response.get("shadow_only") is not True
            or response.get("zero_authority") is not True
            or not zero_authority_is_valid(response.get("authority"))
        ):
            raise BpuShadowProtocolError(
                "ZERO_AUTHORITY_VIOLATION",
                "worker response does not preserve the exact zero-authority contract",
            )
        model = response.get("model")
        if not isinstance(model, Mapping):
            raise BpuShadowProtocolError(
                "INVALID_MODEL_METADATA", "worker response lacks model metadata"
            )
        actual_sha = require_sha256(model.get("sha256"), "response model sha256")
        if actual_sha != self.expected_model_sha256:
            raise BpuShadowProtocolError(
                "MODEL_SHA256_MISMATCH",
                f"worker model sha256 differs: actual={actual_sha} "
                f"expected={self.expected_model_sha256}",
            )
        if (
            model.get("qualification") != "SHADOW_CANDIDATE_NOT_DEFAULT"
            or model.get("selected_bin_changed") is not False
        ):
            raise BpuShadowProtocolError(
                "MODEL_STATUS_VIOLATION",
                "worker attempted to upgrade the r7 shadow qualification/default status",
            )
        backend = response.get("backend")
        if not isinstance(backend, Mapping) or backend.get("mode") != "BPU_SHADOW":
            raise BpuShadowProtocolError(
                "INVALID_BACKEND_METADATA", "worker backend is not BPU_SHADOW"
            )
        actual_metadata = backend.get("actual_metadata")
        if not isinstance(actual_metadata, Mapping):
            raise BpuShadowProtocolError(
                "MISSING_ACTUAL_METADATA",
                "worker backend omits actual input/output metadata evidence",
            )
        compiled_output_shape = actual_metadata.get("compiled_output_shape")
        if compiled_output_shape not in ([1, 4], [1, 4, 1, 1]):
            raise BpuShadowProtocolError(
                "INVALID_COMPILED_OUTPUT_SHAPE",
                "worker compiled output shape is outside the r7 allowlist",
            )
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(tensor_hashes):
            raise BpuShadowProtocolError(
                "RESULT_COUNT_MISMATCH", "worker returned the wrong number of results"
            )
        logits: list[list[float]] = []
        for index, (item, expected_hash) in enumerate(
            zip(results, tensor_hashes, strict=True)
        ):
            if (
                not isinstance(item, Mapping)
                or item.get("index") != index
                or item.get("input_sha256") != expected_hash
            ):
                raise BpuShadowProtocolError(
                    "RESULT_PROVENANCE_MISMATCH",
                    f"worker result {index} is not bound to its input tensor",
                )
            latency = item.get("latency_ms")
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or not math.isfinite(float(latency))
                or float(latency) < 0.0
            ):
                raise BpuShadowProtocolError(
                    "INVALID_LATENCY", f"worker result {index} latency is invalid"
                )
            logits.append(validate_logits(item.get("logits")))
        return logits, model, backend

    @staticmethod
    def _run_cpu_fallback(
        tensors: Sequence[np.ndarray],
        cpu_fallback: CpuFallback,
    ) -> list[list[float]]:
        outputs = cpu_fallback(tensors)
        if not isinstance(outputs, Sequence) or isinstance(
            outputs, (str, bytes, bytearray)
        ):
            raise BpuShadowProtocolError(
                "INVALID_CPU_FALLBACK", "CPU fallback must return a sequence"
            )
        if len(outputs) != len(tensors):
            raise BpuShadowProtocolError(
                "INVALID_CPU_FALLBACK_COUNT",
                "CPU fallback returned the wrong number of results",
            )
        return [validate_logits(output) for output in outputs]

    def infer_tensors(
        self,
        tensors: Sequence[np.ndarray],
        *,
        cpu_fallback: CpuFallback,
    ) -> Mapping[str, Any]:
        """Infer one to four preprocessed tensors, with deterministic CPU fallback.

        Inputs are already-preprocessed uint8 RGB NCHW ``[1,3,224,224]``
        tensors.  This is the stable public entry point used by static replay
        and Competition Live v2.
        """

        request_id = uuid.uuid4().hex
        try:
            canonical = [validate_tensor(tensor) for tensor in tensors]
            request = make_request(request_id, canonical)
            hashes = [tensor_sha256(tensor) for tensor in canonical]
            with self._connect() as connection:
                send_frame(connection, request)
                response = recv_frame(connection)
            logits, model, backend = self._validate_response(
                response,
                request_id=request_id,
                tensor_hashes=hashes,
            )
            return {
                "schema": CLIENT_RESULT_SCHEMA,
                "request_id": request_id,
                "status": "BPU_SHADOW_OK",
                "backend_actual": backend.get("actual"),
                "backend": dict(backend),
                "used_cpu_fallback": False,
                "shadow_only": True,
                "zero_authority": True,
                "authority": dict(ZERO_AUTHORITY),
                "model": dict(model),
                "logits": logits,
                "bpu_batch": dict(response.get("batch") or {}),
                "bpu_results": [dict(item) for item in response.get("results") or []],
                "fallback": None,
            }
        except Exception as bpu_exc:
            try:
                # Invalid input still reaches CPU fallback exactly as supplied;
                # the CPU implementation owns its own input contract.
                cpu_logits = self._run_cpu_fallback(tensors, cpu_fallback)
            except Exception as cpu_exc:
                return {
                    "schema": CLIENT_RESULT_SCHEMA,
                    "request_id": request_id,
                    "status": "CPU_FALLBACK_FAILED_NO_RESULT",
                    "backend_actual": "NONE",
                    "backend": None,
                    "used_cpu_fallback": True,
                    "shadow_only": True,
                    "zero_authority": True,
                    "authority": dict(ZERO_AUTHORITY),
                    "model": None,
                    "logits": None,
                    "bpu_batch": None,
                    "bpu_results": None,
                    "fallback": {
                        "bpu_error_type": type(bpu_exc).__name__,
                        "bpu_error": str(bpu_exc)[:500],
                        "cpu_error_type": type(cpu_exc).__name__,
                        "cpu_error": str(cpu_exc)[:500],
                    },
                }
            return {
                "schema": CLIENT_RESULT_SCHEMA,
                "request_id": request_id,
                "status": "CPU_FALLBACK_OK",
                "backend_actual": "CPU_FALLBACK",
                "backend": None,
                "used_cpu_fallback": True,
                "shadow_only": True,
                "zero_authority": True,
                "authority": dict(ZERO_AUTHORITY),
                "model": None,
                "logits": cpu_logits,
                "bpu_batch": None,
                "bpu_results": None,
                "fallback": {
                    "bpu_error_type": type(bpu_exc).__name__,
                    "bpu_error": str(bpu_exc)[:500],
                },
            }

    def infer_or_cpu(
        self,
        tensors: Sequence[np.ndarray],
        *,
        cpu_fallback: CpuFallback,
    ) -> Mapping[str, Any]:
        """Backward-compatible alias for :meth:`infer_tensors`."""

        return self.infer_tensors(tensors, cpu_fallback=cpu_fallback)


__all__ = ["BpuShadowClient", "CpuFallback", "SocketConnector"]
