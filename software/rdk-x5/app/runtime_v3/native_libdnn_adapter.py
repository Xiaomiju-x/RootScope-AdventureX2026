"""Hash-bound persistent native libdnn adapter for RootScope v3.

This adapter exists because the actual RDK X5 showed a material tensor-contract
difference between ``hbm_runtime`` and the canonical ``hrt_model_exec`` binary
input path.  The paired native worker implements the measured valid-shape
contract while keeping one model resident.

It is a zero-authority inference component: no camera, network, serial, GPIO,
pump, service manager, or action state is opened or changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import select
import stat
import struct
import subprocess
import tempfile
import time
from typing import Any, BinaryIO, Mapping

import numpy as np


INPUT_SHAPE = (1, 3, 224, 224)
INPUT_BYTES = 1 * 3 * 224 * 224
OUTPUT_SHAPE = (1, 4)
EXPECTED_MODEL_NAME = (
    "rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes"
)
PROTOCOL_SCHEMA = "rootscope.native-libdnn.protocol.v1"
REQUEST = struct.Struct("<8sIQI")
RESPONSE = struct.Struct("<8sIQIQI")
LOGITS = struct.Struct("<4f")
REQUEST_MAGIC = b"RSNV3REQ"
RESPONSE_MAGIC = b"RSNV3RSP"
PROTOCOL_VERSION = 1
ZERO_AUTHORITY = {
    "execution_authority": False,
    "serial_write": False,
    "gpio_write": False,
    "pump_command": False,
    "state_machine_write": False,
    "physical_completion": False,
}


class NativeLibdnnContractError(RuntimeError):
    """Raised when a file, process, frame, tensor, or output violates contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_locked_file(
    configured: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise NativeLibdnnContractError(f"{label} expected SHA-256 is invalid")
    expected = expected_sha256.lower()
    if any(character not in "0123456789abcdef" for character in expected):
        raise NativeLibdnnContractError(f"{label} expected SHA-256 is invalid")
    path = Path(configured).expanduser()
    if path.is_symlink():
        raise NativeLibdnnContractError(f"{label} path must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise NativeLibdnnContractError(f"{label} file is unavailable: {exc}") from exc
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise NativeLibdnnContractError(f"{label} must be one regular file")
    if mode & 0o022:
        raise NativeLibdnnContractError(
            f"{label} must not be group/world writable"
        )
    actual = sha256_file(resolved)
    if actual != expected:
        raise NativeLibdnnContractError(
            f"{label} SHA-256 mismatch: actual={actual} expected={expected}"
        )
    return resolved, actual


def _load_compile_contract(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    configured = Path(path).expanduser()
    if configured.is_symlink():
        raise NativeLibdnnContractError(
            "worker compile contract must not be a symlink"
        )
    try:
        resolved = configured.resolve(strict=True)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeLibdnnContractError(
            f"unable to read worker compile contract: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise NativeLibdnnContractError("worker compile contract must be an object")
    required = {
        "schema": "rootscope.native-libdnn.x5-compile-contract.v1",
        "status": "PASS_REPRODUCIBLE_TWO_BUILD",
        "target_arch": "aarch64",
        "protocol": PROTOCOL_SCHEMA,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise NativeLibdnnContractError(
                f"worker compile contract {key!r} mismatch"
            )
    binary = payload.get("binary")
    if not isinstance(binary, Mapping):
        raise NativeLibdnnContractError(
            "worker compile contract binary record is missing"
        )
    sha = binary.get("sha256")
    if not isinstance(sha, str):
        raise NativeLibdnnContractError(
            "worker compile contract binary SHA-256 is missing"
        )
    return resolved, payload


@dataclass(frozen=True)
class WorkerIdentity:
    path: str
    sha256: str
    compile_contract_path: str
    compile_contract_sha256: str
    protocol: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "compile_contract_path": self.compile_contract_path,
            "compile_contract_sha256": self.compile_contract_sha256,
            "protocol": self.protocol,
        }


class PersistentNativeLibdnnR7Adapter:
    """One hash-bound worker process and one model load for all inferences."""

    def __init__(
        self,
        model_path: str | Path,
        expected_model_sha256: str,
        *,
        worker_path: str | Path,
        compile_contract_path: str | Path,
        expected_model_name: str = EXPECTED_MODEL_NAME,
        inference_timeout_s: float = 10.0,
    ) -> None:
        if expected_model_name != EXPECTED_MODEL_NAME:
            raise NativeLibdnnContractError("r7 expected model name changed")
        if not (0.1 <= float(inference_timeout_s) <= 60.0):
            raise NativeLibdnnContractError(
                "inference timeout must be within [0.1, 60] seconds"
            )
        model, model_sha256 = _regular_locked_file(
            model_path,
            expected_sha256=expected_model_sha256,
            label="model",
        )
        contract_path, contract = _load_compile_contract(compile_contract_path)
        binary_record = contract["binary"]
        worker, worker_sha256 = _regular_locked_file(
            worker_path,
            expected_sha256=str(binary_record["sha256"]),
            label="native worker",
        )
        if os.name != "nt" and not os.access(worker, os.X_OK):
            raise NativeLibdnnContractError("native worker is not executable")

        environment = {
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        if os.environ.get("LD_LIBRARY_PATH"):
            environment["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
        self._stderr: BinaryIO = tempfile.TemporaryFile(mode="w+b")
        command = [
            str(worker),
            "--model",
            str(model),
            "--model-sha256",
            model_sha256,
            "--expected-model-name",
            expected_model_name,
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                shell=False,
                close_fds=True,
                env=environment,
            )
        except OSError as exc:
            self._stderr.close()
            raise NativeLibdnnContractError(
                f"native worker launch failed: {exc}"
            ) from exc
        if process.stdin is None or process.stdout is None:
            process.kill()
            process.wait()
            self._stderr.close()
            raise NativeLibdnnContractError("native worker pipes are unavailable")

        self.model_path = model
        self.model_sha256 = model_sha256
        self.expected_model_name = expected_model_name
        self.worker_identity = WorkerIdentity(
            path=str(worker),
            sha256=worker_sha256,
            compile_contract_path=str(contract_path),
            compile_contract_sha256=sha256_file(contract_path),
            protocol=PROTOCOL_SCHEMA,
        )
        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._timeout_s = float(inference_timeout_s)
        self._request_id = 0
        self.inference_count = 0
        self.load_monotonic_ns = time.monotonic_ns()
        self.closed = False
        self.exit_code: int | None = None

    @property
    def worker_pid(self) -> int:
        return int(self._process.pid)

    def _stderr_text(self) -> str:
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            return self._stderr.read().decode("utf-8", errors="replace")[-4096:]
        except (OSError, ValueError):
            return ""

    def _read_exact(self, size: int, deadline: float) -> bytes:
        result = bytearray()
        descriptor = self._stdout.fileno()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NativeLibdnnContractError("native worker response timed out")
            if os.name != "nt":
                ready, _, _ = select.select([descriptor], [], [], remaining)
                if not ready:
                    raise NativeLibdnnContractError(
                        "native worker response timed out"
                    )
                block = os.read(descriptor, size - len(result))
            else:
                block = self._stdout.read(size - len(result))
            if not block:
                code = self._process.poll()
                raise NativeLibdnnContractError(
                    "native worker closed its response pipe; "
                    f"exit={code}; stderr={self._stderr_text()!r}"
                )
            result.extend(block)
        return bytes(result)

    def infer_uint8(self, tensor: np.ndarray) -> dict[str, Any]:
        if self.closed:
            raise NativeLibdnnContractError("native worker adapter is closed")
        if self._process.poll() is not None:
            raise NativeLibdnnContractError(
                "native worker exited before inference; "
                f"exit={self._process.returncode}; stderr={self._stderr_text()!r}"
            )
        array = np.asarray(tensor)
        if tuple(array.shape) != INPUT_SHAPE or array.dtype != np.uint8:
            raise NativeLibdnnContractError(
                f"external tensor must be uint8 {INPUT_SHAPE}, "
                f"got {array.dtype} {tuple(array.shape)}"
            )
        contiguous = np.ascontiguousarray(array)
        payload = contiguous.tobytes(order="C")
        if len(payload) != INPUT_BYTES:
            raise NativeLibdnnContractError("external tensor byte count changed")
        self._request_id += 1
        request_id = self._request_id
        frame = REQUEST.pack(
            REQUEST_MAGIC,
            PROTOCOL_VERSION,
            request_id,
            len(payload),
        )
        started = time.perf_counter_ns()
        try:
            self._stdin.write(frame)
            self._stdin.write(payload)
            self._stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise NativeLibdnnContractError(
                "native worker request write failed; "
                f"stderr={self._stderr_text()!r}"
            ) from exc
        deadline = time.monotonic() + self._timeout_s
        header = self._read_exact(RESPONSE.size, deadline)
        (
            magic,
            version,
            response_id,
            status,
            worker_latency_ns,
            payload_size,
        ) = RESPONSE.unpack(header)
        if magic != RESPONSE_MAGIC or version != PROTOCOL_VERSION:
            raise NativeLibdnnContractError("native worker response header changed")
        if response_id != request_id:
            raise NativeLibdnnContractError("native worker response id mismatch")
        if status != 0 or payload_size != LOGITS.size:
            raise NativeLibdnnContractError(
                "native worker response status/shape mismatch"
            )
        output = np.asarray(
            LOGITS.unpack(self._read_exact(LOGITS.size, deadline)),
            dtype=np.float32,
        ).reshape(OUTPUT_SHAPE)
        if not np.isfinite(output).all():
            raise NativeLibdnnContractError(
                "native worker returned non-finite logits"
            )
        roundtrip_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self.inference_count += 1
        return {
            "schema": "rootscope.runtime-v3.native-libdnn-evidence.v1",
            "status": "QUALIFIED_BACKEND_RUNTIME_EVIDENCE",
            "backend_actual": (
                "rootscope.native_libdnn_valid_shape_bridge/libdnn.so"
            ),
            "persistent_model": True,
            "cold_load_per_inference": False,
            "valid_shape_contract": [1, 3, 224, 224],
            "input_bytes": INPUT_BYTES,
            "model_name": self.expected_model_name,
            "model_sha256": self.model_sha256,
            "worker": self.worker_identity.to_dict(),
            "worker_pid": self.worker_pid,
            "external_tensor_sha256": hashlib.sha256(payload).hexdigest(),
            "logits": output[0].astype(float).tolist(),
            "top1_index": int(np.argmax(output[0])),
            "worker_inference_ms": worker_latency_ns / 1_000_000.0,
            "roundtrip_latency_ms": roundtrip_ms,
            "inference_count_since_load": self.inference_count,
            "authority": dict(ZERO_AUTHORITY),
        }

    def close(self, timeout_s: float = 5.0) -> int:
        if self.closed:
            return int(self.exit_code if self.exit_code is not None else -1)
        self.closed = True
        try:
            self._stdin.close()
        except OSError:
            pass
        try:
            code = self._process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                code = self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                code = self._process.wait(timeout=2.0)
        self.exit_code = int(code)
        try:
            self._stdout.close()
        except OSError:
            pass
        self._stderr.close()
        return self.exit_code

    def __enter__(self) -> "PersistentNativeLibdnnR7Adapter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "EXPECTED_MODEL_NAME",
    "INPUT_BYTES",
    "INPUT_SHAPE",
    "NativeLibdnnContractError",
    "PersistentNativeLibdnnR7Adapter",
    "PROTOCOL_SCHEMA",
    "ZERO_AUTHORITY",
]
