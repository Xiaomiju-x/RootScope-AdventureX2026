"""Hash-bound RootScope r7 BPU shadow worker over AF_UNIX.

The canonical RDK X5 path invokes the vendor ``hrt_model_exec`` input adapter.
That adapter is the only installed runtime path shown by an exact 23/23 replay
to convert the public contiguous uint8 RGB NCHW tensor into the compiled
``RGB_128`` aligned-stride tensor without numerical drift.  It cold-loads the
model for every shadow proposal, so it is explicitly *not* a real-time path.

``pyeasy_dnn`` remains available only through an explicit legacy CLI option.
It is never selected automatically because its observed input-buffer adapter
does not reproduce the x86 quantized r7 reference.

It deliberately has no TCP listener, camera, serial, GPIO, state-machine,
pump, or actuator import.  Its logits are advisory shadow evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from app.edge import bpu_seed17 as frozen_bpu

from .bpu_shadow_protocol import (
    BpuShadowProtocolError,
    OUTPUT_SHAPE,
    PROTOCOL_VERSION,
    R7_REFERENCE_SHA256,
    R7_RELEASE_ID,
    RESPONSE_SCHEMA,
    ZERO_AUTHORITY,
    decode_request,
    recv_frame,
    require_sha256,
    send_frame,
    validate_logits,
    validate_tensor,
)

HRT_MODEL_EXEC_DEFAULT = Path("/usr/sbin/hrt_model_exec")
HRT_MODEL_NAME = (
    "rootscope_seed17_resnet18_224x224_rgb_ddr_"
    "r7_default_int16_all_nodes"
)
HRT_INPUT_VALID_SHAPE = (1, 3, 224, 224)
HRT_INPUT_ALIGNED_SHAPE = (1, 3, 224, 256)
HRT_INPUT_ALIGNED_BYTES = 172_032
HRT_OUTPUT_COMPILED_SHAPE = (1, 4, 1, 1)
HRT_OUTPUT_BYTES = 16

CommandRunner = Callable[..., Any]


class ShadowBackend(Protocol):
    """Minimal interface kept inside the worker's zero-authority boundary."""

    model_sha256: str
    backend_actual: str
    evidence_scope: str
    injected_test_backend: bool
    input_metadata: Mapping[str, Any]
    output_metadata: Mapping[str, Any]
    compiled_output_shape: tuple[int, ...]
    accepted_runtime_input_buffer_dtypes: Sequence[str]
    input_adapter: str
    cold_load_per_inference: bool
    real_time_qualified: bool
    runtime_wrapper_status: str

    def infer_tensor(self, tensor: np.ndarray) -> Sequence[float]:
        """Return four finite logits for one fixed-shape tensor."""


def _validate_hash_bound_model(
    model_path: str | Path,
    expected_sha256: str,
) -> tuple[Path, str]:
    expected = require_sha256(expected_sha256, "expected model sha256")
    configured = Path(model_path).expanduser()
    if configured.is_symlink():
        raise frozen_bpu.Seed17BpuContractError("BPU model must not be a symlink")
    resolved = configured.resolve(strict=True)
    if not resolved.is_file() or resolved.suffix.lower() != ".bin":
        raise frozen_bpu.Seed17BpuContractError(
            "BPU model must be one regular .bin file"
        )
    actual = frozen_bpu.sha256_file(resolved)
    if actual != expected:
        raise frozen_bpu.Seed17BpuContractError(
            f"BPU model SHA-256 mismatch: actual={actual} expected={expected}"
        )
    return resolved, actual


def _require_hrt_model_info_contract(text: str) -> None:
    """Fail closed unless vendor model_info exposes the exact r7 ABI."""

    compact = re.sub(r"\s+", "", text)
    required = (
        f"[modelname]:{HRT_MODEL_NAME}",
        "input[0]:name:image",
        "validshape:(1,3,224,224,)",
        "alignedshape:(1,3,224,256,)",
        "alignedbytesize:172032",
        "tensortype:HB_DNN_IMG_TYPE_RGB",
        "tensorlayout:HB_DNN_LAYOUT_NCHW",
        "stride:(172032,57344,256,1,)",
        "output[0]:name:logits",
        "validshape:(1,4,1,1,)",
        "alignedbytesize:16",
        "tensortype:HB_DNN_TENSOR_TYPE_F32",
        "stride:(16,4,4,4,)",
    )
    missing = [item for item in required if item not in compact]
    if missing:
        raise frozen_bpu.Seed17BpuContractError(
            f"hrt_model_exec model_info differs from the r7 ABI: missing={missing}"
        )


class HrtModelExecR7BpuBackend:
    """Numerically canonical, cold-load vendor CLI adapter for r7.

    The external contract is the same 150,528-byte uint8 RGB NCHW tensor used
    by the x86 replay.  ``hrt_model_exec`` owns uint8 -> RGB_128 conversion and
    row-stride padding.  Passing host-centered int8 or a caller-padded tensor is
    deliberately forbidden because those encodings have different semantics
    in the vendor tool.
    """

    def __init__(
        self,
        model_path: str | Path,
        expected_sha256: str,
        *,
        executable_path: str | Path = HRT_MODEL_EXEC_DEFAULT,
        work_root: str | Path | None = None,
        command_runner: CommandRunner | None = None,
        inference_timeout_s: float = 8.0,
    ) -> None:
        resolved_model, actual = _validate_hash_bound_model(
            model_path, expected_sha256
        )
        if not 1.0 <= float(inference_timeout_s) <= 30.0:
            raise ValueError("inference_timeout_s must be within [1.0, 30.0]")
        self.inference_timeout_s = float(inference_timeout_s)

        configured_executable = Path(executable_path).expanduser()
        if configured_executable.is_symlink():
            raise frozen_bpu.Seed17BpuContractError(
                "hrt_model_exec must not be a symlink"
            )
        resolved_executable = configured_executable.resolve(strict=True)
        executable_stat = resolved_executable.stat()
        if not stat.S_ISREG(executable_stat.st_mode):
            raise frozen_bpu.Seed17BpuContractError(
                "hrt_model_exec must be one regular file"
            )
        if command_runner is None:
            if not os.access(resolved_executable, os.X_OK):
                raise frozen_bpu.Seed17BpuContractError(
                    "hrt_model_exec is not executable"
                )
            if executable_stat.st_mode & 0o022:
                raise frozen_bpu.Seed17BpuContractError(
                    "hrt_model_exec must not be group/world writable"
                )

        root = (
            Path(work_root).expanduser()
            if work_root is not None
            else Path(tempfile.gettempdir()) / "rootscope-hrt-shadow"
        )
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise frozen_bpu.Seed17BpuContractError(
                "hrt_model_exec work root must be a real directory"
            )
        try:
            os.chmod(root, 0o700)
        except OSError:
            if command_runner is None:
                raise

        self._command_runner = command_runner
        self._process_lock = threading.Lock()
        self._active_process: subprocess.Popen[str] | None = None
        self.last_inference_evidence: Mapping[str, Any] = {}
        self.model_path = resolved_model
        self.executable_path = resolved_executable
        self.work_root = root.resolve(strict=True)
        self.model_sha256 = actual
        self.executable_sha256 = frozen_bpu.sha256_file(resolved_executable)
        self.injected_test_backend = command_runner is not None

        version_result = self._run(
            [str(self.executable_path), "--version"],
            timeout_s=5.0,
            error_code="HRT_VERSION_FAILED",
        )
        version = (version_result.stdout or version_result.stderr or "").strip()
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:[^\r\n]*)?", version):
            raise frozen_bpu.Seed17BpuContractError(
                f"unexpected hrt_model_exec version output: {version!r}"
            )
        model_info = self._run(
            [
                str(self.executable_path),
                "model_info",
                "--model_file",
                str(self.model_path),
            ],
            timeout_s=10.0,
            error_code="HRT_MODEL_INFO_FAILED",
        )
        _require_hrt_model_info_contract(
            (model_info.stdout or "") + "\n" + (model_info.stderr or "")
        )

        self.hrt_version = version
        self.input_metadata = {
            "name": "image",
            "valid_shape": list(HRT_INPUT_VALID_SHAPE),
            "aligned_shape": list(HRT_INPUT_ALIGNED_SHAPE),
            "aligned_byte_size": HRT_INPUT_ALIGNED_BYTES,
            "dtype": "uint8",
            "buffer_dtype": "uint8",
            "tensor_type": "HB_DNN_IMG_TYPE_RGB",
            "layout": "HB_DNN_LAYOUT_NCHW",
            "stride": [172032, 57344, 256, 1],
            "host_tensor_bytes": int(np.prod(HRT_INPUT_VALID_SHAPE)),
            "vendor_owned_transform": "UINT8_RGB_TO_RGB_128_AND_ALIGNED_STRIDE",
        }
        self.output_metadata = {
            "name": "logits",
            "valid_shape": list(HRT_OUTPUT_COMPILED_SHAPE),
            "aligned_shape": list(HRT_OUTPUT_COMPILED_SHAPE),
            "aligned_byte_size": HRT_OUTPUT_BYTES,
            "dtype": "float32",
            "tensor_type": "HB_DNN_TENSOR_TYPE_F32",
            "layout": "HB_DNN_LAYOUT_NCHW",
            "stride": [16, 4, 4, 4],
        }
        self.compiled_output_shape = HRT_OUTPUT_COMPILED_SHAPE
        self.accepted_runtime_input_buffer_dtypes = ("uint8",)
        self.input_adapter = (
            "HRT_MODEL_EXEC_VALID_UINT8_VENDOR_OWNS_RGB128_AND_ALIGNMENT"
        )
        self.cold_load_per_inference = True
        self.real_time_qualified = False
        self.runtime_wrapper_status = "CANONICAL_VENDOR_COLD_LOAD_PATH"
        self.backend_actual = (
            "FAKE_HRT_MODEL_EXEC_UNIT_TEST_ONLY"
            if self.injected_test_backend
            else f"drobotics.hrt_model_exec@{self.hrt_version}/cold-load"
        )
        self.evidence_scope = (
            "FAKE_HRT_PROTOCOL_TEST_NOT_BPU_EVIDENCE"
            if self.injected_test_backend
            else "RDK_X5_BPU_CANONICAL_COLD_LOAD_SHADOW_FORWARD"
        )

    def _run(
        self,
        command: Sequence[str],
        *,
        timeout_s: float,
        error_code: str,
    ) -> Any:
        if self._command_runner is not None:
            try:
                result = self._command_runner(
                    list(command),
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_s),
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise frozen_bpu.Seed17BpuContractError(
                    f"{error_code}: {type(exc).__name__}: {exc}"
                ) from exc
        else:
            try:
                process = subprocess.Popen(
                    list(command),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except OSError as exc:
                raise frozen_bpu.Seed17BpuContractError(
                    f"{error_code}: {type(exc).__name__}: {exc}"
                ) from exc
            with self._process_lock:
                self._active_process = process
            try:
                try:
                    stdout, stderr = process.communicate(timeout=float(timeout_s))
                except subprocess.TimeoutExpired as exc:
                    process.terminate()
                    try:
                        stdout, stderr = process.communicate(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate(timeout=1.0)
                    raise frozen_bpu.Seed17BpuContractError(
                        f"{error_code}: TimeoutExpired after {timeout_s}s"
                    ) from exc
                result = subprocess.CompletedProcess(
                    list(command),
                    process.returncode,
                    stdout,
                    stderr,
                )
            finally:
                with self._process_lock:
                    if self._active_process is process:
                        self._active_process = None
        returncode = getattr(result, "returncode", None)
        if returncode != 0:
            stderr = str(getattr(result, "stderr", ""))[-500:]
            raise frozen_bpu.Seed17BpuContractError(
                f"{error_code}: returncode={returncode} stderr={stderr!r}"
            )
        return result

    def cancel_active(self) -> None:
        """Terminate an in-flight vendor child during normal worker shutdown."""

        with self._process_lock:
            process = self._active_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                return

    @staticmethod
    def _require_regular_dump(path: Path, expected_bytes: int) -> bytes:
        if path.is_symlink():
            raise BpuShadowProtocolError(
                "INVALID_HRT_DUMP", f"vendor dump is a symlink: {path.name}"
            )
        try:
            metadata = path.stat()
        except FileNotFoundError as exc:
            raise BpuShadowProtocolError(
                "MISSING_HRT_DUMP", f"vendor dump is missing: {path.name}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
            raise BpuShadowProtocolError(
                "INVALID_HRT_DUMP",
                f"vendor dump {path.name} must be {expected_bytes} bytes",
            )
        return path.read_bytes()

    def infer_tensor(self, tensor: np.ndarray) -> Sequence[float]:
        array = validate_tensor(tensor)
        raw = array.tobytes(order="C")
        if len(raw) != int(np.prod(HRT_INPUT_VALID_SHAPE)):
            raise BpuShadowProtocolError(
                "INVALID_TENSOR_BYTES",
                "canonical hrt input must be 150528 uint8 valid-shape bytes",
            )
        with tempfile.TemporaryDirectory(
            prefix="infer-",
            dir=str(self.work_root),
        ) as run_string:
            run_dir = Path(run_string)
            input_path = run_dir / "input_uint8_rgb_nchw.bin"
            dump_path = run_dir / "dump"
            input_path.write_bytes(raw)
            os.chmod(input_path, 0o600)
            dump_path.mkdir(mode=0o700)
            try:
                infer_result = self._run(
                    [
                        str(self.executable_path),
                        "infer",
                        "--model_file",
                        str(self.model_path),
                        "--model_name",
                        HRT_MODEL_NAME,
                        "--input_file",
                        str(input_path),
                        "--enable_dump",
                        "true",
                        "--dump_path",
                        str(dump_path),
                        "--dump_format",
                        "bin",
                    ],
                    timeout_s=self.inference_timeout_s,
                    error_code="HRT_INFER_FAILED",
                )
            except frozen_bpu.Seed17BpuContractError as exc:
                raise BpuShadowProtocolError(
                    "BPU_FORWARD_FAILED", str(exc)
                ) from exc
            dumped_input = self._require_regular_dump(
                dump_path / "model_infer_input_0_image.bin",
                len(raw),
            )
            if dumped_input != raw:
                raise BpuShadowProtocolError(
                    "HRT_INPUT_PROVENANCE_MISMATCH",
                    "vendor-dumped valid uint8 input differs from the wire tensor",
                )
            output_bytes = self._require_regular_dump(
                dump_path / "model_infer_output_0_logits.bin",
                HRT_OUTPUT_BYTES,
            )
            console = (
                str(getattr(infer_result, "stdout", ""))
                + "\n"
                + str(getattr(infer_result, "stderr", ""))
            )
            latency_matches = re.findall(
                r"Infer\s+time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
                console,
                flags=re.IGNORECASE,
            )
            bpu_infer_latency_ms = (
                float(latency_matches[-1]) if len(latency_matches) == 1 else None
            )
            self.last_inference_evidence = {
                "runtime": "drobotics.hrt_model_exec",
                "version": self.hrt_version,
                "argv": [
                    "hrt_model_exec",
                    "infer",
                    "--model_file",
                    "<sha256-bound-r7.bin>",
                    "--model_name",
                    HRT_MODEL_NAME,
                    "--input_file",
                    "<run-scoped-uint8-rgb-nchw.bin>",
                    "--enable_dump",
                    "true",
                    "--dump_path",
                    "<run-scoped-dump-dir>",
                    "--dump_format",
                    "bin",
                ],
                "returncode": int(getattr(infer_result, "returncode", 0)),
                "bpu_infer_latency_ms": bpu_infer_latency_ms,
                "input_dump_sha256": hashlib.sha256(dumped_input).hexdigest(),
                "output_dump_sha256": hashlib.sha256(output_bytes).hexdigest(),
                "input_dump_bytes": len(dumped_input),
                "output_dump_bytes": len(output_bytes),
                "cold_load_per_inference": True,
                "real_time_qualified": False,
            }
            output = np.frombuffer(output_bytes, dtype="<f4").reshape(
                HRT_OUTPUT_COMPILED_SHAPE
            )
            return validate_logits(np.reshape(output, OUTPUT_SHAPE))


class HashBoundR7BpuBackend(HrtModelExecR7BpuBackend):
    """Default canonical r7 backend retained under the public class name."""


class LegacyPyeasyR7BpuBackend:
    """Explicit legacy pyeasy adapter; never an automatic fallback.

    The immutable ``app.edge.bpu_seed17.Seed17BpuRunner`` intentionally
    remains unchanged.  Actual r7 metadata on X5 exposes ``[1,4,1,1]`` while
    some fake/golden runtimes expose ``[1,4]``.  This adapter accepts exactly
    those two shapes and squeezes only the two trailing singleton axes.
    """

    def __init__(
        self,
        model_path: str | Path,
        expected_sha256: str,
        *,
        dnn_module: Any | None = None,
    ) -> None:
        resolved, actual = _validate_hash_bound_model(model_path, expected_sha256)
        self.injected_test_backend = dnn_module is not None
        if dnn_module is None:
            try:
                from hobot_dnn import pyeasy_dnn as dnn_module  # type: ignore
            except ImportError as exc:
                raise frozen_bpu.Seed17BpuContractError(
                    "hobot_dnn.pyeasy_dnn is unavailable; no fake fallback is allowed"
                ) from exc
        try:
            models = dnn_module.load(str(resolved))
        except Exception as exc:
            raise frozen_bpu.Seed17BpuContractError(
                f"pyeasy_dnn model load failed: {exc}"
            ) from exc
        if not isinstance(models, (list, tuple)) or len(models) != 1:
            count = len(models) if isinstance(models, (list, tuple)) else "non-sequence"
            raise frozen_bpu.Seed17BpuContractError(
                f"BPU bin must expose exactly one model, got {count}"
            )
        model = models[0]
        inputs = list(getattr(model, "inputs", ()))
        outputs = list(getattr(model, "outputs", ()))
        if len(inputs) != 1 or len(outputs) != 1:
            raise frozen_bpu.Seed17BpuContractError(
                f"BPU model must expose one input and one output, got "
                f"{len(inputs)}/{len(outputs)}"
            )
        input_metadata = frozen_bpu._tensor_metadata(inputs[0])
        output_metadata = frozen_bpu._tensor_metadata(outputs[0])
        input_shape = frozen_bpu._effective_shape(inputs[0])
        output_shape = frozen_bpu._effective_shape(outputs[0])
        if input_shape != (1, 3, 224, 224):
            raise frozen_bpu.Seed17BpuContractError(
                f"BPU input shape must be (1, 3, 224, 224), got {input_shape}"
            )
        if output_shape not in {(1, 4), (1, 4, 1, 1)}:
            raise frozen_bpu.Seed17BpuContractError(
                "competition r7 BPU output shape must be exactly "
                f"(1, 4) or (1, 4, 1, 1), got {output_shape}"
            )
        layout = str(frozen_bpu._property_value(inputs[0], "layout") or "").upper()
        if "NCHW" not in layout:
            raise frozen_bpu.Seed17BpuContractError(
                f"BPU input layout must be NCHW, got {layout!r}"
            )
        declared_dtype = str(
            frozen_bpu._property_value(inputs[0], "dtype") or ""
        ).upper()
        declared_tensor_type = str(
            frozen_bpu._property_value(inputs[0], "tensor_type") or ""
        ).upper()
        runtime_buffer_dtype = input_metadata.get("buffer_dtype")
        if "UINT8" not in declared_dtype:
            raise frozen_bpu.Seed17BpuContractError(
                f"BPU input properties dtype must declare uint8, got {declared_dtype!r}"
            )
        if "RGB" not in declared_tensor_type:
            raise frozen_bpu.Seed17BpuContractError(
                "BPU input properties tensor_type must declare RGB, got "
                f"{declared_tensor_type!r}"
            )
        # Actual RDK X5 pyeasy_dnn exposes an int8 backing buffer even though
        # the public tensor properties and compiled DDR contract are uint8.
        # Accept only that observed storage alias (or ordinary uint8); the
        # AF_UNIX wire tensor remains strictly uint8.
        if runtime_buffer_dtype not in {"uint8", "int8"}:
            raise frozen_bpu.Seed17BpuContractError(
                "BPU input backing buffer must be uint8 or the observed pyeasy "
                f"int8 storage alias, got {runtime_buffer_dtype!r}"
            )

        self.model = model
        self.model_path = resolved
        self.model_sha256 = actual
        self.input_metadata = dict(input_metadata)
        self.output_metadata = dict(output_metadata)
        self.compiled_output_shape = tuple(output_shape)
        self.accepted_runtime_input_buffer_dtypes = ("uint8", "int8")
        self.input_adapter = "PYEASY_DNN_LEGACY_NUMERICALLY_UNQUALIFIED"
        self.cold_load_per_inference = False
        self.real_time_qualified = False
        self.runtime_wrapper_status = "LEGACY_WRAPPER_NUMERICALLY_UNQUALIFIED"
        self.backend_actual = (
            "FAKE_PYEASY_MODEL_UNIT_TEST_ONLY"
            if self.injected_test_backend
            else "hobot_dnn.pyeasy_dnn_LEGACY_EXPLICIT_ONLY"
        )
        self.evidence_scope = (
            "FAKE_PYEASY_PROTOCOL_TEST_NOT_BPU_EVIDENCE"
            if self.injected_test_backend
            else "RDK_X5_BPU_LEGACY_WRAPPER_NUMERICALLY_UNQUALIFIED"
        )

    def infer_tensor(self, tensor: np.ndarray) -> Sequence[float]:
        array = validate_tensor(tensor)
        try:
            values = self.model.forward(array)
        except Exception as exc:
            raise BpuShadowProtocolError(
                "BPU_FORWARD_FAILED", f"pyeasy_dnn forward failed: {exc}"
            ) from exc
        if not isinstance(values, (list, tuple)) or len(values) != 1:
            count = len(values) if isinstance(values, (list, tuple)) else "non-sequence"
            raise BpuShadowProtocolError(
                "INVALID_BPU_OUTPUT_COUNT",
                f"BPU forward must return one output, got {count}",
            )
        raw = getattr(values[0], "buffer", values[0])
        array_out = np.asarray(raw)
        observed_shape = tuple(array_out.shape)
        if observed_shape not in {(1, 4), (1, 4, 1, 1)}:
            raise BpuShadowProtocolError(
                "INVALID_BPU_OUTPUT_SHAPE",
                "BPU runtime output must have shape exactly "
                f"(1, 4) or (1, 4, 1, 1), got {observed_shape}",
            )
        canonical = np.reshape(array_out, OUTPUT_SHAPE)
        return validate_logits(canonical)


def _base_response(
    backend: ShadowBackend,
    *,
    request_id: str | None,
    status: str,
) -> dict[str, Any]:
    return {
        "schema": RESPONSE_SCHEMA,
        "protocol": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": status,
        "shadow_only": True,
        "zero_authority": True,
        "authority": dict(ZERO_AUTHORITY),
        "model": {
            "release_id": R7_RELEASE_ID,
            "sha256": backend.model_sha256,
            "qualification": "SHADOW_CANDIDATE_NOT_DEFAULT",
            "selected_bin_changed": False,
        },
        "backend": {
            "mode": "BPU_SHADOW",
            "actual": backend.backend_actual,
            "evidence_scope": backend.evidence_scope,
            "injected_test_backend": backend.injected_test_backend,
            "actual_metadata": {
                "input": dict(backend.input_metadata),
                "output": dict(backend.output_metadata),
                "compiled_output_shape": list(backend.compiled_output_shape),
                "accepted_output_shapes": [[1, 4], [1, 4, 1, 1]],
                "canonical_logits_shape": [1, 4],
                "squeeze_policy": "ONLY_TRAILING_SINGLETON_AXES_FOR_1x4x1x1",
                "wire_input_dtype": "uint8",
                "declared_input_dtype": backend.input_metadata.get("dtype"),
                "runtime_input_buffer_dtype": backend.input_metadata.get(
                    "buffer_dtype"
                ),
                "accepted_runtime_input_buffer_dtypes": list(
                    backend.accepted_runtime_input_buffer_dtypes
                ),
                "input_adapter": backend.input_adapter,
                "cold_load_per_inference": backend.cold_load_per_inference,
                "real_time_qualified": backend.real_time_qualified,
                "runtime_wrapper_status": backend.runtime_wrapper_status,
                "host_must_not_center_or_pad": (
                    backend.input_adapter
                    == "HRT_MODEL_EXEC_VALID_UINT8_VENDOR_OWNS_RGB128_AND_ALIGNMENT"
                ),
            },
        },
        "interface": {
            "input_shape": [1, 3, 224, 224],
            "input_dtype": "uint8",
            "input_layout": "NCHW",
            "input_color_order": "RGB",
            "output_shape": [1, 4],
            "batch_min": 1,
            "batch_max": 4,
        },
    }


class UnixBpuShadowWorker:
    """One persistent model with one framed request per AF_UNIX connection."""

    def __init__(
        self,
        socket_path: str | Path,
        backend: ShadowBackend,
        *,
        connection_timeout_s: float = 1.0,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.backend = backend
        if not 0.05 <= float(connection_timeout_s) <= 5.0:
            raise ValueError("connection_timeout_s must be within [0.05, 5.0]")
        self.connection_timeout_s = float(connection_timeout_s)

    def process_message(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request_id, tensors, hashes = decode_request(payload)
        batch_started = time.perf_counter_ns()
        results: list[dict[str, Any]] = []
        for index, (tensor, digest) in enumerate(zip(tensors, hashes, strict=True)):
            started = time.perf_counter_ns()
            logits = validate_logits(self.backend.infer_tensor(tensor))
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            results.append(
                {
                    "index": index,
                    "input_sha256": digest,
                    "logits": logits,
                    "latency_ms": latency_ms,
                }
            )
        response = _base_response(
            self.backend,
            request_id=request_id,
            status="OK_SHADOW_ONLY",
        )
        response["batch"] = {
            "count": len(results),
            "latency_ms": (time.perf_counter_ns() - batch_started) / 1_000_000.0,
            "sequential_fixed_batch1_forwards": len(results),
        }
        response["results"] = results
        response["error"] = None
        return response

    def _error_response(
        self,
        exc: Exception,
        *,
        request_id: str | None = None,
    ) -> Mapping[str, Any]:
        code = (
            exc.code
            if isinstance(exc, BpuShadowProtocolError)
            else "WORKER_INTERNAL_ERROR"
        )
        response = _base_response(
            self.backend,
            request_id=request_id,
            status="ERROR_FAIL_CLOSED",
        )
        response["batch"] = None
        response["results"] = None
        response["error"] = {
            "code": code,
            "type": type(exc).__name__,
            "message": str(exc)[:500],
        }
        return response

    def serve_connection(self, connection: socket.socket) -> None:
        """Read one request and emit exactly one response before returning."""

        request_id: str | None = None
        try:
            connection.settimeout(self.connection_timeout_s)
            payload = recv_frame(connection)
            raw_id = payload.get("request_id")
            request_id = raw_id if isinstance(raw_id, str) else None
            response = self.process_message(payload)
        except Exception as exc:
            response = self._error_response(exc, request_id=request_id)
        try:
            send_frame(connection, response)
        except (BrokenPipeError, ConnectionError, OSError, socket.timeout):
            # A vanished client has no effect on any other request or system.
            return

    def _prepare_socket_path(self) -> Path:
        path = self.socket_path.expanduser()
        if not path.is_absolute():
            raise ValueError("AF_UNIX socket path must be absolute")
        parent = path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("AF_UNIX socket parent must be a real directory")
        try:
            current = os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(current.st_mode) or not stat.S_ISSOCK(current.st_mode):
                raise ValueError("socket path exists and is not a real Unix socket")
            os.unlink(path)
        return path

    def serve_forever(self, stop_event: threading.Event | None = None) -> None:
        """Bind AF_UNIX only and serve until ``stop_event`` is set."""

        stopper = stop_event or threading.Event()
        path = self._prepare_socket_path()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(path))
            os.chmod(path, 0o600)
            listener.listen(4)
            listener.settimeout(0.2)
            while not stopper.is_set():
                try:
                    connection, _address = listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    self.serve_connection(connection)
        finally:
            listener.close()
            try:
                current = os.lstat(path)
            except FileNotFoundError:
                return
            if stat.S_ISSOCK(current.st_mode):
                os.unlink(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--model-bin", required=True, type=Path)
    parser.add_argument(
        "--expected-model-sha256",
        default=R7_REFERENCE_SHA256,
        help="exact lowercase r7 bin SHA-256",
    )
    parser.add_argument("--connection-timeout-s", type=float, default=1.0)
    parser.add_argument(
        "--backend",
        choices=("canonical_hrt", "legacy_pyeasy"),
        default="canonical_hrt",
        help=(
            "canonical_hrt is the exact vendor input-adapter path; "
            "legacy_pyeasy is explicit, numerically unqualified, and never "
            "selected as an automatic fallback"
        ),
    )
    parser.add_argument(
        "--hrt-model-exec",
        type=Path,
        default=HRT_MODEL_EXEC_DEFAULT,
    )
    parser.add_argument("--hrt-work-root", type=Path)
    parser.add_argument("--hrt-timeout-s", type=float, default=8.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.backend == "canonical_hrt":
        backend: ShadowBackend = HashBoundR7BpuBackend(
            args.model_bin,
            args.expected_model_sha256,
            executable_path=args.hrt_model_exec,
            work_root=args.hrt_work_root,
            inference_timeout_s=args.hrt_timeout_s,
        )
    else:
        backend = LegacyPyeasyR7BpuBackend(
            args.model_bin,
            args.expected_model_sha256,
        )
    worker = UnixBpuShadowWorker(
        args.socket,
        backend,
        connection_timeout_s=args.connection_timeout_s,
    )
    stop_event = threading.Event()

    def _stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    worker.serve_forever(stop_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HashBoundR7BpuBackend",
    "HrtModelExecR7BpuBackend",
    "LegacyPyeasyR7BpuBackend",
    "ShadowBackend",
    "UnixBpuShadowWorker",
    "main",
]
