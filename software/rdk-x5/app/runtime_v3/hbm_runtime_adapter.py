"""Persistent, hash-bound ``hbm_runtime`` candidate for RootScope r7.

The existing vendor CLI adapter remains the numerical oracle.  This module is
intentionally a *candidate*: it keeps one HBM runtime resident and exposes two
explicit input policies for X5 qualification.  A policy is not selected until
same-input replay on the actual board agrees with the frozen oracle.

Nothing in this module can open a camera, serial port, GPIO, service manager,
or actuator.  Returned logits are zero-authority evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_SHAPES = {(1, 4), (1, 4, 1, 1)}
INPUT_POLICIES = frozenset({"RAW_UINT8", "RGB128_CENTERED_INT8"})
ZERO_AUTHORITY = {
    "execution_authority": False,
    "serial_write": False,
    "gpio_write": False,
    "pump_command": False,
    "state_machine_write": False,
    "physical_completion": False,
}


class HbmRuntimeContractError(ValueError):
    """Raised when model, tensor, or vendor runtime metadata is not exact."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).upper()


def _one_model_mapping(value: Any, name: str) -> tuple[str, Any]:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise HbmRuntimeContractError(f"{name} must expose exactly one model")
    return next(iter(value.items()))


@dataclass(frozen=True)
class HbmModelMetadata:
    model_name: str
    input_name: str
    input_shape: tuple[int, ...]
    input_dtype: str
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "output_name": self.output_name,
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
        }


RuntimeFactory = Callable[[str], Any]


class PersistentHbmR7Adapter:
    """Load the exact r7 BIN once and produce finite four-class logits."""

    def __init__(
        self,
        model_path: str | Path,
        expected_sha256: str,
        *,
        input_policy: str,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        if input_policy not in INPUT_POLICIES:
            raise HbmRuntimeContractError(
                f"input_policy must be one of {sorted(INPUT_POLICIES)}"
            )
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise HbmRuntimeContractError("expected_sha256 must be a SHA-256")
        configured = Path(model_path).expanduser()
        if configured.is_symlink():
            raise HbmRuntimeContractError("model path must not be a symlink")
        resolved = configured.resolve(strict=True)
        mode = resolved.stat().st_mode
        if not stat.S_ISREG(mode) or resolved.suffix.lower() != ".bin":
            raise HbmRuntimeContractError("model must be one regular .bin file")
        if mode & 0o022:
            raise HbmRuntimeContractError("model must not be group/world writable")
        actual = _sha256_file(resolved)
        if actual != expected_sha256.lower():
            raise HbmRuntimeContractError(
                f"model SHA-256 mismatch: actual={actual} expected={expected_sha256}"
            )

        injected = runtime_factory is not None
        if runtime_factory is None:
            try:
                from hbm_runtime import HB_HBMRuntime  # type: ignore
            except ImportError as exc:
                raise HbmRuntimeContractError(
                    "hbm_runtime unavailable; fake fallback is forbidden"
                ) from exc
            runtime_factory = HB_HBMRuntime
        try:
            runtime = runtime_factory(str(resolved))
        except Exception as exc:
            raise HbmRuntimeContractError(f"hbm model load failed: {exc}") from exc

        self.runtime = runtime
        self.model_path = resolved
        self.model_sha256 = actual
        self.input_policy = input_policy
        self.injected_test_backend = injected
        self.metadata = self._read_metadata(runtime)
        self.load_monotonic_ns = time.monotonic_ns()
        self.inference_count = 0

    @staticmethod
    def _read_metadata(runtime: Any) -> HbmModelMetadata:
        model_names = list(getattr(runtime, "model_names", ()))
        if len(model_names) != 1:
            raise HbmRuntimeContractError(
                f"runtime must expose one model, got {len(model_names)}"
            )
        model_name = str(model_names[0])
        input_names_map = getattr(runtime, "input_names", None)
        output_names_map = getattr(runtime, "output_names", None)
        input_shapes_map = getattr(runtime, "input_shapes", None)
        output_shapes_map = getattr(runtime, "output_shapes", None)
        input_dtypes_map = getattr(runtime, "input_dtypes", None)
        output_dtypes_map = getattr(runtime, "output_dtypes", None)
        for label, value in (
            ("input_names", input_names_map),
            ("output_names", output_names_map),
            ("input_shapes", input_shapes_map),
            ("output_shapes", output_shapes_map),
            ("input_dtypes", input_dtypes_map),
            ("output_dtypes", output_dtypes_map),
        ):
            if not isinstance(value, Mapping) or model_name not in value:
                raise HbmRuntimeContractError(f"runtime {label} metadata missing")

        input_names = list(input_names_map[model_name])
        output_names = list(output_names_map[model_name])
        if len(input_names) != 1 or len(output_names) != 1:
            raise HbmRuntimeContractError("r7 must expose one input and one output")
        input_name = str(input_names[0])
        output_name = str(output_names[0])
        input_shape = tuple(int(item) for item in input_shapes_map[model_name][input_name])
        output_shape = tuple(
            int(item) for item in output_shapes_map[model_name][output_name]
        )
        if input_shape != INPUT_SHAPE:
            raise HbmRuntimeContractError(
                f"r7 input shape must be {INPUT_SHAPE}, got {input_shape}"
            )
        if output_shape not in OUTPUT_SHAPES:
            raise HbmRuntimeContractError(
                f"r7 output shape must be one of {sorted(OUTPUT_SHAPES)}, "
                f"got {output_shape}"
            )
        input_dtype = _enum_name(input_dtypes_map[model_name][input_name])
        output_dtype = _enum_name(output_dtypes_map[model_name][output_name])
        if not any(item in input_dtype for item in ("RGB", "U8", "S8")):
            raise HbmRuntimeContractError(
                f"r7 input dtype must declare RGB/U8/S8, got {input_dtype!r}"
            )
        if "F32" not in output_dtype:
            raise HbmRuntimeContractError(
                f"r7 output dtype must declare F32, got {output_dtype!r}"
            )
        return HbmModelMetadata(
            model_name=model_name,
            input_name=input_name,
            input_shape=input_shape,
            input_dtype=input_dtype,
            output_name=output_name,
            output_shape=output_shape,
            output_dtype=output_dtype,
        )

    def _prepare(self, tensor: np.ndarray) -> np.ndarray:
        array = np.asarray(tensor)
        if tuple(array.shape) != INPUT_SHAPE or array.dtype != np.uint8:
            raise HbmRuntimeContractError(
                f"external tensor must be uint8 {INPUT_SHAPE}, "
                f"got {array.dtype} {tuple(array.shape)}"
            )
        if not np.isfinite(array).all():
            raise HbmRuntimeContractError("external tensor contains non-finite data")
        contiguous = np.ascontiguousarray(array)
        if self.input_policy == "RAW_UINT8":
            return contiguous
        # Explicit modulo-free centering: 0..255 -> -128..127.
        return np.ascontiguousarray(contiguous.astype(np.int16) - 128, dtype=np.int8)

    def infer_uint8(self, tensor: np.ndarray) -> dict[str, Any]:
        external = np.asarray(tensor)
        prepared = self._prepare(external)
        started = time.perf_counter_ns()
        try:
            raw = self.runtime.run(prepared)
        except Exception as exc:
            raise HbmRuntimeContractError(f"hbm inference failed: {exc}") from exc
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if not isinstance(raw, Mapping) or self.metadata.model_name not in raw:
            raise HbmRuntimeContractError("runtime result does not contain model name")
        model_result = raw[self.metadata.model_name]
        if isinstance(model_result, Mapping):
            if set(model_result) != {self.metadata.output_name}:
                raise HbmRuntimeContractError("runtime result output names changed")
            model_result = model_result[self.metadata.output_name]
        output = np.asarray(model_result)
        if tuple(output.shape) not in OUTPUT_SHAPES:
            raise HbmRuntimeContractError(
                f"runtime output shape changed: {tuple(output.shape)}"
            )
        logits = np.asarray(output, dtype=np.float32).reshape(1, 4)
        if not np.isfinite(logits).all():
            raise HbmRuntimeContractError("runtime produced non-finite logits")
        self.inference_count += 1
        return {
            "schema": "rootscope.runtime-v3.hbm-evidence.v1",
            "status": "CANDIDATE_UNQUALIFIED",
            "backend_actual": (
                "CALLER_INJECTED_FAKE_HBM_TEST_ONLY"
                if self.injected_test_backend
                else "hbm_runtime.HB_HBMRuntime"
            ),
            "persistent_model": True,
            "cold_load_per_inference": False,
            "input_policy": self.input_policy,
            "model_sha256": self.model_sha256,
            "metadata": self.metadata.to_dict(),
            "external_tensor_sha256": hashlib.sha256(
                np.ascontiguousarray(external).tobytes()
            ).hexdigest(),
            "runtime_tensor_sha256": hashlib.sha256(prepared.tobytes()).hexdigest(),
            "runtime_tensor_dtype": str(prepared.dtype),
            "logits": logits[0].astype(float).tolist(),
            "top1_index": int(np.argmax(logits[0])),
            "latency_ms": latency_ms,
            "inference_count_since_load": self.inference_count,
            "qualification": {
                "selected_for_runtime": False,
                "requires_actual_x5_replay": True,
                "requires_oracle_agreement": True,
                "oracle_backend": "drobotics.hrt_model_exec",
            },
            "authority": dict(ZERO_AUTHORITY),
        }


__all__ = [
    "HbmRuntimeContractError",
    "HbmModelMetadata",
    "PersistentHbmR7Adapter",
]
