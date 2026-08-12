"""Strict configuration contract for the RootScope clean-X5 capsule.

This is a deployment *description*, not an activation API.  Every authority
field is required to be the boolean value ``False``.  Optional device entries
are explicit aliases used only by the read-only preflight; this module never
enumerates or opens them.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence, Tuple


CAPSULE_SCHEMA_VERSION = "rootscope.x5-offline-capsule.v1"
CAPSULE_STATUS = "SIMULATED_ONLY_CLEAN_X5_CAPSULE_NOT_X5_QUALIFIED"
CPU_PROVIDER = "CPUExecutionProvider"
PREPROCESS_MODE = "torchvision_resize_short_side_center_crop_rgb_imagenet_v1"
GOLDEN_GENERATOR = "rootscope_simulated_rgb_v1"
ROOTSCOPE_CLASS_ORDER: Tuple[str, ...] = (
    "grass_clump",
    "low_shrub",
    "young_tree",
    "unknown",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

AUTHORITY_FIELDS: Tuple[str, ...] = (
    "hardware_touched",
    "network_touched",
    "ports_enumerated",
    "x5_validated",
    "bpu_ready",
    "bpu_used",
    "model_candidate",
    "model_qualified",
    "physical_authority",
    "execution_authority",
    "physical_completion",
)


def _strict_keys(
    payload: Mapping[str, Any], expected: Sequence[str], context: str
) -> None:
    expected_set = set(expected)
    actual = set(payload)
    missing = expected_set - actual
    unknown = actual - expected_set
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch: missing={sorted(missing)} "
            f"unknown={sorted(unknown)}"
        )


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_sha256(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be null or a lowercase SHA-256 digest")
    return value


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _float_triplet(value: Any, field: str, *, positive: bool) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three numbers")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field} must contain only numbers")
        number = float(item)
        if not math.isfinite(number) or (positive and number <= 0.0):
            raise ValueError(f"{field} contains an invalid number")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True)
class AuthorityBoundary:
    hardware_touched: bool
    network_touched: bool
    ports_enumerated: bool
    x5_validated: bool
    bpu_ready: bool
    bpu_used: bool
    model_candidate: bool
    model_qualified: bool
    physical_authority: bool
    execution_authority: bool
    physical_completion: bool

    def __post_init__(self) -> None:
        for field in AUTHORITY_FIELDS:
            value = getattr(self, field)
            if not isinstance(value, bool):
                raise ValueError(f"authority.{field} must be boolean")
            if value:
                raise ValueError(
                    f"authority.{field} must remain false in the zero-authority capsule"
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AuthorityBoundary":
        if not isinstance(payload, Mapping):
            raise ValueError("authority must be an object")
        _strict_keys(payload, AUTHORITY_FIELDS, "authority")
        return cls(**{field: _bool(payload[field], f"authority.{field}") for field in AUTHORITY_FIELDS})

    def to_dict(self) -> Mapping[str, bool]:
        return {field: getattr(self, field) for field in AUTHORITY_FIELDS}


@dataclass(frozen=True)
class PreprocessConfig:
    mode: str
    short_side: int
    center_crop: Tuple[int, int]
    input_shape: Tuple[int, int, int, int]
    color_order: str
    scale: float
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    interpolation: str
    golden_generator: str
    golden_source_shape: Tuple[int, int, int]
    golden_tensor_sha256: str

    def __post_init__(self) -> None:
        if self.mode != PREPROCESS_MODE:
            raise ValueError(f"preprocess.mode must be {PREPROCESS_MODE}")
        if isinstance(self.short_side, bool) or not isinstance(self.short_side, int) or self.short_side <= 0:
            raise ValueError("preprocess.short_side must be a positive integer")
        if len(self.center_crop) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.center_crop
        ):
            raise ValueError("preprocess.center_crop must be [positive_height, positive_width]")
        if self.short_side < max(self.center_crop):
            raise ValueError("preprocess.short_side must cover the center crop")
        if self.input_shape[0] != 1 or self.input_shape[1] != 3:
            raise ValueError("preprocess.input_shape must be static [1,3,H,W]")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.input_shape):
            raise ValueError("preprocess.input_shape must contain positive integers")
        if self.input_shape[2:] != self.center_crop:
            raise ValueError("preprocess input H/W must equal center_crop H/W")
        if self.color_order != "RGB":
            raise ValueError("only explicit RGB preprocessing is supported")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("preprocess.scale must be finite and positive")
        if self.interpolation != "bilinear":
            raise ValueError("only PIL bilinear interpolation is supported")
        if self.golden_generator != GOLDEN_GENERATOR:
            raise ValueError(f"preprocess.golden_generator must be {GOLDEN_GENERATOR}")
        if len(self.golden_source_shape) != 3 or self.golden_source_shape[2] != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.golden_source_shape
        ):
            raise ValueError("preprocess.golden_source_shape must be [positive_H,positive_W,3]")
        if not _SHA256_RE.fullmatch(self.golden_tensor_sha256):
            raise ValueError("preprocess.golden_tensor_sha256 must be a lowercase SHA-256 digest")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PreprocessConfig":
        expected = (
            "mode",
            "short_side",
            "center_crop",
            "input_shape",
            "color_order",
            "scale",
            "mean",
            "std",
            "interpolation",
            "golden_generator",
            "golden_source_shape",
            "golden_tensor_sha256",
        )
        if not isinstance(payload, Mapping):
            raise ValueError("model.preprocess must be an object")
        _strict_keys(payload, expected, "model.preprocess")
        shape = payload["input_shape"]
        if not isinstance(shape, list) or len(shape) != 4:
            raise ValueError("model.preprocess.input_shape must be a four-item list")
        center_crop = payload["center_crop"]
        if not isinstance(center_crop, list) or len(center_crop) != 2:
            raise ValueError("model.preprocess.center_crop must be a two-item list")
        golden_source_shape = payload["golden_source_shape"]
        if not isinstance(golden_source_shape, list) or len(golden_source_shape) != 3:
            raise ValueError("model.preprocess.golden_source_shape must be a three-item list")
        if isinstance(payload["scale"], bool) or not isinstance(payload["scale"], (int, float)):
            raise ValueError("model.preprocess.scale must be numeric")
        return cls(
            mode=_required_string(payload["mode"], "model.preprocess.mode"),
            short_side=payload["short_side"],
            center_crop=tuple(center_crop),
            input_shape=tuple(shape),
            color_order=_required_string(payload["color_order"], "model.preprocess.color_order"),
            scale=float(payload["scale"]),
            mean=_float_triplet(payload["mean"], "model.preprocess.mean", positive=False),
            std=_float_triplet(payload["std"], "model.preprocess.std", positive=True),
            interpolation=_required_string(
                payload["interpolation"], "model.preprocess.interpolation"
            ),
            golden_generator=_required_string(
                payload["golden_generator"], "model.preprocess.golden_generator"
            ),
            golden_source_shape=tuple(golden_source_shape),
            golden_tensor_sha256=_required_string(
                payload["golden_tensor_sha256"],
                "model.preprocess.golden_tensor_sha256",
            ),
        )


@dataclass(frozen=True)
class ModelConfig:
    enabled: bool
    path: str
    sha256: str | None
    provider: str
    input_name: str | None
    output_name: str | None
    output_shape: Tuple[int, int]
    class_order: Tuple[str, ...]
    model_candidate: bool
    model_qualified: bool
    bpu_ready: bool
    preprocess: PreprocessConfig

    def __post_init__(self) -> None:
        if self.provider != CPU_PROVIDER:
            raise ValueError("the capsule permits only CPUExecutionProvider")
        if self.output_shape != (1, len(ROOTSCOPE_CLASS_ORDER)):
            raise ValueError("model.output_shape must be [1,4]")
        if self.class_order != ROOTSCOPE_CLASS_ORDER:
            raise ValueError(f"model.class_order must be {ROOTSCOPE_CLASS_ORDER}")
        for field in ("model_candidate", "model_qualified", "bpu_ready"):
            value = getattr(self, field)
            if not isinstance(value, bool) or value:
                raise ValueError(f"model.{field} must be boolean false")
        if self.enabled:
            _required_string(self.path, "model.path")
            if self.sha256 is None:
                raise ValueError("model.sha256 is required when model.enabled=true")
        elif self.sha256 is not None and not self.path:
            raise ValueError("model.path is required when model.sha256 is supplied")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelConfig":
        expected = (
            "enabled",
            "path",
            "sha256",
            "provider",
            "input_name",
            "output_name",
            "output_shape",
            "class_order",
            "model_candidate",
            "model_qualified",
            "bpu_ready",
            "preprocess",
        )
        if not isinstance(payload, Mapping):
            raise ValueError("model must be an object")
        _strict_keys(payload, expected, "model")
        for optional_name in ("input_name", "output_name"):
            if payload[optional_name] is not None and (
                not isinstance(payload[optional_name], str) or not payload[optional_name]
            ):
                raise ValueError(f"model.{optional_name} must be null or non-empty string")
        output_shape = payload["output_shape"]
        if not isinstance(output_shape, list) or len(output_shape) != 2:
            raise ValueError("model.output_shape must be a two-item list")
        class_order = payload["class_order"]
        if not isinstance(class_order, list) or not all(
            isinstance(value, str) and value for value in class_order
        ):
            raise ValueError("model.class_order must be a non-empty string list")
        return cls(
            enabled=_bool(payload["enabled"], "model.enabled"),
            path=payload["path"] if isinstance(payload["path"], str) else "",
            sha256=_optional_sha256(payload["sha256"], "model.sha256"),
            provider=_required_string(payload["provider"], "model.provider"),
            input_name=payload["input_name"],
            output_name=payload["output_name"],
            output_shape=tuple(output_shape),
            class_order=tuple(class_order),
            model_candidate=_bool(payload["model_candidate"], "model.model_candidate"),
            model_qualified=_bool(payload["model_qualified"], "model.model_qualified"),
            bpu_ready=_bool(payload["bpu_ready"], "model.bpu_ready"),
            preprocess=PreprocessConfig.from_mapping(payload["preprocess"]),
        )


@dataclass(frozen=True)
class InputEndpoint:
    enabled: bool
    required: bool
    backend: str
    device: str
    use: str

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, name: str, allowed_backends: Sequence[str]
    ) -> "InputEndpoint":
        expected = ("enabled", "required", "backend", "device", "use")
        if not isinstance(payload, Mapping):
            raise ValueError(f"inputs.{name} must be an object")
        _strict_keys(payload, expected, f"inputs.{name}")
        endpoint = cls(
            enabled=_bool(payload["enabled"], f"inputs.{name}.enabled"),
            required=_bool(payload["required"], f"inputs.{name}.required"),
            backend=_required_string(payload["backend"], f"inputs.{name}.backend"),
            device=payload["device"] if isinstance(payload["device"], str) else "",
            use=_required_string(payload["use"], f"inputs.{name}.use"),
        )
        if endpoint.backend not in set(allowed_backends):
            raise ValueError(f"inputs.{name}.backend is unsupported")
        if endpoint.required and not endpoint.enabled:
            raise ValueError(f"inputs.{name} cannot be required while disabled")
        if endpoint.enabled and not endpoint.device:
            raise ValueError(f"inputs.{name}.device is required when enabled")
        return endpoint


@dataclass(frozen=True)
class LlmConfig:
    enabled: bool
    executable: str
    model_path: str
    model_sha256: str | None
    host: str
    port: int
    read_only: bool
    tool_execution: bool
    actuator_access: bool

    def __post_init__(self) -> None:
        if self.host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("llm.host must be loopback")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1024 <= self.port <= 65535:
            raise ValueError("llm.port must be an unprivileged TCP port")
        if self.read_only is not True:
            raise ValueError("llm.read_only must be true")
        if self.tool_execution is not False or self.actuator_access is not False:
            raise ValueError("LLM tool execution and actuator access must remain false")
        if self.enabled:
            _required_string(self.executable, "llm.executable")
            _required_string(self.model_path, "llm.model_path")
            if self.model_sha256 is None:
                raise ValueError("llm.model_sha256 is required when llm.enabled=true")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LlmConfig":
        expected = (
            "enabled",
            "executable",
            "model_path",
            "model_sha256",
            "host",
            "port",
            "read_only",
            "tool_execution",
            "actuator_access",
        )
        if not isinstance(payload, Mapping):
            raise ValueError("llm must be an object")
        _strict_keys(payload, expected, "llm")
        return cls(
            enabled=_bool(payload["enabled"], "llm.enabled"),
            executable=payload["executable"] if isinstance(payload["executable"], str) else "",
            model_path=payload["model_path"] if isinstance(payload["model_path"], str) else "",
            model_sha256=_optional_sha256(payload["model_sha256"], "llm.model_sha256"),
            host=_required_string(payload["host"], "llm.host"),
            port=payload["port"],
            read_only=_bool(payload["read_only"], "llm.read_only"),
            tool_execution=_bool(payload["tool_execution"], "llm.tool_execution"),
            actuator_access=_bool(payload["actuator_access"], "llm.actuator_access"),
        )


@dataclass(frozen=True)
class CapsuleConfig:
    schema_version: str
    status: str
    project_root: str
    python_executable: str
    dashboard_host: str
    dashboard_port: int
    authority: AuthorityBoundary
    model: ModelConfig
    rgb: InputEndpoint
    depth: InputEndpoint
    llm: LlmConfig

    def __post_init__(self) -> None:
        if self.schema_version != CAPSULE_SCHEMA_VERSION:
            raise ValueError("unsupported capsule schema_version")
        if self.status != CAPSULE_STATUS:
            raise ValueError("capsule status must preserve the unqualified claim boundary")
        _required_string(self.project_root, "project_root")
        _required_string(self.python_executable, "python_executable")
        if self.dashboard_host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("dashboard_host must default to loopback in this capsule")
        if (
            isinstance(self.dashboard_port, bool)
            or not isinstance(self.dashboard_port, int)
            or not 1024 <= self.dashboard_port <= 65535
        ):
            raise ValueError("dashboard_port must be an unprivileged TCP port")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CapsuleConfig":
        expected = (
            "schema_version",
            "status",
            "project_root",
            "python_executable",
            "dashboard_host",
            "dashboard_port",
            "authority",
            "model",
            "inputs",
            "llm",
        )
        if not isinstance(payload, Mapping):
            raise ValueError("capsule configuration must be an object")
        _strict_keys(payload, expected, "capsule")
        inputs = payload["inputs"]
        if not isinstance(inputs, Mapping):
            raise ValueError("inputs must be an object")
        _strict_keys(inputs, ("rgb", "depth"), "inputs")
        return cls(
            schema_version=_required_string(payload["schema_version"], "schema_version"),
            status=_required_string(payload["status"], "status"),
            project_root=_required_string(payload["project_root"], "project_root"),
            python_executable=_required_string(payload["python_executable"], "python_executable"),
            dashboard_host=_required_string(payload["dashboard_host"], "dashboard_host"),
            dashboard_port=payload["dashboard_port"],
            authority=AuthorityBoundary.from_mapping(payload["authority"]),
            model=ModelConfig.from_mapping(payload["model"]),
            rgb=InputEndpoint.from_mapping(
                inputs["rgb"], name="rgb", allowed_backends=("disabled", "uvc_v4l2")
            ),
            depth=InputEndpoint.from_mapping(
                inputs["depth"],
                name="depth",
                allowed_backends=("disabled", "depth_v4l2", "vendor_sdk"),
            ),
            llm=LlmConfig.from_mapping(payload["llm"]),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "CapsuleConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(payload)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "project_root": self.project_root,
            "python_executable": self.python_executable,
            "dashboard_host": self.dashboard_host,
            "dashboard_port": self.dashboard_port,
            "authority": self.authority.to_dict(),
            "model": {
                "enabled": self.model.enabled,
                "path": self.model.path,
                "sha256": self.model.sha256,
                "provider": self.model.provider,
                "input_name": self.model.input_name,
                "output_name": self.model.output_name,
                "output_shape": list(self.model.output_shape),
                "class_order": list(self.model.class_order),
                "model_candidate": self.model.model_candidate,
                "model_qualified": self.model.model_qualified,
                "bpu_ready": self.model.bpu_ready,
                "preprocess": {
                    "mode": self.model.preprocess.mode,
                    "short_side": self.model.preprocess.short_side,
                    "center_crop": list(self.model.preprocess.center_crop),
                    "input_shape": list(self.model.preprocess.input_shape),
                    "color_order": self.model.preprocess.color_order,
                    "scale": self.model.preprocess.scale,
                    "mean": list(self.model.preprocess.mean),
                    "std": list(self.model.preprocess.std),
                    "interpolation": self.model.preprocess.interpolation,
                    "golden_generator": self.model.preprocess.golden_generator,
                    "golden_source_shape": list(
                        self.model.preprocess.golden_source_shape
                    ),
                    "golden_tensor_sha256": self.model.preprocess.golden_tensor_sha256,
                },
            },
            "inputs": {
                "rgb": self.rgb.__dict__,
                "depth": self.depth.__dict__,
            },
            "llm": self.llm.__dict__,
        }
