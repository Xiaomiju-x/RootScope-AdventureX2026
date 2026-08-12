"""Zero-authority clean-X5 deployment helpers for RootScope.

The package is intentionally independent from XRD runtime modules.  Importing
it performs no device enumeration, camera open, serial open, network request,
or BPU initialization.
"""

from .capsule import (
    AUTHORITY_FIELDS,
    CAPSULE_SCHEMA_VERSION,
    GOLDEN_GENERATOR,
    PREPROCESS_MODE,
    ROOTSCOPE_CLASS_ORDER,
    AuthorityBoundary,
    CapsuleConfig,
    InputEndpoint,
    LlmConfig,
    ModelConfig,
    PreprocessConfig,
)
from .onnx_cpu import (
    CpuOnnxRunner,
    OnnxCpuContractError,
    make_simulated_rgb,
    preprocess_rgb,
)

__all__ = [
    "AUTHORITY_FIELDS",
    "CAPSULE_SCHEMA_VERSION",
    "GOLDEN_GENERATOR",
    "PREPROCESS_MODE",
    "ROOTSCOPE_CLASS_ORDER",
    "AuthorityBoundary",
    "CapsuleConfig",
    "CpuOnnxRunner",
    "InputEndpoint",
    "LlmConfig",
    "ModelConfig",
    "OnnxCpuContractError",
    "PreprocessConfig",
    "make_simulated_rgb",
    "preprocess_rgb",
]
