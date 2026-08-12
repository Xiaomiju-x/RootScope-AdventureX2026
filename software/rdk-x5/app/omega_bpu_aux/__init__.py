"""Zero-authority generic BPU evidence probe for RootScope-Ω.

The package is intentionally separate from the RootScope classifier and safety
runtime.  Importing it does not import ``hobot_dnn`` or touch any device.
"""

from .probe import (
    BpuAuxProbeError,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    VENDOR_MODEL_PATH,
    run_manifest_probe,
)

__all__ = [
    "BpuAuxProbeError",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "VENDOR_MODEL_PATH",
    "run_manifest_probe",
]
