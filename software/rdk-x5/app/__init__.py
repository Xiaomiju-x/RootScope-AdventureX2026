"""RootScope RDK X5 application core.

This package is intentionally independent from the XRD project runtime.  The
first implementation slice contains only deterministic contracts, the safety
state machine, and local evidence handling.
"""

from .config import ProfileConfig, RootScopeConfig
from .schemas import (
    ArmCommandContext,
    ClearEstopAckEvidence,
    ClearEstopCommandContext,
    CompletionClass,
    ExecutionMode,
    MachineState,
    PerceptionSource,
    StopCommandContext,
    TaskRequest,
    Zone,
)
from .state_machine import RootScopeStateMachine
from .release import (
    BASELINE_EMPTY,
    InstallAcceptanceReceipt,
    RollbackVerificationEvidence,
    ReleaseManifest,
    ReleaseProfile,
    ReleaseProfileContract,
    ServiceIdentitySnapshot,
)
from .runtime import (
    PhysicalActivationUnavailable,
    ProductionRuntime,
    ProductionRuntimeSnapshot,
    ProductionRuntimeState,
    RuntimePreflightFacts,
)

__all__ = [
    "CompletionClass",
    "ArmCommandContext",
    "ClearEstopAckEvidence",
    "ClearEstopCommandContext",
    "ExecutionMode",
    "MachineState",
    "PerceptionSource",
    "ProfileConfig",
    "RootScopeConfig",
    "RootScopeStateMachine",
    "StopCommandContext",
    "TaskRequest",
    "Zone",
    "BASELINE_EMPTY",
    "InstallAcceptanceReceipt",
    "RollbackVerificationEvidence",
    "PhysicalActivationUnavailable",
    "ProductionRuntime",
    "ProductionRuntimeSnapshot",
    "ProductionRuntimeState",
    "ReleaseManifest",
    "ReleaseProfile",
    "ReleaseProfileContract",
    "ServiceIdentitySnapshot",
    "RuntimePreflightFacts",
]
