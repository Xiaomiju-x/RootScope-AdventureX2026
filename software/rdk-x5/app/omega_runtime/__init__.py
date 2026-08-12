"""RootScope-Ω deterministic orchestration and truth-projection runtime."""

from .contracts import (
    AuthorityFlags,
    BackendCapsule,
    DecisionProjection,
    DecisionReceipt,
    EvidenceAction,
    RuntimeMode,
    SafetyDecision,
    TruthRibbon,
    canonical_sha256,
)
from .digital_twin import TwinCaseInput, TwinEvaluation, evaluate_case
from .loopback_llm_cluster import (
    LoopbackLlamaConfig,
    LoopbackLlamaModel,
    run_loopback_role_cluster,
)
from .profiles import EdgeProfile, EdgeProfileRegistry, ResourceSnapshot

__all__ = [
    "AuthorityFlags",
    "BackendCapsule",
    "DecisionProjection",
    "DecisionReceipt",
    "EdgeProfile",
    "EdgeProfileRegistry",
    "EvidenceAction",
    "LoopbackLlamaConfig",
    "LoopbackLlamaModel",
    "ResourceSnapshot",
    "RuntimeMode",
    "SafetyDecision",
    "TruthRibbon",
    "TwinCaseInput",
    "TwinEvaluation",
    "canonical_sha256",
    "evaluate_case",
    "run_loopback_role_cluster",
]
