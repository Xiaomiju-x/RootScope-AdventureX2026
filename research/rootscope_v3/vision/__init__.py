"""RootSight-Delta PC/X5-portable visual evidence primitives."""

from .rootsight_delta import (
    OpticalOODResult,
    RegistrationResult,
    TemporalDecision,
    WettingDeltaConfig,
    WettingDeltaReceipt,
    evaluate_optical_ood,
    evaluate_wetting_delta,
    fuse_temporal_scores,
    register_translation,
)

__all__ = [
    "OpticalOODResult",
    "RegistrationResult",
    "TemporalDecision",
    "WettingDeltaConfig",
    "WettingDeltaReceipt",
    "evaluate_optical_ood",
    "evaluate_wetting_delta",
    "fuse_temporal_scores",
    "register_translation",
]
