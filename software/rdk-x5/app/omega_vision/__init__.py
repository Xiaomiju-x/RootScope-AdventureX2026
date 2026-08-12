"""Fail-closed experimental vision/OOD gates for RootScope Omega."""

from .ood import (
    Calibration,
    Decision,
    QualityMetrics,
    calibrate,
    decide,
    energy_score,
    evaluate_quality,
)

__all__ = [
    "Calibration",
    "Decision",
    "QualityMetrics",
    "calibrate",
    "decide",
    "energy_score",
    "evaluate_quality",
]
