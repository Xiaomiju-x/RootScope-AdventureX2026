"""Relative before/after color-change evidence for the transparent sand fixture.

The result proves only a visible change in a frozen ROI. It does not infer soil
moisture, plant root depth, irrigation need, or field performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from .roi import PixelROI


@dataclass(frozen=True)
class WettingThresholds:
    pixel_delta_threshold: float = 18.0
    min_target_mean_delta: float = 10.0
    min_target_changed_fraction: float = 0.08
    max_neighbor_changed_fraction: float = 0.08
    min_selectivity_ratio: float = 1.8


@dataclass(frozen=True)
class WettingResult:
    passed: bool
    target_roi: str
    target_mean_delta: float
    target_changed_fraction: float
    max_neighbor_changed_fraction: float
    selectivity_ratio: float
    neighbor_changed_fractions: dict[str, float]
    reasons: tuple[str, ...]
    evidence_scope: str = "FIXED_FIXTURE_VISIBLE_CHANGE_ONLY"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def _rgb_float(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in {1, 3, 4}:
        raise ValueError(f"expected HxW, HxWx1, HxWx3 or HxWx4, got {array.shape}")
    values = array.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("frame contains NaN or infinity")
    if np.issubdtype(array.dtype, np.floating) and float(values.max(initial=0.0)) <= 1.5:
        values = values * 255.0
    if values.shape[2] == 1:
        values = np.repeat(values, 3, axis=2)
    return np.clip(values[:, :, :3], 0.0, 255.0)


def verify_wetting_change(
    baseline: np.ndarray,
    result: np.ndarray,
    target_roi: PixelROI,
    neighbor_rois: Iterable[PixelROI] = (),
    thresholds: WettingThresholds | None = None,
) -> WettingResult:
    """Compare frozen before/after frames and enforce target selectivity."""

    thresholds = thresholds or WettingThresholds()
    before = _rgb_float(baseline)
    after = _rgb_float(result)
    if before.shape != after.shape:
        raise ValueError(f"baseline/result shape mismatch: {before.shape} vs {after.shape}")
    height, width = before.shape[:2]
    target_roi.validate_for(width, height)
    neighbors = tuple(neighbor_rois)
    for roi in neighbors:
        roi.validate_for(width, height)

    delta = np.mean(np.abs(after - before), axis=2)
    target_values = delta[target_roi.slices()]
    target_mean = float(np.mean(target_values))
    target_fraction = float(np.mean(target_values >= thresholds.pixel_delta_threshold))

    neighbor_fractions: dict[str, float] = {}
    for roi in neighbors:
        values = delta[roi.slices()]
        neighbor_fractions[roi.name] = float(
            np.mean(values >= thresholds.pixel_delta_threshold)
        )
    max_neighbor = max(neighbor_fractions.values(), default=0.0)
    selectivity = target_fraction / max(max_neighbor, 1e-6)

    reasons: list[str] = []
    if target_mean < thresholds.min_target_mean_delta:
        reasons.append("TARGET_MEAN_CHANGE_TOO_SMALL")
    if target_fraction < thresholds.min_target_changed_fraction:
        reasons.append("TARGET_CHANGED_AREA_TOO_SMALL")
    if max_neighbor > thresholds.max_neighbor_changed_fraction:
        reasons.append("NON_TARGET_CHANGE_TOO_LARGE")
    if neighbors and selectivity < thresholds.min_selectivity_ratio:
        reasons.append("TARGET_NOT_SELECTIVE")

    return WettingResult(
        passed=not reasons,
        target_roi=target_roi.name,
        target_mean_delta=round(target_mean, 4),
        target_changed_fraction=round(target_fraction, 6),
        max_neighbor_changed_fraction=round(max_neighbor, 6),
        selectivity_ratio=round(selectivity, 4),
        neighbor_changed_fractions={
            key: round(value, 6) for key, value in sorted(neighbor_fractions.items())
        },
        reasons=tuple(reasons),
    )
