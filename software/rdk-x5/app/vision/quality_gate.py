"""Deterministic image-quality admission for the fixed RootScope camera."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    min_width: int = 320
    min_height: int = 240
    min_mean_luma: float = 28.0
    max_mean_luma: float = 228.0
    max_dark_fraction: float = 0.35
    max_bright_fraction: float = 0.35
    min_contrast_std: float = 10.0
    min_sharpness_variance: float = 18.0
    clip_low: float = 5.0
    clip_high: float = 250.0


@dataclass(frozen=True)
class FrameQualityResult:
    passed: bool
    width: int
    height: int
    mean_luma: float
    contrast_std: float
    sharpness_variance: float
    dark_fraction: float
    bright_fraction: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def _as_luma(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim not in {2, 3}:
        raise ValueError(f"frame must be HxW or HxWxC, got shape {array.shape}")
    if array.ndim == 3 and array.shape[2] not in {1, 3, 4}:
        raise ValueError(f"unsupported channel count: {array.shape[2]}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("frame must not be empty")
    values = array.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("frame contains NaN or infinity")
    if np.issubdtype(array.dtype, np.floating) and float(values.max(initial=0.0)) <= 1.5:
        values = values * 255.0
    values = np.clip(values, 0.0, 255.0)
    if values.ndim == 2:
        return values
    if values.shape[2] == 1:
        return values[:, :, 0]
    rgb = values[:, :, :3]
    return rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722


def evaluate_frame_quality(
    frame: np.ndarray,
    thresholds: QualityThresholds | None = None,
) -> FrameQualityResult:
    """Evaluate exposure, contrast and focus without opening a camera device."""

    thresholds = thresholds or QualityThresholds()
    luma = _as_luma(frame)
    height, width = luma.shape
    mean_luma = float(np.mean(luma))
    contrast_std = float(np.std(luma))
    dark_fraction = float(np.mean(luma <= thresholds.clip_low))
    bright_fraction = float(np.mean(luma >= thresholds.clip_high))

    if height >= 3 and width >= 3:
        center = luma[1:-1, 1:-1]
        laplacian = (
            -4.0 * center
            + luma[:-2, 1:-1]
            + luma[2:, 1:-1]
            + luma[1:-1, :-2]
            + luma[1:-1, 2:]
        )
        sharpness_variance = float(np.var(laplacian))
    else:
        sharpness_variance = 0.0

    reasons: list[str] = []
    if width < thresholds.min_width or height < thresholds.min_height:
        reasons.append("FRAME_TOO_SMALL")
    if mean_luma < thresholds.min_mean_luma:
        reasons.append("UNDEREXPOSED")
    if mean_luma > thresholds.max_mean_luma:
        reasons.append("OVEREXPOSED")
    if dark_fraction > thresholds.max_dark_fraction:
        reasons.append("DARK_CLIPPING")
    if bright_fraction > thresholds.max_bright_fraction:
        reasons.append("BRIGHT_CLIPPING")
    if contrast_std < thresholds.min_contrast_std:
        reasons.append("LOW_CONTRAST")
    if sharpness_variance < thresholds.min_sharpness_variance:
        reasons.append("BLUR_OR_FLAT_SCENE")

    return FrameQualityResult(
        passed=not reasons,
        width=width,
        height=height,
        mean_luma=round(mean_luma, 4),
        contrast_std=round(contrast_std, 4),
        sharpness_variance=round(sharpness_variance, 4),
        dark_fraction=round(dark_fraction, 6),
        bright_fraction=round(bright_fraction, 6),
        reasons=tuple(reasons),
    )
