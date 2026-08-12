"""RootSight-Delta deterministic visual evidence for the fixed irrigation bay.

This module intentionally has no actuator interface.  It produces observations
and HOLD reasons only.  Arrays are RGB in the 0..255 range.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class OpticalOODResult:
    hold: bool
    score: float
    reasons: tuple[str, ...]
    max_probability: float
    energy: float
    brightness: float
    contrast: float
    sharpness: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


@dataclass(frozen=True)
class TemporalDecision:
    label: str
    confidence: float
    agreement: float
    frames_used: int
    hold: bool
    reasons: tuple[str, ...]
    median_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


@dataclass(frozen=True)
class RegistrationResult:
    dx: int
    dy: int
    confidence: float
    valid_fraction: float
    method: str = "PHASE_CORRELATION_TRANSLATION"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WettingDeltaConfig:
    min_delta_e: float = 6.0
    min_value_drop: float = 7.0
    reference_noise_multiplier: float = 4.0
    min_component_pixels: int = 12
    min_registration_confidence: float = 4.0
    max_registration_shift_px: int = 48
    min_target_coverage: float = 0.08
    max_neighbor_spill: float = 0.10
    max_center_offset: float = 0.75
    min_mass_change_g: float = 0.5


@dataclass(frozen=True)
class WettingDeltaReceipt:
    status: str
    passed: bool
    target_coverage: float
    neighbor_spill: float
    center_offset: float
    wetting_pixels: int
    front_radius_px: float
    delta_e_threshold: float
    reference_noise_delta_e: float
    registration: RegistrationResult
    mass_delta_g: float | None
    mass_visual_consistency: str
    reasons: tuple[str, ...]
    evidence_scope: str = "FIXED_FIXTURE_VISIBLE_CHANGE_AND_MASS_CROSSCHECK_ONLY"
    physical_authority: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


def _rgb(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in {1, 3, 4}:
        raise ValueError(f"expected HxW RGB-like array, got {array.shape}")
    values = array.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("image contains NaN or infinity")
    if np.issubdtype(array.dtype, np.floating) and float(values.max(initial=0)) <= 1.5:
        values = values * 255.0
    if values.shape[2] == 1:
        values = np.repeat(values, 3, axis=2)
    return np.clip(values[:, :, :3], 0, 255)


def _gray(frame: np.ndarray) -> np.ndarray:
    rgb = _rgb(frame)
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _sharpness(gray: np.ndarray) -> float:
    if min(gray.shape) < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = (
        -4 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(lap))


def evaluate_optical_ood(
    frame: np.ndarray,
    logits: Sequence[float],
    *,
    min_max_probability: float = 0.58,
    min_brightness: float = 25.0,
    max_brightness: float = 232.0,
    min_contrast: float = 9.0,
    min_sharpness: float = 12.0,
) -> OpticalOODResult:
    """Fuse model uncertainty and optics health into a fail-closed HOLD flag."""

    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 2 or not np.isfinite(scores).all():
        raise ValueError("logits must be a finite one-dimensional vector")
    shifted = scores - float(np.max(scores))
    probability = np.exp(shifted)
    probability /= float(np.sum(probability))
    max_probability = float(np.max(probability))
    energy = float(-np.log(np.sum(np.exp(np.clip(scores, -50, 50)))))
    gray = _gray(frame)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = _sharpness(gray)

    reasons: list[str] = []
    if max_probability < min_max_probability:
        reasons.append("LOW_MODEL_CONFIDENCE")
    if not min_brightness <= brightness <= max_brightness:
        reasons.append("EXPOSURE_OUT_OF_RANGE")
    if contrast < min_contrast:
        reasons.append("LOW_CONTRAST")
    if sharpness < min_sharpness:
        reasons.append("BLUR_OR_FLAT_SCENE")
    optical_penalty = (
        max(0.0, min_brightness - brightness) / max(min_brightness, 1)
        + max(0.0, brightness - max_brightness) / max(255 - max_brightness, 1)
        + max(0.0, min_contrast - contrast) / max(min_contrast, 1)
        + max(0.0, min_sharpness - sharpness) / max(min_sharpness, 1)
    )
    ood_score = (1.0 - max_probability) + 0.25 * optical_penalty
    return OpticalOODResult(
        hold=bool(reasons),
        score=round(float(ood_score), 6),
        reasons=tuple(reasons),
        max_probability=round(max_probability, 6),
        energy=round(energy, 6),
        brightness=round(brightness, 4),
        contrast=round(contrast, 4),
        sharpness=round(sharpness, 4),
    )


def fuse_temporal_scores(
    observations: Sequence[Mapping[str, float]],
    *,
    ood_holds: Sequence[bool] | None = None,
    min_frames: int = 3,
    min_confidence: float = 0.60,
    min_agreement: float = 0.75,
) -> TemporalDecision:
    """Median-fuse frame scores and require stable frame-level top-1 agreement."""

    if len(observations) < min_frames:
        return TemporalDecision(
            label="unknown",
            confidence=0.0,
            agreement=0.0,
            frames_used=len(observations),
            hold=True,
            reasons=("INSUFFICIENT_FRAMES",),
            median_scores={},
        )
    labels = tuple(sorted(observations[0]))
    if not labels or any(tuple(sorted(row)) != labels for row in observations):
        raise ValueError("every observation must have the same non-empty label set")
    matrix = np.asarray([[float(row[label]) for label in labels] for row in observations])
    if not np.isfinite(matrix).all():
        raise ValueError("temporal scores contain NaN or infinity")
    medians = np.median(matrix, axis=0)
    winner_index = int(np.argmax(medians))
    winner = labels[winner_index]
    confidence = float(medians[winner_index])
    frame_winners = np.argmax(matrix, axis=1)
    agreement = float(np.mean(frame_winners == winner_index))
    holds = tuple(ood_holds or (False,) * len(observations))
    if len(holds) != len(observations):
        raise ValueError("ood_holds length must match observations")
    reasons: list[str] = []
    if confidence < min_confidence:
        reasons.append("LOW_TEMPORAL_CONFIDENCE")
    if agreement < min_agreement:
        reasons.append("TEMPORAL_DISAGREEMENT")
    if sum(bool(value) for value in holds) > len(holds) // 2:
        reasons.append("OOD_MAJORITY")
    return TemporalDecision(
        label="unknown" if reasons else winner,
        confidence=round(confidence, 6),
        agreement=round(agreement, 6),
        frames_used=len(observations),
        hold=bool(reasons),
        reasons=tuple(reasons),
        median_scores={label: round(float(medians[i]), 6) for i, label in enumerate(labels)},
    )


def _shift(frame: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    output = np.zeros_like(frame)
    valid = np.zeros((height, width), dtype=bool)
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    dx0, dx1 = sx0 + dx, sx1 + dx
    dy0, dy1 = sy0 + dy, sy1 + dy
    if sx1 > sx0 and sy1 > sy0:
        output[dy0:dy1, dx0:dx1] = frame[sy0:sy1, sx0:sx1]
        valid[dy0:dy1, dx0:dx1] = True
    return output, valid


def register_translation(
    before: np.ndarray,
    after: np.ndarray,
    *,
    max_shift_px: int = 48,
) -> tuple[np.ndarray, np.ndarray, RegistrationResult]:
    """Align ``after`` to ``before`` with phase correlation and no wraparound."""

    reference = _rgb(before)
    moving = _rgb(after)
    if reference.shape != moving.shape:
        raise ValueError("before/after shape mismatch")
    gray_a, gray_b = _gray(reference), _gray(moving)
    scale = max(1, int(np.ceil(max(gray_a.shape) / 384)))
    a, b = gray_a[::scale, ::scale], gray_b[::scale, ::scale]
    a, b = a - np.mean(a), b - np.mean(b)
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2(a * window), np.fft.fft2(b * window)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-9)
    corr = np.abs(np.fft.ifft2(cross))
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    if py > corr.shape[0] // 2:
        py -= corr.shape[0]
    if px > corr.shape[1] // 2:
        px -= corr.shape[1]
    dx, dy = int(px * scale), int(py * scale)
    peak = float(np.max(corr))
    confidence = peak / max(float(np.mean(corr)), 1e-9)
    if abs(dx) > max_shift_px or abs(dy) > max_shift_px:
        dx = dy = 0
        confidence = 0.0
    aligned, valid = _shift(moving, dx, dy)
    result = RegistrationResult(
        dx=dx,
        dy=dy,
        confidence=round(confidence, 4),
        valid_fraction=round(float(np.mean(valid)), 6),
    )
    return aligned, valid, result


def _roi_mask(shape: tuple[int, int], roi: tuple[int, int, int, int]) -> np.ndarray:
    height, width = shape
    x, y, w, h = (int(value) for value in roi)
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError(f"ROI {roi} outside {width}x{height}")
    mask = np.zeros(shape, dtype=bool)
    mask[y : y + h, x : x + w] = True
    return mask


def _reference_correct(frame: np.ndarray, reference_mask: np.ndarray | None) -> np.ndarray:
    rgb = _rgb(frame)
    if reference_mask is None:
        return rgb
    pixels = rgb[reference_mask]
    if pixels.size == 0:
        raise ValueError("reference ROI is empty")
    channel_medians = np.median(pixels, axis=0)
    neutral = float(np.mean(channel_medians))
    gains = np.clip(neutral / np.maximum(channel_medians, 1.0), 0.65, 1.55)
    return np.clip(rgb * gains[None, None, :], 0, 255)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    values = _rgb(rgb) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.asarray(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    ).T
    xyz /= np.asarray([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6 / 29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])), axis=-1)


def _binary_filter(mask: np.ndarray, *, minimum_neighbors: int) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1)
    counts = sum(
        padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        for dy in range(3)
        for dx in range(3)
    )
    return counts >= minimum_neighbors


def _remove_small_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    if minimum <= 1:
        return mask
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    keep = np.zeros_like(mask, dtype=bool)
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        component: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(component) >= minimum:
            yy, xx = zip(*component)
            keep[np.asarray(yy), np.asarray(xx)] = True
    return keep


def evaluate_wetting_delta(
    before_frames: Sequence[np.ndarray],
    after_frames: Sequence[np.ndarray],
    *,
    target_roi: tuple[int, int, int, int],
    neighbor_rois: Sequence[tuple[int, int, int, int]] = (),
    reference_roi: tuple[int, int, int, int] | None = None,
    mass_delta_g: float | None = None,
    config: WettingDeltaConfig | None = None,
) -> WettingDeltaReceipt:
    """Register and compare multi-frame before/after observations.

    Positive ``mass_delta_g`` means measured delivered water/fixture gain.
    """

    cfg = config or WettingDeltaConfig()
    if not before_frames or not after_frames:
        raise ValueError("before_frames and after_frames must be non-empty")
    before = np.median(np.stack([_rgb(frame) for frame in before_frames]), axis=0)
    after = np.median(np.stack([_rgb(frame) for frame in after_frames]), axis=0)
    if before.shape != after.shape:
        raise ValueError("before/after shape mismatch")
    target = _roi_mask(before.shape[:2], target_roi)
    neighbors = [_roi_mask(before.shape[:2], roi) for roi in neighbor_rois]
    reference = _roi_mask(before.shape[:2], reference_roi) if reference_roi else None
    before = _reference_correct(before, reference)
    after = _reference_correct(after, reference)
    aligned, valid, registration = register_translation(
        before, after, max_shift_px=cfg.max_registration_shift_px
    )
    lab_before, lab_after = _rgb_to_lab(before), _rgb_to_lab(aligned)
    delta_e = np.linalg.norm(lab_after - lab_before, axis=2)
    value_drop = np.mean(before, axis=2) - np.mean(aligned, axis=2)
    reference_values = delta_e[reference & valid] if reference is not None else delta_e[valid & ~target]
    reference_noise = float(np.median(reference_values)) if reference_values.size else 0.0
    reference_mad = float(np.median(np.abs(reference_values - reference_noise))) if reference_values.size else 0.0
    threshold = max(cfg.min_delta_e, reference_noise + cfg.reference_noise_multiplier * max(reference_mad, 0.25))
    wet = valid & (delta_e >= threshold) & (
        (value_drop >= cfg.min_value_drop) | (delta_e >= threshold * 1.5)
    )
    if reference is not None:
        wet &= ~reference
    wet = _binary_filter(_binary_filter(wet, minimum_neighbors=3), minimum_neighbors=5)
    wet = _remove_small_components(wet, cfg.min_component_pixels)

    target_pixels = int(np.sum(target & valid))
    coverage = float(np.sum(wet & target) / max(target_pixels, 1))
    neighbor_pixels = int(sum(np.sum(mask & valid) for mask in neighbors))
    spill = float(sum(np.sum(wet & mask) for mask in neighbors) / max(neighbor_pixels, 1))
    yx = np.argwhere(wet & target)
    x, y, w, h = target_roi
    center = np.asarray([y + (h - 1) / 2, x + (w - 1) / 2])
    diagonal = max(float(np.hypot(w, h)), 1.0)
    if yx.size:
        centroid = np.mean(yx, axis=0)
        offset = float(np.linalg.norm(centroid - center) / diagonal)
        radius = float(np.percentile(np.linalg.norm(yx - center, axis=1), 95))
    else:
        offset, radius = 1.0, 0.0

    visual = coverage >= cfg.min_target_coverage
    mass = None if mass_delta_g is None else mass_delta_g >= cfg.min_mass_change_g
    if mass is None:
        consistency = "MASS_NOT_AVAILABLE"
    elif mass and visual:
        consistency = "CONSISTENT"
    elif mass and not visual:
        consistency = "MASS_ONLY_REVIEW_OPTICS_OR_LEAK"
    elif not mass and visual:
        consistency = "VISION_ONLY_REVIEW_SCALE_OR_LIGHT"
    else:
        consistency = "NO_CHANGE"

    reasons: list[str] = []
    if registration.confidence < cfg.min_registration_confidence:
        reasons.append("REGISTRATION_UNCERTAIN")
    if coverage < cfg.min_target_coverage:
        reasons.append("TARGET_COVERAGE_TOO_SMALL")
    if spill > cfg.max_neighbor_spill:
        reasons.append("NEIGHBOR_SPILL_TOO_LARGE")
    if offset > cfg.max_center_offset:
        reasons.append("WETTING_CENTER_OFFSET_TOO_LARGE")
    if mass is not None and mass != visual:
        reasons.append("MASS_VISUAL_CONFLICT")
    return WettingDeltaReceipt(
        status="HOLD" if reasons else "OBSERVED",
        passed=not reasons,
        target_coverage=round(coverage, 6),
        neighbor_spill=round(spill, 6),
        center_offset=round(offset, 6),
        wetting_pixels=int(np.sum(wet)),
        front_radius_px=round(radius, 4),
        delta_e_threshold=round(float(threshold), 4),
        reference_noise_delta_e=round(reference_noise, 4),
        registration=registration,
        mass_delta_g=None if mass_delta_g is None else round(float(mass_delta_g), 4),
        mass_visual_consistency=consistency,
        reasons=tuple(reasons),
    )
