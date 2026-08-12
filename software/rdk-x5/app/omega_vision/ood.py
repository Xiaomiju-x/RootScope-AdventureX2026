"""Deterministic, zero-authority OOD and abstention logic.

The module intentionally accepts final classifier logits only.  The frozen
seed17 ONNX graph does not expose a penultimate embedding, so a Mahalanobis
detector cannot be fitted honestly and is explicitly excluded from this
implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np


class VisionGateError(ValueError):
    """The caller violated the fail-closed vision contract."""


def _finite_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise VisionGateError(f"{name} must be a non-empty finite vector")
    return array


def _finite_matrix(values: Sequence[Sequence[float]], *, columns: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or array.shape[1] != columns
        or not np.isfinite(array).all()
    ):
        raise VisionGateError(f"logits must be finite [N,{columns}]")
    return array


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def energy_score(logits: Iterable[float], *, temperature: float = 1.0) -> float:
    """Return ``-T logsumexp(logits/T)``; larger values are more OOD-like."""

    vector = _finite_vector(logits, name="logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise VisionGateError("temperature must be finite and positive")
    scaled = vector / temperature
    maximum = float(np.max(scaled))
    return float(-temperature * (maximum + math.log(float(np.exp(scaled - maximum).sum()))))


def finite_upper_quantile(values: Iterable[float], *, alpha: float) -> float:
    """Distribution-free upper threshold using the finite-sample rank."""

    vector = np.sort(_finite_vector(values, name="calibration values"))
    if not 0 < alpha < 1:
        raise VisionGateError("alpha must be in (0,1)")
    rank = min(vector.size, max(1, math.ceil((vector.size + 1) * (1.0 - alpha))))
    return float(vector[rank - 1])


def finite_lower_quantile(values: Iterable[float], *, tail: float) -> float:
    """Distribution-free lower threshold using a conservative finite rank."""

    vector = np.sort(_finite_vector(values, name="calibration values"))
    if not 0 < tail < 1:
        raise VisionGateError("tail must be in (0,1)")
    rank = min(vector.size, max(1, math.floor((vector.size + 1) * tail)))
    return float(vector[rank - 1])


@dataclass(frozen=True)
class QualityMetrics:
    brightness: float
    contrast: float
    sharpness: float
    clipped_fraction: float

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(value) for value in values.values()):
            raise VisionGateError("quality metrics must be finite")
        if not 0.0 <= self.brightness <= 1.0:
            raise VisionGateError("brightness must be in [0,1]")
        if min(self.contrast, self.sharpness, self.clipped_fraction) < 0.0:
            raise VisionGateError("quality magnitudes must be non-negative")
        if self.clipped_fraction > 1.0:
            raise VisionGateError("clipped_fraction must be in [0,1]")


def evaluate_quality(rgb: np.ndarray) -> QualityMetrics:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise VisionGateError("quality input must be uint8 RGB HxWx3")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise VisionGateError("quality input must be at least 2x2")
    floating = array.astype(np.float64)
    luma = 0.2126 * floating[..., 0] + 0.7152 * floating[..., 1] + 0.0722 * floating[..., 2]
    dx = np.abs(np.diff(luma, axis=1)).mean()
    dy = np.abs(np.diff(luma, axis=0)).mean()
    return QualityMetrics(
        brightness=float(luma.mean() / 255.0),
        contrast=float(luma.std() / 255.0),
        sharpness=float((dx + dy) / (2.0 * 255.0)),
        clipped_fraction=float(np.mean((luma <= 3.0) | (luma >= 252.0))),
    )


@dataclass(frozen=True)
class Calibration:
    class_order: tuple[str, ...]
    alpha: float
    temperature: float
    energy_upper: float
    maxprob_lower: float
    brightness_lower: float
    brightness_upper: float
    contrast_lower: float
    sharpness_lower: float
    clipped_upper: float
    conformal_nonconformity: tuple[float, ...]
    calibration_roles: tuple[str, ...] = (
        "EXPERIMENTAL_TRAIN_SUGGESTION",
        "EXPERIMENTAL_VAL_SUGGESTION",
    )
    mahalanobis_status: str = "SKIPPED_NO_VALID_EMBEDDING_OUTPUT"
    status: str = "MACHINE_CURATED_EXPERIMENTAL_NOT_QUALIFIED"

    def __post_init__(self) -> None:
        if len(self.class_order) < 2 or len(set(self.class_order)) != len(self.class_order):
            raise VisionGateError("class_order must contain distinct classes")
        if not 0 < self.alpha < 1:
            raise VisionGateError("alpha must be in (0,1)")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise VisionGateError("temperature must be positive")
        _finite_vector(self.conformal_nonconformity, name="conformal_nonconformity")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calibrate(
    *,
    reference_logits: Sequence[Sequence[float]],
    reference_quality: Sequence[QualityMetrics],
    validation_logits: Sequence[Sequence[float]],
    validation_labels: Sequence[int],
    class_order: Sequence[str],
    alpha: float = 0.20,
    temperature: float = 1.0,
) -> Calibration:
    """Fit gates using train/validation suggestions only.

    ``reference_*`` is train+validation and supplies Energy/max-probability/
    quality limits.  ``validation_*`` alone supplies pooled marginal
    split-conformal nonconformity scores.  Because this experimental
    validation partition was also used by the pre-existing training run for
    checkpoint selection, these scores are a conservative abstention heuristic
    and do not carry a formal distribution-free coverage guarantee.
    """

    classes = tuple(class_order)
    reference = _finite_matrix(reference_logits, columns=len(classes))
    validation = _finite_matrix(validation_logits, columns=len(classes))
    labels = np.asarray(validation_labels, dtype=np.int64)
    if labels.shape != (validation.shape[0],) or np.any(labels < 0) or np.any(labels >= len(classes)):
        raise VisionGateError("validation_labels must be valid class indices")
    if len(reference_quality) != reference.shape[0]:
        raise VisionGateError("reference quality/logit counts differ")
    probabilities = _softmax(reference / temperature)
    validation_probabilities = _softmax(validation / temperature)
    energies = [energy_score(row, temperature=temperature) for row in reference]
    maximum_probabilities = np.max(probabilities, axis=1)
    true_probabilities = validation_probabilities[np.arange(labels.size), labels]
    brightness = [item.brightness for item in reference_quality]
    contrast = [item.contrast for item in reference_quality]
    sharpness = [item.sharpness for item in reference_quality]
    clipped = [item.clipped_fraction for item in reference_quality]
    return Calibration(
        class_order=classes,
        alpha=float(alpha),
        temperature=float(temperature),
        energy_upper=finite_upper_quantile(energies, alpha=0.05),
        maxprob_lower=finite_lower_quantile(maximum_probabilities, tail=0.05),
        brightness_lower=finite_lower_quantile(brightness, tail=0.02),
        brightness_upper=finite_upper_quantile(brightness, alpha=0.02),
        contrast_lower=finite_lower_quantile(contrast, tail=0.05),
        sharpness_lower=finite_lower_quantile(sharpness, tail=0.05),
        clipped_upper=finite_upper_quantile(clipped, alpha=0.05),
        conformal_nonconformity=tuple(float(1.0 - value) for value in true_probabilities),
    )


@dataclass(frozen=True)
class Decision:
    decision: str
    predicted_class: str | None
    raw_top1_class: str
    reasons: tuple[str, ...]
    energy: float
    max_probability: float
    conformal_set: tuple[str, ...]
    conformal_p_values: tuple[float, ...]
    quality: QualityMetrics
    zero_authority: bool = True
    model_qualified: bool = False
    physical_authority: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def decide(
    logits: Sequence[float],
    quality: QualityMetrics,
    calibration: Calibration,
    *,
    unknown_class: str = "unknown",
) -> Decision:
    vector = _finite_vector(logits, name="logits")
    if vector.size != len(calibration.class_order):
        raise VisionGateError("logit dimension does not match class_order")
    probabilities = _softmax((vector / calibration.temperature)[None, :])[0]
    top1 = int(np.argmax(probabilities))
    energy = energy_score(vector, temperature=calibration.temperature)
    calibration_scores = np.asarray(calibration.conformal_nonconformity, dtype=np.float64)
    p_values: list[float] = []
    conformal: list[str] = []
    for index, probability in enumerate(probabilities):
        score = 1.0 - float(probability)
        p_value = float((1 + np.count_nonzero(calibration_scores >= score)) / (calibration_scores.size + 1))
        p_values.append(p_value)
        if p_value > calibration.alpha:
            conformal.append(calibration.class_order[index])

    reasons: list[str] = []
    if energy > calibration.energy_upper:
        reasons.append("ENERGY_OOD")
    if float(probabilities[top1]) < calibration.maxprob_lower:
        reasons.append("LOW_MAX_PROBABILITY")
    if quality.brightness < calibration.brightness_lower:
        reasons.append("QUALITY_TOO_DARK")
    if quality.brightness > calibration.brightness_upper:
        reasons.append("QUALITY_TOO_BRIGHT")
    if quality.contrast < calibration.contrast_lower:
        reasons.append("QUALITY_LOW_CONTRAST")
    if quality.sharpness < calibration.sharpness_lower:
        reasons.append("QUALITY_LOW_SHARPNESS")
    if quality.clipped_fraction > calibration.clipped_upper:
        reasons.append("QUALITY_CLIPPED")
    if len(conformal) != 1:
        reasons.append("CONFORMAL_SET_NOT_SINGLETON")
    elif conformal[0] != calibration.class_order[top1]:
        reasons.append("CONFORMAL_TOP1_MISMATCH")
    if calibration.class_order[top1] == unknown_class:
        reasons.append("UNKNOWN_CLASS")

    accepted = not reasons
    return Decision(
        decision="CLASSIFY" if accepted else "ABSTAIN",
        predicted_class=calibration.class_order[top1] if accepted else None,
        raw_top1_class=calibration.class_order[top1],
        reasons=tuple(reasons),
        energy=float(energy),
        max_probability=float(probabilities[top1]),
        conformal_set=tuple(conformal),
        conformal_p_values=tuple(p_values),
        quality=quality,
    )
