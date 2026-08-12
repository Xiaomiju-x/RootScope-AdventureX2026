"""Evidence-only geometric verification for a registered printed card.

The matcher answers a deliberately narrow question: does a query image contain
an instance whose local features and projective geometry agree with one exact,
registered template?  It does *not* perform plant semantic recognition, and its
output never grants irrigation, pump, serial, or state-machine authority.

AKAZE is attempted first by default.  ORB is an explicit fail-closed fallback.
Both use Hamming KNN matching, Lowe-style ratio filtering in both directions,
mutual consistency, and a RANSAC homography followed by geometry gates.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


SCHEMA_VERSION = "rootscope.known_card_geometric_match.v1"
ALGORITHM_VERSION = "rootscope-known-card-geometry/1.0.0"
CLAIM_SCOPE = "KNOWN_TEMPLATE_INSTANCE_GEOMETRY_ONLY_NOT_SEMANTIC_RECOGNITION"


@dataclass(frozen=True)
class MatcherConfig:
    """Every detector and acceptance threshold used by the matcher."""

    detector_preference: str = "AKAZE"
    allow_orb_fallback: bool = True
    knn_ratio_threshold: float = 0.78
    min_template_keypoints: int = 35
    min_query_keypoints: int = 35
    min_mutual_good_matches: int = 18
    min_inliers: int = 14
    min_inlier_ratio: float = 0.55
    max_median_reprojection_error_px: float = 3.5
    ransac_reprojection_threshold_px: float = 3.0
    ransac_confidence: float = 0.995
    ransac_max_iters: int = 4000
    min_projected_area_ratio: float = 0.08
    max_projected_area_ratio: float = 0.90
    projected_boundary_margin_px: float = 4.0
    max_image_pixels: int = 20_000_000
    akaze_threshold: float = 0.001
    orb_max_features: int = 2500
    orb_fast_threshold: int = 12
    rng_seed: int = 20260717

    def __post_init__(self) -> None:
        preference = self.detector_preference.upper()
        object.__setattr__(self, "detector_preference", preference)
        if preference not in {"AKAZE", "ORB"}:
            raise ValueError("detector_preference must be AKAZE or ORB")
        if not 0.0 < self.knn_ratio_threshold < 1.0:
            raise ValueError("knn_ratio_threshold must be in (0, 1)")
        for name in (
            "min_template_keypoints",
            "min_query_keypoints",
            "min_mutual_good_matches",
            "min_inliers",
            "ransac_max_iters",
            "max_image_pixels",
            "orb_max_features",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_mutual_good_matches < 4:
            raise ValueError("min_mutual_good_matches must be at least 4")
        if self.min_inliers < 4:
            raise ValueError("min_inliers must be at least 4")
        if not 0.0 <= self.min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be in [0, 1]")
        if self.max_median_reprojection_error_px <= 0.0:
            raise ValueError("max_median_reprojection_error_px must be positive")
        if self.ransac_reprojection_threshold_px <= 0.0:
            raise ValueError("ransac_reprojection_threshold_px must be positive")
        if not 0.0 < self.ransac_confidence < 1.0:
            raise ValueError("ransac_confidence must be in (0, 1)")
        if not 0.0 < self.min_projected_area_ratio < self.max_projected_area_ratio:
            raise ValueError("projected area ratio bounds must satisfy 0 < min < max")
        if self.max_projected_area_ratio > 1.0:
            raise ValueError("max_projected_area_ratio must be <= 1")
        if self.projected_boundary_margin_px < 0.0:
            raise ValueError("projected_boundary_margin_px must be non-negative")
        if self.akaze_threshold <= 0.0:
            raise ValueError("akaze_threshold must be positive")
        if self.orb_fast_threshold < 0:
            raise ValueError("orb_fast_threshold must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MatcherConfig":
        known = {item.name for item in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown matcher config fields: {', '.join(unknown)}")
        return cls(**dict(values))


@dataclass(frozen=True)
class GeometricMatchResult:
    schema: str
    status: str
    passed: bool
    claim_scope: str
    irrigation_execution_authority: bool
    template_sha256: str
    query_sha256: str
    template_id: str
    template_class: str
    detector: Mapping[str, Any]
    metrics: Mapping[str, Any]
    gates: Mapping[str, Any]
    reject_reasons: tuple[str, ...]
    config: Mapping[str, Any]
    provenance: Mapping[str, Any]
    authority: Mapping[str, bool] = field(
        default_factory=lambda: {
            "irrigation_execution": False,
            "pump_command": False,
            "serial_write": False,
            "state_machine_write": False,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class _LoadedImage:
    image: np.ndarray
    gray: np.ndarray
    sha256: str
    provenance: Mapping[str, Any]


@dataclass
class _DetectorEvaluation:
    name: str
    template_keypoints: int
    query_keypoints: int
    forward_ratio_good: int
    reverse_ratio_good: int
    mutual_good_matches: int
    homography: np.ndarray | None
    inliers: int
    inlier_ratio: float
    median_reprojection_error_px: float | None
    projected_quadrilateral: list[list[float]] | None
    projected_area_ratio: float | None
    projected_convex: bool
    projected_within_bounds: bool
    gates: dict[str, Any]
    reject_reasons: list[str]

    @property
    def passed(self) -> bool:
        return not self.reject_reasons

    def summary(self) -> dict[str, Any]:
        return {
            "detector": self.name,
            "passed": self.passed,
            "template_keypoints": self.template_keypoints,
            "query_keypoints": self.query_keypoints,
            "mutual_good_matches": self.mutual_good_matches,
            "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio,
            "reject_reasons": list(self.reject_reasons),
        }


ImageInput = str | Path | np.ndarray


def _canonical_pixel_digest(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim not in {2, 3}:
        raise ValueError("image must be a 2D grayscale or 3D color array")
    if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
        raise ValueError("color image must have 1, 3, or 4 channels")
    if image.size == 0 or image.shape[0] < 16 or image.shape[1] < 16:
        raise ValueError("image must be non-empty and at least 16x16 pixels")
    if image.dtype == np.uint8:
        converted = image
    elif np.issubdtype(image.dtype, np.number):
        converted = np.clip(image, 0, 255).astype(np.uint8)
    else:
        raise ValueError("image dtype must be numeric")
    if converted.ndim == 2:
        gray = converted
    elif converted.shape[2] == 1:
        gray = converted[:, :, 0]
    elif converted.shape[2] == 3:
        gray = cv2.cvtColor(converted, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(converted, cv2.COLOR_BGRA2GRAY)
    return np.ascontiguousarray(gray)


def _load_image(source: ImageInput, *, role: str, max_pixels: int) -> _LoadedImage:
    if isinstance(source, np.ndarray):
        image = np.asarray(source)
        source_sha = _canonical_pixel_digest(image)
        source_provenance: dict[str, Any] = {
            "source_kind": "ndarray",
            "sha256": source_sha,
            "sha256_scope": "dtype_shape_and_contiguous_array_bytes",
        }
    else:
        path = Path(source).expanduser().resolve()
        raw = path.read_bytes()
        source_sha = hashlib.sha256(raw).hexdigest()
        encoded = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"{role} is not a decodable image: {path}")
        source_provenance = {
            "source_kind": "file",
            "path": str(path),
            "size_bytes": len(raw),
            "sha256": source_sha,
            "sha256_scope": "raw_file_bytes",
        }
    pixels = int(image.shape[0]) * int(image.shape[1])
    if pixels > max_pixels:
        raise ValueError(f"{role} image has {pixels} pixels, above max_image_pixels={max_pixels}")
    gray = _to_gray_u8(image)
    source_provenance.update(
        {
            "decoded_shape": list(image.shape),
            "decoded_dtype": str(image.dtype),
            "decoded_pixel_sha256": _canonical_pixel_digest(image),
            "grayscale_pixel_sha256": _canonical_pixel_digest(gray),
        }
    )
    return _LoadedImage(image=image, gray=gray, sha256=source_sha, provenance=source_provenance)


def _make_detector(name: str, config: MatcherConfig) -> Any:
    if name == "AKAZE":
        if not hasattr(cv2, "AKAZE_create"):
            raise RuntimeError("OpenCV build has no AKAZE_create")
        return cv2.AKAZE_create(threshold=float(config.akaze_threshold))
    if name == "ORB":
        if not hasattr(cv2, "ORB_create"):
            raise RuntimeError("OpenCV build has no ORB_create")
        return cv2.ORB_create(
            nfeatures=int(config.orb_max_features),
            fastThreshold=int(config.orb_fast_threshold),
        )
    raise ValueError(f"unsupported detector: {name}")


def _ratio_good(knn_matches: Sequence[Sequence[Any]], ratio: float) -> list[Any]:
    good: list[Any] = []
    for neighbors in knn_matches:
        if len(neighbors) < 2:
            continue
        first, second = neighbors[0], neighbors[1]
        if float(first.distance) < ratio * float(second.distance):
            good.append(first)
    return good


def _gate(
    passed: bool,
    value: Any,
    *,
    operator: str,
    threshold: Any,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "value": value,
        "operator": operator,
        "threshold": threshold,
    }


def _evaluate_detector(
    name: str,
    template_gray: np.ndarray,
    query_gray: np.ndarray,
    config: MatcherConfig,
) -> _DetectorEvaluation:
    detector = _make_detector(name, config)
    template_kp, template_desc = detector.detectAndCompute(template_gray, None)
    query_kp, query_desc = detector.detectAndCompute(query_gray, None)
    template_count = len(template_kp)
    query_count = len(query_kp)

    forward_good: list[Any] = []
    reverse_good: list[Any] = []
    mutual: list[Any] = []
    if template_desc is not None and query_desc is not None and template_count >= 2 and query_count >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        forward_good = _ratio_good(
            matcher.knnMatch(template_desc, query_desc, k=2),
            config.knn_ratio_threshold,
        )
        reverse_good = _ratio_good(
            matcher.knnMatch(query_desc, template_desc, k=2),
            config.knn_ratio_threshold,
        )
        reverse_pairs = {(match.queryIdx, match.trainIdx) for match in reverse_good}
        mutual = [
            match
            for match in forward_good
            if (match.trainIdx, match.queryIdx) in reverse_pairs
        ]
        mutual.sort(key=lambda item: (item.queryIdx, item.trainIdx, float(item.distance)))

    homography: np.ndarray | None = None
    mask: np.ndarray | None = None
    if len(mutual) >= 4:
        source_points = np.float32(
            [template_kp[match.queryIdx].pt for match in mutual]
        ).reshape(-1, 1, 2)
        destination_points = np.float32(
            [query_kp[match.trainIdx].pt for match in mutual]
        ).reshape(-1, 1, 2)
        cv2.setRNGSeed(int(config.rng_seed))
        try:
            homography, mask = cv2.findHomography(
                source_points,
                destination_points,
                cv2.RANSAC,
                float(config.ransac_reprojection_threshold_px),
                maxIters=int(config.ransac_max_iters),
                confidence=float(config.ransac_confidence),
            )
        except cv2.error:
            homography, mask = None, None
        if homography is not None and (not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12):
            homography, mask = None, None

    inlier_mask = (
        np.asarray(mask, dtype=np.uint8).reshape(-1).astype(bool)
        if mask is not None
        else np.zeros((len(mutual),), dtype=bool)
    )
    inliers = int(np.count_nonzero(inlier_mask))
    inlier_ratio = float(inliers / len(mutual)) if mutual else 0.0
    median_error: float | None = None
    projected_quad: list[list[float]] | None = None
    projected_area_ratio: float | None = None
    projected_convex = False
    projected_within_bounds = False

    if homography is not None:
        source_points = np.float32(
            [template_kp[match.queryIdx].pt for match in mutual]
        ).reshape(-1, 1, 2)
        destination_points = np.float32(
            [query_kp[match.trainIdx].pt for match in mutual]
        ).reshape(-1, 1, 2)
        projected_points = cv2.perspectiveTransform(source_points, homography)
        errors = np.linalg.norm(
            projected_points.reshape(-1, 2) - destination_points.reshape(-1, 2),
            axis=1,
        )
        if inliers > 0:
            median_error = float(np.median(errors[inlier_mask]))

        template_height, template_width = template_gray.shape[:2]
        corners = np.float32(
            [
                [0.0, 0.0],
                [float(template_width - 1), 0.0],
                [float(template_width - 1), float(template_height - 1)],
                [0.0, float(template_height - 1)],
            ]
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        if np.isfinite(projected).all():
            projected_quad = [[float(x), float(y)] for x, y in projected]
            projected_convex = bool(
                cv2.isContourConvex(np.rint(projected).astype(np.int32).reshape(-1, 1, 2))
            )
            query_height, query_width = query_gray.shape[:2]
            projected_area_ratio = float(
                abs(cv2.contourArea(projected.astype(np.float32)))
                / float(query_width * query_height)
            )
            margin = float(config.projected_boundary_margin_px)
            projected_within_bounds = bool(
                np.all(projected[:, 0] >= -margin)
                and np.all(projected[:, 0] <= (query_width - 1) + margin)
                and np.all(projected[:, 1] >= -margin)
                and np.all(projected[:, 1] <= (query_height - 1) + margin)
            )

    gates: dict[str, Any] = {
        "template_keypoints": _gate(
            template_count >= config.min_template_keypoints,
            template_count,
            operator=">=",
            threshold=config.min_template_keypoints,
        ),
        "query_keypoints": _gate(
            query_count >= config.min_query_keypoints,
            query_count,
            operator=">=",
            threshold=config.min_query_keypoints,
        ),
        "mutual_good_matches": _gate(
            len(mutual) >= config.min_mutual_good_matches,
            len(mutual),
            operator=">=",
            threshold=config.min_mutual_good_matches,
        ),
        "homography_estimated": _gate(
            homography is not None,
            homography is not None,
            operator="is",
            threshold=True,
        ),
        "inliers": _gate(
            inliers >= config.min_inliers,
            inliers,
            operator=">=",
            threshold=config.min_inliers,
        ),
        "inlier_ratio": _gate(
            inlier_ratio >= config.min_inlier_ratio,
            inlier_ratio,
            operator=">=",
            threshold=config.min_inlier_ratio,
        ),
        "median_reprojection_error_px": _gate(
            median_error is not None
            and math.isfinite(median_error)
            and median_error <= config.max_median_reprojection_error_px,
            median_error,
            operator="<=",
            threshold=config.max_median_reprojection_error_px,
        ),
        "projected_quadrilateral_convex": _gate(
            projected_convex,
            projected_convex,
            operator="is",
            threshold=True,
        ),
        "projected_area_ratio": _gate(
            projected_area_ratio is not None
            and config.min_projected_area_ratio
            <= projected_area_ratio
            <= config.max_projected_area_ratio,
            projected_area_ratio,
            operator="within_inclusive",
            threshold={
                "min": config.min_projected_area_ratio,
                "max": config.max_projected_area_ratio,
            },
        ),
        "projected_quadrilateral_within_query_bounds": _gate(
            projected_within_bounds,
            {
                "within": projected_within_bounds,
                "quadrilateral_xy": projected_quad,
            },
            operator="inside_with_margin_px",
            threshold=config.projected_boundary_margin_px,
        ),
    }
    reason_for_gate = {
        "template_keypoints": "TEMPLATE_KEYPOINTS_BELOW_MIN",
        "query_keypoints": "QUERY_KEYPOINTS_BELOW_MIN",
        "mutual_good_matches": "MUTUAL_GOOD_MATCHES_BELOW_MIN",
        "homography_estimated": "HOMOGRAPHY_NOT_ESTIMATED",
        "inliers": "INLIERS_BELOW_MIN",
        "inlier_ratio": "INLIER_RATIO_BELOW_MIN",
        "median_reprojection_error_px": "MEDIAN_REPROJECTION_ERROR_ABOVE_MAX_OR_UNAVAILABLE",
        "projected_quadrilateral_convex": "PROJECTED_QUADRILATERAL_NOT_CONVEX_OR_UNAVAILABLE",
        "projected_area_ratio": "PROJECTED_AREA_RATIO_OUT_OF_RANGE_OR_UNAVAILABLE",
        "projected_quadrilateral_within_query_bounds": "PROJECTED_QUADRILATERAL_OUT_OF_BOUNDS_OR_UNAVAILABLE",
    }
    reject_reasons = [reason_for_gate[name] for name, gate in gates.items() if not gate["passed"]]

    return _DetectorEvaluation(
        name=name,
        template_keypoints=template_count,
        query_keypoints=query_count,
        forward_ratio_good=len(forward_good),
        reverse_ratio_good=len(reverse_good),
        mutual_good_matches=len(mutual),
        homography=homography,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        median_reprojection_error_px=median_error,
        projected_quadrilateral=projected_quad,
        projected_area_ratio=projected_area_ratio,
        projected_convex=projected_convex,
        projected_within_bounds=projected_within_bounds,
        gates=gates,
        reject_reasons=reject_reasons,
    )


def _diagnostic_score(evaluation: _DetectorEvaluation) -> tuple[int, int, float, int]:
    return (
        int(evaluation.passed),
        evaluation.inliers,
        evaluation.inlier_ratio,
        evaluation.mutual_good_matches,
    )


def match_known_card(
    template: ImageInput,
    query: ImageInput,
    *,
    template_id: str,
    template_class: str,
    config: MatcherConfig | None = None,
) -> GeometricMatchResult:
    """Verify query geometry against one exact registered template.

    ``template_id`` and ``template_class`` are caller-supplied bindings.  This
    function does not infer either value and does not turn a match into a plant
    semantic classification.
    """

    if not template_id.strip():
        raise ValueError("template_id must be non-empty")
    if not template_class.strip():
        raise ValueError("template_class must be non-empty")
    effective_config = config or MatcherConfig()
    loaded_template = _load_image(
        template,
        role="template",
        max_pixels=effective_config.max_image_pixels,
    )
    loaded_query = _load_image(
        query,
        role="query",
        max_pixels=effective_config.max_image_pixels,
    )

    order = [effective_config.detector_preference]
    if effective_config.detector_preference == "AKAZE" and effective_config.allow_orb_fallback:
        order.append("ORB")
    evaluations: list[_DetectorEvaluation] = []
    detector_errors: list[dict[str, str]] = []
    for detector_name in order:
        try:
            evaluation = _evaluate_detector(
                detector_name,
                loaded_template.gray,
                loaded_query.gray,
                effective_config,
            )
        except (RuntimeError, cv2.error) as exc:
            detector_errors.append(
                {"detector": detector_name, "error": type(exc).__name__, "message": str(exc)}
            )
            continue
        evaluations.append(evaluation)
        if evaluation.passed:
            break
    if not evaluations:
        details = "; ".join(f"{item['detector']}: {item['message']}" for item in detector_errors)
        raise RuntimeError(f"no configured OpenCV detector could run: {details}")

    passed_evaluations = [item for item in evaluations if item.passed]
    if passed_evaluations:
        selected = passed_evaluations[0]
    else:
        selected = max(evaluations, key=_diagnostic_score)
    selected_index = evaluations.index(selected)
    fallback_used = selected.name != effective_config.detector_preference
    fallback_reason = None
    if fallback_used and evaluations:
        fallback_reason = "PREFERRED_DETECTOR_DID_NOT_PASS_ALL_GEOMETRY_GATES"

    metrics = {
        "template_keypoints": selected.template_keypoints,
        "query_keypoints": selected.query_keypoints,
        "forward_ratio_good_matches": selected.forward_ratio_good,
        "reverse_ratio_good_matches": selected.reverse_ratio_good,
        "mutual_good_matches": selected.mutual_good_matches,
        "inliers": selected.inliers,
        "inlier_ratio": selected.inlier_ratio,
        "median_reprojection_error_px": selected.median_reprojection_error_px,
        "projected_area_ratio": selected.projected_area_ratio,
        "projected_quadrilateral_xy": selected.projected_quadrilateral,
        "homography_template_to_query": (
            selected.homography.astype(float).tolist()
            if selected.homography is not None
            else None
        ),
    }
    detector_provenance = {
        "preference": effective_config.detector_preference,
        "selected": selected.name,
        "selected_attempt_index": selected_index,
        "fallback_allowed": effective_config.allow_orb_fallback,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "descriptor_norm": "HAMMING",
        "knn_k": 2,
        "ratio_filter": "strict_first_distance_lt_ratio_times_second_distance",
        "bidirectional_mutual_consistency": True,
        "homography_method": "RANSAC",
        "attempts": [item.summary() for item in evaluations],
        "detector_errors": detector_errors,
    }
    provenance = {
        "algorithm_version": ALGORITHM_VERSION,
        "opencv_version": cv2.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "template": {
            "template_id": template_id,
            "template_class": template_class,
            **dict(loaded_template.provenance),
        },
        "query": dict(loaded_query.provenance),
        "semantic_recognition_performed": False,
        "physical_hardware_touched": False,
    }
    passed = selected.passed
    return GeometricMatchResult(
        schema=SCHEMA_VERSION,
        status="PASS" if passed else "REJECT",
        passed=passed,
        claim_scope=CLAIM_SCOPE,
        irrigation_execution_authority=False,
        template_sha256=loaded_template.sha256,
        query_sha256=loaded_query.sha256,
        template_id=template_id,
        template_class=template_class,
        detector=detector_provenance,
        metrics=metrics,
        gates=selected.gates,
        reject_reasons=tuple(selected.reject_reasons),
        config=effective_config.to_dict(),
        provenance=provenance,
    )


def fuse_known_card_consensus(
    *,
    semantic_class: str | None,
    semantic_gate_passed: bool,
    template_class: str,
    geometric_result: GeometricMatchResult | Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse independent semantic and exact-template gates, without authority.

    Consensus is emitted only if the semantic gate passed, the geometric gate
    passed, and the semantic class exactly equals the registered template class.
    No output of this function grants irrigation execution authority.
    """

    geometric = (
        geometric_result.to_dict()
        if isinstance(geometric_result, GeometricMatchResult)
        else dict(geometric_result)
    )
    reasons: list[str] = []
    if semantic_gate_passed is not True:
        reasons.append("SEMANTIC_GATE_REJECTED")
    if not semantic_class:
        reasons.append("SEMANTIC_CLASS_MISSING")
    if geometric.get("passed") is not True:
        reasons.append("GEOMETRIC_GATE_REJECTED")
    if geometric.get("status") != "PASS":
        reasons.append("GEOMETRIC_STATUS_MISMATCH")
    if geometric.get("schema") != SCHEMA_VERSION:
        reasons.append("GEOMETRIC_SCHEMA_MISMATCH")
    if geometric.get("claim_scope") != CLAIM_SCOPE:
        reasons.append("GEOMETRIC_CLAIM_SCOPE_MISMATCH")
    if geometric.get("irrigation_execution_authority") is not False:
        reasons.append("GEOMETRIC_AUTHORITY_VIOLATION")
    geometric_authority = geometric.get("authority")
    required_authority_keys = {
        "irrigation_execution",
        "pump_command",
        "serial_write",
        "state_machine_write",
    }
    if (
        not isinstance(geometric_authority, Mapping)
        or set(geometric_authority) != required_authority_keys
        or any(geometric_authority[name] is not False for name in required_authority_keys)
    ):
        reasons.append("GEOMETRIC_NESTED_AUTHORITY_VIOLATION")
    bound_geometric_class = geometric.get("template_class")
    if bound_geometric_class != template_class:
        reasons.append("GEOMETRIC_TEMPLATE_CLASS_BINDING_MISMATCH")
    if semantic_class and semantic_class != template_class:
        reasons.append("SEMANTIC_TEMPLATE_CLASS_DISAGREEMENT")
    consensus = not reasons
    return {
        "schema": "rootscope.known_card_consensus.v1",
        "status": "KNOWN_CARD_CONSENSUS" if consensus else "REJECT",
        "passed": consensus,
        "claim_scope": "CONSENSUS_EVIDENCE_ONLY_NOT_EXECUTION_AUTHORITY",
        "semantic": {
            "class": semantic_class,
            "gate_passed": semantic_gate_passed is True,
        },
        "registered_template": {
            "class": template_class,
            "template_id": geometric.get("template_id"),
            "template_sha256": geometric.get("template_sha256"),
        },
        "geometric": {
            "passed": geometric.get("passed") is True,
            "status": geometric.get("status"),
            "claim_scope": geometric.get("claim_scope"),
        },
        "reject_reasons": reasons,
        "irrigation_execution_authority": False,
        "authority": {
            "irrigation_execution": False,
            "pump_command": False,
            "serial_write": False,
            "state_machine_write": False,
        },
    }


def _load_config(path: str | None) -> MatcherConfig:
    if path is None:
        return MatcherConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config JSON must be an object")
    return MatcherConfig.from_mapping(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one query against one registered printed-card template using "
            "feature geometry only; this is not semantic recognition or execution authority."
        )
    )
    parser.add_argument("--template", help="registered template image path")
    parser.add_argument("--query", help="query/camera image path")
    parser.add_argument("--template-id", help="caller-controlled immutable template ID")
    parser.add_argument("--template-class", help="class bound to the registered template")
    parser.add_argument("--config-json", help="optional strict MatcherConfig JSON")
    parser.add_argument("--output-json", help="optional result output path; stdout is always used")
    parser.add_argument(
        "--dump-default-config",
        action="store_true",
        help="print the complete default threshold configuration and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dump_default_config:
        print(json.dumps(MatcherConfig().to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    missing = [
        flag
        for flag, value in (
            ("--template", args.template),
            ("--query", args.query),
            ("--template-id", args.template_id),
            ("--template-class", args.template_class),
        )
        if not value
    ]
    if missing:
        parser.error("required unless --dump-default-config: " + ", ".join(missing))
    try:
        result = match_known_card(
            args.template,
            args.query,
            template_id=args.template_id,
            template_class=args.template_class,
            config=_load_config(args.config_json),
        )
        payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        print(payload)
        if args.output_json:
            Path(args.output_json).write_text(payload + "\n", encoding="utf-8")
        return 0 if result.passed else 2
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        error = {
            "schema": SCHEMA_VERSION,
            "status": "ERROR",
            "passed": False,
            "claim_scope": CLAIM_SCOPE,
            "irrigation_execution_authority": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(error, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
