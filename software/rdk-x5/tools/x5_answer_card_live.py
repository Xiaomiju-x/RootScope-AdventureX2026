#!/usr/bin/env python3
"""RDK X5 runtime for the four fixed RootScope printed answer cards.

The CPU ONNX classifier is the primary fixed-card predictor.  AKAZE + RANSAC
template verification is an independent second cue.  This program has no
serial, GPIO, pump, or stepper authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_ORDER = ("grass_clump", "low_shrub", "young_tree", "non_target")
ENGLISH_NAMES = {
    "grass_clump": "GRASS CLUMP",
    "low_shrub": "LOW SHRUB",
    "young_tree": "YOUNG TREE",
    "non_target": "PURE SAND / HOLD",
}
LEVELS = {"grass_clump": 1, "low_shrub": 2, "young_tree": 3, "non_target": 0}
COLORS = {
    "grass_clump": (80, 220, 120),
    "low_shrub": (60, 190, 255),
    "young_tree": (220, 160, 70),
    "non_target": (180, 180, 180),
}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def rectify_card(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image_bgr.shape[:2]
    scale = min(1.0, 1280.0 / max(width, height))
    small = cv2.resize(
        image_bgr,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 120)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    frame_area = float(small.shape[0] * small.shape[1])
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.18 * frame_area or area > 0.93 * frame_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = order_quad(approx / scale)
        top = np.linalg.norm(quad[1] - quad[0])
        bottom = np.linalg.norm(quad[2] - quad[3])
        left = np.linalg.norm(quad[3] - quad[0])
        right = np.linalg.norm(quad[2] - quad[1])
        long_side = max((top + bottom) * 0.5, (left + right) * 0.5)
        short_side = min((top + bottom) * 0.5, (left + right) * 0.5)
        if short_side <= 1.0:
            continue
        aspect = long_side / short_side
        if 1.03 <= aspect <= 2.4:
            candidates.append((area, quad))
    if candidates:
        _, quad = max(candidates, key=lambda item: item[0])
        top = np.linalg.norm(quad[1] - quad[0])
        bottom = np.linalg.norm(quad[2] - quad[3])
        left = np.linalg.norm(quad[3] - quad[0])
        right = np.linalg.norm(quad[2] - quad[1])
        target_w = int(max(top, bottom))
        target_h = int(max(left, right))
        if target_h > target_w:
            quad = np.roll(quad, -1, axis=0)
            target_w, target_h = target_h, target_w
        target_w = int(np.clip(target_w, 640, 1600))
        target_h = int(np.clip(target_h, 420, 1200))
        destination = np.array(
            [
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1],
            ],
            dtype=np.float32,
        )
        warped = cv2.warpPerspective(
            image_bgr,
            cv2.getPerspectiveTransform(quad, destination),
            (target_w, target_h),
        )
        return warped, {
            "method": "quadrilateral",
            "quad_xy": np.round(quad, 1).tolist(),
        }
    x0, x1 = int(width * 0.17), int(width * 0.83)
    y0, y1 = int(height * 0.11), int(height * 0.93)
    return image_bgr[y0:y1, x0:x1].copy(), {
        "method": "center_fallback",
        "crop_xyxy": [x0, y0, x1, y1],
    }


def resize_short_side(image: np.ndarray, short_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = short_side / min(height, width)
    return cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_LINEAR,
    )


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = resize_short_side(image_rgb, 256)
    height, width = resized.shape[:2]
    x0 = (width - 224) // 2
    y0 = (height - 224) // 2
    crop = resized[y0 : y0 + 224, x0 : x0 + 224]
    normalized = (crop.astype(np.float32) / 255.0 - MEAN) / STD
    return np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values -= np.max(values)
    exp = np.exp(values)
    return (exp / np.sum(exp)).astype(np.float32)


class AnswerCardRuntime:
    def __init__(self, bundle: Path) -> None:
        self.bundle = bundle.resolve()
        self.manifest = json.loads(
            (self.bundle / "model_manifest.json").read_text("utf-8")
        )
        if tuple(self.manifest["class_order"]) != CLASS_ORDER:
            raise RuntimeError("class order mismatch")
        model_record = self.manifest["artifacts"]["onnx"]
        self.model_path = self.bundle / model_record["path"]
        if sha256_file(self.model_path) != model_record["sha256"]:
            raise RuntimeError("ONNX hash mismatch")
        template_manifest_path = self.bundle / "templates_manifest.json"
        if (
            sha256_file(template_manifest_path)
            != self.manifest["artifacts"]["templates_manifest"]["sha256"]
        ):
            raise RuntimeError("template manifest hash mismatch")
        self.template_records = json.loads(
            template_manifest_path.read_text("utf-8")
        )
        for record in self.template_records:
            path = self.bundle / record["path"]
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"template hash mismatch: {path}")

        # The X5 system OpenCV build has QT5 support, while the isolated
        # inference venv intentionally carries headless OpenCV.  Live mode can
        # therefore run with /usr/bin/python3 and expose only the venv's
        # site-packages here to obtain ONNX Runtime after system cv2/numpy have
        # already been imported.
        ort_site = os.environ.get("ROOTSCOPE_ORT_SITE", "").strip()
        if ort_site and ort_site not in sys.path:
            sys.path.insert(0, ort_site)
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.confidence_threshold = float(
            self.manifest["metrics"]["confidence_threshold"]
        )
        self.margin_threshold = float(self.manifest["metrics"]["margin_threshold"])
        self.akaze = cv2.AKAZE_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.templates: dict[str, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}
        for record in self.template_records:
            image = cv2.imread(str(self.bundle / record["path"]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"cannot decode template: {record['path']}")
            image = resize_short_side(image, 720)
            self.templates[record["class_id"]] = self.akaze.detectAndCompute(
                image, None
            )

    def template_scores(self, roi_bgr: np.ndarray) -> dict[str, dict[str, Any]]:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = resize_short_side(gray, 720)
        query_kp, query_desc = self.akaze.detectAndCompute(gray, None)
        results: dict[str, dict[str, Any]] = {}
        for class_id, (template_kp, template_desc) in self.templates.items():
            good: list[cv2.DMatch] = []
            inliers = 0
            if (
                template_desc is not None
                and query_desc is not None
                and len(template_desc) >= 2
                and len(query_desc) >= 2
            ):
                matches = self.matcher.knnMatch(template_desc, query_desc, k=2)
                good = [
                    first
                    for first, second in matches
                    if first.distance < 0.76 * second.distance
                ]
                if len(good) >= 6:
                    src = np.float32(
                        [template_kp[match.queryIdx].pt for match in good]
                    ).reshape(-1, 1, 2)
                    dst = np.float32(
                        [query_kp[match.trainIdx].pt for match in good]
                    ).reshape(-1, 1, 2)
                    _, mask = cv2.findHomography(
                        src, dst, cv2.RANSAC, 5.0
                    )
                    if mask is not None:
                        inliers = int(mask.sum())
            results[class_id] = {
                "template_keypoints": len(template_kp),
                "good_matches": len(good),
                "homography_inliers": inliers,
                "inlier_ratio": inliers / max(1, len(good)),
                "template_coverage": inliers / max(1, len(template_kp)),
            }
        return results

    def infer(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        roi, geometry = rectify_card(frame_bgr)
        logits = self.session.run(["logits"], {"input": preprocess(roi)})[0][0]
        probabilities = softmax(logits)
        order = np.argsort(probabilities)[::-1]
        predicted = CLASS_ORDER[int(order[0])]
        confidence = float(probabilities[order[0]])
        margin = float(probabilities[order[0]] - probabilities[order[1]])
        scores = self.template_scores(roi)
        template_class = max(
            CLASS_ORDER,
            key=lambda class_id: (
                scores[class_id]["template_coverage"],
                scores[class_id]["inlier_ratio"],
                scores[class_id]["homography_inliers"],
            ),
        )
        template = scores[template_class]
        template_pass = (
            template["homography_inliers"] >= 8
            and template["inlier_ratio"] >= 0.28
            and template["template_coverage"] >= 0.05
        )
        cnn_pass = (
            confidence >= self.confidence_threshold
            and margin >= self.margin_threshold
        )
        evidence_agrees = template_pass and template_class == predicted
        if cnn_pass and evidence_agrees:
            decision = predicted
            state = "CONFIRMED_DUAL_EVIDENCE"
        elif cnn_pass and confidence >= 0.94 and margin >= 0.86:
            decision = predicted
            state = "CNN_HIGH_TEMPLATE_UNCONFIRMED"
        else:
            decision = "non_target"
            state = "HOLD_EVIDENCE_INSUFFICIENT"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "schema": "rootscope.answer_cards.inference.v1",
            "observed_at_utc": utc_now(),
            "decision": decision,
            "state": state,
            "action_level": LEVELS[decision] if state != "HOLD_EVIDENCE_INSUFFICIENT" else 0,
            "cnn": {
                "prediction": predicted,
                "confidence": confidence,
                "margin": margin,
                "probabilities": {
                    class_id: float(probabilities[index])
                    for index, class_id in enumerate(CLASS_ORDER)
                },
                "pass": cnn_pass,
            },
            "template": {
                "prediction": template_class,
                "pass": template_pass,
                **template,
                "all_scores": scores,
            },
            "geometry": geometry,
            "latency_ms": elapsed_ms,
            "physical_action_authority": False,
        }


def annotate(frame: np.ndarray, result: dict[str, Any], fps: float) -> np.ndarray:
    shown = frame.copy()
    decision = str(result["decision"])
    color = COLORS[decision]
    overlay = shown.copy()
    cv2.rectangle(overlay, (0, 0), (shown.shape[1], 190), (8, 20, 30), -1)
    shown = cv2.addWeighted(overlay, 0.88, shown, 0.12, 0)
    cnn = result["cnn"]
    template = result["template"]
    lines = [
        f"RootScope | {ENGLISH_NAMES[decision]} | LEVEL {result['action_level']}",
        f"STATE: {result['state']}",
        f"CNN: {cnn['prediction']}  conf={cnn['confidence']:.3f}  margin={cnn['margin']:.3f}",
        f"AKAZE/RANSAC: {template['prediction']}  inliers={template['homography_inliers']}  coverage={template['template_coverage']:.2f}",
        f"latency={result['latency_ms']:.1f} ms  display={fps:.1f} fps | Q quit | S snapshot",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            shown,
            line,
            (24, 34 + index * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82 if index == 0 else 0.68,
            color if index < 2 else (235, 245, 250),
            2,
            cv2.LINE_AA,
        )
    return shown


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_image(runtime: AnswerCardRuntime, args: argparse.Namespace) -> int:
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode image: {args.image}")
    result = runtime.infer(image)
    if args.result_json:
        write_result(args.result_json, result)
    if args.annotated:
        shown = annotate(image, result, 0.0)
        args.annotated.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.annotated), shown):
            raise RuntimeError(f"cannot write annotated image: {args.annotated}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def open_camera(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    source: int | str = int(device) if device.isdigit() else device
    backend = cv2.CAP_V4L2 if os.name != "nt" else cv2.CAP_DSHOW
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera: {device}")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def run_camera(runtime: AnswerCardRuntime, args: argparse.Namespace) -> int:
    capture = open_camera(args.camera, args.width, args.height, args.fps)
    snapshot_root = args.snapshot_root
    snapshot_root.mkdir(parents=True, exist_ok=True)
    last_result: dict[str, Any] | None = None
    last_shown: np.ndarray | None = None
    frame_index = 0
    display_fps = 0.0
    previous = time.perf_counter()
    inference_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    result_lock = threading.Lock()
    stop_worker = threading.Event()
    worker: threading.Thread | None = None
    try:
        for _ in range(20):
            capture.read()
        if args.once:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("camera frame read failed")
            result = runtime.infer(frame)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = snapshot_root / f"{stamp}_raw.jpg"
            annotated_path = snapshot_root / f"{stamp}_annotated.jpg"
            result_path = snapshot_root / f"{stamp}_result.json"
            cv2.imwrite(str(raw_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(annotated_path), annotate(frame, result, 0.0))
            write_result(result_path, result)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise RuntimeError("initial camera frame read failed")
        last_result = runtime.infer(first_frame)

        def infer_latest() -> None:
            nonlocal last_result
            while not stop_worker.is_set():
                try:
                    queued_frame = inference_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    inferred = runtime.infer(queued_frame)
                    with result_lock:
                        last_result = inferred
                finally:
                    inference_queue.task_done()

        worker = threading.Thread(
            target=infer_latest,
            name="rootscope-answer-inference",
            daemon=True,
        )
        worker.start()
        cv2.namedWindow("RootScope Answer Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RootScope Answer Vision", 1280, 720)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame_index += 1
            if frame_index % max(1, args.infer_every) == 0:
                try:
                    inference_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass
            now = time.perf_counter()
            instantaneous = 1.0 / max(1e-6, now - previous)
            display_fps = instantaneous if display_fps == 0.0 else 0.9 * display_fps + 0.1 * instantaneous
            previous = now
            with result_lock:
                shown_result = dict(last_result)
            last_shown = annotate(frame, shown_result, display_fps)
            cv2.imshow("RootScope Answer Vision", last_shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and last_shown is not None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                cv2.imwrite(str(snapshot_root / f"{stamp}_annotated.jpg"), last_shown)
                write_result(snapshot_root / f"{stamp}_result.json", shown_result)
    finally:
        stop_worker.set()
        if worker is not None:
            worker.join(timeout=4.0)
        capture.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Single-frame/headless validation must still exit cleanly.
            pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path)
    mode.add_argument("--camera")
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--annotated", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--infer-every", type=int, default=3)
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=Path.home() / "rootscope_answer_snapshots",
    )
    args = parser.parse_args()
    runtime = AnswerCardRuntime(args.bundle)
    if args.image is not None:
        return run_image(runtime, args)
    return run_camera(runtime, args)


if __name__ == "__main__":
    raise SystemExit(main())
