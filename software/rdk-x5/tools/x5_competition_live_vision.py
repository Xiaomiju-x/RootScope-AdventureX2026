#!/usr/bin/env python3
"""RootScope competition live vision.

Primary display:
  illumination-robust multi-view CPU classification + temporal smoothing.

Advisory shadow display:
  frozen Omega OOD and registered-card geometry.  Shadow results never suppress
  the visible competition label and never acquire physical authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import threading
import time
import tkinter as tk
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from app.edge.onnx_cpu import preprocess_rgb
from app.omega_vision.ood import Calibration, decide, evaluate_quality
from app.omega_vision.uvc_card_frontend import (
    ExpectedCameraIdentity,
    FrontendRequest,
    LiveUvcFrameSource,
    read_explicit_usb_identity,
)
from app.vision.card_geometric_matcher import MatcherConfig, match_known_card
from app.vision.dual_path_demo import (
    build_seed17_runner_from_capsule,
    load_template_registry,
)


CLASS_ORDER = ("grass_clump", "low_shrub", "young_tree", "unknown")
DISPLAY_NAMES = {
    "grass_clump": "GRASS CLUMP / CAO CONG",
    "low_shrub": "LOW SHRUB / GUAN MU",
    "young_tree": "YOUNG TREE / YOU SHU",
    "unknown": "UNKNOWN / NON-TARGET",
}
FROZEN_HASHES = {
    "capsule": "1e839948e466895e0b416aababb39402380463f9ce125c8aeb59f4087191cb97",
    "model": "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad",
    "registry": "f5328b66c6b385dd081c1a27577d92b5c63fc18dc00d815f7d3d2561bb68e29f",
    "calibration": "e82b196ab627a935fd571bba2b37aa637f040af5ba81fbab5183c1f15aa2e564",
    "matcher": "9952864e50371675e7ea181cc57f2edd9eadd9189bfa0e9eda1e5cdd8f8ca61a",
}
ZERO_AUTHORITY = {
    "execution_authority": False,
    "physical_authority": False,
    "serial_write": False,
    "gpio_access": False,
    "pump_command": False,
    "state_machine_write": False,
    "irrigation_execution": False,
    "physical_completion": False,
}


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


def bind_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    actual = sha256_file(resolved)
    expected = FROZEN_HASHES[label]
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return resolved


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_calibration(path: Path) -> Calibration:
    raw = load_object(path).get("calibration")
    if not isinstance(raw, dict):
        raise RuntimeError("calibration manifest omits calibration")
    normalized = dict(raw)
    for key in ("class_order", "conformal_nonconformity", "calibration_roles"):
        if not isinstance(normalized.get(key), list):
            raise RuntimeError(f"calibration.{key} must be an array")
        normalized[key] = tuple(normalized[key])
    result = Calibration(**normalized)
    if result.class_order != CLASS_ORDER:
        raise RuntimeError("calibration class order changed")
    return result


def gray_world(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    floating = rgb.astype(np.float32)
    means = floating.reshape(-1, 3).mean(axis=0)
    neutral = float(np.mean(means))
    scales = neutral / np.maximum(means, 1.0)
    corrected = np.clip(floating * scales.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    warmth_ratio = float(means[0] / max(float(means[2]), 1.0))
    return corrected, warmth_ratio


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    values = np.exp(shifted)
    return values / float(values.sum())


def display_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


class JsonlLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.handle = path.open("x", encoding="utf-8", newline="\n")
        self.sequence = 0

    def write(self, event: str, payload: dict[str, Any]) -> None:
        with self.lock:
            record = {
                "schema": "rootscope.competition-live-vision.v1",
                "sequence": self.sequence,
                "timestamp_utc": utc_now(),
                "event": event,
                "payload": payload,
                "authority": dict(ZERO_AUTHORITY),
            }
            self.sequence += 1
            self.handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        with self.lock:
            if not self.handle.closed:
                self.handle.close()


class LiveApplication:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_event = threading.Event()
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.geometry_queue: queue.Queue[tuple[np.ndarray, str, int]] = queue.Queue(
            maxsize=1
        )
        self.state_lock = threading.Lock()
        self.state: dict[str, Any] = {
            "label": "unknown",
            "confidence": 0.0,
            "margin": 0.0,
            "stable_votes": 0,
            "window_size": 0,
            "raw_label": "unknown",
            "raw_confidence": 0.0,
            "warmth_ratio": 1.0,
            "ood_decision": "PENDING",
            "ood_reasons": [],
            "geometry": "PENDING",
            "geometry_label": None,
            "semantic_index": 0,
            "last_error": None,
        }
        self.counters: Counter[str] = Counter()
        self.probability_history: deque[np.ndarray] = deque(maxlen=5)
        self.last_geometry_submit = 0.0
        self.last_geometry_label: str | None = None

        self.capsule = bind_file(args.capsule, "capsule")
        self.model = bind_file(args.model, "model")
        self.registry_path = bind_file(args.registry, "registry")
        self.calibration_path = bind_file(args.calibration, "calibration")
        self.matcher_path = bind_file(args.matcher, "matcher")
        self.calibration = load_calibration(self.calibration_path)
        self.matcher_config = MatcherConfig.from_mapping(load_object(self.matcher_path))
        self.registry = load_template_registry(self.registry_path)
        self.templates = {item.class_name: item for item in self.registry.templates}
        self.runner = build_seed17_runner_from_capsule(
            self.capsule, model_path=self.model
        )
        if list(self.runner.providers) != ["CPUExecutionProvider"]:
            raise RuntimeError(f"CPU-only provider contract failed: {self.runner.providers}")

        args.output_dir.mkdir(parents=True, exist_ok=False)
        self.log = JsonlLog(args.output_dir / "live.jsonl")
        request = FrontendRequest(
            device=args.device,
            expected_camera=ExpectedCameraIdentity(
                usb_vid="32e6",
                usb_pid="9228",
                usb_serial="202604081837",
            ),
            print_manifest=args.calibration,
            mode="bounded",
            frames=1,
            warmup_frames=0,
            interval_seconds=0.0,
            width=args.width,
            height=args.height,
            fps=args.fps,
            output_root=args.output_dir,
            jsonl_path=args.output_dir / "unused.jsonl",
        )
        self.source = LiveUvcFrameSource(request)
        self.camera_settings = dict(self.source.negotiated_settings())
        self.log.write(
            "session_start",
            {
                "mode": "COMPETITION_PRIMARY_WITH_SHADOW_SAFETY",
                "camera": self.camera_settings,
                "model_sha256": FROZEN_HASHES["model"],
                "provider": list(self.runner.providers),
                "yolo_used": False,
                "plant_bpu_selected_bin": None,
                "plant_bpu_used": False,
                "primary_pipeline": [
                    "RAW_RGB",
                    "GRAY_WORLD",
                    "HORIZONTAL_FLIP_TTA",
                    "PROBABILITY_ENSEMBLE",
                    "TEMPORAL_MEAN_5",
                ],
                "shadow_pipeline": [
                    "OMEGA_OOD",
                    "CONFORMAL_SET",
                    "QUALITY_GATE",
                    "PREDICTED_CLASS_TEMPLATE_GEOMETRY",
                ],
                "shadow_blocks_primary_display": False,
                "authority": dict(ZERO_AUTHORITY),
            },
        )

        self.root = tk.Tk()
        self.root.title("RootScope Competition Live Vision")
        self.root.geometry("1280x800")
        self.root.configure(bg="#07131d")
        self.preview = tk.Label(self.root, bg="#07131d")
        self.preview.pack(fill=tk.BOTH, expand=True)
        self.root.bind_all("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(150, self.root.focus_force)
        self.root.attributes("-topmost", True)
        self.root.after(1500, lambda: self.root.attributes("-topmost", False))

        self.semantic_thread = threading.Thread(
            target=self.semantic_worker,
            name="rootscope-semantic",
            daemon=True,
        )
        self.geometry_thread = threading.Thread(
            target=self.geometry_worker,
            name="rootscope-geometry",
            daemon=True,
        )
        self.semantic_thread.start()
        self.geometry_thread.start()

    def model_logits(self, rgb: np.ndarray) -> np.ndarray:
        tensor = preprocess_rgb(rgb, self.runner.preprocess)
        values = self.runner._session.run(  # frozen runner; same path as offline CLI
            [self.runner.output_name],
            {self.runner.input_name: tensor},
        )
        logits = np.asarray(values[0], dtype=np.float64)
        if logits.shape != (1, 4) or not np.isfinite(logits).all():
            raise RuntimeError("semantic ONNX output is not finite [1,4]")
        return logits[0]

    def semantic_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                corrected, warmth_ratio = gray_world(frame)
                views = (
                    frame,
                    np.ascontiguousarray(frame[:, ::-1]),
                    corrected,
                    np.ascontiguousarray(corrected[:, ::-1]),
                )
                logits = [self.model_logits(view) for view in views]
                probabilities = np.mean([softmax(item) for item in logits], axis=0)
                self.probability_history.append(probabilities)
                temporal = np.mean(list(self.probability_history), axis=0)
                order = np.argsort(-temporal, kind="stable")
                index = int(order[0])
                label = CLASS_ORDER[index]
                confidence = float(temporal[index])
                margin = confidence - float(temporal[int(order[1])])
                vote_labels = [
                    CLASS_ORDER[int(np.argmax(item))]
                    for item in self.probability_history
                ]
                stable_votes = vote_labels.count(label)
                raw_probability = softmax(logits[0])
                raw_index = int(np.argmax(raw_probability))
                raw_label = CLASS_ORDER[raw_index]
                raw_confidence = float(raw_probability[raw_index])
                ood = decide(
                    logits[0],
                    evaluate_quality(frame),
                    self.calibration,
                )
                with self.state_lock:
                    semantic_index = int(self.state["semantic_index"]) + 1
                    self.state.update(
                        {
                            "label": label,
                            "confidence": confidence,
                            "margin": margin,
                            "stable_votes": stable_votes,
                            "window_size": len(self.probability_history),
                            "raw_label": raw_label,
                            "raw_confidence": raw_confidence,
                            "warmth_ratio": warmth_ratio,
                            "ood_decision": ood.decision,
                            "ood_reasons": list(ood.reasons),
                            "semantic_index": semantic_index,
                            "last_error": None,
                        }
                    )
                self.counters["semantic_updates"] += 1
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                payload = {
                    "semantic_index": semantic_index,
                    "primary_label": label,
                    "primary_confidence": confidence,
                    "primary_margin": margin,
                    "temporal_votes": stable_votes,
                    "temporal_window": len(self.probability_history),
                    "raw_label": raw_label,
                    "raw_confidence": raw_confidence,
                    "warmth_ratio_red_over_blue": warmth_ratio,
                    "illumination_normalization": "GRAY_WORLD_PLUS_RAW_FLIP_TTA",
                    "ood_shadow": ood.to_dict(),
                    "ood_blocks_primary_display": False,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "provider": "CPUExecutionProvider",
                    "yolo_used": False,
                    "plant_bpu_used": False,
                }
                self.log.write("semantic_update", payload)
                print(
                    f"[LIVE {semantic_index:04d}] primary={label:<12} "
                    f"p={confidence:.3f} margin={margin:.3f} "
                    f"votes={stable_votes}/{len(self.probability_history)} "
                    f"raw={raw_label}:{raw_confidence:.3f} "
                    f"warm={warmth_ratio:.2f} "
                    f"ood_shadow={ood.decision} "
                    f"ms={elapsed_ms:.1f}",
                    flush=True,
                )

                now = time.monotonic()
                should_geometry = (
                    label != "unknown"
                    and stable_votes >= min(3, len(self.probability_history))
                    and (
                        label != self.last_geometry_label
                        or now - self.last_geometry_submit >= 4.0
                    )
                )
                if should_geometry and self.geometry_queue.empty():
                    self.geometry_queue.put_nowait((frame.copy(), label, semantic_index))
                    self.last_geometry_submit = now
                    self.last_geometry_label = label
                    with self.state_lock:
                        self.state["geometry"] = "WORKING"
                        self.state["geometry_label"] = label
            except Exception as exc:
                self.counters["semantic_errors"] += 1
                with self.state_lock:
                    self.state["last_error"] = f"{type(exc).__name__}: {exc}"
                self.log.write(
                    "semantic_error",
                    {"error_type": type(exc).__name__, "error": str(exc)},
                )

    def geometry_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame, label, semantic_index = self.geometry_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            template = self.templates.get(label)
            if template is None:
                continue
            started = time.perf_counter()
            try:
                with TemporaryDirectory(prefix="rootscope-live-geometry-") as temporary:
                    query = Path(temporary) / "frame.png"
                    Image.fromarray(frame, mode="RGB").save(query, format="PNG")
                    result = match_known_card(
                        template.path,
                        query,
                        template_id=template.template_id,
                        template_class=template.class_name,
                        config=self.matcher_config,
                    ).to_dict()
                status = "VERIFIED" if result.get("passed") is True else "NO_MATCH"
                with self.state_lock:
                    self.state["geometry"] = status
                    self.state["geometry_label"] = label
                self.counters[f"geometry_{status.lower()}"] += 1
                self.log.write(
                    "geometry_update",
                    {
                        "semantic_index": semantic_index,
                        "label": label,
                        "status": status,
                        "passed": bool(result.get("passed")),
                        "metrics": result.get("metrics"),
                        "reject_reasons": result.get("reject_reasons"),
                        "elapsed_ms": round(
                            (time.perf_counter() - started) * 1000.0, 3
                        ),
                        "blocks_primary_display": False,
                    },
                )
                print(
                    f"[GEOMETRY] label={label} status={status} "
                    f"ms={(time.perf_counter() - started) * 1000.0:.1f}",
                    flush=True,
                )
            except Exception as exc:
                self.counters["geometry_errors"] += 1
                with self.state_lock:
                    self.state["geometry"] = "ERROR"
                self.log.write(
                    "geometry_error",
                    {
                        "semantic_index": semantic_index,
                        "label": label,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )

    def overlay(self, frame: np.ndarray) -> Image.Image:
        with self.state_lock:
            state = dict(self.state)
        image = Image.fromarray(frame, mode="RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        draw.rectangle((0, 0, width, 154), fill=(5, 18, 28, 225))
        draw.rectangle((0, height - 62, width, height), fill=(5, 18, 28, 225))
        font = display_font(25)
        label = state["label"]
        confidence = float(state["confidence"])
        primary_color = (
            (75, 235, 165, 255) if label != "unknown" else (255, 184, 74, 255)
        )
        lines = [
            (
                "RootScope COMPETITION LIVE | ResNet18 CPU + IR-TTA + Temporal Fusion",
                (132, 225, 255, 255),
            ),
            (
                f"PRIMARY: {DISPLAY_NAMES[label]}  {confidence * 100:5.1f}%  "
                f"margin={float(state['margin']):.3f}  "
                f"stable={state['stable_votes']}/{state['window_size']}",
                primary_color,
            ),
            (
                f"ILLUMINATION: Gray-World + Raw/Flip TTA | warm R/B={float(state['warmth_ratio']):.2f}",
                (255, 230, 158, 255),
            ),
            (
                f"SHADOW ONLY: OOD={state['ood_decision']} "
                f"{','.join(state['ood_reasons']) or 'CLEAR'} | "
                f"GEOMETRY={state['geometry']}:{state['geometry_label'] or '-'}",
                (185, 190, 205, 255),
            ),
            (
                "BPU PLANT MODEL: OFF / NOT QUALIFIED | ZERO PUMP/SERIAL/PHYSICAL AUTHORITY",
                (255, 130, 130, 255),
            ),
        ]
        y = 8
        for text_value, color in lines:
            draw.text((12, y), text_value, fill=color, font=font)
            y += 28
        error = state.get("last_error")
        if error:
            draw.text(
                (12, height - 51),
                f"ERROR: {error}",
                fill=(255, 100, 100, 255),
                font=font,
            )
        else:
            draw.text(
                (12, height - 42),
                "Place one card in the center (35%-70% of view). Q / ESC closes safely.",
                fill=(255, 255, 255, 255),
                font=font,
            )
        return image

    def submit_latest(self, frame: np.ndarray) -> None:
        if self.frame_queue.empty():
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    def update_preview(self) -> None:
        if self.stop_event.is_set():
            return
        try:
            frame = self.source.read_rgb()
            self.counters["captured_frames"] += 1
            self.submit_latest(frame)
            shown = self.overlay(frame)
            target_width = max(self.preview.winfo_width(), 960)
            target_height = max(self.preview.winfo_height(), 540)
            scale = min(target_width / shown.width, target_height / shown.height)
            size = (max(1, int(shown.width * scale)), max(1, int(shown.height * scale)))
            shown = shown.resize(size, Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(shown)
            self.preview.configure(image=tk_image)
            self.preview.image = tk_image
        except Exception as exc:
            self.counters["capture_errors"] += 1
            with self.state_lock:
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
        self.root.after(55, self.update_preview)

    def on_key(self, event: tk.Event[Any]) -> None:
        if (event.char or "").lower() == "q" or event.keysym == "Escape":
            self.close()

    def run(self) -> None:
        self.root.after(10, self.update_preview)
        self.root.mainloop()

    def close(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        close_receipt: dict[str, Any]
        try:
            close_receipt = dict(self.source.close())
        except Exception as exc:
            close_receipt = {
                "release_completed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        for thread in (self.semantic_thread, self.geometry_thread):
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=12.0)
        summary = {
            "schema": "rootscope.competition-live-vision-summary.v1",
            "completed_at_utc": utc_now(),
            "counters": dict(self.counters),
            "camera_close_receipt": close_receipt,
            "model_sha256": FROZEN_HASHES["model"],
            "provider": ["CPUExecutionProvider"],
            "yolo_used": False,
            "plant_bpu_selected_bin": None,
            "plant_bpu_used": False,
            "shadow_blocks_primary_display": False,
            "authority": dict(ZERO_AUTHORITY),
        }
        self.log.write("session_end", summary)
        self.log.close()
        (self.args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        try:
            self.root.quit()
            self.root.destroy()
        except tk.TclError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--matcher", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    application: LiveApplication | None = None
    try:
        application = LiveApplication(args)
        application.run()
        return 0
    except Exception as exc:
        print(f"FATAL {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if application is not None and not application.stop_event.is_set():
            application.close()


if __name__ == "__main__":
    raise SystemExit(main())
