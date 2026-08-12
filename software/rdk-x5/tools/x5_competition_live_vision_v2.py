#!/usr/bin/env python3
"""RootScope Competition Live v2: CPU audit + r7 BPU shadow proposal.

This is an additive entry point and does not overwrite Competition Live v1.
It reuses v1 camera ownership, illumination TTA, temporal fusion, OOD shadow,
geometry shadow, GUI, and safe release behavior.  The r7 BPU worker is reached
only through a bounded local AF_UNIX client on a latest-frame, single-slot
background thread.  CPU remains the independent displayed audit result because
r7 is SHADOW_CANDIDATE_NOT_DEFAULT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
import time
import tkinter as tk
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from app.competition_runtime.bpu_shadow_client import BpuShadowClient
from app.competition_runtime.bpu_shadow_protocol import (
    R7_REFERENCE_SHA256,
    ZERO_AUTHORITY,
    tensor_sha256,
    validate_logits,
)
from app.competition_runtime.plant_cpu_bpu_replay import (
    CpuTensorAudit,
    rgb_to_bpu_tensor,
)
from app.omega_vision.ood import decide, evaluate_quality

from tools import x5_competition_live_vision as live_v1

CLASS_ORDER = live_v1.CLASS_ORDER
DISPLAY_NAMES = live_v1.DISPLAY_NAMES


@dataclass(frozen=True)
class BpuShadowJob:
    """One immutable same-tensor comparison submitted by the CPU path."""

    semantic_index: int
    submitted_monotonic: float
    frame_rgb_sha256: str
    decoded_rgb_sha256s: tuple[str, ...]
    tensors: tuple[np.ndarray, ...]
    cpu_audits: tuple[Mapping[str, Any], ...]


class LatestFrameBpuWorker:
    """Run at most one BPU call while retaining only the newest pending job."""

    def __init__(
        self,
        client: BpuShadowClient,
        *,
        on_started: Callable[[BpuShadowJob], None],
        on_result: Callable[[BpuShadowJob, Mapping[str, Any], float], None],
    ) -> None:
        self.client = client
        self.on_started = on_started
        self.on_result = on_result
        self._condition = threading.Condition()
        self._pending: BpuShadowJob | None = None
        self._closed = False
        self.replaced_pending_count = 0
        self.discarded_on_close_count = 0
        self.thread = threading.Thread(
            target=self._run,
            name="rootscope-bpu-latest-shadow",
            daemon=True,
        )
        self.thread.start()

    def submit_latest(self, job: BpuShadowJob) -> bool:
        """Submit without blocking; return true when an older pending job was dropped."""

        with self._condition:
            if self._closed:
                return False
            replaced = self._pending is not None
            if replaced:
                self.replaced_pending_count += 1
            self._pending = job
            self._condition.notify()
            return replaced

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                job = self._pending
                self._pending = None
            assert job is not None
            self.on_started(job)
            tensor_hashes = [tensor_sha256(tensor) for tensor in job.tensors]

            def cpu_fallback(
                fallback_tensors: Sequence[np.ndarray],
            ) -> list[list[float]]:
                fallback_hashes = [
                    tensor_sha256(tensor) for tensor in fallback_tensors
                ]
                if fallback_hashes != tensor_hashes:
                    raise RuntimeError("async CPU fallback tensor order/hash changed")
                return [list(item["logits"]) for item in job.cpu_audits]

            started = time.monotonic()
            try:
                receipt: Mapping[str, Any] = dict(
                    self.client.infer_tensors(
                        job.tensors,
                        cpu_fallback=cpu_fallback,
                    )
                )
            except Exception as exc:
                receipt = {
                    "status": "BPU_ASYNC_EXCEPTION",
                    "backend_actual": None,
                    "bpu_batch": None,
                    "bpu_results": None,
                    "logits": None,
                    "fallback": {
                        "bpu_error": f"{type(exc).__name__}: {exc}",
                    },
                    "zero_authority": True,
                    "authority": dict(ZERO_AUTHORITY),
                }
            elapsed_ms = (time.monotonic() - started) * 1000.0
            self.on_result(job, receipt, elapsed_ms)

    def close(self, *, join_timeout_s: float) -> bool:
        """Discard a pending stale job and join the one bounded in-flight call."""

        with self._condition:
            self._closed = True
            if self._pending is not None:
                self.discarded_on_close_count += 1
                self._pending = None
            self._condition.notify_all()
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=join_timeout_s)
        return not self.thread.is_alive()


class LiveApplicationV2(live_v1.LiveApplication):
    """Add heterogeneous replay while preserving v1's camera/geometry code."""

    def __init__(self, args: argparse.Namespace) -> None:
        if not 0.01 <= args.bpu_timeout_s <= 10.0:
            raise ValueError("--bpu-timeout-s must be within 0.01..10.0")
        if not 0.5 <= args.bpu_interval_s <= 30.0:
            raise ValueError("--bpu-interval-s must be within 0.5..30.0")
        # v1 threads wait on empty queues until run() starts camera reads, so
        # the v2 replay object is installed before the first frame can arrive.
        super().__init__(args)
        self.bpu_client = BpuShadowClient(
            args.bpu_socket,
            expected_model_sha256=args.expected_bpu_model_sha256,
            timeout_s=args.bpu_timeout_s,
        )
        self.cpu_audit = CpuTensorAudit(self.runner)
        self.last_bpu_submit_monotonic = float("-inf")
        with self.state_lock:
            self.state.update(
                {
                    "bpu_status": "IDLE",
                    "bpu_label": None,
                    "bpu_backend": None,
                    "bpu_agreement": None,
                    "bpu_latency_ms": None,
                    "bpu_roundtrip_ms": None,
                    "bpu_source_semantic_index": None,
                    "bpu_completed_monotonic": None,
                    "cpu_latency_ms": None,
                }
            )
        self.root.title("RootScope Competition Live Vision v2")
        # The inherited sequence-0 event truthfully describes the reused v1
        # base initialization.  This event explicitly activates/supersedes its
        # semantic runtime contract before any frame is submitted.
        self.log.write(
            "competition_live_v2_contract_activation",
            {
                "schema": "rootscope.competition-live-vision-contract.v2",
                "supersedes_base_semantic_runtime_after_initialization": True,
                "primary": "CPU_AUDIT_AND_FALLBACK",
                "bpu": {
                    "transport": "AF_UNIX",
                    "role": "SHADOW_PROPOSAL_ONLY",
                    "qualification": "SHADOW_CANDIDATE_NOT_DEFAULT",
                    "expected_model_sha256": args.expected_bpu_model_sha256,
                    "selected_bin_changed": False,
                    "scheduler": "LATEST_FRAME_SINGLE_PENDING_SLOT",
                    "submit_interval_seconds": args.bpu_interval_s,
                    "client_timeout_seconds": args.bpu_timeout_s,
                    "blocks_cpu_primary": False,
                },
                "ood_geometry_role": "SHADOW_DISPLAY_ONLY_NON_BLOCKING",
                "shadow_blocks_primary_display": False,
                "zero_authority": True,
                "authority": dict(ZERO_AUTHORITY),
            },
        )
        self.bpu_worker = LatestFrameBpuWorker(
            self.bpu_client,
            on_started=self._on_bpu_started,
            on_result=self._on_bpu_result,
        )

    def _on_bpu_started(self, job: BpuShadowJob) -> None:
        with self.state_lock:
            self.state.update(
                {
                    "bpu_status": "BPU_SHADOW_RUNNING",
                    "bpu_source_semantic_index": job.semantic_index,
                }
            )
        self.counters["bpu_async_started"] += 1

    def _on_bpu_result(
        self,
        job: BpuShadowJob,
        receipt: Mapping[str, Any],
        roundtrip_ms: float,
    ) -> None:
        status = str(receipt.get("status") or "UNKNOWN")
        bpu_ok = status == "BPU_SHADOW_OK"
        bpu_label: str | None = None
        bpu_agreement: bool | None = None
        bpu_latency_ms: float | None = None
        backend = receipt.get("backend_actual") if bpu_ok else None
        if bpu_ok:
            logits = receipt.get("logits")
            if isinstance(logits, list) and logits:
                first_logits = validate_logits(logits[0])
                bpu_label = CLASS_ORDER[int(np.argmax(np.asarray(first_logits)))]
            agreements: list[bool] = []
            bpu_results = receipt.get("bpu_results")
            if isinstance(bpu_results, list):
                for index, item in enumerate(bpu_results):
                    if not isinstance(item, Mapping):
                        continue
                    item_logits = validate_logits(item.get("logits"))
                    bpu_index = int(np.argmax(np.asarray(item_logits)))
                    cpu_index = int(job.cpu_audits[index]["top1_index"])
                    agreements.append(bpu_index == cpu_index)
            bpu_agreement = all(agreements) if agreements else None
            batch = receipt.get("bpu_batch")
            if isinstance(batch, Mapping):
                latency = batch.get("latency_ms")
                if isinstance(latency, (int, float)):
                    bpu_latency_ms = float(latency)
        with self.state_lock:
            self.state.update(
                {
                    "bpu_status": status,
                    "bpu_label": bpu_label,
                    "bpu_backend": backend,
                    "bpu_agreement": bpu_agreement,
                    "bpu_latency_ms": bpu_latency_ms,
                    "bpu_roundtrip_ms": roundtrip_ms,
                    "bpu_source_semantic_index": job.semantic_index,
                    "bpu_completed_monotonic": time.monotonic(),
                }
            )
        self.counters[f"bpu_async_{status.lower()}"] += 1
        if bpu_agreement is True:
            self.counters["cpu_bpu_batch_top1_agree"] += 1
        elif bpu_agreement is False:
            self.counters["cpu_bpu_batch_top1_disagree"] += 1
        self.log.write(
            "bpu_shadow_update_v2",
            {
                "schema": "rootscope.competition-live-bpu-shadow.v2",
                "semantic_index": job.semantic_index,
                "submitted_monotonic": job.submitted_monotonic,
                "frame_rgb_sha256": job.frame_rgb_sha256,
                "decoded_rgb_sha256s": list(job.decoded_rgb_sha256s),
                "status": status,
                "backend_actual": backend,
                "bpu_label": bpu_label,
                "cpu_bpu_all_view_top1_agreement": bpu_agreement,
                "bpu_batch": receipt.get("bpu_batch") if bpu_ok else None,
                "bpu_results": receipt.get("bpu_results") if bpu_ok else None,
                "fallback": receipt.get("fallback"),
                "roundtrip_ms": round(roundtrip_ms, 3),
                "role": "SHADOW_PROPOSAL_ONLY",
                "qualification": "SHADOW_CANDIDATE_NOT_DEFAULT",
                "selected_bin_changed": False,
                "shadow_blocks_primary_display": False,
                "zero_authority": True,
                "authority": dict(ZERO_AUTHORITY),
            },
        )

    def semantic_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                corrected, warmth_ratio = live_v1.gray_world(frame)
                views = (
                    frame,
                    np.ascontiguousarray(frame[:, ::-1]),
                    corrected,
                    np.ascontiguousarray(corrected[:, ::-1]),
                )
                decoded_hashes = tuple(
                    hashlib.sha256(
                        np.ascontiguousarray(view).tobytes(order="C")
                    ).hexdigest()
                    for view in views
                )
                tensors = tuple(rgb_to_bpu_tensor(view) for view in views)
                cpu_audits = tuple(
                    dict(self.cpu_audit.run_one(tensor)) for tensor in tensors
                )
                cpu_logits = [
                    np.asarray(item["logits"], dtype=np.float64)
                    for item in cpu_audits
                ]
                probabilities = np.mean(
                    [live_v1.softmax(item) for item in cpu_logits],
                    axis=0,
                )
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
                raw_probability = live_v1.softmax(cpu_logits[0])
                raw_index = int(np.argmax(raw_probability))
                raw_label = CLASS_ORDER[raw_index]
                raw_confidence = float(raw_probability[raw_index])
                ood = decide(
                    cpu_logits[0],
                    evaluate_quality(frame),
                    self.calibration,
                )
                cpu_latency_ms = float(
                    sum(item["latency_ms"] for item in cpu_audits)
                )
                frame_sha = hashlib.sha256(
                    np.ascontiguousarray(frame).tobytes(order="C")
                ).hexdigest()

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
                            "cpu_latency_ms": cpu_latency_ms,
                            "last_error": None,
                        }
                    )
                self.counters["semantic_updates_v2"] += 1
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                bpu_submit = False
                bpu_pending_replaced = False
                now = time.monotonic()
                if now - self.last_bpu_submit_monotonic >= self.args.bpu_interval_s:
                    job = BpuShadowJob(
                        semantic_index=semantic_index,
                        submitted_monotonic=now,
                        frame_rgb_sha256=frame_sha,
                        decoded_rgb_sha256s=decoded_hashes,
                        tensors=tensors,
                        cpu_audits=cpu_audits,
                    )
                    bpu_pending_replaced = self.bpu_worker.submit_latest(job)
                    self.last_bpu_submit_monotonic = now
                    bpu_submit = True
                    if bpu_pending_replaced:
                        self.counters["bpu_async_pending_replaced"] += 1
                self.log.write(
                    "semantic_update_v2",
                    {
                        "schema": "rootscope.competition-live-semantic.v2",
                        "semantic_index": semantic_index,
                        "frame_rgb_sha256": frame_sha,
                        "primary_label": label,
                        "primary_confidence": confidence,
                        "primary_margin": margin,
                        "temporal_votes": stable_votes,
                        "temporal_window": len(self.probability_history),
                        "raw_label": raw_label,
                        "raw_confidence": raw_confidence,
                        "warmth_ratio_red_over_blue": warmth_ratio,
                        "illumination_normalization": "GRAY_WORLD_PLUS_RAW_FLIP_TTA",
                        "same_tensor_cpu_audit": [
                            {
                                "index": idx,
                                "decoded_rgb_sha256": decoded_hashes[idx],
                                "input_tensor_sha256": hashlib.sha256(
                                    tensors[idx].tobytes(order="C")
                                ).hexdigest(),
                                "input_tensor_shape": list(tensors[idx].shape),
                                "input_tensor_dtype": "uint8",
                                "cpu_audit": cpu_audits[idx],
                                "display_source": "CPU_AUDIT",
                            }
                            for idx in range(len(tensors))
                        ],
                        "cpu_total_latency_ms": cpu_latency_ms,
                        "bpu_submit": bpu_submit,
                        "bpu_pending_replaced": bpu_pending_replaced,
                        "bpu_role": "SHADOW_PROPOSAL_ONLY",
                        "bpu_qualification": "SHADOW_CANDIDATE_NOT_DEFAULT",
                        "bpu_scheduler": "LATEST_FRAME_SINGLE_PENDING_SLOT",
                        "selected_bin_changed": False,
                        "ood_shadow": ood.to_dict(),
                        "ood_blocks_primary_display": False,
                        "geometry_blocks_primary_display": False,
                        "shadow_blocks_primary_display": False,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "yolo_used": False,
                        "zero_authority": True,
                        "authority": dict(ZERO_AUTHORITY),
                    },
                )
                print(
                    f"[LIVE-V2 {semantic_index:04d}] CPU={label:<12} "
                    f"p={confidence:.3f} margin={margin:.3f} "
                    f"BPU_async_submit={bpu_submit} "
                    f"cpu_ms={cpu_latency_ms:.1f} "
                    f"ood_shadow={ood.decision}",
                    flush=True,
                )
                should_geometry = (
                    label != "unknown"
                    and stable_votes >= min(3, len(self.probability_history))
                    and (
                        label != self.last_geometry_label
                        or now - self.last_geometry_submit >= 4.0
                    )
                )
                if should_geometry and self.geometry_queue.empty():
                    self.geometry_queue.put_nowait(
                        (frame.copy(), label, semantic_index)
                    )
                    self.last_geometry_submit = now
                    self.last_geometry_label = label
                    with self.state_lock:
                        self.state["geometry"] = "WORKING"
                        self.state["geometry_label"] = label
            except Exception as exc:
                self.counters["semantic_errors_v2"] += 1
                with self.state_lock:
                    self.state["last_error"] = f"{type(exc).__name__}: {exc}"
                self.log.write(
                    "semantic_error_v2",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "zero_authority": True,
                        "authority": dict(ZERO_AUTHORITY),
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
        font = live_v1.display_font(25)
        label = state["label"]
        confidence = float(state["confidence"])
        primary_color = (
            (75, 235, 165, 255) if label != "unknown" else (255, 184, 74, 255)
        )
        bpu_label = state.get("bpu_label") or "-"
        bpu_agreement = state.get("bpu_agreement")
        lines = [
            (
                "RootScope LIVE v2 | CPU Audit + r7 BPU Shadow + IR-TTA",
                (132, 225, 255, 255),
            ),
            (
                f"PRIMARY CPU: {DISPLAY_NAMES[label]}  {confidence * 100:5.1f}%  "
                f"margin={float(state['margin']):.3f}  "
                f"stable={state['stable_votes']}/{state['window_size']}",
                primary_color,
            ),
            (
                f"BPU SHADOW: {bpu_label} | {state.get('bpu_status')} | "
                f"CPU/BPU agree={bpu_agreement} | "
                f"{state.get('bpu_latency_ms') or 0.0:.1f} ms",
                (126, 218, 255, 255),
            ),
            (
                f"LIGHT: Gray-World + Raw/Flip TTA | "
                f"warm R/B={float(state['warmth_ratio']):.2f}",
                (255, 230, 158, 255),
            ),
            (
                f"ADVISORY: OOD={state['ood_decision']} | "
                f"GEOMETRY={state['geometry']}:{state['geometry_label'] or '-'} | "
                "never blocks preview",
                (185, 190, 205, 255),
            ),
        ]
        y = 8
        for text_value, color in lines:
            draw.text((12, y), text_value, fill=color, font=font)
            y += 28
        error = state.get("last_error")
        footer = (
            f"ERROR: {error}"
            if error
            else (
                "r7=SHADOW_CANDIDATE_NOT_DEFAULT | "
                "ZERO SERIAL/GPIO/PUMP AUTHORITY | Q/ESC exits"
            )
        )
        draw.text(
            (12, height - 42),
            footer,
            fill=(255, 100, 100, 255) if error else (255, 255, 255, 255),
            font=font,
        )
        return image

    def close(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        try:
            close_receipt = dict(self.source.close())
        except Exception as exc:
            close_receipt = {
                "release_completed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        bpu_thread_joined = self.bpu_worker.close(join_timeout_s=4.0)
        for thread in (self.semantic_thread, self.geometry_thread):
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=12.0)
        with self.state_lock:
            final_state = dict(self.state)
        summary = {
            "schema": "rootscope.competition-live-vision-summary.v2",
            "completed_at_utc": live_v1.utc_now(),
            "counters": dict(self.counters),
            "camera_close_receipt": close_receipt,
            "cpu_model_sha256": live_v1.FROZEN_HASHES["model"],
            "primary_provider": "CPUExecutionProvider",
            "bpu": {
                "transport": "AF_UNIX",
                "status_last": final_state.get("bpu_status"),
                "backend_actual_last": final_state.get("bpu_backend"),
                "role": "SHADOW_PROPOSAL_ONLY",
                "qualification": "SHADOW_CANDIDATE_NOT_DEFAULT",
                "expected_model_sha256": self.args.expected_bpu_model_sha256,
                "plant_bpu_selected_bin": None,
                "selected_bin_changed": False,
                "worker_thread_joined": bpu_thread_joined,
                "pending_jobs_replaced": self.bpu_worker.replaced_pending_count,
                "pending_jobs_discarded_on_close": (
                    self.bpu_worker.discarded_on_close_count
                ),
            },
            "ood_geometry_role": "SHADOW_DISPLAY_ONLY_NON_BLOCKING",
            "shadow_blocks_primary_display": False,
            "yolo_used": False,
            "zero_authority": True,
            "authority": dict(ZERO_AUTHORITY),
        }
        self.log.write("session_end_v2", summary)
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
    parser = live_v1.build_parser()
    parser.description = __doc__
    parser.add_argument("--bpu-socket", required=True, type=Path)
    parser.add_argument(
        "--expected-bpu-model-sha256",
        default=R7_REFERENCE_SHA256,
    )
    parser.add_argument("--bpu-timeout-s", type=float, default=3.0)
    parser.add_argument("--bpu-interval-s", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    application: LiveApplicationV2 | None = None
    try:
        application = LiveApplicationV2(args)
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
