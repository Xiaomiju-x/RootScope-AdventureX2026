#!/usr/bin/env python3
"""Hands-free, persistent RootScope answer-card irrigation kiosk.

This is a second competition entry point.  It deliberately does not replace
the commissioned operator-armed kiosk in ``x5_visual_irrigation_kiosk.py``.

The camera and inference loop stay alive while the serial worker performs one
bounded descent + five-second irrigation cycle.  One commissioned
``CONFIRMED_DUAL_EVIDENCE`` result immediately latches and starts the physical
cycle; there is no dwell timer, progress bar or multi-sample confirmation gate.
After a cycle, the same visible card is latched and cannot retrigger.
Presenting a different confirmed target starts the next round immediately;
clearing the scene rearms the same target class.

The mechanism has no upward drive or top sensor.  Automatic rearming therefore
uses the competition setup contract supplied by the operator: every new card
is presented only after the probe has physically been returned to the top.
The software never claims to have measured or commanded that return.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
try:
    import fcntl
except ImportError:  # Protocol and preflight helpers remain importable on Windows.
    fcntl = None  # type: ignore[assignment]
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


VERSION_ROOT = Path(__file__).resolve().parents[1]
if str(VERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION_ROOT))

from app.hardware.device_identity import UsbDeviceIdentity
from tools import x5_visual_irrigation_cycle as cycle
from tools import x5_visual_irrigation_kiosk as commissioned


WINDOW_NAME = "RootScope Auto Continuous Irrigation"
AUTOPILOT_TOKEN = "START ROOTSCOPE HANDS FREE KIOSK"
STATE_SCANNING = "SCANNING"
STATE_ACTING = "ACTING"
STATE_WAIT_CHANGE = "WAIT_CHANGE"
STATE_FAULT = "FAULT_HOLD"


@dataclass
class InferenceSample:
    """One result tied to the exact frame that produced it."""

    result: dict[str, Any]
    frame: Any
    submitted_at: float
    finished_at: float


class LatestFrameInference:
    """Run expensive vision off the GUI thread, with at most one job queued."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rootsight",
        )
        self.future: Future[dict[str, Any]] | None = None
        self.frame: Any | None = None
        self.submitted_at: float | None = None

    def submit(self, frame: Any, now: float) -> bool:
        if self.future is not None:
            return False
        self.frame = frame.copy()
        self.submitted_at = now
        self.future = self.executor.submit(self.runtime.infer, self.frame)
        return True

    def poll(self, now: float) -> InferenceSample | None:
        if self.future is None or not self.future.done():
            return None
        future = self.future
        frame = self.frame
        submitted_at = self.submitted_at
        self.future = None
        self.frame = None
        self.submitted_at = None
        result = future.result()
        if frame is None or submitted_at is None:
            raise RuntimeError("inference bookkeeping lost its source frame")
        return InferenceSample(
            result=result,
            frame=frame,
            submitted_at=submitted_at,
            finished_at=now,
        )

    @property
    def busy(self) -> bool:
        return self.future is not None

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


def immediate_target(result: dict[str, Any]) -> str | None:
    """Return one commissioned target immediately, with no second-stage gate."""

    if not commissioned.is_single_frame_target(result):
        return None
    return str(result["decision"])


@dataclass
class SceneChangeGate:
    """Prevent the card that just ran from causing an automatic retry."""

    previous_decision: str
    clear_seconds: float
    clear_first_seen: float | None = None

    def observe(
        self, result: dict[str, Any], now: float
    ) -> tuple[bool, str | None]:
        """Return ``(rearmed, target)`` for a cleared or changed scene."""

        if commissioned.is_single_frame_target(result):
            self.clear_first_seen = None
            decision = str(result["decision"])
            if decision == self.previous_decision:
                return False, None
            return True, decision
        if self.clear_first_seen is None:
            self.clear_first_seen = now
            return False, None
        return now - self.clear_first_seen >= self.clear_seconds, None


def fit_to_display(module, image, width: int, height: int):
    """Letterbox one frame to the exact kiosk framebuffer size."""

    source_h, source_w = image.shape[:2]
    scale = min(width / max(source_w, 1), height / max(source_h, 1))
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = module.cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=module.cv2.INTER_AREA,
    )
    canvas = module.np.zeros((height, width, 3), dtype=module.np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def draw_kiosk_status(
    module,
    image,
    *,
    state: str,
    decision: str,
    detail: str,
) -> None:
    cv2 = module.cv2
    width = int(image.shape[1])
    cv2.rectangle(image, (0, 0), (width, 100), (10, 20, 20), -1)
    colors = {
        STATE_SCANNING: (70, 255, 120),
        STATE_ACTING: (40, 190, 255),
        STATE_WAIT_CHANGE: (40, 220, 255),
        STATE_FAULT: (60, 60, 255),
    }
    color = colors.get(state, (255, 255, 255))
    title = {
        STATE_SCANNING: "AUTO SCAN  |  CONFIRMED TARGET -> IMMEDIATE ACTION",
        STATE_ACTING: "PLANT2ACTION  |  DESCENT + 5 s IRRIGATION",
        STATE_WAIT_CHANGE: "CYCLE COMPLETE  |  PRESENT NEXT CARD",
        STATE_FAULT: "FAULT HOLD  |  NO AUTOMATIC RETRY",
    }.get(state, state)
    cv2.putText(
        image,
        title,
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"VISION: {decision.upper()}  |  {detail}",
        (18, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        image,
        (0, int(image.shape[0]) - 38),
        (width, int(image.shape[0])),
        (10, 20, 20),
        -1,
    )
    cv2.putText(
        image,
        "Camera + CPU vision continuously active | Q/Esc: stop safely",
        (18, int(image.shape[0]) - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--camera",
        default="/dev/v4l/by-id/usb-Insta360_Insta360_Link_2C-video-index0",
    )
    parser.add_argument("--serial-device", default="/dev/rootscope_stm32")
    parser.add_argument(
        "--serial-id-path",
        default=os.environ.get(
            "ROOTSCOPE_SERIAL_ID_PATH", "commission-with-udevadm"
        ),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path.home() / ".local/state/rootscope-auto-irrigation",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--infer-every", type=int, default=3)
    parser.add_argument("--clear-seconds", type=float, default=0.6)
    parser.add_argument("--screen-width", type=int, default=1024)
    parser.add_argument("--screen-height", type=int, default=600)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--autopilot-token")
    args = parser.parse_args()

    if args.preflight:
        print(
            json.dumps(
                {
                    "schema": "rootscope.visual_irrigation.autopilot_preflight.v2",
                    "second_entry_point_preserves_commissioned_kiosk": True,
                    "camera_preview_continuous_during_physical_worker": True,
                    "automatic_target_trigger": True,
                    "keyboard_arm_required": False,
                    "gui_inference_thread_decoupled": True,
                    "inference_queue_depth": 1,
                    "latest_frame_backpressure": True,
                    "single_result_immediate_trigger": True,
                    "stable_target_gate_present": False,
                    "progress_bar_present": False,
                    "same_card_automatic_retry": False,
                    "rearm_on_scene_clear_seconds": args.clear_seconds,
                    "different_confirmed_target_immediate_trigger": True,
                    "display": [args.screen_width, args.screen_height],
                    "pump_duration_ms": cycle.CONTINUOUS_PUMP_DURATION_MS,
                    "automatic_return": False,
                    "top_sensor_present": False,
                    "top_position_contract": (
                        "operator returns probe to physical top before each "
                        "new answer card is presented"
                    ),
                    "preflight_opens_camera": False,
                    "preflight_opens_serial": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if fcntl is None:
        raise RuntimeError("physical kiosk execution requires Linux/POSIX")

    if not args.execute or args.autopilot_token != AUTOPILOT_TOKEN:
        raise SystemExit(
            "REFUSED: --execute and exact autopilot token are required; "
            "no camera or serial device was opened"
        )

    state_root = args.state_root.resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (state_root / "cycle.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("REFUSED: another RootScope cycle owns the lock") from exc

    identity = UsbDeviceIdentity(
        alias=args.serial_device,
        vid="1a86",
        pid="7523",
        id_path=args.serial_id_path,
        interface_number="00",
    )
    module = cycle.load_answer_runtime(args.bundle.resolve())
    runtime = module.AnswerCardRuntime(args.bundle.resolve())
    capture = None
    inference = LatestFrameInference(runtime)
    action_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="plant2action",
    )
    action_future: Future[dict[str, Any]] | None = None
    action_decision: str | None = None
    action_result: dict[str, Any] | None = None
    state = STATE_SCANNING
    change_gate: SceneChangeGate | None = None
    latest_result: dict[str, Any] | None = None
    latest_result_frame: Any | None = None
    latest_inference_ms: float | None = None
    frame_counter = 0
    last_detail = "camera online; scanning"
    inference_fault: str | None = None

    try:
        cycle.verify_usb_identity(identity)
        capture = commissioned.open_camera(
            module, args.camera, args.width, args.height, args.fps
        )
        module.cv2.namedWindow(WINDOW_NAME, module.cv2.WINDOW_NORMAL)
        module.cv2.moveWindow(WINDOW_NAME, 0, 0)
        module.cv2.resizeWindow(
            WINDOW_NAME,
            args.screen_width,
            args.screen_height,
        )
        module.cv2.setWindowProperty(
            WINDOW_NAME,
            module.cv2.WND_PROP_FULLSCREEN,
            module.cv2.WINDOW_FULLSCREEN,
        )
        print("RootScope hands-free continuous answer demo is running.")
        print("No SPACE key is required. Q/Esc stops safely.")
        print(
            "Setup contract: return the probe to physical TOP before "
            "presenting each new answer card."
        )

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            frame_counter += 1
            now = time.monotonic()
            if (
                inference_fault is None
                and not inference.busy
                and (
                    latest_result is None
                    or frame_counter % max(args.infer_every, 1) == 0
                )
            ):
                inference.submit(frame, now)

            sample: InferenceSample | None = None
            if inference_fault is None:
                try:
                    sample = inference.poll(now)
                except Exception as exc:
                    inference_fault = f"{type(exc).__name__}: {exc}"
                    state = STATE_FAULT
                    last_detail = f"vision worker failed: {inference_fault}"[:90]
                    print(last_detail, file=sys.stderr, flush=True)

            if sample is not None:
                latest_result = sample.result
                latest_result_frame = sample.frame
                latest_inference_ms = (
                    sample.finished_at - sample.submitted_at
                ) * 1000.0

                trigger: str | None = None
                if state == STATE_SCANNING:
                    trigger = immediate_target(latest_result)
                    if trigger is None:
                        last_detail = "camera live; background vision scanning"
                elif state == STATE_WAIT_CHANGE and change_gate is not None:
                    rearmed, changed_target = change_gate.observe(
                        latest_result,
                        sample.finished_at,
                    )
                    if rearmed:
                        state = STATE_SCANNING
                        change_gate = None
                        if changed_target is None:
                            action_decision = None
                            action_result = None
                            last_detail = "scene cleared; scanning next round"
                        else:
                            trigger = changed_target

                if trigger is not None:
                    action_decision = trigger
                    action_result = dict(latest_result)
                    trigger_frame = latest_result_frame.copy()
                    trigger_result = dict(latest_result)
                    annotated = module.annotate(
                        trigger_frame.copy(),
                        trigger_result,
                        0.0,
                    )
                    top_contract = {
                        "observed_at_utc": commissioned.utc_now(),
                        "method": (
                            "COMPETITION_SETUP_CONTRACT_NO_TOP_SENSOR_"
                            "NEW_CARD_PRESENTED_ONLY_AFTER_MANUAL_TOP_RETURN"
                        ),
                        "operator_key_required": False,
                        "software_measured_top": False,
                    }
                    action_future = action_executor.submit(
                        commissioned.execute_one_cycle,
                        decision=trigger,
                        vision_result=trigger_result,
                        raw_frame=trigger_frame,
                        annotated_frame=annotated,
                        module=module,
                        state_root=state_root,
                        device=args.serial_device,
                        identity=identity,
                        home_attestation=top_contract,
                    )
                    state = STATE_ACTING
                    last_detail = (
                        f"latched {trigger}; level {cycle.CLASS_TO_LEVEL[trigger]}, "
                        f"pump {cycle.CONTINUOUS_PUMP_DURATION_MS // 1000} s"
                    )

            display_result = latest_result
            if (
                state in (STATE_ACTING, STATE_WAIT_CHANGE, STATE_FAULT)
                and action_result is not None
            ):
                display_result = action_result

            if display_result is None:
                decision = "warming_up"
                shown = frame.copy()
            else:
                decision = str(display_result.get("decision", "unknown"))
                shown = module.annotate(frame.copy(), display_result, 0.0)

            if state == STATE_ACTING and action_future is not None:
                if action_future.done():
                    try:
                        receipt = action_future.result()
                    except Exception as exc:
                        receipt = {
                            "passed": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    print(
                        json.dumps(
                            receipt,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                    )
                    if receipt.get("passed"):
                        state = STATE_WAIT_CHANGE
                        change_gate = SceneChangeGate(
                            previous_decision=action_decision or "",
                            clear_seconds=max(args.clear_seconds, 0.0),
                        )
                        last_detail = "done; change/clear card for next round"
                    else:
                        state = STATE_FAULT
                        last_detail = str(
                            receipt.get("error", "physical cycle failed")
                        )[:70]
                    action_future = None

            if (
                latest_inference_ms is not None
                and state == STATE_SCANNING
                and "confirming" not in last_detail
            ):
                last_detail = (
                    f"camera live; vision {latest_inference_ms:.0f} ms "
                    f"in background"
                )
            display = fit_to_display(
                module,
                shown,
                args.screen_width,
                args.screen_height,
            )
            draw_kiosk_status(
                module,
                display,
                state=state,
                decision=decision,
                detail=last_detail,
            )
            module.cv2.imshow(WINDOW_NAME, display)
            key = module.cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                cycle.safe_stop_best_effort(args.serial_device)
                return 0
    except KeyboardInterrupt:
        cycle.safe_stop_best_effort(args.serial_device)
        return 130
    finally:
        if capture is not None:
            capture.release()
        if action_future is not None and not action_future.done():
            cycle.safe_stop_best_effort(args.serial_device)
        inference.shutdown()
        action_executor.shutdown(wait=True, cancel_futures=True)
        try:
            module.cv2.setWindowProperty(
                WINDOW_NAME,
                module.cv2.WND_PROP_FULLSCREEN,
                module.cv2.WINDOW_NORMAL,
            )
            module.cv2.destroyAllWindows()
        except module.cv2.error:
            pass
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
