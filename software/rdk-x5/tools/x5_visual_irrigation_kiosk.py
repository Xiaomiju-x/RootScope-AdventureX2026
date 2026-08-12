#!/usr/bin/env python3
"""Persistent RootScope answer-card irrigation kiosk for the RDK X5.

The kiosk keeps the camera UI alive across multiple plants.  A single frame
may authorize a target only when the existing CNN and AKAZE/RANSAC branches
both pass.  The former three-consecutive-frame gate is intentionally absent.

The commissioned probe is down-only.  After every physical cycle the kiosk
returns to live vision in a disarmed state.  The operator must manually return
the probe to the highest position and press SPACE before another card can
cause motion.  This mechanical re-home attestation is never inferred from
vision or from an STM32 reset.
"""

from __future__ import annotations

import argparse
try:
    import fcntl
except ImportError:  # Protocol and preflight helpers remain importable on Windows.
    fcntl = None  # type: ignore[assignment]
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION_ROOT = Path(__file__).resolve().parents[1]
if str(VERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION_ROOT))

from app.hardware.device_identity import UsbDeviceIdentity
from tools import x5_visual_irrigation_cycle as cycle
from tools.stm32_z3_level1_first_descent import SerialSession


WINDOW_NAME = "RootScope Continuous Irrigation"
KIOSK_TOKEN = "START ROOTSCOPE CONTINUOUS KIOSK"
STATE_WAIT_HOME = "WAIT_HOME"
STATE_ARMED = "ARMED_SINGLE_FRAME_DUAL_EVIDENCE"
STATE_FAULT = "FAULT_HOLD"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def is_single_frame_target(result: dict[str, Any]) -> bool:
    return (
        result.get("state") == "CONFIRMED_DUAL_EVIDENCE"
        and str(result.get("decision")) in cycle.TARGET_CLASSES
        and bool(result.get("cnn", {}).get("pass"))
        and bool(result.get("template", {}).get("pass"))
    )


def overlay_lines(module, image, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    height = int(image.shape[0])
    y = max(34, height - 150)
    for text, color in lines:
        module.cv2.putText(
            image,
            text,
            (24, y),
            module.cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            module.cv2.LINE_AA,
        )
        y += 34


def open_camera(module, camera: str, width: int, height: int, fps: int):
    capture = module.open_camera(camera, width, height, fps)
    for _ in range(20):
        capture.read()
    return capture


def readonly_safe_check(
    *,
    device: str,
    identity: UsbDeviceIdentity,
) -> dict[str, Any]:
    usb = cycle.verify_usb_identity(identity)
    with SerialSession(device) as session:
        version = session.query_ascii("VERSION", "VERSION,")
        status = session.query_ascii("STATUS", "STATUS,")
        io_status = session.query_ascii("IOSTATUS", "IOSTATUS,")
    if cycle.EXPECTED_VERSION not in version:
        raise RuntimeError(f"unexpected STM32 firmware: {version}")
    cycle.verify_safe_locked_state(status, io_status)
    return {
        "observed_at_utc": utc_now(),
        "usb_identity": usb,
        "version": version,
        "status": status,
        "io_status": io_status,
        "physical_commands_sent": False,
    }


def next_run_dir(state_root: Path) -> tuple[str, Path]:
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = state_root / "kiosk_runs" / base
    suffix = 1
    while candidate.exists():
        candidate = state_root / "kiosk_runs" / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate.name, candidate


def execute_one_cycle(
    *,
    decision: str,
    vision_result: dict[str, Any],
    raw_frame,
    annotated_frame,
    module,
    state_root: Path,
    device: str,
    identity: UsbDeviceIdentity,
    home_attestation: dict[str, Any],
) -> dict[str, Any]:
    level = cycle.CLASS_TO_LEVEL[decision]
    run_id, run_dir = next_run_dir(state_root)
    raw_path = run_dir / f"{decision}_raw.jpg"
    annotated_path = run_dir / f"{decision}_annotated.jpg"
    result_path = run_dir / f"{decision}_result.json"
    module.cv2.imwrite(str(raw_path), raw_frame)
    module.cv2.imwrite(str(annotated_path), annotated_frame)
    module.write_result(result_path, vision_result)
    receipt_path = run_dir / "cycle_receipt.json"
    receipt: dict[str, Any] = {
        "schema": "rootscope.visual_irrigation.kiosk_cycle.v1",
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "status": "STARTING_PHYSICAL_CYCLE",
        "passed": False,
        "automatic_retry": False,
        "automatic_return": False,
        "single_frame_dual_evidence": True,
        "former_three_frame_gate_removed": True,
        "home_attestation": home_attestation,
        "vision": {
            "decision": decision,
            "state": vision_result.get("state"),
            "cnn_confidence": vision_result.get("cnn", {}).get("confidence"),
            "cnn_pass": vision_result.get("cnn", {}).get("pass"),
            "template_pass": vision_result.get("template", {}).get("pass"),
            "level": level,
            "steps": cycle.LEVEL_TO_STEPS[level],
        },
        "pump_duration_ms": cycle.CONTINUOUS_PUMP_DURATION_MS,
    }
    cycle.atomic_json(receipt_path, receipt)
    ledger = cycle.SequenceLedger.load_or_create(
        state_root / "stm32_v15_sequence.json",
        identity.identity_sha256,
    )
    try:
        physical = cycle.execute_continuous_timed_cycle(
            device=device,
            level=level,
            ledger=ledger,
            task_state=state_root / "stm32_v15_task.json",
            identity_sha256=identity.identity_sha256,
        )
        receipt["continuous_cycle"] = physical
        receipt["final_readback"] = cycle.final_readback(device)
        receipt["status"] = (
            "COMPLETE_SINGLE_FRAME_DUAL_EVIDENCE_"
            "MOTION_AND_5S_PUMP_STOPPED_LOCKED"
        )
        receipt["passed"] = True
        return receipt
    except BaseException as exc:
        cycle.safe_stop_best_effort(device)
        receipt["status"] = "FAILED_HOLD_STOP_ATTEMPTED_NO_RETRY"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        return receipt
    finally:
        receipt["finished_at_utc"] = utc_now()
        cycle.atomic_json(receipt_path, receipt)


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
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kiosk-token")
    args = parser.parse_args()

    if args.preflight:
        print(
            json.dumps(
                {
                    "schema": "rootscope.visual_irrigation.kiosk_preflight.v1",
                    "persistent_until_operator_quit": True,
                    "single_frame_dual_evidence": True,
                    "former_three_frame_gate_removed": True,
                    "cnn_branch_retained": True,
                    "akaze_ransac_template_branch_retained": True,
                    "pump_duration_ms": cycle.CONTINUOUS_PUMP_DURATION_MS,
                    "class_mapping": {
                        name: {
                            "level": level,
                            "steps": cycle.LEVEL_TO_STEPS[level],
                            "acts": level > 0,
                        }
                        for name, level in cycle.CLASS_TO_LEVEL.items()
                    },
                    "after_cycle": (
                        "live vision resumes automatically; physical actuation "
                        "remains disarmed until manual re-home plus SPACE"
                    ),
                    "automatic_retry": False,
                    "automatic_return": False,
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

    if not args.execute or args.kiosk_token != KIOSK_TOKEN:
        raise SystemExit(
            "REFUSED: --execute and exact kiosk token are required; "
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
    latest_result: dict[str, Any] | None = None
    state = STATE_WAIT_HOME
    home_attestation: dict[str, Any] | None = None
    last_message = "Manual probe return -> place next card -> press SPACE"
    frame_counter = 0

    try:
        cycle.verify_usb_identity(identity)
        capture = open_camera(
            module, args.camera, args.width, args.height, args.fps
        )
        module.cv2.namedWindow(WINDOW_NAME, module.cv2.WINDOW_NORMAL)
        module.cv2.resizeWindow(WINDOW_NAME, 1024, 576)
        print("RootScope continuous answer demo is running.")
        print("SPACE: attest manual probe return and arm one next plant")
        print("Q/Esc: quit; R: read-only recovery check after a fault")

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            frame_counter += 1
            if latest_result is None or frame_counter % max(args.infer_every, 1) == 0:
                latest_result = runtime.infer(frame)
            shown = module.annotate(frame, latest_result, 0.0)
            decision = str(latest_result.get("decision", "unknown"))
            if state == STATE_WAIT_HOME:
                lines = [
                    ("LIVE VISION - ACTUATION DISARMED", (0, 220, 255)),
                    ("Return probe to TOP; place next card; press SPACE", (0, 220, 255)),
                    (f"Detected: {decision} (no action yet)", (255, 255, 255)),
                ]
            elif state == STATE_ARMED:
                lines = [
                    ("ARMED: SINGLE-FRAME CNN + AKAZE/RANSAC", (0, 255, 0)),
                    (f"Detected: {decision}", (255, 255, 255)),
                    ("Q/Esc quits; non-target never actuates", (0, 220, 255)),
                ]
            else:
                lines = [
                    ("FAULT HOLD - NO AUTOMATIC RETRY", (0, 0, 255)),
                    (last_message, (0, 220, 255)),
                    ("Reset/check STM32, then press R for read-only check", (255, 255, 255)),
                ]
            overlay_lines(module, shown, lines)
            module.cv2.imshow(WINDOW_NAME, shown)
            key = module.cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                cycle.safe_stop_best_effort(args.serial_device)
                return 0
            if key == ord(" ") and state == STATE_WAIT_HOME:
                try:
                    check = readonly_safe_check(
                        device=args.serial_device,
                        identity=identity,
                    )
                    home_attestation = {
                        "observed_at_utc": utc_now(),
                        "method": "OPERATOR_SPACE_KEY_AFTER_MANUAL_RETURN",
                        "readonly_safe_check": check,
                    }
                    state = STATE_ARMED
                    last_message = "Armed for exactly one next target."
                    print(last_message)
                except BaseException as exc:
                    state = STATE_FAULT
                    last_message = f"Read-only arm check failed: {type(exc).__name__}: {exc}"
                    print(last_message, file=sys.stderr)
                continue
            if key in (ord("r"), ord("R")) and state == STATE_FAULT:
                try:
                    readonly_safe_check(
                        device=args.serial_device,
                        identity=identity,
                    )
                    state = STATE_WAIT_HOME
                    last_message = (
                        "Recovery check passed. Return probe to TOP, "
                        "then press SPACE."
                    )
                    print(last_message)
                except BaseException as exc:
                    last_message = f"Recovery still blocked: {type(exc).__name__}: {exc}"
                    print(last_message, file=sys.stderr)
                continue

            if state != STATE_ARMED or not is_single_frame_target(latest_result):
                continue

            trigger_frame = frame.copy()
            trigger_shown = shown.copy()
            overlay_lines(
                module,
                trigger_shown,
                [("EXECUTING ONE BOUNDED PHYSICAL CYCLE", (0, 165, 255))],
            )
            module.cv2.imshow(WINDOW_NAME, trigger_shown)
            module.cv2.waitKey(1)
            capture.release()
            capture = None
            print(
                f"Trigger: {decision}; level={cycle.CLASS_TO_LEVEL[decision]}; "
                f"pump={cycle.CONTINUOUS_PUMP_DURATION_MS} ms"
            )
            receipt = execute_one_cycle(
                decision=decision,
                vision_result=latest_result,
                raw_frame=trigger_frame,
                annotated_frame=trigger_shown,
                module=module,
                state_root=state_root,
                device=args.serial_device,
                identity=identity,
                home_attestation=home_attestation or {},
            )
            print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
            latest_result = None
            home_attestation = None
            capture = open_camera(
                module, args.camera, args.width, args.height, args.fps
            )
            if receipt.get("passed"):
                state = STATE_WAIT_HOME
                last_message = (
                    "Cycle complete. Live vision resumed; manually return "
                    "probe and press SPACE for the next plant."
                )
            else:
                state = STATE_FAULT
                last_message = str(receipt.get("error", "physical cycle failed"))
    except KeyboardInterrupt:
        cycle.safe_stop_best_effort(args.serial_device)
        return 130
    finally:
        if capture is not None:
            capture.release()
        try:
            module.cv2.destroyAllWindows()
        except module.cv2.error:
            pass
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
