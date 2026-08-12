#!/usr/bin/env python3
"""Read-only live-camera smoke test for the autopilot inference worker.

This diagnostic never opens the STM32 serial device and has no physical
execution path.  It proves that camera frames continue to be consumed while
the expensive dual-evidence inference worker is busy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOTSCOPE_ROOT = Path(__file__).resolve().parents[1]
if str(ROOTSCOPE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROOTSCOPE_ROOT))

from tools import x5_visual_irrigation_cycle as cycle  # noqa: E402
from tools import x5_visual_irrigation_kiosk as commissioned  # noqa: E402
from tools.x5_visual_irrigation_autopilot import (  # noqa: E402
    LatestFrameInference,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    module = cycle.load_answer_runtime(args.bundle.resolve())
    runtime = module.AnswerCardRuntime(args.bundle.resolve())
    worker = LatestFrameInference(runtime)
    capture = commissioned.open_camera(
        module,
        args.camera,
        args.width,
        args.height,
        args.fps,
    )
    started = time.monotonic()
    deadline = started + max(args.seconds, 0.5)
    frames = 0
    frames_while_inference_busy = 0
    inference_samples = 0
    inference_latencies_ms: list[float] = []
    decisions: list[str] = []
    try:
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frames += 1
            now = time.monotonic()
            if worker.busy:
                frames_while_inference_busy += 1
            else:
                worker.submit(frame, now)
            sample = worker.poll(time.monotonic())
            if sample is not None:
                inference_samples += 1
                inference_latencies_ms.append(
                    (sample.finished_at - sample.submitted_at) * 1000.0
                )
                decisions.append(str(sample.result.get("decision", "unknown")))
    finally:
        capture.release()
        worker.shutdown()

    elapsed = time.monotonic() - started
    passed = (
        frames >= 10
        and frames_while_inference_busy >= 5
        and inference_samples >= 1
    )
    print(
        json.dumps(
            {
                "schema": "rootscope.autopilot.async_vision_smoke.v1",
                "passed": passed,
                "physical_authority": False,
                "serial_opened": False,
                "elapsed_s": elapsed,
                "camera_frames": frames,
                "camera_fps_observed": frames / max(elapsed, 0.001),
                "frames_while_inference_busy": frames_while_inference_busy,
                "inference_samples": inference_samples,
                "inference_latencies_ms": inference_latencies_ms,
                "decisions": decisions,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
