from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np


ROOTSCOPE_ROOT = Path(__file__).resolve().parents[1]
if str(ROOTSCOPE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROOTSCOPE_ROOT))

from tools.x5_visual_irrigation_autopilot import (  # noqa: E402
    LatestFrameInference,
    SceneChangeGate,
    immediate_target,
)


def result(
    *,
    state: str = "CONFIRMED_DUAL_EVIDENCE",
    decision: str = "grass_clump",
    cnn_pass: bool = True,
    template_pass: bool = True,
) -> dict:
    return {
        "state": state,
        "decision": decision,
        "cnn": {"pass": cnn_pass},
        "template": {"pass": template_pass},
    }


class ImmediateTargetTests(unittest.TestCase):
    def test_one_confirmed_result_triggers_immediately(self) -> None:
        self.assertEqual(immediate_target(result()), "grass_clump")

    def test_failed_evidence_does_not_trigger(self) -> None:
        self.assertIsNone(immediate_target(result(template_pass=False)))

    def test_non_target_does_not_trigger(self) -> None:
        self.assertIsNone(immediate_target(result(decision="non_target")))


class SceneChangeGateTests(unittest.TestCase):
    def test_same_card_cannot_automatically_retry(self) -> None:
        gate = SceneChangeGate("grass_clump", 0.5)
        self.assertEqual(gate.observe(result(), 10.0), (False, None))
        self.assertEqual(gate.observe(result(), 20.0), (False, None))

    def test_clear_scene_rearms(self) -> None:
        gate = SceneChangeGate("grass_clump", 0.5)
        clear = result(
            state="HOLD_LOW_CONFIDENCE",
            decision="unknown",
            cnn_pass=False,
            template_pass=False,
        )
        self.assertEqual(gate.observe(clear, 10.0), (False, None))
        self.assertEqual(gate.observe(clear, 10.5), (True, None))

    def test_different_confirmed_target_rearms_and_triggers_immediately(self) -> None:
        gate = SceneChangeGate("grass_clump", 0.5)
        shrub = result(decision="low_shrub")
        self.assertEqual(gate.observe(shrub, 10.0), (True, "low_shrub"))


class LatestFrameInferenceTests(unittest.TestCase):
    def test_inference_does_not_block_gui_caller_and_preserves_frame(self) -> None:
        release = threading.Event()

        class Runtime:
            def infer(self, frame):
                release.wait(timeout=2.0)
                return {"decision": "grass_clump", "pixel": int(frame[0, 0, 0])}

        worker = LatestFrameInference(Runtime())
        frame = np.full((4, 4, 3), 17, dtype=np.uint8)
        started = time.monotonic()
        self.assertTrue(worker.submit(frame, started))
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertFalse(worker.submit(frame, time.monotonic()))
        frame[:] = 99
        self.assertIsNone(worker.poll(time.monotonic()))
        release.set()
        sample = None
        deadline = time.monotonic() + 2.0
        while sample is None and time.monotonic() < deadline:
            sample = worker.poll(time.monotonic())
            time.sleep(0.005)
        worker.shutdown()
        self.assertIsNotNone(sample)
        self.assertEqual(sample.result["pixel"], 17)
        self.assertEqual(int(sample.frame[0, 0, 0]), 17)


if __name__ == "__main__":
    unittest.main()
