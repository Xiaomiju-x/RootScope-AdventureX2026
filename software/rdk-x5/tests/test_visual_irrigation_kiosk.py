from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOTSCOPE_ROOT = Path(__file__).resolve().parents[1]
if str(ROOTSCOPE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROOTSCOPE_ROOT))

from tools.x5_visual_irrigation_kiosk import is_single_frame_target  # noqa: E402


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


class SingleFrameDualEvidenceTests(unittest.TestCase):
    def test_target_passes_in_one_frame_when_both_branches_pass(self) -> None:
        self.assertTrue(is_single_frame_target(result()))

    def test_cnn_only_does_not_actuate(self) -> None:
        self.assertFalse(
            is_single_frame_target(result(template_pass=False))
        )

    def test_template_only_does_not_actuate(self) -> None:
        self.assertFalse(is_single_frame_target(result(cnn_pass=False)))

    def test_non_target_never_actuates(self) -> None:
        self.assertFalse(
            is_single_frame_target(result(decision="non_target"))
        )

    def test_unknown_or_hold_never_actuates(self) -> None:
        self.assertFalse(
            is_single_frame_target(
                result(state="HOLD_LOW_CONFIDENCE", decision="unknown")
            )
        )


if __name__ == "__main__":
    unittest.main()
