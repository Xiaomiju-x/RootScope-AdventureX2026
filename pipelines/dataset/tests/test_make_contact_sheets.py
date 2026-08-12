from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS / "make_contact_sheets.py"
SPEC = importlib.util.spec_from_file_location("make_contact_sheets", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MakeContactSheetsTests(unittest.TestCase):
    def test_review_label_defaults_to_manual_review_wording(self) -> None:
        with mock.patch.object(sys, "argv", [str(SCRIPT)]):
            args = MODULE.parse_args()
        self.assertEqual("pending manual review", args.review_label)

    def test_machine_only_review_label_can_be_explicitly_selected(self) -> None:
        label = "MACHINE ONLY - PENDING MACHINE SCREEN - NOT HUMAN REVIEWED"
        with mock.patch.object(
            sys,
            "argv",
            [str(SCRIPT), "--review-label", label],
        ):
            args = MODULE.parse_args()
        self.assertEqual(label, args.review_label)


if __name__ == "__main__":
    unittest.main()
