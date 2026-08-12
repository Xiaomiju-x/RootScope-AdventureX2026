from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


DATASET_TOOLS = Path(__file__).resolve().parents[1]
if str(DATASET_TOOLS) not in sys.path:
    sys.path.insert(0, str(DATASET_TOOLS))
SCRIPT = DATASET_TOOLS / "audit_machine_curated_provisional_v2.py"
SPEC = importlib.util.spec_from_file_location("audit_machine_curated_provisional_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

WORKSPACE = DATASET_TOOLS.parents[1]
PACK = WORKSPACE / "datasets" / MODULE.v2.OUTPUT_NAME


class AuditMachineCuratedProvisionalV2Tests(unittest.TestCase):
    def test_expected_print_ids_are_frozen(self) -> None:
        self.assertEqual(
            {38233728, 74079996, 66745979, 94700516, 75760716, 98911085},
            MODULE.EXPECTED_PRINT_PAGEIDS,
        )

    @unittest.skipUnless(PACK.is_dir(), "generated v2 pack is not present")
    def test_generated_pack_passes_independent_audit(self) -> None:
        report = MODULE.audit(WORKSPACE, PACK)
        self.assertEqual("PASS", report["status"], report["checks_failed"])
        self.assertEqual(0, report["failure_count"])
        self.assertGreaterEqual(report["check_count"], 25)


if __name__ == "__main__":
    unittest.main()
