from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.config import (
    MAX_HARD_TIMEOUT_MS,
    MAX_TARGET_MASS_MG,
    MIN_HARD_TIMEOUT_MS,
    MIN_TARGET_MASS_MG,
    ProfileConfig,
    RootScopeConfig,
)
from app.schemas import ExecutionMode, FaultCode, MachineState, Zone
from app.serial import fake_f407

from tests.test_state_machine import MemorySink, safety


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_runtime_example_is_physical_but_strictly_uncommissioned(self) -> None:
        config = RootScopeConfig.from_json_file(
            ROOT / "configs" / "runtime_config.example.json"
        )
        self.assertFalse(config.commissioned)
        self.assertEqual(config.execution_mode, ExecutionMode.PHYSICAL)
        self.assertNotEqual(config.required_backend, "FAKE_F407")

        from app.state_machine import RootScopeStateMachine

        machine = RootScopeStateMachine(config, event_sink=MemorySink())
        machine.start_self_check()
        result = machine.complete_self_check(safety(config))
        self.assertFalse(result.accepted)
        self.assertEqual(result.fault_code, FaultCode.CONFIG_MISMATCH)
        self.assertEqual(machine.state, MachineState.ABORTED_LOCKED)

    def test_simulation_config_is_explicitly_fake_only(self) -> None:
        config = RootScopeConfig.from_json_file(
            ROOT / "configs" / "h12_simulation_config.json"
        )
        self.assertEqual(config.execution_mode, ExecutionMode.SIMULATION_ONLY)
        self.assertEqual(config.required_backend, "FAKE_F407")

    def test_protocol_limits_match_fake_and_icd(self) -> None:
        self.assertEqual(MIN_TARGET_MASS_MG, fake_f407.MIN_TARGET_MASS_MG)
        self.assertEqual(MAX_TARGET_MASS_MG, fake_f407.MAX_TARGET_MASS_MG)
        self.assertEqual(MIN_HARD_TIMEOUT_MS, fake_f407.MIN_HARD_TIMEOUT_MS)
        self.assertEqual(MAX_HARD_TIMEOUT_MS, fake_f407.MAX_HARD_TIMEOUT_MS)
        icd = json.loads((ROOT / "configs" / "serial_icd.json").read_text("utf-8"))
        arm = icd.get("commands", {}).get("ARM_TASK")
        self.assertIsNotNone(arm, "serial_icd.json must define commands.ARM_TASK")
        self.assertEqual(
            arm["target_mass_mg_range"],
            [MIN_TARGET_MASS_MG, MAX_TARGET_MASS_MG],
        )
        self.assertEqual(
            arm["hard_timeout_ms_range"],
            [MIN_HARD_TIMEOUT_MS, MAX_HARD_TIMEOUT_MS],
        )

    def test_profile_rejects_values_serial_cannot_encode(self) -> None:
        with self.assertRaises(ValueError):
            ProfileConfig(
                profile_id="Profile-X",
                channel=Zone.Z1,
                morphology_label="fixture",
                target_mass_mg=MIN_TARGET_MASS_MG - 1,
                tolerance_mg=0,
                hard_timeout_ms=MIN_HARD_TIMEOUT_MS,
                settle_ms=0,
                target_wetting_threshold=0.5,
                neighbor_spill_threshold=0.5,
                minimum_mass_samples=1,
                max_final_mass_span_mg=0,
            )


if __name__ == "__main__":
    unittest.main()
