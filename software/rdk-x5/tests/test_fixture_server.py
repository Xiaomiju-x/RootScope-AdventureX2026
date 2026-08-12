from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.web.fixture_server import FixtureActions, build_fixture_server
from app.web.state_store import SnapshotStore


ROOT = Path(__file__).resolve().parents[1]


class FixtureActionsTests(unittest.TestCase):
    def test_start_runs_only_simulated_fixture_and_updates_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore()
            actions = FixtureActions(
                store,
                config_path=ROOT / "configs" / "h12_simulation_config.json",
                evidence_path=Path(tmp) / "dashboard.jsonl",
            )
            result = actions.start({"profile": "Profile-A-SIM"})
            snapshot = store.snapshot()
            self.assertEqual(result["mode"], "SIMULATED_ONLY")
            self.assertFalse(result["hardware_touched"])
            self.assertFalse(result["physical_completion_claim"])
            self.assertEqual(snapshot["mode"], "SIMULATED_ONLY")
            self.assertEqual(snapshot["state"], "TARGET_WETTING_VERIFIED")
            self.assertEqual(snapshot["task"]["completion_class"], "SIMULATED_ONLY")
            self.assertFalse(snapshot["physical_completion_claim"])

    def test_reset_is_view_only_and_unknown_input_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore({"mode": "SIMULATED_ONLY", "state": "READY"})
            actions = FixtureActions(
                store,
                config_path=ROOT / "configs" / "h12_simulation_config.json",
                evidence_path=Path(tmp) / "dashboard.jsonl",
            )
            with self.assertRaises(ValueError):
                actions.start({"task_seq": 1})
            reset = actions.reset_view({})
            self.assertTrue(reset["view_only_reset"])
            self.assertEqual(store.snapshot()["state"], "BOOT_LOCKED")

    def test_fixture_server_refuses_non_loopback_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "only on loopback"):
                build_fixture_server(
                    host="0.0.0.0",
                    port=0,
                    config_path=ROOT / "configs" / "h12_simulation_config.json",
                    evidence_path=Path(tmp) / "dashboard.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
