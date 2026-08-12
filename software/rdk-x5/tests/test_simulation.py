from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from app.config import RootScopeConfig
from app.evidence import verify_live_ledger
from app.evidence.verifier import read_verified_records
from app.simulation import (
    FixtureBackendFacts,
    H12Simulation,
    run_simulated_once,
)
from app.serial.frame import AckReason


ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG = ROOT / "configs" / "h12_simulation_config.json"
RUNTIME_CONFIG = ROOT / "configs" / "runtime_config.example.json"


class SimulationIsolationTests(unittest.TestCase):
    def test_physical_runtime_config_is_rejected_before_evidence_write(self) -> None:
        config = RootScopeConfig.from_json_file(RUNTIME_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "must_not_exist.jsonl"
            with self.assertRaisesRegex(ValueError, "SIMULATION_ONLY"):
                H12Simulation(config, evidence)
            self.assertFalse(evidence.exists())


class SimulationClosureTests(unittest.TestCase):
    def test_independently_offset_host_and_firmware_clocks_close(self) -> None:
        config = RootScopeConfig.from_json_file(SIM_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "offset-clocks.jsonl"
            fixture = H12Simulation(
                config,
                evidence,
                host_monotonic_origin_s=1_000_000.0,
                firmware_monotonic_origin_s=123.0,
            )
            result = fixture.run(task_seq=69_999)
            self.assertEqual(
                result.report["simulated_pipeline_state"],
                "TARGET_WETTING_VERIFIED",
            )
            self.assertEqual(
                result.report["exported_completion_class"], "SIMULATED_ONLY"
            )
            self.assertTrue(verify_live_ledger(evidence).valid)

    def test_fixture_closes_only_as_simulated_with_bound_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "simulation.jsonl"
            result = run_simulated_once(
                evidence,
                SIM_CONFIG,
                task_seq=70_000,
            )
            report = result.report
            dashboard = result.dashboard_snapshot
            self.assertEqual(report["simulated_pipeline_state"], "TARGET_WETTING_VERIFIED")
            self.assertEqual(report["exported_completion_class"], "SIMULATED_ONLY")
            self.assertFalse(report["hardware_touched"])
            self.assertFalse(report["network_touched"])
            self.assertFalse(report["ports_enumerated"])
            self.assertFalse(report["physical_completion_claim"])
            self.assertTrue(report["sequence_domains_bound_separately"])
            self.assertEqual(report["wire_task_id_u32"], 70_000)
            self.assertNotEqual(
                report["wire_task_id_u32"], report["arm_frame_seq_u16"]
            )
            self.assertEqual(report["mass_loss_mg"], report["target_mass_mg"])
            self.assertGreaterEqual(report["firmware_post_stop_sample_count"], 5)
            self.assertLessEqual(report["max_active_pumps_observed"], 1)
            self.assertEqual(report["final_pump_mask"], 0)
            self.assertEqual(report["heartbeat_hz"], 5.0)
            self.assertTrue(all(gap == 0.2 for gap in report["heartbeat_gaps_s"]))
            self.assertEqual(dashboard["mode"], "SIMULATED_ONLY")
            self.assertEqual(dashboard["task"]["completion_class"], "SIMULATED_ONLY")
            self.assertFalse(dashboard["physical_completion_claim"])
            self.assertTrue(all(dashboard["safety"].values()))
            self.assertEqual(dashboard["f407_diagnostics"]["lock_reason"], "NONE")
            self.assertTrue(verify_live_ledger(evidence).valid)
            records = read_verified_records(evidence)
            by_type = {record["event_type"]: record for record in records}
            self.assertIn("clear_estop_command_context_bound", by_type)
            self.assertIn("clear_estop_acknowledged", by_type)
            self.assertIn("arm_command_context_bound", by_type)
            self.assertIn("actuator_ack", by_type)
            self.assertIn("mass_loss_verified", by_type)
            clear_context = by_type["clear_estop_command_context_bound"]["payload"][
                "command_context"
            ]
            clear_ack = by_type["clear_estop_acknowledged"]["payload"]["ack"]
            self.assertEqual(clear_context["transcript_id"], clear_ack["transcript_id"])
            arm_context = by_type["arm_command_context_bound"]["payload"][
                "command_context"
            ]
            actuator_ack = by_type["actuator_ack"]["payload"]
            self.assertEqual(arm_context["transcript_id"], actuator_ack["transcript_id"])
            self.assertEqual(
                by_type["mass_loss_verified"]["payload"]["result_frame_sha256"],
                report["task_result_frame_sha256"],
            )

            # A restart must reuse the live high watermark rather than reusing
            # an F407 wire task id.
            second = run_simulated_once(evidence, SIM_CONFIG)
            self.assertEqual(second.report["task_seq"], 70_001)
            self.assertTrue(verify_live_ledger(evidence).valid)

    def test_failure_uses_exact_stop_context_and_remains_locked(self) -> None:
        config = RootScopeConfig.from_json_file(SIM_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "failed.jsonl"
            fixture = H12Simulation(config, evidence)
            with mock.patch.object(
                fixture.machine,
                "verification_complete",
                side_effect=RuntimeError("injected verification failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    fixture.run(task_seq=81_000)
            snapshot = fixture.machine.snapshot()
            self.assertEqual(snapshot.state.value, "ABORTED_LOCKED")
            self.assertTrue(snapshot.physical_stop_required)
            self.assertTrue(snapshot.physical_stop_confirmed)
            self.assertEqual(fixture.fake.pump_mask, 0)
            records = read_verified_records(evidence)
            failure = records[-1]
            self.assertEqual(failure["event_type"], "simulation_failed")
            self.assertTrue(
                failure["payload"]["exact_stop_evidence_confirmed"]
            )
            self.assertTrue(verify_live_ledger(evidence).valid)

    def test_estop_reaches_f407_before_stop_evidence_binding_failure(self) -> None:
        config = RootScopeConfig.from_json_file(SIM_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "failed-binding.jsonl"
            fixture = H12Simulation(config, evidence)
            with mock.patch.object(
                fixture.machine,
                "verification_complete",
                side_effect=RuntimeError("injected verification failure"),
            ), mock.patch.object(
                fixture.machine,
                "bind_stop_command_context",
                side_effect=RuntimeError("injected stop evidence failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected verification"):
                    fixture.run(task_seq=82_000)

            # The FakeF407 has already consumed the E-stop even though the
            # host evidence binder rejected the post-send transcript.
            self.assertTrue(fixture.fake.locked)
            self.assertEqual(fixture.fake.pump_mask, 0)
            self.assertEqual(
                fixture.link.last_ack.reason,
                int(AckReason.EMERGENCY_STOP),
            )
            records = read_verified_records(evidence)
            failure = records[-1]
            self.assertEqual(failure["event_type"], "simulation_failed")
            self.assertTrue(failure["payload"]["emergency_stop_transmitted"])
            self.assertFalse(failure["payload"]["exact_stop_evidence_confirmed"])
            self.assertIn(
                "injected stop evidence failure",
                failure["payload"]["stop_evidence_error"],
            )
            self.assertTrue(verify_live_ledger(evidence).valid)

    def test_estop_send_survives_host_send_bookkeeping_failure(self) -> None:
        config = RootScopeConfig.from_json_file(SIM_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "failed-mark.jsonl"
            fixture = H12Simulation(config, evidence)
            with mock.patch.object(
                fixture.link,
                "mark_command_sent",
                side_effect=RuntimeError("injected send bookkeeping failure"),
            ):
                fixture._fail_safe(RuntimeError("injected preflight failure"))

            self.assertTrue(fixture.fake.locked)
            self.assertEqual(fixture.fake.pump_mask, 0)
            self.assertEqual(
                fixture.link.last_ack.reason,
                int(AckReason.EMERGENCY_STOP),
            )
            failure = read_verified_records(evidence)[-1]
            self.assertTrue(failure["payload"]["emergency_stop_transmitted"])
            self.assertIn(
                "injected send bookkeeping failure",
                failure["payload"]["stop_mark_error"],
            )
            self.assertTrue(verify_live_ledger(evidence).valid)

    def test_backend_attestation_mismatch_is_rejected_before_write(self) -> None:
        config = RootScopeConfig.from_json_file(SIM_CONFIG)
        mismatched = FixtureBackendFacts(backend_id="ROOTSCOPE_F407_SERIAL_V1")
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "must_not_exist.jsonl"
            with self.assertRaisesRegex(ValueError, "zero-I/O fixture backend"):
                H12Simulation(config, evidence, backend=mismatched)
            self.assertFalse(evidence.exists())


if __name__ == "__main__":
    unittest.main()
