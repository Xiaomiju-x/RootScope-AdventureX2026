from __future__ import annotations

import dataclasses
import unittest

from app.config import ProfileConfig, RootScopeConfig
from app.schemas import (
    ActuatorAckEvidence,
    AdmissionStatus,
    ArmCommandContext,
    BaselineEvidence,
    ClearEstopAckEvidence,
    ClearEstopCommandContext,
    CompletionClass,
    ExecutionMode,
    FaultCode,
    MachineState,
    MassEvidence,
    PerceptionSource,
    PhysicalStopEvidence,
    SafetySnapshot,
    StopCommandContext,
    TaskHistoryEntry,
    TaskRequest,
    WettingEvidence,
    Zone,
)
from app.state_machine import RootScopeStateMachine


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


class MemorySink:
    def __init__(self) -> None:
        self.events = []

    def __call__(self, event_type, payload, task_id=None) -> None:
        self.events.append((event_type, payload, task_id))


def make_config(
    *,
    commissioned: bool = True,
    sources=(PerceptionSource.TAG,),
) -> RootScopeConfig:
    return RootScopeConfig(
        commissioning_id="SIM_TEST_COMMISSIONING",
        commissioned=commissioned,
        execution_mode=ExecutionMode.SIMULATION_ONLY,
        required_backend="FAKE_F407",
        protocol_version=1,
        expected_firmware_build_id="SIM_BUILD_1",
        required_firmware_capabilities=("PUMPS", "MASS", "SAFETY"),
        formal_perception_sources=tuple(sources),
        profiles=(
            ProfileConfig(
                profile_id="Profile-A",
                channel=Zone.Z1,
                morphology_label="grass_clump",
                target_mass_mg=2_000,
                tolerance_mg=100,
                hard_timeout_ms=3_000,
                settle_ms=100,
                target_wetting_threshold=0.70,
                neighbor_spill_threshold=0.20,
                minimum_mass_samples=3,
                max_final_mass_span_mg=20,
            ),
            ProfileConfig(
                profile_id="Profile-B",
                channel=Zone.Z2,
                morphology_label="low_shrub",
                target_mass_mg=2_000,
                tolerance_mg=100,
                hard_timeout_ms=3_000,
                settle_ms=100,
                target_wetting_threshold=0.70,
                neighbor_spill_threshold=0.20,
                minimum_mass_samples=3,
                max_final_mass_span_mg=20,
            ),
            ProfileConfig(
                profile_id="Profile-C",
                channel=Zone.Z3,
                morphology_label="young_tree",
                target_mass_mg=2_000,
                tolerance_mg=100,
                hard_timeout_ms=3_000,
                settle_ms=100,
                target_wetting_threshold=0.70,
                neighbor_spill_threshold=0.20,
                minimum_mass_samples=3,
                max_final_mass_span_mg=20,
            ),
        ),
    )


def safety(config: RootScopeConfig, **changes) -> SafetySnapshot:
    values = dict(
        estop_clear=True,
        leak_clear=True,
        cartridge_present=True,
        guard_closed=True,
        heartbeat_fresh=True,
        telemetry_fresh=True,
        scale_stable=True,
        camera_quality_ok=True,
        firmware_protocol_version=config.protocol_version,
        firmware_build_id=config.expected_firmware_build_id,
        firmware_capabilities=config.required_firmware_capabilities,
        execution_backend=config.required_backend,
        firmware_boot_id="BOOT_SIM_1",
        firmware_uptime_ms=1_000,
        lock_latched=False,
        lock_reason="NONE",
        act_enable=True,
        active_wire_task_id=None,
        pump_z1_on=False,
        pump_z2_on=False,
        pump_z3_on=False,
        observed_at_utc="2026-07-15T00:00:00Z",
    )
    values.update(changes)
    return SafetySnapshot(**values)


def preclear_safety(
    config: RootScopeConfig, *, firmware_boot_id: str = "BOOT_SIM_1"
) -> SafetySnapshot:
    return safety(
        config,
        firmware_boot_id=firmware_boot_id,
        lock_latched=True,
        lock_reason="BOOT_LOCK",
        act_enable=False,
    )


def clear_context(
    config: RootScopeConfig,
    *,
    frame_seq: int = 10,
    boot_id: str = "BOOT_SIM_1",
) -> ClearEstopCommandContext:
    return ClearEstopCommandContext(
        frame_seq=frame_seq,
        raw_frame_sha256=HASH_A,
        transcript_id=f"clear-{frame_seq}",
        decoded_command="CLEAR_ESTOP",
        execution_backend=config.required_backend,
        firmware_boot_id=boot_id,
    )


def clear_ack(
    config: RootScopeConfig,
    *,
    frame_seq: int = 10,
    boot_id: str = "BOOT_SIM_1",
) -> ClearEstopAckEvidence:
    return ClearEstopAckEvidence(
        ack_for_type="CLEAR_ESTOP",
        ack_for_seq=frame_seq,
        ack_frame_sha256=HASH_B,
        transcript_id=f"clear-{frame_seq}",
        acked=True,
        fresh=True,
        firmware_build_id=config.expected_firmware_build_id,
        firmware_boot_id=boot_id,
        execution_backend=config.required_backend,
    )


def drive_ready(
    machine: RootScopeStateMachine,
    config: RootScopeConfig,
    *,
    frame_seq: int = 10,
    boot_id: str = "BOOT_SIM_1",
) -> None:
    assert machine.start_self_check().accepted
    assert machine.bind_clear_estop_command_context(
        clear_context(config, frame_seq=frame_seq, boot_id=boot_id),
        preclear_safety(config, firmware_boot_id=boot_id),
        operator_confirmed=True,
    ).accepted
    assert machine.clear_estop_acknowledged(
        clear_ack(config, frame_seq=frame_seq, boot_id=boot_id),
        safety(config, firmware_boot_id=boot_id),
    ).accepted
    assert machine.complete_self_check(
        safety(config, firmware_boot_id=boot_id)
    ).accepted


def task(config: RootScopeConfig, seq: int = 1, task_id: str = "task-0001"):
    profile = config.profile_for("Profile-A")
    return TaskRequest(
        task_id=task_id,
        task_seq=seq,
        profile_id=profile.profile_id,
        channel=profile.channel,
        target_mass_mg=profile.target_mass_mg,
        tolerance_mg=profile.tolerance_mg,
        hard_timeout_ms=profile.hard_timeout_ms,
        config_hash=config.sha256,
        perception_source=PerceptionSource.TAG,
        perception_label=profile.morphology_label,
        perception_score=1.0,
    )


def baseline(request: TaskRequest) -> BaselineEvidence:
    return BaselineEvidence(
        task_id=request.task_id,
        wire_task_id=request.wire_task_id,
        baseline_id="baseline-001",
        camera_frame_id="frame-before-001",
        camera_frame_sha256=HASH_A,
        baseline_mass_mg=500_000,
        mass_sample_count=3,
        mass_last_sample_seq=10,
        mass_sample_digest=HASH_B,
        config_hash=request.config_hash,
        firmware_boot_id="BOOT_SIM_1",
        firmware_uptime_ms_at_capture=100,
        stable=True,
        fresh=True,
        host_captured_monotonic_ms=100,
    )


def ack(request: TaskRequest, frame_seq: int) -> ActuatorAckEvidence:
    return ActuatorAckEvidence(
        task_id=request.task_id,
        wire_task_id=request.wire_task_id,
        ack_for_type="ARM_TASK",
        ack_for_seq=frame_seq,
        ack_frame_sha256=HASH_C,
        transcript_id=f"arm-{request.task_seq}-{frame_seq}",
        channel=request.channel,
        acked=True,
        fresh=True,
        all_other_pumps_off=True,
        firmware_build_id="SIM_BUILD_1",
        firmware_boot_id="BOOT_SIM_1",
        execution_backend="FAKE_F407",
    )


def arm_context(request: TaskRequest, frame_seq: int) -> ArmCommandContext:
    return ArmCommandContext(
        task_id=request.task_id,
        wire_task_id=request.wire_task_id,
        frame_seq=frame_seq,
        raw_frame_sha256=HASH_A,
        transcript_id=f"arm-{request.task_seq}-{frame_seq}",
        decoded_command="ARM_TASK",
        decoded_channel=request.channel,
        decoded_target_mass_mg=request.target_mass_mg,
        decoded_hard_timeout_ms=request.hard_timeout_ms,
        decoded_config_hash_prefix=request.config_hash[:16],
        execution_backend="FAKE_F407",
        firmware_build_id="SIM_BUILD_1",
        firmware_boot_id="BOOT_SIM_1",
    )


def stop_context(
    config: RootScopeConfig,
    *,
    task_id=None,
    wire_task_id=None,
    frame_seq: int = 55,
    boot_id: str = "BOOT_SIM_1",
) -> StopCommandContext:
    return StopCommandContext(
        task_id=task_id,
        wire_task_id=wire_task_id,
        frame_seq=frame_seq,
        raw_frame_sha256=HASH_A,
        transcript_id=f"stop-{frame_seq}",
        decoded_command="EMERGENCY_STOP",
        execution_backend=config.required_backend,
        firmware_build_id=config.expected_firmware_build_id,
        firmware_boot_id=boot_id,
    )


def mass(request: TaskRequest) -> MassEvidence:
    return MassEvidence(
        task_id=request.task_id,
        wire_task_id=request.wire_task_id,
        result_type="TASK_RESULT",
        result_frame_seq=77,
        result_frame_sha256=HASH_C,
        terminal_reason="TARGET_REACHED",
        firmware_build_id="SIM_BUILD_1",
        firmware_boot_id="BOOT_SIM_1",
        execution_backend="FAKE_F407",
        baseline_id="baseline-001",
        baseline_mass_mg=500_000,
        baseline_sample_digest=HASH_B,
        final_mass_mg=498_000,
        final_mass_min_mg=497_995,
        final_mass_max_mg=498_005,
        first_result_sample_seq=11,
        last_result_sample_seq=15,
        sample_count=5,
        post_stop_sample_count=3,
        firmware_completed_uptime_ms=200,
        host_result_received_monotonic_ms=200,
        stable=True,
        task_result_scale_stable=True,
        pumps_all_off=True,
        fresh=True,
    )


def wetting(request: TaskRequest) -> WettingEvidence:
    return WettingEvidence(
        task_id=request.task_id,
        baseline_id="baseline-001",
        baseline_frame_id="frame-before-001",
        baseline_frame_sha256=HASH_A,
        result_frame_id="frame-after-001",
        result_frame_sha256=HASH_C,
        target_score=0.80,
        target_threshold=0.70,
        neighbor_score=0.10,
        spill_threshold=0.20,
        captured_monotonic_ms=300,
        camera_quality_ok=True,
        fresh=True,
    )


class StateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config()
        self.sink = MemorySink()
        self.machine = RootScopeStateMachine(self.config, event_sink=self.sink)

    def ready(self) -> None:
        self.assertEqual(self.machine.state, MachineState.BOOT_LOCKED)
        drive_ready(self.machine, self.config)
        self.assertEqual(self.machine.state, MachineState.READY)

    def start_dosing(self, request: TaskRequest, arm_frame_seq: int = 65_535):
        self.ready()
        self.assertEqual(
            self.machine.admit_task(request, safety(self.config)).status,
            AdmissionStatus.ACCEPTED,
        )
        self.assertTrue(
            self.machine.baseline_captured(
                request.task_id, baseline(request), safety(self.config)
            ).accepted
        )
        self.assertTrue(
            self.machine.bind_arm_command_context(
                request.task_id,
                arm_context(request, arm_frame_seq),
                safety(self.config),
            ).accepted
        )
        dosing_safety = safety(
            self.config,
            active_wire_task_id=request.wire_task_id,
            pump_z1_on=True,
        )
        self.assertTrue(
            self.machine.actuator_acknowledged(
                request.task_id,
                ack(request, arm_frame_seq),
                dosing_safety,
            ).accepted
        )

    def test_positive_flow_and_independent_u16_arm_seq(self) -> None:
        request = task(self.config, seq=1)
        self.start_dosing(request, arm_frame_seq=65_535)
        self.assertEqual(self.machine.state, MachineState.DOSING_Z1)
        self.assertTrue(
            self.machine.dosing_complete(
                request.task_id, mass(request), safety(self.config)
            ).accepted
        )
        too_early = self.machine.begin_verification(
            request.task_id,
            safety(self.config),
            current_monotonic_ms=299,
        )
        self.assertEqual(too_early.fault_code, FaultCode.SETTLING_NOT_COMPLETE)
        self.assertEqual(self.machine.state, MachineState.SETTLING)
        self.assertTrue(
            self.machine.begin_verification(
                request.task_id,
                safety(self.config),
                current_monotonic_ms=300,
            ).accepted
        )
        self.assertTrue(
            self.machine.verification_complete(
                request.task_id, wetting(request), safety(self.config)
            ).accepted
        )
        self.assertEqual(self.machine.state, MachineState.TARGET_WETTING_VERIFIED)
        self.assertEqual(
            self.machine.snapshot().completion_class,
            CompletionClass.TARGET_WETTING_VERIFIED,
        )
        bad_operator = self.machine.close_verified_task(
            safety(self.config),
            operator_confirmed="true",
            cartridge_changed_or_confirmed=True,
        )
        self.assertEqual(
            bad_operator.fault_code, FaultCode.OPERATOR_CONFIRMATION_REQUIRED
        )
        bad_cartridge = self.machine.close_verified_task(
            safety(self.config),
            operator_confirmed=True,
            cartridge_changed_or_confirmed="true",
        )
        self.assertEqual(
            bad_cartridge.fault_code, FaultCode.CARTRIDGE_CHANGE_REQUIRED
        )
        self.assertTrue(
            self.machine.close_verified_task(
                safety(self.config),
                operator_confirmed=True,
                cartridge_changed_or_confirmed=True,
            ).accepted
        )
        self.assertEqual(self.machine.state, MachineState.READY)

        # A global u16 link sequence may wrap independently of uint32 task_seq.
        second = task(self.config, seq=2, task_id="task-0002")
        self.assertEqual(
            self.machine.admit_task(second, safety(self.config)).status,
            AdmissionStatus.ACCEPTED,
        )
        self.assertTrue(
            self.machine.baseline_captured(
                second.task_id,
                dataclasses.replace(
                    baseline(second), baseline_id="baseline-002"
                ),
                safety(self.config),
            ).accepted
        )
        self.assertTrue(
            self.machine.bind_arm_command_context(
                second.task_id,
                arm_context(second, 1),
                safety(self.config),
            ).accepted
        )
        self.assertTrue(
            self.machine.actuator_acknowledged(
                second.task_id,
                ack(second, 1),
                safety(
                    self.config,
                    active_wire_task_id=second.wire_task_id,
                    pump_z1_on=True,
                ),
            ).accepted
        )

    def test_host_and_firmware_clock_domains_may_have_large_offset(self) -> None:
        request = task(self.config)
        self.start_dosing(request, arm_frame_seq=123)
        offset_mass = dataclasses.replace(
            mass(request),
            firmware_completed_uptime_ms=200,
            host_result_received_monotonic_ms=9_000_000_200,
        )
        self.assertTrue(
            self.machine.dosing_complete(
                request.task_id, offset_mass, safety(self.config)
            ).accepted
        )
        too_early = self.machine.begin_verification(
            request.task_id,
            safety(self.config),
            current_monotonic_ms=9_000_000_299,
        )
        self.assertEqual(too_early.fault_code, FaultCode.SETTLING_NOT_COMPLETE)
        self.assertTrue(
            self.machine.begin_verification(
                request.task_id,
                safety(self.config),
                current_monotonic_ms=9_000_000_300,
            ).accepted
        )

    def test_same_boot_firmware_uptime_must_advance_after_baseline(self) -> None:
        request = task(self.config)
        self.start_dosing(request, arm_frame_seq=124)
        non_advancing = dataclasses.replace(
            mass(request), firmware_completed_uptime_ms=100
        )
        result = self.machine.dosing_complete(
            request.task_id, non_advancing, safety(self.config)
        )
        self.assertEqual(result.fault_code, FaultCode.MASS_OUT_OF_RANGE)
        self.assertEqual(self.machine.state, MachineState.ABORTED_LOCKED)

    def test_duplicate_task_is_idempotent_and_conflict_is_rejected(self) -> None:
        self.ready()
        request = task(self.config)
        first = self.machine.admit_task(request, safety(self.config))
        duplicate = self.machine.admit_task(request, safety(self.config))
        self.assertEqual(first.status, AdmissionStatus.ACCEPTED)
        self.assertEqual(duplicate.status, AdmissionStatus.IDEMPOTENT_REPLAY)
        self.assertFalse(duplicate.may_create_physical_command)
        self.assertEqual(
            sum(event[0] == "task_admitted" for event in self.sink.events), 1
        )
        changed = dataclasses.replace(request, target_mass_mg=2_001)
        conflict = self.machine.admit_task(changed, safety(self.config))
        self.assertEqual(conflict.fault_code, FaultCode.TASK_ID_CONFLICT)

    def test_restored_high_watermark_rejects_old_unseen_task(self) -> None:
        old = task(self.config, seq=10, task_id="task-0010")
        history = (TaskHistoryEntry(old.task_id, old.task_seq, old.fingerprint),)
        machine = RootScopeStateMachine(
            self.config, task_history=history, event_sink=MemorySink()
        )
        self.assertEqual(machine.state, MachineState.BOOT_LOCKED)
        drive_ready(machine, self.config)
        stale = task(self.config, seq=9, task_id="task-0009")
        result = machine.admit_task(stale, safety(self.config))
        self.assertEqual(result.fault_code, FaultCode.STALE_TASK)

    def test_ack_must_bind_actual_arm_frame_context(self) -> None:
        request = task(self.config)
        self.ready()
        self.machine.admit_task(request, safety(self.config))
        self.machine.baseline_captured(
            request.task_id, baseline(request), safety(self.config)
        )
        self.machine.bind_arm_command_context(
            request.task_id,
            arm_context(request, 123),
            safety(self.config),
        )
        bad = self.machine.actuator_acknowledged(
            request.task_id,
            ack(request, 124),
            safety(
                self.config,
                active_wire_task_id=request.wire_task_id,
                pump_z1_on=True,
            ),
        )
        self.assertEqual(bad.fault_code, FaultCode.ACK_INVALID)
        self.assertEqual(self.machine.state, MachineState.ABORTED_LOCKED)
        self.assertTrue(self.machine.snapshot().physical_stop_required)
        self.assertFalse(self.machine.snapshot().physical_stop_confirmed)

    def test_f407_hard_timeout_cannot_be_mass_verified(self) -> None:
        request = task(self.config)
        self.start_dosing(request, arm_frame_seq=7)
        timeout_safety = safety(
            self.config,
            lock_latched=True,
            lock_reason="HARD_TIMEOUT",
            act_enable=False,
        )
        result = self.machine.dosing_complete(
            request.task_id, mass(request), timeout_safety
        )
        self.assertEqual(result.fault_code, FaultCode.F407_LOCK_LATCHED)
        snap = self.machine.snapshot()
        self.assertEqual(snap.state, MachineState.ABORTED_LOCKED)
        self.assertEqual(snap.highest_verified_class, CompletionClass.ACTUATOR_ACK)

    def test_mass_and_wetting_must_bind_frozen_baseline(self) -> None:
        request = task(self.config)
        self.start_dosing(request, arm_frame_seq=9)
        forged = dataclasses.replace(mass(request), baseline_mass_mg=501_000)
        result = self.machine.dosing_complete(
            request.task_id, forged, safety(self.config)
        )
        self.assertEqual(result.fault_code, FaultCode.MASS_OUT_OF_RANGE)

    def test_physical_stop_confirmation_is_separate_from_host_lock(self) -> None:
        self.ready()
        self.machine.abort("operator pressed stop")
        truthy_string = self.machine.begin_operator_reset(
            operator_confirmed="true"
        )
        self.assertEqual(
            truthy_string.fault_code, FaultCode.OPERATOR_CONFIRMATION_REQUIRED
        )
        blocked = self.machine.begin_operator_reset(operator_confirmed=True)
        self.assertEqual(blocked.fault_code, FaultCode.PHYSICAL_STOP_UNCONFIRMED)
        wrong_boot = self.machine.bind_stop_command_context(
            stop_context(self.config, boot_id="BOOT_SIM_2")
        )
        self.assertEqual(wrong_boot.fault_code, FaultCode.COMMAND_CONTEXT_INVALID)
        self.assertTrue(
            self.machine.bind_stop_command_context(stop_context(self.config)).accepted
        )
        stop = PhysicalStopEvidence(
            task_id=None,
            wire_task_id=None,
            stop_frame_seq=55,
            stop_raw_frame_sha256=HASH_A,
            ack_frame_sha256=HASH_B,
            transcript_id="stop-55",
            decoded_command="EMERGENCY_STOP",
            ack_for_type="EMERGENCY_STOP",
            ack_for_seq=55,
            acked=True,
            fresh=True,
            pumps_all_off=True,
            hard_power_cut_confirmed=False,
            firmware_build_id=self.config.expected_firmware_build_id,
            firmware_boot_id="BOOT_SIM_1",
            execution_backend=self.config.required_backend,
        )
        wrong_hash = self.machine.confirm_physical_stop(
            dataclasses.replace(stop, stop_raw_frame_sha256=HASH_C)
        )
        self.assertEqual(
            wrong_hash.fault_code, FaultCode.PHYSICAL_STOP_UNCONFIRMED
        )
        self.assertTrue(self.machine.confirm_physical_stop(stop).accepted)
        self.assertTrue(self.machine.snapshot().physical_stop_confirmed)
        self.assertTrue(
            self.machine.begin_operator_reset(operator_confirmed=True).accepted
        )
        # A new F407 boot identity can be pinned only by a new matching CLEAR ACK.
        boot2 = "BOOT_SIM_2"
        self.assertTrue(
            self.machine.bind_clear_estop_command_context(
                clear_context(self.config, frame_seq=56, boot_id=boot2),
                preclear_safety(self.config, firmware_boot_id=boot2),
                operator_confirmed=True,
            ).accepted
        )
        self.assertTrue(
            self.machine.clear_estop_acknowledged(
                clear_ack(self.config, frame_seq=56, boot_id=boot2),
                safety(self.config, firmware_boot_id=boot2),
            ).accepted
        )
        self.assertTrue(
            self.machine.complete_self_check(
                safety(self.config, firmware_boot_id=boot2)
            ).accepted
        )

    def test_bpu_cannot_admit_when_not_qualified(self) -> None:
        self.ready()
        request = dataclasses.replace(
            task(self.config), perception_source=PerceptionSource.BPU
        )
        result = self.machine.admit_task(request, safety(self.config))
        self.assertEqual(result.fault_code, FaultCode.PERCEPTION_NOT_QUALIFIED)

    def test_evidence_sink_is_mandatory(self) -> None:
        machine = RootScopeStateMachine(self.config)
        result = machine.start_self_check()
        self.assertFalse(result.accepted)
        self.assertEqual(machine.state, MachineState.ABORTED_LOCKED)
        self.assertEqual(result.fault_code, FaultCode.EVIDENCE_WRITE_FAILED)

    def test_clear_estop_is_two_phase_and_identity_bound(self) -> None:
        no_ack = RootScopeStateMachine(self.config, event_sink=MemorySink())
        no_ack.start_self_check()
        result = no_ack.complete_self_check(safety(self.config))
        self.assertEqual(result.fault_code, FaultCode.CLEAR_ACK_INVALID)
        self.assertEqual(no_ack.state, MachineState.ABORTED_LOCKED)

        wrong_backend = RootScopeStateMachine(self.config, event_sink=MemorySink())
        wrong_backend.start_self_check()
        result = wrong_backend.bind_clear_estop_command_context(
            dataclasses.replace(
                clear_context(self.config), execution_backend="WRONG_BACKEND"
            ),
            preclear_safety(self.config),
            operator_confirmed=True,
        )
        self.assertEqual(result.fault_code, FaultCode.COMMAND_CONTEXT_INVALID)

        wrong_seq = RootScopeStateMachine(self.config, event_sink=MemorySink())
        wrong_seq.start_self_check()
        self.assertTrue(
            wrong_seq.bind_clear_estop_command_context(
                clear_context(self.config),
                preclear_safety(self.config),
                operator_confirmed=True,
            ).accepted
        )
        result = wrong_seq.clear_estop_acknowledged(
            clear_ack(self.config, frame_seq=11), safety(self.config)
        )
        self.assertEqual(result.fault_code, FaultCode.CLEAR_ACK_INVALID)

        wrong_boot = RootScopeStateMachine(self.config, event_sink=MemorySink())
        wrong_boot.start_self_check()
        self.assertTrue(
            wrong_boot.bind_clear_estop_command_context(
                clear_context(self.config),
                preclear_safety(self.config),
                operator_confirmed=True,
            ).accepted
        )
        result = wrong_boot.clear_estop_acknowledged(
            clear_ack(self.config, boot_id="BOOT_SIM_2"),
            safety(self.config, firmware_boot_id="BOOT_SIM_2"),
        )
        self.assertEqual(result.fault_code, FaultCode.CLEAR_ACK_INVALID)

    def test_arm_context_requires_exact_payload_hash_and_nonzero_seq(self) -> None:
        request = task(self.config)
        self.ready()
        self.machine.admit_task(request, safety(self.config))
        self.machine.baseline_captured(
            request.task_id, baseline(request), safety(self.config)
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(arm_context(request, 1), frame_seq=0)
        with self.assertRaises(ValueError):
            dataclasses.replace(arm_context(request, 1), raw_frame_sha256="bad")
        mismatched_payload = dataclasses.replace(
            arm_context(request, 100),
            decoded_target_mass_mg=request.target_mass_mg + 1,
        )
        result = self.machine.bind_arm_command_context(
            request.task_id, mismatched_payload, safety(self.config)
        )
        self.assertEqual(result.fault_code, FaultCode.COMMAND_CONTEXT_INVALID)
        self.assertEqual(self.machine.state, MachineState.ABORTED_LOCKED)

    def test_task_result_requires_scale_post_stop_samples_and_span(self) -> None:
        bad_variants = (
            {"task_result_scale_stable": False},
            {"post_stop_sample_count": 2},
            {
                "final_mass_min_mg": 497_900,
                "final_mass_max_mg": 498_100,
            },
        )
        for index, changes in enumerate(bad_variants, start=1):
            with self.subTest(changes=changes):
                machine = RootScopeStateMachine(self.config, event_sink=MemorySink())
                drive_ready(machine, self.config, frame_seq=20 + index)
                request = task(
                    self.config,
                    seq=index,
                    task_id=f"task-mass-{index:02d}",
                )
                self.assertEqual(
                    machine.admit_task(request, safety(self.config)).status,
                    AdmissionStatus.ACCEPTED,
                )
                self.assertTrue(
                    machine.baseline_captured(
                        request.task_id, baseline(request), safety(self.config)
                    ).accepted
                )
                frame_seq = 100 + index
                self.assertTrue(
                    machine.bind_arm_command_context(
                        request.task_id,
                        arm_context(request, frame_seq),
                        safety(self.config),
                    ).accepted
                )
                self.assertTrue(
                    machine.actuator_acknowledged(
                        request.task_id,
                        ack(request, frame_seq),
                        safety(
                            self.config,
                            active_wire_task_id=request.wire_task_id,
                            pump_z1_on=True,
                        ),
                    ).accepted
                )
                bad_mass = dataclasses.replace(mass(request), **changes)
                result = machine.dosing_complete(
                    request.task_id, bad_mass, safety(self.config)
                )
                self.assertEqual(result.fault_code, FaultCode.MASS_OUT_OF_RANGE)
                self.assertEqual(machine.state, MachineState.ABORTED_LOCKED)


class StrictSchemaTests(unittest.TestCase):
    def test_truthy_string_cannot_impersonate_boolean(self) -> None:
        config = make_config()
        with self.assertRaises(ValueError):
            safety(config, estop_clear="false")
        with self.assertRaises(ValueError):
            ActuatorAckEvidence(
                task_id="task-0001",
                wire_task_id=1,
                ack_for_type="ARM_TASK",
                ack_for_seq=1,
                ack_frame_sha256=HASH_A,
                transcript_id="arm-1-1",
                channel=Zone.Z1,
                acked="false",
                fresh=True,
                all_other_pumps_off=True,
                firmware_build_id="SIM_BUILD_1",
                firmware_boot_id="BOOT_SIM_1",
                execution_backend="FAKE_F407",
            )

    def test_wetting_scores_are_bounded(self) -> None:
        with self.assertRaises(ValueError):
            dataclasses.replace(wetting(task(make_config())), target_score=100.0)


if __name__ == "__main__":
    unittest.main()
