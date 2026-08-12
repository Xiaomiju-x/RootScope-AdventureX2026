"""Pure in-memory H12 RootScope integration fixture.

This module intentionally has no hardware adapter.  It does not import
``pyserial`` or OpenCV, enumerate ports, open devices, or contact a network.
Its only actuator backend is :class:`FakeF407`; therefore every exported
result remains ``SIMULATED_ONLY`` even when all simulated gates pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .config import RootScopeConfig
from .evidence import (
    EvidenceWriter,
    load_verified_task_history,
    verify_live_ledger,
)
from .schemas import (
    ActuatorAckEvidence,
    AdmissionStatus,
    ArmCommandContext,
    BaselineEvidence,
    ClearEstopAckEvidence,
    ClearEstopCommandContext,
    ExecutionMode,
    MassEvidence,
    PerceptionSource,
    PhysicalStopEvidence,
    SafetySnapshot,
    StopCommandContext,
    TaskRequest,
    WettingEvidence,
    Zone,
)
from .serial import (
    Ack,
    ArmTask,
    Capability,
    CommandType,
    FakeF407,
    LockReason,
    RootScopeSerialLink,
    SafetyBits,
    TerminalReason,
    decode_frame,
)
from .state_machine import RootScopeStateMachine
from .vision import PixelROI, evaluate_frame_quality, verify_wetting_change
from .web.state_store import SnapshotStore


SIMULATION_COMMISSIONING_PREFIX = "SIMULATION_ONLY_"
SIMULATION_MODE = "SIMULATED_ONLY"
PERIOD_S = 0.2
TELEMETRY_FRESH_S = 0.41


class SimulationInvariantError(RuntimeError):
    """Raised when any fixture or state-machine gate fails closed."""


@dataclass(frozen=True)
class FixtureBackendFacts:
    """Facts derived from the only backend this module is allowed to use."""

    backend_id: str = "FAKE_F407"
    backend_actual: str = "fake_f407_in_memory"
    transport_actual: str = "in_process_bytes"
    mode: str = SIMULATION_MODE
    hardware_touched: bool = False
    network_touched: bool = False
    ports_enumerated: bool = False
    physical_completion_claim: bool = False


@dataclass(frozen=True)
class SimulationRunResult:
    report: Mapping[str, Any]
    dashboard_snapshot: Mapping[str, Any]


class ManualClock:
    """Deterministic host/F407 clocks with deliberately separate origins.

    Host receipt timestamps and firmware uptime are different clock domains.
    Keeping independent origins in the fixture makes an accidental numerical
    comparison between them fail loudly in regression tests.
    """

    def __init__(
        self,
        *,
        host_origin_s: float = 0.0,
        firmware_origin_s: float = 0.0,
    ) -> None:
        if host_origin_s < 0 or firmware_origin_s < 0:
            raise ValueError("fixture clock origins cannot be negative")
        self._elapsed_s = 0.0
        self._host_origin_s = float(host_origin_s)
        self._firmware_origin_s = float(firmware_origin_s)
        self._epoch = datetime(2026, 7, 15, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        """Host monotonic clock used for send/receipt/freshness evidence."""

        return self._host_origin_s + self._elapsed_s

    def firmware_monotonic(self) -> float:
        """Independent firmware clock used only by the FakeF407 backend."""

        return self._firmware_origin_s + self._elapsed_s

    def utc(self) -> str:
        value = self._epoch + timedelta(seconds=self._elapsed_s)
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("fixture clock cannot go backwards")
        self._elapsed_s = round(self._elapsed_s + float(seconds), 9)
        return self._elapsed_s


def _zone_to_channel(zone: Zone) -> int:
    return {Zone.Z1: 1, Zone.Z2: 2, Zone.Z3: 3}[zone]


def _lock_reason_name(value: int | LockReason) -> str:
    try:
        return LockReason(int(value)).name
    except ValueError:
        return f"UNKNOWN_{int(value)}"


def _frame_sha256(frame: np.ndarray) -> str:
    array = np.ascontiguousarray(frame)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\x00" + array.tobytes()).hexdigest()


def _mass_sample_digest(samples: list[Mapping[str, int]]) -> str:
    encoded = json.dumps(
        samples,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _transcript_id(raw_frame_sha256: str) -> str:
    return f"tx-{raw_frame_sha256}"


class H12Simulation:
    """Drive one deterministic end-to-end run against an in-memory fake only."""

    def __init__(
        self,
        config: RootScopeConfig,
        evidence_path: Path,
        *,
        store: Optional[SnapshotStore] = None,
        backend: Optional[FixtureBackendFacts] = None,
        host_monotonic_origin_s: float = 0.0,
        firmware_monotonic_origin_s: float = 0.0,
    ) -> None:
        if (
            not config.commissioned
            or config.execution_mode is not ExecutionMode.SIMULATION_ONLY
            or config.required_backend != "FAKE_F407"
            or not config.commissioning_id.startswith(SIMULATION_COMMISSIONING_PREFIX)
        ):
            raise ValueError(
                "H12 fixture requires the explicitly commissioned SIMULATION_ONLY config"
            )
        self.config = config
        self.evidence_path = Path(evidence_path)
        self.store = store or SnapshotStore()
        self.backend = backend or FixtureBackendFacts()
        if (
            self.backend.mode != SIMULATION_MODE
            or self.backend.backend_id != config.required_backend
            or self.backend.hardware_touched
            or self.backend.network_touched
            or self.backend.ports_enumerated
            or self.backend.physical_completion_claim
        ):
            raise ValueError("H12Simulation accepts only a zero-I/O fixture backend")

        self.clock = ManualClock(
            host_origin_s=host_monotonic_origin_s,
            firmware_origin_s=firmware_monotonic_origin_s,
        )
        self.writer = EvidenceWriter(
            self.evidence_path,
            clock=self.clock.utc,
            initialize=not self.evidence_path.exists(),
        )
        history = load_verified_task_history(self.evidence_path)
        self.machine = RootScopeStateMachine(
            config,
            task_history=history,
            event_sink=self.writer,
            clock=self.clock.utc,
        )
        self.link = RootScopeSerialLink(clock=self.clock.monotonic)
        self.fake = FakeF407(clock=self.clock.firmware_monotonic)
        self._last_telemetry_at: Optional[float] = None
        self._last_safety_at: Optional[float] = None
        self._heartbeat_times: list[float] = []
        self._telemetry_samples: list[dict[str, int]] = []
        self._max_active_pumps = 0

    def _ingest(self, wire: bytes) -> None:
        events = self.link.ingest(wire, now=self.clock.monotonic())
        for event in events:
            if event.kind == "telemetry":
                self._last_telemetry_at = event.received_at
                telemetry = event.value
                pump_mask = int(getattr(telemetry, "pump_mask"))
                self._max_active_pumps = max(
                    self._max_active_pumps, int(pump_mask).bit_count()
                )
                self._telemetry_samples.append(
                    {
                        "at_ms": int(round(event.received_at * 1000)),
                        "wire_task_id": int(getattr(telemetry, "task_id")),
                        "sample_seq": int(getattr(telemetry, "sample_seq")),
                        "pump_mask": pump_mask,
                        "filtered_mass_mg": int(
                            getattr(telemetry, "filtered_mass_mg")
                        ),
                        "safety_bits": int(getattr(telemetry, "safety_bits")),
                        "uptime_ms": int(getattr(telemetry, "uptime_ms")),
                    }
                )
            elif event.kind == "safety":
                self._last_safety_at = event.received_at

    def _exchange(self, wire: bytes) -> None:
        self.link.mark_command_sent(
            wire,
            execution_backend=self.backend.backend_id,
            now=self.clock.monotonic(),
        )
        reply = self.fake.exchange(wire, now=self.clock.firmware_monotonic())
        self._ingest(reply)

    def _tick(self) -> None:
        self._ingest(self.fake.tick(now=self.clock.firmware_monotonic()))
        if not self.fake.at_most_one_pump:
            raise SimulationInvariantError("fake F407 enabled multiple pumps")

    def _heartbeat(self) -> None:
        now = self.clock.monotonic()
        if not self.link.heartbeat_due(now):
            raise SimulationInvariantError("heartbeat emitted faster than 5 Hz")
        self._exchange(self.link.make_heartbeat(now=now))
        self._heartbeat_times.append(now)

    def _periodic_step(self) -> None:
        self.clock.advance(PERIOD_S)
        self._heartbeat()
        self._tick()

    def _core_safety(
        self,
        *,
        camera_quality_ok: bool,
        expected_wire_task_id: Optional[int],
    ) -> SafetySnapshot:
        """Map raw fixture state into the narrower core safety contract.

        ``LOCK_LATCHED`` and a non-zero lock reason revoke ``estop_clear`` and
        ``heartbeat_fresh``.  A mismatched F407 active task id revokes
        ``telemetry_fresh``.  This deliberately loses availability rather than
        allowing an unknown/locked actuator state through the core gate.
        """

        info = self.link.firmware_info
        safety = self.link.last_safety
        telemetry = self.link.last_telemetry
        if info is None or safety is None or telemetry is None:
            raise SimulationInvariantError("identity/safety/telemetry is incomplete")

        now = self.clock.monotonic()
        safety_age = (
            float("inf")
            if self._last_safety_at is None
            else now - self._last_safety_at
        )
        telemetry_age = (
            float("inf")
            if self._last_telemetry_at is None
            else now - self._last_telemetry_at
        )
        safety_bits = SafetyBits(safety.safety_bits)
        telemetry_bits = SafetyBits(telemetry.safety_bits)
        fault_bits = safety_bits | telemetry_bits
        good_bits = safety_bits & telemetry_bits
        lock_latched = bool(fault_bits & SafetyBits.LOCK_LATCHED)
        lock_reason = int(safety.lock_reason)
        fixture_state_consistent = bool(
            lock_latched == self.fake.locked
            and lock_reason == int(self.fake.lock_reason)
            and int(telemetry.pump_mask) == int(self.fake.pump_mask)
        )
        expected_task_on_wire = expected_wire_task_id or 0
        task_identity_ok = int(telemetry.task_id) == expected_task_on_wire
        act_enable = bool(good_bits & SafetyBits.ACT_ENABLE)
        watchdog_fresh = bool(good_bits & SafetyBits.WATCHDOG_FRESH)
        identity_fresh = self.link.identity_fresh(now)
        boot_token = self.link.firmware_boot_id_token
        if boot_token is None:
            raise SimulationInvariantError("firmware boot identity is unavailable")
        telemetry_fresh = bool(
            0.0 <= telemetry_age <= TELEMETRY_FRESH_S
            and 0.0 <= safety_age <= TELEMETRY_FRESH_S
            and task_identity_ok
            and fixture_state_consistent
        )

        snapshot = SafetySnapshot(
            estop_clear=not bool(fault_bits & SafetyBits.ESTOP_ACTIVE),
            leak_clear=not bool(fault_bits & SafetyBits.LEAK_DETECTED),
            cartridge_present=bool(good_bits & SafetyBits.CARTRIDGE_PRESENT),
            guard_closed=bool(good_bits & SafetyBits.GUARD_CLOSED),
            heartbeat_fresh=bool(watchdog_fresh and identity_fresh),
            telemetry_fresh=telemetry_fresh,
            scale_stable=bool(good_bits & SafetyBits.HX711_VALID),
            camera_quality_ok=bool(camera_quality_ok),
            firmware_protocol_version=int(info.protocol_version),
            firmware_build_id=str(info.build_id),
            firmware_capabilities=tuple(
                capability.name.lower()
                for capability in Capability
                if int(info.capabilities) & int(capability)
            ),
            execution_backend=self.backend.backend_id,
            firmware_boot_id=boot_token,
            firmware_uptime_ms=int(telemetry.uptime_ms),
            lock_latched=lock_latched,
            lock_reason=_lock_reason_name(lock_reason),
            act_enable=act_enable,
            active_wire_task_id=(
                int(telemetry.task_id) if int(telemetry.task_id) != 0 else None
            ),
            pump_z1_on=bool(int(telemetry.pump_mask) & 0b001),
            pump_z2_on=bool(int(telemetry.pump_mask) & 0b010),
            pump_z3_on=bool(int(telemetry.pump_mask) & 0b100),
            observed_at_utc=self.clock.utc(),
        )
        self.writer.append(
            "fixture_serial_gate",
            {
                "backend_actual": self.backend.backend_actual,
                "hardware_touched": self.backend.hardware_touched,
                "lock_latched": lock_latched,
                "lock_reason": _lock_reason_name(lock_reason),
                "fake_active_task_id": (
                    self.fake.active_task.task_id if self.fake.active_task else 0
                ),
                "telemetry_task_id": int(telemetry.task_id),
                "expected_wire_task_id": expected_task_on_wire,
                "task_identity_ok": task_identity_ok,
                "fixture_state_consistent": fixture_state_consistent,
                "act_enable": act_enable,
                "max_active_pumps_observed": self._max_active_pumps,
                "mapped_core_safety": snapshot.to_dict(),
            },
            self.machine.snapshot().active_task.task_id
            if self.machine.snapshot().active_task
            else None,
        )
        return snapshot

    @staticmethod
    def _fixture_frames(
        target_zone: Zone,
    ) -> tuple[np.ndarray, PixelROI, tuple[PixelROI, ...]]:
        height, width = 480, 640
        y, x = np.indices((height, width))
        checker = (((x // 12 + y // 12) % 2) * 100 + 70).astype(np.uint8)
        baseline = np.stack(
            (checker, np.clip(checker + 8, 0, 255), np.clip(checker + 16, 0, 255)),
            axis=2,
        ).astype(np.uint8)
        rois = {
            Zone.Z1: PixelROI("Z1", 220, 40, 200, 100),
            Zone.Z2: PixelROI("Z2", 220, 190, 200, 100),
            Zone.Z3: PixelROI("Z3", 220, 340, 200, 100),
        }
        target = rois[target_zone]
        neighbors = tuple(roi for zone, roi in rois.items() if zone is not target_zone)
        return baseline, target, neighbors

    @staticmethod
    def _fixture_result_frame(baseline: np.ndarray, target: PixelROI) -> np.ndarray:
        """Create the after-frame only when the simulated dosing phase is over."""

        result = baseline.copy()
        ys, xs = target.slices()
        result[ys, xs] = np.clip(result[ys, xs].astype(np.int16) - 55, 0, 255).astype(
            np.uint8
        )
        return result

    @staticmethod
    def _must(accepted: bool, detail: str) -> None:
        if not accepted:
            raise SimulationInvariantError(detail)

    def _fail_safe(self, exc: BaseException) -> None:
        """Transmit E-stop first; bind state/evidence only after the send.

        The safety-directional frame must not depend on the host state machine
        or evidence writer being healthy.  Receipt binding remains best effort
        and its failure can never be described as a confirmed physical stop.
        """

        stop_transmitted = False
        stop_confirmed = False
        stop_send_error: Optional[str] = None
        stop_mark_error: Optional[str] = None
        stop_ingest_error: Optional[str] = None
        stop_evidence_error: Optional[str] = None
        stop_seq: Optional[int] = None
        stop_frame_sha256: Optional[str] = None
        stop_wire: Optional[bytes] = None
        reply = b""

        # P0 ordering: generate and deliver before abort/bind/log calls.  A
        # failure in host-side send bookkeeping cannot suppress the frame.
        try:
            stop_wire = self.link.make_emergency_stop(now=self.clock.monotonic())
            stop_frame = decode_frame(stop_wire)
            stop_seq = int.from_bytes(stop_frame.payload, "little")
            stop_frame_sha256 = hashlib.sha256(stop_wire).hexdigest()
        except Exception as stop_exc:
            stop_send_error = f"{type(stop_exc).__name__}: {stop_exc}"

        if stop_wire is not None:
            try:
                self.link.mark_command_sent(
                    stop_wire,
                    execution_backend=self.backend.backend_id,
                    now=self.clock.monotonic(),
                )
            except Exception as mark_exc:
                stop_mark_error = f"{type(mark_exc).__name__}: {mark_exc}"
            try:
                reply = self.fake.exchange(
                    stop_wire, now=self.clock.firmware_monotonic()
                )
                stop_transmitted = True
            except Exception as stop_exc:
                stop_send_error = f"{type(stop_exc).__name__}: {stop_exc}"
            if stop_transmitted:
                try:
                    self._ingest(reply)
                except Exception as ingest_exc:
                    stop_ingest_error = (
                        f"{type(ingest_exc).__name__}: {ingest_exc}"
                    )

        # Host lock/evidence can fail, but only after the lower-controller stop.
        try:
            self.machine.abort(f"simulation fixture failure: {type(exc).__name__}")
        except Exception:
            pass

        try:
            if not stop_transmitted or stop_seq is None:
                raise SimulationInvariantError("EMERGENCY_STOP was not transmitted")
            if self.link.firmware_info is None:
                raise SimulationInvariantError("stop context has no firmware identity")
            command = self.link.command_receipt_for(
                CommandType.EMERGENCY_STOP, stop_seq
            )
            if (
                command is None
                or command.firmware_build_id is None
                or command.firmware_boot_id is None
            ):
                raise SimulationInvariantError("stop command receipt is incomplete")
            active = self.machine.snapshot().active_task
            transcript_id = _transcript_id(command.raw_frame_sha256)
            context = StopCommandContext(
                task_id=active.task_id if active else None,
                wire_task_id=active.wire_task_id if active else None,
                frame_seq=stop_seq,
                raw_frame_sha256=command.raw_frame_sha256,
                transcript_id=transcript_id,
                decoded_command="EMERGENCY_STOP",
                execution_backend=self.backend.backend_id,
                firmware_build_id=command.firmware_build_id,
                firmware_boot_id=command.firmware_boot_id,
            )
            if not self.machine.bind_stop_command_context(context).accepted:
                raise SimulationInvariantError("core rejected exact stop context")
            self._periodic_step()
            ack_receipt = self.link.command_ack_receipt_for(
                CommandType.EMERGENCY_STOP, stop_seq
            )
            if ack_receipt is None:
                raise SimulationInvariantError("stop ACK receipt is missing")
            stop_evidence = PhysicalStopEvidence(
                task_id=context.task_id,
                wire_task_id=context.wire_task_id,
                stop_frame_seq=context.frame_seq,
                stop_raw_frame_sha256=context.raw_frame_sha256,
                ack_frame_sha256=ack_receipt.ack_raw_frame_sha256,
                transcript_id=transcript_id,
                decoded_command="EMERGENCY_STOP",
                ack_for_type="EMERGENCY_STOP",
                ack_for_seq=stop_seq,
                acked=self.link.ack_ok(
                    ack_receipt.ack,
                    expected_type=CommandType.EMERGENCY_STOP,
                ),
                fresh=(
                    0.0
                    <= self.clock.monotonic() - ack_receipt.received_at
                    <= TELEMETRY_FRESH_S
                ),
                pumps_all_off=(
                    self.fake.pump_mask == 0
                    and self.link.last_telemetry is not None
                    and self.link.last_telemetry.pump_mask == 0
                ),
                hard_power_cut_confirmed=False,
                firmware_build_id=ack_receipt.firmware_build_id,
                firmware_boot_id=ack_receipt.firmware_boot_id,
                execution_backend=ack_receipt.execution_backend,
                observed_at_utc=self.clock.utc(),
            )
            stop_confirmed = self.machine.confirm_physical_stop(
                stop_evidence
            ).accepted
        except Exception as evidence_exc:
            stop_evidence_error = (
                f"{type(evidence_exc).__name__}: {evidence_exc}"
            )
        try:
            self.writer.append(
                "simulation_failed",
                {
                    **asdict(self.backend),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "pump_mask_after_stop": int(self.fake.pump_mask),
                    "emergency_stop_transmitted": stop_transmitted,
                    "stop_frame_seq": stop_seq,
                    "stop_frame_sha256": stop_frame_sha256,
                    "stop_send_error": stop_send_error,
                    "stop_mark_error": stop_mark_error,
                    "stop_ingest_error": stop_ingest_error,
                    "stop_evidence_error": stop_evidence_error,
                    "exact_stop_evidence_confirmed": stop_confirmed,
                    "unconfirmed_stop_is_insufficient": not stop_confirmed,
                },
            )
        except Exception:
            pass

    def run(
        self,
        *,
        profile_id: str = "Profile-B-SIM",
        task_seq: Optional[int] = None,
    ) -> SimulationRunResult:
        history = self.machine.task_history()
        next_sequence = (history[-1].task_seq + 1) if history else 1
        if task_seq is None:
            task_seq = next_sequence
        if task_seq < next_sequence:
            raise ValueError(f"task_seq must be >= {next_sequence}")
        profile = self.config.profile_for(profile_id)
        task_id = f"sim-task-{task_seq:010d}"
        baseline_id = f"sim-baseline-{task_seq:010d}"

        self.writer.append(
            "simulation_run_started",
            {
                **asdict(self.backend),
                "config_hash": self.config.sha256,
                "commissioning_id": self.config.commissioning_id,
                "task_seq": task_seq,
                "formal_input_on_core_contract": PerceptionSource.TAG.value,
                "perception_input_actual": "synthetic_tag_fixture",
                "physical_perception_claim": False,
            },
            task_id,
        )

        try:
            # Boot lock -> identity -> 5 Hz heartbeat.  CLEAR_ESTOP is a
            # separate two-phase core gate: bind exact bytes, send once, then
            # bind the exact ACK plus a fresh post-clear snapshot.
            self._exchange(self.link.make_firmware_query())
            self._tick()
            self._heartbeat()
            self._periodic_step()
            if not self.link.identity_valid:
                raise SimulationInvariantError(self.link.identity_error)
            if not self.fake.locked:
                raise SimulationInvariantError("fake F407 did not preserve its boot lock")

            baseline, target_roi, neighbor_rois = self._fixture_frames(
                profile.channel
            )
            baseline_quality = evaluate_frame_quality(baseline)
            self._must(baseline_quality.passed, str(baseline_quality.reasons))

            preclear_safety = self._core_safety(
                camera_quality_ok=True, expected_wire_task_id=None
            )
            self._must(
                self.machine.start_self_check().accepted,
                "state machine rejected startup self-check",
            )

            clear_wire = self.link.make_clear_estop(now=self.clock.monotonic())
            clear_frame = decode_frame(clear_wire)
            clear_seq = int.from_bytes(clear_frame.payload, "little")
            clear_command_receipt = self.link.command_receipt_for(
                CommandType.CLEAR_ESTOP, clear_seq
            )
            if clear_command_receipt is None or clear_command_receipt.firmware_boot_id is None:
                raise SimulationInvariantError("CLEAR_ESTOP command receipt is missing")
            clear_transcript_id = _transcript_id(
                clear_command_receipt.raw_frame_sha256
            )
            clear_context = ClearEstopCommandContext(
                frame_seq=clear_seq,
                raw_frame_sha256=clear_command_receipt.raw_frame_sha256,
                transcript_id=clear_transcript_id,
                decoded_command="CLEAR_ESTOP",
                execution_backend=self.backend.backend_id,
                firmware_boot_id=clear_command_receipt.firmware_boot_id,
            )
            self._must(
                self.machine.bind_clear_estop_command_context(
                    clear_context,
                    preclear_safety,
                    operator_confirmed=True,
                ).accepted,
                "state machine rejected CLEAR_ESTOP command context",
            )
            self._exchange(clear_wire)
            self._periodic_step()
            postclear_safety = self._core_safety(
                camera_quality_ok=True, expected_wire_task_id=None
            )
            clear_ack_receipt = self.link.command_ack_receipt_for(
                CommandType.CLEAR_ESTOP, clear_seq
            )
            if clear_ack_receipt is None:
                raise SimulationInvariantError("CLEAR_ESTOP ACK receipt is missing")
            clear_ack = ClearEstopAckEvidence(
                ack_for_type="CLEAR_ESTOP",
                ack_for_seq=clear_seq,
                ack_frame_sha256=clear_ack_receipt.ack_raw_frame_sha256,
                transcript_id=clear_transcript_id,
                acked=self.link.ack_ok(
                    clear_ack_receipt.ack,
                    expected_type=CommandType.CLEAR_ESTOP,
                ),
                fresh=(
                    0.0
                    <= self.clock.monotonic() - clear_ack_receipt.received_at
                    <= TELEMETRY_FRESH_S
                ),
                firmware_build_id=clear_ack_receipt.firmware_build_id,
                firmware_boot_id=clear_ack_receipt.firmware_boot_id,
                execution_backend=clear_ack_receipt.execution_backend,
                observed_at_utc=self.clock.utc(),
            )
            self._must(
                self.machine.clear_estop_acknowledged(
                    clear_ack, postclear_safety
                ).accepted,
                "state machine rejected CLEAR_ESTOP ACK/post-clear state",
            )
            self._must(
                self.machine.complete_self_check(postclear_safety).accepted,
                "state machine rejected safe fixture identity",
            )
            ready_safety = postclear_safety

            request = TaskRequest(
                task_id=task_id,
                task_seq=task_seq,
                profile_id=profile.profile_id,
                channel=profile.channel,
                target_mass_mg=profile.target_mass_mg,
                tolerance_mg=profile.tolerance_mg,
                hard_timeout_ms=profile.hard_timeout_ms,
                config_hash=self.config.sha256,
                perception_source=PerceptionSource.TAG,
                perception_label=profile.morphology_label,
                perception_score=1.0,
                created_at_utc=self.clock.utc(),
            )
            admission = self.machine.admit_task(request, ready_safety)
            self._must(
                admission.status is AdmissionStatus.ACCEPTED,
                f"task admission failed: {admission.fault_code.value} {admission.detail}",
            )
            baseline_sample_start = len(self._telemetry_samples)
            for _ in range(profile.minimum_mass_samples):
                self._periodic_step()
            baseline_samples = self._telemetry_samples[baseline_sample_start:]
            self._must(
                len(baseline_samples) == profile.minimum_mass_samples,
                "baseline did not receive the frozen number of 5 Hz samples",
            )
            self._must(
                all(
                    sample["wire_task_id"] == 0 and sample["pump_mask"] == 0
                    for sample in baseline_samples
                ),
                "baseline samples were not idle/pump-off",
            )
            self._must(
                len({sample["filtered_mass_mg"] for sample in baseline_samples}) == 1,
                "baseline mass samples were not stable",
            )
            baseline_mass_mg = baseline_samples[-1]["filtered_mass_mg"]
            baseline_frame_id = f"sim-before-{task_seq:010d}"
            baseline_sha256 = _frame_sha256(baseline)
            baseline_evidence = BaselineEvidence(
                task_id=task_id,
                wire_task_id=request.wire_task_id,
                baseline_id=baseline_id,
                camera_frame_id=baseline_frame_id,
                camera_frame_sha256=baseline_sha256,
                baseline_mass_mg=baseline_mass_mg,
                mass_sample_count=len(baseline_samples),
                mass_last_sample_seq=baseline_samples[-1]["sample_seq"],
                mass_sample_digest=_mass_sample_digest(baseline_samples),
                config_hash=self.config.sha256,
                firmware_boot_id=str(self.link.firmware_boot_id_token),
                firmware_uptime_ms_at_capture=baseline_samples[-1]["uptime_ms"],
                stable=True,
                fresh=True,
                host_captured_monotonic_ms=int(
                    round(self.clock.monotonic() * 1000)
                ),
                observed_at_utc=self.clock.utc(),
            )
            baseline_safety = self._core_safety(
                camera_quality_ok=True, expected_wire_task_id=None
            )
            self._must(
                self.machine.baseline_captured(
                    task_id, baseline_evidence, baseline_safety
                ).accepted,
                "baseline admission failed",
            )

            # Construct exactly one ARM frame, decode its actual u16 frame seq,
            # bind that separately from the persistent uint32 wire task id, then
            # and only then deliver the bytes to FakeF407.
            arm_wire = self.link.make_arm_task(
                task_id=request.wire_task_id,
                channel=_zone_to_channel(profile.channel),
                target_mass_mg=profile.target_mass_mg,
                hard_timeout_ms=profile.hard_timeout_ms,
                config_hash_prefix=bytes.fromhex(self.config.sha256[:16]),
                now=self.clock.monotonic(),
            )
            arm_message = ArmTask.from_payload(decode_frame(arm_wire).payload)
            self._must(
                arm_message.task_id == request.wire_task_id,
                "ARM frame task id changed during encoding",
            )
            arm_command_receipt = self.link.command_receipt_for(
                CommandType.ARM_TASK, arm_message.seq
            )
            if arm_command_receipt is None:
                raise SimulationInvariantError("ARM command receipt is missing")
            if (
                arm_command_receipt.firmware_build_id is None
                or arm_command_receipt.firmware_boot_id is None
            ):
                raise SimulationInvariantError("ARM command firmware identity is missing")
            arm_transcript_id = _transcript_id(
                arm_command_receipt.raw_frame_sha256
            )
            arm_context = ArmCommandContext(
                task_id=task_id,
                wire_task_id=request.wire_task_id,
                frame_seq=arm_message.seq,
                raw_frame_sha256=arm_command_receipt.raw_frame_sha256,
                transcript_id=arm_transcript_id,
                decoded_command="ARM_TASK",
                decoded_channel=profile.channel,
                decoded_target_mass_mg=arm_message.target_mass_mg,
                decoded_hard_timeout_ms=arm_message.hard_timeout_ms,
                decoded_config_hash_prefix=arm_message.config_hash_prefix.hex(),
                execution_backend=self.backend.backend_id,
                firmware_build_id=arm_command_receipt.firmware_build_id,
                firmware_boot_id=arm_command_receipt.firmware_boot_id,
            )
            self._must(
                self.machine.bind_arm_command_context(
                    task_id, arm_context, baseline_safety
                ).accepted,
                "state machine rejected ARM frame context",
            )
            self._exchange(arm_wire)
            arm_ack_receipt = self.link.command_ack_receipt_for(
                CommandType.ARM_TASK, arm_message.seq
            )
            if arm_ack_receipt is None:
                raise SimulationInvariantError("ARM ACK receipt is missing")
            arm_ack: Optional[Ack] = arm_ack_receipt.ack
            arm_ack_ok = self.link.ack_ok(
                arm_ack,
                expected_type=CommandType.ARM_TASK,
                expected_task_id=request.wire_task_id,
            )
            self._must(
                arm_ack_ok,
                "fresh task-bound F407 ARM ACK missing",
            )

            # First post-ARM periodic sample is the independent pump/task proof.
            self._periodic_step()
            dosing_safety = self._core_safety(
                camera_quality_ok=True,
                expected_wire_task_id=request.wire_task_id,
            )
            expected_mask = 1 << (_zone_to_channel(profile.channel) - 1)
            self._must(
                int(self.link.last_telemetry.pump_mask) == expected_mask,
                "wrong or multiple pump telemetry after ARM",
            )
            ack_evidence = ActuatorAckEvidence(
                task_id=task_id,
                wire_task_id=request.wire_task_id,
                ack_for_type="ARM_TASK",
                ack_for_seq=arm_message.seq,
                ack_frame_sha256=arm_ack_receipt.ack_raw_frame_sha256,
                transcript_id=arm_transcript_id,
                channel=profile.channel,
                acked=arm_ack_ok,
                fresh=(
                    0.0
                    <= self.clock.monotonic()
                    - arm_ack_receipt.received_at
                    <= TELEMETRY_FRESH_S
                ),
                all_other_pumps_off=(
                    self.link.last_telemetry is not None
                    and int(self.link.last_telemetry.pump_mask) == expected_mask
                    and int(self.link.last_telemetry.pump_mask).bit_count() == 1
                ),
                firmware_build_id=str(self.link.firmware_info.build_id),
                firmware_boot_id=arm_ack_receipt.firmware_boot_id,
                execution_backend=arm_ack_receipt.execution_backend,
                observed_at_utc=self.clock.utc(),
            )
            self._must(
                self.machine.actuator_acknowledged(
                    task_id, ack_evidence, dosing_safety
                ).accepted,
                "state machine rejected independent actuator ACK/telemetry",
            )

            # Continue exact 5 Hz supervision until the firmware authors a
            # TASK_RESULT.  Host-side telemetry observations are not promoted
            # to terminal mass evidence.
            arm_started_at = self.clock.monotonic()
            result_receipt = self.link.task_result_for(request.wire_task_id)
            while result_receipt is None:
                self._periodic_step()
                result_receipt = self.link.task_result_for(request.wire_task_id)
                if (
                    self.clock.monotonic() - arm_started_at
                    > profile.hard_timeout_ms / 1000.0 + 2.0
                ):
                    raise SimulationInvariantError(
                        "firmware did not produce a fresh terminal TASK_RESULT"
                    )

            terminal = result_receipt.result
            self._must(
                terminal.task_id == request.wire_task_id
                and terminal.terminal_reason == int(TerminalReason.TARGET_REACHED)
                and terminal.scale_stable
                and terminal.sample_count >= profile.minimum_mass_samples
                and terminal.final_window_span_mg <= profile.max_final_mass_span_mg
                and terminal.pump_mask == 0
                and not bool(terminal.safety_bits & int(SafetyBits.LOCK_LATCHED))
                and terminal.baseline_mass_mg == baseline_evidence.baseline_mass_mg
                and terminal.first_sample_seq
                > baseline_evidence.mass_last_sample_seq
                and terminal.firmware_completed_uptime_ms
                > baseline_evidence.firmware_uptime_ms_at_capture,
                "firmware TASK_RESULT identity/stability/ordering gate failed",
            )
            self._must(
                result_receipt.firmware_boot_id == self.link.firmware_boot_id_token
                and result_receipt.firmware_build_id
                == str(self.link.firmware_info.build_id),
                "TASK_RESULT firmware identity does not match the live session",
            )

            # TASK_RESULT follows the fifth task-bound post-stop telemetry in
            # the same tick.  One further 5 Hz sample proves the live snapshot
            # has returned to task_id=0 while keeping the receipt fresh.
            self._periodic_step()
            final_mass_mg = int(terminal.final_mass_mg)
            final_safety = self._core_safety(
                camera_quality_ok=True, expected_wire_task_id=None
            )
            mass = MassEvidence(
                task_id=task_id,
                wire_task_id=request.wire_task_id,
                result_type="TASK_RESULT",
                result_frame_seq=result_receipt.result_frame_seq,
                result_frame_sha256=result_receipt.raw_frame_sha256,
                terminal_reason=result_receipt.terminal_reason,
                firmware_build_id=result_receipt.firmware_build_id,
                firmware_boot_id=result_receipt.firmware_boot_id,
                execution_backend=self.backend.backend_id,
                baseline_id=baseline_evidence.baseline_id,
                baseline_mass_mg=baseline_evidence.baseline_mass_mg,
                baseline_sample_digest=baseline_evidence.mass_sample_digest,
                final_mass_mg=final_mass_mg,
                final_mass_min_mg=terminal.final_window_min_mg,
                final_mass_max_mg=terminal.final_window_max_mg,
                first_result_sample_seq=terminal.first_sample_seq,
                last_result_sample_seq=terminal.last_sample_seq,
                sample_count=terminal.sample_count,
                post_stop_sample_count=terminal.post_stop_sample_count,
                firmware_completed_uptime_ms=(
                    terminal.firmware_completed_uptime_ms
                ),
                host_result_received_monotonic_ms=int(
                    round(result_receipt.received_at * 1000)
                ),
                stable=terminal.scale_stable,
                task_result_scale_stable=terminal.scale_stable,
                pumps_all_off=terminal.pump_mask == 0
                and int(self.link.last_telemetry.pump_mask) == 0,
                fresh=(
                    0.0
                    <= self.clock.monotonic() - result_receipt.received_at
                    <= TELEMETRY_FRESH_S
                ),
                observed_at_utc=self.clock.utc(),
            )
            self._must(
                self.machine.dosing_complete(task_id, mass, final_safety).accepted,
                "state machine rejected stable mass-loss evidence",
            )
            verification_started_ms = int(round(self.clock.monotonic() * 1000))
            self._must(
                self.machine.begin_verification(
                    task_id,
                    final_safety,
                    current_monotonic_ms=verification_started_ms,
                ).accepted,
                "state machine rejected the frozen settling interval",
            )
            # The result image is created only after dosing.  Settling itself
            # remains a frozen monotonic gate owned by the state machine/config.
            result_frame = self._fixture_result_frame(baseline, target_roi)
            result_quality = evaluate_frame_quality(result_frame)
            self._must(result_quality.passed, str(result_quality.reasons))
            wetting_result = verify_wetting_change(
                baseline, result_frame, target_roi, neighbor_rois
            )
            self._must(wetting_result.passed, str(wetting_result.reasons))
            wetting = WettingEvidence(
                task_id=task_id,
                baseline_id=baseline_id,
                baseline_frame_id=baseline_frame_id,
                baseline_frame_sha256=baseline_sha256,
                result_frame_id=f"sim-after-{task_seq:010d}",
                result_frame_sha256=_frame_sha256(result_frame),
                target_score=wetting_result.target_changed_fraction,
                target_threshold=profile.target_wetting_threshold,
                neighbor_score=wetting_result.max_neighbor_changed_fraction,
                spill_threshold=profile.neighbor_spill_threshold,
                captured_monotonic_ms=int(round(self.clock.monotonic() * 1000)),
                camera_quality_ok=baseline_quality.passed and result_quality.passed,
                fresh=True,
                observed_at_utc=self.clock.utc(),
            )
            self._must(
                self.machine.verification_complete(
                    task_id, wetting, final_safety
                ).accepted,
                "state machine rejected wetting evidence",
            )

            heartbeat_gaps = [
                round(b - a, 9)
                for a, b in zip(self._heartbeat_times, self._heartbeat_times[1:])
            ]
            self._must(
                bool(self._heartbeat_times)
                and all(abs(gap - PERIOD_S) <= 1e-9 for gap in heartbeat_gaps),
                "heartbeat supervision was not exactly 5 Hz",
            )
            self._must(
                self._max_active_pumps <= 1 and self.fake.at_most_one_pump,
                "pump mutual exclusion failed",
            )

            machine_snapshot = self.machine.snapshot()
            summary = {
                "schema_version": "rootscope.h12-simulation.v1",
                **asdict(self.backend),
                "task_id": task_id,
                "task_seq": task_seq,
                "wire_task_id_u32": request.wire_task_id,
                "arm_frame_seq_u16": arm_message.seq,
                "sequence_domains_bound_separately": True,
                "simulated_pipeline_state": machine_snapshot.state.value,
                "simulated_internal_completion_class": machine_snapshot.completion_class.value,
                "exported_completion_class": SIMULATION_MODE,
                "baseline_mass_mg": baseline_mass_mg,
                "final_mass_mg": final_mass_mg,
                "mass_loss_mg": mass.mass_loss_mg,
                "target_mass_mg": profile.target_mass_mg,
                "firmware_post_stop_sample_count": terminal.sample_count,
                "firmware_final_window_min_mg": terminal.final_window_min_mg,
                "firmware_final_window_max_mg": terminal.final_window_max_mg,
                "firmware_final_window_span_mg": terminal.final_window_span_mg,
                "task_result_frame_sha256": result_receipt.raw_frame_sha256,
                "heartbeat_hz": 5.0,
                "heartbeat_tx_count": len(self._heartbeat_times),
                "heartbeat_gaps_s": heartbeat_gaps,
                "max_active_pumps_observed": self._max_active_pumps,
                "final_pump_mask": int(self.fake.pump_mask),
                "firmware_identity_valid": self.link.identity_valid,
                "firmware_build_id": str(self.link.firmware_info.build_id),
                "final_f407_lock_latched": self.fake.locked,
                "final_f407_lock_reason": _lock_reason_name(self.fake.lock_reason),
                "wetting": wetting_result.to_dict(),
                "baseline_quality": baseline_quality.to_dict(),
                "result_quality": result_quality.to_dict(),
                "physical_completion_claim": False,
                "claim_scope": "IN_MEMORY_PROTOCOL_AND_FIXED_IMAGE_FIXTURE_ONLY",
            }
            self.writer.append("simulation_summary", summary, task_id)
            verified = verify_live_ledger(self.evidence_path).require_valid()
            summary = {
                **summary,
                "evidence_record_count": verified.record_count,
                "evidence_terminal_hash": verified.terminal_hash,
            }

            dashboard = {
                "schema_version": "rootscope.dashboard.v1",
                "generated_at": self.clock.utc(),
                "mode": self.backend.mode,
                "state": machine_snapshot.state.value,
                "backend_actual": self.backend.backend_actual,
                "transport_actual": self.backend.transport_actual,
                "hardware_touched": self.backend.hardware_touched,
                "network_touched": self.backend.network_touched,
                "ports_enumerated": self.backend.ports_enumerated,
                "physical_completion_claim": self.backend.physical_completion_claim,
                "perception": {
                    "source": "synthetic_tag_fixture",
                    "class_id": profile.morphology_label,
                    "confidence": 1.0,
                    "qualified": False,
                    "scope": "SIMULATION_ONLY",
                },
                "task": {
                    "task_id": task_id,
                    "channel": profile.channel.value,
                    "profile": profile.profile_id,
                    "completion_class": SIMULATION_MODE,
                    "simulated_gate_state": machine_snapshot.completion_class.value,
                },
                "safety": {
                    "firmware_identity": self.link.identity_valid,
                    "heartbeat_fresh": final_safety.heartbeat_fresh,
                    "estop_ok": final_safety.estop_clear,
                    "leak_ok": final_safety.leak_clear,
                    "cartridge_ok": final_safety.cartridge_present,
                    "guard_ok": final_safety.guard_closed,
                    "mass_stable": mass.stable,
                    "camera_fresh": wetting.fresh,
                },
                "f407_diagnostics": {
                    "lock_latched": self.fake.locked,
                    "lock_reason": _lock_reason_name(self.fake.lock_reason),
                    "active_task_id": (
                        self.fake.active_task.task_id if self.fake.active_task else 0
                    ),
                },
                "mass": {
                    "target_g": profile.target_mass_mg / 1000.0,
                    "measured_loss_g": mass.mass_loss_mg / 1000.0,
                    "samples": [
                        terminal.final_window_min_mg / 1000.0,
                        terminal.final_window_max_mg / 1000.0,
                    ],
                },
                "wetting": {
                    "passed": wetting_result.passed,
                    "target_changed_fraction": wetting_result.target_changed_fraction,
                    "reasons": list(wetting_result.reasons),
                    "scope": wetting_result.evidence_scope,
                },
                "evidence": {
                    "head_hash": verified.terminal_hash,
                    "record_count": verified.record_count,
                },
                "alerts": [
                    "LOCAL_FIXTURE_ONLY_NO_HARDWARE_TOUCHED",
                    "SIMULATED_GATE_PASS_NOT_PHYSICAL_COMPLETION",
                    "NO_REAL_SERIAL_PORT_OPENED_OR_ENUMERATED",
                ],
            }
            self.store.replace(dashboard)
            return SimulationRunResult(summary, self.store.snapshot())
        except BaseException as exc:
            self._fail_safe(exc)
            raise


def run_simulated_once(
    evidence_path: Path,
    config_path: Path,
    *,
    profile_id: str = "Profile-B-SIM",
    task_seq: Optional[int] = None,
) -> SimulationRunResult:
    """Public one-shot API.  The selected config must be simulation-only."""

    config = RootScopeConfig.from_json_file(Path(config_path))
    return H12Simulation(config, Path(evidence_path)).run(
        profile_id=profile_id, task_seq=task_seq
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one local fixture cycle")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "h12_simulation_config.json",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=project_root / "evidence" / "local_h12" / "simulation_run.jsonl",
    )
    parser.add_argument("--profile", default="Profile-B-SIM")
    parser.add_argument("--task-seq", type=int)
    parser.add_argument(
        "--report",
        type=Path,
        help="optional new JSON report path; existing files are never overwritten",
    )
    args = parser.parse_args()
    if not args.once:
        parser.error("this zero-I/O fixture currently supports only --once")
    result = run_simulated_once(
        args.evidence,
        args.config,
        profile_id=args.profile,
        task_seq=args.task_seq,
    )
    output = json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(output + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
