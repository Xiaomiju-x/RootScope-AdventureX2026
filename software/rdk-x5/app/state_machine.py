"""Deterministic, fail-closed RootScope orchestration state machine.

The class never opens a serial port and never drives a pump.  It is the only
component allowed to *authorise progression* toward such a command.  Callers
must treat a rejected result, an exception, or ``ABORTED_LOCKED`` as an order
to keep all pump outputs off.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from .config import RootScopeConfig
from .schemas import (
    ActuatorAckEvidence,
    AdmissionResult,
    AdmissionStatus,
    ArmCommandContext,
    BaselineEvidence,
    ClearEstopAckEvidence,
    ClearEstopCommandContext,
    CompletionClass,
    FaultCode,
    MachineState,
    MassEvidence,
    OperationResult,
    PhysicalStopEvidence,
    SafetySnapshot,
    StateSnapshot,
    StopCommandContext,
    TaskHistoryEntry,
    TaskRequest,
    WettingEvidence,
    Zone,
    utc_now_iso,
)


EventSink = Callable[[str, Mapping[str, Any], Optional[str]], None]


_DOSING_STATE = {
    Zone.Z1: MachineState.DOSING_Z1,
    Zone.Z2: MachineState.DOSING_Z2,
    Zone.Z3: MachineState.DOSING_Z3,
}
_ACTIVE_TASK_STATES = {
    MachineState.TARGET_IDENTIFIED,
    MachineState.BASELINE_CAPTURED,
    MachineState.DOSING_Z1,
    MachineState.DOSING_Z2,
    MachineState.DOSING_Z3,
    MachineState.SETTLING,
    MachineState.VERIFYING,
}


class RootScopeStateMachine:
    def __init__(
        self,
        config: RootScopeConfig,
        *,
        task_history: Iterable[TaskHistoryEntry] = (),
        event_sink: Optional[EventSink] = None,
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        self.config = config
        self._event_sink = event_sink
        self._clock = clock
        self._lock = threading.RLock()
        self._boot_session_id = str(uuid.uuid4())

        self._state = MachineState.BOOT_LOCKED
        self._completion = CompletionClass.SIMULATED_ONLY
        self._highest_verified = CompletionClass.SIMULATED_ONLY
        self._active_task: Optional[TaskRequest] = None
        self._baseline: Optional[BaselineEvidence] = None
        self._dosing_result_received_monotonic_ms: Optional[int] = None
        self._verification_started_monotonic_ms: Optional[int] = None
        self._pending_arm_context: Optional[ArmCommandContext] = None
        self._pending_clear_context: Optional[ClearEstopCommandContext] = None
        self._pending_stop_context: Optional[StopCommandContext] = None
        self._clear_estop_acknowledged = False
        self._f407_boot_id: Optional[str] = None
        self._physical_stop_required = False
        self._physical_stop_confirmed = False
        self._last_fault = FaultCode.RESTART_RECOVERY
        self._fault_detail = "startup/restart requires explicit self-check"
        self._history_by_id: Dict[str, TaskHistoryEntry] = {}
        self._history_by_seq: Dict[int, TaskHistoryEntry] = {}
        self._high_watermark = 0

        last_sequence = 0
        for entry in task_history:
            if entry.task_id in self._history_by_id:
                raise ValueError(f"duplicate task_id in restored history: {entry.task_id}")
            if entry.task_seq in self._history_by_seq:
                raise ValueError(f"duplicate task_seq in restored history: {entry.task_seq}")
            if entry.task_seq <= last_sequence:
                raise ValueError("restored task history must be strictly increasing")
            self._history_by_id[entry.task_id] = entry
            self._history_by_seq[entry.task_seq] = entry
            self._high_watermark = entry.task_seq
            last_sequence = entry.task_seq

    @property
    def state(self) -> MachineState:
        return self._state

    def snapshot(self) -> StateSnapshot:
        with self._lock:
            return StateSnapshot(
                state=self._state,
                completion_class=self._completion,
                highest_verified_class=self._highest_verified,
                active_task=self._active_task,
                last_fault=self._last_fault,
                fault_detail=self._fault_detail,
                high_watermark_task_seq=self._high_watermark,
                pending_arm_frame_seq=(
                    self._pending_arm_context.frame_seq
                    if self._pending_arm_context
                    else None
                ),
                pending_clear_frame_seq=(
                    self._pending_clear_context.frame_seq
                    if self._pending_clear_context
                    else None
                ),
                pending_stop_frame_seq=(
                    self._pending_stop_context.frame_seq
                    if self._pending_stop_context
                    else None
                ),
                clear_estop_acknowledged=self._clear_estop_acknowledged,
                physical_stop_required=self._physical_stop_required,
                physical_stop_confirmed=self._physical_stop_confirmed,
                boot_session_id=self._boot_session_id,
            )

    def task_history(self) -> Tuple[TaskHistoryEntry, ...]:
        with self._lock:
            return tuple(
                self._history_by_seq[sequence]
                for sequence in sorted(self._history_by_seq)
            )

    def _result(
        self,
        accepted: bool,
        fault: FaultCode = FaultCode.NONE,
        detail: str = "",
    ) -> OperationResult:
        return OperationResult(
            accepted=accepted,
            state=self._state,
            completion_class=self._completion,
            fault_code=fault,
            detail=detail,
        )

    def _emit(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        task_id: Optional[str] = None,
    ) -> bool:
        if self._event_sink is None:
            self._state = MachineState.ABORTED_LOCKED
            self._completion = CompletionClass.ABORTED_LOCKED
            self._last_fault = FaultCode.EVIDENCE_WRITE_FAILED
            self._fault_detail = "no persistent evidence sink configured"
            return False
        try:
            self._event_sink(event_type, payload, task_id)
            return True
        except Exception as exc:  # evidence failure is a safety decision, not recovery
            self._state = MachineState.ABORTED_LOCKED
            self._completion = CompletionClass.ABORTED_LOCKED
            self._last_fault = FaultCode.EVIDENCE_WRITE_FAILED
            self._fault_detail = f"evidence sink failed: {type(exc).__name__}: {exc}"
            return False

    def _transition(
        self,
        target: MachineState,
        reason: str,
        *,
        task_id: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        payload: Dict[str, Any] = {
            "boot_session_id": self._boot_session_id,
            "config_hash": self.config.sha256,
            "from_state": self._state.value,
            "to_state": target.value,
            "reason": reason,
            "transitioned_at_utc": self._clock(),
        }
        if extra:
            payload["detail"] = dict(extra)
        if not self._emit("state_transition", payload, task_id):
            return False
        self._state = target
        return True

    def _safety_problem(
        self,
        safety: SafetySnapshot,
        *,
        require_pumps_off: bool,
        require_camera: bool,
        require_scale: bool,
        expected_active_pump: Optional[Zone] = None,
        expected_active_wire_task_id: Optional[int] = None,
        allow_boot_change: bool = False,
    ) -> Tuple[FaultCode, str]:
        if not safety.estop_clear:
            return FaultCode.ESTOP_ACTIVE, "emergency stop is active"
        if not safety.leak_clear:
            return FaultCode.LEAK_DETECTED, "leak input is active"
        if not safety.cartridge_present:
            return FaultCode.CARTRIDGE_MISSING, "cartridge is not present"
        if not safety.guard_closed:
            return FaultCode.GUARD_OPEN, "guard/interlock is open"
        if not safety.heartbeat_fresh:
            return FaultCode.HEARTBEAT_STALE, "F407 heartbeat is stale"
        if not safety.telemetry_fresh:
            return FaultCode.TELEMETRY_STALE, "F407 telemetry is stale"
        if require_scale and not safety.scale_stable:
            return FaultCode.SCALE_UNSTABLE, "scale is not stable"
        if require_camera and not safety.camera_quality_ok:
            return FaultCode.CAMERA_QUALITY_INVALID, "camera quality gate failed"
        if safety.firmware_protocol_version != self.config.protocol_version:
            return (
                FaultCode.FIRMWARE_IDENTITY_INVALID,
                "firmware protocol version does not match config",
            )
        if safety.firmware_build_id != self.config.expected_firmware_build_id:
            return (
                FaultCode.FIRMWARE_IDENTITY_INVALID,
                "firmware build id does not match config",
            )
        if safety.execution_backend != self.config.required_backend:
            return (
                FaultCode.FIRMWARE_IDENTITY_INVALID,
                "execution backend attestation does not match config",
            )
        if (
            not allow_boot_change
            and self._f407_boot_id is not None
            and safety.firmware_boot_id != self._f407_boot_id
        ):
            return (
                FaultCode.FIRMWARE_REBOOTED,
                "F407 boot identity changed during the host session",
            )
        missing_capabilities = set(self.config.required_firmware_capabilities) - set(
            safety.firmware_capabilities
        )
        if missing_capabilities:
            return (
                FaultCode.FIRMWARE_CAPABILITY_MISSING,
                f"missing capabilities: {sorted(missing_capabilities)}",
            )
        if safety.lock_latched:
            return (
                FaultCode.F407_LOCK_LATCHED,
                f"F407 lock is latched: {safety.lock_reason}",
            )
        if not safety.act_enable:
            return (
                FaultCode.ACT_ENABLE_INVALID,
                "F407 actuator permission is not enabled",
            )
        active = safety.active_pumps
        if len(active) > 1:
            return (
                FaultCode.MULTIPLE_PUMPS_ACTIVE,
                f"multiple pumps active: {[zone.value for zone in active]}",
            )
        if require_pumps_off and active:
            return FaultCode.PUMP_NOT_OFF, f"pump still active: {active[0].value}"
        if expected_active_pump is not None and active != (expected_active_pump,):
            return (
                FaultCode.WRONG_PUMP_ACTIVE,
                f"expected {expected_active_pump.value}, got {[z.value for z in active]}",
            )
        if safety.active_wire_task_id != expected_active_wire_task_id:
            return (
                FaultCode.TASK_CONTEXT_MISMATCH,
                "F407 active wire task id does not match the expected phase",
            )
        return FaultCode.NONE, ""

    def _preclear_safety_problem(
        self, safety: SafetySnapshot
    ) -> Tuple[FaultCode, str]:
        """Validate a locked-but-otherwise-safe F407 before CLEAR_ESTOP."""

        if not safety.estop_clear:
            return FaultCode.ESTOP_ACTIVE, "emergency stop remains active"
        if not safety.leak_clear:
            return FaultCode.LEAK_DETECTED, "leak input is active"
        if not safety.cartridge_present:
            return FaultCode.CARTRIDGE_MISSING, "cartridge is not present"
        if not safety.guard_closed:
            return FaultCode.GUARD_OPEN, "guard/interlock is open"
        if not safety.heartbeat_fresh:
            return FaultCode.HEARTBEAT_STALE, "heartbeat is not fresh before clear"
        if not safety.telemetry_fresh:
            return FaultCode.TELEMETRY_STALE, "telemetry is not fresh before clear"
        if not safety.scale_stable:
            return FaultCode.SCALE_UNSTABLE, "scale is not stable before clear"
        if not safety.camera_quality_ok:
            return FaultCode.CAMERA_QUALITY_INVALID, "camera gate failed before clear"
        if safety.firmware_protocol_version != self.config.protocol_version:
            return FaultCode.FIRMWARE_IDENTITY_INVALID, "protocol version mismatch"
        if safety.firmware_build_id != self.config.expected_firmware_build_id:
            return FaultCode.FIRMWARE_IDENTITY_INVALID, "firmware build mismatch"
        if safety.execution_backend != self.config.required_backend:
            return FaultCode.FIRMWARE_IDENTITY_INVALID, "backend mismatch"
        missing = set(self.config.required_firmware_capabilities) - set(
            safety.firmware_capabilities
        )
        if missing:
            return (
                FaultCode.FIRMWARE_CAPABILITY_MISSING,
                f"missing capabilities: {sorted(missing)}",
            )
        if not safety.lock_latched:
            return FaultCode.F407_LOCK_LATCHED, "expected a latched pre-clear lock"
        if safety.act_enable:
            return FaultCode.ACT_ENABLE_INVALID, "ACT_ENABLE must be off while locked"
        if not safety.pumps_all_off:
            return FaultCode.PUMP_NOT_OFF, "all pumps must be off before clear"
        if safety.active_wire_task_id is not None:
            return FaultCode.TASK_CONTEXT_MISMATCH, "active task exists before clear"
        return FaultCode.NONE, ""

    def _fail_locked(
        self,
        code: FaultCode,
        detail: str,
        *,
        task_id: Optional[str] = None,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> OperationResult:
        self._physical_stop_required = True
        self._physical_stop_confirmed = False
        self._pending_stop_context = None
        payload: Dict[str, Any] = {
            "fault_code": code.value,
            "detail": detail,
            "state_at_fault": self._state.value,
            "highest_verified_class": self._highest_verified.value,
        }
        if evidence:
            payload["evidence"] = dict(evidence)
        if not self._emit("fault_latched", payload, task_id):
            return self._result(False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail)
        if not self._emit(
            "safe_stop_requested",
            {
                "fault_code": code.value,
                "host_state_only": True,
                "physical_stop_confirmed": False,
                "required_external_action": (
                    "execution adapter must send EMERGENCY_STOP and bind its ACK "
                    "or hard power-cut evidence plus fresh pump-off telemetry"
                ),
            },
            task_id,
        ):
            return self._result(False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail)
        if self._state is not MachineState.SAFE_STOP:
            if not self._transition(
                MachineState.SAFE_STOP,
                "host requested physical stop; confirmation pending",
                task_id=task_id,
                extra={"fault_code": code.value, "detail": detail},
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
        if not self._transition(
            MachineState.ABORTED_LOCKED,
            "host lock latched; no physical stop completion claim",
            task_id=task_id,
        ):
            return self._result(False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail)
        self._completion = CompletionClass.ABORTED_LOCKED
        self._last_fault = code
        self._fault_detail = detail
        return self._result(False, code, detail)

    def _reject_operation(self, code: FaultCode, detail: str) -> OperationResult:
        return self._result(False, code, detail)

    def _require_active_task(self, task_id: str) -> Optional[OperationResult]:
        if self._active_task is None or self._active_task.task_id != task_id:
            if self._state in _ACTIVE_TASK_STATES:
                return self._fail_locked(
                    FaultCode.TASK_CONTEXT_MISMATCH,
                    "evidence/command task_id does not match the active task",
                    task_id=self._active_task.task_id if self._active_task else None,
                )
            return self._reject_operation(
                FaultCode.TASK_CONTEXT_MISMATCH, "no matching active task"
            )
        return None

    def start_self_check(self) -> OperationResult:
        with self._lock:
            if self._state is not MachineState.BOOT_LOCKED:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION,
                    "self-check can start only from BOOT_LOCKED",
                )
            if not self._transition(MachineState.SELF_CHECK, "explicit startup self-check"):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._pending_clear_context = None
            self._clear_estop_acknowledged = False
            self._f407_boot_id = None
            return self._result(True)

    def bind_clear_estop_command_context(
        self,
        context: ClearEstopCommandContext,
        safety_before_clear: SafetySnapshot,
        *,
        operator_confirmed: bool,
    ) -> OperationResult:
        """Authorise exactly one already-encoded CLEAR_ESTOP frame context."""

        with self._lock:
            if self._state is not MachineState.SELF_CHECK:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION,
                    "CLEAR_ESTOP context binds only during SELF_CHECK",
                )
            if operator_confirmed is not True:
                return self._reject_operation(
                    FaultCode.OPERATOR_CONFIRMATION_REQUIRED,
                    "operator_confirmed must be the boolean True",
                )
            if not self.config.commissioned:
                return self._fail_locked(
                    FaultCode.CONFIG_MISMATCH,
                    "uncommissioned config cannot clear the F407 lock",
                )
            if self._pending_clear_context is not None:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION,
                    "CLEAR_ESTOP context already bound; do not resend",
                )
            if (
                context.execution_backend != self.config.required_backend
                or context.firmware_boot_id != safety_before_clear.firmware_boot_id
            ):
                return self._fail_locked(
                    FaultCode.COMMAND_CONTEXT_INVALID,
                    "CLEAR_ESTOP backend/boot context mismatch",
                    evidence=context.to_dict(),
                )
            problem, detail = self._preclear_safety_problem(safety_before_clear)
            if problem is not FaultCode.NONE:
                return self._fail_locked(
                    problem,
                    detail,
                    evidence={
                        "command_context": context.to_dict(),
                        "safety": safety_before_clear.to_dict(),
                    },
                )
            if not self._emit(
                "clear_estop_command_context_bound",
                {
                    "command_context": context.to_dict(),
                    "operator_confirmed": True,
                    "physical_command_sent": False,
                },
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._pending_clear_context = context
            self._clear_estop_acknowledged = False
            return self._result(
                True,
                detail="context bound; adapter may send exactly one CLEAR_ESTOP",
            )

    def clear_estop_acknowledged(
        self,
        ack: ClearEstopAckEvidence,
        safety_after_clear: SafetySnapshot,
    ) -> OperationResult:
        """Accept only the ACK and post-clear snapshot bound to the pending frame."""

        with self._lock:
            if self._state is not MachineState.SELF_CHECK:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION,
                    "CLEAR_ESTOP ACK is accepted only during SELF_CHECK",
                )
            context = self._pending_clear_context
            if context is None:
                return self._fail_locked(
                    FaultCode.CLEAR_ACK_INVALID,
                    "no pending CLEAR_ESTOP command context",
                )
            if (
                ack.ack_for_type != "CLEAR_ESTOP"
                or ack.ack_for_seq != context.frame_seq
                or ack.transcript_id != context.transcript_id
                or ack.firmware_build_id != self.config.expected_firmware_build_id
                or ack.firmware_boot_id != context.firmware_boot_id
                or ack.execution_backend != context.execution_backend
                or safety_after_clear.firmware_boot_id != context.firmware_boot_id
                or not ack.acked
                or not ack.fresh
            ):
                return self._fail_locked(
                    FaultCode.CLEAR_ACK_INVALID,
                    "CLEAR_ESTOP ACK type/seq/transcript/identity/freshness mismatch",
                    evidence=ack.to_dict(),
                )
            problem, detail = self._safety_problem(
                safety_after_clear,
                require_pumps_off=True,
                require_camera=True,
                require_scale=True,
                allow_boot_change=True,
            )
            if problem is not FaultCode.NONE:
                return self._fail_locked(
                    problem,
                    detail,
                    evidence={
                        "ack": ack.to_dict(),
                        "safety": safety_after_clear.to_dict(),
                    },
                )
            if not self._emit(
                "clear_estop_acknowledged",
                {
                    "command_context": context.to_dict(),
                    "ack": ack.to_dict(),
                    "post_clear_safety": safety_after_clear.to_dict(),
                },
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._f407_boot_id = safety_after_clear.firmware_boot_id
            self._clear_estop_acknowledged = True
            return self._result(True)

    def complete_self_check(self, safety: SafetySnapshot) -> OperationResult:
        with self._lock:
            if self._state is not MachineState.SELF_CHECK:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION, "not in SELF_CHECK"
                )
            if not self.config.commissioned:
                return self._fail_locked(
                    FaultCode.CONFIG_MISMATCH,
                    "configuration is a fixture/uncommissioned placeholder",
                    evidence={"commissioning_id": self.config.commissioning_id},
                )
            if not self._clear_estop_acknowledged or self._pending_clear_context is None:
                return self._fail_locked(
                    FaultCode.CLEAR_ACK_INVALID,
                    "matching CLEAR_ESTOP command context and ACK are required",
                )
            problem, detail = self._safety_problem(
                safety,
                require_pumps_off=True,
                require_camera=True,
                require_scale=True,
            )
            if problem is not FaultCode.NONE:
                return self._fail_locked(
                    FaultCode.SELF_CHECK_FAILED,
                    f"{problem.value}: {detail}",
                    evidence=safety.to_dict(),
                )
            if not self._emit(
                "self_check_passed",
                {
                    "config_hash": self.config.sha256,
                    "commissioning_id": self.config.commissioning_id,
                    "safety": safety.to_dict(),
                },
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._f407_boot_id = safety.firmware_boot_id
            if not self._transition(MachineState.READY, "all startup gates passed"):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._last_fault = FaultCode.NONE
            self._fault_detail = ""
            self._pending_clear_context = None
            self._physical_stop_required = False
            self._physical_stop_confirmed = False
            return self._result(True)

    def begin_operator_reset(self, *, operator_confirmed: bool) -> OperationResult:
        """Enter the reset self-check; CLEAR_ESTOP is still a separate gate."""

        with self._lock:
            if self._state is not MachineState.ABORTED_LOCKED:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION,
                    "operator reset is allowed only from ABORTED_LOCKED",
                )
            if operator_confirmed is not True:
                return self._reject_operation(
                    FaultCode.OPERATOR_CONFIRMATION_REQUIRED,
                    "operator_confirmed must be the boolean True",
                )
            if self._physical_stop_required and not self._physical_stop_confirmed:
                return self._reject_operation(
                    FaultCode.PHYSICAL_STOP_UNCONFIRMED,
                    "fresh physical pump-off/stop evidence is required before reset",
                )
            old_task_id = self._active_task.task_id if self._active_task else None
            if not self._transition(
                MachineState.SELF_CHECK,
                "manual reset accepted; full self-check still required",
                task_id=old_task_id,
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._active_task = None
            self._baseline = None
            self._dosing_result_received_monotonic_ms = None
            self._verification_started_monotonic_ms = None
            self._pending_arm_context = None
            self._pending_stop_context = None
            self._pending_clear_context = None
            self._clear_estop_acknowledged = False
            # A lower-controller reboot is accepted only through the subsequent
            # matching CLEAR_ESTOP ACK and full post-clear snapshot.
            self._f407_boot_id = None
            self._completion = CompletionClass.SIMULATED_ONLY
            self._highest_verified = CompletionClass.SIMULATED_ONLY
            self._last_fault = FaultCode.NONE
            self._fault_detail = ""
            return self._result(True)

    def bind_stop_command_context(
        self, context: StopCommandContext
    ) -> OperationResult:
        """Record the exact EMERGENCY_STOP transcript after it was sent.

        This method is an evidence-binding step, never an authorization gate.
        The execution adapter must transmit the safety-directional frame first;
        failure here may prevent a *confirmed* stop claim but must not delay or
        suppress the lower-controller stop itself.
        """

        with self._lock:
            if self._state is not MachineState.ABORTED_LOCKED:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION,
                    "stop context binds only while host-locked",
                )
            if not self._physical_stop_required:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION, "no physical stop is pending"
                )
            if self._pending_stop_context is not None:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION,
                    "stop context already bound; do not resend",
                )
            expected_task_id = self._active_task.task_id if self._active_task else None
            expected_wire_id = (
                self._active_task.wire_task_id if self._active_task else None
            )
            if (
                context.task_id != expected_task_id
                or context.wire_task_id != expected_wire_id
                or context.execution_backend != self.config.required_backend
                or context.firmware_build_id
                != self.config.expected_firmware_build_id
                or context.firmware_boot_id != self._f407_boot_id
            ):
                return self._reject_operation(
                    FaultCode.COMMAND_CONTEXT_INVALID,
                    "stop task/backend/firmware context mismatch",
                )
            if not self._emit(
                "stop_command_context_bound",
                {
                    "command_context": context.to_dict(),
                    "physical_command_sent": True,
                    "binding_role": "POST_SEND_EVIDENCE_ONLY",
                },
                expected_task_id,
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._pending_stop_context = context
            return self._result(
                True,
                detail="already-sent EMERGENCY_STOP transcript recorded",
            )

    def confirm_physical_stop(
        self, evidence: PhysicalStopEvidence
    ) -> OperationResult:
        """Bind execution-layer stop ACK/power-cut and fresh pump-off telemetry."""

        with self._lock:
            if self._state is not MachineState.ABORTED_LOCKED:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION,
                    "physical stop confirmation is accepted only while host-locked",
                )
            expected_task_id = self._active_task.task_id if self._active_task else None
            expected_wire_id = (
                self._active_task.wire_task_id if self._active_task else None
            )
            context = self._pending_stop_context
            if context is None:
                return self._reject_operation(
                    FaultCode.PHYSICAL_STOP_UNCONFIRMED,
                    "no stored EMERGENCY_STOP command context",
                )
            if (
                evidence.task_id != expected_task_id
                or evidence.wire_task_id != expected_wire_id
                or evidence.stop_frame_seq != context.frame_seq
                or evidence.stop_raw_frame_sha256 != context.raw_frame_sha256
                or evidence.transcript_id != context.transcript_id
                or evidence.decoded_command != context.decoded_command
                or evidence.firmware_build_id != context.firmware_build_id
                or evidence.firmware_boot_id != context.firmware_boot_id
                or evidence.execution_backend != context.execution_backend
                or not evidence.fresh
                or not evidence.pumps_all_off
            ):
                return self._reject_operation(
                    FaultCode.PHYSICAL_STOP_UNCONFIRMED,
                    "stop evidence task binding/freshness/pump-off gate failed",
                )
            command_stop_confirmed = (
                evidence.acked
                and evidence.ack_for_type == "EMERGENCY_STOP"
                and evidence.ack_for_seq == context.frame_seq
                and evidence.ack_frame_sha256 is not None
            )
            if not (command_stop_confirmed or evidence.hard_power_cut_confirmed):
                return self._reject_operation(
                    FaultCode.PHYSICAL_STOP_UNCONFIRMED,
                    "matching stop ACK or hard power-cut evidence is required",
                )
            if not self._emit(
                "physical_stop_confirmed",
                {
                    "stop_evidence": evidence.to_dict(),
                    "confirmation_source": (
                        "hard_power_cut"
                        if evidence.hard_power_cut_confirmed
                        else "f407_estop_ack"
                    ),
                },
                expected_task_id,
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._physical_stop_confirmed = True
            return self._result(True)

    def admit_task(
        self, request: TaskRequest, safety: SafetySnapshot
    ) -> AdmissionResult:
        with self._lock:
            prior = self._history_by_id.get(request.task_id)
            if prior is not None:
                if prior.request_fingerprint == request.fingerprint:
                    self._emit(
                        "task_idempotent_replay",
                        {
                            "task_id": request.task_id,
                            "task_seq": request.task_seq,
                            "request_fingerprint": request.fingerprint,
                            "state": self._state.value,
                            "physical_command_created": False,
                        },
                        request.task_id,
                    )
                    return AdmissionResult(
                        AdmissionStatus.IDEMPOTENT_REPLAY,
                        self._state,
                        request.task_id,
                        detail="known task replay; no new physical command",
                    )
                self._emit(
                    "task_rejected",
                    {
                        "fault_code": FaultCode.TASK_ID_CONFLICT.value,
                        "task_seq": request.task_seq,
                    },
                    request.task_id,
                )
                return AdmissionResult(
                    AdmissionStatus.REJECTED,
                    self._state,
                    request.task_id,
                    FaultCode.TASK_ID_CONFLICT,
                    "task_id was already admitted with different physical parameters",
                )
            if request.task_seq <= self._high_watermark:
                self._emit(
                    "task_rejected",
                    {
                        "fault_code": FaultCode.STALE_TASK.value,
                        "task_seq": request.task_seq,
                        "high_watermark_task_seq": self._high_watermark,
                    },
                    request.task_id,
                )
                return AdmissionResult(
                    AdmissionStatus.REJECTED,
                    self._state,
                    request.task_id,
                    FaultCode.STALE_TASK,
                    "task_seq is not newer than the persistent high watermark",
                )
            if self._state is not MachineState.READY:
                code = (
                    FaultCode.TASK_BUSY
                    if self._state in _ACTIVE_TASK_STATES
                    else FaultCode.NOT_READY
                )
                return AdmissionResult(
                    AdmissionStatus.REJECTED,
                    self._state,
                    request.task_id,
                    code,
                    "state machine is not READY",
                )
            if request.perception_source not in self.config.formal_perception_sources:
                return AdmissionResult(
                    AdmissionStatus.REJECTED,
                    self._state,
                    request.task_id,
                    FaultCode.PERCEPTION_NOT_QUALIFIED,
                    "perception source is not qualified for physical admission",
                )
            config_error = self.config.task_validation_error(request)
            if config_error:
                return AdmissionResult(
                    AdmissionStatus.REJECTED,
                    self._state,
                    request.task_id,
                    FaultCode.CONFIG_MISMATCH,
                    config_error,
                )
            problem, detail = self._safety_problem(
                safety,
                require_pumps_off=True,
                require_camera=True,
                require_scale=True,
            )
            if problem is not FaultCode.NONE:
                failed = self._fail_locked(
                    problem,
                    detail,
                    task_id=request.task_id,
                    evidence=safety.to_dict(),
                )
                return AdmissionResult(
                    AdmissionStatus.REJECTED,
                    failed.state,
                    request.task_id,
                    failed.fault_code,
                    failed.detail,
                )

            history_entry = TaskHistoryEntry(
                request.task_id, request.task_seq, request.fingerprint
            )
            admitted_payload = dict(request.to_dict())
            admitted_payload["physical_command_created"] = False
            if not self._emit("task_admitted", admitted_payload, request.task_id):
                return AdmissionResult(
                    AdmissionStatus.REJECTED,
                    self._state,
                    request.task_id,
                    FaultCode.EVIDENCE_WRITE_FAILED,
                    self._fault_detail,
                )
            self._history_by_id[request.task_id] = history_entry
            self._history_by_seq[request.task_seq] = history_entry
            self._high_watermark = request.task_seq
            self._active_task = request
            self._baseline = None
            self._dosing_result_received_monotonic_ms = None
            self._verification_started_monotonic_ms = None
            self._pending_arm_context = None
            self._physical_stop_required = False
            self._physical_stop_confirmed = False
            self._completion = CompletionClass.SIMULATED_ONLY
            self._highest_verified = CompletionClass.SIMULATED_ONLY
            if not self._transition(
                MachineState.TARGET_IDENTIFIED,
                "new task admitted",
                task_id=request.task_id,
                extra={
                    "profile_id": request.profile_id,
                    "channel": request.channel.value,
                    "perception_source": request.perception_source.value,
                },
            ):
                return AdmissionResult(
                    AdmissionStatus.REJECTED,
                    self._state,
                    request.task_id,
                    FaultCode.EVIDENCE_WRITE_FAILED,
                    self._fault_detail,
                )
            return AdmissionResult(
                AdmissionStatus.ACCEPTED,
                self._state,
                request.task_id,
            )

    def baseline_captured(
        self, task_id: str, baseline: BaselineEvidence, safety: SafetySnapshot
    ) -> OperationResult:
        with self._lock:
            mismatch = self._require_active_task(task_id)
            if mismatch:
                return mismatch
            if self._state is not MachineState.TARGET_IDENTIFIED:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION, "baseline is not expected in this state"
                )
            assert self._active_task is not None
            profile = self.config.profile_for(self._active_task.profile_id)
            if (
                baseline.task_id != task_id
                or baseline.wire_task_id != self._active_task.wire_task_id
                or baseline.config_hash != self.config.sha256
                or baseline.firmware_boot_id != self._f407_boot_id
                or not baseline.stable
                or not baseline.fresh
                or baseline.mass_sample_count < profile.minimum_mass_samples
            ):
                return self._fail_locked(
                    FaultCode.INVALID_TASK,
                    "baseline task/config/freshness/stability/sample gate failed",
                    task_id=task_id,
                    evidence=baseline.to_dict(),
                )
            problem, detail = self._safety_problem(
                safety,
                require_pumps_off=True,
                require_camera=True,
                require_scale=True,
            )
            if problem is not FaultCode.NONE:
                return self._fail_locked(
                    problem, detail, task_id=task_id, evidence=safety.to_dict()
                )
            if not self._emit(
                "baseline_captured",
                {"baseline": baseline.to_dict(), "safety": safety.to_dict()},
                task_id,
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._baseline = baseline
            if not self._transition(
                MachineState.BASELINE_CAPTURED,
                "visual and mass baseline frozen",
                task_id=task_id,
                extra={
                    "baseline_id": baseline.baseline_id,
                    "camera_frame_id": baseline.camera_frame_id,
                    "mass_sample_digest": baseline.mass_sample_digest,
                },
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            return self._result(True)

    def bind_arm_command_context(
        self, task_id: str, context: ArmCommandContext, safety: SafetySnapshot
    ) -> OperationResult:
        """Bind the exact encoded/decoded ARM_TASK transcript before sending.

        The serial adapter must reserve its next global link sequence first,
        call this method, and send exactly one ARM_TASK only when ``accepted`` is
        true.  Heartbeats use the same link sequence space, so this value is
        intentionally independent from the uint32 ``wire_task_id``.
        """

        with self._lock:
            mismatch = self._require_active_task(task_id)
            if mismatch:
                return mismatch
            if self._state is not MachineState.BASELINE_CAPTURED:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION,
                    "ARM command context can bind only after baseline capture",
                )
            if self._pending_arm_context is not None:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION,
                    "ARM context already bound; do not resend the physical command",
                )
            problem, detail = self._safety_problem(
                safety,
                require_pumps_off=True,
                require_camera=True,
                require_scale=True,
            )
            if problem is not FaultCode.NONE:
                return self._fail_locked(
                    problem, detail, task_id=task_id, evidence=safety.to_dict()
                )
            assert self._active_task is not None
            request = self._active_task
            if (
                context.task_id != request.task_id
                or context.wire_task_id != request.wire_task_id
                or context.decoded_channel is not request.channel
                or context.decoded_target_mass_mg != request.target_mass_mg
                or context.decoded_hard_timeout_ms != request.hard_timeout_ms
                or context.decoded_config_hash_prefix != request.config_hash[:16]
                or context.execution_backend != self.config.required_backend
                or context.firmware_build_id
                != self.config.expected_firmware_build_id
                or context.firmware_boot_id != self._f407_boot_id
            ):
                return self._fail_locked(
                    FaultCode.COMMAND_CONTEXT_INVALID,
                    "ARM raw transcript does not decode to the frozen task",
                    task_id=task_id,
                    evidence=context.to_dict(),
                )
            if not self._emit(
                "arm_command_context_bound",
                {
                    "command_context": context.to_dict(),
                    "physical_command_sent": False,
                },
                task_id,
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._pending_arm_context = context
            return self._result(
                True,
                detail="context bound; serial adapter may send one ARM_TASK frame",
            )

    def actuator_acknowledged(
        self,
        task_id: str,
        ack: ActuatorAckEvidence,
        safety: SafetySnapshot,
    ) -> OperationResult:
        with self._lock:
            mismatch = self._require_active_task(task_id)
            if mismatch:
                return mismatch
            if self._state is not MachineState.BASELINE_CAPTURED:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION, "actuator ACK is not expected"
                )
            assert self._active_task is not None
            assert self._baseline is not None
            request = self._active_task
            context = self._pending_arm_context
            if (
                ack.task_id != request.task_id
                or ack.wire_task_id != request.wire_task_id
                or ack.ack_for_type != "ARM_TASK"
                or context is None
                or ack.ack_for_seq != context.frame_seq
                or ack.transcript_id != context.transcript_id
                or ack.channel is not request.channel
                or ack.firmware_build_id != self.config.expected_firmware_build_id
                or ack.firmware_boot_id != self._f407_boot_id
                or ack.execution_backend != self.config.required_backend
                or not ack.acked
                or not ack.fresh
                or not ack.all_other_pumps_off
            ):
                return self._fail_locked(
                    FaultCode.ACK_INVALID,
                    "ACK identity, channel, freshness, or mutual exclusion failed",
                    task_id=task_id,
                    evidence=ack.to_dict(),
                )
            problem, detail = self._safety_problem(
                safety,
                require_pumps_off=False,
                require_camera=True,
                require_scale=True,
                expected_active_pump=request.channel,
                expected_active_wire_task_id=request.wire_task_id,
            )
            if problem is not FaultCode.NONE:
                return self._fail_locked(
                    problem, detail, task_id=task_id, evidence=safety.to_dict()
                )
            if not self._emit("actuator_ack", ack.to_dict(), task_id):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._completion = CompletionClass.ACTUATOR_ACK
            self._highest_verified = CompletionClass.ACTUATOR_ACK
            if not self._transition(
                _DOSING_STATE[request.channel],
                "fresh matching F407 ACK and single-pump telemetry",
                task_id=task_id,
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            return self._result(True)

    def dosing_complete(
        self, task_id: str, mass: MassEvidence, safety: SafetySnapshot
    ) -> OperationResult:
        with self._lock:
            mismatch = self._require_active_task(task_id)
            if mismatch:
                return mismatch
            if self._state not in set(_DOSING_STATE.values()):
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION, "mass completion is not expected"
                )
            assert self._active_task is not None
            request = self._active_task
            profile = self.config.profile_for(request.profile_id)
            if self._baseline is None:
                return self._fail_locked(
                    FaultCode.INTERNAL_ERROR,
                    "frozen baseline is missing",
                    task_id=task_id,
                )
            problem, detail = self._safety_problem(
                safety,
                require_pumps_off=True,
                require_camera=False,
                require_scale=True,
            )
            if problem is not FaultCode.NONE:
                return self._fail_locked(
                    problem, detail, task_id=task_id, evidence=safety.to_dict()
                )
            if (
                mass.task_id != task_id
                or mass.wire_task_id != request.wire_task_id
                or mass.result_type != "TASK_RESULT"
                or mass.terminal_reason != "TARGET_REACHED"
                or mass.firmware_build_id != self.config.expected_firmware_build_id
                or mass.firmware_boot_id != self._f407_boot_id
                or mass.firmware_boot_id != self._baseline.firmware_boot_id
                or mass.baseline_id != self._baseline.baseline_id
                or mass.baseline_mass_mg != self._baseline.baseline_mass_mg
                or mass.baseline_sample_digest != self._baseline.mass_sample_digest
                or mass.first_result_sample_seq
                <= self._baseline.mass_last_sample_seq
                or mass.firmware_completed_uptime_ms
                <= self._baseline.firmware_uptime_ms_at_capture
                or mass.host_result_received_monotonic_ms
                <= self._baseline.host_captured_monotonic_ms
                or not mass.fresh
                or not mass.stable
                or not mass.task_result_scale_stable
                or not mass.pumps_all_off
            ):
                return self._fail_locked(
                    FaultCode.MASS_OUT_OF_RANGE,
                    "mass evidence identity/stability/pump-off gate failed",
                    task_id=task_id,
                    evidence=mass.to_dict(),
                )
            if (
                mass.sample_count < profile.minimum_mass_samples
                or mass.post_stop_sample_count < profile.minimum_mass_samples
            ):
                return self._fail_locked(
                    FaultCode.MASS_OUT_OF_RANGE,
                    "insufficient stable mass samples",
                    task_id=task_id,
                    evidence=mass.to_dict(),
                )
            if mass.final_mass_span_mg > profile.max_final_mass_span_mg:
                return self._fail_locked(
                    FaultCode.MASS_OUT_OF_RANGE,
                    f"final mass span {mass.final_mass_span_mg} mg exceeds frozen "
                    f"{profile.max_final_mass_span_mg} mg",
                    task_id=task_id,
                    evidence=mass.to_dict(),
                )
            lower = request.target_mass_mg - request.tolerance_mg
            upper = request.target_mass_mg + request.tolerance_mg
            if not lower <= mass.mass_loss_mg <= upper:
                return self._fail_locked(
                    FaultCode.MASS_OUT_OF_RANGE,
                    f"mass loss {mass.mass_loss_mg} mg outside frozen [{lower}, {upper}] mg",
                    task_id=task_id,
                    evidence=mass.to_dict(),
                )
            if not self._emit("mass_loss_verified", mass.to_dict(), task_id):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._completion = CompletionClass.MASS_LOSS_VERIFIED
            self._highest_verified = CompletionClass.MASS_LOSS_VERIFIED
            self._dosing_result_received_monotonic_ms = (
                mass.host_result_received_monotonic_ms
            )
            if not self._transition(
                MachineState.SETTLING,
                "stable mass loss inside frozen tolerance and pumps off",
                task_id=task_id,
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            return self._result(True)

    def begin_verification(
        self,
        task_id: str,
        safety: SafetySnapshot,
        *,
        current_monotonic_ms: int,
    ) -> OperationResult:
        with self._lock:
            mismatch = self._require_active_task(task_id)
            if mismatch:
                return mismatch
            if self._state is not MachineState.SETTLING:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION, "not ready for visual verification"
                )
            if isinstance(current_monotonic_ms, bool) or not isinstance(
                current_monotonic_ms, int
            ):
                return self._reject_operation(
                    FaultCode.INVALID_TASK,
                    "current_monotonic_ms must be an integer",
                )
            if current_monotonic_ms < 0:
                return self._reject_operation(
                    FaultCode.INVALID_TASK,
                    "current_monotonic_ms cannot be negative",
                )
            assert self._active_task is not None
            if self._dosing_result_received_monotonic_ms is None:
                return self._fail_locked(
                    FaultCode.INTERNAL_ERROR,
                    "dosing completion time is missing",
                    task_id=task_id,
                )
            profile = self.config.profile_for(self._active_task.profile_id)
            earliest = self._dosing_result_received_monotonic_ms + profile.settle_ms
            if current_monotonic_ms < earliest:
                return self._reject_operation(
                    FaultCode.SETTLING_NOT_COMPLETE,
                    f"settling incomplete: earliest {earliest}, got {current_monotonic_ms}",
                )
            problem, detail = self._safety_problem(
                safety,
                require_pumps_off=True,
                require_camera=True,
                require_scale=True,
            )
            if problem is not FaultCode.NONE:
                return self._fail_locked(
                    problem, detail, task_id=task_id, evidence=safety.to_dict()
                )
            if not self._transition(
                MachineState.VERIFYING,
                "settling window completed; final frame accepted",
                task_id=task_id,
                extra={
                    "host_result_received_monotonic_ms": (
                        self._dosing_result_received_monotonic_ms
                    ),
                    "settle_ms": profile.settle_ms,
                    "verification_started_monotonic_ms": current_monotonic_ms,
                },
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._verification_started_monotonic_ms = current_monotonic_ms
            return self._result(True)

    def verification_complete(
        self, task_id: str, wetting: WettingEvidence, safety: SafetySnapshot
    ) -> OperationResult:
        with self._lock:
            mismatch = self._require_active_task(task_id)
            if mismatch:
                return mismatch
            if self._state is not MachineState.VERIFYING:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION, "verification result is not expected"
                )
            assert self._active_task is not None
            profile = self.config.profile_for(self._active_task.profile_id)
            if (
                self._baseline is None
                or self._dosing_result_received_monotonic_ms is None
                or self._verification_started_monotonic_ms is None
            ):
                return self._fail_locked(
                    FaultCode.INTERNAL_ERROR,
                    "baseline or dosing completion boundary is missing",
                    task_id=task_id,
                )
            problem, detail = self._safety_problem(
                safety,
                require_pumps_off=True,
                require_camera=True,
                require_scale=True,
            )
            if problem is not FaultCode.NONE:
                return self._fail_locked(
                    problem, detail, task_id=task_id, evidence=safety.to_dict()
                )
            if (
                wetting.task_id != task_id
                or wetting.baseline_id != self._baseline.baseline_id
                or wetting.baseline_frame_id != self._baseline.camera_frame_id
                or wetting.baseline_frame_sha256
                != self._baseline.camera_frame_sha256
                or wetting.captured_monotonic_ms
                < self._verification_started_monotonic_ms
                or not wetting.fresh
                or not wetting.camera_quality_ok
            ):
                return self._fail_locked(
                    FaultCode.WETTING_NOT_VERIFIED,
                    "wetting evidence identity/freshness/camera gate failed",
                    task_id=task_id,
                    evidence=wetting.to_dict(),
                )
            if (
                wetting.target_threshold != profile.target_wetting_threshold
                or wetting.spill_threshold != profile.neighbor_spill_threshold
            ):
                return self._fail_locked(
                    FaultCode.CONFIG_MISMATCH,
                    "vision thresholds do not match the frozen profile",
                    task_id=task_id,
                    evidence=wetting.to_dict(),
                )
            if not self._emit("wetting_evaluated", wetting.to_dict(), task_id):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            if not wetting.target_passed:
                return self._fail_locked(
                    FaultCode.WETTING_NOT_VERIFIED,
                    "target ROI did not reach the frozen threshold",
                    task_id=task_id,
                    evidence=wetting.to_dict(),
                )
            if not wetting.spill_passed:
                return self._fail_locked(
                    FaultCode.NEIGHBOR_SPILL,
                    "neighbor ROI exceeded the frozen spill threshold",
                    task_id=task_id,
                    evidence=wetting.to_dict(),
                )
            self._completion = CompletionClass.TARGET_WETTING_VERIFIED
            self._highest_verified = CompletionClass.TARGET_WETTING_VERIFIED
            if not self._transition(
                MachineState.TARGET_WETTING_VERIFIED,
                "mass and target wetting evidence both verified",
                task_id=task_id,
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            return self._result(True)

    def close_verified_task(
        self,
        safety: SafetySnapshot,
        *,
        operator_confirmed: bool,
        cartridge_changed_or_confirmed: bool,
    ) -> OperationResult:
        with self._lock:
            if self._state is not MachineState.TARGET_WETTING_VERIFIED:
                return self._reject_operation(
                    FaultCode.INVALID_TRANSITION, "no verified task is ready to close"
                )
            if operator_confirmed is not True:
                return self._reject_operation(
                    FaultCode.OPERATOR_CONFIRMATION_REQUIRED,
                    "operator_confirmed must be the boolean True",
                )
            if cartridge_changed_or_confirmed is not True:
                return self._reject_operation(
                    FaultCode.CARTRIDGE_CHANGE_REQUIRED,
                    "cartridge_changed_or_confirmed must be the boolean True",
                )
            problem, detail = self._safety_problem(
                safety,
                require_pumps_off=True,
                require_camera=True,
                require_scale=True,
            )
            if problem is not FaultCode.NONE:
                return self._fail_locked(
                    problem,
                    detail,
                    task_id=self._active_task.task_id if self._active_task else None,
                    evidence=safety.to_dict(),
                )
            task_id = self._active_task.task_id if self._active_task else None
            if not self._emit(
                "task_closed",
                {
                    "completion_class": self._completion.value,
                    "highest_verified_class": self._highest_verified.value,
                    "operator_confirmed": True,
                    "cartridge_changed_or_confirmed": True,
                },
                task_id,
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            if not self._transition(
                MachineState.READY, "verified task closed", task_id=task_id
            ):
                return self._result(
                    False, FaultCode.EVIDENCE_WRITE_FAILED, self._fault_detail
                )
            self._active_task = None
            self._baseline = None
            self._dosing_result_received_monotonic_ms = None
            self._verification_started_monotonic_ms = None
            self._pending_arm_context = None
            self._completion = CompletionClass.SIMULATED_ONLY
            self._highest_verified = CompletionClass.SIMULATED_ONLY
            return self._result(True)

    def abort(
        self, reason: str, *, fault_code: FaultCode = FaultCode.USER_ABORT
    ) -> OperationResult:
        with self._lock:
            if self._state is MachineState.ABORTED_LOCKED:
                return self._result(False, self._last_fault, self._fault_detail)
            task_id = self._active_task.task_id if self._active_task else None
            return self._fail_locked(
                fault_code,
                reason or "abort requested",
                task_id=task_id,
            )
