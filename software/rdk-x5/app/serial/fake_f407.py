"""Deterministic, in-memory RootScope F407 simulator.

The simulator models protocol and safety behavior only.  It does not open a
serial port, touch GPIO, or claim hardware validation.  Callers provide
timestamps to make watchdog and hard-timeout tests deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import secrets
import time
from typing import Callable

from .frame import (
    AbortTask,
    Ack,
    AckReason,
    AckStatus,
    ArmTask,
    CommandType,
    EXPECTED_BUILD_ID,
    EXPECTED_HW_VARIANT,
    FirmwareInfo,
    Frame,
    FrameParser,
    Heartbeat,
    IrrigationTelemetry,
    LockReason,
    PROTOCOL_VERSION,
    REQUIRED_CAPABILITIES,
    ResponseType,
    SafetyBits,
    SafetyState,
    SeqCommand,
    TaskResult,
    TerminalReason,
    PayloadError,
    encode_frame,
    encode_message,
)
from .link import F407_WATCHDOG_TIMEOUT_S, HEARTBEAT_INTERVAL_S


MIN_TARGET_MASS_MG = 100
MAX_TARGET_MASS_MG = 200_000
MIN_HARD_TIMEOUT_MS = 500
MAX_HARD_TIMEOUT_MS = 120_000
TARGET_STABLE_SAMPLE_COUNT = 5
TARGET_STABLE_SPAN_MG = 20


@dataclass
class SafetyInputs:
    estop_active: bool = False
    leak_detected: bool = False
    cartridge_present: bool = True
    guard_closed: bool = True
    hx711_valid: bool = True

    @property
    def all_safe(self) -> bool:
        return bool(
            not self.estop_active
            and not self.leak_detected
            and self.cartridge_present
            and self.guard_closed
            and self.hx711_valid
        )


@dataclass
class ActiveTask:
    task_id: int
    channel: int
    target_mass_mg: int
    hard_timeout_ms: int
    config_hash_prefix: bytes
    started_at: float
    baseline_mass_mg: int
    first_sample_seq: int = 0
    last_sample_seq: int = 0
    sample_count: int = 0
    target_reached_at: float | None = None
    post_stop_sample_count: int = 0
    post_stop_window: list[tuple[int, int]] = field(default_factory=list)


class FakeF407:
    """Protocol-level F407 model with pump, HX711, and safety fixtures."""

    def __init__(
        self,
        *,
        protocol_version: int = PROTOCOL_VERSION,
        capabilities: int = REQUIRED_CAPABILITIES,
        build_id: int = EXPECTED_BUILD_ID,
        hw_variant: int = EXPECTED_HW_VARIANT,
        build_tag: str = "rootscope-fake",
        boot_id: int | None = None,
        initial_mass_mg: int = 500_000,
        pump_rate_mg_s: int = 4_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self.parser = FrameParser()
        actual_boot_id = secrets.randbits(64) if boot_id is None else boot_id
        if actual_boot_id == 0:
            actual_boot_id = 1
        self.firmware_info = FirmwareInfo(
            protocol_version=protocol_version,
            capabilities=capabilities,
            build_id=build_id,
            hw_variant=hw_variant,
            build_tag=build_tag,
            boot_id=actual_boot_id,
        )
        self.inputs = SafetyInputs()
        self.filtered_mass_mg = int(initial_mass_mg)
        self.hx711_raw = int(initial_mass_mg * 8)
        self.pump_rate_mg_s = max(0, int(pump_rate_mg_s))

        # Power-up is intentionally fail-closed.  Heartbeat plus explicit clear
        # are required before ARM_TASK can energize ACT_ENABLE.
        self.locked = True
        self.lock_reason = LockReason.BOOT_LOCK
        self.pump_mask = 0
        self.active_task: ActiveTask | None = None
        self.last_task_id = 0
        self.last_seq: int | None = None
        self.last_heartbeat_at: float | None = None
        self.blocked_count = 0
        self.mass_sample_seq = 0
        self._result_seq = 0
        self.last_task_result: TaskResult | None = None
        self.last_task_result_wire: bytes | None = None
        self._result_replay_pending = False

        self._boot_at: float | None = None
        self._last_advance_at: float | None = None
        self._last_periodic_at: float | None = None
        self._last_identity_at: float | None = None

    def _now(self, now: float | None) -> float:
        current = self._clock() if now is None else float(now)
        if self._boot_at is None:
            self._boot_at = current
            self._last_advance_at = current
        return current

    @staticmethod
    def _sequence_relation(new: int, old: int | None) -> str:
        if old is None:
            return "new"
        delta = (int(new) - int(old)) & 0xFFFF
        if delta == 0:
            return "duplicate"
        if delta < 0x8000:
            return "new"
        return "stale"

    def _accept_sequence(self, seq: int) -> AckReason | None:
        if int(seq) == 0:
            return AckReason.STALE_SEQ
        relation = self._sequence_relation(seq, self.last_seq)
        if relation == "duplicate":
            return AckReason.DUPLICATE_SEQ
        if relation == "stale":
            return AckReason.STALE_SEQ
        self.last_seq = int(seq) & 0xFFFF
        return None

    def _safety_bits(self, now: float) -> int:
        bits = SafetyBits(0)
        if self.inputs.estop_active:
            bits |= SafetyBits.ESTOP_ACTIVE
        if self.inputs.leak_detected:
            bits |= SafetyBits.LEAK_DETECTED
        if self.inputs.cartridge_present:
            bits |= SafetyBits.CARTRIDGE_PRESENT
        if self.inputs.guard_closed:
            bits |= SafetyBits.GUARD_CLOSED
        if self.inputs.hx711_valid:
            bits |= SafetyBits.HX711_VALID
        if self._heartbeat_fresh(now):
            bits |= SafetyBits.WATCHDOG_FRESH
        if self.locked:
            bits |= SafetyBits.LOCK_LATCHED
        if not self.locked and self.inputs.all_safe and self._heartbeat_fresh(now):
            bits |= SafetyBits.ACT_ENABLE
        return int(bits)

    def _heartbeat_fresh(self, now: float) -> bool:
        return bool(
            self.last_heartbeat_at is not None
            and 0.0 <= now - self.last_heartbeat_at <= F407_WATCHDOG_TIMEOUT_S
        )

    def _heartbeat_age_ms(self, now: float) -> int:
        if self.last_heartbeat_at is None:
            return 0xFFFFFFFF
        return min(0xFFFFFFFF, max(0, int((now - self.last_heartbeat_at) * 1000)))

    def _uptime_ms(self, now: float) -> int:
        assert self._boot_at is not None
        return min(0xFFFFFFFF, max(0, int((now - self._boot_at) * 1000)))

    def _stop_pumps(self) -> None:
        self.pump_mask = 0

    def _next_result_seq(self) -> int:
        self._result_seq = (self._result_seq + 1) & 0xFFFF
        if self._result_seq == 0:
            self._result_seq = 1
        return self._result_seq

    def _capture_mass_sample(
        self, task: ActiveTask | None = None, *, post_stop: bool = False
    ) -> None:
        if self.mass_sample_seq >= 0xFFFFFFFF:
            raise RuntimeError("fake F407 mass sample sequence exhausted")
        self.mass_sample_seq += 1
        if task is not None and post_stop:
            task.post_stop_window.append((self.mass_sample_seq, self.filtered_mass_mg))
            if len(task.post_stop_window) > TARGET_STABLE_SAMPLE_COUNT:
                task.post_stop_window.pop(0)
            task.first_sample_seq = task.post_stop_window[0][0]
            task.last_sample_seq = task.post_stop_window[-1][0]
            task.sample_count = len(task.post_stop_window)
            task.post_stop_sample_count = len(task.post_stop_window)

    @staticmethod
    def _stable_window(task: ActiveTask) -> bool:
        if len(task.post_stop_window) < TARGET_STABLE_SAMPLE_COUNT:
            return False
        masses = [mass for _, mass in task.post_stop_window]
        return max(masses) - min(masses) <= TARGET_STABLE_SPAN_MG

    def _finalize_active_task(
        self, terminal_reason: TerminalReason, now: float
    ) -> TaskResult | None:
        task = self.active_task
        if task is None:
            return None
        self._stop_pumps()
        if task.sample_count == 0:
            self._capture_mass_sample(task, post_stop=True)
        window_masses = [mass for _, mass in task.post_stop_window]
        scale_stable = bool(
            terminal_reason is TerminalReason.TARGET_REACHED
            and self._stable_window(task)
        )
        if terminal_reason is TerminalReason.TARGET_REACHED and not scale_stable:
            raise RuntimeError("TARGET_REACHED cannot precede the stable sample gate")
        result = TaskResult(
            boot_id=self.firmware_info.boot_id,
            task_id=task.task_id,
            result_seq=self._next_result_seq(),
            terminal_reason=int(terminal_reason),
            baseline_mass_mg=task.baseline_mass_mg,
            final_mass_mg=self.filtered_mass_mg,
            first_sample_seq=task.first_sample_seq,
            last_sample_seq=task.last_sample_seq,
            sample_count=task.sample_count,
            final_window_min_mg=min(window_masses),
            final_window_max_mg=max(window_masses),
            scale_stable=scale_stable,
            firmware_completed_uptime_ms=self._uptime_ms(now),
            pump_mask=0,
            safety_bits=self._safety_bits(now),
        )
        wire = encode_message(ResponseType.TASK_RESULT, result)
        self.last_task_result = result
        self.last_task_result_wire = wire
        self._result_replay_pending = True
        self.active_task = None
        return result

    def _latch(
        self,
        reason: LockReason,
        now: float,
        *,
        terminal_reason: TerminalReason | None = None,
    ) -> None:
        self._stop_pumps()
        self.locked = True
        self.lock_reason = reason
        if terminal_reason is not None:
            self._finalize_active_task(terminal_reason, now)

    def _unsafe_reason_present(self) -> bool:
        return not self.inputs.all_safe

    def _apply_safety_inputs(self, now: float) -> None:
        if self._unsafe_reason_present():
            self._latch(
                LockReason.UNSAFE_INPUT,
                now,
                terminal_reason=TerminalReason.SAFETY_INPUT,
            )

    def _advance(self, now: float) -> None:
        """Advance mass only until the earliest independent safety deadline."""

        assert self._last_advance_at is not None
        start = self._last_advance_at
        task = self.active_task
        target_reached_first = False
        hard_timeout_due = False
        watchdog_due = False
        if task is not None and self.pump_mask:
            watchdog_deadline = (
                self.last_heartbeat_at + F407_WATCHDOG_TIMEOUT_S
                if self.last_heartbeat_at is not None
                else start
            )
            hard_deadline = task.started_at + task.hard_timeout_ms / 1000.0
            already_dispensed = task.baseline_mass_mg - self.filtered_mass_mg
            remaining_mass = max(0, task.target_mass_mg - already_dispensed)
            target_deadline = (
                start + remaining_mass / self.pump_rate_mg_s
                if self.pump_rate_mg_s > 0
                else float("inf")
            )

            # Target completion must be strictly earlier than a hard deadline.
            # At an exact hard-timeout boundary the safety timeout dominates.
            target_reached_first = bool(
                target_deadline <= now
                and target_deadline < hard_deadline
                and target_deadline <= watchdog_deadline
            )
            hard_timeout_due = bool(
                hard_deadline <= now
                and hard_deadline <= watchdog_deadline
                and not target_reached_first
            )
            watchdog_due = bool(
                now - watchdog_deadline > 1e-9
                and watchdog_deadline < hard_deadline
                and not target_reached_first
            )

            pump_until = min(
                now, watchdog_deadline, hard_deadline, target_deadline
            )
            elapsed = max(0.0, pump_until - start)
            if elapsed:
                dispensed = int(round(elapsed * self.pump_rate_mg_s))
                self.filtered_mass_mg = max(0, self.filtered_mass_mg - dispensed)
                self.hx711_raw = self.filtered_mass_mg * 8

            if target_reached_first:
                # Eliminate float-rounding ambiguity in the fixture evidence.
                self.filtered_mass_mg = min(
                    self.filtered_mass_mg,
                    max(0, task.baseline_mass_mg - task.target_mass_mg),
                )
                self.hx711_raw = self.filtered_mass_mg * 8
                self._stop_pumps()
                task.target_reached_at = target_deadline

        self._last_advance_at = now

        # Inputs and watchdog are independent of the host process and always
        # dominate business completion.
        self._apply_safety_inputs(now)
        if self.active_task is not None and hard_timeout_due:
            self._latch(
                LockReason.HARD_TIMEOUT,
                now,
                terminal_reason=TerminalReason.HARD_TIMEOUT,
            )
        elif self.active_task is not None and (
            watchdog_due
            or self.last_heartbeat_at is None
            or now - self.last_heartbeat_at > F407_WATCHDOG_TIMEOUT_S
        ):
            self._latch(
                LockReason.WATCHDOG_TIMEOUT,
                now,
                terminal_reason=TerminalReason.WATCHDOG_TIMEOUT,
            )
        elif (
            self.active_task is None
            and not self.locked
            and (
                self.last_heartbeat_at is None
                or now - self.last_heartbeat_at > F407_WATCHDOG_TIMEOUT_S
            )
        ):
            # The actuator permission itself is revoked even when no task is
            # currently dosing, so the next ARM requires an explicit clear.
            self._latch(LockReason.WATCHDOG_TIMEOUT, now)

    def _ack(
        self,
        command_type: int,
        seq: int,
        status: AckStatus = AckStatus.OK,
        reason: AckReason = AckReason.NONE,
        task_id: int = 0,
    ) -> bytes:
        return encode_message(
            ResponseType.ACK,
            Ack(
                ack_for_type=int(command_type),
                seq=int(seq) & 0xFFFF,
                status=int(status),
                reason=int(reason),
                task_id=int(task_id),
            ),
        )

    def _reject(
        self,
        command_type: int,
        seq: int,
        reason: AckReason,
        task_id: int = 0,
        *,
        locked: bool | None = None,
    ) -> bytes:
        self.blocked_count = min(0xFFFF, self.blocked_count + 1)
        is_locked = self.locked if locked is None else locked
        status = AckStatus.LOCKED if is_locked else AckStatus.REJECTED
        return self._ack(command_type, seq, status, reason, task_id)

    def _malformed(self, command_type: int, payload: bytes) -> bytes:
        # ARM_TASK and ABORT_TASK begin with task_id:u32; all other downlink
        # payloads begin with seq:u16.  Use zero when the relevant field itself
        # is truncated.  No state is changed.
        seq_offset = (
            4
            if command_type in (CommandType.ARM_TASK, CommandType.ABORT_TASK)
            else 0
        )
        seq = (
            int.from_bytes(payload[seq_offset : seq_offset + 2], "little")
            if len(payload) >= seq_offset + 2
            else 0
        )
        self.blocked_count = min(0xFFFF, self.blocked_count + 1)
        return self._ack(
            command_type,
            seq,
            AckStatus.BAD_PAYLOAD,
            AckReason.MALFORMED_PAYLOAD,
        )

    def _handle_frame(self, frame: Frame, now: float) -> list[bytes]:
        try:
            command_type = CommandType(frame.message_type)
        except ValueError:
            self.blocked_count = min(0xFFFF, self.blocked_count + 1)
            return [encode_frame(ResponseType.ERROR, bytes((AckReason.UNKNOWN_TYPE,)))]

        # E-stop is the one command that always takes effect, including replay.
        if command_type is CommandType.EMERGENCY_STOP:
            try:
                message = SeqCommand.from_payload(frame.payload)
            except PayloadError:
                return [self._malformed(command_type, frame.payload)]
            self._accept_sequence(message.seq)
            had_active_task = self.active_task is not None
            self._latch(
                LockReason.EMERGENCY_STOP,
                now,
                terminal_reason=TerminalReason.EMERGENCY_STOP,
            )
            replies = [
                self._ack(
                    command_type,
                    message.seq,
                    AckStatus.OK,
                    AckReason.EMERGENCY_STOP,
                )
            ]
            if had_active_task and self.last_task_result_wire is not None:
                replies.append(self.last_task_result_wire)
            return replies

        try:
            if command_type is CommandType.HEARTBEAT:
                message: object = Heartbeat.from_payload(frame.payload)
            elif command_type is CommandType.ARM_TASK:
                message = ArmTask.from_payload(frame.payload)
            elif command_type is CommandType.ABORT_TASK:
                message = AbortTask.from_payload(frame.payload)
            else:
                message = SeqCommand.from_payload(frame.payload)
        except PayloadError:
            return [self._malformed(command_type, frame.payload)]

        seq = int(getattr(message, "seq"))
        sequence_issue = self._accept_sequence(seq)
        if sequence_issue is not None:
            return [self._reject(command_type, seq, sequence_issue)]

        if command_type is CommandType.HEARTBEAT:
            self.last_heartbeat_at = now
            return [self._ack(command_type, seq)]

        if command_type is CommandType.QUERY_FIRMWARE:
            return [
                encode_message(ResponseType.FIRMWARE_INFO, self.firmware_info),
                self._ack(command_type, seq),
            ]

        if command_type is CommandType.CLEAR_ESTOP:
            if (
                self.pump_mask != 0
                or not self.inputs.all_safe
                or not self._heartbeat_fresh(now)
            ):
                return [
                    self._reject(
                        command_type,
                        seq,
                        AckReason.CLEAR_CONDITIONS_NOT_MET,
                        locked=True,
                    )
                ]
            self.locked = False
            self.lock_reason = LockReason.NONE
            return [self._ack(command_type, seq)]

        if command_type is CommandType.ARM_TASK:
            assert isinstance(message, ArmTask)
            if self.locked or not self._heartbeat_fresh(now) or not self.inputs.all_safe:
                reason = (
                    AckReason.WATCHDOG_TIMEOUT
                    if not self._heartbeat_fresh(now)
                    else AckReason.UNSAFE_INPUT
                    if not self.inputs.all_safe
                    else AckReason.BOOT_LOCK
                )
                return [self._reject(command_type, seq, reason, message.task_id)]
            if self.active_task is not None or self.pump_mask:
                return [
                    self._reject(
                        command_type,
                        seq,
                        AckReason.BUSY,
                        message.task_id,
                        locked=False,
                    )
                ]
            if message.task_id == 0 or message.task_id < self.last_task_id:
                return [
                    self._reject(
                        command_type,
                        seq,
                        AckReason.STALE_TASK,
                        message.task_id,
                        locked=False,
                    )
                ]
            if message.task_id == self.last_task_id:
                return [
                    self._reject(
                        command_type,
                        seq,
                        AckReason.DUPLICATE_TASK,
                        message.task_id,
                        locked=False,
                    )
                ]
            if message.channel not in (1, 2, 3):
                return [
                    self._reject(
                        command_type,
                        seq,
                        AckReason.INVALID_CHANNEL,
                        message.task_id,
                        locked=False,
                    )
                ]
            if not MIN_TARGET_MASS_MG <= message.target_mass_mg <= MAX_TARGET_MASS_MG:
                return [
                    self._reject(
                        command_type,
                        seq,
                        AckReason.INVALID_TARGET_MASS,
                        message.task_id,
                        locked=False,
                    )
                ]
            if not MIN_HARD_TIMEOUT_MS <= message.hard_timeout_ms <= MAX_HARD_TIMEOUT_MS:
                return [
                    self._reject(
                        command_type,
                        seq,
                        AckReason.INVALID_HARD_TIMEOUT,
                        message.task_id,
                        locked=False,
                    )
                ]

            requested_mask = 1 << (message.channel - 1)
            if requested_mask not in (0b001, 0b010, 0b100):
                # Defensive invariant even though the current wire format is a
                # channel enum, not a multi-pump bitmask.
                return [
                    self._reject(
                        command_type,
                        seq,
                        AckReason.INVALID_CHANNEL,
                        message.task_id,
                        locked=False,
                    )
                ]
            self.last_task_id = message.task_id
            self.active_task = ActiveTask(
                task_id=message.task_id,
                channel=message.channel,
                target_mass_mg=message.target_mass_mg,
                hard_timeout_ms=message.hard_timeout_ms,
                config_hash_prefix=message.config_hash_prefix,
                started_at=now,
                baseline_mass_mg=self.filtered_mass_mg,
            )
            self.pump_mask = requested_mask
            self._result_replay_pending = False
            self._last_advance_at = now
            return [self._ack(command_type, seq, task_id=message.task_id)]

        if command_type is CommandType.ABORT_TASK:
            assert isinstance(message, AbortTask)
            if self.active_task is None:
                return [
                    self._reject(
                        command_type,
                        seq,
                        AckReason.NO_ACTIVE_TASK,
                        message.task_id,
                    )
                ]
            if message.task_id != self.active_task.task_id:
                return [
                    self._reject(
                        command_type,
                        seq,
                        AckReason.TASK_MISMATCH,
                        message.task_id,
                        locked=False,
                    )
                ]
            self._latch(
                LockReason.USER_ABORT,
                now,
                terminal_reason=TerminalReason.USER_ABORT,
            )
            replies = [
                self._ack(
                    command_type,
                    seq,
                    AckStatus.OK,
                    AckReason.USER_ABORT,
                    message.task_id,
                )
            ]
            if self.last_task_result_wire is not None:
                replies.append(self.last_task_result_wire)
            return replies

        return [self._reject(command_type, seq, AckReason.UNKNOWN_TYPE)]

    def exchange(self, data: bytes, *, now: float | None = None) -> bytes:
        """Consume host bytes and return immediate response bytes."""

        current = self._now(now)
        prior_result_seq = (
            self.last_task_result.result_seq if self.last_task_result is not None else None
        )
        self._advance(current)
        replies: list[bytes] = []
        if (
            self.last_task_result is not None
            and self.last_task_result.result_seq != prior_result_seq
            and self.last_task_result_wire is not None
        ):
            replies.append(self.last_task_result_wire)
        for frame in self.parser.feed(data):
            replies.extend(self._handle_frame(frame, current))
        return b"".join(replies)

    def tick(self, *, now: float | None = None) -> bytes:
        """Advance the plant model and emit 5 Hz telemetry/safety snapshots."""

        current = self._now(now)
        self._advance(current)
        replies: list[bytes] = []

        if self._last_identity_at is None or current - self._last_identity_at >= 1.0:
            replies.append(encode_message(ResponseType.FIRMWARE_INFO, self.firmware_info))
            self._last_identity_at = current

        if (
            self._last_periodic_at is None
            or current - self._last_periodic_at >= HEARTBEAT_INTERVAL_S - 1e-9
        ):
            task_at_sample = self.active_task
            post_stop = bool(
                task_at_sample is not None
                and task_at_sample.target_reached_at is not None
                and self.pump_mask == 0
            )
            self._capture_mass_sample(task_at_sample, post_stop=post_stop)
            replies.append(
                encode_message(
                    ResponseType.SAFETY_STATE,
                    SafetyState(
                        boot_id=self.firmware_info.boot_id,
                        safety_bits=self._safety_bits(current),
                        blocked_count=self.blocked_count,
                        lock_reason=int(self.lock_reason),
                        heartbeat_age_ms=self._heartbeat_age_ms(current),
                    ),
                )
            )
            replies.append(
                encode_message(
                    ResponseType.IRRIGATION_TELEMETRY,
                    IrrigationTelemetry(
                        task_id=self.active_task.task_id if self.active_task else 0,
                        sample_seq=self.mass_sample_seq,
                        pump_mask=self.pump_mask,
                        hx711_raw=self.hx711_raw,
                        filtered_mass_mg=self.filtered_mass_mg,
                        safety_bits=self._safety_bits(current),
                        uptime_ms=self._uptime_ms(current),
                    ),
                )
            )
            self._last_periodic_at = current

            if (
                task_at_sample is not None
                and task_at_sample is self.active_task
                and task_at_sample.target_reached_at is not None
                and task_at_sample.post_stop_sample_count
                >= TARGET_STABLE_SAMPLE_COUNT
                and self._stable_window(task_at_sample)
            ):
                self._finalize_active_task(TerminalReason.TARGET_REACHED, current)

            if self._result_replay_pending and self.last_task_result_wire is not None:
                replies.append(self.last_task_result_wire)

        return b"".join(replies)

    def set_safety_inputs(
        self,
        *,
        estop_active: bool | None = None,
        leak_detected: bool | None = None,
        cartridge_present: bool | None = None,
        guard_closed: bool | None = None,
        hx711_valid: bool | None = None,
        now: float | None = None,
    ) -> None:
        current = self._now(now)
        self._advance(current)
        updates = {
            "estop_active": estop_active,
            "leak_detected": leak_detected,
            "cartridge_present": cartridge_present,
            "guard_closed": guard_closed,
            "hx711_valid": hx711_valid,
        }
        for name, value in updates.items():
            if value is not None:
                setattr(self.inputs, name, bool(value))
        self._apply_safety_inputs(current)

    def set_hx711(
        self,
        *,
        filtered_mass_mg: int,
        raw_counts: int | None = None,
        valid: bool = True,
        now: float | None = None,
    ) -> None:
        current = self._now(now)
        self._advance(current)
        self.filtered_mass_mg = max(0, int(filtered_mass_mg))
        self.hx711_raw = (
            int(raw_counts) if raw_counts is not None else self.filtered_mass_mg * 8
        )
        self.inputs.hx711_valid = bool(valid)
        self._capture_mass_sample()
        self._apply_safety_inputs(current)
        if self.inputs.hx711_valid and not self.locked:
            # A fresh external scale sample can itself satisfy the stop target;
            # evaluate it immediately without waiting for another timer tick.
            self._advance(current)

    @property
    def at_most_one_pump(self) -> bool:
        return self.pump_mask in (0, 1, 2, 4)
