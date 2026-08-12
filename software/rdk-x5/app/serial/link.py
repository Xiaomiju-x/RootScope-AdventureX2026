"""Fail-closed X5-side RootScope protocol session.

The class in this file only creates and consumes byte strings.  It never opens
``/dev/tty*`` and does not import pyserial.  A future hardware adapter can own
the file descriptor and feed bytes into :meth:`RootScopeSerialLink.ingest`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import struct
import time
from typing import Callable

from .frame import (
    AbortTask,
    Ack,
    AckReason,
    AckStatus,
    ArmTimedTask,
    ArmTask,
    CommandType,
    EXPECTED_BUILD_ID,
    EXPECTED_HW_VARIANT,
    F103_PB7_BUILD_ID,
    F103_PB7_HW_VARIANT,
    F103_PB7_REQUIRED_CAPABILITIES,
    F103_PB6_BUILD_ID,
    F103_PB6_HW_VARIANT,
    F103_PB6_REQUIRED_CAPABILITIES,
    F103_Z3_PB6_BUILD_ID,
    F103_Z3_PB6_HW_VARIANT,
    F103_Z3_PB6_REQUIRED_CAPABILITIES,
    FirmwareInfo,
    Frame,
    FrameParser,
    Heartbeat,
    IrrigationTelemetry,
    PayloadError,
    PROTOCOL_VERSION,
    REQUIRED_CAPABILITIES,
    ResponseType,
    SafetyState,
    SeqCommand,
    TaskResult,
    boot_id_token,
    decode_frame,
    encode_message,
)


HEARTBEAT_HZ = 5.0
HEARTBEAT_INTERVAL_S = 1.0 / HEARTBEAT_HZ
F407_WATCHDOG_TIMEOUT_S = 1.0
IDENTITY_STALE_TIMEOUT_S = 3.0


class SerialAdmissionError(RuntimeError):
    """Raised when a command that could energize a pump is not admissible."""


@dataclass(frozen=True)
class IdentityExpectation:
    protocol_version: int = PROTOCOL_VERSION
    build_id: int = EXPECTED_BUILD_ID
    hw_variant: int = EXPECTED_HW_VARIANT
    required_capabilities: int = REQUIRED_CAPABILITIES

    def mismatch(self, info: FirmwareInfo) -> str | None:
        reasons: list[str] = []
        if info.protocol_version != self.protocol_version:
            reasons.append(
                f"protocol={info.protocol_version}, expected={self.protocol_version}"
            )
        if info.build_id != self.build_id:
            reasons.append(f"build_id={info.build_id}, expected={self.build_id}")
        if info.hw_variant != self.hw_variant:
            reasons.append(
                f"hw_variant={info.hw_variant}, expected={self.hw_variant}"
            )
        missing = self.required_capabilities & ~info.capabilities
        if missing:
            reasons.append(f"missing_capabilities=0x{missing:08X}")
        return "; ".join(reasons) or None


F103_PB7_IDENTITY_EXPECTATION = IdentityExpectation(
    protocol_version=PROTOCOL_VERSION,
    build_id=F103_PB7_BUILD_ID,
    hw_variant=F103_PB7_HW_VARIANT,
    required_capabilities=F103_PB7_REQUIRED_CAPABILITIES,
)

F103_PB6_IDENTITY_EXPECTATION = IdentityExpectation(
    protocol_version=PROTOCOL_VERSION,
    build_id=F103_PB6_BUILD_ID,
    hw_variant=F103_PB6_HW_VARIANT,
    required_capabilities=F103_PB6_REQUIRED_CAPABILITIES,
)

F103_Z3_PB6_IDENTITY_EXPECTATION = IdentityExpectation(
    protocol_version=PROTOCOL_VERSION,
    build_id=F103_Z3_PB6_BUILD_ID,
    hw_variant=F103_Z3_PB6_HW_VARIANT,
    required_capabilities=F103_Z3_PB6_REQUIRED_CAPABILITIES,
)


CommandPayload = SeqCommand | Heartbeat | ArmTask | ArmTimedTask | AbortTask


@dataclass(frozen=True)
class CommandFrameReceipt:
    """Exact command bytes plus decoded intent; creation is not a send claim."""

    command_type: int
    seq: int
    decoded: CommandPayload
    raw_frame: bytes
    raw_frame_sha256: str
    generated_at: float
    sent_at: float | None
    execution_backend: str
    firmware_build_id: str | None
    firmware_boot_id: str | None

    def __post_init__(self) -> None:
        if not 0 < self.seq <= 0xFFFF:
            raise ValueError("command receipt seq must be a non-zero uint16")
        if hashlib.sha256(self.raw_frame).hexdigest() != self.raw_frame_sha256:
            raise ValueError("command receipt SHA-256 mismatch")
        frame = decode_frame(self.raw_frame)
        if frame.message_type != self.command_type:
            raise ValueError("command receipt type mismatch")
        if self.decoded.seq != self.seq:
            raise ValueError("command receipt decoded/declared sequence mismatch")
        if frame.payload != self.decoded.to_payload():
            raise ValueError("command receipt decoded payload does not match raw frame")
        try:
            command_type = CommandType(self.command_type)
        except ValueError as exc:
            raise ValueError("unknown command receipt type") from exc
        expected_class: type[CommandPayload]
        if command_type is CommandType.HEARTBEAT:
            expected_class = Heartbeat
        elif command_type is CommandType.ARM_TASK:
            expected_class = ArmTask
        elif command_type is CommandType.ARM_TIMED_TASK:
            expected_class = ArmTimedTask
        elif command_type is CommandType.ABORT_TASK:
            expected_class = AbortTask
        else:
            expected_class = SeqCommand
        if not isinstance(self.decoded, expected_class):
            raise ValueError("command receipt decoded class does not match command type")

    @property
    def sent(self) -> bool:
        return self.sent_at is not None

    @property
    def transcript(self) -> str:
        """Lossless lowercase-hex command transcript for evidence binding."""

        return self.raw_frame.hex()


@dataclass(frozen=True)
class CommandAckReceipt:
    command: CommandFrameReceipt
    ack: Ack
    ack_raw_frame: bytes
    ack_raw_frame_sha256: str
    received_at: float
    firmware_build_id: str
    firmware_boot_id: str
    execution_backend: str

    def __post_init__(self) -> None:
        if self.ack.ack_for_type != self.command.command_type:
            raise ValueError("ACK command type mismatch")
        if self.ack.seq != self.command.seq:
            raise ValueError("ACK command sequence mismatch")
        if hashlib.sha256(self.ack_raw_frame).hexdigest() != self.ack_raw_frame_sha256:
            raise ValueError("ACK raw-frame SHA-256 mismatch")
        frame = decode_frame(self.ack_raw_frame)
        if frame.message_type != int(ResponseType.ACK):
            raise ValueError("ACK receipt raw frame is not ACK")
        if Ack.from_payload(frame.payload) != self.ack:
            raise ValueError("ACK receipt decoded value does not match raw frame")


@dataclass(frozen=True)
class TaskResultReceipt:
    """Immutable host receipt over the exact firmware-originated result frame."""

    result: TaskResult
    raw_frame_sha256: str
    raw_frame: bytes
    received_at: float
    result_frame_seq: int
    firmware_build_id: str
    firmware_boot_id: str

    def __post_init__(self) -> None:
        if self.result_frame_seq != self.result.result_seq:
            raise ValueError("receipt/result frame sequence mismatch")
        if self.firmware_boot_id != boot_id_token(self.result.boot_id):
            raise ValueError("receipt/result boot ID mismatch")
        actual = hashlib.sha256(self.raw_frame).hexdigest()
        if actual != self.raw_frame_sha256:
            raise ValueError("receipt raw-frame SHA-256 mismatch")
        frame = decode_frame(self.raw_frame)
        if frame.message_type != int(ResponseType.TASK_RESULT):
            raise ValueError("receipt raw frame is not TASK_RESULT")
        if TaskResult.from_payload(frame.payload) != self.result:
            raise ValueError("receipt decoded result does not match raw frame")

    @property
    def terminal_reason(self) -> str:
        return self.result.terminal_reason_name


@dataclass(frozen=True)
class LinkEvent:
    kind: str
    value: (
        FirmwareInfo
        | Ack
        | IrrigationTelemetry
        | SafetyState
        | TaskResult
        | TaskResultReceipt
        | CommandAckReceipt
        | Frame
    )
    received_at: float


class RootScopeSerialLink:
    """Protocol state for one X5-to-F407 session.

    The global 16-bit sequence counter is shared by all downlink messages.
    F407 uses wrap-aware ordering and rejects duplicate or stale sequences.
    """

    def __init__(
        self,
        *,
        expectation: IdentityExpectation | None = None,
        clock: Callable[[], float] = time.monotonic,
        initial_sequence: int = 0,
    ) -> None:
        if (
            isinstance(initial_sequence, bool)
            or not isinstance(initial_sequence, int)
            or not 0 <= initial_sequence <= 0xFFFF
        ):
            raise ValueError("initial_sequence must be a uint16")
        self.expectation = expectation or IdentityExpectation()
        self._clock = clock
        self._parser = FrameParser()
        # The firmware sequence guard survives host-process restarts. A
        # production bridge must restore the last committed sequence from its
        # durable ledger instead of silently starting again at one.
        self._next_sequence = initial_sequence
        self._last_heartbeat_tx_at: float | None = None
        self._last_arm_task_id = 0
        self._pending_arm_by_seq: dict[int, int] = {}

        self.last_rx_at: float | None = None
        self.last_identity_at: float | None = None
        self.firmware_info: FirmwareInfo | None = None
        self.identity_valid = False
        self.identity_error = "firmware identity has not been received"
        self.last_ack: Ack | None = None
        self.last_telemetry: IrrigationTelemetry | None = None
        self.last_safety: SafetyState | None = None
        self.last_task_result_receipt: TaskResultReceipt | None = None
        self.last_command_receipt: CommandFrameReceipt | None = None
        self.last_command_ack_receipt: CommandAckReceipt | None = None
        self.acks: dict[tuple[int, int], Ack] = {}
        self.command_receipts: dict[tuple[int, int], CommandFrameReceipt] = {}
        self.command_receipts_by_sha256: dict[str, CommandFrameReceipt] = {}
        self.command_receipt_history: list[CommandFrameReceipt] = []
        self.command_ack_receipts: dict[tuple[int, int], CommandAckReceipt] = {}
        self.task_result_receipts: dict[int, TaskResultReceipt] = {}
        # Only non-conflicted terminal receipts remain in this trusted map.
        self.task_result_history: dict[tuple[int, int], TaskResultReceipt] = {}
        # Forensic history is retained separately and is never a trust source.
        self.task_result_forensic_history: dict[
            tuple[int, int], list[TaskResultReceipt]
        ] = {}
        self.task_result_conflicts: list[TaskResultReceipt] = []
        self._task_result_conflicted_keys: set[tuple[int, int]] = set()
        self._task_result_conflict_hashes: dict[tuple[int, int], set[str]] = {}

    @property
    def parser(self) -> FrameParser:
        return self._parser

    @property
    def sequence_checkpoint(self) -> int:
        """Last sequence reserved by this link for durable checkpointing."""

        return self._next_sequence

    @property
    def firmware_boot_id_token(self) -> str | None:
        return (
            boot_id_token(self.firmware_info.boot_id)
            if self.firmware_info is not None
            else None
        )

    @property
    def firmware_build_id_token(self) -> str | None:
        return str(self.firmware_info.build_id) if self.firmware_info is not None else None

    @property
    def task_result_conflicted_keys(self) -> frozenset[tuple[int, int]]:
        """All permanently conflicted ``(boot_id, task_id)`` evidence keys."""

        return frozenset(self._task_result_conflicted_keys)

    @property
    def task_result_conflict_active(self) -> bool:
        """Whether the current firmware boot contains any terminal conflict."""

        if self.firmware_info is None:
            return False
        boot_id = self.firmware_info.boot_id
        return any(key[0] == boot_id for key in self._task_result_conflicted_keys)

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else float(now)

    def next_seq(self) -> int:
        self._next_sequence = (self._next_sequence + 1) & 0xFFFF
        # Reserve zero as an unmistakable uninitialized value on both sides.
        if self._next_sequence == 0:
            self._next_sequence = 1
        return self._next_sequence

    def _record_generated_command(
        self,
        command_type: CommandType,
        decoded: CommandPayload,
        wire: bytes,
        now: float,
    ) -> bytes:
        seq = decoded.seq
        receipt = CommandFrameReceipt(
            command_type=int(command_type),
            seq=seq,
            decoded=decoded,
            raw_frame=wire,
            raw_frame_sha256=hashlib.sha256(wire).hexdigest(),
            generated_at=now,
            sent_at=None,
            execution_backend="NOT_SENT",
            firmware_build_id=self.firmware_build_id_token,
            firmware_boot_id=self.firmware_boot_id_token,
        )
        self.command_receipts[(int(command_type), seq)] = receipt
        self.command_receipts_by_sha256[receipt.raw_frame_sha256] = receipt
        self.command_receipt_history.append(receipt)
        self.last_command_receipt = receipt
        return wire

    def mark_command_sent(
        self,
        raw_frame: bytes,
        *,
        execution_backend: str,
        now: float | None = None,
    ) -> CommandFrameReceipt:
        """Bind an already-generated exact frame to an adapter send event.

        This method records metadata only; it never opens or writes a port.
        """

        if not isinstance(execution_backend, str) or not execution_backend:
            raise ValueError("execution_backend must be a non-empty string")
        if not isinstance(raw_frame, (bytes, bytearray, memoryview)):
            raise TypeError("raw_frame must be bytes-like")
        raw = bytes(raw_frame)
        digest = hashlib.sha256(raw).hexdigest()
        generated = self.command_receipts_by_sha256.get(digest)
        if generated is None or generated.raw_frame != raw:
            raise SerialAdmissionError("cannot mark an unknown command frame as sent")
        sent = replace(
            generated,
            sent_at=self._now(now),
            execution_backend=execution_backend,
            firmware_build_id=self.firmware_build_id_token,
            firmware_boot_id=self.firmware_boot_id_token,
        )
        key = (sent.command_type, sent.seq)
        self.command_receipts[key] = sent
        self.command_receipts_by_sha256[digest] = sent
        self.command_receipt_history.append(sent)
        self.last_command_receipt = sent
        return sent

    def identity_fresh(self, now: float | None = None) -> bool:
        current = self._now(now)
        return bool(
            self.identity_valid
            and self.last_identity_at is not None
            and 0.0 <= current - self.last_identity_at <= IDENTITY_STALE_TIMEOUT_S
        )

    def admission_ready(self, now: float | None = None) -> bool:
        """Return only the host-side firmware identity admission result.

        Physical safety admission remains authoritative on F407 and is also
        checked by the application state machine.  This method intentionally
        does not infer physical readiness from a possibly stale UI snapshot.
        """

        return self.identity_fresh(now)

    def require_admission(self, now: float | None = None) -> None:
        if not self.identity_fresh(now):
            detail = self.identity_error
            if self.identity_valid and self.last_identity_at is not None:
                detail = "firmware identity is stale"
            raise SerialAdmissionError(detail)

    def heartbeat_due(self, now: float | None = None) -> bool:
        current = self._now(now)
        return (
            self._last_heartbeat_tx_at is None
            or current - self._last_heartbeat_tx_at >= HEARTBEAT_INTERVAL_S - 1e-9
        )

    def make_heartbeat(self, host_state: int = 0, *, now: float | None = None) -> bytes:
        current = self._now(now)
        message = Heartbeat(self.next_seq(), host_state)
        self._last_heartbeat_tx_at = current
        wire = encode_message(CommandType.HEARTBEAT, message)
        return self._record_generated_command(
            CommandType.HEARTBEAT, message, wire, current
        )

    def make_firmware_query(self, *, now: float | None = None) -> bytes:
        message = SeqCommand(self.next_seq())
        wire = encode_message(CommandType.QUERY_FIRMWARE, message)
        return self._record_generated_command(
            CommandType.QUERY_FIRMWARE, message, wire, self._now(now)
        )

    def make_emergency_stop(self, *, now: float | None = None) -> bytes:
        # Emergency stop is always constructible, even before identity arrives.
        message = SeqCommand(self.next_seq())
        wire = encode_message(CommandType.EMERGENCY_STOP, message)
        return self._record_generated_command(
            CommandType.EMERGENCY_STOP, message, wire, self._now(now)
        )

    def make_clear_estop(self, *, now: float | None = None) -> bytes:
        self.require_admission(now)
        message = SeqCommand(self.next_seq())
        wire = encode_message(CommandType.CLEAR_ESTOP, message)
        return self._record_generated_command(
            CommandType.CLEAR_ESTOP, message, wire, self._now(now)
        )

    def make_arm_task(
        self,
        *,
        task_id: int,
        channel: int,
        target_mass_mg: int,
        hard_timeout_ms: int,
        config_hash_prefix: bytes,
        now: float | None = None,
    ) -> bytes:
        self.require_admission(now)
        highest_reserved = max(
            [self._last_arm_task_id, *self._pending_arm_by_seq.values()]
        )
        if task_id <= highest_reserved:
            raise SerialAdmissionError(
                f"task_id must increase: got {task_id}, last/reserved={highest_reserved}"
            )
        seq = self.next_seq()
        message = ArmTask(
            task_id=task_id,
            seq=seq,
            channel=channel,
            target_mass_mg=target_mass_mg,
            hard_timeout_ms=hard_timeout_ms,
            config_hash_prefix=config_hash_prefix,
        )
        wire = encode_message(CommandType.ARM_TASK, message)
        # A constructed packet is only reserved here.  Commit the task ID after
        # an explicit successful F407 ACK; a local build call is not execution
        # evidence.
        self._pending_arm_by_seq[seq] = task_id
        return self._record_generated_command(
            CommandType.ARM_TASK, message, wire, self._now(now)
        )

    def make_arm_timed_task(
        self,
        *,
        task_id: int,
        channel: int,
        duration_ms: int,
        hard_timeout_ms: int,
        config_hash_prefix: bytes,
        now: float | None = None,
    ) -> bytes:
        """Build the explicit no-HX711 task used by the F103 pump-only profile."""

        self.require_admission(now)
        highest_reserved = max(
            [self._last_arm_task_id, *self._pending_arm_by_seq.values()]
        )
        if task_id <= highest_reserved:
            raise SerialAdmissionError(
                f"task_id must increase: got {task_id}, last/reserved={highest_reserved}"
            )
        if channel != 1:
            raise SerialAdmissionError(
                "F103 pump-only timed task supports channel=1 only"
            )
        if not 100 <= duration_ms <= 30_000:
            raise SerialAdmissionError("duration_ms must be within [100, 30000]")
        if not 500 <= hard_timeout_ms <= 120_000:
            raise SerialAdmissionError(
                "hard_timeout_ms must be within [500, 120000]"
            )
        if hard_timeout_ms < duration_ms:
            raise SerialAdmissionError(
                "hard_timeout_ms cannot be shorter than duration_ms"
            )
        seq = self.next_seq()
        message = ArmTimedTask(
            task_id=task_id,
            seq=seq,
            channel=channel,
            duration_ms=duration_ms,
            hard_timeout_ms=hard_timeout_ms,
            config_hash_prefix=config_hash_prefix,
        )
        wire = encode_message(CommandType.ARM_TIMED_TASK, message)
        self._pending_arm_by_seq[seq] = task_id
        return self._record_generated_command(
            CommandType.ARM_TIMED_TASK, message, wire, self._now(now)
        )

    def make_abort_task(
        self, *, task_id: int, reason: int = 0, now: float | None = None
    ) -> bytes:
        # Abort is safety-directional and remains available if identity is stale.
        message = AbortTask(task_id=task_id, seq=self.next_seq(), reason=reason)
        wire = encode_message(
            CommandType.ABORT_TASK,
            message,
        )
        return self._record_generated_command(
            CommandType.ABORT_TASK, message, wire, self._now(now)
        )

    def ingest(self, data: bytes, *, now: float | None = None) -> list[LinkEvent]:
        current = self._now(now)
        events: list[LinkEvent] = []
        for frame in self._parser.feed(data):
            self.last_rx_at = current
            try:
                response_type = ResponseType(frame.message_type)
            except ValueError:
                events.append(LinkEvent("unknown", frame, current))
                continue

            try:
                if response_type is ResponseType.FIRMWARE_INFO:
                    info = FirmwareInfo.from_payload(frame.payload)
                    previous_boot = (
                        self.firmware_info.boot_id
                        if self.firmware_info is not None
                        else None
                    )
                    if previous_boot is not None and previous_boot != info.boot_id:
                        # Commands and volatile snapshots from the old boot can
                        # never be rebound to the new firmware session.
                        self._pending_arm_by_seq.clear()
                        self.last_ack = None
                        self.last_telemetry = None
                        self.last_safety = None
                        self.last_task_result_receipt = None
                        self.task_result_receipts = {}
                    self.firmware_info = info
                    self.last_identity_at = current
                    mismatch = self.expectation.mismatch(info)
                    self.identity_valid = mismatch is None
                    self.identity_error = mismatch or ""
                    events.append(LinkEvent("firmware_info", info, current))

                elif response_type is ResponseType.ACK:
                    ack = Ack.from_payload(frame.payload)
                    self.last_ack = ack
                    self.acks[(ack.ack_for_type, ack.seq)] = ack
                    command_receipt = self.command_receipts.get(
                        (ack.ack_for_type, ack.seq)
                    )
                    if ack.ack_for_type in {
                        int(CommandType.ARM_TASK),
                        int(CommandType.ARM_TIMED_TASK),
                    }:
                        pending_task_id = self._pending_arm_by_seq.pop(ack.seq, None)
                        if (
                            pending_task_id is not None
                            and ack.status == int(AckStatus.OK)
                            and ack.reason == int(AckReason.NONE)
                            and ack.task_id == pending_task_id
                        ):
                            self._last_arm_task_id = max(
                                self._last_arm_task_id, pending_task_id
                            )
                    events.append(LinkEvent("ack", ack, current))
                    if command_receipt is not None and self.firmware_info is not None:
                        raw_ack = frame.exact_wire_bytes
                        ack_receipt = CommandAckReceipt(
                            command=command_receipt,
                            ack=ack,
                            ack_raw_frame=raw_ack,
                            ack_raw_frame_sha256=hashlib.sha256(raw_ack).hexdigest(),
                            received_at=current,
                            firmware_build_id=str(self.firmware_info.build_id),
                            firmware_boot_id=boot_id_token(
                                self.firmware_info.boot_id
                            ),
                            execution_backend=(
                                command_receipt.execution_backend
                                if command_receipt.sent
                                else "ACK_OBSERVED_UNMARKED"
                            ),
                        )
                        self.command_ack_receipts[
                            (ack.ack_for_type, ack.seq)
                        ] = ack_receipt
                        self.last_command_ack_receipt = ack_receipt
                        events.append(
                            LinkEvent("command_ack_receipt", ack_receipt, current)
                        )

                elif response_type is ResponseType.IRRIGATION_TELEMETRY:
                    telemetry = IrrigationTelemetry.from_payload(frame.payload)
                    self.last_telemetry = telemetry
                    events.append(LinkEvent("telemetry", telemetry, current))

                elif response_type is ResponseType.SAFETY_STATE:
                    safety = SafetyState.from_payload(frame.payload)
                    if (
                        self.firmware_info is None
                        or safety.boot_id != self.firmware_info.boot_id
                    ):
                        events.append(LinkEvent("safety_boot_mismatch", frame, current))
                    else:
                        self.last_safety = safety
                        events.append(LinkEvent("safety", safety, current))

                elif response_type is ResponseType.TASK_RESULT:
                    result = TaskResult.from_payload(frame.payload)
                    if (
                        self.firmware_info is None
                        or not self.identity_fresh(current)
                        or result.boot_id != self.firmware_info.boot_id
                    ):
                        events.append(LinkEvent("task_result_rejected", frame, current))
                        continue
                    raw_frame = frame.exact_wire_bytes
                    receipt = TaskResultReceipt(
                        result=result,
                        raw_frame_sha256=hashlib.sha256(raw_frame).hexdigest(),
                        raw_frame=raw_frame,
                        received_at=current,
                        result_frame_seq=result.result_seq,
                        firmware_build_id=str(self.firmware_info.build_id),
                        firmware_boot_id=boot_id_token(self.firmware_info.boot_id),
                    )
                    history_key = (result.boot_id, result.task_id)

                    # A conflict is permanent for this boot/task evidence key.
                    # Neither replaying the original frame nor presenting a
                    # third variant can restore it to the trusted maps.
                    if history_key in self._task_result_conflicted_keys:
                        known_hashes = self._task_result_conflict_hashes[
                            history_key
                        ]
                        known_receipts = self.task_result_forensic_history.get(
                            history_key, []
                        )
                        # Raw bytes, rather than a digest alone, define an
                        # exact replay.  The SHA remains the evidence index,
                        # while this comparison also fails closed under a
                        # theoretical digest collision.
                        if any(
                            known.raw_frame == receipt.raw_frame
                            for known in known_receipts
                        ):
                            event_kind = "task_result_conflict_locked_duplicate"
                        else:
                            known_hashes.add(receipt.raw_frame_sha256)
                            self.task_result_conflicts.append(receipt)
                            self.task_result_forensic_history.setdefault(
                                history_key, []
                            ).append(receipt)
                            event_kind = "task_result_conflict_locked"
                        self.task_result_receipts.pop(result.task_id, None)
                        self.task_result_history.pop(history_key, None)
                        if (
                            self.last_task_result_receipt is not None
                            and self.last_task_result_receipt.result.boot_id
                            == result.boot_id
                            and self.last_task_result_receipt.result.task_id
                            == result.task_id
                        ):
                            self.last_task_result_receipt = None
                        events.append(LinkEvent(event_kind, receipt, current))
                        continue

                    previous = self.task_result_history.get(history_key)
                    if previous is not None:
                        if previous.raw_frame == receipt.raw_frame:
                            events.append(
                                LinkEvent("task_result_duplicate", previous, current)
                            )
                        else:
                            self._task_result_conflicted_keys.add(history_key)
                            self._task_result_conflict_hashes[history_key] = {
                                previous.raw_frame_sha256,
                                receipt.raw_frame_sha256,
                            }
                            self.task_result_conflicts.append(receipt)
                            self.task_result_forensic_history.setdefault(
                                history_key, [previous]
                            ).append(receipt)
                            # Conflict revokes the formerly trusted receipt.
                            self.task_result_history.pop(history_key, None)
                            self.task_result_receipts.pop(result.task_id, None)
                            if self.last_task_result_receipt is previous:
                                self.last_task_result_receipt = None
                            events.append(
                                LinkEvent("task_result_conflict", receipt, current)
                            )
                        continue
                    self.task_result_history[history_key] = receipt
                    self.task_result_forensic_history.setdefault(
                        history_key, []
                    ).append(receipt)
                    self.task_result_receipts[result.task_id] = receipt
                    self.last_task_result_receipt = receipt
                    events.append(LinkEvent("task_result", receipt, current))

                else:
                    events.append(LinkEvent("error", frame, current))
            except (PayloadError, ValueError, TypeError, struct.error):
                # A semantically malformed response never mutates the trusted
                # snapshot/receipt state and never escapes the ingest loop.
                events.append(LinkEvent("decode_error", frame, current))
        return events

    def ack_for(self, message_type: int | CommandType, seq: int) -> Ack | None:
        return self.acks.get((int(message_type), int(seq) & 0xFFFF))

    def command_receipt_for(
        self, message_type: int | CommandType, seq: int
    ) -> CommandFrameReceipt | None:
        return self.command_receipts.get((int(message_type), int(seq) & 0xFFFF))

    def command_ack_receipt_for(
        self, message_type: int | CommandType, seq: int
    ) -> CommandAckReceipt | None:
        return self.command_ack_receipts.get(
            (int(message_type), int(seq) & 0xFFFF)
        )

    def task_result_for(self, task_id: int) -> TaskResultReceipt | None:
        """Return the immutable receipt for the current firmware boot."""

        if self.firmware_info is None:
            return None
        if (self.firmware_info.boot_id, task_id) in self._task_result_conflicted_keys:
            return None
        return self.task_result_receipts.get(task_id)

    def task_result_conflicted(
        self, task_id: int, *, boot_id: int | None = None
    ) -> bool:
        """Return the latched conflict state for one terminal evidence key."""

        selected_boot = (
            self.firmware_info.boot_id
            if boot_id is None and self.firmware_info is not None
            else boot_id
        )
        if selected_boot is None:
            return False
        return (selected_boot, task_id) in self._task_result_conflicted_keys

    @staticmethod
    def ack_ok(
        ack: Ack | None,
        *,
        expected_type: int | CommandType | None = None,
        expected_task_id: int | None = None,
    ) -> bool:
        """Validate status *and* action-specific success reason.

        An emergency-stop reason can never make an ARM/CLEAR ACK look
        successful, and task-bearing calls can bind the ACK to their task ID.
        """

        if ack is None or ack.status != int(AckStatus.OK):
            return False
        if expected_type is not None and ack.ack_for_type != int(expected_type):
            return False
        if expected_task_id is not None and ack.task_id != int(expected_task_id):
            return False
        reason_by_type = {
            int(CommandType.EMERGENCY_STOP): int(AckReason.EMERGENCY_STOP),
            int(CommandType.ABORT_TASK): int(AckReason.USER_ABORT),
        }
        expected_reason = reason_by_type.get(ack.ack_for_type, int(AckReason.NONE))
        return ack.reason == expected_reason
