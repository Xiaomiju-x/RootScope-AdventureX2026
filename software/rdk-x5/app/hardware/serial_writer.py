"""Single-owner, bounded RootScope serial write scheduler.

The scheduler is transport-agnostic and has no import-time activity.  Frame
construction (including sequence allocation), queue binding and exact-byte
write receipts all occur behind one lock.  Only the claimed owner thread may
drain the queue or close the transport.  E-stop is always the highest priority
but is still written through this one owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import hashlib
import heapq
import threading
import time
from typing import Callable, Optional, Tuple

from ..serial.frame import (
    AbortTask,
    ArmTimedTask,
    ArmTask,
    CommandType,
    Heartbeat,
    SeqCommand,
    decode_frame,
)
from .physical_serial import SerialByteTransport


class SerialWriterError(RuntimeError):
    """Base class for fail-closed writer errors."""


class SerialWriterOwnershipError(SerialWriterError):
    """Raised when a non-owner attempts transport I/O."""


class SerialWriterLocked(SerialWriterError):
    """Raised when a normal command is attempted behind a safety barrier."""


class SerialWriterQueueFull(SerialWriterError):
    """Raised instead of silently dropping/reordering a command."""


class WriterPriority(IntEnum):
    EMERGENCY_STOP = 0
    ABORT = 10
    CONTROL = 20
    HEARTBEAT_OR_QUERY = 30


class WriterBarrier(str, Enum):
    ESTOP_REQUIRED = "ESTOP_REQUIRED"
    ESTOP_QUEUED = "ESTOP_QUEUED"
    ESTOP_WRITTEN_AWAITING_CONFIRMATION = "ESTOP_WRITTEN_AWAITING_CONFIRMATION"
    NORMAL_COMMANDS_ENABLED = "NORMAL_COMMANDS_ENABLED"
    WRITE_FAULT_LOCKED = "WRITE_FAULT_LOCKED"
    CLOSED = "CLOSED"


class WriteStatus(str, Enum):
    FULLY_WRITTEN = "FULLY_WRITTEN"
    FULLY_WRITTEN_AFTER_SHORT_WRITE = "FULLY_WRITTEN_AFTER_SHORT_WRITE"
    FAILED_NOT_WRITTEN = "FAILED_NOT_WRITTEN"
    PARTIAL_WRITE_AMBIGUOUS = "PARTIAL_WRITE_AMBIGUOUS"
    WRITE_OUTCOME_UNKNOWN = "WRITE_OUTCOME_UNKNOWN"
    FULLY_WRITTEN_RECEIPT_FAILED = "FULLY_WRITTEN_RECEIPT_FAILED"


class CancellationReason(str, Enum):
    EMERGENCY_STOP_INVALIDATED = "EMERGENCY_STOP_INVALIDATED"


@dataclass(frozen=True)
class ScheduledCommand:
    intent_id: str
    command_type: CommandType
    frame_seq: int
    raw_frame: bytes
    raw_frame_sha256: str
    priority: WriterPriority
    queued_at_monotonic: float
    task_id: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.intent_id, str) or not self.intent_id:
            raise ValueError("intent_id must be a non-empty string")
        if not 0 < self.frame_seq <= 0xFFFF:
            raise ValueError("frame_seq must be a non-zero uint16")
        if hashlib.sha256(self.raw_frame).hexdigest() != self.raw_frame_sha256:
            raise ValueError("scheduled command SHA-256 mismatch")


@dataclass(frozen=True)
class SerialWriteReceipt:
    command: ScheduledCommand
    status: WriteStatus
    bytes_confirmed_written: int
    write_calls: int
    started_at_monotonic: float
    completed_at_monotonic: float
    backend_id: str
    device_identity_sha256: str
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0 <= self.bytes_confirmed_written <= len(self.command.raw_frame):
            raise ValueError("invalid confirmed byte count")
        if self.write_calls < 1:
            raise ValueError("a write receipt requires at least one write call")
        if self.completed_at_monotonic < self.started_at_monotonic:
            raise ValueError("write receipt clock moved backwards")
        if self.status in {
            WriteStatus.FULLY_WRITTEN,
            WriteStatus.FULLY_WRITTEN_AFTER_SHORT_WRITE,
            WriteStatus.FULLY_WRITTEN_RECEIPT_FAILED,
        } and self.bytes_confirmed_written != len(self.command.raw_frame):
            raise ValueError("full-write status requires every byte")
        if self.status is WriteStatus.FAILED_NOT_WRITTEN and (
            self.bytes_confirmed_written != 0
        ):
            raise ValueError("not-written status cannot contain written bytes")

    @property
    def complete_frame_written(self) -> bool:
        return self.bytes_confirmed_written == len(self.command.raw_frame)

    @property
    def raw_frame_may_have_reached_device(self) -> bool:
        return self.status is not WriteStatus.FAILED_NOT_WRITTEN


@dataclass(frozen=True)
class SerialCancellationReceipt:
    """Audit record proving a queued frame was invalidated before transport I/O."""

    cancelled_command: ScheduledCommand
    reason: CancellationReason
    invalidated_by_intent_id: str
    invalidating_estop_sha256: str
    cancelled_at_monotonic: float

    def __post_init__(self) -> None:
        if self.cancelled_command.command_type is CommandType.EMERGENCY_STOP:
            raise ValueError("E-stop cannot be cancelled as a stale normal command")
        if not isinstance(self.invalidated_by_intent_id, str) or not (
            self.invalidated_by_intent_id
        ):
            raise ValueError("invalidated_by_intent_id must be non-empty")
        if (
            len(self.invalidating_estop_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.invalidating_estop_sha256
            )
        ):
            raise ValueError("invalidating_estop_sha256 must be lowercase SHA-256")

    @property
    def transport_write_attempted(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "cancelled_intent_id": self.cancelled_command.intent_id,
            "cancelled_command_type": self.cancelled_command.command_type.name,
            "cancelled_frame_seq": self.cancelled_command.frame_seq,
            "cancelled_raw_frame_sha256": (
                self.cancelled_command.raw_frame_sha256
            ),
            "cancelled_task_id": self.cancelled_command.task_id,
            "reason": self.reason.value,
            "invalidated_by_intent_id": self.invalidated_by_intent_id,
            "invalidating_estop_sha256": self.invalidating_estop_sha256,
            "cancelled_at_monotonic": self.cancelled_at_monotonic,
            "transport_write_attempted": False,
        }


@dataclass(frozen=True)
class StopConfirmation:
    command_sha256: str
    ack_matches_command: bool
    firmware_locked: bool
    pumps_all_off: bool
    evidence_fresh: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.command_sha256, str)
            or len(self.command_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.command_sha256)
        ):
            raise ValueError("command_sha256 must be lowercase SHA-256")
        for field_name in (
            "ack_matches_command",
            "firmware_locked",
            "pumps_all_off",
            "evidence_fresh",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")

    @property
    def accepted(self) -> bool:
        return bool(
            self.ack_matches_command
            and self.firmware_locked
            and self.pumps_all_off
            and self.evidence_fresh
        )


FrameBuilder = Callable[[], bytes]
WrittenCallback = Callable[[bytes, float], None]


def _decode_command(raw_frame: bytes) -> Tuple[CommandType, int]:
    frame = decode_frame(raw_frame)
    try:
        command_type = CommandType(frame.message_type)
    except ValueError as exc:
        raise ValueError("scheduled frame is not a known command") from exc
    if command_type is CommandType.ARM_TASK:
        seq = ArmTask.from_payload(frame.payload).seq
    elif command_type is CommandType.ARM_TIMED_TASK:
        seq = ArmTimedTask.from_payload(frame.payload).seq
    elif command_type is CommandType.ABORT_TASK:
        seq = AbortTask.from_payload(frame.payload).seq
    elif command_type is CommandType.HEARTBEAT:
        seq = Heartbeat.from_payload(frame.payload).seq
    else:
        seq = SeqCommand.from_payload(frame.payload).seq
    return command_type, seq


def _priority_for(command_type: CommandType) -> WriterPriority:
    if command_type is CommandType.EMERGENCY_STOP:
        return WriterPriority.EMERGENCY_STOP
    if command_type is CommandType.ABORT_TASK:
        return WriterPriority.ABORT
    if command_type in {
        CommandType.ARM_TASK,
        CommandType.ARM_TIMED_TASK,
        CommandType.CLEAR_ESTOP,
    }:
        return WriterPriority.CONTROL
    return WriterPriority.HEARTBEAT_OR_QUERY


class SerialWriterScheduler:
    """Own one already-open transport and serialize every downlink frame.

    E0 supplies no physical opener; tests use an in-memory transport.  Calling
    the constructor does not start a thread and does not touch the transport.
    """

    def __init__(
        self,
        transport: SerialByteTransport,
        *,
        normal_queue_limit: int = 32,
        emergency_queue_limit: int = 4,
        max_short_write_calls: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for field_name, value in (
            ("normal_queue_limit", normal_queue_limit),
            ("emergency_queue_limit", emergency_queue_limit),
            ("max_short_write_calls", max_short_write_calls),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        self._transport = transport
        self._normal_queue_limit = normal_queue_limit
        self._emergency_queue_limit = emergency_queue_limit
        self._max_short_write_calls = max_short_write_calls
        self._clock = clock
        self._lock = threading.RLock()
        self._queue: list[tuple[int, int, ScheduledCommand, Optional[WrittenCallback]]] = []
        self._queue_counter = 0
        self._normal_queued = 0
        self._emergency_queued = 0
        self._owner_thread_id: Optional[int] = None
        self._barrier = WriterBarrier.ESTOP_REQUIRED
        self._last_estop_write_sha256: Optional[str] = None
        self._receipts: list[SerialWriteReceipt] = []
        self._cancellations: list[SerialCancellationReceipt] = []

    @property
    def barrier(self) -> WriterBarrier:
        with self._lock:
            return self._barrier

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def receipts(self) -> Tuple[SerialWriteReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    @property
    def cancellations(self) -> Tuple[SerialCancellationReceipt, ...]:
        with self._lock:
            return tuple(self._cancellations)

    def _invalidate_queued_non_emergency(
        self, invalidating_command: ScheduledCommand
    ) -> None:
        retained: list[
            tuple[int, int, ScheduledCommand, Optional[WrittenCallback]]
        ] = []
        cancelled_at = float(self._clock())
        for item in self._queue:
            command = item[2]
            if command.command_type is CommandType.EMERGENCY_STOP:
                retained.append(item)
                continue
            self._cancellations.append(
                SerialCancellationReceipt(
                    cancelled_command=command,
                    reason=CancellationReason.EMERGENCY_STOP_INVALIDATED,
                    invalidated_by_intent_id=invalidating_command.intent_id,
                    invalidating_estop_sha256=(
                        invalidating_command.raw_frame_sha256
                    ),
                    cancelled_at_monotonic=cancelled_at,
                )
            )
        self._queue = retained
        heapq.heapify(self._queue)
        self._normal_queued = 0
        self._emergency_queued = len(retained)

    def claim_current_thread(self) -> None:
        thread_id = threading.get_ident()
        with self._lock:
            if self._owner_thread_id is None:
                self._owner_thread_id = thread_id
            elif self._owner_thread_id != thread_id:
                raise SerialWriterOwnershipError("serial writer already has an owner")

    def _require_owner(self) -> None:
        if self._owner_thread_id != threading.get_ident():
            raise SerialWriterOwnershipError(
                "only the claimed serial writer thread may perform transport I/O"
            )

    def schedule(
        self,
        *,
        intent_id: str,
        build_frame: FrameBuilder,
        expected_command_type: CommandType,
        task_id: Optional[str] = None,
        on_fully_written: Optional[WrittenCallback] = None,
    ) -> ScheduledCommand:
        """Build and bind one exact command under the scheduler lock."""

        with self._lock:
            if self._barrier is WriterBarrier.CLOSED:
                raise SerialWriterLocked("serial writer is closed")
            expected_is_estop = (
                expected_command_type is CommandType.EMERGENCY_STOP
            )
            if (
                self._barrier is WriterBarrier.WRITE_FAULT_LOCKED
                and not expected_is_estop
            ):
                raise SerialWriterLocked("writer fault permits E-stop only")
            if (
                self._barrier
                in {
                    WriterBarrier.ESTOP_REQUIRED,
                    WriterBarrier.ESTOP_QUEUED,
                    WriterBarrier.ESTOP_WRITTEN_AWAITING_CONFIRMATION,
                }
                and not expected_is_estop
            ):
                raise SerialWriterLocked(
                    "initial/recent E-stop is not yet confirmed locked and pump-off"
                )
            if expected_is_estop:
                if self._emergency_queued >= self._emergency_queue_limit:
                    raise SerialWriterQueueFull("emergency queue is full")
            elif self._normal_queued >= self._normal_queue_limit:
                raise SerialWriterQueueFull("normal command queue is full")
            raw_frame = bytes(build_frame())
            command_type, frame_seq = _decode_command(raw_frame)
            if command_type is not expected_command_type:
                raise ValueError("built command type does not match expectation")
            is_estop = command_type is CommandType.EMERGENCY_STOP
            priority = _priority_for(command_type)
            command = ScheduledCommand(
                intent_id=intent_id,
                command_type=command_type,
                frame_seq=frame_seq,
                raw_frame=raw_frame,
                raw_frame_sha256=hashlib.sha256(raw_frame).hexdigest(),
                priority=priority,
                queued_at_monotonic=float(self._clock()),
                task_id=task_id,
            )
            if is_estop:
                # A safety stop defines a new command epoch.  Commands built
                # before it must never survive confirmation and execute later.
                self._invalidate_queued_non_emergency(command)
                self._emergency_queued += 1
                self._barrier = WriterBarrier.ESTOP_QUEUED
            else:
                self._normal_queued += 1
            self._queue_counter += 1
            heapq.heappush(
                self._queue,
                (int(priority), self._queue_counter, command, on_fully_written),
            )
            return command

    def write_next(self) -> Optional[SerialWriteReceipt]:
        """Write one entire frame; never interleave another queued frame."""

        with self._lock:
            self._require_owner()
            if not self._queue:
                return None
            next_command = self._queue[0][2]
            if (
                self._barrier
                in {
                    WriterBarrier.ESTOP_REQUIRED,
                    WriterBarrier.ESTOP_QUEUED,
                    WriterBarrier.ESTOP_WRITTEN_AWAITING_CONFIRMATION,
                    WriterBarrier.WRITE_FAULT_LOCKED,
                }
                and next_command.command_type is not CommandType.EMERGENCY_STOP
            ):
                raise SerialWriterLocked(
                    "safety barrier permits only an E-stop write"
                )
            _priority, _order, command, callback = heapq.heappop(self._queue)
            if command.command_type is CommandType.EMERGENCY_STOP:
                self._emergency_queued -= 1
            else:
                self._normal_queued -= 1
            started = float(self._clock())
            offset = 0
            calls = 0
            short_write = False
            status: WriteStatus
            error: Optional[str] = None
            try:
                while offset < len(command.raw_frame):
                    calls += 1
                    if calls > self._max_short_write_calls:
                        status = (
                            WriteStatus.FAILED_NOT_WRITTEN
                            if offset == 0
                            else WriteStatus.PARTIAL_WRITE_AMBIGUOUS
                        )
                        error = "short-write retry limit exceeded"
                        break
                    written = self._transport.write(command.raw_frame[offset:])
                    if isinstance(written, bool) or not isinstance(written, int):
                        raise TypeError("transport write count must be an integer")
                    remaining = len(command.raw_frame) - offset
                    if not 0 <= written <= remaining:
                        raise ValueError("transport returned an impossible write count")
                    if written < remaining:
                        short_write = True
                    offset += written
                else:
                    status = (
                        WriteStatus.FULLY_WRITTEN_AFTER_SHORT_WRITE
                        if short_write
                        else WriteStatus.FULLY_WRITTEN
                    )
            except BaseException as exc:
                status = WriteStatus.WRITE_OUTCOME_UNKNOWN
                error = f"{type(exc).__name__}: {exc}"

            completed = float(self._clock())
            if status in {
                WriteStatus.FULLY_WRITTEN,
                WriteStatus.FULLY_WRITTEN_AFTER_SHORT_WRITE,
            } and callback is not None:
                try:
                    callback(command.raw_frame, completed)
                except BaseException as exc:
                    status = WriteStatus.FULLY_WRITTEN_RECEIPT_FAILED
                    error = f"{type(exc).__name__}: {exc}"

            receipt = SerialWriteReceipt(
                command=command,
                status=status,
                bytes_confirmed_written=offset,
                write_calls=max(calls, 1),
                started_at_monotonic=started,
                completed_at_monotonic=completed,
                backend_id=self._transport.backend_id,
                device_identity_sha256=self._transport.device_identity_sha256,
                error=error,
            )
            self._receipts.append(receipt)

            if command.command_type is CommandType.EMERGENCY_STOP and (
                receipt.complete_frame_written
                and status is not WriteStatus.FULLY_WRITTEN_RECEIPT_FAILED
            ):
                self._last_estop_write_sha256 = command.raw_frame_sha256
                self._barrier = WriterBarrier.ESTOP_WRITTEN_AWAITING_CONFIRMATION
            elif status not in {
                WriteStatus.FULLY_WRITTEN,
                WriteStatus.FULLY_WRITTEN_AFTER_SHORT_WRITE,
            }:
                self._barrier = WriterBarrier.WRITE_FAULT_LOCKED
            return receipt

    def confirm_stop(self, confirmation: StopConfirmation) -> None:
        with self._lock:
            if self._barrier is not WriterBarrier.ESTOP_WRITTEN_AWAITING_CONFIRMATION:
                raise SerialWriterLocked("no E-stop write is awaiting confirmation")
            if confirmation.command_sha256 != self._last_estop_write_sha256:
                self._barrier = WriterBarrier.WRITE_FAULT_LOCKED
                raise SerialWriterLocked("stop confirmation does not bind latest E-stop")
            if not confirmation.accepted:
                self._barrier = WriterBarrier.WRITE_FAULT_LOCKED
                raise SerialWriterLocked("stop confirmation failed closed")
            self._barrier = (
                WriterBarrier.ESTOP_QUEUED
                if self._emergency_queued
                else WriterBarrier.NORMAL_COMMANDS_ENABLED
            )

    def close(self) -> None:
        with self._lock:
            self._require_owner()
            if self._queue:
                raise SerialWriterLocked("refusing to close with queued commands")
            self._transport.close()
            self._barrier = WriterBarrier.CLOSED
