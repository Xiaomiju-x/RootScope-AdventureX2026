"""RootScope v1 wire framing and fixed-width payload codecs.

This module deliberately has no serial-port dependency.  It is the shared,
deterministic protocol layer for the X5 application, the fake F407, and the
future independently-built STM32 firmware.

Wire frame (all integers little-endian inside payloads)::

    0xAA 0x55 TYPE LEN PAYLOAD... SUM8

``SUM8`` is the unsigned eight-bit sum from the first header byte through the
last payload byte.  It is a corruption detector, not a CRC or an authenticity
mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
import struct
from typing import ClassVar, Protocol


HEADER = b"\xAA\x55"
MAX_FRAME_SIZE = 256
MAX_PAYLOAD_SIZE = MAX_FRAME_SIZE - 5

PROTOCOL_VERSION = 1
EXPECTED_BUILD_ID = 2026071501
EXPECTED_HW_VARIANT = 1
F103_PB7_BUILD_ID = 2026072501
F103_PB7_HW_VARIANT = 2
F103_PB6_BUILD_ID = 2026072513
F103_PB6_HW_VARIANT = 2
F103_Z3_PB6_BUILD_ID = 2026072515
F103_Z3_PB6_HW_VARIANT = 2
BUILD_TAG_SIZE = 16
MIN_TARGET_RESULT_SAMPLES = 5


class CommandType(IntEnum):
    """X5 -> F407 message identifiers, frozen for RootScope v1."""

    EMERGENCY_STOP = 0x10
    CLEAR_ESTOP = 0x11
    QUERY_FIRMWARE = 0x12
    ARM_TASK = 0x20
    ABORT_TASK = 0x21
    ARM_TIMED_TASK = 0x22
    HEARTBEAT = 0xFF


class ResponseType(IntEnum):
    """F407 -> X5 message identifiers, frozen for RootScope v1."""

    FIRMWARE_INFO = 0x81
    SAFETY_STATE = 0x82
    IRRIGATION_TELEMETRY = 0x83
    TASK_RESULT = 0x84
    ACK = 0x90
    ERROR = 0x9F


class Capability(IntFlag):
    PUMP_INTERLOCK = 1 << 0
    # Backward-compatible name for the original F407 three-pump profile.
    TRIPLE_PUMP_INTERLOCK = PUMP_INTERLOCK
    HX711_MASS = 1 << 1
    SAFETY_INPUTS = 1 << 2
    HEARTBEAT_WATCHDOG = 1 << 3
    TASK_REPLAY_GUARD = 1 << 4
    PER_TASK_HARD_TIMEOUT = 1 << 5
    TASK_RESULT_RECEIPT = 1 << 6


REQUIRED_CAPABILITIES = int(
    Capability.TRIPLE_PUMP_INTERLOCK
    | Capability.HX711_MASS
    | Capability.SAFETY_INPUTS
    | Capability.HEARTBEAT_WATCHDOG
    | Capability.TASK_REPLAY_GUARD
    | Capability.PER_TASK_HARD_TIMEOUT
    | Capability.TASK_RESULT_RECEIPT
)

F103_PB7_REQUIRED_CAPABILITIES = int(
    Capability.PUMP_INTERLOCK
    | Capability.HEARTBEAT_WATCHDOG
    | Capability.TASK_REPLAY_GUARD
    | Capability.PER_TASK_HARD_TIMEOUT
    | Capability.TASK_RESULT_RECEIPT
)

F103_PB6_REQUIRED_CAPABILITIES = int(
    Capability.PUMP_INTERLOCK
    | Capability.HEARTBEAT_WATCHDOG
    | Capability.TASK_REPLAY_GUARD
    | Capability.PER_TASK_HARD_TIMEOUT
    | Capability.TASK_RESULT_RECEIPT
)

F103_Z3_PB6_REQUIRED_CAPABILITIES = F103_PB6_REQUIRED_CAPABILITIES


class SafetyBits(IntFlag):
    """Normalized F407 input/state bits reported in both safety and telemetry."""

    ESTOP_ACTIVE = 1 << 0
    LEAK_DETECTED = 1 << 1
    CARTRIDGE_PRESENT = 1 << 2
    GUARD_CLOSED = 1 << 3
    HX711_VALID = 1 << 4
    WATCHDOG_FRESH = 1 << 5
    LOCK_LATCHED = 1 << 6
    ACT_ENABLE = 1 << 7


class AckStatus(IntEnum):
    OK = 0
    REJECTED = 1
    LOCKED = 2
    BAD_PAYLOAD = 3


class AckReason(IntEnum):
    NONE = 0
    DUPLICATE_SEQ = 1
    STALE_SEQ = 2
    DUPLICATE_TASK = 3
    STALE_TASK = 4
    INVALID_CHANNEL = 5
    INVALID_TARGET_MASS = 6
    INVALID_HARD_TIMEOUT = 7
    UNSAFE_INPUT = 8
    WATCHDOG_TIMEOUT = 9
    BUSY = 10
    NO_ACTIVE_TASK = 11
    TASK_MISMATCH = 12
    CLEAR_CONDITIONS_NOT_MET = 13
    HARD_TIMEOUT = 14
    MALFORMED_PAYLOAD = 15
    UNKNOWN_TYPE = 16
    EMERGENCY_STOP = 17
    USER_ABORT = 18
    BOOT_LOCK = 19
    TARGET_REACHED = 20
    UNSUPPORTED_CAPABILITY = 21
    TIMED_DOSE_COMPLETE = 22
    INVALID_DURATION = 23


class LockReason(IntEnum):
    NONE = 0
    BOOT_LOCK = 1
    WATCHDOG_TIMEOUT = 2
    HARD_TIMEOUT = 3
    EMERGENCY_STOP = 4
    UNSAFE_INPUT = 5
    USER_ABORT = 6


class TerminalReason(IntEnum):
    """Firmware-originated terminal reason in :class:`TaskResult`."""

    TARGET_REACHED = 1
    HARD_TIMEOUT = 2
    USER_ABORT = 3
    SAFETY_INPUT = 4
    EMERGENCY_STOP = 5
    WATCHDOG_TIMEOUT = 6
    TIMED_DOSE_COMPLETE = 7


class FrameError(ValueError):
    """Base protocol decode failure."""


class HeaderError(FrameError):
    pass


class TruncatedFrame(FrameError):
    pass


class LengthError(FrameError):
    pass


class ChecksumError(FrameError):
    pass


class PayloadError(FrameError):
    pass


def _strict_int(value: object, field_name: str) -> int:
    """Accept Python integers/IntEnums only; reject bool and coercion."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer, not {type(value).__name__}")
    return int(value)


def _bounded_int(value: object, field_name: str, lower: int, upper: int) -> int:
    number = _strict_int(value, field_name)
    if not lower <= number <= upper:
        raise ValueError(f"{field_name} must be within [{lower}, {upper}]")
    return number


def _u8(value: object, field_name: str) -> int:
    return _bounded_int(value, field_name, 0, 0xFF)


def _u16(value: object, field_name: str) -> int:
    return _bounded_int(value, field_name, 0, 0xFFFF)


def _u32(value: object, field_name: str) -> int:
    return _bounded_int(value, field_name, 0, 0xFFFFFFFF)


def _u64(value: object, field_name: str) -> int:
    return _bounded_int(value, field_name, 0, 0xFFFFFFFFFFFFFFFF)


def _i32(value: object, field_name: str) -> int:
    return _bounded_int(value, field_name, -(1 << 31), (1 << 31) - 1)


def _wire_bytes(value: object, field_name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{field_name} must be bytes-like")
    return bytes(value)


def boot_id_token(boot_id: int) -> str:
    """Canonical token used by core/evidence for a wire ``u64`` boot ID."""

    return f"boot-{_u64(boot_id, 'boot_id'):016x}"


@dataclass(frozen=True)
class Frame:
    message_type: int
    payload: bytes = b""
    wire_bytes: bytes | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        message_type = _u8(self.message_type, "message_type")
        payload = _wire_bytes(self.payload, "payload")
        if len(payload) > MAX_PAYLOAD_SIZE:
            raise ValueError(
                f"payload is {len(payload)} bytes; maximum is {MAX_PAYLOAD_SIZE}"
            )
        exact_wire = (
            None
            if self.wire_bytes is None
            else _wire_bytes(self.wire_bytes, "wire_bytes")
        )
        object.__setattr__(self, "message_type", message_type)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "wire_bytes", exact_wire)

    def encode(self) -> bytes:
        return encode_frame(self.message_type, self.payload)

    @property
    def exact_wire_bytes(self) -> bytes:
        """Original bytes from the parser, or canonical encoding for new frames."""

        return self.wire_bytes if self.wire_bytes is not None else self.encode()


def sum8(data: bytes | bytearray | memoryview) -> int:
    """Return the unsigned eight-bit additive checksum."""

    return sum(_wire_bytes(data, "data")) & 0xFF


def encode_frame(message_type: int | IntEnum, payload: bytes = b"") -> bytes:
    frame = Frame(message_type, payload)
    body = HEADER + bytes((frame.message_type, len(frame.payload))) + frame.payload
    return body + bytes((sum8(body),))


def decode_frame(data: bytes | bytearray | memoryview) -> Frame:
    """Strictly decode exactly one complete frame.

    Streaming callers should use :class:`FrameParser`; this function rejects
    both truncation and trailing bytes so tests and file replays fail closed.
    """

    raw = _wire_bytes(data, "data")
    if len(raw) < 2:
        raise TruncatedFrame("frame header is truncated")
    if raw[:2] != HEADER:
        raise HeaderError("missing 0xAA55 header")
    if len(raw) < 4:
        raise TruncatedFrame("type/length field is truncated")
    payload_length = raw[3]
    if payload_length > MAX_PAYLOAD_SIZE:
        raise LengthError(
            f"payload length {payload_length} exceeds maximum {MAX_PAYLOAD_SIZE}"
        )
    expected = 5 + payload_length
    if len(raw) < expected:
        raise TruncatedFrame(f"need {expected} bytes, got {len(raw)}")
    if len(raw) > expected:
        raise LengthError(f"expected exactly {expected} bytes, got {len(raw)}")
    expected_checksum = sum8(raw[:-1])
    if raw[-1] != expected_checksum:
        raise ChecksumError(
            f"sum8 mismatch: wire=0x{raw[-1]:02X}, expected=0x{expected_checksum:02X}"
        )
    return Frame(raw[2], raw[4:-1], wire_bytes=raw)


class FrameParser:
    """Incremental parser that tolerates noise, partial reads, and bad frames."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.frames_decoded = 0
        self.checksum_errors = 0
        self.length_errors = 0
        self.discarded_bytes = 0

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()

    def feed(self, data: bytes | bytearray | memoryview) -> list[Frame]:
        self._buffer.extend(_wire_bytes(data, "data"))
        decoded: list[Frame] = []

        while True:
            start = self._buffer.find(HEADER)
            if start < 0:
                # Preserve a final 0xAA because it may be the first header byte
                # of the next read.
                keep = 1 if self._buffer.endswith(HEADER[:1]) else 0
                drop = len(self._buffer) - keep
                if drop:
                    del self._buffer[:drop]
                    self.discarded_bytes += drop
                break
            if start:
                del self._buffer[:start]
                self.discarded_bytes += start
            if len(self._buffer) < 4:
                break

            payload_length = self._buffer[3]
            if payload_length > MAX_PAYLOAD_SIZE:
                # Invalid lengths (252..255) can otherwise make a parser wait
                # for an impossible >256-byte frame.  Drop one header byte and
                # search again so a following valid frame is recovered.
                del self._buffer[0]
                self.length_errors += 1
                self.discarded_bytes += 1
                continue
            total_length = 5 + payload_length
            if len(self._buffer) < total_length:
                break

            candidate = bytes(self._buffer[:total_length])
            try:
                frame = decode_frame(candidate)
            except (ChecksumError, LengthError):
                # Drop only one byte so an embedded valid 0xAA55 can be found.
                del self._buffer[0]
                if payload_length > MAX_PAYLOAD_SIZE:
                    self.length_errors += 1
                else:
                    self.checksum_errors += 1
                self.discarded_bytes += 1
                continue

            del self._buffer[:total_length]
            decoded.append(frame)
            self.frames_decoded += 1

        return decoded


class PayloadCodec(Protocol):
    def to_payload(self) -> bytes: ...


def encode_message(message_type: int | IntEnum, message: PayloadCodec) -> bytes:
    return encode_frame(message_type, message.to_payload())


def _unpack_exact(fmt: struct.Struct, payload: bytes, name: str) -> tuple[object, ...]:
    if len(payload) != fmt.size:
        raise PayloadError(f"{name} needs {fmt.size} bytes, got {len(payload)}")
    return fmt.unpack(payload)


@dataclass(frozen=True)
class SeqCommand:
    seq: int
    STRUCT: ClassVar[struct.Struct] = struct.Struct("<H")

    def to_payload(self) -> bytes:
        return self.STRUCT.pack(_u16(self.seq, "seq"))

    @classmethod
    def from_payload(cls, payload: bytes) -> "SeqCommand":
        (seq,) = _unpack_exact(cls.STRUCT, payload, cls.__name__)
        return cls(int(seq))


@dataclass(frozen=True)
class Heartbeat:
    seq: int
    host_state: int = 0
    STRUCT: ClassVar[struct.Struct] = struct.Struct("<HB")

    def to_payload(self) -> bytes:
        return self.STRUCT.pack(
            _u16(self.seq, "seq"), _u8(self.host_state, "host_state")
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> "Heartbeat":
        seq, host_state = _unpack_exact(cls.STRUCT, payload, cls.__name__)
        return cls(int(seq), int(host_state))


@dataclass(frozen=True)
class ArmTask:
    task_id: int
    seq: int
    channel: int
    target_mass_mg: int
    hard_timeout_ms: int
    config_hash_prefix: bytes
    STRUCT: ClassVar[struct.Struct] = struct.Struct("<IH BII8s")

    def to_payload(self) -> bytes:
        config_hash = _wire_bytes(self.config_hash_prefix, "config_hash_prefix")
        if len(config_hash) != 8:
            raise ValueError("config_hash_prefix must be exactly 8 raw bytes")
        return self.STRUCT.pack(
            _u32(self.task_id, "task_id"),
            _u16(self.seq, "seq"),
            _u8(self.channel, "channel"),
            _u32(self.target_mass_mg, "target_mass_mg"),
            _u32(self.hard_timeout_ms, "hard_timeout_ms"),
            config_hash,
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> "ArmTask":
        values = _unpack_exact(cls.STRUCT, payload, cls.__name__)
        return cls(
            task_id=int(values[0]),
            seq=int(values[1]),
            channel=int(values[2]),
            target_mass_mg=int(values[3]),
            hard_timeout_ms=int(values[4]),
            config_hash_prefix=bytes(values[5]),
        )


@dataclass(frozen=True)
class ArmTimedTask:
    """Bounded single-pump task for the real F103 pump-only hardware profile.

    This is intentionally distinct from :class:`ArmTask`: the supplied board
    has no HX711, so a duration must never be represented as measured mass.
    """

    task_id: int
    seq: int
    channel: int
    duration_ms: int
    hard_timeout_ms: int
    config_hash_prefix: bytes
    STRUCT: ClassVar[struct.Struct] = struct.Struct("<IH BII8s")

    def to_payload(self) -> bytes:
        config_hash = _wire_bytes(self.config_hash_prefix, "config_hash_prefix")
        if len(config_hash) != 8:
            raise ValueError("config_hash_prefix must be exactly 8 raw bytes")
        return self.STRUCT.pack(
            _u32(self.task_id, "task_id"),
            _u16(self.seq, "seq"),
            _u8(self.channel, "channel"),
            _u32(self.duration_ms, "duration_ms"),
            _u32(self.hard_timeout_ms, "hard_timeout_ms"),
            config_hash,
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> "ArmTimedTask":
        values = _unpack_exact(cls.STRUCT, payload, cls.__name__)
        return cls(
            task_id=int(values[0]),
            seq=int(values[1]),
            channel=int(values[2]),
            duration_ms=int(values[3]),
            hard_timeout_ms=int(values[4]),
            config_hash_prefix=bytes(values[5]),
        )


@dataclass(frozen=True)
class AbortTask:
    task_id: int
    seq: int
    reason: int = 0
    STRUCT: ClassVar[struct.Struct] = struct.Struct("<IHB")

    def to_payload(self) -> bytes:
        return self.STRUCT.pack(
            _u32(self.task_id, "task_id"),
            _u16(self.seq, "seq"),
            _u8(self.reason, "reason"),
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> "AbortTask":
        task_id, seq, reason = _unpack_exact(cls.STRUCT, payload, cls.__name__)
        return cls(int(task_id), int(seq), int(reason))


@dataclass(frozen=True)
class FirmwareInfo:
    protocol_version: int
    capabilities: int
    build_id: int
    hw_variant: int
    build_tag: str
    boot_id: int
    STRUCT: ClassVar[struct.Struct] = struct.Struct("<HII B16sQ")

    def to_payload(self) -> bytes:
        if not isinstance(self.build_tag, str):
            raise TypeError("build_tag must be str")
        encoded_tag = self.build_tag.encode("ascii", errors="strict")
        if len(encoded_tag) > BUILD_TAG_SIZE:
            raise ValueError(f"build_tag must be at most {BUILD_TAG_SIZE} ASCII bytes")
        return self.STRUCT.pack(
            _u16(self.protocol_version, "protocol_version"),
            _u32(self.capabilities, "capabilities"),
            _u32(self.build_id, "build_id"),
            _u8(self.hw_variant, "hw_variant"),
            encoded_tag.ljust(BUILD_TAG_SIZE, b"\x00"),
            _u64(self.boot_id, "boot_id"),
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> "FirmwareInfo":
        protocol, capabilities, build_id, hw_variant, tag, boot_id = _unpack_exact(
            cls.STRUCT, payload, cls.__name__
        )
        return cls(
            int(protocol),
            int(capabilities),
            int(build_id),
            int(hw_variant),
            bytes(tag).split(b"\x00", 1)[0].decode("ascii", errors="replace"),
            int(boot_id),
        )


@dataclass(frozen=True)
class Ack:
    ack_for_type: int
    seq: int
    status: int
    reason: int
    task_id: int = 0
    STRUCT: ClassVar[struct.Struct] = struct.Struct("<B HBBI")

    def to_payload(self) -> bytes:
        return self.STRUCT.pack(
            _u8(self.ack_for_type, "ack_for_type"),
            _u16(self.seq, "seq"),
            _u8(self.status, "status"),
            _u8(self.reason, "reason"),
            _u32(self.task_id, "task_id"),
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> "Ack":
        values = _unpack_exact(cls.STRUCT, payload, cls.__name__)
        return cls(*(int(value) for value in values))


@dataclass(frozen=True)
class IrrigationTelemetry:
    task_id: int
    sample_seq: int
    pump_mask: int
    hx711_raw: int
    filtered_mass_mg: int
    safety_bits: int
    uptime_ms: int
    STRUCT: ClassVar[struct.Struct] = struct.Struct("<IIBiiHI")

    def to_payload(self) -> bytes:
        return self.STRUCT.pack(
            _u32(self.task_id, "task_id"),
            _u32(self.sample_seq, "sample_seq"),
            _u8(self.pump_mask, "pump_mask"),
            _i32(self.hx711_raw, "hx711_raw"),
            _i32(self.filtered_mass_mg, "filtered_mass_mg"),
            _u16(self.safety_bits, "safety_bits"),
            _u32(self.uptime_ms, "uptime_ms"),
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> "IrrigationTelemetry":
        values = _unpack_exact(cls.STRUCT, payload, cls.__name__)
        return cls(*(int(value) for value in values))


@dataclass(frozen=True)
class SafetyState:
    boot_id: int
    safety_bits: int
    blocked_count: int
    lock_reason: int
    heartbeat_age_ms: int
    STRUCT: ClassVar[struct.Struct] = struct.Struct("<QHHBI")

    def to_payload(self) -> bytes:
        return self.STRUCT.pack(
            _u64(self.boot_id, "boot_id"),
            _u16(self.safety_bits, "safety_bits"),
            _u16(self.blocked_count, "blocked_count"),
            _u8(self.lock_reason, "lock_reason"),
            _u32(self.heartbeat_age_ms, "heartbeat_age_ms"),
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> "SafetyState":
        values = _unpack_exact(cls.STRUCT, payload, cls.__name__)
        return cls(*(int(value) for value in values))


@dataclass(frozen=True)
class TaskResult:
    """Terminal result authored by F407 after every accepted physical task."""

    boot_id: int
    task_id: int
    result_seq: int
    terminal_reason: int
    baseline_mass_mg: int
    final_mass_mg: int
    first_sample_seq: int
    last_sample_seq: int
    sample_count: int
    final_window_min_mg: int
    final_window_max_mg: int
    scale_stable: bool
    firmware_completed_uptime_ms: int
    pump_mask: int
    safety_bits: int
    STRUCT: ClassVar[struct.Struct] = struct.Struct("<QIH BiiIIHiiBIBH")

    def to_payload(self) -> bytes:
        boot_id = _u64(self.boot_id, "boot_id")
        task_id = _u32(self.task_id, "task_id")
        result_seq = _u16(self.result_seq, "result_seq")
        terminal_reason = _u8(self.terminal_reason, "terminal_reason")
        try:
            reason = TerminalReason(terminal_reason)
        except ValueError as exc:
            raise ValueError("unknown terminal_reason") from exc
        baseline_mass = _i32(self.baseline_mass_mg, "baseline_mass_mg")
        final_mass = _i32(self.final_mass_mg, "final_mass_mg")
        first_seq = _u32(self.first_sample_seq, "first_sample_seq")
        last_seq = _u32(self.last_sample_seq, "last_sample_seq")
        sample_count = _u16(self.sample_count, "sample_count")
        pump_mask = _u8(self.pump_mask, "pump_mask")
        window_min = _i32(self.final_window_min_mg, "final_window_min_mg")
        window_max = _i32(self.final_window_max_mg, "final_window_max_mg")
        if not isinstance(self.scale_stable, bool):
            raise TypeError("scale_stable must be bool")
        if boot_id == 0 or task_id == 0 or result_seq == 0:
            raise ValueError("boot_id, task_id, and result_seq must be non-zero")
        if baseline_mass < 0 or final_mass < 0:
            raise ValueError("TaskResult masses cannot be negative")
        if pump_mask != 0:
            raise ValueError("terminal TASK_RESULT requires pump_mask=0")
        if last_seq < first_seq:
            raise ValueError("last_sample_seq precedes first_sample_seq")
        if sample_count == 0:
            raise ValueError("sample_count cannot be zero")
        if sample_count > last_seq - first_seq + 1:
            raise ValueError("sample_count exceeds declared sequence window")
        if window_max < window_min:
            raise ValueError("final_window_max_mg precedes final_window_min_mg")
        if reason is TerminalReason.TARGET_REACHED and (
            not self.scale_stable or sample_count < MIN_TARGET_RESULT_SAMPLES
        ):
            raise ValueError(
                "TARGET_REACHED requires firmware-stable post-stop sample window"
            )
        if reason is TerminalReason.TIMED_DOSE_COMPLETE and self.scale_stable:
            raise ValueError(
                "TIMED_DOSE_COMPLETE cannot claim a stable physical scale result"
            )
        return self.STRUCT.pack(
            boot_id,
            task_id,
            result_seq,
            terminal_reason,
            baseline_mass,
            final_mass,
            first_seq,
            last_seq,
            sample_count,
            window_min,
            window_max,
            1 if self.scale_stable else 0,
            _u32(
                self.firmware_completed_uptime_ms,
                "firmware_completed_uptime_ms",
            ),
            pump_mask,
            _u16(self.safety_bits, "safety_bits"),
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> "TaskResult":
        values = _unpack_exact(cls.STRUCT, payload, cls.__name__)
        if int(values[11]) not in (0, 1):
            raise PayloadError("TaskResult scale_stable must be 0 or 1")
        converted = [int(value) for value in values]
        converted[11] = bool(converted[11])
        result = cls(*converted)
        # Apply semantic validation as well as fixed-size decoding.
        result.to_payload()
        if result.task_id == 0:
            raise PayloadError("TaskResult task_id cannot be zero")
        if result.result_seq == 0:
            raise PayloadError("TaskResult result_seq cannot be zero")
        try:
            TerminalReason(result.terminal_reason)
        except ValueError as exc:
            raise PayloadError("unknown TaskResult terminal_reason") from exc
        return result

    @property
    def terminal_reason_name(self) -> str:
        return TerminalReason(self.terminal_reason).name

    @property
    def firmware_boot_id(self) -> str:
        return boot_id_token(self.boot_id)

    @property
    def post_stop_sample_count(self) -> int:
        return self.sample_count

    @property
    def final_window_span_mg(self) -> int:
        return self.final_window_max_mg - self.final_window_min_mg
