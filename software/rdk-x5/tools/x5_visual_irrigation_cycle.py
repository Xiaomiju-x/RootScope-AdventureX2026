#!/usr/bin/env python3
"""One-shot RootScope visual -> probe -> timed-pump physical cycle.

The cycle is intentionally bounded and fail-closed:

1. Three consecutive dual-evidence frames must agree on one target card.
2. The V15 firmware identity, USB identity and boot-safe outputs are checked.
3. Exactly one downward depth preset is executed; no automatic return exists.
4. The commissioned continuous mode runs one fixed 5 s PB6 task after its
   calibrated motion window; the legacy DONE-gated path remains available.
5. The pump task is firmware-timed, heartbeat-guarded, never retried, and ends
   with E-stop plus final safe-state verification.

The program executes at most one physical cycle per invocation.  Pure sand,
unknown/low-confidence frames, conflicting evidence, timeouts and any serial
error produce HOLD/STOP and never advance to the next stage.
"""

from __future__ import annotations

import argparse
try:
    import fcntl
except ImportError:  # Protocol and dry-run helpers remain importable on Windows.
    fcntl = None  # type: ignore[assignment]
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION_ROOT = Path(__file__).resolve().parents[1]
if str(VERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION_ROOT))

from app.hardware.device_identity import UsbDeviceIdentity
from tools import stm32_z3_pb6_5000ms
from tools.stm32_z3_level1_first_descent import (
CMD_CLEAR_ESTOP,
    CMD_HEARTBEAT,
    SerialSession,
    ascii_line,
    encode_frame,
    firmware_query,
    parse_csv_fields,
)


EXPECTED_VERSION = "2026-07-25-RS-F103-Z3-PB6-V15"
EXPECTED_BUILD_ID = 2026072515
EXPECTED_CAPABILITIES = 0x00000079
EXPECTED_VARIANT = 2
CLASS_TO_LEVEL = {
    "grass_clump": 1,
    "low_shrub": 2,
    "young_tree": 3,
    "non_target": 0,
}
LEVEL_TO_STEPS = {0: 0, 1: 1024, 2: 1536, 3: 2048}
LEVEL_TIMEOUT_S = {1: 30.0, 2: 35.0, 3: 45.0}
TARGET_CLASSES = frozenset({"grass_clump", "low_shrub", "young_tree"})
ARM_TOKEN = "ARM ROOTSCOPE ONE CYCLE"
CMD_ARM_TIMED_TASK = 0x22
PUMP_CONFIG_SHA256 = (
    "54f66f0dec8c043623245511972b3d0568f43c77b25323bddc1f1a628ef9426e"
)
PUMP_CONFIG_HASH_PREFIX = bytes.fromhex(PUMP_CONFIG_SHA256[:16])
# Open-loop demo timing requested by the operator.  These values are derived
# from the commissioned 12 ms firmware step interval plus the observed UART /
# scheduler overhead under 5 Hz heartbeat traffic.  This mode intentionally
# does not use DONE as its transition gate; ARM_TIMED_TASK remains fail-closed
# in firmware if the stepper is still busy.
CONTINUOUS_MOTION_WAIT_S = {1: 15.8, 2: 23.7, 3: 31.6}
CONTINUOUS_PUMP_DURATION_MS = 5_000
CONTINUOUS_PUMP_HARD_TIMEOUT_MS = 7_000
CONTINUOUS_PUMP_HOST_WINDOW_S = 5.20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_answer_runtime(bundle: Path):
    path = bundle / "runtime" / "x5_answer_card_live.py"
    spec = importlib.util.spec_from_file_location("rootscope_answer_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import answer runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_usb_identity(identity: UsbDeviceIdentity) -> dict[str, str]:
    completed = subprocess.run(
        ["udevadm", "info", "--query=property", f"--name={identity.alias}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=3.0,
    )
    properties: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    expected = {
        "ID_VENDOR_ID": identity.vid,
        "ID_MODEL_ID": identity.pid,
        "ID_PATH": identity.id_path or "",
        "ID_USB_INTERFACE_NUM": identity.interface_number or "",
    }
    mismatches = {
        key: {"expected": value, "observed": properties.get(key, "")}
        for key, value in expected.items()
        if value and properties.get(key, "").lower() != value.lower()
    }
    if mismatches:
        raise RuntimeError(f"commissioned USB identity mismatch: {mismatches}")
    return {key: properties.get(key, "") for key in expected}


@dataclass
class SequenceLedger:
    path: Path
    identity_sha256: str
    value: int

    @classmethod
    def load_or_create(
        cls, path: Path, identity_sha256: str
    ) -> "SequenceLedger":
        if path.exists():
            payload = json.loads(path.read_text("utf-8"))
            if payload.get("device_identity_sha256") != identity_sha256:
                raise RuntimeError("sequence ledger USB identity mismatch")
            value = payload.get("last_reserved_sequence")
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError("sequence ledger value is invalid")
            return cls(path, identity_sha256, value)
        ledger = cls(path, identity_sha256, 30000)
        ledger.persist("INITIAL_SAFE_QUERY_BASELINE")
        return ledger

    def persist(self, semantics: str) -> None:
        atomic_json(
            self.path,
            {
                "schema": "rootscope.stm32-sequence-ledger.v1",
                "updated_at_utc": utc_now(),
                "device_identity_sha256": self.identity_sha256,
                "firmware_boot_id_token": None,
                "last_reserved_sequence": self.value,
                "reservation_semantics": semantics,
            },
        )

    def reserve_next(self) -> int:
        candidate = (self.value + 1) & 0xFFFF
        if candidate == 0:
            candidate = 1
        self.value = candidate
        self.persist("Persisted before transport write; never reused.")
        return candidate

    def reserve_resync(self) -> int:
        # V15 accepts a forward modular delta in 1..32767.  An exact
        # half-range jump (32768) is deliberately classified as stale, so the
        # recovery reservation must stay one below that boundary.
        candidate = (self.value + 32767) & 0xFFFF
        if candidate == 0:
            candidate = 1
        self.value = candidate
        self.persist(
            "Read-only QUERY_FIRMWARE boundary-safe resynchronization (+32767)."
        )
        return candidate


def reserve_task_id(
    path: Path,
    *,
    identity_sha256: str,
    arm_sequence: int,
) -> int:
    previous = 0
    if path.exists():
        payload = json.loads(path.read_text("utf-8"))
        if payload.get("device_identity_sha256") != identity_sha256:
            raise RuntimeError("task ledger USB identity mismatch")
        value = payload.get("last_reserved_task_id")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("task ledger value is invalid")
        previous = value
    task_id = previous + 1
    if task_id > 0xFFFFFFFF:
        raise RuntimeError("task ID space exhausted")
    atomic_json(
        path,
        {
            "schema": "rootscope.stm32-task-ledger.v1",
            "updated_at_utc": utc_now(),
            "device_identity_sha256": identity_sha256,
            "last_reserved_task_id": task_id,
            "arm_sequence": arm_sequence,
            "duration_ms": CONTINUOUS_PUMP_DURATION_MS,
            "hard_timeout_ms": CONTINUOUS_PUMP_HARD_TIMEOUT_MS,
            "channel": 1,
            "config_sha256": PUMP_CONFIG_SHA256,
            "reservation_semantics": (
                "Reserved before the one allowed continuous-cycle "
                "ARM_TIMED_TASK write and never reused."
            ),
        },
    )
    return task_id

def verify_safe_locked_state(status_line: str, io_line: str) -> None:
    status = parse_csv_fields(status_line)
    expected = {"Z": "0", "P": "0", "LOCK": "1", "TASK": "0"}
    for key, value in expected.items():
        if status.get(key) != value:
            raise RuntimeError(
                f"unsafe initial STATUS {key}={status.get(key)!r}, expected={value}"
            )
    io = parse_csv_fields(io_line)
    if io.get("Z") != "0x0" or io.get("PB6") != "1" or io.get("PLOG") != "0":
        raise RuntimeError(f"unsafe initial IOSTATUS: {io_line}")


def query_firmware_with_safe_resync(
    session: SerialSession, ledger: SequenceLedger
) -> tuple[dict[str, int | str], int]:
    sequence = ledger.reserve_next()
    try:
        return firmware_query(session, sequence), sequence
    except RuntimeError:
        # QUERY_FIRMWARE has no actuation authority. One half-range retry
        # recovers a durable ledger after prior manual qualification tools used
        # a different sequence base. Physical commands are never retried.
        sequence = ledger.reserve_resync()
        return firmware_query(session, sequence), sequence


def query_readonly_ascii_with_retry(
    session: SerialSession,
    command: str,
    prefix: str,
    *,
    attempts: int = 2,
) -> str:
    """Retry only allow-listed, non-actuating ASCII telemetry queries."""

    if command not in {"VERSION", "STATUS", "IOSTATUS"}:
        raise ValueError(f"ASCII retry is forbidden for command: {command}")
    last_error: RuntimeError | None = None
    for _ in range(max(attempts, 1)):
        try:
            return session.query_ascii(command, prefix)
        except RuntimeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def send_ack_checked(
    session: SerialSession,
    ledger: SequenceLedger,
    command_type: int,
    payload_builder,
) -> int:
    """Send a non-actuating control command with one short ACK-loss retry.

    Only heartbeat and clear-E-stop use this helper.  Physical DEPTH and
    ARM_TIMED_TASK writes deliberately use separate one-write paths and are
    never retried.
    """

    if command_type not in {CMD_HEARTBEAT, CMD_CLEAR_ESTOP}:
        raise ValueError(
            f"ACK retry is forbidden for command type 0x{command_type:02X}"
        )
    last_error: RuntimeError | None = None
    for _ in range(2):
        sequence = ledger.reserve_next()
        session.write_once(
            encode_frame(command_type, payload_builder(sequence))
        )
        try:
            status, reason, _ = session.wait_ack(
                command_type,
                sequence,
                timeout_s=0.35,
            )
        except RuntimeError as exc:
            last_error = exc
            continue
        if (status, reason) != (0, 0):
            raise RuntimeError(
                f"binary command 0x{command_type:02X} rejected: "
                f"status={status}, reason={reason}"
            )
        return sequence
    assert last_error is not None
    raise last_error


def execute_depth(
    *,
    device: str,
    level: int,
    ledger: SequenceLedger,
) -> dict[str, Any]:
    steps = LEVEL_TO_STEPS[level]
    receipt: dict[str, Any] = {
        "schema": "rootscope.visual_irrigation.depth.v1",
        "started_at_utc": utc_now(),
        "level": level,
        "steps": steps,
        "automatic_retry": False,
        "automatic_return": False,
        "commands": [],
        "done_received": False,
        "final_safe_state_verified": False,
    }
    with SerialSession(device) as session:
        version_line = query_readonly_ascii_with_retry(
            session, "VERSION", "VERSION,"
        )
        if (
            EXPECTED_VERSION not in version_line
            or "MOTION=Z3_DOWN_ONLY" not in version_line
            or "RETURN=MANUAL" not in version_line
        ):
            raise RuntimeError(f"unexpected V15 firmware contract: {version_line}")
        initial_status = query_readonly_ascii_with_retry(
            session, "STATUS", "STATUS,"
        )
        initial_io = query_readonly_ascii_with_retry(
            session, "IOSTATUS", "IOSTATUS,"
        )
        verify_safe_locked_state(initial_status, initial_io)
        firmware, firmware_sequence = query_firmware_with_safe_resync(
            session, ledger
        )
        if (
            firmware["build_id"] != EXPECTED_BUILD_ID
            or firmware["capabilities"] != EXPECTED_CAPABILITIES
            or firmware["hardware_variant"] != EXPECTED_VARIANT
        ):
            raise RuntimeError(f"binary firmware identity mismatch: {firmware}")
        receipt.update(
            {
                "firmware_ascii": version_line,
                "firmware_binary": firmware,
                "firmware_query_sequence": firmware_sequence,
                "initial_status": initial_status,
                "initial_io_status": initial_io,
            }
        )
        try:
            heartbeat_sequence = send_ack_checked(
                session,
                ledger,
                CMD_HEARTBEAT,
                lambda seq: struct.pack("<HB", seq, 0),
            )
            receipt["commands"].append(f"HEARTBEAT:{heartbeat_sequence}")
            clear_sequence = send_ack_checked(
                session,
                ledger,
                CMD_CLEAR_ESTOP,
                lambda seq: struct.pack("<H", seq),
            )
            receipt["commands"].append(f"CLEAR_ESTOP:{clear_sequence}")
            zhome = session.query_ascii(
                "ZHOME,CONFIRM", "ACK,ZHOME,CONFIRMED,"
            )
            receipt["commands"].append("ZHOME,CONFIRM")
            receipt["zhome_ack"] = zhome
            heartbeat_sequence = send_ack_checked(
                session,
                ledger,
                CMD_HEARTBEAT,
                lambda seq: struct.pack("<HB", seq, 0),
            )
            receipt["commands"].append(f"HEARTBEAT:{heartbeat_sequence}")

            session.write_once(f"DEPTH,{level}\r\n".encode("ascii"))
            receipt["commands"].append(f"DEPTH,{level}")
            ack_prefix = f"ACK,DEPTH,{level},DOWN,STEPS={steps},".encode("ascii")
            done_prefix = f"DONE,Z,DEPTH={level},STEPS={steps},".encode("ascii")
            data = bytearray()
            next_heartbeat = time.monotonic() + 0.20
            deadline = time.monotonic() + LEVEL_TIMEOUT_S[level]
            depth_ack: str | None = None
            done_line: str | None = None
            while time.monotonic() < deadline:
                data.extend(session.read_for(0.04))
                if depth_ack is None:
                    depth_ack = ascii_line(bytes(data), ack_prefix)
                if done_line is None:
                    done_line = ascii_line(bytes(data), done_prefix)
                if done_line is not None:
                    break
                now = time.monotonic()
                if now >= next_heartbeat:
                    sequence = ledger.reserve_next()
                    session.write_once(
                        encode_frame(
                            CMD_HEARTBEAT,
                            struct.pack("<HB", sequence, 0),
                        )
                    )
                    receipt["commands"].append(f"HEARTBEAT:{sequence}")
                    next_heartbeat += 0.20
            if depth_ack is None:
                raise RuntimeError(f"DEPTH,{level} ACK not received")
            if done_line is None:
                raise RuntimeError(
                    f"DEPTH,{level} DONE not received before hard deadline"
                )
            receipt["depth_ack"] = depth_ack
            receipt["done_line"] = done_line
            receipt["done_received"] = True

            session.write_once(b"STOP\r\n")
            receipt["commands"].append("STOP")
            receipt["stop_ack"] = ascii_line(
                session.read_for(0.40), b"ACK,STOP,"
            )
            final_status = query_readonly_ascii_with_retry(
                session, "STATUS", "STATUS,"
            )
            final_io = query_readonly_ascii_with_retry(
                session, "IOSTATUS", "IOSTATUS,"
            )
            status = parse_csv_fields(final_status)
            io = parse_csv_fields(final_io)
            if (
                status.get("Z") != "0"
                or status.get("P") != "0"
                or status.get("LOCK") != "1"
                or status.get("ZUSED") != "1"
                or status.get("ZLEVEL") != str(level)
                or status.get("ZSTEPS") != str(steps)
                or io.get("Z") != "0x0"
                or io.get("PB6") != "1"
                or io.get("PLOG") != "0"
            ):
                raise RuntimeError(
                    f"depth final safe-state mismatch: {final_status} / {final_io}"
                )
            receipt["final_status"] = final_status
            receipt["final_io_status"] = final_io
            receipt["final_safe_state_verified"] = True
            receipt["status"] = "DEPTH_DONE_STOPPED_LOCKED"
        except BaseException:
            session.safe_stop_best_effort()
            receipt["status"] = "FAILED_STOP_SENT_NO_RETRY"
            raise
        finally:
            receipt["finished_at_utc"] = utc_now()
    return receipt


def execute_continuous_timed_cycle(
    *,
    device: str,
    level: int,
    ledger: SequenceLedger,
    task_state: Path,
    identity_sha256: str,
) -> dict[str, Any]:
    """Run depth then pump in one serial session without a DONE transition gate.

    There is no intermediate STOP, port close or second handshake.  The host
    maintains heartbeat traffic for the calibrated motion window, then writes
    exactly one timed pump task.  Firmware still rejects the pump command if
    the stepper remains busy, so a timing underestimate fails closed.
    """

    steps = LEVEL_TO_STEPS[level]
    motion_wait_s = CONTINUOUS_MOTION_WAIT_S[level]
    receipt: dict[str, Any] = {
        "schema": "rootscope.visual_irrigation.continuous_timed_cycle.v1",
        "started_at_utc": utc_now(),
        "level": level,
        "steps": steps,
        "motion_wait_s": motion_wait_s,
        "motor_done_feedback_used_as_transition": False,
        "intermediate_stop_sent": False,
        "serial_reopened_between_motion_and_pump": False,
        "automatic_retry": False,
        "automatic_return": False,
        "pump_duration_ms": CONTINUOUS_PUMP_DURATION_MS,
        "pump_arm_write_count": 0,
        "commands": [],
        "final_safe_state_verified": False,
    }
    with SerialSession(device) as session:
        version_line = query_readonly_ascii_with_retry(
            session, "VERSION", "VERSION,"
        )
        if (
            EXPECTED_VERSION not in version_line
            or "MOTION=Z3_DOWN_ONLY" not in version_line
            or "RETURN=MANUAL" not in version_line
        ):
            raise RuntimeError(f"unexpected V15 firmware contract: {version_line}")
        initial_status = query_readonly_ascii_with_retry(
            session, "STATUS", "STATUS,"
        )
        initial_io = query_readonly_ascii_with_retry(
            session, "IOSTATUS", "IOSTATUS,"
        )
        verify_safe_locked_state(initial_status, initial_io)
        firmware, firmware_sequence = query_firmware_with_safe_resync(
            session, ledger
        )
        if (
            firmware["build_id"] != EXPECTED_BUILD_ID
            or firmware["capabilities"] != EXPECTED_CAPABILITIES
            or firmware["hardware_variant"] != EXPECTED_VARIANT
        ):
            raise RuntimeError(f"binary firmware identity mismatch: {firmware}")
        receipt.update(
            {
                "firmware_ascii": version_line,
                "firmware_binary": firmware,
                "firmware_query_sequence": firmware_sequence,
                "initial_status": initial_status,
                "initial_io_status": initial_io,
            }
        )
        try:
            heartbeat_sequence = send_ack_checked(
                session,
                ledger,
                CMD_HEARTBEAT,
                lambda seq: struct.pack("<HB", seq, 0),
            )
            receipt["commands"].append(f"HEARTBEAT:{heartbeat_sequence}")
            clear_sequence = send_ack_checked(
                session,
                ledger,
                CMD_CLEAR_ESTOP,
                lambda seq: struct.pack("<H", seq),
            )
            receipt["commands"].append(f"CLEAR_ESTOP:{clear_sequence}")
            zhome = session.query_ascii(
                "ZHOME,CONFIRM", "ACK,ZHOME,CONFIRMED,"
            )
            receipt["commands"].append("ZHOME,CONFIRM")
            receipt["zhome_ack"] = zhome
            heartbeat_sequence = send_ack_checked(
                session,
                ledger,
                CMD_HEARTBEAT,
                lambda seq: struct.pack("<HB", seq, 0),
            )
            receipt["commands"].append(f"HEARTBEAT:{heartbeat_sequence}")

            session.write_once(f"DEPTH,{level}\r\n".encode("ascii"))
            receipt["commands"].append(f"DEPTH,{level}")
            ack_prefix = f"ACK,DEPTH,{level},DOWN,STEPS={steps},".encode("ascii")
            data = bytearray()
            ack_deadline = time.monotonic() + 2.0
            depth_ack: str | None = None
            while time.monotonic() < ack_deadline and depth_ack is None:
                data.extend(session.read_for(0.04))
                depth_ack = ascii_line(bytes(data), ack_prefix)
            if depth_ack is None:
                raise RuntimeError(f"DEPTH,{level} ACK not received")
            receipt["depth_ack"] = depth_ack

            motion_started = time.monotonic()
            motion_deadline = motion_started + motion_wait_s
            next_heartbeat = motion_started
            while time.monotonic() < motion_deadline:
                session.read_for(0.02)  # Drain ACK/DONE/telemetry without gating.
                now = time.monotonic()
                if now >= next_heartbeat:
                    sequence = send_ack_checked(
                        session,
                        ledger,
                        CMD_HEARTBEAT,
                        lambda seq: struct.pack("<HB", seq, 0),
                    )
                    receipt["commands"].append(f"HEARTBEAT:{sequence}")
                    next_heartbeat = time.monotonic() + 0.18
            receipt["motion_wait_finished_at_utc"] = utc_now()

            heartbeat_sequence = send_ack_checked(
                session,
                ledger,
                CMD_HEARTBEAT,
                lambda seq: struct.pack("<HB", seq, 0),
            )
            receipt["commands"].append(f"HEARTBEAT:{heartbeat_sequence}")
            arm_sequence = ledger.reserve_next()
            task_id = reserve_task_id(
                task_state,
                identity_sha256=identity_sha256,
                arm_sequence=arm_sequence,
            )
            arm_payload = struct.pack(
                "<IHBII8s",
                task_id,
                arm_sequence,
                1,
                CONTINUOUS_PUMP_DURATION_MS,
                CONTINUOUS_PUMP_HARD_TIMEOUT_MS,
                PUMP_CONFIG_HASH_PREFIX,
            )
            status, reason, ack_task_id = session.send_binary_with_ack(
                CMD_ARM_TIMED_TASK,
                arm_sequence,
                arm_payload,
            )
            receipt["commands"].append(
                f"ARM_TIMED_TASK:{arm_sequence}:TASK={task_id}"
            )
            receipt["pump_arm_write_count"] = 1
            if (status, reason, ack_task_id) != (0, 0, task_id):
                raise RuntimeError(
                    "continuous pump ARM rejected: "
                    f"status={status}, reason={reason}, task={ack_task_id}"
                )
            receipt["pump_arm_ack"] = {
                "sequence": arm_sequence,
                "task_id": task_id,
                "status": status,
                "reason": reason,
            }

            pump_started = time.monotonic()
            pump_deadline = pump_started + CONTINUOUS_PUMP_HOST_WINDOW_S
            next_heartbeat = pump_started
            next_identity_refresh = pump_started + 2.0
            while time.monotonic() < pump_deadline:
                session.read_for(0.02)
                now = time.monotonic()
                if now >= next_heartbeat:
                    sequence = send_ack_checked(
                        session,
                        ledger,
                        CMD_HEARTBEAT,
                        lambda seq: struct.pack("<HB", seq, 0),
                    )
                    receipt["commands"].append(f"HEARTBEAT:{sequence}")
                    next_heartbeat = time.monotonic() + 0.18
                now = time.monotonic()
                if now >= next_identity_refresh:
                    firmware, refresh_sequence = query_firmware_with_safe_resync(
                        session, ledger
                    )
                    if firmware["build_id"] != EXPECTED_BUILD_ID:
                        raise RuntimeError("identity changed during pump window")
                    receipt["commands"].append(
                        f"QUERY_FIRMWARE:{refresh_sequence}"
                    )
                    next_identity_refresh = time.monotonic() + 2.0

            session.write_once(b"STOP\r\n")
            receipt["commands"].append("STOP")
            receipt["final_stop_ack"] = ascii_line(
                session.read_for(0.40), b"ACK,STOP,"
            )
            final_status = query_readonly_ascii_with_retry(
                session, "STATUS", "STATUS,"
            )
            final_io = query_readonly_ascii_with_retry(
                session, "IOSTATUS", "IOSTATUS,"
            )
            status_fields = parse_csv_fields(final_status)
            io_fields = parse_csv_fields(final_io)
            if (
                status_fields.get("Z") != "0"
                or status_fields.get("P") != "0"
                or status_fields.get("LOCK") != "1"
                or status_fields.get("ZUSED") != "1"
                or status_fields.get("ZLEVEL") != str(level)
                or status_fields.get("ZSTEPS") != str(steps)
                or io_fields.get("Z") != "0x0"
                or io_fields.get("PB6") != "1"
                or io_fields.get("PLOG") != "0"
            ):
                raise RuntimeError(
                    "continuous final safe-state mismatch: "
                    f"{final_status} / {final_io}"
                )
            receipt["final_status"] = final_status
            receipt["final_io_status"] = final_io
            receipt["final_safe_state_verified"] = True
            receipt["status"] = "COMPLETE_CONTINUOUS_TIMED_MOTION_AND_5S_PUMP"
        except BaseException:
            session.safe_stop_best_effort()
            receipt["status"] = "FAILED_STOP_SENT_NO_RETRY"
            raise
        finally:
            receipt["finished_at_utc"] = utc_now()
    return receipt


def final_readback(device: str) -> dict[str, str]:
    with SerialSession(device) as session:
        status_line = query_readonly_ascii_with_retry(
            session, "STATUS", "STATUS,"
        )
        io_line = query_readonly_ascii_with_retry(
            session, "IOSTATUS", "IOSTATUS,"
        )
    status = parse_csv_fields(status_line)
    io = parse_csv_fields(io_line)
    if (
        status.get("Z") != "0"
        or status.get("P") != "0"
        or status.get("LOCK") != "1"
        or io.get("Z") != "0x0"
        or io.get("PB6") != "1"
        or io.get("PLOG") != "0"
    ):
        raise RuntimeError(
            f"cycle final safe-state mismatch: {status_line} / {io_line}"
        )
    return {"status": status_line, "io_status": io_line}


def safe_stop_best_effort(device: str) -> None:
    try:
        with SerialSession(device) as session:
            session.safe_stop_best_effort()
    except BaseException:
        pass


def run_vision_consensus(
    *,
    bundle: Path,
    camera: str,
    width: int,
    height: int,
    fps: int,
    confirmations: int,
    timeout_s: float,
    evidence_dir: Path,
) -> tuple[str, list[dict[str, Any]]]:
    module = load_answer_runtime(bundle)
    runtime = module.AnswerCardRuntime(bundle)
    capture = module.open_camera(camera, width, height, fps)
    history: list[dict[str, Any]] = []
    streak_class: str | None = None
    streak = 0
    started = time.monotonic()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    try:
        for _ in range(20):
            capture.read()
        module.cv2.namedWindow("RootScope Auto Irrigation", module.cv2.WINDOW_NORMAL)
        module.cv2.resizeWindow("RootScope Auto Irrigation", 1024, 576)
        while time.monotonic() - started < timeout_s:
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            result = runtime.infer(frame)
            history.append(result)
            if len(history) > 12:
                history.pop(0)
            decision = str(result["decision"])
            confirmed_target = (
                result["state"] == "CONFIRMED_DUAL_EVIDENCE"
                and decision in TARGET_CLASSES
                and bool(result["cnn"]["pass"])
                and bool(result["template"]["pass"])
            )
            if confirmed_target:
                if decision == streak_class:
                    streak += 1
                else:
                    streak_class = decision
                    streak = 1
            else:
                streak_class = None
                streak = 0

            shown = module.annotate(frame, result, 0.0)
            banner = (
                f"AUTO GATE: {streak}/{confirmations} "
                f"{streak_class or 'HOLD'} | one cycle only"
            )
            module.cv2.putText(
                shown,
                banner,
                (24, min(shown.shape[0] - 24, 230)),
                module.cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 220, 255) if streak else (80, 80, 255),
                2,
                module.cv2.LINE_AA,
            )
            module.cv2.imshow("RootScope Auto Irrigation", shown)
            key = module.cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise RuntimeError("operator aborted before physical action")
            if confirmed_target and streak >= confirmations:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                module.cv2.imwrite(
                    str(evidence_dir / f"{stamp}_{decision}_raw.jpg"), frame
                )
                module.cv2.imwrite(
                    str(evidence_dir / f"{stamp}_{decision}_annotated.jpg"), shown
                )
                module.write_result(
                    evidence_dir / f"{stamp}_{decision}_result.json", result
                )
                return decision, history[-confirmations:]
        raise RuntimeError("vision consensus timeout; no physical action")
    finally:
        capture.release()
        try:
            module.cv2.destroyAllWindows()
        except module.cv2.error:
            pass


def dry_run_image(
    bundle: Path,
    image_path: Path,
    *,
    pump_duration_ms: int,
) -> dict[str, Any]:
    module = load_answer_runtime(bundle)
    runtime = module.AnswerCardRuntime(bundle)
    image = module.cv2.imread(str(image_path), module.cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode dry-run image: {image_path}")
    result = runtime.infer(image)
    decision = str(result["decision"])
    eligible = (
        result["state"] == "CONFIRMED_DUAL_EVIDENCE"
        and decision in TARGET_CLASSES
    )
    return {
        "schema": "rootscope.visual_irrigation.dry_run.v1",
        "decision": decision,
        "vision_state": result["state"],
        "would_lower_probe": eligible,
        "would_depth_level": CLASS_TO_LEVEL[decision] if eligible else 0,
        "would_pump_ms": pump_duration_ms if eligible else 0,
        "devices_opened": False,
        "physical_action_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--camera",
        default="/dev/v4l/by-id/usb-Insta360_Insta360_Link_2C-video-index0",
    )
    parser.add_argument("--serial-device", default="/dev/rootscope_stm32")
    parser.add_argument(
        "--serial-id-path",
        default=os.environ.get(
            "ROOTSCOPE_SERIAL_ID_PATH", "commission-with-udevadm"
        ),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path.home() / ".local/state/rootscope-auto-irrigation",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--confirmations", type=int, default=3)
    parser.add_argument("--vision-timeout-s", type=float, default=180.0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run-image", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--continuous-timed-motion",
        action="store_true",
        help=(
            "Use one serial session, no intermediate STOP, and a calibrated "
            "motion-time transition instead of waiting for DONE."
        ),
    )
    parser.add_argument("--manual-home-observed-at-top", action="store_true")
    parser.add_argument("--confirm-independent-motor-power", action="store_true")
    parser.add_argument("--confirm-water-path-safe", action="store_true")
    parser.add_argument("--confirm-emergency-stop-ready", action="store_true")
    parser.add_argument("--arm-token")
    args = parser.parse_args()
    selected_pump_duration_ms = (
        CONTINUOUS_PUMP_DURATION_MS
        if args.continuous_timed_motion
        else 5_000
    )

    if args.preflight:
        print(
            json.dumps(
                {
                    "schema": "rootscope.visual_irrigation.preflight.v1",
                    "mapping": {
                        key: {
                            "level": level,
                            "steps": LEVEL_TO_STEPS[level],
                            "pump_ms": selected_pump_duration_ms if level else 0,
                        }
                        for key, level in CLASS_TO_LEVEL.items()
                    },
                    "required_consecutive_dual_evidence_frames": args.confirmations,
                    "one_physical_cycle_per_invocation": True,
                    "automatic_return": False,
                    "automatic_retry": False,
                    "continuous_timed_motion_available": True,
                    "selected_pump_duration_ms": selected_pump_duration_ms,
                    "continuous_motion_wait_s": CONTINUOUS_MOTION_WAIT_S,
                    "preflight_opens_camera": False,
                    "preflight_opens_serial": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.dry_run_image is not None:
        print(
            json.dumps(
                dry_run_image(
                    args.bundle.resolve(),
                    args.dry_run_image.resolve(),
                    pump_duration_ms=selected_pump_duration_ms,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not args.execute:
        raise SystemExit(
            "REFUSED: choose --preflight, --dry-run-image, or --execute; "
            "no devices were opened"
        )
    if fcntl is None:
        raise RuntimeError("physical irrigation execution requires Linux/POSIX")
    required_flags = {
        "--manual-home-observed-at-top": args.manual_home_observed_at_top,
        "--confirm-independent-motor-power": args.confirm_independent_motor_power,
        "--confirm-water-path-safe": args.confirm_water_path_safe,
        "--confirm-emergency-stop-ready": args.confirm_emergency_stop_ready,
    }
    missing = [name for name, present in required_flags.items() if not present]
    if missing:
        raise SystemExit(
            f"REFUSED: missing physical safety confirmations: {missing}; "
            "no devices were opened"
        )
    arm_token = args.arm_token
    if arm_token is None and sys.stdin.isatty():
        print("This invocation can lower the probe and run the pump once.")
        print(f"Type exactly: {ARM_TOKEN}")
        arm_token = input("> ").strip()
    if arm_token != ARM_TOKEN:
        raise SystemExit("REFUSED: exact one-cycle arm token missing")

    state_root = args.state_root.resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (state_root / "cycle.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("REFUSED: another irrigation cycle owns the lock") from exc

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = state_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = run_dir / "cycle_receipt.json"
    receipt: dict[str, Any] = {
        "schema": "rootscope.visual_irrigation.cycle.v1",
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "status": "ARMED_WAITING_VISION",
        "one_cycle_only": True,
        "automatic_retry": False,
        "automatic_return": False,
        "pump_duration_ms": selected_pump_duration_ms,
        "physical_action_authority": True,
        "checks": required_flags,
    }
    atomic_json(receipt_path, receipt)

    identity = UsbDeviceIdentity(
        alias=args.serial_device,
        vid="1a86",
        pid="7523",
        id_path=args.serial_id_path,
        interface_number="00",
    )
    physical_stage_started = False
    try:
        usb_identity = verify_usb_identity(identity)
        receipt["usb_identity"] = usb_identity
        sequence_state = state_root / "stm32_v15_sequence.json"
        task_state = state_root / "stm32_v15_task.json"
        ledger = SequenceLedger.load_or_create(
            sequence_state, identity.identity_sha256
        )

        decision, vision_frames = run_vision_consensus(
            bundle=args.bundle.resolve(),
            camera=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            confirmations=args.confirmations,
            timeout_s=args.vision_timeout_s,
            evidence_dir=run_dir / "vision",
        )
        level = CLASS_TO_LEVEL[decision]
        if level == 0:
            raise RuntimeError("non-target HOLD cannot enter physical stage")
        receipt["vision"] = {
            "decision": decision,
            "level": level,
            "steps": LEVEL_TO_STEPS[level],
            "consecutive_confirmations": len(vision_frames),
            "states": [frame["state"] for frame in vision_frames],
            "confidences": [frame["cnn"]["confidence"] for frame in vision_frames],
        }
        receipt["status"] = "VISION_LOCKED_STARTING_DEPTH"
        atomic_json(receipt_path, receipt)

        physical_stage_started = True
        if args.continuous_timed_motion:
            continuous = execute_continuous_timed_cycle(
                device=args.serial_device,
                level=level,
                ledger=ledger,
                task_state=task_state,
                identity_sha256=identity.identity_sha256,
            )
            receipt["continuous_cycle"] = continuous
            receipt["final_readback"] = final_readback(args.serial_device)
            receipt["status"] = (
                "COMPLETE_CONTINUOUS_TIMED_MOTION_AND_5S_PUMP_STOPPED_LOCKED"
            )
            receipt["passed"] = True
            return_code = 0
            return return_code

        depth = execute_depth(
            device=args.serial_device,
            level=level,
            ledger=ledger,
        )
        receipt["depth"] = depth
        receipt["status"] = "DEPTH_DONE_STARTING_PUMP"
        atomic_json(receipt_path, receipt)

        pump_output = run_dir / "pump_receipt.json"
        pump_args = argparse.Namespace(
            sequence_state=sequence_state,
            task_state=task_state,
            output=pump_output,
            device_alias=args.serial_device,
            id_path=args.serial_id_path,
            confirm_physical_pb6_pulse=True,
        )
        old_argv = sys.argv
        try:
            sys.argv = [
                "stm32_z3_pb6_5000ms.py",
                "--sequence-state",
                str(pump_args.sequence_state),
                "--task-state",
                str(pump_args.task_state),
                "--output",
                str(pump_args.output),
                "--device-alias",
                pump_args.device_alias,
                "--id-path",
                pump_args.id_path,
                "--confirm-physical-pb6-pulse",
            ]
            pump_exit = stm32_z3_pb6_5000ms.main()
        finally:
            sys.argv = old_argv
        if pump_exit != 0:
            raise RuntimeError(f"bounded pump task failed with exit={pump_exit}")
        pump_receipt = json.loads(pump_output.read_text("utf-8"))
        if not pump_receipt.get("passed"):
            raise RuntimeError("bounded pump receipt did not pass")
        receipt["pump_receipt"] = str(pump_output)
        receipt["final_readback"] = final_readback(args.serial_device)
        receipt["status"] = "COMPLETE_DEPTH_AND_5S_PUMP_STOPPED_LOCKED"
        receipt["passed"] = True
        return_code = 0
    except BaseException as exc:
        if physical_stage_started:
            safe_stop_best_effort(args.serial_device)
        receipt["status"] = "FAILED_HOLD_STOP_ATTEMPTED_NO_RETRY"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["passed"] = False
        return_code = 2
    finally:
        receipt["finished_at_utc"] = utc_now()
        atomic_json(receipt_path, receipt)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
