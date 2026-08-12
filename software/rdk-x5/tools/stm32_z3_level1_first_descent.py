#!/usr/bin/env python3
"""One-shot, fail-closed first physical qualification of V14 depth level 1.

This tool is intentionally separate from the autonomous RootScope runtime.
It has no retry path and no automatic return path.  It may only be invoked
after an operator has manually raised the probe to the top and explicitly
accepted the physical test conditions.
"""

from __future__ import annotations

import argparse
try:
    import fcntl
except ImportError:  # Windows may import the protocol helpers, but cannot execute serial I/O.
    fcntl = None  # type: ignore[assignment]
import json
import os
import select
import struct
try:
    import termios
except ImportError:  # See fcntl note above.
    termios = None  # type: ignore[assignment]
import time
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_VERSION = "2026-07-25-RS-F103-Z3-PB6-V14"
EXPECTED_BUILD_ID = 2026072514
EXPECTED_CAPABILITIES = 0x00000079
EXPECTED_VARIANT = 2
EXPECTED_LEVEL = 1
EXPECTED_STEPS = 512
DEVICE_DEFAULT = "/dev/rootscope_stm32"
ATTEMPT_DEFAULT = (
    "/opt/rootscope/.local/state/rootscope-v14/"
    "level1-first-descent-attempted.json"
)

CMD_CLEAR_ESTOP = 0x11
CMD_QUERY_FIRMWARE = 0x12
CMD_HEARTBEAT = 0xFF
RSP_FIRMWARE_INFO = 0x81
RSP_ACK = 0x90


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode_frame(message_type: int, payload: bytes) -> bytes:
    body = b"\xAA\x55" + bytes((message_type, len(payload))) + payload
    return body + bytes((sum(body) & 0xFF,))


def scan_frames(raw: bytes) -> list[tuple[int, bytes]]:
    frames: list[tuple[int, bytes]] = []
    index = 0
    while index + 5 <= len(raw):
        if raw[index : index + 2] != b"\xAA\x55":
            index += 1
            continue
        payload_length = raw[index + 3]
        end = index + payload_length + 5
        if end > len(raw):
            index += 1
            continue
        wire = raw[index:end]
        if (sum(wire[:-1]) & 0xFF) == wire[-1]:
            frames.append((wire[2], wire[4:-1]))
            index = end
        else:
            index += 1
    return frames


def ascii_line(raw: bytes, prefix: bytes) -> str | None:
    start = raw.find(prefix)
    if start < 0:
        return None
    end = raw.find(b"\r\n", start)
    if end < 0:
        return None
    return raw[start:end].decode("ascii", errors="replace")


def parse_csv_fields(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in line.split(",")[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


class SerialSession:
    def __init__(self, device: str) -> None:
        self.device = device
        self.fd: int | None = None

    def __enter__(self) -> "SerialSession":
        if fcntl is None or termios is None:
            raise RuntimeError("physical serial execution requires Linux/POSIX")
        self.fd = os.open(
            self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK
        )
        fcntl.ioctl(self.fd, getattr(termios, "TIOCEXCL", 0x540C))
        attrs = termios.tcgetattr(self.fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[3] = 0
        attrs[4] = termios.B115200
        attrs[5] = termios.B115200
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 1
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        termios.tcflush(self.fd, termios.TCIFLUSH)
        time.sleep(0.75)
        self.read_for(0.15)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def write_once(self, payload: bytes) -> None:
        if self.fd is None:
            raise RuntimeError("serial session is not open")
        written = os.write(self.fd, payload)
        if written != len(payload):
            raise RuntimeError(
                f"short serial write: {written}/{len(payload)} bytes"
            )

    def read_for(self, duration_s: float) -> bytes:
        if self.fd is None:
            raise RuntimeError("serial session is not open")
        data = bytearray()
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            readable, _, _ = select.select([self.fd], [], [], 0.05)
            if readable:
                try:
                    data.extend(os.read(self.fd, 4096))
                except BlockingIOError:
                    pass
        return bytes(data)

    def query_ascii(self, command: str, prefix: str, timeout_s: float = 2.0) -> str:
        self.write_once(command.encode("ascii") + b"\r\n")
        data = bytearray()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            data.extend(self.read_for(0.10))
            line = ascii_line(bytes(data), prefix.encode("ascii"))
            if line is not None:
                return line
        raise RuntimeError(f"{prefix} reply not received")

    def wait_ack(
        self, command_type: int, sequence: int, timeout_s: float = 2.0
    ) -> tuple[int, int, int]:
        data = bytearray()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            data.extend(self.read_for(0.10))
            for response_type, payload in scan_frames(bytes(data)):
                if response_type != RSP_ACK or len(payload) != 9:
                    continue
                ack_type, ack_seq, status, reason, task_id = struct.unpack(
                    "<B H B B I", payload
                )
                if ack_type == command_type and ack_seq == sequence:
                    return status, reason, task_id
        raise RuntimeError(
            f"ACK not received: type=0x{command_type:02X}, seq={sequence}"
        )

    def send_binary_with_ack(
        self, command_type: int, sequence: int, payload: bytes
    ) -> tuple[int, int, int]:
        self.write_once(encode_frame(command_type, payload))
        return self.wait_ack(command_type, sequence)

    def safe_stop_best_effort(self) -> None:
        try:
            self.write_once(b"STOP\r\n")
            self.read_for(0.30)
        except Exception:
            pass


def firmware_query(
    session: SerialSession, sequence: int
) -> dict[str, int | str]:
    session.write_once(
        encode_frame(CMD_QUERY_FIRMWARE, struct.pack("<H", sequence))
    )
    data = bytearray()
    deadline = time.monotonic() + 2.0
    firmware_payload: bytes | None = None
    ack: tuple[int, int, int] | None = None
    while time.monotonic() < deadline:
        data.extend(session.read_for(0.10))
        for response_type, payload in scan_frames(bytes(data)):
            if response_type == RSP_FIRMWARE_INFO and len(payload) == 35:
                firmware_payload = payload
            if response_type == RSP_ACK and len(payload) == 9:
                ack_type, ack_seq, status, reason, task_id = struct.unpack(
                    "<B H B B I", payload
                )
                if ack_type == CMD_QUERY_FIRMWARE and ack_seq == sequence:
                    ack = (status, reason, task_id)
        if firmware_payload is not None and ack is not None:
            break
    if firmware_payload is None or ack is None:
        raise RuntimeError("complete firmware identity transaction not received")
    if ack[:2] != (0, 0):
        raise RuntimeError(f"firmware query rejected: status/reason={ack[:2]}")
    protocol, caps, build, variant, tag, boot_id = struct.unpack(
        "<HII B16sQ", firmware_payload
    )
    return {
        "protocol_version": protocol,
        "capabilities": caps,
        "build_id": build,
        "hardware_variant": variant,
        "build_tag_wire": tag.split(b"\0", 1)[0].decode("ascii", "strict"),
        "boot_id": f"{boot_id:016x}",
    }


def verify_initial_state(
    version_line: str, status_line: str, io_line: str
) -> None:
    if EXPECTED_VERSION not in version_line:
        raise RuntimeError(f"unexpected firmware version: {version_line}")
    if "MOTION=Z3_DOWN_ONLY" not in version_line or "RETURN=MANUAL" not in version_line:
        raise RuntimeError(f"unexpected motion contract: {version_line}")
    status = parse_csv_fields(status_line)
    expected_status = {
        "Z": "0",
        "P": "0",
        "LOCK": "1",
        "HB": "0",
        "TASK": "0",
        "ZHOME": "0",
        "ZUSED": "0",
        "ZLEVEL": "0",
        "ZSTEPS": "0",
    }
    for key, expected in expected_status.items():
        if status.get(key) != expected:
            raise RuntimeError(
                f"unsafe initial STATUS {key}={status.get(key)!r}, expected={expected}"
            )
    io = parse_csv_fields(io_line)
    if io.get("Z") != "0x0" or io.get("PB6") != "1" or io.get("PLOG") != "0":
        raise RuntimeError(f"unsafe initial IOSTATUS: {io_line}")


def reserve_attempt(path: Path, initial_receipt: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = json.dumps(
            initial_receipt, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8") + b"\n"
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def replace_receipt(path: Path, receipt: dict[str, object]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE_DEFAULT)
    parser.add_argument("--attempt-record", default=ATTEMPT_DEFAULT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--confirm-physical-level1-descent",
        action="store_true",
        help="required physical-action acknowledgement",
    )
    parser.add_argument(
        "--manual-home-observed-at-top",
        action="store_true",
        help="operator has manually raised and visually checked the probe",
    )
    args = parser.parse_args()

    contract = {
        "schema": "rootscope.stm32_z3_level1_first_descent.v1",
        "expected_version": EXPECTED_VERSION,
        "expected_build_id": EXPECTED_BUILD_ID,
        "expected_capabilities": f"0x{EXPECTED_CAPABILITIES:08X}",
        "expected_variant": EXPECTED_VARIANT,
        "depth_level": EXPECTED_LEVEL,
        "steps": EXPECTED_STEPS,
        "direction": "DOWN_ONLY",
        "automatic_return": False,
        "automatic_retry": False,
        "preflight_opens_serial": False,
    }
    if args.preflight:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    if not (
        args.confirm_physical_level1_descent
        and args.manual_home_observed_at_top
    ):
        raise SystemExit(
            "REFUSED: both physical confirmation flags are required; "
            "serial was not opened"
        )

    attempt_path = Path(args.attempt_record)
    if attempt_path.exists():
        raise SystemExit(
            f"REFUSED: one-shot attempt record already exists: {attempt_path}"
        )

    receipt: dict[str, object] = {
        **contract,
        "started_at_utc": utc_now(),
        "status": "ATTEMPT_RESERVED_NO_MOTION_YET",
        "device": args.device,
        "commands": [],
        "heartbeat_sent_count": 0,
        "heartbeat_ack_count": 0,
        "depth_ack_received": False,
        "done_received": False,
        "final_safe_state_verified": False,
    }

    with SerialSession(args.device) as session:
        version_line = session.query_ascii("VERSION", "VERSION,")
        status_line = session.query_ascii("STATUS", "STATUS,")
        io_line = session.query_ascii("IOSTATUS", "IOSTATUS,")
        verify_initial_state(version_line, status_line, io_line)
        firmware = firmware_query(session, 100)
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
                "initial_status": status_line,
                "initial_io_status": io_line,
            }
        )
        reserve_attempt(attempt_path, receipt)

        sequence = 101
        try:
            status, reason, _ = session.send_binary_with_ack(
                CMD_HEARTBEAT,
                sequence,
                struct.pack("<HB", sequence, 0),
            )
            receipt["commands"].append(f"HEARTBEAT:{sequence}")  # type: ignore[index]
            if (status, reason) != (0, 0):
                raise RuntimeError(
                    f"initial heartbeat rejected: status={status}, reason={reason}"
                )
            receipt["heartbeat_sent_count"] = 1
            receipt["heartbeat_ack_count"] = 1

            sequence += 1
            status, reason, _ = session.send_binary_with_ack(
                CMD_CLEAR_ESTOP, sequence, struct.pack("<H", sequence)
            )
            receipt["commands"].append(f"CLEAR_ESTOP:{sequence}")  # type: ignore[index]
            if (status, reason) != (0, 0):
                raise RuntimeError(
                    f"clear rejected: status={status}, reason={reason}"
                )

            zhome_line = session.query_ascii(
                "ZHOME,CONFIRM", "ACK,ZHOME,CONFIRMED,"
            )
            receipt["commands"].append("ZHOME,CONFIRM")  # type: ignore[index]
            receipt["zhome_ack"] = zhome_line

            sequence += 1
            session.write_once(
                encode_frame(
                    CMD_HEARTBEAT, struct.pack("<HB", sequence, 0)
                )
            )
            status, reason, _ = session.wait_ack(CMD_HEARTBEAT, sequence)
            receipt["commands"].append(f"HEARTBEAT:{sequence}")  # type: ignore[index]
            if (status, reason) != (0, 0):
                raise RuntimeError(
                    f"pre-motion heartbeat rejected: status={status}, reason={reason}"
                )
            receipt["heartbeat_sent_count"] = 2
            receipt["heartbeat_ack_count"] = 2

            session.write_once(b"DEPTH,1\r\n")
            receipt["commands"].append("DEPTH,1")  # type: ignore[index]
            data = bytearray()
            next_heartbeat = time.monotonic() + 0.20
            deadline = time.monotonic() + 15.0
            depth_ack: str | None = None
            done_line: str | None = None

            while time.monotonic() < deadline:
                data.extend(session.read_for(0.04))
                if depth_ack is None:
                    depth_ack = ascii_line(
                        bytes(data), b"ACK,DEPTH,1,DOWN,STEPS=512,"
                    )
                    if depth_ack is not None:
                        receipt["depth_ack_received"] = True
                        receipt["depth_ack"] = depth_ack
                if done_line is None:
                    done_line = ascii_line(
                        bytes(data), b"DONE,Z,DEPTH=1,STEPS=512,"
                    )
                if done_line is not None:
                    receipt["done_received"] = True
                    receipt["done_line"] = done_line
                    break
                now = time.monotonic()
                if now >= next_heartbeat:
                    sequence += 1
                    session.write_once(
                        encode_frame(
                            CMD_HEARTBEAT,
                            struct.pack("<HB", sequence, 0),
                        )
                    )
                    receipt["commands"].append(  # type: ignore[index]
                        f"HEARTBEAT:{sequence}"
                    )
                    receipt["heartbeat_sent_count"] = int(
                        receipt["heartbeat_sent_count"]
                    ) + 1
                    next_heartbeat += 0.20

            if depth_ack is None:
                raise RuntimeError("DEPTH,1 acknowledgement not received")
            if done_line is None:
                raise RuntimeError("level-1 completion not received before timeout")

            session.write_once(b"STOP\r\n")
            receipt["commands"].append("STOP")  # type: ignore[index]
            stop_data = session.read_for(0.40)
            receipt["stop_ack"] = ascii_line(stop_data, b"ACK,STOP,")

            final_status = session.query_ascii("STATUS", "STATUS,")
            final_io = session.query_ascii("IOSTATUS", "IOSTATUS,")
            receipt["commands"].extend(["STATUS", "IOSTATUS"])  # type: ignore[index]
            final_fields = parse_csv_fields(final_status)
            final_io_fields = parse_csv_fields(final_io)
            if (
                final_fields.get("Z") != "0"
                or final_fields.get("P") != "0"
                or final_fields.get("LOCK") != "1"
                or final_fields.get("ZHOME") != "0"
                or final_fields.get("ZUSED") != "1"
                or final_fields.get("ZLEVEL") != "1"
                or final_fields.get("ZSTEPS") != "512"
                or final_io_fields.get("Z") != "0x0"
                or final_io_fields.get("PB6") != "1"
                or final_io_fields.get("PLOG") != "0"
            ):
                raise RuntimeError(
                    f"final fail-closed state mismatch: "
                    f"{final_status} / {final_io}"
                )
            receipt["final_status"] = final_status
            receipt["final_io_status"] = final_io
            receipt["final_safe_state_verified"] = True
            receipt["status"] = (
                "PROTOCOL_LEVEL1_COMPLETE_PHYSICAL_OBSERVATION_PENDING"
            )
        except Exception as exc:
            session.safe_stop_best_effort()
            receipt["status"] = "FAILED_STOP_SENT_NO_RETRY"
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            receipt["finished_at_utc"] = utc_now()
            replace_receipt(attempt_path, receipt)

    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
