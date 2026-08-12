#!/usr/bin/env python3
"""One-shot physical qualification for V15 depth levels 2 and 3."""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

from stm32_z3_level1_first_descent import (
    CMD_CLEAR_ESTOP,
    CMD_HEARTBEAT,
    DEVICE_DEFAULT,
    SerialSession,
    ascii_line,
    encode_frame,
    firmware_query,
    parse_csv_fields,
    replace_receipt,
    reserve_attempt,
    utc_now,
)


EXPECTED_VERSION = "2026-07-25-RS-F103-Z3-PB6-V15"
EXPECTED_BUILD_ID = 2026072515
EXPECTED_CAPABILITIES = 0x00000079
EXPECTED_VARIANT = 2

PROFILES = {
    2: {
        "steps": 1536,
        "label": "medium",
        "confirm_flag": "confirm_v15_depth2_1536",
        "attempt": (
            "/opt/rootscope/.local/state/rootscope-v15/"
            "depth2-1536-attempted.json"
        ),
        "sequence_base": 2000,
        "timeout_s": 35.0,
    },
    3: {
        "steps": 2048,
        "label": "deep",
        "confirm_flag": "confirm_v15_depth3_2048",
        "attempt": (
            "/opt/rootscope/.local/state/rootscope-v15/"
            "depth3-2048-attempted.json"
        ),
        "sequence_base": 10000,
        "timeout_s": 45.0,
    },
}


def verify_safe_locked_state(status_line: str, io_line: str) -> None:
    status = parse_csv_fields(status_line)
    for key, expected in {
        "Z": "0",
        "P": "0",
        "LOCK": "1",
        "TASK": "0",
    }.items():
        if status.get(key) != expected:
            raise RuntimeError(
                f"unsafe initial STATUS {key}={status.get(key)!r}, expected={expected}"
            )
    io = parse_csv_fields(io_line)
    if io.get("Z") != "0x0" or io.get("PB6") != "1" or io.get("PLOG") != "0":
        raise RuntimeError(f"unsafe initial IOSTATUS: {io_line}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=sorted(PROFILES))
    parser.add_argument("--device", default=DEVICE_DEFAULT)
    parser.add_argument("--attempt-record")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-v15-depth2-1536", action="store_true")
    parser.add_argument("--confirm-v15-depth3-2048", action="store_true")
    parser.add_argument("--manual-home-observed-at-top", action="store_true")
    args = parser.parse_args()

    if args.preflight:
        print(
            json.dumps(
                {
                    "schema": "rootscope.stm32_z3_v15_depth_qualifier.v1",
                    "expected_version": EXPECTED_VERSION,
                    "expected_build_id": EXPECTED_BUILD_ID,
                    "profiles": {
                        str(level): {
                            "steps": spec["steps"],
                            "label": spec["label"],
                            "automatic_retry": False,
                            "automatic_return": False,
                        }
                        for level, spec in PROFILES.items()
                    },
                    "preflight_opens_serial": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.level not in PROFILES:
        raise SystemExit("REFUSED: --level 2 or --level 3 is required")
    profile = PROFILES[args.level]
    exact_confirmation = bool(getattr(args, str(profile["confirm_flag"])))
    if not exact_confirmation or not args.manual_home_observed_at_top:
        raise SystemExit(
            "REFUSED: exact depth confirmation and manual-home confirmation "
            "are both required; serial was not opened"
        )

    attempt_path = Path(args.attempt_record or str(profile["attempt"]))
    if attempt_path.exists():
        raise SystemExit(
            f"REFUSED: one-shot attempt record already exists: {attempt_path}"
        )

    steps = int(profile["steps"])
    level = int(args.level)
    sequence = int(profile["sequence_base"])
    receipt: dict[str, object] = {
        "schema": "rootscope.stm32_z3_v15_depth_qualifier.v1",
        "started_at_utc": utc_now(),
        "status": "ATTEMPT_RESERVED_NO_MOTION_YET",
        "device": args.device,
        "firmware_depth_level": level,
        "steps": steps,
        "label": str(profile["label"]),
        "direction": "DOWN_ONLY",
        "automatic_retry": False,
        "automatic_return": False,
        "commands": [],
        "heartbeat_sent_count": 0,
        "heartbeat_ack_count": 0,
        "depth_ack_received": False,
        "done_received": False,
        "final_safe_state_verified": False,
    }

    with SerialSession(args.device) as session:
        version_line = session.query_ascii("VERSION", "VERSION,")
        if (
            EXPECTED_VERSION not in version_line
            or "MOTION=Z3_DOWN_ONLY" not in version_line
            or "RETURN=MANUAL" not in version_line
        ):
            raise RuntimeError(f"unexpected firmware contract: {version_line}")
        initial_status = session.query_ascii("STATUS", "STATUS,")
        initial_io = session.query_ascii("IOSTATUS", "IOSTATUS,")
        verify_safe_locked_state(initial_status, initial_io)

        firmware = firmware_query(session, sequence)
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
                "initial_status": initial_status,
                "initial_io_status": initial_io,
            }
        )
        reserve_attempt(attempt_path, receipt)

        try:
            sequence += 1
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
                CMD_CLEAR_ESTOP,
                sequence,
                struct.pack("<H", sequence),
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
            status, reason, _ = session.send_binary_with_ack(
                CMD_HEARTBEAT,
                sequence,
                struct.pack("<HB", sequence, 0),
            )
            receipt["commands"].append(f"HEARTBEAT:{sequence}")  # type: ignore[index]
            if (status, reason) != (0, 0):
                raise RuntimeError(
                    f"pre-motion heartbeat rejected: status={status}, reason={reason}"
                )
            receipt["heartbeat_sent_count"] = 2
            receipt["heartbeat_ack_count"] = 2

            session.write_once(f"DEPTH,{level}\r\n".encode("ascii"))
            receipt["commands"].append(f"DEPTH,{level}")  # type: ignore[index]
            ack_prefix = (
                f"ACK,DEPTH,{level},DOWN,STEPS={steps},".encode("ascii")
            )
            done_prefix = (
                f"DONE,Z,DEPTH={level},STEPS={steps},".encode("ascii")
            )
            data = bytearray()
            next_heartbeat = time.monotonic() + 0.20
            deadline = time.monotonic() + float(profile["timeout_s"])
            depth_ack: str | None = None
            done_line: str | None = None

            while time.monotonic() < deadline:
                data.extend(session.read_for(0.04))
                if depth_ack is None:
                    depth_ack = ascii_line(bytes(data), ack_prefix)
                    if depth_ack is not None:
                        receipt["depth_ack_received"] = True
                        receipt["depth_ack"] = depth_ack
                if done_line is None:
                    done_line = ascii_line(bytes(data), done_prefix)
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
                raise RuntimeError(
                    f"DEPTH,{level} acknowledgement not received"
                )
            if done_line is None:
                raise RuntimeError(
                    f"{steps}-step completion not received before timeout"
                )

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
                or final_fields.get("ZLEVEL") != str(level)
                or final_fields.get("ZSTEPS") != str(steps)
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
                f"PROTOCOL_V15_DEPTH{level}_{steps}_COMPLETE_"
                "PHYSICAL_OBSERVATION_PENDING"
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
