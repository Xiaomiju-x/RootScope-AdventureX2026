#!/usr/bin/env python3
"""Qualify the real RootScope STM32 link without ever arming the pump.

The only emitted command types are EMERGENCY_STOP, QUERY_FIRMWARE,
HEARTBEAT and CLEAR_ESTOP.  ARM_TASK, ARM_TIMED_TASK and ABORT_TASK are
forbidden by an explicit post-run command audit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

from app.hardware.device_identity import UsbDeviceIdentity
from app.hardware.physical_serial import (
    PhysicalSerialOpenRequest,
    PosixExplicitSerialOpener,
)
from app.hardware.serial_writer import (
    SerialWriterScheduler,
    StopConfirmation,
    WriteStatus,
)
from app.serial.frame import (
    AckReason,
    AckStatus,
    CommandType,
    LockReason,
    SafetyBits,
    SafetyState,
)
from app.serial.link import (
    F103_PB6_IDENTITY_EXPECTATION,
    RootScopeSerialLink,
)


SAFE_COMMANDS = frozenset(
    {
        CommandType.EMERGENCY_STOP,
        CommandType.QUERY_FIRMWARE,
        CommandType.HEARTBEAT,
        CommandType.CLEAR_ESTOP,
    }
)
FULL_WRITE = frozenset(
    {
        WriteStatus.FULLY_WRITTEN,
        WriteStatus.FULLY_WRITTEN_AFTER_SHORT_WRITE,
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def atomic_json(path: Path, payload: dict[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_sequence(path: Path, identity_sha256: str, missing: int | None) -> int:
    if not path.exists():
        if missing is None:
            raise RuntimeError(
                "sequence ledger is absent; supply --initial-sequence-if-missing"
            )
        return missing
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "rootscope.stm32-sequence-ledger.v1":
        raise RuntimeError("sequence ledger schema mismatch")
    if value.get("device_identity_sha256") != identity_sha256:
        raise RuntimeError("sequence ledger device identity mismatch")
    sequence = value.get("last_reserved_sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise RuntimeError("sequence ledger value is not an integer")
    if not 0 <= sequence <= 0xFFFF:
        raise RuntimeError("sequence ledger value is outside uint16")
    return sequence


def checkpoint_sequence(
    path: Path,
    *,
    identity_sha256: str,
    sequence: int,
    boot_id_token: str | None,
) -> None:
    atomic_json(
        path,
        {
            "schema": "rootscope.stm32-sequence-ledger.v1",
            "updated_at_utc": utc_now(),
            "device_identity_sha256": identity_sha256,
            "firmware_boot_id_token": boot_id_token,
            "last_reserved_sequence": sequence,
            "reservation_semantics": (
                "Persisted before transport write; unused reserved values may be "
                "skipped but must never be reused."
            ),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        type=Path,
        default=Path.home() / ".local/state/rootscope/stm32_sequence.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home()
        / ".local/state/rootscope/evidence/stm32_no_pump_dry_run.json",
    )
    parser.add_argument("--initial-sequence-if-missing", type=int)
    args = parser.parse_args()

    identity = UsbDeviceIdentity(
        alias="/dev/rootscope_stm32",
        vid="1a86",
        pid="7523",
        id_path=os.environ.get(
            "ROOTSCOPE_SERIAL_ID_PATH", "commission-with-udevadm"
        ),
        interface_number="00",
    )
    initial_sequence = load_sequence(
        args.state, identity.identity_sha256, args.initial_sequence_if_missing
    )
    link = RootScopeSerialLink(
        expectation=F103_PB6_IDENTITY_EXPECTATION,
        initial_sequence=initial_sequence,
    )
    transport = PosixExplicitSerialOpener().open_explicit(
        PhysicalSerialOpenRequest(
            identity=identity,
            physical_authority_granted=True,
        )
    )
    writer = SerialWriterScheduler(transport)
    writer.claim_current_thread()

    started_at_utc = utc_now()
    event_kinds: list[str] = []
    observed_pump_masks: list[int] = []
    heartbeat_acks = 0
    clear_ack_ok = False
    unlocked_observed = False
    watchdog_relocked = False
    initial_stop_confirmed = False
    provisional_stop_safety: SafetyState | None = None
    failure: str | None = None

    def snapshot() -> None:
        if link.last_telemetry is not None:
            observed_pump_masks.append(int(link.last_telemetry.pump_mask))

    def read_until(
        timeout_s: float,
        predicate: Callable[[], bool] | None = None,
    ) -> bool:
        nonlocal provisional_stop_safety
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            data = transport.read(4096)
            if data:
                events = link.ingest(data)
                event_kinds.extend(event.kind for event in events)
                for event in events:
                    if event.kind == "safety_boot_mismatch":
                        # Before QUERY_FIRMWARE, the parser intentionally
                        # refuses to bind a safety frame to an unknown boot.
                        # Its exact payload may only confirm the initial
                        # stop/locked condition; it is never admitted as an
                        # action or completion authority.
                        provisional_stop_safety = SafetyState.from_payload(
                            event.value.payload
                        )
                snapshot()
            if predicate is not None and predicate():
                return True
        return bool(predicate is None)

    def send(
        intent_id: str,
        command_type: CommandType,
        builder: Callable[[], bytes],
    ):
        command = writer.schedule(
            intent_id=intent_id,
            build_frame=builder,
            expected_command_type=command_type,
            on_fully_written=lambda raw, stamp: link.mark_command_sent(
                raw,
                execution_backend=transport.backend_id,
                now=stamp,
            ),
        )
        if command.command_type not in SAFE_COMMANDS:
            raise RuntimeError(f"forbidden command constructed: {command.command_type}")
        checkpoint_sequence(
            args.state,
            identity_sha256=identity.identity_sha256,
            sequence=command.frame_seq,
            boot_id_token=link.firmware_boot_id_token,
        )
        receipt = writer.write_next()
        if receipt is None or receipt.status not in FULL_WRITE:
            raise RuntimeError(f"serial write failed: {receipt}")
        return command, receipt

    try:
        stop_command, _ = send(
            "dry-run-initial-estop",
            CommandType.EMERGENCY_STOP,
            link.make_emergency_stop,
        )
        read_until(
            1.0,
            lambda: (
                link.last_ack is not None
                and link.last_ack.ack_for_type == int(CommandType.EMERGENCY_STOP)
                and link.last_ack.seq == stop_command.frame_seq
                and provisional_stop_safety is not None
                and bool(
                    provisional_stop_safety.safety_bits
                    & SafetyBits.LOCK_LATCHED
                )
                and link.last_telemetry is not None
                and link.last_telemetry.pump_mask == 0
            ),
        )
        ack = link.last_ack
        initial_stop_confirmed = bool(
            ack is not None
            and ack.ack_for_type == int(CommandType.EMERGENCY_STOP)
            and ack.seq == stop_command.frame_seq
            and ack.status == AckStatus.OK
            and ack.reason == AckReason.EMERGENCY_STOP
            and provisional_stop_safety is not None
            and bool(
                provisional_stop_safety.safety_bits & SafetyBits.LOCK_LATCHED
            )
            and link.last_telemetry is not None
            and link.last_telemetry.pump_mask == 0
        )
        writer.confirm_stop(
            StopConfirmation(
                command_sha256=stop_command.raw_frame_sha256,
                ack_matches_command=initial_stop_confirmed,
                firmware_locked=initial_stop_confirmed,
                pumps_all_off=initial_stop_confirmed,
                evidence_fresh=initial_stop_confirmed,
            )
        )

        query, _ = send(
            "dry-run-firmware-query",
            CommandType.QUERY_FIRMWARE,
            link.make_firmware_query,
        )
        if not read_until(
            1.0,
            lambda: (
                link.identity_valid
                and link.last_ack is not None
                and link.last_ack.ack_for_type == int(CommandType.QUERY_FIRMWARE)
                and link.last_ack.seq == query.frame_seq
                and link.last_ack.status == AckStatus.OK
            ),
        ):
            raise RuntimeError("firmware identity/ACK timeout")
        checkpoint_sequence(
            args.state,
            identity_sha256=identity.identity_sha256,
            sequence=link.sequence_checkpoint,
            boot_id_token=link.firmware_boot_id_token,
        )

        heartbeat, _ = send(
            "dry-run-heartbeat-before-clear",
            CommandType.HEARTBEAT,
            link.make_heartbeat,
        )
        if not read_until(
            0.5,
            lambda: (
                link.last_ack is not None
                and link.last_ack.ack_for_type == int(CommandType.HEARTBEAT)
                and link.last_ack.seq == heartbeat.frame_seq
                and link.last_ack.status == AckStatus.OK
            ),
        ):
            raise RuntimeError("pre-clear heartbeat ACK timeout")
        heartbeat_acks += 1

        clear, _ = send(
            "dry-run-clear-estop",
            CommandType.CLEAR_ESTOP,
            link.make_clear_estop,
        )
        if not read_until(
            0.6,
            lambda: (
                link.last_ack is not None
                and link.last_ack.ack_for_type == int(CommandType.CLEAR_ESTOP)
                and link.last_ack.seq == clear.frame_seq
                and link.last_ack.status == AckStatus.OK
                and link.last_ack.reason == AckReason.NONE
                and link.last_safety is not None
                and not bool(
                    link.last_safety.safety_bits & SafetyBits.LOCK_LATCHED
                )
                and link.last_telemetry is not None
                and link.last_telemetry.pump_mask == 0
            ),
        ):
            raise RuntimeError("clear/unlocked state timeout")
        clear_ack_ok = True
        unlocked_observed = True

        for index in range(5):
            heartbeat, _ = send(
                f"dry-run-heartbeat-{index + 1}",
                CommandType.HEARTBEAT,
                link.make_heartbeat,
            )
            if not read_until(
                0.35,
                lambda: (
                    link.last_ack is not None
                    and link.last_ack.ack_for_type == int(CommandType.HEARTBEAT)
                    and link.last_ack.seq == heartbeat.frame_seq
                    and link.last_ack.status == AckStatus.OK
                ),
            ):
                raise RuntimeError(f"heartbeat {index + 1} ACK timeout")
            heartbeat_acks += 1
            if link.last_telemetry is not None and link.last_telemetry.pump_mask != 0:
                raise RuntimeError("pump became active during forbidden no-pump run")
            sleep_remaining = 0.2 - (
                time.monotonic() - heartbeat.queued_at_monotonic
            )
            if sleep_remaining > 0:
                time.sleep(sleep_remaining)

        watchdog_relocked = read_until(
            1.8,
            lambda: (
                link.last_safety is not None
                and bool(link.last_safety.safety_bits & SafetyBits.LOCK_LATCHED)
                and link.last_safety.lock_reason == LockReason.WATCHDOG_TIMEOUT
                and link.last_telemetry is not None
                and link.last_telemetry.pump_mask == 0
            ),
        )
        if not watchdog_relocked:
            raise RuntimeError("watchdog did not relock within qualification window")
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        try:
            send(
                "dry-run-exception-estop",
                CommandType.EMERGENCY_STOP,
                link.make_emergency_stop,
            )
            read_until(0.6)
        except BaseException as stop_exc:
            failure += f"; fail-safe estop error={type(stop_exc).__name__}: {stop_exc}"
    finally:
        snapshot()
        command_names = [
            receipt.command.command_type.name for receipt in writer.receipts
        ]
        forbidden_commands = sorted(
            {
                name
                for name in command_names
                if CommandType[name] not in SAFE_COMMANDS
            }
        )
        pumps_always_off = bool(observed_pump_masks) and all(
            mask == 0 for mask in observed_pump_masks
        )
        passed = bool(
            failure is None
            and initial_stop_confirmed
            and link.identity_valid
            and clear_ack_ok
            and unlocked_observed
            and heartbeat_acks >= 6
            and watchdog_relocked
            and pumps_always_off
            and not forbidden_commands
        )
        firmware = link.firmware_info
        payload = {
            "schema": "rootscope.stm32-no-pump-dry-run.v1",
            "started_at_utc": started_at_utc,
            "finished_at_utc": utc_now(),
            "status": "PASS" if passed else "FAIL",
            "failure": failure,
            "device_identity": dict(identity.to_dict()),
            "device_identity_sha256": identity.identity_sha256,
            "firmware": (
                None
                if firmware is None
                else {
                    "protocol_version": firmware.protocol_version,
                    "capabilities": firmware.capabilities,
                    "build_id": firmware.build_id,
                    "hw_variant": firmware.hw_variant,
                    "build_tag": firmware.build_tag,
                    "boot_id_token": link.firmware_boot_id_token,
                }
            ),
            "checks": {
                "initial_estop_confirmed": initial_stop_confirmed,
                "identity_valid": link.identity_valid,
                "clear_ack_ok": clear_ack_ok,
                "unlocked_observed": unlocked_observed,
                "heartbeat_acks": heartbeat_acks,
                "watchdog_relocked": watchdog_relocked,
                "pumps_always_off": pumps_always_off,
                "observed_pump_masks": observed_pump_masks,
                "forbidden_commands": forbidden_commands,
            },
            "event_kinds": event_kinds,
            "writes": [
                {
                    "intent_id": receipt.command.intent_id,
                    "command_type": receipt.command.command_type.name,
                    "sequence": receipt.command.frame_seq,
                    "raw_frame_sha256": receipt.command.raw_frame_sha256,
                    "status": receipt.status.value,
                    "bytes_confirmed_written": receipt.bytes_confirmed_written,
                }
                for receipt in writer.receipts
            ],
            "final": {
                "sequence_checkpoint": link.sequence_checkpoint,
                "lock_reason": (
                    None
                    if link.last_safety is None
                    else int(link.last_safety.lock_reason)
                ),
                "lock_latched": (
                    None
                    if link.last_safety is None
                    else bool(
                        link.last_safety.safety_bits & SafetyBits.LOCK_LATCHED
                    )
                ),
                "pump_mask": (
                    None
                    if link.last_telemetry is None
                    else int(link.last_telemetry.pump_mask)
                ),
            },
            "authority": {
                "arm_task_sent": False,
                "arm_timed_task_sent": False,
                "abort_task_sent": False,
                "pump_command_sent": False,
                "physical_completion": False,
            },
            "claim_boundary": (
                "Proves actual X5/F103 heartbeat, explicit unlock, sequence/ACK "
                "and watchdog relock while every observed pump mask remained zero. "
                "It does not prove PB6 actuation or irrigation completion."
            ),
        }
        atomic_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        try:
            writer.close()
        except BaseException:
            transport.close()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
