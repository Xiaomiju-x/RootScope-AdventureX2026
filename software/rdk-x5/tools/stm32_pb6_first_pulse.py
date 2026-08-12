#!/usr/bin/env python3
"""Execute exactly one commissioned 100 ms PB6 pump pulse on RootScope F103.

This is a one-shot qualification tool, not the autonomous irrigation service.
It requires the already-commissioned USB identity and durable sequence ledger,
uses an initial and final E-stop, and never retries ARM_TIMED_TASK.
"""

from __future__ import annotations

import argparse
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
    SafetyBits,
    SafetyState,
    TerminalReason,
)
from app.serial.link import (
    F103_PB6_IDENTITY_EXPECTATION,
    RootScopeSerialLink,
)


DURATION_MS = 100
HARD_TIMEOUT_MS = 500
CHANNEL = 1
DEVICE_ALIAS = "/dev/rootscope_stm32"
DEVICE_ID_PATH = os.environ.get(
    "ROOTSCOPE_SERIAL_ID_PATH", "commission-with-udevadm"
)
IDENTITY_EXPECTATION = F103_PB6_IDENTITY_EXPECTATION
CONFIG_SHA256 = "54f66f0dec8c043623245511972b3d0568f43c77b25323bddc1f1a628ef9426e"
CONFIG_HASH_PREFIX = bytes.fromhex(CONFIG_SHA256[:16])
ALLOWED_COMMANDS = frozenset(
    {
        CommandType.EMERGENCY_STOP,
        CommandType.QUERY_FIRMWARE,
        CommandType.HEARTBEAT,
        CommandType.CLEAR_ESTOP,
        CommandType.ARM_TIMED_TASK,
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


def load_ledger(path: Path, identity_sha256: str, field: str) -> int:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("device_identity_sha256") != identity_sha256:
        raise RuntimeError(f"{path.name} device identity mismatch")
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise RuntimeError(f"{path.name} has invalid {field}")
    return result


def checkpoint_sequence(
    path: Path,
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


def checkpoint_task(
    path: Path,
    identity_sha256: str,
    task_id: int,
    sequence: int,
) -> None:
    atomic_json(
        path,
        {
            "schema": "rootscope.stm32-task-ledger.v1",
            "updated_at_utc": utc_now(),
            "device_identity_sha256": identity_sha256,
            "last_reserved_task_id": task_id,
            "arm_sequence": sequence,
            "duration_ms": DURATION_MS,
            "hard_timeout_ms": HARD_TIMEOUT_MS,
            "channel": CHANNEL,
            "config_sha256": CONFIG_SHA256,
            "reservation_semantics": (
                "Reserved before the one allowed ARM_TIMED_TASK write and never reused."
            ),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-state",
        type=Path,
        default=Path.home() / ".local/state/rootscope/stm32_sequence.json",
    )
    parser.add_argument(
        "--task-state",
        type=Path,
        default=Path.home() / ".local/state/rootscope/stm32_task.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home()
        / ".local/state/rootscope/evidence/stm32_pb6_first_pulse.json",
    )
    parser.add_argument(
        "--confirm-physical-pb6-pulse",
        action="store_true",
        help=(
            "Required physical-action gate: confirms an operator is present, "
            "the water path is safe, and pump power can be disconnected."
        ),
    )
    args = parser.parse_args()
    if not args.confirm_physical_pb6_pulse:
        parser.error(
            "--confirm-physical-pb6-pulse is required; no serial port was opened"
        )

    identity = UsbDeviceIdentity(
        alias=DEVICE_ALIAS,
        vid="1a86",
        pid="7523",
        id_path=DEVICE_ID_PATH,
        interface_number="00",
    )
    initial_sequence = load_ledger(
        args.sequence_state,
        identity.identity_sha256,
        "last_reserved_sequence",
    )
    previous_task_id = (
        load_ledger(
            args.task_state,
            identity.identity_sha256,
            "last_reserved_task_id",
        )
        if args.task_state.exists()
        else 0
    )
    task_id = previous_task_id + 1
    if task_id > 0xFFFFFFFF:
        raise RuntimeError("task ID space exhausted")

    link = RootScopeSerialLink(
        expectation=IDENTITY_EXPECTATION,
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
    pump_masks: list[int] = []
    provisional_safety: SafetyState | None = None
    initial_stop_confirmed = False
    final_stop_confirmed = False
    identity_valid = False
    clear_ok = False
    arm_ack_ok = False
    task_result_ok = False
    arm_write_count = 0
    failure: str | None = None

    def snapshot() -> None:
        if link.last_telemetry is not None:
            pump_masks.append(int(link.last_telemetry.pump_mask))

    def read_until(timeout_s: float, predicate: Callable[[], bool]) -> bool:
        nonlocal provisional_safety
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            data = transport.read(4096)
            if data:
                events = link.ingest(data)
                event_kinds.extend(event.kind for event in events)
                for event in events:
                    if event.kind == "safety_boot_mismatch":
                        provisional_safety = SafetyState.from_payload(
                            event.value.payload
                        )
                snapshot()
            if predicate():
                return True
        return False

    def send(
        intent_id: str,
        command_type: CommandType,
        builder: Callable[[], bytes],
    ):
        nonlocal arm_write_count
        command = writer.schedule(
            intent_id=intent_id,
            build_frame=builder,
            expected_command_type=command_type,
            task_id=str(task_id) if command_type is CommandType.ARM_TIMED_TASK else None,
            on_fully_written=lambda raw, stamp: link.mark_command_sent(
                raw,
                execution_backend=transport.backend_id,
                now=stamp,
            ),
        )
        if command.command_type not in ALLOWED_COMMANDS:
            raise RuntimeError(f"forbidden command constructed: {command.command_type}")
        checkpoint_sequence(
            args.sequence_state,
            identity.identity_sha256,
            command.frame_seq,
            link.firmware_boot_id_token,
        )
        if command_type is CommandType.ARM_TIMED_TASK:
            checkpoint_task(
                args.task_state,
                identity.identity_sha256,
                task_id,
                command.frame_seq,
            )
        receipt = writer.write_next()
        if receipt is None or receipt.status not in FULL_WRITE:
            raise RuntimeError(f"serial write failed: {receipt}")
        if command_type is CommandType.ARM_TIMED_TASK:
            arm_write_count += 1
            if arm_write_count > 1:
                raise RuntimeError("ARM_TIMED_TASK retry forbidden")
        return command

    def confirm_estop(command, *, provisional: bool) -> bool:
        safety = provisional_safety if provisional else link.last_safety
        ack = link.last_ack
        accepted = bool(
            ack is not None
            and ack.ack_for_type == int(CommandType.EMERGENCY_STOP)
            and ack.seq == command.frame_seq
            and ack.status == AckStatus.OK
            and ack.reason == AckReason.EMERGENCY_STOP
            and safety is not None
            and bool(safety.safety_bits & SafetyBits.LOCK_LATCHED)
            and link.last_telemetry is not None
            and link.last_telemetry.pump_mask == 0
        )
        writer.confirm_stop(
            StopConfirmation(
                command_sha256=command.raw_frame_sha256,
                ack_matches_command=accepted,
                firmware_locked=accepted,
                pumps_all_off=accepted,
                evidence_fresh=accepted,
            )
        )
        return accepted

    try:
        initial_stop = send(
            "first-pulse-initial-estop",
            CommandType.EMERGENCY_STOP,
            link.make_emergency_stop,
        )
        read_until(
            1.0,
            lambda: (
                link.last_ack is not None
                and link.last_ack.seq == initial_stop.frame_seq
                and provisional_safety is not None
                and bool(provisional_safety.safety_bits & SafetyBits.LOCK_LATCHED)
                and link.last_telemetry is not None
                and link.last_telemetry.pump_mask == 0
            ),
        )
        initial_stop_confirmed = confirm_estop(initial_stop, provisional=True)

        query = send(
            "first-pulse-firmware-query",
            CommandType.QUERY_FIRMWARE,
            link.make_firmware_query,
        )
        identity_valid = read_until(
            1.0,
            lambda: (
                link.identity_valid
                and link.last_ack is not None
                and link.last_ack.ack_for_type == int(CommandType.QUERY_FIRMWARE)
                and link.last_ack.seq == query.frame_seq
                and link.last_ack.status == AckStatus.OK
            ),
        )
        if not identity_valid:
            raise RuntimeError("firmware identity/ACK timeout")

        heartbeat = send(
            "first-pulse-heartbeat",
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
            raise RuntimeError("heartbeat ACK timeout")

        clear = send(
            "first-pulse-clear-estop",
            CommandType.CLEAR_ESTOP,
            link.make_clear_estop,
        )
        clear_ok = read_until(
            0.6,
            lambda: (
                link.last_ack is not None
                and link.last_ack.ack_for_type == int(CommandType.CLEAR_ESTOP)
                and link.last_ack.seq == clear.frame_seq
                and link.last_ack.status == AckStatus.OK
                and link.last_ack.reason == AckReason.NONE
                and link.last_safety is not None
                and not bool(link.last_safety.safety_bits & SafetyBits.LOCK_LATCHED)
                and link.last_telemetry is not None
                and link.last_telemetry.pump_mask == 0
            ),
        )
        if not clear_ok:
            raise RuntimeError("clear/unlocked state timeout")

        arm = send(
            "first-pulse-arm-timed-task",
            CommandType.ARM_TIMED_TASK,
            lambda: link.make_arm_timed_task(
                task_id=task_id,
                channel=CHANNEL,
                duration_ms=DURATION_MS,
                hard_timeout_ms=HARD_TIMEOUT_MS,
                config_hash_prefix=CONFIG_HASH_PREFIX,
            ),
        )
        arm_ack_ok = read_until(
            0.25,
            lambda: (
                link.last_ack is not None
                and link.last_ack.ack_for_type == int(CommandType.ARM_TIMED_TASK)
                and link.last_ack.seq == arm.frame_seq
                and link.last_ack.task_id == task_id
                and link.last_ack.status == AckStatus.OK
                and link.last_ack.reason == AckReason.NONE
            ),
        )
        if not arm_ack_ok:
            raise RuntimeError("ARM_TIMED_TASK ACK rejected or timed out")

        # Refresh the firmware watchdog immediately after ARM.  Waiting one
        # heartbeat period here can push the gap from the pre-clear heartbeat
        # past the 1 s firmware watchdog once CLEAR_ESTOP and ARM ACK latency
        # are included.
        heartbeat_index = 0
        immediate_heartbeat = send(
            f"first-pulse-task-heartbeat-{heartbeat_index:03d}",
            CommandType.HEARTBEAT,
            link.make_heartbeat,
        )
        if not read_until(
            0.2,
            lambda: (
                link.last_ack is not None
                and link.last_ack.ack_for_type == int(CommandType.HEARTBEAT)
                and link.last_ack.seq == immediate_heartbeat.frame_seq
                and link.last_ack.status == AckStatus.OK
            ),
        ):
            raise RuntimeError("immediate task heartbeat ACK timeout")
        heartbeat_index += 1

        # The commissioned pulse is 100 ms, while explicitly authorized
        # diagnostics may substitute a longer bounded duration.  Always allow
        # the requested dose plus 900 ms for the terminal result, but still
        # issue the exception E-stop before the independent hard timeout.
        deadline = time.monotonic() + (DURATION_MS / 1000.0) + 0.9
        next_heartbeat = time.monotonic() + 0.18
        next_identity_refresh = time.monotonic() + 2.0
        identity_refresh_index = 0
        while time.monotonic() < deadline and link.last_task_result_receipt is None:
            data = transport.read(4096)
            if data:
                event_kinds.extend(event.kind for event in link.ingest(data))
                snapshot()
            now = time.monotonic()
            if now >= next_heartbeat and link.last_task_result_receipt is None:
                followup = send(
                    f"first-pulse-task-heartbeat-{heartbeat_index:03d}",
                    CommandType.HEARTBEAT,
                    link.make_heartbeat,
                )
                if not read_until(
                    0.2,
                    lambda: (
                        link.last_ack is not None
                        and link.last_ack.ack_for_type == int(CommandType.HEARTBEAT)
                        and link.last_ack.seq == followup.frame_seq
                        and link.last_ack.status == AckStatus.OK
                    ),
                ):
                    raise RuntimeError("task heartbeat ACK timeout")
                heartbeat_index += 1
                next_heartbeat = time.monotonic() + 0.18
            now = time.monotonic()
            if (
                now >= next_identity_refresh
                and link.last_task_result_receipt is None
            ):
                refresh = send(
                    f"first-pulse-task-identity-{identity_refresh_index:03d}",
                    CommandType.QUERY_FIRMWARE,
                    link.make_firmware_query,
                )
                if not read_until(
                    0.25,
                    lambda: (
                        link.identity_valid
                        and link.identity_fresh()
                        and link.last_ack is not None
                        and link.last_ack.ack_for_type
                        == int(CommandType.QUERY_FIRMWARE)
                        and link.last_ack.seq == refresh.frame_seq
                        and link.last_ack.status == AckStatus.OK
                    ),
                ):
                    raise RuntimeError("task identity refresh timeout")
                identity_refresh_index += 1
                next_identity_refresh = time.monotonic() + 2.0

        result_receipt = link.last_task_result_receipt
        task_result_ok = bool(
            result_receipt is not None
            and result_receipt.result.task_id == task_id
            and result_receipt.result.terminal_reason
            == TerminalReason.TIMED_DOSE_COMPLETE
            and result_receipt.result.pump_mask == 0
            and not result_receipt.result.scale_stable
        )
        if not task_result_ok:
            raise RuntimeError("missing or invalid TIMED_DOSE_COMPLETE result")

        final_stop = send(
            "first-pulse-final-estop",
            CommandType.EMERGENCY_STOP,
            link.make_emergency_stop,
        )
        read_until(
            0.8,
            lambda: (
                link.last_ack is not None
                and link.last_ack.ack_for_type == int(CommandType.EMERGENCY_STOP)
                and link.last_ack.seq == final_stop.frame_seq
                and link.last_safety is not None
                and bool(link.last_safety.safety_bits & SafetyBits.LOCK_LATCHED)
                and link.last_telemetry is not None
                and link.last_telemetry.pump_mask == 0
            ),
        )
        final_stop_confirmed = confirm_estop(final_stop, provisional=False)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        try:
            emergency = send(
                "first-pulse-exception-estop",
                CommandType.EMERGENCY_STOP,
                link.make_emergency_stop,
            )
            read_until(
                0.8,
                lambda: (
                    link.last_ack is not None
                    and link.last_ack.seq == emergency.frame_seq
                    and link.last_telemetry is not None
                    and link.last_telemetry.pump_mask == 0
                ),
            )
        except BaseException as stop_exc:
            failure += f"; fail-safe estop error={type(stop_exc).__name__}: {stop_exc}"
    finally:
        snapshot()
        arm_receipts = [
            receipt
            for receipt in writer.receipts
            if receipt.command.command_type is CommandType.ARM_TIMED_TASK
        ]
        forbidden = sorted(
            {
                receipt.command.command_type.name
                for receipt in writer.receipts
                if receipt.command.command_type not in ALLOWED_COMMANDS
            }
        )
        result_receipt = link.last_task_result_receipt
        passed = bool(
            failure is None
            and initial_stop_confirmed
            and identity_valid
            and clear_ok
            and arm_ack_ok
            and task_result_ok
            and final_stop_confirmed
            and arm_write_count == 1
            and len(arm_receipts) == 1
            and not forbidden
            and link.last_telemetry is not None
            and link.last_telemetry.pump_mask == 0
            and link.last_safety is not None
            and bool(link.last_safety.safety_bits & SafetyBits.LOCK_LATCHED)
        )
        firmware = link.firmware_info
        payload = {
            "schema": "rootscope.stm32-pb6-first-pulse.v1",
            "started_at_utc": started_at_utc,
            "finished_at_utc": utc_now(),
            "status": "PASS" if passed else "FAIL",
            "failure": failure,
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
            "command": {
                "task_id": task_id,
                "channel": CHANNEL,
                "duration_ms": DURATION_MS,
                "hard_timeout_ms": HARD_TIMEOUT_MS,
                "config_sha256": CONFIG_SHA256,
                "config_hash_prefix_hex": CONFIG_HASH_PREFIX.hex(),
                "arm_write_count": arm_write_count,
                "retry_performed": False,
            },
            "checks": {
                "initial_stop_confirmed": initial_stop_confirmed,
                "identity_valid": identity_valid,
                "clear_ok": clear_ok,
                "arm_ack_ok": arm_ack_ok,
                "task_result_ok": task_result_ok,
                "final_stop_confirmed": final_stop_confirmed,
                "forbidden_commands": forbidden,
                "observed_pump_masks": pump_masks,
            },
            "task_result": (
                None
                if result_receipt is None
                else {
                    "terminal_reason": result_receipt.terminal_reason,
                    "raw_frame_sha256": result_receipt.raw_frame_sha256,
                    "task_id": result_receipt.result.task_id,
                    "pump_mask": result_receipt.result.pump_mask,
                    "scale_stable": result_receipt.result.scale_stable,
                    "firmware_completed_uptime_ms": (
                        result_receipt.result.firmware_completed_uptime_ms
                    ),
                }
            ),
            "event_kinds": event_kinds,
            "writes": [
                {
                    "intent_id": receipt.command.intent_id,
                    "command_type": receipt.command.command_type.name,
                    "sequence": receipt.command.frame_seq,
                    "raw_frame_sha256": receipt.command.raw_frame_sha256,
                    "status": receipt.status.value,
                }
                for receipt in writer.receipts
            ],
            "final": {
                "sequence_checkpoint": link.sequence_checkpoint,
                "lock_latched": (
                    None
                    if link.last_safety is None
                    else bool(link.last_safety.safety_bits & SafetyBits.LOCK_LATCHED)
                ),
                "pump_mask": (
                    None
                    if link.last_telemetry is None
                    else link.last_telemetry.pump_mask
                ),
            },
            "authority": {
                "single_pb6_timed_task_executed": passed,
                "measured_water_mass": False,
                "visual_wetting_confirmed": False,
                "physical_completion": False,
            },
            "claim_boundary": (
                f"PASS proves one firmware-bounded {DURATION_MS} ms PB6 timed task with "
                "matching ACK/result and final E-stop. It does not prove water "
                "mass, wetting coverage or irrigation completion."
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
