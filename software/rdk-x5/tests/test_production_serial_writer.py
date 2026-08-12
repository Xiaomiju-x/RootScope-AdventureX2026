from __future__ import annotations

import threading
import unittest

from app.hardware import (
    CancellationReason,
    DisabledPhysicalSerialOpener,
    PhysicalSerialDisabled,
    PhysicalSerialOpenRequest,
    ROOTSCOPE_F407_ALIAS,
    SerialWriterLocked,
    SerialWriterOwnershipError,
    SerialWriterScheduler,
    StopConfirmation,
    UsbDeviceIdentity,
    WriteStatus,
    WriterBarrier,
)
from app.serial import CommandType, FakeF407, RootScopeSerialLink


DEVICE_SHA = "d" * 64


class MemoryTransport:
    def __init__(self, write_plan: list[object] | None = None) -> None:
        self.backend_id = "MEMORY_TRANSPORT_TEST_ONLY"
        self.device_identity_sha256 = DEVICE_SHA
        self.is_open = True
        self.write_plan = list(write_plan or [])
        self.write_arguments: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.write_arguments.append(bytes(data))
        if self.write_plan:
            result = self.write_plan.pop(0)
            if isinstance(result, BaseException):
                raise result
            return int(result)
        return len(data)

    def read(self, size: int) -> bytes:
        del size
        return b""

    def close(self) -> None:
        self.closed = True
        self.is_open = False


def valid_identity() -> UsbDeviceIdentity:
    return UsbDeviceIdentity(
        alias=ROOTSCOPE_F407_ALIAS,
        vid="1a86",
        pid="7523",
        id_path="pci-0000:01:00.0-usb-0:2:1.0",
        interface_number="00",
    )


def stop_confirmation(command_sha256: str) -> StopConfirmation:
    return StopConfirmation(
        command_sha256=command_sha256,
        ack_matches_command=True,
        firmware_locked=True,
        pumps_all_off=True,
        evidence_fresh=True,
    )


class PhysicalSerialBoundaryTests(unittest.TestCase):
    def test_disabled_opener_refuses_without_touching_any_port(self) -> None:
        request = PhysicalSerialOpenRequest(identity=valid_identity())
        self.assertFalse(request.physical_authority_granted)
        with self.assertRaisesRegex(PhysicalSerialDisabled, "no port was opened"):
            DisabledPhysicalSerialOpener().open_explicit(request)

    def test_identity_requires_explicit_alias_and_stable_usb_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "serial_number or frozen id_path"):
            UsbDeviceIdentity(alias=ROOTSCOPE_F407_ALIAS, vid="1a86", pid="7523")
        with self.assertRaisesRegex(ValueError, "physical serial alias"):
            PhysicalSerialOpenRequest(
                identity=UsbDeviceIdentity(
                    alias="/dev/ttyUSB0",
                    vid="1a86",
                    pid="7523",
                    serial_number="fixture",
                )
            )

    def test_open_request_forbids_enumeration_and_non_boolean_authority(self) -> None:
        with self.assertRaisesRegex(ValueError, "enumeration is forbidden"):
            PhysicalSerialOpenRequest(
                identity=valid_identity(), allow_port_enumeration=True
            )
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            PhysicalSerialOpenRequest(
                identity=valid_identity(), physical_authority_granted="yes"  # type: ignore[arg-type]
            )


class SerialWriterSchedulerTests(unittest.TestCase):
    def _scheduler(
        self, transport: MemoryTransport | None = None
    ) -> tuple[RootScopeSerialLink, MemoryTransport, SerialWriterScheduler]:
        link = RootScopeSerialLink(clock=lambda: 1.0)
        actual_transport = transport or MemoryTransport()
        writer = SerialWriterScheduler(actual_transport, clock=lambda: 1.0)
        writer.claim_current_thread()
        return link, actual_transport, writer

    @staticmethod
    def _schedule_estop(
        writer: SerialWriterScheduler, link: RootScopeSerialLink, intent: str = "stop-1"
    ):
        return writer.schedule(
            intent_id=intent,
            build_frame=lambda: link.make_emergency_stop(now=1.0),
            expected_command_type=CommandType.EMERGENCY_STOP,
            on_fully_written=lambda raw, stamp: link.mark_command_sent(
                raw, execution_backend="MEMORY_TRANSPORT_TEST_ONLY", now=stamp
            ),
        )

    def _unlock(self, writer: SerialWriterScheduler, link: RootScopeSerialLink) -> None:
        command = self._schedule_estop(writer, link)
        receipt = writer.write_next()
        self.assertEqual(receipt.status, WriteStatus.FULLY_WRITTEN)
        writer.confirm_stop(stop_confirmation(command.raw_frame_sha256))

    def test_first_semantic_write_must_be_estop_and_confirmation_bound(self) -> None:
        link, transport, writer = self._scheduler()
        with self.assertRaisesRegex(SerialWriterLocked, "E-stop"):
            writer.schedule(
                intent_id="query-too-early",
                build_frame=link.make_firmware_query,
                expected_command_type=CommandType.QUERY_FIRMWARE,
            )
        self.assertEqual(link.command_receipt_history, [])
        command = self._schedule_estop(writer, link)
        receipt = writer.write_next()
        self.assertEqual(receipt.command, command)
        self.assertEqual(b"".join(transport.write_arguments), command.raw_frame)
        self.assertEqual(
            writer.barrier, WriterBarrier.ESTOP_WRITTEN_AWAITING_CONFIRMATION
        )
        with self.assertRaisesRegex(SerialWriterLocked, "not yet confirmed"):
            writer.schedule(
                intent_id="query-before-confirm",
                build_frame=link.make_firmware_query,
                expected_command_type=CommandType.QUERY_FIRMWARE,
            )
        writer.confirm_stop(stop_confirmation(command.raw_frame_sha256))
        self.assertEqual(writer.barrier, WriterBarrier.NORMAL_COMMANDS_ENABLED)

    def test_estop_preempts_and_cancels_queued_normal_permanently(self) -> None:
        link, transport, writer = self._scheduler()
        self._unlock(writer, link)
        normal = writer.schedule(
            intent_id="query-1",
            build_frame=link.make_firmware_query,
            expected_command_type=CommandType.QUERY_FIRMWARE,
        )
        stop = self._schedule_estop(writer, link, "stop-preempt")
        self.assertEqual(writer.barrier, WriterBarrier.ESTOP_QUEUED)
        self.assertEqual(writer.queue_depth, 1)
        self.assertEqual(len(writer.cancellations), 1)
        cancellation = writer.cancellations[0]
        self.assertEqual(cancellation.cancelled_command, normal)
        self.assertEqual(
            cancellation.reason, CancellationReason.EMERGENCY_STOP_INVALIDATED
        )
        self.assertFalse(cancellation.transport_write_attempted)
        self.assertEqual(
            cancellation.to_dict()["invalidating_estop_sha256"],
            stop.raw_frame_sha256,
        )
        first = writer.write_next()
        self.assertEqual(first.command, stop)
        writer.confirm_stop(stop_confirmation(stop.raw_frame_sha256))
        self.assertEqual(writer.barrier, WriterBarrier.NORMAL_COMMANDS_ENABLED)
        self.assertIsNone(writer.write_next())
        self.assertNotIn(normal.raw_frame, transport.write_arguments)

    def test_estop_cancels_stale_arm_and_normal_across_confirmation(self) -> None:
        link, transport, writer = self._scheduler()
        self._unlock(writer, link)
        fake = FakeF407()
        query = link.make_firmware_query(now=1.0)
        link.ingest(fake.exchange(query, now=1.0), now=1.0)
        self.assertTrue(link.identity_valid)

        callback_calls: list[str] = []
        arm = writer.schedule(
            intent_id="arm-stale",
            task_id="task-00000001",
            build_frame=lambda: link.make_arm_task(
                task_id=1,
                channel=1,
                target_mass_mg=2_000,
                hard_timeout_ms=3_000,
                config_hash_prefix=bytes.fromhex("0123456789abcdef"),
                now=1.0,
            ),
            expected_command_type=CommandType.ARM_TASK,
            on_fully_written=lambda _raw, _stamp: callback_calls.append("arm"),
        )
        normal = writer.schedule(
            intent_id="heartbeat-stale",
            build_frame=lambda: link.make_heartbeat(now=1.0),
            expected_command_type=CommandType.HEARTBEAT,
            on_fully_written=lambda _raw, _stamp: callback_calls.append("heartbeat"),
        )
        stop = self._schedule_estop(writer, link, "stop-invalidates-task-epoch")
        self.assertEqual(
            {item.cancelled_command.intent_id for item in writer.cancellations[-2:]},
            {"arm-stale", "heartbeat-stale"},
        )
        self.assertTrue(
            all(
                not item.transport_write_attempted
                for item in writer.cancellations[-2:]
            )
        )
        self.assertEqual(writer.write_next().command, stop)
        writer.confirm_stop(stop_confirmation(stop.raw_frame_sha256))
        self.assertIsNone(writer.write_next())
        self.assertEqual(callback_calls, [])
        self.assertNotIn(arm.raw_frame, transport.write_arguments)
        self.assertNotIn(normal.raw_frame, transport.write_arguments)

    def test_multiple_estops_keep_normal_commands_locked_until_each_is_confirmed(self) -> None:
        link, _transport, writer = self._scheduler()
        first = self._schedule_estop(writer, link, "stop-multi-1")
        second = self._schedule_estop(writer, link, "stop-multi-2")
        self.assertEqual(writer.barrier, WriterBarrier.ESTOP_QUEUED)
        with self.assertRaises(SerialWriterLocked):
            writer.schedule(
                intent_id="normal-before-first-write",
                build_frame=link.make_firmware_query,
                expected_command_type=CommandType.QUERY_FIRMWARE,
            )

        self.assertEqual(writer.write_next().command, first)
        self.assertEqual(
            writer.barrier, WriterBarrier.ESTOP_WRITTEN_AWAITING_CONFIRMATION
        )
        with self.assertRaises(SerialWriterLocked):
            writer.schedule(
                intent_id="normal-before-first-confirm",
                build_frame=link.make_firmware_query,
                expected_command_type=CommandType.QUERY_FIRMWARE,
            )
        writer.confirm_stop(stop_confirmation(first.raw_frame_sha256))
        self.assertEqual(writer.barrier, WriterBarrier.ESTOP_QUEUED)
        with self.assertRaises(SerialWriterLocked):
            writer.schedule(
                intent_id="normal-between-stops",
                build_frame=link.make_firmware_query,
                expected_command_type=CommandType.QUERY_FIRMWARE,
            )

        self.assertEqual(writer.write_next().command, second)
        self.assertEqual(
            writer.barrier, WriterBarrier.ESTOP_WRITTEN_AWAITING_CONFIRMATION
        )
        with self.assertRaises(SerialWriterLocked):
            writer.schedule(
                intent_id="normal-before-final-confirm",
                build_frame=link.make_firmware_query,
                expected_command_type=CommandType.QUERY_FIRMWARE,
            )
        writer.confirm_stop(stop_confirmation(second.raw_frame_sha256))
        self.assertEqual(writer.barrier, WriterBarrier.NORMAL_COMMANDS_ENABLED)
        normal = writer.schedule(
            intent_id="normal-after-all-confirmed",
            build_frame=link.make_firmware_query,
            expected_command_type=CommandType.QUERY_FIRMWARE,
        )
        self.assertEqual(writer.write_next().command, normal)

    def test_stale_confirmation_after_two_written_estops_faults_and_new_estop_recovers(self) -> None:
        link, _transport, writer = self._scheduler()
        first = self._schedule_estop(writer, link, "stop-written-1")
        second = self._schedule_estop(writer, link, "stop-written-2")
        self.assertEqual(writer.write_next().command, first)
        self.assertEqual(writer.write_next().command, second)
        self.assertEqual(
            writer.barrier, WriterBarrier.ESTOP_WRITTEN_AWAITING_CONFIRMATION
        )

        with self.assertRaisesRegex(SerialWriterLocked, "latest E-stop"):
            writer.confirm_stop(stop_confirmation(first.raw_frame_sha256))
        self.assertEqual(writer.barrier, WriterBarrier.WRITE_FAULT_LOCKED)
        with self.assertRaisesRegex(SerialWriterLocked, "fault permits E-stop only"):
            writer.schedule(
                intent_id="normal-after-stale-confirmation",
                build_frame=link.make_firmware_query,
                expected_command_type=CommandType.QUERY_FIRMWARE,
            )

        recovery = self._schedule_estop(writer, link, "stop-recovery")
        self.assertEqual(writer.barrier, WriterBarrier.ESTOP_QUEUED)
        self.assertEqual(writer.write_next().command, recovery)
        writer.confirm_stop(stop_confirmation(recovery.raw_frame_sha256))
        self.assertEqual(writer.barrier, WriterBarrier.NORMAL_COMMANDS_ENABLED)

    def test_short_writes_are_retried_without_frame_interleaving(self) -> None:
        transport = MemoryTransport(write_plan=[2])
        link, _transport, writer = self._scheduler(transport)
        command = self._schedule_estop(writer, link)
        receipt = writer.write_next()
        self.assertEqual(receipt.status, WriteStatus.FULLY_WRITTEN_AFTER_SHORT_WRITE)
        self.assertEqual(receipt.write_calls, 2)
        written = transport.write_arguments[0][:2] + transport.write_arguments[1]
        self.assertEqual(written, command.raw_frame)
        self.assertTrue(receipt.complete_frame_written)

    def test_zero_length_writes_fail_known_not_written_and_lock(self) -> None:
        transport = MemoryTransport(write_plan=[0, 0])
        link = RootScopeSerialLink(clock=lambda: 1.0)
        writer = SerialWriterScheduler(
            transport, max_short_write_calls=2, clock=lambda: 1.0
        )
        writer.claim_current_thread()
        self._schedule_estop(writer, link)
        receipt = writer.write_next()
        self.assertEqual(receipt.status, WriteStatus.FAILED_NOT_WRITTEN)
        self.assertFalse(receipt.raw_frame_may_have_reached_device)
        self.assertEqual(writer.barrier, WriterBarrier.WRITE_FAULT_LOCKED)

    def test_write_exception_is_unknown_not_falsely_not_written(self) -> None:
        transport = MemoryTransport(write_plan=[OSError("disconnected")])
        link, _transport, writer = self._scheduler(transport)
        self._schedule_estop(writer, link)
        receipt = writer.write_next()
        self.assertEqual(receipt.status, WriteStatus.WRITE_OUTCOME_UNKNOWN)
        self.assertTrue(receipt.raw_frame_may_have_reached_device)
        self.assertIn("disconnected", receipt.error)
        self.assertEqual(writer.barrier, WriterBarrier.WRITE_FAULT_LOCKED)

    def test_only_claimed_thread_can_drain_transport(self) -> None:
        link, transport, writer = self._scheduler()
        self._schedule_estop(writer, link)
        errors: list[BaseException] = []

        def intruder() -> None:
            try:
                writer.write_next()
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=intruder)
        thread.start()
        thread.join(timeout=2.0)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], SerialWriterOwnershipError)
        self.assertEqual(transport.write_arguments, [])
        self.assertEqual(writer.write_next().status, WriteStatus.FULLY_WRITTEN)


if __name__ == "__main__":
    unittest.main()
