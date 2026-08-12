from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import sys
import unittest


ROOTSCOPE_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOTSCOPE_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    # Keep the protocol fixture test independent from sibling app modules that
    # are implemented concurrently during H0-H12.
    sys.path.insert(0, str(APP_ROOT))

from serial.fake_f407 import FakeF407  # noqa: E402
from serial.frame import (  # noqa: E402
    Ack,
    AckReason,
    AckStatus,
    ArmTask,
    CommandType,
    EXPECTED_BUILD_ID,
    Heartbeat,
    LockReason,
    ResponseType,
    SafetyBits,
    SeqCommand,
    TaskResult,
    TerminalReason,
    boot_id_token,
    decode_frame,
    encode_frame,
    encode_message,
)
from serial.link import (  # noqa: E402
    CommandAckReceipt,
    CommandFrameReceipt,
    IdentityExpectation,
    RootScopeSerialLink,
    SerialAdmissionError,
    TaskResultReceipt,
)


CONFIG_HASH = bytes.fromhex("0123456789abcdef")


def exchange(
    link: RootScopeSerialLink, fake: FakeF407, wire: bytes, now: float
) -> None:
    try:
        link.mark_command_sent(wire, execution_backend="FAKE_F407", now=now)
    except SerialAdmissionError:
        # Some replay/fuzz tests deliberately inject frames that were not made
        # by the host link; those must not gain a fabricated send receipt.
        pass
    reply = fake.exchange(wire, now=now)
    link.ingest(reply, now=now)


def ready_pair(now: float = 0.0) -> tuple[RootScopeSerialLink, FakeF407]:
    link = RootScopeSerialLink()
    fake = FakeF407()
    exchange(link, fake, link.make_firmware_query(), now)
    if not link.identity_valid:
        raise AssertionError(link.identity_error)
    exchange(link, fake, link.make_heartbeat(now=now + 0.01), now + 0.01)
    exchange(link, fake, link.make_clear_estop(now=now + 0.02), now + 0.02)
    if fake.locked:
        raise AssertionError("fixture failed to leave boot lock")
    return link, fake


def collect_target_result(
    link: RootScopeSerialLink, fake: FakeF407
) -> None:
    for stamp in (0.10, 0.30, 0.50, 0.70, 0.90):
        exchange(link, fake, link.make_heartbeat(now=stamp), stamp)
        link.ingest(fake.tick(now=stamp), now=stamp)


class IdentityAndHeartbeatTests(unittest.TestCase):
    def test_host_sequence_can_resume_from_durable_checkpoint(self) -> None:
        link = RootScopeSerialLink(initial_sequence=41)
        wire = link.make_firmware_query(now=0.0)
        self.assertEqual(
            SeqCommand.from_payload(decode_frame(wire).payload).seq,
            42,
        )
        self.assertEqual(link.sequence_checkpoint, 42)

    def test_initial_sequence_rejects_non_uint16_values(self) -> None:
        for value in (-1, 0x10000, True, 1.5, "1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RootScopeSerialLink(initial_sequence=value)

    def test_wrong_firmware_identity_blocks_arm(self) -> None:
        link = RootScopeSerialLink()
        fake = FakeF407(build_id=EXPECTED_BUILD_ID + 1)
        exchange(link, fake, link.make_firmware_query(), 0.0)
        self.assertFalse(link.identity_valid)
        self.assertIn("build_id", link.identity_error)
        with self.assertRaises(SerialAdmissionError):
            link.make_arm_task(
                task_id=1,
                channel=1,
                target_mass_mg=5_000,
                hard_timeout_ms=5_000,
                config_hash_prefix=CONFIG_HASH,
                now=0.1,
            )
        self.assertEqual(fake.pump_mask, 0)

    def test_missing_capability_blocks_arm(self) -> None:
        link = RootScopeSerialLink(expectation=IdentityExpectation())
        fake = FakeF407(capabilities=0x1F)
        exchange(link, fake, link.make_firmware_query(), 0.0)
        self.assertFalse(link.identity_valid)
        self.assertIn("missing_capabilities", link.identity_error)

    def test_task_result_from_wrong_boot_is_not_receipted(self) -> None:
        link = RootScopeSerialLink()
        fake = FakeF407(boot_id=100)
        exchange(link, fake, link.make_firmware_query(now=0.0), 0.0)
        forged_other_boot = TaskResult(
            boot_id=101,
            task_id=1,
            result_seq=1,
            terminal_reason=TerminalReason.HARD_TIMEOUT,
            baseline_mass_mg=500_000,
            final_mass_mg=499_000,
            first_sample_seq=1,
            last_sample_seq=1,
            sample_count=1,
            final_window_min_mg=499_000,
            final_window_max_mg=499_000,
            scale_stable=False,
            firmware_completed_uptime_ms=10,
            pump_mask=0,
            safety_bits=SafetyBits.LOCK_LATCHED,
        )
        events = link.ingest(
            encode_message(ResponseType.TASK_RESULT, forged_other_boot), now=0.1
        )
        self.assertIn("task_result_rejected", [event.kind for event in events])
        self.assertIsNone(link.last_task_result_receipt)

    def test_conflicting_task_result_permanently_revokes_trust(self) -> None:
        link = RootScopeSerialLink()
        fake = FakeF407(boot_id=0xCAFE)
        exchange(link, fake, link.make_firmware_query(now=0.0), 0.0)
        first = TaskResult(
            boot_id=fake.firmware_info.boot_id,
            task_id=17,
            result_seq=5,
            terminal_reason=TerminalReason.HARD_TIMEOUT,
            baseline_mass_mg=500_000,
            final_mass_mg=499_000,
            first_sample_seq=9,
            last_sample_seq=9,
            sample_count=1,
            final_window_min_mg=499_000,
            final_window_max_mg=499_000,
            scale_stable=False,
            firmware_completed_uptime_ms=100,
            pump_mask=0,
            safety_bits=SafetyBits.LOCK_LATCHED,
        )
        first_wire = encode_message(ResponseType.TASK_RESULT, first)
        events = link.ingest(first_wire, now=0.1)
        self.assertIn("task_result", [event.kind for event in events])
        first_receipt = link.task_result_for(first.task_id)
        self.assertIsNotNone(first_receipt)
        self.assertEqual(first_receipt.raw_frame, first_wire)
        self.assertEqual(
            first_receipt.raw_frame_sha256,
            hashlib.sha256(first_wire).hexdigest(),
        )

        conflicting = replace(first, final_mass_mg=498_999)
        conflict_wire = encode_message(ResponseType.TASK_RESULT, conflicting)
        events = link.ingest(conflict_wire, now=0.2)
        self.assertIn("task_result_conflict", [event.kind for event in events])
        key = (first.boot_id, first.task_id)
        self.assertTrue(link.task_result_conflict_active)
        self.assertTrue(link.task_result_conflicted(first.task_id))
        self.assertIn(key, link.task_result_conflicted_keys)
        self.assertNotIn(key, link.task_result_history)
        self.assertNotIn(first.task_id, link.task_result_receipts)
        self.assertIsNone(link.task_result_for(first.task_id))
        self.assertIsNone(link.last_task_result_receipt)

        # Replaying either the original or conflicting frame cannot restore
        # trust after the key has been conflict-latched.
        original_replay = link.ingest(first_wire, now=0.3)
        conflict_replay = link.ingest(conflict_wire, now=0.4)
        self.assertIn(
            "task_result_conflict_locked_duplicate",
            [event.kind for event in original_replay],
        )
        self.assertIn(
            "task_result_conflict_locked_duplicate",
            [event.kind for event in conflict_replay],
        )
        self.assertIsNone(link.task_result_for(first.task_id))
        self.assertTrue(link.task_result_conflict_active)

        # A third distinct result is forensic-only as well.  Its exact raw
        # frame/hash are retained without ever repopulating a trusted map.
        third = replace(first, final_mass_mg=498_998)
        third_wire = encode_message(ResponseType.TASK_RESULT, third)
        third_events = link.ingest(third_wire, now=0.5)
        self.assertIn(
            "task_result_conflict_locked",
            [event.kind for event in third_events],
        )
        forensic = link.task_result_forensic_history[key]
        self.assertEqual(
            {receipt.raw_frame for receipt in forensic},
            {first_wire, conflict_wire, third_wire},
        )
        self.assertEqual(
            {receipt.raw_frame_sha256 for receipt in forensic},
            {
                hashlib.sha256(first_wire).hexdigest(),
                hashlib.sha256(conflict_wire).hexdigest(),
                hashlib.sha256(third_wire).hexdigest(),
            },
        )
        self.assertIsNone(link.task_result_for(first.task_id))
        self.assertNotIn(key, link.task_result_history)

    def test_identity_expires_on_host(self) -> None:
        link = RootScopeSerialLink()
        fake = FakeF407()
        exchange(link, fake, link.make_firmware_query(), 0.0)
        self.assertTrue(link.identity_fresh(3.0))
        self.assertFalse(link.identity_fresh(3.001))

    def test_heartbeat_scheduler_is_five_hz(self) -> None:
        link = RootScopeSerialLink()
        self.assertTrue(link.heartbeat_due(10.0))
        link.make_heartbeat(now=10.0)
        self.assertFalse(link.heartbeat_due(10.199))
        self.assertTrue(link.heartbeat_due(10.2))

    def test_idle_tick_exposes_same_boot_id_and_monotonic_sample(self) -> None:
        link = RootScopeSerialLink()
        fake = FakeF407(boot_id=0x1234)
        exchange(link, fake, link.make_firmware_query(now=0.0), 0.0)
        link.ingest(fake.tick(now=0.1), now=0.1)
        self.assertEqual(link.firmware_info.boot_id, 0x1234)
        self.assertEqual(link.last_safety.boot_id, 0x1234)
        self.assertEqual(link.last_telemetry.sample_seq, 1)
        link.ingest(fake.tick(now=0.3), now=0.3)
        self.assertEqual(link.last_telemetry.sample_seq, 2)

    def test_command_and_ack_receipts_bind_exact_decoded_bytes(self) -> None:
        link, fake = ready_pair()
        clear_ack = link.last_ack
        clear_command = link.command_receipt_for(
            CommandType.CLEAR_ESTOP, clear_ack.seq
        )
        self.assertTrue(clear_command.sent)
        self.assertEqual(clear_command.execution_backend, "FAKE_F407")
        self.assertEqual(clear_command.transcript, clear_command.raw_frame.hex())
        clear_receipt = link.command_ack_receipt_for(
            CommandType.CLEAR_ESTOP, clear_ack.seq
        )
        self.assertEqual(clear_receipt.firmware_build_id, str(EXPECTED_BUILD_ID))
        self.assertEqual(clear_receipt.firmware_boot_id, link.firmware_boot_id_token)
        self.assertEqual(
            clear_receipt.ack_raw_frame_sha256,
            hashlib.sha256(clear_receipt.ack_raw_frame).hexdigest(),
        )

        arm_wire = link.make_arm_task(
            task_id=11,
            channel=3,
            target_mass_mg=2_000,
            hard_timeout_ms=5_000,
            config_hash_prefix=CONFIG_HASH,
            now=0.03,
        )
        arm_frame = decode_frame(arm_wire)
        decoded = ArmTask.from_payload(arm_frame.payload)
        generated = link.command_receipt_for(CommandType.ARM_TASK, decoded.seq)
        self.assertFalse(generated.sent)
        self.assertEqual(generated.decoded, decoded)
        self.assertEqual(generated.raw_frame_sha256, hashlib.sha256(arm_wire).hexdigest())
        exchange(link, fake, arm_wire, 0.03)
        self.assertTrue(link.command_receipt_for(CommandType.ARM_TASK, decoded.seq).sent)
        self.assertEqual(
            link.command_ack_receipt_for(CommandType.ARM_TASK, decoded.seq).ack.task_id,
            11,
        )

    def test_public_command_receipt_rejects_mismatched_decoded_payload(self) -> None:
        raw = encode_message(CommandType.CLEAR_ESTOP, SeqCommand(7))
        with self.assertRaises(ValueError):
            CommandFrameReceipt(
                command_type=CommandType.CLEAR_ESTOP,
                seq=7,
                decoded=SeqCommand(8),
                raw_frame=raw,
                raw_frame_sha256=hashlib.sha256(raw).hexdigest(),
                generated_at=0.0,
                sent_at=None,
                execution_backend="NOT_SENT",
                firmware_build_id=None,
                firmware_boot_id=None,
            )

    def test_public_ack_receipt_rejects_mismatched_decoded_raw_frame(self) -> None:
        link, _fake = ready_pair()
        valid = link.last_command_ack_receipt
        different_ack = Ack(
            valid.ack.ack_for_type,
            valid.ack.seq,
            valid.ack.status,
            valid.ack.reason,
            valid.ack.task_id + 1,
        )
        different_raw = encode_message(ResponseType.ACK, different_ack)
        with self.assertRaises(ValueError):
            CommandAckReceipt(
                command=valid.command,
                ack=valid.ack,
                ack_raw_frame=different_raw,
                ack_raw_frame_sha256=hashlib.sha256(different_raw).hexdigest(),
                received_at=valid.received_at,
                firmware_build_id=valid.firmware_build_id,
                firmware_boot_id=valid.firmware_boot_id,
                execution_backend=valid.execution_backend,
            )

    def test_public_task_result_receipt_rejects_mismatched_decoded_raw_frame(
        self,
    ) -> None:
        result = TaskResult(
            boot_id=0x1234,
            task_id=9,
            result_seq=3,
            terminal_reason=TerminalReason.HARD_TIMEOUT,
            baseline_mass_mg=500_000,
            final_mass_mg=499_000,
            first_sample_seq=7,
            last_sample_seq=7,
            sample_count=1,
            final_window_min_mg=499_000,
            final_window_max_mg=499_000,
            scale_stable=False,
            firmware_completed_uptime_ms=50,
            pump_mask=0,
            safety_bits=SafetyBits.LOCK_LATCHED,
        )
        different_result = TaskResult(
            boot_id=result.boot_id,
            task_id=result.task_id,
            result_seq=result.result_seq,
            terminal_reason=result.terminal_reason,
            baseline_mass_mg=result.baseline_mass_mg,
            final_mass_mg=498_999,
            first_sample_seq=result.first_sample_seq,
            last_sample_seq=result.last_sample_seq,
            sample_count=result.sample_count,
            final_window_min_mg=498_999,
            final_window_max_mg=498_999,
            scale_stable=result.scale_stable,
            firmware_completed_uptime_ms=result.firmware_completed_uptime_ms,
            pump_mask=result.pump_mask,
            safety_bits=result.safety_bits,
        )
        different_raw = encode_message(ResponseType.TASK_RESULT, different_result)
        with self.assertRaises(ValueError):
            TaskResultReceipt(
                result=result,
                raw_frame_sha256=hashlib.sha256(different_raw).hexdigest(),
                raw_frame=different_raw,
                received_at=0.1,
                result_frame_seq=result.result_seq,
                firmware_build_id=str(EXPECTED_BUILD_ID),
                firmware_boot_id=boot_id_token(result.boot_id),
            )

    def test_ack_success_is_action_specific_and_task_bound(self) -> None:
        misleading = Ack(
            CommandType.ARM_TASK,
            7,
            AckStatus.OK,
            AckReason.EMERGENCY_STOP,
            42,
        )
        self.assertFalse(
            RootScopeSerialLink.ack_ok(
                misleading,
                expected_type=CommandType.ARM_TASK,
                expected_task_id=42,
            )
        )
        valid = Ack(CommandType.ARM_TASK, 7, AckStatus.OK, AckReason.NONE, 42)
        self.assertTrue(
            RootScopeSerialLink.ack_ok(
                valid,
                expected_type=CommandType.ARM_TASK,
                expected_task_id=42,
            )
        )
        self.assertFalse(
            RootScopeSerialLink.ack_ok(
                valid,
                expected_type=CommandType.ARM_TASK,
                expected_task_id=43,
            )
        )

    def test_watchdog_gap_over_one_second_stops_and_latches(self) -> None:
        link, fake = ready_pair()
        arm = link.make_arm_task(
            task_id=1,
            channel=1,
            target_mass_mg=100_000,
            hard_timeout_ms=10_000,
            config_hash_prefix=CONFIG_HASH,
            now=0.03,
        )
        exchange(link, fake, arm, 0.03)
        self.assertEqual(fake.pump_mask, 1)

        link.ingest(fake.tick(now=1.02), now=1.02)
        self.assertEqual(fake.pump_mask, 0)
        self.assertTrue(fake.locked)
        self.assertEqual(fake.lock_reason, LockReason.WATCHDOG_TIMEOUT)
        self.assertTrue(link.last_safety.safety_bits & SafetyBits.LOCK_LATCHED)
        self.assertEqual(
            link.last_task_result_receipt.terminal_reason, "WATCHDOG_TIMEOUT"
        )
        self.assertFalse(link.last_task_result_receipt.result.scale_stable)

        # A late heartbeat cannot silently resume the old task.
        exchange(link, fake, link.make_heartbeat(now=1.03), 1.03)
        self.assertTrue(fake.locked)
        self.assertIsNone(fake.active_task)
        self.assertEqual(fake.pump_mask, 0)


class TaskAndInterlockTests(unittest.TestCase):
    def test_each_channel_maps_to_exactly_one_pump(self) -> None:
        for channel, mask in ((1, 1), (2, 2), (3, 4)):
            with self.subTest(channel=channel):
                link, fake = ready_pair()
                wire = link.make_arm_task(
                    task_id=channel,
                    channel=channel,
                    target_mass_mg=10_000,
                    hard_timeout_ms=5_000,
                    config_hash_prefix=CONFIG_HASH,
                    now=0.03,
                )
                exchange(link, fake, wire, 0.03)
                self.assertEqual(fake.pump_mask, mask)
                self.assertTrue(fake.at_most_one_pump)
                self.assertEqual(fake.pump_mask.bit_count(), 1)

    def test_duplicate_frame_is_rejected_without_reexecution(self) -> None:
        link, fake = ready_pair()
        wire = link.make_arm_task(
            task_id=7,
            channel=2,
            target_mass_mg=10_000,
            hard_timeout_ms=5_000,
            config_hash_prefix=CONFIG_HASH,
            now=0.03,
        )
        exchange(link, fake, wire, 0.03)
        accepted_task = fake.active_task
        exchange(link, fake, wire, 0.04)
        self.assertEqual(link.last_ack.status, AckStatus.REJECTED)
        self.assertEqual(link.last_ack.reason, AckReason.DUPLICATE_SEQ)
        self.assertIs(fake.active_task, accepted_task)
        self.assertEqual(fake.pump_mask, 2)

    def test_busy_command_cannot_enable_second_pump(self) -> None:
        link, fake = ready_pair()
        exchange(
            link,
            fake,
            link.make_arm_task(
                task_id=1,
                channel=1,
                target_mass_mg=10_000,
                hard_timeout_ms=5_000,
                config_hash_prefix=CONFIG_HASH,
                now=0.03,
            ),
            0.03,
        )
        exchange(
            link,
            fake,
            link.make_arm_task(
                task_id=2,
                channel=3,
                target_mass_mg=10_000,
                hard_timeout_ms=5_000,
                config_hash_prefix=CONFIG_HASH,
                now=0.04,
            ),
            0.04,
        )
        self.assertEqual(link.last_ack.reason, AckReason.BUSY)
        self.assertEqual(fake.pump_mask, 1)
        self.assertTrue(fake.at_most_one_pump)

    def test_duplicate_and_old_task_ids_are_rejected_after_clear(self) -> None:
        link, fake = ready_pair()
        exchange(
            link,
            fake,
            link.make_arm_task(
                task_id=10,
                channel=1,
                target_mass_mg=10_000,
                hard_timeout_ms=5_000,
                config_hash_prefix=CONFIG_HASH,
                now=0.03,
            ),
            0.03,
        )
        exchange(link, fake, link.make_abort_task(task_id=10), 0.04)
        self.assertEqual(link.last_task_result_receipt.terminal_reason, "USER_ABORT")
        exchange(link, fake, link.make_heartbeat(now=0.05), 0.05)
        exchange(link, fake, link.make_clear_estop(now=0.06), 0.06)
        self.assertFalse(fake.locked)

        duplicate = encode_message(
            CommandType.ARM_TASK,
            ArmTask(10, link.next_seq(), 1, 5_000, 5_000, CONFIG_HASH),
        )
        exchange(link, fake, duplicate, 0.07)
        self.assertEqual(link.last_ack.reason, AckReason.DUPLICATE_TASK)
        self.assertEqual(fake.pump_mask, 0)

        old = encode_message(
            CommandType.ARM_TASK,
            ArmTask(9, link.next_seq(), 1, 5_000, 5_000, CONFIG_HASH),
        )
        exchange(link, fake, old, 0.08)
        self.assertEqual(link.last_ack.reason, AckReason.STALE_TASK)
        self.assertEqual(fake.pump_mask, 0)

    def test_sequence_wrap_accepts_new_value_and_rejects_old(self) -> None:
        fake = FakeF407()
        fake.last_seq = 0xFFFF
        heartbeat = encode_message(CommandType.HEARTBEAT, Heartbeat(1, 0))
        reply = fake.exchange(heartbeat, now=0.0)
        link = RootScopeSerialLink()
        link.ingest(reply, now=0.0)
        self.assertEqual(link.last_ack.status, AckStatus.OK)
        self.assertEqual(fake.last_seq, 1)

        stale = encode_message(CommandType.HEARTBEAT, Heartbeat(0xFF00, 0))
        link.ingest(fake.exchange(stale, now=0.01), now=0.01)
        self.assertEqual(link.last_ack.reason, AckReason.STALE_SEQ)

    def test_reserved_zero_sequence_is_rejected(self) -> None:
        fake = FakeF407()
        wire = encode_message(CommandType.HEARTBEAT, Heartbeat(0, 0))
        link = RootScopeSerialLink()
        link.ingest(fake.exchange(wire, now=0.0), now=0.0)
        self.assertEqual(link.last_ack.status, AckStatus.LOCKED)
        self.assertEqual(link.last_ack.reason, AckReason.STALE_SEQ)
        self.assertIsNone(fake.last_heartbeat_at)

    def test_target_mass_waits_for_firmware_stable_result(self) -> None:
        link, fake = ready_pair()
        exchange(
            link,
            fake,
            link.make_arm_task(
                task_id=1,
                channel=1,
                target_mass_mg=100,
                hard_timeout_ms=5_000,
                config_hash_prefix=CONFIG_HASH,
                now=0.03,
            ),
            0.03,
        )
        exchange(link, fake, link.make_heartbeat(now=0.10), 0.10)
        link.ingest(fake.tick(now=0.10), now=0.10)
        self.assertEqual(fake.pump_mask, 0)
        self.assertIsNotNone(fake.active_task)
        self.assertEqual(link.last_telemetry.task_id, 1)
        self.assertIsNone(link.last_task_result_receipt)
        self.assertFalse(fake.locked)

        for stamp in (0.30, 0.50, 0.70, 0.90):
            exchange(link, fake, link.make_heartbeat(now=stamp), stamp)
            link.ingest(fake.tick(now=stamp), now=stamp)
        self.assertIsNone(fake.active_task)
        receipt = link.last_task_result_receipt
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.terminal_reason, "TARGET_REACHED")
        self.assertEqual(receipt.result.post_stop_sample_count, 5)
        self.assertTrue(receipt.result.scale_stable)
        self.assertLessEqual(receipt.result.final_window_span_mg, 20)
        self.assertEqual(receipt.result.pump_mask, 0)
        self.assertEqual(receipt.result.boot_id, fake.firmware_info.boot_id)
        self.assertEqual(
            receipt.raw_frame_sha256, hashlib.sha256(receipt.raw_frame).hexdigest()
        )
        self.assertEqual(decode_frame(receipt.raw_frame).message_type, ResponseType.TASK_RESULT)
        self.assertEqual(receipt.result_frame_seq, receipt.result.result_seq)
        exchange(link, fake, link.make_heartbeat(now=1.10), 1.10)
        events = link.ingest(fake.tick(now=1.10), now=1.10)
        self.assertIn("task_result_duplicate", [event.kind for event in events])
        self.assertIs(link.last_task_result_receipt, receipt)
        self.assertEqual(len(link.task_result_history), 1)

    def test_fresh_external_hx711_sample_can_stop_target_immediately(self) -> None:
        link, fake = ready_pair()
        exchange(
            link,
            fake,
            link.make_arm_task(
                task_id=1,
                channel=1,
                target_mass_mg=1_000,
                hard_timeout_ms=5_000,
                config_hash_prefix=CONFIG_HASH,
                now=0.03,
            ),
            0.03,
        )
        baseline = fake.active_task.baseline_mass_mg
        fake.set_hx711(filtered_mass_mg=baseline - 1_000, valid=True, now=0.04)
        self.assertEqual(fake.pump_mask, 0)
        self.assertIsNotNone(fake.active_task)
        self.assertFalse(fake.locked)
        collect_target_result(link, fake)
        self.assertIsNone(fake.active_task)
        self.assertEqual(
            link.last_task_result_receipt.result.terminal_reason,
            TerminalReason.TARGET_REACHED,
        )
        self.assertTrue(link.last_task_result_receipt.result.scale_stable)

    def test_hard_timeout_is_independent_of_fresh_heartbeat(self) -> None:
        link, fake = ready_pair()
        exchange(
            link,
            fake,
            link.make_arm_task(
                task_id=1,
                channel=1,
                target_mass_mg=100_000,
                hard_timeout_ms=500,
                config_hash_prefix=CONFIG_HASH,
                now=0.03,
            ),
            0.03,
        )
        exchange(link, fake, link.make_heartbeat(now=0.40), 0.40)
        link.ingest(fake.tick(now=0.54), now=0.54)
        self.assertTrue(fake.locked)
        self.assertEqual(fake.lock_reason, LockReason.HARD_TIMEOUT)
        self.assertEqual(fake.pump_mask, 0)
        self.assertEqual(link.last_task_result_receipt.terminal_reason, "HARD_TIMEOUT")

    def test_hard_timeout_wins_exact_target_boundary(self) -> None:
        link = RootScopeSerialLink()
        fake = FakeF407(pump_rate_mg_s=1_000)
        exchange(link, fake, link.make_firmware_query(), 0.0)
        exchange(link, fake, link.make_heartbeat(now=0.01), 0.01)
        exchange(link, fake, link.make_clear_estop(now=0.02), 0.02)
        exchange(
            link,
            fake,
            link.make_arm_task(
                task_id=1,
                channel=1,
                target_mass_mg=500,
                hard_timeout_ms=500,
                config_hash_prefix=CONFIG_HASH,
                now=0.03,
            ),
            0.03,
        )
        exchange(link, fake, link.make_heartbeat(now=0.40), 0.40)
        link.ingest(fake.tick(now=0.53), now=0.53)
        self.assertTrue(fake.locked)
        self.assertEqual(fake.lock_reason, LockReason.HARD_TIMEOUT)
        self.assertEqual(fake.pump_mask, 0)
        self.assertEqual(link.last_task_result_receipt.terminal_reason, "HARD_TIMEOUT")

    def test_malformed_arm_ack_recovers_seq_from_correct_offset(self) -> None:
        link, fake = ready_pair()
        seq = link.next_seq()
        payload = ArmTask(1, seq, 1, 5_000, 5_000, CONFIG_HASH).to_payload()[:-1]
        malformed = encode_frame(CommandType.ARM_TASK, payload)
        exchange(link, fake, malformed, 0.03)
        self.assertEqual(link.last_ack.seq, seq)
        self.assertEqual(link.last_ack.status, AckStatus.BAD_PAYLOAD)
        self.assertEqual(link.last_ack.reason, AckReason.MALFORMED_PAYLOAD)
        self.assertEqual(fake.pump_mask, 0)


class SafetyFixtureTests(unittest.TestCase):
    def test_leak_input_immediately_stops_and_latches(self) -> None:
        link, fake = ready_pair()
        exchange(
            link,
            fake,
            link.make_arm_task(
                task_id=1,
                channel=3,
                target_mass_mg=10_000,
                hard_timeout_ms=5_000,
                config_hash_prefix=CONFIG_HASH,
                now=0.03,
            ),
            0.03,
        )
        fake.set_safety_inputs(leak_detected=True, now=0.04)
        link.ingest(fake.tick(now=0.05), now=0.05)
        self.assertEqual(fake.pump_mask, 0)
        self.assertTrue(fake.locked)
        self.assertEqual(fake.lock_reason, LockReason.UNSAFE_INPUT)
        self.assertEqual(link.last_task_result_receipt.terminal_reason, "SAFETY_INPUT")

    def test_invalid_hx711_immediately_stops_and_latches(self) -> None:
        link, fake = ready_pair()
        exchange(
            link,
            fake,
            link.make_arm_task(
                task_id=1,
                channel=2,
                target_mass_mg=10_000,
                hard_timeout_ms=5_000,
                config_hash_prefix=CONFIG_HASH,
                now=0.03,
            ),
            0.03,
        )
        fake.set_hx711(filtered_mass_mg=499_900, valid=False, now=0.04)
        link.ingest(fake.tick(now=0.05), now=0.05)
        self.assertEqual(fake.pump_mask, 0)
        self.assertTrue(fake.locked)
        self.assertEqual(link.last_task_result_receipt.terminal_reason, "SAFETY_INPUT")

    def test_emergency_stop_ack_and_terminal_result_share_task(self) -> None:
        link, fake = ready_pair()
        exchange(
            link,
            fake,
            link.make_arm_task(
                task_id=8,
                channel=2,
                target_mass_mg=10_000,
                hard_timeout_ms=5_000,
                config_hash_prefix=CONFIG_HASH,
                now=0.03,
            ),
            0.03,
        )
        stop_wire = link.make_emergency_stop(now=0.04)
        stop_seq = SeqCommand.from_payload(decode_frame(stop_wire).payload).seq
        exchange(link, fake, stop_wire, 0.04)
        self.assertEqual(link.last_ack.reason, AckReason.EMERGENCY_STOP)
        self.assertEqual(link.last_task_result_receipt.terminal_reason, "EMERGENCY_STOP")
        self.assertEqual(link.last_task_result_receipt.result.task_id, 8)
        ack_receipt = link.command_ack_receipt_for(
            CommandType.EMERGENCY_STOP, stop_seq
        )
        self.assertEqual(ack_receipt.command.transcript, stop_wire.hex())
        self.assertEqual(ack_receipt.execution_backend, "FAKE_F407")

    def test_clear_is_rejected_while_physical_input_is_unsafe(self) -> None:
        link = RootScopeSerialLink()
        fake = FakeF407()
        exchange(link, fake, link.make_firmware_query(), 0.0)
        exchange(link, fake, link.make_heartbeat(now=0.01), 0.01)
        fake.set_safety_inputs(cartridge_present=False, now=0.02)
        exchange(link, fake, link.make_clear_estop(now=0.03), 0.03)
        self.assertEqual(link.last_ack.status, AckStatus.LOCKED)
        self.assertEqual(link.last_ack.reason, AckReason.CLEAR_CONDITIONS_NOT_MET)
        self.assertTrue(fake.locked)


if __name__ == "__main__":
    unittest.main()
