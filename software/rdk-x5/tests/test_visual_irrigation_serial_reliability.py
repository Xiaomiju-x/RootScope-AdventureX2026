from __future__ import annotations

from pathlib import Path
import unittest


ROOTSCOPE_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOTSCOPE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROOTSCOPE_ROOT))

from tools.x5_visual_irrigation_cycle import (  # noqa: E402
    CMD_ARM_TIMED_TASK,
    CMD_HEARTBEAT,
    query_readonly_ascii_with_retry,
    send_ack_checked,
)


class FakeLedger:
    def __init__(self) -> None:
        self.value = 100

    def reserve_next(self) -> int:
        self.value += 1
        return self.value


class ReadonlyAsciiRetryTests(unittest.TestCase):
    def test_lost_first_version_reply_is_retried(self) -> None:
        class Session:
            calls = 0

            def query_ascii(self, command, prefix):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("VERSION, reply not received")
                return "VERSION,2026-07-25-RS-F103-Z3-PB6-V15"

        session = Session()
        reply = query_readonly_ascii_with_retry(
            session, "VERSION", "VERSION,"
        )
        self.assertTrue(reply.startswith("VERSION,"))
        self.assertEqual(session.calls, 2)

    def test_actuating_ascii_command_cannot_use_retry_helper(self) -> None:
        with self.assertRaises(ValueError):
            query_readonly_ascii_with_retry(
                object(), "DEPTH,1", "ACK,DEPTH,1,"
            )


class NonActuatingAckRetryTests(unittest.TestCase):
    def test_lost_first_heartbeat_ack_uses_a_new_sequence(self) -> None:
        class Session:
            def __init__(self) -> None:
                self.writes = []
                self.waits = []

            def write_once(self, payload):
                self.writes.append(payload)

            def wait_ack(self, command_type, sequence, timeout_s):
                self.waits.append((command_type, sequence, timeout_s))
                if len(self.waits) == 1:
                    raise RuntimeError(
                        f"ACK not received: type=0x{command_type:02X}, "
                        f"seq={sequence}"
                    )
                return 0, 0, 0

        session = Session()
        ledger = FakeLedger()
        sequence = send_ack_checked(
            session,
            ledger,
            CMD_HEARTBEAT,
            lambda seq: seq.to_bytes(2, "little") + b"\0",
        )
        self.assertEqual(sequence, 102)
        self.assertEqual([item[1] for item in session.waits], [101, 102])
        self.assertEqual(len(session.writes), 2)
        self.assertTrue(all(item[2] == 0.35 for item in session.waits))

    def test_physical_command_cannot_use_retry_helper(self) -> None:
        with self.assertRaises(ValueError):
            send_ack_checked(
                object(),
                FakeLedger(),
                CMD_ARM_TIMED_TASK,
                lambda seq: b"",
            )


if __name__ == "__main__":
    unittest.main()
