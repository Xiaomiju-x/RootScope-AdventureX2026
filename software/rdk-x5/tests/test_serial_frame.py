from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import unittest


ROOTSCOPE_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOTSCOPE_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    # Import the serial subpackage directly so this focused suite remains
    # runnable while other H0-H12 app modules are being landed in parallel.
    sys.path.insert(0, str(APP_ROOT))

from serial.frame import (  # noqa: E402
    Ack,
    ArmTimedTask,
    ArmTask,
    ChecksumError,
    CommandType,
    EXPECTED_BUILD_ID,
    EXPECTED_HW_VARIANT,
    F103_PB6_BUILD_ID,
    F103_PB6_HW_VARIANT,
    F103_PB6_REQUIRED_CAPABILITIES,
    F103_Z3_PB6_BUILD_ID,
    F103_Z3_PB6_HW_VARIANT,
    F103_Z3_PB6_REQUIRED_CAPABILITIES,
    FirmwareInfo,
    Frame,
    FrameParser,
    Heartbeat,
    IrrigationTelemetry,
    LengthError,
    MAX_PAYLOAD_SIZE,
    PROTOCOL_VERSION,
    REQUIRED_CAPABILITIES,
    ResponseType,
    SafetyState,
    TaskResult,
    TerminalReason,
    TruncatedFrame,
    decode_frame,
    encode_frame,
)


class FrameCodecTests(unittest.TestCase):
    def test_round_trip_and_checksum_scope(self) -> None:
        wire = encode_frame(CommandType.HEARTBEAT, b"\x34\x12\x02")
        self.assertEqual(wire[:5], b"\xAA\x55\xFF\x03\x34")
        self.assertEqual(wire[-1], sum(wire[:-1]) & 0xFF)
        frame = decode_frame(wire)
        self.assertEqual(frame.message_type, int(CommandType.HEARTBEAT))
        self.assertEqual(frame.payload, b"\x34\x12\x02")

    def test_strict_decode_rejects_truncation(self) -> None:
        wire = encode_frame(CommandType.QUERY_FIRMWARE, b"\x01\x00")
        for cut in range(len(wire)):
            with self.assertRaises(TruncatedFrame):
                decode_frame(wire[:cut])

    def test_strict_decode_rejects_corruption(self) -> None:
        wire = bytearray(encode_frame(CommandType.HEARTBEAT, b"\x01\x00\x00"))
        wire[4] ^= 0x80
        with self.assertRaises(ChecksumError):
            decode_frame(wire)

    def test_stream_parser_buffers_partial_frame(self) -> None:
        parser = FrameParser()
        wire = encode_frame(CommandType.HEARTBEAT, b"\x01\x00\x00")
        self.assertEqual(parser.feed(wire[:3]), [])
        self.assertEqual(parser.feed(wire[3:6]), [])
        self.assertGreater(parser.buffered_bytes, 0)
        frames = parser.feed(wire[6:])
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].payload, b"\x01\x00\x00")
        self.assertEqual(parser.buffered_bytes, 0)

    def test_stream_parser_resynchronizes_after_noise_and_bad_frame(self) -> None:
        parser = FrameParser()
        bad = bytearray(encode_frame(CommandType.HEARTBEAT, b"\x01\x00\x00"))
        bad[-1] ^= 0x01
        good = encode_frame(CommandType.QUERY_FIRMWARE, b"\x02\x00")
        frames = parser.feed(b"noise" + bytes(bad) + good)
        self.assertEqual([f.message_type for f in frames], [CommandType.QUERY_FIRMWARE])
        self.assertEqual(parser.checksum_errors, 1)
        self.assertGreaterEqual(parser.discarded_bytes, len(b"noise") + 1)

    def test_parser_preserves_split_header(self) -> None:
        parser = FrameParser()
        wire = encode_frame(CommandType.QUERY_FIRMWARE, b"\x01\x00")
        self.assertEqual(parser.feed(b"junk\xAA"), [])
        frames = parser.feed(wire[1:])
        self.assertEqual(len(frames), 1)

    def test_fixed_payload_round_trips(self) -> None:
        arm = ArmTask(42, 65535, 3, 12_500, 9_000, bytes.fromhex("0123456789abcdef"))
        self.assertEqual(ArmTask.from_payload(arm.to_payload()), arm)
        self.assertEqual(len(arm.to_payload()), 23)

        timed = ArmTimedTask(
            43,
            1,
            1,
            2_500,
            3_000,
            bytes.fromhex("fedcba9876543210"),
        )
        self.assertEqual(ArmTimedTask.from_payload(timed.to_payload()), timed)
        self.assertEqual(len(timed.to_payload()), 23)

        info = FirmwareInfo(
            PROTOCOL_VERSION,
            REQUIRED_CAPABILITIES,
            EXPECTED_BUILD_ID,
            EXPECTED_HW_VARIANT,
            "rootscope-v1",
            0x123456789ABCDEF0,
        )
        self.assertEqual(FirmwareInfo.from_payload(info.to_payload()), info)
        self.assertEqual(len(info.to_payload()), 35)

        ack = Ack(CommandType.ARM_TASK, 9, 0, 0, 42)
        self.assertEqual(Ack.from_payload(ack.to_payload()), ack)
        self.assertEqual(len(ack.to_payload()), 9)

        result = TaskResult(
            boot_id=0x123456789ABCDEF0,
            task_id=42,
            result_seq=9,
            terminal_reason=TerminalReason.TARGET_REACHED,
            baseline_mass_mg=500_000,
            final_mass_mg=495_000,
            first_sample_seq=101,
            last_sample_seq=105,
            sample_count=5,
            final_window_min_mg=494_998,
            final_window_max_mg=495_002,
            scale_stable=True,
            firmware_completed_uptime_ms=1234,
            pump_mask=0,
            safety_bits=0x003C,
        )
        self.assertEqual(TaskResult.from_payload(result.to_payload()), result)
        self.assertEqual(len(result.to_payload()), 49)

    def test_decode_explicitly_rejects_impossible_length_bytes(self) -> None:
        for length in range(MAX_PAYLOAD_SIZE + 1, 256):
            with self.subTest(length=length):
                with self.assertRaises(LengthError):
                    decode_frame(b"\xAA\x55\x20" + bytes((length,)))

    def test_parser_resynchronizes_after_each_impossible_length(self) -> None:
        good = encode_frame(CommandType.QUERY_FIRMWARE, b"\x01\x00")
        for length in range(MAX_PAYLOAD_SIZE + 1, 256):
            with self.subTest(length=length):
                parser = FrameParser()
                frames = parser.feed(b"\xAA\x55\x20" + bytes((length,)) + good)
                self.assertEqual(len(frames), 1)
                self.assertEqual(frames[0].message_type, CommandType.QUERY_FIRMWARE)
                self.assertEqual(parser.length_errors, 1)

    def test_integer_encoders_reject_coercible_non_integers(self) -> None:
        bad_values = (True, False, 1.0, "1")
        for bad in bad_values:
            with self.subTest(frame_type=bad):
                with self.assertRaises((TypeError, ValueError)):
                    Frame(bad, b"")
            with self.subTest(seq=bad):
                with self.assertRaises((TypeError, ValueError)):
                    Heartbeat(bad, 0).to_payload()
            with self.subTest(telemetry_sample=bad):
                with self.assertRaises((TypeError, ValueError)):
                    IrrigationTelemetry(1, bad, 0, 1, 1, 0, 1).to_payload()
            with self.subTest(safety_boot=bad):
                with self.assertRaises((TypeError, ValueError)):
                    SafetyState(bad, 0, 0, 0, 0).to_payload()
        with self.assertRaises(TypeError):
            encode_frame(1, 3)
        with self.assertRaises(TypeError):
            ArmTask(1, 1, 1, 100, 500, "12345678").to_payload()

    def test_fuzz_like_stream_never_raises_on_arbitrary_bytes(self) -> None:
        rng = random.Random(20260715)
        parser = FrameParser()
        for _ in range(500):
            blob = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 40)))
            if rng.random() < 0.15:
                blob += encode_frame(CommandType.QUERY_FIRMWARE, b"\x01\x00")
            parser.feed(blob)
        parser.reset()
        self.assertEqual(
            len(parser.feed(encode_frame(CommandType.QUERY_FIRMWARE, b"\x01\x00"))),
            1,
        )


class IcdParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOTSCOPE_ROOT / "configs" / "serial_icd.json"
        cls.icd = json.loads(path.read_text(encoding="utf-8"))

    def test_identity_constants_match_icd(self) -> None:
        identity = self.icd["identity"]
        self.assertEqual(identity["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(identity["required_capabilities_mask"], REQUIRED_CAPABILITIES)
        self.assertEqual(identity["expected_build_id"], EXPECTED_BUILD_ID)
        self.assertEqual(identity["expected_hw_variant"], EXPECTED_HW_VARIANT)

        pump_only = self.icd["hardware_profiles"]["f103_pb6_pump_only_v13"]
        self.assertEqual(pump_only["expected_build_id"], F103_PB6_BUILD_ID)
        self.assertEqual(pump_only["hw_variant"], F103_PB6_HW_VARIANT)
        self.assertEqual(
            int(pump_only["required_capabilities_hex"], 16),
            F103_PB6_REQUIRED_CAPABILITIES,
        )

        z3 = self.icd["hardware_profiles"]["f103_z3_pb6_v15"]
        self.assertEqual(z3["expected_build_id"], F103_Z3_PB6_BUILD_ID)
        self.assertEqual(z3["hw_variant"], F103_Z3_PB6_HW_VARIANT)
        self.assertEqual(
            int(z3["required_capabilities_hex"], 16),
            F103_Z3_PB6_REQUIRED_CAPABILITIES,
        )

    def test_message_type_values_match_icd(self) -> None:
        for name, spec in self.icd["commands"].items():
            self.assertEqual(spec["type"], int(CommandType[name]))
        for name, spec in self.icd["responses"].items():
            self.assertEqual(spec["type"], int(ResponseType[name]))


if __name__ == "__main__":
    unittest.main()
