from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.vision.uvc_card_capture import (
    DISPOSITION,
    EXIT_CAPTURE_ERROR,
    EXIT_QUALITY_REJECTED,
    EXIT_USAGE,
    EXPECTED_CARD_LAYOUT,
    EXPECTED_PDF_RELATIVE_PATH,
    EXPECTED_PRINT_STATUS,
    REGISTERED_ROLE,
    UNKNOWN_ROLE,
    CaptureRequest,
    LiveUVCBackend,
    _atomic_write_bytes,
    capture_card,
    main,
)


class FakeBackend:
    def __init__(self, request: CaptureRequest, frames: list[np.ndarray]) -> None:
        self.request = request
        self.frames = list(frames)
        self.closed = False
        self.restore_called = False
        self.read_count = 0

    def negotiated_settings(self) -> dict[str, object]:
        return {
            "backend": "fixture",
            "fourcc": "MJPG",
            "width": self.request.width,
            "height": self.request.height,
            "fps": 30.0,
        }

    def snapshot_controls(self) -> dict[str, float]:
        return {
            "auto_exposure": 3.0,
            "exposure": 313.0,
            "auto_white_balance": 1.0,
            "white_balance_temperature": 4000.0,
        }

    def apply_controls(self, request: CaptureRequest) -> dict[str, object]:
        return {
            "policy": "FIXTURE",
            "before": self.snapshot_controls(),
            "operations": [],
            "effective": self.snapshot_controls(),
            "all_set_acknowledged": True,
            "all_set_confirmed": True,
            "persistence_requested": False,
        }

    def restore_controls(self) -> dict[str, object]:
        self.restore_called = True
        return {
            "required": False,
            "attempted": False,
            "all_restore_acknowledged": True,
            "all_restore_confirmed": True,
            "operations": [],
        }

    def read_rgb(self) -> np.ndarray:
        self.read_count += 1
        if not self.frames:
            raise RuntimeError("fixture depleted")
        return self.frames.pop(0)

    def close(self) -> dict[str, object]:
        self.closed = True
        return {
            "release_called": True,
            "opened_after_release": False,
            "release_error": None,
            "release_completed": True,
        }


class IneffectiveApplyBackend(FakeBackend):
    def apply_controls(self, request: CaptureRequest) -> dict[str, object]:
        result = super().apply_controls(request)
        result["operations"] = [
            {
                "name": "exposure",
                "requested": 100.0,
                "set_acknowledged": True,
                "effective_after_set": 313.0,
                "confirmed": False,
            }
        ]
        result["all_set_confirmed"] = False
        return result


class IneffectiveRestoreBackend(FakeBackend):
    def restore_controls(self) -> dict[str, object]:
        self.restore_called = True
        return {
            "required": True,
            "attempted": True,
            "all_restore_acknowledged": True,
            "all_restore_confirmed": False,
            "operations": [
                {
                    "name": "exposure",
                    "restore_requested": 313.0,
                    "set_acknowledged": True,
                    "effective_after_restore": 100.0,
                    "confirmed": False,
                }
            ],
        }


class IdentityVerifier:
    def __init__(self, identities: list[dict[str, object]] | None = None) -> None:
        default = {
            "device_path": (
                "/dev/v4l/by-id/"
                "usb-Web_Camera_Web_Camera_202604081837-video-index0"
            ),
            "resolved_device": "/dev/video0",
            "target_stat": {"major": 81, "minor": 0},
            "usb": {"vid_pid": "32e6:9228", "serial": "202604081837"},
        }
        self.identities = list(identities or [default, default, default])
        self.calls = 0

    def __call__(self, request: CaptureRequest) -> dict[str, object]:
        self.calls += 1
        if not self.identities:
            raise AssertionError("unexpected verifier call")
        return self.identities.pop(0)


def textured_frame(width: int, height: int) -> np.ndarray:
    y, x = np.indices((height, width))
    checker = (((x // 12 + y // 12) % 2) * 150 + 50).astype(np.uint8)
    return np.repeat(checker[:, :, None], 3, axis=2)


class UVCCardCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.print_manifest = self.root / "four_up.json"
        self.print_manifest_sha = self._write_print_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _manifest_payload(self) -> dict[str, object]:
        cards = []
        for index, (class_id, (position, role)) in enumerate(
            EXPECTED_CARD_LAYOUT.items()
        ):
            cards.append(
                {
                    "class_id": class_id,
                    "role": role,
                    "position": position,
                    "sha256": f"{index + 1:x}" * 64,
                    "holdout_claimed": False,
                    "accuracy_evidence": False,
                }
            )
        return {
            "schema": "rootscope.event-demo-four-up-print-sheet.v1",
            "status": EXPECTED_PRINT_STATUS,
            "pdf": {
                "path_relative_to_adventurex": EXPECTED_PDF_RELATIVE_PATH,
                "sha256": "a" * 64,
            },
            "cards": cards,
        }

    def _write_print_manifest(
        self, payload: dict[str, object] | None = None
    ) -> str:
        raw = json.dumps(
            payload or self._manifest_payload(),
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        self.print_manifest.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def _request(
        self,
        output_name: str,
        *,
        card_id: str = "grass_clump",
        role: str = REGISTERED_ROLE,
        expected_manifest_sha: str | None = None,
    ) -> CaptureRequest:
        return CaptureRequest(
            device_path=(
                "/dev/v4l/by-id/"
                "usb-Web_Camera_Web_Camera_202604081837-video-index0"
            ),
            expected_vid_pid="32e6:9228",
            expected_serial="202604081837",
            print_manifest=self.print_manifest,
            expected_print_manifest_sha256=(
                expected_manifest_sha or self.print_manifest_sha
            ),
            card_id=card_id,
            class_role=role,
            output_root=self.root,
            output_dir=self.root / output_name,
            width=1280,
            height=720,
            warmup_frames=1,
            frame_count=2,
            interval_seconds=0.05,
        )

    def _capture(
        self,
        request: CaptureRequest,
        backend: FakeBackend,
        verifier: IdentityVerifier | None = None,
    ):
        verifier = verifier or IdentityVerifier()
        outcome = capture_card(
            request,
            backend_factory=lambda _: backend,
            device_verifier=verifier,
            sleep_fn=lambda _: None,
        )
        return outcome, verifier

    def test_good_fixture_is_event_only_and_releases_backend(self) -> None:
        request = self._request("accepted")
        frame = textured_frame(request.width, request.height)
        backend = FakeBackend(request, [frame, frame, frame])

        outcome, verifier = self._capture(request, backend)

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(verifier.calls, 3)
        self.assertTrue(backend.closed)
        self.assertTrue(backend.restore_called)
        manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["disposition"], DISPOSITION)
        self.assertTrue(manifest["device_identity"]["identity_unchanged"])
        self.assertTrue(
            manifest["device_identity"]["identity_unchanged_across_lifecycle"]
        )
        self.assertTrue(manifest["controls"]["close"]["release_completed"])
        self.assertFalse(
            manifest["controls"]["close"]["opened_after_release"]
        )
        self.assertTrue(manifest["quality_gate"]["accepted"])
        self.assertEqual(len(manifest["frames"]), 2)
        self.assertFalse(manifest["truth_boundary"]["auto_train"])
        self.assertFalse(manifest["truth_boundary"]["registry_mutated"])
        self.assertFalse(manifest["truth_boundary"]["physical_or_irrigation_authority"])
        self.assertEqual(list(request.output_dir.glob("*.tmp")), [])
        for frame_receipt in manifest["frames"]:
            self.assertEqual(len(frame_receipt["sha256"]), 64)
            self.assertTrue((request.output_dir / frame_receipt["file"]).is_file())

    def test_flat_frames_are_saved_but_quality_has_distinct_exit(self) -> None:
        request = self._request("rejected")
        flat = np.full((request.height, request.width, 3), 128, dtype=np.uint8)
        backend = FakeBackend(request, [flat, flat, flat])

        outcome, _ = self._capture(request, backend)

        self.assertEqual(outcome.exit_code, EXIT_QUALITY_REJECTED)
        self.assertNotEqual(outcome.exit_code, EXIT_USAGE)
        manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["status"],
            "EVENT_OPTICAL_CAPTURE_REJECTED_NOT_AUTO_TRAIN",
        )
        self.assertFalse(manifest["quality_gate"]["all_frames_passed"])
        self.assertIn("LOW_CONTRAST", manifest["frames"][0]["quality"]["reasons"])
        self.assertTrue(backend.closed)

    def test_capture_error_still_releases_and_writes_manifest(self) -> None:
        request = self._request("capture_error")
        backend = FakeBackend(request, [])

        outcome, _ = self._capture(request, backend)

        self.assertEqual(outcome.exit_code, EXIT_CAPTURE_ERROR)
        self.assertTrue(backend.closed)
        manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
        self.assertIn("fixture depleted", manifest["error"])
        self.assertFalse(manifest["quality_gate"]["accepted"])

    def test_existing_output_is_never_overwritten(self) -> None:
        request = self._request("existing")
        request.output_dir.mkdir()
        marker = request.output_dir / "keep.txt"
        marker.write_text("preserve", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "overwrite refused"):
            capture_card(request)

        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_unknown_can_only_use_unregistered_negative_role(self) -> None:
        invalid = self._request(
            "bad_unknown",
            card_id="unknown",
            role=REGISTERED_ROLE,
        )
        with self.assertRaisesRegex(ValueError, "role mismatch"):
            capture_card(invalid)

        valid = self._request(
            "valid_unknown",
            card_id="unknown",
            role=UNKNOWN_ROLE,
        )
        frame = textured_frame(valid.width, valid.height)
        backend = FakeBackend(valid, [frame, frame, frame])
        outcome, _ = self._capture(valid, backend)
        manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["truth_boundary"]["unknown_registration_allowed"])
        self.assertTrue(manifest["truth_boundary"]["unknown_is_unregistered_negative"])

    def test_unsupported_resolution_fails_before_device_open(self) -> None:
        request = replace(
            self._request("bad_resolution"),
            width=640,
            height=480,
        )
        opened = False

        def factory(_: CaptureRequest) -> FakeBackend:
            nonlocal opened
            opened = True
            raise AssertionError("must not open")

        with self.assertRaisesRegex(ValueError, "resolution"):
            capture_card(request, backend_factory=factory)
        self.assertFalse(opened)

    def test_numeric_and_parent_traversal_device_paths_are_refused(self) -> None:
        numeric = replace(self._request("numeric_alias"), device_path="/dev/video0")
        with self.assertRaisesRegex(ValueError, "direct-child symlink"):
            capture_card(numeric)

        traversal = replace(
            self._request("traversal"),
            device_path="/dev/v4l/by-id/../by-id/camera",
        )
        with self.assertRaisesRegex(ValueError, "must not contain"):
            capture_card(traversal)

    def test_output_dir_must_be_direct_child_of_explicit_root(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        request = replace(
            self._request("unused"),
            output_dir=nested / "take01",
        )
        with self.assertRaisesRegex(ValueError, "direct child"):
            capture_card(request)

    def test_print_manifest_sha_mismatch_fails_before_device_open(self) -> None:
        request = self._request(
            "bad_manifest_hash",
            expected_manifest_sha="0" * 64,
        )
        opened = False

        def factory(_: CaptureRequest) -> FakeBackend:
            nonlocal opened
            opened = True
            raise AssertionError("must not open")

        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            capture_card(request, backend_factory=factory)
        self.assertFalse(opened)

    def test_print_manifest_requires_exact_four_card_layout(self) -> None:
        payload = self._manifest_payload()
        payload["cards"][0]["position"] = "BOTTOM_RIGHT"  # type: ignore[index]
        mutated_sha = self._write_print_manifest(payload)
        request = self._request(
            "bad_layout",
            expected_manifest_sha=mutated_sha,
        )
        with self.assertRaisesRegex(ValueError, "layout/role mismatch"):
            capture_card(request)

    def test_silent_ineffective_control_set_fails_closed(self) -> None:
        request = self._request("ineffective_set")
        frame = textured_frame(request.width, request.height)
        backend = IneffectiveApplyBackend(request, [frame, frame, frame])

        outcome, _ = self._capture(request, backend)

        self.assertEqual(outcome.exit_code, EXIT_CAPTURE_ERROR)
        self.assertTrue(backend.closed)
        manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
        self.assertIn("not confirmed by readback", manifest["error"])

    def test_silent_ineffective_restore_fails_closed(self) -> None:
        request = self._request("ineffective_restore")
        frame = textured_frame(request.width, request.height)
        backend = IneffectiveRestoreBackend(request, [frame, frame, frame])

        outcome, _ = self._capture(request, backend)

        self.assertEqual(outcome.exit_code, EXIT_CAPTURE_ERROR)
        manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(
            manifest["controls"]["restore"]["all_restore_confirmed"]
        )

    def test_identity_change_after_open_releases_and_fails_closed(self) -> None:
        request = self._request("identity_changed")
        frame = textured_frame(request.width, request.height)
        backend = FakeBackend(request, [frame, frame, frame])
        first = {
            "resolved_device": "/dev/video0",
            "target_stat": {"major": 81, "minor": 0},
            "usb": {"vid_pid": "32e6:9228", "serial": "202604081837"},
        }
        second = {
            "resolved_device": "/dev/video1",
            "target_stat": {"major": 81, "minor": 1},
            "usb": {"vid_pid": "32e6:9228", "serial": "202604081837"},
        }
        verifier = IdentityVerifier([first, second, second])

        outcome, _ = self._capture(request, backend, verifier)

        self.assertEqual(outcome.exit_code, EXIT_CAPTURE_ERROR)
        self.assertTrue(backend.closed)
        manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["device_identity"]["identity_unchanged"])
        self.assertIn("identity changed", manifest["error"])

    def test_identity_change_after_close_releases_and_fails_closed(self) -> None:
        request = self._request("identity_changed_after_close")
        frame = textured_frame(request.width, request.height)
        backend = FakeBackend(request, [frame, frame, frame])
        stable = {
            "device_path": request.device_path,
            "resolved_device": "/dev/video0",
            "target_stat": {"major": 81, "minor": 0},
            "usb": {"vid_pid": "32e6:9228", "serial": "202604081837"},
        }
        changed_after_close = {
            **stable,
            "resolved_device": "/dev/video1",
            "target_stat": {"major": 81, "minor": 1},
        }
        verifier = IdentityVerifier([stable, stable, changed_after_close])

        outcome, checked = self._capture(request, backend, verifier)

        self.assertEqual(outcome.exit_code, EXIT_CAPTURE_ERROR)
        self.assertEqual(checked.calls, 3)
        self.assertTrue(backend.closed)
        manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
        identity = manifest["device_identity"]
        self.assertTrue(identity["identity_unchanged_before_and_after_open"])
        self.assertFalse(identity["identity_unchanged_after_close"])
        self.assertFalse(identity["identity_unchanged_across_lifecycle"])
        self.assertEqual(
            identity["after_close"]["resolved_device"], "/dev/video1"
        )
        self.assertIn("identity changed after camera close", manifest["error"])

    def test_live_backend_releases_if_set_raises_after_open(self) -> None:
        request = self._request("never_created")

        class ExplodingCapture:
            def __init__(self) -> None:
                self.released = False

            def isOpened(self) -> bool:
                return True

            def set(self, property_id: int, value: float) -> bool:
                raise RuntimeError("fixture set failed")

            def release(self) -> None:
                self.released = True

        capture = ExplodingCapture()

        class FakeCV2:
            CAP_V4L2 = 200
            CAP_PROP_FOURCC = 1
            CAP_PROP_FRAME_WIDTH = 2
            CAP_PROP_FRAME_HEIGHT = 3
            CAP_PROP_FPS = 4

            @staticmethod
            def VideoCapture(device: str, backend: int) -> ExplodingCapture:
                return capture

            @staticmethod
            def VideoWriter_fourcc(*letters: str) -> int:
                return 0

        with self.assertRaisesRegex(RuntimeError, "fixture set failed"):
            LiveUVCBackend(request, cv2_module=FakeCV2)
        self.assertTrue(capture.released)

    def test_live_backend_releases_on_baseexception_during_construction(self) -> None:
        request = self._request("never_created_baseexception")

        class InterruptingCapture:
            def __init__(self) -> None:
                self.released = False

            def isOpened(self) -> bool:
                raise KeyboardInterrupt("fixture interrupt")

            def release(self) -> None:
                self.released = True

        capture = InterruptingCapture()

        class FakeCV2:
            CAP_V4L2 = 200

            @staticmethod
            def VideoCapture(device: str, backend: int) -> InterruptingCapture:
                return capture

        with self.assertRaises(KeyboardInterrupt):
            LiveUVCBackend(request, cv2_module=FakeCV2)
        self.assertTrue(capture.released)

    def test_silent_ineffective_live_release_is_runtime_failure(self) -> None:
        request = self._request("silent_release")
        frame = textured_frame(request.width, request.height)

        class SilentReleaseBackend(FakeBackend):
            def close(self) -> dict[str, object]:
                self.closed = True
                return {
                    "release_called": True,
                    "opened_after_release": True,
                    "release_error": None,
                    "release_completed": False,
                }

        backend = SilentReleaseBackend(request, [frame, frame, frame])
        outcome, verifier = self._capture(request, backend)

        self.assertEqual(outcome.exit_code, EXIT_CAPTURE_ERROR)
        self.assertEqual(verifier.calls, 3)
        manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
        close = manifest["controls"]["close"]
        self.assertTrue(close["release_called"])
        self.assertTrue(close["opened_after_release"])
        self.assertFalse(close["release_completed"])
        self.assertFalse(close["contract_satisfied"])
        self.assertIn(
            "CAMERA_RELEASE_NOT_CONFIRMED_ISOPENED_FALSE", manifest["error"]
        )

    def test_live_close_proves_isopened_false_and_caches_receipt(self) -> None:
        class ClosingCapture:
            def __init__(self) -> None:
                self.opened = True
                self.release_calls = 0

            def release(self) -> None:
                self.release_calls += 1
                self.opened = False

            def isOpened(self) -> bool:
                return self.opened

        capture = ClosingCapture()
        backend = LiveUVCBackend.__new__(LiveUVCBackend)
        backend._capture = capture
        backend._closed = False
        backend._close_receipt = None

        first = backend.close()
        second = backend.close()

        self.assertEqual(capture.release_calls, 1)
        self.assertEqual(first, second)
        self.assertTrue(first["release_called"])
        self.assertFalse(first["opened_after_release"])
        self.assertIsNone(first["release_error"])
        self.assertTrue(first["release_completed"])

    def test_live_close_detects_silent_ineffective_release(self) -> None:
        class SilentCapture:
            def release(self) -> None:
                return None

            def isOpened(self) -> bool:
                return True

        backend = LiveUVCBackend.__new__(LiveUVCBackend)
        backend._capture = SilentCapture()
        backend._closed = False
        backend._close_receipt = None

        receipt = backend.close()

        self.assertTrue(receipt["release_called"])
        self.assertTrue(receipt["opened_after_release"])
        self.assertIsNone(receipt["release_error"])
        self.assertFalse(receipt["release_completed"])

    def test_live_control_readback_detects_silent_ineffective_changes(self) -> None:
        class FakeControlCV2:
            CAP_PROP_AUTO_EXPOSURE = 10
            CAP_PROP_EXPOSURE = 11
            CAP_PROP_AUTO_WB = 12
            CAP_PROP_WB_TEMPERATURE = 13

        class SilentCapture:
            def __init__(self, values: dict[int, float]) -> None:
                self.values = values

            def get(self, property_id: int) -> float:
                return self.values[property_id]

            def set(self, property_id: int, value: float) -> bool:
                return True

        request = replace(
            self._request("not_created"),
            exposure_mode="manual",
            exposure_value=100.0,
        )
        backend = LiveUVCBackend.__new__(LiveUVCBackend)
        backend._cv2 = FakeControlCV2
        backend._capture = SilentCapture(
            {
                FakeControlCV2.CAP_PROP_AUTO_EXPOSURE: 3.0,
                FakeControlCV2.CAP_PROP_EXPOSURE: 313.0,
                FakeControlCV2.CAP_PROP_AUTO_WB: 1.0,
                FakeControlCV2.CAP_PROP_WB_TEMPERATURE: 4000.0,
            }
        )
        backend._before_controls = None
        backend._touched_controls = []
        applied = backend.apply_controls(request)
        self.assertTrue(applied["all_set_acknowledged"])
        self.assertFalse(applied["all_set_confirmed"])

        restore_backend = LiveUVCBackend.__new__(LiveUVCBackend)
        restore_backend._cv2 = FakeControlCV2
        restore_backend._capture = SilentCapture(
            {
                FakeControlCV2.CAP_PROP_AUTO_EXPOSURE: 3.0,
                FakeControlCV2.CAP_PROP_EXPOSURE: 100.0,
                FakeControlCV2.CAP_PROP_AUTO_WB: 1.0,
                FakeControlCV2.CAP_PROP_WB_TEMPERATURE: 4000.0,
            }
        )
        restore_backend._before_controls = {
            "auto_exposure": 3.0,
            "exposure": 313.0,
            "auto_white_balance": 1.0,
            "white_balance_temperature": 4000.0,
        }
        restore_backend._touched_controls = ["exposure"]
        restored = restore_backend.restore_controls()
        self.assertTrue(restored["all_restore_acknowledged"])
        self.assertFalse(restored["all_restore_confirmed"])

    def test_validation_main_exit_is_distinct_from_quality_exit(self) -> None:
        request = self._request("not_created")
        exit_code = main(
            [
                "--device",
                request.device_path,
                "--expected-vid-pid",
                request.expected_vid_pid,
                "--expected-serial",
                request.expected_serial,
                "--print-manifest",
                str(request.print_manifest),
                "--expected-print-manifest-sha256",
                request.expected_print_manifest_sha256,
                "--card-id",
                "unknown",
                "--class-role",
                REGISTERED_ROLE,
                "--output-root",
                str(request.output_root),
                "--output-dir",
                str(request.output_dir),
                "--resolution",
                "1280x720",
            ]
        )
        self.assertEqual(exit_code, EXIT_USAGE)
        self.assertNotEqual(exit_code, EXIT_QUALITY_REJECTED)

    def test_atomic_publish_never_replaces_a_racing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "capture_manifest.json"
            real_link = __import__("os").link

            def racing_link(source, destination, *args, **kwargs):
                target.write_bytes(b"racing-owner")
                return real_link(source, destination, *args, **kwargs)

            with mock.patch(
                "app.vision.uvc_card_capture.os.link",
                side_effect=racing_link,
            ):
                with self.assertRaises(FileExistsError):
                    _atomic_write_bytes(target, b"new-content")
            self.assertEqual(target.read_bytes(), b"racing-owner")


if __name__ == "__main__":
    unittest.main()
