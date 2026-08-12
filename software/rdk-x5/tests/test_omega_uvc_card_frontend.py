from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.edge.capsule import CapsuleConfig, ROOTSCOPE_CLASS_ORDER
from app.omega_vision.ood import Calibration
from app.omega_vision.uvc_card_frontend import (
    FINAL_ACCEPT,
    FINAL_REJECT,
    ZERO_AUTHORITY,
    CalibrationBinding,
    ExpectedCameraIdentity,
    FrontendRequest,
    LiveUvcFrameSource,
    PrintManifestBinding,
    SafeOutputDirectory,
    UvcFrontendError,
    FROZEN_THRESHOLDS_SHA256,
    _bind_frozen_file,
    bind_print_manifest_to_registry,
    evaluate_rgb_frame,
    load_calibration_binding,
    load_print_manifest_binding,
    run_bounded_frontend,
    validate_request,
    validate_stable_device_syntax,
)
from app.vision.card_geometric_matcher import (
    CLAIM_SCOPE as GEOMETRIC_CLAIM_SCOPE,
    SCHEMA_VERSION as GEOMETRIC_SCHEMA_VERSION,
    MatcherConfig,
)
from app.vision.dual_path_demo import (
    REGISTERED_ROLE,
    REGISTRY_FROZEN_STATUS,
    REGISTRY_SCHEMA_VERSION,
    SEED17_MODEL_SHA256,
    DemoThresholds,
)


ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / "deploy" / "x5" / "capsule_config.seed17_cpu_experimental.json"
CALIBRATION_MANIFEST = (
    ROOT / "configs" / "omega" / "vision_board_replay_new_x5_20260723.json"
)
PRINT_MANIFEST = (
    ROOT.parent
    / "output"
    / "pdf"
    / "RootScope_A4_four_up_field_cards_20260723_manifest.json"
)
REGISTRY = (
    ROOT / "app" / "vision" / "known_card_template_registry.frozen.experimental.json"
)
STABLE_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-Web_Camera_Web_Camera_202604081837-video-index0"
)
EXPECTED_CAMERA = ExpectedCameraIdentity(
    usb_vid="1a2b",
    usb_pid="3c4d",
    usb_serial="fixture-camera-serial",
)


class _FakeSession:
    def __init__(self, logits: list[float]) -> None:
        self.logits = np.asarray([logits], dtype=np.float32)

    def run(self, output_names, feeds):
        if output_names != ["logits"] or set(feeds) != {"image"}:
            raise AssertionError("unexpected fake ONNX call")
        return [self.logits.copy()]


class _FakeRunner:
    def __init__(self, logits: list[float]) -> None:
        model = CapsuleConfig.from_json_file(CAPSULE).model
        self.model_sha256 = SEED17_MODEL_SHA256
        self.class_order = ROOTSCOPE_CLASS_ORDER
        self.expected_output_shape = (1, 4)
        self.providers = ["CPUExecutionProvider"]
        self.preprocess = model.preprocess
        self.input_name = "image"
        self.output_name = "logits"
        self._session = _FakeSession(logits)


def _textured_rgb(width: int = 640, height: int = 480) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.uint16)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (40 + x * 3 + y * 2) % 256
    rgb[..., 1] = (80 + x * 2 + y * 5) % 256
    rgb[..., 2] = (120 + x * 7 + y * 3) % 256
    return rgb


def _write_image(path: Path, rgb: np.ndarray) -> str:
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_registry(root: Path, class_name: str = "young_tree") -> tuple[Path, str]:
    templates = root / "templates"
    templates.mkdir()
    template = templates / "registered.png"
    template_sha = _write_image(template, _textured_rgb())
    payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "status": REGISTRY_FROZEN_STATUS,
        "template_root": "templates",
        "templates": [
            {
                "template_id": "fixture-young-tree",
                "class_name": class_name,
                "relative_path": template.name,
                "raw_sha256": template_sha,
                "role": REGISTERED_ROLE,
                "dataset_record": {
                    "record_id": "fixture-young-tree",
                    "source_manifest": "datasets/fixture/manifest.jsonl",
                    "source_url": "https://example.invalid/fixture",
                    "attribution": {
                        "creator": "fixture",
                        "license": "fixture-only",
                        "license_url": "https://example.invalid/license",
                    },
                },
            }
        ],
    }
    registry = root / "registry.json"
    registry.write_text(json.dumps(payload), encoding="utf-8")
    return registry, template_sha


def _matcher(pass_geometry: bool):
    def matcher(
        template_path,
        query_path,
        *,
        template_id,
        template_class,
        config,
    ):
        del config
        return {
            "schema": GEOMETRIC_SCHEMA_VERSION,
            "status": "PASS" if pass_geometry else "REJECT",
            "passed": pass_geometry,
            "claim_scope": GEOMETRIC_CLAIM_SCOPE,
            "irrigation_execution_authority": False,
            "template_sha256": hashlib.sha256(
                Path(template_path).read_bytes()
            ).hexdigest(),
            "query_sha256": hashlib.sha256(Path(query_path).read_bytes()).hexdigest(),
            "template_id": template_id,
            "template_class": template_class,
            "authority": {
                "irrigation_execution": False,
                "pump_command": False,
                "serial_write": False,
                "state_machine_write": False,
            },
            "provenance": {
                "semantic_recognition_performed": False,
                "physical_hardware_touched": False,
            },
        }

    return matcher


def _calibration_binding(*, strict_quality: bool = False) -> CalibrationBinding:
    calibration = Calibration(
        class_order=tuple(ROOTSCOPE_CLASS_ORDER),
        alpha=0.2,
        temperature=1.0,
        energy_upper=100.0,
        maxprob_lower=0.0,
        brightness_lower=0.1 if strict_quality else 0.0,
        brightness_upper=1.0,
        contrast_lower=0.01 if strict_quality else 0.0,
        sharpness_lower=0.001 if strict_quality else 0.0,
        clipped_upper=1.0,
        conformal_nonconformity=(0.2,) * 9,
    )
    return CalibrationBinding(
        calibration=calibration,
        manifest_path=CALIBRATION_MANIFEST,
        manifest_sha256="a" * 64,
        provenance={
            "formal_distribution_free_coverage_guarantee": False,
            "holdout_reevaluated_for_board_replay": False,
        },
    )


def _assets(registry: Path) -> dict:
    return {
        "seed17_onnx": {
            "sha256": SEED17_MODEL_SHA256,
            "provider": "CPUExecutionProvider",
        },
        "registered_template_registry": {
            "path": str(registry.resolve()),
            "sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        },
        "four_up_print_manifest": {
            "sha256": hashlib.sha256(PRINT_MANIFEST.read_bytes()).hexdigest(),
            "card_count": 4,
        },
        "plant_bpu": {"selected_bin": None, "used": False},
    }


class _FakeSource:
    configured_device = STABLE_DEVICE
    resolved_device = "/dev/video-fixture"

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = list(frames)
        self.closed = False

    def negotiated_settings(self):
        identity = {
            "configured_device": self.configured_device,
            "resolved_device": self.resolved_device,
            "kernel_video_node": "video-fixture",
            "usb_vid": EXPECTED_CAMERA.usb_vid,
            "usb_pid": EXPECTED_CAMERA.usb_pid,
            "usb_serial": EXPECTED_CAMERA.usb_serial,
            "usb_device_sysfs": "/sys/fixture/usb-camera",
            "identity_match": True,
            "device_discovery_used": False,
        }
        return {
            "backend": "fixture_no_camera",
            "configured_device": self.configured_device,
            "resolved_device": self.resolved_device,
            "fourcc": "FIXT",
            "width": 640,
            "height": 480,
            "fps": 30.0,
            "expected_identity": EXPECTED_CAMERA.to_dict(),
            "identity_before_open": dict(identity),
            "identity_after_open": dict(identity),
            "identity_match_before_and_after_open": True,
            "device_discovery_used": False,
        }

    def read_rgb(self) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("fixture frame source depleted")
        return self.frames.pop(0)

    def close(self):
        self.closed = True
        return {
            "release_completed": True,
            "identity_after_close": {
                "configured_device": self.configured_device,
                "resolved_device": self.resolved_device,
                "kernel_video_node": "video-fixture",
                "usb_vid": EXPECTED_CAMERA.usb_vid,
                "usb_pid": EXPECTED_CAMERA.usb_pid,
                "usb_serial": EXPECTED_CAMERA.usb_serial,
                "usb_device_sysfs": "/sys/fixture/usb-camera",
                "identity_match": True,
                "device_discovery_used": False,
            },
            "identity_match_after_close": True,
            "identity_error": None,
            "device_discovery_used": False,
        }


class UvcCardFrontendFrameTests(unittest.TestCase):
    def test_good_fixture_exposes_four_separate_layers_and_cpu_only_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, _ = _write_registry(Path(directory))
            result = evaluate_rgb_frame(
                rgb=_textured_rgb(),
                frame_index=1,
                runner=_FakeRunner([0.0, -1.0, 9.0, -2.0]),
                calibration_binding=_calibration_binding(),
                registry_path=registry,
                thresholds=DemoThresholds(),
                matcher_config=MatcherConfig(),
                asset_binding=_assets(registry),
                matcher=_matcher(True),
            )

        self.assertEqual(result["semantic_hypothesis"]["raw_top1_class"], "young_tree")
        self.assertEqual(result["omega_ood_abstention"]["decision"], "CLASSIFY")
        self.assertEqual(
            result["registered_card_geometry"]["contract_valid_pass_count"], 1
        )
        self.assertEqual(result["final_consensus"]["status"], FINAL_ACCEPT)
        self.assertTrue(result["final_consensus"]["passed"])
        self.assertEqual(result["final_consensus"]["display_class"], "young_tree")
        self.assertIsNone(result["compute_boundary"]["plant_bpu_selected_bin"])
        self.assertFalse(result["compute_boundary"]["plant_bpu_used"])
        self.assertEqual(result["authority"], ZERO_AUTHORITY)
        self.assertFalse(any(result["authority"].values()))

    def test_bad_quality_forces_omega_abstain_despite_semantic_and_geometry(self) -> None:
        flat_dark = np.zeros((480, 640, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            registry, _ = _write_registry(Path(directory))
            result = evaluate_rgb_frame(
                rgb=flat_dark,
                frame_index=1,
                runner=_FakeRunner([0.0, -1.0, 9.0, -2.0]),
                calibration_binding=_calibration_binding(strict_quality=True),
                registry_path=registry,
                thresholds=DemoThresholds(),
                matcher_config=MatcherConfig(),
                asset_binding=_assets(registry),
                matcher=_matcher(True),
            )

        self.assertEqual(result["omega_ood_abstention"]["decision"], "ABSTAIN")
        self.assertEqual(result["final_consensus"]["status"], FINAL_REJECT)
        self.assertFalse(result["final_consensus"]["passed"])
        self.assertIn(
            "OMEGA_OOD_ABSTAIN", result["final_consensus"]["reject_reasons"]
        )

    def test_unknown_semantics_always_abstains_and_never_displays_a_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, _ = _write_registry(Path(directory))
            result = evaluate_rgb_frame(
                rgb=_textured_rgb(),
                frame_index=1,
                runner=_FakeRunner([0.0, -1.0, -2.0, 9.0]),
                calibration_binding=_calibration_binding(),
                registry_path=registry,
                thresholds=DemoThresholds(),
                matcher_config=MatcherConfig(),
                asset_binding=_assets(registry),
                matcher=_matcher(False),
            )

        self.assertEqual(result["semantic_hypothesis"]["raw_top1_class"], "unknown")
        self.assertEqual(result["omega_ood_abstention"]["decision"], "ABSTAIN")
        self.assertIsNone(result["final_consensus"]["display_class"])
        self.assertIn(
            "UNKNOWN_CLASS_FAIL_CLOSED",
            result["final_consensus"]["reject_reasons"],
        )

    def test_missing_geometry_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, _ = _write_registry(Path(directory))
            result = evaluate_rgb_frame(
                rgb=_textured_rgb(),
                frame_index=1,
                runner=_FakeRunner([0.0, -1.0, 9.0, -2.0]),
                calibration_binding=_calibration_binding(),
                registry_path=registry,
                thresholds=DemoThresholds(),
                matcher_config=MatcherConfig(),
                asset_binding=_assets(registry),
                matcher=_matcher(False),
            )
        self.assertFalse(result["final_consensus"]["passed"])
        self.assertIn(
            "GEOMETRY_NOT_EXACTLY_ONE_REGISTERED_PASS",
            result["final_consensus"]["reject_reasons"],
        )


class UvcCardFrontendSessionTests(unittest.TestCase):
    def _request(
        self,
        root: Path,
        *,
        mode: str = "bounded",
        frames: int = 2,
        annotated: bool = True,
    ) -> FrontendRequest:
        return FrontendRequest(
            device=STABLE_DEVICE,
            expected_camera=EXPECTED_CAMERA,
            print_manifest=PRINT_MANIFEST.resolve(),
            mode=mode,
            frames=frames,
            warmup_frames=1,
            interval_seconds=0.0,
            width=640,
            height=480,
            fps=30.0,
            output_root=root.resolve(),
            jsonl_path=root / "session.jsonl",
            annotated_dir=root / "annotated" if annotated else None,
        )

    def test_bounded_fixture_writes_hash_chain_annotations_and_releases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, _ = _write_registry(root)
            frame = _textured_rgb()
            source = _FakeSource([frame, frame, frame])

            def processor(rgb: np.ndarray, index: int):
                return evaluate_rgb_frame(
                    rgb=rgb,
                    frame_index=index,
                    runner=_FakeRunner([0.0, -1.0, 9.0, -2.0]),
                    calibration_binding=_calibration_binding(),
                    registry_path=registry,
                    thresholds=DemoThresholds(),
                    matcher_config=MatcherConfig(),
                    asset_binding=_assets(registry),
                    matcher=_matcher(True),
                )

            request = self._request(root)
            outcome = run_bounded_frontend(
                request,
                print_binding=load_print_manifest_binding(PRINT_MANIFEST),
                frame_processor=processor,
                source_factory=lambda _: source,
                sleep_fn=lambda _: None,
            )
            lines = [
                json.loads(line)
                for line in request.jsonl_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(outcome.exit_code, 0)
            self.assertTrue(outcome.camera_released)
            self.assertTrue(source.closed)
            self.assertEqual(
                [line["event"] for line in lines],
                [
                    "session_start",
                    "camera_opened",
                    "frame_result",
                    "frame_result",
                    "camera_closed",
                    "session_end",
                ],
            )
            self.assertIsNone(lines[0]["previous_record_sha256"])
            for index in range(1, len(lines)):
                self.assertEqual(
                    lines[index]["previous_record_sha256"],
                    lines[index - 1]["record_sha256"],
                )
            self.assertTrue(lines[-1]["payload"]["camera_released"])
            self.assertFalse(lines[-1]["payload"]["device_discovery_used"])
            self.assertIsNone(lines[-1]["payload"]["plant_bpu_selected_bin"])
            annotations = sorted(request.annotated_dir.glob("*.jpg"))
            self.assertEqual(len(annotations), 2)
            self.assertTrue(all(path.stat().st_size > 0 for path in annotations))
            self.assertTrue(outcome.final_manifest_path.is_file())
            final = json.loads(
                outcome.final_manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                final["jsonl"]["sha256"],
                hashlib.sha256(request.jsonl_path.read_bytes()).hexdigest(),
            )
            self.assertTrue(final["final_manifest_atomic_write"])

    def test_processing_error_is_recorded_and_camera_is_still_released(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = _textured_rgb()
            source = _FakeSource([frame, frame])
            request = self._request(root, mode="one-shot", frames=1, annotated=False)

            def processor(_rgb: np.ndarray, _index: int):
                raise RuntimeError("fixture processing failure")

            outcome = run_bounded_frontend(
                request,
                print_binding=load_print_manifest_binding(PRINT_MANIFEST),
                frame_processor=processor,
                source_factory=lambda _: source,
                sleep_fn=lambda _: None,
            )
            lines = [
                json.loads(line)
                for line in request.jsonl_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(outcome.exit_code, 3)
            self.assertTrue(source.closed)
            frame_record = next(
                line["payload"] for line in lines if line["event"] == "frame_result"
            )
            self.assertIsNone(frame_record["semantic_hypothesis"])
            self.assertIsNone(frame_record["omega_ood_abstention"])
            self.assertIsNone(frame_record["registered_card_geometry"])
            self.assertFalse(frame_record["final_consensus"]["passed"])
            self.assertFalse(any(frame_record["authority"].values()))

    def test_numeric_alias_and_unbounded_count_are_rejected_before_source_open(self) -> None:
        with self.assertRaisesRegex(UvcFrontendError, "/dev/v4l/by-id"):
            validate_stable_device_syntax("/dev/video0")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._request(root, frames=31, annotated=False)
            with self.assertRaisesRegex(UvcFrontendError, r"\[1,30\]"):
                validate_request(request)

    def test_one_shot_cannot_silently_process_multiple_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._request(
                Path(directory),
                mode="one-shot",
                frames=2,
                annotated=False,
            )
            with self.assertRaisesRegex(UvcFrontendError, "exactly one"):
                validate_request(request)

    def test_existing_receipt_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._request(root, annotated=False)
            request.jsonl_path.write_text("preserve\n", encoding="utf-8")
            with self.assertRaisesRegex(UvcFrontendError, "overwrite refused"):
                validate_request(request)
            self.assertEqual(
                request.jsonl_path.read_text(encoding="utf-8"), "preserve\n"
            )

    def test_output_must_be_a_direct_child_of_non_symlink_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            request = self._request(root, annotated=False)
            request = FrontendRequest(
                **{
                    **request.__dict__,
                    "jsonl_path": nested / "escape.jsonl",
                }
            )
            with self.assertRaisesRegex(UvcFrontendError, "direct child"):
                validate_request(request)

    def test_camera_identity_mismatch_is_fail_closed_and_source_released(self) -> None:
        class WrongIdentitySource(_FakeSource):
            def negotiated_settings(self):
                result = dict(super().negotiated_settings())
                after = dict(result["identity_after_open"])
                after["usb_serial"] = "wrong-camera"
                result["identity_after_open"] = after
                return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = _textured_rgb()
            source = WrongIdentitySource([frame])
            request = self._request(
                root, mode="one-shot", frames=1, annotated=False
            )
            outcome = run_bounded_frontend(
                request,
                print_binding=load_print_manifest_binding(PRINT_MANIFEST),
                frame_processor=lambda _rgb, _index: {},
                source_factory=lambda _: source,
                sleep_fn=lambda _: None,
            )
            self.assertEqual(outcome.exit_code, 3)
            self.assertTrue(source.closed)
            self.assertTrue(outcome.camera_released)
            records = [
                json.loads(line)
                for line in request.jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(row["event"] == "session_error" for row in records))
            self.assertEqual(records[-1]["payload"]["captured_frames"], 0)

    def test_live_constructor_releases_fake_capture_when_post_open_identity_fails(self) -> None:
        class FakeCapture:
            def __init__(self) -> None:
                self.released = False
                self.values = {
                    1: 1196444237.0,
                    2: 640.0,
                    3: 480.0,
                    4: 30.0,
                }

            def isOpened(self):
                return True

            def set(self, prop, value):
                self.values[prop] = float(value)
                return True

            def get(self, prop):
                return self.values.get(prop, 0.0)

            def release(self):
                self.released = True

        capture = FakeCapture()
        fake_cv2 = SimpleNamespace(
            CAP_V4L2=200,
            CAP_PROP_FOURCC=1,
            CAP_PROP_FRAME_WIDTH=2,
            CAP_PROP_FRAME_HEIGHT=3,
            CAP_PROP_FPS=4,
            VideoWriter_fourcc=lambda *_args: 1196444237,
            VideoCapture=lambda *_args: capture,
        )
        identity = {
            "configured_device": STABLE_DEVICE,
            "resolved_device": "/dev/video-fixture",
            "kernel_video_node": "video-fixture",
            "usb_vid": EXPECTED_CAMERA.usb_vid,
            "usb_pid": EXPECTED_CAMERA.usb_pid,
            "usb_serial": EXPECTED_CAMERA.usb_serial,
            "usb_device_sysfs": "/sys/fixture/usb-camera",
            "identity_match": True,
            "device_discovery_used": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            request = self._request(
                Path(directory), mode="one-shot", frames=1, annotated=False
            )
            with (
                mock.patch.dict("sys.modules", {"cv2": fake_cv2}),
                mock.patch(
                    "app.omega_vision.uvc_card_frontend.read_explicit_usb_identity",
                    side_effect=[identity, UvcFrontendError("identity changed")],
                ),
            ):
                with self.assertRaisesRegex(UvcFrontendError, "identity changed"):
                    LiveUvcFrameSource(request)
        self.assertTrue(capture.released)

    def test_live_constructor_releases_on_baseexception_after_open(self) -> None:
        class InterruptingCapture:
            def __init__(self) -> None:
                self.released = False

            def isOpened(self):
                raise KeyboardInterrupt("fixture interrupt")

            def release(self):
                self.released = True

        capture = InterruptingCapture()
        fake_cv2 = SimpleNamespace(
            CAP_V4L2=200,
            VideoCapture=lambda *_args: capture,
        )
        identity = {
            "configured_device": STABLE_DEVICE,
            "resolved_device": "/dev/video-fixture",
        }
        with tempfile.TemporaryDirectory() as directory:
            request = self._request(
                Path(directory), mode="one-shot", frames=1, annotated=False
            )
            with (
                mock.patch.dict("sys.modules", {"cv2": fake_cv2}),
                mock.patch(
                    "app.omega_vision.uvc_card_frontend.read_explicit_usb_identity",
                    return_value=identity,
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    LiveUvcFrameSource(request)
        self.assertTrue(capture.released)

    def test_silent_ineffective_live_release_is_not_reported_complete(self) -> None:
        class SilentCapture:
            def isOpened(self):
                return True

            def release(self):
                return None

        source = LiveUvcFrameSource.__new__(LiveUvcFrameSource)
        source.configured_device = STABLE_DEVICE
        source.resolved_device = "/dev/video-fixture"
        source._capture = SilentCapture()
        source._closed = False
        source._close_receipt = None
        source._expected_camera = EXPECTED_CAMERA
        source._identity_before_open = {
            "configured_device": STABLE_DEVICE,
            "resolved_device": "/dev/video-fixture",
            "kernel_video_node": "video-fixture",
            "usb_vid": EXPECTED_CAMERA.usb_vid,
            "usb_pid": EXPECTED_CAMERA.usb_pid,
            "usb_serial": EXPECTED_CAMERA.usb_serial,
            "usb_device_sysfs": "/sys/fixture/usb-camera",
        }
        with mock.patch(
            "app.omega_vision.uvc_card_frontend.read_explicit_usb_identity",
            return_value=dict(source._identity_before_open),
        ):
            receipt = source.close()
        self.assertTrue(receipt["release_called"])
        self.assertTrue(receipt["opened_after_release"])
        self.assertFalse(receipt["release_completed"])

    def test_hardlink_publish_never_replaces_racing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            guard = SafeOutputDirectory(root)
            real_link = __import__("os").link

            def racing_link(source, target, *args, **kwargs):
                (root / "result.json").write_bytes(b"racing-owner")
                return real_link(source, target, *args, **kwargs)

            try:
                with mock.patch(
                    "app.omega_vision.uvc_card_frontend.os.link",
                    side_effect=racing_link,
                ):
                    with self.assertRaises(FileExistsError):
                        guard.publish_bytes("result.json", b"new-content")
                self.assertEqual((root / "result.json").read_bytes(), b"racing-owner")
            finally:
                guard.close()

    def test_print_manifest_is_exact_four_card_byte_binding_and_registry_matches(self) -> None:
        binding = load_print_manifest_binding(PRINT_MANIFEST)
        self.assertEqual(binding.byte_count, len(PRINT_MANIFEST.read_bytes()))
        self.assertEqual(
            binding.sha256,
            hashlib.sha256(PRINT_MANIFEST.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            set(binding.cards),
            {"grass_clump", "low_shrub", "young_tree", "unknown"},
        )
        bind_print_manifest_to_registry(binding, REGISTRY)

        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "tampered.json"
            payload = json.loads(PRINT_MANIFEST.read_text(encoding="utf-8"))
            payload["cards"].pop()
            tampered.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(UvcFrontendError, "SHA-256 mismatch"):
                load_print_manifest_binding(tampered)

            payload = json.loads(PRINT_MANIFEST.read_text(encoding="utf-8"))
            payload["cards"][0]["geometry"]["card_width_mm"] = 999.0
            tampered.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(UvcFrontendError, "SHA-256 mismatch"):
                load_print_manifest_binding(tampered)

    def test_frozen_threshold_file_rejects_wrong_expected_or_changed_bytes(self) -> None:
        thresholds = ROOT / "app" / "vision" / "dual_path_demo.thresholds.example.json"
        binding = _bind_frozen_file(
            thresholds,
            expected_sha256=FROZEN_THRESHOLDS_SHA256,
            frozen_sha256=FROZEN_THRESHOLDS_SHA256,
            label="demo thresholds",
        )
        self.assertEqual(binding.sha256, FROZEN_THRESHOLDS_SHA256)
        with self.assertRaisesRegex(UvcFrontendError, "frozen contract"):
            _bind_frozen_file(
                thresholds,
                expected_sha256="0" * 64,
                frozen_sha256=FROZEN_THRESHOLDS_SHA256,
                label="demo thresholds",
            )

    def test_frozen_calibration_load_preserves_nonqualification_boundary(self) -> None:
        binding = load_calibration_binding(CALIBRATION_MANIFEST)
        self.assertEqual(
            binding.calibration.status,
            "MACHINE_CURATED_EXPERIMENTAL_NOT_QUALIFIED",
        )
        self.assertEqual(binding.calibration.class_order, tuple(ROOTSCOPE_CLASS_ORDER))
        self.assertFalse(
            binding.provenance["formal_distribution_free_coverage_guarantee"]
        )


if __name__ == "__main__":
    unittest.main()
