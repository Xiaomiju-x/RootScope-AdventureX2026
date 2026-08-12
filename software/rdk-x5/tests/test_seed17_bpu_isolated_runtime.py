from __future__ import annotations

import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from app.edge.bpu_seed17 import (
    CLASS_ORDER,
    INPUT_SHAPE,
    OUTPUT_SHAPE,
    Seed17BpuContractError,
    Seed17BpuRunner,
    load_hash_bound_image_bgr,
    preprocess_bgr_uint8,
)
from app.edge.capsule import CapsuleConfig
from app.edge.onnx_cpu import make_simulated_rgb, preprocess_rgb
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VENV_INSTALLER = ROOT / "deploy/x5/scripts/prepare_bpu_system_site_venv.py"
VENV_HELPERS = runpy.run_path(str(VENV_INSTALLER))


class _Properties:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        layout: str,
        dtype: str,
        tensor_type: str,
    ) -> None:
        self.name = "image" if shape == INPUT_SHAPE else "logits"
        self.shape = list(shape)
        self.validShape = list(shape)
        self.alignedShape = list(shape)
        self.layout = layout
        self.dtype = dtype
        self.tensor_type = tensor_type


class _Tensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        layout: str,
        dtype: str,
        buffer_dtype: np.dtype,
        tensor_type: str,
    ) -> None:
        self.properties = _Properties(
            shape, layout=layout, dtype=dtype, tensor_type=tensor_type
        )
        self.buffer = np.zeros(shape, dtype=buffer_dtype)


class _Value:
    def __init__(self, buffer: np.ndarray) -> None:
        self.buffer = buffer


class _FakeModel:
    def __init__(
        self,
        *,
        input_shape: tuple[int, ...] = INPUT_SHAPE,
        output_shape: tuple[int, ...] = OUTPUT_SHAPE,
        layout: str = "HB_DNN_LAYOUT_NCHW",
        input_buffer_dtype: np.dtype | None = None,
        logits: np.ndarray | None = None,
        forward_error: Exception | None = None,
    ) -> None:
        if input_buffer_dtype is None:
            input_buffer_dtype = np.dtype(np.uint8)
        self.inputs = [
            _Tensor(
                input_shape,
                layout=layout,
                dtype="uint8",
                buffer_dtype=input_buffer_dtype,
                tensor_type="HB_DNN_IMG_TYPE_RGB",
            )
        ]
        self.outputs = [
            _Tensor(
                output_shape,
                layout="HB_DNN_LAYOUT_NC",
                dtype="float32",
                buffer_dtype=np.dtype(np.float32),
                tensor_type="HB_DNN_TENSOR_TYPE_F32",
            )
        ]
        self.logits = (
            np.asarray([[0.1, 1.25, -0.5, 0.0]], dtype=np.float32)
            if logits is None
            else logits
        )
        self.forward_error = forward_error
        self.forward_calls = 0
        self.last_input: np.ndarray | None = None

    def forward(self, tensor: np.ndarray):
        self.forward_calls += 1
        self.last_input = tensor
        if self.forward_error is not None:
            raise self.forward_error
        return [_Value(self.logits)]


class _FakeDnn:
    def __init__(self, models: list[_FakeModel]) -> None:
        self.models = models
        self.load_calls: list[str] = []

    def load(self, path: str):
        self.load_calls.append(path)
        return self.models


class Seed17BpuRuntimeTests(unittest.TestCase):
    def _artifact(self, directory: Path) -> tuple[Path, str]:
        model = directory / "rootscope_seed17_test.bin"
        model.write_bytes(b"fake-bpu-bin-contract-bytes")
        return model, hashlib.sha256(model.read_bytes()).hexdigest()

    def _runner(
        self, directory: Path, model: _FakeModel | None = None
    ) -> tuple[Seed17BpuRunner, _FakeModel, _FakeDnn]:
        artifact, digest = self._artifact(directory)
        fake_model = model or _FakeModel()
        fake_dnn = _FakeDnn([fake_model])
        runner = Seed17BpuRunner(artifact, digest, dnn_module=fake_dnn)
        return runner, fake_model, fake_dnn

    def test_bgr_is_converted_to_rgb_nchw_uint8_without_host_normalization(self) -> None:
        bgr = np.empty((224, 224, 3), dtype=np.uint8)
        bgr[:, :, 0] = 11
        bgr[:, :, 1] = 22
        bgr[:, :, 2] = 233
        tensor = preprocess_bgr_uint8(bgr)
        self.assertEqual(tensor.shape, INPUT_SHAPE)
        self.assertEqual(tensor.dtype, np.uint8)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertTrue(np.all(tensor[0, 0] == 233))
        self.assertTrue(np.all(tensor[0, 1] == 22))
        self.assertTrue(np.all(tensor[0, 2] == 11))

    def test_resize_and_center_crop_geometry_is_frozen(self) -> None:
        bgr = np.zeros((200, 400, 3), dtype=np.uint8)
        bgr[:, :200] = (10, 20, 30)
        bgr[:, 200:] = (110, 120, 130)
        first = preprocess_bgr_uint8(bgr)
        second = preprocess_bgr_uint8(bgr.copy())
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, INPUT_SHAPE)

    def test_bpu_geometry_matches_frozen_cpu_training_transform_before_normalization(self) -> None:
        contract = CapsuleConfig.from_json_file(
            ROOT / "deploy/x5/capsule_config.seed17_cpu_experimental.json"
        ).model.preprocess
        rgb = make_simulated_rgb(257, 401)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        bpu_uint8 = preprocess_bgr_uint8(bgr).astype(np.float32)
        mean = np.asarray(contract.mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(contract.std, dtype=np.float32).reshape(1, 3, 1, 1)
        normalized_from_bpu_input = (
            bpu_uint8 * np.float32(contract.scale) - mean
        ) / std
        cpu_reference = preprocess_rgb(rgb, contract)
        self.assertLessEqual(
            float(np.max(np.abs(normalized_from_bpu_input - cpu_reference))),
            1.0e-6,
        )

    def test_preprocess_rejects_wrong_shape_and_dtype(self) -> None:
        with self.assertRaisesRegex(Seed17BpuContractError, "HxWx3"):
            preprocess_bgr_uint8(np.zeros((224, 224), dtype=np.uint8))
        with self.assertRaisesRegex(Seed17BpuContractError, "uint8 BGR"):
            preprocess_bgr_uint8(np.zeros((224, 224, 3), dtype=np.float32))

    def test_hash_mismatch_fails_before_fake_dnn_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            artifact, _digest = self._artifact(directory)
            fake_dnn = _FakeDnn([_FakeModel()])
            with self.assertRaisesRegex(Seed17BpuContractError, "SHA-256 mismatch"):
                Seed17BpuRunner(artifact, "0" * 64, dnn_module=fake_dnn)
            self.assertEqual(fake_dnn.load_calls, [])

    def test_input_shape_is_checked_at_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            artifact, digest = self._artifact(directory)
            with self.assertRaisesRegex(Seed17BpuContractError, "input shape"):
                Seed17BpuRunner(
                    artifact,
                    digest,
                    dnn_module=_FakeDnn([_FakeModel(input_shape=(1, 224, 224, 3))]),
                )

    def test_output_shape_is_checked_at_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            artifact, digest = self._artifact(directory)
            with self.assertRaisesRegex(Seed17BpuContractError, "output shape"):
                Seed17BpuRunner(
                    artifact,
                    digest,
                    dnn_module=_FakeDnn([_FakeModel(output_shape=(1, 5))]),
                )

    def test_layout_is_strictly_nchw(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            artifact, digest = self._artifact(directory)
            with self.assertRaisesRegex(Seed17BpuContractError, "NCHW"):
                Seed17BpuRunner(
                    artifact,
                    digest,
                    dnn_module=_FakeDnn([_FakeModel(layout="NHWC")]),
                )

    def test_input_buffer_must_be_uint8(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            artifact, digest = self._artifact(directory)
            with self.assertRaisesRegex(Seed17BpuContractError, "uint8 RGB"):
                Seed17BpuRunner(
                    artifact,
                    digest,
                    dnn_module=_FakeDnn(
                        [_FakeModel(input_buffer_dtype=np.dtype(np.float32))]
                    ),
                )

    def test_exactly_one_model_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            artifact, digest = self._artifact(directory)
            with self.assertRaisesRegex(Seed17BpuContractError, "exactly one model"):
                Seed17BpuRunner(artifact, digest, dnn_module=_FakeDnn([]))

    def test_class_order_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            artifact, digest = self._artifact(directory)
            with self.assertRaisesRegex(Seed17BpuContractError, "class order"):
                Seed17BpuRunner(
                    artifact,
                    digest,
                    class_order=tuple(reversed(CLASS_ORDER)),
                    dnn_module=_FakeDnn([_FakeModel()]),
                )

    def test_fake_dnn_success_cannot_be_reported_as_bpu_or_qualification_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            runner, model, _dnn = self._runner(Path(directory_string))
            bgr = np.zeros((240, 320, 3), dtype=np.uint8)
            report = runner.run_bgr(
                bgr,
                source_provenance={"source_kind": "UNIT_TEST", "camera_opened": False},
            )
        self.assertEqual(report["status"], "FAKE_DNN_REPLAY_PASS_NOT_BPU_EVIDENCE")
        self.assertTrue(report["runtime"]["injected_test_backend"])
        self.assertEqual(model.forward_calls, 1)
        self.assertIsNotNone(model.last_input)
        self.assertEqual(model.last_input.shape, INPUT_SHAPE)
        self.assertEqual(model.last_input.dtype, np.uint8)
        self.assertTrue(model.last_input.flags.c_contiguous)
        self.assertEqual(report["inference"]["output_shape"], list(OUTPUT_SHAPE))
        self.assertTrue(report["inference"]["output_finite"])
        self.assertFalse(report["authority"]["hardware_touched"])
        self.assertFalse(report["authority"]["bpu_used"])
        self.assertFalse(report["claims"]["x5_ready"])
        self.assertFalse(report["claims"]["camera_qualified"])
        self.assertFalse(report["claims"]["model_qualified"])
        self.assertFalse(report["claims"]["production_integration_allowed"])

    def test_preflight_checks_interface_but_does_not_forward(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            runner, model, _dnn = self._runner(Path(directory_string))
            report = runner.preflight_report()
        self.assertEqual(report["status"], "FAKE_DNN_INTERFACE_PASS_NOT_BPU_EVIDENCE")
        self.assertEqual(model.forward_calls, 0)
        self.assertIsNone(report["inference"])
        self.assertFalse(report["authority"]["bpu_used"])

    def test_runtime_output_shape_and_finiteness_fail_closed(self) -> None:
        cases = (
            (np.zeros((4,), dtype=np.float32), "runtime output shape"),
            (
                np.asarray([[0.0, np.nan, 1.0, 2.0]], dtype=np.float32),
                "finite",
            ),
        )
        for logits, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory_string:
                runner, _model, _dnn = self._runner(
                    Path(directory_string), _FakeModel(logits=logits)
                )
                with self.assertRaisesRegex(Seed17BpuContractError, message):
                    runner.run_bgr(np.zeros((224, 224, 3), dtype=np.uint8))

    def test_backend_forward_exception_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            runner, _model, _dnn = self._runner(
                Path(directory_string),
                _FakeModel(forward_error=RuntimeError("simulated backend failure")),
            )
            with self.assertRaisesRegex(Seed17BpuContractError, "forward failed"):
                runner.run_bgr(np.zeros((224, 224, 3), dtype=np.uint8))

    def test_hash_bound_image_loader_preserves_bgr_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            image_path = Path(directory_string) / "golden.png"
            rgb = np.zeros((16, 24, 3), dtype=np.uint8)
            rgb[:, :, 0] = 201
            rgb[:, :, 1] = 52
            rgb[:, :, 2] = 7
            Image.fromarray(rgb, mode="RGB").save(image_path)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            bgr, provenance = load_hash_bound_image_bgr(image_path, digest)
            self.assertTrue(np.all(bgr[:, :, 0] == 7))
            self.assertTrue(np.all(bgr[:, :, 1] == 52))
            self.assertTrue(np.all(bgr[:, :, 2] == 201))
            self.assertEqual(provenance["source_file_sha256"], digest)
            self.assertFalse(provenance["camera_opened"])
            with self.assertRaisesRegex(Seed17BpuContractError, "SHA-256 mismatch"):
                load_hash_bound_image_bgr(image_path, "0" * 64)

    def test_adapter_import_has_no_eager_bpu_or_authority_integration(self) -> None:
        source = (ROOT / "app/edge/bpu_seed17.py").read_text(encoding="utf-8")
        prefix = source.split("class Seed17BpuRunner", 1)[0]
        self.assertNotIn("from hobot_dnn import", prefix)
        for forbidden in (
            "serial.Serial(",
            "requests.get(",
            "requests.post(",
            "socket.socket(",
            "from app.state_machine import",
            "from app.hardware import",
            "from app.serial import",
            "subprocess.run(",
            "glob(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_cli_defaults_to_no_camera_and_has_no_action_imports(self) -> None:
        source = (
            ROOT / "deploy/x5/scripts/bpu_seed17_isolated_readonly.py"
        ).read_text(encoding="utf-8")
        self.assertIn("elif args.camera_device is not None", source)
        self.assertIn("report = dict(runner.preflight_report())", source)
        for forbidden in (
            "VideoCapture(0",
            "glob(",
            "from app.state_machine import",
            "from app.hardware import",
            "from app.serial import",
            "systemctl",
            "subprocess",
            "socket",
            "requests",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_machine_contract_freezes_rgb_ddr_interface_and_all_claims_false(self) -> None:
        contract = json.loads(
            (ROOT / "deploy/x5/seed17_bpu_isolated_runtime_contract.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(contract["input"]["runtime_color_order"], "RGB")
        self.assertEqual(contract["input"]["runtime_layout"], "NCHW")
        self.assertEqual(contract["input"]["runtime_dtype"], "uint8")
        self.assertEqual(contract["input"]["runtime_shape"], list(INPUT_SHAPE))
        self.assertEqual(contract["input"]["runtime_source"], "DDR")
        self.assertFalse(contract["input"]["host_normalization"])
        self.assertEqual(contract["output"]["shape"], list(OUTPUT_SHAPE))
        self.assertEqual(contract["output"]["class_order"], list(CLASS_ORDER))
        self.assertFalse(any(contract["claims"].values()))
        self.assertFalse(any(contract["authority"].values()))
        self.assertFalse(contract["modes"]["device_enumeration"])
        self.assertFalse(contract["modes"]["continuous_camera_loop"])
        environment = contract["python_environment"]
        self.assertTrue(environment["independent_bpu_venv_required"])
        self.assertTrue(environment["include_system_site_packages"])
        self.assertFalse(environment["core_v1_venv_allowed"])
        self.assertEqual(environment["local_wheel_install_allowlist"], ["Pillow"])
        self.assertFalse(environment["venv_numpy_install_allowed"])
        self.assertFalse(environment["venv_hobot_dnn_install_allowed"])

    def test_bpu_venv_config_parser_requires_explicit_system_site_packages(self) -> None:
        parser = VENV_HELPERS["parse_pyvenv_config"]
        with tempfile.TemporaryDirectory() as directory_string:
            config_path = Path(directory_string) / "pyvenv.cfg"
            config_path.write_text(
                "home = /usr/bin\ninclude-system-site-packages = true\nversion = 3.10.12\n",
                encoding="utf-8",
            )
            value = parser(config_path)
        self.assertEqual(value["include-system-site-packages"], "true")
        self.assertEqual(value["version"], "3.10.12")

    def test_only_hash_bound_pillow_wheel_is_accepted_by_installer_helper(self) -> None:
        validator = VENV_HELPERS["validate_pillow_wheel"]
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            wheel = directory / "pillow-11.3.0-cp310-cp310-manylinux2014_aarch64.whl"
            wheel.write_bytes(b"fake-pillow-wheel-for-contract-test")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            result = validator(wheel, digest)
            self.assertEqual(result["sha256"], digest)
            numpy_wheel = directory / "numpy-2.2.6-cp310-cp310-manylinux.whl"
            numpy_wheel.write_bytes(b"must-never-be-accepted")
            numpy_digest = hashlib.sha256(numpy_wheel.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "Pillow wheel"):
                validator(numpy_wheel, numpy_digest)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validator(wheel, "0" * 64)

    def test_bpu_venv_installer_is_import_only_offline_and_never_loads_model(self) -> None:
        source = VENV_INSTALLER.read_text(encoding="utf-8")
        self.assertIn("system_site_packages=True", source)
        self.assertIn('"--no-index"', source)
        self.assertIn('"--no-deps"', source)
        self.assertIn('"local_install_allowlist": ["Pillow"]', source)
        self.assertIn('"venv_numpy_install_allowed": False', source)
        self.assertIn('"venv_hobot_dnn_install_allowed": False', source)
        for forbidden in (
            ".load(",
            ".forward(",
            "VideoCapture(",
            "serial.Serial(",
            "socket.socket(",
            "requests.",
            "urllib.request",
            "systemctl",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_bpu_cli_rejects_core_venv_and_checks_system_module_origins(self) -> None:
        source = (
            ROOT / "deploy/x5/scripts/bpu_seed17_isolated_readonly.py"
        ).read_text(encoding="utf-8")
        self.assertIn("include-system-site-packages", source)
        self.assertIn("core v1 venv is forbidden", source)
        self.assertIn('(\"numpy\", \"hobot_dnn\")', source)
        self.assertIn("resolves inside the BPU venv", source)


if __name__ == "__main__":
    unittest.main()
