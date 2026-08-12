from __future__ import annotations

import hashlib
import json
import math
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.omega_bpu_aux.probe import (
    BpuAuxProbeError,
    CLASS_COUNT,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    RECEIPT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    GenericBpuAuxRunner,
    _build_parser,
    describe_output,
    preprocess_rgb_to_nv12,
    rgb_to_nv12,
    run_manifest_probe,
    sha256_file,
    write_receipt_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
PROBE_SOURCE = ROOT / "app/omega_bpu_aux/probe.py"


class _FakeTensor:
    def __init__(self, *, name: str, shape: list[int], layout: str, dtype: str):
        self.name = name
        self.properties = types.SimpleNamespace(
            shape=shape,
            validShape=shape,
            alignedShape=shape,
            layout=layout,
            dtype=dtype,
        )


class _FakeOutput:
    def __init__(self, buffer: np.ndarray):
        self.buffer = buffer


class _FakeModel:
    def __init__(
        self,
        output: np.ndarray,
        *,
        forward_error: Exception | None = None,
        output_count: int = 1,
    ) -> None:
        self.name = "fake_mobilenetv2_224x224_nv12"
        self.inputs = [
            _FakeTensor(
                name="input",
                shape=[1, 3, INPUT_HEIGHT, INPUT_WIDTH],
                layout="NCHW",
                dtype="uint8",
            )
        ]
        self.outputs = [
            _FakeTensor(
                name=f"output_{index}",
                shape=[1, CLASS_COUNT],
                layout="NCHW",
                dtype="float32",
            )
            for index in range(output_count)
        ]
        self.output = output
        self.forward_error = forward_error
        self.forward_calls: list[np.ndarray] = []

    def forward(self, tensor: np.ndarray):
        if self.forward_error is not None:
            raise self.forward_error
        self.forward_calls.append(np.array(tensor, copy=True))
        return [_FakeOutput(np.array(self.output, copy=True))]


class _FakeDnn:
    def __init__(
        self,
        model: _FakeModel,
        *,
        load_error: Exception | None = None,
        model_count: int = 1,
    ) -> None:
        self.model = model
        self.load_error = load_error
        self.model_count = model_count
        self.load_calls: list[str] = []

    def load(self, path: str):
        self.load_calls.append(path)
        if self.load_error is not None:
            raise self.load_error
        return [self.model for _ in range(self.model_count)]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BpuAuxProbeTests(unittest.TestCase):
    def _fixture(
        self,
        directory: Path,
        *,
        output: np.ndarray | None = None,
        output_semantics: str = "LOGITS",
        warmup_runs: int = 1,
        with_labels: bool = False,
    ) -> tuple[Path, Path, Path, _FakeModel, _FakeDnn]:
        model_path = directory / "mobilenetv2_224x224_nv12.bin"
        model_path.write_bytes(b"fake-bayes-e-bpu-model-for-contract-tests")

        image_path = directory / "explicit.png"
        rgb = np.zeros((18, 26, 3), dtype=np.uint8)
        rgb[:, :, 0] = 207
        rgb[:, :, 1] = 91
        rgb[:, :, 2] = 33
        Image.fromarray(rgb, mode="RGB").save(image_path)

        if output is None:
            output = np.linspace(-3.0, 3.0, CLASS_COUNT, dtype=np.float32).reshape(
                1, CLASS_COUNT
            )
        model = _FakeModel(output)
        dnn = _FakeDnn(model)

        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": "unit-test-run",
            "model": {
                "path": str(model_path.resolve()),
                "sha256": _digest(model_path),
                "output_semantics": output_semantics,
            },
            "top_k": 5,
            "warmup_runs": warmup_runs,
            "images": [
                {
                    "image_id": "explicit-001",
                    "path": str(image_path.resolve()),
                    "sha256": _digest(image_path),
                }
            ],
        }
        if with_labels:
            labels_path = directory / "labels.txt"
            labels_path.write_text(
                repr({index: f"generic_{index}" for index in range(CLASS_COUNT)}),
                encoding="utf-8",
            )
            manifest["labels"] = {
                "path": str(labels_path.resolve()),
                "sha256": _digest(labels_path),
                "format": "PYTHON_LITERAL_DICT_INT_TO_STRING",
            }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest_path, model_path, image_path, model, dnn

    def test_rgb_to_nv12_contract_and_neutral_gray_chroma(self) -> None:
        gray = np.full((4, 6, 3), 128, dtype=np.uint8)
        nv12 = rgb_to_nv12(gray)
        self.assertEqual(nv12.shape, (4 * 6 * 3 // 2,))
        self.assertEqual(nv12.dtype, np.uint8)
        self.assertTrue(nv12.flags.c_contiguous)
        uv = nv12[4 * 6 :]
        self.assertTrue(np.all(uv[0::2] == 128))
        self.assertTrue(np.all(uv[1::2] == 128))

    def test_rgb_to_nv12_rejects_wrong_dtype_shape_and_odd_size(self) -> None:
        invalid = (
            np.zeros((4, 4), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.float32),
            np.zeros((3, 4, 3), dtype=np.uint8),
        )
        for value in invalid:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaisesRegex(BpuAuxProbeError, "positive even-sized"):
                    rgb_to_nv12(value)

    def test_preprocess_is_pillow_only_flat_nv12_and_deterministic(self) -> None:
        rgb = np.zeros((15, 37, 3), dtype=np.uint8)
        rgb[:, :, 0] = 250
        first, first_receipt = preprocess_rgb_to_nv12(rgb)
        second, second_receipt = preprocess_rgb_to_nv12(rgb)
        self.assertEqual(
            first.shape, (INPUT_WIDTH * INPUT_HEIGHT * 3 // 2,)
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(
            first_receipt["nv12_sha256"], second_receipt["nv12_sha256"]
        )
        self.assertEqual(first_receipt["runtime_color_format"], "NV12")
        self.assertEqual(first_receipt["resize_interpolation"], "PIL_BILINEAR")

    def test_describe_logits_computes_finite_probability_entropy_energy_topk(self) -> None:
        logits = np.zeros((1, CLASS_COUNT), dtype=np.float32)
        logits[0, 17] = 4.0
        report = describe_output(
            logits,
            output_semantics="LOGITS",
            top_k=3,
            labels={17: "generic_seventeen"},
        )
        self.assertEqual(report["class_count"], CLASS_COUNT)
        self.assertAlmostEqual(report["probabilities_sum"], 1.0, places=7)
        self.assertTrue(report["all_canonical_logits_finite"])
        self.assertTrue(report["all_probabilities_finite"])
        self.assertEqual(len(report["vectors"]["raw_model_values"]), CLASS_COUNT)
        self.assertEqual(len(report["vectors"]["canonical_logits"]), CLASS_COUNT)
        self.assertEqual(len(report["vectors"]["probabilities"]), CLASS_COUNT)
        self.assertTrue(
            all(math.isfinite(value) for value in report["vectors"]["canonical_logits"])
        )
        self.assertTrue(
            all(math.isfinite(value) for value in report["vectors"]["probabilities"])
        )
        self.assertEqual(
            report["top_k"][0]["generic_imagenet_class_id"], 17
        )
        self.assertEqual(report["top_k"][0]["vendor_label"], "generic_seventeen")
        descriptors = report["generic_descriptors"]
        self.assertTrue(math.isfinite(descriptors["predictive_entropy_nats"]))
        self.assertTrue(math.isfinite(descriptors["energy_score_temperature_1"]))
        self.assertTrue(descriptors["energy_raw_logit_comparable"])
        self.assertIsNone(report["ood_interpretation"]["ood_decision"])
        self.assertFalse(report["ood_interpretation"]["plant_domain_ood_claim"])

    def test_probability_output_is_explicitly_not_raw_logit_energy_comparable(self) -> None:
        probabilities = np.full(
            (1, CLASS_COUNT), 1.0 / CLASS_COUNT, dtype=np.float32
        )
        report = describe_output(
            probabilities,
            output_semantics="PROBABILITIES",
            top_k=2,
            labels={},
        )
        descriptors = report["generic_descriptors"]
        self.assertEqual(
            descriptors["energy_basis"], "LOG_PROBABILITY_CANONICALIZATION"
        )
        self.assertFalse(descriptors["energy_raw_logit_comparable"])
        self.assertAlmostEqual(
            descriptors["normalized_predictive_entropy"], 1.0, places=6
        )

    def test_nonfinite_wrong_count_and_invalid_probability_output_fail_closed(self) -> None:
        invalid = (
            (np.zeros((999,), dtype=np.float32), "exactly 1000"),
            (
                np.concatenate(
                    (
                        np.zeros(CLASS_COUNT - 1, dtype=np.float32),
                        np.asarray([np.nan], dtype=np.float32),
                    )
                ),
                "finite",
            ),
        )
        for values, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(BpuAuxProbeError, message):
                    describe_output(
                        values,
                        output_semantics="LOGITS",
                        top_k=3,
                        labels={},
                    )
        bad_probability = np.full(
            (CLASS_COUNT,), 0.5 / CLASS_COUNT, dtype=np.float32
        )
        with self.assertRaisesRegex(BpuAuxProbeError, "sum to 1"):
            describe_output(
                bad_probability,
                output_semantics="PROBABILITIES",
                top_k=3,
                labels={},
            )

    def test_fake_manifest_run_is_permanently_non_bpu_zero_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            manifest_path, model_path, _image_path, model, dnn = self._fixture(
                directory, with_labels=True
            )
            receipt = run_manifest_probe(
                manifest_path.resolve(),
                dnn_module=dnn,
                injected_model_path=model_path.resolve(),
            )
        self.assertEqual(receipt["schema_version"], RECEIPT_SCHEMA_VERSION)
        self.assertEqual(
            receipt["status"], "FAKE_BACKEND_TEST_PASS_NOT_BPU_EVIDENCE"
        )
        self.assertTrue(receipt["runtime"]["injected_test_backend"])
        self.assertFalse(receipt["runtime"]["bpu_forward_executed"])
        self.assertTrue(receipt["runtime"]["fake_forward_executed"])
        self.assertEqual(receipt["runtime"]["warmup_forward_count"], 1)
        self.assertEqual(receipt["runtime"]["measured_forward_count"], 1)
        self.assertEqual(len(model.forward_calls), 2)
        for tensor in model.forward_calls:
            self.assertEqual(
                tensor.shape, (INPUT_WIDTH * INPUT_HEIGHT * 3 // 2,)
            )
            self.assertEqual(tensor.dtype, np.uint8)
        self.assertFalse(any(receipt["effects_and_authority"].values()))
        self.assertFalse(receipt["claims"]["plant_classification"])
        self.assertFalse(receipt["claims"]["plant_species_identification"])
        self.assertFalse(receipt["claims"]["ood_threshold_calibrated"])
        self.assertIsNone(receipt["claims"]["rootscope_classifier_selected_bin"])
        self.assertTrue(
            receipt["claims"]["rootscope_classifier_selected_bin_remains_null"]
        )
        self.assertFalse(
            receipt["integration_boundary"]["safety_compiler_influence"]
        )
        self.assertEqual(
            receipt["images"][0]["output"]["top_k"][0]["vendor_label"],
            "generic_999",
        )
        self.assertRegex(receipt["receipt_sha256"], r"^[0-9a-f]{64}$")

    def test_manifest_rejects_unknown_keys_duplicates_relative_paths_and_wildcards(self) -> None:
        mutations = (
            lambda payload: payload.update({"unexpected": True}),
            lambda payload: payload["images"].append(dict(payload["images"][0])),
            lambda payload: payload["images"][0].update({"path": "relative.png"}),
            lambda payload: payload["images"][0].update(
                {"path": str(Path(payload["images"][0]["path"]).parent / "*.png")}
            ),
        )
        messages = ("unknown", "duplicate image_id", "absolute", "wildcard")
        for mutation, message in zip(mutations, messages):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory_string:
                directory = Path(directory_string)
                manifest_path, model_path, _image_path, _model, dnn = self._fixture(
                    directory
                )
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutation(payload)
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(BpuAuxProbeError, message):
                    run_manifest_probe(
                        manifest_path.resolve(),
                        dnn_module=dnn,
                        injected_model_path=model_path.resolve(),
                    )

    def test_image_hash_mismatch_occurs_before_backend_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            manifest_path, model_path, image_path, _model, dnn = self._fixture(directory)
            image_path.write_bytes(image_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(BpuAuxProbeError, "SHA-256 mismatch"):
                run_manifest_probe(
                    manifest_path.resolve(),
                    dnn_module=dnn,
                    injected_model_path=model_path.resolve(),
                )
            self.assertEqual(dnn.load_calls, [])

    def test_model_hash_mismatch_occurs_before_backend_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            manifest_path, model_path, _image_path, _model, dnn = self._fixture(
                directory
            )
            model_path.write_bytes(model_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(BpuAuxProbeError, "SHA-256 mismatch"):
                run_manifest_probe(
                    manifest_path.resolve(),
                    dnn_module=dnn,
                    injected_model_path=model_path.resolve(),
                )
            self.assertEqual(dnn.load_calls, [])

    def test_backend_load_forward_and_interface_errors_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            manifest_path, model_path, _image_path, model, _dnn = self._fixture(
                directory
            )
            load_dnn = _FakeDnn(model, load_error=RuntimeError("simulated load"))
            with self.assertRaisesRegex(BpuAuxProbeError, "model load failed"):
                run_manifest_probe(
                    manifest_path.resolve(),
                    dnn_module=load_dnn,
                    injected_model_path=model_path.resolve(),
                )
            multiple_dnn = _FakeDnn(model, model_count=2)
            with self.assertRaisesRegex(BpuAuxProbeError, "exactly one model"):
                run_manifest_probe(
                    manifest_path.resolve(),
                    dnn_module=multiple_dnn,
                    injected_model_path=model_path.resolve(),
                )
            forward_model = _FakeModel(
                np.zeros((1, CLASS_COUNT), dtype=np.float32),
                forward_error=RuntimeError("simulated forward"),
            )
            with self.assertRaisesRegex(BpuAuxProbeError, "forward failed"):
                run_manifest_probe(
                    manifest_path.resolve(),
                    dnn_module=_FakeDnn(forward_model),
                    injected_model_path=model_path.resolve(),
                )
            bad_interface_model = _FakeModel(
                np.zeros((1, CLASS_COUNT), dtype=np.float32),
                output_count=2,
            )
            with self.assertRaisesRegex(BpuAuxProbeError, "one input and one output"):
                GenericBpuAuxRunner(
                    model_path=model_path,
                    expected_sha256=sha256_file(model_path),
                    dnn_module=_FakeDnn(bad_interface_model),
                )

    def test_runtime_output_nan_fails_without_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            output = np.zeros((1, CLASS_COUNT), dtype=np.float32)
            output[0, 42] = np.nan
            manifest_path, model_path, _image_path, _model, dnn = self._fixture(
                directory, output=output
            )
            with self.assertRaisesRegex(BpuAuxProbeError, "finite"):
                run_manifest_probe(
                    manifest_path.resolve(),
                    dnn_module=dnn,
                    injected_model_path=model_path.resolve(),
                )

    def test_injected_model_path_cannot_be_used_with_real_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            manifest_path, model_path, _image_path, _model, _dnn = self._fixture(
                Path(directory_string)
            )
            with self.assertRaisesRegex(BpuAuxProbeError, "only with an injected"):
                run_manifest_probe(
                    manifest_path.resolve(),
                    injected_model_path=model_path.resolve(),
                )

    def test_receipt_writer_is_exclusive_and_json_has_no_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            output = Path(directory_string) / "receipt.json"
            payload = {"status": "TEST", "value": 1.25}
            resolved = write_receipt_exclusive(output, payload)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            self.assertEqual(resolved, output.resolve())
            with self.assertRaisesRegex(BpuAuxProbeError, "refusing to overwrite"):
                write_receipt_exclusive(output, payload)

    def test_cli_has_only_manifest_and_output_no_model_camera_or_device_arguments(self) -> None:
        parser = _build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("--manifest", option_strings)
        self.assertIn("--out", option_strings)
        for forbidden in ("--model", "--camera", "--camera-device", "--device", "--directory"):
            self.assertNotIn(forbidden, option_strings)

    def test_import_is_lazy_and_source_has_no_control_or_discovery_dependencies(self) -> None:
        source = PROBE_SOURCE.read_text(encoding="utf-8")
        prefix = source.split("class GenericBpuAuxRunner", 1)[0]
        self.assertNotIn("from hobot_dnn import", prefix)
        for forbidden in (
            "import cv2",
            "VideoCapture(",
            "serial.Serial(",
            "socket.socket(",
            "requests.get(",
            "requests.post(",
            "subprocess.run(",
            "from app.state_machine import",
            "from app.hardware import",
            "from app.serial import",
            "from app.omega_runtime import",
            "os.listdir(",
            ".iterdir(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_schema_and_documentation_freeze_claim_boundary(self) -> None:
        schema = json.loads(
            (
                ROOT / "configs/omega/bpu_aux_probe_input.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["model"]["properties"]["path"]["const"],
                         "/opt/hobot/model/x5/basic/mobilenetv2_224x224_nv12.bin")
        readme = (
            ROOT / "app/omega_bpu_aux/README.md"
        ).read_text(encoding="utf-8")
        for phrase in (
            "uncalibrated",
            "not a plant-domain OOD decision",
            "`selected_bin=null`",
            "does not qualify a RootScope BPU classifier",
            "Safety Compiler",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, readme)


if __name__ == "__main__":
    unittest.main()
