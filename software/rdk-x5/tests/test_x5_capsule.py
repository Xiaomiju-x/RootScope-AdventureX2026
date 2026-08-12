from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from app.edge.capsule import (
    CAPSULE_STATUS,
    GOLDEN_GENERATOR,
    PREPROCESS_MODE,
    ROOTSCOPE_CLASS_ORDER,
    CapsuleConfig,
    PreprocessConfig,
)
from app.edge.onnx_cpu import (
    CpuOnnxRunner,
    OnnxCpuContractError,
    make_simulated_rgb,
    preprocess_rgb,
)
from app.edge.preflight import run_preflight
from app.edge.selftest import run_simulated_selftest
from app.edge.service import locked_snapshot


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy" / "x5"
EXAMPLE = DEPLOY / "capsule_config.example.json"
SEED17_CONFIG = DEPLOY / "capsule_config.seed17_cpu_experimental.json"
SEED17_MANIFEST = DEPLOY / "seed17_cpu_deployment_manifest.json"
SEED17_MODEL = DEPLOY / "models" / "rootscope_seed17_cpu_experimental_opset11.onnx"
SEED17_RECEIPT = DEPLOY / "evidence" / "seed17_cpu_pc_selftest_receipt.json"
SEED17_SHA256 = "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad"
GOLDEN_TENSOR_SHA256 = "3cf32b73011e6f72a242041767e8a5a263fd354e9f2469accec3d82ce6015dc1"


def example_payload() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


class _Meta:
    def __init__(self, name: str, shape: list[int], type_name: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = type_name


class _FakeSession:
    def __init__(self, path: str, providers: list[str]) -> None:
        self.path = path
        self.requested_providers = providers

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def get_inputs(self) -> list[_Meta]:
        return [_Meta("image", [1, 3, 2, 2])]

    def get_outputs(self) -> list[_Meta]:
        return [_Meta("scores", [1, 4])]

    def run(self, outputs, feeds):
        if outputs is not None:
            assert outputs == ["scores"]
        assert list(feeds) == ["image"]
        value = feeds["image"]
        assert value.shape == (1, 3, 2, 2)
        return [np.asarray([[value.mean(), 0.25, -0.5, 1.0]], dtype=np.float32)]


class _FakeOrt:
    @staticmethod
    def get_available_providers() -> list[str]:
        return ["CPUExecutionProvider", "NotRequestedProvider"]

    InferenceSession = _FakeSession


class CapsuleContractTests(unittest.TestCase):
    def test_example_is_strictly_zero_authority_and_disabled(self) -> None:
        config = CapsuleConfig.from_json_file(EXAMPLE)
        self.assertEqual(config.status, CAPSULE_STATUS)
        self.assertFalse(any(config.authority.to_dict().values()))
        self.assertFalse(config.model.enabled)
        self.assertFalse(config.rgb.enabled)
        self.assertFalse(config.depth.enabled)
        self.assertFalse(config.llm.enabled)
        self.assertEqual(config.model.provider, "CPUExecutionProvider")
        self.assertEqual(config.model.output_shape, (1, 4))
        self.assertEqual(config.model.class_order, ROOTSCOPE_CLASS_ORDER)
        self.assertEqual(config.model.preprocess.mode, PREPROCESS_MODE)
        self.assertEqual(config.model.preprocess.short_side, 256)
        self.assertEqual(config.model.preprocess.center_crop, (224, 224))

    def test_any_authority_true_is_rejected(self) -> None:
        payload = example_payload()
        payload["authority"]["x5_validated"] = True
        with self.assertRaisesRegex(ValueError, "x5_validated.*false"):
            CapsuleConfig.from_mapping(payload)

    def test_bpu_or_model_candidate_claim_is_rejected(self) -> None:
        for field in ("bpu_ready", "model_candidate", "model_qualified"):
            with self.subTest(field=field):
                payload = example_payload()
                payload["model"][field] = True
                with self.assertRaisesRegex(ValueError, field):
                    CapsuleConfig.from_mapping(payload)

    def test_non_cpu_provider_is_rejected(self) -> None:
        payload = example_payload()
        payload["model"]["provider"] = "CUDAExecutionProvider"
        with self.assertRaisesRegex(ValueError, "CPUExecutionProvider"):
            CapsuleConfig.from_mapping(payload)

    def test_class_order_and_output_shape_are_frozen(self) -> None:
        payload = example_payload()
        payload["model"]["class_order"] = list(reversed(payload["model"]["class_order"]))
        with self.assertRaisesRegex(ValueError, "class_order"):
            CapsuleConfig.from_mapping(payload)
        payload = example_payload()
        payload["model"]["output_shape"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "output_shape"):
            CapsuleConfig.from_mapping(payload)

    def test_enabled_model_requires_path_and_hash(self) -> None:
        payload = example_payload()
        payload["model"]["enabled"] = True
        payload["model"]["path"] = "/opt/rootscope/models/example.onnx"
        with self.assertRaisesRegex(ValueError, "sha256"):
            CapsuleConfig.from_mapping(payload)

    def test_llm_must_be_loopback_and_cannot_execute_tools(self) -> None:
        payload = example_payload()
        payload["llm"]["host"] = "0.0.0.0"
        with self.assertRaisesRegex(ValueError, "loopback"):
            CapsuleConfig.from_mapping(payload)
        payload = example_payload()
        payload["llm"]["tool_execution"] = True
        with self.assertRaisesRegex(ValueError, "tool execution"):
            CapsuleConfig.from_mapping(payload)


class CpuOnnxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocess = PreprocessConfig(
            mode=PREPROCESS_MODE,
            short_side=2,
            center_crop=(2, 2),
            input_shape=(1, 3, 2, 2),
            color_order="RGB",
            scale=1.0 / 255.0,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            interpolation="bilinear",
            golden_generator=GOLDEN_GENERATOR,
            golden_source_shape=(2, 3, 3),
            golden_tensor_sha256="0" * 64,
        )

    def test_simulated_input_is_deterministic(self) -> None:
        first = make_simulated_rgb(12, 17)
        second = make_simulated_rgb(12, 17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape, (12, 17, 3))

    def test_cpu_runner_verifies_hash_shape_provider_and_finite_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"fake-static-onnx-for-contract-test")
            expected = hashlib.sha256(model.read_bytes()).hexdigest()
            runner = CpuOnnxRunner(
                model,
                expected,
                self.preprocess,
                input_name="image",
                output_name="scores",
                expected_output_shape=(1, 4),
                class_order=ROOTSCOPE_CLASS_ORDER,
                ort_module=_FakeOrt,
            )
            report = runner.run_rgb(make_simulated_rgb(2, 2))
        self.assertEqual(report["providers_actual"], ["CPUExecutionProvider"])
        self.assertTrue(report["output_finite"])
        self.assertFalse(report["bpu_ready"])
        self.assertFalse(report["bpu_used"])
        self.assertFalse(report["x5_validated"])
        self.assertFalse(report["execution_authority"])
        self.assertEqual(report["class_order"], list(ROOTSCOPE_CLASS_ORDER))
        self.assertIn(report["simulated_prediction_class"], ROOTSCOPE_CLASS_ORDER)

    def test_hash_mismatch_fails_before_session_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"tampered")
            with self.assertRaisesRegex(OnnxCpuContractError, "SHA-256 mismatch"):
                CpuOnnxRunner(
                    model,
                    "0" * 64,
                    self.preprocess,
                    expected_output_shape=(1, 4),
                    class_order=ROOTSCOPE_CLASS_ORDER,
                    ort_module=_FakeOrt,
                )

    def test_deployment_preprocess_matches_training_transform(self) -> None:
        adventurex = ROOT.parent
        if str(adventurex) not in sys.path:
            sys.path.insert(0, str(adventurex))
        from tools.training.rootscope_machine_curated_pipeline import build_transforms

        reference = build_transforms(train=False)
        contract = CapsuleConfig.from_json_file(EXAMPLE).model.preprocess
        for height, width in ((173, 311), (311, 173), (229, 229), (257, 401)):
            with self.subTest(height=height, width=width):
                image = make_simulated_rgb(height, width)
                actual = preprocess_rgb(image, contract)
                expected = reference(Image.fromarray(image, mode="RGB")).unsqueeze(0).numpy()
                self.assertEqual(actual.shape, (1, 3, 224, 224))
                self.assertLessEqual(float(np.max(np.abs(actual - expected))), 1.0e-6)

    def test_preprocess_golden_tensor_hash_is_frozen(self) -> None:
        contract = CapsuleConfig.from_json_file(EXAMPLE).model.preprocess
        height, width, _channels = contract.golden_source_shape
        tensor = preprocess_rgb(make_simulated_rgb(height, width), contract)
        actual = hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
        self.assertEqual(actual, GOLDEN_TENSOR_SHA256)
        self.assertEqual(actual, contract.golden_tensor_sha256)


class CapsulePreflightTests(unittest.TestCase):
    def _local_config(self, directory: Path) -> CapsuleConfig:
        payload = example_payload()
        payload["project_root"] = str(ROOT)
        payload["python_executable"] = sys.executable
        path = directory / "capsule.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return CapsuleConfig.from_json_file(path)

    def test_readonly_preflight_passes_core_with_model_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_preflight(self._local_config(Path(directory)))
        self.assertNotEqual(report["status"], "FAIL")
        self.assertEqual(report["summary"]["fail"], 0)
        self.assertEqual(
            report["device_policy"],
            "EXPLICIT_ALIAS_EXISTENCE_ONLY_NOT_OPENED_NOT_ENUMERATED",
        )
        self.assertFalse(any(report["authority"].values()))

    def test_optional_missing_device_is_warning_not_enumeration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = example_payload()
            payload["project_root"] = str(ROOT)
            payload["python_executable"] = sys.executable
            payload["inputs"]["rgb"].update(
                {
                    "enabled": True,
                    "required": False,
                    "backend": "uvc_v4l2",
                    "device": str(Path(directory) / "explicit-missing-device"),
                }
            )
            config = CapsuleConfig.from_mapping(payload)
            report = run_preflight(config)
        item = next(check for check in report["checks"] if check["name"] == "rgb_input_alias")
        self.assertEqual(item["status"], "WARN")
        self.assertIn("not opened", item["detail"])
        self.assertFalse(report["authority"]["ports_enumerated"])

    def test_disabled_model_selftest_is_preprocess_only(self) -> None:
        config = CapsuleConfig.from_json_file(EXAMPLE)
        first = run_simulated_selftest(config)
        second = run_simulated_selftest(config)
        self.assertEqual(first["status"], "PASS_PREPROCESS_ONLY_MODEL_DISABLED_SIMULATED_ONLY")
        self.assertFalse(first["onnx_executed"])
        self.assertFalse(first["accuracy_evidence"])
        self.assertTrue(first["preprocess_golden_matched"])
        self.assertEqual(first["preprocess_mode"], PREPROCESS_MODE)
        self.assertEqual(first["preprocessed_tensor_sha256"], second["preprocessed_tensor_sha256"])
        self.assertFalse(first["hardware_touched"])
        self.assertFalse(first["bpu_ready"])

    def test_locked_service_snapshot_exposes_no_actions_or_authority(self) -> None:
        snapshot = locked_snapshot(CapsuleConfig.from_json_file(EXAMPLE))
        self.assertEqual(snapshot["state"], "BOOT_LOCKED")
        self.assertEqual(snapshot["mode"], "SIMULATED_ONLY")
        self.assertFalse(snapshot["capsule"]["x5_validated"])
        self.assertFalse(snapshot["capsule"]["bpu_ready"])
        self.assertIn("NO_ACTION_ENDPOINTS_REGISTERED", snapshot["alerts"])

    def test_seed17_cpu_config_manifest_and_copied_model_are_hash_bound(self) -> None:
        config = CapsuleConfig.from_json_file(SEED17_CONFIG)
        manifest = json.loads(SEED17_MANIFEST.read_text(encoding="utf-8"))
        receipt = json.loads(SEED17_RECEIPT.read_text(encoding="utf-8"))
        model_hash = hashlib.sha256(SEED17_MODEL.read_bytes()).hexdigest()
        config_hash = hashlib.sha256(SEED17_CONFIG.read_bytes()).hexdigest()
        receipt_hash = hashlib.sha256(SEED17_RECEIPT.read_bytes()).hexdigest()
        self.assertTrue(config.model.enabled)
        self.assertEqual(config.model.sha256, SEED17_SHA256)
        self.assertEqual(model_hash, SEED17_SHA256)
        self.assertEqual(manifest["onnx"]["source_sha256"], SEED17_SHA256)
        self.assertEqual(manifest["onnx"]["deployment_sha256"], model_hash)
        self.assertTrue(manifest["onnx"]["byte_identity_verified"])
        self.assertEqual(manifest["deployment_config"]["sha256"], config_hash)
        self.assertEqual(
            manifest["pc_simulated_input_selftest_receipt"]["sha256"], receipt_hash
        )
        self.assertEqual(manifest["onnx"]["output"]["shape"], [1, 4])
        self.assertEqual(
            manifest["onnx"]["output"]["class_order"], list(ROOTSCOPE_CLASS_ORDER)
        )
        self.assertEqual(
            manifest["preprocess"]["golden_tensor_sha256"], GOLDEN_TENSOR_SHA256
        )
        self.assertFalse(any(config.authority.to_dict().values()))
        for field in ("model_candidate", "model_qualified", "x5_validated", "bpu_ready"):
            self.assertFalse(manifest["claims"][field])
            self.assertFalse(receipt["claims"][field])
        self.assertTrue(receipt["preprocess"]["golden_matched"])
        self.assertEqual(receipt["inference"]["output_shape"], [1, 4])
        self.assertFalse(any(receipt["authority"].values()))

    def test_seed17_manifest_source_and_reference_hashes_are_current(self) -> None:
        manifest = json.loads(SEED17_MANIFEST.read_text(encoding="utf-8"))
        source_root = ROOT.parent
        deployment_root = ROOT
        bound_files = (
            (
                source_root / manifest["onnx"]["source_path"],
                manifest["onnx"]["source_sha256"],
            ),
            (
                deployment_root / manifest["onnx"]["deployment_path"],
                manifest["onnx"]["deployment_sha256"],
            ),
            (
                source_root / manifest["source_experimental_run"]["run_receipt_path"],
                manifest["source_experimental_run"]["run_receipt_sha256"],
            ),
            (
                source_root / manifest["source_experimental_run"]["model_provenance_path"],
                manifest["source_experimental_run"]["model_provenance_sha256"],
            ),
            (
                source_root / manifest["source_experimental_run"]["calibration_path"],
                manifest["source_experimental_run"]["calibration_sha256"],
            ),
            (
                source_root / manifest["preprocess"]["reference_path"],
                manifest["preprocess"]["reference_sha256"],
            ),
        )
        for path, expected in bound_files:
            with self.subTest(path=str(path)):
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_preflight_implementation_has_no_device_open_or_network_client(self) -> None:
        source = (ROOT / "app" / "edge" / "preflight.py").read_text(encoding="utf-8")
        for forbidden in (
            "VideoCapture(",
            "serial.Serial(",
            "socket.socket(",
            "requests.get(",
            "requests.post(",
            "subprocess.run(",
            ".glob(",
            ".iterdir(",
            "hobot_dnn",
            "hbm_runtime",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_templates_have_no_remote_or_online_install_path(self) -> None:
        script_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((DEPLOY / "scripts").glob("*.sh"))
        ).lower()
        for forbidden in ("ssh ", "scp ", "curl ", "wget ", "apt "):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, script_text)
        installer = (
            DEPLOY / "scripts" / "install_cpu_venv_candidate.sh"
        ).read_text(encoding="utf-8").lower()
        self.assertIn("pip install", installer)
        self.assertIn("--no-index", installer)
        self.assertIn("--require-hashes", installer)
        runtime_text = "\n".join(
            (DEPLOY / "scripts" / name).read_text(encoding="utf-8").lower()
            for name in ("preflight.sh", "start_rootscope.sh")
        )
        self.assertNotIn("pip install", runtime_text)
        service = (DEPLOY / "systemd" / "rootscope-edge.service").read_text(
            encoding="utf-8"
        )
        llm = (
            DEPLOY / "systemd" / "rootscope-llm-readonly.service.disabled-template"
        ).read_text(encoding="utf-8")
        llm_launcher = (
            DEPLOY / "scripts" / "start_readonly_llm.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("PrivateDevices=true", service)
        self.assertIn("ConditionPathExists=@GATE_FILE@", llm)
        self.assertIn("IPAddressDeny=any", llm)
        self.assertIn("IPAddressAllow=localhost", llm)
        self.assertNotIn("User=", llm)
        self.assertNotIn("[Install]", llm)
        self.assertIn("--host 127.0.0.1", llm_launcher)
        self.assertIn("ROOTSCOPE_LLM_MANUAL_ACK", llm_launcher)
        self.assertNotIn("0.0.0.0", llm)


if __name__ == "__main__":
    unittest.main()
