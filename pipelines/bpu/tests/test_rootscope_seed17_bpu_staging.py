from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


TOOLS = Path(__file__).resolve().parents[1]
PREP_PATH = TOOLS / "prepare_rootscope_seed17_bpu_staging.py"
AUDIT_PATH = TOOLS / "audit_rootscope_seed17_bpu_staging.py"
PREFLIGHT_PATH = TOOLS / "preflight_rootscope_seed17_bpu_staging.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREP = load_module("rootscope_bpu_prepare_tests", PREP_PATH)
AUDIT = load_module("rootscope_bpu_audit_tests", AUDIT_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def make_small_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 224, 224])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4])
    conv_w = helper.make_tensor("conv_w", TensorProto.FLOAT, [1, 3, 1, 1], [0.1, 0.2, 0.3])
    conv_b = helper.make_tensor("conv_b", TensorProto.FLOAT, [1], [0.0])
    gemm_w = helper.make_tensor("gemm_w", TensorProto.FLOAT, [1, 4], [0.1, 0.2, 0.3, 0.4])
    gemm_b = helper.make_tensor("gemm_b", TensorProto.FLOAT, [4], [0.0, 0.0, 0.0, 0.0])
    nodes = [
        helper.make_node("Conv", ["image", "conv_w", "conv_b"], ["conv"]),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node("MaxPool", ["relu"], ["pool"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Add", ["pool", "pool"], ["added"]),
        helper.make_node("AveragePool", ["added"], ["average"], kernel_shape=[112, 112]),
        helper.make_node("Flatten", ["average"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "gemm_w", "gemm_b"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "small_rootscope_resnet_contract",
        [image],
        [logits],
        [conv_w, conv_b, gemm_w, gemm_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def rewrite_sums(staging: Path) -> None:
    files = {
        path.relative_to(staging).as_posix(): sha256(path)
        for path in sorted(staging.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.name != "STAGING_SHA256SUMS"
    }
    (staging / "STAGING_SHA256SUMS").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in files.items()),
        encoding="utf-8",
    )


class MiniFixture:
    def __init__(self, root: Path) -> None:
        self.workspace = root / "adventurex"
        self.xrd_root = root
        self.pack = self.workspace / PREP.PACK_REL
        self.run = self.workspace / PREP.RUN_REL
        self.output = self.workspace / "output/mini_bpu_staging"
        self.workspace.mkdir()

        generator_copy = self.workspace / PREP.GENERATOR_REL if hasattr(PREP, "GENERATOR_REL") else self.workspace / "tools/bpu/prepare_rootscope_seed17_bpu_staging.py"
        generator_copy.parent.mkdir(parents=True)
        shutil.copyfile(PREP_PATH, generator_copy)

        rows: list[dict] = []
        sizes = [(320, 240), (240, 320), (300, 300), (400, 250)]
        for index, (class_id, size) in enumerate(zip(PREP.CLASS_ORDER, sizes, strict=True)):
            relative = f"images/{class_id}/{class_id}.jpg"
            target = self.pack / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", size, (30 + index * 40, 80 + index * 20, 140 - index * 20))
            image.save(target, quality=91)
            digest = sha256(target)
            rows.append(
                {
                    "asset": f"synthetic:{class_id}",
                    "class_id": class_id,
                    "experimental_split_suggestion": PREP.TRAIN_ROLE,
                    "filename": relative,
                    "copied_image_sha256": digest,
                    "source_image_sha256": digest,
                    "source_group": f"source:{class_id}",
                    "creator_group": f"creator:{class_id}",
                    "source_dataset": "SYNTHETIC",
                    "pageid": index + 1,
                }
            )
        self.rows = rows
        write_jsonl(self.pack / "manifest.jsonl", rows)
        write_json(
            self.pack / "receipt.json",
            {
                "manifest_sha256": sha256(self.pack / "manifest.jsonl"),
                "human_reviewed": False,
                "rights_approved": False,
                "data_locked": False,
                "training_eligible": False,
            },
        )

        onnx_path = self.run / PREP.SOURCE_ONNX_REL
        make_small_onnx(onnx_path)
        write_json(
            self.run / "seed_00017/model_provenance.json",
            {
                "architecture": "torchvision.resnet18",
                "class_order": list(PREP.CLASS_ORDER),
                "input_shape": PREP.INPUT_SHAPE,
            },
        )
        write_json(
            self.run / "run_receipt.json",
            {
                "status": "MACHINE_CURATED_EXPERIMENTAL_MODEL_NOT_QUALIFIED",
                "model_candidate": False,
                "model_qualified": False,
                "x5_ready": False,
                "bpu_compiled": False,
                "selected_seed": {
                    "seed": 17,
                    "artifacts": {"onnx": PREP.SOURCE_ONNX_REL.as_posix()},
                },
            },
        )

        for relative in PREP.XRD_REFERENCE_PATHS.values():
            target = self.xrd_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"reference:{relative.as_posix()}\n", encoding="utf-8")


class BpuStagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = MiniFixture(Path(self.temporary.name))
        self.stack = contextlib.ExitStack()
        class_counts = {name: 1 for name in PREP.CLASS_ORDER}
        expected_manifest = sha256(self.fixture.pack / "manifest.jsonl")
        expected_run = sha256(self.fixture.run / "run_receipt.json")
        expected_onnx = sha256(self.fixture.run / PREP.SOURCE_ONNX_REL)
        for module in (PREP, AUDIT):
            self.stack.enter_context(mock.patch.object(module, "WORKSPACE", self.fixture.workspace))
            self.stack.enter_context(mock.patch.object(module, "XRD_ROOT", self.fixture.xrd_root))
            self.stack.enter_context(mock.patch.object(module, "EXPECTED_PACK_MANIFEST_SHA256", expected_manifest))
            self.stack.enter_context(mock.patch.object(module, "EXPECTED_RUN_RECEIPT_SHA256", expected_run))
            self.stack.enter_context(mock.patch.object(module, "EXPECTED_ONNX_SHA256", expected_onnx))
            self.stack.enter_context(mock.patch.object(module, "EXPECTED_TRAIN_CLASS_COUNTS", class_counts))
            self.stack.enter_context(mock.patch.object(module, "EXPECTED_TRAIN_SOURCE_COUNT", 4))
            self.stack.enter_context(mock.patch.object(module, "SAMPLES_PER_CLASS", 1))
            self.stack.enter_context(mock.patch.object(module, "EXPECTED_SAMPLE_COUNT", 4))
        PREP.build_staging(self.fixture.output)

    def tearDown(self) -> None:
        self.stack.close()
        self.temporary.cleanup()

    def test_valid_train_only_mini_staging_passes(self) -> None:
        report = AUDIT.audit_staging(self.fixture.output)
        self.assertEqual("PASS", report["status"])
        self.assertEqual(4, report["calibration"]["sample_count"])
        self.assertEqual(4, report["calibration"]["unique_train_sources_covered"])
        self.assertEqual(0, report["calibration"]["non_train_sources"])
        self.assertFalse(report["formal_flags"]["bpu_compiled"])
        self.assertFalse(report["formal_flags"]["x5_ready"])
        self.assertEqual("rgb", report["mapper_config"]["runtime_input_type"])
        self.assertEqual("NCHW", report["mapper_config"]["runtime_layout"])
        self.assertEqual({"image": "ddr"}, report["mapper_config"]["input_source"])
        contract = json.loads(
            (self.fixture.output / "preprocess_contract.json").read_text(encoding="utf-8")
        )
        runtime = contract["target_runtime"]
        self.assertEqual("ddr", runtime["mapper_input_source"])
        self.assertEqual("uint8", runtime["host_tensor_contract"]["dtype"])
        self.assertFalse(runtime["host_tensor_contract"]["normalization_on_host"])

    def test_calibration_payload_tamper_fails_after_sums_are_rewritten(self) -> None:
        target = next((self.fixture.output / "calibration_data_rgb_f32").glob("*.rgb"))
        payload = bytearray(target.read_bytes())
        payload[0] ^= 1
        target.write_bytes(payload)
        rewrite_sums(self.fixture.output)
        with self.assertRaisesRegex(AUDIT.AuditError, "payload reproduction failed"):
            AUDIT.audit_staging(self.fixture.output)

    def test_false_formal_flag_is_fail_closed_even_when_rehashed(self) -> None:
        receipt_path = self.fixture.output / "staging_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["formal_flags"]["bpu_compiled"] = True
        write_json(receipt_path, receipt)
        rewrite_sums(self.fixture.output)
        with self.assertRaisesRegex(AUDIT.AuditError, "bpu_compiled"):
            AUDIT.audit_staging(self.fixture.output)

    def test_builder_rejects_cross_partition_creator_overlap(self) -> None:
        rows = list(self.fixture.rows)
        extra = dict(rows[0])
        extra["asset"] = "synthetic:validation"
        extra["experimental_split_suggestion"] = "EXPERIMENTAL_VAL_SUGGESTION"
        rows.append(extra)
        write_jsonl(self.fixture.pack / "manifest.jsonl", rows)
        receipt = json.loads((self.fixture.pack / "receipt.json").read_text(encoding="utf-8"))
        receipt["manifest_sha256"] = sha256(self.fixture.pack / "manifest.jsonl")
        write_json(self.fixture.pack / "receipt.json", receipt)
        with mock.patch.object(
            PREP, "EXPECTED_PACK_MANIFEST_SHA256", sha256(self.fixture.pack / "manifest.jsonl")
        ):
            with self.assertRaisesRegex(PREP.StagingError, "cross-partition"):
                PREP.build_staging(self.fixture.workspace / "output/cross_partition_reject")

    def test_auditor_is_independent_of_builder(self) -> None:
        source = AUDIT_PATH.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\s*(?:from|import)\s+prepare_rootscope_seed17")

    def test_preflight_source_has_no_process_launcher(self) -> None:
        source = PREFLIGHT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("os.system", source)
        self.assertNotIn("Popen(", source)
        self.assertIn('"docker_daemon_state": "NOT_QUERIED"', source)

    def test_output_outside_adventurex_is_rejected(self) -> None:
        with self.assertRaisesRegex(PREP.StagingError, "under AdventureX"):
            PREP.build_staging(self.fixture.xrd_root / "outside")


if __name__ == "__main__":
    unittest.main()
