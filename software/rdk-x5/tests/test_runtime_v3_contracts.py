from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.action_v3.contracts import ActionContractCompiler, PhysicalReceiptCompiler
from app.runtime_v3.hbm_runtime_adapter import (
    HbmRuntimeContractError,
    PersistentHbmR7Adapter,
)
from app.runtime_v3.resource_broker import (
    ResourceBroker,
    ResourceSnapshot,
    RuntimePhase,
    Workload,
)


class _DType:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeHbm:
    def __init__(self, _path: str) -> None:
        self.model_names = ["rootscope_r7"]
        self.input_names = {"rootscope_r7": ["image"]}
        self.output_names = {"rootscope_r7": ["logits"]}
        self.input_shapes = {"rootscope_r7": {"image": [1, 3, 224, 224]}}
        self.output_shapes = {"rootscope_r7": {"logits": [1, 4, 1, 1]}}
        self.input_dtypes = {"rootscope_r7": {"image": _DType("RGB")}}
        self.output_dtypes = {"rootscope_r7": {"logits": _DType("F32")}}
        self.calls = 0

    def run(self, tensor: np.ndarray):
        self.calls += 1
        assert tensor.flags.c_contiguous
        return {
            "rootscope_r7": {
                "logits": np.array([[[[1]], [[2]], [[3]], [[4]]]], dtype=np.float32)
            }
        }


class RuntimeV3ContractTests(unittest.TestCase):
    def _model(self) -> tuple[tempfile.TemporaryDirectory[str], Path, str]:
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "r7.bin"
        path.write_bytes(b"test-hbm")
        os.chmod(path, 0o400)
        return temp, path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_persistent_hbm_candidate_is_zero_authority(self) -> None:
        temp, path, digest = self._model()
        self.addCleanup(temp.cleanup)
        adapter = PersistentHbmR7Adapter(
            path,
            digest,
            input_policy="RAW_UINT8",
            runtime_factory=_FakeHbm,
        )
        tensor = np.zeros((1, 3, 224, 224), dtype=np.uint8)
        first = adapter.infer_uint8(tensor)
        second = adapter.infer_uint8(tensor)
        self.assertEqual(first["top1_index"], 3)
        self.assertEqual(second["inference_count_since_load"], 2)
        self.assertFalse(first["qualification"]["selected_for_runtime"])
        self.assertTrue(all(value is False for value in first["authority"].values()))
        self.assertEqual(adapter.runtime.calls, 2)

    def test_centered_policy_is_explicit_and_hash_bound(self) -> None:
        temp, path, digest = self._model()
        self.addCleanup(temp.cleanup)
        adapter = PersistentHbmR7Adapter(
            path,
            digest,
            input_policy="RGB128_CENTERED_INT8",
            runtime_factory=_FakeHbm,
        )
        report = adapter.infer_uint8(
            np.full((1, 3, 224, 224), 128, dtype=np.uint8)
        )
        self.assertEqual(report["runtime_tensor_dtype"], "int8")
        with self.assertRaises(HbmRuntimeContractError):
            PersistentHbmR7Adapter(
                path,
                "0" * 64,
                input_policy="RAW_UINT8",
                runtime_factory=_FakeHbm,
            )

    def test_resource_broker_excludes_deep_model_during_action(self) -> None:
        broker = ResourceBroker()
        snapshot = ResourceSnapshot(2200, 220, 55, 1.0)
        deep = broker.decide(
            Workload.DEEP_LLM, RuntimePhase.IRRIGATION_CRITICAL, snapshot
        )
        self.assertFalse(deep.admitted)
        self.assertIn(
            "OPTIONAL_MODEL_EXCLUDED_DURING_IRRIGATION", deep.reason_codes
        )
        cpu = broker.decide(
            Workload.CPU_VISION, RuntimePhase.IRRIGATION_CRITICAL, snapshot
        )
        self.assertTrue(cpu.admitted)

    def test_action_contract_holds_ood_and_receipt_needs_three_signals(self) -> None:
        compiler = ActionContractCompiler(
            release_sha256="1" * 64,
            config_sha256="2" * 64,
        )
        held = compiler.compile(
            contract_id="contract-1",
            sequence=7,
            boot_id="boot-1",
            evidence_root_sha256="3" * 64,
            plant_class="grass_clump",
            plant_confidence=0.9,
            ood_hold=True,
            target_zone="zone-1",
            proposed_volume_ml=20.0,
            evidence_fresh=True,
            interlocks_clear=True,
            reason_codes=["MODEL_PROPOSAL"],
        )
        self.assertEqual(held.proposed_volume_ml, 0)
        self.assertFalse(held.payload()["authority"]["pump_command"])

        action = compiler.compile(
            contract_id="contract-2",
            sequence=8,
            boot_id="boot-1",
            evidence_root_sha256="4" * 64,
            plant_class="low_shrub",
            plant_confidence=0.95,
            ood_hold=False,
            target_zone="zone-2",
            proposed_volume_ml=25.0,
            evidence_fresh=True,
            interlocks_clear=True,
            reason_codes=["FUSED_EVIDENCE_PASS"],
        )
        receipt_compiler = PhysicalReceiptCompiler()
        bad = receipt_compiler.compile(
            receipt_id="receipt-bad",
            contract=action,
            device_identity_sha256="5" * 64,
            ack_boot_id="boot-1",
            ack_sequence=8,
            ack_payload_sha256="6" * 64,
            ack_fresh=True,
            expected_mass_loss_g=25,
            observed_mass_loss_g=25,
            target_wetting_coverage=0.30,
            neighbor_spill_ratio=0.20,
        )
        self.assertFalse(bad.completed)
        good = receipt_compiler.compile(
            receipt_id="receipt-good",
            contract=action,
            device_identity_sha256="5" * 64,
            ack_boot_id="boot-1",
            ack_sequence=8,
            ack_payload_sha256="6" * 64,
            ack_fresh=True,
            expected_mass_loss_g=25,
            observed_mass_loss_g=24,
            target_wetting_coverage=0.30,
            neighbor_spill_ratio=0.02,
        )
        self.assertTrue(good.completed)
        self.assertFalse(good.payload()["authority"]["physical_completion"])


if __name__ == "__main__":
    unittest.main()
