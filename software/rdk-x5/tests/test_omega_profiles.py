from __future__ import annotations

import unittest
from pathlib import Path

from app.omega_runtime.contracts import RuntimeMode
from app.omega_runtime.profiles import EdgeProfileRegistry, ResourceSnapshot


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "configs" / "omega" / "edge_profiles.v1.json"


class OmegaProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = EdgeProfileRegistry.from_file(PROFILES)

    def test_unqualified_bpu_falls_back_to_safe_cpu(self) -> None:
        capsule = self.registry.select(
            "LOCAL_HYBRID",
            ResourceSnapshot(1800, 60.0, False, True, False),
            runtime_mode=RuntimeMode.SIMULATION,
            release_id="rootscope-omega-v3-alpha",
        )
        self.assertEqual(capsule.profile, "SAFE_CPU")
        self.assertIn("BPU_MODEL_NOT_QUALIFIED", capsule.fallback_reasons)
        self.assertFalse(capsule.bpu_model_qualified)
        self.assertFalse(capsule.local_llm_active)

    def test_low_memory_and_temperature_both_recorded(self) -> None:
        capsule = self.registry.select(
            "LOCAL_HYBRID",
            ResourceSnapshot(256, 90.0, True, True, False),
            runtime_mode=RuntimeMode.REPLAY,
            release_id="rootscope-omega-v3-alpha",
        )
        self.assertEqual(capsule.profile, "SAFE_CPU")
        self.assertIn("MEMORY_RESERVE_GATE", capsule.fallback_reasons)
        self.assertIn("THERMAL_GATE", capsule.fallback_reasons)

    def test_qualified_local_hybrid_is_explicit(self) -> None:
        capsule = self.registry.select(
            "LOCAL_HYBRID",
            ResourceSnapshot(1800, 60.0, True, True, False),
            runtime_mode=RuntimeMode.REPLAY,
            release_id="rootscope-omega-v3-alpha",
        )
        self.assertEqual(capsule.profile, "LOCAL_HYBRID")
        self.assertTrue(capsule.bpu_model_qualified)
        self.assertTrue(capsule.local_llm_active)
        self.assertFalse(capsule.remote_shadow_active)


if __name__ == "__main__":
    unittest.main()
