import unittest

from app.action_v3.probe_depth import (
    CLASS_TO_DEPTH,
    compile_probe_depth_plan,
)


class ProbeDepthPlanTests(unittest.TestCase):
    def compile(self, plant_class: str, **overrides):
        values = {
            "plant_class": plant_class,
            "confidence": 0.95,
            "ood_hold": False,
            "evidence_fresh": True,
            "temporal_support": 3,
            "interlocks_clear": True,
            "manual_home_confirmed": True,
            "descent_available": True,
        }
        values.update(overrides)
        return compile_probe_depth_plan(**values)

    def test_frozen_class_mapping(self):
        self.assertEqual(
            CLASS_TO_DEPTH,
            {
                "non_target": (0, "hold", 0),
                "grass_clump": (1, "shallow", 1024),
                "low_shrub": (2, "medium", 1536),
                "young_tree": (3, "deep", 2048),
            },
        )

    def test_three_target_classes_compile_to_distinct_downward_presets(self):
        expected = {
            "grass_clump": (1, "DEPTH,1"),
            "low_shrub": (2, "DEPTH,2"),
            "young_tree": (3, "DEPTH,3"),
        }
        for plant_class, (level, command) in expected.items():
            with self.subTest(plant_class=plant_class):
                plan = self.compile(plant_class)
                self.assertTrue(plan.admitted)
                self.assertEqual(plan.requested_level, level)
                self.assertEqual(plan.command_preview, command)
                self.assertTrue(plan.manual_return_required)
                self.assertEqual(plan.calibration_state, "UNQUALIFIED_STEPS_ONLY")
                self.assertFalse(plan.payload()["authority"]["serial_write"])

    def test_non_target_is_no_motion(self):
        plan = self.compile("non_target")
        self.assertFalse(plan.admitted)
        self.assertEqual(plan.requested_level, 0)
        self.assertEqual(plan.configured_steps, 0)
        self.assertIsNone(plan.command_preview)
        self.assertIn("NON_TARGET_HOLD", plan.reason_codes)

    def test_every_safety_gate_fails_closed(self):
        cases = (
            ({"confidence": 0.5}, "VISION_CONFIDENCE_LOW"),
            ({"ood_hold": True}, "VISION_OOD_HOLD"),
            ({"evidence_fresh": False}, "EVIDENCE_STALE"),
            ({"temporal_support": 2}, "TEMPORAL_SUPPORT_LOW"),
            ({"interlocks_clear": False}, "INTERLOCK_ACTIVE"),
            ({"manual_home_confirmed": False}, "MANUAL_HOME_NOT_CONFIRMED"),
            ({"descent_available": False}, "DESCENT_ALREADY_USED"),
        )
        for overrides, reason in cases:
            with self.subTest(reason=reason):
                plan = self.compile("young_tree", **overrides)
                self.assertFalse(plan.admitted)
                self.assertIsNone(plan.command_preview)
                self.assertIn(reason, plan.reason_codes)

    def test_plan_hash_is_deterministic(self):
        self.assertEqual(
            self.compile("low_shrub").sha256,
            self.compile("low_shrub").sha256,
        )


if __name__ == "__main__":
    unittest.main()
