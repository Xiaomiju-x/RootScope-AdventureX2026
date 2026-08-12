from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


DATASET_TOOLS = Path(__file__).resolve().parents[1]
if str(DATASET_TOOLS) not in sys.path:
    sys.path.insert(0, str(DATASET_TOOLS))
SCRIPT = DATASET_TOOLS / "build_machine_curated_provisional_v2.py"
SPEC = importlib.util.spec_from_file_location("build_machine_curated_provisional_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fake_record(pageid: int, class_id: str, creator: str, *, print_holdout: bool = False) -> dict:
    return {
        "pageid": pageid,
        "class_id": class_id,
        "creator_group": creator,
        "print_holdout_candidate": print_holdout,
    }


class MachineCuratedProvisionalV2Tests(unittest.TestCase):
    def test_v2_is_independent_from_v1(self) -> None:
        self.assertNotEqual(MODULE.OUTPUT_NAME, MODULE.V1_NAME)
        self.assertEqual("rootscope_machine_curated_provisional_v2", MODULE.OUTPUT_NAME)
        self.assertIn("NOT_HUMAN_REVIEWED", MODULE.STATUS)
        self.assertIn("NOT_A1", MODULE.STATUS)
        self.assertTrue(all(value is False for value in MODULE.authority_false().values()))

    def test_print_creator_group_is_held_out(self) -> None:
        rows = [
            fake_record(1, "grass_clump", "creator:held", print_holdout=True),
            fake_record(2, "grass_clump", "creator:held"),
            fake_record(3, "grass_clump", "creator:free-a"),
            fake_record(4, "grass_clump", "creator:free-b"),
        ]
        roles, _attainment = MODULE.plan_group_roles(
            rows,
            train_minimums={"grass_clump": 1},
            val_minimums={"grass_clump": 1},
        )
        self.assertEqual("CREATOR_GROUP_HOLDOUT_NOT_TRAIN", roles["creator:held"])

    def test_group_planner_meets_feasible_minimums(self) -> None:
        rows = []
        pageid = 1
        for class_id in ("grass_clump", "low_shrub", "young_tree", "unknown"):
            for index in range(8):
                rows.append(fake_record(pageid, class_id, f"creator:{class_id}:{index}"))
                pageid += 1
        roles, attainment = MODULE.plan_group_roles(
            rows,
            train_minimums={class_id: 5 for class_id in ("grass_clump", "low_shrub", "young_tree", "unknown")},
            val_minimums={class_id: 2 for class_id in ("grass_clump", "low_shrub", "young_tree", "unknown")},
        )
        self.assertTrue(all(values["both_met"] for values in attainment.values()))
        self.assertIn("EXPERIMENTAL_VAL_SUGGESTION", set(roles.values()))

    def test_group_planner_reports_infeasible_validation(self) -> None:
        rows = [
            fake_record(1, "young_tree", "creator:a"),
            fake_record(2, "young_tree", "creator:b"),
        ]
        _roles, attainment = MODULE.plan_group_roles(
            rows,
            train_minimums={"young_tree": 2},
            val_minimums={"young_tree": 1},
        )
        self.assertFalse(attainment["young_tree"]["both_met"])
        self.assertEqual(1, attainment["young_tree"]["val_deficit"])

    def test_young_supplement_scope_is_explicit(self) -> None:
        self.assertEqual(6, len(MODULE.SUPPLEMENT_REQUESTS["young_tree"]))
        self.assertIn(108010572, MODULE.SUPPLEMENT_REQUESTS["young_tree"])
        self.assertTrue(
            set(MODULE.SUPPLEMENT_REQUESTS["young_tree"]).issubset(
                MODULE.SUPPLEMENT_VISUAL_EXCLUSIONS
            )
        )
        self.assertEqual(5, MODULE.TRAIN_MINIMUMS["young_tree"])
        self.assertEqual(2, MODULE.VAL_MINIMUMS["young_tree"])

    def test_known_cross_class_grass_candidate_is_excluded(self) -> None:
        self.assertIn(21981205, MODULE.SUPPLEMENT_VISUAL_EXCLUSIONS)
        self.assertIn("class-mismatch", MODULE.SUPPLEMENT_VISUAL_EXCLUSIONS[21981205])

    def test_hamming_distance(self) -> None:
        self.assertEqual(0, MODULE.hamming64("0000000000000000", "0000000000000000"))
        self.assertEqual(64, MODULE.hamming64("0000000000000000", "ffffffffffffffff"))


if __name__ == "__main__":
    unittest.main()
