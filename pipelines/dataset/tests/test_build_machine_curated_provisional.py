from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


DATASET_TOOLS = Path(__file__).resolve().parents[1]
if str(DATASET_TOOLS) not in sys.path:
    sys.path.insert(0, str(DATASET_TOOLS))
SCRIPT = DATASET_TOOLS / "build_machine_curated_provisional.py"
SPEC = importlib.util.spec_from_file_location("build_machine_curated_provisional", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MachineCuratedProvisionalTests(unittest.TestCase):
    def test_status_and_authority_are_fail_closed(self) -> None:
        self.assertIn("NOT_HUMAN_REVIEWED", MODULE.STATUS)
        self.assertIn("NOT_A1", MODULE.STATUS)
        self.assertIn("NOT_DATA_LOCKED", MODULE.STATUS)
        self.assertTrue(MODULE.AUTHORITY_FALSE)
        self.assertTrue(all(value is False for value in MODULE.AUTHORITY_FALSE.values()))
        fields = MODULE.machine_status_fields()
        for key in ("data_locked", "human_reviewed", "print_eligible", "rights_approved", "training_eligible"):
            self.assertIs(fields[key], False)
        self.assertEqual("UNASSIGNED_DO_NOT_TRAIN", fields["split"])

    def test_unknown_selection_is_deterministic_and_creator_capped(self) -> None:
        strict = [
            {"pageid": index, "final_label": "UNKNOWN_CANDIDATE", "source_group": f"commons:{index}"}
            for index in range(1, 8)
        ]
        e0 = {
            index: {
                "pageid": index,
                "creator_group": "creator:shared" if index in (1, 2, 3) else f"creator:{index}",
                "source_group": f"commons:{index}",
            }
            for index in range(1, 8)
        }
        first, decisions_first = MODULE.select_unknown_records(strict, e0, limit=4)
        second, decisions_second = MODULE.select_unknown_records(strict, e0, limit=4)
        self.assertEqual(first, second)
        self.assertEqual(decisions_first, decisions_second)
        self.assertLessEqual(len(first), 4)
        creators = [row["creator_group"] for row in first]
        self.assertEqual(len(creators), len(set(creators)))

    def test_unknown_visual_exclusion_is_fail_closed(self) -> None:
        strict = [{"pageid": 10, "final_label": "UNKNOWN_CANDIDATE", "source_group": "commons:10"}]
        e0 = {10: {"pageid": 10, "creator_group": "creator:10", "source_group": "commons:10"}}
        selected, decisions = MODULE.select_unknown_records(
            strict, e0, limit=1, exclusions={10: "obvious target plant"}
        )
        self.assertEqual([], selected)
        self.assertEqual("EXCLUDE_VISUAL_UNKNOWN_GATE", decisions[0]["disposition"])

    def test_creator_group_print_holdout_never_enters_train_or_validation(self) -> None:
        print_groups = {"creator:held"}
        self.assertEqual(
            "CREATOR_GROUP_HOLDOUT_NOT_TRAIN",
            MODULE.experimental_role("creator:held", print_groups),
        )
        role = MODULE.experimental_role("creator:free", print_groups)
        self.assertIn(role, {"EXPERIMENTAL_TRAIN_SUGGESTION", "EXPERIMENTAL_VAL_SUGGESTION"})

    def test_historical_bad_ids_are_recorded_without_substitution(self) -> None:
        self.assertEqual([137981651, 183947751, 227016131], MODULE.HISTORICAL_UNRESOLVED_IDS)
        requested = {
            pageid
            for request in MODULE.TARGET_REQUESTS.values()
            for pageid in request["pageids"]
        }
        self.assertTrue(set(MODULE.HISTORICAL_UNRESOLVED_IDS).isdisjoint(requested))

    def test_conservative_visual_exclusions_are_explicit(self) -> None:
        self.assertIn(38234300, MODULE.TARGET_REQUESTS["grass_clump"]["visual_exclusions"])
        young_excluded = MODULE.TARGET_REQUESTS["young_tree"]["visual_exclusions"]
        self.assertEqual(
            {133359583, 137881651, 18394775, 22701613, 25062664, 59265209},
            set(young_excluded),
        )


if __name__ == "__main__":
    unittest.main()
