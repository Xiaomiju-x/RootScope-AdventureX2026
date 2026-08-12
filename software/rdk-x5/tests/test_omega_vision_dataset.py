from __future__ import annotations

import unittest

from training.omega_vision.build_evidence import (
    CREATOR,
    PRINT,
    TRAIN,
    VAL,
    audit_dataset,
)


class OmegaVisionDatasetTests(unittest.TestCase):
    def test_actual_78_image_pack_is_byte_bound_and_zero_authority(self) -> None:
        rows, audit = audit_dataset()
        self.assertEqual(78, len(rows))
        self.assertEqual("PASS_MACHINE_CURATED_PROVISIONAL_IDENTITY_ONLY", audit["status"])
        self.assertEqual(
            {TRAIN: 55, VAL: 9, PRINT: 6, CREATOR: 8},
            audit["role_counts"],
        )
        self.assertEqual(78, audit["byte_hash_verified_count"])
        self.assertFalse(audit["human_reviewed"])
        self.assertFalse(audit["formal_split_assigned"])
        self.assertFalse(audit["training_eligible"])

    def test_no_source_group_leaks_across_any_role(self) -> None:
        _, audit = audit_dataset()
        self.assertTrue(all(value == 0 for value in audit["source_group_intersection_counts"].values()))


if __name__ == "__main__":
    unittest.main()
