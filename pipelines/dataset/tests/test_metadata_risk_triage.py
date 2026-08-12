from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DATASET = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DATASET))

import metadata_risk_triage as triage  # noqa: E402


def fixture_row(
    index: int,
    *,
    class_hint: str,
    query: str,
    title: str,
    species_hint: str,
    creator_group: str,
) -> dict[str, str]:
    return {
        "asset": f"fixture:{index}",
        "source_group": f"fixture-source:{index}",
        "local_path": f"images/{class_hint}/{index}.jpg",
        "class_hint": class_hint,
        "acquisition_query": query,
        "title": title,
        "species_hint": species_hint,
        "creator_group": creator_group,
    }


class MetadataRiskTriageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.review = self.root / "review"
        self.review.mkdir()
        self.queue = self.review / "candidate_review_queue.jsonl"
        self.output = self.review / triage.OUTPUT_DIR_NAME

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_queue(self, rows: list[dict[str, str]]) -> None:
        self.queue.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _risk_ids(record: dict) -> set[str]:
        return {flag["id"] for flag in record["risk_flags"]}

    def test_lexical_signals_are_metadata_only(self) -> None:
        rows = [
            fixture_row(1, class_hint="grass_clump", query="Grass", title="Grass seed head close-up.jpg", species_hint="Grass", creator_group="c1"),
            fixture_row(2, class_hint="low_shrub", query="Shrub", title="Shrub herbarium specimen sheet.jpg", species_hint="Shrub", creator_group="c2"),
            fixture_row(3, class_hint="grass_clump", query="Grass", title="Distribution map botanical plate.jpg", species_hint="Grass", creator_group="c3"),
            fixture_row(4, class_hint="young_tree", query="Tree", title="Old tree stump in woodland.jpg", species_hint="Tree", creator_group="c4"),
            fixture_row(5, class_hint="low_shrub", query="Shrub", title="Shrub in desert landscape panorama.jpg", species_hint="Shrub", creator_group="c5"),
        ]
        records = triage.analyze_rows(rows)
        self.assertIn("detail_crop_signal", self._risk_ids(records[0]))
        self.assertIn("herbarium_specimen_signal", self._risk_ids(records[1]))
        self.assertIn("map_text_illustration_signal", self._risk_ids(records[2]))
        self.assertIn("mature_dead_tree_signal", self._risk_ids(records[3]))
        self.assertIn("landscape_many_subjects_signal", self._risk_ids(records[3]))
        self.assertIn("landscape_many_subjects_signal", self._risk_ids(records[4]))
        for record in records:
            self.assertTrue(record["metadata_only"])
            self.assertFalse(record["visual_truth_established"])

    def test_young_tree_age_support_and_unverified_are_distinct(self) -> None:
        rows = [
            fixture_row(1, class_hint="young_tree", query="Acacia", title="Young Acacia sapling.jpg", species_hint="Acacia", creator_group="c1"),
            fixture_row(2, class_hint="young_tree", query="Acacia", title="Acacia tortilis.jpg", species_hint="Acacia", creator_group="c2"),
        ]
        records = triage.analyze_rows(rows)
        self.assertNotIn("young_tree_age_unverified", self._risk_ids(records[0]))
        self.assertEqual({"young_tree_title_support"}, {x["id"] for x in records[0]["support_signals"]})
        self.assertIn("young_tree_age_unverified", self._risk_ids(records[1]))

    def test_unknown_is_context_not_automatic_high_risk(self) -> None:
        row = fixture_row(
            1,
            class_hint="unknown",
            query="Gravel",
            title="Five little stones.jpg",
            species_hint="negative:rocks",
            creator_group="c1",
        )
        record = triage.analyze_rows([row])[0]
        self.assertEqual("NO_OBVIOUS_METADATA_RISK_SIGNAL", record["risk_priority"])
        self.assertEqual({"unknown_acquisition_bucket"}, {x["id"] for x in record["context_flags"]})

    def test_creator_series_concentration_is_flagged(self) -> None:
        rows = [
            fixture_row(index, class_hint="grass_clump", query="Grass", title=f"Grass {index}.jpg", species_hint="Grass", creator_group="series")
            for index in range(1, 9)
        ]
        records = triage.analyze_rows(rows)
        self.assertTrue(all("creator_series_concentration" in self._risk_ids(row) for row in records))
        evidence = records[0]["risk_flags"][0]["evidence"]
        self.assertEqual("8", evidence["same_class_query_count"])

    def test_output_boundary_rejects_any_other_directory(self) -> None:
        self._write_queue(
            [fixture_row(1, class_hint="unknown", query="Sand", title="Sand.jpg", species_hint="negative:bare_sand", creator_group="c1")]
        )
        with self.assertRaises(ValueError):
            triage.run(self.queue, self.review / "human_decisions")
        with self.assertRaises(ValueError):
            triage.run(self.queue, self.review / "somewhere_else")

    def test_run_is_deterministic_and_has_no_authority(self) -> None:
        self._write_queue(
            [
                fixture_row(1, class_hint="grass_clump", query="Grass", title="Grass close view.jpg", species_hint="Grass", creator_group="c1"),
                fixture_row(2, class_hint="unknown", query="Hands", title="Hand.jpg", species_hint="negative:hand", creator_group="c2"),
            ]
        )
        first = triage.run(self.queue, self.output)
        first_bytes = {path.name: path.read_bytes() for path in self.output.iterdir()}
        second = triage.run(self.queue, self.output)
        second_bytes = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.assertEqual(first, second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertTrue(all(value is False for value in first["authority"].values()))
        self.assertEqual(2, first["rows"])
        for name, digest in first["artifacts_sha256"].items():
            self.assertEqual(hashlib.sha256(first_bytes[name]).hexdigest(), digest)

    def test_load_queue_fails_on_duplicate_asset(self) -> None:
        row = fixture_row(1, class_hint="grass_clump", query="Grass", title="Grass.jpg", species_hint="Grass", creator_group="c1")
        self._write_queue([row, row])
        with self.assertRaisesRegex(ValueError, "duplicate asset"):
            triage.load_queue(self.queue)


if __name__ == "__main__":
    unittest.main()
