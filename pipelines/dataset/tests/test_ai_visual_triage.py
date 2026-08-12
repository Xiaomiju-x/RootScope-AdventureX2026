from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "ai_visual_triage.py"
POLICY_PATH = MODULE_PATH.with_name("ai_visual_triage_policy_v1.json")
SPEC = importlib.util.spec_from_file_location("rootscope_ai_visual_triage", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


CLASSES = ["grass_clump", "low_shrub", "young_tree", "unknown"]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class AITriageFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.staging = root / "staging"
        self.review = self.staging / "review"
        self.queue = self.review / "candidate_review_queue.jsonl"
        self.manifest = self.staging / "manifest.jsonl"
        self.summary = self.review / "review_queue_summary.json"
        self.integrity = self.staging / "integrity_audit.json"
        self.contract = root / "class_contract.json"
        self.policy = root / "ai_visual_triage_policy_v1.json"
        self.model = root / "fixture-model.bin"
        self.model_outputs = root / "model_outputs.jsonl"
        self.output = self.review / "ai_triage_fixture"
        self.human_dir = self.review / "human_decisions"
        self.model.write_bytes(b"fixture-local-model-artifact-v1")
        self.rows = [
            self._make_candidate(101, "grass_clump", (54, 129, 55)),
            self._make_candidate(102, "low_shrub", (126, 93, 49)),
            self._make_candidate(103, "unknown", (174, 161, 126)),
        ]
        self.manifest_rows = [item[0] for item in self.rows]
        self.queue_rows = [item[1] for item in self.rows]
        write_jsonl(self.manifest, self.manifest_rows)
        write_jsonl(self.queue, self.queue_rows)
        queue_sha = sha256_bytes(self.queue.read_bytes())
        manifest_sha = sha256_bytes(self.manifest.read_bytes())
        write_json(
            self.summary,
            {
                "schema_version": "rootscope.wikimedia_human_review_queue_summary.v1",
                "candidate_count": len(self.queue_rows),
                "inputs": {"staging_manifest_sha256": manifest_sha},
                "outputs": {"candidate_review_queue.jsonl": queue_sha},
            },
        )
        write_json(
            self.integrity,
            {
                "schema_version": "rootscope.wikimedia_staging_integrity_audit.v2",
                "result": "PASS_STAGING_INTEGRITY_NOT_TRAIN_READY",
                "failure_count": 0,
                "failures": [],
                "manifest_sha256": manifest_sha,
                "image_constraints": {
                    "allowed_mime": ["image/jpeg", "image/png"],
                    "minimum_downloaded_side": 32,
                },
            },
        )
        write_json(
            self.contract,
            {
                "schema_version": "2.0.0",
                "class_order": CLASSES,
            },
        )
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        policy["production_input_roots"] = {
            "candidate_review_queue_sha256": queue_sha,
            "staging_manifest_sha256": manifest_sha,
            "review_queue_summary_sha256": sha256_bytes(self.summary.read_bytes()),
            "integrity_audit_sha256": sha256_bytes(self.integrity.read_bytes()),
            "class_contract_sha256": sha256_bytes(self.contract.read_bytes()),
        }
        policy["expected_candidate_count"] = len(self.queue_rows)
        policy["expected_acquisition_hint_counts"] = {
            "grass_clump": 1,
            "low_shrub": 1,
            "young_tree": 0,
            "unknown": 1,
        }
        policy["image_contract"]["minimum_width"] = 32
        policy["image_contract"]["minimum_height"] = 32
        write_json(self.policy, policy)
        self.policy_value = policy
        self._write_scores()
        self.human_dir.mkdir(parents=True)
        (self.human_dir / "decision_journal.jsonl").write_bytes(b"human-journal-sentinel\n")
        write_json(self.human_dir / "human_review_receipt.json", {"sentinel": "human-only"})

    def _make_candidate(
        self, pageid: int, class_hint: str, color: tuple[int, int, int]
    ) -> tuple[dict, dict]:
        relative = Path("images") / class_hint / f"candidate_{pageid}.png"
        absolute = self.staging / relative
        absolute.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (64, 64), color)
        for offset in range(8):
            image.putpixel(
                ((pageid + offset * 5) % 64, (pageid + offset * 7) % 64),
                tuple(255 - value for value in color),
            )
        image.save(absolute, format="PNG")
        payload = absolute.read_bytes()
        sha = sha256_bytes(payload)
        source_group = f"commons:{pageid}"
        source_page = f"https://commons.wikimedia.org/wiki/File:fixture_{pageid}.png"
        download_url = f"https://upload.wikimedia.org/wikipedia/commons/fixture_{pageid}.png"
        artist = f"Fixture Author {pageid}"
        manifest = {
            "schema_version": "rootscope.wikimedia_candidate.v1",
            "class_id": class_hint,
            "domain": "natural_web_candidate",
            "split": "UNASSIGNED_DO_NOT_TRAIN",
            "review_status": "pending_human_visual_and_license_review",
            "training_eligible": False,
            "print_eligible": False,
            "source_provider": "Wikimedia Commons",
            "source_group": source_group,
            "pageid": pageid,
            "source_page": source_page,
            "download_url": download_url,
            "artist": artist,
            "license_canonical_name": "CC BY-SA 4.0",
            "license_canonical_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "filename": relative.as_posix(),
            "download_sha256": sha,
            "download_width": 64,
            "download_height": 64,
            "download_mime": "image/png",
        }
        queue = {
            "acquisition_mode": "fixture",
            "acquisition_query": f"fixture query {pageid}",
            "asset": f"wikimedia:{pageid}@sha256:{sha}",
            "class_hint": class_hint,
            "class_hint_status": "ACQUISITION_HINT_ONLY_UNREVIEWED",
            "creator": artist,
            "creator_group": f"commons-creator:{pageid}",
            "dhash64": f"{pageid:016x}",
            "download_height": 64,
            "download_mime": "image/png",
            "download_width": 64,
            "license": "CC BY-SA 4.0",
            "license_policy_sha256": "a" * 64,
            "license_raw_name": "CC BY-SA 4.0",
            "license_raw_url": "https://creativecommons.org/licenses/by-sa/4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "license_url_basis": "fixture",
            "local_path": relative.as_posix(),
            "near_duplicate_family": "",
            "notes": "",
            "pageid": pageid,
            "print_eligible": False,
            "review_status": "UNREVIEWED",
            "reviewed_source_group": "",
            "reviewer": "",
            "rights_decision": "",
            "schema_version": "rootscope.wikimedia_human_review_queue.v1",
            "sha256": sha,
            "source_group": source_group,
            "source_url": source_page,
            "species_hint": "fixture only",
            "species_hint_status": "ACQUISITION_HINT_ONLY_UNREVIEWED",
            "split": "UNASSIGNED_DO_NOT_TRAIN",
            "target_class": "",
            "title": f"File:fixture {pageid}.png",
            "training_eligible": False,
            "visual_decision": "",
        }
        return manifest, queue

    def _write_scores(self, mutate: callable | None = None) -> None:
        model_root = MODULE._artifact_root(self.model)["sha256"]
        prompt_root = MODULE._sha256_bytes(
            MODULE._canonical_bytes(self.policy_value["inference"]["prompts"])
        )
        probabilities = [
            {
                "grass_clump": 0.82,
                "low_shrub": 0.08,
                "young_tree": 0.05,
                "unknown": 0.05,
            },
            {
                "grass_clump": 0.13,
                "low_shrub": 0.42,
                "young_tree": 0.35,
                "unknown": 0.10,
            },
            {
                "grass_clump": 0.10,
                "low_shrub": 0.10,
                "young_tree": 0.08,
                "unknown": 0.72,
            },
        ]
        rows = [
            {
                "schema_version": "rootscope.ai_visual_model_output.v1",
                "asset": queue["asset"],
                "candidate_sha256": queue["sha256"],
                "model_id": "fixture/siglip-v1",
                "model_artifact_sha256": model_root,
                "prompt_set_sha256": prompt_root,
                "class_probabilities": score,
            }
            for queue, score in zip(self.queue_rows, probabilities, strict=True)
        ]
        if mutate is not None:
            mutate(rows)
        write_jsonl(self.model_outputs, rows)

    def config(self, output: Path | None = None, *, fixture_mode: bool = True) -> MODULE.TriageConfig:
        return MODULE.TriageConfig(
            queue_path=self.queue,
            manifest_path=self.manifest,
            queue_summary_path=self.summary,
            integrity_audit_path=self.integrity,
            class_contract_path=self.contract,
            policy_path=self.policy,
            output_dir=output or self.output,
            model_path=self.model,
            model_id="fixture/siglip-v1",
            backend="external_scores",
            model_output_path=self.model_outputs,
            fixture_mode=fixture_mode,
        )

    def human_hashes(self) -> dict[str, str]:
        return {
            path.name: sha256_bytes(path.read_bytes())
            for path in sorted(self.human_dir.iterdir())
            if path.is_file()
        }


class AIVisualTriageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = AITriageFixture(Path(self.temporary.name))

    def test_preflight_is_read_only_and_fully_bound(self) -> None:
        before = self.fixture.human_hashes()
        triage = MODULE.AIVisualTriage(self.fixture.config())
        report = triage.preflight()
        self.assertEqual("FIXTURE", report["mode"])
        self.assertEqual(3, report["candidate_count"])
        self.assertTrue(report["status"].startswith("FIXTURE_AI_VISUAL_TRIAGE_PREFLIGHT_PASS"))
        self.assertFalse(report["writes_performed"])
        self.assertFalse(self.fixture.output.exists())
        self.assertEqual(before, self.fixture.human_hashes())
        self.assertTrue(all(value is False for value in report["authority"].values()))

    def test_run_writes_strict_results_low_queue_stats_and_hash_receipt(self) -> None:
        human_before = self.fixture.human_hashes()
        result = MODULE.AIVisualTriage(self.fixture.config()).run()
        receipt = result["receipt"]
        self.assertEqual("FIXTURE", receipt["mode"])
        self.assertIn("NOT_HUMAN_REVIEWED_NOT_DATA_LOCKED", receipt["status"])
        self.assertEqual(3, receipt["candidate_count"])
        self.assertEqual(
            {"high_confidence_suggestion": 2, "low_confidence": 1}, receipt["counts"]
        )
        self.assertFalse(receipt["human_review_files_touched"])
        self.assertFalse(receipt["dataset_manifest_written"])
        self.assertTrue(all(value is False for value in receipt["authority"].values()))
        self.assertEqual(human_before, self.fixture.human_hashes())
        results = read_jsonl(self.fixture.output / "ai_visual_triage_results.jsonl")
        low = read_jsonl(self.fixture.output / "low_confidence_queue.jsonl")
        stats = read_json(self.fixture.output / "ai_visual_triage_stats.json")
        self.assertEqual(3, len(results))
        self.assertEqual(1, len(low))
        self.assertEqual(self.fixture.queue_rows[1]["asset"], low[0]["asset"])
        self.assertEqual(
            [
                "TOP1_PROBABILITY_BELOW_CLASS_THRESHOLD",
                "TOP1_TOP2_MARGIN_BELOW_THRESHOLD",
            ],
            low[0]["low_confidence_reasons"],
        )
        self.assertEqual(2, stats["high_confidence_suggestion_count"])
        self.assertEqual(1, stats["low_confidence_count"])
        for filename, expected_sha in receipt["outputs"].items():
            self.assertEqual(expected_sha, sha256_bytes((self.fixture.output / filename).read_bytes()))
        self.assertFalse((self.fixture.output / "human_review_receipt.json").exists())
        self.assertFalse((self.fixture.output / "decision_journal.jsonl").exists())

    def test_missing_model_fails_closed_before_any_output(self) -> None:
        self.fixture.model.unlink()
        with self.assertRaisesRegex(MODULE.TriageError, "model artifact does not exist"):
            MODULE.AIVisualTriage(self.fixture.config())
        self.assertFalse(self.fixture.output.exists())

    def test_tampered_image_fails_preflight(self) -> None:
        relative = self.fixture.queue_rows[0]["local_path"]
        (self.fixture.staging / relative).write_bytes(b"tampered")
        triage = MODULE.AIVisualTriage(self.fixture.config())
        with self.assertRaisesRegex(MODULE.TriageError, "image bytes changed"):
            triage.preflight()
        self.assertFalse(self.fixture.output.exists())

    def test_model_output_extra_field_is_rejected(self) -> None:
        self.fixture._write_scores(lambda rows: rows[0].update({"human_label": "grass_clump"}))
        triage = MODULE.AIVisualTriage(self.fixture.config())
        with self.assertRaisesRegex(MODULE.TriageError, "strict schema"):
            triage.preflight()

    def test_model_output_probability_sum_is_rejected(self) -> None:
        def corrupt(rows: list[dict]) -> None:
            rows[0]["class_probabilities"]["unknown"] = 0.25

        self.fixture._write_scores(corrupt)
        triage = MODULE.AIVisualTriage(self.fixture.config())
        with self.assertRaisesRegex(MODULE.TriageError, "do not sum to one"):
            triage.preflight()

    def test_probability_mapping_order_is_not_semantic(self) -> None:
        def reverse_probability_keys(rows: list[dict]) -> None:
            for row in rows:
                row["class_probabilities"] = dict(
                    reversed(list(row["class_probabilities"].items()))
                )

        self.fixture._write_scores(reverse_probability_keys)
        report = MODULE.AIVisualTriage(self.fixture.config()).preflight()
        self.assertEqual(3, report["candidate_count"])

    def test_model_output_order_and_candidate_binding_are_strict(self) -> None:
        def swap(rows: list[dict]) -> None:
            rows[0], rows[1] = rows[1], rows[0]

        self.fixture._write_scores(swap)
        triage = MODULE.AIVisualTriage(self.fixture.config())
        with self.assertRaisesRegex(MODULE.TriageError, "asset binding mismatch"):
            triage.preflight()

    def test_custom_inputs_require_explicit_fixture_mode(self) -> None:
        with self.assertRaisesRegex(MODULE.TriageError, "require explicit --fixture-mode"):
            MODULE.AIVisualTriage(self.fixture.config(fixture_mode=False))

    def test_fixture_cannot_write_production_output(self) -> None:
        config = self.fixture.config(output=MODULE.DEFAULT_OUTPUT_DIR)
        with self.assertRaisesRegex(MODULE.TriageError, "may not write the production"):
            MODULE.AIVisualTriage(config)

    def test_ai_output_cannot_be_nested_under_human_decisions(self) -> None:
        config = self.fixture.config(output=self.fixture.human_dir / "ai")
        with self.assertRaisesRegex(MODULE.TriageError, "human_decisions"):
            MODULE.AIVisualTriage(config)

    def test_immutable_output_directory_is_never_overwritten(self) -> None:
        MODULE.AIVisualTriage(self.fixture.config()).run()
        before = {
            path.name: sha256_bytes(path.read_bytes())
            for path in self.fixture.output.iterdir()
            if path.is_file()
        }
        with self.assertRaisesRegex(MODULE.TriageError, "already exists"):
            MODULE.AIVisualTriage(self.fixture.config()).run()
        after = {
            path.name: sha256_bytes(path.read_bytes())
            for path in self.fixture.output.iterdir()
            if path.is_file()
        }
        self.assertEqual(before, after)

    def test_model_artifact_mutation_after_binding_fails_closed(self) -> None:
        triage = MODULE.AIVisualTriage(self.fixture.config())
        self.fixture.model.write_bytes(b"mutated-model-artifact")
        with self.assertRaisesRegex(MODULE.TriageError, "changed after its initial SHA-256 binding"):
            triage.preflight()
        self.assertFalse(self.fixture.output.exists())

    def test_transformers_backend_rejects_non_siglip_config(self) -> None:
        model_dir = self.fixture.root / "not-siglip"
        model_dir.mkdir()
        write_json(model_dir / "config.json", {"model_type": "clip"})
        (model_dir / "model.safetensors").write_bytes(b"safe-fixture-weights")
        config = replace(
            self.fixture.config(),
            model_path=model_dir,
            backend="transformers_siglip",
            model_output_path=None,
        )
        with self.assertRaisesRegex(MODULE.TriageError, "not SigLIP/SigLIP2"):
            MODULE.AIVisualTriage(config)

    def test_transformers_backend_refuses_pickle_only_weights(self) -> None:
        model_dir = self.fixture.root / "pickle-only-siglip"
        model_dir.mkdir()
        write_json(model_dir / "config.json", {"model_type": "siglip2"})
        (model_dir / "pytorch_model.bin").write_bytes(b"unsafe-pickle-placeholder")
        (model_dir / ".cache").mkdir()
        (model_dir / ".cache" / "dummy.safetensors").write_bytes(b"not-root-weights")
        config = replace(
            self.fixture.config(),
            model_path=model_dir,
            backend="transformers_siglip",
            model_output_path=None,
        )
        with self.assertRaisesRegex(MODULE.TriageError, "no complete root-level safetensors weights"):
            MODULE.AIVisualTriage(config)

    def test_two_independent_fixture_runs_are_byte_deterministic(self) -> None:
        output_one = self.fixture.review / "ai_triage_one"
        output_two = self.fixture.review / "ai_triage_two"
        MODULE.AIVisualTriage(self.fixture.config(output_one)).run()
        MODULE.AIVisualTriage(self.fixture.config(output_two)).run()
        names = {
            "normalized_model_outputs.jsonl",
            "ai_visual_triage_results.jsonl",
            "low_confidence_queue.jsonl",
            "ai_visual_triage_stats.json",
            "ai_visual_triage_receipt.json",
        }
        for name in names:
            self.assertEqual((output_one / name).read_bytes(), (output_two / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
