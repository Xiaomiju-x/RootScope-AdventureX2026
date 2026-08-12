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


MODULE_PATH = Path(__file__).resolve().parents[1] / "ai_siglip2_ensemble.py"
POLICY_PATH = MODULE_PATH.with_name("ai_siglip2_ensemble_policy_v1.json")
SPEC = importlib.util.spec_from_file_location("rootscope_ai_siglip2_ensemble", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TARGET_CLASSES = ["grass_clump", "low_shrub", "young_tree"]
ALL_HINTS = [*TARGET_CLASSES, "unknown"]


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
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class FakePromptScorer:
    def __init__(self, malformed: str | None = None) -> None:
        self.calls = 0
        self.malformed = malformed

    def score(self, images, prompts):
        self.calls += 1
        matrix: list[list[float]] = []
        for image in images:
            pageid = int(image.stem.rsplit("_", 1)[-1])
            row: list[float] = []
            for prompt in prompts:
                prompt_id = prompt["id"]
                if pageid == 201:  # clear grass target
                    if prompt_id.startswith("quality."):
                        value = 5.0
                    elif prompt_id.startswith("class.grass_clump."):
                        value = 5.0
                    elif prompt_id.startswith("class.low_shrub."):
                        value = 0.0
                    elif prompt_id.startswith("class.young_tree."):
                        value = -2.0
                    else:
                        value = -5.0
                elif pageid == 202:  # strong non-target; no morphology dominates
                    if prompt_id.startswith("quality."):
                        value = -5.0
                    elif prompt_id.startswith("reject.no_target_object."):
                        value = 5.0
                    elif prompt_id.startswith("reject."):
                        value = -1.0
                    else:
                        value = 0.0
                else:  # admissible but shrub/tree ambiguous => HOLD
                    if prompt_id.startswith("quality."):
                        value = 5.0
                    elif prompt_id.startswith("reject."):
                        value = -5.0
                    elif prompt_id.startswith("class.low_shrub."):
                        value = 1.0
                    elif prompt_id.startswith("class.young_tree."):
                        value = 0.9
                    else:
                        value = 0.0
                row.append(value)
            matrix.append(row)
        if self.malformed == "short_row":
            matrix[0].pop()
        elif self.malformed == "nan":
            matrix[0][0] = float("nan")
        elif self.malformed == "short_matrix":
            matrix.pop()
        return matrix


class EnsembleFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging = root / "staging"
        self.review = self.staging / "review"
        self.queue = self.review / "candidate_review_queue.jsonl"
        self.manifest = self.staging / "manifest.jsonl"
        self.summary = self.review / "review_queue_summary.json"
        self.integrity = self.staging / "integrity_audit.json"
        self.contract = root / "class_contract.json"
        self.policy = root / "ai_siglip2_ensemble_policy_v1.json"
        self.model = root / "fixture-model.bin"
        self.output = self.review / "ai_ensemble_v1"
        self.human = self.review / "human_decisions"
        self.model.write_bytes(b"fixture-siglip2-artifact-v1")
        candidates = [
            self._candidate(201, "grass_clump", (42, 122, 52)),
            self._candidate(202, "unknown", (182, 163, 123)),
            self._candidate(203, "low_shrub", (105, 88, 51)),
        ]
        self.manifest_rows = [item[0] for item in candidates]
        self.queue_rows = [item[1] for item in candidates]
        write_jsonl(self.manifest, self.manifest_rows)
        write_jsonl(self.queue, self.queue_rows)
        manifest_sha = sha256_bytes(self.manifest.read_bytes())
        queue_sha = sha256_bytes(self.queue.read_bytes())
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
            },
        )
        write_json(self.contract, {"schema_version": "2.0.0", "class_order": ALL_HINTS})
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
        self.human.mkdir(parents=True)
        (self.human / "decision_journal.jsonl").write_bytes(b"human-sentinel\n")
        write_json(self.human / "human_review_receipt.json", {"sentinel": "formal-human-only"})

    def _candidate(self, pageid: int, hint: str, color: tuple[int, int, int]) -> tuple[dict, dict]:
        relative = Path("images") / hint / f"candidate_{pageid}.png"
        absolute = self.staging / relative
        absolute.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (64, 64), color)
        for offset in range(8):
            image.putpixel(
                ((pageid + offset * 5) % 64, (pageid + offset * 9) % 64),
                tuple(255 - channel for channel in color),
            )
        image.save(absolute, format="PNG")
        sha = sha256_bytes(absolute.read_bytes())
        source_group = f"commons:{pageid}"
        source_page = f"https://commons.wikimedia.org/wiki/File:fixture_{pageid}.png"
        manifest = {
            "schema_version": "rootscope.wikimedia_candidate.v1",
            "class_id": hint,
            "domain": "natural_web_candidate",
            "split": "UNASSIGNED_DO_NOT_TRAIN",
            "review_status": "pending_human_visual_and_license_review",
            "training_eligible": False,
            "print_eligible": False,
            "source_provider": "Wikimedia Commons",
            "source_group": source_group,
            "pageid": pageid,
            "source_page": source_page,
            "download_url": f"https://upload.wikimedia.org/fixture_{pageid}.png",
            "artist": f"Fixture Author {pageid}",
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
            "acquisition_query": f"fixture {pageid}",
            "asset": f"wikimedia:{pageid}@sha256:{sha}",
            "class_hint": hint,
            "class_hint_status": "ACQUISITION_HINT_ONLY_UNREVIEWED",
            "creator": f"Fixture Author {pageid}",
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

    def config(self, **changes) -> MODULE.EnsembleConfig:
        value = MODULE.EnsembleConfig(
            queue_path=self.queue,
            manifest_path=self.manifest,
            queue_summary_path=self.summary,
            integrity_audit_path=self.integrity,
            class_contract_path=self.contract,
            policy_path=self.policy,
            output_dir=self.output,
            model_path=self.model,
            model_id="fixture/google-siglip2",
            backend="fixture_fake",
            fixture_mode=True,
        )
        return replace(value, **changes)

    def human_hashes(self) -> dict[str, str]:
        return {
            path.name: sha256_bytes(path.read_bytes())
            for path in sorted(self.human.iterdir())
            if path.is_file()
        }


class AISigLIP2EnsembleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = EnsembleFixture(Path(self.temporary.name))

    def ensemble(self, scorer: FakePromptScorer | None = None):
        return MODULE.AISigLIP2Ensemble(
            self.fixture.config(), scorer or FakePromptScorer()
        )

    def test_preflight_is_read_only_and_does_not_call_scorer(self) -> None:
        scorer = FakePromptScorer()
        before_human = self.fixture.human_hashes()
        before_manifest = sha256_bytes(self.fixture.manifest.read_bytes())
        report = MODULE.AISigLIP2Ensemble(self.fixture.config(), scorer).preflight()
        self.assertEqual("FIXTURE", report["mode"])
        self.assertEqual(3, report["candidate_count"])
        self.assertEqual(32, report["prompt_count"])
        self.assertFalse(report["writes_performed"])
        self.assertFalse(report["output_exists"])
        self.assertEqual(0, scorer.calls)
        self.assertFalse(self.fixture.output.exists())
        self.assertEqual(before_human, self.fixture.human_hashes())
        self.assertEqual(before_manifest, sha256_bytes(self.fixture.manifest.read_bytes()))

    def test_run_emits_three_conservative_decisions_and_full_score_evidence(self) -> None:
        human_before = self.fixture.human_hashes()
        manifest_before = sha256_bytes(self.fixture.manifest.read_bytes())
        result = self.ensemble().run()
        receipt = result["receipt"]
        self.assertEqual(
            {"auto_target": 1, "auto_unknown": 1, "hold": 1}, receipt["counts"]
        )
        self.assertFalse(receipt["human_review_files_touched"])
        self.assertFalse(receipt["dataset_manifest_written"])
        self.assertTrue(all(value is False for value in receipt["authority"].values()))
        rows = read_jsonl(self.fixture.output / "ai_siglip2_ensemble_results.jsonl")
        self.assertEqual(["AUTO_TARGET", "AUTO_UNKNOWN", "HOLD"], [row["decision"] for row in rows])
        self.assertEqual("grass_clump", rows[0]["suggested_class"])
        self.assertEqual("unknown", rows[1]["suggested_class"])
        self.assertEqual("no_target_object", rows[1]["dominant_reject_family"])
        self.assertEqual(32, len(rows[0]["prompt_scores"]))
        self.assertEqual(TARGET_CLASSES, list(rows[0]["class_scores"]))
        self.assertEqual(
            set(MODULE.REJECT_FAMILIES), set(rows[0]["reject_family_scores"])
        )
        holds = read_jsonl(self.fixture.output / "hold_queue.jsonl")
        self.assertEqual(1, len(holds))
        self.assertEqual(self.fixture.queue_rows[2]["asset"], holds[0]["asset"])
        self.assertIn("TOP1_PROBABILITY_BELOW_CLASS_THRESHOLD", holds[0]["decision_reasons"])
        self.assertIn("TOP1_TOP2_MARGIN_BELOW_THRESHOLD", holds[0]["decision_reasons"])
        for filename, expected_sha in receipt["outputs"].items():
            self.assertEqual(
                expected_sha,
                sha256_bytes((self.fixture.output / filename).read_bytes()),
            )
        self.assertEqual(human_before, self.fixture.human_hashes())
        self.assertEqual(manifest_before, sha256_bytes(self.fixture.manifest.read_bytes()))
        self.assertFalse((self.fixture.output / "decision_journal.jsonl").exists())
        self.assertFalse((self.fixture.output / "human_review_receipt.json").exists())

    def test_tampered_image_fails_before_scorer_and_output(self) -> None:
        image = self.fixture.staging.joinpath(*self.fixture.queue_rows[0]["local_path"].split("/"))
        image.write_bytes(b"tampered")
        scorer = FakePromptScorer()
        ensemble = MODULE.AISigLIP2Ensemble(self.fixture.config(), scorer)
        with self.assertRaisesRegex(MODULE.EnsembleError, "image bytes changed"):
            ensemble.preflight()
        self.assertEqual(0, scorer.calls)
        self.assertFalse(self.fixture.output.exists())

    def test_model_mutation_after_initial_binding_fails_closed(self) -> None:
        ensemble = self.ensemble()
        self.fixture.model.write_bytes(b"mutated-model")
        with self.assertRaisesRegex(MODULE.EnsembleError, "changed after its initial"):
            ensemble.preflight()
        self.assertFalse(self.fixture.output.exists())

    def test_bad_fake_score_shapes_and_nan_fail_without_output(self) -> None:
        for malformed in ("short_row", "short_matrix", "nan"):
            with self.subTest(malformed=malformed):
                root = Path(self.temporary.name) / malformed
                fixture = EnsembleFixture(root)
                ensemble = MODULE.AISigLIP2Ensemble(
                    fixture.config(), FakePromptScorer(malformed)
                )
                with self.assertRaises(MODULE.EnsembleError):
                    ensemble.run()
                self.assertFalse(fixture.output.exists())

    def test_output_scope_is_exact_and_human_directory_is_forbidden(self) -> None:
        with self.assertRaisesRegex(MODULE.EnsembleError, "exactly review/ai_ensemble_v1"):
            MODULE.AISigLIP2Ensemble(
                self.fixture.config(output_dir=self.fixture.review / "another_name"),
                FakePromptScorer(),
            )
        with self.assertRaisesRegex(MODULE.EnsembleError, "exactly review/ai_ensemble_v1"):
            MODULE.AISigLIP2Ensemble(
                self.fixture.config(output_dir=self.fixture.human / "ai_ensemble_v1"),
                FakePromptScorer(),
            )

    def test_custom_paths_without_fixture_boundary_are_rejected(self) -> None:
        with self.assertRaisesRegex(MODULE.EnsembleError, "explicit --fixture-mode"):
            MODULE.AISigLIP2Ensemble(
                self.fixture.config(fixture_mode=False), FakePromptScorer()
            )

    def test_fake_scorer_cannot_cross_fixture_boundary(self) -> None:
        with self.assertRaisesRegex(MODULE.EnsembleError, "injected scorer"):
            MODULE.AISigLIP2Ensemble(
                self.fixture.config(backend="local_siglip2"), FakePromptScorer()
            )

    def test_local_backend_rejects_non_siglip_and_pickle_only_models(self) -> None:
        non_siglip = self.fixture.root / "non-siglip"
        non_siglip.mkdir()
        write_json(non_siglip / "config.json", {"model_type": "clip"})
        (non_siglip / "model.safetensors").write_bytes(b"safe-placeholder")
        with self.assertRaisesRegex(MODULE.EnsembleError, "not SigLIP/SigLIP2"):
            MODULE.AISigLIP2Ensemble(
                self.fixture.config(model_path=non_siglip, backend="local_siglip2")
            )

        pickle_only = self.fixture.root / "pickle-only"
        pickle_only.mkdir()
        write_json(pickle_only / "config.json", {"model_type": "siglip2"})
        (pickle_only / "pytorch_model.bin").write_bytes(b"pickle-placeholder")
        with self.assertRaisesRegex(MODULE.EnsembleError, "no non-empty safetensors"):
            MODULE.AISigLIP2Ensemble(
                self.fixture.config(model_path=pickle_only, backend="local_siglip2")
            )

    def test_duplicate_prompt_id_is_rejected(self) -> None:
        policy = json.loads(self.fixture.policy.read_text(encoding="utf-8"))
        policy["inference"]["class_prompts"]["grass_clump"][0]["id"] = policy[
            "inference"
        ]["whole_quality_prompts"][0]["id"]
        write_json(self.fixture.policy, policy)
        with self.assertRaisesRegex(MODULE.EnsembleError, "globally unique"):
            self.ensemble()

    def test_output_is_immutable(self) -> None:
        self.ensemble().run()
        before = {
            path.name: sha256_bytes(path.read_bytes())
            for path in self.fixture.output.iterdir()
            if path.is_file()
        }
        with self.assertRaisesRegex(MODULE.EnsembleError, "already exists"):
            self.ensemble().run()
        after = {
            path.name: sha256_bytes(path.read_bytes())
            for path in self.fixture.output.iterdir()
            if path.is_file()
        }
        self.assertEqual(before, after)

    def test_independent_fixture_runs_are_byte_deterministic(self) -> None:
        one = EnsembleFixture(Path(self.temporary.name) / "one")
        two = EnsembleFixture(Path(self.temporary.name) / "two")
        MODULE.AISigLIP2Ensemble(one.config(), FakePromptScorer()).run()
        MODULE.AISigLIP2Ensemble(two.config(), FakePromptScorer()).run()
        for name in (
            "ai_siglip2_ensemble_results.jsonl",
            "hold_queue.jsonl",
            "ai_siglip2_ensemble_stats.json",
            "ai_siglip2_ensemble_receipt.json",
        ):
            self.assertEqual((one.output / name).read_bytes(), (two.output / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
