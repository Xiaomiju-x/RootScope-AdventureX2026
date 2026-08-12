from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
ADVENTUREX_ROOT = V3_ROOT.parent


def load_json(relative: str):
    return json.loads((V3_ROOT / relative).read_text(encoding="utf-8"))


def load_verifier_module():
    path = V3_ROOT / "tools" / "verify_e0.py"
    spec = importlib.util.spec_from_file_location("rootscope_v3_verify_e0", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class E0ContractTests(unittest.TestCase):
    def test_full_e0_verifier(self):
        result = load_verifier_module().verify(ADVENTUREX_ROOT)
        self.assertEqual(
            result["status"],
            "PASS_E0_FACTS_REGISTRIES_SCHEMAS_CANDIDATE_ZERO_AUTHORITY",
        )
        self.assertFalse(result["x5_contacted"])
        self.assertFalse(result["environment_values_read"])

    def test_v2_hash_is_bound_everywhere(self):
        expected = "03ca7b8d9ff8b691f1fd61dc696601ba30f494377a0b2a3cfadb66c19478ed94"
        baseline = load_json("governance/baseline_v2_snapshot.json")
        state = load_json(
            "candidates/rootscope_v3_candidate_unqualified/CANDIDATE_STATE.json"
        )
        manifest = load_json(
            "candidates/rootscope_v3_candidate_unqualified/MANIFEST.template.json"
        )
        self.assertEqual(baseline["release"]["archive_sha256"], expected)
        self.assertEqual(state["baseline_v2_archive_sha256"], expected)
        self.assertEqual(manifest["baseline_v2_archive_sha256"], expected)

    def test_all_ai_registry_entries_are_zero_authority(self):
        models = load_json("registries/models.v1.json")["models"]
        teachers = load_json("registries/teachers.v1.json")["teachers"]
        self.assertTrue(models)
        self.assertTrue(teachers)
        self.assertTrue(all(item["physical_authority"] is False for item in models))
        self.assertTrue(all(item["physical_authority"] is False for item in teachers))

    def test_planned_assets_have_no_fake_hash(self):
        models = load_json("registries/models.v1.json")["models"]
        planned = [item for item in models if item["status"] == "PLANNED_NOT_ACQUIRED"]
        self.assertGreaterEqual(len(planned), 4)
        for item in planned:
            self.assertIsNone(item["relative_locator"])
            self.assertIsNone(item["sha256"])
            self.assertIsNone(item["size_bytes"])

    def test_teacher_registry_contains_names_not_values(self):
        registry = load_json("registries/teachers.v1.json")
        allowed = set(registry["secret_policy"]["allowed_environment_variable_names"])
        for teacher in registry["teachers"]:
            name = teacher["credential_environment_variable"]
            if name is not None:
                self.assertIn(name, allowed)
                self.assertRegex(name, r"^ROOTSCOPE_[A-Z0-9_]+$")
        serialized = json.dumps(registry).lower()
        self.assertNotIn("bearer ", serialized)
        self.assertNotIn("sk-", serialized)

    def test_five_schema_fixture_pairs_exist(self):
        names = ("vision", "llm", "rag", "resource", "physical_loop")
        for name in names:
            schema = load_json(f"schemas/evaluation/{name}_evaluation.schema.json")
            fixture = load_json(f"examples/evaluation/{name}.fixture.json")
            self.assertEqual(
                fixture["schema"], schema["properties"]["schema"]["const"]
            )
            self.assertEqual(
                schema["$schema"],
                "https://json-schema.org/draft/2020-12/schema",
            )

    def test_candidate_is_not_deployable(self):
        state = load_json(
            "candidates/rootscope_v3_candidate_unqualified/CANDIDATE_STATE.json"
        )
        self.assertEqual(state["status"], "SKELETON_NOT_A_RELEASE_DO_NOT_DEPLOY")
        self.assertTrue(state["gates"]["e0_contracts_frozen"])
        for key, value in state["gates"].items():
            if key != "e0_contracts_frozen":
                self.assertFalse(value, key)
        self.assertTrue(all(value is False for value in state["authority"].values()))


if __name__ == "__main__":
    unittest.main()
