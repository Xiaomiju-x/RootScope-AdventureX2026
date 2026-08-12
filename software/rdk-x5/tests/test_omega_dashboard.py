from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from app.omega_runtime.omega_server import build_omega_server


ROOT = Path(__file__).resolve().parents[1]


class OmegaDashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = build_omega_server(
            host="127.0.0.1",
            port=0,
            cases_path=ROOT / "configs" / "omega" / "locked_replay_cases.v1.json",
            profiles_path=ROOT / "configs" / "omega" / "edge_profiles.v1.json",
            corpus_path=ROOT / "configs" / "omega" / "field_knowledge.v1.md",
        )
        self.server.start()
        host, port = self.server.address
        self.base = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.close()

    def test_status_exposes_five_receipts_and_zero_authority(self) -> None:
        with urllib.request.urlopen(self.base + "/api/status", timeout=3) as response:
            report = json.loads(response.read().decode("utf-8"))
        self.assertTrue(report["all_locked_cases_passed"])
        self.assertEqual(len(report["cases"]), 5)
        self.assertFalse(report["authority"]["execution_authority"])
        self.assertEqual(
            report["selected_backend"]["decision_backend_actual"],
            "deterministic_cpu",
        )
        self.assertEqual(
            report["selected_backend"]["vision_backend_actual"],
            "onnxruntime_cpu",
        )
        self.assertEqual(
            report["selected_backend"]["retrieval_backend_actual"],
            "sqlite_fts5_bm25",
        )
        self.assertEqual(
            report["selected_backend"]["explanation_backend_actual"],
            "deterministic_template",
        )
        self.assertEqual(
            report["selected_backend"]["fallback_reasons"],
            ["BPU_MODEL_NOT_QUALIFIED", "LOCAL_LLM_UNAVAILABLE"],
        )
        for item in report["cases"]:
            self.assertEqual(
                item["truth_ribbon"]["receipt_sha256"],
                item["decision_receipt"]["receipt_sha256"],
            )
            self.assertTrue(
                all(value is False for value in item["truth_ribbon"]["authority"].values())
            )
            self.assertFalse(item["truth_ribbon"]["physical_completion_claim"])

    def test_page_contains_truth_ribbon_and_chain(self) -> None:
        with urllib.request.urlopen(self.base + "/", timeout=3) as response:
            html = response.read().decode("utf-8")
        self.assertIn("Truth Ribbon", html)
        self.assertIn("Evidence DAG", html)
        self.assertIn("H=2 RB-VoE", html)
        self.assertIn("五个锁定回放案例", html)
        self.assertIn(".case.accept", html)
        self.assertIn(".accept .decision", html)
        self.assertIn("decision_backend_actual", html)
        self.assertIn("vision_backend_actual", html)
        self.assertIn("retrieval_backend_actual", html)
        self.assertIn("explanation_backend_actual", html)
        self.assertIn("fallback_reasons", html)
        self.assertIn("report.authority", html)
        self.assertIn("report.runtime_boundary", html)
        self.assertIn("item.truth_ribbon", html)
        self.assertIn('id="authority">—</span>', html)
        self.assertIn('id="physical">—</span>', html)
        self.assertIn("UNVERIFIED", html)

    def test_server_rejects_non_loopback_and_has_no_actions(self) -> None:
        for host in ("0.0.0.0", "127.0.0.2", "localhost", "::1"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                build_omega_server(
                    host=host,
                    port=0,
                    cases_path=ROOT / "configs" / "omega" / "locked_replay_cases.v1.json",
                    profiles_path=ROOT / "configs" / "omega" / "edge_profiles.v1.json",
                    corpus_path=ROOT / "configs" / "omega" / "field_knowledge.v1.md",
                )
        request = urllib.request.Request(
            self.base + "/api/command",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=3)
        self.assertEqual(raised.exception.code, 409)


if __name__ == "__main__":
    unittest.main()
