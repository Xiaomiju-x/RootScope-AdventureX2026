from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import unittest

from app.omega_runtime.loopback_llm_cluster import (
    LoopbackLlamaConfig,
    LoopbackLlamaModel,
    run_loopback_role_cluster,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "6c8adc95de81b1d0103fcf900a5f369d5936d234e42ed64881dad56e0104e77b"


class _ValidHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, object]] = []

    def log_message(self, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        size = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(size))
        self.__class__.requests.append(request)
        user = json.loads(request["messages"][-1]["content"])
        role = user["request"]["role"]
        citation = user["citation_allowlist"][0]
        model_output = {
            "schema_version": "rootscope.omega.llm-model-output.v1",
            "role": role,
            "status": "READ_ONLY",
            "summary": "The response is limited to the cited RootScope evidence.",
            "claims": [
                {
                    "text": "The cited passage preserves a zero-authority boundary.",
                    "support_citation_ids": [citation],
                    "contradiction_citation_ids": [],
                    "safety_critical": role == "SAFETY_AUDITOR",
                }
            ],
            "uncertainties": ["No physical result is inferred."],
            "suggested_checks": ["Verify the cited source hash."],
            "authority": {
                "external_network": False,
                "tool_execution": False,
                "serial_write": False,
                "gpio_write": False,
                "state_machine_write": False,
                "actuator_access": False,
                "irrigation_execution": False,
            },
        }
        raw = json.dumps(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": json.dumps(model_output)}}
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class OmegaLoopbackClusterTests(unittest.TestCase):
    def setUp(self) -> None:
        _ValidHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ValidHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = int(self.server.server_address[1])

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _config(self) -> LoopbackLlamaConfig:
        return LoopbackLlamaConfig(
            endpoint=f"http://127.0.0.1:{self.port}",
            model_id="qwen2-0.5b-q4km-rootscope-read-only",
            model_sha256=MODEL_SHA,
            timeout_seconds=2,
        )

    def test_three_roles_share_one_loopback_model_and_are_cited(self) -> None:
        report = run_loopback_role_cluster(
            case_id="CASE01_NORMAL_VERIFIED",
            evidence_refs=("case01-source",),
            corpus_path=ROOT / "configs/omega/field_knowledge.v1.md",
            config=self._config(),
        )
        self.assertEqual(report["cluster_topology"]["resident_model_count"], 1)
        self.assertEqual(report["accepted_model_role_count"], 3)
        self.assertEqual(report["deterministic_fallback_role_count"], 0)
        self.assertEqual(len(report["transport_attempts"]), 3)
        self.assertEqual(
            report["cluster_topology"]["prompt_profile"]["name"],
            "X5_COMPACT_CITED_V1",
        )
        self.assertEqual(len(_ValidHandler.requests), 3)
        for request in _ValidHandler.requests:
            self.assertEqual(request["max_tokens"], 192)
            user = json.loads(request["messages"][-1]["content"])
            self.assertEqual(len(user["retrieval"]), 1)
            self.assertNotIn("required_model_schema", user)
            self.assertLess(
                sum(
                    len(message["content"].encode("utf-8"))
                    for message in request["messages"]
                ),
                2_200,
            )
        self.assertFalse(report["runtime_boundary"]["external_network_touched"])
        self.assertFalse(report["authority"]["tool_execution"])

    def test_non_loopback_endpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            LoopbackLlamaConfig(
                endpoint="http://192.0.2.10:9080",
                model_id="model",
                model_sha256=MODEL_SHA,
            )

    def test_bad_transport_falls_back_for_all_roles(self) -> None:
        config = LoopbackLlamaConfig(
            endpoint="http://127.0.0.1:65534",
            model_id="unavailable-loopback-fixture",
            model_sha256=MODEL_SHA,
            timeout_seconds=0.2,
        )
        report = run_loopback_role_cluster(
            case_id="CASE02_OOD_RECAPTURE",
            evidence_refs=("case02-ood",),
            corpus_path=ROOT / "configs/omega/field_knowledge.v1.md",
            config=config,
        )
        self.assertEqual(report["accepted_model_role_count"], 0)
        self.assertEqual(report["deterministic_fallback_role_count"], 3)
        self.assertEqual(
            {item["transport_status"] for item in report["transport_attempts"]},
            {"FAILED_CLOSED"},
        )

    def test_adapter_exposes_text_only_generate_contract(self) -> None:
        model = LoopbackLlamaModel(self._config())
        with self.assertRaisesRegex(ValueError, "text-only"):
            model.generate(
                [{"role": "tool", "content": "do something"}],
                response_schema={},
                temperature=0.0,
                max_tokens=64,
            )


if __name__ == "__main__":
    unittest.main()
