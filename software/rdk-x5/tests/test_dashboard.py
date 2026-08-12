from __future__ import annotations

import json
import http.client
import unittest
import urllib.error
import urllib.request

from app.web.server import DashboardServer
from app.web.state_store import SnapshotStore


class DashboardServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = DashboardServer(
            SnapshotStore(),
            port=0,
            actions={"/api/test": lambda payload: {}},
        )
        self.server.start()
        host, port = self.server.address
        self.base = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.close()

    def get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_health_is_explicitly_non_hardware(self) -> None:
        health = self.get_json("/api/health")
        self.assertTrue(health["ok"])
        self.assertFalse(health["hardware_touched"])

    def test_default_status_is_locked_fixture(self) -> None:
        status = self.get_json("/api/status")
        self.assertEqual(status["state"], "BOOT_LOCKED")
        self.assertEqual(status["mode"], "SIMULATED_ONLY")
        self.assertEqual(status["backend_actual"], "fixture")

    def test_action_without_callback_fails_closed(self) -> None:
        request = urllib.request.Request(
            self.base + "/api/simulate/start",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=2)
        self.assertEqual(raised.exception.code, 409)

    def test_negative_content_length_is_rejected_without_reading_body(self) -> None:
        host, port = self.server.address
        connection = http.client.HTTPConnection(host, port, timeout=2)
        try:
            connection.putrequest("POST", "/api/test")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", "-1")
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            payload = json.loads(response.read().decode("utf-8"))
            self.assertFalse(payload["ok"])
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
