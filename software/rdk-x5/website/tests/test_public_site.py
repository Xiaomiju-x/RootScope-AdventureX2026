from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rootscope_public_app", ROOT / "app.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class PublicSiteContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = module.ThreadingHTTPServer(("127.0.0.1", 0), module.RootScopeHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path="/", body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=4)
        conn.request(method, path, body=body)
        response = conn.getresponse()
        payload = response.read()
        headers = {key.lower(): value for key, value in response.getheaders()}
        conn.close()
        return response.status, headers, payload

    def test_public_home_has_project_identity_and_no_login(self):
        status, headers, body = self.request("GET", "/")
        text = body.decode("utf-8")
        self.assertEqual(status, 200)
        self.assertIn("RootScope", text)
        self.assertIn("固定式根区灌溉舱", text)
        self.assertNotIn('type="password"', text)
        self.assertNotIn("/login", text)
        self.assertEqual(headers["cache-control"], "no-store")

    def test_health_is_read_only_and_local_service_is_identifiable(self):
        status, _, body = self.request("GET", "/healthz")
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["service"], "rootscope-public")
        self.assertEqual(payload["mode"], "read-only")

    def test_write_methods_are_rejected(self):
        for method in ("POST", "PUT", "PATCH", "DELETE", "CONNECT", "TRACE"):
            status, headers, body = self.request(method, "/", b"test=1")
            self.assertEqual(status, 405, method)
            self.assertEqual(headers["allow"], "GET, HEAD, OPTIONS")
            self.assertIn(b"method_not_allowed", body)

    def test_security_headers_are_present(self):
        status, headers, _ = self.request("HEAD", "/")
        self.assertEqual(status, 200)
        self.assertIn("frame-ancestors 'none'", headers["content-security-policy"])
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertEqual(headers["x-frame-options"], "DENY")
        self.assertIn("camera=()", headers["permissions-policy"])
        self.assertIn("usb=()", headers["permissions-policy"])

    def test_path_traversal_is_not_served(self):
        status, _, _ = self.request("GET", "/%2e%2e/app.py")
        self.assertIn(status, (400, 404))

    def test_static_contract_contains_no_active_device_interfaces(self):
        html = (ROOT / "public" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "public" / "app.js").read_text(encoding="utf-8")
        combined = html + "\n" + js
        forbidden = (
            "WebSocket(",
            "EventSource(",
            "navigator.usb",
            "getUserMedia(",
            "/api/pump",
            "/api/serial",
            "fetch(",
        )
        for token in forbidden:
            self.assertNotIn(token, combined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
