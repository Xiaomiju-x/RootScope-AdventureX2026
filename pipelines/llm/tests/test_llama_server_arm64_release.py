from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ADVENTUREX = Path(__file__).resolve().parents[3]
MODULE_DIR = ADVENTUREX / "tools" / "llm" / "llama_server_arm64"
sys.path.insert(0, str(MODULE_DIR))

from audit_release import audit_release, parse_sha256sums, version_tuple  # noqa: E402


class LlamaServerArm64ReleaseTests(unittest.TestCase):
    def test_frozen_release_passes_read_only_audit(self) -> None:
        release = ADVENTUREX / "output" / "rootscope_llama_server_arm64_b9637_v1"
        result = audit_release(release)
        failures = [item for item in result["checks"] if not item["passed"]]
        self.assertEqual("PASS", result["status"], failures)
        self.assertEqual(0, result["checks_failed"])

    def test_sha_parser_rejects_unfrozen_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SHA256SUMS"
            path.write_text("not-a-hash  bin/llama-server\n", encoding="ascii")
            with self.assertRaises(ValueError):
                parse_sha256sums(path)

    def test_numeric_version_order(self) -> None:
        self.assertLess(version_tuple("2.9"), version_tuple("2.34"))
        self.assertLessEqual(version_tuple("2.34"), (2, 35))


if __name__ == "__main__":
    unittest.main()
