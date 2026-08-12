from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.evidence.verifier import (
    EvidenceVerificationError,
    load_verified_task_history,
    verify_against_terminal_manifest,
    verify_jsonl,
    verify_live_ledger,
)
from app.evidence.writer import EvidenceWriter
from app.schemas import AdmissionStatus, MachineState
from app.state_machine import RootScopeStateMachine

from tests.test_state_machine import drive_ready, make_config, safety, task


class EvidenceTests(unittest.TestCase):
    def test_initialisation_is_explicit_and_missing_ledger_never_recreated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            with self.assertRaises(EvidenceVerificationError):
                EvidenceWriter(ledger)
            writer = EvidenceWriter(ledger, initialize=True)
            writer.append("startup", {"mode": "fixture"})
            self.assertTrue(verify_live_ledger(ledger).valid)
            ledger.unlink()
            with self.assertRaises(EvidenceVerificationError):
                EvidenceWriter(ledger)
            with self.assertRaises(EvidenceVerificationError):
                EvidenceWriter(ledger, initialize=True)

    def test_chain_detects_payload_edit_and_writer_refuses_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            writer = EvidenceWriter(ledger, initialize=True)
            writer.append("startup", {"value": 1})
            writer.append("health", {"ok": True})
            self.assertTrue(verify_jsonl(ledger).valid)
            lines = ledger.read_text("utf-8").splitlines()
            record = json.loads(lines[0])
            record["payload"]["value"] = 2
            lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
            ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertFalse(verify_jsonl(ledger).valid)
            with self.assertRaises(EvidenceVerificationError):
                EvidenceWriter(ledger)

    def test_live_state_and_terminal_manifest_detect_clean_tail_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            manifest = Path(temporary) / "external" / "terminal.json"
            writer = EvidenceWriter(ledger, initialize=True)
            writer.append("startup", {"value": 1})
            writer.append("health", {"ok": True})
            writer.freeze_terminal_manifest(manifest)

            first_line = ledger.read_text("utf-8").splitlines()[0]
            ledger.write_text(first_line + "\n", encoding="utf-8")
            # A bare chain cannot detect removal at a complete record boundary.
            self.assertTrue(verify_jsonl(ledger).valid)
            self.assertFalse(verify_live_ledger(ledger).valid)
            anchored = verify_against_terminal_manifest(ledger, manifest)
            self.assertFalse(anchored.valid)
            self.assertTrue(
                {issue.code for issue in anchored.issues}
                & {"ANCHOR_COUNT_MISMATCH", "ANCHOR_HASH_MISMATCH"}
            )

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            writer = EvidenceWriter(ledger, initialize=True)
            writer.append("startup", {"value": 1})
            text = ledger.read_text("utf-8")
            text = text.replace('"payload":{', '"payload":{},"payload":{', 1)
            ledger.write_text(text, encoding="utf-8")
            report = verify_jsonl(ledger)
            self.assertFalse(report.valid)
            self.assertIn("INVALID_JSON", {issue.code for issue in report.issues})

    def test_verified_history_preserves_idempotency_across_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            writer = EvidenceWriter(ledger, initialize=True)
            config = make_config()
            machine = RootScopeStateMachine(config, event_sink=writer)
            drive_ready(machine, config)
            request = task(config)
            self.assertEqual(
                machine.admit_task(request, safety(config)).status,
                AdmissionStatus.ACCEPTED,
            )

            history = load_verified_task_history(ledger)
            self.assertEqual(len(history), 1)
            restarted = RootScopeStateMachine(
                config, task_history=history, event_sink=writer
            )
            self.assertEqual(restarted.state, MachineState.BOOT_LOCKED)
            drive_ready(restarted, config, frame_seq=11)
            replay = restarted.admit_task(request, safety(config))
            self.assertEqual(replay.status, AdmissionStatus.IDEMPOTENT_REPLAY)
            self.assertFalse(replay.may_create_physical_command)
            admitted = [
                json.loads(line)
                for line in ledger.read_text("utf-8").splitlines()
                if json.loads(line)["event_type"] == "task_admitted"
            ]
            self.assertEqual(len(admitted), 1)

    def test_history_loader_requires_live_state_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            writer = EvidenceWriter(ledger, initialize=True)
            writer.append("startup", {})
            writer.live_state_path.unlink()
            with self.assertRaises(EvidenceVerificationError):
                load_verified_task_history(ledger)


if __name__ == "__main__":
    unittest.main()
