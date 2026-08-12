"""Fail-closed append-only JSONL evidence writer."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..schemas import utc_now_iso
from .verifier import (
    EVIDENCE_SCHEMA_VERSION,
    EvidenceVerificationError,
    LiveLedgerState,
    TerminalManifest,
    compute_record_hash,
    create_terminal_manifest,
    default_live_state_path,
    make_live_state,
    read_live_state,
    verify_live_ledger,
    verify_jsonl,
    write_live_state,
)


_EVENT_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class EvidenceReceipt:
    record_index: int
    record_hash: str
    previous_hash: str
    event_type: str
    task_id: Optional[str]
    ledger_id: str


def _normalise_json(value: Any, path: str = "payload") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            result[key] = _normalise_json(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _normalise_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} contains unsupported type {type(value).__name__}")


class EvidenceWriter:
    """Single-process append writer that verifies the chain before every write."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], str] = utc_now_iso,
        create_parent: bool = True,
        initialize: bool = False,
        live_state_path: Optional[Path] = None,
    ) -> None:
        self.path = Path(path)
        self.live_state_path = (
            Path(live_state_path)
            if live_state_path is not None
            else default_live_state_path(self.path)
        )
        self._clock = clock
        self._lock = threading.RLock()
        if create_parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            if not initialize:
                raise EvidenceVerificationError(
                    "live ledger is missing; explicit initialize=True is required"
                )
            if self.live_state_path.exists():
                raise EvidenceVerificationError(
                    "ledger missing while persistent live state exists"
                )
            self.path.touch(exist_ok=False)
            initial_state = make_live_state(
                ledger_id=str(uuid.uuid4()),
                evidence_file=self.path.name,
                record_count=0,
                terminal_hash="0" * 64,
                high_watermark_task_seq=0,
                updated_at_utc=self._clock(),
            )
            write_live_state(self.live_state_path, initial_state, exclusive=True)
        elif initialize:
            raise EvidenceVerificationError(
                "refusing to initialize over an existing ledger"
            )
        self._live_state = read_live_state(self.live_state_path)
        verify_live_ledger(self.path, self.live_state_path).require_valid()

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        task_id: Optional[str] = None,
    ) -> EvidenceReceipt:
        if not isinstance(event_type, str) or not _EVENT_RE.fullmatch(event_type):
            raise ValueError("event_type must be a lowercase safe token")
        if task_id is not None and (not isinstance(task_id, str) or not task_id):
            raise ValueError("task_id must be a non-empty string or None")
        normalised_payload = _normalise_json(payload)
        if not isinstance(normalised_payload, dict):
            raise ValueError("payload must be a mapping")

        with self._lock:
            before = verify_live_ledger(
                self.path, self.live_state_path
            ).require_valid()
            live_before = read_live_state(self.live_state_path)
            next_high_watermark = live_before.high_watermark_task_seq
            if event_type == "task_admitted":
                task_seq = normalised_payload.get("task_seq")
                if (
                    isinstance(task_seq, bool)
                    or not isinstance(task_seq, int)
                    or not 0 < task_seq <= 0xFFFFFFFF
                ):
                    raise ValueError("task_admitted requires a uint32 task_seq")
                if task_seq <= next_high_watermark:
                    raise EvidenceVerificationError(
                        "task_admitted task_seq does not advance live high watermark"
                    )
                next_high_watermark = task_seq
            unsigned = {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "ledger_id": live_before.ledger_id,
                "record_index": before.record_count,
                "recorded_at_utc": self._clock(),
                "event_type": event_type,
                "task_id": task_id,
                "payload": normalised_payload,
                "prev_hash": before.terminal_hash,
            }
            record_hash = compute_record_hash(unsigned)
            record = dict(unsigned)
            record["record_hash"] = record_hash
            line = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")

            with self.path.open("ab", buffering=0) as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

            after = verify_jsonl(
                self.path, expected_ledger_id=live_before.ledger_id
            )
            if not after.valid or after.record_count != before.record_count + 1:
                raise EvidenceVerificationError(
                    "evidence append did not produce a valid one-record extension"
                )
            live_after = make_live_state(
                ledger_id=live_before.ledger_id,
                evidence_file=self.path.name,
                record_count=after.record_count,
                terminal_hash=after.terminal_hash,
                high_watermark_task_seq=next_high_watermark,
                updated_at_utc=self._clock(),
            )
            write_live_state(self.live_state_path, live_after, exclusive=False)
            verify_live_ledger(self.path, self.live_state_path).require_valid()
            self._live_state = live_after
            return EvidenceReceipt(
                record_index=before.record_count,
                record_hash=record_hash,
                previous_hash=before.terminal_hash,
                event_type=event_type,
                task_id=task_id,
                ledger_id=live_before.ledger_id,
            )

    def write_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        task_id: Optional[str] = None,
    ) -> None:
        """State-machine event sink interface."""

        self.append(event_type, payload, task_id)

    __call__ = write_event

    def freeze_terminal_manifest(self, manifest_path: Path) -> TerminalManifest:
        """Create a non-overwriting terminal manifest for external anchoring."""

        with self._lock:
            return create_terminal_manifest(
                self.path, Path(manifest_path), clock=self._clock
            )
