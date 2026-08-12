"""Strict verification for the local RootScope JSONL SHA-256 chain.

This detects accidental or unsynchronised modification.  It is not a digital
signature and must not be described as third-party attestation or blockchain.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..schemas import TaskHistoryEntry, utc_now_iso


EVIDENCE_SCHEMA_VERSION = "rootscope.evidence.v1"
TERMINAL_MANIFEST_SCHEMA_VERSION = "rootscope.evidence-terminal.v1"
LIVE_STATE_SCHEMA_VERSION = "rootscope.evidence-live-state.v1"
GENESIS_HASH = "0" * 64
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_EXPECTED_KEYS = {
    "schema_version",
    "ledger_id",
    "record_index",
    "recorded_at_utc",
    "event_type",
    "task_id",
    "payload",
    "prev_hash",
    "record_hash",
}


class EvidenceVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    line_number: int
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    record_count: int
    terminal_hash: str
    issues: Tuple[VerificationIssue, ...]

    def require_valid(self) -> "VerificationReport":
        if not self.valid:
            summary = "; ".join(
                f"{issue.code}@{issue.line_number}: {issue.detail}"
                for issue in self.issues[:5]
            )
            raise EvidenceVerificationError(summary or "invalid evidence chain")
        return self


@dataclass(frozen=True)
class TerminalManifest:
    evidence_file: str
    record_count: int
    terminal_hash: str
    created_at_utc: str
    manifest_hash: str


@dataclass(frozen=True)
class LiveLedgerState:
    ledger_id: str
    evidence_file: str
    record_count: int
    terminal_hash: str
    high_watermark_task_seq: int
    updated_at_utc: str
    state_hash: str


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def canonical_record_bytes(record_without_hash: Mapping[str, Any]) -> bytes:
    return json.dumps(
        record_without_hash,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_record_hash(record_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_record_bytes(record_without_hash)).hexdigest()


def _read_lines(path: Path) -> Tuple[List[str], List[VerificationIssue]]:
    issues: List[VerificationIssue] = []
    if not path.exists():
        issues.append(VerificationIssue("FILE_MISSING", 0, str(path)))
        return [], issues
    if not path.is_file():
        issues.append(VerificationIssue("NOT_A_FILE", 0, str(path)))
        return [], issues
    raw = path.read_bytes()
    if not raw:
        return [], issues
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        issues.append(VerificationIssue("INVALID_UTF8", 0, str(exc)))
        return [], issues
    if not text.endswith("\n"):
        issues.append(
            VerificationIssue(
                "MISSING_FINAL_NEWLINE",
                text.count("\n") + 1,
                "possible interrupted append",
            )
        )
    return text.splitlines(), issues


def _validate_record_shape(
    record: Mapping[str, Any], expected_index: int, line_number: int
) -> List[VerificationIssue]:
    issues: List[VerificationIssue] = []
    missing = _EXPECTED_KEYS - set(record)
    unknown = set(record) - _EXPECTED_KEYS
    if missing or unknown:
        issues.append(
            VerificationIssue(
                "INVALID_KEYS",
                line_number,
                f"missing={sorted(missing)}, unknown={sorted(unknown)}",
            )
        )
        return issues
    if record["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        issues.append(
            VerificationIssue(
                "SCHEMA_MISMATCH", line_number, str(record["schema_version"])
            )
        )
    if not isinstance(record["ledger_id"], str) or not record["ledger_id"]:
        issues.append(
            VerificationIssue("INVALID_LEDGER_ID", line_number, "must be non-empty")
        )
    index = record["record_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index != expected_index:
        issues.append(
            VerificationIssue(
                "INDEX_MISMATCH",
                line_number,
                f"expected {expected_index}, got {index!r}",
            )
        )
    if not isinstance(record["recorded_at_utc"], str) or not record[
        "recorded_at_utc"
    ]:
        issues.append(
            VerificationIssue("INVALID_TIMESTAMP", line_number, "must be non-empty")
        )
    if not isinstance(record["event_type"], str) or not _EVENT_RE.fullmatch(
        record["event_type"]
    ):
        issues.append(
            VerificationIssue(
                "INVALID_EVENT_TYPE", line_number, repr(record["event_type"])
            )
        )
    if record["task_id"] is not None and not isinstance(record["task_id"], str):
        issues.append(
            VerificationIssue("INVALID_TASK_ID", line_number, "must be string or null")
        )
    if not isinstance(record["payload"], dict):
        issues.append(
            VerificationIssue("INVALID_PAYLOAD", line_number, "must be object")
        )
    for field_name in ("prev_hash", "record_hash"):
        value = record[field_name]
        if not isinstance(value, str) or not _HEX_64_RE.fullmatch(value):
            issues.append(
                VerificationIssue(
                    "INVALID_HASH_ENCODING", line_number, f"{field_name}={value!r}"
                )
            )
    return issues


def verify_jsonl(
    path: Path,
    *,
    expected_record_count: Optional[int] = None,
    expected_terminal_hash: Optional[str] = None,
    expected_ledger_id: Optional[str] = None,
) -> VerificationReport:
    evidence_path = Path(path)
    lines, issues = _read_lines(evidence_path)
    previous_hash = GENESIS_HASH
    valid_count = 0
    observed_ledger_id: Optional[str] = None

    for expected_index, line in enumerate(lines):
        line_number = expected_index + 1
        if not line.strip():
            issues.append(
                VerificationIssue("BLANK_LINE", line_number, "blank records forbidden")
            )
            continue
        try:
            record = strict_json_loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            issues.append(VerificationIssue("INVALID_JSON", line_number, str(exc)))
            continue
        if not isinstance(record, dict):
            issues.append(
                VerificationIssue("NOT_AN_OBJECT", line_number, type(record).__name__)
            )
            continue

        shape_issues = _validate_record_shape(record, expected_index, line_number)
        issues.extend(shape_issues)
        if shape_issues:
            continue
        if observed_ledger_id is None:
            observed_ledger_id = record["ledger_id"]
        elif record["ledger_id"] != observed_ledger_id:
            issues.append(
                VerificationIssue(
                    "LEDGER_ID_MISMATCH",
                    line_number,
                    f"expected {observed_ledger_id}, got {record['ledger_id']}",
                )
            )
        if expected_ledger_id is not None and record["ledger_id"] != expected_ledger_id:
            issues.append(
                VerificationIssue(
                    "LEDGER_ID_MISMATCH",
                    line_number,
                    f"expected live ledger {expected_ledger_id}",
                )
            )
        if record["prev_hash"] != previous_hash:
            issues.append(
                VerificationIssue(
                    "PREV_HASH_MISMATCH",
                    line_number,
                    f"expected {previous_hash}, got {record['prev_hash']}",
                )
            )
        unsigned = dict(record)
        stored_hash = unsigned.pop("record_hash")
        try:
            calculated_hash = compute_record_hash(unsigned)
        except (TypeError, ValueError) as exc:
            issues.append(
                VerificationIssue("NON_CANONICAL_PAYLOAD", line_number, str(exc))
            )
            continue
        if stored_hash != calculated_hash:
            issues.append(
                VerificationIssue(
                    "RECORD_HASH_MISMATCH",
                    line_number,
                    f"expected {calculated_hash}, got {stored_hash}",
                )
            )
        previous_hash = stored_hash
        valid_count += 1

    terminal_hash = previous_hash if valid_count else GENESIS_HASH
    if expected_record_count is not None and valid_count != expected_record_count:
        issues.append(
            VerificationIssue(
                "ANCHOR_COUNT_MISMATCH",
                0,
                f"expected {expected_record_count}, got {valid_count}",
            )
        )
    if expected_terminal_hash is not None and terminal_hash != expected_terminal_hash:
        issues.append(
            VerificationIssue(
                "ANCHOR_HASH_MISMATCH",
                0,
                f"expected {expected_terminal_hash}, got {terminal_hash}",
            )
        )
    return VerificationReport(
        valid=not issues,
        record_count=valid_count,
        terminal_hash=terminal_hash,
        issues=tuple(issues),
    )


def create_terminal_manifest(
    evidence_path: Path,
    manifest_path: Path,
    *,
    clock: Callable[[], str] = utc_now_iso,
) -> TerminalManifest:
    """Freeze a non-overwritable terminal root for export outside the ledger.

    Store the resulting manifest on a separate release/evidence medium.  A
    manifest beside a writable ledger improves accident detection but is not an
    external trust anchor and can still be replaced by an attacker.
    """

    evidence_path = Path(evidence_path)
    manifest_path = Path(manifest_path)
    report = verify_jsonl(evidence_path).require_valid()
    unsigned = {
        "schema_version": TERMINAL_MANIFEST_SCHEMA_VERSION,
        "evidence_file": evidence_path.name,
        "record_count": report.record_count,
        "terminal_hash": report.terminal_hash,
        "created_at_utc": clock(),
    }
    manifest_hash = hashlib.sha256(canonical_record_bytes(unsigned)).hexdigest()
    data = dict(unsigned)
    data["manifest_hash"] = manifest_hash
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with manifest_path.open("xb", buffering=0) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return TerminalManifest(
        evidence_file=evidence_path.name,
        record_count=report.record_count,
        terminal_hash=report.terminal_hash,
        created_at_utc=unsigned["created_at_utc"],
        manifest_hash=manifest_hash,
    )


def verify_against_terminal_manifest(
    evidence_path: Path, manifest_path: Path
) -> VerificationReport:
    manifest_path = Path(manifest_path)
    try:
        raw = manifest_path.read_text(encoding="utf-8")
        if not raw.endswith("\n"):
            raise ValueError("manifest must end with newline")
        data = strict_json_loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return VerificationReport(
            valid=False,
            record_count=0,
            terminal_hash=GENESIS_HASH,
            issues=(VerificationIssue("INVALID_TERMINAL_MANIFEST", 0, str(exc)),),
        )
    expected_keys = {
        "schema_version",
        "evidence_file",
        "record_count",
        "terminal_hash",
        "created_at_utc",
        "manifest_hash",
    }
    if not isinstance(data, dict) or set(data) != expected_keys:
        return VerificationReport(
            valid=False,
            record_count=0,
            terminal_hash=GENESIS_HASH,
            issues=(
                VerificationIssue(
                    "INVALID_TERMINAL_MANIFEST", 0, "unexpected manifest shape"
                ),
            ),
        )
    unsigned = dict(data)
    stored_manifest_hash = unsigned.pop("manifest_hash")
    try:
        calculated_manifest_hash = hashlib.sha256(
            canonical_record_bytes(unsigned)
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        return VerificationReport(
            valid=False,
            record_count=0,
            terminal_hash=GENESIS_HASH,
            issues=(VerificationIssue("INVALID_TERMINAL_MANIFEST", 0, str(exc)),),
        )
    if (
        data["schema_version"] != TERMINAL_MANIFEST_SCHEMA_VERSION
        or stored_manifest_hash != calculated_manifest_hash
        or data["evidence_file"] != Path(evidence_path).name
        or isinstance(data["record_count"], bool)
        or not isinstance(data["record_count"], int)
        or data["record_count"] < 0
        or not isinstance(data["terminal_hash"], str)
        or not _HEX_64_RE.fullmatch(data["terminal_hash"])
    ):
        return VerificationReport(
            valid=False,
            record_count=0,
            terminal_hash=GENESIS_HASH,
            issues=(
                VerificationIssue(
                    "INVALID_TERMINAL_MANIFEST", 0, "manifest field/hash mismatch"
                ),
            ),
        )
    return verify_jsonl(
        evidence_path,
        expected_record_count=data["record_count"],
        expected_terminal_hash=data["terminal_hash"],
    )


def default_live_state_path(evidence_path: Path) -> Path:
    evidence_path = Path(evidence_path)
    return evidence_path.with_name(evidence_path.name + ".state.json")


def _live_state_unsigned(state: LiveLedgerState) -> Mapping[str, Any]:
    return {
        "schema_version": LIVE_STATE_SCHEMA_VERSION,
        "ledger_id": state.ledger_id,
        "evidence_file": state.evidence_file,
        "record_count": state.record_count,
        "terminal_hash": state.terminal_hash,
        "high_watermark_task_seq": state.high_watermark_task_seq,
        "updated_at_utc": state.updated_at_utc,
    }


def make_live_state(
    *,
    ledger_id: str,
    evidence_file: str,
    record_count: int,
    terminal_hash: str,
    high_watermark_task_seq: int,
    updated_at_utc: str,
) -> LiveLedgerState:
    unsigned = {
        "schema_version": LIVE_STATE_SCHEMA_VERSION,
        "ledger_id": ledger_id,
        "evidence_file": evidence_file,
        "record_count": record_count,
        "terminal_hash": terminal_hash,
        "high_watermark_task_seq": high_watermark_task_seq,
        "updated_at_utc": updated_at_utc,
    }
    state_hash = hashlib.sha256(canonical_record_bytes(unsigned)).hexdigest()
    return LiveLedgerState(
        ledger_id=ledger_id,
        evidence_file=evidence_file,
        record_count=record_count,
        terminal_hash=terminal_hash,
        high_watermark_task_seq=high_watermark_task_seq,
        updated_at_utc=updated_at_utc,
        state_hash=state_hash,
    )


def write_live_state(
    path: Path, state: LiveLedgerState, *, exclusive: bool
) -> None:
    path = Path(path)
    unsigned = dict(_live_state_unsigned(state))
    if hashlib.sha256(canonical_record_bytes(unsigned)).hexdigest() != state.state_hash:
        raise EvidenceVerificationError("refusing to write invalid live state hash")
    data = dict(unsigned)
    data["state_hash"] = state.state_hash
    encoded = (
        json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb", buffering=0) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb", buffering=0) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_live_state(path: Path) -> LiveLedgerState:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.endswith("\n"):
            raise ValueError("live state must end with newline")
        data = strict_json_loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceVerificationError(f"invalid/missing live ledger state: {exc}") from exc
    expected = {
        "schema_version",
        "ledger_id",
        "evidence_file",
        "record_count",
        "terminal_hash",
        "high_watermark_task_seq",
        "updated_at_utc",
        "state_hash",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise EvidenceVerificationError("invalid live ledger state shape")
    for field_name in ("record_count", "high_watermark_task_seq"):
        if isinstance(data[field_name], bool) or not isinstance(data[field_name], int):
            raise EvidenceVerificationError(f"invalid live state {field_name}")
        if data[field_name] < 0:
            raise EvidenceVerificationError(f"negative live state {field_name}")
    if (
        data["schema_version"] != LIVE_STATE_SCHEMA_VERSION
        or not isinstance(data["ledger_id"], str)
        or not data["ledger_id"]
        or not isinstance(data["evidence_file"], str)
        or not isinstance(data["terminal_hash"], str)
        or not _HEX_64_RE.fullmatch(data["terminal_hash"])
        or not isinstance(data["updated_at_utc"], str)
        or not isinstance(data["state_hash"], str)
        or not _HEX_64_RE.fullmatch(data["state_hash"])
    ):
        raise EvidenceVerificationError("invalid live ledger state field")
    unsigned = dict(data)
    state_hash = unsigned.pop("state_hash")
    if hashlib.sha256(canonical_record_bytes(unsigned)).hexdigest() != state_hash:
        raise EvidenceVerificationError("live ledger state hash mismatch")
    return LiveLedgerState(
        ledger_id=data["ledger_id"],
        evidence_file=data["evidence_file"],
        record_count=data["record_count"],
        terminal_hash=data["terminal_hash"],
        high_watermark_task_seq=data["high_watermark_task_seq"],
        updated_at_utc=data["updated_at_utc"],
        state_hash=data["state_hash"],
    )


def verify_live_ledger(
    evidence_path: Path, live_state_path: Optional[Path] = None
) -> VerificationReport:
    evidence_path = Path(evidence_path)
    state_path = (
        Path(live_state_path)
        if live_state_path is not None
        else default_live_state_path(evidence_path)
    )
    try:
        state = read_live_state(state_path)
    except EvidenceVerificationError as exc:
        return VerificationReport(
            valid=False,
            record_count=0,
            terminal_hash=GENESIS_HASH,
            issues=(VerificationIssue("LIVE_STATE_INVALID", 0, str(exc)),),
        )
    if state.evidence_file != evidence_path.name:
        return VerificationReport(
            valid=False,
            record_count=0,
            terminal_hash=GENESIS_HASH,
            issues=(
                VerificationIssue(
                    "LIVE_STATE_FILE_MISMATCH", 0, state.evidence_file
                ),
            ),
        )
    return verify_jsonl(
        evidence_path,
        expected_record_count=state.record_count,
        expected_terminal_hash=state.terminal_hash,
        expected_ledger_id=state.ledger_id,
    )


def read_verified_records(path: Path) -> Tuple[Mapping[str, Any], ...]:
    report = verify_jsonl(path).require_valid()
    if report.record_count == 0:
        return ()
    lines, issues = _read_lines(Path(path))
    if issues:
        raise EvidenceVerificationError(str(issues))
    return tuple(strict_json_loads(line) for line in lines)


def load_verified_task_history(
    path: Path,
    *,
    live_state_path: Optional[Path] = None,
    require_live_state: bool = True,
) -> Tuple[TaskHistoryEntry, ...]:
    """Extract admitted task identities only after full chain verification."""

    live_state: Optional[LiveLedgerState] = None
    if require_live_state:
        state_path = (
            Path(live_state_path)
            if live_state_path is not None
            else default_live_state_path(Path(path))
        )
        verify_live_ledger(Path(path), state_path).require_valid()
        live_state = read_live_state(state_path)
    entries: List[TaskHistoryEntry] = []
    by_id: Dict[str, TaskHistoryEntry] = {}
    by_seq: Dict[int, TaskHistoryEntry] = {}
    high_watermark = 0
    for record in read_verified_records(path):
        if record["event_type"] != "task_admitted":
            continue
        payload = record["payload"]
        try:
            entry = TaskHistoryEntry(
                task_id=payload["task_id"],
                task_seq=payload["task_seq"],
                request_fingerprint=payload["request_fingerprint"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceVerificationError(
                f"malformed task_admitted record {record['record_index']}: {exc}"
            ) from exc
        if record["task_id"] != entry.task_id:
            raise EvidenceVerificationError(
                f"task_id mismatch in record {record['record_index']}"
            )
        if entry.task_id in by_id:
            raise EvidenceVerificationError(f"duplicate task_id admission: {entry.task_id}")
        if entry.task_seq in by_seq:
            raise EvidenceVerificationError(
                f"duplicate task_seq admission: {entry.task_seq}"
            )
        if entry.task_seq <= high_watermark:
            raise EvidenceVerificationError(
                "task_seq admissions must be strictly increasing"
            )
        high_watermark = entry.task_seq
        by_id[entry.task_id] = entry
        by_seq[entry.task_seq] = entry
        entries.append(entry)
    if live_state is not None and high_watermark != live_state.high_watermark_task_seq:
        raise EvidenceVerificationError(
            "task history high watermark does not match persistent live state"
        )
    return tuple(entries)
