#!/usr/bin/env python3
"""Offline A1 anchor for a completed RootScope human-review receipt.

The production ``preflight`` command is deliberately read-only.  It does not
create the A1 evidence directory, instantiate :class:`EvidenceWriter`, or write
an authority event.  Ledger creation and append are separate, explicit
commands so an incomplete human-review session can never be anchored by a
routine readiness check.

This is a local hash-chain anchor.  It is not a digital signature, blockchain,
third-party attestation, dataset lock, or grant of training/print authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence


ROOTSCOPE_ROOT = Path(__file__).resolve().parents[1]
ADVENTUREX_ROOT = ROOTSCOPE_ROOT.parent
if str(ROOTSCOPE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROOTSCOPE_ROOT))

from app.evidence import EvidenceVerificationError, EvidenceWriter, verify_live_ledger
from app.evidence.verifier import read_verified_records, strict_json_loads


REVIEW_RELATIVE_PATH = Path(
    "datasets/desert_plants_wikimedia_staging_e0/review/human_decisions"
)
DEFAULT_REVIEW_DIR = ADVENTUREX_ROOT / REVIEW_RELATIVE_PATH
DEFAULT_DATASET_ROOT = ADVENTUREX_ROOT / "datasets/desert_plants_wikimedia_staging_e0"
DEFAULT_QUEUE = DEFAULT_DATASET_ROOT / "review/candidate_review_queue.jsonl"
DEFAULT_POLICY = ADVENTUREX_ROOT / "tools/dataset/human_review_policy_v1.json"
DEFAULT_IMPLEMENTATION = ADVENTUREX_ROOT / "tools/dataset/human_review_server.py"
DEFAULT_UI = ADVENTUREX_ROOT / "tools/dataset/human_review_app.html"
DEFAULT_PRODUCTION_INPUT_FILES = {
    "review_queue_summary_sha256": DEFAULT_DATASET_ROOT
    / "review/review_queue_summary.json",
    "integrity_audit_sha256": DEFAULT_DATASET_ROOT / "integrity_audit.json",
    "dataset_manifest_schema_v2_sha256": ROOTSCOPE_ROOT
    / "training/dataset_manifest_schema_v2.json",
    "class_contract_sha256": ROOTSCOPE_ROOT / "configs/class_contract.json",
    "class_contract_lock_sha256": ROOTSCOPE_ROOT / "configs/class_contract.lock.json",
}
DEFAULT_A1_DIR = ROOTSCOPE_ROOT / "evidence/a1"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
UUID_TEXT = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
ZERO_HASH = "0" * 64
AUTHORITY = {
    "dataset_manifest_write": False,
    "training_eligibility": False,
    "split_assignment": False,
    "print_eligibility": False,
    "data_locked": False,
}
EXPLICIT_NON_CLAIMS = [
    "DATA_LOCKED",
    "TRAIN_READY",
    "SPLIT_READY",
    "FINAL_OPTICS_READY",
    "MODEL_TRAINED",
]
VALIDATION_KEYS = {
    "runtime_inputs_revalidated",
    "journal_revalidated",
    "all_referenced_rights_attestations_revalidated",
    "all_candidate_payloads_revalidated",
}
COUNT_KEYS = {
    "candidate",
    "event",
    "evented_asset",
    "reviewed_asset",
    "finalized_asset",
    "needs_review_asset",
    "reviewed_source_family",
    "family_issue",
    "referenced_rights_evidence",
}
RECEIPT_KEYS = {
    "schema_version",
    "mode",
    "status",
    "session_id",
    "session_sha256",
    "complete",
    "production_binding_enforced",
    "verified_production_input_roots",
    "queue_sha256",
    "policy_sha256",
    "implementation_sha256",
    "ui_sha256",
    "journal_sha256",
    "journal_checkpoint_sha256",
    "last_event_sha256",
    "human_review_decisions_sha256",
    "reviewed_source_families_sha256",
    "rights_evidence_reference_list_sha256",
    "snapshot_sha256",
    "rights_attestation_scope",
    "rights_source_page_payload_archived",
    "counts",
    "review_status_counts",
    "visual_axis_counts",
    "rights_axis_counts",
    "visual_pass_target_class_counts",
    "global_near_duplicate_issues",
    "validation_scope",
    "authority",
    "explicit_non_claims",
}
ANCHOR_EVENT_TYPE = "a1_human_review_complete_receipt_anchored"
ANCHOR_PAYLOAD_SCHEMA = "rootscope.a1_human_review_anchor.v1"


class AnchorError(RuntimeError):
    """A fail-closed A1 anchor contract violation."""


class AnchorNotReady(AnchorError):
    """The current review state is valid but is not eligible for anchoring."""


@dataclass(frozen=True)
class AnchorPaths:
    dataset_root: Path
    review_dir: Path
    receipt: Path
    queue: Path
    policy: Path
    implementation: Path
    ui: Path
    production_input_files: Mapping[str, Path]
    review_lock: Path
    ledger: Path
    ledger_lock: Path

    @classmethod
    def production(cls) -> "AnchorPaths":
        return cls(
            dataset_root=DEFAULT_DATASET_ROOT,
            review_dir=DEFAULT_REVIEW_DIR,
            receipt=DEFAULT_REVIEW_DIR / "human_review_receipt.json",
            queue=DEFAULT_QUEUE,
            policy=DEFAULT_POLICY,
            implementation=DEFAULT_IMPLEMENTATION,
            ui=DEFAULT_UI,
            production_input_files=DEFAULT_PRODUCTION_INPUT_FILES,
            review_lock=DEFAULT_REVIEW_DIR / ".human_review.lock",
            ledger=DEFAULT_A1_DIR / "a1_gate_ledger.jsonl",
            ledger_lock=DEFAULT_A1_DIR / ".a1_gate_ledger.lock",
        )


@dataclass(frozen=True)
class AnchorCandidate:
    receipt: Mapping[str, Any]
    receipt_sha256: str
    receipt_size_bytes: int
    eligible_to_anchor: bool


class ExclusiveFileLock:
    """Non-blocking one-byte OS lock, including across processes."""

    def __init__(self, path: Path, *, create: bool = False) -> None:
        self.path = Path(path)
        self.create = create
        self.handle: Any = None

    def __enter__(self) -> "ExclusiveFileLock":
        if self.create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                try:
                    with self.path.open("xb", buffering=0) as handle:
                        handle.write(b"\0")
                        handle.flush()
                        os.fsync(handle.fileno())
                except FileExistsError:
                    pass
        try:
            self.handle = self.path.open("r+b", buffering=0)
        except OSError as exc:
            raise AnchorError(f"required lock file is missing/unreadable: {self.path}") from exc
        try:
            self.handle.seek(0, os.SEEK_END)
            if self.handle.tell() < 1:
                raise AnchorError(f"lock file has no lock byte: {self.path}")
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ImportError, AnchorError) as exc:
            self.handle.close()
            self.handle = None
            if isinstance(exc, AnchorError):
                raise
            raise AnchorError(f"evidence path is locked by another process: {self.path}") from exc
        return self

    def __exit__(self, *_: Any) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AnchorError(f"cannot hash required artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _read_object(path: Path, context: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8", errors="strict")
        value = strict_json_loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AnchorError(f"cannot strictly parse {context}: {exc}") from exc
    if not raw.endswith(b"\n"):
        raise AnchorError(f"{context} must end with a newline")
    if not isinstance(value, dict):
        raise AnchorError(f"{context} must be a JSON object")
    return raw, value


def _read_jsonl(path: Path, context: str) -> tuple[bytes, list[Mapping[str, Any]]]:
    try:
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise AnchorError(f"cannot read {context}: {exc}") from exc
    if raw and not raw.endswith(b"\n"):
        raise AnchorError(f"{context} must end with a newline")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise AnchorError(f"{context} contains blank line {line_number}")
        try:
            value = strict_json_loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise AnchorError(f"cannot strictly parse {context} line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise AnchorError(f"{context} line {line_number} must be an object")
        rows.append(value)
    return raw, rows


def _require_hex64(value: Any, context: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise AnchorError(f"{context} must be a lowercase SHA-256")
    return value


def _require_uint(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnchorError(f"{context} must be a non-negative integer")
    return value


def _require_file_hash(path: Path, expected: Any, context: str) -> str:
    expected_hash = _require_hex64(expected, f"{context} expected hash")
    actual = _sha256_file(path)
    if actual != expected_hash:
        raise AnchorError(f"{context} SHA-256 mismatch")
    return actual


def _validate_candidate_payloads(paths: AnchorPaths, candidate_count: int) -> None:
    _, rows = _read_jsonl(paths.queue, "candidate review queue")
    if len(rows) != candidate_count:
        raise AnchorError("candidate queue length conflicts with COMPLETE receipt")
    seen: set[str] = set()
    root = paths.dataset_root.resolve(strict=True)
    for index, row in enumerate(rows, start=1):
        local_path = row.get("local_path")
        claimed = row.get("sha256")
        if not isinstance(local_path, str) or not local_path or "\\" in local_path:
            raise AnchorError(f"candidate row {index} has an unsafe local_path")
        pure = PurePosixPath(local_path)
        if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
            raise AnchorError(f"candidate row {index} has an unsafe local_path")
        candidate = root.joinpath(*pure.parts).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise AnchorError(f"candidate row {index} escapes dataset root") from exc
        if local_path in seen:
            raise AnchorError(f"duplicate candidate local_path: {local_path}")
        seen.add(local_path)
        _require_file_hash(candidate, claimed, f"candidate row {index}")


def _validate_receipt(paths: AnchorPaths, *, require_complete: bool) -> AnchorCandidate:
    receipt_raw, receipt = _read_object(paths.receipt, "human review receipt")
    if set(receipt) != RECEIPT_KEYS:
        raise AnchorError("human review receipt fields do not match the frozen schema")
    if receipt["schema_version"] != "rootscope.wikimedia_human_review_receipt.v1":
        raise AnchorError("human review receipt schema mismatch")
    if receipt["mode"] != "PRODUCTION" or receipt["production_binding_enforced"] is not True:
        raise AnchorError("only a production-bound human review receipt can be anchored")
    if receipt["authority"] != AUTHORITY:
        raise AnchorError("human review receipt claims forbidden authority")
    if receipt["explicit_non_claims"] != EXPLICIT_NON_CLAIMS:
        raise AnchorError("human review receipt non-claims changed")
    if receipt["rights_attestation_scope"] != "PERMALINK_BOUND_HUMAN_ATTESTATION_NO_PAGE_SNAPSHOT":
        raise AnchorError("human review rights attestation scope changed")
    if receipt["rights_source_page_payload_archived"] is not False:
        raise AnchorError("human review receipt makes an unsupported page archive claim")
    session_id = receipt["session_id"]
    if not isinstance(session_id, str) or UUID_TEXT.fullmatch(session_id) is None:
        raise AnchorError("human review receipt session_id is invalid")

    for key in (
        "session_sha256",
        "queue_sha256",
        "policy_sha256",
        "implementation_sha256",
        "ui_sha256",
        "journal_sha256",
        "journal_checkpoint_sha256",
        "last_event_sha256",
        "human_review_decisions_sha256",
        "reviewed_source_families_sha256",
        "rights_evidence_reference_list_sha256",
        "snapshot_sha256",
    ):
        _require_hex64(receipt[key], f"receipt.{key}")

    counts = receipt["counts"]
    if not isinstance(counts, dict) or set(counts) != COUNT_KEYS:
        raise AnchorError("human review receipt counts shape changed")
    for key, value in counts.items():
        _require_uint(value, f"receipt.counts.{key}")
    if counts["candidate"] <= 0:
        raise AnchorError("human review receipt candidate count must be positive")

    validation = receipt["validation_scope"]
    if not isinstance(validation, dict) or set(validation) != VALIDATION_KEYS:
        raise AnchorError("human review receipt validation scope changed")
    if any(type(value) is not bool for value in validation.values()):
        raise AnchorError("human review receipt validation flags must be booleans")
    for required_true in (
        "runtime_inputs_revalidated",
        "journal_revalidated",
        "all_referenced_rights_attestations_revalidated",
    ):
        if validation[required_true] is not True:
            raise AnchorError(f"human review receipt lacks {required_true}")

    complete = receipt["complete"]
    if type(complete) is not bool:
        raise AnchorError("human review receipt complete flag must be boolean")
    expected_status = (
        "HUMAN_REVIEW_COMPLETE_NOT_DATA_LOCKED"
        if complete
        else "HUMAN_REVIEW_IN_PROGRESS_NOT_DATA_LOCKED"
    )
    if receipt["status"] != expected_status:
        raise AnchorError("human review receipt complete/status fields conflict")

    _require_file_hash(paths.queue, receipt["queue_sha256"], "candidate queue")
    _require_file_hash(paths.policy, receipt["policy_sha256"], "review policy")
    _require_file_hash(
        paths.implementation, receipt["implementation_sha256"], "review implementation"
    )
    _require_file_hash(paths.ui, receipt["ui_sha256"], "review UI")

    _, policy = _read_object(paths.policy, "review policy")
    roots = receipt["verified_production_input_roots"]
    expected_root_keys = {"candidate_review_queue_sha256", *paths.production_input_files.keys()}
    if not isinstance(roots, dict) or set(roots) != expected_root_keys:
        raise AnchorError("verified production input root set changed")
    if roots["candidate_review_queue_sha256"] != receipt["queue_sha256"]:
        raise AnchorError("candidate queue root conflicts inside receipt")
    if policy.get("production_input_roots") != roots:
        raise AnchorError("review policy production roots conflict with receipt")
    for key, path in paths.production_input_files.items():
        _require_file_hash(path, roots[key], f"production input root {key}")

    session_raw, session = _read_object(paths.review_dir / "session.json", "review session")
    if _sha256_bytes(session_raw) != receipt["session_sha256"]:
        raise AnchorError("review session SHA-256 mismatch")
    if session.get("session_id") != session_id:
        raise AnchorError("review session identity conflicts with receipt")

    journal_raw, journal_rows = _read_jsonl(
        paths.review_dir / "decision_journal.jsonl", "decision journal"
    )
    if _sha256_bytes(journal_raw) != receipt["journal_sha256"]:
        raise AnchorError("decision journal SHA-256 mismatch")
    if len(journal_rows) != counts["event"]:
        raise AnchorError("decision journal record count conflicts with receipt")
    observed_last = ZERO_HASH if not journal_rows else journal_rows[-1].get("event_sha256")
    if observed_last != receipt["last_event_sha256"]:
        raise AnchorError("decision journal terminal event conflicts with receipt")

    checkpoint_raw, checkpoint = _read_object(
        paths.review_dir / "journal_checkpoint.json", "journal checkpoint"
    )
    if _sha256_bytes(checkpoint_raw) != receipt["journal_checkpoint_sha256"]:
        raise AnchorError("journal checkpoint SHA-256 mismatch")
    checkpoint_expected = {
        "session_id": session_id,
        "session_sha256": receipt["session_sha256"],
        "event_count": counts["event"],
        "journal_sha256": receipt["journal_sha256"],
        "last_event_sha256": receipt["last_event_sha256"],
    }
    for key, value in checkpoint_expected.items():
        if checkpoint.get(key) != value:
            raise AnchorError(f"journal checkpoint conflicts at {key}")

    decisions_raw, decisions = _read_jsonl(
        paths.review_dir / "human_review_decisions.jsonl", "human review decisions"
    )
    if _sha256_bytes(decisions_raw) != receipt["human_review_decisions_sha256"]:
        raise AnchorError("human review decisions SHA-256 mismatch")
    if len(decisions) != counts["evented_asset"]:
        raise AnchorError("decision export count conflicts with receipt")

    families_raw, families = _read_jsonl(
        paths.review_dir / "reviewed_source_families.jsonl", "reviewed source families"
    )
    if _sha256_bytes(families_raw) != receipt["reviewed_source_families_sha256"]:
        raise AnchorError("reviewed source families SHA-256 mismatch")
    if len(families) != counts["reviewed_source_family"]:
        raise AnchorError("reviewed source family count conflicts with receipt")

    snapshot_raw, snapshot = _read_object(
        paths.review_dir / "latest_decisions.json", "latest decision snapshot"
    )
    if _sha256_bytes(snapshot_raw) != receipt["snapshot_sha256"]:
        raise AnchorError("latest decision snapshot SHA-256 mismatch")
    for key, value in {
        "session_id": session_id,
        "status": receipt["status"],
        "queue_sha256": receipt["queue_sha256"],
        "policy_sha256": receipt["policy_sha256"],
    }.items():
        if snapshot.get(key) != value:
            raise AnchorError(f"latest decision snapshot conflicts at {key}")

    rights_references: set[str] = set()
    for index, record in enumerate(decisions, start=1):
        decision = record.get("decision")
        if not isinstance(decision, dict):
            raise AnchorError(f"decision export row {index} lacks a decision object")
        reference = decision.get("rights_evidence_sha256", "")
        if reference:
            rights_references.add(_require_hex64(reference, f"decision row {index} rights evidence"))
    rights_list = "".join(f"{value}\n" for value in sorted(rights_references)).encode("ascii")
    if _sha256_bytes(rights_list) != receipt["rights_evidence_reference_list_sha256"]:
        raise AnchorError("rights evidence reference list SHA-256 mismatch")
    if len(rights_references) != counts["referenced_rights_evidence"]:
        raise AnchorError("rights evidence reference count conflicts with receipt")
    for reference in rights_references:
        _require_file_hash(
            paths.review_dir / "rights_evidence" / f"{reference}.json",
            reference,
            f"rights evidence {reference}",
        )

    eligible = complete
    if complete:
        if validation["all_candidate_payloads_revalidated"] is not True:
            raise AnchorError("COMPLETE receipt lacks full candidate payload revalidation")
        completion_counts = {
            "evented_asset": counts["candidate"],
            "reviewed_asset": counts["candidate"],
            "finalized_asset": counts["candidate"],
            "needs_review_asset": 0,
            "family_issue": 0,
        }
        for key, value in completion_counts.items():
            if counts[key] != value:
                raise AnchorError(f"COMPLETE receipt count conflicts at {key}")
        if receipt["global_near_duplicate_issues"] != []:
            raise AnchorError("COMPLETE receipt contains unresolved near-duplicate issues")
        _validate_candidate_payloads(paths, counts["candidate"])

    receipt_raw_after = paths.receipt.read_bytes()
    if receipt_raw_after != receipt_raw:
        raise AnchorError("human review receipt changed during validation")
    candidate = AnchorCandidate(
        receipt=receipt,
        receipt_sha256=_sha256_bytes(receipt_raw),
        receipt_size_bytes=len(receipt_raw),
        eligible_to_anchor=eligible,
    )
    if require_complete and not eligible:
        raise AnchorNotReady("human review is valid but not COMPLETE; no anchor was written")
    return candidate


def preflight(paths: AnchorPaths) -> Mapping[str, Any]:
    """Read-only validation.  This function never opens or initializes a ledger."""

    with ExclusiveFileLock(paths.review_lock, create=False):
        candidate = _validate_receipt(paths, require_complete=False)
    return {
        "schema_version": "rootscope.a1_human_review_anchor_preflight.v1",
        "status": "READY_TO_ANCHOR" if candidate.eligible_to_anchor else "WAITING_FOR_HUMAN_REVIEW",
        "eligible_to_anchor": candidate.eligible_to_anchor,
        "observed_receipt_sha256": candidate.receipt_sha256,
        "session_id": candidate.receipt["session_id"],
        "human_review_status": candidate.receipt["status"],
        "human_review_event_count": candidate.receipt["counts"]["event"],
        "a1_ledger_exists": paths.ledger.is_file(),
        "authoritative_event_written": False,
        "authority": AUTHORITY,
    }


def initialize_ledger(paths: AnchorPaths) -> Mapping[str, Any]:
    """Explicitly create an empty local A1 ledger; never called by preflight."""

    paths.ledger.parent.mkdir(parents=True, exist_ok=True)
    with ExclusiveFileLock(paths.ledger_lock, create=True):
        writer = EvidenceWriter(paths.ledger, initialize=True)
        report = verify_live_ledger(paths.ledger).require_valid()
        del writer
    return {
        "status": "EMPTY_A1_LEDGER_INITIALIZED",
        "record_count": report.record_count,
        "terminal_hash": report.terminal_hash,
        "authority": AUTHORITY,
    }


def _anchor_payload(candidate: AnchorCandidate, *, supersedes: Optional[str]) -> Mapping[str, Any]:
    receipt = candidate.receipt
    return {
        "schema_version": ANCHOR_PAYLOAD_SCHEMA,
        "anchor_kind": "HUMAN_REVIEW_COMPLETE_RECEIPT",
        "claim_scope": "LOCAL_DIRECTORY_EXTERNAL_HASH_ANCHOR_NOT_SIGNATURE_NOT_DATA_LOCK",
        "source_relpath": REVIEW_RELATIVE_PATH.joinpath("human_review_receipt.json").as_posix(),
        "receipt_sha256": candidate.receipt_sha256,
        "receipt_size_bytes": candidate.receipt_size_bytes,
        "receipt_schema_version": receipt["schema_version"],
        "session_id": receipt["session_id"],
        "session_sha256": receipt["session_sha256"],
        "mode": receipt["mode"],
        "status": receipt["status"],
        "complete": receipt["complete"],
        "queue_sha256": receipt["queue_sha256"],
        "policy_sha256": receipt["policy_sha256"],
        "implementation_sha256": receipt["implementation_sha256"],
        "ui_sha256": receipt["ui_sha256"],
        "journal_sha256": receipt["journal_sha256"],
        "journal_checkpoint_sha256": receipt["journal_checkpoint_sha256"],
        "last_event_sha256": receipt["last_event_sha256"],
        "human_review_decisions_sha256": receipt["human_review_decisions_sha256"],
        "reviewed_source_families_sha256": receipt["reviewed_source_families_sha256"],
        "snapshot_sha256": receipt["snapshot_sha256"],
        "rights_evidence_reference_list_sha256": receipt[
            "rights_evidence_reference_list_sha256"
        ],
        "counts": receipt["counts"],
        "verified_production_input_roots": receipt["verified_production_input_roots"],
        "validation_scope": receipt["validation_scope"],
        "authority": AUTHORITY,
        "supersedes_receipt_sha256": supersedes,
    }


def anchor_complete(
    paths: AnchorPaths, *, supersedes_receipt_sha256: Optional[str] = None
) -> Mapping[str, Any]:
    """Append one COMPLETE receipt root to an explicitly initialized ledger."""

    if supersedes_receipt_sha256 is not None:
        _require_hex64(supersedes_receipt_sha256, "supersedes_receipt_sha256")
    with ExclusiveFileLock(paths.review_lock, create=False):
        with ExclusiveFileLock(paths.ledger_lock, create=False):
            candidate = _validate_receipt(paths, require_complete=True)
            before = verify_live_ledger(paths.ledger).require_valid()
            records = read_verified_records(paths.ledger)
            anchors = [
                record
                for record in records
                if record.get("event_type") == ANCHOR_EVENT_TYPE
                and isinstance(record.get("payload"), dict)
                and record["payload"].get("session_id") == candidate.receipt["session_id"]
            ]
            latest = anchors[-1] if anchors else None
            if latest is not None and latest["payload"].get("receipt_sha256") == candidate.receipt_sha256:
                return {
                    "status": "ALREADY_ANCHORED_IDEMPOTENT",
                    "record_count": before.record_count,
                    "record_hash": latest["record_hash"],
                    "receipt_sha256": candidate.receipt_sha256,
                    "authority": AUTHORITY,
                }
            if latest is None:
                if supersedes_receipt_sha256 is not None:
                    raise AnchorError("supersedes was supplied but this session has no prior anchor")
            else:
                latest_sha = latest["payload"].get("receipt_sha256")
                if supersedes_receipt_sha256 != latest_sha:
                    raise AnchorError("a changed receipt requires the exact latest anchor in --supersedes")

            writer = EvidenceWriter(paths.ledger, initialize=False)
            evidence_receipt = writer.append(
                ANCHOR_EVENT_TYPE,
                _anchor_payload(candidate, supersedes=supersedes_receipt_sha256),
                task_id=None,
            )
            after = verify_live_ledger(paths.ledger).require_valid()
            if after.record_count != before.record_count + 1:
                raise AnchorError("A1 ledger did not extend by exactly one record")
            if after.terminal_hash != evidence_receipt.record_hash:
                raise AnchorError("A1 ledger terminal hash conflicts with append receipt")
            return {
                "status": "COMPLETE_RECEIPT_ANCHORED_NOT_DATA_LOCKED",
                "record_count": after.record_count,
                "record_index": evidence_receipt.record_index,
                "record_hash": evidence_receipt.record_hash,
                "receipt_sha256": candidate.receipt_sha256,
                "authority": AUTHORITY,
            }


def verify_current(paths: AnchorPaths) -> Mapping[str, Any]:
    """Verify that the current COMPLETE receipt is the session's latest anchor."""

    with ExclusiveFileLock(paths.review_lock, create=False):
        with ExclusiveFileLock(paths.ledger_lock, create=False):
            candidate = _validate_receipt(paths, require_complete=True)
            report = verify_live_ledger(paths.ledger).require_valid()
            anchors = [
                record
                for record in read_verified_records(paths.ledger)
                if record.get("event_type") == ANCHOR_EVENT_TYPE
                and isinstance(record.get("payload"), dict)
                and record["payload"].get("session_id") == candidate.receipt["session_id"]
            ]
            if not anchors:
                raise AnchorError("current COMPLETE receipt has no A1 anchor")
            latest = anchors[-1]
            if latest["payload"].get("receipt_sha256") != candidate.receipt_sha256:
                raise AnchorError("current receipt SHA-256 is not the session's latest A1 anchor")
            return {
                "status": "CURRENT_COMPLETE_RECEIPT_ANCHOR_VERIFIED_NOT_DATA_LOCKED",
                "record_count": report.record_count,
                "record_hash": latest["record_hash"],
                "receipt_sha256": candidate.receipt_sha256,
                "authority": AUTHORITY,
            }


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="read-only production receipt readiness check")
    init_parser = subparsers.add_parser("init-ledger", help="explicitly initialize an empty A1 ledger")
    init_parser.add_argument("--confirm-empty-ledger-init", action="store_true")
    anchor_parser = subparsers.add_parser("anchor-complete", help="append a COMPLETE receipt root")
    anchor_parser.add_argument("--confirm-authoritative-append", action="store_true")
    anchor_parser.add_argument("--supersedes")
    subparsers.add_parser("verify-current", help="read-only current receipt/anchor verification")
    args = parser.parse_args(argv)
    paths = AnchorPaths.production()
    try:
        if args.command == "preflight":
            result = preflight(paths)
            _emit(result)
            return 0 if result["eligible_to_anchor"] else 3
        if args.command == "init-ledger":
            if not args.confirm_empty_ledger_init:
                raise AnchorError("init-ledger requires --confirm-empty-ledger-init")
            _emit(initialize_ledger(paths))
            return 0
        if args.command == "anchor-complete":
            if not args.confirm_authoritative_append:
                raise AnchorError("anchor-complete requires --confirm-authoritative-append")
            _emit(anchor_complete(paths, supersedes_receipt_sha256=args.supersedes))
            return 0
        if args.command == "verify-current":
            _emit(verify_current(paths))
            return 0
        raise AnchorError("unsupported command")
    except AnchorNotReady as exc:
        _emit({"ok": False, "status": "WAITING_FOR_HUMAN_REVIEW", "error": str(exc)})
        return 3
    except (AnchorError, EvidenceVerificationError, OSError) as exc:
        _emit({"ok": False, "status": "FAIL_CLOSED", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
