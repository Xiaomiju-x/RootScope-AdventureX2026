#!/usr/bin/env python3
"""Loopback-only, append-only reviewer for the RootScope candidate queue.

This tool records human decisions but never edits the staging manifest and never
grants training or print eligibility.  A later, separately audited A1 export is
required before any decision can enter a production dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import secrets
import tempfile
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlparse


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGING = DEFAULT_ROOT / "datasets" / "desert_plants_wikimedia_staging_e0"
DEFAULT_QUEUE = DEFAULT_STAGING / "review" / "candidate_review_queue.jsonl"
DEFAULT_POLICY = Path(__file__).with_name("human_review_policy_v1.json")
DEFAULT_UI = Path(__file__).with_name("human_review_app.html")
DEFAULT_OUTPUT = DEFAULT_STAGING / "review" / "human_decisions"
DEFAULT_PRODUCTION_INPUTS = {
    "review_queue_summary_sha256": DEFAULT_STAGING / "review" / "review_queue_summary.json",
    "integrity_audit_sha256": DEFAULT_STAGING / "integrity_audit.json",
    "dataset_manifest_schema_v2_sha256": DEFAULT_ROOT / "rootscope" / "training" / "dataset_manifest_schema_v2.json",
    "class_contract_sha256": DEFAULT_ROOT / "rootscope" / "configs" / "class_contract.json",
    "class_contract_lock_sha256": DEFAULT_ROOT / "rootscope" / "configs" / "class_contract.lock.json",
}
PRODUCTION_ROOT_KEYS = {
    "candidate_review_queue_sha256",
    *DEFAULT_PRODUCTION_INPUTS.keys(),
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UUID_TEXT = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
UTC_TEXT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
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
CLIENT_DECISION_FIELDS = {
    "visual_decision",
    "rights_decision",
    "target_class",
    "unknown_scenario",
    "reviewed_source_group",
    "family_role",
    "near_duplicate_family",
    "visual_reviewer",
    "rights_reviewer",
    "rights_source_page_checked",
    "rights_evidence_sha256",
    "source_page_revision_id",
    "visual_reason_codes",
    "rights_reason_codes",
    "notes",
}
PERSISTED_DECISION_FIELDS = CLIENT_DECISION_FIELDS | {
    "visual_reviewed_at_utc",
    "rights_reviewed_at_utc",
}


class ReviewError(RuntimeError):
    """Fail-closed review contract violation."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_bytes(raw: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ReviewError) as exc:
        raise ReviewError(f"cannot parse {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewError(f"{context} must be a JSON object")
    return value


def _read_json_and_sha(path: Path, context: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReviewError(f"cannot read {context}: {exc}") from exc
    return _parse_json_bytes(raw, context), _sha256_bytes(raw)


def _read_json(path: Path, context: str) -> dict[str, Any]:
    return _read_json_and_sha(path, context)[0]


def _read_jsonl(path: Path, context: str) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReviewError(f"cannot read {context}: {exc}") from exc
    return raw, _parse_jsonl_bytes(raw, context)


def _parse_jsonl_bytes(raw: bytes, context: str) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewError(f"cannot decode {context}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ReviewError(f"{context} has blank line {line_number}")
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, ReviewError) as exc:
            raise ReviewError(f"{context} line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ReviewError(f"{context} line {line_number} is not an object")
        rows.append(value)
    return rows


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    line = _canonical_bytes(value) + b"\n"
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


def _source_permanent_url(source_url: str, revision_id: str) -> str:
    parsed = urlparse(source_url)
    query = [(key, value) for key, value in parse_qsl(parsed.query) if key != "oldid"]
    query.append(("oldid", revision_id))
    return parsed._replace(query=urlencode(query)).geturl()


def _acquire_process_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ImportError) as exc:
        handle.close()
        raise ReviewError("review output directory is already locked by another process") from exc
    return handle


def _release_process_lock(handle: Any) -> None:
    if handle is None or handle.closed:
        return
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


class ReviewStore:
    """Hash-bound queue plus a single-writer decision journal."""

    def __init__(
        self,
        queue_path: Path,
        output_dir: Path,
        policy_path: Path,
        ui_path: Path = DEFAULT_UI,
    ):
        self.queue_path = Path(queue_path).resolve(strict=True)
        self.dataset_root = self.queue_path.parent.parent.resolve(strict=True)
        self.output_dir = Path(output_dir).resolve(strict=False)
        self.policy_path = Path(policy_path).resolve(strict=True)
        self.ui_path = Path(ui_path).resolve(strict=True)
        self.tool_path = Path(__file__).resolve(strict=True)
        self.production_mode = (
            self.queue_path == DEFAULT_QUEUE.resolve(strict=False)
            and self.policy_path == DEFAULT_POLICY.resolve(strict=False)
            and self.ui_path == DEFAULT_UI.resolve(strict=False)
            and self.output_dir == DEFAULT_OUTPUT.resolve(strict=False)
        )
        if (
            self.output_dir == DEFAULT_OUTPUT.resolve(strict=False)
            and not self.production_mode
        ):
            raise ReviewError(
                "fixture/non-default review inputs may not use the production output directory"
            )
        self.policy, self.policy_sha256 = _read_json_and_sha(
            self.policy_path, "review policy"
        )
        self.tool_sha256 = _sha256_file(self.tool_path)
        self.ui_sha256 = _sha256_file(self.ui_path)
        self._validate_policy()

        queue_raw, rows = _read_jsonl(self.queue_path, "candidate review queue")
        self.queue_sha256 = _sha256_bytes(queue_raw)
        if (
            self.queue_sha256
            != self.policy["production_input_roots"]["candidate_review_queue_sha256"]
        ):
            raise ReviewError("candidate queue does not match the policy-bound queue SHA-256")
        self.items: list[dict[str, Any]] = []
        self.items_by_asset: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(rows, start=1):
            self._validate_queue_row(row, index)
            asset = row["asset"]
            if asset in self.items_by_asset:
                raise ReviewError(f"duplicate queue asset: {asset}")
            self.items.append(row)
            self.items_by_asset[asset] = row
        if not self.items:
            raise ReviewError("candidate review queue is empty")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_path = self.output_dir / "session.json"
        self.journal_path = self.output_dir / "decision_journal.jsonl"
        self.pending_path = self.output_dir / "pending_event.json"
        self.snapshot_path = self.output_dir / "latest_decisions.json"
        self.decisions_path = self.output_dir / "human_review_decisions.jsonl"
        self.families_path = self.output_dir / "reviewed_source_families.jsonl"
        self.receipt_path = self.output_dir / "human_review_receipt.json"
        self.checkpoint_path = self.output_dir / "journal_checkpoint.json"
        self.rights_evidence_dir = self.output_dir / "rights_evidence"
        self.recovery_dir = self.output_dir / "recovery"
        self.process_lock_path = self.output_dir / ".human_review.lock"
        self._lock = threading.RLock()
        self._process_lock_handle = None
        self._closed = False
        self.degraded = False
        self.latest_by_asset: dict[str, dict[str, Any]] = {}
        self.last_event_sha256 = "0" * 64
        self.last_event: dict[str, Any] | None = None
        self.event_count = 0
        self.event_ids: set[str] = set()
        self.events: list[dict[str, Any]] = []
        self.journal_size = 0
        self.journal_file_sha256 = _sha256_bytes(b"")
        self.verified_production_input_roots: dict[str, str] = {}
        self.session_sha256 = ""
        self._startup_verified_candidate_shas: set[str] = set()
        self._process_lock_handle = _acquire_process_lock(self.process_lock_path)
        try:
            self._verify_runtime_inputs()
            self._open_or_create_session()
            self._load_journal()
            self._recover_pending_event()
            self._write_snapshot()
            if self.pending_path.exists():
                self.pending_path.unlink()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _release_process_lock(self._process_lock_handle)
        self._process_lock_handle = None

    def __enter__(self) -> "ReviewStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _verify_runtime_inputs(self) -> None:
        expected_files = {
            "candidate review queue": (self.queue_path, self.queue_sha256),
            "review policy": (self.policy_path, self.policy_sha256),
            "review server implementation": (self.tool_path, self.tool_sha256),
            "review UI": (self.ui_path, self.ui_sha256),
        }
        for context, (path, expected_sha) in expected_files.items():
            if _sha256_file(path) != expected_sha:
                raise ReviewError(f"{context} changed after the review session was bound")
        if self.session_sha256:
            if not self.session_path.is_file() or _sha256_file(self.session_path) != self.session_sha256:
                raise ReviewError("review session file changed after startup")
        if not self.production_mode:
            self.verified_production_input_roots = {}
            return
        roots = self.policy["production_input_roots"]
        verified = {"candidate_review_queue_sha256": self.queue_sha256}
        for root_name, path in DEFAULT_PRODUCTION_INPUTS.items():
            expected_sha = roots.get(root_name)
            actual_sha = _sha256_file(path.resolve(strict=True))
            if expected_sha is None or actual_sha != expected_sha:
                raise ReviewError(f"production input root mismatch: {root_name}")
            verified[root_name] = actual_sha
        contract_path = DEFAULT_PRODUCTION_INPUTS["class_contract_sha256"]
        lock_path = DEFAULT_PRODUCTION_INPUTS["class_contract_lock_sha256"]
        contract, contract_sha = _read_json_and_sha(contract_path, "class contract")
        lock, lock_sha = _read_json_and_sha(lock_path, "class contract lock")
        if contract_sha != roots["class_contract_sha256"]:
            raise ReviewError("class contract byte root changed")
        if lock_sha != roots["class_contract_lock_sha256"]:
            raise ReviewError("class contract lock byte root changed")
        if lock != {
            "schema_version": "1.0.0",
            "profile": "rootscope.dataset_contract.production.v2",
            "contract_version": "2.0.0",
            "class_contract_sha256": contract_sha,
        }:
            raise ReviewError("class contract lock semantics changed")
        dataset_contract = contract.get("dataset_contract")
        if contract.get("class_order") != self.policy["target_classes"] or not isinstance(
            dataset_contract, dict
        ) or dataset_contract.get("unknown_scenarios") != self.policy["unknown_scenarios"]:
            raise ReviewError("review policy taxonomy conflicts with the frozen class contract")
        if verified != roots:
            raise ReviewError("verified production roots do not exactly match the frozen policy")
        self.verified_production_input_roots = verified

    def _validate_policy(self) -> None:
        expected = {
            "schema_version": "rootscope.wikimedia_human_review_policy.v1",
            "queue_schema_version": "rootscope.wikimedia_human_review_queue.v1",
            "decision_schema_version": "rootscope.wikimedia_human_review_decision.v1",
            "session_schema_version": "rootscope.wikimedia_human_review_session.v1",
            "snapshot_schema_version": "rootscope.wikimedia_human_review_snapshot.v1",
        }
        for key, value in expected.items():
            if self.policy.get(key) != value:
                raise ReviewError(f"unsupported review policy {key}")
        for key in (
            "visual_decisions",
            "rights_decisions",
            "target_classes",
            "unknown_scenarios",
            "family_roles",
            "visual_reason_codes",
            "rights_reason_codes",
        ):
            values = self.policy.get(key)
            if not isinstance(values, list) or not values or len(values) != len(set(values)):
                raise ReviewError(f"review policy {key} must be a unique non-empty list")
        if self.policy["visual_decisions"] != ["UNREVIEWED", "PASS", "REJECT", "NEEDS_REVIEW"]:
            raise ReviewError("visual decision state machine changed")
        if self.policy["rights_decisions"] != ["UNREVIEWED", "PASS", "REJECT", "NEEDS_REVIEW"]:
            raise ReviewError("rights decision state machine changed")
        if self.policy["family_roles"] != [
            "UNASSIGNED",
            "CANONICAL_REPRESENTATIVE",
            "SERIES_SIBLING_EXCLUDED",
            "HOLD",
        ]:
            raise ReviewError("family role state machine changed")
        roots = self.policy.get("production_input_roots")
        if not isinstance(roots, dict) or not roots:
            raise ReviewError("production_input_roots must be a non-empty object")
        if set(roots) != PRODUCTION_ROOT_KEYS:
            raise ReviewError("production_input_roots keys do not match the frozen set")
        if any(not isinstance(value, str) or HEX64.fullmatch(value) is None for value in roots.values()):
            raise ReviewError("production_input_roots must contain SHA-256 values")
        for key in (
            "reviewer_pattern",
            "reviewed_source_group_pattern",
            "near_duplicate_family_pattern",
            "source_page_revision_id_pattern",
        ):
            try:
                re.compile(self.policy[key])
            except (KeyError, re.error) as exc:
                raise ReviewError(f"invalid review policy regex {key}") from exc
        if type(self.policy.get("notes_max_length")) is not int or self.policy["notes_max_length"] < 1:
            raise ReviewError("invalid notes_max_length")
        expected_rules = {
            "pass_requires_visual_and_rights_pass": True,
            "visual_pass_requires_target_class": True,
            "rights_pass_or_reject_requires_source_page_check": True,
            "rights_pass_or_reject_requires_source_page_revision_id": True,
            "rights_pass_or_reject_requires_evidence_sha256": True,
            "rights_attestation_archives_page_snapshot": False,
            "pass_pair_requires_reviewed_source_group": True,
            "pass_pair_requires_family_role": True,
            "series_sibling_requires_near_duplicate_family": True,
            "reject_requires_axis_reason": True,
            "every_non_unreviewed_axis_requires_its_reviewer": True,
            "reviewed_source_group_is_independence_family_not_v2_source_group": True,
            "one_canonical_representative_per_reviewed_source_group": True,
            "acquisition_hint_must_not_auto_map_unknown_scenario": True,
            "decisions_do_not_grant_training_or_print_eligibility": True,
            "export_status_before_dataset_audit": "HUMAN_REVIEW_IN_PROGRESS_NOT_DATA_LOCKED",
        }
        if self.policy.get("rules") != expected_rules:
            raise ReviewError("review policy rules changed")

    def _validate_queue_row(self, row: dict[str, Any], index: int) -> None:
        context = f"queue line {index}"
        if row.get("schema_version") != self.policy["queue_schema_version"]:
            raise ReviewError(f"{context}: unsupported schema")
        if row.get("review_status") != "UNREVIEWED":
            raise ReviewError(f"{context}: input queue is not UNREVIEWED")
        if row.get("split") != "UNASSIGNED_DO_NOT_TRAIN":
            raise ReviewError(f"{context}: input queue is not locked from training")
        if row.get("training_eligible") is not False or row.get("print_eligible") is not False:
            raise ReviewError(f"{context}: input queue has forbidden eligibility")
        for field in ("asset", "local_path", "source_url", "sha256", "class_hint", "source_group"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ReviewError(f"{context}: invalid {field}")
        if HEX64.fullmatch(row["sha256"]) is None:
            raise ReviewError(f"{context}: invalid SHA-256")
        if row["class_hint"] not in self.policy["target_classes"]:
            raise ReviewError(f"{context}: invalid class hint")
        self._bound_image_path(row)

    def _bound_image_path(self, row: Mapping[str, Any]) -> Path:
        relative = row.get("local_path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ReviewError("queue image path is not a POSIX relative path")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ReviewError("queue image path is unsafe")
        windows_reserved = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{index}" for index in range(1, 10)),
            *(f"LPT{index}" for index in range(1, 10)),
        }
        for part in pure.parts:
            stem = part.split(".", 1)[0].upper()
            if ":" in part or part.endswith((" ", ".")) or stem in windows_reserved:
                raise ReviewError("queue image path is unsafe on Windows")
        candidate = self.dataset_root.joinpath(*pure.parts).resolve(strict=True)
        try:
            candidate.relative_to(self.dataset_root)
        except ValueError as exc:
            raise ReviewError("queue image path escapes dataset root") from exc
        if not candidate.is_file():
            raise ReviewError("queue image is not a regular file")
        return candidate

    def _open_or_create_session(self) -> None:
        expected = {
            "schema_version": self.policy["session_schema_version"],
            "mode": "PRODUCTION" if self.production_mode else "FIXTURE",
            "queue_path": self.queue_path.as_posix(),
            "queue_sha256": self.queue_sha256,
            "policy_path": self.policy_path.as_posix(),
            "policy_sha256": self.policy_sha256,
            "implementation_sha256": self.tool_sha256,
            "ui_path": self.ui_path.as_posix(),
            "ui_sha256": self.ui_sha256,
            "candidate_count": len(self.items),
            "production_binding_enforced": self.production_mode,
            "verified_production_input_roots": self.verified_production_input_roots,
            "authority": AUTHORITY,
            "explicit_non_claims": EXPLICIT_NON_CLAIMS,
        }
        if self.session_path.is_file():
            session_raw = self.session_path.read_bytes()
            session = _parse_json_bytes(session_raw, "review session")
            required_session_fields = set(expected) | {
                "created_at_utc",
                "session_id",
                "csrf_token",
            }
            if set(session) != required_session_fields:
                raise ReviewError("existing review session fields do not match schema")
            for key, value in expected.items():
                if session.get(key) != value:
                    raise ReviewError(f"existing review session conflicts at {key}")
            if not isinstance(session.get("created_at_utc"), str) or UTC_TEXT.fullmatch(
                session["created_at_utc"]
            ) is None:
                raise ReviewError("existing review session has invalid created_at_utc")
            token = session.get("csrf_token")
            if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{64}", token) is None:
                raise ReviewError("existing review session has invalid CSRF token")
            session_id = session.get("session_id")
            if not isinstance(session_id, str) or UUID_TEXT.fullmatch(session_id) is None:
                raise ReviewError("existing review session has invalid session_id")
            self.csrf_token = token
            self.session_id = session_id
            self.session_sha256 = _sha256_bytes(session_raw)
            if not self.journal_path.exists():
                other_outputs = (
                    self.pending_path,
                    self.checkpoint_path,
                    self.snapshot_path,
                    self.decisions_path,
                    self.families_path,
                    self.receipt_path,
                    self.rights_evidence_dir,
                    self.recovery_dir,
                )
                if any(path.exists() for path in other_outputs):
                    raise ReviewError("session exists without journal but other outputs are present")
                _atomic_write(self.journal_path, b"")
                self._write_journal_checkpoint()
            elif not self.checkpoint_path.is_file():
                raise ReviewError("existing review session is missing its journal checkpoint")
            return
        non_journal_outputs = (
            self.pending_path,
            self.checkpoint_path,
            self.snapshot_path,
            self.decisions_path,
            self.families_path,
            self.receipt_path,
            self.rights_evidence_dir,
            self.recovery_dir,
        )
        if any(path.exists() for path in non_journal_outputs):
            raise ReviewError("review outputs exist without a bound session")
        if self.journal_path.exists() and self.journal_path.stat().st_size != 0:
            raise ReviewError("non-empty decision journal exists without a bound session")
        self.csrf_token = secrets.token_hex(32)
        self.session_id = str(uuid.uuid4())
        session = {
            **expected,
            "created_at_utc": _utc_now(),
            "session_id": self.session_id,
            "csrf_token": self.csrf_token,
        }
        session_bytes = _json_bytes(session)
        _atomic_write(self.session_path, session_bytes)
        self.session_sha256 = _sha256_bytes(session_bytes)
        if not self.journal_path.exists():
            _atomic_write(self.journal_path, b"")
        self._write_journal_checkpoint()

    def _checkpoint_body(self) -> dict[str, Any]:
        return {
            "schema_version": "rootscope.wikimedia_human_review_journal_checkpoint.v1",
            "updated_at_utc": _utc_now(),
            "session_id": self.session_id,
            "session_sha256": self.session_sha256,
            "queue_sha256": self.queue_sha256,
            "policy_sha256": self.policy_sha256,
            "implementation_sha256": self.tool_sha256,
            "ui_sha256": self.ui_sha256,
            "event_count": self.event_count,
            "journal_size": self.journal_size,
            "journal_sha256": self.journal_file_sha256,
            "last_event_sha256": self.last_event_sha256,
            "authority": AUTHORITY,
        }

    def _write_journal_checkpoint(self) -> None:
        body = self._checkpoint_body()
        checkpoint = {
            **body,
            "checkpoint_sha256": _sha256_bytes(_canonical_bytes(body)),
        }
        _atomic_write(self.checkpoint_path, _json_bytes(checkpoint))

    def _read_journal_checkpoint(self) -> dict[str, Any]:
        checkpoint = _read_json(self.checkpoint_path, "journal checkpoint")
        required = {
            "schema_version",
            "updated_at_utc",
            "session_id",
            "session_sha256",
            "queue_sha256",
            "policy_sha256",
            "implementation_sha256",
            "ui_sha256",
            "event_count",
            "journal_size",
            "journal_sha256",
            "last_event_sha256",
            "authority",
            "checkpoint_sha256",
        }
        if set(checkpoint) != required:
            raise ReviewError("journal checkpoint fields do not match schema")
        static_expected = {
            "schema_version": "rootscope.wikimedia_human_review_journal_checkpoint.v1",
            "session_id": self.session_id,
            "session_sha256": self.session_sha256,
            "queue_sha256": self.queue_sha256,
            "policy_sha256": self.policy_sha256,
            "implementation_sha256": self.tool_sha256,
            "ui_sha256": self.ui_sha256,
            "authority": AUTHORITY,
        }
        for key, value in static_expected.items():
            if checkpoint.get(key) != value:
                raise ReviewError(f"journal checkpoint conflicts at {key}")
        if not isinstance(checkpoint["updated_at_utc"], str) or UTC_TEXT.fullmatch(
            checkpoint["updated_at_utc"]
        ) is None:
            raise ReviewError("journal checkpoint timestamp is invalid")
        if type(checkpoint["event_count"]) is not int or checkpoint["event_count"] < 0:
            raise ReviewError("journal checkpoint event_count is invalid")
        if type(checkpoint["journal_size"]) is not int or checkpoint["journal_size"] < 0:
            raise ReviewError("journal checkpoint journal_size is invalid")
        for key in ("journal_sha256", "last_event_sha256", "checkpoint_sha256"):
            if not isinstance(checkpoint[key], str) or HEX64.fullmatch(checkpoint[key]) is None:
                raise ReviewError(f"journal checkpoint {key} is invalid")
        body = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
        if _sha256_bytes(_canonical_bytes(body)) != checkpoint["checkpoint_sha256"]:
            raise ReviewError("journal checkpoint SHA-256 mismatch")
        return checkpoint

    def _assert_checkpoint_matches_memory(self) -> None:
        checkpoint = self._read_journal_checkpoint()
        expected = {
            "event_count": self.event_count,
            "journal_size": self.journal_size,
            "journal_sha256": self.journal_file_sha256,
            "last_event_sha256": self.last_event_sha256,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ReviewError(f"journal checkpoint does not match current {key}")

    def _empty_decision(self) -> dict[str, Any]:
        return {
            "visual_decision": "UNREVIEWED",
            "rights_decision": "UNREVIEWED",
            "target_class": "",
            "unknown_scenario": "",
            "reviewed_source_group": "",
            "family_role": "UNASSIGNED",
            "near_duplicate_family": "",
            "visual_reviewer": "",
            "rights_reviewer": "",
            "rights_source_page_checked": False,
            "rights_evidence_sha256": "",
            "source_page_revision_id": "",
            "visual_reason_codes": [],
            "rights_reason_codes": [],
            "notes": "",
            "visual_reviewed_at_utc": "",
            "rights_reviewed_at_utc": "",
        }

    def _validate_client_decision(self, value: Any) -> tuple[dict[str, Any], str]:
        if not isinstance(value, dict) or set(value) != CLIENT_DECISION_FIELDS:
            raise ReviewError("decision fields do not match the frozen policy")
        visual = value["visual_decision"]
        rights = value["rights_decision"]
        target = value["target_class"]
        scenario = value["unknown_scenario"]
        group = value["reviewed_source_group"]
        role = value["family_role"]
        family = value["near_duplicate_family"]
        visual_reviewer = value["visual_reviewer"]
        rights_reviewer = value["rights_reviewer"]
        checked = value["rights_source_page_checked"]
        evidence_sha = value["rights_evidence_sha256"]
        revision_id = value["source_page_revision_id"]
        visual_reasons = value["visual_reason_codes"]
        rights_reasons = value["rights_reason_codes"]
        notes = value["notes"]

        if visual not in self.policy["visual_decisions"] or rights not in self.policy["rights_decisions"]:
            raise ReviewError("decision enum is invalid")
        if not isinstance(target, str) or target not in {"", *self.policy["target_classes"]}:
            raise ReviewError("target_class is invalid")
        if not isinstance(scenario, str) or scenario not in {"", *self.policy["unknown_scenarios"]}:
            raise ReviewError("unknown_scenario is invalid")
        if target != "unknown" and scenario:
            raise ReviewError("unknown_scenario is only allowed for target_class=unknown")
        if not isinstance(group, str) or (
            group and re.fullmatch(self.policy["reviewed_source_group_pattern"], group) is None
        ):
            raise ReviewError("reviewed_source_group is invalid")
        if role not in self.policy["family_roles"]:
            raise ReviewError("family_role is invalid")
        if not isinstance(family, str) or re.fullmatch(
            self.policy["near_duplicate_family_pattern"], family
        ) is None:
            raise ReviewError("near_duplicate_family is invalid")
        for axis, reviewer in (("visual", visual_reviewer), ("rights", rights_reviewer)):
            if not isinstance(reviewer, str):
                raise ReviewError(f"{axis}_reviewer is invalid")
            if reviewer and re.fullmatch(self.policy["reviewer_pattern"], reviewer) is None:
                raise ReviewError(f"{axis}_reviewer does not match policy")
        if type(checked) is not bool:
            raise ReviewError("rights_source_page_checked must be boolean")
        if not isinstance(evidence_sha, str) or (
            evidence_sha and HEX64.fullmatch(evidence_sha) is None
        ):
            raise ReviewError("rights_evidence_sha256 is invalid")
        if not isinstance(revision_id, str) or re.fullmatch(
            self.policy["source_page_revision_id_pattern"], revision_id
        ) is None:
            raise ReviewError("source_page_revision_id is invalid")
        for axis, reasons, allowed in (
            ("visual", visual_reasons, self.policy["visual_reason_codes"]),
            ("rights", rights_reasons, self.policy["rights_reason_codes"]),
        ):
            if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
                raise ReviewError(f"{axis}_reason_codes must be a list of strings")
            if len(reasons) != len(set(reasons)) or any(reason not in allowed for reason in reasons):
                raise ReviewError(f"{axis}_reason_codes are invalid")
        if not isinstance(notes, str) or len(notes) > self.policy["notes_max_length"]:
            raise ReviewError("notes are invalid or too long")

        if visual == "UNREVIEWED" and visual_reviewer:
            raise ReviewError("UNREVIEWED visual axis must not claim a reviewer")
        if visual != "UNREVIEWED" and not visual_reviewer:
            raise ReviewError("non-UNREVIEWED visual axis requires visual_reviewer")
        if rights == "UNREVIEWED" and rights_reviewer:
            raise ReviewError("UNREVIEWED rights axis must not claim a reviewer")
        if rights != "UNREVIEWED" and not rights_reviewer:
            raise ReviewError("non-UNREVIEWED rights axis requires rights_reviewer")
        if visual == "PASS" and not target:
            raise ReviewError("visual PASS requires target_class")
        if visual != "PASS" and (target or scenario):
            raise ReviewError("target_class/unknown_scenario require visual PASS")
        if visual == "REJECT" and not visual_reasons:
            raise ReviewError("visual REJECT requires visual_reason_codes")
        if visual != "REJECT" and visual_reasons:
            raise ReviewError("visual_reason_codes require visual REJECT")
        if rights == "REJECT" and not rights_reasons:
            raise ReviewError("rights REJECT requires rights_reason_codes")
        if rights != "REJECT" and rights_reasons:
            raise ReviewError("rights_reason_codes require rights REJECT")
        if rights in {"PASS", "REJECT"} and (
            not checked or not evidence_sha or not revision_id
        ):
            raise ReviewError(
                "rights PASS/REJECT requires a checked permanent source-page revision and evidence SHA-256"
            )
        if rights == "UNREVIEWED" and (checked or evidence_sha or revision_id):
            raise ReviewError("UNREVIEWED rights axis must not claim page evidence")

        pass_pair = visual == rights == "PASS"
        if pass_pair:
            if not group:
                raise ReviewError("PASS/PASS requires reviewed_source_group")
            if role not in {
                "CANONICAL_REPRESENTATIVE",
                "SERIES_SIBLING_EXCLUDED",
                "HOLD",
            }:
                raise ReviewError("PASS/PASS requires a reviewed family role")
        elif role in {"CANONICAL_REPRESENTATIVE", "SERIES_SIBLING_EXCLUDED"}:
            raise ReviewError("canonical/sibling family roles require PASS/PASS")
        if role != "UNASSIGNED" and not group:
            raise ReviewError("assigned family_role requires reviewed_source_group")
        if role == "SERIES_SIBLING_EXCLUDED" and not family:
            raise ReviewError("series sibling requires near_duplicate_family")

        untouched = visual == rights == "UNREVIEWED"
        if untouched:
            if target or scenario or group or role != "UNASSIGNED" or family or notes:
                raise ReviewError("fully UNREVIEWED decision must be empty")
            status = "UNREVIEWED"
        elif visual == "REJECT":
            status = "REJECTED_VISUAL_PENDING_EXPORT_NOT_DATA_LOCKED"
        elif rights == "REJECT":
            status = "REJECTED_RIGHTS_PENDING_EXPORT_NOT_DATA_LOCKED"
        elif pass_pair and role == "CANONICAL_REPRESENTATIVE":
            status = "CANONICAL_READY_FOR_FAMILY_AUDIT_NOT_DATA_LOCKED"
        elif pass_pair and role == "SERIES_SIBLING_EXCLUDED":
            status = "SERIES_SIBLING_EXCLUDED_NOT_DATA_LOCKED"
        else:
            status = "NEEDS_REVIEW"

        normalized = {
            "visual_decision": visual,
            "rights_decision": rights,
            "target_class": target,
            "unknown_scenario": scenario,
            "reviewed_source_group": group,
            "family_role": role,
            "near_duplicate_family": family,
            "visual_reviewer": visual_reviewer,
            "rights_reviewer": rights_reviewer,
            "rights_source_page_checked": checked,
            "rights_evidence_sha256": evidence_sha,
            "source_page_revision_id": revision_id,
            "visual_reason_codes": sorted(visual_reasons),
            "rights_reason_codes": sorted(rights_reasons),
            "notes": notes,
        }
        return normalized, status

    def _recordable_decision(
        self,
        value: Any,
        previous: Mapping[str, Any] | None,
        event_at_utc: str,
    ) -> tuple[dict[str, Any], str]:
        normalized, status = self._validate_client_decision(value)
        old = self._empty_decision() if previous is None else dict(previous)
        visual_fields = (
            "visual_decision",
            "target_class",
            "unknown_scenario",
            "reviewed_source_group",
            "family_role",
            "near_duplicate_family",
            "visual_reviewer",
            "visual_reason_codes",
        )
        rights_fields = (
            "rights_decision",
            "rights_reviewer",
            "rights_source_page_checked",
            "rights_evidence_sha256",
            "source_page_revision_id",
            "rights_reason_codes",
        )
        visual_same = all(old.get(key) == normalized[key] for key in visual_fields)
        rights_same = all(old.get(key) == normalized[key] for key in rights_fields)
        visual_time = "" if normalized["visual_decision"] == "UNREVIEWED" else (
            str(old.get("visual_reviewed_at_utc", "")) if visual_same else event_at_utc
        )
        rights_time = "" if normalized["rights_decision"] == "UNREVIEWED" else (
            str(old.get("rights_reviewed_at_utc", "")) if rights_same else event_at_utc
        )
        if normalized["visual_decision"] != "UNREVIEWED" and not visual_time:
            visual_time = event_at_utc
        if normalized["rights_decision"] != "UNREVIEWED" and not rights_time:
            rights_time = event_at_utc
        return {
            **normalized,
            "visual_reviewed_at_utc": visual_time,
            "rights_reviewed_at_utc": rights_time,
        }, status

    def _validate_persisted_decision(self, value: Any) -> tuple[dict[str, Any], str]:
        if not isinstance(value, dict) or set(value) != PERSISTED_DECISION_FIELDS:
            raise ReviewError("persisted decision fields do not match schema")
        client_value = {key: value[key] for key in CLIENT_DECISION_FIELDS}
        normalized, status = self._validate_client_decision(client_value)
        if any(value[key] != normalized[key] for key in CLIENT_DECISION_FIELDS):
            raise ReviewError("persisted decision is not normalized")
        for axis in ("visual", "rights"):
            decision = normalized[f"{axis}_decision"]
            reviewed_at = value[f"{axis}_reviewed_at_utc"]
            if not isinstance(reviewed_at, str):
                raise ReviewError(f"{axis}_reviewed_at_utc is invalid")
            if decision == "UNREVIEWED" and reviewed_at:
                raise ReviewError(f"UNREVIEWED {axis} axis must not have a timestamp")
            if decision != "UNREVIEWED" and UTC_TEXT.fullmatch(reviewed_at) is None:
                raise ReviewError(f"reviewed {axis} axis requires a UTC timestamp")
        return dict(value), status

    def _validate_event(self, event: dict[str, Any], *, expected_previous: str) -> tuple[str, str]:
        required = {
            "schema_version",
            "event_id",
            "event_at_utc",
            "session_id",
            "session_sha256",
            "queue_sha256",
            "policy_sha256",
            "implementation_sha256",
            "ui_sha256",
            "asset",
            "pageid",
            "candidate_sha256",
            "revision",
            "previous_event_sha256",
            "decision",
            "review_status",
            "authority",
            "event_sha256",
        }
        if set(event) != required:
            raise ReviewError("journal event fields do not match schema")
        if event["schema_version"] != self.policy["decision_schema_version"]:
            raise ReviewError("journal event schema mismatch")
        if not isinstance(event["event_id"], str) or UUID_TEXT.fullmatch(event["event_id"]) is None:
            raise ReviewError("journal event_id is invalid")
        if event["event_id"] in self.event_ids:
            raise ReviewError("journal event_id is duplicated")
        if not isinstance(event["event_at_utc"], str) or UTC_TEXT.fullmatch(event["event_at_utc"]) is None:
            raise ReviewError("journal event_at_utc is invalid")
        if event["session_id"] != self.session_id:
            raise ReviewError("journal event session binding mismatch")
        if event["session_sha256"] != self.session_sha256:
            raise ReviewError("journal event session root mismatch")
        if event["queue_sha256"] != self.queue_sha256 or event["policy_sha256"] != self.policy_sha256:
            raise ReviewError("journal event input root mismatch")
        if (
            event["implementation_sha256"] != self.tool_sha256
            or event["ui_sha256"] != self.ui_sha256
        ):
            raise ReviewError("journal event implementation/UI binding mismatch")
        if event["previous_event_sha256"] != expected_previous:
            raise ReviewError("journal hash chain is broken")
        asset = event["asset"]
        row = self.items_by_asset.get(asset)
        if row is None or event["candidate_sha256"] != row["sha256"] or event["pageid"] != row["pageid"]:
            raise ReviewError("journal event candidate binding mismatch")
        image_path = self._bound_image_path(row)
        if _sha256_file(image_path) != row["sha256"]:
            raise ReviewError("journal event candidate payload changed")
        if type(event["revision"]) is not int or event["revision"] < 1:
            raise ReviewError("journal revision is invalid")
        current = self.latest_by_asset.get(asset)
        expected_revision = 1 if current is None else current["revision"] + 1
        if event["revision"] != expected_revision:
            raise ReviewError("journal per-asset revision is not monotonic")
        decision, status = self._validate_persisted_decision(event["decision"])
        if decision != event["decision"] or status != event["review_status"]:
            raise ReviewError("journal decision/status mismatch")
        self._validate_rights_evidence_binding(row, decision)
        if event["authority"] != AUTHORITY:
            raise ReviewError("journal event claims forbidden authority")
        claimed = event["event_sha256"]
        if not isinstance(claimed, str) or HEX64.fullmatch(claimed) is None:
            raise ReviewError("journal event SHA-256 is invalid")
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        actual = _sha256_bytes(_canonical_bytes(body))
        if actual != claimed:
            raise ReviewError("journal event SHA-256 mismatch")
        return claimed, asset

    def _apply_event(self, event: dict[str, Any]) -> None:
        claimed, asset = self._validate_event(event, expected_previous=self.last_event_sha256)
        self.latest_by_asset[asset] = event
        self.last_event_sha256 = claimed
        self.last_event = event
        self.event_count += 1
        self.event_ids.add(event["event_id"])
        self.events.append(event)

    def _load_journal(self) -> None:
        raw = self.journal_path.read_bytes()
        checkpoint = self._read_journal_checkpoint()
        pending = (
            _read_json(self.pending_path, "pending review event")
            if self.pending_path.is_file()
            else None
        )

        def preserve_recovery(
            original: bytes, retained: bytes, trailing: bytes, action: str
        ) -> None:
            recovery_sha = _sha256_bytes(original)
            self.recovery_dir.mkdir(parents=True, exist_ok=True)
            partial_path = self.recovery_dir / f"partial_journal_{recovery_sha}.bin"
            if partial_path.exists() and partial_path.read_bytes() != original:
                raise ReviewError("journal recovery evidence collision")
            if not partial_path.exists():
                _atomic_write(partial_path, original)
            report = {
                "schema_version": "rootscope.wikimedia_journal_recovery.v1",
                "recovered_at_utc": _utc_now(),
                "session_id": self.session_id,
                "session_sha256": self.session_sha256,
                "original_journal_sha256": recovery_sha,
                "original_journal_bytes": len(original),
                "retained_journal_sha256": _sha256_bytes(retained),
                "retained_journal_bytes": len(retained),
                "discarded_partial_bytes": len(trailing),
                "pending_event_sha256": None if pending is None else pending.get("event_sha256"),
                "action": action,
                "authority": AUTHORITY,
            }
            _atomic_write(
                self.recovery_dir / f"journal_recovery_{recovery_sha}.json",
                _json_bytes(report),
            )

        checkpoint_size = checkpoint["journal_size"]
        checkpoint_sha = checkpoint["journal_sha256"]
        if (
            pending is not None
            and raw
            and not raw.endswith(b"\n")
            and len(raw) + 1 == checkpoint_size
            and _sha256_bytes(raw + b"\n") == checkpoint_sha
        ):
            preserve_recovery(
                raw,
                raw + b"\n",
                b"",
                "PRESERVED_FULL_EVENT_THEN_RESTORED_MISSING_TERMINAL_NEWLINE",
            )
            raw += b"\n"
            _atomic_write(self.journal_path, raw)

        if len(raw) < checkpoint_size:
            raise ReviewError("decision journal is shorter than its monotonic checkpoint")
        retained = raw[:checkpoint_size]
        trailing = raw[checkpoint_size:]
        if _sha256_bytes(retained) != checkpoint_sha:
            raise ReviewError("decision journal checkpointed prefix changed")
        if retained and not retained.endswith(b"\n"):
            raise ReviewError("checkpointed decision journal lacks a terminal newline")

        canonical_pending_line = None
        if trailing:
            if pending is None:
                raise ReviewError("decision journal advanced without a pending WAL event")
            canonical_pending_line = _canonical_bytes(pending) + b"\n"
            if not canonical_pending_line.startswith(trailing):
                raise ReviewError("journal bytes after checkpoint do not match pending event")
            if trailing != canonical_pending_line:
                preserve_recovery(
                    raw,
                    retained,
                    trailing,
                    "PRESERVED_PARTIAL_BYTES_THEN_REPLAY_PENDING_EVENT",
                )
                _atomic_write(self.journal_path, retained)
                raw = retained
                trailing = b""

        events = _parse_jsonl_bytes(retained, "checkpointed decision journal") if retained else []
        for event in events:
            self._apply_event(event)
        self.journal_size = len(retained)
        self.journal_file_sha256 = _sha256_bytes(retained)
        checkpoint_expected = {
            "event_count": self.event_count,
            "last_event_sha256": self.last_event_sha256,
        }
        for key, value in checkpoint_expected.items():
            if checkpoint.get(key) != value:
                raise ReviewError(f"journal replay does not match checkpoint {key}")

        if trailing:
            if canonical_pending_line is None or trailing != canonical_pending_line:
                raise ReviewError("pending journal tail is not a complete canonical event")
            assert pending is not None
            self._apply_event(pending)
            self._refresh_journal_binding()

    def _assert_journal_unchanged(self) -> None:
        try:
            raw = self.journal_path.read_bytes()
        except OSError as exc:
            raise ReviewError(f"cannot re-read decision journal: {exc}") from exc
        if len(raw) != self.journal_size or _sha256_bytes(raw) != self.journal_file_sha256:
            raise ReviewError("decision journal changed outside the locked review process")

    def _refresh_journal_binding(self) -> None:
        raw = self.journal_path.read_bytes()
        self.journal_size = len(raw)
        self.journal_file_sha256 = _sha256_bytes(raw)

    def _recover_pending_event(self) -> None:
        self._assert_journal_unchanged()
        if not self.pending_path.is_file():
            self._assert_checkpoint_matches_memory()
            return
        event = _read_json(self.pending_path, "pending review event")
        claimed = event.get("event_sha256")
        if claimed == self.last_event_sha256:
            if self.last_event is None or event != self.last_event:
                raise ReviewError("pending event hash collides with a different journal tail")
            checkpoint = self._read_journal_checkpoint()
            if (
                checkpoint["event_count"] != self.event_count
                or checkpoint["journal_size"] != self.journal_size
                or checkpoint["journal_sha256"] != self.journal_file_sha256
                or checkpoint["last_event_sha256"] != self.last_event_sha256
            ):
                self._write_journal_checkpoint()
            self._assert_checkpoint_matches_memory()
            return
        self._validate_event(event, expected_previous=self.last_event_sha256)
        _append_jsonl(self.journal_path, event)
        self._apply_event(event)
        self._refresh_journal_binding()
        self._write_journal_checkpoint()
        self._assert_checkpoint_matches_memory()

    def create_rights_evidence(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.degraded:
                raise ReviewError("review store is degraded; restart for fail-closed recovery")
            self._verify_runtime_inputs()
            self._assert_journal_unchanged()
            self._assert_checkpoint_matches_memory()
            required = {
                "asset",
                "candidate_sha256",
                "reviewer",
                "source_page_revision_id",
                "confirmed_source_page_checked",
                "confirmed_creator_license_attribution",
                "confirmed_non_copyright_rights_reviewed",
            }
            if set(payload) != required:
                raise ReviewError("rights-evidence request fields do not match schema")
            asset = payload["asset"]
            row = self.items_by_asset.get(asset)
            if row is None or payload["candidate_sha256"] != row["sha256"]:
                raise ReviewError("rights evidence candidate binding mismatch")
            reviewer = payload["reviewer"]
            if not isinstance(reviewer, str) or re.fullmatch(
                self.policy["reviewer_pattern"], reviewer
            ) is None:
                raise ReviewError("rights evidence reviewer is invalid")
            revision_id = payload["source_page_revision_id"]
            if not isinstance(revision_id, str) or re.fullmatch(
                self.policy["source_page_revision_id_pattern"], revision_id
            ) is None or not revision_id:
                raise ReviewError("rights evidence source-page revision is invalid")
            confirmations = (
                "confirmed_source_page_checked",
                "confirmed_creator_license_attribution",
                "confirmed_non_copyright_rights_reviewed",
            )
            if any(payload[key] is not True for key in confirmations):
                raise ReviewError("all three rights-review attestations must be true")
            body = {
                "schema_version": "rootscope.wikimedia_rights_review_evidence.v1",
                "evidence_scope": "HUMAN_ATTESTATION_BOUND_TO_PERMALINK_NO_PAGE_SNAPSHOT",
                "reviewed_at_utc": _utc_now(),
                "session_id": self.session_id,
                "session_sha256": self.session_sha256,
                "queue_sha256": self.queue_sha256,
                "policy_sha256": self.policy_sha256,
                "implementation_sha256": self.tool_sha256,
                "ui_sha256": self.ui_sha256,
                "asset": asset,
                "pageid": row["pageid"],
                "candidate_sha256": row["sha256"],
                "source_url": row["source_url"],
                "source_page_permanent_url": _source_permanent_url(
                    row["source_url"], revision_id
                ),
                "creator": row["creator"],
                "license": row["license"],
                "license_url": row["license_url"],
                "reviewer": reviewer,
                "source_page_revision_id": revision_id,
                **{key: True for key in confirmations},
                "authority": AUTHORITY,
            }
            evidence_bytes = _canonical_bytes(body)
            evidence_sha256 = _sha256_bytes(evidence_bytes)
            self.rights_evidence_dir.mkdir(parents=True, exist_ok=True)
            evidence_path = self.rights_evidence_dir / f"{evidence_sha256}.json"
            if evidence_path.exists():
                if evidence_path.read_bytes() != evidence_bytes:
                    raise ReviewError("content-addressed rights evidence collision")
            else:
                _atomic_write(evidence_path, evidence_bytes)
            return {
                "rights_evidence_sha256": evidence_sha256,
                "evidence_path": evidence_path.relative_to(self.output_dir).as_posix(),
                "reviewed_at_utc": body["reviewed_at_utc"],
                "data_locked": False,
            }

    def _validate_rights_evidence_binding(
        self, row: Mapping[str, Any], decision: Mapping[str, Any]
    ) -> None:
        evidence_sha = decision["rights_evidence_sha256"]
        if not evidence_sha:
            return
        evidence_path = self.rights_evidence_dir / f"{evidence_sha}.json"
        if not evidence_path.is_file():
            raise ReviewError("rights evidence file is missing or changed")
        try:
            evidence_raw = evidence_path.read_bytes()
        except OSError as exc:
            raise ReviewError(f"cannot read rights evidence: {exc}") from exc
        if _sha256_bytes(evidence_raw) != evidence_sha:
            raise ReviewError("rights evidence file is missing or changed")
        evidence = _parse_json_bytes(evidence_raw, "rights review evidence")
        required = {
            "schema_version",
            "evidence_scope",
            "reviewed_at_utc",
            "session_id",
            "session_sha256",
            "queue_sha256",
            "policy_sha256",
            "implementation_sha256",
            "ui_sha256",
            "asset",
            "pageid",
            "candidate_sha256",
            "source_url",
            "source_page_permanent_url",
            "creator",
            "license",
            "license_url",
            "reviewer",
            "source_page_revision_id",
            "confirmed_source_page_checked",
            "confirmed_creator_license_attribution",
            "confirmed_non_copyright_rights_reviewed",
            "authority",
        }
        if set(evidence) != required:
            raise ReviewError("rights evidence fields do not match schema")
        expected = {
            "schema_version": "rootscope.wikimedia_rights_review_evidence.v1",
            "evidence_scope": "HUMAN_ATTESTATION_BOUND_TO_PERMALINK_NO_PAGE_SNAPSHOT",
            "session_id": self.session_id,
            "session_sha256": self.session_sha256,
            "queue_sha256": self.queue_sha256,
            "policy_sha256": self.policy_sha256,
            "implementation_sha256": self.tool_sha256,
            "ui_sha256": self.ui_sha256,
            "asset": row["asset"],
            "pageid": row["pageid"],
            "candidate_sha256": row["sha256"],
            "source_url": row["source_url"],
            "source_page_permanent_url": _source_permanent_url(
                row["source_url"], decision["source_page_revision_id"]
            ),
            "creator": row["creator"],
            "license": row["license"],
            "license_url": row["license_url"],
            "reviewer": decision["rights_reviewer"],
            "source_page_revision_id": decision["source_page_revision_id"],
            "confirmed_source_page_checked": True,
            "confirmed_creator_license_attribution": True,
            "confirmed_non_copyright_rights_reviewed": True,
            "authority": AUTHORITY,
        }
        for key, value in expected.items():
            if evidence.get(key) != value:
                raise ReviewError(f"rights evidence conflicts at {key}")
        if not isinstance(evidence["reviewed_at_utc"], str) or UTC_TEXT.fullmatch(
            evidence["reviewed_at_utc"]
        ) is None:
            raise ReviewError("rights evidence timestamp is invalid")

    def _write_snapshot(self) -> None:
        self._verify_runtime_inputs()
        self._assert_journal_unchanged()
        self._assert_checkpoint_matches_memory()
        checkpoint_file_sha256 = _sha256_file(self.checkpoint_path)
        for historical_event in self.events:
            historical_row = self.items_by_asset[historical_event["asset"]]
            self._validate_rights_evidence_binding(
                historical_row, historical_event["decision"]
            )
        records: list[dict[str, Any]] = []
        status_counts: Counter[str] = Counter()
        visual_axis_counts: Counter[str] = Counter()
        rights_axis_counts: Counter[str] = Counter()
        target_class_counts: Counter[str] = Counter()
        family_members: dict[str, list[dict[str, Any]]] = {}
        for row in self.items:
            event = self.latest_by_asset.get(row["asset"])
            status = "UNREVIEWED" if event is None else event["review_status"]
            status_counts[status] += 1
            if event is None:
                visual_axis_counts["UNREVIEWED"] += 1
                rights_axis_counts["UNREVIEWED"] += 1
                continue
            decision = event["decision"]
            visual_axis_counts[decision["visual_decision"]] += 1
            rights_axis_counts[decision["rights_decision"]] += 1
            if decision["visual_decision"] == "PASS":
                target_class_counts[decision["target_class"]] += 1
            self._validate_rights_evidence_binding(row, decision)
            record = {
                "schema_version": "rootscope.wikimedia_human_review_export.v1",
                "mode": "PRODUCTION" if self.production_mode else "FIXTURE",
                "session_id": self.session_id,
                "session_sha256": self.session_sha256,
                "implementation_sha256": self.tool_sha256,
                "ui_sha256": self.ui_sha256,
                "asset": row["asset"],
                "pageid": row["pageid"],
                "candidate_sha256": row["sha256"],
                "origin_source_group": row["source_group"],
                "source_url": row["source_url"],
                "revision": event["revision"],
                "review_status": status,
                "decision": event["decision"],
                "event_sha256": event["event_sha256"],
                "authority": AUTHORITY,
            }
            records.append(record)
            group = decision["reviewed_source_group"]
            if (
                group
                and decision["visual_decision"] == "PASS"
                and decision["rights_decision"] == "PASS"
                and decision["family_role"] != "UNASSIGNED"
            ):
                family_members.setdefault(group, []).append(
                    {
                        "asset": row["asset"],
                        "pageid": row["pageid"],
                        "origin_source_group": row["source_group"],
                        "target_class": decision["target_class"],
                        "family_role": decision["family_role"],
                        "near_duplicate_family": decision["near_duplicate_family"],
                        "review_status": status,
                    }
                )

        family_records: list[dict[str, Any]] = []
        family_issue_count = 0
        near_family_index: dict[str, list[dict[str, Any]]] = {}
        for group in sorted(family_members):
            members = sorted(family_members[group], key=lambda item: item["asset"])
            canonicals = [
                item["asset"]
                for item in members
                if item["family_role"] == "CANONICAL_REPRESENTATIVE"
            ]
            siblings = [
                item["asset"]
                for item in members
                if item["family_role"] == "SERIES_SIBLING_EXCLUDED"
            ]
            holds = [
                item["asset"] for item in members if item["family_role"] == "HOLD"
            ]
            pass_classes = sorted(
                {
                    item["target_class"]
                    for item in members
                    if item["family_role"]
                    in {"CANONICAL_REPRESENTATIVE", "SERIES_SIBLING_EXCLUDED"}
                }
            )
            issues: list[str] = []
            if len(canonicals) > 1:
                issues.append("MULTIPLE_CANONICAL_REPRESENTATIVES")
            if siblings and not canonicals:
                issues.append("SIBLINGS_WITHOUT_CANONICAL_REPRESENTATIVE")
            if len(pass_classes) > 1:
                issues.append("TARGET_CLASS_CONFLICT_WITHIN_INDEPENDENCE_FAMILY")
            for member in members:
                near_family = member["near_duplicate_family"]
                if near_family:
                    near_family_index.setdefault(near_family, []).append(
                        {**member, "reviewed_source_group": group}
                    )
            sibling_near_families = {
                item["near_duplicate_family"]
                for item in members
                if item["family_role"] == "SERIES_SIBLING_EXCLUDED"
            }
            for near_family in sorted(sibling_near_families):
                matching_canonicals = [
                    item
                    for item in members
                    if item["family_role"] == "CANONICAL_REPRESENTATIVE"
                    and item["near_duplicate_family"] == near_family
                ]
                if len(matching_canonicals) != 1:
                    issues.append(
                        f"NEAR_DUPLICATE_FAMILY_REQUIRES_ONE_MATCHING_CANONICAL:{near_family}"
                    )
            family_issue_count += len(issues)
            family_records.append(
                {
                    "schema_version": "rootscope.wikimedia_reviewed_source_family.v1",
                    "mode": "PRODUCTION" if self.production_mode else "FIXTURE",
                    "session_id": self.session_id,
                    "session_sha256": self.session_sha256,
                    "implementation_sha256": self.tool_sha256,
                    "ui_sha256": self.ui_sha256,
                    "reviewed_source_group": group,
                    "member_count": len(members),
                    "canonical_assets": canonicals,
                    "series_sibling_assets": siblings,
                    "hold_assets": holds,
                    "target_classes": pass_classes,
                    "issues": issues,
                    "members": members,
                    "authority": AUTHORITY,
                }
            )

        global_near_issues: list[str] = []
        for near_family, members in sorted(near_family_index.items()):
            groups = {item["reviewed_source_group"] for item in members}
            classes = {item["target_class"] for item in members}
            if len(groups) > 1:
                global_near_issues.append(
                    f"NEAR_DUPLICATE_FAMILY_SPANS_REVIEWED_GROUPS:{near_family}"
                )
            if len(classes) > 1:
                global_near_issues.append(
                    f"NEAR_DUPLICATE_FAMILY_SPANS_TARGET_CLASSES:{near_family}"
                )
        family_issue_count += len(global_near_issues)

        final_statuses = {
            "REJECTED_VISUAL_PENDING_EXPORT_NOT_DATA_LOCKED",
            "REJECTED_RIGHTS_PENDING_EXPORT_NOT_DATA_LOCKED",
            "CANONICAL_READY_FOR_FAMILY_AUDIT_NOT_DATA_LOCKED",
            "SERIES_SIBLING_EXCLUDED_NOT_DATA_LOCKED",
        }
        finalized_count = sum(status_counts[status] for status in final_statuses)
        reviewed_count = len(self.items) - status_counts["UNREVIEWED"]
        needs_review_count = status_counts["NEEDS_REVIEW"]
        complete_candidate = finalized_count == len(self.items) and family_issue_count == 0
        full_candidate_payload_revalidation = False
        if complete_candidate:
            for row in self.items:
                if _sha256_file(self._bound_image_path(row)) != row["sha256"]:
                    raise ReviewError("full completion validation found a changed candidate payload")
            self._verify_runtime_inputs()
            self._assert_journal_unchanged()
            self._assert_checkpoint_matches_memory()
            full_candidate_payload_revalidation = True
        complete = complete_candidate and full_candidate_payload_revalidation
        status_base = (
            "HUMAN_REVIEW_COMPLETE_NOT_DATA_LOCKED"
            if complete
            else "HUMAN_REVIEW_IN_PROGRESS_NOT_DATA_LOCKED"
        )
        status = status_base if self.production_mode else f"FIXTURE_{status_base}"
        self.current_status = status
        decisions_payload = b"".join(_canonical_bytes(record) + b"\n" for record in records)
        families_payload = b"".join(
            _canonical_bytes(record) + b"\n" for record in family_records
        )
        _atomic_write(self.decisions_path, decisions_payload)
        _atomic_write(self.families_path, families_payload)
        payload = {
            "schema_version": self.policy["snapshot_schema_version"],
            "mode": "PRODUCTION" if self.production_mode else "FIXTURE",
            "status": status,
            "session_id": self.session_id,
            "session_sha256": self.session_sha256,
            "queue_sha256": self.queue_sha256,
            "policy_sha256": self.policy_sha256,
            "implementation_sha256": self.tool_sha256,
            "ui_sha256": self.ui_sha256,
            "candidate_count": len(self.items),
            "evented_asset_count": len(records),
            "reviewed_asset_count": reviewed_count,
            "finalized_asset_count": finalized_count,
            "needs_review_asset_count": needs_review_count,
            "event_count": self.event_count,
            "last_event_sha256": self.last_event_sha256,
            "journal_checkpoint_sha256": checkpoint_file_sha256,
            "review_status_counts": dict(sorted(status_counts.items())),
            "visual_axis_counts": dict(sorted(visual_axis_counts.items())),
            "rights_axis_counts": dict(sorted(rights_axis_counts.items())),
            "visual_pass_target_class_counts": dict(sorted(target_class_counts.items())),
            "reviewed_source_family_count": len(family_records),
            "family_issue_count": family_issue_count,
            "global_near_duplicate_issues": global_near_issues,
            "validation_scope": {
                "runtime_inputs_revalidated": True,
                "journal_revalidated": True,
                "all_referenced_rights_attestations_revalidated": True,
                "all_candidate_payloads_revalidated": full_candidate_payload_revalidation,
            },
            "records": records,
            "authority": AUTHORITY,
            "explicit_non_claims": EXPLICIT_NON_CLAIMS,
        }
        snapshot_bytes = _json_bytes(payload)
        _atomic_write(self.snapshot_path, snapshot_bytes)
        rights_evidence_references = sorted(
            {
                record["decision"]["rights_evidence_sha256"]
                for record in records
                if record["decision"]["rights_evidence_sha256"]
            }
        )
        rights_reference_payload = (
            "".join(f"{value}\n" for value in rights_evidence_references).encode("ascii")
        )
        receipt = {
            "schema_version": "rootscope.wikimedia_human_review_receipt.v1",
            "mode": "PRODUCTION" if self.production_mode else "FIXTURE",
            "status": status,
            "session_id": self.session_id,
            "session_sha256": self.session_sha256,
            "complete": complete,
            "production_binding_enforced": self.production_mode,
            "verified_production_input_roots": self.verified_production_input_roots,
            "queue_sha256": self.queue_sha256,
            "policy_sha256": self.policy_sha256,
            "implementation_sha256": self.tool_sha256,
            "ui_sha256": self.ui_sha256,
            "journal_sha256": self.journal_file_sha256,
            "journal_checkpoint_sha256": checkpoint_file_sha256,
            "last_event_sha256": self.last_event_sha256,
            "human_review_decisions_sha256": _sha256_bytes(decisions_payload),
            "reviewed_source_families_sha256": _sha256_bytes(families_payload),
            "rights_evidence_reference_list_sha256": _sha256_bytes(rights_reference_payload),
            "snapshot_sha256": _sha256_bytes(snapshot_bytes),
            "rights_attestation_scope": "PERMALINK_BOUND_HUMAN_ATTESTATION_NO_PAGE_SNAPSHOT",
            "rights_source_page_payload_archived": False,
            "counts": {
                "candidate": len(self.items),
                "event": self.event_count,
                "evented_asset": len(records),
                "reviewed_asset": reviewed_count,
                "finalized_asset": finalized_count,
                "needs_review_asset": needs_review_count,
                "reviewed_source_family": len(family_records),
                "family_issue": family_issue_count,
                "referenced_rights_evidence": len(rights_evidence_references),
            },
            "review_status_counts": dict(sorted(status_counts.items())),
            "visual_axis_counts": dict(sorted(visual_axis_counts.items())),
            "rights_axis_counts": dict(sorted(rights_axis_counts.items())),
            "visual_pass_target_class_counts": dict(sorted(target_class_counts.items())),
            "global_near_duplicate_issues": global_near_issues,
            "validation_scope": payload["validation_scope"],
            "authority": AUTHORITY,
            "explicit_non_claims": EXPLICIT_NON_CLAIMS,
        }
        _atomic_write(self.receipt_path, _json_bytes(receipt))

    def record_decision(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.degraded:
                raise ReviewError("review store is degraded; restart for fail-closed recovery")
            self._verify_runtime_inputs()
            self._assert_journal_unchanged()
            self._assert_checkpoint_matches_memory()
            if self.pending_path.exists():
                raise ReviewError("a pending event appeared after startup; restart for fail-closed recovery")
            required = {"asset", "candidate_sha256", "expected_revision", "decision"}
            if set(payload) != required:
                raise ReviewError("decision request fields do not match schema")
            asset = payload["asset"]
            row = self.items_by_asset.get(asset)
            if row is None:
                raise ReviewError("decision references an unknown asset")
            if payload["candidate_sha256"] != row["sha256"]:
                raise ReviewError("decision candidate SHA-256 mismatch")
            current = self.latest_by_asset.get(asset)
            current_revision = 0 if current is None else current["revision"]
            if type(payload["expected_revision"]) is not int or payload["expected_revision"] != current_revision:
                raise ReviewError(f"stale decision revision; current={current_revision}")
            image_path = self._bound_image_path(row)
            if _sha256_file(image_path) != row["sha256"]:
                raise ReviewError("candidate payload changed before decision")
            event_at_utc = _utc_now()
            previous_decision = None if current is None else current["decision"]
            decision, status = self._recordable_decision(
                payload["decision"], previous_decision, event_at_utc
            )
            baseline_decision = (
                self._empty_decision() if previous_decision is None else previous_decision
            )
            if decision == baseline_decision:
                raise ReviewError("decision is unchanged; no-op events are forbidden")
            body = {
                "schema_version": self.policy["decision_schema_version"],
                "event_id": str(uuid.uuid4()),
                "event_at_utc": event_at_utc,
                "session_id": self.session_id,
                "session_sha256": self.session_sha256,
                "queue_sha256": self.queue_sha256,
                "policy_sha256": self.policy_sha256,
                "implementation_sha256": self.tool_sha256,
                "ui_sha256": self.ui_sha256,
                "asset": asset,
                "pageid": row["pageid"],
                "candidate_sha256": row["sha256"],
                "revision": current_revision + 1,
                "previous_event_sha256": self.last_event_sha256,
                "decision": decision,
                "review_status": status,
                "authority": AUTHORITY,
            }
            event = {**body, "event_sha256": _sha256_bytes(_canonical_bytes(body))}
            self._validate_event(event, expected_previous=self.last_event_sha256)
            _atomic_write(self.pending_path, _json_bytes(event))
            try:
                _append_jsonl(self.journal_path, event)
                self._apply_event(event)
                self._refresh_journal_binding()
                self._write_journal_checkpoint()
                self._write_snapshot()
                self.pending_path.unlink()
            except Exception:
                self.degraded = True
                raise
            return self.public_item(asset)

    def public_item(self, asset: str) -> dict[str, Any]:
        row = self.items_by_asset[asset]
        event = self.latest_by_asset.get(asset)
        decision = (
            event["decision"]
            if event is not None
            else self._empty_decision()
        )
        return {
            **row,
            "review_status": "UNREVIEWED" if event is None else event["review_status"],
            "review_revision": 0 if event is None else event["revision"],
            "decision": decision,
            "image_url": f"/image?asset={quote(asset, safe='')}",
        }

    def list_items(
        self,
        *,
        class_hint: str = "",
        species_hint: str = "",
        review_status: str = "",
        query: str = "",
        offset: int = 0,
        limit: int = 24,
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 100:
            raise ReviewError("invalid pagination")
        query_folded = query.casefold().strip()
        selected = []
        for row in self.items:
            item = self.public_item(row["asset"])
            if class_hint and item["class_hint"] != class_hint:
                continue
            if species_hint and item.get("species_hint") != species_hint:
                continue
            if review_status and item["review_status"] != review_status:
                continue
            if query_folded:
                haystack = " ".join(
                    str(item.get(key, ""))
                    for key in ("pageid", "title", "creator", "species_hint", "acquisition_query", "source_group")
                ).casefold()
                if query_folded not in haystack:
                    continue
            selected.append(item)
        return {
            "status": self.current_status,
            "queue_sha256": self.queue_sha256,
            "policy_sha256": self.policy_sha256,
            "candidate_count": len(self.items),
            "matched_count": len(selected),
            "offset": offset,
            "limit": limit,
            "items": selected[offset : offset + limit],
            "policy": {
                "visual_decisions": self.policy["visual_decisions"],
                "rights_decisions": self.policy["rights_decisions"],
                "target_classes": self.policy["target_classes"],
                "unknown_scenarios": self.policy["unknown_scenarios"],
                "family_roles": self.policy["family_roles"],
                "visual_reason_codes": self.policy["visual_reason_codes"],
                "rights_reason_codes": self.policy["rights_reason_codes"],
            },
        }

    def bootstrap(self) -> dict[str, Any]:
        return {
            "status": self.current_status,
            "mode": "PRODUCTION" if self.production_mode else "FIXTURE",
            "csrf_token": self.csrf_token,
            "stats": self.stats(),
            "policy": {
                "visual_decisions": self.policy["visual_decisions"],
                "rights_decisions": self.policy["rights_decisions"],
                "target_classes": self.policy["target_classes"],
                "unknown_scenarios": self.policy["unknown_scenarios"],
                "family_roles": self.policy["family_roles"],
                "visual_reason_codes": self.policy["visual_reason_codes"],
                "rights_reason_codes": self.policy["rights_reason_codes"],
                "notes_max_length": self.policy["notes_max_length"],
                "reviewer_pattern": self.policy["reviewer_pattern"],
                "reviewed_source_group_pattern": self.policy[
                    "reviewed_source_group_pattern"
                ],
                "near_duplicate_family_pattern": self.policy[
                    "near_duplicate_family_pattern"
                ],
                "source_page_revision_id_pattern": self.policy[
                    "source_page_revision_id_pattern"
                ],
            },
            "rules": {
                "reviewed_source_group_is_independence_family_not_v2_source_group": True,
                "acquisition_hint_must_not_auto_map_unknown_scenario": True,
            },
            "authority": AUTHORITY,
            "explicit_non_claims": EXPLICIT_NON_CLAIMS,
            "rights_attestation_scope": "PERMALINK_BOUND_HUMAN_ATTESTATION_NO_PAGE_SNAPSHOT",
        }

    def stats(self) -> dict[str, Any]:
        statuses = Counter()
        classes = Counter()
        visual_axes = Counter()
        rights_axes = Counter()
        target_classes = Counter()
        visual_reviewers = Counter()
        rights_reviewers = Counter()
        canonical_groups: set[str] = set()
        for row in self.items:
            item = self.public_item(row["asset"])
            statuses[item["review_status"]] += 1
            classes[row["class_hint"]] += 1
            visual_axes[item["decision"]["visual_decision"]] += 1
            rights_axes[item["decision"]["rights_decision"]] += 1
            if item["decision"]["visual_decision"] == "PASS":
                target_classes[item["decision"]["target_class"]] += 1
            visual_reviewer = item["decision"]["visual_reviewer"]
            rights_reviewer = item["decision"]["rights_reviewer"]
            if visual_reviewer:
                visual_reviewers[visual_reviewer] += 1
            if rights_reviewer:
                rights_reviewers[rights_reviewer] += 1
            if item["decision"]["family_role"] == "CANONICAL_REPRESENTATIVE":
                canonical_groups.add(item["decision"]["reviewed_source_group"])
        return {
            "status": self.current_status,
            "candidate_count": len(self.items),
            "event_count": self.event_count,
            "evented_asset_count": len(self.latest_by_asset),
            "reviewed_asset_count": len(self.items) - statuses["UNREVIEWED"],
            "finalized_asset_count": sum(
                count
                for status, count in statuses.items()
                if status
                in {
                    "REJECTED_VISUAL_PENDING_EXPORT_NOT_DATA_LOCKED",
                    "REJECTED_RIGHTS_PENDING_EXPORT_NOT_DATA_LOCKED",
                    "CANONICAL_READY_FOR_FAMILY_AUDIT_NOT_DATA_LOCKED",
                    "SERIES_SIBLING_EXCLUDED_NOT_DATA_LOCKED",
                }
            ),
            "needs_review_asset_count": statuses["NEEDS_REVIEW"],
            "review_status_counts": dict(sorted(statuses.items())),
            "visual_axis_counts": dict(sorted(visual_axes.items())),
            "rights_axis_counts": dict(sorted(rights_axes.items())),
            "visual_pass_target_class_counts": dict(sorted(target_classes.items())),
            "class_hint_counts": dict(sorted(classes.items())),
            "visual_reviewer_counts": dict(sorted(visual_reviewers.items())),
            "rights_reviewer_counts": dict(sorted(rights_reviewers.items())),
            "canonical_independence_family_count": len(canonical_groups),
            "last_event_sha256": self.last_event_sha256,
            "data_locked": False,
            "training_authority": False,
            "print_authority": False,
        }

    def verify_readonly_state(self) -> None:
        with self._lock:
            if self.degraded:
                raise ReviewError("review store is degraded; restart for fail-closed recovery")
            try:
                self._verify_runtime_inputs()
                self._assert_journal_unchanged()
                self._assert_checkpoint_matches_memory()
                for event in self.events:
                    self._validate_rights_evidence_binding(
                        self.items_by_asset[event["asset"]], event["decision"]
                    )
                if self.current_status.endswith("HUMAN_REVIEW_COMPLETE_NOT_DATA_LOCKED"):
                    for row in self.items:
                        if _sha256_file(self._bound_image_path(row)) != row["sha256"]:
                            raise ReviewError(
                                "completed review candidate payload is missing or changed"
                            )
            except Exception:
                self.degraded = True
                raise

    def verify_operational_state(self) -> None:
        with self._lock:
            if self.degraded:
                raise ReviewError("review store is degraded; restart for fail-closed recovery")
            try:
                self._verify_runtime_inputs()
                self._assert_journal_unchanged()
                self._assert_checkpoint_matches_memory()
            except Exception:
                self.degraded = True
                raise

    def image(self, asset: str) -> tuple[bytes, str]:
        row = self.items_by_asset.get(asset)
        if row is None:
            raise ReviewError("unknown image asset")
        path = self._bound_image_path(row)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ReviewError(f"cannot read image payload: {exc}") from exc
        if _sha256_bytes(payload) != row["sha256"]:
            raise ReviewError("image payload SHA-256 mismatch")
        mime = row.get("download_mime")
        if mime not in {"image/jpeg", "image/png"}:
            raise ReviewError("image MIME is not allowed")
        return payload, mime


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server_version = "RootScopeHumanReview/1.0"

    @property
    def app(self) -> "ReviewHTTPServer":
        return self.server  # type: ignore[return-value]

    def _headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()

    def _json(self, status: int, value: Mapping[str, Any]) -> None:
        payload = _json_bytes(value)
        self._headers(status, "application/json; charset=utf-8", len(payload))
        self.wfile.write(payload)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"ok": False, "error": message, "data_locked": False})

    def _validate_host(self) -> None:
        try:
            client_ip = ipaddress.ip_address(self.client_address[0])
        except ValueError as exc:
            raise ReviewError("client address is invalid") from exc
        if not client_ip.is_loopback:
            raise ReviewError("review requests are accepted only from loopback clients")
        host_header = self.headers.get("Host")
        if not host_header or len(host_header) > 255:
            raise ReviewError("a loopback Host header is required")
        parsed = urlparse(f"//{host_header}")
        if parsed.username or parsed.password or parsed.path not in {"", "/"}:
            raise ReviewError("Host header is invalid")
        if parsed.hostname != "127.0.0.1":
            raise ReviewError("Host header must use numeric 127.0.0.1")
        if parsed.port is None or parsed.port != self.app.server_address[1]:
            raise ReviewError("Host port does not match the review server")

    def _validate_post_origin(self) -> None:
        if self.headers.get("Sec-Fetch-Site", "").lower() in {"cross-site", "none"}:
            raise ReviewError("cross-site review submissions are forbidden")
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlparse(origin)
            if (
                parsed.scheme != "http"
                or parsed.hostname != "127.0.0.1"
                or parsed.port != self.app.server_address[1]
                or parsed.netloc.lower() != self.headers["Host"].lower()
            ):
                raise ReviewError("submission Origin does not exactly match Host")
        token = self.headers.get("X-RootScope-Review-Token", "")
        if not secrets.compare_digest(token, self.app.store.csrf_token):
            raise ReviewError("review token is missing or invalid")

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._validate_host()
            if len(self.path) > 4096:
                raise ReviewError("request target is too long")
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.app.store.verify_operational_state()
                if parsed.query:
                    raise ReviewError("unexpected query parameters")
                payload = self.app.ui_path.read_bytes()
                if _sha256_bytes(payload) != self.app.store.ui_sha256:
                    raise ReviewError("review UI changed after server startup")
                self._headers(HTTPStatus.OK, "text/html; charset=utf-8", len(payload))
                self.wfile.write(payload)
                return
            if parsed.path == "/api/stats":
                self.app.store.verify_readonly_state()
                if parsed.query:
                    raise ReviewError("unexpected query parameters")
                self._json(HTTPStatus.OK, self.app.store.stats())
                return
            if parsed.path == "/api/bootstrap":
                self.app.store.verify_readonly_state()
                if parsed.query:
                    raise ReviewError("unexpected query parameters")
                self._json(HTTPStatus.OK, self.app.store.bootstrap())
                return
            if parsed.path == "/api/items":
                self.app.store.verify_readonly_state()
                values = parse_qs(parsed.query, keep_blank_values=True)
                allowed = {"class_hint", "species_hint", "review_status", "q", "offset", "limit"}
                if set(values) - allowed or any(len(entries) != 1 for entries in values.values()):
                    raise ReviewError("invalid or repeated item query parameter")
                result = self.app.store.list_items(
                    class_hint=values.get("class_hint", [""])[0],
                    species_hint=values.get("species_hint", [""])[0],
                    review_status=values.get("review_status", [""])[0],
                    query=values.get("q", [""])[0],
                    offset=int(values.get("offset", ["0"])[0]),
                    limit=int(values.get("limit", ["24"])[0]),
                )
                self._json(HTTPStatus.OK, result)
                return
            if parsed.path == "/image":
                self.app.store.verify_operational_state()
                values = parse_qs(parsed.query, keep_blank_values=True)
                if set(values) != {"asset"} or len(values["asset"]) != 1:
                    raise ReviewError("image request requires exactly one asset")
                asset = values.get("asset", [""])[0]
                payload, mime = self.app.store.image(asset)
                self._headers(HTTPStatus.OK, mime, len(payload))
                self.wfile.write(payload)
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except (ReviewError, OSError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._validate_host()
            self._validate_post_origin()
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/decision", "/api/rights-evidence"} or parsed.query:
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise ReviewError("Content-Type must be application/json")
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ReviewError("Content-Length is required")
            length = int(raw_length)
            if not 1 <= length <= 65536:
                raise ReviewError("request body length is invalid")
            payload = self.rfile.read(length)
            value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
            if not isinstance(value, dict):
                raise ReviewError("request body must be a JSON object")
            if parsed.path == "/api/decision":
                item = self.app.store.record_decision(value)
                result = {"ok": True, "item": item, "data_locked": False}
            else:
                evidence = self.app.store.create_rights_evidence(value)
                result = {"ok": True, "evidence": evidence, "data_locked": False}
            self._json(HTTPStatus.OK, result)
        except (ReviewError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {format % args}")


class ReviewHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], store: ReviewStore, ui_path: Path):
        self.store = store
        self.ui_path = Path(ui_path).resolve(strict=True)
        if address[0] != "127.0.0.1" or self.ui_path != store.ui_path:
            raise ReviewError("HTTP server requires the store-bound numeric loopback/UI")
        super().__init__(address, ReviewRequestHandler)
        if self.server_address[0] != "127.0.0.1":
            super().server_close()
            raise ReviewError("HTTP server did not bind numeric IPv4 loopback")

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            self.store.close()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ui", type=Path, default=DEFAULT_UI)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fixture-mode", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        print("Review server refused: host must be numeric 127.0.0.1")
        return 2
    if not 1 <= args.port <= 65535:
        print("Review server refused: invalid port")
        return 2
    try:
        requested_production = (
            args.queue.resolve(strict=True) == DEFAULT_QUEUE.resolve(strict=True)
            and args.policy.resolve(strict=True) == DEFAULT_POLICY.resolve(strict=True)
            and args.ui.resolve(strict=True) == DEFAULT_UI.resolve(strict=True)
            and args.output_dir.resolve(strict=False) == DEFAULT_OUTPUT.resolve(strict=False)
        )
    except OSError as exc:
        print(f"Review server refused before creating outputs: {exc}")
        return 2
    if requested_production and args.fixture_mode:
        print("Review server refused before creating outputs: production tuple cannot use fixture mode")
        return 2
    if not requested_production:
        if not args.fixture_mode:
            print(
                "Review server refused before creating outputs: non-default inputs require explicit fixture mode"
            )
            return 2
        if args.output_dir.resolve(strict=False) == DEFAULT_OUTPUT.resolve(strict=False):
            print(
                "Review server refused before creating outputs: fixture mode cannot use the production output directory"
            )
            return 2
    store: ReviewStore | None = None
    try:
        store = ReviewStore(args.queue, args.output_dir, args.policy, args.ui)
        if store.production_mode != requested_production:
            raise ReviewError("resolved review mode changed during startup")
        server = ReviewHTTPServer((args.host, args.port), store, args.ui)
    except (ReviewError, OSError) as exc:
        if store is not None:
            store.close()
        print(f"Review server refused: {exc}")
        return 2
    print(
        f"RootScope reviewer: http://{args.host}:{args.port}/ | "
        f"mode={'PRODUCTION' if store.production_mode else 'FIXTURE'} | "
        f"candidates={len(store.items)} | DATA_LOCKED=false"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
