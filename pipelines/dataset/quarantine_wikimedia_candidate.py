#!/usr/bin/env python3
"""Reversibly quarantine one fail-closed Wikimedia staging candidate.

The operation is explicit, hash-bound and recoverable.  It never deletes the
payload: the exact file and original manifest record are retained under
``quarantine/<receipt_id>/``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image


SCHEMA_VERSION = "rootscope.wikimedia_candidate_quarantine.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,95}$")


class QuarantineError(RuntimeError):
    """Raised when the quarantine transaction must fail closed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QuarantineError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_manifest(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise QuarantineError(f"cannot read UTF-8 manifest: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise QuarantineError(f"blank manifest line {line_number}")
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, QuarantineError) as exc:
            raise QuarantineError(f"invalid manifest line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise QuarantineError(f"manifest line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise QuarantineError("manifest is empty")
    return raw, rows


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, QuarantineError) as exc:
        raise QuarantineError(f"cannot read {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise QuarantineError(f"{context} is not a JSON object")
    return value


def _remove_manifest_row(
    raw: bytes, rows: list[dict[str, Any]], pageid: int
) -> tuple[bytes, dict[str, Any], int, str, str]:
    """Remove only the target raw line while preserving every other byte."""

    text = raw.decode("utf-8")
    raw_lines = text.splitlines(keepends=True)
    if len(raw_lines) != len(rows):
        raise QuarantineError("manifest raw-line/object count mismatch")
    indexes = [
        index
        for index, row in enumerate(rows)
        if type(row.get("pageid")) is int and row["pageid"] == pageid
    ]
    if len(indexes) != 1:
        raise QuarantineError(
            f"manifest must contain exactly one pageid={pageid}, found {len(indexes)}"
        )
    index = indexes[0]
    raw_line = raw_lines[index]
    after = "".join(raw_lines[:index] + raw_lines[index + 1 :]).encode("utf-8")
    return (
        after,
        rows[index],
        index + 1,
        _sha256_bytes(raw_line.encode("utf-8")),
        raw_line,
    )


def _manifest_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        )
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


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


@contextlib.contextmanager
def _dataset_writer_lock(dataset: Path):
    """Mutually exclude this transaction and the PowerShell collector."""

    lock_path = dataset / ".collector.lock"
    try:
        handle = lock_path.open("a+b")
    except OSError as exc:
        raise QuarantineError(f"cannot open collector lock: {exc}") from exc
    locked = False
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except (OSError, ImportError) as exc:
            raise QuarantineError("another collector or quarantine transaction owns the lock") from exc
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _bound_file(dataset: Path, filename: Any) -> tuple[Path, str]:
    if not isinstance(filename, str) or not filename or "\\" in filename:
        raise QuarantineError("record filename must be a non-empty POSIX relative path")
    pure = PurePosixPath(filename)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise QuarantineError("record filename is unsafe")
    root = dataset.resolve(strict=True)
    candidate = dataset.joinpath(*pure.parts).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise QuarantineError("record filename escapes dataset root") from exc
    return candidate, pure.as_posix()


def _complete_receipt(intent: dict[str, Any]) -> dict[str, Any]:
    return {
        **intent,
        "status": "COMPLETE",
        "file_move_performed": True,
        "manifest_row_removed": True,
        "reversible": True,
        "restore_contract": (
            "Verify the receipt, payload and manifest-before backup hashes; move the "
            "payload to source, then atomically restore manifest.before.jsonl."
        ),
    }


def _recover_intent(
    dataset: Path,
    manifest_path: Path,
    intent_path: Path,
    receipt_path: Path,
    *,
    pageid: int,
    expected_manifest_sha256: str,
    receipt_id: str,
    reason: str,
) -> dict[str, Any]:
    intent = _read_json_object(intent_path, "quarantine intent")
    expected_binding = {
        "schema_version": SCHEMA_VERSION,
        "status": "INTENT",
        "receipt_id": receipt_id,
        "pageid": pageid,
        "reason": reason,
        "manifest_before_sha256": expected_manifest_sha256,
    }
    for key, expected in expected_binding.items():
        if intent.get(key) != expected:
            raise QuarantineError(f"existing intent conflicts at {key}")

    source, source_relative = _bound_file(dataset, intent.get("source"))
    destination, destination_relative = _bound_file(dataset, intent.get("destination"))
    backup, backup_relative = _bound_file(dataset, intent.get("manifest_before_backup"))
    if source_relative != intent.get("source") or destination_relative != intent.get("destination"):
        raise QuarantineError("intent paths are not canonical")
    if backup_relative != intent.get("manifest_before_backup"):
        raise QuarantineError("intent manifest backup path is not canonical")
    if not backup.is_file() or _sha256_file(backup) != expected_manifest_sha256:
        raise QuarantineError("manifest-before backup is missing or changed")

    raw = manifest_path.read_bytes()
    rows = _read_manifest(manifest_path)[1] if raw else []
    current_sha = _sha256_bytes(raw)
    before_sha = intent["manifest_before_sha256"]
    after_sha = intent.get("manifest_after_sha256")
    file_sha = intent.get("file_sha256")
    if not isinstance(after_sha, str) or HEX64.fullmatch(after_sha) is None:
        raise QuarantineError("intent manifest-after SHA-256 is invalid")
    if not isinstance(file_sha, str) or HEX64.fullmatch(file_sha) is None:
        raise QuarantineError("intent file SHA-256 is invalid")

    source_exists = source.is_file()
    destination_exists = destination.is_file()
    if source_exists == destination_exists:
        raise QuarantineError("recovery requires the payload at exactly one bound path")
    payload = source if source_exists else destination
    if _sha256_file(payload) != file_sha:
        raise QuarantineError("recovery payload SHA-256 mismatch")

    if current_sha == before_sha:
        after_bytes, record, line_number, raw_line_sha, raw_line = _remove_manifest_row(
            raw, rows, pageid
        )
        if _sha256_bytes(after_bytes) != after_sha:
            raise QuarantineError("recovered manifest-after projection disagrees with intent")
        if record != intent.get("record"):
            raise QuarantineError("recovered manifest record disagrees with intent")
        if (
            line_number != intent.get("manifest_line_number")
            or raw_line_sha != intent.get("manifest_raw_line_sha256")
            or raw_line != intent.get("manifest_raw_line")
        ):
            raise QuarantineError("recovered raw manifest line disagrees with intent")
        if source_exists:
            source.rename(destination)
        if _sha256_file(manifest_path) != before_sha:
            raise QuarantineError("manifest changed before recovery commit")
        _atomic_write(manifest_path, after_bytes)
    elif current_sha == after_sha:
        if source_exists or not destination_exists:
            raise QuarantineError("manifest is committed but payload state is inconsistent")
        if any(type(row.get("pageid")) is int and row["pageid"] == pageid for row in rows):
            raise QuarantineError("committed manifest still contains quarantined pageid")
    else:
        raise QuarantineError("manifest does not match a recoverable transaction state")

    if not destination.is_file() or _sha256_file(destination) != file_sha:
        raise QuarantineError("post-recovery payload verification failed")
    if _sha256_file(manifest_path) != after_sha:
        raise QuarantineError("post-recovery manifest verification failed")
    receipt = _complete_receipt(intent)
    _atomic_write(receipt_path, _json_bytes(receipt))
    return receipt


def _quarantine_candidate_locked(
    dataset: Path,
    pageid: int,
    expected_manifest_sha256: str,
    receipt_id: str,
    reason: str,
) -> dict[str, Any]:
    manifest_path = dataset / "manifest.jsonl"
    quarantine_dir = dataset / "quarantine" / receipt_id
    intent_path = quarantine_dir / "intent.json"
    receipt_path = quarantine_dir / "receipt.json"

    if receipt_path.is_file():
        receipt = _read_json_object(receipt_path, "quarantine receipt")
        if (
            receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("status") != "COMPLETE"
            or receipt.get("pageid") != pageid
            or receipt.get("manifest_before_sha256") != expected_manifest_sha256
            or receipt.get("receipt_id") != receipt_id
            or receipt.get("reason") != reason
        ):
            raise QuarantineError("existing receipt conflicts with requested transaction")
        destination, canonical_destination = _bound_file(dataset, receipt.get("destination"))
        if canonical_destination != receipt.get("destination"):
            raise QuarantineError("receipt destination is not canonical")
        if not destination.is_file() or _sha256_file(destination) != receipt["file_sha256"]:
            raise QuarantineError("completed quarantine payload is missing or changed")
        current = manifest_path.read_bytes()
        rows = _read_manifest(manifest_path)[1] if current else []
        if any(type(row.get("pageid")) is int and row["pageid"] == pageid for row in rows):
            raise QuarantineError("completed quarantine pageid reappeared in the manifest")
        return receipt

    if intent_path.is_file():
        return _recover_intent(
            dataset,
            manifest_path,
            intent_path,
            receipt_path,
            pageid=pageid,
            expected_manifest_sha256=expected_manifest_sha256,
            receipt_id=receipt_id,
            reason=reason,
        )

    raw, rows = _read_manifest(manifest_path)
    current_sha = _sha256_bytes(raw)
    if current_sha != expected_manifest_sha256:
        raise QuarantineError(
            f"manifest SHA mismatch: expected {expected_manifest_sha256}, got {current_sha}"
        )

    after_bytes, record, line_number, raw_line_sha, raw_line = _remove_manifest_row(
        raw, rows, pageid
    )
    source, relative_source = _bound_file(dataset, record.get("filename"))
    expected_file_sha = record.get("download_sha256")
    if not isinstance(expected_file_sha, str) or HEX64.fullmatch(expected_file_sha) is None:
        raise QuarantineError("record download_sha256 is invalid")
    if not source.is_file() or _sha256_file(source) != expected_file_sha:
        raise QuarantineError("source payload is missing or does not match the manifest")
    try:
        with Image.open(source) as image:
            image.load()
            width, height = image.size
            decoded_format = image.format
    except (OSError, ValueError) as exc:
        raise QuarantineError(f"source payload cannot be decoded: {exc}") from exc

    after_sha = _sha256_bytes(after_bytes)
    destination_relative = (
        PurePosixPath("quarantine") / receipt_id / "payload" / Path(relative_source).name
    ).as_posix()
    destination, canonical_destination = _bound_file(dataset, destination_relative)
    if canonical_destination != destination_relative or destination.exists():
        raise QuarantineError("quarantine destination is unsafe or already exists")
    backup_relative = (
        PurePosixPath("quarantine") / receipt_id / "manifest.before.jsonl"
    ).as_posix()
    backup, canonical_backup = _bound_file(dataset, backup_relative)
    if canonical_backup != backup_relative or backup.exists():
        raise QuarantineError("manifest-before backup is unsafe or already exists")
    intent = {
        "schema_version": SCHEMA_VERSION,
        "status": "INTENT",
        "receipt_id": receipt_id,
        "pageid": pageid,
        "reason": reason,
        "manifest_before_sha256": current_sha,
        "manifest_after_sha256": after_sha,
        "manifest_before_backup": backup_relative,
        "manifest_line_number": line_number,
        "manifest_raw_line_sha256": raw_line_sha,
        "manifest_raw_line": raw_line,
        "file_sha256": expected_file_sha,
        "decoded_width": width,
        "decoded_height": height,
        "decoded_format": decoded_format,
        "source": relative_source,
        "destination": destination_relative,
        "record": record,
        "delete_performed": False,
    }
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(backup, raw)
    if _sha256_file(backup) != current_sha:
        raise QuarantineError("manifest-before backup verification failed")
    _atomic_write(intent_path, _json_bytes(intent))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _sha256_file(manifest_path) != current_sha:
        raise QuarantineError("manifest changed before quarantine commit")
    if not source.is_file() or _sha256_file(source) != expected_file_sha:
        raise QuarantineError("source payload changed before quarantine commit")
    source.rename(destination)
    _atomic_write(manifest_path, after_bytes)
    if _sha256_file(manifest_path) != after_sha:
        raise QuarantineError("post-write manifest verification failed")

    receipt = _complete_receipt(intent)
    _atomic_write(receipt_path, _json_bytes(receipt))
    return receipt


def quarantine_candidate(
    dataset: Path,
    pageid: int,
    expected_manifest_sha256: str,
    receipt_id: str,
    reason: str,
) -> dict[str, Any]:
    dataset = Path(dataset).resolve(strict=True)
    if type(pageid) is not int or pageid <= 0:
        raise QuarantineError("pageid must be a positive integer")
    if HEX64.fullmatch(expected_manifest_sha256) is None:
        raise QuarantineError("expected manifest SHA-256 is invalid")
    if TOKEN.fullmatch(receipt_id) is None or TOKEN.fullmatch(reason) is None:
        raise QuarantineError("receipt_id and reason must be safe explicit tokens")
    with _dataset_writer_lock(dataset):
        return _quarantine_candidate_locked(
            dataset, pageid, expected_manifest_sha256, receipt_id, reason
        )


def main(argv: list[str] | None = None) -> int:
    default_dataset = Path(__file__).resolve().parents[2] / "datasets" / "desert_plants_wikimedia_staging_e0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--pageid", type=int, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--receipt-id", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = quarantine_candidate(
            args.dataset,
            args.pageid,
            args.expected_manifest_sha256,
            args.receipt_id,
            args.reason,
        )
    except (QuarantineError, OSError, json.JSONDecodeError) as exc:
        print(f"Quarantine failed: {exc}")
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
