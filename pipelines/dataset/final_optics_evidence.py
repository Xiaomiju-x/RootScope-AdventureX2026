#!/usr/bin/env python3
"""Build and verify RootScope final-optics evidence-package roots.

This tool is deliberately narrower than a physical acceptance procedure.  It
binds regular files into deterministic manifests, verifies those manifests,
and checks that a human-authored B/C/D receipt refers to the same bytes.  It
does not create receipts, sign approvals, validate physical claims, or grant
training/printing authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


MANIFEST_SCHEMA_VERSION = "rootscope.final_optics.evidence_manifest.v1"
PREFLIGHT_SCHEMA_VERSION = "rootscope.final_optics.structural_preflight.v1"
RECEIPT_SCHEMA_VERSION = "1.0.0"
EVIDENCE_KINDS = ("uvc", "lighting", "paper", "printer", "geometry")
ROLE_MEMBERS = {"hardware": "B", "mechanical": "C", "operations": "D"}
RECEIPT_TOP_LEVEL_FIELDS = {
    "schema_version",
    "receipt_id",
    "signed_roles",
    "evidence_roots",
}
ROLE_ENTRY_FIELDS = {"member", "signed", "signer", "approval_evidence_sha256"}
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_CONTROL_FILE_BYTES = 16 * 1024 * 1024
WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class EvidenceError(RuntimeError):
    """Raised when evidence cannot be bound or verified safely."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise EvidenceError(f"non-finite JSON constant: {value}")


def _strict_json_object(raw: bytes, context: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, EvidenceError) as exc:
        raise EvidenceError(f"cannot parse {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{context} must be a JSON object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"value cannot be canonically serialized: {exc}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_reparse(stat_result: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(stat_result, "st_file_attributes", 0) & flag)


def _fingerprint(stat_result: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        stat.S_IFMT(stat_result.st_mode),
        stat_result.st_size,
        getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)),
        getattr(stat_result, "st_ctime_ns", int(stat_result.st_ctime * 1_000_000_000)),
        stat_result.st_dev,
        stat_result.st_ino,
    )


def _open_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    """Fields that Windows reports consistently for path-stat and handle-stat.

    On Windows, ``st_ctime_ns`` obtained through an open handle can differ from
    the immediately preceding ``lstat`` value even when the file is unchanged.
    Path-to-path fingerprints still include ctime; this reduced tuple is used
    only to prove that the opened handle names the same regular file.
    """

    return (
        stat.S_IFMT(stat_result.st_mode),
        stat_result.st_size,
        getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)),
        stat_result.st_dev,
        stat_result.st_ino,
    )


def _ensure_real_directory(path: Path, context: str) -> Path:
    candidate = Path(path)
    try:
        info = os.lstat(candidate)
    except OSError as exc:
        raise EvidenceError(f"{context} is missing or unreadable: {candidate}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
        raise EvidenceError(f"{context} must not be a symlink, junction, or reparse point: {candidate}")
    if not stat.S_ISDIR(info.st_mode):
        raise EvidenceError(f"{context} is not a directory: {candidate}")
    return candidate.resolve(strict=True)


def _validate_relative_path(relative: str) -> str:
    if not relative or "\\" in relative:
        raise EvidenceError("evidence path must be a non-empty POSIX relative path")
    if unicodedata.normalize("NFC", relative) != relative:
        raise EvidenceError(f"evidence path is not NFC-normalized: {relative!r}")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise EvidenceError(f"unsafe evidence path: {relative!r}")
    for part in pure.parts:
        if any(unicodedata.category(character).startswith("C") for character in part):
            raise EvidenceError(f"evidence path contains a control character: {relative!r}")
        stem = part.split(".", 1)[0].upper()
        if ":" in part or part.endswith((" ", ".")) or stem in WINDOWS_RESERVED:
            raise EvidenceError(f"evidence path is unsafe on Windows: {relative!r}")
    try:
        relative.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise EvidenceError(f"evidence path is not valid UTF-8: {relative!r}") from exc
    return relative


def _path_identity(relative: str) -> str:
    return unicodedata.normalize("NFKC", relative).casefold()


def _regular_file_lstat(path: Path, context: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise EvidenceError(f"cannot inspect {context}: {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
        raise EvidenceError(f"{context} must not be a symlink or reparse point: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise EvidenceError(f"{context} is not a regular file: {path}")
    return info


def _hash_regular_file(path: Path, context: str = "evidence file") -> tuple[str, int, tuple[int, int, int, int, int, int]]:
    before = _regular_file_lstat(path, context)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot open {context}: {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if _open_identity(opened) != _open_identity(before):
            raise EvidenceError(f"{context} changed while it was opened: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after_handle = os.fstat(handle.fileno())
            if _fingerprint(after_handle) != _fingerprint(opened):
                raise EvidenceError(f"{context} changed while it was hashed: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after_path = _regular_file_lstat(path, context)
    if _fingerprint(after_path) != _fingerprint(before):
        raise EvidenceError(f"{context} changed while it was hashed: {path}")
    return digest.hexdigest(), before.st_size, _fingerprint(before)


def _read_stable_regular_bytes(path: Path, context: str, *, maximum: int) -> bytes:
    before = _regular_file_lstat(path, context)
    if before.st_size > maximum:
        raise EvidenceError(f"{context} exceeds {maximum} bytes: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read {context}: {path}: {exc}") from exc
    after = _regular_file_lstat(path, context)
    if _fingerprint(after) != _fingerprint(before) or len(raw) != before.st_size:
        raise EvidenceError(f"{context} changed while it was read: {path}")
    return raw


def _collect_regular_files(payload_root: Path) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    identities: dict[str, str] = {}
    for current_raw, directory_names, file_names in os.walk(payload_root, followlinks=False):
        current = Path(current_raw)
        current_info = os.lstat(current)
        if stat.S_ISLNK(current_info.st_mode) or _is_reparse(current_info):
            raise EvidenceError(f"evidence directory is a symlink or reparse point: {current}")
        for name in directory_names:
            directory = current / name
            info = os.lstat(directory)
            relative = directory.relative_to(payload_root).as_posix()
            _validate_relative_path(relative)
            if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
                raise EvidenceError(f"evidence directory is a symlink or reparse point: {directory}")
            if not stat.S_ISDIR(info.st_mode):
                raise EvidenceError(f"evidence directory entry is not a directory: {directory}")
        for name in file_names:
            path = current / name
            _regular_file_lstat(path, "evidence file")
            relative = _validate_relative_path(path.relative_to(payload_root).as_posix())
            try:
                path.resolve(strict=True).relative_to(payload_root)
            except (OSError, ValueError) as exc:
                raise EvidenceError(f"evidence file escapes payload root: {path}") from exc
            identity = _path_identity(relative)
            prior = identities.get(identity)
            if prior is not None:
                raise EvidenceError(f"evidence paths collide after normalization: {prior!r}, {relative!r}")
            identities[identity] = relative
            records.append((relative, path))
    records.sort(key=lambda item: item[0].encode("utf-8"))
    if not records:
        raise EvidenceError(f"evidence package is empty: {payload_root}")
    return records


def _validate_kind(evidence_kind: str) -> str:
    if evidence_kind not in EVIDENCE_KINDS:
        raise EvidenceError(f"unknown evidence kind: {evidence_kind!r}")
    return evidence_kind


def build_package_manifest(payload_dir: Path, evidence_kind: str) -> tuple[dict[str, Any], str]:
    """Build an in-memory deterministic manifest without changing the payload."""

    evidence_kind = _validate_kind(evidence_kind)
    payload_root = _ensure_real_directory(Path(payload_dir), f"{evidence_kind} payload directory")
    initial_files = _collect_regular_files(payload_root)
    file_records: list[dict[str, Any]] = []
    fingerprints: dict[str, tuple[int, int, int, int, int, int]] = {}
    for relative, path in initial_files:
        digest, byte_count, fingerprint = _hash_regular_file(path)
        file_records.append({"path": relative, "bytes": byte_count, "sha256": digest})
        fingerprints[relative] = fingerprint

    final_files = _collect_regular_files(payload_root)
    if [item[0] for item in final_files] != [item[0] for item in initial_files]:
        raise EvidenceError("evidence package changed while its manifest was being built")
    for relative, path in final_files:
        if _fingerprint(_regular_file_lstat(path, "evidence file")) != fingerprints[relative]:
            raise EvidenceError(f"evidence file changed after hashing: {relative}")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "evidence_kind": evidence_kind,
        "files": file_records,
    }
    return manifest, _sha256_bytes(_canonical_bytes(manifest))


def _ensure_outside_payload(output: Path, payload_root: Path, context: str) -> None:
    resolved = Path(output).resolve(strict=False)
    try:
        resolved.relative_to(payload_root)
    except ValueError:
        return
    raise EvidenceError(f"{context} must be outside the evidence payload directory")


def _atomic_create(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = _ensure_real_directory(path.parent, "manifest output parent")
    target = parent / path.name
    if target.exists() or target.is_symlink():
        raise EvidenceError(f"refusing to overwrite existing manifest: {target}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise EvidenceError(f"refusing to overwrite existing manifest: {target}") from exc
        except OSError as exc:
            raise EvidenceError(f"cannot atomically publish manifest: {target}: {exc}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_package_manifest(payload_dir: Path, evidence_kind: str, manifest_path: Path) -> dict[str, Any]:
    payload_root = _ensure_real_directory(Path(payload_dir), f"{evidence_kind} payload directory")
    _ensure_outside_payload(Path(manifest_path), payload_root, "manifest")
    manifest, evidence_root = build_package_manifest(payload_root, evidence_kind)
    canonical = _canonical_bytes(manifest)
    _atomic_create(Path(manifest_path), canonical)
    return {
        "evidence_kind": evidence_kind,
        "evidence_root": evidence_root,
        "file_count": len(manifest["files"]),
        "total_bytes": sum(record["bytes"] for record in manifest["files"]),
        "manifest": str(Path(manifest_path).resolve(strict=True)),
    }


def _validate_manifest_shape(value: dict[str, Any], expected_kind: str) -> None:
    if set(value) != {"schema_version", "evidence_kind", "files"}:
        raise EvidenceError("evidence manifest has unexpected top-level fields")
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise EvidenceError("evidence manifest schema_version is not frozen v1")
    if value.get("evidence_kind") != expected_kind:
        raise EvidenceError("evidence manifest kind does not match its package")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise EvidenceError("evidence manifest files must be a non-empty list")
    paths: list[str] = []
    identities: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise EvidenceError(f"evidence manifest file record {index} has an invalid shape")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise EvidenceError(f"evidence manifest file record {index} has an invalid path")
        relative = _validate_relative_path(relative)
        if type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise EvidenceError(f"evidence manifest file record {index} has invalid bytes")
        digest = record.get("sha256")
        if type(digest) is not str or HEX64_RE.fullmatch(digest) is None:
            raise EvidenceError(f"evidence manifest file record {index} has invalid SHA-256")
        identity = _path_identity(relative)
        if identity in identities:
            raise EvidenceError("evidence manifest paths collide after normalization")
        identities.add(identity)
        paths.append(relative)
    if paths != sorted(paths, key=lambda item: item.encode("utf-8")):
        raise EvidenceError("evidence manifest files are not UTF-8 path sorted")


def verify_package_manifest(payload_dir: Path, evidence_kind: str, manifest_path: Path) -> dict[str, Any]:
    evidence_kind = _validate_kind(evidence_kind)
    payload_root = _ensure_real_directory(Path(payload_dir), f"{evidence_kind} payload directory")
    _ensure_outside_payload(Path(manifest_path), payload_root, "manifest")
    raw = _read_stable_regular_bytes(Path(manifest_path), "evidence manifest", maximum=MAX_CONTROL_FILE_BYTES)
    stored = _strict_json_object(raw, "evidence manifest")
    _validate_manifest_shape(stored, evidence_kind)
    canonical = _canonical_bytes(stored)
    if raw != canonical:
        raise EvidenceError("evidence manifest bytes are not canonical compact sorted-key JSON")
    rebuilt, rebuilt_root = build_package_manifest(payload_root, evidence_kind)
    if stored != rebuilt:
        raise EvidenceError("evidence manifest does not match the current package bytes")
    stored_root = _sha256_bytes(raw)
    if stored_root != rebuilt_root:
        raise EvidenceError("evidence manifest root does not match the rebuilt root")
    return {
        "evidence_kind": evidence_kind,
        "evidence_root": stored_root,
        "file_count": len(stored["files"]),
        "total_bytes": sum(record["bytes"] for record in stored["files"]),
        "manifest": str(Path(manifest_path).resolve(strict=True)),
    }


def _canonical_signer_identity(value: Any) -> str | None:
    if type(value) is not str:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        return None
    canonical = " ".join(normalized.split()).casefold()
    return canonical or None


def _validate_receipt(receipt_path: Path) -> tuple[dict[str, Any], str]:
    raw = _read_stable_regular_bytes(Path(receipt_path), "optical-domain receipt", maximum=MAX_CONTROL_FILE_BYTES)
    receipt = _strict_json_object(raw, "optical-domain receipt")
    if set(receipt) != RECEIPT_TOP_LEVEL_FIELDS:
        raise EvidenceError("optical-domain receipt has unexpected top-level fields")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise EvidenceError("optical-domain receipt schema_version is not 1.0.0")
    if type(receipt.get("receipt_id")) is not str or not receipt["receipt_id"].strip():
        raise EvidenceError("optical-domain receipt_id must be a non-empty string")

    signed_roles = receipt.get("signed_roles")
    if not isinstance(signed_roles, dict) or set(signed_roles) != set(ROLE_MEMBERS):
        raise EvidenceError("optical-domain receipt must contain exactly hardware/mechanical/operations")
    signer_identities: list[str] = []
    approval_hashes: list[str] = []
    for role, member in ROLE_MEMBERS.items():
        entry = signed_roles.get(role)
        if not isinstance(entry, dict) or set(entry) != ROLE_ENTRY_FIELDS:
            raise EvidenceError(f"receipt role {role} has an invalid shape")
        signer = _canonical_signer_identity(entry.get("signer"))
        approval_hash = entry.get("approval_evidence_sha256")
        if (
            entry.get("member") != member
            or entry.get("signed") is not True
            or signer is None
            or type(approval_hash) is not str
            or HEX64_RE.fullmatch(approval_hash) is None
        ):
            raise EvidenceError(f"receipt role {role} is not a valid explicit human declaration")
        signer_identities.append(signer)
        approval_hashes.append(approval_hash)
    if len(set(signer_identities)) != len(ROLE_MEMBERS):
        raise EvidenceError("B/C/D signer identities are not distinct after normalization")
    if len(set(approval_hashes)) != len(ROLE_MEMBERS):
        raise EvidenceError("B/C/D approval evidence hashes are not distinct")

    roots = receipt.get("evidence_roots")
    if not isinstance(roots, dict) or set(roots) != set(EVIDENCE_KINDS):
        raise EvidenceError("receipt must contain exactly the five frozen evidence roots")
    for kind in EVIDENCE_KINDS:
        value = roots.get(kind)
        if type(value) is not str or HEX64_RE.fullmatch(value) is None:
            raise EvidenceError(f"receipt evidence root is invalid: {kind}")
    return receipt, _sha256_bytes(_canonical_bytes(receipt))


def _nonclaims() -> dict[str, bool]:
    return {
        "physical_truth_verified": False,
        "data_locked": False,
        "training_authorized": False,
        "print_authorized": False,
        "cryptographic_human_signatures_verified": False,
    }


def _layout_names(directory: Path) -> set[str]:
    return {entry.name for entry in directory.iterdir()}


def preflight_bundle(bundle_dir: Path, receipt_path: Path | None = None) -> dict[str, Any]:
    """Verify bundle byte bindings while preserving explicit non-claims."""

    errors: list[dict[str, Any]] = []

    def add_error(code: str, message: str, **context: Any) -> None:
        errors.append({"code": code, "message": message, **context})

    requested_bundle = Path(bundle_dir)
    report: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "FINAL_OPTICS_NOT_READY",
        "bundle_dir": str(requested_bundle.resolve(strict=False)),
        "evidence": {kind: {"status": "NOT_VERIFIED", "evidence_root": None} for kind in EVIDENCE_KINDS},
        "approvals": {role: {"status": "NOT_VERIFIED", "sha256": None} for role in ROLE_MEMBERS},
        "receipt": {"status": "NOT_VERIFIED", "canonical_sha256": None},
        "optical_domain_root": None,
        "byte_integrity_verified": False,
        "claims": _nonclaims(),
        "error_count": 0,
        "errors": errors,
    }
    try:
        bundle = _ensure_real_directory(requested_bundle, "final-optics bundle")
    except EvidenceError as exc:
        add_error("BUNDLE_MISSING_OR_UNSAFE", str(exc))
        report["error_count"] = len(errors)
        return report

    evidence_dir = bundle / "evidence"
    manifests_dir = bundle / "manifests"
    approvals_dir = bundle / "approvals"
    expected_layouts = (
        (evidence_dir, set(EVIDENCE_KINDS), "EVIDENCE_LAYOUT_INVALID"),
        (manifests_dir, {f"{kind}.manifest.json" for kind in EVIDENCE_KINDS}, "MANIFEST_LAYOUT_INVALID"),
        (approvals_dir, {f"{role}.approval" for role in ROLE_MEMBERS}, "APPROVAL_LAYOUT_INVALID"),
    )
    for directory, expected_names, code in expected_layouts:
        try:
            resolved = _ensure_real_directory(directory, directory.name)
            actual_names = _layout_names(resolved)
            if actual_names != expected_names:
                add_error(
                    code,
                    f"{directory.name} must contain exactly the frozen entries",
                    expected=sorted(expected_names),
                    actual=sorted(actual_names),
                )
        except (EvidenceError, OSError) as exc:
            add_error(code, str(exc))

    computed_roots: dict[str, str] = {}
    for kind in EVIDENCE_KINDS:
        try:
            result = verify_package_manifest(
                evidence_dir / kind,
                kind,
                manifests_dir / f"{kind}.manifest.json",
            )
            computed_roots[kind] = result["evidence_root"]
            report["evidence"][kind] = {"status": "BYTE_INTEGRITY_PASS", **result}
        except EvidenceError as exc:
            add_error("EVIDENCE_PACKAGE_INVALID", str(exc), evidence_kind=kind)
            report["evidence"][kind]["error"] = str(exc)

    actual_approval_hashes: dict[str, str] = {}
    for role in ROLE_MEMBERS:
        path = approvals_dir / f"{role}.approval"
        try:
            digest, byte_count, _ = _hash_regular_file(path, f"{role} approval evidence")
            if byte_count == 0:
                raise EvidenceError(f"{role} approval evidence is empty: {path}")
            actual_approval_hashes[role] = digest
            report["approvals"][role] = {
                "status": "RAW_BYTES_HASHED",
                "sha256": digest,
                "bytes": byte_count,
            }
        except EvidenceError as exc:
            add_error("APPROVAL_EVIDENCE_INVALID", str(exc), role=role)
            report["approvals"][role]["error"] = str(exc)
    if len(actual_approval_hashes) == len(ROLE_MEMBERS) and len(set(actual_approval_hashes.values())) != len(ROLE_MEMBERS):
        add_error("APPROVAL_EVIDENCE_NOT_DISTINCT", "B/C/D approval files do not have distinct byte hashes")

    chosen_receipt = Path(receipt_path) if receipt_path is not None else bundle / "final_optics_receipt.json"
    receipt: dict[str, Any] | None = None
    receipt_root: str | None = None
    receipt_inside_payload = False
    try:
        chosen_receipt.resolve(strict=False).relative_to(evidence_dir.resolve(strict=False))
        receipt_inside_payload = True
    except ValueError:
        pass
    if receipt_inside_payload:
        message = "optical-domain receipt must not be stored inside an evidence payload"
        add_error("RECEIPT_LOCATION_INVALID", message)
        report["receipt"]["error"] = message
    else:
        try:
            receipt, receipt_root = _validate_receipt(chosen_receipt)
            report["receipt"] = {
                "status": "STRUCTURE_PASS_HUMAN_DECLARATIONS_ONLY",
                "canonical_sha256": receipt_root,
                "path": str(chosen_receipt.resolve(strict=True)),
            }
        except EvidenceError as exc:
            add_error("RECEIPT_INVALID", str(exc))
            report["receipt"]["error"] = str(exc)

    if receipt is not None:
        for kind in EVIDENCE_KINDS:
            actual = computed_roots.get(kind)
            expected = receipt["evidence_roots"][kind]
            if actual is not None and actual != expected:
                add_error(
                    "RECEIPT_EVIDENCE_ROOT_MISMATCH",
                    "receipt evidence root does not match the rebuilt package root",
                    evidence_kind=kind,
                    expected=expected,
                    actual=actual,
                )
        for role in ROLE_MEMBERS:
            actual = actual_approval_hashes.get(role)
            expected = receipt["signed_roles"][role]["approval_evidence_sha256"]
            if actual is not None and actual != expected:
                add_error(
                    "RECEIPT_APPROVAL_HASH_MISMATCH",
                    "receipt approval hash does not match the actual approval file",
                    role=role,
                    expected=expected,
                    actual=actual,
                )

    if not errors and receipt_root is not None:
        report["status"] = "STRUCTURAL_PREFLIGHT_PASS_HUMAN_PHYSICAL_ATTESTATION_ONLY"
        report["byte_integrity_verified"] = True
        report["optical_domain_root"] = receipt_root
    report["error_count"] = len(errors)
    return report


def _atomic_replace(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _output_is_inside_evidence(output: Path, bundle: Path) -> bool:
    candidate = Path(output).resolve(strict=False)
    evidence = (Path(bundle).resolve(strict=False) / "evidence").resolve(strict=False)
    try:
        candidate.relative_to(evidence)
        return True
    except ValueError:
        return False


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-package", help="Create one immutable canonical package manifest.")
    build_parser.add_argument("--kind", choices=EVIDENCE_KINDS, required=True)
    build_parser.add_argument("--payload-dir", type=Path, required=True)
    build_parser.add_argument("--manifest", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify-package", help="Rebuild and verify one package manifest.")
    verify_parser.add_argument("--kind", choices=EVIDENCE_KINDS, required=True)
    verify_parser.add_argument("--payload-dir", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)

    preflight_parser = subparsers.add_parser("preflight", help="Verify the complete five-package bundle and receipt.")
    preflight_parser.add_argument("--bundle-dir", type=Path, required=True)
    preflight_parser.add_argument("--receipt", type=Path, default=None)
    preflight_parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "build-package":
            result = write_package_manifest(args.payload_dir, args.kind, args.manifest)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "verify-package":
            result = verify_package_manifest(args.payload_dir, args.kind, args.manifest)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0

        report = preflight_bundle(args.bundle_dir, args.receipt)
        if args.output is not None:
            if _output_is_inside_evidence(args.output, args.bundle_dir):
                raise EvidenceError("preflight report must not be written inside an evidence payload")
            _atomic_replace(args.output, _pretty_json_bytes(report))
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0 if report["status"] == "STRUCTURAL_PREFLIGHT_PASS_HUMAN_PHYSICAL_ATTESTATION_ONLY" else 2
    except (EvidenceError, OSError) as exc:
        print(f"final-optics evidence error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
