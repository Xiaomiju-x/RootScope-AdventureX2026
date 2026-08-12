#!/usr/bin/env python3
"""Exact, read-only RootMind GGUF page-cache release helper.

The helper deliberately has no service, device, network, or physical execution
authority.  ``bind`` pins one manifest-owned GGUF to a candidate identity.
``release`` revalidates that binding and asks Linux to evict only that open
file's clean page-cache pages with POSIX_FADV_DONTNEED.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence


BINDING_SCHEMA = "rootscope.v3.rootmind-gguf-cache-binding.v1"
RECEIPT_SCHEMA = "rootscope.v3.rootmind-gguf-cache-release.v1"
MANIFEST_SCHEMA = "rootscope.v3.candidate-manifest.v1"
ROLE_CATEGORY = {
    "fast": "ROOTMIND_FAST_MODEL",
    "deep": "ROOTMIND_DEEP_MODEL",
}
ROLE_DIRECTORY = {
    "fast": "models/llm/fast",
    "deep": "models/llm/deep",
}
ENDPOINT_HOST = "127.0.0.1"
ENDPOINT_PORT = 9080
CMA_FREE_MINIMUM_KIB = 131_072
RESIDENT_LIMIT_BYTES = 4_096
SAMPLE_INTERVAL_SECONDS = 0.25

ZERO_AUTHORITY = {
    "execution_authority": False,
    "physical_authority": False,
    "external_network": False,
    "service_started": False,
    "serial_opened": False,
    "serial_write": False,
    "gpio_touched": False,
    "pump_command": False,
    "state_machine_write": False,
    "model_modified": False,
}
CANDIDATE_ZERO_AUTHORITY = {
    "execution_authority": False,
    "external_network": False,
    "gpio_write": False,
    "physical_completion": False,
    "pump_command": False,
    "serial_write": False,
    "state_machine_write": False,
}


class GateError(RuntimeError):
    """A fail-closed binding or cache-release gate."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant forbidden: {value}")


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key forbidden: {key}")
        output[key] = value
    return output


def _load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise GateError("INVALID_JSON", f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise GateError("INVALID_JSON", f"{label} must be a JSON object")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stat_fingerprint(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "nlink": int(value.st_nlink),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _open_nofollow_regular(path: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    if sys.platform.startswith("linux"):
        try:
            nofollow = os.O_NOFOLLOW
        except AttributeError as exc:
            raise GateError("NO_NOFOLLOW", "Linux O_NOFOLLOW is unavailable") from exc
    else:
        nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        # This branch exists for mocked Windows unit tests.  Production X5
        # execution is Linux and must supply O_NOFOLLOW.
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            raise GateError("SYMLINK_REJECTED", f"symlink forbidden: {path}")
        fd = os.open(path, flags)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            os.close(fd)
            raise GateError("PATH_RACE", f"path identity changed while opening: {path}")
    else:
        try:
            fd = os.open(path, flags | nofollow)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EMLINK}:
                raise GateError("SYMLINK_REJECTED", f"symlink forbidden: {path}") from exc
            raise
    opened = os.fstat(fd)
    if not stat.S_ISREG(opened.st_mode):
        os.close(fd)
        raise GateError("NOT_REGULAR", f"regular file required: {path}")
    return fd


def _read_exact_open_file(
    path: Path, *, hash_content: bool = True
) -> tuple[bytes, dict[str, int], str | None]:
    try:
        fd = _open_nofollow_regular(path)
    except (OSError, GateError) as exc:
        if isinstance(exc, GateError):
            raise
        raise GateError("OPEN_FAILED", f"unable to open {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        chunks: list[bytes] = []
        digest = hashlib.sha256() if hash_content else None
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            if digest is not None:
                digest.update(block)
        after = os.fstat(fd)
        if _stat_fingerprint(before) != _stat_fingerprint(after):
            raise GateError("FILE_CHANGED_DURING_READ", f"file changed while reading: {path}")
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            raise GateError("SHORT_READ", f"short read from {path}")
        return raw, _stat_fingerprint(before), digest.hexdigest() if digest else None
    finally:
        os.close(fd)


def _hash_open_fd(fd: int) -> tuple[str, dict[str, int]]:
    before = os.fstat(fd)
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    after = os.fstat(fd)
    if _stat_fingerprint(before) != _stat_fingerprint(after):
        raise GateError("MODEL_CHANGED_DURING_HASH", "model changed while hashing")
    return digest.hexdigest(), _stat_fingerprint(before)


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(os.path.abspath(os.fspath(path)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise GateError("OUTPUT_EXISTS", f"refusing to replace existing output: {destination}")
    payload = _canonical_json(value)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    fd: int | None = None
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        os.close(fd)
        fd = None
        # A hard link gives no-clobber atomic publication on the same filesystem.
        os.link(temporary, destination)
        os.unlink(temporary)
    except FileExistsError as exc:
        raise GateError(
            "OUTPUT_EXISTS", f"refusing to replace existing output: {destination}"
        ) from exc
    except OSError as exc:
        raise GateError("OUTPUT_WRITE_FAILED", f"unable to create {destination}: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)
        try:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
        except OSError:
            pass


def _canonical_release_root(raw: str | os.PathLike[str]) -> Path:
    supplied = Path(raw)
    try:
        root = supplied.resolve(strict=True)
    except OSError as exc:
        raise GateError("RELEASE_ROOT_INVALID", f"release root unavailable: {supplied}") from exc
    if not root.is_dir():
        raise GateError("RELEASE_ROOT_INVALID", f"release root is not a directory: {root}")
    return root


def _safe_manifest_relative(raw: Any) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise GateError("MANIFEST_RECORD_INVALID", "manifest model path is invalid")
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise GateError("MANIFEST_RECORD_INVALID", "manifest model path is not canonical")
    normalized = candidate.as_posix()
    if normalized != raw:
        raise GateError("MANIFEST_RECORD_INVALID", "manifest model path is not canonical")
    return normalized


def _validate_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise GateError("MANIFEST_RECORD_INVALID", f"{label} must be lowercase SHA-256")
    return value


def _manifest_and_role_model(
    release_root: Path, role: str
) -> tuple[dict[str, Any], bytes, str, dict[str, Any], Path]:
    manifest_path = release_root / "candidate_manifest.json"
    raw, _, manifest_sha = _read_exact_open_file(manifest_path)
    assert manifest_sha is not None
    manifest = _load_json_bytes(raw, "candidate manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise GateError("MANIFEST_SCHEMA_CHANGED", "candidate manifest schema changed")
    if manifest.get("candidate_id") != release_root.name:
        raise GateError("CANDIDATE_ID_CHANGED", "candidate manifest identity changed")
    authority = manifest.get("authority")
    if authority != CANDIDATE_ZERO_AUTHORITY:
        raise GateError("CANDIDATE_AUTHORITY", "candidate manifest is not zero-authority")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise GateError("MANIFEST_RECORD_INVALID", "candidate manifest files must be a list")
    category = ROLE_CATEGORY[role]
    matches = [
        item
        for item in files
        if isinstance(item, dict) and item.get("category") == category
    ]
    if len(matches) != 1:
        raise GateError(
            "ROLE_MODEL_NOT_UNIQUE",
            f"manifest must contain exactly one {role} RootMind GGUF",
        )
    record = matches[0]
    relative = _safe_manifest_relative(record.get("path"))
    expected_prefix = ROLE_DIRECTORY[role] + "/"
    if not relative.startswith(expected_prefix) or not relative.endswith(".gguf"):
        raise GateError("MANIFEST_RECORD_INVALID", "role GGUF path is outside its role")
    if type(record.get("bytes")) is not int or record["bytes"] <= 0:
        raise GateError("MANIFEST_RECORD_INVALID", "manifest model bytes are invalid")
    _validate_sha(record.get("sha256"), "manifest model hash")
    role_dir = release_root / ROLE_DIRECTORY[role]
    try:
        disk_ggufs = sorted(
            entry.name
            for entry in os.scandir(role_dir)
            if entry.name.endswith(".gguf")
        )
    except OSError as exc:
        raise GateError("ROLE_DIRECTORY_INVALID", f"unable to inspect {role_dir}") from exc
    if disk_ggufs != [Path(relative).name]:
        raise GateError(
            "ROLE_MODEL_NOT_UNIQUE",
            f"role directory must contain exactly manifest GGUF, observed={disk_ggufs}",
        )
    model_path = release_root / Path(relative)
    return manifest, raw, manifest_sha, record, model_path


def _exact_model_identity(
    model_path: Path, record: Mapping[str, Any]
) -> tuple[dict[str, int], str]:
    try:
        fd = _open_nofollow_regular(model_path)
    except (OSError, GateError) as exc:
        if isinstance(exc, GateError):
            raise
        raise GateError("MODEL_OPEN_FAILED", f"unable to open role model: {exc}") from exc
    try:
        digest, fingerprint = _hash_open_fd(fd)
    finally:
        os.close(fd)
    if fingerprint["size"] != record["bytes"]:
        raise GateError("MODEL_BYTES_MISMATCH", "model bytes differ from candidate manifest")
    if digest != record["sha256"]:
        raise GateError("MODEL_HASH_MISMATCH", "model SHA-256 differs from candidate manifest")
    return fingerprint, digest


def build_binding(
    *,
    release_root_raw: str | os.PathLike[str],
    role: str,
    model_raw: str | os.PathLike[str],
) -> dict[str, Any]:
    release_root = _canonical_release_root(release_root_raw)
    manifest, _, manifest_sha, record, manifest_model_path = _manifest_and_role_model(
        release_root, role
    )
    supplied_model = Path(os.path.abspath(os.fspath(model_raw)))
    expected_model = Path(os.path.abspath(os.fspath(manifest_model_path)))
    if supplied_model != expected_model:
        raise GateError(
            "MODEL_PATH_MISMATCH",
            "explicit model path is not the unique role model in candidate manifest",
        )
    fingerprint, digest = _exact_model_identity(expected_model, record)
    return {
        "schema": BINDING_SCHEMA,
        "status": "BOUND",
        "created_utc": _utc_now(),
        "candidate": {
            "id": manifest["candidate_id"],
            "release_root": str(release_root),
            "manifest_path": str(release_root / "candidate_manifest.json"),
            "manifest_sha256": manifest_sha,
        },
        "role": role,
        "model": {
            "path": str(expected_model),
            "relative_path": record["path"],
            "category": record["category"],
            "bytes": record["bytes"],
            "sha256": digest,
            "stat_fingerprint": fingerprint,
        },
        "integrity": {
            "manifest_record_count": len(manifest["files"]),
            "unique_role_gguf": True,
            "content_sha256_verified": True,
            "regular_file": True,
            "nofollow_open": True,
        },
        "authority": dict(ZERO_AUTHORITY),
    }


def _strict_binding(value: Mapping[str, Any]) -> None:
    expected_top = {
        "schema",
        "status",
        "created_utc",
        "candidate",
        "role",
        "model",
        "integrity",
        "authority",
    }
    if set(value) != expected_top:
        raise GateError("BINDING_SCHEMA_CHANGED", "binding top-level fields changed")
    if value.get("schema") != BINDING_SCHEMA or value.get("status") != "BOUND":
        raise GateError("BINDING_SCHEMA_CHANGED", "binding schema or status changed")
    if value.get("role") not in ROLE_CATEGORY:
        raise GateError("BINDING_SCHEMA_CHANGED", "binding role is invalid")
    candidate = value.get("candidate")
    model = value.get("model")
    integrity = value.get("integrity")
    if not isinstance(candidate, dict) or set(candidate) != {
        "id",
        "release_root",
        "manifest_path",
        "manifest_sha256",
    }:
        raise GateError("BINDING_SCHEMA_CHANGED", "binding candidate fields changed")
    if not isinstance(model, dict) or set(model) != {
        "path",
        "relative_path",
        "category",
        "bytes",
        "sha256",
        "stat_fingerprint",
    }:
        raise GateError("BINDING_SCHEMA_CHANGED", "binding model fields changed")
    fingerprint = model.get("stat_fingerprint")
    if not isinstance(fingerprint, dict) or set(fingerprint) != {
        "device",
        "inode",
        "mode",
        "nlink",
        "uid",
        "gid",
        "size",
        "mtime_ns",
        "ctime_ns",
    }:
        raise GateError("BINDING_SCHEMA_CHANGED", "binding stat fingerprint changed")
    if any(type(item) is not int for item in fingerprint.values()):
        raise GateError("BINDING_SCHEMA_CHANGED", "binding stat values must be integers")
    if (
        not isinstance(integrity, dict)
        or integrity
        != {
            "manifest_record_count": integrity.get("manifest_record_count"),
            "unique_role_gguf": True,
            "content_sha256_verified": True,
            "regular_file": True,
            "nofollow_open": True,
        }
        or type(integrity.get("manifest_record_count")) is not int
    ):
        raise GateError("BINDING_SCHEMA_CHANGED", "binding integrity fields changed")
    if value.get("authority") != ZERO_AUTHORITY:
        raise GateError("BINDING_AUTHORITY_CHANGED", "binding is not zero-authority")
    if candidate.get("id") != Path(str(candidate.get("release_root"))).name:
        raise GateError("BINDING_SCHEMA_CHANGED", "binding candidate identity is inconsistent")
    _validate_sha(candidate.get("manifest_sha256"), "binding manifest hash")
    _validate_sha(model.get("sha256"), "binding model hash")
    if type(model.get("bytes")) is not int or model["bytes"] <= 0:
        raise GateError("BINDING_SCHEMA_CHANGED", "binding model bytes are invalid")


def _scan_llama_server_processes() -> list[dict[str, Any]]:
    observed: list[dict[str, Any]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise GateError("PROC_UNAVAILABLE", "/proc is required for residual process gate")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        tokens = [
            item.decode("utf-8", errors="replace")
            for item in raw.split(b"\0")
            if item
        ]
        if any(Path(token).name == "llama-server" for token in tokens):
            observed.append({"pid": int(entry.name), "cmdline": tokens})
    return sorted(observed, key=lambda item: item["pid"])


def _loopback_port_closed() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.25)
        result = probe.connect_ex((ENDPOINT_HOST, ENDPOINT_PORT))
        return result != 0
    finally:
        probe.close()


def _memory_sample() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise GateError("MEMINFO_UNAVAILABLE", "unable to read /proc/meminfo") from exc
    expected = {
        "MemAvailable": "mem_available_kib",
        "CmaFree": "cma_free_kib",
        "Cached": "cached_kib",
    }
    for line in lines:
        key, separator, remainder = line.partition(":")
        if not separator or key not in expected:
            continue
        fields = remainder.split()
        if len(fields) != 2 or fields[1] != "kB" or not fields[0].isdigit():
            raise GateError("MEMINFO_INVALID", f"unexpected /proc/meminfo line: {line}")
        values[expected[key]] = int(fields[0])
    if set(values) != set(expected.values()):
        raise GateError("MEMINFO_INVALID", "required memory counters are unavailable")
    return values


def _resident_bytes(fd: int, size: int) -> int:
    if size <= 0:
        return 0
    if not sys.platform.startswith("linux"):
        raise GateError("MINCORE_UNAVAILABLE", "resident-page evidence requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    mmap_fn = libc.mmap
    mmap_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong,
    ]
    mmap_fn.restype = ctypes.c_void_p
    mincore_fn = libc.mincore
    mincore_fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    mincore_fn.restype = ctypes.c_int
    munmap_fn = libc.munmap
    munmap_fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    munmap_fn.restype = ctypes.c_int
    prot_read = 0x1
    map_shared = 0x01
    address = mmap_fn(None, size, prot_read, map_shared, fd, 0)
    map_failed = ctypes.c_void_p(-1).value
    if address in {None, map_failed}:
        current_errno = ctypes.get_errno()
        raise GateError(
            "MINCORE_MAP_FAILED", f"read-only model mapping failed: errno={current_errno}"
        )
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = (size + page_size - 1) // page_size
    vector = (ctypes.c_ubyte * pages)()
    try:
        if mincore_fn(address, size, vector) != 0:
            current_errno = ctypes.get_errno()
            raise GateError(
                "MINCORE_FAILED", f"resident-page query failed: errno={current_errno}"
            )
        resident_pages = sum(1 for value in vector if value & 1)
        return min(size, resident_pages * page_size)
    finally:
        munmap_fn(address, size)


def _apply_exact_fadvise(fd: int) -> None:
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or dontneed is None:
        raise GateError("FADVISE_UNAVAILABLE", "POSIX_FADV_DONTNEED is unavailable")
    try:
        advise(fd, 0, 0, dontneed)
    except OSError as exc:
        raise GateError("FADVISE_FAILED", f"exact-file fadvise failed: {exc}") from exc


def _empty_release_receipt(binding_sha: str | None = None) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "FAIL",
        "created_utc": _utc_now(),
        "binding_sha256": binding_sha,
        "candidate": None,
        "role": None,
        "model": None,
        "integrity": {
            "binding_valid": False,
            "release_root_unchanged": False,
            "manifest_path_unchanged": False,
            "manifest_sha256_unchanged": False,
            "manifest_record_unchanged": False,
            "model_path_unchanged": False,
            "model_stat_unchanged": False,
            "model_sha256_verified": False,
            "model_stat_unchanged_after": False,
            "model_modified": False,
        },
        "preconditions": {
            "llama_server_processes": None,
            "no_llama_server": False,
            "endpoint": f"{ENDPOINT_HOST}:{ENDPOINT_PORT}",
            "port_closed": False,
        },
        "cache": {
            "method": "POSIX_FADV_DONTNEED",
            "exact_file_only": True,
            "global_drop_caches": False,
            "sync_called": False,
            "compact_memory_called": False,
            "fadvise_applied": False,
            "resident_bytes_before": None,
            "resident_bytes_after": None,
            "resident_limit_bytes": RESIDENT_LIMIT_BYTES,
            "window_reached": False,
        },
        "memory": {
            "before": None,
            "after": None,
            "samples": [],
            "observe_seconds": None,
            "cma_free_minimum_kib": CMA_FREE_MINIMUM_KIB,
            "window_reached": False,
        },
        "authority": dict(ZERO_AUTHORITY),
        "error": None,
    }


def _validate_release_receipt(receipt: Mapping[str, Any]) -> None:
    expected_top = {
        "schema",
        "status",
        "created_utc",
        "binding_sha256",
        "candidate",
        "role",
        "model",
        "integrity",
        "preconditions",
        "cache",
        "memory",
        "authority",
        "error",
    }
    if set(receipt) != expected_top:
        raise GateError("RECEIPT_SCHEMA_CHANGED", "release receipt fields changed")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") not in {
        "PASS",
        "FAIL",
    }:
        raise GateError("RECEIPT_SCHEMA_CHANGED", "release receipt schema/status changed")
    expected_cache_keys = {
        "method",
        "exact_file_only",
        "global_drop_caches",
        "sync_called",
        "compact_memory_called",
        "fadvise_applied",
        "resident_bytes_before",
        "resident_bytes_after",
        "resident_limit_bytes",
        "window_reached",
    }
    cache = receipt.get("cache")
    if not isinstance(cache, dict) or set(cache) != expected_cache_keys:
        raise GateError("RECEIPT_SCHEMA_CHANGED", "release cache fields changed")
    policy = {
        "method": "POSIX_FADV_DONTNEED",
        "exact_file_only": True,
        "global_drop_caches": False,
        "sync_called": False,
        "compact_memory_called": False,
        "resident_limit_bytes": RESIDENT_LIMIT_BYTES,
    }
    if any(cache.get(key) != value for key, value in policy.items()):
        raise GateError("GLOBAL_CACHE_POLICY_FORBIDDEN", "exact-file cache policy changed")
    if receipt.get("authority") != ZERO_AUTHORITY:
        raise GateError("RECEIPT_AUTHORITY_CHANGED", "release receipt is not zero-authority")
    if receipt["status"] == "PASS":
        if receipt.get("error") is not None:
            raise GateError("RECEIPT_SCHEMA_CHANGED", "PASS receipt contains an error")
        if (
            cache.get("fadvise_applied") is not True
            or cache.get("window_reached") is not True
            or type(cache.get("resident_bytes_before")) is not int
            or type(cache.get("resident_bytes_after")) is not int
            or cache["resident_bytes_after"] > RESIDENT_LIMIT_BYTES
        ):
            raise GateError("RECEIPT_GATE_INCONSISTENT", "PASS cache evidence is inconsistent")
    elif not isinstance(receipt.get("error"), dict):
        raise GateError("RECEIPT_SCHEMA_CHANGED", "FAIL receipt must contain an error object")


def execute_release(
    *,
    release_root_raw: str | os.PathLike[str],
    role: str,
    binding_path: Path,
    observe_seconds: float,
    process_scanner: Callable[[], list[dict[str, Any]]] = _scan_llama_server_processes,
    port_checker: Callable[[], bool] = _loopback_port_closed,
    memory_sampler: Callable[[], dict[str, int]] = _memory_sample,
    resident_sampler: Callable[[int, int], int] = _resident_bytes,
    fadvise: Callable[[int], None] = _apply_exact_fadvise,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if not math.isfinite(observe_seconds) or not 0 <= observe_seconds <= 60:
        raise GateError(
            "OBSERVE_SECONDS_INVALID", "observe-seconds must be finite and within 0..60"
        )
    binding_raw, _, _ = _read_exact_open_file(binding_path)
    binding_sha = _sha256_bytes(binding_raw)
    receipt = _empty_release_receipt(binding_sha)
    binding = _load_json_bytes(binding_raw, "binding")
    _strict_binding(binding)
    receipt["integrity"]["binding_valid"] = True
    receipt["candidate"] = binding["candidate"]
    receipt["role"] = binding["role"]
    receipt["model"] = binding["model"]
    receipt["memory"]["observe_seconds"] = float(observe_seconds)
    if binding["role"] != role:
        raise GateError("ROLE_CHANGED", "requested role differs from binding")
    release_root = _canonical_release_root(release_root_raw)
    if str(release_root) != binding["candidate"]["release_root"]:
        raise GateError("RELEASE_ROOT_CHANGED", "release root differs from binding")
    receipt["integrity"]["release_root_unchanged"] = True
    expected_manifest = release_root / "candidate_manifest.json"
    if str(expected_manifest) != binding["candidate"]["manifest_path"]:
        raise GateError("MANIFEST_PATH_CHANGED", "manifest path differs from binding")
    receipt["integrity"]["manifest_path_unchanged"] = True
    manifest, _, manifest_sha, record, model_path = _manifest_and_role_model(
        release_root, role
    )
    if manifest_sha != binding["candidate"]["manifest_sha256"]:
        raise GateError("MANIFEST_CHANGED", "candidate manifest SHA-256 changed")
    receipt["integrity"]["manifest_sha256_unchanged"] = True
    if manifest["candidate_id"] != binding["candidate"]["id"]:
        raise GateError("CANDIDATE_ID_CHANGED", "candidate id differs from binding")
    record_projection = {
        "path": record["path"],
        "category": record["category"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
    }
    binding_projection = {
        key: binding["model"][key]
        for key in ("relative_path", "category", "bytes", "sha256")
    }
    binding_projection["path"] = binding_projection.pop("relative_path")
    if record_projection != binding_projection:
        raise GateError("MANIFEST_RECORD_CHANGED", "manifest role record differs from binding")
    receipt["integrity"]["manifest_record_unchanged"] = True
    expected_model_path = Path(os.path.abspath(os.fspath(model_path)))
    if str(expected_model_path) != binding["model"]["path"]:
        raise GateError("MODEL_PATH_CHANGED", "model path differs from binding")
    receipt["integrity"]["model_path_unchanged"] = True
    processes = process_scanner()
    receipt["preconditions"]["llama_server_processes"] = processes
    if processes:
        raise GateError("LLAMA_SERVER_RESIDUAL", "llama-server process remains")
    receipt["preconditions"]["no_llama_server"] = True
    if not port_checker():
        raise GateError("PORT_STILL_OPEN", "127.0.0.1:9080 remains open")
    receipt["preconditions"]["port_closed"] = True
    try:
        fd = _open_nofollow_regular(expected_model_path)
    except (OSError, GateError) as exc:
        if isinstance(exc, GateError):
            raise
        raise GateError("MODEL_OPEN_FAILED", f"unable to open bound model: {exc}") from exc
    try:
        model_sha, fingerprint_before = _hash_open_fd(fd)
        if fingerprint_before != binding["model"]["stat_fingerprint"]:
            raise GateError("MODEL_STAT_CHANGED", "bound model stat fingerprint changed")
        receipt["integrity"]["model_stat_unchanged"] = True
        if (
            model_sha != binding["model"]["sha256"]
            or model_sha != record["sha256"]
            or fingerprint_before["size"] != binding["model"]["bytes"]
        ):
            raise GateError("MODEL_HASH_CHANGED", "bound model content changed")
        receipt["integrity"]["model_sha256_verified"] = True
        before_memory = memory_sampler()
        before_resident = resident_sampler(fd, fingerprint_before["size"])
        receipt["memory"]["before"] = before_memory
        receipt["cache"]["resident_bytes_before"] = before_resident
        fadvise(fd)
        receipt["cache"]["fadvise_applied"] = True
        started = monotonic()
        samples: list[dict[str, Any]] = []
        sleeper(min(SAMPLE_INTERVAL_SECONDS, observe_seconds))
        while True:
            elapsed = max(0.0, monotonic() - started)
            current_memory = memory_sampler()
            current_resident = resident_sampler(fd, fingerprint_before["size"])
            gate_pass = (
                current_memory["cma_free_kib"] >= CMA_FREE_MINIMUM_KIB
                and current_resident <= RESIDENT_LIMIT_BYTES
            )
            samples.append(
                {
                    "elapsed_ms": int(round(elapsed * 1000)),
                    **current_memory,
                    "resident_bytes": current_resident,
                    "gate_pass": gate_pass,
                }
            )
            if elapsed >= observe_seconds and len(samples) >= 2:
                break
            if elapsed < observe_seconds:
                sleeper(min(SAMPLE_INTERVAL_SECONDS, observe_seconds - elapsed))
        after = samples[-1]
        receipt["memory"]["samples"] = samples
        receipt["memory"]["after"] = {
            key: after[key]
            for key in ("mem_available_kib", "cma_free_kib", "cached_kib")
        }
        receipt["cache"]["resident_bytes_after"] = after["resident_bytes"]
        window_reached = len(samples) >= 2 and all(
            sample["gate_pass"] for sample in samples
        )
        receipt["cache"]["window_reached"] = window_reached
        receipt["memory"]["window_reached"] = window_reached
        fingerprint_after = _stat_fingerprint(os.fstat(fd))
        if fingerprint_after != fingerprint_before:
            receipt["integrity"]["model_modified"] = True
            receipt["authority"]["model_modified"] = True
            raise GateError("MODEL_STAT_CHANGED_AFTER", "model changed during cache release")
        receipt["integrity"]["model_stat_unchanged_after"] = True
        terminal_gate = (
            after["cma_free_kib"] >= CMA_FREE_MINIMUM_KIB
            and after["resident_bytes"] <= RESIDENT_LIMIT_BYTES
        )
        if not window_reached or not terminal_gate:
            raise GateError(
                "CACHE_RESOURCE_GATE_FAILED",
                "exact model cache/CMA gate was not met at the end of observation",
            )
    finally:
        os.close(fd)
    receipt["status"] = "PASS"
    receipt["error"] = None
    _validate_release_receipt(receipt)
    return receipt


def _run_bind(args: argparse.Namespace) -> int:
    try:
        binding = build_binding(
            release_root_raw=args.release_root,
            role=args.role,
            model_raw=args.model,
        )
        _atomic_create_json(Path(args.output), binding)
    except GateError as exc:
        print(f"bind failed [{exc.code}]: {exc}", file=sys.stderr)
        return 31 if exc.code.startswith("OUTPUT_") else 30
    print(json.dumps(binding, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


def _run_release(args: argparse.Namespace) -> int:
    binding_sha: str | None = None
    try:
        try:
            binding_raw, _, _ = _read_exact_open_file(Path(args.binding))
            binding_sha = _sha256_bytes(binding_raw)
        except GateError:
            pass
        receipt = execute_release(
            release_root_raw=args.release_root,
            role=args.role,
            binding_path=Path(args.binding),
            observe_seconds=args.observe_seconds,
        )
    except GateError as exc:
        receipt = _empty_release_receipt(binding_sha)
        receipt["role"] = args.role
        receipt["memory"]["observe_seconds"] = float(args.observe_seconds)
        receipt["error"] = {"code": exc.code, "message": str(exc)}
        try:
            # If validation reached far enough, execute_release's partial receipt
            # is intentionally not guessed here; failure is still explicit.
            _validate_release_receipt(receipt)
            _atomic_create_json(Path(args.output), receipt)
        except GateError as write_exc:
            print(
                f"release failed [{exc.code}] and receipt write failed "
                f"[{write_exc.code}]: {write_exc}",
                file=sys.stderr,
            )
            return 41
        print(json.dumps(receipt, ensure_ascii=True, sort_keys=True, allow_nan=False))
        return 40
    try:
        _validate_release_receipt(receipt)
        _atomic_create_json(Path(args.output), receipt)
    except GateError as exc:
        print(f"release receipt write failed [{exc.code}]: {exc}", file=sys.stderr)
        return 41
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind and release one exact RootMind GGUF page-cache footprint"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    bind = subparsers.add_parser("bind")
    bind.add_argument("--release-root", required=True)
    bind.add_argument("--role", required=True, choices=sorted(ROLE_CATEGORY))
    bind.add_argument("--model", required=True)
    bind.add_argument("--output", required=True)
    bind.set_defaults(handler=_run_bind)
    release = subparsers.add_parser("release")
    release.add_argument("--release-root", required=True)
    release.add_argument("--role", required=True, choices=sorted(ROLE_CATEGORY))
    release.add_argument("--binding", required=True)
    release.add_argument("--output", required=True)
    release.add_argument("--observe-seconds", type=float, default=2.0)
    release.set_defaults(handler=_run_release)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
