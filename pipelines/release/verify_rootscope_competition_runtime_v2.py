#!/usr/bin/env python3
"""Verify one extracted RootScope competition runtime v2 without opening hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
from typing import Any, Mapping, Sequence


EXPECTED_ID = "rootscope_competition_runtime_v2_candidate_20260723"
EXPECTED_IDENTITY = {
    "hostname": "rootscope-x5",
    "serial": "3281556110258c1902ab5d9b0012004",
    "machine_id": "<redacted-device-boot-id>",
    "wlan_mac": "02:00:00:00:00:01",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
IGNORED_RUNTIME_PARTS = frozenset({"__pycache__", ".pytest_cache"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_object(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, Mapping):
        raise ValueError("manifest must be an object")
    return value


def safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return Path(*pure.parts)


def read_identity() -> dict[str, str]:
    serial_path = Path("/proc/device-tree/serial-number")
    if not serial_path.exists():
        serial_path = Path("/sys/firmware/devicetree/base/serial-number")
    return {
        "hostname": platform.node(),
        "serial": serial_path.read_bytes().replace(b"\x00", b"").decode("ascii"),
        "machine_id": Path("/etc/machine-id").read_text(encoding="ascii").strip(),
        "wlan_mac": Path("/sys/class/net/wlan0/address")
        .read_text(encoding="ascii")
        .strip(),
    }


def verify(root: Path) -> dict[str, Any]:
    release = root.resolve(strict=True)
    if release.is_symlink() or not release.is_dir() or release.name != EXPECTED_ID:
        raise ValueError("release root identity/path is invalid")
    manifest_path = release / "candidate_manifest.json"
    manifest = strict_object(manifest_path)
    if (
        manifest.get("schema") != "rootscope.competition-runtime-candidate.v2"
        or manifest.get("candidate_id") != EXPECTED_ID
    ):
        raise ValueError("candidate manifest identity changed")
    authority = manifest.get("authority")
    if not isinstance(authority, Mapping) or not authority or any(
        value is not False for value in authority.values()
    ):
        raise ValueError("candidate authority must remain entirely false")
    qualification = manifest.get("qualification")
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("previous_release_selected_bin_remains_null") is not True
        or qualification.get("r7_passed_previous_probability_drift_gate") is not False
        or qualification.get("production_integration_allowed") is not False
        or qualification.get("physical_closure") is not False
    ):
        raise ValueError("candidate truth boundary changed")
    build_gates = manifest.get("build_gates")
    if not isinstance(build_gates, Mapping):
        raise ValueError("candidate build gates are missing")
    structural_gate = build_gates.get("rag_structural")
    retrieval_gate = build_gates.get("rag_fts5_bm25_retrieval")
    if (
        not isinstance(structural_gate, Mapping)
        or structural_gate.get("status") != "PASS"
        or structural_gate.get("source_count") != 15
        or structural_gate.get("chunk_count") != 24
        or structural_gate.get("gold_qa_count") != 20
        or structural_gate.get("forbidden_qa_count") != 20
        or structural_gate.get("secret_count") != 0
        or structural_gate.get("control_instruction_count") != 0
    ):
        raise ValueError("candidate structural RAG gate changed")
    if (
        not isinstance(retrieval_gate, Mapping)
        or retrieval_gate.get("status") != "PASS"
        or retrieval_gate.get("retrieval_backend") != "SQLITE_FTS5_BM25"
        or retrieval_gate.get("indexed_chunk_count") != 24
        or retrieval_gate.get("gold_expected_citation_top5") != 20
        or (
            retrieval_gate.get("forbidden_expected_citation_top5", 0)
            + retrieval_gate.get("forbidden_direct_guard", 0)
        )
        != 20
        or retrieval_gate.get("zero_authority_response_count") != 40
        or retrieval_gate.get("citation_escape_count") != 0
        or retrieval_gate.get("command_response_count") != 0
    ):
        raise ValueError("candidate FTS5/BM25 RAG gate changed")

    file_records = manifest.get("files")
    if not isinstance(file_records, list) or not file_records:
        raise ValueError("manifest files are missing")
    expected: dict[str, str] = {}
    for record in file_records:
        if not isinstance(record, Mapping):
            raise ValueError("manifest file record must be an object")
        name = record.get("path")
        digest = record.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str) or not SHA_RE.fullmatch(digest):
            raise ValueError("manifest file record is invalid")
        safe_relative(name)
        if name in expected:
            raise ValueError(f"duplicate manifest file: {name}")
        expected[name] = digest
    expected["candidate_manifest.json"] = sha256_file(manifest_path)
    actual_files: dict[str, Path] = {}
    for path in release.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"release symlink is forbidden: {path}")
        relative_parts = path.relative_to(release).parts
        if any(part in IGNORED_RUNTIME_PARTS for part in relative_parts):
            continue
        if path.is_file() and path.name != "SHA256SUMS":
            relative = path.relative_to(release).as_posix()
            actual_files[relative] = path
    if set(actual_files) != set(expected):
        raise ValueError(
            f"manifest coverage mismatch missing={sorted(set(expected)-set(actual_files))} "
            f"extra={sorted(set(actual_files)-set(expected))}"
        )
    for name, digest in expected.items():
        if sha256_file(actual_files[name]) != digest:
            raise ValueError(f"payload SHA-256 mismatch: {name}")

    sum_lines = (release / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    parsed: dict[str, str] = {}
    for line in sum_lines:
        digest, separator, name = line.partition("  ")
        if separator != "  " or not SHA_RE.fullmatch(digest) or name in parsed:
            raise ValueError("SHA256SUMS is malformed")
        safe_relative(name)
        parsed[name] = digest
    if parsed != dict(sorted(expected.items())):
        raise ValueError("SHA256SUMS exact coverage/content mismatch")

    identity = read_identity()
    if identity != EXPECTED_IDENTITY or platform.machine() != "aarch64":
        raise ValueError(
            f"X5 identity mismatch: identity={identity} arch={platform.machine()}"
        )
    external_results: list[dict[str, Any]] = []
    external = manifest.get("external_components")
    if not isinstance(external, list) or not external:
        raise ValueError("external component contracts are missing")
    for item in external:
        if not isinstance(item, Mapping):
            raise ValueError("external component record must be an object")
        configured = item.get("x5_path")
        expected_sha = item.get("sha256")
        if not isinstance(configured, str) or not isinstance(expected_sha, str):
            raise ValueError("external component path/hash is invalid")
        path = Path(configured).expanduser().resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"external component is not a regular file: {path}")
        actual = sha256_file(path)
        if actual != expected_sha:
            raise ValueError(f"external component SHA mismatch: {item.get('id')}")
        external_results.append(
            {
                "id": item.get("id"),
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )

    camera = Path(
        "/dev/v4l/by-id/"
        "usb-Web_Camera_Web_Camera_202604081837-video-index0"
    )
    camera_present = camera.exists()
    camera_owner = ""
    if camera_present:
        resolved = camera.resolve(strict=True)
        probe = subprocess.run(
            ["fuser", str(resolved)],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        camera_owner = (probe.stdout + probe.stderr).strip()
    return {
        "schema": "rootscope.competition-runtime-verify-receipt.v2",
        "status": "PASS_HASHES_IDENTITY_EXTERNALS_ZERO_AUTHORITY_NOT_PHYSICAL_QUALIFICATION",
        "candidate_id": EXPECTED_ID,
        "identity": identity,
        "architecture": platform.machine(),
        "files_verified": len(expected),
        "external_components_verified": external_results,
        "camera": {
            "present": camera_present,
            "opened_by_verifier": False,
            "owner_observed": camera_owner,
        },
        "authority": dict(authority),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.release_root), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
