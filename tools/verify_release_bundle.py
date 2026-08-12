#!/usr/bin/env python3
"""Verify RootScope release checksums, deterministic tar metadata and SPDX SBOM."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._+-]*)\Z")
RELEASE_SCHEMA = "rootscope.public-release.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path.name}") from error
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path.name}")
    return value


def parse_checksums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("missing or invalid SHA256SUMS") from error
    result: dict[str, str] = {}
    for line in lines:
        match = CHECKSUM_LINE.fullmatch(line)
        if not match:
            fail("malformed SHA256SUMS entry")
        digest, name = match.groups()
        if name in result:
            fail("duplicate SHA256SUMS entry")
        result[name] = digest
    if not result:
        fail("SHA256SUMS is empty")
    return result


def validate_tar(
    path: Path, prefix: str, epoch: int, expected_files: int
) -> dict[str, str]:
    header = path.read_bytes()[:10]
    if len(header) != 10 or header[:2] != b"\x1f\x8b":
        fail(f"not a gzip stream: {path.name}")
    if struct.unpack("<I", header[4:8])[0] != epoch:
        fail(f"non-deterministic gzip timestamp: {path.name}")
    seen: set[str] = set()
    inventory: dict[str, str] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) != expected_files:
            fail(f"archive file count mismatch: {path.name}")
        for member in members:
            pure = PurePosixPath(member.name)
            if (
                not member.isfile()
                or pure.is_absolute()
                or ".." in pure.parts
                or not member.name.startswith(prefix + "/")
            ):
                fail(f"unsafe archive member: {path.name}")
            if member.name in seen:
                fail(f"duplicate archive member: {path.name}")
            seen.add(member.name)
            if member.mtime != epoch or member.uid != 0 or member.gid != 0:
                fail(f"non-deterministic tar metadata: {path.name}")
            if member.uname or member.gname or member.mode not in {0o644, 0o755}:
                fail(f"non-portable tar metadata: {path.name}")
            stream = archive.extractfile(member)
            if stream is None:
                fail(f"unreadable archive member: {path.name}")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            relative = PurePosixPath(*pure.parts[1:]).as_posix()
            inventory[relative] = digest.hexdigest()
    return inventory


def checksum_from_spdx(entry: dict[str, object]) -> str:
    checksums = entry.get("checksums")
    if not isinstance(checksums, list):
        fail("SPDX file has no checksums")
    for checksum in checksums:
        if isinstance(checksum, dict) and checksum.get("algorithm") == "SHA256":
            value = checksum.get("checksumValue")
            if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                return value
    fail("SPDX file has no valid SHA256 checksum")
    return ""  # unreachable, keeps static type checkers satisfied


def validate_spdx(path: Path, archive_inventory: dict[str, str]) -> int:
    sbom = load_json(path)
    if sbom.get("spdxVersion") != "SPDX-2.3" or sbom.get("dataLicense") != "CC0-1.0":
        fail("unsupported SPDX document")
    namespace = sbom.get("documentNamespace")
    if not isinstance(namespace, str) or not namespace.startswith("https://github.com/"):
        fail("invalid SPDX namespace")
    files = sbom.get("files")
    if not isinstance(files, list) or not files:
        fail("SPDX file inventory is empty")
    names: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            fail("invalid SPDX file entry")
        name = entry.get("fileName")
        if not isinstance(name, str) or not name.startswith("./"):
            fail("invalid SPDX file name")
        relative = name[2:]
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or relative in names:
            fail("unsafe or duplicate SPDX file name")
        names.add(relative)
        checksum = checksum_from_spdx(entry)
        if archive_inventory.get(relative) != checksum:
            fail("SPDX checksum does not match release archives")
    if names != set(archive_inventory):
        fail("SPDX inventory does not exactly match release archives")
    return len(files)


def verify(directory: Path) -> tuple[int, int]:
    directory = directory.resolve()
    manifest_path = directory / "RELEASE_MANIFEST.json"
    checksums_path = directory / "SHA256SUMS"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != RELEASE_SCHEMA:
        fail("unsupported release manifest schema")
    version = manifest.get("version")
    epoch = manifest.get("source_date_epoch")
    bundles = manifest.get("bundles")
    sbom = manifest.get("sbom")
    if not isinstance(version, str) or not isinstance(epoch, int) or epoch < 0:
        fail("invalid release identity")
    if not isinstance(bundles, list) or len(bundles) != 3 or not isinstance(sbom, dict):
        fail("release must contain source, evidence and models bundles plus an SBOM")
    roles = {entry.get("role") for entry in bundles if isinstance(entry, dict)}
    if roles != {"source", "evidence", "models"}:
        fail("release bundle roles are incomplete")

    expected: dict[str, str] = {}
    prefix = f"RootScope-AdventureX2026-{version}"
    total_archived = 0
    archive_inventory: dict[str, str] = {}
    for entry in bundles:
        if not isinstance(entry, dict):
            fail("invalid bundle manifest entry")
        name, digest, size, count = (
            entry.get("name"),
            entry.get("sha256"),
            entry.get("bytes"),
            entry.get("files"),
        )
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*\.tar\.gz", name)
            or not isinstance(digest, str)
            or not isinstance(size, int)
            or not isinstance(count, int)
            or count <= 0
        ):
            fail("invalid bundle metadata")
        path = directory / name
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            fail(f"bundle digest mismatch: {name}")
        bundle_inventory = validate_tar(path, prefix, epoch, count)
        overlap = set(bundle_inventory) & set(archive_inventory)
        if overlap:
            fail("a public file appears in more than one release bundle")
        archive_inventory.update(bundle_inventory)
        expected[name] = digest
        total_archived += count

    sbom_name, sbom_digest, sbom_size = sbom.get("name"), sbom.get("sha256"), sbom.get("bytes")
    if not isinstance(sbom_name, str) or not isinstance(sbom_digest, str) or not isinstance(sbom_size, int):
        fail("invalid SBOM metadata")
    sbom_path = directory / sbom_name
    if not sbom_path.is_file() or sbom_path.stat().st_size != sbom_size or sha256_file(sbom_path) != sbom_digest:
        fail("SBOM digest mismatch")
    inventory_count = validate_spdx(sbom_path, archive_inventory)
    if inventory_count != total_archived:
        fail("SPDX and archive file counts differ")
    expected[sbom_name] = sbom_digest
    expected[manifest_path.name] = sha256_file(manifest_path)

    declared = parse_checksums(checksums_path)
    if declared != expected:
        fail("SHA256SUMS does not exactly match release artifacts")
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_names != set(expected) | {checksums_path.name}:
        fail("unexpected or missing file in release directory")
    return total_archived, inventory_count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        archived, inventoried = verify(args.directory)
    except ValueError as error:
        print(f"RELEASE_VERIFY=FAIL reason={error}")
        return 1
    print(f"RELEASE_VERIFY=PASS archived_files={archived} sbom_files={inventoried}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
