#!/usr/bin/env python3
"""Verify or install the compositional RootScope X5 field bundle v2.

The default install is deliberately conservative: it verifies every outer and
nested artifact, installs the frozen v1 CPU core and runs its simulated-input
self-test, stages the read-only LLM model/server with a disabled manual-ack user
unit, and prepares a separate BPU ``--system-site-packages`` venv.  It never
starts a service, creates an activation gate, opens/enumerates a device, loads a
BPU binary, runs BPU inference, accesses a network, writes serial data, or sends
an irrigation command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
from typing import Any, BinaryIO, Mapping, Sequence


BUNDLE_SCHEMA = "rootscope.x5-field-bundle.v2"
BUNDLE_ID = "rootscope_x5_field_bundle_v2"
BUNDLE_STATUS = "HASH_LOCKED_COMPOSITION_NOT_X5_QUALIFIED"
CORE_ARCHIVE = "rootscope_x5_offline_core_v1.tar"
CORE_ROOT = "rootscope_x5_offline_core_v1"
LLM_MODEL_ARCHIVE = "rootscope_x5_readonly_llm_model_v1.tar"
LLM_MODEL_ROOT = "rootscope_x5_readonly_llm_model_v1"
LLAMA_SERVER_ARCHIVE = "rootscope_llama_server_arm64_b9637_v1.tar"
LLAMA_SERVER_ROOT = "rootscope_llama_server_arm64_b9637_v1"
BPU_SUPPORT_ARCHIVE = "rootscope_seed17_bpu_support_v1.tar"
BPU_SUPPORT_ROOT = "rootscope_seed17_bpu_support_v1"
PRINT_PDF = "RootScope_demo_reference_candidate_cards_A4.pdf"
SHA256_LENGTH = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(handle: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _portable_relative(value: Any, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context} must be one portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError(f"{context} is not normalized")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{context} contains an unsafe component")
    return path


def _valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_false_mapping(value: Any, context: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{context} must be a non-empty object")
    changed = sorted(name for name, flag in value.items() if flag is not False)
    if changed:
        raise ValueError(f"{context} must remain false: {changed}")


def _parse_sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        if len(line) < SHA256_LENGTH + 3 or line[SHA256_LENGTH : SHA256_LENGTH + 2] != "  ":
            raise ValueError(f"invalid SHA256SUMS line {line_number}")
        digest = line[:SHA256_LENGTH]
        relative = _portable_relative(line[SHA256_LENGTH + 2 :], f"SHA256SUMS line {line_number}")
        name = relative.as_posix()
        if not _valid_sha(digest) or name in records:
            raise ValueError(f"invalid or duplicate SHA256SUMS line {line_number}")
        records[name] = digest
    return records


def _safe_tar_members(path: Path, expected_root: str) -> list[tarfile.TarInfo]:
    expected_prefix = f"{expected_root}/"
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        names: set[str] = set()
        for member in members:
            name = _portable_relative(member.name, f"{path.name} member").as_posix()
            if name != expected_root and not name.startswith(expected_prefix):
                raise ValueError(f"{path.name} member is outside {expected_root}: {name}")
            if name == expected_root and not member.isdir():
                raise ValueError(f"{path.name} root member must be a directory")
            if member.isfile() and not name.startswith(expected_prefix):
                raise ValueError(f"{path.name} file must be below {expected_root}/")
            if name in names:
                raise ValueError(f"{path.name} contains a duplicate member: {name}")
            names.add(name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"{path.name} contains a link or special member: {name}")
        if not members:
            raise ValueError(f"{path.name} is empty")
        file_names = {
            PurePosixPath(member.name) for member in members if member.isfile()
        }
        explicit_dirs = {
            PurePosixPath(member.name) for member in members if member.isdir()
        }
        allowed_dirs = {
            parent
            for name in file_names
            for parent in name.parents
            if parent.as_posix() != "."
        }
        if not explicit_dirs.issubset(allowed_dirs):
            extra = sorted(path.as_posix() for path in explicit_dirs - allowed_dirs)
            raise ValueError(f"{path.name} contains extra empty directories: {extra}")
        return members


def _read_tar_json(path: Path, member_name: str) -> Mapping[str, Any]:
    with tarfile.open(path, mode="r:") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise ValueError(f"{path.name} is missing {member_name}") from exc
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"{path.name} member is not readable: {member_name}")
        payload = json.loads(handle.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{member_name} must contain an object")
    return payload


def _read_tar_member_hash(
    path: Path, member_name: str
) -> tuple[str, int, bytes]:
    with tarfile.open(path, mode="r:") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise ValueError(f"{path.name} is missing {member_name}") from exc
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"{path.name} member is not readable: {member_name}")
        prefix = handle.read(64)
        digest = hashlib.sha256(prefix)
        size = len(prefix)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size, prefix


def _audit_bpu_nested(path: Path, record: Mapping[str, Any]) -> Mapping[str, Any]:
    members = _safe_tar_members(path, BPU_SUPPORT_ROOT)
    names = {member.name for member in members if member.isfile()}
    manifest = _read_tar_json(
        path, f"{BPU_SUPPORT_ROOT}/component_manifest.json"
    )
    if manifest.get("schema") != "rootscope.seed17-bpu-support-component.v1":
        raise ValueError("unsupported BPU support component schema")
    _strict_false_mapping(manifest.get("formal_flags"), "BPU component formal_flags")
    _strict_false_mapping(manifest.get("authority"), "BPU component authority")
    file_records = manifest.get("files")
    if not isinstance(file_records, list) or not file_records:
        raise ValueError("BPU component file manifest is missing")
    expected_files: dict[str, tuple[str, int, int]] = {}
    for index, file_record in enumerate(file_records):
        if not isinstance(file_record, Mapping):
            raise ValueError(f"BPU files[{index}] must be an object")
        relative = _portable_relative(
            file_record.get("path"), f"BPU files[{index}].path"
        ).as_posix()
        digest = file_record.get("sha256")
        size = file_record.get("bytes")
        mode = file_record.get("mode")
        if (
            not _valid_sha(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or isinstance(mode, bool)
            or not isinstance(mode, int)
            or relative in expected_files
        ):
            raise ValueError(f"BPU files[{index}] hash/size/mode is invalid")
        expected_files[relative] = (str(digest), size, mode)
    manifest_member = f"{BPU_SUPPORT_ROOT}/component_manifest.json"
    sums_member = f"{BPU_SUPPORT_ROOT}/SHA256SUMS"
    expected_members = {
        f"{BPU_SUPPORT_ROOT}/{name}" for name in expected_files
    } | {manifest_member, sums_member}
    if names != expected_members:
        raise ValueError("BPU component manifest does not exactly cover tar files")
    with tarfile.open(path, mode="r:") as archive:
        sums_handle = archive.extractfile(archive.getmember(sums_member))
        manifest_handle = archive.extractfile(archive.getmember(manifest_member))
        if sums_handle is None or manifest_handle is None:
            raise ValueError("BPU component manifest/SHA256SUMS is unreadable")
        sums_text = sums_handle.read().decode("utf-8")
        manifest_digest = hashlib.sha256(manifest_handle.read()).hexdigest()
        actual_sums: dict[str, str] = {}
        for line_number, line in enumerate(sums_text.splitlines(), 1):
            if len(line) < 67 or line[64:66] != "  " or not _valid_sha(line[:64]):
                raise ValueError(f"invalid BPU SHA256SUMS line {line_number}")
            relative = _portable_relative(
                line[66:], f"BPU SHA256SUMS line {line_number}"
            ).as_posix()
            if relative in actual_sums:
                raise ValueError("duplicate BPU SHA256SUMS path")
            actual_sums[relative] = line[:64]
    expected_sums = {name: values[0] for name, values in expected_files.items()}
    expected_sums["component_manifest.json"] = manifest_digest
    if actual_sums != expected_sums:
        raise ValueError("BPU SHA256SUMS does not exactly match component_manifest")
    with tarfile.open(path, mode="r:") as archive:
        for relative, (digest, size, mode) in expected_files.items():
            member = archive.getmember(f"{BPU_SUPPORT_ROOT}/{relative}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"BPU component member is unreadable: {relative}")
            actual_digest, actual_size = sha256_stream(handle)
            if actual_digest != digest or actual_size != size or member.mode != mode:
                raise ValueError(f"BPU component member hash/size/mode mismatch: {relative}")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("BPU component selection is missing")
    selected = selection.get("selected_bin")
    bins = sorted(name for name in names if name.lower().endswith(".bin"))
    if selected is None:
        if bins or manifest.get("bpu_binary_included") is not False:
            raise ValueError("null BPU selection must package no .bin")
    else:
        if not isinstance(selected, Mapping) or len(bins) != 1:
            raise ValueError("selected BPU component must contain exactly one .bin")
        package_path = selected.get("package_path")
        if package_path != bins[0]:
            raise ValueError("selected BPU package path does not match tar member")
        if selection.get("all_predeclared_replay_gates_passed") is not True:
            raise ValueError("a packaged BPU bin requires every frozen replay gate")
        with tarfile.open(path, mode="r:") as archive:
            handle = archive.extractfile(archive.getmember(bins[0]))
            if handle is None:
                raise ValueError("selected BPU bin is unreadable")
            digest, size = sha256_stream(handle)
        if digest != selected.get("sha256") or size != selected.get("bytes"):
            raise ValueError("selected BPU bin hash/size mismatch")
    if record.get("bpu_binary_included") is not bool(bins):
        raise ValueError("outer BPU binary claim does not match nested component")
    wheels = sorted(name for name in names if name.lower().endswith(".whl"))
    pillow = manifest.get("pillow_wheel")
    if len(wheels) != 1 or not isinstance(pillow, Mapping):
        raise ValueError("BPU support must contain exactly one Pillow wheel")
    relative_wheel = _portable_relative(
        pillow.get("package_path"), "BPU Pillow package_path"
    ).as_posix()
    expected_wheel = f"{BPU_SUPPORT_ROOT}/{relative_wheel}"
    if wheels != [expected_wheel] or not PurePosixPath(relative_wheel).name.lower().startswith("pillow-"):
        raise ValueError("BPU wheel must be the bound Pillow wheel")
    lowered_wheels = " ".join(wheels).lower()
    if "numpy" in lowered_wheels or "hobot_dnn" in lowered_wheels:
        raise ValueError("NumPy/hobot_dnn wheels are forbidden in the BPU component")
    wheel_sha, wheel_size, _prefix = _read_tar_member_hash(path, expected_wheel)
    if wheel_sha != pillow.get("sha256") or wheel_size != pillow.get("bytes"):
        raise ValueError("BPU Pillow wheel hash/size mismatch")
    policy = manifest.get("dependency_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("BPU dependency policy is missing")
    if (
        policy.get("independent_system_site_packages_venv") is not True
        or policy.get("system_numpy_required") is not True
        or policy.get("system_hobot_dnn_required") is not True
        or
        policy.get("core_v1_venv_allowed") is not False
        or policy.get("numpy_wheel_included") is not False
        or policy.get("local_wheel_install_allowlist") != ["Pillow"]
    ):
        raise ValueError("BPU isolated environment policy changed")
    return {
        "schema": "rootscope.bpu-support-nested-verification.v1",
        "status": "PASS_SUPPORT_COMPONENT_NOT_X5_QUALIFICATION",
        "members": len(members),
        "bpu_binary_included": bool(bins),
        "selected_bin": selected,
        "wheels": wheels,
        "pillow_sha256": wheel_sha,
        "bpu_model_loaded": False,
        "bpu_forward_executed": False,
        "camera_opened": False,
        "x5_validated": False,
        "model_qualified": False,
        "execution_authority": False,
    }


def _audit_llama_nested(path: Path) -> Mapping[str, Any]:
    manifest_name = f"{LLAMA_SERVER_ROOT}/release_manifest.json"
    manifest = _read_tar_json(path, manifest_name)
    if (
        manifest.get("schema") != "rootscope.llama_server_arm64_release.v1"
        or manifest.get("release_id") != LLAMA_SERVER_ROOT
        or manifest.get("status") != "CROSS_BUILD_QEMU_SMOKE_PASS_NOT_X5_VALIDATED"
    ):
        raise ValueError("llama-server release identity/status mismatch")
    _strict_false_mapping(manifest.get("formal_flags"), "llama-server formal_flags")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("llama-server artifact contract is missing")
    relative = _portable_relative(artifact.get("path"), "llama-server artifact.path")
    member_name = f"{LLAMA_SERVER_ROOT}/{relative.as_posix()}"
    digest, size, prefix = _read_tar_member_hash(path, member_name)
    if digest != artifact.get("sha256") or size != artifact.get("size_bytes"):
        raise ValueError("llama-server binary hash/size mismatch")
    if len(prefix) < 20 or prefix[:4] != b"\x7fELF" or prefix[4:6] != b"\x02\x01":
        raise ValueError("llama-server is not a 64-bit little-endian ELF")
    if int.from_bytes(prefix[18:20], byteorder="little") != 183:
        raise ValueError("llama-server ELF machine is not AArch64")
    if (
        artifact.get("elf_machine") != "AArch64"
        or artifact.get("glibc_max") != "2.34"
        or artifact.get("rpath") != []
        or artifact.get("runpath") != []
    ):
        raise ValueError("llama-server ELF compatibility contract changed")
    qemu = manifest.get("qemu_smoke")
    if not isinstance(qemu, Mapping) or qemu.get("status") != "PASS":
        raise ValueError("llama-server QEMU smoke receipt is missing")
    if (
        qemu.get("loopback_health_passed") is not True
        or qemu.get("minimal_completion_passed") is not True
        or qemu.get("version_executed") is not True
        or qemu.get("network_mode") != "none"
    ):
        raise ValueError("llama-server QEMU smoke boundary changed")
    sums_member = f"{LLAMA_SERVER_ROOT}/SHA256SUMS"
    with tarfile.open(path, mode="r:") as archive:
        regular = {member.name: member for member in archive.getmembers() if member.isfile()}
        sums_handle = archive.extractfile(archive.getmember(sums_member))
        if sums_handle is None:
            raise ValueError("llama-server SHA256SUMS is unreadable")
        sums_records: dict[str, str] = {}
        for line_number, line in enumerate(
            sums_handle.read().decode("utf-8").splitlines(), 1
        ):
            if len(line) < 67 or line[64:66] != "  " or not _valid_sha(line[:64]):
                raise ValueError(f"invalid llama-server SHA256SUMS line {line_number}")
            relative = _portable_relative(
                line[66:], f"llama-server SHA256SUMS line {line_number}"
            ).as_posix()
            if relative in sums_records:
                raise ValueError("duplicate llama-server SHA256SUMS path")
            sums_records[relative] = line[:64]
        expected_relative = {
            PurePosixPath(name).relative_to(LLAMA_SERVER_ROOT).as_posix()
            for name in regular
            if name != sums_member
        }
        if set(sums_records) != expected_relative:
            raise ValueError("llama-server SHA256SUMS does not exactly cover regular files")
        for relative, expected_digest in sums_records.items():
            member = regular[f"{LLAMA_SERVER_ROOT}/{relative}"]
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"llama-server member is unreadable: {relative}")
            actual_digest, actual_size = sha256_stream(handle)
            if actual_digest != expected_digest or actual_size != member.size:
                raise ValueError(f"llama-server member hash/size mismatch: {relative}")
    return {
        "status": "PASS_ARM64_ELF_AND_QEMU_RECEIPT_NOT_X5_VALIDATION",
        "artifact_sha256": digest,
        "artifact_bytes": size,
        "elf_machine": "AArch64",
        "qemu_smoke_passed": True,
        "sha256sums_files_verified": len(sums_records),
        "x5_validated": False,
        "llama_server_qualified": False,
    }


def verify_bundle(bundle_root: Path) -> Mapping[str, Any]:
    """Strictly verify the extracted outer bundle without writing anything."""

    root = bundle_root.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("bundle_root must be a real directory")
    manifest_path = root / "bundle_manifest.json"
    sums_path = root / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        raise ValueError("bundle_manifest.json and SHA256SUMS are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("bundle manifest must be an object")
    if (
        manifest.get("schema") != BUNDLE_SCHEMA
        or manifest.get("bundle_id") != BUNDLE_ID
        or manifest.get("status") != BUNDLE_STATUS
    ):
        raise ValueError("bundle identity/status mismatch")
    _strict_false_mapping(manifest.get("formal_flags"), "bundle formal_flags")
    _strict_false_mapping(manifest.get("authority"), "bundle authority")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("bundle manifest files must be non-empty")
    expected: dict[str, tuple[str, int, Mapping[str, Any]]] = {}
    for index, record in enumerate(files):
        if not isinstance(record, Mapping):
            raise ValueError(f"files[{index}] must be an object")
        name = _portable_relative(record.get("path"), f"files[{index}].path").as_posix()
        digest = record.get("sha256")
        size = record.get("bytes")
        if not _valid_sha(digest) or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"files[{index}] hash/size is invalid")
        if name in expected:
            raise ValueError(f"duplicate bundle file: {name}")
        expected[name] = (str(digest), size, record)
    sums = _parse_sums(sums_path)
    expected_sums = {name: value[0] for name, value in expected.items()}
    expected_sums["bundle_manifest.json"] = sha256_file(manifest_path)
    if sums != expected_sums:
        raise ValueError("SHA256SUMS does not exactly cover the bundle manifest")

    actual: set[str] = set()
    actual_dirs: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"bundle symlink is forbidden: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        elif path.is_dir():
            actual_dirs.add(path.relative_to(root).as_posix())
        else:
            raise ValueError(f"bundle special filesystem entry is forbidden: {path}")
    if actual != set(sums) | {"SHA256SUMS"}:
        raise ValueError("bundle contains missing or extra files")
    allowed_dirs = {
        parent.as_posix()
        for name in actual
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    if actual_dirs != allowed_dirs:
        raise ValueError("bundle contains missing or extra directories")
    for name, (digest, size, _record) in expected.items():
        path = root / Path(*PurePosixPath(name).parts)
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"bundle file hash/size mismatch: {name}")

    components = manifest.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("bundle components object is missing")
    component_contract = {
        "core_v1": (CORE_ARCHIVE, CORE_ROOT),
        "readonly_llm_model_v1": (LLM_MODEL_ARCHIVE, LLM_MODEL_ROOT),
        "llama_server_arm64_v1": (LLAMA_SERVER_ARCHIVE, LLAMA_SERVER_ROOT),
        "bpu_support_v1": (BPU_SUPPORT_ARCHIVE, BPU_SUPPORT_ROOT),
    }
    nested: dict[str, Any] = {}
    for key, (filename, archive_root) in component_contract.items():
        record = components.get(key)
        if not isinstance(record, Mapping) or record.get("filename") != filename:
            raise ValueError(f"component contract missing: {key}")
        relative = f"components/{filename}"
        file_record = expected.get(relative)
        if file_record is None or file_record[0] != record.get("sha256") or file_record[1] != record.get("bytes"):
            raise ValueError(f"component/file record mismatch: {key}")
        path = root / "components" / filename
        members = _safe_tar_members(path, archive_root)
        nested[key] = {"members": len(members), "safe_paths": True}
    nested["llama_server_arm64_v1"] = _audit_llama_nested(
        root / "components" / LLAMA_SERVER_ARCHIVE
    )
    nested["bpu_support_v1"] = _audit_bpu_nested(
        root / "components" / BPU_SUPPORT_ARCHIVE,
        components["bpu_support_v1"],
    )
    return {
        "schema": "rootscope.x5-field-bundle-verification.v2",
        "status": "PASS_HASHES_AND_SAFE_PATHS_NOT_X5_QUALIFIED",
        "bundle_id": BUNDLE_ID,
        "manifest_sha256": sha256_file(manifest_path),
        "files_verified": len(expected),
        "nested": nested,
        "verify_only_capable": True,
        "hardware_touched": False,
        "network_touched": False,
        "device_enumerated": False,
        "service_started": False,
        "systemctl_invoked": False,
        "activation_gate_created": False,
        "bpu_model_loaded": False,
        "bpu_forward_executed": False,
        "camera_opened": False,
        "x5_validated": False,
        "model_qualified": False,
        "execution_authority": False,
        "physical_authority": False,
    }


def assert_supported_install_host() -> None:
    facts = (
        platform.system(),
        platform.machine().lower(),
        platform.python_implementation(),
        sys.version_info[:2],
        sys.prefix == sys.base_prefix,
    )
    if facts not in {
        ("Linux", "aarch64", "CPython", (3, 10), True),
        ("Linux", "arm64", "CPython", (3, 10), True),
    }:
        raise RuntimeError(
            "installation requires RDK Linux/aarch64 system CPython 3.10 "
            "outside every venv"
        )


def _tar_expected_files(path: Path, expected_root: str) -> dict[str, tuple[str, int, int]]:
    expected: dict[str, tuple[str, int, int]] = {}
    with tarfile.open(path, mode="r:") as archive:
        _safe_tar_members(path, expected_root)
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"unreadable nested member: {member.name}")
            digest, size = sha256_stream(handle)
            expected[member.name] = (digest, size, member.mode)
    return expected


def _verify_staged_tree(destination: Path, expected: Mapping[str, tuple[str, int, int]]) -> None:
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f"unsafe staged destination: {destination}")
    actual: set[str] = set()
    actual_dirs: set[str] = set()
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"staged symlink is forbidden: {path}")
        if path.is_file():
            actual.add(path.relative_to(destination).as_posix())
        elif path.is_dir():
            actual_dirs.add(path.relative_to(destination).as_posix())
        else:
            raise ValueError(f"staged special filesystem entry is forbidden: {path}")
    expected_relative = {
        PurePosixPath(name).relative_to(destination.name).as_posix()
        for name in expected
    }
    if actual != expected_relative:
        raise ValueError(f"staged tree coverage mismatch: {destination.name}")
    expected_dirs = {
        parent.as_posix()
        for name in expected_relative
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    if actual_dirs != expected_dirs:
        raise ValueError(f"staged tree directory coverage mismatch: {destination.name}")
    for name, (digest, size, _mode) in expected.items():
        relative = PurePosixPath(name).relative_to(destination.name)
        path = destination / Path(*relative.parts)
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"staged tree hash mismatch: {name}")


def _stage_nested_tar(path: Path, staging_parent: Path, expected_root: str) -> Path:
    expected = _tar_expected_files(path, expected_root)
    destination = staging_parent / expected_root
    if destination.exists():
        _verify_staged_tree(destination, expected)
        return destination
    staging_parent.mkdir(parents=True, exist_ok=True)
    partial = staging_parent / f".{expected_root}.partial"
    if partial.exists():
        if partial.is_symlink() or partial.parent.resolve() != staging_parent.resolve():
            raise ValueError("unsafe partial staging destination")
        shutil.rmtree(partial)
    partial.mkdir()
    with tarfile.open(path, mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = PurePosixPath(member.name).relative_to(expected_root)
            target = partial / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"unreadable nested member: {member.name}")
            with target.open("wb") as output:
                shutil.copyfileobj(handle, output, 1024 * 1024)
            os.chmod(target, 0o755 if member.mode & 0o111 else 0o644)
    _verify_staged_tree(partial, {
        f"{partial.name}/{PurePosixPath(name).relative_to(expected_root).as_posix()}": value
        for name, value in expected.items()
    })
    os.replace(partial, destination)
    _verify_staged_tree(destination, expected)
    return destination


def _extract_expected_receipt(text: str, expected_schema: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()
    matches: list[Mapping[str, Any]] = []
    offset = 0
    while offset < len(text):
        line_end = text.find("\n", offset)
        if line_end < 0:
            line_end = len(text)
        line = text[offset:line_end]
        if not line.startswith("{"):
            offset = line_end + (line_end < len(text))
            continue
        index = offset
        try:
            candidate, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            offset = line_end + (line_end < len(text))
            continue
        if isinstance(candidate, Mapping) and candidate.get("schema") == expected_schema:
            matches.append(candidate)
        # Skip the entire successfully decoded top-level value.  This prevents
        # nested objects in a wrapper from being mistaken for receipts while
        # still counting two consecutive top-level matching receipts.
        offset = index + end
        while offset < len(text) and text[offset] in " \t\r\n":
            offset += 1
    if len(matches) != 1:
        raise RuntimeError(
            f"offline component emitted {len(matches)} receipts for schema {expected_schema}"
        )
    return matches[0]


def _require_disk_receipt_matches(
    stdout_receipt: Mapping[str, Any], path: Path, expected_schema: str
) -> Mapping[str, Any]:
    if path.is_symlink():
        raise RuntimeError(f"fixed component receipt symlink is forbidden: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"fixed component receipt is missing or unsafe: {path}")
    disk = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(disk, Mapping) or disk.get("schema") != expected_schema:
        raise RuntimeError(f"fixed component receipt schema mismatch: {path}")
    if dict(disk) != dict(stdout_receipt):
        raise RuntimeError(f"stdout/disk component receipt divergence: {path}")
    return disk


def _run_json(
    command: Sequence[str], *, cwd: Path, expected_schema: str
) -> Mapping[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=1800,
        env={
            **os.environ,
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise RuntimeError(f"offline component command failed: {detail}")
    # A fresh core install runs the offline pip shell and therefore may emit
    # progress lines before its final receipt.  Extract exactly one object with
    # the explicitly requested schema; never accept an arbitrary JSON-looking
    # fragment from dependency output.
    return _extract_expected_receipt(completed.stdout, expected_schema)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    os.chmod(partial, 0o600)
    os.replace(partial, path)


def install_bundle(
    bundle_root: Path,
    install_base: Path,
    config_dir: Path,
    systemd_user_dir: Path,
) -> Mapping[str, Any]:
    assert_supported_install_host()
    root = bundle_root.expanduser().resolve(strict=True)
    verification = verify_bundle(root)
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    base = install_base.expanduser().resolve()
    stage = base / "staged_components"

    core = _stage_nested_tar(root / "components" / CORE_ARCHIVE, stage, CORE_ROOT)
    model = _stage_nested_tar(root / "components" / LLM_MODEL_ARCHIVE, stage, LLM_MODEL_ROOT)
    server = _stage_nested_tar(root / "components" / LLAMA_SERVER_ARCHIVE, stage, LLAMA_SERVER_ROOT)
    bpu = _stage_nested_tar(root / "components" / BPU_SUPPORT_ARCHIVE, stage, BPU_SUPPORT_ROOT)

    core_receipt = _run_json(
        [
            sys.executable,
            str(core / "rootscope/deploy/x5/scripts/install_offline_core.py"),
            "--package-root",
            str(core),
            "--install-base",
            str(base / "core_v1"),
        ],
        cwd=core,
        expected_schema="rootscope.x5-offline-user-install-receipt.v1",
    )
    core_receipt = _require_disk_receipt_matches(
        core_receipt,
        base / "core_v1/evidence/rootscope_x5_offline_core_v1/install_receipt.json",
        "rootscope.x5-offline-user-install-receipt.v1",
    )
    if (
        core_receipt.get("status") != "PASS_LOCAL_AARCH64_CPU_SMOKE_NOT_X5_QUALIFIED"
        or core_receipt.get("cpu_onnx_simulated_selftest_passed") is not True
    ):
        raise RuntimeError("frozen core v1 simulated CPU self-test did not pass")
    project_root = Path(str(core_receipt["project_root"])).resolve(strict=True)
    core_python = Path(str(core_receipt["python_executable"])).resolve(strict=True)

    server_manifest = json.loads((server / "release_manifest.json").read_text(encoding="utf-8"))
    server_artifact = server_manifest.get("artifact")
    if not isinstance(server_artifact, Mapping):
        raise ValueError("staged llama-server artifact contract is missing")
    server_relative = _portable_relative(
        server_artifact.get("path"), "staged llama-server artifact.path"
    )
    server_path = server / Path(*server_relative.parts)
    server_sha = str(server_artifact["sha256"])
    config_root = config_dir.expanduser().resolve()
    user_unit_root = systemd_user_dir.expanduser().resolve()
    llm_receipt = _run_json(
        [
            str(core_python),
            str(project_root / "deploy/x5/scripts/install_readonly_llm.py"),
            "--release-dir",
            str(model),
            "--project-root",
            str(project_root),
            "--python",
            str(core_python),
            "--llama-server",
            str(server_path),
            "--llama-server-sha256",
            server_sha,
            "--prefix",
            str(base / "readonly_llm"),
            "--config-dir",
            str(config_root),
            "--systemd-user-dir",
            str(user_unit_root),
        ],
        cwd=project_root,
        expected_schema="rootscope.readonly_llm_install_receipt.v1",
    )
    llm_receipt = _require_disk_receipt_matches(
        llm_receipt,
        base / "readonly_llm/install_receipt.json",
        "rootscope.readonly_llm_install_receipt.v1",
    )
    if (
        llm_receipt.get("schema") != "rootscope.readonly_llm_install_receipt.v1"
        or llm_receipt.get("status")
        != "INSTALLED_DISABLED_MANUAL_ACK_REQUIRED_NOT_X5_QUALIFIED"
        or llm_receipt.get("manual_acknowledged") is not False
    ):
        raise RuntimeError("read-only LLM receipt identity/disabled state changed")
    for name in ("activation_gate_created", "service_started", "systemctl_invoked"):
        if llm_receipt.get(name) is not False:
            raise RuntimeError(f"read-only LLM boundary changed: {name}")
    gate = config_root / "enable-readonly-llm"
    if gate.exists() or gate.is_symlink():
        raise RuntimeError("read-only LLM activation gate already exists; refusing disabled claim")
    env_path = config_root / "rootscope-llm.env"
    if env_path.is_symlink() or not env_path.is_file():
        raise RuntimeError("read-only LLM environment file is missing or unsafe")
    env_values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env_values[key] = value
    if (
        env_values.get("ROOTSCOPE_LLM_MANUAL_ACK") != "NOT_ACKNOWLEDGED"
        or env_values.get("ROOTSCOPE_LLM_READ_ONLY") != "true"
        or env_values.get("ROOTSCOPE_LLM_EXTERNAL_NETWORK") != "false"
        or env_values.get("ROOTSCOPE_LLM_TOOL_EXECUTION") != "false"
        or env_values.get("ROOTSCOPE_LLM_ACTUATOR_ACCESS") != "false"
    ):
        raise RuntimeError("read-only LLM environment is not disabled/zero-authority")
    runtime_config_path = Path(str(llm_receipt.get("runtime_config_path", "")))
    if runtime_config_path.is_symlink() or not runtime_config_path.is_file():
        raise RuntimeError("read-only LLM runtime config is missing or unsafe")
    runtime_config = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    if (
        runtime_config.get("default_enabled") is not False
        or runtime_config.get("manual_start_only") is not True
        or runtime_config.get("manual_acknowledged") is not False
        or runtime_config.get("read_only") is not True
        or runtime_config.get("execution_authority") is not False
        or runtime_config.get("physical_authority") is not False
    ):
        raise RuntimeError("read-only LLM runtime config is not disabled/zero-authority")
    unit_path = Path(str(llm_receipt.get("user_unit_path", "")))
    if unit_path.is_symlink() or not unit_path.is_file():
        raise RuntimeError("read-only LLM user unit is missing or unsafe")
    unit_text = unit_path.read_text(encoding="utf-8")
    if "[Install]" in unit_text or f"ConditionPathExists={gate}" not in unit_text:
        raise RuntimeError("read-only LLM user unit can be enabled without the absent gate")

    bpu_manifest = json.loads((bpu / "component_manifest.json").read_text(encoding="utf-8"))
    pillow = bpu_manifest.get("pillow_wheel")
    if not isinstance(pillow, Mapping):
        raise ValueError("BPU Pillow wheel binding is missing")
    bpu_receipt_path = base / "evidence/seed17_bpu_system_site_venv.json"
    bpu_receipt = _run_json(
        [
            sys.executable,
            str(bpu / "rootscope/deploy/x5/scripts/prepare_bpu_system_site_venv.py"),
            "--venv",
            str(base / "venvs/rootscope_seed17_bpu_system_site"),
            "--pillow-wheel",
            str(bpu / str(pillow["package_path"])),
            "--expected-pillow-sha256",
            str(pillow["sha256"]),
            "--output-json",
            str(bpu_receipt_path),
        ],
        cwd=bpu,
        expected_schema="rootscope.seed17-bpu-system-site-venv-receipt.v1",
    )
    bpu_receipt = _require_disk_receipt_matches(
        bpu_receipt,
        bpu_receipt_path,
        "rootscope.seed17-bpu-system-site-venv-receipt.v1",
    )
    if bpu_receipt.get("claims", {}).get("bpu_model_loaded") is not False:
        raise RuntimeError("BPU environment preparation unexpectedly loaded a model")
    if (
        bpu_receipt.get("status")
        != "PASS_IMPORT_ONLY_ENVIRONMENT_NOT_MODEL_OR_X5_QUALIFICATION"
        or bpu_receipt.get("venv", {}).get("include_system_site_packages") is not True
        or bpu_receipt.get("venv", {}).get("core_v1_venv_allowed") is not False
        or bpu_receipt.get("dependency_policy", {}).get("local_install_allowlist")
        != ["Pillow"]
        or bpu_receipt.get("dependency_policy", {}).get("venv_numpy_install_allowed")
        is not False
        or bpu_receipt.get("dependency_policy", {}).get("venv_hobot_dnn_install_allowed")
        is not False
    ):
        raise RuntimeError("BPU isolated environment receipt boundary changed")
    _strict_false_mapping(bpu_receipt.get("claims"), "BPU preparation claims")
    for name, value in bpu_receipt.get("authority", {}).items():
        if name != "hardware_touched" and value is not False:
            raise RuntimeError(f"BPU preparation authority changed: {name}")
    if bpu_receipt.get("authority", {}).get("hardware_touched") is not False:
        raise RuntimeError("BPU import-only environment preparation touched hardware")

    receipt = {
        "schema": "rootscope.x5-field-bundle-install-receipt.v2",
        "status": "PASS_OFFLINE_STAGING_AND_CPU_SIMULATED_SELFTEST_NOT_X5_QUALIFIED",
        "bundle_manifest_sha256": verification["manifest_sha256"],
        "install_base": str(base),
        "staged_components": {
            "core_v1": str(core),
            "readonly_llm_model_v1": str(model),
            "llama_server_arm64_v1": str(server),
            "bpu_support_v1": str(bpu),
        },
        "core": core_receipt,
        "readonly_llm": llm_receipt,
        "bpu_environment": bpu_receipt,
        "bpu_binary_included": bool(manifest["components"]["bpu_support_v1"]["bpu_binary_included"]),
        "cpu_simulated_selftest_passed": True,
        "llm_installed_disabled": True,
        "llm_manual_ack_required": True,
        "activation_gate_created": False,
        "service_started": False,
        "systemctl_invoked": False,
        "network_touched": False,
        "device_enumerated": False,
        "serial_write": False,
        "pump_command": False,
        "bpu_model_loaded": False,
        "bpu_forward_executed": False,
        "camera_opened": False,
        "x5_validated": False,
        "model_qualified": False,
        "llama_server_qualified": False,
        "execution_authority": False,
        "physical_authority": False,
        "physical_completion": False,
    }
    _atomic_json(base / "evidence/field_bundle_v2_install_receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--install-base",
        type=Path,
        default=Path.home() / ".local/share/rootscope-field-v2",
    )
    parser.add_argument(
        "--config-dir", type=Path, default=Path.home() / ".config/rootscope"
    )
    parser.add_argument(
        "--systemd-user-dir",
        type=Path,
        default=Path.home() / ".config/systemd/user",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verify_only:
            payload = verify_bundle(args.bundle_root)
        else:
            payload = install_bundle(
                args.bundle_root,
                args.install_base,
                args.config_dir,
                args.systemd_user_dir,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": "rootscope.x5-field-bundle-install-error.v2",
            "status": "FAIL_CLOSED_NO_AUTHORITY",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "activation_gate_created": False,
            "service_started": False,
            "systemctl_invoked": False,
            "network_touched": False,
            "device_enumerated": False,
            "serial_write": False,
            "pump_command": False,
            "bpu_model_loaded": False,
            "bpu_forward_executed": False,
            "camera_opened": False,
            "x5_validated": False,
            "model_qualified": False,
            "execution_authority": False,
            "physical_authority": False,
        }
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
