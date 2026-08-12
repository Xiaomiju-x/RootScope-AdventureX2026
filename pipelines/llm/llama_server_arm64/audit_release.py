#!/usr/bin/env python3
"""Independent, read-only audit for the frozen RootScope arm64 llama-server release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import tarfile
from pathlib import Path
from typing import Any


EXPECTED_RELEASE_ID = "rootscope_llama_server_arm64_b9637_v1"
EXPECTED_COMMIT = "aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3"
EXPECTED_TAG = "b9637"
EXPECTED_MODEL_SHA = "6c8adc95de81b1d0103fcf900a5f369d5936d234e42ed64881dad56e0104e77b"
EXPECTED_NEEDED = ["libm.so.6", "libc.so.6", "ld-linux-aarch64.so.1"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any = None) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks)


def parse_sha256sums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"invalid SHA256SUMS line: {line!r}")
        values[match.group(2)] = match.group(1)
    return values


def version_tuple(value: str | None) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(".")) if value else ()


def audit_release(release: Path, archive: Path | None = None) -> dict[str, Any]:
    release = release.resolve()
    audit = Audit()
    manifest_path = release / "release_manifest.json"
    sums_path = release / "SHA256SUMS"
    audit.check("release_directory_exists", release.is_dir(), str(release))
    audit.check("manifest_exists", manifest_path.is_file())
    audit.check("sha256sums_exists", sums_path.is_file())
    if not manifest_path.is_file() or not sums_path.is_file():
        return {"status": "FAIL", "checks": audit.checks}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit.check("schema", manifest.get("schema") == "rootscope.llama_server_arm64_release.v1", manifest.get("schema"))
    audit.check("release_id", manifest.get("release_id") == EXPECTED_RELEASE_ID, manifest.get("release_id"))
    audit.check("status_boundary", manifest.get("status") == "CROSS_BUILD_QEMU_SMOKE_PASS_NOT_X5_VALIDATED", manifest.get("status"))

    source = manifest.get("source", {})
    audit.check("source_repo_official", source.get("repository") == "https://github.com/ggml-org/llama.cpp.git")
    audit.check("source_tag", source.get("tag") == EXPECTED_TAG, source.get("tag"))
    audit.check("source_commit", source.get("commit") == EXPECTED_COMMIT, source.get("commit"))
    audit.check("source_release_url", source.get("official_release") == f"https://github.com/ggml-org/llama.cpp/releases/tag/{EXPECTED_TAG}")
    audit.check("source_workflow_url_official", str(source.get("official_release_workflow", "")).startswith("https://github.com/ggml-org/llama.cpp/"))
    source_archive = release / str(source.get("archive", ""))
    audit.check("source_archive_exists", source_archive.is_file(), str(source_archive))
    if source_archive.is_file():
        audit.check("source_archive_sha", sha256_file(source_archive) == source.get("archive_sha256"), sha256_file(source_archive))
        audit.check("source_archive_size", source_archive.stat().st_size == source.get("archive_size_bytes"), source_archive.stat().st_size)
        with tarfile.open(source_archive, "r") as source_tar:
            names = source_tar.getnames()
        audit.check("source_archive_nonempty", len(names) > 1000, len(names))
        audit.check("source_archive_prefix", all(name == f"llama.cpp-{EXPECTED_TAG}" or name.startswith(f"llama.cpp-{EXPECTED_TAG}/") for name in names))
        audit.check("source_archive_no_git_metadata", not any(".git" in Path(name).parts for name in names))
        audit.check("source_archive_license", f"llama.cpp-{EXPECTED_TAG}/LICENSE" in names)

    artifact = manifest.get("artifact", {})
    server = release / str(artifact.get("path", ""))
    audit.check("server_exists", server.is_file(), str(server))
    if server.is_file():
        server_hash = sha256_file(server)
        audit.check("server_sha", server_hash == artifact.get("sha256"), server_hash)
        audit.check("server_size", server.stat().st_size == artifact.get("size_bytes"), server.stat().st_size)
        header = server.read_bytes()[:64]
        audit.check("elf_magic", header[:4] == b"\x7fELF")
        audit.check("elf_64", len(header) >= 64 and header[4] == 2 and artifact.get("elf_class") == 64)
        audit.check("elf_little_endian", len(header) >= 64 and header[5] == 1 and artifact.get("elf_little_endian") is True)
        machine = struct.unpack_from("<H", header, 18)[0] if len(header) >= 20 else -1
        audit.check("elf_machine_aarch64", machine == 183 and artifact.get("elf_machine") == "AArch64", machine)
        binary = server.read_bytes()
        forbidden = [b"C:\\Users\\", b"xrd_backup", b"/out/"]
        for pattern in forbidden:
            audit.check(f"binary_no_host_path_{pattern.decode('ascii').replace('/', '_')}", pattern not in binary)
        printable = re.findall(rb"[\x20-\x7e]{4,}", binary)
        bad_source_paths = [
            value.decode("ascii", errors="replace")
            for value in printable
            if b"/src/" in value and b"/usr/src/llama.cpp-b9637/" not in value
        ]
        audit.check("binary_only_sanitized_source_prefix", not bad_source_paths, bad_source_paths[:20])

    needed = artifact.get("needed")
    audit.check("needed_exact_ubuntu_runtime", needed == EXPECTED_NEEDED, needed)
    audit.check("no_libstdcxx_needed", "libstdc++.so.6" not in (needed or []))
    audit.check("no_libgcc_needed", "libgcc_s.so.1" not in (needed or []))
    audit.check("no_libgomp_needed", "libgomp.so.1" not in (needed or []))
    audit.check("no_rpath", artifact.get("rpath") == [], artifact.get("rpath"))
    audit.check("no_runpath", artifact.get("runpath") == [], artifact.get("runpath"))
    audit.check("glibc_max_present", bool(artifact.get("glibc_max")), artifact.get("glibc_max"))
    audit.check("glibc_max_ubuntu22_compatible", version_tuple(artifact.get("glibc_max")) <= (2, 35), artifact.get("glibc_max"))
    audit.check("no_glibcxx_dynamic_symbols", artifact.get("glibcxx_versions") == [], artifact.get("glibcxx_versions"))

    build = manifest.get("build", {})
    audit.check("cross_build_kind", build.get("kind") == "AMD64_UBUNTU_22_04_GCC11_AARCH64_CROSS_BUILD")
    audit.check("target_linux_aarch64", build.get("target") == "linux/aarch64")
    audit.check("conservative_armv8_baseline", build.get("cpu_baseline") == "armv8-a")
    for key in ("openssl", "https", "openmp", "shared_libraries_built"):
        audit.check(f"build_{key}_false", build.get(key) is False, build.get(key))
    audit.check("libstdcxx_static", build.get("libstdcxx_static") is True)
    audit.check("libgcc_static", build.get("libgcc_static") is True)
    audit.check("base_image_digest_frozen", re.fullmatch(r"sha256:[0-9a-f]{64}", str(build.get("amd64_base_image_digest", ""))) is not None)
    audit.check("cross_image_id_frozen", re.fullmatch(r"sha256:[0-9a-f]{64}", str(build.get("cross_toolchain_image_id", ""))) is not None)

    metadata = release / "metadata"
    required_metadata = [
        "file.txt", "readelf_header.txt", "readelf_dynamic.txt", "readelf_notes.txt",
        "readelf_version_info.txt", "objdump_private_headers.txt", "toolchain_versions.txt",
        "cmake_config.log", "build.log", "qemu_version.txt", "qemu_ldd.txt",
        "qemu_health_headers.txt", "qemu_health_response.txt",
        "qemu_completion_headers.txt", "qemu_completion_response.txt", "qemu_server.log",
    ]
    for name in required_metadata:
        audit.check(f"metadata_{name}_exists", (metadata / name).is_file(), name)

    def read(name: str) -> str:
        path = metadata / name
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""

    audit.check("file_report_aarch64", "ARM aarch64" in read("file.txt"))
    audit.check("file_report_interpreter", "/lib/ld-linux-aarch64.so.1" in read("file.txt"))
    audit.check("readelf_report_aarch64", "Machine:" in read("readelf_header.txt") and "AArch64" in read("readelf_header.txt"))
    audit.check("readelf_no_rpath_text", "(RPATH)" not in read("readelf_dynamic.txt"))
    audit.check("readelf_no_runpath_text", "(RUNPATH)" not in read("readelf_dynamic.txt"))
    audit.check("toolchain_ubuntu2204", "BASE_OS=ubuntu 22.04" in read("toolchain_versions.txt"))
    audit.check("toolchain_gcc11", "11.4.0" in read("toolchain_versions.txt"))
    audit.check("cmake_commit_exact", EXPECTED_COMMIT in read("cmake_config.log"))
    audit.check("cmake_armv8a", "-march=armv8-a" in read("cmake_config.log"))
    audit.check("build_target_linked", "Linking CXX executable bin/llama-server" in read("build.log"))

    qemu = manifest.get("qemu_smoke", {})
    audit.check("qemu_scope_boundary", qemu.get("scope") == "CROSS_BUILD_QEMU_SMOKE_NOT_X5_VALIDATION")
    audit.check("qemu_status", qemu.get("status") == "PASS")
    audit.check("qemu_network_none", qemu.get("network_mode") == "none")
    audit.check("qemu_platform_arm64", qemu.get("platform") == "linux/arm64")
    audit.check("qemu_model_sha", qemu.get("model_sha256") == EXPECTED_MODEL_SHA)
    audit.check("qemu_version_exact", f"version: 9637 ({EXPECTED_COMMIT})" in read("qemu_version.txt"))
    ldd_text = read("qemu_ldd.txt")
    audit.check("qemu_ldd_no_missing", "not found" not in ldd_text.lower(), ldd_text)
    for library in EXPECTED_NEEDED:
        audit.check(f"qemu_ldd_{library}", library in ldd_text, ldd_text)
    health_headers = read("qemu_health_headers.txt")
    health_body = read("qemu_health_response.txt")
    audit.check("qemu_health_http200", " 200 " in health_headers, health_headers)
    audit.check("qemu_health_json_ok", json.loads(health_body).get("status") == "ok" if health_body else False, health_body)
    completion_headers = read("qemu_completion_headers.txt")
    completion_body = read("qemu_completion_response.txt")
    audit.check("qemu_completion_http200", " 200 " in completion_headers, completion_headers)
    try:
        completion = json.loads(completion_body)
    except json.JSONDecodeError:
        completion = {}
    choices = completion.get("choices", []) if isinstance(completion, dict) else []
    audit.check("qemu_completion_one_choice", isinstance(choices, list) and len(choices) == 1, choices)
    content = choices[0].get("message", {}).get("content") if choices else None
    audit.check("qemu_completion_nonempty", isinstance(content, str) and bool(content.strip()), content)
    audit.check("qemu_completion_fingerprint", completion.get("system_fingerprint") == f"b9637-{EXPECTED_COMMIT}", completion.get("system_fingerprint"))
    qemu_log = read("qemu_server.log")
    audit.check("qemu_log_model_loaded", "model loaded" in qemu_log)
    audit.check("qemu_log_loopback_listening", "http://127.0.0.1:19080" in qemu_log)
    audit.check("qemu_log_completion_timing", "prompt eval time" in qemu_log)

    flags = manifest.get("formal_flags", {})
    audit.check("formal_flags_present", bool(flags), flags)
    for key, value in flags.items():
        audit.check(f"formal_flag_{key}_false", value is False, value)
    audit.check("formal_x5_not_validated", flags.get("x5_validated") is False)
    audit.check("formal_no_authority", flags.get("execution_authority") is False and flags.get("physical_authority") is False)

    launcher = release / "bin" / "run_loopback_readonly.sh"
    launcher_text = launcher.read_text(encoding="utf-8") if launcher.is_file() else ""
    audit.check("launcher_exists", launcher.is_file())
    audit.check("launcher_loopback_only", "--host 127.0.0.1" in launcher_text and "0.0.0.0" not in launcher_text)
    audit.check("launcher_server_hash_frozen", artifact.get("sha256") in launcher_text)
    audit.check("launcher_model_hash_frozen", EXPECTED_MODEL_SHA in launcher_text)
    audit.check("launcher_manual_gate", "ROOTSCOPE_LLM_MANUAL_ACK" in launcher_text and "ROOTSCOPE_LLM_GATE_FILE" in launcher_text)
    audit.check("launcher_no_tools_or_actuators", not re.search(r"serial|gpio|pump|actuator|/dev/tty", launcher_text, re.IGNORECASE))

    shared_objects = [path.relative_to(release).as_posix() for path in release.rglob("*.so*") if path.is_file()]
    audit.check("no_bundled_shared_objects", not shared_objects, shared_objects)
    sums = parse_sha256sums(sums_path)
    actual_members = sorted(path.relative_to(release).as_posix() for path in release.rglob("*") if path.is_file() and path != sums_path)
    audit.check("sha_member_set_exact", sorted(sums) == actual_members, {"listed": len(sums), "actual": len(actual_members)})
    for relative, expected in sums.items():
        path = release / relative
        audit.check(f"sha_{relative}", path.is_file() and sha256_file(path) == expected, relative)

    if archive is not None:
        archive = archive.resolve()
        audit.check("release_tar_exists", archive.is_file(), str(archive))
        if archive.is_file():
            with tarfile.open(archive, "r") as handle:
                members = handle.getmembers()
            audit.check("release_tar_root", all(member.name == release.name or member.name.startswith(release.name + "/") for member in members))
            audit.check("release_tar_no_absolute", not any(member.name.startswith("/") or ".." in Path(member.name).parts for member in members))
            audit.check("release_tar_fixed_owner", all(member.uid == 0 and member.gid == 0 for member in members))
            audit.check("release_tar_fixed_mtime", all(member.mtime == source.get("source_date_epoch") for member in members))
            sha_sidecar = archive.with_suffix(".tar.sha256")
            audit.check("release_tar_sidecar", sha_sidecar.is_file() and sha256_file(archive) in sha_sidecar.read_text(encoding="ascii"))

    passed_count = sum(1 for item in audit.checks if item["passed"])
    result = {
        "schema": "rootscope.llama_server_arm64_independent_audit.v1",
        "status": "PASS" if audit.passed else "FAIL",
        "scope": "CROSS_BUILD_QEMU_SMOKE_NOT_X5_VALIDATION",
        "release": str(release),
        "checks_total": len(audit.checks),
        "checks_passed": passed_count,
        "checks_failed": len(audit.checks) - passed_count,
        "checks": audit.checks,
        "formal_conclusion": "HASH_FROZEN_UBUNTU22_ARM64_CROSS_BUILD_QEMU_SMOKE_PASS_X5_BOARD_VALIDATION_PENDING" if audit.passed else "AUDIT_FAILED",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_release(args.release_dir, args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_total", "checks_passed", "checks_failed")}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
