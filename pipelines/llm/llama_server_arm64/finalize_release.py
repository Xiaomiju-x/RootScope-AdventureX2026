#!/usr/bin/env python3
"""Assemble and hash-freeze the RootScope Ubuntu 22.04/aarch64 llama-server release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import tarfile
from pathlib import Path


TAG = "b9637"
COMMIT = "aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3"
SOURCE_DATE_EPOCH = 1781461060
RELEASE_ID = "rootscope_llama_server_arm64_b9637_v1"
AMD64_BASE_DIGEST = "sha256:0e0a0fc6d18feda9db1590da249ac93e8d5abfea8f4c3c0c849ce512b5ef8982"
ARM64_BASE_DIGEST = "sha256:7d6d8dd7b545fa3b9a0dd6f33eb981e697996ed533ae2c0b2b7b5ac99fb9dafe"
MODEL_SHA256 = "6c8adc95de81b1d0103fcf900a5f369d5936d234e42ed64881dad56e0104e77b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: list[str], *, stdout: Path | None = None) -> str:
    if stdout is None:
        return subprocess.check_output(args, text=True, encoding="utf-8").strip()
    with stdout.open("wb") as handle:
        subprocess.run(args, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return ""


def docker_image_id(name: str) -> str:
    return run(["docker", "image", "inspect", name, "--format", "{{.Id}}"])


def parse_dynamic(path: Path) -> tuple[list[str], list[str], list[str]]:
    text = path.read_text(encoding="utf-8")
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", text)
    rpath = re.findall(r"\(RPATH\).*?\[(.*?)\]", text)
    runpath = re.findall(r"\(RUNPATH\).*?\[(.*?)\]", text)
    return needed, rpath, runpath


def versions(path: Path, prefix: str) -> list[str]:
    values = set(re.findall(rf"\b{re.escape(prefix)}_(\d+(?:\.\d+)*)\b", path.read_text(encoding="utf-8")))
    return sorted(values, key=lambda value: tuple(int(part) for part in value.split(".")))


def write_launcher(path: Path, server_sha256: str) -> None:
    text = f"""#!/usr/bin/env bash
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
SERVER="$SERVER_DIR/llama-server"
MODEL="${{1:-}}"
GATE="${{ROOTSCOPE_LLM_GATE_FILE:-}}"

[[ "$(uname -s)" == "Linux" ]] || {{ echo "Linux required" >&2; exit 64; }}
[[ "$(uname -m)" == "aarch64" ]] || {{ echo "aarch64 required" >&2; exit 64; }}
[[ -n "$MODEL" && -f "$MODEL" && ! -L "$MODEL" ]] || {{ echo "regular GGUF path required" >&2; exit 64; }}
[[ "${{ROOTSCOPE_LLM_MANUAL_ACK:-}}" == "READ_ONLY_EXPLANATION_ONLY" ]] || {{ echo "manual acknowledgement missing" >&2; exit 65; }}
[[ -n "$GATE" && -f "$GATE" ]] || {{ echo "manual gate file missing" >&2; exit 65; }}
[[ "$(tr -d '\\r\\n' < "$GATE")" == "READ_ONLY_EXPLANATION_ONLY" ]] || {{ echo "manual gate mismatch" >&2; exit 65; }}
[[ "$(sha256sum "$SERVER" | awk '{{print $1}}')" == "{server_sha256}" ]] || {{ echo "llama-server SHA mismatch" >&2; exit 66; }}
[[ "$(sha256sum "$MODEL" | awk '{{print $1}}')" == "{MODEL_SHA256}" ]] || {{ echo "GGUF SHA mismatch" >&2; exit 66; }}

exec "$SERVER" --model "$MODEL" --host 127.0.0.1 --port 9080 --ctx-size 2048 --threads 2 --parallel 1 --no-ui
"""
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def write_readme(path: Path, server_sha256: str, needed: list[str], glibc_max: str | None) -> None:
    path.write_text(
        f"""# RootScope llama-server aarch64 b9637

This is a source-frozen cross-build of official `ggml-org/llama.cpp` tag `{TAG}` / commit `{COMMIT}` for Ubuntu 22.04 aarch64. It uses conservative `armv8-a`; it does not claim unverified X5 CPU extensions.

Artifact SHA-256: `{server_sha256}`. Dynamic dependencies: `{', '.join(needed)}`. Highest referenced GLIBC symbol: `{glibc_max or 'none'}`. libstdc++ and libgcc are statically linked; no release `.so` and no RPATH/RUNPATH are required.

Verify with `sha256sum -c SHA256SUMS`. On an RDK X5, run `bin/llama-server --version`, then pass this exact binary and SHA to RootScope `install_readonly_llm.py`. Installation remains disabled until its manual acknowledgement and gate are created.

`bin/run_loopback_readonly.sh MODEL.gguf` is an additional fail-closed manual launcher. It requires `ROOTSCOPE_LLM_MANUAL_ACK=READ_ONLY_EXPLANATION_ONLY` and a gate file named by `ROOTSCOPE_LLM_GATE_FILE` containing the same exact text. It always binds `127.0.0.1:9080`, disables UI, and verifies both server and GGUF hashes.

The included smoke receipt proves only an Ubuntu 22.04 arm64 QEMU run with Docker `--network none`, loopback `/health`, and one minimal completion. It is `CROSS_BUILD_QEMU_SMOKE_NOT_X5_VALIDATION`: X5 execution, latency, thermals, and service qualification remain pending.
""",
        encoding="utf-8",
        newline="\n",
    )


def deterministic_tar(release: Path, target: Path) -> None:
    with tarfile.open(target, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted([release, *release.rglob("*")], key=lambda item: item.relative_to(release.parent).as_posix()):
            relative = path.relative_to(release.parent).as_posix()
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = SOURCE_DATE_EPOCH
            if path.is_dir():
                info.mode = 0o755
                archive.addfile(info)
            else:
                executable = path.name in {"llama-server", "run_loopback_readonly.sh"} or path.suffix == ".sh"
                info.mode = 0o755 if executable else 0o644
                with path.open("rb") as handle:
                    archive.addfile(info, handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-stage", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--cross-image", default="rootscope/llama-server-cross:ubuntu22-b9637-r3")
    parser.add_argument("--smoke-image", default="rootscope/llama-server-smoke:ubuntu22-arm64-v1")
    args = parser.parse_args()

    stage = args.build_stage.resolve()
    source = args.source_repo.resolve()
    model = args.model.resolve()
    release = args.release_dir.resolve()
    tools = args.tools_dir.resolve()
    if release.exists():
        raise SystemExit(f"release directory already exists: {release}")
    if sha256_file(model) != MODEL_SHA256:
        raise SystemExit("model SHA mismatch before QEMU smoke")
    if run(["git", "-C", str(source), "rev-parse", "HEAD"]) != COMMIT:
        raise SystemExit("source commit mismatch")

    shutil.copytree(stage, release)
    (release / "source").mkdir()
    (release / "repro").mkdir()
    source_tar = release / "source" / f"llama.cpp-{TAG}-{COMMIT[:12]}.tar"
    run(["git", "-C", str(source), "archive", "--format=tar", f"--prefix=llama.cpp-{TAG}/", "-o", str(source_tar), COMMIT])
    for item in sorted(tools.iterdir()):
        if item.is_file():
            shutil.copy2(item, release / "repro" / item.name)

    cross_id = docker_image_id(args.cross_image)
    smoke_id = docker_image_id(args.smoke_image)
    run(
        [
            "docker", "run", "--rm", "--platform", "linux/amd64",
            "-v", f"{release}:/release", args.cross_image,
            "aarch64-linux-gnu-readelf", "--version-info", "/release/bin/llama-server",
        ],
        stdout=release / "metadata" / "readelf_version_info.txt",
    )
    subprocess.run(
        [
            "docker", "run", "--rm", "--platform", "linux/arm64", "--network", "none",
            "-v", f"{release}:/release",
            "-v", f"{model}:/model/qwen2_05b_distill.Q4_K_M.gguf:ro",
            "-v", f"{tools}:/tools:ro",
            args.smoke_image, "bash", "/tools/qemu_smoke.sh",
        ],
        check=True,
    )

    server = release / "bin" / "llama-server"
    server_sha = sha256_file(server)
    needed, rpath, runpath = parse_dynamic(release / "metadata" / "readelf_dynamic.txt")
    glibc = versions(release / "metadata" / "readelf_version_info.txt", "GLIBC")
    glibcxx = versions(release / "metadata" / "readelf_version_info.txt", "GLIBCXX")
    write_launcher(release / "bin" / "run_loopback_readonly.sh", server_sha)
    write_readme(release / "README.md", server_sha, needed, glibc[-1] if glibc else None)

    header = server.read_bytes()[:64]
    elf_machine = struct.unpack_from("<H", header, 18)[0]
    manifest = {
        "schema": "rootscope.llama_server_arm64_release.v1",
        "release_id": RELEASE_ID,
        "status": "CROSS_BUILD_QEMU_SMOKE_PASS_NOT_X5_VALIDATED",
        "source": {
            "repository": "https://github.com/ggml-org/llama.cpp.git",
            "official_release": f"https://github.com/ggml-org/llama.cpp/releases/tag/{TAG}",
            "official_release_workflow": "https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/release.yml",
            "tag": TAG,
            "commit": COMMIT,
            "source_date_epoch": SOURCE_DATE_EPOCH,
            "archive": str(source_tar.relative_to(release)).replace(os.sep, "/"),
            "archive_sha256": sha256_file(source_tar),
            "archive_size_bytes": source_tar.stat().st_size,
        },
        "build": {
            "kind": "AMD64_UBUNTU_22_04_GCC11_AARCH64_CROSS_BUILD",
            "target": "linux/aarch64",
            "cpu_baseline": "armv8-a",
            "openssl": False,
            "https": False,
            "openmp": False,
            "shared_libraries_built": False,
            "libstdcxx_static": True,
            "libgcc_static": True,
            "amd64_base_image_digest": AMD64_BASE_DIGEST,
            "cross_toolchain_image": args.cross_image,
            "cross_toolchain_image_id": cross_id,
        },
        "artifact": {
            "path": "bin/llama-server",
            "sha256": server_sha,
            "size_bytes": server.stat().st_size,
            "elf_class": 64 if header[4] == 2 else None,
            "elf_little_endian": header[5] == 1,
            "elf_machine_id": elf_machine,
            "elf_machine": "AArch64" if elf_machine == 183 else "UNKNOWN",
            "needed": needed,
            "rpath": rpath,
            "runpath": runpath,
            "glibc_versions": glibc,
            "glibc_max": glibc[-1] if glibc else None,
            "glibcxx_versions": glibcxx,
            "glibcxx_max": glibcxx[-1] if glibcxx else None,
        },
        "qemu_smoke": {
            "status": "PASS",
            "scope": "CROSS_BUILD_QEMU_SMOKE_NOT_X5_VALIDATION",
            "platform": "linux/arm64",
            "network_mode": "none",
            "base_image_digest": ARM64_BASE_DIGEST,
            "smoke_image": args.smoke_image,
            "smoke_image_id": smoke_id,
            "version_executed": True,
            "loopback_health_passed": True,
            "minimal_completion_passed": True,
            "model_sha256": MODEL_SHA256,
        },
        "formal_flags": {
            "hardware_touched": False,
            "network_configuration_touched": False,
            "service_started_on_host_or_x5": False,
            "x5_validated": False,
            "latency_measured_on_x5": False,
            "thermal_measured_on_x5": False,
            "model_qualified": False,
            "execution_authority": False,
            "physical_authority": False,
            "actuator_access": False,
            "tool_execution": False,
            "external_network_runtime_allowed": False,
        },
    }
    manifest_path = release / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    members = sorted(path for path in release.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    sums = "".join(f"{sha256_file(path)}  {path.relative_to(release).as_posix()}\n" for path in members)
    (release / "SHA256SUMS").write_text(sums, encoding="ascii", newline="\n")

    tar_path = release.with_suffix(".tar")
    deterministic_tar(release, tar_path)
    tar_path.with_suffix(".tar.sha256").write_text(f"{sha256_file(tar_path)}  {tar_path.name}\n", encoding="ascii", newline="\n")
    print(json.dumps({"release": str(release), "server_sha256": server_sha, "tar": str(tar_path), "tar_sha256": sha256_file(tar_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
