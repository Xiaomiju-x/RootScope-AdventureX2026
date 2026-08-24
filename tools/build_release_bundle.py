#!/usr/bin/env python3
"""Build deterministic, checksummed RootScope public release bundles.

The builder cannot bypass publication, model, or licence gates. It separates
portable source, public evidence/media, and the four reviewed LFS model
artifacts so downstream users can fetch only what they need.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}\Z")
RELEASE_SCHEMA = "rootscope.public-release.v1"
SBOM_SCHEMA = "SPDX-2.3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha1_file(path: Path) -> str:
    """Return SHA-1 only for the SPDX package verification-code algorithm."""

    digest = hashlib.sha1()  # noqa: S324 - required by SPDX 2.3 section 7.9
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_gate(script: str) -> None:
    command = [sys.executable, str(ROOT / "tools" / script)]
    if script == "audit_public_release.py":
        command.append("--include-git-history")
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(f"release gate failed: tools/{script}")


def run_release_gates() -> None:
    for script in (
        "audit_public_release.py",
        "verify_model_assets.py",
        "check_license_inventory.py",
    ):
        run_gate(script)


def source_date_epoch() -> int:
    configured = os.environ.get("SOURCE_DATE_EPOCH")
    if configured is not None:
        try:
            value = int(configured)
        except ValueError as error:
            raise SystemExit("SOURCE_DATE_EPOCH must be an integer") from error
        if value < 0:
            raise SystemExit("SOURCE_DATE_EPOCH must not be negative")
        return value
    try:
        value = subprocess.run(
            ["git", "-C", str(ROOT), "show", "-s", "--format=%ct", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="ascii",
        ).stdout.strip()
        return int(value)
    except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError, ValueError):
        return 0


def git_revision() -> str:
    try:
        revision = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="ascii",
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
        return "0" * 40
    return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else "0" * 40


def public_files() -> list[Path]:
    # Build from the Git publication candidate: tracked files plus untracked,
    # non-ignored additions. The mandatory scanner still inspects ignored
    # local material for leakage, but caches and private collections can never
    # enter an archive merely because they happen to exist in the worktree.
    try:
        raw = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit("Git file inventory is required for a release") from error
    try:
        relatives = sorted(value.decode("utf-8") for value in raw.split(b"\0") if value)
    except UnicodeDecodeError as error:
        raise SystemExit("release path is not valid UTF-8") from error
    files: list[Path] = []
    for relative in relatives:
        path = ROOT / Path(*PurePosixPath(relative).parts)
        if not path.is_file():
            raise SystemExit(f"release candidate is missing a Git path: {relative}")
        files.append(path)
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            raise SystemExit(f"release refuses symbolic link: {relative}")
        if "\n" in relative or "\r" in relative:
            raise SystemExit("release refuses a path containing a newline")
    return files


def bundle_role(relative: str) -> str:
    if relative.startswith("model-assets/"):
        return "models"
    if relative.startswith("evidence/") or relative.startswith("assets/media/"):
        return "evidence"
    return "source"


def archive_mode(relative: str) -> int:
    return 0o755 if PurePosixPath(relative).suffix in {".sh"} else 0o644


def write_deterministic_tar_gz(
    destination: Path,
    files: Iterable[Path],
    prefix: str,
    epoch: int,
) -> int:
    selected = list(files)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in selected:
                    relative = path.relative_to(ROOT).as_posix()
                    payload = path.read_bytes()
                    info = tarfile.TarInfo(f"{prefix}/{relative}")
                    info.size = len(payload)
                    info.mtime = epoch
                    info.mode = archive_mode(relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.pax_headers = {}
                    archive.addfile(info, fileobj=_BytesReader(payload))
    return len(selected)


class _BytesReader:
    """Minimal read-only stream accepted by tarfile.addfile."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.payload) - self.offset
        start = self.offset
        self.offset = min(len(self.payload), self.offset + size)
        return self.payload[start : self.offset]


def licence_for_path(relative: str) -> str:
    if relative.startswith("model-assets/"):
        return "Apache-2.0"
    if relative.startswith("hardware/design/"):
        return "CERN-OHL-S-2.0"
    if relative.startswith("assets/print-cards/") or relative.startswith("LICENSES/"):
        return "NOASSERTION"
    if relative.startswith("firmware/stm32f103-v15/Drivers/STM32F1xx_HAL_Driver/"):
        return "BSD-3-Clause"
    if relative.startswith("firmware/stm32f103-v15/Drivers/CMSIS/Include/"):
        return "Apache-2.0"
    if relative.startswith("firmware/stm32f103-v15/Drivers/CMSIS/Device/ST/"):
        return "NOASSERTION"
    if relative.startswith("firmware/stm32f103-v15/") and not relative.startswith(
        "firmware/stm32f103-v15/Tests/"
    ):
        return "NOASSERTION"
    if relative.startswith(("docs/", "assets/", "evidence/")) or relative.endswith((".md", ".cff")):
        return "CC-BY-4.0"
    return "Apache-2.0"


def package_for_path(relative: str) -> str:
    if relative.startswith("model-assets/"):
        return "SPDXRef-Package-Models"
    if relative.startswith("hardware/design/"):
        return "SPDXRef-Package-Hardware"
    if relative.startswith(("docs/", "assets/", "evidence/")) or relative.endswith((".md", ".cff")):
        return "SPDXRef-Package-Content"
    if relative.startswith("firmware/"):
        return "SPDXRef-Package-Firmware"
    return "SPDXRef-Package-Software"


def spdx_id(relative: str) -> str:
    return "SPDXRef-File-" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]


def build_spdx(files: list[Path], version: str, revision: str, epoch: int) -> dict[str, object]:
    created = dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    packages = [
        ("SPDXRef-Package-Software", "RootScope public software", "Apache-2.0"),
        ("SPDXRef-Package-Firmware", "RootScope STM32F103 V15 firmware", "NOASSERTION"),
        ("SPDXRef-Package-Hardware", "RootScope hardware design", "CERN-OHL-S-2.0"),
        ("SPDXRef-Package-Content", "RootScope documentation and public evidence", "CC-BY-4.0"),
        ("SPDXRef-Package-Models", "RootScope reviewed model artifacts", "Apache-2.0"),
    ]
    package_records = [
        {
            "SPDXID": identifier,
            "name": name,
            "versionInfo": version,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": True,
            "licenseConcluded": licence,
            "licenseDeclared": licence,
            "copyrightText": "NOASSERTION",
            "supplier": "Organization: RootScope Team",
        }
        for identifier, name, licence in packages
    ]
    package_by_id = {record["SPDXID"]: record for record in package_records}
    verification_inputs: dict[str, list[str]] = {identifier: [] for identifier, _, _ in packages}
    file_records: list[dict[str, object]] = []
    relationships: list[dict[str, str]] = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": identifier,
        }
        for identifier, _, _ in packages
    ]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        identifier = spdx_id(relative)
        licence = licence_for_path(relative)
        package = package_for_path(relative)
        file_sha1 = sha1_file(path)
        file_records.append(
            {
                "SPDXID": identifier,
                "fileName": f"./{relative}",
                "checksums": [
                    {"algorithm": "SHA1", "checksumValue": file_sha1},
                    {"algorithm": "SHA256", "checksumValue": sha256_file(path)},
                ],
                "licenseConcluded": licence,
                "licenseInfoInFiles": [licence],
                "copyrightText": "NOASSERTION",
            }
        )
        relationships.append(
            {
                "spdxElementId": package,
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": identifier,
            }
        )
        verification_inputs[package].append(file_sha1)
    for package, checksums in verification_inputs.items():
        if not checksums:
            continue
        verification = hashlib.sha1(  # noqa: S324 - SPDX 2.3 mandates SHA-1 here
            "".join(sorted(checksums)).encode("ascii")
        ).hexdigest()
        package_by_id[package]["packageVerificationCode"] = {
            "packageVerificationCodeValue": verification
        }
    namespace_version = urllib.parse.quote(version, safe="._+-")
    return {
        "spdxVersion": SBOM_SCHEMA,
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"RootScope-AdventureX2026-{version}",
        "documentNamespace": (
            "https://github.com/Xiaomiju-x/RootScope-AdventureX2026/"
            f"releases/{namespace_version}/spdx/{revision}"
        ),
        "creationInfo": {
            "created": created,
            "creators": ["Tool: RootScope tools/build_release_bundle.py"],
            "licenseListVersion": "3.25",
        },
        "documentDescribes": [identifier for identifier, _, _ in packages],
        "packages": package_records,
        "files": file_records,
        "relationships": relationships,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--version", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not VERSION_PATTERN.fullmatch(args.version):
        raise SystemExit("version must be a portable 1-64 character release identifier")
    run_release_gates()
    files = public_files()
    if not files:
        raise SystemExit("public tree is empty")

    output = args.output.resolve()
    if output == ROOT.resolve():
        raise SystemExit("output directory must not be the repository root")
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)

    epoch = source_date_epoch()
    revision = git_revision()
    prefix = f"RootScope-AdventureX2026-{args.version}"
    bundle_records: list[dict[str, object]] = []
    for role in ("source", "evidence", "models"):
        selected = [path for path in files if bundle_role(path.relative_to(ROOT).as_posix()) == role]
        if not selected:
            raise SystemExit(f"release bundle would be empty: {role}")
        destination = output / f"rootscope-{args.version}-{role}.tar.gz"
        count = write_deterministic_tar_gz(destination, selected, prefix, epoch)
        bundle_records.append(
            {
                "bytes": destination.stat().st_size,
                "files": count,
                "name": destination.name,
                "role": role,
                "sha256": sha256_file(destination),
            }
        )

    sbom_path = output / f"rootscope-{args.version}.spdx.json"
    write_json(sbom_path, build_spdx(files, args.version, revision, epoch))
    manifest_path = output / "RELEASE_MANIFEST.json"
    manifest = {
        "schema": RELEASE_SCHEMA,
        "version": args.version,
        "revision": revision,
        "source_date_epoch": epoch,
        "bundles": bundle_records,
        "sbom": {
            "bytes": sbom_path.stat().st_size,
            "format": SBOM_SCHEMA,
            "name": sbom_path.name,
            "sha256": sha256_file(sbom_path),
        },
    }
    write_json(manifest_path, manifest)

    checksummed = [
        *(output / str(record["name"]) for record in bundle_records),
        sbom_path,
        manifest_path,
    ]
    checksums_path = output / "SHA256SUMS"
    checksums_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in sorted(checksummed)),
        encoding="ascii",
        newline="\n",
    )
    print(
        "RELEASE_BUILD=PASS "
        f"version={args.version} bundles={len(bundle_records)} files={len(files)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
