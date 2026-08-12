"""Fail closed when the public tree contains secrets, private identities or drift.

The scanner is dependency-free so it can run before installing project code. It
intentionally inspects ignored files too: a local secret that is ignored by Git
is still unsafe input to a release builder. Findings print only category, path
and line number; the matched value is never echoed into CI logs.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_BYTES = 50 * 1024 * 1024

SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "venv",
}

TEXT_SUFFIXES = {
    "",
    ".cff",
    ".c",
    ".cmake",
    ".cpp",
    ".css",
    ".desktop",
    ".example",
    ".h",
    ".html",
    ".ini",
    ".ioc",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".rules",
    ".s",
    ".service",
    ".sh",
    ".svg",
    ".tag",
    ".toml",
    ".txt",
    ".uvprojx",
    ".xml",
    ".yml",
    ".yaml",
}

# Generated object formats do not belong in ordinary Git. The one reviewed V15
# Intel HEX is permitted only at this exact path and only with this digest.
BANNED_SUFFIXES = {
    ".axf",
    ".bin",
    ".ckpt",
    ".db",
    ".dll",
    ".elf",
    ".engine",
    ".exe",
    ".gguf",
    ".hbm",
    ".key",
    ".onnx",
    ".p12",
    ".pem",
    ".pfx",
    ".pth",
    ".pt",
    ".safetensors",
    ".so",
    ".sqlite",
    ".tflite",
    ".whl",
}
REVIEWED_BINARY_ALLOWLIST = {
    "firmware/stm32f103-v15/release/FLASH_THIS_Z3_PB6_V15.hex": {
        "sha256": "5016b96d138d4ffad2088dd5da288b4d68c5deba781555ad82eb6f7fb4bfd887",
        "bytes": 72141,
    }
}
REVIEWED_MODEL_ARTIFACTS = {
    "model-assets/vision/rootscope_answer_cards_resnet18_opset11.onnx": {
        "sha256": "7ad94a092a86f1c3706a86137b017d501b1b9e62833d68c6a14fab0b110a3b95",
        "bytes": 44704833,
        "lfs": True,
    },
    "model-assets/bpu/rootscope_seed17_resnet18_static_b1x3x224x224_opset11.onnx": {
        "sha256": "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad",
        "bytes": 44704833,
        "lfs": True,
    },
    "model-assets/bpu/rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes.bin": {
        "sha256": "4dc4bc24741332bb6bc6da184dd1d6f954ae539ba740330490339d76eb200285",
        "bytes": 12519415,
        "lfs": True,
    },
    "model-assets/rootmind-adapter/adapter_model.safetensors": {
        "sha256": "5720045c92e88096e1b3e6dc819e59e14b4ae2aac2c47e0ac8e0d5d7a1bd67c1",
        "bytes": 34916720,
        "lfs": True,
    },
}

REQUIRED_GOVERNANCE_FILES = {
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "LICENSES/BSD-3-Clause.txt",
    "LICENSES/CC-BY-4.0.txt",
    "LICENSES/CC-BY-SA-4.0.txt",
    "LICENSES/CERN-OHL-S-2.0.txt",
    "LICENSE_MATRIX.md",
    "NOTICE",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
}

SECRET_PATTERNS = {
    "github_token": re.compile(r"\b(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}\b"),
    "generic_sk_token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA |PGP )?PRIVATE KEY-----"),
    "url_userinfo": re.compile(r"\bhttps?://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
    "assigned_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
        r"password|passwd|private[_-]?key|secret)\b\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']"
    ),
}

ABSOLUTE_USER_PATH_PATTERNS = {
    "windows_user_path": re.compile(
        r"(?i)(?:[A-Z]:[\\/](?:Users|Documents and Settings)[\\/])[^\\/\s`\"']+"
    ),
    "workspace_absolute_path": re.compile(
        r"(?i)[A-Z]:[\\/](?:WorkData|workspace|workspaces|projects)[\\/]"
    ),
    "linux_user_home": re.compile(r"/(?:home|Users)/(?!(?:user|runner|example)(?:/|\b))[^/\s`\"']+/"),
}

IDENTITY_PATTERNS = {
    "ssh_target": re.compile(r"(?i)\b(?:root|admin|ubuntu|sunrise)@[A-Za-z0-9_.-]+"),
    "persistent_device_path": re.compile(r"/dev/serial/by-id/[^\s`\"']+"),
    "usb_topology_identity": re.compile(
        r"(?i)(?:ENV\{)?ID_PATH(?:\})?\s*(?:==|=)\s*[\"']?platform-xhci[^\s`\"']+"
    ),
    "machine_id_value": re.compile(
        r"(?i)[\"']?(?:machine[_-]?id|boot[_-]?id)[\"']?\s*[:=]\s*[\"'][0-9a-f-]{24,}[\"']"
    ),
    "mac_address": re.compile(r"(?i)\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b"),
}

IPV4_PATTERN = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
SCANNER_FIXTURE_PATHS = {
    "tools/audit_public_release.py",  # scanner implementation and unit fixtures
    "tests/test_publication_audit.py",
    "tools/sanitize_public_import.py",  # sanitizer patterns and safe substitutions
}

# These exact source lines exercise downstream rejection. A second secret in
# either test file remains a finding; there is no directory- or file-wide
# exemption.
INTENTIONAL_SECRET_FIXTURES = {
    "pipelines/release/tests/test_rootscope_omega_v3_delta.py": (
        re.compile(
            r'''^\s*'api_key = "abcdefghijklmnopqrstuvwxyz123456"\\n',\s*$'''
        ),
    ),
    "software/rdk-x5/tests/test_read_only_explainer.py": (
        re.compile(r'''^\s*"http://user:pass@127[.]0[.]0[.]1:9080",\s*$'''),
    ),
}

SAFE_EXAMPLE_MACS = {"02:00:00:00:00:01"}
SAFE_EXAMPLE_MACHINE_IDS = {"00000000000000000000000000000001"}
SAFE_DEVICE_PATH_FRAGMENTS = (
    "/dev/serial/by-id/*",
    "/dev/serial/by-id/usb-<",
    "/dev/serial/by-id/usb-vendor_product-serial",
)

# Static OS/tool installation paths contain no user identity and are portable
# fallbacks, so they are not treated as private home/workspace paths.
SAFE_PATH_FRAGMENTS = (
    "C:\\Windows\\Fonts",
    "C:/Windows/Fonts",
    "/home/user/",
    "/home/runner/",
    "/home/example/",
)


@dataclass(frozen=True, order=True)
class Finding:
    category: str
    path: str
    line: int | None = None

    def render(self) -> str:
        location = self.path if self.line is None else f"{self.path}:{self.line}"
        return f"{self.category}: {location}"


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_public_files() -> Iterator[Path]:
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(
            part in SKIP_PARTS or part.endswith(".egg-info") for part in path.parts
        ):
            continue
        yield path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return path.name in {"LICENSE", "NOTICE", ".gitattributes", ".gitignore"}


def read_text(path: Path) -> str | None:
    if not is_text_file(path):
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def is_private_ipv4(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if address.version != 4:
        return False
    # Documentation, loopback, link-local, multicast and unspecified ranges are
    # safe examples. RFC1918 addresses reveal deployment topology and fail.
    if (
        address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address in ipaddress.ip_network("192.0.2.0/24")
        or address in ipaddress.ip_network("198.51.100.0/24")
        or address in ipaddress.ip_network("203.0.113.0/24")
    ):
        return False
    return (
        address in ipaddress.ip_network("10.0.0.0/8")
        or address in ipaddress.ip_network("172.16.0.0/12")
        or address in ipaddress.ip_network("192.168.0.0/16")
        or address in ipaddress.ip_network("100.64.0.0/10")
    )


def scan_text(path: Path, text: str) -> list[Finding]:
    rel = relative(path)
    if rel in SCANNER_FIXTURE_PATHS:
        return []
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        intentional_fixture = any(
            pattern.fullmatch(line)
            for pattern in INTENTIONAL_SECRET_FIXTURES.get(rel, ())
        )
        if not intentional_fixture:
            for category, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append(Finding(category, rel, line_number))
        if not any(fragment in line for fragment in SAFE_PATH_FRAGMENTS):
            for category, pattern in ABSOLUTE_USER_PATH_PATTERNS.items():
                if pattern.search(line):
                    findings.append(Finding(category, rel, line_number))
        for category, pattern in IDENTITY_PATTERNS.items():
            matches = list(pattern.finditer(line))
            if not matches:
                continue
            if category == "mac_address" and all(
                match.group(0).lower() in SAFE_EXAMPLE_MACS for match in matches
            ):
                continue
            if category == "machine_id_value" and any(
                example in line.lower() for example in SAFE_EXAMPLE_MACHINE_IDS
            ):
                continue
            if category == "persistent_device_path" and any(
                fragment in line for fragment in SAFE_DEVICE_PATH_FRAGMENTS
            ):
                continue
            findings.append(Finding(category, rel, line_number))
        if any(is_private_ipv4(match.group(0)) for match in IPV4_PATTERN.finditer(line)):
            findings.append(Finding("private_ipv4", rel, line_number))
    return findings


def scan_files(files: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    seen_rel: set[str] = set()
    for path in files:
        rel = relative(path)
        seen_rel.add(rel)
        if path.is_symlink():
            findings.append(Finding("symlink_not_allowed", rel))
        suffix = path.suffix.lower()
        size = path.stat().st_size
        if size > MAX_PUBLIC_FILE_BYTES:
            findings.append(Finding("oversized_file", rel))
        reviewed_model = REVIEWED_MODEL_ARTIFACTS.get(rel)
        if suffix in BANNED_SUFFIXES and reviewed_model is None:
            findings.append(Finding("banned_artifact", rel))
        if reviewed_model is not None:
            if size != reviewed_model["bytes"] or sha256_file(path) != reviewed_model["sha256"]:
                findings.append(Finding("model_artifact_digest_mismatch", rel))
        if suffix == ".hex":
            expected = REVIEWED_BINARY_ALLOWLIST.get(rel)
            if expected is None:
                findings.append(Finding("unreviewed_firmware_image", rel))
            elif size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
                findings.append(Finding("firmware_allowlist_digest_mismatch", rel))
        text = read_text(path)
        if text is not None:
            findings.extend(scan_text(path, text))

    for required in sorted(REQUIRED_GOVERNANCE_FILES - seen_rel):
        findings.append(Finding("missing_governance_file", required))
    for reviewed in sorted(REVIEWED_BINARY_ALLOWLIST.keys() - seen_rel):
        findings.append(Finding("missing_reviewed_firmware_image", reviewed))
    for reviewed in sorted(REVIEWED_MODEL_ARTIFACTS.keys() - seen_rel):
        findings.append(Finding("missing_reviewed_model_artifact", reviewed))
    return findings


def git_lfs_scan() -> list[Finding]:
    """Require every reviewed model to be tracked by LFS and stored as a pointer."""

    findings: list[Finding] = []
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    required_rules = {
        "model-assets/**/*.onnx filter=lfs diff=lfs merge=lfs -text",
        "model-assets/**/*.safetensors filter=lfs diff=lfs merge=lfs -text",
        "model-assets/**/*.bin filter=lfs diff=lfs merge=lfs -text",
    }
    for rule in sorted(required_rules):
        if rule not in attributes:
            findings.append(Finding("missing_lfs_attribute", ".gitattributes"))

    if not (ROOT / ".git").exists():
        return findings
    try:
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--stage", "--", "model-assets"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
        return findings + [Finding("git_lfs_scan_unavailable", ".git")]

    # Before the release files enter the index, attributes and worktree digests
    # are still checked. Once staged/committed, the Git blob must be an LFS
    # pointer bound to the same SHA-256 and size as the worktree artifact.
    staged_paths = {
        record.rsplit("\t", 1)[-1].replace("\\", "/")
        for record in tracked.splitlines()
        if "\t" in record
    }
    for rel, expected in REVIEWED_MODEL_ARTIFACTS.items():
        expected_pointer = (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{expected['sha256']}\n"
            f"size {expected['bytes']}\n"
        )
        try:
            attribute_output = subprocess.run(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "check-attr",
                    "filter",
                    "diff",
                    "merge",
                    "text",
                    "--",
                    rel,
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout
            attributes = {}
            for line in attribute_output.splitlines():
                _path, separator, remainder = line.partition(": ")
                if not separator:
                    continue
                attribute, separator, value = remainder.partition(": ")
                if separator:
                    attributes[attribute] = value
            if attributes != {
                "filter": "lfs",
                "diff": "lfs",
                "merge": "lfs",
                "text": "unset",
            }:
                findings.append(Finding("ineffective_lfs_attributes", rel))
        except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
            findings.append(Finding("git_lfs_scan_unavailable", rel))
        if rel not in staged_paths:
            # Pre-stage review: prove that the installed LFS client derives
            # the exact pointer bound by the public model manifest. Once the
            # file enters the index, the stricter blob check below applies.
            try:
                pointer_output = subprocess.run(
                    ["git", "-C", str(ROOT), "lfs", "pointer", f"--file={rel}"],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                ).stdout.replace("\r\n", "\n")
            except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
                findings.append(Finding("git_lfs_pointer_generation_failed", rel))
                continue
            if pointer_output != expected_pointer:
                findings.append(Finding("invalid_generated_lfs_pointer", rel))
            continue
        try:
            pointer = subprocess.run(
                ["git", "-C", str(ROOT), "show", f":{rel}"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout
        except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
            findings.append(Finding("lfs_pointer_unavailable", rel))
            continue
        if pointer.replace("\r\n", "\n") != expected_pointer:
            findings.append(Finding("invalid_lfs_pointer", rel))
    return findings


def git_history_secret_scan() -> list[Finding]:
    """Scan committed text blobs across all refs without printing secret values."""

    if not (ROOT / ".git").exists():
        return []
    try:
        objects = subprocess.run(
            ["git", "-C", str(ROOT), "rev-list", "--objects", "--all"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
        return [Finding("git_history_scan_unavailable", ".git")]

    findings: list[Finding] = []
    seen: set[str] = set()
    for record in objects:
        object_id, separator, name = record.partition(" ")
        if not separator or not name or object_id in seen:
            continue
        seen.add(object_id)
        suffix = Path(name).suffix.lower()
        if suffix not in TEXT_SUFFIXES and Path(name).name not in {"LICENSE", "NOTICE"}:
            continue
        try:
            object_type = subprocess.run(
                ["git", "-C", str(ROOT), "cat-file", "-t", object_id],
                check=True,
                capture_output=True,
                text=True,
                encoding="ascii",
            ).stdout.strip()
            if object_type != "blob":
                continue
            content = subprocess.run(
                ["git", "-C", str(ROOT), "cat-file", "blob", object_id],
                check=True,
                capture_output=True,
            ).stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            return [Finding("git_history_scan_unavailable", ".git")]
        if len(content) > 5 * 1024 * 1024:
            continue
        text = content.decode("utf-8", errors="ignore")
        rel_name = name.replace("\\", "/")
        # Scanner implementation/tests necessarily contain credential-shaped
        # regexes and synthetic negative fixtures. Gitleaks independently scans
        # these blobs; excluding the exact paths here prevents self-recursion.
        if rel_name in SCANNER_FIXTURE_PATHS:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(
                pattern.fullmatch(line)
                for pattern in INTENTIONAL_SECRET_FIXTURES.get(rel_name, ())
            ):
                continue
            if any(pattern.search(line) for pattern in SECRET_PATTERNS.values()):
                findings.append(Finding("secret_in_git_history", name, line_number))
    return findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-git-history",
        action="store_true",
        help="also scan committed text blobs reachable from every local ref",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = list(iter_public_files())
    findings = scan_files(files)
    findings.extend(git_lfs_scan())
    if args.include_git_history:
        findings.extend(git_history_secret_scan())

    unique = sorted(set(findings))
    if unique:
        print("PUBLIC_RELEASE_AUDIT=FAIL")
        for finding in unique:
            print(f"- {finding.render()}")
        return 1

    total_bytes = sum(path.stat().st_size for path in files)
    print(
        "PUBLIC_RELEASE_AUDIT=PASS "
        f"files={len(files)} bytes={total_bytes} "
        f"git_history={'checked' if args.include_git_history else 'not_requested'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
