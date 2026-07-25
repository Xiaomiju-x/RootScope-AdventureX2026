"""Fail a public release that contains secret-shaped or device-control content."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cff",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
BANNED_SUFFIXES = {
    ".axf",
    ".bin",
    ".ckpt",
    ".db",
    ".engine",
    ".gguf",
    ".hbm",
    ".hex",
    ".key",
    ".onnx",
    ".p12",
    ".pem",
    ".pth",
    ".pt",
    ".safetensors",
    ".sqlite",
    ".tflite",
}
SECRET_PATTERNS = {
    "github_token": re.compile(r"(?:github_pat_|ghp_)[A-Za-z0-9_]{20,}"),
    "generic_sk_token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private_key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "private_ipv4": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
}
SOURCE_ONLY_PATTERNS = {
    "device_path": re.compile(r"(?:/dev/(?:tty|gpio)|COM\d+\b)", re.IGNORECASE),
    "device_libraries": re.compile(
        r"^\s*(?:from|import)\s+(?:serial|RPi\.GPIO|gpiod|socket)\b",
        re.MULTILINE,
    ),
    "process_or_shell": re.compile(
        r"\b(?:subprocess\.(?:run|Popen)|os\.system)\s*\("
    ),
}


def iter_public_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
    ]


def main() -> int:
    findings: list[str] = []
    files = iter_public_files()

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in BANNED_SUFFIXES:
            findings.append(f"banned artifact: {rel}")
        if path.stat().st_size > 5 * 1024 * 1024:
            findings.append(f"oversized file: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".gitignore",
            "LICENSE",
            "NOTICE",
        }:
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{name}: {rel}")
        if path.parts[-2:-1] == ("rootscope_public",):
            for name, pattern in SOURCE_ONLY_PATTERNS.items():
                if pattern.search(text):
                    findings.append(f"{name}: {rel}")

    if findings:
        print("PUBLIC_RELEASE_AUDIT=FAIL")
        for finding in sorted(set(findings)):
            print(f"- {finding}")
        return 1

    print(f"PUBLIC_RELEASE_AUDIT=PASS files={len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
