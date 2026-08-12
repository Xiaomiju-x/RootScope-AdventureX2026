#!/usr/bin/env python3
"""Scrub machine-specific identifiers from an allow-listed public source import.

Run this only in a disposable public staging tree. It deliberately substitutes
documentation-range addresses and locally administered example identities so the
code remains readable without publishing a competition device or workstation ID.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    "",
    ".c",
    ".cmd",
    ".cpp",
    ".csv",
    ".desktop",
    ".h",
    ".html",
    ".ioc",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".service",
    ".sh",
    ".svg",
    ".toml",
    ".txt",
    ".uvprojx",
    ".xml",
    ".yaml",
    ".yml",
}

PRIVATE_IP = re.compile(
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
MAC = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")
LINUX_HOME = re.compile(r"/home/[A-Za-z0-9._-]+")
WINDOWS_USER = re.compile(r"(?i)[A-Z]:\\Users\\[^\\/\s\"']+")
WINDOWS_USER_ESCAPED = re.compile(r"(?i)[A-Z]:\\\\Users\\\\[^\\/\s\"']+")
WINDOWS_USER_FORWARD = re.compile(r"(?i)[A-Z]:/Users/[^/\s\"']+")
UUID = re.compile(r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b")
MACHINE_HEX = re.compile(r"\b[0-9A-Fa-f]{32}\b")
BOOT_TOKEN = re.compile(r"\bboot-[0-9A-Fa-f]{8,64}\b")


def scrub(text: str) -> str:
    text = PRIVATE_IP.sub("192.0.2.42", text)
    text = MAC.sub("02:00:00:00:00:01", text)
    text = LINUX_HOME.sub("/opt/rootscope", text)
    text = WINDOWS_USER_ESCAPED.sub(lambda _: r"C:\\Users\\example", text)
    text = WINDOWS_USER.sub(lambda _: r"C:\Users\example", text)
    text = WINDOWS_USER_FORWARD.sub("C:/Users/example", text)
    text = re.sub(r"(?i)\bsunrise\b", "rootscope", text)
    text = BOOT_TOKEN.sub("boot-<redacted-device-boot-id>", text)

    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if re.search(r"(?i)(boot[_ -]?id|machine[_ -]?id)", line):
            line = UUID.sub("00000000-0000-4000-8000-000000000001", line)
            line = MACHINE_HEX.sub("00000000000000000000000000000001", line)
            line = re.sub(
                r'(:\s*")[0-9A-Fa-f]{8,64}(")',
                r'\1<redacted-device-boot-id>\2',
                line,
            )
        if re.search(r'(?i)"ID_PATH"\s*:', line):
            line = re.sub(r'(:\s*")[^"]+("\s*[,}]?)', r'\1<redacted-usb-topology>\2', line)
        lines.append(line)
    return "".join(lines)


def main() -> None:
    changed = 0
    for path in sorted(REPO.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "assets" in path.parts and "media" in path.parts:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        public = scrub(original)
        if public != original:
            path.write_text(public, encoding="utf-8", newline="")
            changed += 1
    print(f"Sanitized {changed} text files")


if __name__ == "__main__":
    main()
