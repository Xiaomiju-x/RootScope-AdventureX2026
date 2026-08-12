#!/usr/bin/env python3
"""Fail closed unless a release tag, commit, and package version agree."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEMVER_TAG = re.compile(
    r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?\Z"
)
PACKAGE_VERSION = re.compile(r'(?m)^__version__\s*=\s*["\']([^"\']+)["\']\s*$')
PROJECT_TABLE = re.compile(r"(?ms)^\[project\]\s*$\n(.*?)(?=^\[|\Z)")
PROJECT_VERSION = re.compile(r'(?m)^version\s*=\s*["\']([^"\']+)["\']\s*$')


def project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    table = PROJECT_TABLE.search(text)
    match = PROJECT_VERSION.search(table.group(1)) if table else None
    if not match:
        raise ValueError("pyproject.toml has no unambiguous string project.version")
    return match.group(1)


def package_version() -> str:
    text = (ROOT / "src/rootscope_public/__init__.py").read_text(encoding="utf-8")
    match = PACKAGE_VERSION.search(text)
    if not match:
        raise ValueError("package __version__ is missing or ambiguous")
    return match.group(1)


def validate_versions(tag: str, metadata_version: str, code_version: str) -> str:
    match = SEMVER_TAG.fullmatch(tag)
    if not match:
        raise ValueError("tag must be strict SemVer in vX.Y.Z form without prerelease/build metadata")
    if match.group(4) is not None or match.group(5) is not None:
        raise ValueError("release tags must not contain prerelease or build metadata")
    tag_version = tag[1:]
    if tag_version != metadata_version or tag_version != code_version:
        raise ValueError("tag, pyproject.toml, and package versions do not agree")
    return tag_version


def git(*args: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise ValueError("Git release ancestry check failed")


def verify_main_ancestry(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("commit must be a full 40-character Git object ID")
    git("rev-parse", "--verify", "origin/main^{commit}")
    git("rev-parse", "--verify", f"{commit}^{{commit}}")
    git("merge-base", "--is-ancestor", commit, "origin/main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit")
    parser.add_argument("--skip-git", action="store_true", help="only verify local versions")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        version = validate_versions(args.tag, project_version(), package_version())
        if not args.skip_git:
            if not args.commit:
                raise ValueError("--commit is required unless --skip-git is used")
            verify_main_ancestry(args.commit)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        print(f"RELEASE_IDENTITY=FAIL reason={error}")
        return 1
    print(f"RELEASE_IDENTITY=PASS tag={args.tag} version={version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
