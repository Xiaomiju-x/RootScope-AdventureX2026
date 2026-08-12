#!/usr/bin/env python3
"""Refresh SHA256SUMS and the deterministic tar after reproducibility-file updates."""

from __future__ import annotations

import argparse
from pathlib import Path

from finalize_release import deterministic_tar, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    release = args.release_dir.resolve()
    sums_path = release / "SHA256SUMS"
    members = sorted(path for path in release.rglob("*") if path.is_file() and path != sums_path)
    sums_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(release).as_posix()}\n" for path in members),
        encoding="ascii",
        newline="\n",
    )
    archive = release.with_suffix(".tar")
    deterministic_tar(release, archive)
    archive.with_suffix(".tar.sha256").write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="ascii", newline="\n"
    )
    print(f"{sha256_file(archive)}  {archive.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
