#!/usr/bin/env python3
"""Verify every published model asset against the root content manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "model-assets" / "MANIFEST.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for asset in manifest["assets"]:
        path = ROOT / asset["path"]
        if not path.is_file():
            failures.append(f"missing: {asset['path']}")
            continue
        if path.stat().st_size != asset["bytes"]:
            failures.append(f"size mismatch: {asset['path']}")
        if sha256(path) != asset["sha256"]:
            failures.append(f"sha256 mismatch: {asset['path']}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(manifest['assets'])} model assets content-bound")


if __name__ == "__main__":
    main()
