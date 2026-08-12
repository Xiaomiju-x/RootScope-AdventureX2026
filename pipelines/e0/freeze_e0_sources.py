#!/usr/bin/env python3
"""Create a deterministic source inventory for the RootScope E0 freeze.

The inventory deliberately excludes generated evidence and dataset payloads so
that a completion receipt can bind the implementation without becoming
self-referential or hashing hundreds of candidate images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "rootscope.e0.source_inventory.v1"
ADVENTUREX_ROOT = Path(__file__).resolve().parents[2]

SOURCE_GLOBS = (
    "rootscope/app/**/*.py",
    "rootscope/app/web/index.html",
    "rootscope/training/**/*.py",
    "rootscope/training/**/*.json",
    "rootscope/configs/**/*.json",
    "rootscope/tests/**/*.py",
    "tools/dataset/*.py",
    "tools/dataset/*.ps1",
    "tools/dataset/*.json",
    "tools/dataset/tests/**/*.py",
    "tools/e0/**/*.py",
)

SOURCE_FILES = (
    "rootscope/pyproject.toml",
    "rootscope/README.md",
    "rootscope/H12_IMPLEMENTATION_STATUS.md",
    "rootscope/BLOCKERS.md",
    "rootscope/PREEXISTING.md",
    "rootscope/BUILT_DURING_EVENT.md",
    "rootscope_work/execution/08_ROOTSCOPE_ONE_WEEK_DEEP_PLAN.md",
)


class FreezeError(RuntimeError):
    """Raised when the source inventory cannot be frozen safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_eligible(path: Path) -> bool:
    return path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"


def collect_sources(root: Path = ADVENTUREX_ROOT) -> list[Path]:
    root = root.resolve(strict=True)
    selected: set[Path] = set()
    for pattern in SOURCE_GLOBS:
        selected.update(path for path in root.glob(pattern) if _is_eligible(path))
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FreezeError(f"required E0 source is missing: {relative}")
        selected.add(path)

    resolved: list[Path] = []
    for path in selected:
        candidate = path.resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise FreezeError(f"source escapes AdventureX root: {path}") from exc
        resolved.append(candidate)
    return sorted(resolved, key=lambda item: item.relative_to(root).as_posix())


def build_inventory(root: Path = ADVENTUREX_ROOT) -> tuple[list[dict[str, object]], str]:
    root = root.resolve(strict=True)
    records: list[dict[str, object]] = []
    canonical_lines: list[str] = []
    for path in collect_sources(root):
        relative = path.relative_to(root).as_posix()
        digest = _sha256_file(path)
        byte_count = path.stat().st_size
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "path": relative,
                "sha256": digest,
                "bytes": byte_count,
            }
        )
        canonical_lines.append(f"{relative}\t{digest}\t{byte_count}\n")
    canonical = "".join(canonical_lines).encode("utf-8")
    return records, hashlib.sha256(canonical).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def freeze(output: Path, root: Path = ADVENTUREX_ROOT) -> dict[str, object]:
    records, root_sha256 = build_inventory(root)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    ).encode("utf-8")
    _atomic_write(output, payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_file_count": len(records),
        "source_root_sha256": root_sha256,
        "inventory_sha256": hashlib.sha256(payload).hexdigest(),
        "inventory": output.resolve().as_posix(),
        "canonicalization": (
            "UTF-8 lines sorted by normalized AdventureX-relative path; each canonical "
            "line is path<TAB>lowercase_sha256<TAB>byte_count; final newline included."
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ADVENTUREX_ROOT
        / "rootscope"
        / "evidence"
        / "e0"
        / "e0_source_inventory.jsonl",
    )
    args = parser.parse_args(argv)
    try:
        summary = freeze(args.output)
    except (FreezeError, OSError) as exc:
        print(f"E0 source freeze failed: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
