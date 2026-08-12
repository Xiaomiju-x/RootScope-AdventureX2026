from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def curated_markdown() -> list[Path]:
    roots = [
        ROOT / name
        for name in (
            "README.md",
            "CHANGELOG.md",
            "CODE_OF_CONDUCT.md",
            "CONTRIBUTING.md",
            "GOVERNANCE.md",
            "SECURITY.md",
            "SUPPORT.md",
            "THIRD_PARTY_NOTICES.md",
        )
    ]
    roots.extend(sorted((ROOT / "docs").glob("*.md")))
    roots.extend(sorted((ROOT / "model-assets").glob("*/MODEL_CARD.md")))
    return roots


def local_target(document: Path, raw: str) -> Path | None:
    value = raw.strip().strip("<>")
    if " " in value and not value.startswith("assets/"):
        value = value.split(maxsplit=1)[0]
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return (document.parent / unquote(parsed.path)).resolve()


def test_curated_documentation_has_no_broken_local_links() -> None:
    broken: list[str] = []
    for document in curated_markdown():
        text = document.read_text(encoding="utf-8")
        targets = MARKDOWN_LINK.findall(text) + HTML_LINK.findall(text)
        for raw in targets:
            target = local_target(document, raw)
            if target is not None and not target.exists():
                broken.append(f"{document.relative_to(ROOT).as_posix()} -> {raw}")
    assert not broken, "broken local documentation links:\n" + "\n".join(broken)


def test_public_media_manifest_binds_every_derivative() -> None:
    manifest = json.loads(
        (ROOT / "assets/media/ASSET_PROVENANCE.json").read_text(encoding="utf-8")
    )
    entries = manifest["assets"]
    assert len(entries) == 19
    seen: set[str] = set()
    for entry in entries:
        relative = entry["public_path"]
        assert relative not in seen
        seen.add(relative)
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.stat().st_size == entry["bytes"], relative
        assert sha256_file(path) == entry["public_sha256"], relative
