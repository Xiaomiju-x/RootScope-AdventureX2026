from __future__ import annotations

import csv
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
    seen: set[str] = set()
    for entry in entries:
        relative = entry["public_path"]
        assert relative not in seen
        seen.add(relative)
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.stat().st_size == entry["bytes"], relative
        assert sha256_file(path) == entry["public_sha256"], relative

    manifest_files = {
        "assets/media/ASSET_PROVENANCE.json",
        "assets/media/ASSET_PROVENANCE.csv",
    }
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "assets/media").rglob("*")
        if path.is_file() and path.relative_to(ROOT).as_posix() not in manifest_files
    }
    assert seen == actual


def test_public_media_csv_matches_json_inventory() -> None:
    manifest = json.loads(
        (ROOT / "assets/media/ASSET_PROVENANCE.json").read_text(encoding="utf-8")
    )
    with (ROOT / "assets/media/ASSET_PROVENANCE.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    json_index = {
        entry["public_path"]: (entry["public_sha256"], str(entry["bytes"]))
        for entry in manifest["assets"]
    }
    csv_index = {
        row["public_path"]: (row["public_sha256"], row["bytes"]) for row in rows
    }
    assert csv_index == json_index


def test_readme_video_previews_are_lightweight_gifs() -> None:
    for relative in (
        "assets/media/demo/probe-descent-preview.gif",
        "assets/media/demo/water-delivery-preview.gif",
    ):
        path = ROOT / relative
        assert path.stat().st_size < 5 * 1024 * 1024, relative
        assert path.read_bytes()[:6] in {b"GIF87a", b"GIF89a"}, relative


def test_readme_references_every_public_media_asset() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    manifest = json.loads(
        (ROOT / "assets/media/ASSET_PROVENANCE.json").read_text(encoding="utf-8")
    )
    missing = [
        entry["public_path"]
        for entry in manifest["assets"]
        if entry["public_path"] not in readme
    ]
    assert not missing, "README does not expose public media:\n" + "\n".join(missing)


def test_awards_are_stated_naturally_without_gallery_screenshots() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "二等奖（银奖 · 最终第 2 名）" in readme
    assert "赛道三等奖" in readme
    assert "superun.ai「不比代码，比谁离问题最近」" in readme

    manifest = json.loads(
        (ROOT / "assets/media/ASSET_PROVENANCE.json").read_text(encoding="utf-8")
    )
    public_paths = {entry["public_path"] for entry in manifest["assets"]}
    assert not any("gallery" in path for path in public_paths)
    assert "adventurex-gallery" not in readme
    assert "同一奖项" not in readme
    assert "同奖" not in readme


def test_official_certificates_are_originals_with_full_page_previews() -> None:
    manifest = json.loads(
        (ROOT / "assets/media/ASSET_PROVENANCE.json").read_text(encoding="utf-8")
    )
    entries = {
        entry["public_path"]: entry
        for entry in manifest["assets"]
        if "-certificate." in entry["public_path"]
    }
    expected = {
        "assets/media/award/adventurex2026-drobotics-second-prize-certificate.pdf",
        "assets/media/award/adventurex2026-drobotics-second-prize-certificate.png",
        "assets/media/award/adventurex2026-superun-third-prize-certificate.pdf",
        "assets/media/award/adventurex2026-superun-third-prize-certificate.png",
    }
    assert set(entries) == expected
    for relative, entry in entries.items():
        assert "no repository content licence is asserted" in entry["rights_basis"]
        payload = (ROOT / relative).read_bytes()
        if relative.endswith(".pdf"):
            assert payload.startswith(b"%PDF-")
            assert "byte-for-byte" in entry["transform"]
            assert entry["public_sha256"] == entry["source_sha256"]
        else:
            assert payload.startswith(b"\x89PNG\r\n\x1a\n")
            assert "no crop or redaction" in entry["transform"]


def test_demo_bottle_is_preserved_in_all_video_derivatives() -> None:
    manifest = json.loads(
        (ROOT / "assets/media/ASSET_PROVENANCE.json").read_text(encoding="utf-8")
    )
    demo_entries = [
        entry
        for entry in manifest["assets"]
        if entry["source_name"] == "mmexport1786529983379.mp4"
    ]
    assert len(demo_entries) == 6
    assert all("bottle" in entry["transform"] for entry in demo_entries)
    assert all("preserved" in entry["transform"] for entry in demo_entries)
    assert all("redacted" not in entry["transform"] for entry in demo_entries)
