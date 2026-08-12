#!/usr/bin/env python3
"""Collect a small, traceable Wikimedia Commons seed set for RootScope.

This is a data-acquisition utility, not model-training or inference code.
Every downloaded item remains pending until a human reviews its file page.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "RootScopeAdventureX/1.0 "
    "(educational dataset audit; https://github.com/Xiaomiju-x/xrd; "
    "xiaomiju-x@users.noreply.github.com)"
)
ALLOWED_MIME = {"image/jpeg", "image/png"}
MIN_SIDE = 600
BLOCKED_PAGEIDS = {
    11201948,  # irrelevant adult-content false positive from a broad rock query
    39195578,  # irrelevant military image
    39223649,  # person/climbing image, not a useful RootScope negative
}

SOURCE_PLAN: dict[str, list[tuple[str, str, str]]] = {
    "grass_clump": [
        ("category", "Stipagrostis plumosa", "Stipagrostis plumosa"),
        ("category", "Panicum turgidum", "Panicum turgidum"),
    ],
    "low_shrub": [
        ("category", "Larrea tridentata", "Larrea tridentata"),
        ("category", "Haloxylon ammodendron", "Haloxylon ammodendron"),
        ("category", "Artemisia tridentata", "Artemisia tridentata"),
    ],
    "young_tree": [
        ("category", "Acacia tortilis", "Vachellia tortilis (syn. Acacia tortilis)"),
        ("category", "Prosopis cineraria", "Prosopis cineraria"),
        ("category", "Populus euphratica", "Populus euphratica"),
    ],
    "unknown": [
        ("search", "bare sand texture desert", "bare sand"),
        ("search", "desert rock closeup geology", "desert rocks"),
        ("search", "3x5 blank notecard", "blank card"),
        ("search", "desert wildlife animal", "desert animal"),
    ],
}

REJECT_TEXT = re.compile(
    r"\b(map|distribution|range|diagram|drawing|illustration|herbarium|specimen|"
    r"logo|stamp|coin|painting|painted|poster|book|page|satellite|modis|landsat|"
    r"coat of arms|flag|icon|symbol|chart|graph|microscope|fossil)\b",
    re.IGNORECASE,
)


def strip_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value or "")
    return " ".join(html.unescape(value).split())


def meta_value(ext: dict[str, Any], name: str) -> str:
    raw = ext.get(name, {})
    return raw.get("value", "") if isinstance(raw, dict) else ""


def request_json(params: dict[str, Any], retries: int = 5) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise
            wait = min(30, 2 ** (attempt + 1))
            print(f"HTTP {exc.code}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
            time.sleep(min(30, 2 ** (attempt + 1)))
    raise RuntimeError("unreachable")


def search_commons(query: str, limit: int = 50) -> list[dict[str, Any]]:
    payload = request_json(
        {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "generator": "search",
            "gsrnamespace": 6,
            "gsrlimit": min(limit, 50),
            "gsrsearch": query,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|sha1|extmetadata",
            "iiurlwidth": 1280,
        }
    )
    return payload.get("query", {}).get("pages", [])


def category_commons(category: str, limit: int = 50) -> list[dict[str, Any]]:
    payload = request_json(
        {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": min(limit, 50),
            "prop": "imageinfo",
            "iiprop": "url|mime|size|sha1|extmetadata",
            "iiurlwidth": 1280,
        }
    )
    return payload.get("query", {}).get("pages", [])


def license_allowed(name: str) -> bool:
    normalized = strip_html(name).lower()
    if not normalized:
        return False
    if any(bad in normalized for bad in ("-nc", "-nd", "noncommercial", "no derivatives")):
        return False
    return normalized.startswith("cc by") or "cc0" in normalized or "public domain" in normalized


def candidate_from_page(class_id: str, query: str, species_hint: str, page: dict[str, Any]) -> dict[str, Any] | None:
    if int(page.get("pageid") or 0) in BLOCKED_PAGEIDS:
        return None
    infos = page.get("imageinfo") or []
    if not infos:
        return None
    info = infos[0]
    ext = info.get("extmetadata") or {}
    mime = info.get("mime", "")
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    title = page.get("title", "")
    description_html = meta_value(ext, "ImageDescription")
    description = strip_html(description_html)
    license_name = strip_html(meta_value(ext, "LicenseShortName") or meta_value(ext, "UsageTerms"))
    license_url = strip_html(meta_value(ext, "LicenseUrl"))
    artist_html = meta_value(ext, "Artist") or meta_value(ext, "Credit")
    artist = strip_html(artist_html)

    if mime not in ALLOWED_MIME or min(width, height) < MIN_SIDE:
        return None
    if REJECT_TEXT.search(f"{title} {description}"):
        return None
    if not license_allowed(license_name) or not artist:
        return None

    download_url = info.get("thumburl") or info.get("url")
    if not download_url:
        return None
    pageid = int(page["pageid"])
    return {
        "class_id": class_id,
        "species_hint": species_hint,
        "query": query,
        "domain": "natural_web",
        "split": "unassigned",
        "review_status": "pending",
        "print_eligible": False,
        "source_provider": "Wikimedia Commons",
        "source_group": f"commons:{pageid}",
        "pageid": pageid,
        "title": title,
        "source_page": info.get("descriptionurl") or f"https://commons.wikimedia.org/?curid={pageid}",
        "original_url": info.get("url", ""),
        "download_url": download_url,
        "commons_sha1": info.get("sha1", ""),
        "mime": mime,
        "original_width": width,
        "original_height": height,
        "artist": artist,
        "artist_html": artist_html,
        "credit": strip_html(meta_value(ext, "Credit")),
        "license": license_name,
        "license_url": license_url,
        "attribution_required": strip_html(meta_value(ext, "AttributionRequired")),
        "copyrighted": strip_html(meta_value(ext, "Copyrighted")),
        "description": description[:1000],
        "accessed_at": date.today().isoformat(),
    }


def download_bytes(url: str, source_page: str, retries: int = 3) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": source_page,
            "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*;q=0.5",
        },
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if attempt == retries - 1:
                raise
            retry_after = int(exc.headers.get("Retry-After", "0") or 0)
            time.sleep(max(retry_after, min(30, 5 * (attempt + 1))))
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
            time.sleep(min(20, 3 * (attempt + 1)))
    raise RuntimeError("unreachable")


def safe_extension(candidate: dict[str, Any]) -> str:
    extension = mimetypes.guess_extension(candidate["mime"]) or ".jpg"
    return ".jpg" if extension in {".jpe", ".jpeg"} else extension


def write_outputs(output: Path, records: list[dict[str, Any]]) -> None:
    manifest = output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records, key=lambda item: (item["class_id"], item["pageid"])):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    counts = Counter(item["class_id"] for item in records)
    licenses = Counter(item["license"] for item in records)
    summary = {
        "status": "SEED_MANUAL_REVIEW_REQUIRED_NOT_TRAIN_READY",
        "generated_at": date.today().isoformat(),
        "total": len(records),
        "class_counts": dict(sorted(counts.items())),
        "license_counts": dict(sorted(licenses.items())),
        "all_splits": "unassigned",
        "all_review_status": "pending",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# RootScope Desert Plants v1｜图片署名清单",
        "",
        "> 自动生成；使用或打印前必须逐张打开来源页复核。",
        "",
    ]
    for record in sorted(records, key=lambda item: (item["class_id"], item["filename"])):
        license_text = record["license"]
        if record["license_url"]:
            license_text = f"[{license_text}]({record['license_url']})"
        lines.append(
            f"- `{record['filename']}` — {record['artist']} — "
            f"[{record['title']}]({record['source_page']}) — {license_text}"
        )
    (output / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect(output: Path, per_class: int, reset: bool) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    images_root = output / "images"
    images_root.mkdir(exist_ok=True)

    if reset:
        for class_id in SOURCE_PLAN:
            class_dir = images_root / class_id
            if class_dir.exists():
                for path in class_dir.glob("*"):
                    if path.is_file():
                        path.unlink()

    records: list[dict[str, Any]] = []
    used_pageids: set[int] = set()
    used_hashes: set[str] = set()

    for class_id, source_items in SOURCE_PLAN.items():
        class_dir = images_root / class_id
        class_dir.mkdir(exist_ok=True)
        accepted = 0
        source_cap = max(1, (per_class + len(source_items) - 1) // len(source_items))
        for source_mode, query, species_hint in source_items:
            if accepted >= per_class:
                break
            source_accepted = 0
            print(f"[{class_id}] {source_mode}: {query}")
            pages = category_commons(query, limit=50) if source_mode == "category" else search_commons(query, limit=50)
            for page in pages:
                if accepted >= per_class or source_accepted >= source_cap:
                    break
                candidate = candidate_from_page(class_id, query, species_hint, page)
                if candidate is None or candidate["pageid"] in used_pageids:
                    continue
                try:
                    payload = download_bytes(candidate["download_url"], candidate["source_page"])
                except Exception as exc:  # keep the rest of the acquisition usable
                    print(f"  skip download {candidate['title']}: {exc}", file=sys.stderr)
                    continue
                digest = hashlib.sha256(payload).hexdigest()
                if digest in used_hashes:
                    continue
                extension = safe_extension(candidate)
                filename = f"{class_id}_{candidate['pageid']}_{digest[:10]}{extension}"
                path = class_dir / filename
                path.write_bytes(payload)
                candidate.update(
                    {
                        "filename": f"images/{class_id}/{filename}",
                        "download_sha256": digest,
                        "download_bytes": len(payload),
                    }
                )
                records.append(candidate)
                used_pageids.add(candidate["pageid"])
                used_hashes.add(digest)
                accepted += 1
                source_accepted += 1
                print(f"  {accepted:02d}/{per_class}: {filename}")
                time.sleep(1.1)
            time.sleep(0.8)
        print(f"[{class_id}] collected {accepted}/{per_class}")
        # Preserve a usable partial manifest if a later class is rate-limited.
        write_outputs(output, records)
    return records


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_output = script.parents[2] / "datasets" / "desert_plants_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--reset", action="store_true", help="remove previously downloaded image files first")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.per_class <= 100:
        raise SystemExit("--per-class must be between 1 and 100")
    records = collect(args.output.resolve(), args.per_class, args.reset)
    write_outputs(args.output.resolve(), records)
    print(json.dumps({"output": str(args.output.resolve()), "records": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
