#!/usr/bin/env python3
"""Acquire traceable whole-plant candidates from Wikimedia Commons.

This collector deliberately writes to a separate reacquisition staging directory.
It does not edit the E0 manifest, the formal review queue, or human decisions.  A
search/category hit is only an acquisition hint: every record stays unassigned,
not train eligible, and pending visual and rights review.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "RootScopeAdventureX/2.1 (whole-plant educational dataset provenance audit; "
    "xiaomiju-x@users.noreply.github.com)"
)
SCHEMA_VERSION = "rootscope.wikimedia_candidate.v1"
STATUS = "MACHINE_ACQUIRED_WHOLE_PLANT_CANDIDATES_NOT_TRAIN_READY"
ALLOWED_MIME = {"image/jpeg", "image/png"}
MIN_ORIGINAL_SIDE = 720
MIN_DOWNLOADED_SIDE = 448
DHASH_ALGORITHM = "rootscope_rgb_center_sample_9x8_v1"

# Metadata-only screening.  This cannot prove a whole plant is visible; it only
# removes obvious maps, specimens and macro-detail images before visual triage.
REJECT_COMMON = re.compile(
    r"\b(map|distribution|range map|diagram|drawing|illustration|cross[- ]section|"
    r"herbarium|specimen|logo|stamp|coin|painting|poster|book page|satellite|"
    r"modis|landsat|coat of arms|flag|icon|symbol|chart|graph|microscope|fossil|"
    r"pressed plant|botanical plate)\b",
    re.IGNORECASE,
)
REJECT_DETAIL = re.compile(
    r"\b(close[- ]?up|macro|flower detail|flowers? close|leaf detail|leaves close|"
    r"fruit detail|seed detail|bark detail|twig|branch detail|inflorescence detail)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Source:
    class_id: str
    retrieval_mode: str
    retrieval_query: str
    species_hint: str
    acquisition_query: str


def source(class_id: str, mode: str, retrieval: str, hint: str, intent: str) -> Source:
    return Source(
        class_id=class_id,
        retrieval_mode=mode,
        retrieval_query=retrieval,
        species_hint=hint,
        acquisition_query=f"{retrieval}: {intent}",
    )


GRASS_INTENT = (
    "whole desert bunchgrass/tussock plant; base visible; crown visible; "
    "isolated single plant; not a flower, seedhead or leaf close-up"
)
SHRUB_INTENT = (
    "whole low desert shrub; base visible; full canopy/crown visible; isolated "
    "single plant; not a flower, fruit, leaf or branch close-up"
)
TREE_INTENT = (
    "whole young desert tree sapling/seedling; trunk base visible; entire crown "
    "visible; isolated single young plant; not a mature tree or branch close-up"
)


SOURCE_PLAN: tuple[Source, ...] = tuple(
    [
        source("grass_clump", "category", q, q, GRASS_INTENT)
        for q in (
            "Panicum turgidum",
            "Stipagrostis plumosa",
            "Cenchrus ciliaris",
            "Cenchrus setaceus",
            "Muhlenbergia porteri",
            "Sporobolus airoides",
            "Achnatherum hymenoides",
            "Bouteloua eriopoda",
            "Pleuraphis rigida",
            "Aristida purpurea",
            "Pleuraphis mutica",
            "Sporobolus cryptandrus",
            "Hesperostipa comata",
            "Elymus elymoides",
            "Dasyochloa pulchella",
        )
    ]
    + [
        source("grass_clump", "search", q, "whole desert bunchgrass morphology", GRASS_INTENT)
        for q in (
            '"whole plant" desert bunchgrass',
            '"whole plant" desert grass tussock',
            '"whole plant" Panicum turgidum',
            '"whole plant" Stipagrostis',
            '"base visible" bunchgrass',
            'isolated desert grass tussock plant',
        )
    ]
    + [
        source("low_shrub", "category", q, q, SHRUB_INTENT)
        for q in (
            "Larrea tridentata",
            "Artemisia tridentata",
            "Atriplex canescens",
            "Haloxylon persicum",
            "Calligonum comosum",
            "Atriplex halimus",
            "Artemisia herba-alba",
            "Ephedra alata",
            "Rhanterium epapposum",
            "Zygophyllum dumosum",
            "Ambrosia dumosa",
            "Encelia farinosa",
            "Atriplex confertifolia",
            "Ephedra californica",
            "Krascheninnikovia lanata",
            "Coleogyne ramosissima",
        )
    ]
    + [
        source("low_shrub", "search", q, "whole low desert shrub morphology", SHRUB_INTENT)
        for q in (
            '"whole plant" desert shrub isolated',
            '"whole shrub" desert isolated',
            '"base visible" desert shrub',
            'isolated low arid shrub whole plant',
            'isolated xerophytic shrub whole plant',
        )
    ]
    + [
        source("young_tree", "category", q, q, TREE_INTENT)
        for q in (
            "Vachellia tortilis",
            "Acacia tortilis",
            "Prosopis cineraria",
            "Populus euphratica",
            "Tamarix aphylla",
            "Vachellia erioloba",
            "Vachellia nilotica",
            "Senegalia senegal",
            "Prosopis juliflora",
            "Parkinsonia aculeata",
            "Boscia albitrunca",
            "Balanites aegyptiaca",
        )
    ]
    + [
        source("young_tree", "search", q, "young desert tree morphology", TREE_INTENT)
        for q in (
            'Acacia sapling "whole plant"',
            'Vachellia sapling "whole plant"',
            'desert tree sapling "whole plant"',
            'desert tree seedling "whole plant"',
            'young Acacia isolated sapling',
            'young Vachellia isolated sapling',
            '"base visible" tree sapling desert',
            'tree seedling growing in sand whole plant',
        )
    ]
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_html(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(text).split())


def meta_value(metadata: dict[str, Any], name: str) -> str:
    raw = metadata.get(name)
    if not isinstance(raw, dict):
        return ""
    return strip_html(raw.get("value", ""))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def append_event(path: Path, event: str, **fields: Any) -> None:
    record = {"at_utc": utc_now(), "event": event, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def request_json(params: dict[str, Any], *, retries: int = 6) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {429, 500, 502, 503, 504}
            if attempt == retries or not retryable:
                raise
            time.sleep(min(20.0, 0.8 * (2 ** (attempt - 1))))
    raise RuntimeError("unreachable")


def commons_pages(source_item: Source, batches: int) -> Iterable[dict[str, Any]]:
    common: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|sha1|extmetadata",
        "iiurlwidth": 1280,
    }
    if source_item.retrieval_mode == "category":
        common.update(
            generator="categorymembers",
            gcmtitle=f"Category:{source_item.retrieval_query}",
            gcmtype="file",
            gcmlimit=50,
        )
    else:
        common.update(
            generator="search",
            gsrnamespace=6,
            gsrlimit=50,
            gsrsearch=source_item.retrieval_query,
        )
    continuation: dict[str, Any] = {}
    for _ in range(batches):
        payload = request_json({**common, **continuation})
        for page in payload.get("query", {}).get("pages", []):
            yield page
        continuation = payload.get("continue", {})
        if not continuation:
            return


def download(url: str, source_page: str, retries: int = 5) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": source_page,
            "Accept": "image/jpeg,image/png,*/*;q=0.2",
        },
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {429, 500, 502, 503, 504}
            if attempt == retries or not retryable:
                raise
            time.sleep(min(20.0, 1.0 * (2 ** (attempt - 1))))
    raise RuntimeError("unreachable")


def image_facts(payload: bytes) -> tuple[int, int, str, str]:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        mime = Image.MIME.get(image.format, "")
        if mime not in ALLOWED_MIME:
            raise ValueError(f"unsupported decoded image format {image.format!r}")
        rgb = image.convert("RGB")
        bits: list[str] = []
        for y in range(8):
            source_y = min(rgb.height - 1, int(((y + 0.5) * rgb.height) // 8))
            for x in range(8):
                left_x = min(rgb.width - 1, int(((x + 0.5) * rgb.width) // 9))
                right_x = min(rgb.width - 1, int(((x + 1.5) * rgb.width) // 9))
                left = rgb.getpixel((left_x, source_y))
                right = rgb.getpixel((right_x, source_y))
                left_luma = 299 * left[0] + 587 * left[1] + 114 * left[2]
                right_luma = 299 * right[0] + 587 * right[1] + 114 * right[2]
                bits.append("1" if left_luma > right_luma else "0")
        dhash = f"{int(''.join(bits), 2):016x}"
        return rgb.width, rgb.height, mime, dhash


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                records.append(value)
    return records


def license_table(policy: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for license_item in policy["licenses"]:
        for binding in license_item["raw_bindings"]:
            key = (binding["raw_name"], binding["raw_url"])
            if key in table:
                raise ValueError(f"duplicate license binding: {key}")
            table[key] = license_item
    return table


def resolve_license(
    table: dict[tuple[str, str], dict[str, Any]], raw_name: str, raw_url: str, copyrighted: str
) -> dict[str, Any] | None:
    item = table.get((raw_name, raw_url))
    if item is None:
        return None
    required = item.get("copyrighted_exact_values", [])
    if required and copyrighted not in required:
        return None
    return item


def normalized_creator_group(artist: str) -> str:
    return "commons-creator:" + sha256_bytes(artist.encode("utf-8"))[:16]


def save_outputs(
    output: Path,
    records: list[dict[str, Any]],
    policy_sha: str,
    source_plan_sha: str,
    existing_count: int,
) -> None:
    ordered = sorted(records, key=lambda item: (item["class_id"], int(item["pageid"])))
    manifest_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in ordered
    )
    atomic_text(output / "manifest.jsonl", manifest_text)
    manifest_sha = sha256_bytes(manifest_text.encode("utf-8"))
    class_counts = Counter(record["class_id"] for record in ordered)
    license_counts = Counter(record["license_canonical_name"] for record in ordered)
    query_counts = Counter(record["acquisition_query"] for record in ordered)
    creator_counts = Counter(record["creator_group"] for record in ordered)
    summary = {
        "schema_version": "rootscope.whole_plant_reacquisition_summary.v1",
        "status": STATUS,
        "generated_at_utc": utc_now(),
        "total": len(ordered),
        "class_counts": dict(sorted(class_counts.items())),
        "license_counts": dict(sorted(license_counts.items())),
        "query_counts": dict(sorted(query_counts.items())),
        "existing_records_screened_for_overlap": existing_count,
        "exact_overlap_with_existing": 0,
        "all_splits": "UNASSIGNED_DO_NOT_TRAIN",
        "all_training_eligible": False,
        "all_print_eligible": False,
        "formal_human_review_authority": False,
        "machine_query_labels_are_ground_truth": False,
        "license_policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
        "manifest_sha256": manifest_sha,
        "max_records_from_one_creator_group": max(creator_counts.values(), default=0),
        "legal_note": (
            "Commons metadata is machine-screened, not a warranty. Re-open the canonical "
            "file page and verify attribution, license, privacy/personality rights and visual "
            "class/whole-plant conformance before training, printing or publication."
        ),
    }
    atomic_text(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    source_plan_document = {
        "schema_version": "rootscope.whole_plant_reacquisition_source_plan.v1",
        "source_provider": "Wikimedia Commons",
        "api_endpoint": API_URL,
        "official_api_documentation": [
            "https://www.mediawiki.org/wiki/API:Search",
            "https://www.mediawiki.org/wiki/API:Imageinfo",
        ],
        "source_plan_payload_sha256": source_plan_sha,
        "candidate_generation_only": True,
        "visual_ground_truth_authority": False,
        "required_downstream_visual_gate": {
            "one_dominant_plant": True,
            "entire_base_visible": True,
            "entire_crown_or_canopy_visible": True,
            "reject_closeup_or_plant_part": True,
            "reject_hand_or_person": True,
            "young_tree_must_be_sapling_or_seedling_not_mature_tree": True,
            "failure_action": "EXCLUDE_OR_HOLD_DO_NOT_FORCE_TO_TARGET_CLASS",
        },
        "sources": [item.__dict__ for item in SOURCE_PLAN],
    }
    atomic_text(
        output / "source_plan.json",
        json.dumps(source_plan_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    attribution = [
        "# RootScope 整株沙漠植物定向再采集｜来源与署名",
        "",
        "> 机器采集候选；不是审核结论。训练、打印或公开前必须重新打开每个 Commons 文件页复核。",
        "",
    ]
    for record in ordered:
        license_label = record["license_canonical_name"]
        if record["license_canonical_url"]:
            license_label = f"[{license_label}]({record['license_canonical_url']})"
        attribution.append(
            f"- `{record['filename']}` — {record['artist']} — "
            f"[{record['title']}]({record['source_page']}) — {license_label}"
        )
    atomic_text(output / "ATTRIBUTION.md", "\n".join(attribution) + "\n")

    readme = f"""# RootScope whole-plant reacquisition staging

Status: `{STATUS}`

This directory is isolated from `desert_plants_wikimedia_staging_e0`.  It contains
Wikimedia Commons candidates acquired with explicit whole-plant structural intent:
base visible, crown/canopy visible, one dominant isolated plant, and sapling/seedling
for `young_tree`.  Search/category metadata does **not** establish visual truth.

- `manifest.jsonl`: `{SCHEMA_VERSION}` candidate records with source, creator,
  exact license binding, page ID, query, UTC download time and SHA-256.
- `summary.json`: counts and provenance roots.
- `source_plan.json`: exact retrieval queries, explicit structural intent, and the
  mandatory downstream visual gate.
- `ATTRIBUTION.md`: machine-generated attribution aid.
- `recovery_log.jsonl`: append-only acquisition events.
- `images/<class_id>/`: downloaded 1280-pixel derivatives.

No record is training-eligible or print-eligible. Formal E0 human decisions and the
frozen E0 manifest are outside this directory and are not changed by this collector.
"""
    atomic_text(output / "README.md", readme)


def collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    script = Path(__file__).resolve()
    adventurex = script.parents[2]
    output = args.output.resolve()
    e0 = args.existing_e0.resolve()
    legacy = args.existing_legacy.resolve()
    policy_path = args.license_policy.resolve()
    if output == e0 or output == legacy:
        raise ValueError("reacquisition output must be distinct from existing datasets")

    policy_bytes = policy_path.read_bytes()
    policy_sha = sha256_bytes(policy_bytes)
    policy = json.loads(policy_bytes.decode("utf-8-sig"))
    if policy.get("schema_version") != "rootscope.wikimedia_license_policy.v1":
        raise ValueError("unexpected Wikimedia license policy schema")
    bindings = license_table(policy)
    source_plan_payload = [item.__dict__ for item in SOURCE_PLAN]
    source_plan_sha = sha256_bytes(
        json.dumps(source_plan_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    output.mkdir(parents=True, exist_ok=True)
    event_log = output / "recovery_log.jsonl"
    records = load_jsonl(output / "manifest.jsonl")
    existing_records = load_jsonl(e0 / "manifest.jsonl") + load_jsonl(legacy / "manifest.jsonl")

    used_pageids = {int(record["pageid"]) for record in existing_records + records if record.get("pageid")}
    used_sha1 = {
        str(record["commons_sha1"]).lower()
        for record in existing_records + records
        if record.get("commons_sha1")
    }
    used_sha256 = {
        str(record["download_sha256"]).lower()
        for record in existing_records + records
        if record.get("download_sha256")
    }
    known_dhashes = [
        str(record["dhash64"]).lower()
        for record in existing_records + records
        if re.fullmatch(r"[0-9a-fA-F]{16}", str(record.get("dhash64", "")))
    ]
    creator_counts = Counter(record.get("creator_group", "") for record in records)
    class_counts = Counter(record["class_id"] for record in records)

    append_event(
        event_log,
        "run_started",
        output=str(output),
        targets=args.per_class,
        license_policy_sha256=policy_sha,
        source_plan_sha256=source_plan_sha,
    )
    save_outputs(output, records, policy_sha, source_plan_sha, len(existing_records))

    for source_item in SOURCE_PLAN:
        class_id = source_item.class_id
        if class_counts[class_id] >= args.per_class:
            continue
        print(
            f"[{class_id}] {source_item.retrieval_mode}: {source_item.retrieval_query} "
            f"({class_counts[class_id]}/{args.per_class})",
            flush=True,
        )
        try:
            pages = commons_pages(source_item, args.api_batches)
            for page in pages:
                if class_counts[class_id] >= args.per_class:
                    break
                pageid = int(page.get("pageid") or 0)
                if not pageid or pageid in used_pageids:
                    continue
                infos = page.get("imageinfo") or []
                if not infos:
                    continue
                info = infos[0]
                metadata = info.get("extmetadata") or {}
                title = str(page.get("title") or "")
                description = meta_value(metadata, "ImageDescription")
                visible_text = f"{title} {description}"
                if REJECT_COMMON.search(visible_text) or REJECT_DETAIL.search(visible_text):
                    continue
                mime = str(info.get("mime") or "")
                original_width = int(info.get("width") or 0)
                original_height = int(info.get("height") or 0)
                if mime not in ALLOWED_MIME or min(original_width, original_height) < MIN_ORIGINAL_SIDE:
                    continue
                artist = meta_value(metadata, "Artist") or meta_value(metadata, "Credit")
                if not artist:
                    continue
                creator_group = normalized_creator_group(artist)
                if creator_counts[creator_group] >= args.max_per_creator:
                    continue
                raw_name = meta_value(metadata, "LicenseShortName") or meta_value(metadata, "UsageTerms")
                raw_url = meta_value(metadata, "LicenseUrl")
                copyrighted = meta_value(metadata, "Copyrighted")
                canonical = resolve_license(bindings, raw_name, raw_url, copyrighted)
                if canonical is None:
                    continue
                commons_sha1 = str(info.get("sha1") or "").lower()
                if not commons_sha1 or commons_sha1 in used_sha1:
                    continue
                source_page = str(info.get("descriptionurl") or f"https://commons.wikimedia.org/?curid={pageid}")
                original_url = str(info.get("url") or "")
                download_url = str(info.get("thumburl") or original_url)
                if not download_url:
                    continue
                try:
                    downloaded_at = utc_now()
                    payload = download(download_url, source_page)
                    digest = sha256_bytes(payload)
                    if digest in used_sha256:
                        continue
                    width, height, decoded_mime, dhash = image_facts(payload)
                    if decoded_mime != mime or min(width, height) < MIN_DOWNLOADED_SIDE:
                        continue
                    min_distance = min((hamming(dhash, known) for known in known_dhashes), default=64)
                    if min_distance <= args.dhash_distance:
                        continue
                    extension = ".png" if mime == "image/png" else ".jpg"
                    filename_only = f"{class_id}_{pageid}_{digest[:12]}{extension}"
                    relative = f"images/{class_id}/{filename_only}"
                    destination = output / Path(relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists() and sha256_file(destination) != digest:
                        raise RuntimeError(f"refusing to replace a different file: {destination}")
                    if not destination.exists():
                        temporary = destination.with_suffix(destination.suffix + ".download")
                        temporary.write_bytes(payload)
                        temporary.replace(destination)
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "class_id": class_id,
                        "species_hint": source_item.species_hint,
                        "species_hint_status": "acquisition_hint_not_a_reviewed_species_or_shape_label",
                        "candidate_label_status": "targeted_query_or_category_derived_unverified",
                        "acquisition_mode": source_item.retrieval_mode,
                        "acquisition_query": source_item.acquisition_query,
                        "query": source_item.acquisition_query,
                        "retrieval_query": source_item.retrieval_query,
                        "structural_intent": {
                            "whole_plant": True,
                            "base_visible": True,
                            "crown_or_canopy_visible": True,
                            "isolated_single_plant": True,
                            "young_tree_requires_sapling_or_seedling": class_id == "young_tree",
                            "status": "query_intent_not_visually_verified",
                        },
                        "domain": "natural_web_candidate",
                        "split": "UNASSIGNED_DO_NOT_TRAIN",
                        "review_status": "pending_machine_visual_triage_and_human_rights_review",
                        "training_eligible": False,
                        "print_eligible": False,
                        "source_provider": "Wikimedia Commons",
                        "source_group": f"commons:{pageid}",
                        "source_group_basis": (
                            "one authoritative Commons original file page; crops/augmentations/prints/"
                            "recaptures must inherit this group"
                        ),
                        "creator_group": creator_group,
                        "pageid": pageid,
                        "title": title,
                        "source_page": source_page,
                        "original_url": original_url,
                        "download_url": download_url,
                        "commons_sha1": commons_sha1,
                        "mime": mime,
                        "original_width": original_width,
                        "original_height": original_height,
                        "artist": artist,
                        "credit": meta_value(metadata, "Credit"),
                        "license": raw_name,
                        "license_url": raw_url,
                        "license_raw_name": raw_name,
                        "license_raw_url": raw_url,
                        "license_canonical_id": canonical["canonical_id"],
                        "license_canonical_name": canonical["canonical_name"],
                        "license_canonical_url": canonical["canonical_url"],
                        "license_binding_id": f"policy:{canonical['canonical_id']}:{raw_name}|{raw_url}",
                        "license_allowlist_rule": f"policy:{canonical['canonical_id']}:{raw_name}|{raw_url}",
                        "license_policy_sha256": policy_sha,
                        "usage_terms": meta_value(metadata, "UsageTerms"),
                        "attribution_required": meta_value(metadata, "AttributionRequired"),
                        "copyrighted": copyrighted,
                        "restrictions": meta_value(metadata, "Restrictions"),
                        "description": description[:1500],
                        "license_metadata_source": (
                            "Wikimedia Commons action=query imageinfo extmetadata and canonical file page"
                        ),
                        "rights_review_status": (
                            "machine_allowlist_pass_human_file_page_and_non_copyright_rights_review_pending"
                        ),
                        "accessed_at_utc": downloaded_at,
                        "downloaded_at_utc": downloaded_at,
                        "filename": relative,
                        "download_bytes": len(payload),
                        "download_sha256": digest,
                        "download_width": width,
                        "download_height": height,
                        "download_mime": decoded_mime,
                        "dhash64_algorithm": DHASH_ALGORITHM,
                        "dhash64": dhash,
                        "minimum_prior_or_reacquisition_dhash_distance": min_distance,
                    }
                    records.append(record)
                    used_pageids.add(pageid)
                    used_sha1.add(commons_sha1)
                    used_sha256.add(digest)
                    known_dhashes.append(dhash)
                    creator_counts[creator_group] += 1
                    class_counts[class_id] += 1
                    save_outputs(output, records, policy_sha, source_plan_sha, len(existing_records))
                    append_event(
                        event_log,
                        "candidate_saved",
                        class_id=class_id,
                        pageid=pageid,
                        filename=relative,
                        sha256=digest,
                        acquisition_query=source_item.acquisition_query,
                    )
                    print(f"  saved {class_counts[class_id]:02d}/{args.per_class}: {relative}", flush=True)
                    time.sleep(args.delay)
                except Exception as exc:  # preserve the usable partial manifest
                    append_event(
                        event_log,
                        "candidate_failed",
                        class_id=class_id,
                        pageid=pageid,
                        title=title,
                        error=str(exc),
                    )
        except Exception as exc:
            append_event(
                event_log,
                "source_query_failed",
                class_id=class_id,
                retrieval_mode=source_item.retrieval_mode,
                retrieval_query=source_item.retrieval_query,
                error=str(exc),
            )

    save_outputs(output, records, policy_sha, source_plan_sha, len(existing_records))
    append_event(event_log, "run_finished", total=len(records), class_counts=dict(class_counts))
    return records


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    adventurex = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=adventurex / "datasets" / "desert_plants_whole_plant_reacquisition_e1",
    )
    parser.add_argument(
        "--existing-e0",
        type=Path,
        default=adventurex / "datasets" / "desert_plants_wikimedia_staging_e0",
    )
    parser.add_argument(
        "--existing-legacy",
        type=Path,
        default=adventurex / "datasets" / "desert_plants_v1",
    )
    parser.add_argument(
        "--license-policy",
        type=Path,
        default=script.with_name("wikimedia_license_policy_v1.json"),
    )
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--api-batches", type=int, default=5)
    parser.add_argument("--max-per-creator", type=int, default=5)
    parser.add_argument("--dhash-distance", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    if not 1 <= args.per_class <= 200:
        parser.error("--per-class must be between 1 and 200")
    if not 1 <= args.api_batches <= 20:
        parser.error("--api-batches must be between 1 and 20")
    if not 1 <= args.max_per_creator <= 50:
        parser.error("--max-per-creator must be between 1 and 50")
    if not 0 <= args.dhash_distance <= 16:
        parser.error("--dhash-distance must be between 0 and 16")
    if not 0 <= args.delay <= 10:
        parser.error("--delay must be between 0 and 10")
    return args


def main() -> int:
    args = parse_args()
    records = collect(args)
    counts = Counter(record["class_id"] for record in records)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": STATUS,
                "total": len(records),
                "class_counts": dict(sorted(counts.items())),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    missing = [name for name in ("grass_clump", "low_shrub", "young_tree") if counts[name] < args.per_class]
    if missing:
        print(f"targets not reached: {', '.join(missing)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
