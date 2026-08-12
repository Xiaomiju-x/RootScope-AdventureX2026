#!/usr/bin/env python3
"""Acquire a second, metadata-gated young-tree candidate pool.

Unlike the broad E1 pass, this collector uses Commons full-text search only and
requires youth evidence (young/sapling/seedling/juvenile/nursery or a narrow
morphological equivalent) in the Commons title or ImageDescription.  Explicit
mature/ancient/old/large-tree language is rejected before download.

The output is candidate-only and isolated from E0, E1, and formal review files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import collect_whole_plant_reacquisition as base


STATUS = "MACHINE_ACQUIRED_YOUNG_TREE_METADATA_GATED_CANDIDATES_NOT_TRAIN_READY"
YOUTH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("young", re.compile(r"\byoung\b", re.IGNORECASE)),
    ("sapling", re.compile(r"\bsaplings?\b", re.IGNORECASE)),
    ("seedling", re.compile(r"\bseedlings?\b", re.IGNORECASE)),
    ("juvenile", re.compile(r"\bjuveniles?\b", re.IGNORECASE)),
    ("nursery", re.compile(r"\bnurser(?:y|ies)\b", re.IGNORECASE)),
    ("plantlet", re.compile(r"\bplantlets?\b", re.IGNORECASE)),
    ("treelet", re.compile(r"\btreelets?\b", re.IGNORECASE)),
    ("newly_planted", re.compile(r"\b(?:newly|recently) planted\b", re.IGNORECASE)),
)
MATURE_REJECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mature", re.compile(r"\bmature\b", re.IGNORECASE)),
    ("ancient", re.compile(r"\bancient\b", re.IGNORECASE)),
    ("old_tree", re.compile(r"\bold (?:growth )?(?:tree|acacia|prosopis|vachellia|tamarix)\b", re.IGNORECASE)),
    ("large_tree", re.compile(r"\blarge (?:old )?(?:tree|acacia|prosopis|vachellia|tamarix)\b", re.IGNORECASE)),
    ("veteran_tree", re.compile(r"\bveteran tree\b", re.IGNORECASE)),
    ("monumental_tree", re.compile(r"\bmonumental tree\b", re.IGNORECASE)),
    ("centenarian", re.compile(r"\bcentenarian\b", re.IGNORECASE)),
)
DETAIL_REJECT = re.compile(
    r"\b(close[- ]?up|macro|herbarium|specimen|leaf detail|leaves detail|"
    r"flower detail|fruit detail|seed pod detail|bark detail|branch detail|"
    r"twig detail|thorn detail|spine detail|root section|microscope)\b",
    re.IGNORECASE,
)


def young_source(query: str, hint: str) -> base.Source:
    return base.Source(
        class_id="young_tree",
        retrieval_mode="search",
        retrieval_query=query,
        species_hint=hint,
        acquisition_query=(
            f"{query}: whole young desert tree sapling/seedling; trunk base visible; "
            "entire crown visible; isolated single young plant; metadata must explicitly "
            "state young/sapling/seedling/juvenile/nursery; reject mature/ancient/old/large tree"
        ),
    )


SOURCE_PLAN: tuple[base.Source, ...] = tuple(
    young_source(query, hint)
    for query, hint in (
        ('Acacia seedlings nursery', "Acacia spp. nursery seedlings"),
        ('Acacia plantlet', "Acacia spp. plantlet"),
        ('mesquite sapling', "Prosopis spp. (mesquite) sapling"),
        ('mesquite seedling', "Prosopis spp. (mesquite) seedling"),
        ('"young mesquite" tree', "young Prosopis spp. (mesquite)"),
        ('ghaf sapling', "Prosopis cineraria (ghaf) sapling"),
        ('ghaf seedling', "Prosopis cineraria (ghaf) seedling"),
        ('palo verde sapling', "Parkinsonia spp. (palo verde) sapling"),
        ('palo verde seedling', "Parkinsonia spp. (palo verde) seedling"),
        ('desert willow sapling', "Chilopsis linearis sapling"),
        ('desert willow seedling', "Chilopsis linearis seedling"),
        ('Argania spinosa sapling', "Argania spinosa sapling"),
        ('Argania spinosa seedling', "Argania spinosa seedling"),
        ('Moringa peregrina seedling', "Moringa peregrina seedling"),
        ('dryland reforestation seedlings', "dryland tree nursery seedlings"),
        ('arid afforestation seedlings', "arid tree nursery seedlings"),
        ('"desert tree" sapling', "desert tree sapling"),
        ('"desert tree" seedling', "desert tree seedling"),
        ('"young tree" desert', "young desert tree"),
        ('"arid tree" sapling', "arid tree sapling"),
        ('"arid tree" seedling', "arid tree seedling"),
        ('"tree nursery" desert sapling', "desert nursery sapling"),
        ('"tree nursery" arid seedling', "arid nursery seedling"),
        ('Acacia sapling', "Acacia spp. sapling"),
        ('Acacia seedling', "Acacia spp. seedling"),
        ('"young Acacia" tree', "young Acacia spp."),
        ('Acacia nursery seedling', "Acacia spp. nursery seedling"),
        ('Acacia juvenile tree', "juvenile Acacia spp."),
        ('Vachellia sapling', "Vachellia spp. sapling"),
        ('Vachellia seedling', "Vachellia spp. seedling"),
        ('"young Vachellia" tree', "young Vachellia spp."),
        ('Vachellia tortilis sapling', "Vachellia tortilis sapling"),
        ('Vachellia tortilis seedling', "Vachellia tortilis seedling"),
        ('Vachellia erioloba sapling', "Vachellia erioloba sapling"),
        ('Vachellia nilotica seedling', "Vachellia nilotica seedling"),
        ('Prosopis sapling', "Prosopis spp. sapling"),
        ('Prosopis seedling', "Prosopis spp. seedling"),
        ('"young Prosopis" tree', "young Prosopis spp."),
        ('Prosopis cineraria sapling', "Prosopis cineraria sapling"),
        ('Prosopis cineraria seedling', "Prosopis cineraria seedling"),
        ('Prosopis juliflora seedling', "Prosopis juliflora seedling"),
        ('Tamarix sapling', "Tamarix spp. sapling"),
        ('Tamarix seedling', "Tamarix spp. seedling"),
        ('Populus euphratica sapling', "Populus euphratica sapling"),
        ('Populus euphratica seedling', "Populus euphratica seedling"),
        ('Balanites aegyptiaca seedling', "Balanites aegyptiaca seedling"),
        ('Senegalia senegal seedling', "Senegalia senegal seedling"),
        ('Parkinsonia aculeata seedling', "Parkinsonia aculeata seedling"),
        ('Boscia albitrunca seedling', "Boscia albitrunca seedling"),
        ('"tree seedling" growing sand', "tree seedling in sand"),
        ('"tree sapling" desert nursery', "tree sapling in desert nursery"),
        ('"young tree" dryland', "young dryland tree"),
        ('"young tree" arid', "young arid tree"),
    )
)


def metadata_gate(title: str, description: str) -> tuple[bool, list[str], list[str]]:
    text = f"{title} {description}"
    youth = [name for name, pattern in YOUTH_PATTERNS if pattern.search(text)]
    mature = [name for name, pattern in MATURE_REJECT_PATTERNS if pattern.search(text)]
    if not youth or mature:
        return False, youth, mature
    if base.REJECT_COMMON.search(text) or base.REJECT_DETAIL.search(text) or DETAIL_REJECT.search(text):
        return False, youth, mature
    return True, youth, mature


def source_plan_hash() -> str:
    payload = [item.__dict__ for item in SOURCE_PLAN]
    return base.sha256_bytes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def save_outputs(
    output: Path,
    records: list[dict[str, Any]],
    policy_sha: str,
    existing_count: int,
) -> None:
    ordered = sorted(records, key=lambda item: int(item["pageid"]))
    manifest_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in ordered
    )
    base.atomic_text(output / "manifest.jsonl", manifest_text)
    plan_sha = source_plan_hash()
    summary = {
        "schema_version": "rootscope.young_tree_reacquisition_summary.v1",
        "status": STATUS,
        "generated_at_utc": base.utc_now(),
        "total": len(ordered),
        "class_counts": dict(Counter(record["class_id"] for record in ordered)),
        "provider_counts": dict(Counter(record["source_provider"] for record in ordered)),
        "license_counts": dict(Counter(record["license_canonical_name"] for record in ordered)),
        "metadata_youth_term_counts": dict(
            Counter(term for record in ordered for term in record["metadata_youth_matches"])
        ),
        "existing_records_screened_for_overlap": existing_count,
        "exact_pageid_overlap_with_commons_existing": 0,
        "exact_commons_sha1_overlap_with_existing": 0,
        "exact_download_sha256_overlap_with_existing": 0,
        "all_metadata_youth_gate_passed": True,
        "all_mature_reject_matches_empty": True,
        "all_splits": "UNASSIGNED_DO_NOT_TRAIN",
        "all_training_eligible": False,
        "all_print_eligible": False,
        "formal_human_review_authority": False,
        "machine_candidate_class_is_ground_truth": False,
        "manifest_sha256": base.sha256_bytes(manifest_text.encode("utf-8")),
        "license_policy_sha256": policy_sha,
        "source_plan_sha256": plan_sha,
        "legal_note": (
            "Metadata youth terms improve retrieval precision but do not establish whole-plant "
            "visual truth, age, class, license warranty, or non-copyright rights clearance."
        ),
    }
    base.atomic_text(
        output / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    plan = {
        "schema_version": "rootscope.young_tree_reacquisition_source_plan.v1",
        "source_provider": "Wikimedia Commons",
        "api_endpoint": base.API_URL,
        "official_api_documentation": "https://www.mediawiki.org/wiki/API:Search",
        "source_plan_sha256": plan_sha,
        "metadata_gate": {
            "required_youth_terms": [name for name, _ in YOUTH_PATTERNS],
            "mature_reject_terms": [name for name, _ in MATURE_REJECT_PATTERNS],
            "title_or_image_description_only": True,
        },
        "visual_ground_truth_authority": False,
        "required_downstream_action": (
            "Run the strict whole-plant visual gate; failures become EXCLUDE/HOLD, never an "
            "automatic young_tree positive."
        ),
        "sources": [item.__dict__ for item in SOURCE_PLAN],
    }
    base.atomic_text(
        output / "source_plan.json",
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    attribution = [
        "# RootScope 幼树专向 E2｜来源与署名",
        "",
        "> 机器采集、元数据硬筛候选；不是视觉标签或权利审核结论。",
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
    base.atomic_text(output / "ATTRIBUTION.md", "\n".join(attribution) + "\n")
    readme = f"""# RootScope young-tree-only reacquisition E2

Status: `{STATUS}`

This directory is independent from E0 and E1. Candidates are generated using
explicit `sapling`, `seedling`, `young tree`, `juvenile`, and nursery searches.
The Commons title or ImageDescription must contain a youth term, and explicit
`mature`, `ancient`, `old tree`, or `large tree` language is rejected before
download. These are metadata constraints, not visual truth.

- `manifest.jsonl`: provenance-rich unassigned candidate records.
- `source_plan.json`: exact queries and hard metadata gate.
- `summary.json`: counts and evidence roots.
- `ATTRIBUTION.md`: attribution aid, pending file-page verification.
- `recovery_log.jsonl`: append-only acquisition events.
- `images/young_tree/`: downloaded Commons derivatives.

Every record remains `UNASSIGNED_DO_NOT_TRAIN`, `training_eligible=false`, and
`print_eligible=false`. A strict pixel-level whole-sapling gate is mandatory.
"""
    base.atomic_text(output / "README.md", readme)


def collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    output = args.output.resolve()
    existing_roots = [path.resolve() for path in args.existing]
    if output in existing_roots:
        raise ValueError("E2 output must be independent from every existing dataset")
    output.mkdir(parents=True, exist_ok=True)
    event_log = output / "recovery_log.jsonl"

    policy_bytes = args.license_policy.resolve().read_bytes()
    policy_sha = base.sha256_bytes(policy_bytes)
    policy = json.loads(policy_bytes.decode("utf-8-sig"))
    bindings = base.license_table(policy)

    records = base.load_jsonl(output / "manifest.jsonl")
    existing_records: list[dict[str, Any]] = []
    for root in existing_roots:
        existing_records.extend(base.load_jsonl(root / "manifest.jsonl"))
    all_known = existing_records + records
    used_pageids = {int(record["pageid"]) for record in all_known if record.get("pageid")}
    used_sha1 = {
        str(record["commons_sha1"]).lower() for record in all_known if record.get("commons_sha1")
    }
    used_sha256 = {
        str(record["download_sha256"]).lower()
        for record in all_known
        if record.get("download_sha256")
    }
    known_dhashes = [
        str(record["dhash64"]).lower()
        for record in all_known
        if re.fullmatch(r"[0-9a-fA-F]{16}", str(record.get("dhash64", "")))
    ]
    creator_counts = Counter(record.get("creator_group", "") for record in records)

    base.append_event(
        event_log,
        "run_started",
        target=args.target,
        existing_records=len(existing_records),
        source_plan_sha256=source_plan_hash(),
        license_policy_sha256=policy_sha,
    )
    save_outputs(output, records, policy_sha, len(existing_records))

    for item in SOURCE_PLAN:
        if len(records) >= args.target:
            break
        print(f"[young_tree E2] {item.retrieval_query} ({len(records)}/{args.target})", flush=True)
        try:
            pages = base.commons_pages(item, args.api_batches)
            for page in pages:
                if len(records) >= args.target:
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
                description = base.meta_value(metadata, "ImageDescription")
                passed, youth_matches, mature_matches = metadata_gate(title, description)
                if not passed:
                    continue
                mime = str(info.get("mime") or "")
                original_width = int(info.get("width") or 0)
                original_height = int(info.get("height") or 0)
                if mime not in base.ALLOWED_MIME or min(original_width, original_height) < base.MIN_ORIGINAL_SIDE:
                    continue
                artist = base.meta_value(metadata, "Artist") or base.meta_value(metadata, "Credit")
                if not artist:
                    continue
                creator_group = base.normalized_creator_group(artist)
                if creator_counts[creator_group] >= args.max_per_creator:
                    continue
                raw_name = base.meta_value(metadata, "LicenseShortName") or base.meta_value(metadata, "UsageTerms")
                raw_url = base.meta_value(metadata, "LicenseUrl")
                copyrighted = base.meta_value(metadata, "Copyrighted")
                canonical = base.resolve_license(bindings, raw_name, raw_url, copyrighted)
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
                    downloaded_at = base.utc_now()
                    payload = base.download(download_url, source_page)
                    digest = base.sha256_bytes(payload)
                    if digest in used_sha256:
                        continue
                    width, height, decoded_mime, dhash = base.image_facts(payload)
                    if decoded_mime != mime or min(width, height) < base.MIN_DOWNLOADED_SIDE:
                        continue
                    min_distance = min(
                        (base.hamming(dhash, known) for known in known_dhashes), default=64
                    )
                    if min_distance <= args.dhash_distance:
                        continue
                    extension = ".png" if mime == "image/png" else ".jpg"
                    filename_only = f"young_tree_{pageid}_{digest[:12]}{extension}"
                    relative = f"images/young_tree/{filename_only}"
                    destination = output / Path(relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists() and base.sha256_file(destination) != digest:
                        raise RuntimeError(f"refusing to replace a different file: {destination}")
                    if not destination.exists():
                        temporary = destination.with_suffix(destination.suffix + ".download")
                        temporary.write_bytes(payload)
                        temporary.replace(destination)
                    record = {
                        "schema_version": base.SCHEMA_VERSION,
                        "class_id": "young_tree",
                        "species_hint": item.species_hint,
                        "species_hint_status": "acquisition_hint_not_a_reviewed_species_or_shape_label",
                        "candidate_label_status": "youth_metadata_gated_unverified_visual_candidate",
                        "acquisition_mode": "search",
                        "acquisition_query": item.acquisition_query,
                        "query": item.acquisition_query,
                        "retrieval_query": item.retrieval_query,
                        "metadata_gate_version": "rootscope_young_tree_metadata_gate_v1",
                        "metadata_gate_passed": True,
                        "metadata_gate_scope": "commons_title_plus_image_description",
                        "metadata_youth_matches": youth_matches,
                        "metadata_mature_reject_matches": mature_matches,
                        "structural_intent": {
                            "whole_plant": True,
                            "trunk_base_visible": True,
                            "entire_crown_visible": True,
                            "isolated_single_plant": True,
                            "sapling_or_seedling_not_mature_tree": True,
                            "status": "query_and_metadata_intent_not_visually_verified",
                        },
                        "domain": "natural_web_candidate",
                        "split": "UNASSIGNED_DO_NOT_TRAIN",
                        "review_status": "pending_strict_machine_visual_triage_and_human_rights_review",
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
                        "credit": base.meta_value(metadata, "Credit"),
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
                        "usage_terms": base.meta_value(metadata, "UsageTerms"),
                        "attribution_required": base.meta_value(metadata, "AttributionRequired"),
                        "copyrighted": copyrighted,
                        "restrictions": base.meta_value(metadata, "Restrictions"),
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
                        "dhash64_algorithm": base.DHASH_ALGORITHM,
                        "dhash64": dhash,
                        "minimum_prior_or_reacquisition_dhash_distance": min_distance,
                    }
                    records.append(record)
                    used_pageids.add(pageid)
                    used_sha1.add(commons_sha1)
                    used_sha256.add(digest)
                    known_dhashes.append(dhash)
                    creator_counts[creator_group] += 1
                    save_outputs(output, records, policy_sha, len(existing_records))
                    base.append_event(
                        event_log,
                        "candidate_saved",
                        pageid=pageid,
                        filename=relative,
                        sha256=digest,
                        retrieval_query=item.retrieval_query,
                        metadata_youth_matches=youth_matches,
                    )
                    print(f"  saved {len(records):02d}/{args.target}: {relative}", flush=True)
                    time.sleep(args.delay)
                except Exception as exc:
                    base.append_event(
                        event_log,
                        "candidate_failed",
                        pageid=pageid,
                        title=title,
                        error=str(exc),
                    )
        except Exception as exc:
            base.append_event(
                event_log,
                "source_query_failed",
                retrieval_query=item.retrieval_query,
                error=str(exc),
            )

    save_outputs(output, records, policy_sha, len(existing_records))
    base.append_event(event_log, "run_finished", total=len(records))
    return records


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    adventurex = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=adventurex / "datasets" / "desert_plants_young_tree_reacquisition_e2",
    )
    parser.add_argument(
        "--existing",
        type=Path,
        action="append",
        default=None,
        help="dataset root to exclude; repeatable",
    )
    parser.add_argument(
        "--license-policy",
        type=Path,
        default=script.with_name("wikimedia_license_policy_v1.json"),
    )
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--api-batches", type=int, default=8)
    parser.add_argument("--max-per-creator", type=int, default=5)
    parser.add_argument("--dhash-distance", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args()
    if args.existing is None:
        args.existing = [
            adventurex / "datasets" / "desert_plants_v1",
            adventurex / "datasets" / "desert_plants_wikimedia_staging_e0",
            adventurex / "datasets" / "desert_plants_whole_plant_reacquisition_e1",
        ]
    if not 1 <= args.target <= 300:
        parser.error("--target must be between 1 and 300")
    if not 1 <= args.api_batches <= 20:
        parser.error("--api-batches must be between 1 and 20")
    if not 1 <= args.max_per_creator <= 50:
        parser.error("--max-per-creator must be between 1 and 50")
    if not 0 <= args.dhash_distance <= 16:
        parser.error("--dhash-distance must be between 0 and 16")
    return args


def main() -> int:
    args = parse_args()
    records = collect(args)
    result = {
        "output": str(args.output.resolve()),
        "status": STATUS,
        "total": len(records),
        "target": args.target,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if len(records) < args.target:
        print(f"target not reached: {len(records)}/{args.target}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
