"""Freeze three machine-curated RootScope demo references.

This is deliberately a *demo-reference* operation, not a data-lock, human
review, rights approval, holdout evaluation, or model qualification step.  It
copies three already selected positive source images byte-for-byte into the
RootScope application and emits the strict registry consumed by
``app.vision.dual_path_demo``.  The unknown/sand-dune card is never registered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


DATASET_MANIFEST_SHA256 = (
    "bf14c7423aad965b8af736c7d77cef1ba134d78dd1f905c03cc14cff1192f3fe"
)
REGISTRY_STATUS = "FROZEN_EXPERIMENTAL_DEMO_REFERENCES"
REGISTERED_ROLE = "DEMO_REFERENCE_NOT_HOLDOUT_ONCE_REGISTERED"
NEGATIVE_PAGE_ID = 157364276

TEMPLATES = (
    {
        "pageid": 163498042,
        "template_id": "grass-clump-163498042",
        "class_name": "grass_clump",
        "source": "images/grass_clump/grass_clump_163498042_b1f6262895c3.jpg",
        "destination": "grass_clump_163498042.jpg",
        "sha256": "b1f6262895c31e8e507be31cebba09140e2a2582aa4f266ab05261fe50751d23",
        "source_url": "https://commons.wikimedia.org/wiki/File:Stipagrostis_plumosa_kz06.jpg",
        "creator": "Krzysztof Ziarnek, Kenraiz",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    },
    {
        "pageid": 68787114,
        "template_id": "low-shrub-68787114",
        "class_name": "low_shrub",
        "source": "images/low_shrub/low_shrub_68787114_810c7649ac72.jpg",
        "destination": "low_shrub_68787114.jpg",
        "sha256": "810c7649ac729105367b3213bfafc467a036f4054244c424613da6c027c73610",
        "source_url": "https://commons.wikimedia.org/wiki/File:Plants22_(27104657009).jpg",
        "creator": "USDA NRCS Montana",
        "license": "Public domain",
        "license_url": "https://commons.wikimedia.org/wiki/Commons:Public_domain",
    },
    {
        "pageid": 92774234,
        "template_id": "young-tree-92774234",
        "class_name": "young_tree",
        "source": "images/young_tree/young_tree_92774234_0d994e838a2d.jpg",
        "destination": "young_tree_92774234.jpg",
        "sha256": "0d994e838a2d7787ab3edfd8646e317390c790d92588c7ef9109778b843b40eb",
        "source_url": "https://commons.wikimedia.org/wiki/File:A_newly_planted_tree.jpg",
        "creator": "Wogatha Kanyi",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def load_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        pageid = value.get("pageid")
        if not isinstance(pageid, int) or pageid in records:
            raise ValueError(f"invalid or duplicate pageid at manifest line {line_number}")
        records[pageid] = value
    return records


def build(adventurex: Path) -> dict[str, Any]:
    adventurex = adventurex.resolve(strict=True)
    dataset = adventurex / "datasets" / "rootscope_machine_curated_provisional_v3"
    manifest = dataset / "manifest.jsonl"
    if sha256_file(manifest) != DATASET_MANIFEST_SHA256:
        raise ValueError("v3 dataset manifest hash mismatch")

    records = load_records(manifest)
    negative = records.get(NEGATIVE_PAGE_ID)
    if not negative or negative.get("class_id") != "unknown":
        raise ValueError("the frozen negative card identity is missing or no longer unknown")

    vision_root = adventurex / "rootscope" / "app" / "vision"
    template_root = vision_root / "known_card_templates"
    template_root.mkdir(parents=True, exist_ok=True)
    registry_items: list[dict[str, Any]] = []
    receipt_items: list[dict[str, Any]] = []

    expected_destinations = {str(item["destination"]) for item in TEMPLATES}
    existing_files = {path.name for path in template_root.iterdir() if path.is_file()}
    unexpected = sorted(existing_files - expected_destinations)
    if unexpected:
        raise ValueError(f"unexpected files already exist in template root: {unexpected}")

    for item in TEMPLATES:
        record = records.get(int(item["pageid"]))
        if record is None:
            raise ValueError(f"dataset record missing for pageid={item['pageid']}")
        required = {
            "class_id": item["class_name"],
            "filename": item["source"],
            "copied_image_sha256": item["sha256"],
            "source_page": item["source_url"],
            "artist": item["creator"],
            "license": item["license"],
            "experimental_split_suggestion": "EXPERIMENTAL_TRAIN_SUGGESTION",
            "human_reviewed": False,
            "rights_approved": False,
            "print_eligible": False,
            "formal_a1_dataset": False,
        }
        mismatches = {
            key: {"actual": record.get(key), "expected": expected}
            for key, expected in required.items()
            if record.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"source record mismatch for pageid={item['pageid']}: {mismatches}")

        source = dataset / str(item["source"])
        if not source.is_file() or sha256_file(source) != item["sha256"]:
            raise ValueError(f"source image hash mismatch for pageid={item['pageid']}")
        destination = template_root / str(item["destination"])
        temporary = destination.with_suffix(destination.suffix + ".partial")
        if temporary.exists():
            temporary.unlink()
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != item["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"mechanical copy mismatch for pageid={item['pageid']}")
        temporary.replace(destination)

        registry_items.append(
            {
                "template_id": item["template_id"],
                "class_name": item["class_name"],
                "relative_path": item["destination"],
                "raw_sha256": item["sha256"],
                "role": REGISTERED_ROLE,
                "dataset_record": {
                    "record_id": f"commons-{item['pageid']}",
                    "source_manifest": "datasets/rootscope_machine_curated_provisional_v3/manifest.jsonl",
                    "source_url": item["source_url"],
                    "attribution": {
                        "creator": item["creator"],
                        "license": item["license"],
                        "license_url": item["license_url"],
                    },
                },
            }
        )
        receipt_items.append(
            {
                "pageid": item["pageid"],
                "class_name": item["class_name"],
                "source_relative_to_dataset": item["source"],
                "destination_relative_to_vision": f"known_card_templates/{item['destination']}",
                "raw_sha256": item["sha256"],
                "role": REGISTERED_ROLE,
                "holdout_evidence": False,
                "generalization_evidence": False,
            }
        )

    registry = {
        "schema_version": "rootscope.known-card-template-registry.v1",
        "status": REGISTRY_STATUS,
        "template_root": "known_card_templates",
        "templates": registry_items,
    }
    registry_path = vision_root / "known_card_template_registry.frozen.experimental.json"
    write_json(registry_path, registry)

    receipt = {
        "schema": "rootscope.demo-template-registration-receipt.v1",
        "status": "FROZEN_EXPERIMENTAL_DEMO_REFERENCES_NOT_HOLDOUT_NOT_QUALIFIED",
        "dataset_manifest": {
            "relative_path": "datasets/rootscope_machine_curated_provisional_v3/manifest.jsonl",
            "sha256": DATASET_MANIFEST_SHA256,
            "mutated": False,
        },
        "registry": {
            "relative_path": "rootscope/app/vision/known_card_template_registry.frozen.experimental.json",
            "sha256": sha256_file(registry_path),
            "template_count": len(registry_items),
        },
        "templates": receipt_items,
        "negative_card": {
            "pageid": NEGATIVE_PAGE_ID,
            "class_name": "unknown",
            "registered": False,
            "use": "NEGATIVE_REJECTION_DEMO_ONLY",
        },
        "registration_state": {
            "positive_templates_registered": True,
            "positive_template_count": len(registry_items),
            "unknown_negative_registered": False,
        },
        "authority": {
            "template_registry_write_authority": False,
            "human_reviewed": False,
            "rights_approved": False,
            "print_eligible": False,
            "data_locked": False,
            "model_candidate": False,
            "model_qualified": False,
            "x5_validated": False,
            "bpu_compiled": False,
            "execution_authority": False,
            "physical_authority": False,
            "physical_completion": False,
        },
    }
    receipt_path = adventurex / "evidence" / "rootscope_demo_template_registry_receipt_20260717.json"
    write_json(receipt_path, receipt)
    return {
        "status": "PASS",
        "registry": str(registry_path),
        "registry_sha256": receipt["registry"]["sha256"],
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "templates": len(registry_items),
        "negative_registered": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adventurex",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    print(json.dumps(build(args.adventurex), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
