"""Fail-closed RootScope v3 E0 verifier.

This verifier is intentionally standard-library only.  It reads local files,
does not read environment-variable values, does not access the network, and
does not open devices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


EXPECTED_V2_SHA256 = (
    "03ca7b8d9ff8b691f1fd61dc696601ba30f494377a0b2a3cfadb66c19478ed94"
)
EXPECTED_V2_SIZE = 13_486_080
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SECRET_KEY_NAMES = {
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "private_key",
}
ALLOWED_TEACHER_ENV_NAMES = {
    "ROOTSCOPE_QWEN_API_KEY",
    "ROOTSCOPE_DEEPSEEK_API_KEY",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _walk(value: Any, prefix: str = "$"):
    if isinstance(value, dict):
        for key, child in value.items():
            yield f"{prefix}.{key}", key, child
            yield from _walk(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{prefix}[{index}]")


def _require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def verify(
    adventurex_root: Path, *, require_private_archive: bool = False
) -> dict[str, Any]:
    root = adventurex_root.resolve()
    v3 = root / "rootscope_v3"
    checks: list[str] = []

    _require(v3.is_dir(), "v3 work root exists", checks)

    baseline = _load(v3 / "governance" / "baseline_v2_snapshot.json")
    archive = (
        root
        / baseline["release"]["relative_path"]
        / baseline["release"]["archive_name"]
    )
    archive_present = archive.is_file()
    if require_private_archive:
        _require(archive_present, "v2 archive exists", checks)
    if archive_present:
        _require(
            archive.stat().st_size == EXPECTED_V2_SIZE,
            "v2 archive size unchanged",
            checks,
        )
        _require(
            _sha256(archive) == EXPECTED_V2_SHA256,
            "v2 archive SHA-256 unchanged",
            checks,
        )
    else:
        checks.append(
            "private v2 archive intentionally absent; verify recorded digest only"
        )
    _require(
        baseline["release"]["archive_sha256"] == EXPECTED_V2_SHA256,
        "baseline snapshot binds expected v2 SHA-256",
        checks,
    )
    _require(
        all(value is False for value in baseline["authority"].values()),
        "baseline snapshot preserves zero physical authority",
        checks,
    )

    registry_names = ("models", "datasets", "teachers", "dependencies")
    registries = {
        name: _load(v3 / "registries" / f"{name}.v1.json")
        for name in registry_names
    }
    _require(len(registries["models"]["models"]) >= 8, "model registry has baseline and challengers", checks)
    _require(len(registries["datasets"]["datasets"]) >= 7, "dataset registry has baseline and planned domains", checks)
    _require(len(registries["teachers"]["teachers"]) >= 5, "teacher registry has cloud and local candidates", checks)
    _require(len(registries["dependencies"]["dependencies"]) >= 8, "dependency registry freezes X5 ABI boundary", checks)

    for model in registries["models"]["models"]:
        locator = model["relative_locator"]
        if locator is not None:
            artifact = root / locator
            _require(artifact.exists(), f"model locator exists for {model['model_id']}", checks)
            expected_hash = model["sha256"]
            if expected_hash is not None:
                _require(artifact.is_file(), f"hashed model is a file for {model['model_id']}", checks)
                _require(HEX64.fullmatch(expected_hash) is not None, f"model hash format is valid for {model['model_id']}", checks)
                _require(_sha256(artifact) == expected_hash, f"model hash matches for {model['model_id']}", checks)

    for dataset in registries["datasets"]["datasets"]:
        locator = dataset["relative_locator"]
        if locator is not None:
            dataset_exists = (root / locator).exists()
            _require(
                dataset_exists
                or dataset.get("public_redistribution") is False,
                f"dataset locator exists or is explicitly non-redistributed for {dataset['dataset_id']}",
                checks,
            )
        manifest_locator = dataset["manifest_relative_locator"]
        expected_hash = dataset.get("manifest_sha256")
        if manifest_locator is not None:
            manifest = root / manifest_locator
            manifest_exists = manifest.is_file()
            _require(
                manifest_exists
                or dataset.get("public_redistribution") is False,
                f"dataset manifest exists or is explicitly non-redistributed for {dataset['dataset_id']}",
                checks,
            )
            if manifest_exists and expected_hash is not None:
                _require(_sha256(manifest) == expected_hash, f"dataset manifest hash matches for {dataset['dataset_id']}", checks)

    for name, registry in registries.items():
        for location, key, value in _walk(registry):
            lowered = key.lower()
            _require(
                lowered not in SECRET_KEY_NAMES,
                f"{name} registry has no secret-bearing field at {location}",
                checks,
            )
            if key == "credential_environment_variable" and value is not None:
                _require(
                    value in ALLOWED_TEACHER_ENV_NAMES,
                    f"teacher credential reference is an allowlisted variable name at {location}",
                    checks,
                )
    _require(
        registries["teachers"]["secret_policy"]["secret_values_stored"] is False,
        "teacher registry explicitly forbids stored secret values",
        checks,
    )
    for teacher in registries["teachers"]["teachers"]:
        _require(teacher["physical_authority"] is False, f"teacher {teacher['teacher_id']} has zero authority", checks)
        if teacher["provider"] != "LOCAL_PC":
            _require(teacher["status"] == "PLANNED_NOT_CALLED", f"teacher {teacher['teacher_id']} was not called in E0", checks)

    schema_dir = v3 / "schemas" / "evaluation"
    example_dir = v3 / "examples" / "evaluation"
    schema_stems = ("vision", "llm", "rag", "resource", "physical_loop")
    for stem in schema_stems:
        schema = _load(schema_dir / f"{stem}_evaluation.schema.json")
        example = _load(example_dir / f"{stem}.fixture.json")
        _require(
            schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
            f"{stem} uses JSON Schema 2020-12",
            checks,
        )
        _require(schema.get("type") == "object", f"{stem} schema root is object", checks)
        _require(bool(schema.get("required")), f"{stem} schema has required fields", checks)
        _require(
            example.get("schema") == schema["properties"]["schema"]["const"],
            f"{stem} fixture binds schema identity",
            checks,
        )

    state = _load(
        v3
        / "candidates"
        / "rootscope_v3_candidate_unqualified"
        / "CANDIDATE_STATE.json"
    )
    _require(
        state["status"] == "SKELETON_NOT_A_RELEASE_DO_NOT_DEPLOY",
        "candidate skeleton is explicitly non-deployable",
        checks,
    )
    _require(
        state["baseline_v2_archive_sha256"] == EXPECTED_V2_SHA256,
        "candidate skeleton binds v2 rollback hash",
        checks,
    )
    _require(
        all(value is False for value in state["authority"].values()),
        "candidate skeleton has zero authority",
        checks,
    )
    _require(
        state["gates"]["e0_contracts_frozen"] is True
        and all(
            value is False
            for key, value in state["gates"].items()
            if key != "e0_contracts_frozen"
        ),
        "only the E0 contract gate is passed",
        checks,
    )

    return {
        "schema": "rootscope.v3.e0-verification.v1",
        "status": "PASS_E0_FACTS_REGISTRIES_SCHEMAS_CANDIDATE_ZERO_AUTHORITY",
        "check_count": len(checks),
        "v2_archive_sha256": EXPECTED_V2_SHA256,
        "v2_archive_present": archive_present,
        "x5_contacted": False,
        "network_accessed": False,
        "environment_values_read": False,
        "device_opened": False,
        "physical_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adventurex-root", type=Path, required=True)
    parser.add_argument(
        "--require-private-archive",
        action="store_true",
        help="also require and hash the non-redistributed v2 archive",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(
            args.adventurex_root,
            require_private_archive=args.require_private_archive,
        )
    except (AssertionError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 1
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
