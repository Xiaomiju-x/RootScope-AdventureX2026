#!/usr/bin/env python3
"""Mechanically stage the frozen XRD GGUF inside AdventureX.

This script only reads the exact source named by the frozen spec and writes an
exact copy, a deterministic manifest, and SHA256SUMS below AdventureX/output.
It never starts llama.cpp, opens a device, or changes networking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


SPEC_SCHEMA = "rootscope.readonly_llm_release_spec.v1"
SPEC_STATUS = "STAGING_SPEC_READ_ONLY_LOCAL_LLM_NOT_X5_QUALIFIED"
MANIFEST_SCHEMA = "rootscope.readonly_llm_release_manifest.v1"
MANIFEST_STATUS = "STAGED_READ_ONLY_LOCAL_LLM_NOT_X5_QUALIFIED"
_FALSE_FLAGS = (
    "human_reviewed",
    "data_locked",
    "model_candidate",
    "model_qualified",
    "x5_validated",
    "latency_measured",
    "llama_cpp_bundled",
    "service_started",
    "hardware_touched",
    "network_touched",
    "external_network_allowed",
    "tool_execution",
    "actuator_access",
    "execution_authority",
    "physical_authority",
    "physical_completion",
)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _load_spec(path: Path) -> Mapping[str, Any]:
    spec = _require_mapping(json.loads(path.read_text(encoding="utf-8")), "spec")
    if spec.get("schema") != SPEC_SCHEMA or spec.get("status") != SPEC_STATUS:
        raise ValueError("unsupported read-only LLM staging spec")
    artifact = _require_mapping(spec.get("selected_artifact"), "selected_artifact")
    required_artifact = {
        "source_relative_to_adventurex",
        "output_relative_to_adventurex",
        "filename",
        "size_bytes",
        "sha256",
        "copy_contract",
    }
    if set(artifact) != required_artifact:
        raise ValueError("selected_artifact keys mismatch")
    flags = _require_mapping(spec.get("formal_flags"), "formal_flags")
    if set(flags) != set(_FALSE_FLAGS) or any(flags[name] is not False for name in _FALSE_FLAGS):
        raise ValueError("every formal flag in the staging spec must be boolean false")
    runtime = _require_mapping(spec.get("runtime_contract"), "runtime_contract")
    if (
        runtime.get("host") != "127.0.0.1"
        or runtime.get("default_enabled") is not False
        or runtime.get("manual_start_only") is not True
        or runtime.get("external_network_allowed") is not False
        or runtime.get("read_only") is not True
        or runtime.get("tool_execution") is not False
        or runtime.get("actuator_access") is not False
    ):
        raise ValueError("runtime contract is not loopback-only/read-only/manual")
    dependency = _require_mapping(spec.get("llama_cpp_dependency"), "llama_cpp_dependency")
    if dependency.get("bundled") is not False or dependency.get("x5_binary_selected") is not False:
        raise ValueError("llama.cpp must remain an unbundled, unselected X5 dependency")
    return spec


def stage(adventurex_root: Path, spec_path: Path) -> Mapping[str, Any]:
    root = adventurex_root.resolve(strict=True)
    spec_path = spec_path.resolve(strict=True)
    if not _under(spec_path, root):
        raise ValueError("staging spec must be inside AdventureX")
    spec = _load_spec(spec_path)
    artifact = spec["selected_artifact"]

    source = (root / str(artifact["source_relative_to_adventurex"])).resolve(strict=True)
    expected_source = (root.parent / "models" / str(artifact["filename"])).resolve(strict=True)
    if source != expected_source or not source.is_file():
        raise ValueError("source must be the exact frozen XRD models artifact")
    expected_size = int(artifact["size_bytes"])
    expected_sha = str(artifact["sha256"])
    if source.stat().st_size != expected_size or _sha256_file(source) != expected_sha:
        raise ValueError("source GGUF size or SHA-256 does not match the frozen spec")

    output_dir = (root / str(artifact["output_relative_to_adventurex"])).resolve()
    allowed_output_root = (root / "output").resolve(strict=True)
    if not _under(output_dir, allowed_output_root):
        raise ValueError("release output must stay below AdventureX/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / str(artifact["filename"])
    if destination.exists():
        if destination.is_symlink() or destination.stat().st_size != expected_size:
            raise ValueError("existing staged GGUF is not the exact expected regular file")
        if _sha256_file(destination) != expected_sha:
            raise ValueError("existing staged GGUF hash mismatch; refusing overwrite")
    else:
        partial = destination.with_suffix(destination.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        try:
            shutil.copyfile(source, partial)
            if partial.stat().st_size != expected_size or _sha256_file(partial) != expected_sha:
                raise ValueError("mechanical copy verification failed")
            os.replace(partial, destination)
        finally:
            if partial.exists():
                partial.unlink()

    spec_sha = _sha256_file(spec_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": MANIFEST_STATUS,
        "release_id": spec["release_id"],
        "artifact_staged": True,
        "artifact": {
            "filename": destination.name,
            "size_bytes": destination.stat().st_size,
            "sha256": _sha256_file(destination),
            "source_relative_to_adventurex": artifact["source_relative_to_adventurex"],
            "destination_relative_to_adventurex": destination.relative_to(root).as_posix(),
            "copy_contract": artifact["copy_contract"],
        },
        "staging_spec": {
            "relative_to_adventurex": spec_path.relative_to(root).as_posix(),
            "sha256": spec_sha,
        },
        "runtime_contract": dict(spec["runtime_contract"]),
        "llama_cpp_dependency": dict(spec["llama_cpp_dependency"]),
        "formal_flags": dict(spec["formal_flags"]),
    }
    manifest_path = output_dir / "release_manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    manifest_sha = _sha256_file(manifest_path)
    sums_path = output_dir / "SHA256SUMS"
    sums_path.write_text(
        f"{expected_sha}  {destination.name}\n{manifest_sha}  {manifest_path.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": MANIFEST_STATUS,
        "release_directory": str(output_dir),
        "artifact_sha256": expected_sha,
        "artifact_size_bytes": expected_size,
        "manifest_sha256": manifest_sha,
        "sha256sums_sha256": _sha256_file(sums_path),
        "formal_flags": dict(spec["formal_flags"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[4]
    parser.add_argument("--adventurex-root", type=Path, default=default_root)
    parser.add_argument(
        "--spec",
        type=Path,
        default=default_root / "rootscope/deploy/x5/readonly_llm_release_spec.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(stage(args.adventurex_root, args.spec), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
