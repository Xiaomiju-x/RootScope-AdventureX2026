#!/usr/bin/env python3
"""Prove that the v2 bundle and installed core remain byte-identical."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
from typing import Any


sys.dont_write_bytecode = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_installer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("rootscope_v2_installer_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load immutable v2 installer for read-only audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(partial, 0o600)
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-tar", type=Path, required=True)
    parser.add_argument("--expected-outer-sha256", required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--installed-project", type=Path, required=True)
    parser.add_argument("--shim-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    outer_tar = args.outer_tar.resolve(strict=True)
    bundle = args.bundle_root.resolve(strict=True)
    project = args.installed_project.resolve(strict=True)
    shim = args.shim_root.resolve(strict=True)
    outer_sha = sha256_file(outer_tar)
    if outer_sha != args.expected_outer_sha256:
        raise ValueError("outer tar SHA-256 mismatch")

    installer = load_installer(bundle / "install_field_bundle_v2.py")
    bundle_verification = installer.verify_bundle(bundle)
    if bundle_verification["status"] != "PASS_HASHES_AND_SAFE_PATHS_NOT_X5_QUALIFIED":
        raise ValueError("outer bundle verification did not pass")

    nested = bundle / "components/rootscope_x5_offline_core_v1.tar"
    with tarfile.open(nested, "r:") as archive:
        member = archive.getmember(
            "rootscope_x5_offline_core_v1/release_manifest.json"
        )
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("nested core manifest is unreadable")
        manifest = json.loads(handle.read().decode("utf-8"))

    expected: dict[str, tuple[str, int]] = {}
    for record in manifest["files"]:
        name = str(record["path"])
        if name.startswith("rootscope/"):
            relative = name[len("rootscope/") :]
            expected[relative] = (str(record["sha256"]), int(record["bytes"]))
    actual_files = {
        path.relative_to(project).as_posix(): path
        for path in project.rglob("*")
        if path.is_file()
    }
    if set(actual_files) != set(expected):
        raise ValueError("installed core coverage differs from nested manifest")
    for relative, (expected_sha, expected_size) in expected.items():
        path = actual_files[relative]
        if (
            path.is_symlink()
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected_sha
        ):
            raise ValueError(f"installed core differs: {relative}")

    if shim.is_relative_to(bundle) or shim.is_relative_to(project):
        raise ValueError("compatibility shim must remain external")
    shim_files = {
        path.relative_to(shim).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(shim.rglob("*"))
        if path.is_file() and path.suffix != ".pyc"
    }
    if set(shim_files) != {"app/__init__.py", "sitecustomize.py"}:
        raise ValueError("external shim file set changed")

    receipt = {
        "schema": "rootscope.x5-v2-immutable-install-audit.v1",
        "status": "PASS_IMMUTABLE_V2_UNCHANGED_EXTERNAL_SHIM_ONLY",
        "outer_tar": {
            "path": str(outer_tar),
            "sha256": outer_sha,
            "size_bytes": outer_tar.stat().st_size,
        },
        "outer_bundle_verification": bundle_verification,
        "installed_core": {
            "path": str(project),
            "manifest_bound_files": len(expected),
            "exact_file_coverage": True,
            "all_hashes_and_sizes_match": True,
            "extra_files": [],
        },
        "compatibility_shim": {
            "path": str(shim),
            "external_to_bundle_and_installed_core": True,
            "files": shim_files,
            "persistent_environment_modified": False,
            "invocation_scope": (
                "PYTHONPATH=$HOME/.local/share/rootscope-field-v2/compat/"
                "field_bundle_v2_app_init_shim ./install_and_verify.sh"
            ),
            "execution_authority": False,
            "physical_authority": False,
        },
        "camera_opened": False,
        "device_enumerated": False,
        "service_started": False,
        "bpu_model_loaded": False,
        "bpu_forward_executed": False,
        "execution_authority": False,
        "physical_authority": False,
    }
    atomic_json(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
