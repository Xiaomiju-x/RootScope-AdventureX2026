#!/usr/bin/env python3
"""Independently audit the raw RootScope X5 field-bundle v2 tar."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


BUNDLE_ID = "rootscope_x5_field_bundle_v2"
BUNDLE_NAME = f"{BUNDLE_ID}.tar"
CORE_NAME = "rootscope_x5_offline_core_v1.tar"
CORE_SHA = "19f29d5be629bfdc8f66a77119c1391c5c609ff2a83f4a3f6059c65ed768391f"
CORE_BYTES = 122_664_960
MODEL_NAME = "rootscope_x5_readonly_llm_model_v1.tar"
MODEL_SHA = "2c39dd6a8bebbb62e7f27b7a1ddf2ed93356f696c2dc1bee1bba70f9c0098652"
MODEL_BYTES = 397_813_760
LLAMA_NAME = "rootscope_llama_server_arm64_b9637_v1.tar"
LLAMA_SHA = "48f2048a9e207ff4215c8867447a8546ac9f438705731b20f6d2905440a167c2"
LLAMA_BYTES = 167_772_160
PRINT_NAME = "RootScope_demo_reference_candidate_cards_A4.pdf"
PRINT_SHA = "dfd4b2e9524f2a37fbe39b9f1911b441c0b44565da93e4cfd321c2afe248070a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise ValueError(f"unsafe tar member: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError(f"unsafe tar member: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe tar member: {value!r}")
    if path.parts[0] != BUNDLE_ID or len(path.parts) < 2:
        raise ValueError(f"outer member escapes {BUNDLE_ID}: {value}")
    return path


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def result(self, archive: Path) -> Mapping[str, Any]:
        failed = [item for item in self.checks if not item["passed"]]
        return {
            "schema": "rootscope.x5-field-bundle-independent-audit.v2",
            "status": "PASS" if not failed else "FAIL",
            "passed": not failed,
            "checks_passed": len(self.checks) - len(failed),
            "checks_failed": len(failed),
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive),
            "checks": self.checks,
            "authority": {
                "hardware_touched": False,
                "network_touched": False,
                "device_enumerated": False,
                "service_started": False,
                "systemctl_invoked": False,
                "serial_write": False,
                "pump_command": False,
                "execution_authority": False,
                "physical_authority": False,
            },
        }


def _load_installer(path: Path):
    spec = importlib.util.spec_from_file_location("rootscope_field_installer_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load field installer for cross-check")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _extract_outer(archive_path: Path, destination_parent: Path, audit: Audit) -> Path:
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        names: set[str] = set()
        metadata_ok = True
        for member in members:
            path = _safe_name(member.name)
            if member.name in names:
                raise ValueError(f"duplicate outer member: {member.name}")
            names.add(member.name)
            if not member.isfile():
                raise ValueError(f"outer tar links/directories/devices are forbidden: {member.name}")
            metadata_ok = metadata_ok and (
                member.mtime == 0
                and member.uid == 0
                and member.gid == 0
                and member.uname == ""
                and member.gname == ""
                and not member.pax_headers
            )
            target = destination_parent / Path(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"outer member is unreadable: {member.name}")
            with target.open("wb") as output:
                shutil.copyfileobj(handle, output, 1024 * 1024)
            os.chmod(target, 0o755 if member.mode & 0o111 else 0o644)
    audit.check("outer_members_present", bool(names), len(names))
    audit.check("outer_members_unique", len(names) == len(members), len(names))
    audit.check("outer_regular_files_only", all(member.isfile() for member in members), len(members))
    audit.check("outer_deterministic_metadata", metadata_ok, "mtime/uid/gid/names/pax")
    return destination_parent / BUNDLE_ID


def audit_bundle(archive_path: Path, adventurex_root: Path) -> Mapping[str, Any]:
    archive = archive_path.resolve(strict=True)
    adventurex = adventurex_root.resolve(strict=True)
    audit = Audit()
    sha_path = archive.with_name(archive.name + ".sha256")
    digest = sha256_file(archive)
    audit.check(
        "outer_sha_sidecar",
        sha_path.is_file() and sha_path.read_text(encoding="ascii") == f"{digest}  {archive.name}\n",
        str(sha_path),
    )
    temp_parent = adventurex / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="field_bundle_v2_audit_", dir=temp_parent) as temporary:
        extracted = _extract_outer(archive, Path(temporary), audit)
        packaged_installer = extracted / "install_field_bundle_v2.py"
        workspace_installer = adventurex / "rootscope/deploy/x5/scripts/install_field_bundle_v2.py"
        audit.check(
            "packaged_installer_matches_workspace",
            packaged_installer.read_bytes() == workspace_installer.read_bytes(),
            sha256_file(packaged_installer),
        )
        installer = _load_installer(packaged_installer)
        verification = installer.verify_bundle(extracted)
        audit.check(
            "installer_independent_verify_pass",
            verification.get("status") == "PASS_HASHES_AND_SAFE_PATHS_NOT_X5_QUALIFIED",
            verification.get("status"),
        )
        for name in (
            "hardware_touched",
            "network_touched",
            "device_enumerated",
            "service_started",
            "systemctl_invoked",
            "activation_gate_created",
            "bpu_model_loaded",
            "bpu_forward_executed",
            "camera_opened",
            "x5_validated",
            "model_qualified",
            "execution_authority",
            "physical_authority",
        ):
            audit.check(f"verify_{name}_false", verification.get(name) is False, verification.get(name))
        expected = {
            f"components/{CORE_NAME}": (CORE_SHA, CORE_BYTES),
            f"components/{MODEL_NAME}": (MODEL_SHA, MODEL_BYTES),
            f"components/{LLAMA_NAME}": (LLAMA_SHA, LLAMA_BYTES),
            PRINT_NAME: (PRINT_SHA, None),
        }
        for relative, (expected_sha, expected_size) in expected.items():
            path = extracted / relative
            audit.check(f"fixed_{Path(relative).name}_sha", sha256_file(path) == expected_sha, sha256_file(path))
            if expected_size is not None:
                audit.check(f"fixed_{Path(relative).name}_bytes", path.stat().st_size == expected_size, path.stat().st_size)
        bpu = extracted / "components/rootscope_seed17_bpu_support_v1.tar"
        with tarfile.open(bpu, mode="r:") as nested:
            files = [member.name for member in nested.getmembers() if member.isfile()]
        audit.check("bpu_no_bin", not any(name.lower().endswith(".bin") for name in files), [name for name in files if name.lower().endswith(".bin")])
        wheels = [name for name in files if name.lower().endswith(".whl")]
        audit.check("bpu_exactly_one_pillow_wheel", len(wheels) == 1 and PurePosixPath(wheels[0]).name.lower().startswith("pillow-"), wheels)
        audit.check("bpu_no_numpy_or_hobot_wheel", not any("numpy" in name.lower() or "hobot_dnn" in name.lower() for name in wheels), wheels)
        manifest = json.loads((extracted / "bundle_manifest.json").read_text(encoding="utf-8"))
        audit.check(
            "target_contract_exact",
            manifest.get("target")
            == {
                "system": "Linux",
                "architecture": "aarch64",
                "python": "CPython 3.10",
                "install_python": "RDK_SYSTEM_PYTHON_OUTSIDE_VENV",
                "offline_only": True,
            },
            manifest.get("target"),
        )
        workflow = manifest.get("default_workflow", {})
        expected_true = (
            "strict_verify_before_extract",
            "safe_nested_tar_paths",
            "core_v1_cpu_simulated_selftest",
            "llm_staged_disabled_manual_ack",
            "bpu_separate_system_site_packages_venv",
        )
        expected_false = (
            "llm_service_start",
            "activation_gate_created",
            "bpu_core_v1_venv_allowed",
            "bpu_model_load",
            "bpu_forward",
            "camera_open",
        )
        audit.check("default_workflow_true_boundaries", all(workflow.get(name) is True for name in expected_true), workflow)
        audit.check("default_workflow_false_boundaries", all(workflow.get(name) is False for name in expected_false), workflow)
        audit.check("default_bpu_wheel_allowlist", workflow.get("bpu_local_wheel_allowlist") == ["Pillow"], workflow.get("bpu_local_wheel_allowlist"))
        bpu_record = manifest["components"]["bpu_support_v1"]
        audit.check("manifest_bpu_binary_false", bpu_record.get("bpu_binary_included") is False, bpu_record)
        audit.check("manifest_selected_bin_null", bpu_record.get("selected_bin") is None, bpu_record.get("selected_bin"))

    receipt_path = archive.parent / "release_build_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    audit.check("build_receipt_schema", receipt.get("schema") == "rootscope.x5-field-bundle-build-receipt.v2", receipt.get("schema"))
    audit.check("build_receipt_bundle_sha", receipt.get("bundle", {}).get("sha256") == digest, receipt.get("bundle", {}).get("sha256"))
    audit.check("build_receipt_bundle_bytes", receipt.get("bundle", {}).get("bytes") == archive.stat().st_size, receipt.get("bundle", {}).get("bytes"))
    audit.check("v1_immutable_claim", receipt.get("immutable_v1_unchanged") is True and receipt.get("immutable_v1_before") == receipt.get("immutable_v1_after"), receipt.get("immutable_v1_unchanged"))
    for name, expected_sha, expected_size in (
        (CORE_NAME, CORE_SHA, CORE_BYTES),
        (MODEL_NAME, MODEL_SHA, MODEL_BYTES),
    ):
        path = adventurex / "output/releases/rootscope_x5_offline_v1" / name
        audit.check(f"source_v1_{name}_sha", sha256_file(path) == expected_sha, sha256_file(path))
        audit.check(f"source_v1_{name}_bytes", path.stat().st_size == expected_size, path.stat().st_size)
    source = (adventurex / "rootscope/deploy/x5/scripts/install_field_bundle_v2.py").read_text(encoding="utf-8")
    for token in ("import requests", "import urllib", "import socket", "import serial", "import cv2", "import hobot_dnn"):
        audit.check(f"installer_no_{token.replace(' ', '_')}", token not in source, token)
    return audit.result(archive)


def _parser() -> argparse.ArgumentParser:
    adventurex = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adventurex-root", type=Path, default=adventurex)
    parser.add_argument(
        "--archive",
        type=Path,
        default=adventurex / "output/releases/rootscope_x5_field_bundle_v2" / BUNDLE_NAME,
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_bundle(args.archive, args.adventurex_root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json is not None:
        output = args.output_json.resolve()
        adventurex = args.adventurex_root.resolve(strict=True)
        if adventurex not in output.parents:
            raise ValueError("audit output must stay below AdventureX")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
