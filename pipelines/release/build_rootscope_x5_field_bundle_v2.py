#!/usr/bin/env python3
"""Build the deterministic, compositional RootScope X5 field bundle v2."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ADVENTUREX_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(ADVENTUREX_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(ADVENTUREX_IMPORT_ROOT))

from tools.release.build_rootscope_x5_offline_release import (  # noqa: E402
    build_deterministic_tar,
    canonical_json,
    sha256_file,
)


BUILD_DATE = "2026-07-17"
BUNDLE_ID = "rootscope_x5_field_bundle_v2"
BUNDLE_ARCHIVE = f"{BUNDLE_ID}.tar"
BUNDLE_ROOT = BUNDLE_ID
BUNDLE_SCHEMA = "rootscope.x5-field-bundle.v2"
BUNDLE_STATUS = "HASH_LOCKED_COMPOSITION_NOT_X5_QUALIFIED"
BPU_ID = "rootscope_seed17_bpu_support_v1"
BPU_ARCHIVE = f"{BPU_ID}.tar"
CORE_ARCHIVE = "rootscope_x5_offline_core_v1.tar"
CORE_SHA256 = "19f29d5be629bfdc8f66a77119c1391c5c609ff2a83f4a3f6059c65ed768391f"
CORE_BYTES = 122_664_960
LLM_MODEL_ARCHIVE = "rootscope_x5_readonly_llm_model_v1.tar"
LLM_MODEL_SHA256 = "2c39dd6a8bebbb62e7f27b7a1ddf2ed93356f696c2dc1bee1bba70f9c0098652"
LLM_MODEL_BYTES = 397_813_760
LLAMA_ARCHIVE = "rootscope_llama_server_arm64_b9637_v1.tar"
LLAMA_SHA256 = "48f2048a9e207ff4215c8867447a8546ac9f438705731b20f6d2905440a167c2"
LLAMA_BYTES = 167_772_160
PRINT_PDF = "RootScope_demo_reference_candidate_cards_A4.pdf"
PRINT_SHA256 = "dfd4b2e9524f2a37fbe39b9f1911b441c0b44565da93e4cfd321c2afe248070a"
PILLOW_FILENAME = "pillow-11.3.0-cp310-cp310-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"
PILLOW_SHA256 = "7107195ddc914f656c7fc8e4a5e1c25f32e9236ea3ea860f257b0436011fddd0"


def _strict_false(value: Any, context: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{context} must be a non-empty object")
    bad = sorted(name for name, flag in value.items() if flag is not False)
    if bad:
        raise ValueError(f"{context} must remain false: {bad}")


def _inside(path: Path, root: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"source symlink is forbidden: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"source must stay below AdventureX: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"source must be one regular file: {path}")
    return resolved


def _assert_file(path: Path, digest: str, size: int | None = None) -> Path:
    if path.is_symlink():
        raise ValueError(f"required artifact symlink is forbidden: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"required artifact is unsafe: {path}")
    if size is not None and resolved.stat().st_size != size:
        raise ValueError(f"required artifact size changed: {path.name}")
    actual = sha256_file(resolved)
    if actual != digest:
        raise ValueError(f"required artifact hash changed: {path.name} actual={actual}")
    return resolved


def validate_selection_receipt(path: Path, adventurex: Path) -> Mapping[str, Any]:
    receipt_path = _inside(path, adventurex)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        raise ValueError("BPU selection receipt must be an object")
    if receipt.get("schema") != "rootscope.seed17-bpu-field-selection.v1":
        raise ValueError("unsupported BPU field selection receipt")
    _strict_false(receipt.get("claims"), "BPU selection claims")
    _strict_false(receipt.get("authority"), "BPU selection authority")
    selection = receipt.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("BPU selection object is missing")
    selected = selection.get("selected_bin")
    source: Path | None = None
    if selected is None:
        if selection.get("selected_variant") is not None:
            raise ValueError("null selected_bin requires null selected_variant")
        if selection.get("publishable_default_bpu_bin") is not None:
            raise ValueError("null selected_bin cannot publish a default bin")
    else:
        if not isinstance(selected, Mapping):
            raise ValueError("selected_bin must be null or an object")
        if selection.get("all_predeclared_replay_gates_passed") is not True:
            raise ValueError("a selected BPU bin requires all frozen gates")
        source_value = selected.get("source_path")
        if not isinstance(source_value, str):
            raise ValueError("selected BPU source_path is missing")
        source = _inside(adventurex / Path(*PurePosixPath(source_value).parts), adventurex)
        if source.suffix.lower() != ".bin":
            raise ValueError("selected BPU artifact must be one .bin")
        if selected.get("sha256") != sha256_file(source) or selected.get("bytes") != source.stat().st_size:
            raise ValueError("selected BPU artifact hash/size mismatch")
    for item in receipt.get("source_results", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ValueError("selection source result record is invalid")
        source_result = _inside(
            adventurex / Path(*PurePosixPath(str(item["path"])).parts), adventurex
        )
        if sha256_file(source_result) != item.get("sha256"):
            raise ValueError(f"selection source result changed: {item['path']}")
    return {
        "path": receipt_path,
        "payload": receipt,
        "sha256": sha256_file(receipt_path),
        "selected_source": source,
    }


def _entry(path: str, source: Path, category: str, mode: int = 0o644) -> dict[str, Any]:
    return {
        "path": path,
        "source": source,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "mode": mode,
        "category": category,
    }


def build_bpu_support(
    adventurex: Path, output_dir: Path, selection_path: Path
) -> Mapping[str, Any]:
    validated = validate_selection_receipt(selection_path, adventurex)
    rootscope = adventurex / "rootscope"
    sources = [
        _entry("rootscope/app/__init__.py", rootscope / "app/__init__.py", "python"),
        _entry("rootscope/app/edge/__init__.py", rootscope / "app/edge/__init__.py", "python"),
        _entry("rootscope/app/edge/bpu_seed17.py", rootscope / "app/edge/bpu_seed17.py", "bpu_adapter"),
        _entry(
            "rootscope/deploy/x5/scripts/bpu_seed17_isolated_readonly.py",
            rootscope / "deploy/x5/scripts/bpu_seed17_isolated_readonly.py",
            "manual_cli",
            0o755,
        ),
        _entry(
            "rootscope/deploy/x5/scripts/prepare_bpu_system_site_venv.py",
            rootscope / "deploy/x5/scripts/prepare_bpu_system_site_venv.py",
            "environment",
            0o755,
        ),
        _entry(
            "rootscope/deploy/x5/seed17_bpu_isolated_runtime_contract.json",
            rootscope / "deploy/x5/seed17_bpu_isolated_runtime_contract.json",
            "contract",
        ),
        _entry(
            "rootscope/deploy/x5/ROOTSCOPE_SEED17_BPU_ISOLATED_RUNBOOK_ZH.md",
            rootscope / "deploy/x5/ROOTSCOPE_SEED17_BPU_ISOLATED_RUNBOOK_ZH.md",
            "runbook",
        ),
        _entry(
            f"wheelhouse/{PILLOW_FILENAME}",
            _assert_file(
                rootscope / "deploy/x5/wheelhouse/candidate_cp310_aarch64" / PILLOW_FILENAME,
                PILLOW_SHA256,
            ),
            "pillow_only",
        ),
        _entry(
            "selection_receipt.json",
            validated["path"],
            "selection",
        ),
        _entry(
            "evidence/quant_variant_search_generation1_result.json",
            adventurex
            / "evidence/rootscope_seed17_bpu_compile_20260717/quant_variant_search_generation1_result.json",
            "compile_evidence",
        ),
        _entry(
            "evidence/quant_variant_search_generation2_result.json",
            adventurex
            / "evidence/rootscope_seed17_bpu_compile_20260717/quant_variant_search_generation2_result.json",
            "compile_evidence",
        ),
        _entry(
            "evidence/postbuild_independent_audit.json",
            adventurex
            / "evidence/rootscope_seed17_bpu_compile_20260717/postbuild_independent_audit.json",
            "postbuild_evidence",
        ),
        _entry(
            "evidence/rootscope_seed17_bpu_postbuild_receipt.json",
            adventurex
            / "evidence/rootscope_seed17_bpu_compile_20260717/rootscope_seed17_bpu_postbuild_receipt.json",
            "postbuild_evidence",
        ),
        _entry(
            "evidence/rootscope_seed17_bpu_isolated_runtime_support_pc_audit_20260717.json",
            adventurex
            / "evidence/rootscope_seed17_bpu_isolated_runtime_support_pc_audit_20260717.json",
            "runtime_support_evidence",
        ),
    ]
    selected_source = validated["selected_source"]
    selected_package: Mapping[str, Any] | None = None
    if selected_source is not None:
        selected_package_path = f"models/{selected_source.name}"
        sources.append(_entry(selected_package_path, selected_source, "selected_bpu_bin"))
        selected_package = {
            "package_path": f"{BPU_ID}/{selected_package_path}",
            "filename": selected_source.name,
            "bytes": selected_source.stat().st_size,
            "sha256": sha256_file(selected_source),
        }
    selection = validated["payload"]["selection"]
    file_records = [
        {key: value for key, value in record.items() if key != "source"}
        for record in sources
    ]
    manifest = {
        "schema": "rootscope.seed17-bpu-support-component.v1",
        "component_id": BPU_ID,
        "build_date": BUILD_DATE,
        "status": (
            "HASH_BOUND_SELECTED_BIN_NOT_X5_QUALIFIED"
            if selected_package is not None
            else "SUPPORT_ONLY_NO_SELECTED_BIN_NOT_X5_QUALIFIED"
        ),
        "selection": {
            "receipt_sha256": validated["sha256"],
            "selected_variant": selection.get("selected_variant"),
            "selected_bin": selected_package,
            "all_predeclared_replay_gates_passed": selection.get(
                "all_predeclared_replay_gates_passed"
            ),
        },
        "bpu_binary_included": selected_package is not None,
        "pillow_wheel": {
            "package_path": f"wheelhouse/{PILLOW_FILENAME}",
            "sha256": PILLOW_SHA256,
            "bytes": next(record["bytes"] for record in file_records if record["path"].endswith(PILLOW_FILENAME)),
            "pip_no_index": True,
            "pip_no_deps": True,
        },
        "dependency_policy": {
            "independent_system_site_packages_venv": True,
            "core_v1_venv_allowed": False,
            "system_numpy_required": True,
            "system_hobot_dnn_required": True,
            "numpy_wheel_included": False,
            "local_wheel_install_allowlist": ["Pillow"],
        },
        "formal_flags": {
            "x5_ready": False,
            "x5_validated": False,
            "camera_qualified": False,
            "model_candidate": False,
            "model_qualified": False,
            "production_integration_allowed": False,
        },
        "authority": {
            "hardware_touched": False,
            "network_touched": False,
            "device_enumerated": False,
            "serial_write": False,
            "state_machine_write": False,
            "pump_command": False,
            "irrigation_execution": False,
            "execution_authority": False,
            "physical_authority": False,
            "physical_completion": False,
        },
        "files": file_records,
    }
    manifest_bytes = canonical_json(manifest)
    sums = sorted(
        [f"{record['sha256']}  {record['path']}" for record in file_records]
        + [f"{hashlib.sha256(manifest_bytes).hexdigest()}  component_manifest.json"]
    )
    archive = build_deterministic_tar(
        output_dir / BPU_ARCHIVE,
        file_entries=[
            (f"{BPU_ID}/{record['path']}", record["source"], int(record["mode"]))
            for record in sources
        ],
        byte_entries=[
            (f"{BPU_ID}/component_manifest.json", manifest_bytes, 0o644),
            (f"{BPU_ID}/SHA256SUMS", ("\n".join(sums) + "\n").encode("utf-8"), 0o644),
        ],
    )
    archive.update(
        {
            "component_id": BPU_ID,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "selection_receipt_sha256": validated["sha256"],
            "bpu_binary_included": selected_package is not None,
            "selected_bin": selected_package,
            "x5_validated": False,
            "model_qualified": False,
            "execution_authority": False,
        }
    )
    (output_dir / f"{BPU_ARCHIVE}.sha256").write_text(
        f"{archive['sha256']}  {BPU_ARCHIVE}\n", encoding="ascii"
    )
    return archive


def _install_wrapper() -> bytes:
    return b'''#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
exec "${ROOTSCOPE_SYSTEM_PYTHON:-/usr/bin/python3}" \
  "$BUNDLE_ROOT/install_field_bundle_v2.py" --bundle-root "$BUNDLE_ROOT" "$@"
'''


def build_bundle(
    adventurex_root: Path,
    output_dir: Path,
    selection_receipt: Path,
) -> Mapping[str, Any]:
    adventurex = adventurex_root.resolve(strict=True)
    output = output_dir.resolve()
    if adventurex not in output.parents:
        raise ValueError("field bundle output must stay below AdventureX")
    if output.name == "rootscope_x5_offline_v1":
        raise ValueError("v2 output must not target the immutable v1 directory")
    output.mkdir(parents=True, exist_ok=True)

    v1 = adventurex / "output/releases/rootscope_x5_offline_v1"
    core = _assert_file(v1 / CORE_ARCHIVE, CORE_SHA256, CORE_BYTES)
    llm_model = _assert_file(v1 / LLM_MODEL_ARCHIVE, LLM_MODEL_SHA256, LLM_MODEL_BYTES)
    llama = _assert_file(adventurex / "output" / LLAMA_ARCHIVE, LLAMA_SHA256, LLAMA_BYTES)
    print_pdf = _assert_file(adventurex / "output/pdf" / PRINT_PDF, PRINT_SHA256)
    v1_before = {CORE_ARCHIVE: sha256_file(core), LLM_MODEL_ARCHIVE: sha256_file(llm_model)}
    bpu = build_bpu_support(adventurex, output, selection_receipt)

    installer = adventurex / "rootscope/deploy/x5/scripts/install_field_bundle_v2.py"
    runbook = adventurex / "rootscope/deploy/x5/ROOTSCOPE_X5_FIELD_BUNDLE_V2_RUNBOOK_ZH.md"
    wrapper = _install_wrapper()
    file_sources = [
        ("components/core_v1", CORE_ARCHIVE, core, 0o644),
        ("components/readonly_llm_model_v1", LLM_MODEL_ARCHIVE, llm_model, 0o644),
        ("components/llama_server_arm64_v1", LLAMA_ARCHIVE, llama, 0o644),
        ("components/bpu_support_v1", BPU_ARCHIVE, output / BPU_ARCHIVE, 0o644),
        ("print_reference", PRINT_PDF, print_pdf, 0o644),
        ("installer", "install_field_bundle_v2.py", installer, 0o755),
        ("runbook", "ROOTSCOPE_X5_FIELD_BUNDLE_V2_RUNBOOK_ZH.md", runbook, 0o644),
    ]
    records = []
    for category, filename, source, mode in file_sources:
        path = f"components/{filename}" if category.startswith("components/") else filename
        records.append(
            {
                "path": path,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "mode": mode,
                "category": category,
            }
        )
    records.append(
        {
            "path": "install_and_verify.sh",
            "bytes": len(wrapper),
            "sha256": hashlib.sha256(wrapper).hexdigest(),
            "mode": 0o755,
            "category": "entrypoint",
        }
    )
    records.sort(key=lambda record: record["path"])
    components = {
        "core_v1": {
            "filename": CORE_ARCHIVE,
            "bytes": CORE_BYTES,
            "sha256": CORE_SHA256,
            "immutable_existing_v1": True,
        },
        "readonly_llm_model_v1": {
            "filename": LLM_MODEL_ARCHIVE,
            "bytes": LLM_MODEL_BYTES,
            "sha256": LLM_MODEL_SHA256,
            "immutable_existing_v1": True,
        },
        "llama_server_arm64_v1": {
            "filename": LLAMA_ARCHIVE,
            "bytes": LLAMA_BYTES,
            "sha256": LLAMA_SHA256,
            "cross_built": True,
            "qemu_smoke_passed": True,
            "x5_validated": False,
            "llama_server_qualified": False,
        },
        "bpu_support_v1": {
            "filename": BPU_ARCHIVE,
            "bytes": bpu["bytes"],
            "sha256": bpu["sha256"],
            "selection_receipt_sha256": bpu["selection_receipt_sha256"],
            "bpu_binary_included": bpu["bpu_binary_included"],
            "selected_bin": bpu["selected_bin"],
            "x5_validated": False,
            "model_qualified": False,
        },
        "print_reference_pdf": {
            "filename": PRINT_PDF,
            "bytes": print_pdf.stat().st_size,
            "sha256": PRINT_SHA256,
            "machine_input": False,
        },
    }
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "bundle_id": BUNDLE_ID,
        "build_date": BUILD_DATE,
        "status": BUNDLE_STATUS,
        "target": {
            "system": "Linux",
            "architecture": "aarch64",
            "python": "CPython 3.10",
            "install_python": "RDK_SYSTEM_PYTHON_OUTSIDE_VENV",
            "offline_only": True,
        },
        "default_workflow": {
            "strict_verify_before_extract": True,
            "safe_nested_tar_paths": True,
            "core_v1_cpu_simulated_selftest": True,
            "llm_staged_disabled_manual_ack": True,
            "llm_service_start": False,
            "activation_gate_created": False,
            "bpu_separate_system_site_packages_venv": True,
            "bpu_core_v1_venv_allowed": False,
            "bpu_local_wheel_allowlist": ["Pillow"],
            "bpu_model_load": False,
            "bpu_forward": False,
            "camera_open": False,
        },
        "components": components,
        "formal_flags": {
            "x5_ready": False,
            "x5_validated": False,
            "camera_qualified": False,
            "model_candidate": False,
            "model_qualified": False,
            "llama_server_qualified": False,
            "production_integration_allowed": False,
        },
        "authority": {
            "hardware_touched": False,
            "network_touched": False,
            "device_enumerated": False,
            "service_started": False,
            "systemctl_invoked": False,
            "activation_gate_created": False,
            "serial_write": False,
            "state_machine_write": False,
            "pump_command": False,
            "irrigation_execution": False,
            "execution_authority": False,
            "physical_authority": False,
            "physical_completion": False,
        },
        "files": records,
    }
    manifest_bytes = canonical_json(manifest)
    sums = sorted(
        [f"{record['sha256']}  {record['path']}" for record in records]
        + [f"{hashlib.sha256(manifest_bytes).hexdigest()}  bundle_manifest.json"]
    )
    archive = build_deterministic_tar(
        output / BUNDLE_ARCHIVE,
        file_entries=[
            (
                f"{BUNDLE_ROOT}/{record['path']}",
                next(source for _category, filename, source, _mode in file_sources if (f"components/{filename}" if _category.startswith('components/') else filename) == record["path"]),
                int(record["mode"]),
            )
            for record in records
            if record["path"] != "install_and_verify.sh"
        ],
        byte_entries=[
            (f"{BUNDLE_ROOT}/install_and_verify.sh", wrapper, 0o755),
            (f"{BUNDLE_ROOT}/bundle_manifest.json", manifest_bytes, 0o644),
            (f"{BUNDLE_ROOT}/SHA256SUMS", ("\n".join(sums) + "\n").encode("utf-8"), 0o644),
        ],
    )
    archive.update(
        {
            "bundle_id": BUNDLE_ID,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "bpu_binary_included": bpu["bpu_binary_included"],
            "x5_validated": False,
            "model_qualified": False,
            "llama_server_qualified": False,
            "execution_authority": False,
        }
    )
    (output / f"{BUNDLE_ARCHIVE}.sha256").write_text(
        f"{archive['sha256']}  {BUNDLE_ARCHIVE}\n", encoding="ascii"
    )
    v1_after = {CORE_ARCHIVE: sha256_file(core), LLM_MODEL_ARCHIVE: sha256_file(llm_model)}
    if v1_after != v1_before:
        raise RuntimeError("immutable v1 archives changed during v2 build")
    receipt = {
        "schema": "rootscope.x5-field-bundle-build-receipt.v2",
        "status": "PASS_DETERMINISTIC_COMPOSITION_NOT_X5_QUALIFIED",
        "build_date": BUILD_DATE,
        "output_relative_to_adventurex": output.relative_to(adventurex).as_posix(),
        "bundle": archive,
        "bpu_support": bpu,
        "immutable_v1_before": v1_before,
        "immutable_v1_after": v1_after,
        "immutable_v1_unchanged": v1_before == v1_after,
        "authority": manifest["authority"],
    }
    (output / "release_build_receipt.json").write_bytes(canonical_json(receipt))
    return receipt


def _parser() -> argparse.ArgumentParser:
    adventurex = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adventurex-root", type=Path, default=adventurex)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=adventurex / "output/releases/rootscope_x5_field_bundle_v2",
    )
    parser.add_argument(
        "--selection-receipt",
        type=Path,
        default=adventurex / "evidence/rootscope_seed17_bpu_field_selection_20260717.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_bundle(args.adventurex_root, args.output_dir, args.selection_receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
