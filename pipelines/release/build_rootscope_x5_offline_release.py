#!/usr/bin/env python3
"""Build deterministic RootScope core and optional read-only LLM tar packages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Iterable, Mapping, Sequence


RELEASE_ID = "rootscope_x5_offline_core_v1"
CORE_ROOT_NAME = RELEASE_ID
CORE_ARCHIVE_NAME = f"{RELEASE_ID}.tar"
LLM_PACKAGE_ID = "rootscope_x5_readonly_llm_model_v1"
LLM_ARCHIVE_NAME = f"{LLM_PACKAGE_ID}.tar"
OUTPUT_FOLDER = "rootscope_x5_offline_v1"
BUILD_DATE = "2026-07-17"
CORE_STATUS = "HASH_LOCKED_CROSS_BUILT_NOT_EXACT_TWIN_X5_QUALIFIED"
MODEL_SHA256 = "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad"
REGISTRY_SHA256 = "f5328b66c6b385dd081c1a27577d92b5c63fc18dc00d815f7d3d2561bb68e29f"
REGISTRY_RECEIPT_SHA256 = "fc6a4ba195625fb01f4f7301edd5c6225b97342788a7b0ee7b2b22252b77ea32"
DUAL_PATH_FIXTURE_AUDIT_SHA256 = "3992b7e27438261ef5c817ddc49abd5d54e445a0cfd6f193b27f38cce7db7b2d"
PRINT_CARD_PDF_SHA256 = "dfd4b2e9524f2a37fbe39b9f1911b441c0b44565da93e4cfd321c2afe248070a"


@dataclass(frozen=True)
class SourceEntry:
    source: Path
    package_path: str
    mode: int
    category: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _portable_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe package path: {value!r}")
    return value


def _entry_mode(path: Path) -> int:
    return 0o755 if path.suffix == ".sh" else 0o644


def _iter_regular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"source symlink is forbidden: {path}")
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        yield path


def collect_core_sources(adventurex_root: Path) -> list[SourceEntry]:
    """Return the explicit, dataset-free core payload allowlist."""

    adventurex = adventurex_root.resolve(strict=True)
    rootscope = adventurex / "rootscope"
    if not rootscope.is_dir():
        raise ValueError("AdventureX rootscope directory is missing")
    entries: dict[str, SourceEntry] = {}

    def add_file(source: Path, package_relative: str, category: str) -> None:
        source = source.resolve(strict=True)
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"source must be a regular file: {source}")
        package_path = _portable_path(f"rootscope/{package_relative}")
        if package_path in entries:
            raise ValueError(f"duplicate package path: {package_path}")
        entries[package_path] = SourceEntry(
            source=source,
            package_path=package_path,
            mode=_entry_mode(source),
            category=category,
        )

    def add_tree(relative: str, category: str) -> None:
        base = rootscope / relative
        if not base.is_dir():
            raise ValueError(f"required source tree is missing: {relative}")
        for source in _iter_regular_files(base):
            relative_path = source.relative_to(rootscope).as_posix()
            if relative_path == "deploy/x5/systemd/rootscope-edge.service":
                # The legacy system unit hard-codes an account.  The release is
                # deliberately current-user/manual and never packages it.
                continue
            add_file(source, relative_path, category)

    add_file(rootscope / "app/__init__.py", "app/__init__.py", "python_core")
    for tree, category in (
        ("app/edge", "cpu_capsule"),
        ("app/vision", "vision_dual_path"),
        ("app/llm", "readonly_llm_client"),
        ("app/web", "locked_dashboard"),
        ("deploy/x5", "offline_deployment"),
    ):
        add_tree(tree, category)
    for relative in (
        "configs/class_contract.json",
        "configs/class_contract.lock.json",
        "pyproject.toml",
    ):
        add_file(rootscope / relative, relative, "contract")

    registry_receipt = adventurex / "evidence/rootscope_demo_template_registry_receipt_20260717.json"
    add_file(
        registry_receipt,
        "evidence/rootscope_demo_template_registry_receipt_20260717.json",
        "demo_template_provenance",
    )
    dual_path_audit = adventurex / "evidence/rootscope_dual_path_pc_fixture_audit_20260717.json"
    add_file(
        dual_path_audit,
        "evidence/rootscope_dual_path_pc_fixture_audit_20260717.json",
        "demo_fixture_evidence",
    )

    forbidden_tokens = ("datasets/", "training/", "output/rootscope_machine_curated")
    for package_path, entry in entries.items():
        lowered = package_path.lower()
        if any(token in lowered for token in forbidden_tokens):
            raise ValueError(f"dataset/training payload is forbidden: {package_path}")
        if entry.source.suffix.lower() in {".bin", ".pt", ".pth"}:
            raise ValueError(f"BPU/training binary is forbidden: {package_path}")

    model = entries.get(
        "rootscope/deploy/x5/models/rootscope_seed17_cpu_experimental_opset11.onnx"
    )
    if model is None or sha256_file(model.source) != MODEL_SHA256:
        raise ValueError("frozen seed17 CPU ONNX is missing or changed")
    registry = entries.get(
        "rootscope/app/vision/known_card_template_registry.frozen.experimental.json"
    )
    if registry is None or sha256_file(registry.source) != REGISTRY_SHA256:
        raise ValueError("frozen experimental template registry is missing or changed")
    receipt = entries.get(
        "rootscope/evidence/rootscope_demo_template_registry_receipt_20260717.json"
    )
    if receipt is None or sha256_file(receipt.source) != REGISTRY_RECEIPT_SHA256:
        raise ValueError("template registry receipt is missing or changed")
    dual_path_receipt = entries.get(
        "rootscope/evidence/rootscope_dual_path_pc_fixture_audit_20260717.json"
    )
    if (
        dual_path_receipt is None
        or sha256_file(dual_path_receipt.source) != DUAL_PATH_FIXTURE_AUDIT_SHA256
    ):
        raise ValueError("dual-path fixture audit is missing or changed")
    print_card_pdf = adventurex / "output/pdf/RootScope_demo_reference_candidate_cards_A4.pdf"
    if not print_card_pdf.is_file() or sha256_file(print_card_pdf) != PRINT_CARD_PDF_SHA256:
        raise ValueError("external print-card PDF is missing or changed")
    return [entries[name] for name in sorted(entries)]


def _install_wrapper() -> bytes:
    return b"""#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SYSTEM_PYTHON="${ROOTSCOPE_SYSTEM_PYTHON:-python3}"
exec "$SYSTEM_PYTHON" \
  "$PACKAGE_ROOT/rootscope/deploy/x5/scripts/install_offline_core.py" \
  --package-root "$PACKAGE_ROOT" "$@"
"""


def _tar_add_bytes(archive: tarfile.TarFile, name: str, payload: bytes, mode: int) -> None:
    info = tarfile.TarInfo(_portable_path(name))
    info.size = len(payload)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.type = tarfile.REGTYPE
    archive.addfile(info, BytesIO(payload))


def _tar_add_file(
    archive: tarfile.TarFile, name: str, source: Path, mode: int
) -> None:
    info = tarfile.TarInfo(_portable_path(name))
    info.size = source.stat().st_size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.type = tarfile.REGTYPE
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def build_deterministic_tar(
    destination: Path,
    *,
    file_entries: Sequence[tuple[str, Path, int]],
    byte_entries: Sequence[tuple[str, bytes, int]],
) -> dict[str, Any]:
    """Create a byte-repeatable, uncompressed USTAR archive."""

    names = [name for name, _source, _mode in file_entries] + [
        name for name, _payload, _mode in byte_entries
    ]
    if len(names) != len(set(names)):
        raise ValueError("tar member names must be unique")
    for name in names:
        _portable_path(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    file_map = {name: (source, mode) for name, source, mode in file_entries}
    byte_map = {name: (payload, mode) for name, payload, mode in byte_entries}
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name in sorted(names):
                if name in file_map:
                    source, mode = file_map[name]
                    _tar_add_file(archive, name, source, mode)
                else:
                    payload, mode = byte_map[name]
                    _tar_add_bytes(archive, name, payload, mode)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "filename": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "compression": "none",
        "tar_format": "USTAR",
    }


def _validate_llm_release(llm_release: Path) -> dict[str, Any]:
    release = llm_release.resolve(strict=True)
    if not release.is_dir():
        raise ValueError("LLM release must be a directory")
    expected_names = {"release_manifest.json", "SHA256SUMS"}
    manifest_path = release / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("LLM release manifest must be an object")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("filename"), str):
        raise ValueError("LLM release artifact contract missing")
    expected_names.add(artifact["filename"])
    actual_names = {path.name for path in release.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ValueError("LLM release has missing or extra files")
    if any(path.is_symlink() for path in release.iterdir()):
        raise ValueError("LLM release symlinks are forbidden")
    formal = manifest.get("formal_flags")
    if not isinstance(formal, dict) or not formal or any(value is not False for value in formal.values()):
        raise ValueError("LLM formal flags must all remain false")
    model = release / artifact["filename"]
    if model.stat().st_size != artifact.get("size_bytes") or sha256_file(model) != artifact.get("sha256"):
        raise ValueError("LLM GGUF size/hash mismatch")
    sums_lines = (release / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    expected_lines = sorted(
        [
            f"{sha256_file(model)}  {model.name}",
            f"{sha256_file(manifest_path)}  release_manifest.json",
        ]
    )
    if sorted(sums_lines) != expected_lines:
        raise ValueError("LLM SHA256SUMS mismatch")
    return {
        "release": release,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_filename": model.name,
        "artifact_sha256": sha256_file(model),
        "artifact_bytes": model.stat().st_size,
    }


def build_llm_archive(llm_release: Path, output_dir: Path) -> dict[str, Any]:
    validated = _validate_llm_release(llm_release)
    release: Path = validated["release"]
    destination = output_dir / LLM_ARCHIVE_NAME
    archive_record = build_deterministic_tar(
        destination,
        file_entries=[
            (f"{LLM_PACKAGE_ID}/{path.name}", path, 0o644)
            for path in sorted(release.iterdir(), key=lambda item: item.name)
            if path.is_file()
        ],
        byte_entries=[],
    )
    archive_record.update(
        {
            "package_id": LLM_PACKAGE_ID,
            "source_release_manifest_sha256": validated["manifest_sha256"],
            "model_filename": validated["artifact_filename"],
            "model_sha256": validated["artifact_sha256"],
            "model_bytes": validated["artifact_bytes"],
            "llama_server_bundled": False,
            "llama_server_qualified": False,
            "x5_validated": False,
            "model_qualified": False,
            "execution_authority": False,
        }
    )
    (output_dir / f"{LLM_ARCHIVE_NAME}.sha256").write_text(
        f"{archive_record['sha256']}  {LLM_ARCHIVE_NAME}\n", encoding="ascii"
    )
    return archive_record


def build_core_archive(
    adventurex_root: Path,
    output_dir: Path,
    llm_archive: Mapping[str, Any] | None,
) -> dict[str, Any]:
    entries = collect_core_sources(adventurex_root)
    wrapper = _install_wrapper()
    file_records: list[dict[str, Any]] = []
    for entry in entries:
        file_records.append(
            {
                "path": entry.package_path,
                "bytes": entry.source.stat().st_size,
                "sha256": sha256_file(entry.source),
                "mode": format(entry.mode, "04o"),
                "category": entry.category,
            }
        )
    file_records.append(
        {
            "path": "install_and_selftest.sh",
            "bytes": len(wrapper),
            "sha256": hashlib.sha256(wrapper).hexdigest(),
            "mode": "0755",
            "category": "entrypoint",
        }
    )
    file_records.sort(key=lambda item: item["path"])
    category_counts: dict[str, int] = {}
    for record in file_records:
        category_counts[record["category"]] = category_counts.get(record["category"], 0) + 1

    llm_reference: dict[str, Any]
    if llm_archive is None:
        llm_reference = {
            "separate_archive_present": False,
            "llama_server_bundled": False,
            "llama_server_qualified": False,
            "x5_validated": False,
        }
    else:
        llm_reference = {
            "separate_archive_present": True,
            "archive_filename": llm_archive["filename"],
            "archive_sha256": llm_archive["sha256"],
            "archive_bytes": llm_archive["bytes"],
            "source_release_manifest_sha256": llm_archive[
                "source_release_manifest_sha256"
            ],
            "model_filename": llm_archive["model_filename"],
            "model_sha256": llm_archive["model_sha256"],
            "model_bytes": llm_archive["model_bytes"],
            "llama_server_bundled": False,
            "llama_server_qualified": False,
            "x5_validated": False,
        }
    manifest = {
        "schema": "rootscope.x5-offline-core-release.v1",
        "release_id": RELEASE_ID,
        "build_date": BUILD_DATE,
        "status": CORE_STATUS,
        "target_contract": {
            "system": "Linux",
            "architecture": "aarch64",
            "python_implementation": "CPython",
            "python_version": "3.10",
            "install_scope": "CURRENT_USER_MANUAL_NO_SUDO_NO_SYSTEM_SERVICE",
        },
        "contents": {
            "files": len(file_records),
            "bytes": sum(record["bytes"] for record in file_records),
            "category_counts": category_counts,
            "dataset_included": False,
            "training_outputs_included": False,
            "cpu_onnx_sha256": MODEL_SHA256,
            "frozen_demo_registry_sha256": REGISTRY_SHA256,
            "frozen_demo_registry_templates": 3,
            "unknown_negative_registered": False,
            "pc_fixture_dual_path_audit_sha256": DUAL_PATH_FIXTURE_AUDIT_SHA256,
            "pc_fixture_dual_path_checks": "40/40_PASS_NOT_X5_EVIDENCE",
            "print_card_pdf_included": False,
            "external_print_card_pdf_sha256": PRINT_CARD_PDF_SHA256,
            "bpu_binary_included": False,
        },
        "llm": llm_reference,
        "formal_flags": {
            "human_reviewed": False,
            "rights_approved": False,
            "data_locked": False,
            "wheelhouse_qualified": False,
            "model_candidate": False,
            "model_qualified": False,
            "camera_qualified": False,
            "llama_server_qualified": False,
            "bpu_compiled": False,
            "bpu_ready": False,
            "x5_ready": False,
            "x5_validated": False,
            "physical_completion": False,
        },
        "authority": {
            "hardware_touched": False,
            "network_touched": False,
            "service_started": False,
            "systemctl_invoked": False,
            "pump_command": False,
            "serial_write": False,
            "state_machine_write": False,
            "execution_authority": False,
            "physical_authority": False,
        },
        "files": file_records,
    }
    manifest_bytes = canonical_json(manifest)
    sums = [f"{record['sha256']}  {record['path']}" for record in file_records]
    sums.append(
        f"{hashlib.sha256(manifest_bytes).hexdigest()}  release_manifest.json"
    )
    sums_bytes = ("\n".join(sorted(sums)) + "\n").encode("utf-8")
    destination = output_dir / CORE_ARCHIVE_NAME
    archive_record = build_deterministic_tar(
        destination,
        file_entries=[
            (f"{CORE_ROOT_NAME}/{entry.package_path}", entry.source, entry.mode)
            for entry in entries
        ],
        byte_entries=[
            (f"{CORE_ROOT_NAME}/install_and_selftest.sh", wrapper, 0o755),
            (f"{CORE_ROOT_NAME}/release_manifest.json", manifest_bytes, 0o644),
            (f"{CORE_ROOT_NAME}/SHA256SUMS", sums_bytes, 0o644),
        ],
    )
    archive_record.update(
        {
            "release_id": RELEASE_ID,
            "internal_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "internal_files": len(file_records),
            "bpu_binary_present": False,
            "bpu_compiled": False,
            "bpu_ready": False,
            "x5_validated": False,
            "model_qualified": False,
            "execution_authority": False,
        }
    )
    (output_dir / f"{CORE_ARCHIVE_NAME}.sha256").write_text(
        f"{archive_record['sha256']}  {CORE_ARCHIVE_NAME}\n", encoding="ascii"
    )
    return archive_record


def build_release(
    adventurex_root: Path,
    output_dir: Path,
    llm_release: Path | None,
) -> dict[str, Any]:
    adventurex = adventurex_root.resolve(strict=True)
    output = output_dir.resolve()
    if adventurex not in output.parents:
        raise ValueError("release output must stay below AdventureX")
    output.mkdir(parents=True, exist_ok=True)
    llm_record = build_llm_archive(llm_release, output) if llm_release is not None else None
    core_record = build_core_archive(adventurex, output, llm_record)
    receipt = {
        "schema": "rootscope.x5-offline-release-build-receipt.v1",
        "build_date": BUILD_DATE,
        "status": "PASS_DETERMINISTIC_PACKAGING_NOT_X5_QUALIFIED",
        "output_relative_to_adventurex": output.relative_to(adventurex).as_posix(),
        "core": core_record,
        "readonly_llm_model": llm_record,
        "formal_flags": {
            "human_reviewed": False,
            "rights_approved": False,
            "data_locked": False,
            "model_candidate": False,
            "model_qualified": False,
            "llama_server_qualified": False,
            "bpu_compiled": False,
            "bpu_ready": False,
            "x5_ready": False,
            "x5_validated": False,
        },
        "authority": {
            "hardware_touched": False,
            "network_touched": False,
            "service_started": False,
            "execution_authority": False,
            "physical_authority": False,
        },
    }
    receipt_path = output / "release_build_receipt.json"
    receipt_path.write_bytes(canonical_json(receipt))
    return receipt


def _parser() -> argparse.ArgumentParser:
    script = Path(__file__).resolve()
    adventurex = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adventurex-root", type=Path, default=adventurex)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=adventurex / "output/releases" / OUTPUT_FOLDER,
    )
    parser.add_argument(
        "--llm-release",
        type=Path,
        default=adventurex / "output/rootscope_llm_readonly_release_v1",
    )
    parser.add_argument("--without-llm", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = build_release(
            args.adventurex_root,
            args.output_dir,
            None if args.without_llm else args.llm_release,
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        error = {
            "schema": "rootscope.x5-offline-release-build-error.v1",
            "status": "FAIL_CLOSED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "hardware_touched": False,
            "network_touched": False,
            "x5_validated": False,
            "model_qualified": False,
            "bpu_ready": False,
            "execution_authority": False,
        }
        print(json.dumps(error, ensure_ascii=False, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
