#!/usr/bin/env python3
"""Independently audit RootScope X5 core/LLM release archives without extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Mapping, Sequence


CORE_NAME = "rootscope_x5_offline_core_v1.tar"
CORE_ROOT = "rootscope_x5_offline_core_v1"
LLM_NAME = "rootscope_x5_readonly_llm_model_v1.tar"
LLM_ROOT = "rootscope_x5_readonly_llm_model_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(handle: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        total += len(chunk)
    return digest.hexdigest(), total


def _safe_member(name: str, expected_root: str) -> bool:
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and path.parts
        and path.parts[0] == expected_root
        and all(part not in {"", ".", ".."} for part in path.parts)
        and path.as_posix() == name
    )


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def result(self, release_dir: Path) -> dict[str, Any]:
        passed = all(item["passed"] for item in self.checks)
        return {
            "schema": "rootscope.x5-offline-release-independent-audit.v1",
            "status": "PASS_NOT_X5_QUALIFIED" if passed else "FAIL",
            "passed": passed,
            "checks_passed": sum(item["passed"] for item in self.checks),
            "checks_total": len(self.checks),
            "release_dir": str(release_dir),
            "hardware_touched": False,
            "network_touched": False,
            "service_started": False,
            "x5_validated": False,
            "model_qualified": False,
            "bpu_ready": False,
            "execution_authority": False,
            "physical_authority": False,
            "checks": self.checks,
        }


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"tar member is not readable: {name}")
    return handle.read()


def _audit_outer(audit: Audit, release: Path, receipt: Mapping[str, Any]) -> None:
    for key, filename in (("core", CORE_NAME), ("readonly_llm_model", LLM_NAME)):
        archive_path = release / filename
        record = receipt.get(key)
        audit.check(f"{key}_receipt_object", isinstance(record, Mapping), type(record).__name__)
        if not isinstance(record, Mapping):
            continue
        audit.check(f"{key}_archive_present", archive_path.is_file(), str(archive_path))
        if archive_path.is_file():
            digest = sha256_file(archive_path)
            audit.check(f"{key}_archive_sha", digest == record.get("sha256"), digest)
            audit.check(f"{key}_archive_bytes", archive_path.stat().st_size == record.get("bytes"), archive_path.stat().st_size)
            sha_path = release / f"{filename}.sha256"
            expected = f"{digest}  {filename}\n"
            audit.check(f"{key}_outer_sha_file", sha_path.is_file() and sha_path.read_text(encoding="ascii") == expected, str(sha_path))
        for name in ("x5_validated", "model_qualified", "execution_authority"):
            audit.check(f"{key}_{name}_false", record.get(name) is False, record.get(name))
    for context in ("formal_flags", "authority"):
        flags = receipt.get(context)
        audit.check(f"receipt_{context}_object", isinstance(flags, Mapping) and bool(flags), flags)
        if isinstance(flags, Mapping):
            for name, value in flags.items():
                audit.check(f"receipt_{context}_{name}_false", value is False, value)


def _audit_core(audit: Audit, path: Path, receipt: Mapping[str, Any]) -> None:
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        audit.check("core_members_unique", len(names) == len(set(names)), len(names))
        audit.check("core_members_safe", all(_safe_member(name, CORE_ROOT) for name in names), names[:3])
        audit.check("core_members_regular_only", all(member.isfile() for member in members), [member.name for member in members if not member.isfile()])
        manifest_name = f"{CORE_ROOT}/release_manifest.json"
        sums_name = f"{CORE_ROOT}/SHA256SUMS"
        audit.check("core_manifest_present", manifest_name in names, manifest_name)
        audit.check("core_sums_present", sums_name in names, sums_name)
        manifest_bytes = _read_member(archive, manifest_name)
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        audit.check("core_manifest_schema", manifest.get("schema") == "rootscope.x5-offline-core-release.v1", manifest.get("schema"))
        audit.check("core_manifest_status", manifest.get("status") == "HASH_LOCKED_CROSS_BUILT_NOT_EXACT_TWIN_X5_QUALIFIED", manifest.get("status"))
        for context in ("formal_flags", "authority"):
            flags = manifest.get(context)
            audit.check(f"core_{context}_object", isinstance(flags, Mapping) and bool(flags), flags)
            if isinstance(flags, Mapping):
                for name, value in flags.items():
                    audit.check(f"core_{context}_{name}_false", value is False, value)
        contents = manifest.get("contents", {})
        audit.check("core_dataset_absent", contents.get("dataset_included") is False, contents.get("dataset_included"))
        audit.check("core_training_absent", contents.get("training_outputs_included") is False, contents.get("training_outputs_included"))
        audit.check("core_bpu_binary_absent_claim", contents.get("bpu_binary_included") is False, contents.get("bpu_binary_included"))
        audit.check("core_three_templates", contents.get("frozen_demo_registry_templates") == 3, contents.get("frozen_demo_registry_templates"))
        audit.check("core_unknown_unregistered", contents.get("unknown_negative_registered") is False, contents.get("unknown_negative_registered"))

        records = manifest.get("files")
        audit.check("core_file_records_array", isinstance(records, list) and bool(records), len(records) if isinstance(records, list) else None)
        record_map = {record["path"]: record for record in records if isinstance(record, Mapping) and isinstance(record.get("path"), str)} if isinstance(records, list) else {}
        expected_names = {f"{CORE_ROOT}/{name}" for name in record_map} | {manifest_name, sums_name}
        audit.check("core_exact_member_coverage", set(names) == expected_names, {"actual": len(names), "expected": len(expected_names)})
        sums_lines = _read_member(archive, sums_name).decode("utf-8").splitlines()
        sums: dict[str, str] = {}
        for line in sums_lines:
            if len(line) >= 66 and line[64:66] == "  ":
                sums[line[66:]] = line[:64]
        expected_sums = {name: str(record["sha256"]) for name, record in record_map.items()}
        expected_sums["release_manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
        audit.check("core_sha256sums_exact", sums == expected_sums, {"actual": len(sums), "expected": len(expected_sums)})
        for relative, record in record_map.items():
            member_name = f"{CORE_ROOT}/{relative}"
            member = archive.getmember(member_name)
            handle = archive.extractfile(member)
            if handle is None:
                audit.check(f"core_file_{relative}_readable", False, member_name)
                continue
            digest, size = sha256_stream(handle)
            audit.check(f"core_file_{relative}_size", size == record.get("bytes"), size)
            audit.check(f"core_file_{relative}_sha", digest == record.get("sha256"), digest)
        lowered = [name.lower() for name in names]
        audit.check("core_no_bpu_bin", not any(name.endswith(".bin") for name in lowered), [name for name in lowered if name.endswith(".bin")])
        audit.check("core_no_dataset_tree", not any("/datasets/" in name for name in lowered), [])
        audit.check("core_no_hardcoded_edge_system_unit", f"{CORE_ROOT}/rootscope/deploy/x5/systemd/rootscope-edge.service" not in names, "manual/current-user only")
        wrapper = _read_member(archive, f"{CORE_ROOT}/install_and_selftest.sh").decode("utf-8")
        audit.check("core_wrapper_no_network_or_service", not any(token in wrapper.lower() for token in ("curl ", "wget ", "apt ", "ssh ", "systemctl", "sudo ")), wrapper)
        registry = json.loads(_read_member(archive, f"{CORE_ROOT}/rootscope/app/vision/known_card_template_registry.frozen.experimental.json").decode("utf-8"))
        templates = registry.get("templates", [])
        audit.check("core_registry_three_positive", len(templates) == 3 and {item.get("class_name") for item in templates} == {"grass_clump", "low_shrub", "young_tree"}, [item.get("class_name") for item in templates])
        audit.check("core_registry_unknown_absent", all(item.get("class_name") != "unknown" for item in templates), templates)
        audit.check("core_manifest_bound_to_receipt", hashlib.sha256(manifest_bytes).hexdigest() == receipt.get("internal_manifest_sha256"), hashlib.sha256(manifest_bytes).hexdigest())


def _audit_llm(audit: Audit, path: Path, receipt: Mapping[str, Any]) -> None:
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        audit.check("llm_members_safe", all(_safe_member(name, LLM_ROOT) for name in names), names)
        audit.check("llm_regular_only", all(member.isfile() for member in members), names)
        audit.check("llm_three_files", len(names) == 3, names)
        manifest_name = f"{LLM_ROOT}/release_manifest.json"
        manifest = json.loads(_read_member(archive, manifest_name).decode("utf-8"))
        flags = manifest.get("formal_flags")
        audit.check("llm_formal_flags_false", isinstance(flags, Mapping) and bool(flags) and all(value is False for value in flags.values()), flags)
        artifact = manifest.get("artifact", {})
        model_name = f"{LLM_ROOT}/{artifact.get('filename')}"
        model = archive.getmember(model_name)
        handle = archive.extractfile(model)
        if handle is None:
            audit.check("llm_model_readable", False, model_name)
        else:
            digest, size = sha256_stream(handle)
            audit.check("llm_model_sha", digest == artifact.get("sha256") == receipt.get("model_sha256"), digest)
            audit.check("llm_model_bytes", size == artifact.get("size_bytes") == receipt.get("model_bytes"), size)
        audit.check("llm_server_not_bundled", receipt.get("llama_server_bundled") is False, receipt.get("llama_server_bundled"))
        audit.check("llm_server_not_qualified", receipt.get("llama_server_qualified") is False, receipt.get("llama_server_qualified"))
        audit.check("llm_no_executable_member", not any("llama-server" in name or name.endswith(".so") for name in names), names)


def audit_release(release_dir: Path) -> dict[str, Any]:
    release = release_dir.resolve(strict=True)
    audit = Audit()
    receipt_path = release / "release_build_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    audit.check("receipt_schema", receipt.get("schema") == "rootscope.x5-offline-release-build-receipt.v1", receipt.get("schema"))
    audit.check("receipt_status", receipt.get("status") == "PASS_DETERMINISTIC_PACKAGING_NOT_X5_QUALIFIED", receipt.get("status"))
    _audit_outer(audit, release, receipt)
    core_record = receipt.get("core", {})
    llm_record = receipt.get("readonly_llm_model", {})
    if (release / CORE_NAME).is_file() and isinstance(core_record, Mapping):
        _audit_core(audit, release / CORE_NAME, core_record)
    if (release / LLM_NAME).is_file() and isinstance(llm_record, Mapping):
        _audit_llm(audit, release / LLM_NAME, llm_record)
    return audit.result(release)


def _parser() -> argparse.ArgumentParser:
    adventurex = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=adventurex / "output/releases/rootscope_x5_offline_v1",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_release(args.release_dir)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
