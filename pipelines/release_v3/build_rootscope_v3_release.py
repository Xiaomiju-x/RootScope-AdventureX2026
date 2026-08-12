#!/usr/bin/env python3
"""Build a deterministic, immutable RootScope v3 X5 candidate archive.

The input allowlist is explicit and hash-bound.  Sources must remain under the
AdventureX root; previous releases and the frozen XRD runtime are never
modified.  The archive may contain models and public evaluation fixtures, but
its authority remains entirely false until separate hardware qualification.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import Any, Mapping


SCHEMA = "rootscope.v3.release-inputs.v1"
MANIFEST_SCHEMA = "rootscope.v3.candidate-manifest.v1"
E0_CONTRACT_ROOT = "43882938b7bb3ef34b8febf51ac1a8bbc92c8cc815e848b8b5c61d371768eaa3"
V2_ROLLBACK_SHA256 = (
    "03ca7b8d9ff8b691f1fd61dc696601ba30f494377a0b2a3cfadb66c19478ed94"
)
TEXT_SUFFIXES = {
    ".py", ".json", ".jsonl", ".md", ".sh", ".ps1", ".toml", ".yaml", ".yml",
    ".txt", ".csv", ".tsv", ".log", ".env",
}
FORBIDDEN_PARTS = {
    ".git", ".ssh", "__pycache__", ".pytest_cache", "credentials", "secrets",
    "private_keys",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"(?i)\bBearer[ \t]+[A-Za-z0-9._~+/=-]{8,}\b"),
)
ZERO_AUTHORITY = {
    "execution_authority": False,
    "external_network": False,
    "serial_write": False,
    "gpio_write": False,
    "pump_command": False,
    "state_machine_write": False,
    "physical_completion": False,
}
FORBIDDEN_DENSE_RELEASE_MARKERS = (
    "corpus_embeddings",
    "bge-small-zh",
    "dense_encoder",
    "RAG_DENSE_CHALLENGER",
)
REQUIRED_RAG_PATHS = {
    "rootscope_v3/rag2/bm25_runtime.py",
    "rootscope_v3/rag2/pack/rag2_index.sqlite3",
    "rootscope_v3/rag2/pack/rootscope_rag_citation_allowlist.v2.json",
    "rootscope_v3/rag2/pack/rootscope_rag_corpus.v2.jsonl",
}
REQUIRED_RUNTIME_PATHS = {
    "tools/release_v3/x5_bootstrap_runtime_v3.sh",
    "tools/release_v3/x5_wheelhouse_lock.v1.json",
    "tools/release_v3/requirements-x5-cpu-v3.txt",
    "bin/llama-server",
    "bin/rootscope-native-libdnn-worker",
    "models/rootscope_seed17_cpu.onnx",
    "models/rootscope_seed17_resnet18_224x224_rgb_ddr_r7.bin",
    "rootscope/app/runtime_v3/native/compile_contract_x5.v1.json",
    "rootscope/app/runtime_v3/native/rootscope_libdnn_worker.cpp",
    "rootscope/app/runtime_v3/native_libdnn_adapter.py",
    "rootscope/tools/x5_native_libdnn_qualify_v3.py",
    "rootscope/tools/x5_rootmind_cache_release_v3.py",
    "rootscope/tools/x5_v3_live_camera_gate.py",
    "tools/release_v3/x5_rootmind_smoke_v3.sh",
    "tools/release_v3/x5_accept_candidate_v3.sh",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_compact(value: Any) -> bytes:
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


def safe_relative(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe archive path: {value!r}")
    if {part.casefold() for part in pure.parts} & FORBIDDEN_PARTS:
        raise ValueError(f"forbidden archive path: {value!r}")
    return value


def source_file(adventurex: Path, relative: str) -> Path:
    safe_relative(relative)
    configured = adventurex / Path(*PurePosixPath(relative).parts)
    if configured.is_symlink():
        raise ValueError(f"source symlink forbidden: {relative}")
    resolved = configured.resolve(strict=True)
    try:
        resolved.relative_to(adventurex.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"source leaves AdventureX: {relative}") from exc
    if not resolved.is_file() or not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError(f"source is not a regular file: {relative}")
    return resolved


def scan_text(path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    text = path.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"possible embedded secret in {path}")


def tar_info(name: str, size: int, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def build(adventurex: Path, input_manifest: Path, output_root: Path) -> dict[str, Any]:
    spec = json.loads(input_manifest.read_text(encoding="utf-8"))
    if spec.get("schema") != SCHEMA:
        raise ValueError("release input schema mismatch")
    candidate_id = safe_relative(str(spec.get("candidate_id")))
    test_fixture_only = spec.get("test_fixture_only") is True
    candidate_pattern = (
        r"rootscope_v3_test_fixture_20260724_[0-9a-f]{12}"
        if test_fixture_only
        else r"rootscope_v3_pc_ready_20260724_[0-9a-f]{12}"
    )
    if "/" in candidate_id or re.fullmatch(candidate_pattern, candidate_id) is None:
        raise ValueError("candidate_id must be one content-addressed RootScope v3 directory name")
    entries = spec.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("release entries must be a non-empty list")
    if spec.get("rag_default") != "SQLITE_FTS5_BM25_V2":
        raise ValueError("RAG default must remain the PC-qualified BM25 v2 backend")
    if spec.get("rag_dense_challenger_packaged") is not False:
        raise ValueError("non-eligible dense RAG challenger must not enter X5 release")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, Mapping):
            raise ValueError("release entry must be an object")
        source_relative = safe_relative(str(item.get("source")))
        package_path = safe_relative(str(item.get("path")))
        if package_path in {"candidate_manifest.json", "SHA256SUMS"}:
            raise ValueError(f"reserved package path: {package_path}")
        if package_path in seen:
            raise ValueError(f"duplicate package path: {package_path}")
        seen.add(package_path)
        source = source_file(adventurex, source_relative)
        scan_text(source)
        actual = sha256_file(source)
        expected = item.get("sha256")
        if expected is not None and actual != expected:
            raise ValueError(f"source SHA-256 changed: {source_relative}")
        mode = int(item.get("mode", 0o644))
        if mode not in {0o444, 0o544, 0o555, 0o644, 0o755}:
            raise ValueError(f"unsupported archive mode for {package_path}")
        records.append(
            {
                "source": source,
                "source_relative": source_relative,
                "path": package_path,
                "bytes": source.stat().st_size,
                "sha256": actual,
                "mode": mode,
                "category": str(item.get("category", "UNSPECIFIED")),
            }
        )
    for item in records:
        marker_text = item["path"] + "\n" + item["category"]
        if any(marker.casefold() in marker_text.casefold() for marker in FORBIDDEN_DENSE_RELEASE_MARKERS):
            raise ValueError(f"non-eligible dense RAG asset in release: {item['path']}")
    missing_rag = REQUIRED_RAG_PATHS - {item["path"] for item in records}
    if missing_rag:
        raise ValueError(f"required BM25 runtime assets missing: {sorted(missing_rag)}")
    gate_records = [item for item in records if item["category"] == "PC_GATE_RECEIPT"]
    if len(gate_records) != 1 or gate_records[0]["path"] != "evidence/pc_gate_receipt.json":
        raise ValueError("exactly one canonical PC gate receipt is required")
    gate_value = json.loads(gate_records[0]["source"].read_text(encoding="utf-8"))
    if (
        gate_value.get("status")
        != "PASS_PC_COMPLETE_X5_FINAL_CANDIDATE_ACCEPTANCE_PENDING"
        or gate_records[0]["sha256"] != spec.get("pc_gate_receipt_sha256")
    ):
        raise ValueError("PC gate receipt status/hash mismatch")
    if bool(gate_value.get("test_fixture_only")) != test_fixture_only:
        raise ValueError("test fixture identity does not match the PC gate receipt")
    if not test_fixture_only:
        contract_keys = (
            "receipts",
            "models",
            "e0",
            "bpu_oracle",
            "llm_training_contract",
            "tests",
            "authority",
            "pending_x5_gates",
            "pc_completion_scope",
        )
        gate_contract_root = hashlib.sha256(
            canonical_compact(
                {key: gate_value[key] for key in contract_keys}
            )
        ).hexdigest()
        if gate_value.get("contract_root_sha256") != gate_contract_root:
            raise ValueError("PC gate contract root mismatch")
        by_source = {item["source_relative"]: item for item in records}
        referenced: list[tuple[str, Mapping[str, Any]]] = []
        gate_receipts = gate_value.get("receipts")
        llm_contract = gate_value.get("llm_training_contract")
        if not isinstance(gate_receipts, Mapping) or not isinstance(
            llm_contract, Mapping
        ):
            raise ValueError("PC gate packaged reference sets missing")
        for name, value in gate_receipts.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"PC gate {name} receipt reference malformed")
            referenced.append((f"receipt:{name}", value))
        for name in (
            "seal",
            "strict_binding",
            "evaluation_seal",
            "safety_compiler",
            "training_receipt",
            "merge_receipt",
            "model_export_seal",
        ):
            value = llm_contract.get(name)
            if not isinstance(value, Mapping):
                raise ValueError(f"PC gate LLM {name} reference missing")
            referenced.append((f"llm:{name}", value))
        for label, reference in referenced:
            path = reference.get("path")
            digest = reference.get("sha256")
            record = by_source.get(path)
            if (
                not isinstance(path, str)
                or not isinstance(digest, str)
                or record is None
                or record["sha256"] != digest
            ):
                raise ValueError(f"PC gate packaged reference mismatch: {label}")
    gate_models = gate_value.get("models")
    if not isinstance(gate_models, dict):
        raise ValueError("PC gate receipt does not bind RootMind models")
    for role, category, prefix in (
        ("fast", "ROOTMIND_FAST_MODEL", "models/llm/fast/"),
        ("deep", "ROOTMIND_DEEP_MODEL", "models/llm/deep/"),
    ):
        model_records = [item for item in records if item["path"].startswith(prefix)]
        observed = gate_models.get(role)
        if (
            len(model_records) != 1
            or model_records[0]["category"] != category
            or not isinstance(observed, dict)
            or model_records[0]["source_relative"] != observed.get("path")
            or model_records[0]["bytes"] != observed.get("bytes")
            or model_records[0]["sha256"] != observed.get("sha256")
        ):
            raise ValueError(f"{role} RootMind model is not exactly bound to PC gate")
    if not test_fixture_only:
        by_path = {item["path"]: item for item in records}
        missing_runtime = REQUIRED_RUNTIME_PATHS - set(by_path)
        if missing_runtime:
            raise ValueError(f"required X5 runtime assets missing: {sorted(missing_runtime)}")
        for path, category, minimum_bytes in (
            ("bin/llama-server", "ARM64_RUNTIME", 1_000_000),
            (
                "bin/rootscope-native-libdnn-worker",
                "ARM64_BPU_RUNTIME",
                10_000,
            ),
            ("models/rootscope_seed17_cpu.onnx", "CPU_MODEL", 100_000),
            (
                "models/rootscope_seed17_resnet18_224x224_rgb_ddr_r7.bin",
                "BPU_MODEL",
                100_000,
            ),
        ):
            item = by_path[path]
            if item["category"] != category or item["bytes"] < minimum_bytes:
                raise ValueError(f"invalid required X5 runtime asset: {path}")
        native_contract = json.loads(
            by_path[
                "rootscope/app/runtime_v3/native/compile_contract_x5.v1.json"
            ]["source"].read_text(encoding="utf-8")
        )
        native_source = by_path[
            "rootscope/app/runtime_v3/native/rootscope_libdnn_worker.cpp"
        ]
        native_binary = by_path["bin/rootscope-native-libdnn-worker"]
        native_repro = native_contract.get("reproducibility", {})
        if (
            native_contract.get("schema")
            != "rootscope.native-libdnn.x5-compile-contract.v1"
            or native_contract.get("status") != "PASS_REPRODUCIBLE_TWO_BUILD"
            or native_contract.get("target_arch") != "aarch64"
            or native_contract.get("protocol")
            != "rootscope.native-libdnn.protocol.v1"
            or native_contract.get("source", {}).get("path")
            != "rootscope/app/runtime_v3/native/rootscope_libdnn_worker.cpp"
            or native_contract.get("source", {}).get("sha256")
            != native_source["sha256"]
            or native_contract.get("binary", {}).get("package_path")
            != "bin/rootscope-native-libdnn-worker"
            or native_contract.get("binary", {}).get("source_path")
            != (
                "output/rootscope_native_libdnn_bridge_x5_20260724/bin/"
                "rootscope-native-libdnn-worker"
            )
            or native_contract.get("binary", {}).get("sha256")
            != native_binary["sha256"]
            or native_contract.get("binary", {}).get("bytes")
            != native_binary["bytes"]
            or native_contract.get("binary", {}).get("mode") != "0555"
            or native_binary["mode"] != 0o555
            or native_repro.get("independent_build_count") != 2
            or native_repro.get("byte_identical") is not True
            or native_repro.get("build_1_sha256") != native_binary["sha256"]
            or native_repro.get("build_2_sha256") != native_binary["sha256"]
        ):
            raise ValueError("native libdnn compile contract is not hash-bound")
        for role, category in (
            ("fast", "ROOTMIND_FAST_MODEL"),
            ("deep", "ROOTMIND_DEEP_MODEL"),
        ):
            item = next(record for record in records if record["category"] == category)
            if not item["path"].endswith(".gguf") or item["bytes"] < 100_000_000:
                raise ValueError(f"{role} RootMind artifact is not a plausible GGUF")
        lock_record = by_path["tools/release_v3/x5_wheelhouse_lock.v1.json"]
        wheel_lock = json.loads(lock_record["source"].read_text(encoding="utf-8"))
        expected_wheels = wheel_lock.get("wheels")
        if not isinstance(expected_wheels, dict) or len(expected_wheels) != 12:
            raise ValueError("X5 wheel lock must contain exactly 12 artifacts")
        wheel_records = [
            item for item in records if item["path"].startswith("wheelhouse/")
        ]
        observed_wheels = {
            Path(item["path"]).name: item["sha256"] for item in wheel_records
        }
        if observed_wheels != expected_wheels or any(
            item["category"] != "OFFLINE_AARCH64_WHEEL"
            or not item["path"].endswith(".whl")
            or item["path"] != f"wheelhouse/{Path(item['path']).name}"
            for item in wheel_records
        ):
            raise ValueError("X5 wheelhouse does not exactly match the locked 12-wheel set")
    records.sort(key=lambda item: item["path"])

    contract_root = sha256_bytes(
        canonical_json(
            [
                {
                    key: item[key]
                    for key in ("path", "bytes", "sha256", "mode", "category")
                }
                for item in records
            ]
        )
    )
    if not candidate_id.endswith("_" + contract_root[:12]):
        raise ValueError("candidate id is not bound to the entry contract root")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "candidate_id": candidate_id,
        "build_date": "2026-07-24",
        "release_state": (
            "TEST_FIXTURE_NOT_DEPLOYABLE"
            if test_fixture_only
            else "PC_COMPLETE_X5_QUALIFICATION_PENDING"
        ),
        "test_fixture_only": test_fixture_only,
        "rollback": {
            "v2_archive_sha256": V2_ROLLBACK_SHA256,
            "v2_must_remain_unchanged": True,
        },
        "contracts": {
            "registry_and_schema_root_sha256": spec.get(
                "registry_and_schema_root_sha256"
            ),
            "entry_contract_root_sha256": contract_root,
        },
        "qualification": {
            "pc_gates_passed": True,
            "pc_gate_receipt_sha256": spec.get("pc_gate_receipt_sha256"),
            "x5_identity_pending": True,
            "x5_cpu_replay_pending": True,
            "x5_persistent_bpu_replay_pending": True,
            "x5_resource_soak_pending": True,
            "camera_live_pending": True,
            "stm32_pending": True,
            "physical_closure": False,
        },
        "runtime_selection": {
            "rag_default": "SQLITE_FTS5_BM25_V2",
            "rag_dense_challenger_packaged": False,
        },
        "authority": dict(ZERO_AUTHORITY),
        "files": [
            {
                key: item[key]
                for key in ("path", "bytes", "sha256", "mode", "category")
            }
            for item in records
        ],
    }
    if manifest["contracts"]["registry_and_schema_root_sha256"] != E0_CONTRACT_ROOT:
        raise ValueError("E0 registry/schema contract root changed")
    if not manifest["qualification"]["pc_gates_passed"]:
        raise ValueError("PC gates have not been declared complete")
    manifest_bytes = canonical_json(manifest)
    file_hashes = {item["path"]: item["sha256"] for item in records}
    file_hashes["candidate_manifest.json"] = sha256_bytes(manifest_bytes)
    sums_bytes = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(file_hashes.items())
    ).encode("ascii")

    output_root.mkdir(parents=True, exist_ok=True)
    archive = output_root / f"{candidate_id}.tar"
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
        prefix = f"{candidate_id}/"
        tar.addfile(
            tar_info(prefix + "candidate_manifest.json", len(manifest_bytes), 0o444),
            BytesIO(manifest_bytes),
        )
        tar.addfile(
            tar_info(prefix + "SHA256SUMS", len(sums_bytes), 0o444),
            BytesIO(sums_bytes),
        )
        for item in records:
            with item["source"].open("rb") as handle:
                tar.addfile(
                    tar_info(prefix + item["path"], item["bytes"], item["mode"]),
                    handle,
                )
    archive_sha = sha256_file(archive)
    sha_path = output_root / f"{candidate_id}.tar.sha256"
    sha_path.write_text(f"{archive_sha}  {archive.name}\n", encoding="ascii")
    receipt = {
        "schema": "rootscope.v3.release-build-receipt.v1",
        "status": (
            "PASS_TEST_FIXTURE_ONLY_NOT_DEPLOYABLE"
            if test_fixture_only
            else "PASS_PC_COMPLETE_X5_FINAL_CANDIDATE_ACCEPTANCE_PENDING"
        ),
        "candidate_id": candidate_id,
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
        "file_count": len(records) + 2,
        "entry_contract_root_sha256": contract_root,
        "authority": dict(ZERO_AUTHORITY),
    }
    receipt_path = output_root / "release_build_receipt.json"
    receipt_path.write_bytes(canonical_json(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adventurex", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(
        args.adventurex.resolve(strict=True),
        args.inputs.resolve(strict=True),
        args.output.resolve(),
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
