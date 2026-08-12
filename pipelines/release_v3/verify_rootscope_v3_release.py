#!/usr/bin/env python3
"""Verify one extracted RootScope v3 candidate without opening hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import stat
from typing import Any, Mapping


MANIFEST_SCHEMA = "rootscope.v3.candidate-manifest.v1"
E0_CONTRACT_ROOT = "43882938b7bb3ef34b8febf51ac1a8bbc92c8cc815e848b8b5c61d371768eaa3"
V2_ROLLBACK_SHA256 = "03ca7b8d9ff8b691f1fd61dc696601ba30f494377a0b2a3cfadb66c19478ed94"
ZERO_AUTHORITY = {
    "execution_authority": False,
    "external_network": False,
    "serial_write": False,
    "gpio_write": False,
    "pump_command": False,
    "state_machine_write": False,
    "physical_completion": False,
}
PENDING_TRUE = {
    "x5_identity_pending",
    "x5_cpu_replay_pending",
    "x5_persistent_bpu_replay_pending",
    "x5_resource_soak_pending",
    "camera_live_pending",
    "stm32_pending",
}
EXPECTED_IDENTITY = {
    "hostname": "rootscope-x5",
    "serial": "3281556110220e0c002bdeab0012004",
    "machine_id": "<redacted-device-boot-id>",
    "wlan_mac": "02:00:00:00:00:01",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
IGNORED = {"__pycache__", ".pytest_cache"}
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
LEGACY_ROLLBACK_MIGRATIONS = {
    "rootscope_v3_pc_ready_20260724_45d0b6fa434b": {
        "entry_contract_root_sha256": (
            "45d0b6fa434b2d9c24401fffccac9b4eba2482e48f0785729ce800d570a25038"
        ),
        "allowed_missing_runtime_paths": {
            "rootscope/tools/x5_rootmind_cache_release_v3.py",
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
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


def safe_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return Path(*pure.parts)


def read_identity() -> dict[str, str]:
    serial = Path("/proc/device-tree/serial-number")
    if not serial.exists():
        serial = Path("/sys/firmware/devicetree/base/serial-number")
    return {
        "hostname": platform.node(),
        "serial": serial.read_bytes().replace(b"\x00", b"").decode("ascii"),
        "machine_id": Path("/etc/machine-id").read_text(encoding="ascii").strip(),
        "wlan_mac": Path("/sys/class/net/wlan0/address")
        .read_text(encoding="ascii")
        .strip(),
    }


def verify(
    root: Path,
    *,
    require_x5: bool,
    allow_legacy_rollback: bool = False,
) -> dict[str, Any]:
    release = root.resolve(strict=True)
    test_fixture_only = release.name.startswith("rootscope_v3_test_fixture_")
    release_pattern = (
        r"rootscope_v3_test_fixture_20260724_[0-9a-f]{12}"
        if test_fixture_only
        else r"rootscope_v3_pc_ready_20260724_[0-9a-f]{12}"
    )
    if (
        release.is_symlink()
        or not release.is_dir()
        or re.fullmatch(release_pattern, release.name) is None
    ):
        raise ValueError("release root identity/path is invalid")
    manifest_path = release / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("candidate_id") != release.name
        or manifest.get("test_fixture_only") is not test_fixture_only
    ):
        raise ValueError("candidate manifest identity changed")
    if test_fixture_only:
        if (
            require_x5
            or allow_legacy_rollback
            or manifest.get("release_state") != "TEST_FIXTURE_NOT_DEPLOYABLE"
        ):
            raise ValueError("test fixture candidate is never X5 deployable")
    elif manifest.get("release_state") != "PC_COMPLETE_X5_QUALIFICATION_PENDING":
        raise ValueError("production candidate release state changed")
    authority = manifest.get("authority")
    if authority != ZERO_AUTHORITY:
        raise ValueError("candidate authority must remain entirely false")
    qualification = manifest.get("qualification")
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("pc_gates_passed") is not True
        or any(qualification.get(key) is not True for key in PENDING_TRUE)
        or qualification.get("physical_closure") is not False
        or not SHA_RE.fullmatch(str(qualification.get("pc_gate_receipt_sha256", "")))
    ):
        raise ValueError("candidate truth boundary changed")
    if manifest.get("runtime_selection") != {
        "rag_default": "SQLITE_FTS5_BM25_V2",
        "rag_dense_challenger_packaged": False,
    }:
        raise ValueError("runtime selection changed")
    if manifest.get("rollback") != {
        "v2_archive_sha256": V2_ROLLBACK_SHA256,
        "v2_must_remain_unchanged": True,
    }:
        raise ValueError("rollback contract changed")
    contracts = manifest.get("contracts")
    if (
        not isinstance(contracts, Mapping)
        or contracts.get("registry_and_schema_root_sha256") != E0_CONTRACT_ROOT
        or not SHA_RE.fullmatch(str(contracts.get("entry_contract_root_sha256", "")))
    ):
        raise ValueError("contract roots changed")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("candidate file records missing")
    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("file record must be an object")
        name = record.get("path")
        digest = record.get("sha256")
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or not SHA_RE.fullmatch(digest)
        ):
            raise ValueError("invalid file record")
        safe_path(name)
        if name in expected:
            raise ValueError(f"duplicate file record: {name}")
        size = record.get("bytes")
        mode = record.get("mode")
        category = record.get("category")
        if (
            not isinstance(size, int)
            or size < 0
            or mode not in {0o444, 0o544, 0o555, 0o644, 0o755}
            or not isinstance(category, str)
            or not category
        ):
            raise ValueError("invalid file bytes/mode/category")
        marker = f"{name}\n{category}".casefold()
        if any(value in marker for value in ("corpus_embeddings", "bge-small-zh", "rag_dense_challenger")):
            raise ValueError("dense RAG asset is forbidden")
        expected[name] = dict(record)
    calculated_contract = hashlib.sha256(
        canonical_json(
            [
                {
                    key: expected[name][key]
                    for key in ("path", "bytes", "sha256", "mode", "category")
                }
                for name in sorted(expected)
            ]
        )
    ).hexdigest()
    if calculated_contract != contracts["entry_contract_root_sha256"]:
        raise ValueError("entry contract root mismatch")
    if not release.name.endswith("_" + calculated_contract[:12]):
        raise ValueError("candidate id is not bound to the entry contract root")
    gate = expected.get("evidence/pc_gate_receipt.json")
    if (
        gate is None
        or gate["category"] != "PC_GATE_RECEIPT"
        or gate["sha256"] != qualification["pc_gate_receipt_sha256"]
    ):
        raise ValueError("PC gate receipt binding changed")
    expected_hashes = {name: record["sha256"] for name, record in expected.items()}
    expected_hashes["candidate_manifest.json"] = sha256_file(manifest_path)
    actual: dict[str, Path] = {}
    for path in release.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink forbidden inside candidate: {path}")
        relative_parts = path.relative_to(release).parts
        if any(part in IGNORED for part in relative_parts):
            continue
        if path.is_file() and path.name != "SHA256SUMS":
            actual[path.relative_to(release).as_posix()] = path
    if set(actual) != set(expected_hashes):
        raise ValueError(
            f"manifest coverage mismatch missing={sorted(set(expected_hashes)-set(actual))} "
            f"extra={sorted(set(actual)-set(expected_hashes))}"
        )
    for name, digest in expected_hashes.items():
        if sha256_file(actual[name]) != digest:
            raise ValueError(f"payload SHA-256 mismatch: {name}")
        if name in expected and actual[name].stat().st_size != expected[name]["bytes"]:
            raise ValueError(f"payload byte count mismatch: {name}")
        if (
            name in expected
            and platform.system() != "Windows"
            and stat.S_IMODE(actual[name].stat().st_mode) != expected[name]["mode"]
        ):
            raise ValueError(f"payload mode mismatch: {name}")
    legacy_rollback_migration = False
    if not test_fixture_only:
        missing_runtime = REQUIRED_RUNTIME_PATHS - set(expected)
        if missing_runtime:
            migration = LEGACY_ROLLBACK_MIGRATIONS.get(release.name)
            if (
                not allow_legacy_rollback
                or not require_x5
                or not isinstance(migration, Mapping)
                or contracts.get("entry_contract_root_sha256")
                != migration.get("entry_contract_root_sha256")
                or missing_runtime
                != migration.get("allowed_missing_runtime_paths")
            ):
                raise ValueError(
                    f"required X5 runtime assets missing: {sorted(missing_runtime)}"
                )
            legacy_rollback_migration = True
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
            item = expected[path]
            if item["category"] != category or item["bytes"] < minimum_bytes:
                raise ValueError(f"invalid required X5 runtime asset: {path}")
        for role, category in (
            ("fast", "ROOTMIND_FAST_MODEL"),
            ("deep", "ROOTMIND_DEEP_MODEL"),
        ):
            matches = [
                (name, item)
                for name, item in expected.items()
                if item["category"] == category
            ]
            if (
                len(matches) != 1
                or not matches[0][0].endswith(".gguf")
                or matches[0][1]["bytes"] < 100_000_000
            ):
                raise ValueError(f"{role} RootMind artifact is not a plausible GGUF")
        wheel_lock = json.loads(
            actual["tools/release_v3/x5_wheelhouse_lock.v1.json"].read_text(
                encoding="utf-8"
            )
        )
        expected_wheels = wheel_lock.get("wheels")
        observed_wheels = {
            Path(name).name: item["sha256"]
            for name, item in expected.items()
            if name.startswith("wheelhouse/")
        }
        if (
            not isinstance(expected_wheels, dict)
            or len(expected_wheels) != 12
            or observed_wheels != expected_wheels
            or any(
                item["category"] != "OFFLINE_AARCH64_WHEEL"
                or not name.endswith(".whl")
                or name != f"wheelhouse/{Path(name).name}"
                for name, item in expected.items()
                if name.startswith("wheelhouse/")
            )
        ):
            raise ValueError("X5 wheelhouse does not exactly match the locked 12-wheel set")
        native_contract = json.loads(
            actual[
                "rootscope/app/runtime_v3/native/compile_contract_x5.v1.json"
            ].read_text(encoding="utf-8")
        )
        native_source = expected[
            "rootscope/app/runtime_v3/native/rootscope_libdnn_worker.cpp"
        ]
        native_binary = expected["bin/rootscope-native-libdnn-worker"]
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
    gate_value = json.loads(
        actual["evidence/pc_gate_receipt.json"].read_text(encoding="utf-8")
    )
    if not test_fixture_only:
        if (
            gate_value.get("schema") != "rootscope.v3.pc-gate-receipt.v1"
            or gate_value.get("status")
            != "PASS_PC_COMPLETE_X5_FINAL_CANDIDATE_ACCEPTANCE_PENDING"
            or gate_value.get("authority") != ZERO_AUTHORITY
            or gate_value.get("pending_x5_gates")
            != [
                "identity",
                "cpu_replay",
                "persistent_native_libdnn_replay",
                "resource_soak",
                "live_camera",
                "stm32",
                "physical_closure",
            ]
        ):
            raise ValueError("PC gate truth boundary changed")
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
        contract_root = hashlib.sha256(
            canonical_compact(
                {key: gate_value[key] for key in contract_keys}
            )
        ).hexdigest()
        if gate_value.get("contract_root_sha256") != contract_root:
            raise ValueError("PC gate contract root mismatch")

        gate_references: list[tuple[str, Mapping[str, Any]]] = []
        gate_receipts = gate_value.get("receipts")
        if not isinstance(gate_receipts, Mapping):
            raise ValueError("PC gate receipt references missing")
        for name, value in gate_receipts.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"PC gate {name} receipt reference malformed")
            gate_references.append((f"receipt:{name}", value))
        llm_contract = gate_value.get("llm_training_contract")
        if not isinstance(llm_contract, Mapping):
            raise ValueError("PC gate LLM contract missing")
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
            gate_references.append((f"llm:{name}", value))
        for label, reference in gate_references:
            path = reference.get("path")
            digest = reference.get("sha256")
            if (
                not isinstance(path, str)
                or not isinstance(digest, str)
                or not SHA_RE.fullmatch(digest)
                or path not in expected
                or expected[path]["sha256"] != digest
                or sha256_file(actual[path]) != digest
            ):
                raise ValueError(f"packaged PC gate reference mismatch: {label}")

        expected_qualification = {
            "vision": "PASS",
            "llm": "PASS",
            "rag": "PASS",
            "resource": "NOT_EVALUATED",
            "physical": "SIMULATED_ONLY",
        }
        for name, expected_status in expected_qualification.items():
            value = json.loads(
                actual[gate_receipts[name]["path"]].read_text(encoding="utf-8")
            )
            qualification_value = value.get(
                "qualification", value.get("outcome")
            )
            if (
                value.get("schema") != gate_receipts[name].get("schema")
                or not isinstance(qualification_value, Mapping)
                or qualification_value.get("status") != expected_status
            ):
                raise ValueError(f"packaged {name} qualification changed")

        compiler_reference = llm_contract["safety_compiler"]
        compiler = json.loads(
            actual[compiler_reference["path"]].read_text(encoding="utf-8")
        )
        compiler_results = compiler.get("results")
        if (
            compiler.get("schema")
            != "rootscope.v3.llm-safety-compiler-evaluation.v1"
            or compiler.get("status") != "PASS_END_TO_END_FAIL_CLOSED"
            or compiler.get("compiler_metrics")
            != {
                "case_count": 32,
                "raw_accept_count": 32,
                "deterministic_fallback_count": 0,
                "no_valid_citation_count": 0,
                "unsafe_escape_count": 0,
            }
            or compiler.get("end_to_end_metrics")
            != {"contract_pass_count": 32, "contract_pass_rate": 1.0}
            or not isinstance(compiler_results, list)
            or len(compiler_results) != 32
            or hashlib.sha256(canonical_compact(compiler_results)).hexdigest()
            != compiler_reference.get("results_root_sha256")
        ):
            raise ValueError("packaged Safety Compiler qualification changed")
    gate_models = gate_value.get("models")
    if not isinstance(gate_models, Mapping):
        raise ValueError("PC gate receipt does not bind RootMind models")
    for role, category, prefix in (
        ("fast", "ROOTMIND_FAST_MODEL", "models/llm/fast/"),
        ("deep", "ROOTMIND_DEEP_MODEL", "models/llm/deep/"),
    ):
        model_records = [
            (name, record)
            for name, record in expected.items()
            if name.startswith(prefix)
        ]
        observed = gate_models.get(role)
        if (
            len(model_records) != 1
            or model_records[0][1]["category"] != category
            or not isinstance(observed, Mapping)
            or model_records[0][1]["bytes"] != observed.get("bytes")
            or model_records[0][1]["sha256"] != observed.get("sha256")
        ):
            raise ValueError(f"{role} RootMind model is not exactly bound to PC gate")
    sums = {}
    for line in (release / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or not SHA_RE.fullmatch(digest) or name in sums:
            raise ValueError("SHA256SUMS malformed")
        safe_path(name)
        sums[name] = digest
    if sums != dict(sorted(expected_hashes.items())):
        raise ValueError("SHA256SUMS exact coverage mismatch")

    identity: dict[str, str] | None = None
    if require_x5:
        if platform.machine() != "aarch64":
            raise ValueError(f"expected aarch64, got {platform.machine()}")
        identity = read_identity()
        if identity != EXPECTED_IDENTITY:
            raise ValueError(f"X5 identity mismatch: {identity}")
    return {
        "schema": "rootscope.v3.release-verification-receipt.v1",
        "status": (
            "PASS_TEST_FIXTURE_ONLY_NOT_DEPLOYABLE"
            if test_fixture_only
            else "PASS_X5_PINNED_LEGACY_ROLLBACK_MIGRATION_ZERO_AUTHORITY"
            if legacy_rollback_migration
            else "PASS_X5_STAGED_ZERO_AUTHORITY_LIVE_QUALIFICATION_PENDING"
            if require_x5
            else "PASS_PC_ARCHIVE_CONTENT_ZERO_AUTHORITY"
        ),
        "legacy_rollback_migration": legacy_rollback_migration,
        "candidate_id": release.name,
        "files_verified": len(expected_hashes),
        "identity": identity,
        "architecture": platform.machine(),
        "hardware_opened": False,
        "camera_opened": False,
        "serial_opened": False,
        "pump_touched": False,
        "authority": dict(authority),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--require-x5", action="store_true")
    parser.add_argument("--allow-legacy-rollback", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.release_root,
                require_x5=args.require_x5,
                allow_legacy_rollback=args.allow_legacy_rollback,
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
