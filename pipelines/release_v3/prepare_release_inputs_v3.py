#!/usr/bin/env python3
"""Create the explicit hash-bound input allowlist for the final v3 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CANDIDATE_PREFIX = "rootscope_v3_pc_ready_20260724"
E0_CONTRACT_ROOT = (
    "43882938b7bb3ef34b8febf51ac1a8bbc92c8cc815e848b8b5c61d371768eaa3"
)
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".cache",
    ".git",
    "logs",
    "pc_gate_receipt_20260724.json",
    "pc_gate_receipt_v2_20260724.json",
    "pc_gate_receipt_v3_final_20260724.json",
    "pc_gate_receipt_v4_final_20260724.json",
    "pc_gate_receipt_v5_final_20260724.json",
    "pc_gate_receipt_v6_camera_hashfix_20260724.json",
    "pc_gate_receipt_v7_camera_hashfix_final_20260724.json",
    "pc_gate_receipt_v8_camera_fuser_final_20260724.json",
    "pc_gate_receipt_v9_rootmind_cache_release_final_20260724.json",
    "pc_gate_receipt_v10_rootmind_cache_release_deploy_final_20260724.json",
    "pc_gate_receipt_v11_rootmind_cache_migration_final_20260724.json",
    "pc_gate_receipt_v12_rootmind_precondition_final_20260724.json",
    "pc_gate_receipt_v13_x5_8gb_identity_final_20260724.json",
    "pc_gate_receipt_v14_x5_8gb_activation_contract_final_20260724.json",
    "release_inputs_v3_20260724.json",
    "release_inputs_v3_final_20260724.json",
    "release_inputs_v3_final_camera_hashfix_20260724.json",
    "release_inputs_v3_v7_camera_hashfix_final_20260724.json",
    "release_inputs_v3_v8_camera_fuser_final_20260724.json",
    "release_inputs_v3_v9_rootmind_cache_release_final_20260724.json",
    "release_inputs_v3_v10_rootmind_cache_release_deploy_final_20260724.json",
    "release_inputs_v3_v11_rootmind_cache_migration_final_20260724.json",
    "release_inputs_v3_v12_rootmind_precondition_final_20260724.json",
    "release_inputs_v3_v13_x5_8gb_identity_final_20260724.json",
    "release_inputs_v3_v14_x5_8gb_activation_contract_final_20260724.json",
    "final_release_receipt_20260724.json",
    "rootscope_v3_candidate_unqualified",
}
RESEARCH_ONLY_RELEASE_EXCLUSIONS = {
    "rootscope_v3/rag2/pack/corpus_embeddings.f16.npy",
    "rootscope_v3/rag2/pack/corpus_embeddings.v1.json",
}
TEXT_TREES = (
    "rootscope/app/competition_runtime",
    "rootscope/app/competition_llm",
    "rootscope/app/competition_rag",
    "rootscope/app/edge",
    "rootscope/app/omega",
    "rootscope/app/omega_knowledge",
    "rootscope/app/omega_vision",
    "rootscope/app/vision",
    "rootscope/app/runtime_v3",
    "rootscope/app/action_v3",
    "rootscope/app/rootmind_v3",
    "rootscope/app/system_v3",
    "rootscope/configs/competition",
    "rootscope/configs/competition_v3",
    "rootscope/configs/omega",
    "rootscope_v3/governance",
    "rootscope_v3/registries",
    "rootscope_v3/schemas",
    "rootscope_v3/examples",
    "rootscope_v3/evaluations",
    "rootscope_v3/evidence",
    "rootscope_v3/vision",
    "rootscope_v3/llm",
    "tools/release_v3",
)
RAG_RUNTIME_FILES = (
    "rootscope_v3/rag2/deploy_selection.v1.json",
    "rootscope_v3/rag2/bm25_runtime.py",
    "rootscope_v3/rag2/pack/rootscope_rag_sources.v2.json",
    "rootscope_v3/rag2/pack/rootscope_rag_corpus.v2.jsonl",
    "rootscope_v3/rag2/pack/rootscope_rag_citation_allowlist.v2.json",
    "rootscope_v3/rag2/pack/rag2_index.sqlite3",
)
RAG_QUALIFICATION_FILES = (
    "rootscope_v3/rag2/pack/rootscope_rag_gold_qa.v2.jsonl",
    "rootscope_v3/rag2/pack/rootscope_rag_forbidden_qa.v2.jsonl",
    "rootscope_v3/rag2/pack/manifest.v2.json",
    "rootscope_v3/rag2/pack/index_build_receipt.v1.json",
    "rootscope_v3/rag2/evidence/rag2_audit_20260724.json",
    "rootscope_v3/rag2/RAG2_HANDOFF_20260724.md",
)
EXPLICIT_FILES = (
    "rootscope/pyproject.toml",
    "rootscope/deploy/x5/capsule_config.seed17_cpu_experimental.json",
    "rootscope/ROOTSCOPE_RULE_DRIVEN_ALGORITHM_UPGRADE_PLAN_V3_20260724.md",
    "rootscope/ROOTSCOPE_COMPETITION_RUNTIME_V2_X5_HANDOFF_20260723.md",
    "rootscope/tools/x5_hbm_persistent_qualify_v3.py",
    "rootscope/tools/x5_hrt_oracle_qualify_v3.py",
    "rootscope/tools/x5_native_libdnn_qualify_v3.py",
    "rootscope/tools/x5_rootmind_cache_release_v3.py",
    "rootscope/tools/run_v3_pc_fault_matrix.py",
    "rootscope/tools/x5_competition_live_vision.py",
    "rootscope/tools/x5_competition_live_vision_v2.py",
    "rootscope/tools/x5_competition_static_cpu_bpu_replay.py",
    "rootscope/tools/x5_competition_resource_monitor.py",
    "rootscope/tools/x5_v3_live_camera_gate.py",
    "rootscope/tools/start_x5_competition_runtime_v2.sh",
    "rootscope/tools/start_x5_competition_live_vision_v2.sh",
    "rootscope/tests/test_runtime_v3_contracts.py",
    "rootscope/tests/test_v3_fault_matrix.py",
    "rootscope/tests/test_system_v3_coordinator.py",
    "rootscope/tests/test_release_v3_builder.py",
    "rootscope/tests/test_native_libdnn_contract_v3.py",
    "rootscope/tests/test_x5_v3_live_camera_gate.py",
    "rootscope/tests/test_rootmind_v3.py",
    "rootscope/tests/test_x5_rootmind_cache_release_v3.py",
    "rootscope/tests/test_x5_accept_cache_contract_v3.py",
    "rootscope_v3/README.md",
    "rootscope_v3/FINAL_PC_HANDOFF_20260724.md",
    (
        "rootscope_v3/models/llm/"
        "rootscope_qwen3_17b_qlora_final_v6_adv96_bound/"
        "training_receipt.json"
    ),
    (
        "rootscope_v3/models/llm/"
        "rootscope_qwen3_17b_merged_fp16_final_v6/"
        "merge_receipt.json"
    ),
    "rootscope_v3/E0_HANDOFF_20260724.md",
    "rootscope_v3/evidence/e0_verification_receipt_20260724.json",
    "rootscope/evidence/v3_pc_readiness_20260724/fault_matrix.json",
)
BINARY_FILES = (
    (
        "output/rootscope_bpu_seed17_quant_variant_r7_default_int16_all_nodes/"
        "model_output/"
        "rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes.bin",
        "models/rootscope_seed17_resnet18_224x224_rgb_ddr_r7.bin",
        "BPU_MODEL",
        0o444,
    ),
    (
        "rootscope/deploy/x5/models/"
        "rootscope_seed17_cpu_experimental_opset11.onnx",
        "models/rootscope_seed17_cpu.onnx",
        "CPU_MODEL",
        0o444,
    ),
    (
        "output/rootscope_llama_server_arm64_b9637_v1/bin/llama-server",
        "bin/llama-server",
        "ARM64_RUNTIME",
        0o555,
    ),
    (
        (
            "output/rootscope_native_libdnn_bridge_x5_20260724/bin/"
            "rootscope-native-libdnn-worker"
        ),
        "bin/rootscope-native-libdnn-worker",
        "ARM64_BPU_RUNTIME",
        0o555,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible(path: Path) -> bool:
    return path.is_file() and not any(part in EXCLUDED_PARTS for part in path.parts)


def tree_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (item for item in root.rglob("*") if eligible(item))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adventurex", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fast-llm", type=Path, required=True)
    parser.add_argument("--deep-llm", type=Path, required=True)
    parser.add_argument("--pc-gate-receipt", type=Path, required=True)
    parser.add_argument("--rag-model-dir", type=Path)
    args = parser.parse_args()
    adventurex = args.adventurex.resolve(strict=True)
    entries: dict[str, dict[str, Any]] = {}

    def add(
        source: Path,
        package_path: str,
        category: str,
        mode: int | None = None,
    ) -> None:
        resolved = source.resolve(strict=True)
        resolved.relative_to(adventurex)
        relative = resolved.relative_to(adventurex).as_posix()
        selected_mode = mode
        if selected_mode is None:
            selected_mode = (
                0o555 if resolved.suffix == ".sh" or resolved.name.endswith(".py") else 0o444
            )
        value = {
            "source": relative,
            "path": package_path,
            "sha256": sha256_file(resolved),
            "mode": selected_mode,
            "category": category,
        }
        previous = entries.get(package_path)
        if previous is not None and previous != value:
            raise ValueError(f"duplicate package path: {package_path}")
        entries[package_path] = value

    for tree in TEXT_TREES:
        root = adventurex / tree
        for path in tree_files(root):
            relative = path.relative_to(adventurex).as_posix()
            if relative in RESEARCH_ONLY_RELEASE_EXCLUSIONS:
                continue
            add(path, relative, "CODE_CONFIG_EVIDENCE")
    for relative in EXPLICIT_FILES:
        path = adventurex / relative
        if path.exists():
            add(path, relative, "CODE_CONFIG_EVIDENCE")
    for relative in RAG_RUNTIME_FILES:
        add(adventurex / relative, relative, "RAG_BM25_RUNTIME", 0o444)
    for relative in RAG_QUALIFICATION_FILES:
        add(adventurex / relative, relative, "RAG_QUALIFICATION_EVIDENCE", 0o444)
    for source, package_path, category, mode in BINARY_FILES:
        add(adventurex / source, package_path, category, mode)

    static_root = (
        adventurex
        / "rootscope"
        / "evidence"
        / "rootscope_cpu_bpu_replay_inputs_43_20260723"
    )
    for path in tree_files(static_root):
        relative_below = path.relative_to(static_root).as_posix()
        add(path, f"inputs/static43/{relative_below}", "BPU_ORACLE_INPUT", 0o444)

    wheel_roots = (
        adventurex / "output" / "x5_wheelhouse_probe_cp310_aarch64",
        adventurex / "output" / "x5_opencv_wheel_probe_cp310_aarch64",
    )
    wheel_names: set[str] = set()
    for root in wheel_roots:
        for path in sorted(root.glob("*.whl")):
            if path.name in wheel_names:
                continue
            wheel_names.add(path.name)
            add(path, f"wheelhouse/{path.name}", "OFFLINE_AARCH64_WHEEL", 0o444)

    fast = args.fast_llm.resolve(strict=True)
    add(fast, f"models/llm/fast/{fast.name}", "ROOTMIND_FAST_MODEL", 0o444)
    deep = args.deep_llm.resolve(strict=True)
    add(deep, f"models/llm/deep/{deep.name}", "ROOTMIND_DEEP_MODEL", 0o444)
    gate = args.pc_gate_receipt.resolve(strict=True)
    gate.relative_to(adventurex)
    gate_value = json.loads(gate.read_text(encoding="utf-8"))
    if (
        gate_value.get("status")
        != "PASS_PC_COMPLETE_X5_FINAL_CANDIDATE_ACCEPTANCE_PENDING"
    ):
        raise ValueError("PC gate receipt is not passing")
    gate_models = gate_value.get("models")
    if not isinstance(gate_models, dict):
        raise ValueError("PC gate receipt does not bind RootMind models")
    for role, model_path in (("fast", fast), ("deep", deep)):
        observed = gate_models.get(role)
        expected_relative = model_path.relative_to(adventurex).as_posix()
        if (
            not isinstance(observed, dict)
            or observed.get("path") != expected_relative
            or observed.get("bytes") != model_path.stat().st_size
            or observed.get("sha256") != sha256_file(model_path)
        ):
            raise ValueError(f"{role} RootMind model differs from the PC gate receipt")
    add(gate, "evidence/pc_gate_receipt.json", "PC_GATE_RECEIPT", 0o444)
    if args.rag_model_dir is not None:
        rag_root = args.rag_model_dir.resolve(strict=True)
        rag_root.relative_to(adventurex)
        for path in tree_files(rag_root):
            relative = path.relative_to(rag_root).as_posix()
            add(path, f"models/rag/{relative}", "RAG_DENSE_CHALLENGER", 0o444)

    entry_contract = [
        {
            "path": item["path"],
            "bytes": (adventurex / item["source"]).stat().st_size,
            "sha256": item["sha256"],
            "mode": item["mode"],
            "category": item["category"],
        }
        for item in (entries[key] for key in sorted(entries))
    ]
    entry_contract_root = hashlib.sha256(
        (
            json.dumps(
                entry_contract,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    candidate_id = (
        f"{CANDIDATE_PREFIX}_"
        f"{entry_contract_root[:12]}"
    )
    payload = {
        "schema": "rootscope.v3.release-inputs.v1",
        "candidate_id": candidate_id,
        "registry_and_schema_root_sha256": E0_CONTRACT_ROOT,
        "pc_gate_receipt_sha256": sha256_file(gate),
        "rag_default": "SQLITE_FTS5_BM25_V2",
        "rag_dense_challenger_packaged": args.rag_model_dir is not None,
        "entries": [entries[key] for key in sorted(entries)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    total = sum(
        (adventurex / item["source"]).stat().st_size for item in payload["entries"]
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "entry_count": len(payload["entries"]),
                "input_bytes": total,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
