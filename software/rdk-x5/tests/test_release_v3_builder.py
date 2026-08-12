from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import tempfile


def test_release_v3_requires_rootmind_lifecycle_assets() -> None:
    adventurex = Path(__file__).resolve().parents[2]
    required = {
        "rootscope/tools/x5_rootmind_cache_release_v3.py",
        "tools/release_v3/x5_rootmind_smoke_v3.sh",
        "tools/release_v3/x5_accept_candidate_v3.sh",
    }
    for relative in (
        "tools/release_v3/build_rootscope_v3_release.py",
        "tools/release_v3/verify_rootscope_v3_release.py",
    ):
        namespace = runpy.run_path(str(adventurex / relative))
        assert required <= namespace["REQUIRED_RUNTIME_PATHS"]


def test_release_v3_legacy_rollback_migration_is_exactly_pinned() -> None:
    adventurex = Path(__file__).resolve().parents[2]
    verifier = runpy.run_path(
        str(adventurex / "tools/release_v3/verify_rootscope_v3_release.py")
    )
    assert verifier["LEGACY_ROLLBACK_MIGRATIONS"] == {
        "rootscope_v3_pc_ready_20260724_45d0b6fa434b": {
            "entry_contract_root_sha256": (
                "45d0b6fa434b2d9c24401fffccac9b4eba2482e48f0785729ce800d570a25038"
            ),
            "allowed_missing_runtime_paths": {
                "rootscope/tools/x5_rootmind_cache_release_v3.py",
            },
        },
    }
    stage_source = (
        adventurex / "tools/release_v3/x5_stage_candidate_v3.sh"
    ).read_text(encoding="utf-8")
    assert "--allow-legacy-rollback" in stage_source


def test_release_v3_build_and_pc_verify() -> None:
    adventurex = Path(__file__).resolve().parents[2]
    build_script = adventurex / "tools" / "release_v3" / "build_rootscope_v3_release.py"
    verify_script = (
        adventurex / "tools" / "release_v3" / "verify_rootscope_v3_release.py"
    )
    source = adventurex / "rootscope" / "app" / "runtime_v3" / "__init__.py"
    sources = [
        (source, "rootscope/app/runtime_v3/__init__.py", "CODE"),
        (
            adventurex / "rootscope_v3/rag2/bm25_runtime.py",
            "rootscope_v3/rag2/bm25_runtime.py",
            "RAG_BM25_RUNTIME",
        ),
        (
            adventurex / "rootscope_v3/rag2/pack/rag2_index.sqlite3",
            "rootscope_v3/rag2/pack/rag2_index.sqlite3",
            "RAG_BM25_RUNTIME",
        ),
        (
            adventurex / "rootscope_v3/rag2/pack/rootscope_rag_citation_allowlist.v2.json",
            "rootscope_v3/rag2/pack/rootscope_rag_citation_allowlist.v2.json",
            "RAG_BM25_RUNTIME",
        ),
        (
            adventurex / "rootscope_v3/rag2/pack/rootscope_rag_corpus.v2.jsonl",
            "rootscope_v3/rag2/pack/rootscope_rag_corpus.v2.jsonl",
            "RAG_BM25_RUNTIME",
        ),
        (
            adventurex / "rootscope/tests/fixtures/pc_gate_pass_fixture.json",
            "evidence/pc_gate_receipt.json",
            "PC_GATE_RECEIPT",
        ),
        (
            source,
            "models/llm/fast/test-fast.gguf",
            "ROOTMIND_FAST_MODEL",
        ),
        (
            source,
            "models/llm/deep/test-deep.gguf",
            "ROOTMIND_DEEP_MODEL",
        ),
    ]
    entries = [
        {
            "source": path.relative_to(adventurex).as_posix(),
            "path": package_path,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mode": 420,
            "category": category,
        }
        for path, package_path, category in sources
    ]
    gate_sha = next(
        item["sha256"] for item in entries if item["category"] == "PC_GATE_RECEIPT"
    )
    contract_rows = [
        {
            "path": item["path"],
            "bytes": (adventurex / item["source"]).stat().st_size,
            "sha256": item["sha256"],
            "mode": item["mode"],
            "category": item["category"],
        }
        for item in sorted(entries, key=lambda value: value["path"])
    ]
    contract_root = hashlib.sha256(
        (
            json.dumps(
                contract_rows,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    candidate_id = f"rootscope_v3_test_fixture_20260724_{contract_root[:12]}"
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        inputs = temporary_path / "inputs.json"
        output = temporary_path / "output"
        inputs.write_text(
            json.dumps(
                {
                    "schema": "rootscope.v3.release-inputs.v1",
                    "test_fixture_only": True,
                    "candidate_id": candidate_id,
                    "registry_and_schema_root_sha256": (
                        "43882938b7bb3ef34b8febf51ac1a8bbc"
                        "92c8cc815e848b8b5c61d371768eaa3"
                    ),
                    "pc_gate_receipt_sha256": gate_sha,
                    "rag_default": "SQLITE_FTS5_BM25_V2",
                    "rag_dense_challenger_packaged": False,
                    "entries": entries,
                }
            ),
            encoding="utf-8",
        )
        built = subprocess.run(
            [
                sys.executable,
                str(build_script),
                "--adventurex",
                str(adventurex),
                "--inputs",
                str(inputs),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert built.returncode == 0, built.stderr
        archive = output / f"{candidate_id}.tar"
        extract = temporary_path / "extract"
        extract.mkdir()
        with tarfile.open(archive, "r") as handle:
            handle.extractall(extract, filter="data")
        verified = subprocess.run(
            [
                sys.executable,
                str(verify_script),
                "--release-root",
                str(extract / candidate_id),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert verified.returncode == 0, verified.stderr
        report = json.loads(verified.stdout)
        assert report["status"] == "PASS_TEST_FIXTURE_ONLY_NOT_DEPLOYABLE"
        assert report["files_verified"] == len(entries) + 1
        assert report["pump_touched"] is False

        mismatched_entries = [dict(item) for item in entries]
        wrong_source = (
            adventurex
            / "rootscope"
            / "app"
            / "runtime_v3"
            / "hbm_runtime_adapter.py"
        )
        wrong_record = next(
            item
            for item in mismatched_entries
            if item["category"] == "ROOTMIND_DEEP_MODEL"
        )
        wrong_record["source"] = wrong_source.relative_to(adventurex).as_posix()
        wrong_record["sha256"] = hashlib.sha256(wrong_source.read_bytes()).hexdigest()
        mismatched_rows = [
            {
                "path": item["path"],
                "bytes": (adventurex / item["source"]).stat().st_size,
                "sha256": item["sha256"],
                "mode": item["mode"],
                "category": item["category"],
            }
            for item in sorted(mismatched_entries, key=lambda value: value["path"])
        ]
        mismatched_root = hashlib.sha256(
            (
                json.dumps(
                    mismatched_rows,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        tampered = json.loads(inputs.read_text(encoding="utf-8"))
        tampered["candidate_id"] = (
            f"rootscope_v3_test_fixture_20260724_{mismatched_root[:12]}"
        )
        tampered["entries"] = mismatched_entries
        inputs.write_text(json.dumps(tampered), encoding="utf-8")
        rejected = subprocess.run(
            [
                sys.executable,
                str(build_script),
                "--adventurex",
                str(adventurex),
                "--inputs",
                str(inputs),
                "--output",
                str(temporary_path / "tampered-output"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert rejected.returncode != 0
        assert "deep RootMind model is not exactly bound to PC gate" in rejected.stderr
