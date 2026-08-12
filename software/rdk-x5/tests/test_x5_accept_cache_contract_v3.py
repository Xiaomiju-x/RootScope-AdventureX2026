from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Callable

import pytest


ADVENTUREX = Path(__file__).resolve().parents[2]
ACCEPT_SCRIPT = ADVENTUREX / "tools" / "release_v3" / "x5_accept_candidate_v3.sh"
SMOKE_SCRIPT = ADVENTUREX / "tools" / "release_v3" / "x5_rootmind_smoke_v3.sh"
CACHE_HELPER = ADVENTUREX / "rootscope" / "tools" / "x5_rootmind_cache_release_v3.py"
DEPLOY_SCRIPT = (
    ADVENTUREX / "tools" / "release_v3" / "deploy_rootscope_v3_to_x5.ps1"
)
STAGE_SCRIPT = ADVENTUREX / "tools" / "release_v3" / "x5_stage_candidate_v3.sh"

ROOTMIND_SCHEMA = "rootscope.v3.x5-rootmind-smoke.v3"
CACHE_SCHEMA = "rootscope.v3.rootmind-gguf-cache-release.v1"
FAST_STATUS = "PASS_X5_ROOTMIND_CHAT_TEMPLATE_SCHEMA_LOCKED_READ_ONLY"
DEEP_STATUS = (
    "PASS_X5_ROOTMIND_CHAT_TEMPLATE_EXPLICIT_GBNF_EXACT_READ_ONLY_"
    "SCHEMA_RUNTIME_INCOMPATIBLE"
)
CMA_MINIMUM_KIB = 131_072
RESIDENT_LIMIT_BYTES = 4_096
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64

EXPECTED_ACCEPTANCE_RECEIPT_NAMES = {
    "00_runtime_bootstrap.json",
    "01_release_verify.json",
    "02_cpu_bm25.json",
    "03_hrt_oracle.json",
    "04_hbm_execution.json",
    "05_native_libdnn.json",
    "06_rootmind_fast_receipt.json",
    "07_rootmind_deep_receipt.json",
    "rootmind_fast_model_binding.json",
    "rootmind_fast_model_page_cache_release.json",
    "rootmind_deep_model_binding.json",
    "rootmind_deep_model_page_cache_release.json",
    "rootmind_precondition_deep_model_binding.json",
    "rootmind_precondition_deep_model_page_cache_release.json",
    "rootmind_precondition_fast_model_binding.json",
    "rootmind_precondition_fast_model_page_cache_release.json",
}

CACHE_ZERO_AUTHORITY = {
    "execution_authority": False,
    "physical_authority": False,
    "external_network": False,
    "service_started": False,
    "serial_opened": False,
    "serial_write": False,
    "gpio_touched": False,
    "pump_command": False,
    "state_machine_write": False,
    "model_modified": False,
}


def _acceptance_python() -> str:
    """Return the acceptance-summary Python heredoc that owns validate_rootmind."""

    source = ACCEPT_SCRIPT.read_text(encoding="utf-8")
    blocks = re.findall(
        r"<<'PY'\r?\n(?P<body>.*?)\r?\nPY(?=\r?\n|$)",
        source,
        flags=re.DOTALL,
    )
    matches = [block for block in blocks if "def validate_rootmind(" in block]
    assert len(matches) == 1, "acceptance must have one validate_rootmind heredoc"
    return matches[0]


def _stage_acceptance_python() -> str:
    """Return the activation-stage heredoc that validates acceptance hashes."""

    source = STAGE_SCRIPT.read_text(encoding="utf-8")
    blocks = re.findall(
        r"<<'PY'\r?\n(?P<body>.*?)\r?\nPY(?=\r?\n|$)",
        source,
        flags=re.DOTALL,
    )
    matches = [
        block
        for block in blocks
        if "acceptance receipt hash coverage mismatch" in block
    ]
    assert len(matches) == 1, "stage must have one acceptance-hash validator"
    return matches[0]


def _python_receipt_names(source: str) -> set[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "receipt_names"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError("receipt_names literal is missing")


def _validation_namespace(release_root: Path) -> dict[str, Any]:
    """Compile production validation helpers without running acceptance top level."""

    tree = ast.parse(_acceptance_python(), filename=str(ACCEPT_SCRIPT))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            selected.append(node)
            continue
        # Preserve literal module constants if the production validator introduces
        # one, but never execute receipt loading or summary publication here.
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                continue
            try:
                ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
            selected.append(node)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace: dict[str, Any] = {}
    exec(compile(module, str(ACCEPT_SCRIPT), "exec"), namespace)
    namespace["release_root"] = release_root
    namespace["evidence"] = release_root.parent / "evidence"
    return namespace


def _valid_cache_release(
    release_root: Path,
    *,
    role: str = "fast",
    model_sha256: str = SHA_A,
    model_bytes: int = 1_048_576,
) -> dict[str, Any]:
    evidence = release_root.parent / "evidence"
    external = evidence / f"rootmind_{role}_model_page_cache_release.json"
    if external.is_file():
        return json.loads(external.read_text(encoding="utf-8"))
    relative = f"models/llm/{role}/fixture-{role}.gguf"
    model_path = release_root / relative
    category = "ROOTMIND_FAST_MODEL" if role == "fast" else "ROOTMIND_DEEP_MODEL"
    manifest_path = release_root / "candidate_manifest.json"
    manifest_sha = (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if manifest_path.is_file()
        else SHA_C
    )
    stat_fingerprint = {
        "device": 11,
        "inode": 22,
        "mode": stat.S_IFREG | 0o444,
        "nlink": 1,
        "uid": 1000,
        "gid": 1000,
        "size": model_bytes,
        "mtime_ns": 1_727_000_000_000_000_000,
        "ctime_ns": 1_727_000_000_000_000_001,
    }
    memory_sample = {
        "elapsed_ms": 0,
        "mem_available_kib": 2_000_000,
        "cma_free_kib": 315_000,
        "cached_kib": 1_000_000,
        "resident_bytes": 0,
        "gate_pass": True,
    }
    return {
        "schema": CACHE_SCHEMA,
        "status": "PASS",
        "created_utc": "2026-07-24T13:00:00.000000Z",
        "binding_sha256": SHA_D,
        "candidate": {
            "id": release_root.name,
            "release_root": str(release_root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
        },
        "role": role,
        "model": {
            "path": str(model_path),
            "relative_path": relative,
            "category": category,
            "bytes": model_bytes,
            "sha256": model_sha256,
            "stat_fingerprint": stat_fingerprint,
        },
        "integrity": {
            "binding_valid": True,
            "release_root_unchanged": True,
            "manifest_path_unchanged": True,
            "manifest_sha256_unchanged": True,
            "manifest_record_unchanged": True,
            "model_path_unchanged": True,
            "model_stat_unchanged": True,
            "model_sha256_verified": True,
            "model_stat_unchanged_after": True,
            "model_modified": False,
        },
        "preconditions": {
            "llama_server_processes": [],
            "no_llama_server": True,
            "endpoint": "127.0.0.1:9080",
            "port_closed": True,
        },
        "cache": {
            "method": "POSIX_FADV_DONTNEED",
            "fadvise_applied": True,
            "exact_file_only": True,
            "global_drop_caches": False,
            "sync_called": False,
            "compact_memory_called": False,
            "resident_bytes_before": model_bytes,
            "resident_bytes_after": 0,
            "resident_limit_bytes": RESIDENT_LIMIT_BYTES,
            "window_reached": True,
        },
        "memory": {
            "before": {
                "mem_available_kib": 2_000_000,
                "cma_free_kib": 2_000,
                "cached_kib": 2_000_000,
            },
            "after": {
                "mem_available_kib": 2_000_000,
                "cma_free_kib": 315_000,
                "cached_kib": 1_000_000,
            },
            "samples": [
                memory_sample,
                {**memory_sample, "elapsed_ms": 2_000},
            ],
            "observe_seconds": 2.0,
            "cma_free_minimum_kib": CMA_MINIMUM_KIB,
            "window_reached": True,
        },
        "authority": deepcopy(CACHE_ZERO_AUTHORITY),
        "error": None,
    }


def _valid_binding(
    release_root: Path,
    *,
    role: str = "fast",
    model_sha256: str = SHA_A,
    model_bytes: int = 1_048_576,
) -> dict[str, Any]:
    cache = _valid_cache_release(
        release_root,
        role=role,
        model_sha256=model_sha256,
        model_bytes=model_bytes,
    )
    manifest = json.loads(
        (release_root / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "schema": "rootscope.v3.rootmind-gguf-cache-binding.v1",
        "status": "BOUND",
        "created_utc": "2026-07-24T12:59:00.000000Z",
        "candidate": deepcopy(cache["candidate"]),
        "role": role,
        "model": deepcopy(cache["model"]),
        "integrity": {
            "manifest_record_count": len(manifest["files"]),
            "unique_role_gguf": True,
            "content_sha256_verified": True,
            "regular_file": True,
            "nofollow_open": True,
        },
        "authority": deepcopy(CACHE_ZERO_AUTHORITY),
    }


def _valid_rootmind_receipt(
    release_root: Path,
    *,
    role: str = "fast",
) -> dict[str, Any]:
    model_sha = SHA_A if role == "fast" else SHA_B
    cache_release = _valid_cache_release(
        release_root,
        role=role,
        model_sha256=model_sha,
    )
    deep = role == "deep"
    return {
        "schema": ROOTMIND_SCHEMA,
        "status": DEEP_STATUS if deep else FAST_STATUS,
        "role": role,
        "candidate": {
            "candidate_id": release_root.name,
            "release_root": str(release_root),
            "manifest_sha256": cache_release["candidate"]["manifest_sha256"],
        },
        "runtime": {
            "model_path": cache_release["model"]["path"],
            "model_relative_path": cache_release["model"]["relative_path"],
            "model_sha256": model_sha,
            "model_bytes": cache_release["model"]["bytes"],
            "server_sha256": SHA_B,
        },
        "transport": {
            "loopback_only": True,
            "external_network_touched": False,
        },
        "contract": {
            "exact_output": {"authority": False, "status": "READ_ONLY"},
            "tool_interface_supplied": False,
            "tool_calls_observed": False,
            "single_explicit_gbnf_retry_used": deep,
            "schema_primary_passed": not deep,
            "json_schema_strict": not deep,
            "explicit_gbnf_strict": deep,
            "compatibility_downgrade_reason": (
                "B9637_QWEN3_ASSISTANT_PREFIX_JSON_SCHEMA_GRAMMAR_SAMPLER_INIT"
                if deep
                else None
            ),
        },
        "shutdown": {
            "process_stopped": True,
            "port_closed_after_stop": True,
            "forced_kill": False,
            "model_page_cache_release": cache_release,
        },
        "execution_authority": False,
        "physical_authority": False,
        "service_started": False,
        "serial_opened": False,
        "serial_write": False,
        "gpio_touched": False,
        "pump_command": False,
        "state_machine_write": False,
        "physical_completion": False,
    }


def _validator(tmp_path: Path) -> tuple[Callable[[dict[str, Any], str], Any], Path]:
    release_root = tmp_path / "candidate-fixture"
    release_root.mkdir()
    manifest = {
        "schema": "rootscope.v3.candidate-manifest.v1",
        "candidate_id": release_root.name,
        # Deliberately use more than one record: production manifests contain
        # hundreds, and acceptance must not confuse total records with role hits.
        "files": [
            {
                "path": "models/llm/fast/fixture-fast.gguf",
                "bytes": 1_048_576,
                "sha256": SHA_A,
                "category": "ROOTMIND_FAST_MODEL",
            },
            {
                "path": "models/llm/deep/fixture-deep.gguf",
                "bytes": 1_048_576,
                "sha256": SHA_B,
                "category": "ROOTMIND_DEEP_MODEL",
            },
            {"path": "rootscope/app/a.py", "bytes": 2, "sha256": SHA_B},
            {"path": "rootscope/app/b.py", "bytes": 3, "sha256": SHA_C},
        ],
    }
    (release_root / "candidate_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for role, model_sha in (("fast", SHA_A), ("deep", SHA_B)):
        binding = _valid_binding(
            release_root,
            role=role,
            model_sha256=model_sha,
        )
        binding_path = evidence / f"rootmind_{role}_model_binding.json"
        binding_path.write_text(
            json.dumps(binding, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        cache_release = _valid_cache_release(
            release_root,
            role=role,
            model_sha256=model_sha,
        )
        cache_release["binding_sha256"] = hashlib.sha256(
            binding_path.read_bytes()
        ).hexdigest()
        (evidence / f"rootmind_{role}_model_page_cache_release.json").write_text(
            json.dumps(cache_release, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    namespace = _validation_namespace(release_root)
    return namespace["validate_rootmind"], release_root


def _set_path(value: dict[str, Any], dotted: str, replacement: Any) -> None:
    target: Any = value
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = replacement


def _assert_rejected(
    validator: Callable[[dict[str, Any], str], Any],
    receipt: dict[str, Any],
    *,
    role: str = "fast",
) -> None:
    # Keep the separately stored release receipt byte-for-byte equivalent to the
    # nested receipt.  Otherwise every mutation would be rejected by only the
    # outer equality gate and would not exercise the field-specific validator.
    release_root = Path(receipt["candidate"]["release_root"])
    external = (
        release_root.parent
        / "evidence"
        / f"rootmind_{role}_model_page_cache_release.json"
    )
    external.write_text(
        json.dumps(
            receipt["shutdown"]["model_page_cache_release"],
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        validator(receipt, role)


def test_acceptance_statically_binds_the_cache_release_contract() -> None:
    source = ACCEPT_SCRIPT.read_text(encoding="utf-8")
    helper = CACHE_HELPER.read_text(encoding="utf-8")
    smoke = SMOKE_SCRIPT.read_text(encoding="utf-8")

    required_acceptance_literals = {
        "model_page_cache_release",
        CACHE_SCHEMA,
        "POSIX_FADV_DONTNEED",
        "binding_sha256",
        "model_sha256_verified",
        "model_stat_unchanged_after",
        "model_modified",
        "no_llama_server",
        "port_closed",
        "resident_bytes_after",
        "resident_limit_bytes",
        "cma_free_minimum_kib",
        "06_rootmind_fast_receipt.json",
        "07_rootmind_deep_receipt.json",
        "receipts_sha256",
    }
    missing = sorted(item for item in required_acceptance_literals if item not in source)
    assert not missing, f"acceptance omits cache-release contract literals: {missing}"
    assert str(CMA_MINIMUM_KIB) in source or "131_072" in source
    assert str(RESIDENT_LIMIT_BYTES) in source or "4_096" in source

    # The production helper, not acceptance, owns the exact-file syscall.  Static
    # tests ensure that no global cache mutation was smuggled into this chain.
    assert "os.O_NOFOLLOW" in helper
    assert "stat.S_ISREG" in helper
    assert '"posix_fadvise"' in helper
    assert '"POSIX_FADV_DONTNEED"' in helper
    forbidden_global_mutations = (
        "/proc/sys/vm/drop_caches",
        "echo 3 >",
        "sysctl -w vm.drop_caches",
    )
    combined = "\n".join((helper, smoke, source))
    assert not any(token in combined for token in forbidden_global_mutations)


@pytest.mark.parametrize(
    ("role", "expected_status", "expected_model_sha"),
    [
        ("fast", FAST_STATUS, SHA_A),
        ("deep", DEEP_STATUS, SHA_B),
    ],
)
def test_valid_cache_release_fixture_is_accepted_and_summarized(
    tmp_path: Path,
    role: str,
    expected_status: str,
    expected_model_sha: str,
) -> None:
    validator, release_root = _validator(tmp_path)
    result = validator(_valid_rootmind_receipt(release_root, role=role), role)
    assert result["status"] == expected_status
    assert result["model_sha256"] == expected_model_sha
    cache_summary = result.get("model_page_cache_release")
    assert isinstance(cache_summary, dict)
    assert cache_summary["schema"] == CACHE_SCHEMA
    assert cache_summary["status"] == "PASS"
    binding_path = (
        release_root.parent / "evidence" / f"rootmind_{role}_model_binding.json"
    )
    assert cache_summary["binding_sha256"] == hashlib.sha256(
        binding_path.read_bytes()
    ).hexdigest()
    assert cache_summary["resident_bytes_after"] <= RESIDENT_LIMIT_BYTES
    assert cache_summary["cma_free_after_kib"] >= CMA_MINIMUM_KIB
    assert cache_summary["cma_free_minimum_kib"] == CMA_MINIMUM_KIB


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("shutdown.model_page_cache_release.schema", "wrong.schema"),
        ("shutdown.model_page_cache_release.status", "FAIL"),
        ("shutdown.model_page_cache_release.role", "deep"),
        ("shutdown.model_page_cache_release.binding_sha256", "not-a-sha"),
        ("shutdown.model_page_cache_release.candidate.id", "other-candidate"),
        ("shutdown.model_page_cache_release.candidate.release_root", "/other"),
        (
            "shutdown.model_page_cache_release.candidate.manifest_path",
            "/other/candidate_manifest.json",
        ),
        ("shutdown.model_page_cache_release.candidate.manifest_sha256", SHA_D),
        ("shutdown.model_page_cache_release.model.path", "/other/model.gguf"),
        (
            "shutdown.model_page_cache_release.model.relative_path",
            "models/llm/deep/fixture-deep.gguf",
        ),
        (
            "shutdown.model_page_cache_release.model.category",
            "ROOTMIND_DEEP_MODEL",
        ),
        ("shutdown.model_page_cache_release.model.sha256", SHA_B),
        ("shutdown.model_page_cache_release.model.bytes", 1_048_577),
        (
            "shutdown.model_page_cache_release.model.stat_fingerprint.mode",
            stat.S_IFDIR | 0o555,
        ),
    ],
)
def test_identity_and_candidate_binding_mutations_fail_closed(
    tmp_path: Path,
    path: str,
    replacement: Any,
) -> None:
    validator, release_root = _validator(tmp_path)
    receipt = _valid_rootmind_receipt(release_root)
    _set_path(receipt, path, replacement)
    _assert_rejected(validator, receipt)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("shutdown.model_page_cache_release.integrity.binding_valid", False),
        (
            "shutdown.model_page_cache_release.integrity.release_root_unchanged",
            False,
        ),
        (
            "shutdown.model_page_cache_release.integrity.manifest_path_unchanged",
            False,
        ),
        (
            "shutdown.model_page_cache_release.integrity.manifest_sha256_unchanged",
            False,
        ),
        (
            "shutdown.model_page_cache_release.integrity.manifest_record_unchanged",
            False,
        ),
        (
            "shutdown.model_page_cache_release.integrity.model_path_unchanged",
            False,
        ),
        (
            "shutdown.model_page_cache_release.integrity.model_stat_unchanged",
            False,
        ),
        (
            "shutdown.model_page_cache_release.integrity.model_sha256_verified",
            False,
        ),
        (
            "shutdown.model_page_cache_release.integrity.model_stat_unchanged_after",
            False,
        ),
        ("shutdown.model_page_cache_release.integrity.model_modified", True),
        (
            "shutdown.model_page_cache_release.preconditions.llama_server_processes",
            [{"pid": 123, "cmdline": ["llama-server"]}],
        ),
        (
            "shutdown.model_page_cache_release.preconditions.no_llama_server",
            False,
        ),
        ("shutdown.model_page_cache_release.preconditions.port_closed", False),
        ("shutdown.model_page_cache_release.cache.method", "GLOBAL_DROP_CACHES"),
        ("shutdown.model_page_cache_release.cache.fadvise_applied", False),
        ("shutdown.model_page_cache_release.cache.exact_file_only", False),
        ("shutdown.model_page_cache_release.cache.global_drop_caches", True),
        ("shutdown.model_page_cache_release.cache.sync_called", True),
        ("shutdown.model_page_cache_release.cache.compact_memory_called", True),
        (
            "shutdown.model_page_cache_release.cache.resident_bytes_after",
            RESIDENT_LIMIT_BYTES + 1,
        ),
        (
            "shutdown.model_page_cache_release.cache.resident_limit_bytes",
            RESIDENT_LIMIT_BYTES + 1,
        ),
        ("shutdown.model_page_cache_release.cache.window_reached", False),
        ("shutdown.model_page_cache_release.memory.window_reached", False),
        ("shutdown.model_page_cache_release.error", {"code": "FAIL"}),
    ],
)
def test_integrity_process_port_and_exact_file_mutations_fail_closed(
    tmp_path: Path,
    path: str,
    replacement: Any,
) -> None:
    validator, release_root = _validator(tmp_path)
    receipt = _valid_rootmind_receipt(release_root)
    _set_path(receipt, path, replacement)
    _assert_rejected(validator, receipt)


def test_cma_after_and_every_sample_must_meet_the_frozen_minimum(
    tmp_path: Path,
) -> None:
    validator, release_root = _validator(tmp_path)

    low_after = _valid_rootmind_receipt(release_root)
    low_after["shutdown"]["model_page_cache_release"]["memory"]["after"][
        "cma_free_kib"
    ] = CMA_MINIMUM_KIB - 1
    _assert_rejected(validator, low_after)

    low_sample = _valid_rootmind_receipt(release_root)
    low_sample["shutdown"]["model_page_cache_release"]["memory"]["samples"][0][
        "cma_free_kib"
    ] = CMA_MINIMUM_KIB - 1
    _assert_rejected(validator, low_sample)

    weak_threshold = _valid_rootmind_receipt(release_root)
    weak_threshold["shutdown"]["model_page_cache_release"]["memory"][
        "cma_free_minimum_kib"
    ] = CMA_MINIMUM_KIB - 1
    _assert_rejected(validator, weak_threshold)


@pytest.mark.parametrize("authority_key", sorted(CACHE_ZERO_AUTHORITY))
def test_each_cache_release_authority_bit_is_fail_closed(
    tmp_path: Path,
    authority_key: str,
) -> None:
    validator, release_root = _validator(tmp_path)
    receipt = _valid_rootmind_receipt(release_root)
    receipt["shutdown"]["model_page_cache_release"]["authority"][authority_key] = True
    _assert_rejected(validator, receipt)


@pytest.mark.parametrize(
    ("binding_path", "replacement"),
    [
        ("integrity.manifest_record_count", 5),
        ("integrity.unique_role_gguf", False),
        ("integrity.content_sha256_verified", False),
        ("integrity.regular_file", False),
        ("integrity.nofollow_open", False),
    ],
)
def test_external_binding_integrity_is_fail_closed(
    tmp_path: Path,
    binding_path: str,
    replacement: Any,
) -> None:
    validator, release_root = _validator(tmp_path)
    receipt = _valid_rootmind_receipt(release_root)
    evidence = release_root.parent / "evidence"
    external_binding = evidence / "rootmind_fast_model_binding.json"
    binding = json.loads(external_binding.read_text(encoding="utf-8"))
    _set_path(binding, binding_path, replacement)
    external_binding.write_text(
        json.dumps(binding, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    release = receipt["shutdown"]["model_page_cache_release"]
    release["binding_sha256"] = hashlib.sha256(
        external_binding.read_bytes()
    ).hexdigest()
    (evidence / "rootmind_fast_model_page_cache_release.json").write_text(
        json.dumps(release, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _assert_rejected(validator, receipt)


@pytest.mark.parametrize("authority_key", sorted(CACHE_ZERO_AUTHORITY))
def test_each_external_binding_authority_bit_is_fail_closed(
    tmp_path: Path,
    authority_key: str,
) -> None:
    validator, release_root = _validator(tmp_path)
    receipt = _valid_rootmind_receipt(release_root)
    evidence = release_root.parent / "evidence"
    external_binding = evidence / "rootmind_fast_model_binding.json"
    binding = json.loads(external_binding.read_text(encoding="utf-8"))
    binding["authority"][authority_key] = True
    external_binding.write_text(
        json.dumps(binding, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    release = receipt["shutdown"]["model_page_cache_release"]
    release["binding_sha256"] = hashlib.sha256(
        external_binding.read_bytes()
    ).hexdigest()
    (evidence / "rootmind_fast_model_page_cache_release.json").write_text(
        json.dumps(release, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _assert_rejected(validator, receipt)


def test_cache_release_binding_sha_is_checked_against_external_bytes(
    tmp_path: Path,
) -> None:
    validator, release_root = _validator(tmp_path)
    receipt = _valid_rootmind_receipt(release_root)
    binding_path = (
        release_root.parent / "evidence" / "rootmind_fast_model_binding.json"
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["created_utc"] = "2026-07-24T13:01:00.000000Z"
    binding_path.write_text(
        json.dumps(binding, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Deliberately leave release.binding_sha256 at the digest of the prior bytes.
    _assert_rejected(validator, receipt)


def test_rootmind_receipt_hashes_are_part_of_acceptance_hash_chain() -> None:
    source = _acceptance_python()
    receipt_names = _python_receipt_names(source)
    assert receipt_names == EXPECTED_ACCEPTANCE_RECEIPT_NAMES
    assert "06_rootmind_fast_receipt.json" in receipt_names
    assert "07_rootmind_deep_receipt.json" in receipt_names
    assert "rootmind_fast_model_binding.json" in receipt_names
    assert "rootmind_fast_model_page_cache_release.json" in receipt_names
    assert "rootmind_deep_model_binding.json" in receipt_names
    assert "rootmind_deep_model_page_cache_release.json" in receipt_names
    assert "rootmind_precondition_deep_model_binding.json" in receipt_names
    assert (
        "rootmind_precondition_deep_model_page_cache_release.json"
        in receipt_names
    )
    assert "rootmind_precondition_fast_model_binding.json" in receipt_names
    assert (
        "rootmind_precondition_fast_model_page_cache_release.json"
        in receipt_names
    )
    assert "receipts_sha256" in source
    assert "sha256_file(evidence / name)" in source
    # The nested cache-release object must survive into the summarized RootMind
    # result; otherwise the acceptance hash chain is not inspectable downstream.
    assert '"model_page_cache_release":' in source


def test_accept_stage_and_deployer_use_the_same_exact_receipt_hash_contract() -> None:
    accept_names = _python_receipt_names(_acceptance_python())
    stage_names = _python_receipt_names(_stage_acceptance_python())
    deploy_source = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r"\$ExpectedReceiptNames\s*=\s*@\((?P<body>.*?)\r?\n\s*\)",
        deploy_source,
        flags=re.DOTALL,
    )
    assert match is not None, "deployer ExpectedReceiptNames literal is missing"
    deploy_names = set(re.findall(r'"([^"]+\.json)"', match.group("body")))

    assert accept_names == EXPECTED_ACCEPTANCE_RECEIPT_NAMES
    assert stage_names == EXPECTED_ACCEPTANCE_RECEIPT_NAMES
    assert deploy_names == EXPECTED_ACCEPTANCE_RECEIPT_NAMES


def test_deployer_requires_cache_release_receipts_and_summary_gates() -> None:
    source = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    for name in (
        "rootmind_fast_model_binding.json",
        "rootmind_fast_model_page_cache_release.json",
        "rootmind_deep_model_binding.json",
        "rootmind_deep_model_page_cache_release.json",
        "rootmind_precondition_deep_model_binding.json",
        "rootmind_precondition_deep_model_page_cache_release.json",
        "rootmind_precondition_fast_model_binding.json",
        "rootmind_precondition_fast_model_page_cache_release.json",
    ):
        assert f'"{name}"' in source
    for token in (
        "rootscope.v3.rootmind-gguf-cache-release.v1",
        "POSIX_FADV_DONTNEED",
        "exact_file_only",
        "global_drop_caches",
        "resident_bytes_after",
        "cma_free_after_kib",
        "binding_sha256",
        "receipt_sha256",
        "rootmind_cache_precondition",
    ):
        assert token in source


def test_acceptance_preconditions_deep_then_fast_with_exact_helper() -> None:
    source = ACCEPT_SCRIPT.read_text(encoding="utf-8")
    loop = "for role in deep fast; do"
    assert loop in source
    assert source.index(loop) < source.index(
        'bash "${RELEASE_ROOT}/tools/release_v3/x5_rootmind_smoke_v3.sh"'
    )
    assert '"${ROOTSCOPE_CPU_PYTHON}" -I "${ROOTMIND_CACHE_HELPER}" bind' in source
    assert '"${ROOTSCOPE_CPU_PYTHON}" -I "${ROOTMIND_CACHE_HELPER}" release' in source
    assert "--observe-seconds 2" in source
    assert '"rootmind_cache_precondition": cache_precondition' in source
