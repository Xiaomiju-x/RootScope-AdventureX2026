from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest


ROOTSCOPE = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOTSCOPE / "tools" / "x5_rootmind_cache_release_v3.py"
SPEC = importlib.util.spec_from_file_location("x5_rootmind_cache_release_v3", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def candidate(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "rootscope_v3_pc_ready_20260724_aaaaaaaaaaaa"
    model = root / "models" / "llm" / "fast" / "fast.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"GGUF" + bytes(range(64)) * 8)
    digest = _sha256(model)
    manifest = {
        "schema": helper.MANIFEST_SCHEMA,
        "candidate_id": root.name,
        "authority": dict(helper.CANDIDATE_ZERO_AUTHORITY),
        "files": [
            {
                "path": "models/llm/fast/fast.gguf",
                "category": "ROOTMIND_FAST_MODEL",
                "bytes": model.stat().st_size,
                "sha256": digest,
                "mode": 0o444,
            }
        ],
    }
    manifest_path = root / "candidate_manifest.json"
    _write_json(manifest_path, manifest)
    binding = helper.build_binding(
        release_root_raw=root,
        role="fast",
        model_raw=model,
    )
    binding_path = tmp_path / "binding.json"
    _write_json(binding_path, binding)
    return {
        "root": root,
        "model": model,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "binding": binding,
        "binding_path": binding_path,
    }


def _memory(cma: int = 200_000) -> dict[str, int]:
    return {
        "mem_available_kib": 2_000_000,
        "cma_free_kib": cma,
        "cached_kib": 500_000,
    }


def _release(
    candidate: dict[str, Any],
    *,
    processes: list[dict[str, Any]] | None = None,
    port_closed: bool = True,
    cma: int = 200_000,
    residents: list[int] | None = None,
    fadvise: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    resident_values = iter(residents or [8192, 0, 0])
    clock = {"seconds": 100.0}

    def monotonic() -> float:
        return clock["seconds"]

    def sleeper(seconds: float) -> None:
        clock["seconds"] += seconds

    return helper.execute_release(
        release_root_raw=candidate["root"],
        role="fast",
        binding_path=candidate["binding_path"],
        observe_seconds=0.25,
        process_scanner=lambda: list(processes or []),
        port_checker=lambda: port_closed,
        memory_sampler=lambda: _memory(cma),
        resident_sampler=lambda _fd, _size: next(resident_values),
        fadvise=fadvise or (lambda _fd: None),
        monotonic=monotonic,
        sleeper=sleeper,
    )


def test_bind_and_release_success_are_exact_and_zero_authority(
    candidate: dict[str, Any],
) -> None:
    binding = candidate["binding"]
    assert binding["schema"] == helper.BINDING_SCHEMA
    assert binding["status"] == "BOUND"
    assert binding["candidate"]["manifest_sha256"] == _sha256(
        candidate["manifest_path"]
    )
    assert binding["model"]["sha256"] == _sha256(candidate["model"])
    assert set(binding["model"]["stat_fingerprint"]) == {
        "device",
        "inode",
        "mode",
        "nlink",
        "uid",
        "gid",
        "size",
        "mtime_ns",
        "ctime_ns",
    }
    assert binding["authority"] == helper.ZERO_AUTHORITY

    receipt = _release(candidate)
    assert receipt["schema"] == helper.RECEIPT_SCHEMA
    assert receipt["status"] == "PASS"
    assert receipt["cache"] == {
        "method": "POSIX_FADV_DONTNEED",
        "exact_file_only": True,
        "global_drop_caches": False,
        "sync_called": False,
        "compact_memory_called": False,
        "fadvise_applied": True,
        "resident_bytes_before": 8192,
        "resident_bytes_after": 0,
        "resident_limit_bytes": 4096,
        "window_reached": True,
    }
    assert receipt["memory"]["after"]["cma_free_kib"] == 200_000
    assert receipt["memory"]["samples"][0]["gate_pass"] is True
    assert receipt["integrity"]["model_stat_unchanged_after"] is True
    assert receipt["authority"] == helper.ZERO_AUTHORITY


def test_bind_rejects_symlink_without_opening_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.gguf"
    target.write_bytes(b"GGUF")
    monkeypatch.delattr(helper.os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(
        helper.Path,
        "lstat",
        lambda _self: SimpleNamespace(st_mode=helper.stat.S_IFLNK),
    )
    with pytest.raises(helper.GateError, match="symlink forbidden") as caught:
        helper._open_nofollow_regular(target)
    assert caught.value.code == "SYMLINK_REJECTED"


def test_release_rejects_manifest_change(candidate: dict[str, Any]) -> None:
    candidate["manifest"]["build_date"] = "tampered"
    _write_json(candidate["manifest_path"], candidate["manifest"])
    with pytest.raises(helper.GateError) as caught:
        _release(candidate)
    assert caught.value.code == "MANIFEST_CHANGED"


def test_release_rejects_model_hash_change_even_with_rebound_stat(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprint = candidate["binding"]["model"]["stat_fingerprint"]
    monkeypatch.setattr(
        helper,
        "_hash_open_fd",
        lambda _fd: ("0" * 64, fingerprint),
    )
    with pytest.raises(helper.GateError) as caught:
        _release(candidate)
    assert caught.value.code == "MODEL_HASH_CHANGED"


def test_release_rejects_model_stat_change(candidate: dict[str, Any]) -> None:
    stat_result = candidate["model"].stat()
    os.utime(
        candidate["model"],
        ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000_000),
    )
    with pytest.raises(helper.GateError) as caught:
        _release(candidate)
    assert caught.value.code == "MODEL_STAT_CHANGED"


def test_release_fails_closed_on_fadvise_error(candidate: dict[str, Any]) -> None:
    def fail(_fd: int) -> None:
        raise helper.GateError("FADVISE_FAILED", "synthetic failure")

    with pytest.raises(helper.GateError) as caught:
        _release(candidate, fadvise=fail, residents=[8192])
    assert caught.value.code == "FADVISE_FAILED"


def test_release_fails_when_cma_or_resident_gate_is_low(
    candidate: dict[str, Any],
) -> None:
    with pytest.raises(helper.GateError) as caught:
        _release(candidate, cma=131_071, residents=[8192, 0, 0])
    assert caught.value.code == "CACHE_RESOURCE_GATE_FAILED"
    with pytest.raises(helper.GateError) as caught:
        _release(candidate, residents=[8192, 4097, 4097])
    assert caught.value.code == "CACHE_RESOURCE_GATE_FAILED"


def test_release_fails_if_any_observation_sample_misses_gate(
    candidate: dict[str, Any],
) -> None:
    with pytest.raises(helper.GateError) as caught:
        _release(candidate, residents=[8192, 0, 4097])
    assert caught.value.code == "CACHE_RESOURCE_GATE_FAILED"


@pytest.mark.parametrize(
    ("processes", "port_closed", "code"),
    [
        ([{"pid": 123, "cmdline": ["/opt/llama-server"]}], True, "LLAMA_SERVER_RESIDUAL"),
        ([], False, "PORT_STILL_OPEN"),
    ],
)
def test_release_rejects_residual_process_or_port(
    candidate: dict[str, Any],
    processes: list[dict[str, Any]],
    port_closed: bool,
    code: str,
) -> None:
    with pytest.raises(helper.GateError) as caught:
        _release(candidate, processes=processes, port_closed=port_closed)
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("exact_file_only", False),
        ("global_drop_caches", True),
        ("sync_called", True),
        ("compact_memory_called", True),
    ],
)
def test_receipt_rejects_any_global_policy_tamper(
    field: str, tampered: bool
) -> None:
    receipt = helper._empty_release_receipt("a" * 64)
    receipt["error"] = {"code": "SYNTHETIC", "message": "expected"}
    receipt["cache"][field] = tampered
    with pytest.raises(helper.GateError) as caught:
        helper._validate_release_receipt(receipt)
    assert caught.value.code == "GLOBAL_CACHE_POLICY_FORBIDDEN"


def test_failure_cli_creates_new_atomic_fail_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failure.json"
    monkeypatch.setattr(
        helper,
        "execute_release",
        lambda **_kwargs: (_ for _ in ()).throw(
            helper.GateError("PORT_STILL_OPEN", "synthetic")
        ),
    )
    args = argparse.Namespace(
        release_root="unused",
        role="fast",
        binding=str(tmp_path / "missing-binding.json"),
        output=str(output),
        observe_seconds=2.0,
    )
    assert helper._run_release(args) == 40
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert receipt["error"]["code"] == "PORT_STILL_OPEN"
    assert receipt["cache"]["exact_file_only"] is True
    assert receipt["authority"] == helper.ZERO_AUTHORITY
    assert helper._run_release(args) == 41


def test_implementation_contains_no_global_reclaim_calls() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    forbidden_calls: set[tuple[str | None, str]] = {
        ("os", "sync"),
        ("subprocess", "run"),
        ("subprocess", "call"),
        ("subprocess", "Popen"),
    }
    observed: set[tuple[str | None, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            observed.add((node.func.value.id, node.func.attr))
        elif isinstance(node.func, ast.Name):
            observed.add((None, node.func.id))
    assert not (observed & forbidden_calls)
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "/proc/sys/vm/" not in source
