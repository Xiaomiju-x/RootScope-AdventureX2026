#!/usr/bin/env python3
"""Fail-closed local checks for the optional RootScope llama.cpp service.

The default mode reads files only.  ``--health`` performs one direct GET to a
numeric 127.0.0.1 endpoint; redirects and non-loopback hosts are not supported.
It never starts a process, scans devices, or changes networking.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "rootscope.readonly_llm_release_manifest.v1"
MANIFEST_STATUS = "STAGED_READ_ONLY_LOCAL_LLM_NOT_X5_QUALIFIED"
MAX_HEALTH_BYTES = 16_384
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_release_manifest(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("release manifest must be an object")
    if value.get("schema") != MANIFEST_SCHEMA or value.get("status") != MANIFEST_STATUS:
        raise ValueError("unsupported read-only LLM release manifest")
    if value.get("artifact_staged") is not True:
        raise ValueError("release manifest does not attest an exact staged artifact")
    flags = value.get("formal_flags")
    if not isinstance(flags, Mapping) or not flags or any(item is not False for item in flags.values()):
        raise ValueError("all release formal flags must remain boolean false")
    runtime = value.get("runtime_contract")
    if not isinstance(runtime, Mapping) or (
        runtime.get("host") != "127.0.0.1"
        or runtime.get("default_enabled") is not False
        or runtime.get("manual_start_only") is not True
        or runtime.get("external_network_allowed") is not False
        or runtime.get("read_only") is not True
        or runtime.get("tool_execution") is not False
        or runtime.get("actuator_access") is not False
    ):
        raise ValueError("release runtime contract is not manual loopback-only/read-only")
    dependency = value.get("llama_cpp_dependency")
    if not isinstance(dependency, Mapping) or dependency.get("bundled") is not False:
        raise ValueError("llama.cpp must remain an explicit external dependency")
    return value


def validate_release_model(manifest_path: Path, model_path: Path) -> Mapping[str, Any]:
    manifest = load_release_manifest(manifest_path)
    artifact = manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("release artifact entry is missing")
    expected_sha = artifact.get("sha256")
    expected_size = artifact.get("size_bytes")
    if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha):
        raise ValueError("release artifact SHA-256 is invalid")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
        raise ValueError("release artifact size is invalid")
    if model_path.is_symlink():
        raise ValueError("installed GGUF must not be a symlink")
    model = model_path.resolve(strict=True)
    if not model.is_file():
        raise ValueError("installed GGUF must be one regular, non-symlink file")
    if model.stat().st_size != expected_size or sha256_file(model) != expected_sha:
        raise ValueError("installed GGUF does not match the release manifest")
    return manifest


def validate_external_llama_server(path: Path, expected_sha256: str) -> str:
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("llama-server expected SHA-256 must be lowercase hexadecimal")
    if path.is_symlink():
        raise ValueError("llama-server must not be a symlink")
    executable = path.resolve(strict=True)
    if not executable.is_file():
        raise ValueError("llama-server must be one regular, non-symlink external file")
    if os.name != "nt" and not os.access(executable, os.X_OK):
        raise ValueError("external llama-server is not executable")
    actual = sha256_file(executable)
    if actual != expected_sha256:
        raise ValueError("external llama-server SHA-256 mismatch")
    return actual


def health_probe(host: str, port: int, path: str = "/health") -> Mapping[str, Any]:
    if host != "127.0.0.1":
        raise ValueError("health endpoint host must be the numeric IPv4 loopback")
    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError("health endpoint port must be unprivileged")
    if path != "/health":
        raise ValueError("health path must be /health")
    connection = http.client.HTTPConnection(host, port, timeout=3.0)
    try:
        connection.request("GET", path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"llama-server health status must be 200, got {response.status}")
        raw = response.read(MAX_HEALTH_BYTES + 1)
    finally:
        connection.close()
    if len(raw) > MAX_HEALTH_BYTES:
        raise ValueError("llama-server health response exceeds the byte limit")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("llama-server health response must be a JSON object")
    return payload


def preflight(
    *,
    manifest_path: Path,
    model_path: Path,
    llama_server: Path,
    llama_server_sha256: str,
    host: str,
    port: int,
    health: bool,
) -> Mapping[str, Any]:
    manifest = validate_release_model(manifest_path, model_path)
    executable_sha = validate_external_llama_server(llama_server, llama_server_sha256)
    health_payload = health_probe(host, port) if health else None
    return {
        "schema": "rootscope.readonly_llm_preflight_receipt.v1",
        "status": "PASS_READ_ONLY_LOCAL_PREFLIGHT",
        "release_id": manifest["release_id"],
        "model_sha256": manifest["artifact"]["sha256"],
        "model_size_bytes": manifest["artifact"]["size_bytes"],
        "llama_server_sha256": executable_sha,
        "llama_cpp_bundled": False,
        "health_checked": health,
        "health_payload": health_payload,
        "host": host,
        "port": port,
        "external_network_touched": False,
        "service_started_by_preflight": False,
        "tool_execution": False,
        "actuator_access": False,
        "execution_authority": False,
        "physical_authority": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--llama-server-sha256", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9080)
    parser.add_argument("--health", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = preflight(
        manifest_path=args.manifest,
        model_path=args.model,
        llama_server=args.llama_server,
        llama_server_sha256=args.llama_server_sha256,
        host=args.host,
        port=args.port,
        health=args.health,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
