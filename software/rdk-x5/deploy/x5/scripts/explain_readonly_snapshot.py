#!/usr/bin/env python3
"""Explicit stdout-only CLI for one RootScope loopback LLM explanation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from readonly_llm_preflight import preflight


CONFIG_SCHEMA = "rootscope.readonly_llm_runtime_config.v1"
CONFIG_STATUS = "INSTALLED_DISABLED_MANUAL_ACK_REQUIRED_NOT_X5_QUALIFIED"


def _load_runtime_config(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("runtime config must be an object")
    expected = {
        "schema",
        "status",
        "project_root",
        "python_executable",
        "release_manifest",
        "model_path",
        "model_sha256",
        "llama_server",
        "llama_server_sha256",
        "host",
        "port",
        "read_only",
        "default_enabled",
        "manual_start_only",
        "manual_acknowledged",
        "external_network_allowed",
        "tool_execution",
        "actuator_access",
        "execution_authority",
        "physical_authority",
    }
    if set(value) != expected:
        raise ValueError("runtime config keys mismatch")
    if value["schema"] != CONFIG_SCHEMA or value["status"] != CONFIG_STATUS:
        raise ValueError("unsupported runtime config")
    if (
        value["host"] != "127.0.0.1"
        or value["read_only"] is not True
        or value["default_enabled"] is not False
        or value["manual_start_only"] is not True
        or value["manual_acknowledged"] is not False
        or value["external_network_allowed"] is not False
        or value["tool_execution"] is not False
        or value["actuator_access"] is not False
        or value["execution_authority"] is not False
        or value["physical_authority"] is not False
    ):
        raise ValueError("runtime config violates the read-only authority boundary")
    return value


def explain(config_path: Path, snapshot_path: Path, question: str) -> Mapping[str, Any]:
    config = _load_runtime_config(config_path)
    preflight(
        manifest_path=Path(str(config["release_manifest"])),
        model_path=Path(str(config["model_path"])),
        llama_server=Path(str(config["llama_server"])),
        llama_server_sha256=str(config["llama_server_sha256"]),
        host=str(config["host"]),
        port=int(config["port"]),
        health=True,
    )
    project_root = Path(str(config["project_root"])).resolve(strict=True)
    if not (project_root / "app/llm/read_only_explainer.py").is_file():
        raise ValueError("project_root does not contain the read-only explainer")
    sys.path.insert(0, str(project_root))
    from app.llm.read_only_explainer import ExplanationConfig, explain_snapshot

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot must be one JSON object")
    return explain_snapshot(
        snapshot,
        ExplanationConfig(
            enabled=True,
            endpoint=f"http://127.0.0.1:{int(config['port'])}",
            model_sha256=str(config["model_sha256"]),
        ),
        question=question,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--snapshot-json", type=Path, required=True)
    parser.add_argument("--question", default="请解释当前 RootScope 证据、风险和不确定性。")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = explain(args.runtime_config, args.snapshot_json, args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["provenance"]["model_output_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
