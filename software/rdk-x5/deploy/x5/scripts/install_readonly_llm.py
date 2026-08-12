#!/usr/bin/env python3
"""Install a staged RootScope GGUF for one *user* without activating it.

The caller must provide a separately obtained, hash-frozen X5 llama-server.
This installer copies no executable, creates no activation gate, invokes no
systemctl command, and starts no process.  The generated user service therefore
remains manual and disabled after installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from readonly_llm_preflight import (
    load_release_manifest,
    sha256_file,
    validate_external_llama_server,
    validate_release_model,
)


def _safe_absolute(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_absolute() or any(character.isspace() for character in str(resolved)):
        raise ValueError(f"{name} must be an absolute path without whitespace")
    forbidden = ['"', "'", "\n", "\r"]
    if os.name != "nt":
        forbidden.append("\\")
    if any(character in str(resolved) for character in forbidden):
        raise ValueError(f"{name} contains unsafe service/environment characters")
    return resolved


def _atomic_write(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(payload)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _copy_exact(source: Path, destination: Path, expected_sha: str, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or destination.stat().st_size != expected_size:
            raise ValueError("existing installed GGUF is not the frozen artifact")
        if sha256_file(destination) != expected_sha:
            raise ValueError("existing installed GGUF hash mismatch; refusing overwrite")
        return
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != expected_size or sha256_file(temporary) != expected_sha:
            raise ValueError("installed GGUF copy verification failed")
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def install(
    *,
    release_dir: Path,
    project_root: Path,
    python_executable: Path,
    llama_server: Path,
    llama_server_sha256: str,
    prefix: Path,
    config_dir: Path,
    systemd_user_dir: Path,
    template_path: Path,
) -> Mapping[str, Any]:
    release = _safe_absolute(release_dir, "release_dir")
    project = _safe_absolute(project_root, "project_root")
    python = _safe_absolute(python_executable, "python_executable")
    external_server = _safe_absolute(llama_server, "llama_server")
    install_root = _safe_absolute(prefix, "prefix")
    config_root = _safe_absolute(config_dir, "config_dir")
    user_units = _safe_absolute(systemd_user_dir, "systemd_user_dir")
    template = template_path.resolve(strict=True)

    manifest_source = release / "release_manifest.json"
    manifest = load_release_manifest(manifest_source)
    artifact = manifest["artifact"]
    source_model = release / str(artifact["filename"])
    validate_release_model(manifest_source, source_model)
    validate_external_llama_server(external_server, llama_server_sha256)
    if not project.is_dir() or not (project / "app/llm/read_only_explainer.py").is_file():
        raise ValueError("project_root does not contain the RootScope read-only explainer")
    if not python.is_file() or (os.name != "nt" and not os.access(python, os.X_OK)):
        raise ValueError("python_executable is missing or not executable")

    installed_model = install_root / "models" / str(artifact["filename"])
    installed_manifest = install_root / "release_manifest.json"
    _copy_exact(source_model, installed_model, str(artifact["sha256"]), int(artifact["size_bytes"]))
    _atomic_write(installed_manifest, manifest_source.read_bytes(), mode=0o400)
    if sha256_file(installed_manifest) != sha256_file(manifest_source):
        raise ValueError("installed manifest verification failed")

    env_path = config_root / "rootscope-llm.env"
    gate_path = config_root / "enable-readonly-llm"
    runtime_config_path = config_root / "rootscope-llm-runtime.json"
    start_script = project / "deploy/x5/scripts/start_readonly_llm.sh"
    preflight_script = project / "deploy/x5/scripts/readonly_llm_preflight.py"
    for required in (start_script, preflight_script):
        if not required.is_file():
            raise ValueError(f"required deployment script missing: {required.name}")

    env_values = {
        "ROOTSCOPE_PROJECT_ROOT": str(project),
        "ROOTSCOPE_PYTHON": str(python),
        "ROOTSCOPE_LLM_MANIFEST": str(installed_manifest),
        "ROOTSCOPE_LLM_MODEL": str(installed_model),
        "ROOTSCOPE_LLM_MODEL_SHA256": str(artifact["sha256"]),
        "ROOTSCOPE_LLAMA_SERVER": str(external_server),
        "ROOTSCOPE_LLAMA_SERVER_SHA256": llama_server_sha256,
        "ROOTSCOPE_LLM_GATE_FILE": str(gate_path),
        "ROOTSCOPE_LLM_HOST": "127.0.0.1",
        "ROOTSCOPE_LLM_PORT": "9080",
        "ROOTSCOPE_LLM_THREADS": "2",
        "ROOTSCOPE_LLM_CONTEXT": "2048",
        "ROOTSCOPE_LLM_READ_ONLY": "true",
        "ROOTSCOPE_LLM_EXTERNAL_NETWORK": "false",
        "ROOTSCOPE_LLM_TOOL_EXECUTION": "false",
        "ROOTSCOPE_LLM_ACTUATOR_ACCESS": "false",
        "ROOTSCOPE_LLM_MANUAL_ACK": "NOT_ACKNOWLEDGED",
    }
    env_text = "\n".join(f"{name}={value}" for name, value in env_values.items()) + "\n"
    _atomic_write(env_path, env_text.encode("utf-8"), mode=0o600)

    runtime_config = {
        "schema": "rootscope.readonly_llm_runtime_config.v1",
        "status": "INSTALLED_DISABLED_MANUAL_ACK_REQUIRED_NOT_X5_QUALIFIED",
        "project_root": str(project),
        "python_executable": str(python),
        "release_manifest": str(installed_manifest),
        "model_path": str(installed_model),
        "model_sha256": artifact["sha256"],
        "llama_server": str(external_server),
        "llama_server_sha256": llama_server_sha256,
        "host": "127.0.0.1",
        "port": 9080,
        "read_only": True,
        "default_enabled": False,
        "manual_start_only": True,
        "manual_acknowledged": False,
        "external_network_allowed": False,
        "tool_execution": False,
        "actuator_access": False,
        "execution_authority": False,
        "physical_authority": False,
    }
    _atomic_write(
        runtime_config_path,
        (json.dumps(runtime_config, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        mode=0o600,
    )

    unit_template = template.read_text(encoding="utf-8")
    replacements = {
        "@PROJECT_ROOT@": str(project),
        "@ENV_FILE@": str(env_path),
        "@GATE_FILE@": str(gate_path),
    }
    for marker, replacement in replacements.items():
        unit_template = unit_template.replace(marker, replacement)
    if "@" in unit_template or "User=" in unit_template or "Group=" in unit_template:
        raise ValueError("rendered unit contains an unresolved marker or hard-coded account")
    unit_path = user_units / "rootscope-llm-readonly.service"
    _atomic_write(unit_path, unit_template.encode("utf-8"), mode=0o600)

    receipt = {
        "schema": "rootscope.readonly_llm_install_receipt.v1",
        "status": "INSTALLED_DISABLED_MANUAL_ACK_REQUIRED_NOT_X5_QUALIFIED",
        "artifact_installed": True,
        "artifact_sha256": artifact["sha256"],
        "artifact_size_bytes": artifact["size_bytes"],
        "external_llama_server_sha256": llama_server_sha256,
        "llama_cpp_bundled": False,
        "env_path": str(env_path),
        "runtime_config_path": str(runtime_config_path),
        "user_unit_path": str(unit_path),
        "activation_gate_created": False,
        "manual_acknowledged": False,
        "service_started": False,
        "systemctl_invoked": False,
        "x5_validated": False,
        "external_network_touched": False,
        "tool_execution": False,
        "actuator_access": False,
        "execution_authority": False,
        "physical_authority": False,
    }
    _atomic_write(
        install_root / "install_receipt.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        mode=0o600,
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--llama-server-sha256", required=True)
    parser.add_argument("--prefix", type=Path, default=Path.home() / ".local/share/rootscope-readonly")
    parser.add_argument("--config-dir", type=Path, default=Path.home() / ".config/rootscope")
    parser.add_argument("--systemd-user-dir", type=Path, default=Path.home() / ".config/systemd/user")
    parser.add_argument(
        "--unit-template",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "systemd/rootscope-llm-readonly.service.disabled-template",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = install(
        release_dir=args.release_dir,
        project_root=args.project_root,
        python_executable=args.python,
        llama_server=args.llama_server,
        llama_server_sha256=args.llama_server_sha256,
        prefix=args.prefix,
        config_dir=args.config_dir,
        systemd_user_dir=args.systemd_user_dir,
        template_path=args.unit_template,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
