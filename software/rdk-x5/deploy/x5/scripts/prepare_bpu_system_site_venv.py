#!/usr/bin/env python3
"""Create/verify the isolated RootScope BPU system-site-packages venv.

Run this file with the RDK OS ``/usr/bin/python3``.  It creates one CPython
3.10 venv with ``system_site_packages=True``, verifies that NumPy and
``hobot_dnn`` still resolve outside that venv, and optionally installs only a
hash-bound local Pillow wheel with pip ``--no-index --no-deps``.  It never
loads a BPU model, opens a device, starts a service, or accesses a network.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import venv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PILLOW_WHEEL_RE = re.compile(r"^pillow-[^-]+-.+\.whl$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pyvenv_config(path: Path) -> Mapping[str, str]:
    if not path.is_file():
        raise ValueError("BPU venv is missing pyvenv.cfg")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip().lower()] = value.strip()
    return result


def path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=True).relative_to(parent.resolve(strict=True))
    except ValueError:
        return False
    return True


def validate_pillow_wheel(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("expected Pillow SHA-256 must be 64 lowercase hex characters")
    configured = path.expanduser()
    if configured.is_symlink():
        raise ValueError("Pillow wheel must not be a symlink")
    resolved = configured.resolve(strict=True)
    if not resolved.is_file() or not _PILLOW_WHEEL_RE.fullmatch(resolved.name):
        raise ValueError("offline dependency must be one Pillow wheel")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ValueError(
            f"Pillow wheel SHA-256 mismatch: actual={actual} expected={expected_sha256}"
        )
    return {"path": str(resolved), "sha256": actual, "size_bytes": resolved.stat().st_size}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser()
    if target.is_symlink():
        raise ValueError("--output-json must not be a symlink")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    os.chmod(partial, 0o600)
    os.replace(partial, target)


def _require_rdk_system_python() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "aarch64",
        "arm64",
    }:
        raise RuntimeError("BPU venv preparation requires Linux/aarch64")
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 10):
        raise RuntimeError("BPU venv preparation requires RDK OS CPython 3.10")
    if sys.prefix != sys.base_prefix:
        raise RuntimeError(
            "run preparation with RDK /usr/bin/python3, not from core or another venv"
        )
    for module_name in ("numpy", "hobot_dnn"):
        if importlib.util.find_spec(module_name) is None:
            raise RuntimeError(f"RDK system Python cannot resolve {module_name}")


def _venv_python(venv_root: Path) -> Path:
    executable = venv_root / "bin/python"
    if not executable.is_file():
        raise RuntimeError("BPU venv Python executable is missing")
    return executable


def _probe_runtime(python: Path) -> Mapping[str, Any]:
    probe = """
import importlib.util
import json
import sys
import numpy
import hobot_dnn
from hobot_dnn import pyeasy_dnn

pil_spec = importlib.util.find_spec("PIL")
pil_version = None
pil_origin = None
if pil_spec is not None:
    import PIL
    pil_version = getattr(PIL, "__version__", None)
    pil_origin = getattr(PIL, "__file__", None)

print(json.dumps({
    "python_version": list(sys.version_info[:3]),
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "numpy_version": numpy.__version__,
    "numpy_origin": numpy.__file__,
    "hobot_dnn_origin": getattr(hobot_dnn, "__file__", None),
    "pyeasy_dnn_imported": pyeasy_dnn is not None,
    "pillow_available": pil_spec is not None,
    "pillow_version": pil_version,
    "pillow_origin": pil_origin,
}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    result = subprocess.run(
        [str(python), "-I", "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(f"BPU venv import-only probe failed: {detail}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("BPU venv probe did not emit one JSON object") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("BPU venv probe result is not an object")
    return value


def prepare(
    *,
    venv_root: Path,
    pillow_wheel: Path | None,
    expected_pillow_sha256: str | None,
) -> Mapping[str, Any]:
    _require_rdk_system_python()
    target = venv_root.expanduser()
    if target.is_symlink():
        raise ValueError("BPU venv root must not be a symlink")
    created = False
    if target.exists() and not target.is_dir():
        raise ValueError("BPU venv root exists but is not a directory")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(
            system_site_packages=True,
            clear=False,
            symlinks=True,
            with_pip=True,
        ).create(target)
        created = True
    target = target.resolve(strict=True)

    config = parse_pyvenv_config(target / "pyvenv.cfg")
    if config.get("include-system-site-packages", "").lower() != "true":
        raise RuntimeError(
            "existing venv is incompatible: include-system-site-packages must be true"
        )
    python = _venv_python(target)
    probe = dict(_probe_runtime(python))
    if probe.get("python_version", [])[:2] != [3, 10]:
        raise RuntimeError("BPU venv is not CPython 3.10")
    for module_name, key in (
        ("numpy", "numpy_origin"),
        ("hobot_dnn", "hobot_dnn_origin"),
    ):
        origin = probe.get(key)
        if not isinstance(origin, str) or not origin:
            raise RuntimeError(f"BPU venv did not report {module_name} origin")
        if path_is_within(Path(origin), target):
            raise RuntimeError(
                f"{module_name} resolves inside the BPU venv; system ABI must be preserved"
            )
    if probe.get("pyeasy_dnn_imported") is not True:
        raise RuntimeError("pyeasy_dnn import-only probe failed")

    pillow_install: Mapping[str, Any] | None = None
    if pillow_wheel is None and expected_pillow_sha256 is not None:
        raise ValueError("--expected-pillow-sha256 requires --pillow-wheel")
    if pillow_wheel is not None and expected_pillow_sha256 is None:
        raise ValueError("--pillow-wheel requires --expected-pillow-sha256")
    if probe.get("pillow_available") is not True:
        if pillow_wheel is None or expected_pillow_sha256 is None:
            raise RuntimeError(
                "Pillow is absent; provide one hash-bound local Pillow wheel"
            )
        pillow_install = validate_pillow_wheel(
            pillow_wheel, expected_pillow_sha256
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INDEX": "1",
                "PIP_REQUIRE_VIRTUALENV": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        result = subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                str(pillow_install["path"]),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
            env=environment,
        )
        if result.returncode != 0:
            raise RuntimeError(f"offline Pillow install failed: {result.stderr[-2000:]}")
        probe = dict(_probe_runtime(python))
    elif pillow_wheel is not None and expected_pillow_sha256 is not None:
        # Validate the supplied handoff even though system Pillow made an
        # installation unnecessary.  Do not mutate a working environment.
        pillow_install = validate_pillow_wheel(
            pillow_wheel, expected_pillow_sha256
        )

    if probe.get("pillow_available") is not True:
        raise RuntimeError("Pillow remains unavailable after offline preparation")
    return {
        "schema": "rootscope.seed17-bpu-system-site-venv-receipt.v1",
        "status": "PASS_IMPORT_ONLY_ENVIRONMENT_NOT_MODEL_OR_X5_QUALIFICATION",
        "venv": {
            "path": str(target),
            "created_this_run": created,
            "python": str(python),
            "python_version": probe["python_version"],
            "include_system_site_packages": True,
            "core_v1_venv_allowed": False,
        },
        "runtime_import_probe": probe,
        "pillow_wheel_handoff": pillow_install,
        "dependency_policy": {
            "network_install_allowed": False,
            "pip_no_index": True,
            "pip_no_deps": True,
            "local_install_allowlist": ["Pillow"],
            "venv_numpy_install_allowed": False,
            "venv_hobot_dnn_install_allowed": False,
            "system_numpy_abi_preserved": True,
            "system_hobot_dnn_preserved": True,
        },
        "claims": {
            "bpu_model_loaded": False,
            "bpu_forward_executed": False,
            "x5_ready": False,
            "x5_validated": False,
            "camera_qualified": False,
            "model_candidate": False,
            "model_qualified": False,
            "production_integration_allowed": False,
        },
        "authority": {
            "hardware_touched": False,
            "network_touched": False,
            "device_enumerated": False,
            "serial_write": False,
            "state_machine_write": False,
            "pump_command": False,
            "irrigation_execution": False,
            "execution_authority": False,
            "physical_authority": False,
            "physical_completion": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", required=True, type=Path)
    parser.add_argument("--pillow-wheel", type=Path)
    parser.add_argument("--expected-pillow-sha256")
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = prepare(
            venv_root=args.venv,
            pillow_wheel=args.pillow_wheel,
            expected_pillow_sha256=args.expected_pillow_sha256,
        )
        if args.output_json is not None:
            _atomic_write_json(args.output_json, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": "rootscope.seed17-bpu-system-site-venv-error.v1",
            "status": "FAIL_CLOSED_NO_MODEL_LOAD_NO_AUTHORITY",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "claims": {
                "bpu_model_loaded": False,
                "bpu_forward_executed": False,
                "x5_ready": False,
                "x5_validated": False,
                "camera_qualified": False,
                "model_candidate": False,
                "model_qualified": False,
                "production_integration_allowed": False,
            },
            "authority": {
                "hardware_touched": False,
                "network_touched": False,
                "device_enumerated": False,
                "serial_write": False,
                "state_machine_write": False,
                "pump_command": False,
                "irrigation_execution": False,
                "execution_authority": False,
                "physical_authority": False,
                "physical_completion": False,
            },
        }
        try:
            if args.output_json is not None:
                _atomic_write_json(args.output_json, failure)
        finally:
            print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
