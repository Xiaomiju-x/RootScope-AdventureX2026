#!/usr/bin/env python3
"""Install and smoke-test the hash-locked RootScope core without networking.

This script is intentionally conservative.  It accepts only the deterministic
release package, requires Linux/aarch64/CPython 3.10, installs into the current
user's home by default, and runs only filesystem/import/CPU simulated-input
checks.  It never starts a service, opens a device, calls a network client, or
grants execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


RELEASE_SCHEMA = "rootscope.x5-offline-core-release.v1"
RELEASE_ID = "rootscope_x5_offline_core_v1"
RELEASE_STATUS = "HASH_LOCKED_CROSS_BUILT_NOT_EXACT_TWIN_X5_QUALIFIED"
INSTALL_STATUS = "PASS_LOCAL_AARCH64_CPU_SMOKE_NOT_X5_QUALIFIED"
CAPSULE_STATUS = "SIMULATED_ONLY_CLEAN_X5_CAPSULE_NOT_X5_QUALIFIED"
SHA256_LENGTH = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(payload)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _portable_relative_path(value: Any, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context} must be a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError(f"{context} must be normalized")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{context} contains an unsafe component")
    return path


def _strict_false_mapping(value: Any, context: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{context} must be a non-empty object")
    bad = sorted(name for name, flag in value.items() if flag is not False)
    if bad:
        raise ValueError(f"{context} must remain false: {bad}")


def assert_supported_host(
    *,
    system: str | None = None,
    machine: str | None = None,
    version: tuple[int, int] | None = None,
    implementation: str | None = None,
) -> None:
    """Fail before any installation write unless the frozen target matches."""

    actual_system = platform.system() if system is None else system
    actual_machine = platform.machine() if machine is None else machine
    actual_version = sys.version_info[:2] if version is None else version
    actual_implementation = (
        platform.python_implementation()
        if implementation is None
        else implementation
    )
    facts = (
        actual_system,
        actual_machine,
        actual_implementation,
        f"{actual_version[0]}.{actual_version[1]}",
    )
    if facts != ("Linux", "aarch64", "CPython", "3.10"):
        raise RuntimeError(
            "frozen target requires Linux/aarch64/CPython 3.10; "
            f"got {facts[0]}/{facts[1]}/{facts[2]} {facts[3]}"
        )


def _parse_sha256sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            continue
        if len(raw) < SHA256_LENGTH + 2 or raw[SHA256_LENGTH : SHA256_LENGTH + 2] != "  ":
            raise ValueError(f"invalid SHA256SUMS line {line_number}")
        digest = raw[:SHA256_LENGTH]
        relative = raw[SHA256_LENGTH + 2 :]
        if (
            len(digest) != SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid SHA-256 on line {line_number}")
        normalized = _portable_relative_path(relative, f"SHA256SUMS line {line_number}")
        key = normalized.as_posix()
        if key in records:
            raise ValueError(f"duplicate SHA256SUMS path: {key}")
        records[key] = digest
    return records


def verify_package(package_root: Path) -> Mapping[str, Any]:
    """Verify the complete extracted package without importing project code."""

    root = package_root.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("package_root must be a real directory")
    manifest_path = root / "release_manifest.json"
    sums_path = root / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        raise ValueError("release_manifest.json and SHA256SUMS are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("release manifest must be an object")
    if manifest.get("schema") != RELEASE_SCHEMA:
        raise ValueError("unsupported release manifest schema")
    if manifest.get("release_id") != RELEASE_ID or manifest.get("status") != RELEASE_STATUS:
        raise ValueError("release identity/status mismatch")
    _strict_false_mapping(manifest.get("formal_flags"), "formal_flags")
    _strict_false_mapping(manifest.get("authority"), "authority")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("release manifest files must be non-empty")
    expected: dict[str, tuple[str, int]] = {}
    for index, record in enumerate(files):
        if not isinstance(record, Mapping):
            raise ValueError(f"files[{index}] must be an object")
        relative = _portable_relative_path(record.get("path"), f"files[{index}].path")
        name = relative.as_posix()
        digest = record.get("sha256")
        size = record.get("bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"files[{index}].sha256 is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"files[{index}].bytes is invalid")
        if name in expected:
            raise ValueError(f"duplicate release path: {name}")
        expected[name] = (digest, size)

    sums = _parse_sha256sums(sums_path)
    manifest_digest = sha256_file(manifest_path)
    expected_sums = {name: digest for name, (digest, _size) in expected.items()}
    expected_sums["release_manifest.json"] = manifest_digest
    if sums != expected_sums:
        raise ValueError("SHA256SUMS does not exactly cover manifest files")

    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"package symlink is forbidden: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != set(sums) | {"SHA256SUMS"}:
        raise ValueError("extracted package contains missing or extra files")

    for name, (expected_digest, expected_size) in expected.items():
        path = root / Path(*PurePosixPath(name).parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"release file missing or unsafe: {name}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"release file size mismatch: {name}")
        if sha256_file(path) != expected_digest:
            raise ValueError(f"release file hash mismatch: {name}")

    return {
        "schema": "rootscope.x5-offline-package-verification.v1",
        "status": "PASS_PACKAGE_HASHES_NOT_X5_QUALIFIED",
        "release_id": RELEASE_ID,
        "manifest_sha256": manifest_digest,
        "files_verified": len(expected),
        "hardware_touched": False,
        "network_touched": False,
        "service_started": False,
        "x5_validated": False,
        "model_qualified": False,
        "execution_authority": False,
        "physical_authority": False,
    }


def render_runtime_config(template_path: Path, project_root: Path, python: Path) -> dict[str, Any]:
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    if payload.get("status") != CAPSULE_STATUS:
        raise ValueError("capsule status changed")
    _strict_false_mapping(payload.get("authority"), "capsule authority")
    model = payload.get("model")
    llm = payload.get("llm")
    if not isinstance(model, dict) or not isinstance(llm, dict):
        raise ValueError("capsule model/llm contracts are missing")
    for name in ("model_candidate", "model_qualified", "bpu_ready"):
        if model.get(name) is not False:
            raise ValueError(f"model.{name} must remain false")
    if llm.get("enabled") is not False:
        raise ValueError("core install must keep the LLM disabled")
    project = project_root.resolve(strict=True)
    executable = Path(os.path.abspath(str(python.expanduser())))
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("runtime Python is missing or not executable")
    model_path = project / "deploy/x5/models/rootscope_seed17_cpu_experimental_opset11.onnx"
    if not model_path.is_file():
        raise ValueError("frozen CPU ONNX is missing")
    payload["project_root"] = str(project)
    payload["python_executable"] = str(executable)
    model["path"] = str(model_path)
    return payload


def _verify_installed_project(project: Path, file_records: Sequence[Mapping[str, Any]]) -> None:
    expected: dict[str, tuple[str, int]] = {}
    for record in file_records:
        name = str(record["path"])
        if not name.startswith("rootscope/"):
            continue
        relative = name[len("rootscope/") :]
        expected[relative] = (str(record["sha256"]), int(record["bytes"]))
    actual = {
        path.relative_to(project).as_posix()
        for path in project.rglob("*")
        if path.is_file()
    }
    if actual != set(expected):
        raise ValueError("installed project file coverage mismatch")
    for relative, (digest, size) in expected.items():
        path = project / Path(*PurePosixPath(relative).parts)
        if path.is_symlink() or path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"installed project verification failed: {relative}")


def _run_json(command: Sequence[str], *, cwd: Path, output: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    if output is not None:
        payload = json.loads(output.read_text(encoding="utf-8"))
    else:
        payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("command did not emit a JSON object")
    return payload


def install_and_selftest(package_root: Path, install_base: Path) -> Mapping[str, Any]:
    assert_supported_host()
    package = package_root.expanduser().resolve(strict=True)
    package_receipt = verify_package(package)
    manifest = json.loads((package / "release_manifest.json").read_text(encoding="utf-8"))

    base = install_base.expanduser().resolve()
    release_parent = base / "releases"
    release_root = release_parent / RELEASE_ID
    project = release_root / "rootscope"
    venv_parent = base / "venvs"
    venv = venv_parent / RELEASE_ID
    config_path = base / "config" / f"{RELEASE_ID}.capsule.json"
    evidence_dir = base / "evidence" / RELEASE_ID

    if release_root.exists():
        if release_root.is_symlink() or not project.is_dir():
            raise ValueError("existing release destination is unsafe")
        _verify_installed_project(project, manifest["files"])
    else:
        release_parent.mkdir(parents=True, exist_ok=True)
        partial = release_parent / f".{RELEASE_ID}.partial"
        if partial.exists():
            if partial.is_symlink() or partial.parent.resolve() != release_parent.resolve():
                raise ValueError("unsafe partial release path")
            shutil.rmtree(partial)
        shutil.copytree(package / "rootscope", partial / "rootscope", symlinks=False)
        _verify_installed_project(partial / "rootscope", manifest["files"])
        os.replace(partial, release_root)

    if venv.exists() and (venv.is_symlink() or not venv.is_dir()):
        raise ValueError("existing candidate venv destination is unsafe")
    if not venv.exists():
        venv_parent.mkdir(parents=True, exist_ok=True)
        partial_venv = venv_parent / f".{RELEASE_ID}.partial"
        if partial_venv.exists():
            if partial_venv.is_symlink() or partial_venv.parent.resolve() != venv_parent.resolve():
                raise ValueError("unsafe partial venv path")
            shutil.rmtree(partial_venv)
        installer = project / "deploy/x5/scripts/install_cpu_venv_candidate.sh"
        subprocess.run(
            ["/bin/bash", str(installer)],
            cwd=project,
            check=True,
            env={
                **os.environ,
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "ROOTSCOPE_PROJECT_ROOT": str(project),
                "ROOTSCOPE_SYSTEM_PYTHON": sys.executable,
                "ROOTSCOPE_VENV_DIR": str(partial_venv),
            },
        )
        os.replace(partial_venv, venv)
    python = venv / "bin/python3"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("candidate venv Python is missing or not executable")

    import_receipt = _run_json(
        [
            str(python),
            "-c",
            (
                "import json,cv2,numpy,onnxruntime; from PIL import Image; "
                "print(json.dumps({'status':'PASS_IMPORTS_NOT_X5_QUALIFIED',"
                "'numpy':numpy.__version__,'Pillow':Image.__version__,"
                "'onnxruntime':onnxruntime.__version__,'opencv':cv2.__version__,"
                "'providers':onnxruntime.get_available_providers(),"
                "'x5_validated':False,'model_qualified':False,"
                "'execution_authority':False},sort_keys=True))"
            ),
        ],
        cwd=project,
    )
    if "CPUExecutionProvider" not in import_receipt.get("providers", []):
        raise ValueError("CPUExecutionProvider is unavailable")

    runtime_config = render_runtime_config(
        project / "deploy/x5/capsule_config.seed17_cpu_experimental.json",
        project,
        python,
    )
    _atomic_write(
        config_path,
        (json.dumps(runtime_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = evidence_dir / "cpu_preflight.json"
    selftest_path = evidence_dir / "cpu_simulated_selftest.json"
    preflight = _run_json(
        [
            str(python),
            "-m",
            "app.edge.cli",
            "preflight",
            "--config",
            str(config_path),
            "--output",
            str(preflight_path),
        ],
        cwd=project,
        output=preflight_path,
    )
    if preflight.get("status") == "FAIL" or any(preflight.get("authority", {}).values()):
        raise ValueError("read-only CPU preflight failed or mutated authority")
    selftest = _run_json(
        [
            str(python),
            "-m",
            "app.edge.cli",
            "selftest",
            "--config",
            str(config_path),
            "--output",
            str(selftest_path),
        ],
        cwd=project,
        output=selftest_path,
    )
    if selftest.get("status") != "PASS_CPU_ONNX_SIMULATED_INPUT_NOT_ACCURACY_EVIDENCE":
        raise ValueError("CPU simulated-input ONNX self-test failed")
    for name in (
        "hardware_touched",
        "network_touched",
        "x5_validated",
        "bpu_ready",
        "bpu_used",
        "model_candidate",
        "model_qualified",
        "physical_authority",
        "execution_authority",
        "physical_completion",
    ):
        if selftest.get(name) is not False:
            raise ValueError(f"self-test authority/claim changed: {name}")

    receipt = {
        "schema": "rootscope.x5-offline-user-install-receipt.v1",
        "status": INSTALL_STATUS,
        "release_id": RELEASE_ID,
        "package_manifest_sha256": package_receipt["manifest_sha256"],
        "project_root": str(project),
        "python_executable": str(python),
        "capsule_config": str(config_path),
        "preflight_receipt": str(preflight_path),
        "selftest_receipt": str(selftest_path),
        "imports": import_receipt,
        "cpu_onnx_simulated_selftest_passed": True,
        "accuracy_evidence": False,
        "camera_opened": False,
        "device_opened": False,
        "service_started": False,
        "systemctl_invoked": False,
        "network_touched": False,
        "hardware_touched": False,
        "exact_rdk_identity_verified": False,
        "x5_validated": False,
        "wheelhouse_qualified": False,
        "model_candidate": False,
        "model_qualified": False,
        "bpu_binary_present": False,
        "bpu_compiled": False,
        "bpu_ready": False,
        "llm_enabled": False,
        "llama_server_qualified": False,
        "execution_authority": False,
        "physical_authority": False,
        "physical_completion": False,
    }
    _atomic_write(
        evidence_dir / "install_receipt.json",
        (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument(
        "--install-base",
        type=Path,
        default=Path.home() / ".local/share/rootscope",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify target contract and package only; write nothing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        assert_supported_host()
        if args.verify_only:
            payload = verify_package(args.package_root)
        else:
            payload = install_and_selftest(args.package_root, args.install_base)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {
            "schema": "rootscope.x5-offline-install-error.v1",
            "status": "FAIL_CLOSED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "service_started": False,
            "systemctl_invoked": False,
            "network_touched": False,
            "hardware_touched": False,
            "x5_validated": False,
            "model_qualified": False,
            "bpu_ready": False,
            "execution_authority": False,
            "physical_authority": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
