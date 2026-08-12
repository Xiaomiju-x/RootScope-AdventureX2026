"""Read-only clean-X5 environment preflight.

The preflight checks only explicit filesystem paths, Python module metadata,
hashes, and ONNX provider availability.  It never scans ``/dev``, opens a
camera or serial port, starts a service, invokes a network client, or imports a
BPU runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

from .capsule import AUTHORITY_FIELDS, CPU_PROVIDER, CapsuleConfig


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "WARN", "FAIL"}:
            raise ValueError("preflight status must be PASS, WARN or FAIL")

    def to_dict(self) -> Mapping[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_check(
    name: str, path: Path, *, required: bool, executable: bool = False
) -> PreflightCheck:
    if not path.is_file():
        return PreflightCheck(
            name,
            "FAIL" if required else "WARN",
            f"explicit path is missing: {path}",
        )
    if executable and sys.platform != "win32" and not path.stat().st_mode & 0o111:
        return PreflightCheck(name, "FAIL", f"path is not executable: {path}")
    return PreflightCheck(name, "PASS", f"explicit path exists: {path}")


def _module_check(import_name: str, *, required: bool) -> PreflightCheck:
    found = importlib.util.find_spec(import_name) is not None
    return PreflightCheck(
        f"python_module:{import_name}",
        "PASS" if found else "FAIL" if required else "WARN",
        "module metadata found" if found else "module metadata missing",
    )


def _explicit_device_check(
    name: str, device: str, *, enabled: bool, required: bool
) -> PreflightCheck:
    if not enabled:
        return PreflightCheck(name, "PASS", "disabled; no device path inspected")
    # This is a lookup of one configured alias only.  It is intentionally not a
    # glob, /dev listing, VideoCapture call, SDK call, or serial enumeration.
    exists = Path(device).exists()
    return PreflightCheck(
        name,
        "PASS" if exists else "FAIL" if required else "WARN",
        f"configured alias {'exists' if exists else 'is missing'}: {device}; not opened",
    )


def run_preflight(config: CapsuleConfig) -> Mapping[str, Any]:
    checks: list[PreflightCheck] = []
    version_ok = sys.version_info >= (3, 10)
    checks.append(
        PreflightCheck(
            "python_version",
            "PASS" if version_ok else "FAIL",
            platform.python_version(),
        )
    )
    project_root = Path(config.project_root)
    checks.append(
        PreflightCheck(
            "project_root",
            "PASS" if project_root.is_dir() else "FAIL",
            f"explicit project root {'exists' if project_root.is_dir() else 'is missing'}: {project_root}",
        )
    )
    checks.append(
        _path_check(
            "python_executable",
            Path(config.python_executable),
            required=True,
            executable=True,
        )
    )
    checks.extend((_module_check("numpy", required=True), _module_check("PIL", required=True)))

    if config.model.enabled:
        checks.append(_module_check("onnxruntime", required=True))
        model_path = Path(config.model.path)
        path_result = _path_check("onnx_model", model_path, required=True)
        checks.append(path_result)
        if path_result.status == "PASS":
            actual = _sha256_file(model_path)
            checks.append(
                PreflightCheck(
                    "onnx_model_sha256",
                    "PASS" if actual == config.model.sha256 else "FAIL",
                    f"actual={actual} expected={config.model.sha256}",
                )
            )
        try:
            import onnxruntime as ort  # type: ignore

            providers = list(ort.get_available_providers())
            checks.append(
                PreflightCheck(
                    "onnx_cpu_provider",
                    "PASS" if CPU_PROVIDER in providers else "FAIL",
                    f"available={providers}; requested capsule provider={CPU_PROVIDER}",
                )
            )
        except ImportError:
            checks.append(
                PreflightCheck(
                    "onnx_cpu_provider",
                    "FAIL",
                    "onnxruntime import failed; no provider claim made",
                )
            )
    else:
        checks.append(
            PreflightCheck(
                "onnx_model",
                "WARN",
                "disabled; preprocessing-only simulated self-test is available",
            )
        )

    camera_enabled = config.rgb.enabled or config.depth.enabled
    checks.append(_module_check("cv2", required=camera_enabled))
    checks.append(
        _explicit_device_check(
            "rgb_input_alias",
            config.rgb.device,
            enabled=config.rgb.enabled,
            required=config.rgb.required,
        )
    )
    checks.append(
        _explicit_device_check(
            "depth_input_alias",
            config.depth.device,
            enabled=config.depth.enabled,
            required=config.depth.required,
        )
    )

    if config.llm.enabled:
        executable_result = _path_check(
            "llm_executable", Path(config.llm.executable), required=True, executable=True
        )
        model_result = _path_check(
            "llm_model", Path(config.llm.model_path), required=True
        )
        checks.extend((executable_result, model_result))
        if model_result.status == "PASS":
            actual = _sha256_file(Path(config.llm.model_path))
            checks.append(
                PreflightCheck(
                    "llm_model_sha256",
                    "PASS" if actual == config.llm.model_sha256 else "FAIL",
                    f"actual={actual} expected={config.llm.model_sha256}",
                )
            )
    else:
        checks.append(
            PreflightCheck(
                "local_llm",
                "PASS",
                "disabled; deterministic template explanation remains the fallback",
            )
        )

    failures = sum(check.status == "FAIL" for check in checks)
    warnings = sum(check.status == "WARN" for check in checks)
    status = "FAIL" if failures else "PASS_WITH_WARN" if warnings else "PASS"
    authority = dict(config.authority.to_dict())
    if set(authority) != set(AUTHORITY_FIELDS) or any(authority.values()):
        raise AssertionError("zero-authority contract mutated after configuration load")
    return {
        "schema_version": "rootscope.x5-readonly-preflight.v1",
        "status": status,
        "capsule_status": config.status,
        "host_facts": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "checks": [check.to_dict() for check in checks],
        "summary": {"fail": failures, "warn": warnings, "pass": len(checks) - failures - warnings},
        "device_policy": "EXPLICIT_ALIAS_EXISTENCE_ONLY_NOT_OPENED_NOT_ENUMERATED",
        "network_policy": "NO_NETWORK_CHECK_OR_MUTATION",
        "provider_policy": "CPUExecutionProvider_ONLY_NO_BPU_IMPORT",
        "authority": authority,
    }


def failed_check_names(report: Mapping[str, Any]) -> Sequence[str]:
    return tuple(
        item["name"]
        for item in report.get("checks", [])
        if isinstance(item, Mapping) and item.get("status") == "FAIL"
    )
