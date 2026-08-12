"""Fail-closed audit of the cross-downloaded CPython 3.10/aarch64 wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_FILENAME_TOKENS = ("x86_64", "amd64", "win32", "win_amd64", "macosx")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_manifest(
    manifest_path: str | Path, *, require_wheel_files: bool = False
) -> dict:
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("schema", payload.get("schema") == "rootscope.x5-candidate-wheelhouse.v1", payload.get("schema"))
    target = payload.get("target_contract", {})
    check("target_linux", target.get("os_family") == "linux", target)
    check("target_aarch64", target.get("architecture") == "aarch64", target)
    check("target_cp310", target.get("python_version") == "3.10" and target.get("abi") == "cp310", target)
    claims = payload.get("claims", {})
    check(
        "wheel_files_not_redistributed_in_git",
        claims.get("wheel_files_redistributed_in_git") is False,
        claims.get("wheel_files_redistributed_in_git"),
    )
    for name in (
        "exact_twin_install_tested",
        "rdk_x5_import_tested",
        "golden_preprocess_replayed_on_x5",
        "onnx_replayed_on_x5",
        "wheelhouse_qualified",
        "x5_ready",
    ):
        check(f"claim_{name}_false", claims.get(name) is False, claims.get(name))
    authority = payload.get("authority", {})
    for name in ("hardware_touched", "network_configuration_changed", "x5_validated", "bpu_ready", "execution_authority", "physical_authority"):
        check(f"authority_{name}_false", authority.get(name) is False, authority.get(name))

    wheel_dir = (root / payload.get("wheel_directory", "")).resolve()
    check("wheel_dir_contained", wheel_dir.parent == root, str(wheel_dir))
    records = payload.get("wheels", [])
    check("eleven_wheel_records", isinstance(records, list) and len(records) == 11, len(records) if isinstance(records, list) else None)
    expected_names: set[str] = set()
    hashes: set[str] = set()
    distributions: set[str] = set()
    for index, record in enumerate(records if isinstance(records, list) else []):
        filename = record.get("filename")
        distribution = str(record.get("distribution", "")).lower()
        prefix = f"wheel_{index}_{distribution or 'missing'}"
        safe_name = isinstance(filename, str) and filename == Path(filename).name and filename.endswith(".whl")
        check(prefix + "_safe_filename", safe_name, filename)
        if not safe_name:
            continue
        expected_names.add(filename)
        distributions.add(distribution)
        path = wheel_dir / filename
        check(
            prefix + "_present_or_manifest_only",
            path.is_file() or not require_wheel_files,
            {"path": str(path), "require_wheel_files": require_wheel_files},
        )
        if not path.is_file():
            continue
        check(prefix + "_bytes", path.stat().st_size == record.get("bytes"), path.stat().st_size)
        digest = sha256_file(path)
        check(prefix + "_sha_format", isinstance(record.get("sha256"), str) and bool(SHA_RE.fullmatch(record["sha256"])), record.get("sha256"))
        check(prefix + "_sha", digest == record.get("sha256"), digest)
        check(prefix + "_unique_sha", digest not in hashes, digest)
        hashes.add(digest)
        lowered = filename.lower()
        check(prefix + "_not_foreign_arch", not any(token in lowered for token in FORBIDDEN_FILENAME_TOKENS), filename)
        scope = record.get("platform_scope")
        tag_ok = (scope == "pure_python" and "none-any.whl" in lowered) or (
            isinstance(scope, str) and scope.startswith("linux_aarch64") and "aarch64.whl" in lowered
        )
        check(prefix + "_platform_tag", tag_ok, {"scope": scope, "filename": filename})
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                check(prefix + "_wheel_zip", archive.testzip() is None, "zip CRC")
                check(prefix + "_metadata_present", any(name.endswith(".dist-info/METADATA") for name in names), "METADATA")
                check(prefix + "_wheel_metadata_present", any(name.endswith(".dist-info/WHEEL") for name in names), "WHEEL")
        except zipfile.BadZipFile as exc:
            check(prefix + "_wheel_zip", False, str(exc))

    actual_names = {path.name for path in wheel_dir.glob("*.whl")} if wheel_dir.is_dir() else set()
    coverage_ok = (
        actual_names == expected_names
        if require_wheel_files
        else actual_names.issubset(expected_names)
    )
    check(
        "wheel_file_coverage",
        coverage_ok,
        {
            "mode": "required" if require_wheel_files else "manifest_only",
            "missing": sorted(expected_names - actual_names),
            "extra": sorted(actual_names - expected_names),
        },
    )
    check("unique_distributions", len(distributions) == len(records), sorted(distributions))
    required = {"numpy", "pillow", "onnxruntime", "opencv-python-headless", "coloredlogs", "flatbuffers", "packaging", "protobuf", "sympy", "humanfriendly", "mpmath"}
    check("required_distributions", distributions == required, {"actual": sorted(distributions), "required": sorted(required)})

    requirements = (root / payload.get("requirements_path", "")).resolve()
    check("requirements_contained", requirements.parent == root, str(requirements))
    check("requirements_present", requirements.is_file(), str(requirements))
    if requirements.is_file():
        requirements_text = requirements.read_text(encoding="utf-8")
        for record in records:
            check(
                "requirements_hash_" + str(record.get("distribution", "missing")).lower(),
                f"--hash=sha256:{record.get('sha256')}" in requirements_text,
                record.get("sha256"),
            )
    passed = all(item["passed"] for item in checks)
    return {
        "schema": "rootscope.x5-candidate-wheelhouse-audit.v1",
        "status": "PASS_NOT_X5_QUALIFIED" if passed else "FAIL",
        "passed": passed,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "manifest_sha256": sha256_file(manifest_path),
        "hardware_touched": False,
        "network_touched": False,
        "x5_validated": False,
        "wheelhouse_qualified": False,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--require-wheel-files",
        action="store_true",
        help="fail unless every content-bound wheel is present and verified",
    )
    args = parser.parse_args()
    result = audit_manifest(
        args.manifest, require_wheel_files=args.require_wheel_files
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
