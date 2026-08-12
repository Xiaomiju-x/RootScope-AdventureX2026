#!/usr/bin/env python3
"""Build the deterministic RootScope competition runtime v2 candidate.

The archive is an AdventureX-only, zero-physical-authority overlay.  It carries
the plant r7 BPU binary as an explicitly unqualified shadow candidate and
hash-binds the already installed CPU model, Qwen2 0.5B model, llama-server and
immutable field-v2 support files.  It never edits the frozen XRD project or any
previous RootScope release.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
from typing import Any, Iterable, Mapping, Sequence


BUILD_DATE = "2026-07-23"
CANDIDATE_ID = "rootscope_competition_runtime_v2_candidate_20260723"
ARCHIVE_NAME = f"{CANDIDATE_ID}.tar"
R7_SHA256 = "4dc4bc24741332bb6bc6da184dd1d6f954ae539ba740330490339d76eb200285"
R7_SOURCE = (
    "output/rootscope_bpu_seed17_quant_variant_r7_default_int16_all_nodes/"
    "model_output/"
    "rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes.bin"
)
R7_PACKAGE = (
    "models/rootscope_seed17_resnet18_224x224_rgb_ddr_"
    "r7_default_int16_all_nodes.bin"
)

EXTERNAL_COMPONENTS = (
    {
        "id": "seed17_cpu_onnx",
        "path": (
            "~/.local/share/rootscope-field-v2/core_v1/releases/"
            "rootscope_x5_offline_core_v1/rootscope/deploy/x5/models/"
            "rootscope_seed17_cpu_experimental_opset11.onnx"
        ),
        "sha256": "50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad",
        "local_source": "rootscope/deploy/x5/models/rootscope_seed17_cpu_experimental_opset11.onnx",
    },
    {
        "id": "field_v2_capsule",
        "path": (
            "~/.local/share/rootscope-field-v2/core_v1/config/"
            "rootscope_x5_offline_core_v1.capsule.json"
        ),
        "sha256": "1e839948e466895e0b416aababb39402380463f9ce125c8aeb59f4087191cb97",
        "local_source": None,
    },
    {
        "id": "known_card_registry",
        "path": (
            "~/.local/share/rootscope-field-v2/core_v1/releases/"
            "rootscope_x5_offline_core_v1/rootscope/app/vision/"
            "known_card_template_registry.frozen.experimental.json"
        ),
        "sha256": "f5328b66c6b385dd081c1a27577d92b5c63fc18dc00d815f7d3d2561bb68e29f",
        "local_source": "rootscope/app/vision/known_card_template_registry.frozen.experimental.json",
    },
    {
        "id": "card_matcher_config",
        "path": (
            "~/.local/share/rootscope-field-v2/core_v1/releases/"
            "rootscope_x5_offline_core_v1/rootscope/app/vision/"
            "card_geometric_matcher.config.example.json"
        ),
        "sha256": "9952864e50371675e7ea181cc57f2edd9eadd9189bfa0e9eda1e5cdd8f8ca61a",
        "local_source": "rootscope/app/vision/card_geometric_matcher.config.example.json",
    },
    {
        "id": "qwen2_05b_q4_gguf",
        "path": (
            "~/.local/share/rootscope-field-v2/readonly_llm/models/"
            "qwen2_05b_distill.Q4_K_M.gguf"
        ),
        "sha256": "6c8adc95de81b1d0103fcf900a5f369d5936d234e42ed64881dad56e0104e77b",
        "local_source": "output/rootscope_llm_readonly_release_v1/qwen2_05b_distill.Q4_K_M.gguf",
    },
    {
        "id": "llama_server_arm64_b9637",
        "path": (
            "~/.local/share/rootscope-field-v2/staged_components/"
            "rootscope_llama_server_arm64_b9637_v1/bin/llama-server"
        ),
        "sha256": "dcb636215243b8911488b8ca96f0c39bedee14e92f44f7d0ef6c599419acf9b9",
        "local_source": "output/rootscope_llama_server_arm64_b9637_v1/bin/llama-server",
    },
    {
        "id": "drobotics_hrt_model_exec_1_24_5",
        "path": "/usr/sbin/hrt_model_exec",
        "sha256": "c3a47c77889bc82c8519b68a86b75f8205c6a4f9695339bb3d01da2713abcb04",
        "local_source": None,
    },
    {
        "id": "drobotics_libdnn_runtime",
        "path": "/usr/lib/libdnn.so",
        "sha256": "661bac161124921eb9065fb9cd8d311144ea3f899fee28d61dfd0d7255074ace",
        "local_source": None,
    },
    {
        "id": "drobotics_libhbrt_bayes_e",
        "path": "/usr/lib/libhbrt_bayes_aarch64.so",
        "sha256": "8b4719d147a53a4adb215f0307a732a420b4dc2ffaa893f4a8fcd02a7e88fc9a",
        "local_source": None,
    },
)

TEXT_TREES = (
    "rootscope/app/competition_runtime",
    "rootscope/app/competition_llm",
    "rootscope/app/competition_rag",
    "rootscope/app/edge",
    "rootscope/app/omega_knowledge",
    "rootscope/app/omega_vision",
    "rootscope/app/vision",
    "rootscope/configs/competition",
)
REQUIRED_FILES = (
    "rootscope/deploy/x5/omega_standalone_app_init.py",
    "rootscope/configs/omega/field_knowledge.v1.md",
    "rootscope/configs/omega/vision_board_replay_new_x5_20260723.json",
    "rootscope/evidence/physical_laptop_batch_20260723T131242Z/output/summary.json",
    "rootscope/OMEGA_V3_X5_FINAL_HANDOFF_20260723.md",
    "rootscope/pyproject.toml",
    "rootscope/tools/x5_competition_live_vision.py",
    "rootscope/tools/x5_competition_live_vision_v2.py",
    "rootscope/tools/start_x5_competition_runtime_v2.sh",
    "rootscope/tools/start_x5_competition_live_vision_v2.sh",
    "rootscope/tools/x5_competition_static_cpu_bpu_replay.py",
    "rootscope/tools/x5_competition_resource_monitor.py",
    "rootscope/ROOTSCOPE_X5_COMPLETION_PLAN_20260723.md",
    "tools/release/build_rootscope_competition_runtime_v2.py",
    "tools/release/verify_rootscope_competition_runtime_v2.py",
)
TEST_GLOB = "test_competition*.py"
TEXT_SUFFIXES = frozenset(
    {".py", ".json", ".jsonl", ".md", ".sh", ".toml", ".yaml", ".yml", ".txt"}
)
FORBIDDEN_PARTS = frozenset(
    {"__pycache__", ".git", ".ssh", "credentials", "secrets", "private_keys"}
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"(?i)\bBearer[ \t]+[A-Za-z0-9._~+/=-]{8,}\b"),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\."
        r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    ),
    re.compile(
        r"(?i)\b(?:https?|ftp)://"
        r"[^/\s:@]+:[^/\s@]{4,}@[^/\s]+"
    ),
    re.compile(
        r"""(?ix)
        (?:
            ["'](?:password|passwd|pwd|api[_-]?key|access[_-]?token|
                  refresh[_-]?token|client[_-]?secret|secret[_-]?key|token)["']
            |
            \b(?:password|passwd|pwd|api[_-]?key|access[_-]?token|
                refresh[_-]?token|client[_-]?secret|secret[_-]?key|token)\b
        )
        \s*[:=]\s*
        ["'][^"'\r\n]{4,}["']
        """
    ),
)
STRUCTURED_SECRET_PATTERNS = (
    re.compile(
        r"""(?imx)
        ^[ \t]*["']?
        (?:password|passwd|pwd|api[_-]?key|access[_-]?token|
           refresh[_-]?token|client[_-]?secret|secret[_-]?key|token)
        ["']?[ \t]*:[ \t]*
        (?!null\b|false\b|true\b|["']?redacted["']?\b|
           ["']?placeholder["']?\b|["']?change[_-]?me["']?\b)
        (?:
            ["'][^"'\r\n]{4,}["']
            |
            [A-Za-z0-9_./+@=~-]{4,}
        )
        [ \t]*(?:,[ \t]*)?(?:\#.*)?$
        """
    ),
)


@dataclass(frozen=True)
class Entry:
    source: Path
    package_path: str
    mode: int
    category: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def safe_package_path(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe package path: {value!r}")
    return value


def regular_below(path: Path, root: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} leaves AdventureX: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    return resolved


def scan_text(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"source must be UTF-8: {path}") from exc
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"possible embedded secret in {path}")
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        for pattern in STRUCTURED_SECRET_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"possible structured secret in {path}")


def run_json_audit(
    *,
    script: Path,
    rootscope_root: Path,
) -> dict[str, Any]:
    """Run one mandatory release gate and accept only a final JSON PASS."""

    if script.is_symlink() or not script.is_file():
        raise ValueError(f"required release audit is missing or unsafe: {script}")
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(rootscope_root)
        if not existing_pythonpath
        else os.pathsep.join((str(rootscope_root), existing_pythonpath))
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=rootscope_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ValueError(
            f"mandatory audit failed ({script.name}): {detail[:1000]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"mandatory audit emitted no report: {script.name}")
    try:
        report = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"mandatory audit final line is not JSON: {script.name}"
        ) from exc
    if not isinstance(report, dict) or report.get("status") != "PASS":
        raise ValueError(f"mandatory audit did not report PASS: {script.name}")
    return report


def run_mandatory_rag_audits(adventurex: Path) -> dict[str, Any]:
    rootscope = adventurex / "rootscope"
    competition = rootscope / "configs/competition"
    structural = run_json_audit(
        script=competition / "audit_competition_rag.py",
        rootscope_root=rootscope,
    )
    retrieval = run_json_audit(
        script=competition / "audit_competition_rag_retrieval.py",
        rootscope_root=rootscope,
    )
    return {
        "structural": structural,
        "fts5_bm25_retrieval": retrieval,
    }


def collect_entries(adventurex: Path) -> list[Entry]:
    root = adventurex.resolve(strict=True)
    entries: dict[str, Entry] = {}

    def add(
        source_relative: str,
        package_path: str | None = None,
        *,
        category: str,
        mode: int = 0o644,
    ) -> None:
        source = regular_below(root / source_relative, root, category)
        relative = source.relative_to(root)
        if {part.casefold() for part in relative.parts} & FORBIDDEN_PARTS:
            raise ValueError(f"forbidden source path: {relative.as_posix()}")
        target = safe_package_path(package_path or relative.as_posix())
        if target in entries:
            raise ValueError(f"duplicate package path: {target}")
        if source.suffix.lower() not in TEXT_SUFFIXES:
            raise ValueError(f"non-text source not allowlisted: {source}")
        scan_text(source)
        entries[target] = Entry(source, target, mode, category)

    for tree_relative in TEXT_TREES:
        tree = root / tree_relative
        if not tree.exists():
            if tree_relative.endswith(("competition_rag",)):
                continue
            raise ValueError(f"required source tree is missing: {tree_relative}")
        if not tree.is_dir() or tree.is_symlink():
            raise ValueError(f"unsafe source tree: {tree_relative}")
        count = 0
        for source in sorted(tree.rglob("*"), key=lambda item: item.as_posix()):
            if not source.is_file() or source.is_symlink():
                continue
            if "__pycache__" in source.parts or source.suffix.lower() in {".pyc", ".pyo"}:
                continue
            if source.suffix.lower() not in TEXT_SUFFIXES:
                continue
            add(
                source.relative_to(root).as_posix(),
                category=tree_relative.replace("/", "_"),
            )
            count += 1
        if count == 0:
            raise ValueError(f"required source tree is empty: {tree_relative}")

    add(
        "rootscope/deploy/x5/omega_standalone_app_init.py",
        "rootscope/app/__init__.py",
        category="minimal_zero_authority_app_init",
    )
    for item in REQUIRED_FILES:
        if item == "rootscope/deploy/x5/omega_standalone_app_init.py":
            continue
        mode = 0o755 if item.endswith((".py", ".sh")) and (
            "/tools/" in item or item.startswith("tools/")
        ) else 0o644
        add(item, category="required_runtime_or_contract", mode=mode)
    tests_root = root / "rootscope/tests"
    for source in sorted(tests_root.glob(TEST_GLOB), key=lambda item: item.name):
        add(
            source.relative_to(root).as_posix(),
            category="portable_competition_tests",
        )
    if not any(entry.category == "portable_competition_tests" for entry in entries.values()):
        raise ValueError("no portable competition tests were found")
    return [entries[name] for name in sorted(entries)]


def validate_external_sources(adventurex: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in EXTERNAL_COMPONENTS:
        local_relative = item["local_source"]
        local_status: dict[str, Any] = {
            "available_during_build": local_relative is not None,
            "source_path_relative_to_adventurex": local_relative,
        }
        if local_relative is not None:
            source = regular_below(
                adventurex / str(local_relative),
                adventurex,
                f"external component {item['id']}",
            )
            actual = sha256_file(source)
            if actual != item["sha256"]:
                raise ValueError(
                    f"external component hash changed: {item['id']} {actual}"
                )
            local_status["bytes"] = source.stat().st_size
        records.append(
            {
                "id": item["id"],
                "x5_path": item["path"],
                "sha256": item["sha256"],
                "bundled": False,
                "required": True,
                "build_source_check": local_status,
            }
        )
    return records


def add_bytes(
    archive: tarfile.TarFile, name: str, data: bytes, mode: int
) -> None:
    info = tarfile.TarInfo(safe_package_path(name))
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, BytesIO(data))


def add_file(
    archive: tarfile.TarFile, name: str, source: Path, mode: int
) -> None:
    info = tarfile.TarInfo(safe_package_path(name))
    info.size = source.stat().st_size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with source.open("rb") as stream:
        archive.addfile(info, stream)


def build(adventurex: Path, output_dir: Path) -> dict[str, Any]:
    root = adventurex.resolve(strict=True)
    output = output_dir.resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must stay below AdventureX") from exc
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rag_audits = run_mandatory_rag_audits(root)
    entries = collect_entries(root)
    r7 = regular_below(root / R7_SOURCE, root, "r7 BPU shadow candidate")
    if r7.stat().st_size != 12_519_415 or sha256_file(r7) != R7_SHA256:
        raise ValueError("r7 BPU candidate size/SHA-256 mismatch")
    entries.append(Entry(r7, R7_PACKAGE, 0o644, "plant_bpu_shadow_candidate"))
    entries.sort(key=lambda item: item.package_path)
    external = validate_external_sources(root)

    files = [
        {
            "path": entry.package_path,
            "sha256": sha256_file(entry.source),
            "bytes": entry.source.stat().st_size,
            "mode": format(entry.mode, "04o"),
            "category": entry.category,
            "source_path_relative_to_adventurex": entry.source.relative_to(root).as_posix(),
        }
        for entry in entries
    ]
    authority = {
        "serial_open": False,
        "serial_write": False,
        "gpio_access": False,
        "pump_command": False,
        "state_machine_write": False,
        "execution_authority": False,
        "physical_authority": False,
        "irrigation_execution": False,
        "physical_completion": False,
        "systemd_write": False,
        "network_configuration_write": False,
    }
    manifest_base: dict[str, Any] = {
        "schema": "rootscope.competition-runtime-candidate.v2",
        "candidate_id": CANDIDATE_ID,
        "build_date": BUILD_DATE,
        "status": "SHADOW_CANDIDATE_REQUIRES_X5_QUALIFICATION",
        "target": {
            "hostname": "rootscope-x5",
            "architecture": "aarch64",
            "serial": "3281556110258c1902ab5d9b0012004",
            "machine_id": "<redacted-device-boot-id>",
            "wlan_mac": "02:00:00:00:00:01",
            "physical_ram_gib": 4,
        },
        "architecture": {
            "plant_bpu_role": "SHADOW_CANDIDATE_NOT_DEFAULT",
            "plant_bpu_sha256": R7_SHA256,
            "cpu_role": "SPARSE_AUDIT_AND_FAIL_CLOSED_FALLBACK",
            "rag_backend": "SQLITE_FTS5_BM25",
            "llm_topology": "ONE_QWEN2_05B_RESIDENT_ONE_CALL_THREE_LOGICAL_ROLES",
            "serial_adapter": "DISABLED_PLACEHOLDER_ONLY",
            "yolo_used": False,
        },
        "qualification": {
            "previous_release_selected_bin_remains_null": True,
            "r7_passed_previous_probability_drift_gate": False,
            "r7_actual_x5_forward_required": True,
            "plant_domain_accuracy_qualified": False,
            "camera_generalization_qualified": False,
            "production_integration_allowed": False,
            "physical_closure": False,
        },
        "build_gates": {
            "rag_structural": rag_audits["structural"],
            "rag_fts5_bm25_retrieval": rag_audits["fts5_bm25_retrieval"],
        },
        "external_components": external,
        "authority": authority,
        "files": files,
    }
    composition = hashlib.sha256(
        json.dumps(
            manifest_base,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest = {**manifest_base, "composition_root_sha256": composition}
    manifest_bytes = canonical_json(manifest)
    sums = [f"{item['sha256']}  {item['path']}" for item in files]
    sums.append(
        f"{hashlib.sha256(manifest_bytes).hexdigest()}  candidate_manifest.json"
    )
    sums_bytes = ("\n".join(sorted(sums)) + "\n").encode("ascii")

    archive_path = output / ARCHIVE_NAME
    temporary = output / f".{ARCHIVE_NAME}.{os.getpid()}.partial"
    if archive_path.exists() or temporary.exists():
        raise FileExistsError("candidate archive or temporary already exists")
    try:
        with tarfile.open(temporary, "x", format=tarfile.USTAR_FORMAT) as archive:
            members: list[tuple[str, str, Any, int]] = [
                (
                    f"{CANDIDATE_ID}/{entry.package_path}",
                    "file",
                    entry.source,
                    entry.mode,
                )
                for entry in entries
            ]
            members.extend(
                [
                    (
                        f"{CANDIDATE_ID}/candidate_manifest.json",
                        "bytes",
                        manifest_bytes,
                        0o644,
                    ),
                    (
                        f"{CANDIDATE_ID}/SHA256SUMS",
                        "bytes",
                        sums_bytes,
                        0o644,
                    ),
                ]
            )
            for name, kind, value, mode in sorted(members, key=lambda item: item[0]):
                if kind == "file":
                    add_file(archive, name, value, mode)
                else:
                    add_bytes(archive, name, value, mode)
        with temporary.open("rb") as source, archive_path.open("xb") as destination:
            shutil.copyfileobj(source, destination, 1024 * 1024)
    finally:
        if temporary.exists():
            temporary.unlink()

    archive_sha = sha256_file(archive_path)
    sidecar = output / f"{ARCHIVE_NAME}.sha256"
    sidecar.write_text(f"{archive_sha}  {ARCHIVE_NAME}\n", encoding="ascii", newline="\n")
    receipt = {
        "schema": "rootscope.competition-runtime-build-receipt.v2",
        "candidate_id": CANDIDATE_ID,
        "archive": {
            "path_relative_to_adventurex": archive_path.relative_to(root).as_posix(),
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha,
            "format": "USTAR_UNCOMPRESSED_DETERMINISTIC",
        },
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "composition_root_sha256": composition,
        "file_count": len(files),
        "build_gates": {
            "rag_structural_status": rag_audits["structural"]["status"],
            "rag_fts5_bm25_retrieval_status": (
                rag_audits["fts5_bm25_retrieval"]["status"]
            ),
        },
        "authority": authority,
    }
    (output / "release_build_receipt.json").write_bytes(canonical_json(receipt))
    return receipt


def parser() -> argparse.ArgumentParser:
    adventurex = Path(__file__).resolve().parents[2]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--adventurex-root", type=Path, default=adventurex)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=adventurex / "output/releases" / CANDIDATE_ID,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(
        json.dumps(
            build(args.adventurex_root, args.output_dir),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
