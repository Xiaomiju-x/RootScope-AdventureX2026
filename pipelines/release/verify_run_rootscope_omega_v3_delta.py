#!/usr/bin/env python3
"""Verify a RootScope-Ω v3 delta tree and optionally run one pure-CPU smoke.

This helper is deliberately incapable of installing services or touching
network, camera, serial, GPIO, STM32, or pump interfaces.  The optional smoke
only imports the packaged evidence core and evaluates one sealed fixture in
memory.  It does not grant execution authority and it does not write a receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True


CANDIDATE_SCHEMA = "rootscope.omega-v3-delta-candidate.v1"
CANDIDATE_ID = "rootscope_omega_v3_delta_candidate_v1"
BASE_SHA256 = "e6627685170252004d118bf77a690a9f89ad3afa274910697554d0f5cc8c3ebb"
BASE_BYTES = 696_832_000


class DeltaVerificationError(ValueError):
    """The extracted delta does not match its frozen manifest."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise DeltaVerificationError(f"unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DeltaVerificationError(f"unsafe relative path: {value!r}")
    return path


def _load_json(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise DeltaVerificationError(f"non-finite JSON constant: {value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, Mapping):
        raise DeltaVerificationError(f"{path.name} must contain one JSON object")
    return payload


def _parse_sums(path: Path) -> Mapping[str, str]:
    result: dict[str, str] = {}
    lines = path.read_text(encoding="ascii").splitlines()
    if lines != sorted(lines):
        raise DeltaVerificationError("SHA256SUMS must be bytewise sorted")
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise DeltaVerificationError("malformed SHA256SUMS line")
        digest, relative = line[:64], line[66:]
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise DeltaVerificationError("malformed SHA-256 digest")
        _safe_relative(relative)
        if relative in result:
            raise DeltaVerificationError(f"duplicate SHA256SUMS path: {relative}")
        result[relative] = digest
    return result


def verify_extracted_delta(bundle_root: Path) -> Mapping[str, Any]:
    """Verify exact files, hashes, immutable-base reference, and truth bounds."""

    root = bundle_root.resolve(strict=True)
    if not root.is_dir() or root.name != CANDIDATE_ID:
        raise DeltaVerificationError(
            f"bundle root must be the extracted {CANDIDATE_ID} directory"
        )
    manifest_path = root / "candidate_manifest.json"
    sums_path = root / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        raise DeltaVerificationError("candidate manifest or SHA256SUMS is missing")
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != CANDIDATE_SCHEMA:
        raise DeltaVerificationError("candidate schema mismatch")
    if manifest.get("candidate_id") != CANDIDATE_ID:
        raise DeltaVerificationError("candidate id mismatch")

    base = manifest.get("immutable_base_v2")
    if not isinstance(base, Mapping):
        raise DeltaVerificationError("immutable v2 base reference is missing")
    if (
        base.get("sha256") != BASE_SHA256
        or base.get("bytes") != BASE_BYTES
        or base.get("bundled_in_delta") is not False
        or base.get("immutable_reference_only") is not True
    ):
        raise DeltaVerificationError("immutable v2 base contract mismatch")

    qualification = manifest.get("qualification")
    authority = manifest.get("authority")
    if not isinstance(qualification, Mapping) or not isinstance(authority, Mapping):
        raise DeltaVerificationError("qualification or authority boundary is missing")
    for name in (
        "bpu_plant_model_qualified",
        "readonly_llm_long_run_qualified",
        "physical_closure",
        "production_integration_allowed",
    ):
        if qualification.get(name) is not False:
            raise DeltaVerificationError(f"qualification.{name} must remain false")
    if qualification.get("selected_bin") is not None:
        raise DeltaVerificationError("qualification.selected_bin must remain null")
    if not authority or any(value is not False for value in authority.values()):
        raise DeltaVerificationError("every authority field must remain exactly false")

    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise DeltaVerificationError("manifest files must be a non-empty array")
    record_paths: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise DeltaVerificationError("manifest file record must be an object")
        relative = str(record.get("path", ""))
        _safe_relative(relative)
        if relative in record_paths:
            raise DeltaVerificationError(f"duplicate manifest path: {relative}")
        record_paths.add(relative)
        source = root / Path(*PurePosixPath(relative).parts)
        if source.is_symlink() or not source.is_file():
            raise DeltaVerificationError(f"missing or unsafe payload: {relative}")
        if source.stat().st_size != record.get("bytes"):
            raise DeltaVerificationError(f"payload size mismatch: {relative}")
        if _sha256_file(source) != record.get("sha256"):
            raise DeltaVerificationError(f"payload hash mismatch: {relative}")

    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise DeltaVerificationError(f"symlink is forbidden: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    expected_files = record_paths | {"candidate_manifest.json", "SHA256SUMS"}
    if actual_files != expected_files:
        raise DeltaVerificationError(
            f"extracted file coverage mismatch: extra={sorted(actual_files - expected_files)}, "
            f"missing={sorted(expected_files - actual_files)}"
        )

    sums = _parse_sums(sums_path)
    covered = expected_files - {"SHA256SUMS"}
    if set(sums) != covered:
        raise DeltaVerificationError("SHA256SUMS does not exactly cover candidate files")
    for relative, expected_digest in sums.items():
        source = root / Path(*PurePosixPath(relative).parts)
        if _sha256_file(source) != expected_digest:
            raise DeltaVerificationError(f"SHA256SUMS mismatch: {relative}")

    return {
        "schema": "rootscope.omega-v3-delta-board-verification.v1",
        "status": "PASS_HASHES_ZERO_AUTHORITY_NOT_PHYSICAL_QUALIFICATION",
        "candidate_id": CANDIDATE_ID,
        "manifest_sha256": _sha256_file(manifest_path),
        "files_verified": len(covered),
        "pure_cpu_smoke_executed": False,
        "authority": {
            "network_touched": False,
            "camera_opened": False,
            "serial_opened": False,
            "gpio_opened": False,
            "pump_touched": False,
            "systemd_touched": False,
            "execution_authority": False,
            "physical_authority": False,
            "physical_closure": False,
        },
    }


def run_pure_cpu_smoke(bundle_root: Path) -> Mapping[str, Any]:
    """Exercise one deterministic evidence case without external interfaces."""

    verification = dict(verify_extracted_delta(bundle_root))
    rootscope = bundle_root.resolve(strict=True) / "rootscope"
    sys.path.insert(0, str(rootscope))
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from app.omega import (  # type: ignore[import-not-found]
            EvidenceKind,
            EvidenceMode,
            EvidenceNode,
            EvidenceVerdict,
        )
        from app.omega_runtime.digital_twin import (  # type: ignore[import-not-found]
            TwinCaseInput,
        )
        from app.omega_runtime.evidence_pipeline import (  # type: ignore[import-not-found]
            build_evidence_context,
        )

        node = EvidenceNode.create(
            node_id="board-pure-cpu-smoke",
            kind=EvidenceKind.SOURCE,
            verdict=EvidenceVerdict.PASS,
            mode=EvidenceMode.SEALED_REPLAY,
            source_id="delta-board-helper",
            observed_at_ms=0,
            payload={"fixture": "pure_cpu_zero_authority"},
        )
        case = TwinCaseInput.from_mapping(
            {
                "camera_quality_ok": True,
                "ood_detected": False,
                "evidence_fresh": True,
                "payload_hash_valid": True,
                "firmware_connected": True,
                "estop_clear": True,
                "ack_ok": True,
                "target_mass_mg": 100,
                "measured_mass_loss_mg": 100,
                "tolerance_mg": 5,
                "target_wetting_score": 0.8,
                "target_wetting_threshold": 0.7,
                "neighbor_wetting_score": 0.1,
                "neighbor_spill_threshold": 0.3,
            }
        )
        context = build_evidence_context("BOARD_PURE_CPU_SMOKE", case)
        if node.authority.execution_authority is not False:
            raise DeltaVerificationError("evidence node unexpectedly gained authority")
        if context.belief.authority.execution_authority is not False:
            raise DeltaVerificationError("belief state unexpectedly gained authority")
        for digest in (
            node.content_sha256,
            context.evidence_dag_root,
            context.belief_state_hash,
            context.failure_core_hash,
            context.rb_voe_plan_hash,
        ):
            if len(digest) != 64:
                raise DeltaVerificationError("pure-CPU smoke produced an invalid digest")
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        try:
            sys.path.remove(str(rootscope))
        except ValueError:
            pass

    verification.update(
        {
            "status": "PASS_PURE_CPU_SEALED_FIXTURE_ZERO_AUTHORITY",
            "pure_cpu_smoke_executed": True,
            "evidence_node_sha256": node.content_sha256,
            "evidence_dag_root": context.evidence_dag_root,
            "belief_state_hash": context.belief_state_hash,
            "failure_core_hash": context.failure_core_hash,
            "rb_voe_plan_hash": context.rb_voe_plan_hash,
        }
    )
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--verify-only", action="store_true")
    modes.add_argument("--run-pure-cpu-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        run_pure_cpu_smoke(args.bundle_root)
        if args.run_pure_cpu_smoke
        else verify_extracted_delta(args.bundle_root)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
