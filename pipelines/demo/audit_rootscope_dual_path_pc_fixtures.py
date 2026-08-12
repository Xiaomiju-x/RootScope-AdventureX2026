"""Audit the deterministic PC-only RootScope dual-path smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = (
    ("grass", "grass_clump", True),
    ("shrub", "low_shrub", True),
    ("young_tree", "young_tree", True),
    ("unknown_negative", "unknown", False),
)
REGISTRY_SHA256 = "f5328b66c6b385dd081c1a27577d92b5c63fc18dc00d815f7d3d2561bb68e29f"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(adventurex: Path) -> dict[str, Any]:
    adventurex = adventurex.resolve(strict=True)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    inputs: list[dict[str, Any]] = []
    for stem, expected_class, expected_consensus in EXPECTED:
        path = adventurex / "evidence" / f"rootscope_dual_path_{stem}_pc_simulated_frame_20260717.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        prefix = f"{stem}:"
        check(prefix + "schema", value.get("schema") == "rootscope.dual-path-demo.v1", value.get("schema"))
        check(prefix + "registry_sha", value.get("registry", {}).get("raw_sha256") == REGISTRY_SHA256, value.get("registry", {}).get("raw_sha256"))
        check(prefix + "semantic_class", value.get("semantic", {}).get("raw_top1_class") == expected_class, value.get("semantic", {}).get("raw_top1_class"))
        check(prefix + "consensus", value.get("experimental_consensus_passed") is expected_consensus, value.get("status"))
        expected_geometry = 1 if expected_consensus else 0
        check(prefix + "geometry_count", value.get("geometry", {}).get("contract_valid_pass_count") == expected_geometry, value.get("geometry", {}).get("contract_valid_pass_count"))
        check(prefix + "formal_gate_rejects", value.get("semantic", {}).get("formal_rejection_gate", {}).get("passed") is False, value.get("semantic", {}).get("formal_rejection_gate", {}).get("status"))
        check(prefix + "authority", all(flag is False for flag in value.get("authority", {}).values()), value.get("authority"))
        check(prefix + "claims", all(flag is False for flag in value.get("claims", {}).values()), value.get("claims"))
        check(prefix + "no_hardware", value.get("hardware_touched") is False, value.get("hardware_touched"))
        check(prefix + "no_network", value.get("network_touched") is False, value.get("network_touched"))
        inputs.append({
            "name": stem,
            "relative_path": str(path.relative_to(adventurex)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "expected_class": expected_class,
            "expected_consensus": expected_consensus,
            "observed_status": value.get("status"),
        })

    failures = [item for item in checks if not item["passed"]]
    report = {
        "schema": "rootscope.dual-path-pc-fixture-audit.v1",
        "status": "PASS_SIMULATED_ONLY" if not failures else "FAIL",
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failures),
        "checks_failed": len(failures),
        "checks": checks,
        "inputs": inputs,
        "claim_scope": "PC_SIMULATED_FRAME_SOFTWARE_SMOKE_ONLY_NOT_UVC_NOT_X5_NOT_ACCURACY",
        "authority": {
            "camera_qualified": False,
            "model_qualified": False,
            "x5_validated": False,
            "bpu_compiled": False,
            "execution_authority": False,
            "physical_authority": False,
            "physical_completion": False,
        },
    }
    output = adventurex / "evidence" / "rootscope_dual_path_pc_fixture_audit_20260717.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report["report_path"] = str(output)
    report["report_sha256"] = sha256_file(output)
    if failures:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adventurex", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    report = audit(args.adventurex)
    print(json.dumps({key: report[key] for key in ("status", "checks_total", "checks_passed", "checks_failed", "report_path", "report_sha256")}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
