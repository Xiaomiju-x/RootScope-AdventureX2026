"""Evaluate the proposal-only DR-MPC kernel and fifteen fault mutations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import AuthorityFlags, canonical_sha256
from .dr_mpc import DrMpcScenario, solve_dr_mpc
from .fault_injection import run_fault_injection


def evaluate_algorithms(
    *,
    dr_mpc_path: Path,
    locked_cases_path: Path,
) -> Mapping[str, Any]:
    dr_document = json.loads(Path(dr_mpc_path).read_text(encoding="utf-8"))
    if set(dr_document) != {"schema_version", "scenarios"}:
        raise ValueError("DR-MPC scenario document has unknown or missing fields")
    if dr_document["schema_version"] != "rootscope.omega.dr-mpc-scenarios.v1":
        raise ValueError("DR-MPC scenario schema mismatch")
    proposals = []
    for item in dr_document["scenarios"]:
        if "expected_status" not in item:
            raise ValueError("DR-MPC scenario lacks expected_status")
        expected = item["expected_status"]
        scenario = DrMpcScenario.from_mapping(
            {key: value for key, value in item.items() if key != "expected_status"}
        )
        proposal = solve_dr_mpc(scenario)
        proposals.append(
            {
                "expected_status": expected,
                "expected_status_matched": proposal.status == expected,
                "proposal": proposal.to_dict(),
            }
        )
    case_document = json.loads(
        Path(locked_cases_path).read_text(encoding="utf-8")
    )
    normal = case_document["cases"][0]["inputs"]
    fault_report = run_fault_injection(normal)
    report = {
        "schema_version": "rootscope.omega.algorithm-evaluation.v1",
        "dr_mpc": {
            "scenario_count": len(proposals),
            "matched_count": sum(
                item["expected_status_matched"] for item in proposals
            ),
            "all_expected_statuses_matched": all(
                item["expected_status_matched"] for item in proposals
            ),
            "proposals": proposals,
        },
        "fault_injection": fault_report,
        "authority": AuthorityFlags().to_dict(),
        "runtime_boundary": {
            "hardware_touched": False,
            "network_touched": False,
            "serial_opened": False,
            "physical_command_count": 0,
            "proposal_only": True,
        },
    }
    passed = (
        report["dr_mpc"]["all_expected_statuses_matched"]
        and fault_report["passed"]
    )
    report = {**report, "passed": passed}
    return {**report, "report_sha256": canonical_sha256(report)}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dr-mpc",
        type=Path,
        default=root / "configs" / "omega" / "dr_mpc_scenarios.v1.json",
    )
    parser.add_argument(
        "--locked-cases",
        type=Path,
        default=root / "configs" / "omega" / "locked_replay_cases.v1.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_algorithms(
        dr_mpc_path=args.dr_mpc,
        locked_cases_path=args.locked_cases,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
