"""Run the five locked RootScope-Ω cases through the complete CPU advisory chain."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    AuthorityFlags,
    DecisionReceipt,
    RuntimeMode,
    TruthRibbon,
    canonical_sha256,
)
from .digital_twin import TwinCaseInput, evaluate_case
from .evidence_pipeline import build_evidence_context
from .knowledge_pipeline import run_knowledge_roles
from .profiles import EdgeProfileRegistry, ResourceSnapshot


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _validate_document(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "release_id",
        "execution_mode",
        "execution_authority",
        "cases",
    }
    if set(payload) != expected:
        raise ValueError("locked replay document has unknown or missing fields")
    if payload["schema_version"] != "rootscope.omega.locked-replay-cases.v1":
        raise ValueError("locked replay schema mismatch")
    if payload["execution_mode"] != "SIMULATION":
        raise ValueError("locked replay must remain SIMULATION")
    if payload["execution_authority"] is not False:
        raise ValueError("locked replay execution_authority must be false")
    if not isinstance(payload["cases"], list) or len(payload["cases"]) != 5:
        raise ValueError("exactly five locked cases are required")


def _expected_match(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> bool:
    expected_keys = {
        "safety_decision",
        "evidence_action",
        "terminal_state",
        "completion_claim",
    }
    if set(expected) != expected_keys:
        raise ValueError("case expected projection contract changed")
    return all(actual[key] == expected[key] for key in expected_keys)


def run_locked_replay(
    *,
    cases_path: Path,
    profiles_path: Path,
    corpus_path: Path,
) -> Mapping[str, Any]:
    document = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    _validate_document(document)
    release_id = document["release_id"]
    registry = EdgeProfileRegistry.from_file(profiles_path)
    resources = ResourceSnapshot(
        available_memory_mib=2048,
        cpu_temperature_c=55.0,
        bpu_model_qualified=False,
        local_llm_available=False,
        remote_shadow_available=False,
    )
    backend = registry.select(
        "LOCAL_HYBRID",
        resources,
        runtime_mode=RuntimeMode.SIMULATION,
        release_id=release_id,
    )
    authority = AuthorityFlags()
    results = []
    for index, item in enumerate(document["cases"], start=1):
        if set(item) != {"case_id", "description", "inputs", "expected"}:
            raise ValueError("locked case has unknown or missing fields")
        case_id = item["case_id"]
        case = TwinCaseInput.from_mapping(item["inputs"])
        twin = evaluate_case(case)
        evidence = build_evidence_context(case_id, case)
        knowledge = run_knowledge_roles(
            case_id=case_id,
            evidence_refs=evidence.dag.node_ids,
            corpus_path=corpus_path,
        )
        receipt = DecisionReceipt(
            run_id=f"run-{case_id.lower()}",
            event_id=f"evaluate-{index:02d}",
            case_id=case_id,
            evidence_dag_root=evidence.evidence_dag_root,
            belief_state_hash=evidence.belief_state_hash,
            failure_core_hash=evidence.failure_core_hash,
            rb_voe_plan_hash=evidence.rb_voe_plan_hash,
            claim_ledger_root=knowledge.claim_ledger_root,
            projection=twin.projection,
            backend=backend,
            authority=authority,
            generated_at_utc=_now(),
        )
        warnings = [
            "SIMULATION_ONLY",
            "ZERO_EXECUTION_AUTHORITY",
            "NO_PHYSICAL_COMPLETION_CLAIM",
            "LLM_DETERMINISTIC_FALLBACK",
        ]
        if not backend.bpu_model_qualified:
            warnings.append("BPU_MODEL_NOT_QUALIFIED")
        ribbon = TruthRibbon(
            mode=RuntimeMode.SIMULATION,
            profile=backend.profile,
            backend_actual=backend.decision_backend_actual,
            evidence_state=(
                "FRESH"
                if case.evidence_fresh and case.payload_hash_valid
                else "STALE_OR_INVALID"
            ),
            evidence_fresh=case.evidence_fresh and case.payload_hash_valid,
            receipt_sha256=receipt.receipt_sha256,
            authority=authority,
            physical_completion_claim=False,
            warnings=tuple(warnings),
        )
        projection = twin.projection.to_dict()
        matched = _expected_match(item["expected"], projection)
        results.append(
            {
                "case_id": case_id,
                "description": item["description"],
                "expected_projection": item["expected"],
                "expected_projection_matched": matched,
                "twin_evaluation": twin.to_dict(),
                "evidence_dag": evidence.dag.snapshot().to_dict(),
                "belief_state": evidence.belief.to_dict(),
                "failure_core": evidence.failure_core.to_dict(),
                "rb_voe_h2": evidence.rb_voe_plan.to_dict(),
                "knowledge": knowledge.to_dict(),
                "decision_receipt": receipt.to_dict(),
                "truth_ribbon": ribbon.to_dict(),
            }
        )
    receipt_hashes = [
        result["decision_receipt"]["receipt_sha256"] for result in results
    ]
    matched_count = sum(
        bool(result["expected_projection_matched"]) for result in results
    )
    report = {
        "schema_version": "rootscope.omega.locked-replay-report.v1",
        "generated_at_utc": _now(),
        "release_id": release_id,
        "requested_profile": "LOCAL_HYBRID",
        "selected_backend": backend.to_dict(),
        "case_count": len(results),
        "matched_case_count": matched_count,
        "all_locked_cases_passed": matched_count == len(results),
        "case_receipts_root": canonical_sha256(receipt_hashes),
        "cases": results,
        "authority": authority.to_dict(),
        "runtime_boundary": {
            "hardware_touched": False,
            "network_touched": False,
            "camera_opened": False,
            "serial_opened": False,
            "pump_touched": False,
            "physical_completion_claim": False,
        },
    }
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=root / "configs" / "omega" / "locked_replay_cases.v1.json",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=root / "configs" / "omega" / "edge_profiles.v1.json",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=root / "configs" / "omega" / "field_knowledge.v1.md",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_locked_replay(
        cases_path=args.cases,
        profiles_path=args.profiles,
        corpus_path=args.corpus,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    print(text, end="")
    return 0 if report["all_locked_cases_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
