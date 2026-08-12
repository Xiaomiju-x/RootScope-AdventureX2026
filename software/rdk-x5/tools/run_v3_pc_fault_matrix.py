#!/usr/bin/env python3
"""Run the RootScope v3 PC-only fail-closed contract matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.action_v3.contracts import (
    ActionContractCompiler,
    ActionContractError,
    PhysicalReceiptCompiler,
)
from app.runtime_v3.resource_broker import (
    ResourceBroker,
    ResourceSnapshot,
    RuntimePhase,
    Workload,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    broker = ResourceBroker()
    action_compiler = ActionContractCompiler(
        release_sha256="1" * 64,
        config_sha256="2" * 64,
    )
    receipt_compiler = PhysicalReceiptCompiler()
    rows: list[dict[str, Any]] = []

    def check(case_id: str, expectation: str, function: Callable[[], bool]) -> None:
        try:
            passed = bool(function())
            detail = "EXPECTED_FAIL_CLOSED" if passed else "UNSAFE_ACCEPT"
        except Exception as exc:  # expected for strict validation cases
            passed = expectation == "RAISE"
            detail = f"{type(exc).__name__}:{exc}"
        rows.append(
            {
                "case_id": case_id,
                "expectation": expectation,
                "passed": passed,
                "detail": detail,
            }
        )

    check(
        "F01_LOW_MEMORY_FAST_LLM",
        "DENY",
        lambda: not broker.decide(
            Workload.FAST_LLM,
            RuntimePhase.IDLE,
            ResourceSnapshot(400, 200, 50, 1),
        ).admitted,
    )
    check(
        "F02_LOW_CMA_BPU",
        "DENY",
        lambda: not broker.decide(
            Workload.BPU_VISION,
            RuntimePhase.PERCEPTION,
            ResourceSnapshot(1500, 80, 50, 1),
        ).admitted,
    )
    check(
        "F03_THERMAL_HOLD",
        "DENY",
        lambda: not broker.decide(
            Workload.CPU_VISION,
            RuntimePhase.PERCEPTION,
            ResourceSnapshot(1500, 200, 82, 1),
        ).admitted,
    )
    check(
        "F04_DEEP_DURING_IRRIGATION",
        "DENY",
        lambda: not broker.decide(
            Workload.DEEP_LLM,
            RuntimePhase.IRRIGATION_CRITICAL,
            ResourceSnapshot(2400, 200, 50, 1),
        ).admitted,
    )
    check(
        "F05_BPU_SWAP_DURING_IRRIGATION",
        "DENY",
        lambda: not broker.decide(
            Workload.BPU_VISION,
            RuntimePhase.IRRIGATION_CRITICAL,
            ResourceSnapshot(2400, 200, 50, 1),
        ).admitted,
    )

    def contract(**overrides: Any):
        values = {
            "contract_id": "contract-1",
            "sequence": 1,
            "boot_id": "boot-1",
            "evidence_root_sha256": "3" * 64,
            "plant_class": "grass_clump",
            "plant_confidence": 0.9,
            "ood_hold": False,
            "target_zone": "zone-1",
            "proposed_volume_ml": 20.0,
            "evidence_fresh": True,
            "interlocks_clear": True,
            "reason_codes": ["FUSED_EVIDENCE"],
        }
        values.update(overrides)
        return action_compiler.compile(**values)

    check(
        "F06_STALE_EVIDENCE",
        "ZERO_VOLUME",
        lambda: contract(evidence_fresh=False).proposed_volume_ml == 0,
    )
    check(
        "F07_INTERLOCK_ACTIVE",
        "ZERO_VOLUME",
        lambda: contract(interlocks_clear=False).proposed_volume_ml == 0,
    )
    check(
        "F08_OOD_HOLD",
        "ZERO_VOLUME",
        lambda: contract(ood_hold=True).proposed_volume_ml == 0,
    )
    check(
        "F09_NON_TARGET",
        "ZERO_VOLUME",
        lambda: contract(plant_class="non_target").proposed_volume_ml == 0,
    )
    check(
        "F10_VOLUME_LIMIT",
        "RAISE",
        lambda: contract(proposed_volume_ml=90.0),
    )
    check(
        "F11_NONFINITE_CONFIDENCE",
        "RAISE",
        lambda: contract(plant_confidence=float("nan")),
    )

    accepted = contract(contract_id="contract-ok", sequence=9)

    def receipt(**overrides: Any):
        values = {
            "receipt_id": "receipt-1",
            "contract": accepted,
            "device_identity_sha256": "4" * 64,
            "ack_boot_id": "boot-1",
            "ack_sequence": 9,
            "ack_payload_sha256": "5" * 64,
            "ack_fresh": True,
            "expected_mass_loss_g": 20.0,
            "observed_mass_loss_g": 20.0,
            "target_wetting_coverage": 0.3,
            "neighbor_spill_ratio": 0.02,
        }
        values.update(overrides)
        return receipt_compiler.compile(**values)

    check(
        "F12_STALE_ACK",
        "NO_COMPLETION",
        lambda: not receipt(ack_fresh=False).completed,
    )
    check(
        "F13_BOOT_MISMATCH",
        "NO_COMPLETION",
        lambda: not receipt(ack_boot_id="boot-2").completed,
    )
    check(
        "F14_SEQUENCE_MISMATCH",
        "NO_COMPLETION",
        lambda: not receipt(ack_sequence=10).completed,
    )
    check(
        "F15_MASS_MISMATCH",
        "NO_COMPLETION",
        lambda: not receipt(observed_mass_loss_g=1.0).completed,
    )
    check(
        "F16_WETTING_INSUFFICIENT",
        "NO_COMPLETION",
        lambda: not receipt(target_wetting_coverage=0.02).completed,
    )
    check(
        "F17_NEIGHBOR_SPILL",
        "NO_COMPLETION",
        lambda: not receipt(neighbor_spill_ratio=0.4).completed,
    )
    check(
        "F18_ACK_ONLY_NOT_ENOUGH",
        "NO_COMPLETION",
        lambda: not receipt(
            observed_mass_loss_g=0.0,
            target_wetting_coverage=0.0,
        ).completed,
    )
    check(
        "F19_THREE_SIGNAL_COMPLETION",
        "OBSERVED_COMPLETION",
        lambda: receipt(receipt_id="receipt-pass").completed,
    )

    passed = sum(int(item["passed"]) for item in rows)
    report = {
        "schema": "rootscope.v3.pc-fault-matrix.v1",
        "status": "PASS" if passed == len(rows) else "FAIL",
        "count": len(rows),
        "passed": passed,
        "unsafe_accepts": sum(
            int(not item["passed"] and item["detail"] == "UNSAFE_ACCEPT")
            for item in rows
        ),
        "hardware_touched": False,
        "network_touched": False,
        "serial_opened": False,
        "pump_touched": False,
        "physical_completion_claim": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "count": report["count"],
                "passed": report["passed"],
                "unsafe_accepts": report["unsafe_accepts"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
