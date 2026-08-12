#!/usr/bin/env python3
"""Generate schema-valid PC-only resource and physical-simulation receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADVENTUREX = PROJECT_ROOT.parent
V3_ROOT = ADVENTUREX / "rootscope_v3"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.action_v3.contracts import ActionContractCompiler  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=V3_ROOT / "evaluations",
        help="Directory for generated receipts; tests must use an isolated directory.",
    )
    args = parser.parse_args()
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    process = psutil.Process(os.getpid())
    created = utc_now()
    resource = {
        "schema": "rootscope.v3.resource-evaluation.v1",
        "evaluation_id": "resource-pc-contract-20260724",
        "created_at_utc": created,
        "platform": "PC",
        "scenario": {
            "scenario_id": "PC_CONTRACT_AND_FAULT_MATRIX_ONLY",
            "components_active": [
                "resource_broker",
                "action_contract_compiler",
                "fault_matrix",
            ],
            "camera_live": False,
            "physical_execution": False,
        },
        "sampling": {
            "duration_seconds": 0.1,
            "interval_seconds": 0.1,
            "sample_count": 1,
        },
        "memory": {
            "mem_available_min_kib": int(memory.available // 1024),
            "cma_free_min_kib": None,
            "swap_used_max_kib": int(swap.used // 1024),
            "oom_kill_delta": 0,
        },
        "temperature": {
            "max_millicelsius": None,
            "thermal_throttle_observed": False,
        },
        "bpu": {
            "actual_inference_executed": False,
            "utilization_percent_max": None,
            "cma_leak_observed": False,
        },
        "processes": [
            {
                "process_id": "generate_v3_pc_evaluations",
                "observed_running": True,
                "rss_mib_max": process.memory_info().rss / (1024 * 1024),
            }
        ],
        "qualification": {
            "status": "NOT_EVALUATED",
            "mem_gate_kib": 512 * 1024,
            "cma_gate_kib": 128 * 1024,
            "claim_boundary": (
                "PC contract test only; no RDK X5 memory, CMA, temperature, "
                "BPU utilization, soak, or live qualification."
            ),
        },
        "authority": {
            "serial_opened": False,
            "gpio_touched": False,
            "pump_touched": False,
        },
    }
    compiler = ActionContractCompiler(
        release_sha256="1" * 64,
        config_sha256="2" * 64,
    )
    contract = compiler.compile(
        contract_id="pc-simulation-contract-1",
        sequence=0,
        boot_id="simulation-boot",
        evidence_root_sha256="3" * 64,
        plant_class="grass_clump",
        plant_confidence=0.9,
        ood_hold=False,
        target_zone="simulation-zone",
        proposed_volume_ml=15.0,
        evidence_fresh=True,
        interlocks_clear=True,
        reason_codes=["PC_FIXTURE_ONLY"],
    )
    physical = {
        "schema": "rootscope.v3.physical-loop-evaluation.v1",
        "trial_id": "physical-pc-simulation-20260724",
        "created_at_utc": created,
        "execution_mode": "SIMULATION",
        "hardware": {
            "x5_identity": None,
            "stm32_boot_id": None,
            "firmware_build_id": None,
            "pump_channel": None,
            "hx711_id": None,
        },
        "action_contract": {
            "contract_id": contract.contract_id,
            "contract_sha256": contract.sha256,
            "sequence": contract.sequence,
            "requested_mass_g": contract.proposed_volume_ml,
            "max_mass_g": contract.maximum_volume_ml,
        },
        "safety": {
            "operator_present": False,
            "estop_tested": False,
            "watchdog_tested": False,
            "link_loss_tested": False,
            "actuator_power_connected": False,
        },
        "actuation": {
            "serial_opened": False,
            "serial_write_count": 0,
            "ack_received": False,
            "ack_sequence": None,
            "pump_energized": False,
        },
        "feedback": {
            "mass_before_g": None,
            "mass_after_g": None,
            "mass_delta_g": None,
            "target_coverage": None,
            "neighbor_spill": None,
            "center_offset_ratio": None,
            "moisture_delta": None,
        },
        "outcome": {
            "status": "SIMULATED_ONLY",
            "mass_gate_passed": None,
            "wetting_gate_passed": None,
            "spill_gate_passed": None,
            "manual_repair_required": False,
        },
        "claim_boundary": (
            "PC schema/logic simulation only. No X5, STM32, USB-TTL, HX711, "
            "moisture sensor, camera, pump, water, ACK, or physical closure."
        ),
    }
    evaluation_root = args.output_root.resolve()
    resource_path = evaluation_root / "resource_pc_contract_20260724.json"
    physical_path = evaluation_root / "physical_loop_pc_simulation_20260724.json"
    write_json(resource_path, resource)
    write_json(physical_path, physical)
    print(
        json.dumps(
            {
                "status": "PASS_PC_ONLY",
                "resource": str(resource_path),
                "physical": str(physical_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
