"""Bind perception evidence, resource admission, and Plant2Action contracts.

The coordinator deliberately stops at a signed/hashable *proposal*.  The
future qualified STM32 bridge is outside this package and is the only
component that may ever own the serial writer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Sequence

from app.action_v3.contracts import ActionContractCompiler
from app.runtime_v3.resource_broker import (
    ResourceBroker,
    ResourceSnapshot,
    RuntimePhase,
    Workload,
)


@dataclass(frozen=True)
class PerceptionEvidence:
    plant_class: str
    confidence: float
    ood_hold: bool
    temporal_support: int
    target_zone: str
    evidence_root_sha256: str
    fresh: bool

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.confidence)):
            raise ValueError("confidence must be finite")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be within [0,1]")
        if not 1 <= int(self.temporal_support) <= 120:
            raise ValueError("temporal_support must be within [1,120]")


class RootScopeV3Coordinator:
    """Produce an auditable proposal and Truth Ribbon for one pre-action cycle."""

    def __init__(
        self,
        *,
        release_sha256: str,
        config_sha256: str,
        broker: ResourceBroker | None = None,
    ) -> None:
        self.broker = broker or ResourceBroker()
        self.compiler = ActionContractCompiler(
            release_sha256=release_sha256,
            config_sha256=config_sha256,
        )

    def pre_action_cycle(
        self,
        *,
        contract_id: str,
        sequence: int,
        boot_id: str,
        evidence: PerceptionEvidence,
        deterministic_volume_ml: float,
        interlocks_clear: bool,
        requested_backend: str,
        resources: ResourceSnapshot,
        upstream_reason_codes: Sequence[str],
    ) -> dict[str, Any]:
        workload = (
            Workload.BPU_VISION
            if requested_backend == "BPU"
            else Workload.CPU_VISION
        )
        admission = self.broker.decide(
            workload, RuntimePhase.PERCEPTION, resources
        )
        reasons = set(str(item) for item in upstream_reason_codes)
        if not admission.admitted:
            reasons.update(admission.reason_codes)
            reasons.add("VISION_BACKEND_FALLBACK_CPU")
        else:
            reasons.add(
                "BPU_SHADOW_EVIDENCE_AVAILABLE"
                if workload == Workload.BPU_VISION
                else "CPU_PRIMARY"
            )
        if evidence.temporal_support < 3:
            reasons.add("TEMPORAL_SUPPORT_LOW")
        effective_fresh = evidence.fresh and evidence.temporal_support >= 3
        contract = self.compiler.compile(
            contract_id=contract_id,
            sequence=sequence,
            boot_id=boot_id,
            evidence_root_sha256=evidence.evidence_root_sha256,
            plant_class=evidence.plant_class,
            plant_confidence=evidence.confidence,
            ood_hold=evidence.ood_hold,
            target_zone=evidence.target_zone,
            proposed_volume_ml=deterministic_volume_ml,
            evidence_fresh=effective_fresh,
            interlocks_clear=interlocks_clear,
            reason_codes=sorted(reasons),
        )
        payload = {
            "schema": "rootscope.v3.pre-action-cycle.v1",
            "perception": asdict(evidence),
            "resource_decision": admission.to_dict(),
            "action_contract": contract.payload(),
            "action_contract_sha256": contract.sha256,
            "truth_ribbon": {
                "primary_backend": (
                    workload.value if admission.admitted else Workload.CPU_VISION.value
                ),
                "bpu_qualified": False,
                "llm_authority": False,
                "proposal_only": True,
                "physical_completion_claim": False,
                "hold": contract.proposed_volume_ml == 0.0,
                "reason_codes": list(contract.reason_codes),
            },
            "authority": {
                "execution_authority": False,
                "serial_write": False,
                "gpio_write": False,
                "pump_command": False,
                "state_machine_write": False,
                "physical_completion": False,
            },
        }
        payload["cycle_sha256"] = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return payload


__all__ = ["PerceptionEvidence", "RootScopeV3Coordinator"]
