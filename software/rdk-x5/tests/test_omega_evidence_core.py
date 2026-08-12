"""Contract and integration tests for the RootScope-Ω evidence core."""

from __future__ import annotations

import ast
import dataclasses
import json
import unittest
from pathlib import Path

from app.omega import (
    AuthorityBoundary,
    BeliefState,
    BeliefUpdateError,
    BoundedMeasurement,
    CalibrationLevel,
    ContinuousEstimate,
    CoreStatus,
    CounterfactualFailureCore,
    EvidenceActionType,
    EvidenceDAG,
    EvidenceDagError,
    EvidenceKind,
    EvidenceMode,
    EvidenceRecord,
    EvidenceVerdict,
    FailureCorePolicy,
    FailureMode,
    OmegaContractError,
    RbVoePlan,
    RbVoePlanner,
    default_evidence_actions,
)


def record(
    node_id: str,
    kind: EvidenceKind,
    verdict: EvidenceVerdict = EvidenceVerdict.PASS,
    *,
    parents: tuple[str, ...] = (),
    at: int = 1,
    **payload,
) -> EvidenceRecord:
    return EvidenceRecord.create(
        node_id=node_id,
        kind=kind,
        verdict=verdict,
        mode=EvidenceMode.SEALED_REPLAY,
        source_id="locked-replay",
        observed_at_ms=at,
        payload=payload,
        parents=parents,
    )


def likelihoods(
    *,
    target: FailureMode = FailureMode.CAMERA_QUALITY,
    target_value: float = 0.9,
    other_value: float = 0.1,
) -> dict[FailureMode, float]:
    return {
        mode: target_value if mode is target else other_value for mode in FailureMode
    }


def perception_dag(*, conflict: bool = False, safety_fail: bool = False) -> EvidenceDAG:
    dag = EvidenceDAG()
    dag.add(record("quality-001", EvidenceKind.QUALITY, label="clear"))
    dag.add(record("semantic-001", EvidenceKind.SEMANTIC, label="low_shrub"))
    dag.add(
        record(
            "geometry-001",
            EvidenceKind.GEOMETRY,
            label="young_tree" if conflict else "low_shrub",
        )
    )
    dag.add(
        record(
            "safety-001",
            EvidenceKind.SAFETY,
            EvidenceVerdict.FAIL if safety_fail else EvidenceVerdict.PASS,
            interlocks="fail" if safety_fail else "pass",
        )
    )
    return dag


class StrictEvidenceContractTests(unittest.TestCase):
    def test_machine_readable_schema_is_strict_and_complete(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (
                root
                / "configs"
                / "omega"
                / "rootscope_omega_contracts.schema.json"
            ).read_text(encoding="utf-8")
        )
        definitions = payload["$defs"]
        self.assertEqual(
            {
                "actionEvaluation",
                "authority",
                "beliefState",
                "continuousEstimate",
                "evidenceAction",
                "evidenceNode",
                "failureCore",
                "failureProbabilities",
                "failureSignal",
                "planBranch",
                "rbVoePlan",
                "safeToken",
                "sha256",
            },
            set(definitions),
        )
        for name in (
            "authority",
            "beliefState",
            "continuousEstimate",
            "evidenceNode",
            "failureCore",
            "failureSignal",
            "evidenceAction",
            "failureProbabilities",
            "planBranch",
            "actionEvaluation",
            "rbVoePlan",
        ):
            self.assertIs(definitions[name]["additionalProperties"], False)

    def test_evidence_package_has_no_hardware_or_network_imports(self) -> None:
        package = Path(__file__).resolve().parents[1] / "app" / "omega"
        forbidden = {
            "can",
            "cv2",
            "ftplib",
            "gpiozero",
            "http",
            "paramiko",
            "requests",
            "rclpy",
            "serial",
            "smbus",
            "socket",
            "spidev",
            "subprocess",
            "urllib",
        }
        imported: set[str] = set()
        for source in package.glob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(
                        alias.name.partition(".")[0] for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    imported.add((node.module or "").partition(".")[0])
        self.assertEqual(forbidden & imported, set())

    def test_authority_must_be_exact_false_booleans(self) -> None:
        with self.assertRaises(OmegaContractError):
            AuthorityBoundary(execution_authority=True)
        with self.assertRaises(OmegaContractError):
            AuthorityBoundary(serial_write=0)  # type: ignore[arg-type]

    def test_payload_cannot_smuggle_authority(self) -> None:
        with self.assertRaises(OmegaContractError):
            record(
                "quality-001",
                EvidenceKind.QUALITY,
                execution_authority=False,
            )

    def test_evidence_round_trip_is_hash_bound_and_extra_keys_fail(self) -> None:
        node = record("quality-001", EvidenceKind.QUALITY, reason="glare")
        self.assertEqual(EvidenceRecord.from_dict(node.to_dict()), node)
        tampered = node.to_dict()
        tampered["payload"]["reason"] = "clear"
        with self.assertRaises(OmegaContractError):
            EvidenceRecord.from_dict(tampered)
        extra = node.to_dict()
        extra["unexpected"] = False
        with self.assertRaises(OmegaContractError):
            EvidenceRecord.from_dict(extra)

    def test_nonfinite_and_excessive_payloads_fail(self) -> None:
        with self.assertRaises(OmegaContractError):
            record("quality-001", EvidenceKind.QUALITY, score=float("nan"))
        with self.assertRaises(OmegaContractError):
            record("quality-001", EvidenceKind.QUALITY, values=list(range(257)))


class EvidenceDagTests(unittest.TestCase):
    def test_unknown_parent_is_atomic_and_same_node_is_idempotent(self) -> None:
        dag = EvidenceDAG()
        orphan = record(
            "semantic-001",
            EvidenceKind.SEMANTIC,
            parents=("missing-001",),
            label="grass_clump",
        )
        before = dag.root_sha256
        with self.assertRaises(EvidenceDagError):
            dag.add(orphan)
        self.assertEqual(len(dag), 0)
        self.assertEqual(dag.root_sha256, before)

        root = record("quality-001", EvidenceKind.QUALITY, label="clear")
        self.assertTrue(dag.add(root))
        self.assertFalse(dag.add(root))
        conflicting = record(
            "quality-001", EvidenceKind.QUALITY, EvidenceVerdict.FAIL, label="blur"
        )
        with self.assertRaises(EvidenceDagError):
            dag.add(conflicting)
        self.assertEqual(len(dag), 1)

    def test_snapshot_root_is_independent_of_sibling_insertion_order(self) -> None:
        root = record("quality-001", EvidenceKind.QUALITY, label="clear")
        left = record(
            "semantic-001",
            EvidenceKind.SEMANTIC,
            parents=(root.node_id,),
            label="low_shrub",
        )
        right = record(
            "geometry-001",
            EvidenceKind.GEOMETRY,
            parents=(root.node_id,),
            label="low_shrub",
        )
        dag_a = EvidenceDAG((root, left, right))
        dag_b = EvidenceDAG((root, right, left))
        self.assertEqual(dag_a.root_sha256, dag_b.root_sha256)
        self.assertEqual(dag_a.validate(), dag_b.validate())
        self.assertEqual(dag_a.ancestors(left.node_id), (root.node_id,))


class HybridBeliefTests(unittest.TestCase):
    def test_posterior_normalizes_and_binds_evidence(self) -> None:
        dag = EvidenceDAG()
        node = record(
            "quality-001", EvidenceKind.QUALITY, EvidenceVerdict.FAIL, reason="glare"
        )
        dag.add(node)
        belief = BeliefState.create()
        updated = belief.update(
            dag=dag,
            evidence_node_id=node.node_id,
            likelihoods=likelihoods(),
        )
        self.assertEqual(updated.revision, 1)
        self.assertEqual(updated.evidence_node_ids, (node.node_id,))
        self.assertAlmostEqual(sum(updated.probability_map.values()), 1.0, places=12)
        self.assertGreater(
            updated.probability(FailureMode.CAMERA_QUALITY),
            belief.probability(FailureMode.CAMERA_QUALITY),
        )
        self.assertEqual(BeliefState.from_dict(updated.to_dict()), updated)

    def test_bounded_continuous_update_intersects_not_invents_precision(self) -> None:
        dag = EvidenceDAG()
        node = record("mass-001", EvidenceKind.MASS, mass_mg=1000)
        dag.add(node)
        belief = BeliefState.create(
            continuous=(
                ContinuousEstimate(
                    name="delivered_mass_mg",
                    mean=1000,
                    variance=100,
                    lower=970,
                    upper=1030,
                    hard_lower=0,
                    hard_upper=5000,
                ),
            ),
            calibration_level=CalibrationLevel.INTERVAL_ONLY,
        )
        updated = belief.update(
            dag=dag,
            evidence_node_id=node.node_id,
            likelihoods={mode: 1.0 for mode in FailureMode},
            measurements=(
                BoundedMeasurement(
                    name="delivered_mass_mg",
                    value=1010,
                    variance=25,
                    lower=995,
                    upper=1020,
                    hard_lower=0,
                    hard_upper=5000,
                ),
            ),
        )
        estimate = updated.continuous_map["delivered_mass_mg"]
        self.assertEqual((estimate.lower, estimate.upper), (995.0, 1020.0))
        self.assertLess(estimate.variance, 100)

    def test_disjoint_interval_and_missing_node_fail_without_mutation(self) -> None:
        dag = EvidenceDAG()
        belief = BeliefState.create(
            continuous=(
                ContinuousEstimate(
                    name="wetting_fraction",
                    mean=0.4,
                    variance=0.01,
                    lower=0.3,
                    upper=0.5,
                    hard_lower=0,
                    hard_upper=1,
                ),
            )
        )
        before = belief.state_sha256
        with self.assertRaises(EvidenceDagError):
            belief.update(
                dag=dag,
                evidence_node_id="missing-001",
                likelihoods=likelihoods(),
            )
        node = record("wetting-001", EvidenceKind.WETTING, fraction=0.9)
        dag.add(node)
        with self.assertRaises(BeliefUpdateError):
            belief.update(
                dag=dag,
                evidence_node_id=node.node_id,
                likelihoods=likelihoods(),
                measurements=(
                    BoundedMeasurement(
                        name="wetting_fraction",
                        value=0.9,
                        variance=0.01,
                        lower=0.8,
                        upper=1.0,
                        hard_lower=0,
                        hard_upper=1,
                    ),
                ),
            )
        self.assertEqual(belief.state_sha256, before)


class FailureCoreAndPlannerTests(unittest.TestCase):
    def test_cross_path_conflict_requests_evidence_without_authority(self) -> None:
        dag = perception_dag(conflict=True)
        belief = BeliefState.create()
        core = CounterfactualFailureCore().analyze(dag, belief)
        self.assertEqual(core.status, CoreStatus.NEEDS_EVIDENCE)
        self.assertIn(
            "perception-cross-path-conflict",
            {signal.signal_id for signal in core.signals},
        )
        self.assertIn(
            EvidenceActionType.REQUEST_OPERATOR_REVIEW, core.requested_actions
        )
        self.assertFalse(core.authority.execution_authority)
        plan = RbVoePlanner().plan(
            dag=dag, belief=belief, failure_core=core, horizon=2
        )
        self.assertNotEqual(plan.selected_action, EvidenceActionType.HOLD)
        self.assertFalse(plan.authority.execution_authority)
        self.assertEqual(len(plan.evaluation.branches), 3)  # type: ignore[union-attr]

    def test_safety_failure_is_blocking_and_planner_holds(self) -> None:
        dag = perception_dag(safety_fail=True)
        belief = BeliefState.create()
        core = CounterfactualFailureCore().analyze(dag, belief)
        self.assertEqual(core.status, CoreStatus.BLOCKING)
        plan = RbVoePlanner().plan(
            dag=dag, belief=belief, failure_core=core, horizon=2
        )
        self.assertEqual(plan.action, "HOLD")
        self.assertEqual(plan.status, "HOLD_BLOCKING")
        self.assertIsNone(plan.evaluation)

    def test_ack_requires_mass_counterfactual(self) -> None:
        policy = FailureCorePolicy(required_kinds=())
        dag = EvidenceDAG()
        dag.add(record("ack-001", EvidenceKind.ACK, frame="ack"))
        belief = BeliefState.create()
        core = CounterfactualFailureCore(policy).analyze(dag, belief)
        self.assertIn(
            "mass-after-ack-missing", {signal.signal_id for signal in core.signals}
        )
        self.assertIn(EvidenceActionType.REWEIGH, core.requested_actions)

    def test_h2_is_deterministic_and_not_worse_than_h1_objective(self) -> None:
        dag = perception_dag(conflict=True)
        belief = BeliefState.create()
        core = CounterfactualFailureCore().analyze(dag, belief)
        planner = RbVoePlanner()
        h1 = planner.plan(dag=dag, belief=belief, failure_core=core, horizon=1)
        h2a = planner.plan(dag=dag, belief=belief, failure_core=core, horizon=2)
        h2b = planner.plan(dag=dag, belief=belief, failure_core=core, horizon=2)
        self.assertEqual(h2a, h2b)
        self.assertEqual(h2a.plan_sha256, h2b.plan_sha256)
        self.assertGreaterEqual(h2a.value_of_evidence, h1.value_of_evidence - 1e-12)
        self.assertTrue(
            all(
                branch.second_action
                in set(EvidenceActionType)
                for branch in h2a.evaluation.branches  # type: ignore[union-attr]
            )
        )

    def test_action_and_plan_round_trip_reject_nested_unknown_keys(self) -> None:
        action = default_evidence_actions()[0]
        self.assertEqual(type(action).from_dict(action.to_dict()), action)

        dag = perception_dag(conflict=True)
        belief = BeliefState.create()
        core = CounterfactualFailureCore().analyze(dag, belief)
        plan = RbVoePlanner().plan(
            dag=dag, belief=belief, failure_core=core, horizon=2
        )
        self.assertEqual(RbVoePlan.from_dict(plan.to_dict()), plan)
        tampered = plan.to_dict()
        self.assertIsNotNone(tampered["evaluation"])
        tampered["evaluation"]["branches"][0]["unexpected"] = False
        with self.assertRaises(OmegaContractError):
            RbVoePlan.from_dict(tampered)

    def test_planner_rejects_stale_core_binding_and_invalid_horizon(self) -> None:
        dag = perception_dag(conflict=True)
        belief = BeliefState.create()
        core = CounterfactualFailureCore().analyze(dag, belief)
        dag.add(record("ood-001", EvidenceKind.OOD, score=0.9))
        planner = RbVoePlanner()
        with self.assertRaises(OmegaContractError):
            planner.plan(dag=dag, belief=belief, failure_core=core, horizon=2)
        fresh = CounterfactualFailureCore().analyze(dag, belief)
        with self.assertRaises(OmegaContractError):
            planner.plan(dag=dag, belief=belief, failure_core=fresh, horizon=3)

    def test_clear_core_holds(self) -> None:
        dag = perception_dag()
        belief = BeliefState.create()
        core = CounterfactualFailureCore().analyze(dag, belief)
        self.assertEqual(core.status, CoreStatus.CLEAR)
        plan = RbVoePlanner().plan(
            dag=dag, belief=belief, failure_core=core, horizon=1
        )
        self.assertEqual(plan.status, "HOLD_CLEAR")
        self.assertEqual(plan.action, "HOLD")


if __name__ == "__main__":
    unittest.main()
