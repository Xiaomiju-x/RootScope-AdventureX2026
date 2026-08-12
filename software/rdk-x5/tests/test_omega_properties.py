"""Deterministic property-style and fail-closed tests for RootScope-Ω."""

from __future__ import annotations

import dataclasses
import random
import unittest

from app.omega import (
    BeliefState,
    BeliefUpdateError,
    EvidenceDAG,
    EvidenceDagError,
    EvidenceKind,
    EvidenceMode,
    EvidenceRecord,
    EvidenceVerdict,
    FailureMode,
    OmegaContractError,
    default_evidence_actions,
)


def make_node(
    node_id: str, kind: EvidenceKind, parents: tuple[str, ...] = ()
) -> EvidenceRecord:
    return EvidenceRecord.create(
        node_id=node_id,
        kind=kind,
        verdict=EvidenceVerdict.PASS,
        mode=EvidenceMode.SIMULATION,
        source_id="property-fixture",
        observed_at_ms=1,
        payload={"label": "low_shrub"},
        parents=parents,
    )


class OmegaPropertyTests(unittest.TestCase):
    def test_random_topological_dags_validate_and_hash_stably(self) -> None:
        rng = random.Random(20260723)
        kinds = list(EvidenceKind)
        for case in range(40):
            dag = EvidenceDAG()
            records: list[EvidenceRecord] = []
            for index in range(1, rng.randint(3, 18)):
                parent_pool = [record.node_id for record in records]
                parent_count = rng.randint(0, min(3, len(parent_pool)))
                parents = tuple(sorted(rng.sample(parent_pool, parent_count)))
                node = make_node(
                    f"case-{case:02d}-node-{index:03d}",
                    rng.choice(kinds),
                    parents,
                )
                dag.add(node)
                records.append(node)
            snapshot = dag.validate()
            self.assertEqual(snapshot.node_count, len(records))
            self.assertEqual(snapshot.root_sha256, dag.root_sha256)
            self.assertEqual(len(snapshot.root_sha256), 64)

    def test_random_likelihood_updates_remain_finite_and_normalized(self) -> None:
        rng = random.Random(1701)
        dag = EvidenceDAG()
        node = make_node("property-evidence-001", EvidenceKind.QUALITY)
        dag.add(node)
        for _ in range(100):
            raw = {mode: rng.random() for mode in FailureMode}
            belief = BeliefState.create().update(
                dag=dag,
                evidence_node_id=node.node_id,
                likelihoods=raw,
            )
            probabilities = tuple(belief.probability_map.values())
            self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities))

    def test_zero_likelihood_and_incomplete_modes_fail_closed(self) -> None:
        belief = BeliefState.create()
        with self.assertRaises(BeliefUpdateError):
            belief.posterior_from_likelihoods({mode: 0.0 for mode in FailureMode})
        with self.assertRaises(BeliefUpdateError):
            belief.posterior_from_likelihoods({FailureMode.NORMAL: 1.0})

    def test_cycle_and_self_parent_are_unrepresentable(self) -> None:
        with self.assertRaises(OmegaContractError):
            make_node(
                "self-parent-001",
                EvidenceKind.QUALITY,
                parents=("self-parent-001",),
            )
        dag = EvidenceDAG()
        first = make_node(
            "future-parent-001",
            EvidenceKind.QUALITY,
            parents=("future-child-001",),
        )
        with self.assertRaises(EvidenceDagError):
            dag.add(first)
        self.assertEqual(len(dag), 0)

    def test_default_action_catalog_probabilities_are_complete(self) -> None:
        actions = default_evidence_actions()
        self.assertEqual(len(actions), 5)
        for action in actions:
            model = action.model_map
            for mode in FailureMode:
                self.assertAlmostEqual(
                    sum(model[outcome][mode] for outcome in model), 1.0, places=12
                )
            self.assertFalse(action.authority.execution_authority)
            self.assertFalse(action.authority.actuator_access)

    def test_action_model_mutation_is_rejected(self) -> None:
        action = default_evidence_actions()[0]
        model = list(action.observation_model)
        outcome, values = model[0]
        changed = list(values)
        changed[0] = (changed[0][0], changed[0][1] + 0.1)
        model[0] = (outcome, tuple(changed))
        with self.assertRaises(OmegaContractError):
            dataclasses.replace(action, observation_model=tuple(model))


if __name__ == "__main__":
    unittest.main()
