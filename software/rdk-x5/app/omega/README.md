# RootScope-Ω Evidence Core v1

This package is a deterministic, dependency-free, read-only planning core. It
does not import camera, serial, F407, state-machine, service, network, or pump
modules. Every evidence record, DAG snapshot, belief state, failure core, and
RB-VoE plan carries a strict authority capsule whose fields are all exactly
`false`.

Stable integration API:

```python
from app.omega import *

dag = EvidenceDAG()
dag.add(EvidenceRecord.create(
    node_id="quality-001",
    kind=EvidenceKind.QUALITY,
    verdict=EvidenceVerdict.FAIL,
    mode=EvidenceMode.SEALED_REPLAY,
    source_id="locked-replay-1",
    observed_at_ms=1,
    payload={"reason": "glare"},
))

belief = BeliefState.create()
core = CounterfactualFailureCore().analyze(dag, belief)
plan = RbVoePlanner().plan(
    dag=dag,
    belief=belief,
    failure_core=core,
    horizon=2,
)

print(dag.root_sha256)
print(belief.belief_hash)
print(core.failure_core_hash)
print(plan.action)
```

`plan.action` is only a proposed evidence-acquisition action. `HOLD` is
returned for a blocking failure core, a clear core, a failed risk bound, or
non-positive Value of Evidence. There is deliberately no execution method.

The matching JSON Schema bundle is
`configs/omega/rootscope_omega_contracts.schema.json`. Python constructors and
`from_dict` methods additionally enforce canonical ordering, exact key sets,
finite numbers, probability normalization, content hashes, parent existence,
cross-object hash binding, and all-false authority.
