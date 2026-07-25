# Architecture

## Authority separation

RootScope is organized around evidence and authority rather than around one
large model.

```mermaid
flowchart TB
    subgraph READ["Read-only intelligence"]
      V["RootSight visual evidence"]
      K["RAG2 retrieved evidence"]
      L["RootMind explanation"]
      O["RootScope-Ω analysis"]
    end

    subgraph DET["Deterministic boundary"]
      S["Safety Compiler"]
      A["Plant2Action proposal"]
      C["Claim Ledger"]
    end

    subgraph PHYS["Private physical executor"]
      M["STM32 safety state machine"]
      X["Probe / relay / pump"]
    end

    V --> S
    K --> L
    L --> C
    O --> S
    S --> A
    A -. "bounded private protocol" .-> M
    M --> X
    X -. "receipt / observed state" .-> C
    L -.-|"no write authority"| X
```

The dotted boundary between `ActionProposal` and the STM32 is intentionally not
implemented in this repository.

## Public data contract

`EvidenceBundle` represents only the minimum public signals needed to explain
the safety pattern:

- semantic label;
- independent geometric label;
- image quality state;
- OOD state;
- evidence freshness;
- lower-controller safety state.

`ActionProposal` contains an abstract tier and reason codes. It explicitly keeps
`hardware_command = null`.

## Fail-closed ordering

1. Reject an unknown class.
2. Reject inadequate image quality.
3. Reject OOD input.
4. Reject stale evidence.
5. Reject unsafe device state.
6. Reject missing geometry verification.
7. Reject semantic/geometric disagreement.
8. Hold non-target scenes.
9. Only then emit an abstract tier.

The public confidence field is recorded but no production confidence threshold
is released. High confidence never overrides an independent safety failure.

