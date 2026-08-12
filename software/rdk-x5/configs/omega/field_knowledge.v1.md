# RootScope-Ω Field Knowledge v1

## K01 Product Boundary

RootScope is a fixed root-zone irrigation chamber, not a navigation vehicle. The field profile does not depend on LiDAR, SLAM, Nav2, a depth camera, chassis motion, or autonomous route planning. RDK X5 performs perception, evidence fusion, decision explanation, and deployment telemetry; the deterministic safety state machine remains the only future owner of physical task commands.

## K02 Zero-Authority AI Boundary

LLM, VLM, BPU Evidence Probe, retrieval, GraphRAG, RB-VoE, and DR-MPC are advisory components. They have no serial, GPIO, pump, state-machine, reset, tool-use, or physical execution authority. A malformed response, missing citation, stale evidence, prompt injection, resource shortage, or backend failure must fall back to deterministic templates or SAFE_CPU and must not create an actuator command.

## K03 Completion Evidence

An actuator ACK is necessary but never sufficient. A valid completion requires a fresh identity-bound ACK, mass loss within the frozen target tolerance, qualified target-zone wetting, and no neighbor-zone spill. ACK without mass loss indicates an empty tank, blockage, or metering fault. Normal mass loss with excessive neighbor wetting indicates leakage or hydraulic crosstalk and requires ABORTED_LOCKED.

## K04 Perception and OOD

The field perception chain combines image quality checks, semantic classification, geometric template evidence, Energy score, Mahalanobis distance, and a conformal prediction set. Occlusion, glare, disagreement, low support, or out-of-distribution input must produce HOLD, RECAPTURE, or operator review. Safety-critical OOD is never automatically accepted.

## K05 Evidence DAG and RB-VoE

Every observation is an immutable EvidenceRecord with provenance, freshness, a payload SHA-256, and zero authority. Evidence DAG roots, Hybrid Belief State hashes, Counterfactual Failure Core hashes, and H=1 or H=2 Risk-Bounded Value-of-Evidence plans are bound into a Decision Receipt. RB-VoE may recommend RECAPTURE, REWEIGH, WAIT, REVIEW, or HOLD, but the deterministic Safety Compiler owns the final advisory projection.

## K06 X5 Qualification State

The new RDK X5 is an aarch64 RDK X5 V1.0 board with Ubuntu 22.04 and hbm_runtime 3.0.9. The immutable RootScope v2 field bundle already contains an offline CPU ONNX core and a read-only Qwen2 0.5B model path. The existing BPU component is support-only and selected_bin remains null until a new candidate passes PC quantized replay and actual hbm_runtime golden replay. No camera, STM32, pump, or physical completion claim follows from software installation alone.

## K07 EdgeOS Resource Policy

SAFE_CPU uses deterministic CPU decisions, ONNX Runtime CPU vision, SQLite FTS5/BM25 retrieval, and template explanations. LOCAL_HYBRID may add one qualified BPU evidence probe and a sequential Qwen2 0.5B read-only explanation service. DEEP_SHADOW may use a PC or remote model only for non-authoritative annotations. Memory reserve, thermal, model qualification, or connectivity gates force an explicit fallback whose reason is displayed in the Truth Ribbon.
