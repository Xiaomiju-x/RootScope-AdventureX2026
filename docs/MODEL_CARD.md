# RootScope System Model Card

This card summarizes the deployed model ensemble. Individual redistributable assets must also carry their own model card and SHA-256 manifest; see [`MODEL_ASSETS.md`](MODEL_ASSETS.md).

## Intended use

RootScope is a stationary AdventureX 2026 prototype for controlled printed-card demonstrations and evidence/safety research. The model ensemble may classify one of three visible morphology cards or reject an unknown/non-target input, retrieve bounded explanatory evidence, and create a read-only explanation.

It is not intended for species identification, biological root-depth estimation, agronomic water-demand prediction, field irrigation prescription, unattended actuation, or safety-critical deployment.

## Components

| Component | Deployment role | Runtime | Physical authority |
|---|---|---|---|
| CPU ONNX visual classifier | Fixed-card semantic evidence / audit fallback | RDK X5 CPU | None |
| AKAZE/RANSAC matcher | Independent geometric evidence | CPU/OpenCV | None |
| Bayes-e visual model | BPU execution / auxiliary evidence | RDK X5 BPU | None |
| RootMind Fast | Compact structured explanation | CPU, on demand | None |
| RootMind Deep | Domain explanation with retrieval | CPU, on demand | None |
| BM25/RAG2 | Retrieval and deterministic HOLD fallback | CPU | None |

Only a deterministic gate can map verified evidence to `0/1024/1536/2048`; STM32 V15 independently enforces the physical transaction. LLM text cannot change a tier or create a hardware command.

## Inputs and outputs

The visual contract uses a perspective-rectified fixed-card ROI, 224×224 RGB, with classes `grass_clump`, `low_shrub`, `young_tree`, and unknown/non-target. Exact preprocessing and output order are bound to each asset manifest.

The system output includes semantic and geometric evidence, image quality/OOD/freshness state, decision/reason code, provenance, and receipt hash. Missing, stale, conflicting, OOD, or unsafe evidence must return HOLD.

## Frozen evaluation snapshot

- Four controlled physical printed cards: 4/4 live qualification.
- Same-scene frozen holdout: 8/8.
- Two fixed-input BPU paths: 43/43 with recorded mean cosine similarity 1.0.
- RAG2: BM25 Recall@5 92.19%; hard Top-3 84.09%; Forbidden Recall@5 94.44%; Citation Escape 0.
- Two supervised full grass-card physical cycles.

These figures are not open-world accuracy or sustained performance. Fixed cards, same-scene images, and frozen BPU replay inputs are strongly scoped evidence.

## Training and data

Training and curation code is under `pipelines/training/` and `pipelines/dataset/`; registries and evaluation contracts are under `research/rootscope_v3/`. Some development assets are not redistributed because their redistribution rights are not approved. Machine/VLM labels are treated as candidates, not ground truth. See [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md).

## Limitations and risks

- Strong domain shift from fixed printed cards to natural plants.
- Sensitivity to camera pose, printer/paper, illumination, glare, blur, and occlusion.
- Dataset scale and source diversity are insufficient for field claims.
- Retrieval/LLM can be wrong or malformed; deterministic validation and HOLD are mandatory.
- BPU quantization agreement on frozen inputs does not establish semantic generalization.
- The mechanism has no automatic retraction or limit sensor.

## Ethical and safety considerations

Do not use morphology classes as species, ecological, or agronomic judgments. Do not use training images or people/event media without checking their licenses and rights. Models have zero actuator authority; hardware requires independent power isolation, emergency stop, watchdog, timeout, mechanical limits, leak protection, and human supervision.

## Licensing

Project-authored code is Apache-2.0. Each model asset follows its manifest and upstream license; base model licenses are not replaced by this repository license. Git LFS availability does not imply a right to redistribute or deploy an asset.
