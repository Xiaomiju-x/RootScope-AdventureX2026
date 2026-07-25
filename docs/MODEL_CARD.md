# Public Model Card

## Intended system

RootScope uses multiple bounded components rather than a single end-to-end
actuator model.

| Component | Public role | Deployment statement |
|---|---|---|
| CPU ONNX classifier | semantic plant-card evidence | current controlled-demo authority branch |
| AKAZE/RANSAC | independent geometric verification | required by the demonstrated dual-evidence path |
| Fast local LLM | compact structured explanation | CPU, read-only, swap-loaded |
| Deep local LLM | deeper domain explanation | CPU, read-only, swap-loaded |
| BM25/RAG2 | evidence retrieval and deterministic fallback | read-only |
| BPU model | acceleration qualification and auxiliary evidence | 43/43 replay on two paths; not current action authority |

## Public evaluation snapshot

- Four controlled physical cards: 4/4 live recognition.
- Same-scene holdout: 8/8.
- RAG2 BM25 Recall@5: 92.19%.
- RAG2 hard Top-3: 84.09%.
- Forbidden Recall@5: 94.44%.
- Citation escape: 0.
- BPU replay: 43/43, mean cosine 1.0, Top-1 43/43.

These are frozen project snapshots, not claims of open-world agricultural
generalization or sustained performance statistics.

## Not released

Weights, adapters, prompts, corpora, templates, thresholds, splits, calibration,
quantization files and deployment binaries are intentionally excluded.

## Known limitations

- Controlled printed cards are not wild plants.
- The current live path prioritizes stable 1080p30 capture; 4K is a camera
  capability, not the demonstrated live pipeline.
- The probe is down-only in the current prototype and needs manual return.
- BPU evidence does not directly authorize an actuator.
- No water-saving percentage or biological outcome has been established.

