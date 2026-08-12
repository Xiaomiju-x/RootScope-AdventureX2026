# Published model assets

Only the three final, compact, redistributable RootScope artifacts are stored with
Git LFS. Base models, merged/f16 models, training caches, OpenExplorer intermediates,
third-party runtimes and repeated checkpoints are intentionally not duplicated.

| Asset | Purpose | Boundary |
|---|---|---|
| `vision/*.onnx` | CPU reference for four fixed printed answer cards | Not open-world plant recognition |
| `bpu/*.bin` | Horizon Bayes-e compiled auxiliary replay model | BPU evidence only; no actuator authority |
| `rootmind-adapter/*.safetensors` | RootMind read-only QLoRA adapter | Requires upstream Qwen3-1.7B; no control authority |

Every binary is content-bound in [`MANIFEST.json`](MANIFEST.json). Clone with Git
LFS enabled (`git lfs install`) and run `python tools/verify_model_assets.py` before
using an artifact.

The 1.1 GB merged Q4 GGUF is not duplicated in Git. Its published content identity
is retained in the final X5 acceptance receipt: SHA-256
`0bd32a4d943db70ca2e7859906aa23cd7773ef80982680c454178a26b513aeec`.
Recreate it from the Apache-2.0 upstream base plus this adapter by following the
merge/conversion tools under `research/rootscope-v3/llm/`.
