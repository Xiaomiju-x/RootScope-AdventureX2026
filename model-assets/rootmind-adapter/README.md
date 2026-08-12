---
base_model: Qwen/Qwen3-1.7B
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
  - qwen3
  - lora
  - rootscope
  - read-only
---

# RootMind Qwen3-1.7B QLoRA adapter

This is the final RootScope event adapter, trained on deterministic structured
supervision for evidence summaries, abstention and safety-boundary responses.
It is an adapter only; fetch the upstream `Qwen/Qwen3-1.7B` base model separately.

## Scope

- Base: `Qwen/Qwen3-1.7B`, Apache-2.0.
- Method: NF4 QLoRA, rank 8, alpha 16, 96 adversarial-refinement optimizer steps.
- Hardware: a laptop RTX 4050; approximately 296 seconds for the final refinement.
- Intended use: RootScope read-only explanations and structured JSON responses.
- Authority: none. The model cannot write serial, GPIO, pump, motion or state-machine commands.

## Evaluation boundary

The accompanying receipts concern the fixed RootScope structured task. They are
not evidence of general agriculture expertise, general dialogue quality, physical
control safety, X5 accelerator execution, or field deployment. The LLM layer is
always subordinate to deterministic evidence and safety logic.

## Loading

Use a PEFT-compatible Transformers version, load `Qwen/Qwen3-1.7B`, then attach
`adapter_model.safetensors` and `adapter_config.json`. See
[`docs/MODEL_ASSETS.md`](../../docs/MODEL_ASSETS.md) for the exact reproducibility boundary.

## License

The RootScope adapter files are Apache-2.0. The upstream base model is not copied
into this repository and remains subject to its own Apache-2.0 license and notices.
