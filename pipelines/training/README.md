# RootScope machine-curated experimental training

`rootscope_machine_curated_pipeline.py` is the only training entry point for the
frozen `rootscope_machine_curated_provisional_v1`, v2, and v3 packs. The CLI
defaults to v3. It fails closed unless the operator explicitly acknowledges that
the pack is machine-curated experimental evidence and has no formal authority.

The current safe verification command is a one-batch CPU smoke test:

```powershell
& .\.ai_curation_venv\Scripts\python.exe tools\training\rootscope_machine_curated_pipeline.py `
  --ack-machine-curated-experimental-only `
  --smoke --epochs 1 --seeds 17 --max-train-batches 1 `
  --random-init --batch-size 2 --samples-per-class 1 --device cpu `
  --run-id v3_cpu_smoke_unique_id
```

The frozen v2 remains audit-readable but its independently recomputed long-run
gate is false. V3 is checked independently for image count, unique source-group
count, and unique creator-group count in every train/validation class. It also
recomputes every source image SHA-256 and dHash, binds creator/path/source fields
to the acquisition manifests, and rejects cross-partition dHash distance <= 4.
`--smoke` may exercise one epoch and at most two batches, but every resulting artifact is marked
`MACHINE_CURATED_SMOKE_ONLY_NOT_MODEL_CANDIDATE`.

The default model initialization is official torchvision ImageNet pretraining.
`--random-init` exists only for a dependency/graph smoke test and is rejected
outside `--smoke`.

Non-smoke execution does not trust a bare visual-audit CLI acknowledgement. It
requires v3's receipt-bound `machine_visual_review_evidence.json`, verifies its
SHA-256 and selected ID/role bindings, and preserves the scope distinction: E3
has a machine screen only, while dual-machine review and root adjudication apply
only to the selected E4 records. This evidence does not make the pack
human-reviewed, A1, rights-approved, formally training-eligible, print-eligible,
data-locked, or model-qualified.

The fixed contracts are:

- class order: `grass_clump, low_shrub, young_tree, unknown`;
- static RGB input: `1x3x224x224`;
- torchvision ResNet18 with fixed `AvgPool2d(7, 1)`, not adaptive pooling;
- static ONNX opset 11 plus CPU Torch/ONNX logit consistency on the first
  validation image of every present class and deterministic zero/ramp probes;
- confidence/margin rejection plus a per-predicted-class support and Wilson
  lower-bound gate; unsupported classes are forced to reject;
- print-card-oriented training augmentation (mild perspective/rotation/crop,
  illumination/color-temperature drift, blur, JPEG, paper border/shadow);
- separate `NATURAL_WEB_VALIDATION` and
  `DIGITAL_PRINT_SOURCE_HOLDOUT_NOT_UVC_RECAPTURE` reports.

The digital print-source holdout contains original files selected for future
printing. It is never used for weights, checkpoint selection, temperature, or
rejection calibration, and it must never be described as a physical print/UVC
camera-domain result.

Every run receipt keeps `rights=false`, every authority field false,
`model_candidate=false`, and `x5_ready=false`. A non-smoke run may set only
`experimental_model_candidate=true`; that is not a formal model-candidate or
deployment claim.
