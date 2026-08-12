# RootSight-Delta candidate

This independent candidate extends the frozen RootScope v2 visual chain without
overwriting it. It contains deterministic, X5-portable functions for:

- reproducible yellow booth light, print, blur, perspective, moire and JPEG
  domain augmentation;
- capture-session/source-group splitting with group and SHA leakage checks;
- optical/OOD admission and temporal multi-frame consensus;
- before/after translation registration, reference-patch correction and Lab
  color difference;
- wetting target coverage, neighbor spill, center offset/front radius and a
  mass/visual consistency HOLD gate.

The package has **no camera opener, serial writer, GPIO or pump API**.
`physical_authority` is always false. PC synthetic tests are not field
accuracy, and a learned wetting segmenter is intentionally not trained until
physical paired images exist.

Verification from the `adventurex` directory:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m unittest rootscope_v3.evaluations.test_rootsight_delta
python rootscope_v3/evaluations/run_vision_evaluation.py `
  --adventurex-root . `
  --output rootscope_v3/evaluations/evidence/rootsight_delta_pc_receipt.json
```
