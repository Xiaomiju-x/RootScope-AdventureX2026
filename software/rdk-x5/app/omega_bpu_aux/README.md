# RootScope-Ω generic BPU auxiliary probe

This package is a standalone evidence probe for the vendor Bayes-e
MobileNetV2 binary:

`/opt/hobot/model/x5/basic/mobilenetv2_224x224_nv12.bin`

It accepts only an explicit JSON manifest. Every model, optional label file,
and image is bound to an absolute path and SHA-256 digest. It never scans an
image directory, enumerates a camera, opens `/dev`, or imports the RootScope
Safety Compiler, state machine, serial, GPIO, pump, network, or tool layers.

## What the receipt proves

On a real, non-injected RDK X5 runtime, a successful receipt proves that the
hash-bound generic ImageNet-1000 model was loaded by
`hobot_dnn.pyeasy_dnn` and that the listed explicit images produced finite
outputs. Per-image receipts contain:

- the source-file and decoded-RGB hashes;
- the exact Pillow RGB-to-NV12 preprocessing contract and NV12 hash;
- finite raw/canonical-logit/probability vectors and their float32 hashes;
- top-k generic ImageNet class IDs/optional vendor labels;
- softmax probability, entropy, normalized entropy, energy, and timing;
- immutable zero-authority and claim-boundary fields.

The entropy, energy, and maximum probability values are **uncalibrated
descriptors only**. They are not a plant-domain OOD decision. Generic
ImageNet labels are not desert-plant or RootScope classifications. This probe
does not qualify a RootScope BPU classifier and does not change the immutable
`selected_bin=null` state.

## Input and execution

Start from `configs/omega/bpu_aux_probe.manifest.example.json`, replace every
zero digest with the actual SHA-256 value, and list each input image explicitly.
The normal CLI has no model override and no camera argument:

```bash
cd ~/rootscope
python3 -m app.omega_bpu_aux.probe \
  --manifest /opt/rootscope/rootscope_inputs/bpu_aux_manifest.json \
  --out /opt/rootscope/rootscope_evidence/bpu_aux_receipt.json
```

The intended environment is the isolated v2 BPU virtual environment created
with `include-system-site-packages=true`: vendor `numpy` and `hobot_dnn` remain
system packages, while Pillow is the only local wheel needed by this probe.
The CLI creates its receipt exclusively and refuses to overwrite existing
evidence.

## Provenance boundary

The frozen XRD file
`workstation/dual_arm/overhead_bpu_aux_probe_x5.py` was inspected only as an
architectural reference. No XRD source was copied, imported, executed, or
modified. This implementation lives entirely in the AdventureX RootScope tree.
