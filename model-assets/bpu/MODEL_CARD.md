# RootSight Bayes-e compiled auxiliary model

This `.bin` is the final Horizon toolchain compilation of the RootSight fixed-card
model. Its exact source graph is published beside it as
`rootscope_seed17_resnet18_static_b1x3x224x224_opset11.onnx` (SHA-256
`50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad`).
The content-bound X5 acceptance replay ran 43 fixed samples with 43/43 top-1
agreement and mean cosine 1.0 through both canonical and persistent native paths.

Those numbers are replay equivalence on a frozen fixture, not live-camera accuracy,
latency, field performance or physical-loop success. The BPU path is auxiliary and
never holds pump/motor authority. `config.yaml` and `*_quant_info.json` preserve the
compiler contract; proprietary compiler/runtime binaries and uncleared calibration
images are not redistributed. Regenerate calibration tensors with the published
dataset pipeline after supplying rights-cleared images.
