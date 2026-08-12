# RootScope native libdnn bridge

This directory contains the narrow RDK X5 runtime bridge used to replay the
frozen RootScope r7 BPU model with the exact valid-shape input contract observed
from D-Robotics `hrt_model_exec`.

The worker is deliberately not a robot controller. It loads one hash-bound
model once, accepts only fixed-size `uint8 [1,3,224,224]` tensor frames over
stdin, and returns four finite float32 logits over stdout. It has no camera,
network, serial, GPIO, pump, service-manager, or state-machine API.

Protocol v1 is little-endian:

- request header: `<8sIQI>` = magic `RSNV3REQ`, version `1`, non-zero request
  id, payload length `150528`; followed by exactly `150528` tensor bytes;
- response header: `<8sIQIQI>` = magic `RSNV3RSP`, version `1`, matching request
  id, status `0`, worker inference nanoseconds, payload length `16`; followed by
  four float32 logits.

The packaged `compile_contract_x5.v1.json` binds source, compiler, headers,
`libdnn.so`, two deterministic builds, and the selected executable. The Python
adapter validates both that contract and the executable/model SHA-256 before
starting the worker.
