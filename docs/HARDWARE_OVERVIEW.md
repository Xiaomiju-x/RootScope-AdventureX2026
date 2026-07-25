# Hardware Overview

## Public block diagram

```mermaid
flowchart LR
    CAM["USB camera"] --> X5["RDK X5<br/>perception + evidence + proposal"]
    X5 -. "bounded private link" .-> MCU["STM32<br/>real-time safety executor"]
    MCU --> DRV["isolated driver stage"]
    DRV --> PROBE["single-axis probe"]
    DRV --> PUMP["relay + pump"]
    ESTOP["physical E-stop / power isolation"] --> DRV
```

## Responsibility split

RDK X5:

- camera acquisition and visual evidence;
- CPU/BPU inference experiments;
- local read-only LLM and RAG;
- deterministic, bounded action proposal;
- evidence receipt generation.

STM32:

- real-time actuator sequencing;
- watchdog and heartbeat supervision;
- timeout and emergency stop;
- final pump-off latch;
- lower-controller identity and state reporting.

## Deliberately omitted

This public document does not provide pin numbers, voltage/current design,
relay polarity, serial frames, unlocking sequence, heartbeat timing, motor step
tables, firmware, calibration, wiring harness details or commissioning
instructions.

Do not infer a safe physical design from this block diagram.

