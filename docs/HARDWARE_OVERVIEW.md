# Hardware Overview / 硬件总览

> This compatibility page points to the maintained final-hardware guides. Earlier public snapshots intentionally omitted implementation details; the current source release includes the final V15 firmware, pin map, build procedure, and staged reproduction gates.

RootScope is a stationary system:

```mermaid
flowchart LR
    CAM["Fixed UVC camera"] --> X5["RDK X5<br/>vision · evidence · read-only LLM/RAG"]
    X5 -->|"bounded serial transaction"| MCU["STM32F103C8T6 V15"]
    MCU -->|"PA0–PA3 via ULN2003"| PROBE["Down-only probe"]
    MCU -->|"PB6 active-low open-drain"| RELAY["Relay + one pump"]
    CUT["Physical power isolation / supervision"] --> PROBE
    CUT --> RELAY
```

The wheeled chassis visible in photographs is only a carrier/power stand; locomotion is not part of the final chain.

Canonical guides:

- [Final hardware and wiring](HARDWARE_WIRING.md)
- [Mechanical design](MECHANICAL.md)
- [STM32 V15 build and flash](STM32_BUILD.md)
- [RDK X5 staged deployment](RDK_X5_DEPLOYMENT.md)
- [System architecture and authority separation](ARCHITECTURE.md)

Do not connect actuator power before completing the staged checks in those guides. This prototype is not safety-certified.
