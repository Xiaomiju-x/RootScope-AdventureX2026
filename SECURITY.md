# Security Policy

## Scope

This repository contains a software-only, proposal-only reference layer. It must
not be used to operate pumps, motors, relays, valves, GPIO, serial devices or
other physical actuators.

The public package deliberately has:

- no serial or GPIO dependency;
- no hardware device path;
- no actuator command encoder;
- no network control endpoint;
- no credential loader;
- no production thresholds or calibration;
- no automatic retry of a physical action.

## Reporting a vulnerability

Please open a GitHub security advisory for issues in the public reference code.
Do not publish credentials, private device identities, precise network topology
or details that would facilitate unsafe actuator control in a public issue.

## Hardware safety

Real irrigation hardware requires an independent lower-controller safety state
machine, watchdog, emergency stop, power-stage interlock, electrical isolation,
mechanical limits, leak protection and a trained operator. A software `HOLD`
result is not a substitute for physical safety engineering.

## Supported versions

Only the latest tagged public release is supported. Competition prototypes and
private deployment bundles are outside the scope of this repository.

