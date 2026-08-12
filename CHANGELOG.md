# Changelog

All notable changes to the public RootScope release are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for public interfaces and artifact contracts.

## [Unreleased]

### Planned

- Community-reviewed reproduction reports and portability fixes.
- Documentation corrections that preserve the frozen competition facts.

## [1.0.1] - 2026-08-13

### Security

- Split read-only release verification, provenance attestation, and publishing
  into least-privilege jobs with an exact six-asset handoff.
- Require strict SemVer tags on commits contained in `origin/main`, with the
  tag version bound to both package metadata and runtime `__version__`.
- Hash-lock the Python 3.12 Linux dependencies used by release verification.
- Publish releases through a complete draft before making them immutable.

## [1.0.0] - 2026-08-12

### Added

- Full reproducible source release for the AdventureX 2026 D-Robotics “Give AI a Body” Silver Award, 2nd-place RootScope project.
- Final RDK X5 application, RootSight/RootMind/RAG2 research code, tests, and release/data/model pipelines.
- STM32F103C8T6 V15 firmware source and archived release checksum contract.
- Hardware, electrical, mechanical, build, deployment, reproduction, safety, data-governance, and troubleshooting documentation.
- Redistributable final model assets through Git LFS with model/asset manifests.
- Sanitized competition photos, demo chapters, and public machine-readable evidence.
- Community health files, issue forms, CODEOWNERS, Dependabot, release drafting, and public-release auditing.

### Changed

- Replaced the earlier minimal open-core narrative with an auditable public-release boundary.
- Clarified that RootScope is stationary and the visible wheeled chassis is only a carrier/power stand.
- Bound every award, vision, BPU, LLM, mechanical, and physical claim to its verified scope.

### Security

- Excluded and detected credentials, private device/network identity, absolute user paths, unsanitized receipts, and unlicensed assets.
- Documented actuator-power isolation, staged bring-up, watchdog/timeout/latching behavior, and private vulnerability reporting.

[Unreleased]: https://github.com/Xiaomiju-x/RootScope-AdventureX2026/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/Xiaomiju-x/RootScope-AdventureX2026/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Xiaomiju-x/RootScope-AdventureX2026/releases/tag/v1.0.0
