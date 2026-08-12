## Summary / 摘要

<!-- What changed and why? Keep the PR focused. -->

## Evidence / 验证

<!-- Commands, tests, commit/model/input hashes, and evidence level (R0–R5). -->

```text
python -m pytest
python tools/audit_public_release.py
```

## Truth and safety boundary / 真实性与安全边界

<!-- State what was NOT tested or claimed. If hardware is involved, say: not run / read-only / no-action / supervised physical. -->

## Checklist

- [ ] I read `CONTRIBUTING.md`, `SECURITY.md`, and the relevant architecture/runbook.
- [ ] Tests pass; new behavior has missing/conflict/OOD/timeout/failure coverage.
- [ ] The public-release audit passes.
- [ ] No credentials, private IP/MAC/device identity, absolute user paths, or unsanitized logs/media are included.
- [ ] New data/models/media have source, license, redistribution permission, manifest, and SHA-256.
- [ ] LLM/RAG output cannot directly control serial, GPIO, motors, relays, or pumps.
- [ ] Physical failures do not automatically retry, unlock, or resume.
- [ ] Documentation/schema/model cards are updated.
- [ ] Benchmarks state environment, sample size, evidence level, and limitations.
- [ ] I have the right to submit this contribution under the repository’s licenses.

## Related issues

<!-- Closes #... -->
