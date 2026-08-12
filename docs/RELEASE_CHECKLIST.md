# Release Checklist / 发布清单

Use this checklist from a clean clone of the candidate tag. A release is rejected on any credential/privacy/licensing failure, missing LFS object, hash mismatch, or overstated claim.

## Source and tests

- [ ] Candidate is a reviewed commit on the intended branch; worktree is clean.
- [ ] `python -m pip install -e ".[dev]"` succeeds in a fresh environment.
- [ ] Root tests pass on supported Python versions.
- [ ] Relevant `research/`, `software/`, firmware-host, and website tests pass.
- [ ] `python tools/audit_public_release.py` passes.
- [ ] `git diff --check` and internal Markdown link checks pass.

## Secrets, privacy, and licences

- [ ] No credential, key, token, password, private endpoint, or server backup exists in Git/LFS/history being published.
- [ ] No real private IP, MAC, machine-id, boot-id, serial number, username path, or device topology remains.
- [ ] Media has consent, EXIF/GPS removal, screen/badge/QR review, provenance, hashes, and a licence.
- [ ] Data/model assets have verified redistribution permission; unapproved source images and base-model duplicates are absent.
- [ ] `LICENSE_MATRIX.md`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, model/data/media manifests, and vendored headers agree.

## Artifacts

- [ ] `git lfs pull` retrieves every pointer; `git lfs fsck` passes.
- [ ] Every released binary has a relative path, byte size, SHA-256, role, upstream/source, licence, and validation boundary.
- [ ] Firmware release hash is tied to source commit and toolchain.
- [ ] CPU/BPU/LLM/RAG assets are not confused; `physical_authority` is false for models.
- [ ] Rebuilding twice from the same frozen inputs is compared or any nondeterminism is documented.

## Documentation and claims

- [ ] Title/description state AdventureX 2026 D-Robotics “Give AI a Body,” Silver Award, 2nd Place.
- [ ] README says stationary chamber; wheeled base is carrier/power stand only.
- [ ] Fixed cards are not described as open-world plants.
- [ ] Steps are not converted to length/root depth; 5 seconds is not converted to water volume/savings.
- [ ] PC, X5 CPU, actual BPU, replay, read-only, and physical observations are distinct.
- [ ] Stage photo is event context, not standalone rank proof.
- [ ] All commands use placeholders and no competition device entry points.

## GitHub release

- [ ] Version in `CITATION.cff`, package metadata, changelog, manifest, and tag agree.
- [ ] Annotated, signed tag is created from the reviewed commit where signing is available.
- [ ] Release notes list changes, checksums, migrations, limitations, and safety warnings.
- [ ] CI succeeds on the tag; GitHub dependency/security features are enabled as appropriate.
- [ ] Repository description/topics no longer say open-core.
- [ ] Remote tag/release assets are fetched back and hashes verified.

## After release

- [ ] Verify README images/video links and LFS downloads in a logged-out browser.
- [ ] Verify CITATION rendering, issue forms, security reporting, and release notes.
- [ ] Keep the previous release available; do not rewrite a published tag for ordinary fixes.
- [ ] If any secret was exposed, revoke/rotate first, then coordinate history cleanup and disclosure.
