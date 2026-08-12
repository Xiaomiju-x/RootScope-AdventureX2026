# Repository licence matrix

This repository is a multi-licence distribution. The top-level Apache-2.0
[`LICENSE`](LICENSE) does **not** replace third-party, hardware-design, media or
dataset terms. Resolve a file's licence in this order:

1. a licence or copyright notice inside the file;
2. an adjacent manifest or licence file;
3. the most specific path row below;
4. the repository default for that class of material.

| Path or material | Licence / status | Required notice |
|---|---|---|
| `src/**`, `software/**`, `pipelines/**`, `tools/**`, `tests/**`, `examples/**` | Apache-2.0, except a nearer notice | `LICENSE`, `NOTICE` |
| `research/**` authored source, configs and synthetic fixtures | Apache-2.0, except source-bound data or a nearer notice | `LICENSE`, relevant model/data card |
| `model-assets/**` team-authored final artifacts and metadata | Apache-2.0; upstream base weights are not redistributed | `model-assets/MANIFEST.json`, per-directory model cards, `THIRD_PARTY_NOTICES.md` |
| `hardware/design/**` original editable hardware source | CERN-OHL-S-2.0 | `LICENSES/CERN-OHL-S-2.0.txt`, source location |
| `docs/**`, root Markdown/CFF policy files | CC-BY-4.0, except quoted/vendor material and nearer notices | `LICENSES/CC-BY-4.0.txt`, attribution |
| `assets/media/**`, `evidence/public/images/**`, `evidence/public/video/**` created by the RootScope team | CC-BY-4.0, subject to recorded portrait/privacy consent | `LICENSES/CC-BY-4.0.txt`, asset manifest |
| `assets/print-cards/**` | Per-source CC-BY-SA-4.0 or public domain | adjacent manifest and `THIRD_PARTY_NOTICES.md` |
| `firmware/stm32f103-v15/Drivers/CMSIS/Include/**` | Apache-2.0 | original file headers, `LICENSE` |
| `firmware/stm32f103-v15/Drivers/STM32F1xx_HAL_Driver/**` | BSD-3-Clause | original file headers, `LICENSES/BSD-3-Clause.txt` |
| `firmware/stm32f103-v15/Drivers/CMSIS/Device/ST/**` and CubeMX-generated ST source/project files | File-level ST notice; BSD-3-Clause where stated | original file headers, `LICENSES/BSD-3-Clause.txt` |
| RootScope-authored firmware additions in `firmware/stm32f103-v15/Core/**` and `Tests/**` | Apache-2.0 for the team's additions; generated scaffolding retains ST terms | file headers, both applicable notices |
| `firmware/stm32f103-v15/release/*.hex` | Object form of the mixed-licence firmware source | checksum manifest, `NOTICE`, `THIRD_PARTY_NOTICES.md` |
| Public evidence JSON and receipts | CC-BY-4.0 for the authored record; embedded facts/source hashes are not separately relicensed | evidence index and privacy policy |
| Upstream model weights, training datasets, vendor SDKs, Keil binaries | **Not included unless separately and explicitly licensed** | upstream licence and checksum manifest |
| Names, logos and marks | Not licensed by code/content licences | `TRADEMARKS.md` |

## Hardware source location notice

The source location for the original RootScope hardware design is:

<https://github.com/Xiaomiju-x/RootScope-AdventureX2026/tree/main/hardware/design>

When conveying a product based on CERN-OHL-S-2.0 covered source, comply with
the Complete Source and Source Location obligations in that licence.

## Contributions

By submitting a contribution, contributors certify they have the right to
submit it under the licence applicable to the destination path. Moving a file
between the classes above does not silently relicense it. Record deliberate
licence changes explicitly in the pull request and retain upstream notices.
