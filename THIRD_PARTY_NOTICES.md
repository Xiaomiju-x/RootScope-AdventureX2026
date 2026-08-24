# Third-party notices

This file records third-party material that is physically distributed in this
repository. It does not grant rights beyond each upstream licence. A nearer
file header or asset manifest takes precedence when it is more specific.

## Arm CMSIS 5

- Paths: `firmware/stm32f103-v15/Drivers/CMSIS/Include/**`
- Upstream: <https://github.com/ARM-software/CMSIS_5>
- Copyright: Arm Limited and contributors, as stated in each file
- Licence: Apache-2.0
- Local text: [`LICENSE`](LICENSE)
- Notes: source headers are marked vendored; upstream copyright and SPDX
  notices must be retained.

## STMicroelectronics STM32CubeF1 components

- Paths:
  - `firmware/stm32f103-v15/Drivers/STM32F1xx_HAL_Driver/**`
  - `firmware/stm32f103-v15/Drivers/CMSIS/Device/ST/STM32F1xx/**`
  - generated CubeMX startup, system, peripheral and project files beneath
    `firmware/stm32f103-v15/Core/**` and `firmware/stm32f103-v15/MDK-ARM/**`
- Upstream: <https://github.com/STMicroelectronics/STM32CubeF1>
- HAL driver upstream: <https://github.com/STMicroelectronics/stm32f1xx-hal-driver>
- Copyright: STMicroelectronics, as stated in each file
- Licence: BSD-3-Clause for the included HAL/Cube-generated files that carry
  that notice; some included CMSIS-derived files carry Apache-2.0. Preserve the
  file-level notice.
- Local BSD text: [`LICENSES/BSD-3-Clause.txt`](LICENSES/BSD-3-Clause.txt)

The firmware release image is an object-form build of this mixed-licence source
tree. Its accompanying documentation and checksum manifest reproduce this
notice as required by BSD-3-Clause.

## Wikimedia Commons print-card sources

- Paths: `assets/print-cards/**`
- Source and author details:
  [`assets/print-cards/RootScope_A4_half_page_cards_20260724_manifest.json`](assets/print-cards/RootScope_A4_half_page_cards_20260724_manifest.json)
- Licences: source-specific `CC-BY-SA-4.0` or public-domain status recorded in
  that manifest
- CC BY-SA text/URI notice:
  [`LICENSES/CC-BY-SA-4.0.txt`](LICENSES/CC-BY-SA-4.0.txt)

Do not strip the manifest or its attribution when redistributing the cards or
derivatives. RootScope does not relicense upstream images under CC-BY-4.0.

## Reviewed model artifacts and upstream components

Exactly four team-produced final artifacts are redistributed through Git LFS
and content-bound by `model-assets/MANIFEST.json`:

- the fixed-answer-card ONNX vision model;
- the exact static source ONNX used for BPU compilation;
- the corresponding D-Robotics Bayes-e BPU compiled artifact; and
- a RootMind QLoRA adapter trained for the Apache-2.0 Qwen3-1.7B base model.

The repository does **not** redistribute the Qwen base model, torchvision
initialization weights, training datasets, BGE weights, the D-Robotics SDK or
OpenExplorer intermediate products. The BPU artifact is a compiled form of the
published team model, not a copy of the vendor toolchain. All four published
artifacts are licensed Apache-2.0 by the RootScope team and have no physical
execution authority; upstream names, licences and limitations remain recorded
in the manifest and model cards. Consult [`docs/MODEL_ASSETS.md`](docs/MODEL_ASSETS.md)
before downloading or republishing an artifact.

## Tool and platform names

RDK, D-Robotics, Arm, Keil, STM32, STM32Cube, GitHub, Qwen and other names are
marks of their respective owners. Their mention is descriptive and does not
imply endorsement.

## Reporting omissions

Please report a missing attribution or licence through the private security
channel described in [`SECURITY.md`](SECURITY.md) when public disclosure could
also expose a credential, device identity or private dataset. Ordinary licence
corrections may be submitted as pull requests.
