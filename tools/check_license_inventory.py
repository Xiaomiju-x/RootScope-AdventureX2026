#!/usr/bin/env python3
"""Fail closed when redistributed files are not covered by the licence map.

This is deliberately a repository-specific policy check, not a legal-opinion
generator. It binds the known vendored firmware and the reviewed model
artifacts to their notices and catches accidental additions outside the map.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST = ROOT / "model-assets" / "MANIFEST.json"
EXPECTED_MODEL_PATHS = {
    "model-assets/vision/rootscope_answer_cards_resnet18_opset11.onnx",
    "model-assets/bpu/rootscope_seed17_resnet18_static_b1x3x224x224_opset11.onnx",
    "model-assets/bpu/rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes.bin",
    "model-assets/rootmind-adapter/adapter_model.safetensors",
}
EXPECTED_FIRMWARE_SHA256 = (
    "5016b96d138d4ffad2088dd5da288b4d68c5deba781555ad82eb6f7fb4bfd887"
)


@dataclass(frozen=True, order=True)
class Failure:
    category: str
    path: str

    def render(self) -> str:
        return f"{self.category}: {self.path}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_text(
    failures: list[Failure], relative: str, required_fragments: tuple[str, ...]
) -> str:
    path = ROOT / relative
    if not path.is_file():
        failures.append(Failure("missing_licence_material", relative))
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        failures.append(Failure("unreadable_licence_material", relative))
        return ""
    for fragment in required_fragments:
        if fragment not in text:
            failures.append(Failure("incomplete_licence_material", relative))
            break
    return text


def check_licence_texts(failures: list[Failure]) -> None:
    require_text(
        failures,
        "LICENSE",
        ("Apache License", "Version 2.0, January 2004", "END OF TERMS AND CONDITIONS"),
    )
    require_text(
        failures,
        "LICENSES/BSD-3-Clause.txt",
        (
            "Redistribution and use in source and binary forms",
            "Neither the name of the copyright holder",
            "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS",
        ),
    )
    require_text(
        failures,
        "LICENSES/CERN-OHL-S-2.0.txt",
        (
            "CERN Open Hardware Licence Version 2 - Strongly Reciprocal",
            "4 Making and Conveying Products",
            "6 DISCLAIMER AND LIABILITY",
            "8 General",
        ),
    )
    require_text(
        failures,
        "LICENSES/CC-BY-4.0.txt",
        (
            "Attribution 4.0 International",
            "Creative Commons Corporation",
            "Section 3",
            "License Conditions",
        ),
    )
    require_text(
        failures,
        "LICENSES/CC-BY-SA-4.0.txt",
        (
            "Attribution-ShareAlike 4.0 International",
            "Creative Commons Corporation",
            "Section 3",
            "License Conditions",
        ),
    )


def check_policy_documents(failures: list[Failure]) -> None:
    matrix = require_text(
        failures,
        "LICENSE_MATRIX.md",
        (
            "model-assets/**",
            "hardware/design/**",
            "firmware/stm32f103-v15/Drivers/CMSIS/Include/**",
            "firmware/stm32f103-v15/Drivers/STM32F1xx_HAL_Driver/**",
            "assets/print-cards/**",
        ),
    )
    notices = require_text(
        failures,
        "THIRD_PARTY_NOTICES.md",
        ("Arm CMSIS", "STMicroelectronics", "Wikimedia", "Qwen"),
    )
    require_text(
        failures,
        "NOTICE",
        ("Apache", "License 2.0", "CERN-OHL-S-2.0", "CC-BY-4.0"),
    )
    if "NOASSERTION" in matrix or "UNKNOWN" in matrix:
        failures.append(Failure("unresolved_licence_matrix_entry", "LICENSE_MATRIX.md"))
    if "credentials" in notices.lower() and "not" not in notices.lower():
        failures.append(Failure("unsafe_notice_wording", "THIRD_PARTY_NOTICES.md"))


def check_vendor_headers(failures: list[Failure]) -> int:
    groups = (
        (
            "firmware/stm32f103-v15/Drivers/CMSIS/Include",
            (".h",),
            ("Copyright",),
            ("Apache-2.0", "Apache License"),
            "cmsis_header_notice_missing",
        ),
        (
            "firmware/stm32f103-v15/Drivers/STM32F1xx_HAL_Driver",
            (".c", ".h"),
            ("Copyright", "STMicroelectronics"),
            ("licensed under terms", "BSD"),
            "stm32_hal_notice_missing",
        ),
        (
            "firmware/stm32f103-v15/Drivers/CMSIS/Device/ST",
            (".c", ".h", ".s"),
            ("Copyright", "STMicroelectronics"),
            ("licensed under terms", "BSD", "Apache License"),
            "stm32_device_notice_missing",
        ),
    )
    checked = 0
    for directory, suffixes, fragments, alternatives, category in groups:
        root = ROOT / directory
        if not root.is_dir():
            failures.append(Failure("missing_vendor_source", directory))
            continue
        files = sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
        )
        if not files:
            failures.append(Failure("missing_vendor_source", directory))
            continue
        for path in files:
            checked += 1
            try:
                # Vendor notices are at the top; a bounded read keeps this gate fast.
                text = path.read_text(encoding="utf-8", errors="replace")[:8192]
            except OSError:
                failures.append(Failure("unreadable_vendor_source", path.relative_to(ROOT).as_posix()))
                continue
            lowered = text.lower()
            if (
                not all(fragment.lower() in lowered for fragment in fragments)
                or not any(fragment.lower() in lowered for fragment in alternatives)
            ):
                failures.append(Failure(category, path.relative_to(ROOT).as_posix()))
    return checked


def check_model_inventory(failures: list[Failure]) -> int:
    if not MODEL_MANIFEST.is_file():
        failures.append(Failure("missing_model_manifest", "model-assets/MANIFEST.json"))
        return 0
    try:
        manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        failures.append(Failure("invalid_model_manifest", "model-assets/MANIFEST.json"))
        return 0
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        failures.append(Failure("invalid_model_manifest", "model-assets/MANIFEST.json"))
        return 0
    seen: set[str] = set()
    for entry in assets:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            failures.append(Failure("invalid_model_manifest_entry", "model-assets/MANIFEST.json"))
            continue
        path = entry["path"]
        seen.add(path)
        if entry.get("license") != "Apache-2.0":
            failures.append(Failure("unapproved_model_licence", path))
        if entry.get("storage") != "git-lfs":
            failures.append(Failure("model_not_bound_to_lfs", path))
        if entry.get("physical_authority") is not False:
            failures.append(Failure("model_authority_not_denied", path))
        if not isinstance(entry.get("upstream"), str) or not entry["upstream"].strip():
            failures.append(Failure("model_upstream_missing", path))
        if not isinstance(entry.get("scope"), str) or not entry["scope"].strip():
            failures.append(Failure("model_scope_missing", path))
    if seen != EXPECTED_MODEL_PATHS:
        failures.append(Failure("model_inventory_drift", "model-assets/MANIFEST.json"))
    return len(assets)


def check_firmware_image(failures: list[Failure]) -> None:
    relative = "firmware/stm32f103-v15/release/FLASH_THIS_Z3_PB6_V15.hex"
    path = ROOT / relative
    if not path.is_file():
        failures.append(Failure("missing_reviewed_firmware", relative))
    elif sha256_file(path) != EXPECTED_FIRMWARE_SHA256:
        failures.append(Failure("reviewed_firmware_digest_mismatch", relative))


def main() -> int:
    failures: list[Failure] = []
    check_licence_texts(failures)
    check_policy_documents(failures)
    vendor_count = check_vendor_headers(failures)
    model_count = check_model_inventory(failures)
    check_firmware_image(failures)
    unique = sorted(set(failures))
    if unique:
        print("LICENSE_INVENTORY=FAIL")
        for failure in unique:
            print(f"- {failure.render()}")
        return 1
    print(
        "LICENSE_INVENTORY=PASS "
        f"vendor_files={vendor_count} model_artifacts={model_count}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
