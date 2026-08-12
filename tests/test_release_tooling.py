from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load_tool("build_release_bundle")
verifier = load_tool("verify_release_bundle")


def test_bundle_role_separates_large_public_payloads():
    assert builder.bundle_role("src/rootscope_public/cli.py") == "source"
    assert builder.bundle_role("assets/print-cards/card.pdf") == "source"
    assert builder.bundle_role("assets/media/demo.mp4") == "evidence"
    assert builder.bundle_role("evidence/public/receipt.json") == "evidence"
    assert builder.bundle_role("model-assets/vision/model.onnx") == "models"


def test_licence_mapping_is_path_specific():
    assert builder.licence_for_path("hardware/design/controller.kicad_sch") == "CERN-OHL-S-2.0"
    assert builder.licence_for_path(
        "firmware/stm32f103-v15/Drivers/STM32F1xx_HAL_Driver/Src/hal.c"
    ) == "BSD-3-Clause"
    assert builder.licence_for_path("docs/QUICKSTART.md") == "CC-BY-4.0"
    assert builder.licence_for_path("assets/print-cards/card.pdf") == "NOASSERTION"
    assert builder.licence_for_path("model-assets/vision/MODEL_CARD.md") == "Apache-2.0"
    assert builder.licence_for_path("software/rdk-x5/app.py") == "Apache-2.0"


def test_deterministic_archives_have_identical_hashes(tmp_path: Path):
    selected = [ROOT / "LICENSE", ROOT / "NOTICE"]
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    count_a = builder.write_deterministic_tar_gz(first, selected, "fixture", 1234567890)
    count_b = builder.write_deterministic_tar_gz(second, selected, "fixture", 1234567890)
    assert count_a == count_b == 2
    assert builder.sha256_file(first) == builder.sha256_file(second)
    inventory = verifier.validate_tar(first, "fixture", 1234567890, 2)
    assert set(inventory) == {"LICENSE", "NOTICE"}


def test_checksum_parser_rejects_duplicate_or_unsafe_names(tmp_path: Path):
    digest = "a" * 64
    valid = tmp_path / "valid"
    valid.write_text(f"{digest}  artifact.tar.gz\n", encoding="ascii")
    assert verifier.parse_checksums(valid) == {"artifact.tar.gz": digest}

    duplicate = tmp_path / "duplicate"
    duplicate.write_text(
        f"{digest}  artifact.tar.gz\n{digest}  artifact.tar.gz\n", encoding="ascii"
    )
    try:
        verifier.parse_checksums(duplicate)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate checksum entry was accepted")

    unsafe = tmp_path / "unsafe"
    unsafe.write_text(f"{digest}  ../artifact.tar.gz\n", encoding="ascii")
    try:
        verifier.parse_checksums(unsafe)
    except ValueError as error:
        assert "malformed" in str(error)
    else:
        raise AssertionError("unsafe checksum path was accepted")
