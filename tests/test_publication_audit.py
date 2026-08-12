from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publication_audit", ROOT / "tools" / "audit_public_release.py"
)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def categories(text: str, name: str = "fixture.txt") -> set[str]:
    return {
        finding.category
        for finding in audit.scan_text(ROOT / name, text)
    }


def test_private_deployment_addresses_fail_but_documentation_ranges_pass():
    assert "private_ipv4" in categories("host = 192.168.23.19")
    assert "private_ipv4" in categories("host = 10.23.45.67")
    assert "private_ipv4" in categories("host = 100.64.12.9")
    assert "private_ipv4" not in categories("host = 192.0.2.19")
    assert "private_ipv4" not in categories("host = 127.0.0.1")


def test_user_paths_fail_but_portable_examples_pass():
    assert "windows_user_path" in categories(
        "root = C:\\Users\\example-person\\Downloads\\release"
    )
    assert "linux_user_home" in categories("root = /home/operator-name/private")
    assert "windows_user_path" not in categories("font = C:\\Windows\\Fonts\\arial.ttf")
    assert "linux_user_home" not in categories("root = /home/user/rootscope")


def test_secret_shapes_are_blocked_without_echoing_the_value():
    fake = "ghp_" + "A" * 36
    findings = audit.scan_text(ROOT / "fixture.txt", f"token={fake}")
    assert any(finding.category == "github_token" for finding in findings)
    assert all(fake not in finding.render() for finding in findings)


def test_only_exact_synthetic_secret_lines_are_exempted():
    fixture_name = "pipelines/release/tests/test_rootscope_omega_v3_delta.py"
    allowed = '                \'api_key = "abcdefghijklmnopqrstuvwxyz123456"\\n\','
    assert "assigned_secret" not in categories(allowed, fixture_name)
    assert "assigned_secret" in categories(
        'api_key = "not-the-allowlisted-synthetic-value"', fixture_name
    )


def test_device_identity_shapes_are_blocked():
    assert "ssh_target" in categories("ssh sunrise@device-name")
    assert "persistent_device_path" in categories(
        "port=/dev/serial/by-id/usb-1a86_USB_Serial_REALDEVICE-if00-port0"
    )
    assert "mac_address" in categories("mac=02:11:22:33:44:55")


def test_explicit_identity_placeholders_are_safe():
    assert "persistent_device_path" not in categories("port=/dev/serial/by-id/*")
    assert "persistent_device_path" not in categories(
        "port=/dev/serial/by-id/usb-vendor_product-serial"
    )
    assert "mac_address" not in categories("mac=02:00:00:00:00:01")
    assert "machine_id_value" not in categories(
        'machine_id="00000000000000000000000000000001"'
    )


def test_only_exact_reviewed_firmware_path_is_allowlisted():
    expected = audit.REVIEWED_BINARY_ALLOWLIST[
        "firmware/stm32f103-v15/release/FLASH_THIS_Z3_PB6_V15.hex"
    ]
    assert expected["bytes"] == 72141
    assert len(expected["sha256"]) == 64
    assert "firmware/other/image.hex" not in audit.REVIEWED_BINARY_ALLOWLIST


def test_only_four_content_bound_model_artifacts_are_allowlisted():
    assert set(audit.REVIEWED_MODEL_ARTIFACTS) == {
        "model-assets/vision/rootscope_answer_cards_resnet18_opset11.onnx",
        "model-assets/bpu/rootscope_seed17_resnet18_static_b1x3x224x224_opset11.onnx",
        "model-assets/bpu/rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes.bin",
        "model-assets/rootmind-adapter/adapter_model.safetensors",
    }
    for contract in audit.REVIEWED_MODEL_ARTIFACTS.values():
        assert contract["lfs"] is True
        assert contract["bytes"] > 0
        assert len(contract["sha256"]) == 64


def test_reviewed_model_paths_resolve_to_effective_lfs_attributes():
    assert not [
        finding
        for finding in audit.git_lfs_scan()
        if finding.category in {"missing_lfs_attribute", "ineffective_lfs_attributes"}
    ]
