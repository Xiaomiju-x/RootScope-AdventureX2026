from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from app.config import RootScopeConfig
from app.release import ReleaseManifest, ReleaseProfileContract
from app.runtime import (
    PhysicalActivationUnavailable,
    ProductionRuntime,
    ProductionRuntimeState,
    RuntimePreflightFacts,
)


ROOT = Path(__file__).resolve().parents[1]
H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
H4 = "4" * 64
H5 = "5" * 64


def accepted_preflight() -> RuntimePreflightFacts:
    return RuntimePreflightFacts(
        release_hash_verified=True,
        immutable_capsule_matched=True,
        provisioning_receipt_trusted=True,
        preinstall_state_policy_passed=True,
        runtime_limits_passed=True,
        target_enrolled=True,
        dpkg_state_unchanged=True,
    )


def tag_manifest(config: RootScopeConfig) -> ReleaseManifest:
    return ReleaseManifest(
        release_id="RC1-TAG-TEMPLATE",
        release_root_sha256=H1,
        immutable_capsule_root_sha256=H2,
        image_provisioning_receipt_schema_sha256=H3,
        preinstall_state_policy_sha256=H4,
        runtime_preflight_limits_sha256=H5,
        config_sha256=config.sha256,
        profile_contract=ReleaseProfileContract.tag_template(),
    )


class ProductionRuntimeTests(unittest.TestCase):
    def test_valid_software_boot_stays_commissioning_locked_and_zero_io(self) -> None:
        config = RootScopeConfig.from_json_file(
            ROOT / "configs" / "runtime_config.example.json"
        )
        runtime = ProductionRuntime(config, tag_manifest(config), accepted_preflight())
        snapshot = runtime.start_locked()
        self.assertEqual(snapshot.state, ProductionRuntimeState.COMMISSIONING_LOCKED)
        self.assertEqual(snapshot.blockers, ())
        self.assertFalse(snapshot.hardware_touched)
        self.assertFalse(snapshot.ports_enumerated)
        self.assertEqual(snapshot.serial_state, "NOT_OPENED")
        self.assertFalse(snapshot.execution_authority)
        self.assertFalse(snapshot.physical_completion)
        self.assertFalse(snapshot.commissioned)

    def test_any_failed_preflight_gate_stays_recovery_locked(self) -> None:
        config = RootScopeConfig.from_json_file(
            ROOT / "configs" / "runtime_config.example.json"
        )
        preflight = replace(accepted_preflight(), target_enrolled=False)
        snapshot = ProductionRuntime(config, tag_manifest(config), preflight).start_locked()
        self.assertEqual(snapshot.state, ProductionRuntimeState.RECOVERY_REQUIRED_LOCKED)
        self.assertIn("target_enrolled", snapshot.blockers)
        self.assertFalse(snapshot.execution_authority)

    def test_release_config_hash_mismatch_fails_closed(self) -> None:
        config = RootScopeConfig.from_json_file(
            ROOT / "configs" / "runtime_config.example.json"
        )
        bad_manifest = replace(tag_manifest(config), config_sha256="3" * 64)
        snapshot = ProductionRuntime(
            config, bad_manifest, accepted_preflight()
        ).start_locked()
        self.assertEqual(snapshot.state, ProductionRuntimeState.RECOVERY_REQUIRED_LOCKED)
        self.assertIn("release/config SHA-256 mismatch", snapshot.blockers)

    def test_simulation_config_cannot_enter_production_runtime(self) -> None:
        config = RootScopeConfig.from_json_file(
            ROOT / "configs" / "h12_simulation_config.json"
        )
        snapshot = ProductionRuntime(
            config, tag_manifest(config), accepted_preflight()
        ).start_locked()
        self.assertEqual(snapshot.state, ProductionRuntimeState.RECOVERY_REQUIRED_LOCKED)
        self.assertIn("production runtime requires PHYSICAL config", snapshot.blockers)

    def test_e0_physical_activation_is_unconditionally_unavailable(self) -> None:
        config = RootScopeConfig.from_json_file(
            ROOT / "configs" / "runtime_config.example.json"
        )
        runtime = ProductionRuntime(config, tag_manifest(config), accepted_preflight())
        runtime.start_locked()
        with self.assertRaisesRegex(PhysicalActivationUnavailable, "NOT_OPENED"):
            runtime.request_physical_activation()
        self.assertFalse(runtime.snapshot.hardware_touched)


if __name__ == "__main__":
    unittest.main()
