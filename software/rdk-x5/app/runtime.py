"""Zero-authority production runtime bootstrap for RootScope E0.

The runtime validates immutable software facts and exposes a locked status
snapshot.  It does not import pyserial, enumerate devices, open a port, start a
background thread, or grant execution authority.  Physical activation is a
future, separately reviewed slice after commissioning evidence exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Tuple

from .config import RootScopeConfig
from .release import ReleaseManifest
from .schemas import ExecutionMode


RUNTIME_SNAPSHOT_SCHEMA_VERSION = "rootscope.production-runtime.v1"


class ProductionRuntimeState(str, Enum):
    BOOT_LOCKED = "BOOT_LOCKED"
    COMMISSIONING_LOCKED = "COMMISSIONING_LOCKED"
    RECOVERY_REQUIRED_LOCKED = "RECOVERY_REQUIRED_LOCKED"


class PhysicalActivationUnavailable(RuntimeError):
    """E0 never grants actuator authority."""


@dataclass(frozen=True)
class RuntimePreflightFacts:
    """Read-only facts supplied by future installer/preflight code."""

    release_hash_verified: bool
    immutable_capsule_matched: bool
    provisioning_receipt_trusted: bool
    preinstall_state_policy_passed: bool
    runtime_limits_passed: bool
    target_enrolled: bool
    dpkg_state_unchanged: bool

    def __post_init__(self) -> None:
        for field_name in (
            "release_hash_verified",
            "immutable_capsule_matched",
            "provisioning_receipt_trusted",
            "preinstall_state_policy_passed",
            "runtime_limits_passed",
            "target_enrolled",
            "dpkg_state_unchanged",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")

    @property
    def accepted(self) -> bool:
        return all(
            (
                self.release_hash_verified,
                self.immutable_capsule_matched,
                self.provisioning_receipt_trusted,
                self.preinstall_state_policy_passed,
                self.runtime_limits_passed,
                self.target_enrolled,
                self.dpkg_state_unchanged,
            )
        )

    def failed_gates(self) -> Tuple[str, ...]:
        return tuple(
            field_name
            for field_name in (
                "release_hash_verified",
                "immutable_capsule_matched",
                "provisioning_receipt_trusted",
                "preinstall_state_policy_passed",
                "runtime_limits_passed",
                "target_enrolled",
                "dpkg_state_unchanged",
            )
            if not getattr(self, field_name)
        )


@dataclass(frozen=True)
class ProductionRuntimeSnapshot:
    state: ProductionRuntimeState
    release_id: str
    release_profile: str
    config_sha256: str
    commissioned: bool
    blockers: Tuple[str, ...]
    hardware_touched: bool = False
    ports_enumerated: bool = False
    serial_state: str = "NOT_OPENED"
    execution_authority: bool = False
    physical_completion: bool = False
    schema_version: str = RUNTIME_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported runtime snapshot schema")
        for field_name in (
            "commissioned",
            "hardware_touched",
            "ports_enumerated",
            "execution_authority",
            "physical_completion",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if (
            self.hardware_touched
            or self.ports_enumerated
            or self.serial_state != "NOT_OPENED"
            or self.execution_authority
            or self.physical_completion
        ):
            raise ValueError("E0 runtime snapshot must remain zero-I/O and locked")
        if self.state not in {
            ProductionRuntimeState.BOOT_LOCKED,
            ProductionRuntimeState.COMMISSIONING_LOCKED,
            ProductionRuntimeState.RECOVERY_REQUIRED_LOCKED,
        }:
            raise ValueError("runtime snapshot is not fail-closed")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state.value,
            "release_id": self.release_id,
            "release_profile": self.release_profile,
            "config_sha256": self.config_sha256,
            "commissioned": self.commissioned,
            "blockers": list(self.blockers),
            "hardware_touched": self.hardware_touched,
            "ports_enumerated": self.ports_enumerated,
            "serial_state": self.serial_state,
            "execution_authority": self.execution_authority,
            "physical_completion": self.physical_completion,
        }


class ProductionRuntime:
    """Validate release/config/preflight facts and remain locked."""

    def __init__(
        self,
        config: RootScopeConfig,
        release_manifest: ReleaseManifest,
        preflight: RuntimePreflightFacts,
    ) -> None:
        self.config = config
        self.release_manifest = release_manifest
        self.preflight = preflight
        self._snapshot = ProductionRuntimeSnapshot(
            state=ProductionRuntimeState.BOOT_LOCKED,
            release_id=release_manifest.release_id,
            release_profile=release_manifest.profile_contract.profile.value,
            config_sha256=config.sha256,
            commissioned=config.commissioned,
            blockers=("locked bootstrap has not run",),
        )

    @property
    def snapshot(self) -> ProductionRuntimeSnapshot:
        return self._snapshot

    def start_locked(self) -> ProductionRuntimeSnapshot:
        """Perform pure contract checks and expose a local locked snapshot."""

        blockers = list(self.preflight.failed_gates())
        if self.config.execution_mode is not ExecutionMode.PHYSICAL:
            blockers.append("production runtime requires PHYSICAL config")
        if self.release_manifest.config_sha256 != self.config.sha256:
            blockers.append("release/config SHA-256 mismatch")
        release_sources = set(
            self.release_manifest.profile_contract.formal_perception_sources
        )
        if release_sources != set(self.config.formal_perception_sources):
            blockers.append("release/config formal perception sources mismatch")
        state = (
            ProductionRuntimeState.RECOVERY_REQUIRED_LOCKED
            if blockers
            else ProductionRuntimeState.COMMISSIONING_LOCKED
        )
        self._snapshot = ProductionRuntimeSnapshot(
            state=state,
            release_id=self.release_manifest.release_id,
            release_profile=self.release_manifest.profile_contract.profile.value,
            config_sha256=self.config.sha256,
            commissioned=self.config.commissioned,
            blockers=tuple(blockers),
        )
        return self._snapshot

    def request_physical_activation(self) -> None:
        """Refuse physical authority until the separately reviewed E5 slice."""

        raise PhysicalActivationUnavailable(
            "E0 production runtime has no physical activation implementation; "
            "serial remains NOT_OPENED"
        )
