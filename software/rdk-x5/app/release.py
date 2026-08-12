"""Strict RootScope release-profile and install-receipt contracts.

These are pure schemas: they inspect no host state and perform no installation.
The installer must collect facts independently and construct one of these
objects only after validation.  The three release profiles cannot silently
degrade into one another.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping, Optional, Tuple

from .schemas import PerceptionSource


RELEASE_SCHEMA_VERSION = "rootscope.release.v1"
RECEIPT_SCHEMA_VERSION = "rootscope.install-receipt.v1"
APPLICATION_TUPLE_SCHEMA_VERSION = "rootscope.application-tuple.v1"
SERVICE_IDENTITY_SCHEMA_VERSION = "rootscope.service-identity.v1"
ROLLBACK_VERIFICATION_SCHEMA_VERSION = "rootscope.rollback-verification.v1"

_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_token(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _TOKEN_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty safe token")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class ReleaseProfile(str, Enum):
    FULL_BPU_LLM = "FULL_BPU_LLM"
    BPU_TEMPLATE = "BPU_TEMPLATE"
    TAG_TEMPLATE = "TAG_TEMPLATE"


class CPUShadowState(str, Enum):
    VALIDATED = "VALIDATED"
    UNAVAILABLE = "UNAVAILABLE"


class BPUArtifactState(str, Enum):
    QUALIFIED = "QUALIFIED"
    SHADOW_LOCKED = "SHADOW_LOCKED"
    NOT_LOADED = "NOT_LOADED"


class LLMArtifactState(str, Enum):
    QUALIFIED = "QUALIFIED"
    DISABLED_TEMPLATE_ONLY = "DISABLED_TEMPLATE_ONLY"


class ReceiptKind(str, Enum):
    STAGED_INSTALL_RECEIPT = "STAGED_INSTALL_RECEIPT"
    FINAL_ACCEPTANCE_RECEIPT = "FINAL_ACCEPTANCE_RECEIPT"


class AcceptanceStatus(str, Enum):
    PASS = "PASS"
    FAIL_LOCKED = "FAIL_LOCKED"
    RECOVERY_REQUIRED_LOCKED = "RECOVERY_REQUIRED_LOCKED"


class ApplicationTupleKind(str, Enum):
    BASELINE_EMPTY = "BASELINE_EMPTY"
    COMPLETE_PREVIOUS_TUPLE = "COMPLETE_PREVIOUS_TUPLE"


@dataclass(frozen=True)
class ReleaseProfileContract:
    """Truth table binding a named release to the services it may load."""

    profile: ReleaseProfile
    formal_perception_sources: Tuple[PerceptionSource, ...]
    cpu_shadow_state: CPUShadowState
    bpu_artifact_state: BPUArtifactState
    llm_artifact_state: LLMArtifactState
    bpu_formal_backend_loaded: bool
    llm_service_loaded: bool
    deterministic_template_available: bool

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ReleaseProfile):
            raise ValueError("profile must be a ReleaseProfile")
        if not isinstance(self.cpu_shadow_state, CPUShadowState):
            raise ValueError("cpu_shadow_state must be a CPUShadowState")
        if not isinstance(self.bpu_artifact_state, BPUArtifactState):
            raise ValueError("bpu_artifact_state must be a BPUArtifactState")
        if not isinstance(self.llm_artifact_state, LLMArtifactState):
            raise ValueError("llm_artifact_state must be an LLMArtifactState")
        if not self.formal_perception_sources:
            raise ValueError("formal_perception_sources cannot be empty")
        if any(
            not isinstance(source, PerceptionSource)
            for source in self.formal_perception_sources
        ):
            raise ValueError("formal_perception_sources must contain PerceptionSource values")
        if len(set(self.formal_perception_sources)) != len(
            self.formal_perception_sources
        ):
            raise ValueError("formal perception sources contain duplicates")
        if not set(self.formal_perception_sources).issubset(
            {PerceptionSource.TAG, PerceptionSource.BPU}
        ):
            raise ValueError("only tag and qualified BPU may be formal sources")
        for field_name in (
            "bpu_formal_backend_loaded",
            "llm_service_loaded",
            "deterministic_template_available",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if self.deterministic_template_available is not True:
            raise ValueError("every release profile requires deterministic templates")

        if self.profile is ReleaseProfile.FULL_BPU_LLM:
            valid = bool(
                self.cpu_shadow_state is CPUShadowState.VALIDATED
                and self.bpu_artifact_state is BPUArtifactState.QUALIFIED
                and self.llm_artifact_state is LLMArtifactState.QUALIFIED
                and self.formal_perception_sources
                == (PerceptionSource.TAG, PerceptionSource.BPU)
                and self.bpu_formal_backend_loaded
                and self.llm_service_loaded
            )
        elif self.profile is ReleaseProfile.BPU_TEMPLATE:
            valid = bool(
                self.cpu_shadow_state is CPUShadowState.VALIDATED
                and self.bpu_artifact_state is BPUArtifactState.QUALIFIED
                and self.llm_artifact_state
                is LLMArtifactState.DISABLED_TEMPLATE_ONLY
                and self.formal_perception_sources
                == (PerceptionSource.TAG, PerceptionSource.BPU)
                and self.bpu_formal_backend_loaded
                and not self.llm_service_loaded
            )
        else:
            valid = bool(
                self.formal_perception_sources == (PerceptionSource.TAG,)
                and self.bpu_artifact_state
                in {BPUArtifactState.SHADOW_LOCKED, BPUArtifactState.NOT_LOADED}
                and self.llm_artifact_state
                is LLMArtifactState.DISABLED_TEMPLATE_ONLY
                and not self.bpu_formal_backend_loaded
                and not self.llm_service_loaded
            )
        if not valid:
            raise ValueError(f"illegal artifact/service combination for {self.profile.value}")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "profile": self.profile.value,
            "formal_perception_sources": [
                source.value for source in self.formal_perception_sources
            ],
            "cpu_shadow_state": self.cpu_shadow_state.value,
            "bpu_artifact_state": self.bpu_artifact_state.value,
            "llm_artifact_state": self.llm_artifact_state.value,
            "bpu_formal_backend_loaded": self.bpu_formal_backend_loaded,
            "llm_service_loaded": self.llm_service_loaded,
            "deterministic_template_available": (
                self.deterministic_template_available
            ),
        }

    @classmethod
    def full_bpu_llm(cls) -> "ReleaseProfileContract":
        return cls(
            profile=ReleaseProfile.FULL_BPU_LLM,
            formal_perception_sources=(PerceptionSource.TAG, PerceptionSource.BPU),
            cpu_shadow_state=CPUShadowState.VALIDATED,
            bpu_artifact_state=BPUArtifactState.QUALIFIED,
            llm_artifact_state=LLMArtifactState.QUALIFIED,
            bpu_formal_backend_loaded=True,
            llm_service_loaded=True,
            deterministic_template_available=True,
        )

    @classmethod
    def bpu_template(cls) -> "ReleaseProfileContract":
        return cls(
            profile=ReleaseProfile.BPU_TEMPLATE,
            formal_perception_sources=(PerceptionSource.TAG, PerceptionSource.BPU),
            cpu_shadow_state=CPUShadowState.VALIDATED,
            bpu_artifact_state=BPUArtifactState.QUALIFIED,
            llm_artifact_state=LLMArtifactState.DISABLED_TEMPLATE_ONLY,
            bpu_formal_backend_loaded=True,
            llm_service_loaded=False,
            deterministic_template_available=True,
        )

    @classmethod
    def tag_template(
        cls, *, cpu_shadow_available: bool = True
    ) -> "ReleaseProfileContract":
        if not isinstance(cpu_shadow_available, bool):
            raise ValueError("cpu_shadow_available must be boolean")
        return cls(
            profile=ReleaseProfile.TAG_TEMPLATE,
            formal_perception_sources=(PerceptionSource.TAG,),
            cpu_shadow_state=(
                CPUShadowState.VALIDATED
                if cpu_shadow_available
                else CPUShadowState.UNAVAILABLE
            ),
            bpu_artifact_state=BPUArtifactState.NOT_LOADED,
            llm_artifact_state=LLMArtifactState.DISABLED_TEMPLATE_ONLY,
            bpu_formal_backend_loaded=False,
            llm_service_loaded=False,
            deterministic_template_available=True,
        )


@dataclass(frozen=True)
class ReleaseManifest:
    release_id: str
    release_root_sha256: str
    immutable_capsule_root_sha256: str
    image_provisioning_receipt_schema_sha256: str
    preinstall_state_policy_sha256: str
    runtime_preflight_limits_sha256: str
    config_sha256: str
    profile_contract: ReleaseProfileContract
    schema_version: str = RELEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELEASE_SCHEMA_VERSION:
            raise ValueError("unsupported release schema version")
        _require_token(self.release_id, "release_id")
        for field_name in (
            "release_root_sha256",
            "immutable_capsule_root_sha256",
            "image_provisioning_receipt_schema_sha256",
            "preinstall_state_policy_sha256",
            "runtime_preflight_limits_sha256",
            "config_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if not isinstance(self.profile_contract, ReleaseProfileContract):
            raise ValueError("profile_contract must be validated")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "release_root_sha256": self.release_root_sha256,
            "immutable_capsule_root_sha256": self.immutable_capsule_root_sha256,
            "image_provisioning_receipt_schema_sha256": (
                self.image_provisioning_receipt_schema_sha256
            ),
            "preinstall_state_policy_sha256": self.preinstall_state_policy_sha256,
            "runtime_preflight_limits_sha256": self.runtime_preflight_limits_sha256,
            "config_sha256": self.config_sha256,
            "profile_contract": self.profile_contract.to_dict(),
        }

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


MANAGED_APPLICATION_COMPONENTS = (
    "app_release",
    "virtualenv",
    "commissioning_config",
    "models",
    "labels",
    "loader_unit",
    "udev_rule_f407",
    "udev_rule_uvc",
    "opt_current_symlink",
    "etc_current_symlink",
    "service_user",
    "service_group",
    "wants_symlink",
)

_CONTENT_BEARING_COMPONENTS = frozenset(MANAGED_APPLICATION_COMPONENTS) - {
    "service_user",
    "service_group",
}
_REQUIRED_COMPLETE_COMPONENTS = frozenset(MANAGED_APPLICATION_COMPONENTS) - {
    "wants_symlink",
}


@dataclass(frozen=True)
class ManagedComponentState:
    component: str
    existed: bool
    content_sha256: Optional[str] = None
    enabled: Optional[bool] = None
    active: Optional[bool] = None
    masked: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.component not in MANAGED_APPLICATION_COMPONENTS:
            raise ValueError("unknown managed application component")
        if not isinstance(self.existed, bool):
            raise ValueError("existed must be boolean")
        if self.content_sha256 is not None:
            _require_sha256(self.content_sha256, "content_sha256")
        for field_name in ("enabled", "active", "masked"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field_name} must be boolean or null")
        if not self.existed and (
            self.content_sha256 is not None
            or self.enabled is not None
            or self.active is not None
            or self.masked is not None
        ):
            raise ValueError("absent component cannot carry residual state")
        if self.existed and self.component in _CONTENT_BEARING_COMPONENTS:
            if self.content_sha256 is None:
                raise ValueError(
                    f"existing {self.component} requires content_sha256"
                )
        if self.component in {"service_user", "service_group"}:
            if self.content_sha256 is not None:
                raise ValueError(f"{self.component} cannot use a file content hash")
        if self.component == "loader_unit" and self.existed:
            if any(
                getattr(self, field_name) is None
                for field_name in ("enabled", "active", "masked")
            ):
                raise ValueError(
                    "existing loader_unit requires enabled/active/masked facts"
                )
        elif self.component != "loader_unit" and any(
            getattr(self, field_name) is not None
            for field_name in ("enabled", "active", "masked")
        ):
            raise ValueError("systemd state facts are valid only for loader_unit")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "component": self.component,
            "existed": self.existed,
            "content_sha256": self.content_sha256,
            "enabled": self.enabled,
            "active": self.active,
            "masked": self.masked,
        }


@dataclass(frozen=True)
class ServiceIdentitySnapshot:
    """Exact pre-install service identity, including explicit absence."""

    user_existed: bool
    group_existed: bool
    uid: Optional[int]
    primary_gid: Optional[int]
    service_group_gid: Optional[int]
    home: Optional[str]
    shell: Optional[str]
    supplementary_group_gids: Tuple[int, ...]
    schema_version: str = SERVICE_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SERVICE_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported service identity schema")
        for field_name in ("user_existed", "group_existed"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if not isinstance(self.supplementary_group_gids, tuple):
            raise ValueError("supplementary_group_gids must be a tuple")
        if any(type(gid) is not int or gid < 0 for gid in self.supplementary_group_gids):
            raise ValueError("supplementary_group_gids must contain non-negative integers")
        if tuple(sorted(set(self.supplementary_group_gids))) != self.supplementary_group_gids:
            raise ValueError("supplementary_group_gids must be unique and sorted")

        if self.user_existed:
            for field_name in ("uid", "primary_gid"):
                value = getattr(self, field_name)
                if type(value) is not int or value < 0:
                    raise ValueError(f"existing service user requires non-negative {field_name}")
            for field_name in ("home", "shell"):
                value = getattr(self, field_name)
                if (
                    not isinstance(value, str)
                    or not value.startswith("/")
                    or "\x00" in value
                    or "\n" in value
                    or "\r" in value
                ):
                    raise ValueError(
                        f"existing service user requires absolute, single-line {field_name}"
                    )
        elif any(
            value is not None
            for value in (self.uid, self.primary_gid, self.home, self.shell)
        ) or self.supplementary_group_gids:
            raise ValueError("absent service user cannot carry identity facts")

        if self.group_existed:
            if type(self.service_group_gid) is not int or self.service_group_gid < 0:
                raise ValueError("existing service group requires non-negative service_group_gid")
        elif self.service_group_gid is not None:
            raise ValueError("absent service group cannot carry service_group_gid")

    @classmethod
    def absent(cls) -> "ServiceIdentitySnapshot":
        return cls(
            user_existed=False,
            group_existed=False,
            uid=None,
            primary_gid=None,
            service_group_gid=None,
            home=None,
            shell=None,
            supplementary_group_gids=(),
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "user_existed": self.user_existed,
            "group_existed": self.group_existed,
            "uid": self.uid,
            "primary_gid": self.primary_gid,
            "service_group_gid": self.service_group_gid,
            "home": self.home,
            "shell": self.shell,
            "supplementary_group_gids": list(self.supplementary_group_gids),
        }


@dataclass(frozen=True)
class ApplicationTupleSnapshot:
    tuple_kind: ApplicationTupleKind
    components: Tuple[ManagedComponentState, ...]
    dpkg_state_changed: bool
    service_identity: ServiceIdentitySnapshot
    schema_version: str = APPLICATION_TUPLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != APPLICATION_TUPLE_SCHEMA_VERSION:
            raise ValueError("unsupported application tuple schema")
        if not isinstance(self.tuple_kind, ApplicationTupleKind):
            raise ValueError("tuple_kind must be an ApplicationTupleKind")
        if not isinstance(self.dpkg_state_changed, bool):
            raise ValueError("dpkg_state_changed must be boolean")
        if not isinstance(self.service_identity, ServiceIdentitySnapshot):
            raise ValueError("service_identity must be a ServiceIdentitySnapshot")
        if any(
            not isinstance(component, ManagedComponentState)
            for component in self.components
        ):
            raise ValueError("components must contain ManagedComponentState values")
        names = tuple(component.component for component in self.components)
        if len(set(names)) != len(names):
            raise ValueError("application tuple contains duplicate components")
        if names != MANAGED_APPLICATION_COMPONENTS:
            raise ValueError("application tuple inventory must be complete and ordered")
        by_name = {component.component: component for component in self.components}
        if by_name["service_user"].existed is not self.service_identity.user_existed:
            raise ValueError("service_user component existence must match service identity")
        if by_name["service_group"].existed is not self.service_identity.group_existed:
            raise ValueError("service_group component existence must match service identity")
        if self.tuple_kind is ApplicationTupleKind.BASELINE_EMPTY:
            if self.dpkg_state_changed or any(
                component.existed for component in self.components
            ):
                raise ValueError("BASELINE_EMPTY must contain only absent components")
        else:
            missing = sorted(
                name
                for name in _REQUIRED_COMPLETE_COMPONENTS
                if not by_name[name].existed
            )
            if missing:
                raise ValueError(
                    f"COMPLETE_PREVIOUS_TUPLE missing installed state: {missing}"
                )
            loader = by_name["loader_unit"]
            wants = by_name["wants_symlink"]
            if wants.existed is not loader.enabled:
                raise ValueError(
                    "wants_symlink existence must match loader_unit enabled state"
                )
            if self.dpkg_state_changed:
                raise ValueError("application tuple cannot include a dpkg state change")

    @classmethod
    def baseline_empty(cls) -> "ApplicationTupleSnapshot":
        return cls(
            tuple_kind=ApplicationTupleKind.BASELINE_EMPTY,
            components=tuple(
                ManagedComponentState(component=name, existed=False)
                for name in MANAGED_APPLICATION_COMPONENTS
            ),
            dpkg_state_changed=False,
            service_identity=ServiceIdentitySnapshot.absent(),
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tuple_kind": self.tuple_kind.value,
            "components": [component.to_dict() for component in self.components],
            "dpkg_state_changed": self.dpkg_state_changed,
            "service_identity": self.service_identity.to_dict(),
        }

    @property
    def tuple_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


BASELINE_EMPTY = ApplicationTupleSnapshot.baseline_empty()


@dataclass(frozen=True)
class RollbackVerificationEvidence:
    """Independent evidence of the exact tuple observed after rollback."""

    observed_post_rollback_tuple: ApplicationTupleSnapshot
    independent_audit_root_sha256: str
    audit_receipt_observed_tuple_sha256: str
    audit_tool_id: str
    auditor_id: str
    schema_version: str = ROLLBACK_VERIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROLLBACK_VERIFICATION_SCHEMA_VERSION:
            raise ValueError("unsupported rollback verification schema")
        if not isinstance(self.observed_post_rollback_tuple, ApplicationTupleSnapshot):
            raise ValueError(
                "observed_post_rollback_tuple must be an ApplicationTupleSnapshot"
            )
        _require_sha256(
            self.independent_audit_root_sha256,
            "independent_audit_root_sha256",
        )
        _require_sha256(
            self.audit_receipt_observed_tuple_sha256,
            "audit_receipt_observed_tuple_sha256",
        )
        _require_token(self.audit_tool_id, "audit_tool_id")
        _require_token(self.auditor_id, "auditor_id")
        if (
            self.audit_receipt_observed_tuple_sha256
            != self.observed_post_rollback_tuple.tuple_sha256
        ):
            raise ValueError(
                "rollback audit receipt must bind the observed tuple SHA-256"
            )
        if (
            self.independent_audit_root_sha256
            == self.observed_post_rollback_tuple.tuple_sha256
        ):
            raise ValueError("independent audit root cannot be the observed tuple hash")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observed_post_rollback_tuple_sha256": (
                self.observed_post_rollback_tuple.tuple_sha256
            ),
            "observed_post_rollback_tuple": (
                self.observed_post_rollback_tuple.to_dict()
            ),
            "independent_audit_root_sha256": self.independent_audit_root_sha256,
            "audit_receipt_observed_tuple_sha256": (
                self.audit_receipt_observed_tuple_sha256
            ),
            "audit_tool_id": self.audit_tool_id,
            "auditor_id": self.auditor_id,
        }


@dataclass(frozen=True)
class InstallAcceptanceReceipt:
    """Fact-only install receipt; never claims physical completion."""

    receipt_id: str
    kind: ReceiptKind
    status: AcceptanceStatus
    release_manifest: ReleaseManifest
    previous_tuple: ApplicationTupleSnapshot
    image_provisioning_receipt_sha256: str
    preinstall_state_audit_sha256: str
    runtime_preflight_receipt_sha256: str
    target_identity_sha256: str
    software_installed: bool
    release_hash_verified: bool
    os_capsule_matched: bool
    dashboard_local_ready: bool
    commissioning_locked: bool
    service_enabled_and_boot_locked: bool
    rollback_attempted: bool
    rollback_verified: bool
    recovery_required: bool
    rollback_verification: Optional[RollbackVerificationEvidence] = None
    physical_completion: bool = False
    schema_version: str = RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA_VERSION:
            raise ValueError("unsupported install receipt schema")
        _require_token(self.receipt_id, "receipt_id")
        if not isinstance(self.kind, ReceiptKind):
            raise ValueError("kind must be a ReceiptKind")
        if not isinstance(self.status, AcceptanceStatus):
            raise ValueError("status must be an AcceptanceStatus")
        if not isinstance(self.release_manifest, ReleaseManifest):
            raise ValueError("release_manifest must be a ReleaseManifest")
        if not isinstance(self.previous_tuple, ApplicationTupleSnapshot):
            raise ValueError("previous_tuple must be an ApplicationTupleSnapshot")
        for field_name in (
            "image_provisioning_receipt_sha256",
            "preinstall_state_audit_sha256",
            "runtime_preflight_receipt_sha256",
            "target_identity_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        for field_name in (
            "software_installed",
            "release_hash_verified",
            "os_capsule_matched",
            "dashboard_local_ready",
            "commissioning_locked",
            "service_enabled_and_boot_locked",
            "rollback_attempted",
            "rollback_verified",
            "recovery_required",
            "physical_completion",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if self.physical_completion:
            raise ValueError("install receipts can never claim physical completion")
        if self.rollback_verified and not self.rollback_attempted:
            raise ValueError("rollback_verified requires rollback_attempted")
        if self.rollback_verification is not None and not isinstance(
            self.rollback_verification, RollbackVerificationEvidence
        ):
            raise ValueError(
                "rollback_verification must be RollbackVerificationEvidence or null"
            )
        if self.rollback_verified:
            if self.rollback_verification is None:
                raise ValueError(
                    "rollback_verified requires observed tuple and independent audit root"
                )
            observed = self.rollback_verification.observed_post_rollback_tuple
            if observed.tuple_sha256 != self.previous_tuple.tuple_sha256:
                raise ValueError(
                    "observed post-rollback tuple does not exactly match previous_tuple"
                )
            audit_root = self.rollback_verification.independent_audit_root_sha256
            non_independent_roots = {
                self.previous_tuple.tuple_sha256,
                self.release_manifest.manifest_sha256,
                self.release_manifest.release_root_sha256,
                self.release_manifest.immutable_capsule_root_sha256,
                self.release_manifest.image_provisioning_receipt_schema_sha256,
                self.release_manifest.preinstall_state_policy_sha256,
                self.release_manifest.runtime_preflight_limits_sha256,
                self.release_manifest.config_sha256,
                self.image_provisioning_receipt_sha256,
                self.preinstall_state_audit_sha256,
                self.runtime_preflight_receipt_sha256,
                self.target_identity_sha256,
            }
            if audit_root in non_independent_roots:
                raise ValueError(
                    "rollback independent audit root reuses another receipt evidence root"
                )
        elif self.rollback_verification is not None:
            raise ValueError("rollback evidence is valid only when rollback_verified is true")
        if self.kind is ReceiptKind.STAGED_INSTALL_RECEIPT:
            if self.service_enabled_and_boot_locked:
                raise ValueError("staged receipt cannot claim enabled cold boot")
        elif (
            self.status is AcceptanceStatus.PASS
            and not self.service_enabled_and_boot_locked
        ):
            raise ValueError("final acceptance requires enabled locked cold boot")
        if self.status is AcceptanceStatus.PASS:
            if self.recovery_required:
                raise ValueError("PASS cannot require recovery")
            required = (
                self.software_installed,
                self.release_hash_verified,
                self.os_capsule_matched,
                self.dashboard_local_ready,
                self.commissioning_locked,
            )
            if not all(required):
                raise ValueError("PASS receipt is missing a mandatory software claim")
            if self.rollback_attempted and not self.rollback_verified:
                raise ValueError("PASS requires an attempted rollback to be verified")
        if self.status is AcceptanceStatus.RECOVERY_REQUIRED_LOCKED:
            if not self.recovery_required or not self.commissioning_locked:
                raise ValueError("recovery-required receipt must remain locked")
        elif self.status is AcceptanceStatus.FAIL_LOCKED:
            if not self.commissioning_locked:
                raise ValueError("FAIL_LOCKED receipt must remain commissioning locked")
            if self.recovery_required:
                raise ValueError("recovery_required must match receipt status")
        elif self.recovery_required:
            raise ValueError("recovery_required must match receipt status")

    def to_dict(self) -> Mapping[str, Any]:
        contract = self.release_manifest.profile_contract
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "receipt_kind": self.kind.value,
            "acceptance_status": self.status.value,
            "release_manifest_sha256": self.release_manifest.manifest_sha256,
            "release_id": self.release_manifest.release_id,
            "release_root_sha256": self.release_manifest.release_root_sha256,
            "image_provisioning_receipt_schema_sha256": (
                self.release_manifest.image_provisioning_receipt_schema_sha256
            ),
            "preinstall_state_policy_sha256": (
                self.release_manifest.preinstall_state_policy_sha256
            ),
            "runtime_preflight_limits_sha256": (
                self.release_manifest.runtime_preflight_limits_sha256
            ),
            "image_provisioning_receipt_sha256": (
                self.image_provisioning_receipt_sha256
            ),
            "preinstall_state_audit_sha256": self.preinstall_state_audit_sha256,
            "runtime_preflight_receipt_sha256": (
                self.runtime_preflight_receipt_sha256
            ),
            "target_identity_sha256": self.target_identity_sha256,
            "release_profile": contract.profile.value,
            "cpu_shadow_state": contract.cpu_shadow_state.value,
            "bpu_artifact_state": contract.bpu_artifact_state.value,
            "llm_artifact_state": contract.llm_artifact_state.value,
            "previous_tuple": self.previous_tuple.to_dict(),
            "software_installed": self.software_installed,
            "release_hash_verified": self.release_hash_verified,
            "os_capsule_matched": self.os_capsule_matched,
            "dashboard_local_ready": self.dashboard_local_ready,
            "commissioning_locked": self.commissioning_locked,
            "service_enabled_and_boot_locked": (
                self.service_enabled_and_boot_locked
            ),
            "rollback_attempted": self.rollback_attempted,
            "rollback_verified": self.rollback_verified,
            "rollback_verification": (
                self.rollback_verification.to_dict()
                if self.rollback_verification is not None
                else None
            ),
            "recovery_required": self.recovery_required,
            "physical_completion": self.physical_completion,
        }

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()
