from __future__ import annotations

import unittest

from app.release import (
    AcceptanceStatus,
    ApplicationTupleKind,
    ApplicationTupleSnapshot,
    BASELINE_EMPTY,
    BPUArtifactState,
    CPUShadowState,
    InstallAcceptanceReceipt,
    LLMArtifactState,
    MANAGED_APPLICATION_COMPONENTS,
    ManagedComponentState,
    ReceiptKind,
    ReleaseManifest,
    ReleaseProfile,
    ReleaseProfileContract,
    RollbackVerificationEvidence,
    ServiceIdentitySnapshot,
)
from app.schemas import PerceptionSource


H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
H4 = "4" * 64
H5 = "5" * 64
H6 = "6" * 64
H7 = "7" * 64
H8 = "8" * 64
H9 = "9" * 64
HA = "a" * 64
HB = "b" * 64


def manifest(contract: ReleaseProfileContract) -> ReleaseManifest:
    return ReleaseManifest(
        release_id=f"RC1-{contract.profile.value}",
        release_root_sha256=H1,
        immutable_capsule_root_sha256=H2,
        image_provisioning_receipt_schema_sha256=H4,
        preinstall_state_policy_sha256=H5,
        runtime_preflight_limits_sha256=H6,
        config_sha256=H3,
        profile_contract=contract,
    )


def rollback_evidence(
    observed: ApplicationTupleSnapshot,
    *,
    audit_root: str = HB,
    observed_binding: str | None = None,
) -> RollbackVerificationEvidence:
    return RollbackVerificationEvidence(
        observed_post_rollback_tuple=observed,
        independent_audit_root_sha256=audit_root,
        audit_receipt_observed_tuple_sha256=(
            observed.tuple_sha256 if observed_binding is None else observed_binding
        ),
        audit_tool_id="rootscope-rollback-audit-v1",
        auditor_id="independent-auditor-01",
    )


class ReleaseProfileContractTests(unittest.TestCase):
    def test_all_three_canonical_profiles_validate(self) -> None:
        full = ReleaseProfileContract.full_bpu_llm()
        bpu = ReleaseProfileContract.bpu_template()
        tag = ReleaseProfileContract.tag_template(cpu_shadow_available=False)
        self.assertEqual(full.profile, ReleaseProfile.FULL_BPU_LLM)
        self.assertTrue(full.bpu_formal_backend_loaded)
        self.assertTrue(full.llm_service_loaded)
        self.assertEqual(bpu.profile, ReleaseProfile.BPU_TEMPLATE)
        self.assertTrue(bpu.bpu_formal_backend_loaded)
        self.assertFalse(bpu.llm_service_loaded)
        self.assertEqual(tag.formal_perception_sources, (PerceptionSource.TAG,))
        self.assertEqual(tag.cpu_shadow_state, CPUShadowState.UNAVAILABLE)
        self.assertFalse(tag.bpu_formal_backend_loaded)

    def test_full_profile_cannot_silently_disable_llm(self) -> None:
        with self.assertRaisesRegex(ValueError, "illegal artifact"):
            ReleaseProfileContract(
                profile=ReleaseProfile.FULL_BPU_LLM,
                formal_perception_sources=(PerceptionSource.TAG, PerceptionSource.BPU),
                cpu_shadow_state=CPUShadowState.VALIDATED,
                bpu_artifact_state=BPUArtifactState.QUALIFIED,
                llm_artifact_state=LLMArtifactState.DISABLED_TEMPLATE_ONLY,
                bpu_formal_backend_loaded=True,
                llm_service_loaded=False,
                deterministic_template_available=True,
            )

    def test_full_and_bpu_profiles_require_exact_tag_fallback_sources(self) -> None:
        for profile, llm_state, llm_loaded in (
            (ReleaseProfile.FULL_BPU_LLM, LLMArtifactState.QUALIFIED, True),
            (
                ReleaseProfile.BPU_TEMPLATE,
                LLMArtifactState.DISABLED_TEMPLATE_ONLY,
                False,
            ),
        ):
            with self.subTest(profile=profile):
                with self.assertRaisesRegex(ValueError, "illegal artifact"):
                    ReleaseProfileContract(
                        profile=profile,
                        formal_perception_sources=(PerceptionSource.BPU,),
                        cpu_shadow_state=CPUShadowState.VALIDATED,
                        bpu_artifact_state=BPUArtifactState.QUALIFIED,
                        llm_artifact_state=llm_state,
                        bpu_formal_backend_loaded=True,
                        llm_service_loaded=llm_loaded,
                        deterministic_template_available=True,
                    )

    def test_artifact_state_fields_require_enum_instances(self) -> None:
        with self.assertRaisesRegex(ValueError, "cpu_shadow_state"):
            ReleaseProfileContract(
                profile=ReleaseProfile.TAG_TEMPLATE,
                formal_perception_sources=(PerceptionSource.TAG,),
                cpu_shadow_state="VALIDATED",  # type: ignore[arg-type]
                bpu_artifact_state=BPUArtifactState.NOT_LOADED,
                llm_artifact_state=LLMArtifactState.DISABLED_TEMPLATE_ONLY,
                bpu_formal_backend_loaded=False,
                llm_service_loaded=False,
                deterministic_template_available=True,
            )

    def test_tag_profile_cannot_load_bpu_formally(self) -> None:
        with self.assertRaisesRegex(ValueError, "illegal artifact"):
            ReleaseProfileContract(
                profile=ReleaseProfile.TAG_TEMPLATE,
                formal_perception_sources=(PerceptionSource.TAG,),
                cpu_shadow_state=CPUShadowState.VALIDATED,
                bpu_artifact_state=BPUArtifactState.NOT_LOADED,
                llm_artifact_state=LLMArtifactState.DISABLED_TEMPLATE_ONLY,
                bpu_formal_backend_loaded=True,
                llm_service_loaded=False,
                deterministic_template_available=True,
            )

    def test_every_profile_requires_template_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "deterministic templates"):
            ReleaseProfileContract(
                profile=ReleaseProfile.TAG_TEMPLATE,
                formal_perception_sources=(PerceptionSource.TAG,),
                cpu_shadow_state=CPUShadowState.UNAVAILABLE,
                bpu_artifact_state=BPUArtifactState.NOT_LOADED,
                llm_artifact_state=LLMArtifactState.DISABLED_TEMPLATE_ONLY,
                bpu_formal_backend_loaded=False,
                llm_service_loaded=False,
                deterministic_template_available=False,
            )


class ApplicationTupleTests(unittest.TestCase):
    def test_baseline_empty_is_complete_absence_inventory(self) -> None:
        self.assertEqual(BASELINE_EMPTY.tuple_kind, ApplicationTupleKind.BASELINE_EMPTY)
        self.assertEqual(
            {item.component for item in BASELINE_EMPTY.components},
            set(MANAGED_APPLICATION_COMPONENTS),
        )
        self.assertTrue(all(not item.existed for item in BASELINE_EMPTY.components))
        self.assertFalse(BASELINE_EMPTY.dpkg_state_changed)
        self.assertFalse(BASELINE_EMPTY.service_identity.user_existed)
        self.assertFalse(BASELINE_EMPTY.service_identity.group_existed)

    def test_baseline_empty_rejects_residual_unit(self) -> None:
        states = tuple(
            ManagedComponentState(
                component=name,
                existed=name == "loader_unit",
                content_sha256=H1 if name == "loader_unit" else None,
                enabled=False if name == "loader_unit" else None,
                active=False if name == "loader_unit" else None,
                masked=False if name == "loader_unit" else None,
            )
            for name in MANAGED_APPLICATION_COMPONENTS
        )
        with self.assertRaisesRegex(ValueError, "only absent"):
            ApplicationTupleSnapshot(
                tuple_kind=ApplicationTupleKind.BASELINE_EMPTY,
                components=states,
                dpkg_state_changed=False,
                service_identity=ServiceIdentitySnapshot.absent(),
            )

    def test_complete_previous_tuple_requires_hashes_and_service_facts(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires content_sha256"):
            ManagedComponentState(component="models", existed=True)
        with self.assertRaisesRegex(ValueError, "enabled/active/masked"):
            ManagedComponentState(
                component="loader_unit", existed=True, content_sha256=H1
            )

    def test_complete_previous_tuple_is_canonical_and_verifiable(self) -> None:
        snapshot = self._complete_previous_tuple(enabled=True)
        self.assertEqual(
            snapshot.tuple_kind, ApplicationTupleKind.COMPLETE_PREVIOUS_TUPLE
        )
        loader = next(
            item for item in snapshot.components if item.component == "loader_unit"
        )
        self.assertTrue(loader.enabled)
        self.assertTrue(
            next(
                item
                for item in snapshot.components
                if item.component == "wants_symlink"
            ).existed
        )
        self.assertEqual(snapshot.service_identity.uid, 991)
        self.assertEqual(snapshot.service_identity.primary_gid, 992)
        self.assertEqual(snapshot.service_identity.service_group_gid, 992)
        self.assertEqual(snapshot.service_identity.home, "/var/lib/rootscope")
        self.assertEqual(snapshot.service_identity.shell, "/usr/sbin/nologin")
        self.assertEqual(snapshot.service_identity.supplementary_group_gids, (20, 44))
        self.assertEqual(len(snapshot.tuple_sha256), 64)

    def test_complete_previous_tuple_rejects_wants_state_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "wants_symlink"):
            self._complete_previous_tuple(enabled=True, wants_exists=False)

    def test_service_identity_must_match_component_presence(self) -> None:
        complete = self._complete_previous_tuple(enabled=False)
        with self.assertRaisesRegex(ValueError, "service_user component existence"):
            ApplicationTupleSnapshot(
                tuple_kind=ApplicationTupleKind.COMPLETE_PREVIOUS_TUPLE,
                components=complete.components,
                dpkg_state_changed=False,
                service_identity=ServiceIdentitySnapshot.absent(),
            )

    def test_existing_service_identity_requires_full_passwd_and_group_facts(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires non-negative uid"):
            ServiceIdentitySnapshot(
                user_existed=True,
                group_existed=True,
                uid=None,
                primary_gid=992,
                service_group_gid=992,
                home="/var/lib/rootscope",
                shell="/usr/sbin/nologin",
                supplementary_group_gids=(20, 44),
            )

    @staticmethod
    def _complete_previous_tuple(
        *, enabled: bool, wants_exists: bool | None = None, uid: int = 991
    ) -> ApplicationTupleSnapshot:
        if wants_exists is None:
            wants_exists = enabled
        states = []
        for name in MANAGED_APPLICATION_COMPONENTS:
            existed = wants_exists if name == "wants_symlink" else True
            is_identity = name in {"service_user", "service_group"}
            is_loader = name == "loader_unit"
            states.append(
                ManagedComponentState(
                    component=name,
                    existed=existed,
                    content_sha256=(H1 if existed and not is_identity else None),
                    enabled=enabled if is_loader else None,
                    active=False if is_loader else None,
                    masked=False if is_loader else None,
                )
            )
        return ApplicationTupleSnapshot(
            tuple_kind=ApplicationTupleKind.COMPLETE_PREVIOUS_TUPLE,
            components=tuple(states),
            dpkg_state_changed=False,
            service_identity=ServiceIdentitySnapshot(
                user_existed=True,
                group_existed=True,
                uid=uid,
                primary_gid=992,
                service_group_gid=992,
                home="/var/lib/rootscope",
                shell="/usr/sbin/nologin",
                supplementary_group_gids=(20, 44),
            ),
        )


class InstallReceiptTests(unittest.TestCase):
    def test_staged_pass_cannot_claim_enabled_cold_boot(self) -> None:
        with self.assertRaisesRegex(ValueError, "staged receipt"):
            self._receipt(
                kind=ReceiptKind.STAGED_INSTALL_RECEIPT,
                service_enabled_and_boot_locked=True,
            )

    def test_final_pass_requires_enabled_locked_cold_boot(self) -> None:
        with self.assertRaisesRegex(ValueError, "final acceptance"):
            self._receipt(
                kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
                service_enabled_and_boot_locked=False,
            )
        accepted = self._receipt(
            kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
            service_enabled_and_boot_locked=True,
        )
        self.assertFalse(accepted.to_dict()["physical_completion"])
        self.assertEqual(len(accepted.receipt_sha256), 64)

    def test_install_receipt_can_never_claim_physical_completion(self) -> None:
        with self.assertRaisesRegex(ValueError, "never claim physical"):
            self._receipt(
                kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
                service_enabled_and_boot_locked=True,
                physical_completion=True,
            )

    def test_final_failure_receipt_may_report_service_not_enabled(self) -> None:
        receipt = self._receipt(
            kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
            status=AcceptanceStatus.FAIL_LOCKED,
            service_enabled_and_boot_locked=False,
        )
        self.assertEqual(receipt.status, AcceptanceStatus.FAIL_LOCKED)
        self.assertTrue(receipt.commissioning_locked)

    def test_fail_locked_must_actually_be_commissioning_locked(self) -> None:
        with self.assertRaisesRegex(ValueError, "must remain commissioning locked"):
            self._receipt(
                kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
                status=AcceptanceStatus.FAIL_LOCKED,
                service_enabled_and_boot_locked=False,
                commissioning_locked=False,
            )

    def test_receipt_requires_typed_release_and_previous_tuple(self) -> None:
        values = dict(
            receipt_id="receipt-typed",
            kind=ReceiptKind.STAGED_INSTALL_RECEIPT,
            status=AcceptanceStatus.FAIL_LOCKED,
            release_manifest=manifest(ReleaseProfileContract.tag_template()),
            previous_tuple=BASELINE_EMPTY,
            image_provisioning_receipt_sha256=H7,
            preinstall_state_audit_sha256=H8,
            runtime_preflight_receipt_sha256=H9,
            target_identity_sha256=HA,
            software_installed=False,
            release_hash_verified=False,
            os_capsule_matched=False,
            dashboard_local_ready=False,
            commissioning_locked=True,
            service_enabled_and_boot_locked=False,
            rollback_attempted=False,
            rollback_verified=False,
            recovery_required=False,
        )
        with self.assertRaisesRegex(ValueError, "release_manifest"):
            InstallAcceptanceReceipt(**{**values, "release_manifest": "bad"})  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "previous_tuple"):
            InstallAcceptanceReceipt(**{**values, "previous_tuple": "bad"})  # type: ignore[arg-type]

    def test_pass_cannot_hide_failed_attempted_rollback(self) -> None:
        with self.assertRaisesRegex(ValueError, "attempted rollback"):
            self._receipt(
                kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
                service_enabled_and_boot_locked=True,
                rollback_attempted=True,
                rollback_verified=False,
            )

    def test_rollback_verified_cannot_be_a_naked_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "observed tuple and independent audit root"):
            self._receipt(
                kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
                service_enabled_and_boot_locked=True,
                rollback_attempted=True,
                rollback_verified=True,
            )

    def test_rollback_observation_must_match_expected_previous_tuple(self) -> None:
        observed = ApplicationTupleTests._complete_previous_tuple(enabled=False)
        evidence = rollback_evidence(observed)
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            self._receipt(
                kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
                service_enabled_and_boot_locked=True,
                rollback_attempted=True,
                rollback_verified=True,
                rollback_verification=evidence,
            )

    def test_rollback_evidence_requires_independent_audit_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "independent_audit_root_sha256"):
            RollbackVerificationEvidence(
                observed_post_rollback_tuple=BASELINE_EMPTY,
                independent_audit_root_sha256="",
                audit_receipt_observed_tuple_sha256=BASELINE_EMPTY.tuple_sha256,
                audit_tool_id="rootscope-rollback-audit-v1",
                auditor_id="independent-auditor-01",
            )

    def test_rollback_audit_receipt_must_bind_observed_tuple(self) -> None:
        with self.assertRaisesRegex(ValueError, "must bind the observed tuple"):
            rollback_evidence(BASELINE_EMPTY, observed_binding=H1)

    def test_rollback_detects_previous_service_identity_mismatch(self) -> None:
        expected = ApplicationTupleTests._complete_previous_tuple(enabled=False, uid=991)
        observed = ApplicationTupleTests._complete_previous_tuple(enabled=False, uid=1991)
        evidence = rollback_evidence(observed)
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            self._receipt(
                kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
                service_enabled_and_boot_locked=True,
                previous_tuple=expected,
                rollback_attempted=True,
                rollback_verified=True,
                rollback_verification=evidence,
            )

    def test_verified_rollback_binds_observed_tuple_and_audit_root(self) -> None:
        evidence = rollback_evidence(BASELINE_EMPTY)
        receipt = self._receipt(
            kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
            service_enabled_and_boot_locked=True,
            rollback_attempted=True,
            rollback_verified=True,
            rollback_verification=evidence,
        )
        payload = receipt.to_dict()["rollback_verification"]
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["independent_audit_root_sha256"], HB)
        self.assertEqual(
            payload["observed_post_rollback_tuple_sha256"],
            BASELINE_EMPTY.tuple_sha256,
        )
        self.assertEqual(
            payload["audit_receipt_observed_tuple_sha256"],
            BASELINE_EMPTY.tuple_sha256,
        )
        self.assertEqual(payload["audit_tool_id"], "rootscope-rollback-audit-v1")
        self.assertEqual(payload["auditor_id"], "independent-auditor-01")

    def test_rollback_audit_root_cannot_reuse_existing_receipt_root(self) -> None:
        evidence = rollback_evidence(BASELINE_EMPTY, audit_root=H8)
        with self.assertRaisesRegex(ValueError, "reuses another receipt evidence root"):
            self._receipt(
                kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
                service_enabled_and_boot_locked=True,
                rollback_attempted=True,
                rollback_verified=True,
                rollback_verification=evidence,
            )

    def test_release_and_receipt_bind_all_external_evidence_roots(self) -> None:
        receipt = self._receipt(
            kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
            service_enabled_and_boot_locked=True,
        )
        payload = receipt.to_dict()
        self.assertEqual(payload["image_provisioning_receipt_schema_sha256"], H4)
        self.assertEqual(payload["preinstall_state_policy_sha256"], H5)
        self.assertEqual(payload["runtime_preflight_limits_sha256"], H6)
        self.assertEqual(payload["image_provisioning_receipt_sha256"], H7)
        self.assertEqual(payload["preinstall_state_audit_sha256"], H8)
        self.assertEqual(payload["runtime_preflight_receipt_sha256"], H9)
        self.assertEqual(payload["target_identity_sha256"], HA)

    def test_malformed_external_evidence_root_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_identity_sha256"):
            self._receipt(
                kind=ReceiptKind.FINAL_ACCEPTANCE_RECEIPT,
                service_enabled_and_boot_locked=True,
                target_identity_sha256="missing",
            )

    @staticmethod
    def _receipt(
        *,
        kind: ReceiptKind,
        status: AcceptanceStatus = AcceptanceStatus.PASS,
        service_enabled_and_boot_locked: bool,
        physical_completion: bool = False,
        commissioning_locked: bool = True,
        rollback_attempted: bool = False,
        rollback_verified: bool = False,
        rollback_verification: RollbackVerificationEvidence | None = None,
        previous_tuple: ApplicationTupleSnapshot = BASELINE_EMPTY,
        target_identity_sha256: str = HA,
    ) -> InstallAcceptanceReceipt:
        return InstallAcceptanceReceipt(
            receipt_id="receipt-0001",
            kind=kind,
            status=status,
            release_manifest=manifest(ReleaseProfileContract.tag_template()),
            previous_tuple=previous_tuple,
            image_provisioning_receipt_sha256=H7,
            preinstall_state_audit_sha256=H8,
            runtime_preflight_receipt_sha256=H9,
            target_identity_sha256=target_identity_sha256,
            software_installed=True,
            release_hash_verified=True,
            os_capsule_matched=True,
            dashboard_local_ready=True,
            commissioning_locked=commissioning_locked,
            service_enabled_and_boot_locked=service_enabled_and_boot_locked,
            rollback_attempted=rollback_attempted,
            rollback_verified=rollback_verified,
            recovery_required=False,
            rollback_verification=rollback_verification,
            physical_completion=physical_completion,
        )


if __name__ == "__main__":
    unittest.main()
