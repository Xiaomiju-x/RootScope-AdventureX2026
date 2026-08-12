#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

EXPECTED_ARCHIVE_SHA="${1:?expected archive SHA-256 required}"
ARCHIVE="${2:?candidate archive required}"
ACTIVATE="${3:-0}"
TRUSTED_VERIFIER="${4:?uploaded trusted verifier required}"
ACCEPTANCE_SUMMARY="${5:-}"
EXPECTED_ACCEPTANCE_SHA="${6:-}"

if [[ "${ACTIVATE}" != "0" && "${ACTIVATE}" != "1" ]]; then
  echo "activate flag must be 0 or 1" >&2
  exit 10
fi
if [[ ! "${EXPECTED_ARCHIVE_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "expected archive SHA-256 is malformed" >&2
  exit 10
fi
if [[ "${ACTIVATE}" == "0" \
      && ( -n "${ACCEPTANCE_SUMMARY}" || -n "${EXPECTED_ACCEPTANCE_SHA}" ) ]]; then
  echo "StageOnly forbids an acceptance receipt" >&2
  exit 10
fi
if [[ "${ACTIVATE}" == "1" \
      && ( -z "${ACCEPTANCE_SUMMARY}" \
           || ! "${EXPECTED_ACCEPTANCE_SHA}" =~ ^[0-9a-f]{64}$ ) ]]; then
  echo "activation requires an exact acceptance summary path and SHA-256" >&2
  exit 10
fi

first_member="$(
  python3 - "${ARCHIVE}" <<'PY'
import sys
import tarfile

with tarfile.open(sys.argv[1], "r") as archive:
    first = archive.next()
    if first is None:
        raise SystemExit("empty candidate archive")
    print(first.name)
PY
)"
CANDIDATE_ID="${first_member%%/*}"
if [[ ! "${CANDIDATE_ID}" =~ ^rootscope_v3_pc_ready_20260724_[0-9a-f]{12}$ ]]; then
  echo "invalid content-addressed candidate id" >&2
  exit 10
fi

STATE_ROOT="${HOME}/.local/share/rootscope-v3"
CANDIDATES="${STATE_ROOT}/candidates"
TARGET="${CANDIDATES}/${CANDIDATE_ID}"
EVIDENCE_PARENT="${STATE_ROOT}/evidence"
RUN_ID="stage-$(date -u +%Y%m%dT%H%M%S%NZ)-$$"
EVIDENCE="${EVIDENCE_PARENT}/${RUN_ID}"

mkdir -p -m 700 "${CANDIDATES}" "${EVIDENCE_PARENT}"
mkdir -m 700 "${EVIDENCE}"
chmod 700 "${STATE_ROOT}" "${CANDIDATES}" "${EVIDENCE_PARENT}" "${EVIDENCE}"
command -v flock >/dev/null 2>&1 || {
  echo "flock is required for serialized stage/activation" >&2
  exit 10
}
exec 9>"${STATE_ROOT}/stage-activation.lock"
chmod 600 "${STATE_ROOT}/stage-activation.lock"
flock -n 9 || {
  echo "another stage/activation transaction is already running" >&2
  exit 15
}

staging="${CANDIDATES}/.${CANDIDATE_ID}.staging.$$"
next_link="${STATE_ROOT}/.current.next.$$"
rollback_link="${STATE_ROOT}/.current.rollback.$$"
activation_in_progress=0
activation_committed=0
previous=""

cleanup_staging() {
  if [[ -n "${staging:-}" && -d "${staging}" ]]; then
    chmod -R u+w "${staging}" 2>/dev/null || true
    rm -rf -- "${staging}"
  fi
}

restore_previous_current() {
  local observed=""
  if [[ -L "${STATE_ROOT}/current" ]]; then
    observed="$(readlink "${STATE_ROOT}/current")"
  elif [[ -e "${STATE_ROOT}/current" ]]; then
    echo "CRITICAL: current became a non-symlink during activation" >&2
    return 1
  fi

  if [[ -n "${previous}" ]]; then
    if [[ "${observed}" == "${previous}" ]]; then
      return 0
    fi
    if [[ -n "${observed}" && "${observed}" != "${TARGET}" ]]; then
      echo "CRITICAL: current changed to an unexpected target during activation" >&2
      return 1
    fi
    if [[ ! -L "${rollback_link}" ]]; then
      echo "CRITICAL: prepared rollback link is unavailable" >&2
      return 1
    fi
    mv -Tf "${rollback_link}" "${STATE_ROOT}/current"
    [[ -L "${STATE_ROOT}/current" \
       && "$(readlink "${STATE_ROOT}/current")" == "${previous}" ]]
    return
  fi

  if [[ -z "${observed}" ]]; then
    return 0
  fi
  if [[ "${observed}" != "${TARGET}" ]]; then
    echo "CRITICAL: current changed to an unexpected target during activation" >&2
    return 1
  fi
  rm -f -- "${STATE_ROOT}/current"
  [[ ! -e "${STATE_ROOT}/current" && ! -L "${STATE_ROOT}/current" ]]
}

record_activation_rollback() {
  python3 - \
    "${EVIDENCE}/activation_rollback.json" \
    "${EVIDENCE}/stage_receipt.json" \
    "${previous}" "${TARGET}" "${CANDIDATE_ID}" \
    "${ACCEPTANCE_SUMMARY}" "${EXPECTED_ACCEPTANCE_SHA}" <<'PY'
import json
import os
from pathlib import Path
import sys

(
    rollback_path_raw,
    stage_path_raw,
    previous,
    target,
    candidate_id,
    acceptance_summary,
    acceptance_sha,
) = sys.argv[1:]
rollback_path = Path(rollback_path_raw)
stage_path = Path(stage_path_raw)
rollback = {
    "schema": "rootscope.v3.x5-activation-rollback.v1",
    "status": "ACTIVATION_FAILED_CURRENT_ATOMICALLY_ROLLED_BACK",
    "candidate_id": candidate_id,
    "failed_target": target,
    "restored_current": previous or None,
    "acceptance_summary": acceptance_summary or None,
    "acceptance_summary_sha256": acceptance_sha or None,
    "rollback_completed": True,
}
temporary = rollback_path.with_name(
    f".{rollback_path.name}.tmp-{os.getpid()}"
)
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(rollback, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, rollback_path)

if stage_path.is_file() and not stage_path.is_symlink():
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["status"] = "ACTIVATION_FAILED_CURRENT_ATOMICALLY_ROLLED_BACK"
    stage["current"] = previous or None
    stage["current_symlink_changed"] = False
    stage["activation_transaction_committed"] = False
    stage["rollback_completed"] = True
    stage_temporary = stage_path.with_name(
        f".{stage_path.name}.rollback-{os.getpid()}"
    )
    with stage_temporary.open("x", encoding="utf-8") as handle:
        json.dump(stage, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(stage_temporary, stage_path)

directory_fd = os.open(rollback_path.parent, os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
}

cleanup_all() {
  local rc="$?"
  trap - EXIT HUP INT TERM
  if [[ "${activation_in_progress}" == "1" \
        && "${activation_committed}" != "1" ]]; then
    if ! restore_previous_current; then
      echo "CRITICAL: activation failed and automatic rollback also failed" >&2
      rc=99
    elif ! record_activation_rollback; then
      echo "CRITICAL: current was restored but rollback evidence could not be sealed" >&2
      rc=98
    fi
  fi
  rm -f -- "${next_link}" "${rollback_link}" 2>/dev/null || true
  cleanup_staging
  exit "${rc}"
}
trap cleanup_all EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

actual_sha="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
if [[ "${actual_sha}" != "${EXPECTED_ARCHIVE_SHA}" ]]; then
  echo "archive SHA-256 mismatch" >&2
  exit 11
fi

mkdir -m 700 "${staging}"
(
  # Candidate modes are part of the verified manifest.  A 022 extraction
  # umask preserves the builder's allowed 0444/0544/0555/0644/0755 modes while
  # still refusing archive ownership.
  umask 022
  tar -xf "${ARCHIVE}" -C "${staging}" \
    --no-same-owner --no-same-permissions
)
extracted="${staging}/${CANDIDATE_ID}"
if [[ ! -d "${extracted}" || -L "${extracted}" ]]; then
  echo "extracted candidate root is not one regular directory" >&2
  exit 12
fi
python3 "${TRUSTED_VERIFIER}" \
  --release-root "${extracted}" --require-x5 >"${EVIDENCE}/verify_new.json"
if [[ -L "${TARGET}" ]]; then
  echo "candidate store target must not be a symlink" >&2
  exit 12
fi
if [[ -e "${TARGET}" ]]; then
  python3 "${TRUSTED_VERIFIER}" \
    --release-root "${TARGET}" --require-x5 >"${EVIDENCE}/verify_existing.json"
  if ! cmp -s "${extracted}/candidate_manifest.json" \
      "${TARGET}/candidate_manifest.json"; then
    echo "content-address collision or stale target mismatch" >&2
    exit 12
  fi
  chmod -R u+w "${extracted}"
  rm -rf "${extracted}"
else
  mv "${extracted}" "${TARGET}"
fi
chmod -R a-w "${TARGET}"
rmdir "${staging}"
staging=""

if [[ -e "${STATE_ROOT}/current" || -L "${STATE_ROOT}/current" ]]; then
  if [[ ! -L "${STATE_ROOT}/current" ]]; then
    echo "current exists but is not a symlink" >&2
    exit 13
  fi
  previous="$(readlink "${STATE_ROOT}/current")"
  previous_resolved="$(readlink -f "${STATE_ROOT}/current")"
  previous_name="$(basename "${previous_resolved}")"
  previous_parent="$(dirname "${previous_resolved}")"
  if [[ "${previous_parent}" != "${CANDIDATES}" \
        || ! "${previous_name}" =~ ^rootscope_v3_pc_ready_20260724_[0-9a-f]{12}$ \
        || ! -d "${previous_resolved}" \
        || -L "${previous_resolved}" ]]; then
    echo "previous current target is outside the verified candidate store" >&2
    exit 13
  fi
  python3 "${TRUSTED_VERIFIER}" \
    --release-root "${previous_resolved}" --require-x5 \
    --allow-legacy-rollback \
    >"${EVIDENCE}/verify_previous_current.json"
fi

acceptance_validation_sha=""
if [[ "${ACTIVATE}" == "1" ]]; then
  python3 - \
    "${ACCEPTANCE_SUMMARY}" "${EXPECTED_ACCEPTANCE_SHA}" \
    "${EVIDENCE_PARENT}" "${TARGET}" "${CANDIDATE_ID}" \
    >"${EVIDENCE}/acceptance_validation.json" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import stat
import sys

summary_arg, expected_sha, evidence_parent_arg, target_arg, candidate_id = sys.argv[1:]
sha_re = re.compile(r"[0-9a-f]{64}")
accept_re = re.compile(r"accept-[0-9]{8}T[0-9]{6}[0-9]{9}Z-[0-9]+")


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant forbidden: {value}")


def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key forbidden: {key}")
        result[key] = value
    return result


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=reject_constant,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(value, expected, label):
    if value != expected or type(value) is not type(expected):
        raise SystemExit(f"{label} mismatch")


summary_raw = Path(summary_arg)
if not summary_raw.is_absolute():
    raise SystemExit("acceptance summary path must be absolute")
if summary_raw.is_symlink() or not summary_raw.exists():
    raise SystemExit("acceptance summary must be an existing non-symlink")
if not stat.S_ISREG(summary_raw.lstat().st_mode):
    raise SystemExit("acceptance summary must be a regular file")
summary = summary_raw.resolve(strict=True)
if summary != summary_raw:
    raise SystemExit("acceptance summary path must already be canonical")

evidence_parent = Path(evidence_parent_arg).resolve(strict=True)
if evidence_parent.is_symlink():
    raise SystemExit("evidence parent must not be a symlink")
accept_root = summary.parent
if (
    accept_root.parent != evidence_parent
    or accept_root.is_symlink()
    or not accept_root.is_dir()
    or accept_re.fullmatch(accept_root.name) is None
    or summary.name != "08_acceptance_summary.json"
):
    raise SystemExit("acceptance summary is outside one exact evidence/accept-* run")
if not sha_re.fullmatch(expected_sha) or sha256_file(summary) != expected_sha:
    raise SystemExit("acceptance summary SHA-256 mismatch")

target = Path(target_arg).resolve(strict=True)
value = load_json(summary)
exact(value.get("schema"), "rootscope.v3.x5-software-acceptance.v2", "schema")
exact(
    value.get("status"),
    (
        "PASS_X5_OFFLINE_ZERO_AUTHORITY_SOFTWARE_NATIVE_LIBDNN_"
        "LIVE_RESOURCE_STM32_PHYSICAL_PENDING"
    ),
    "status",
)
exact(value.get("candidate_id"), candidate_id, "candidate_id")
exact(value.get("release_root"), str(target), "release_root")
exact(value.get("evidence_root"), str(accept_root), "evidence_root")
exact(
    value.get("acceptance_scope"),
    "STAGED_CANDIDATE_OFFLINE_ZERO_AUTHORITY_SOFTWARE_AND_STATIC43_BPU",
    "acceptance_scope",
)
exact(
    value.get("mutation_boundary"),
    {
        "current_selected_or_modified": False,
        "service_started": False,
        "camera_opened": False,
        "serial_opened": False,
        "serial_write": False,
        "gpio_touched": False,
        "pump_touched": False,
        "physical_completion": False,
    },
    "mutation_boundary",
)
exact(
    value.get("pending"),
    {
        "live_camera": "PENDING_NOT_RUN_BY_ACCEPTANCE",
        "resource_soak": "PENDING_NOT_RUN_BY_ACCEPTANCE",
        "thermal_cma_full_stack": "PENDING_SEPARATE_QUALIFICATION",
        "stm32": "PENDING_HARDWARE_NOT_TOUCHED",
        "physical_loop": "PENDING_HARDWARE_NOT_TOUCHED",
    },
    "pending",
)

for label in ("runtime_bootstrap", "release_verification", "cpu_onnx_bm25"):
    section = value.get(label)
    if not isinstance(section, dict):
        raise SystemExit(f"{label} section missing")
    exact(section.get("passed"), True, f"{label}.passed")
native = value.get("bpu", {}).get(
    "qualification_persistent_native_libdnn", {}
)
exact(native.get("status"), "PASS_X5_PERSISTENT_NATIVE_LIBDNN", "native.status")
exact(native.get("passed"), True, "native.passed")
exact(native.get("selected_for_runtime"), False, "native.selected_for_runtime")
exact(native.get("clean_worker_exit"), True, "native.clean_worker_exit")
exact(native.get("count"), 43, "native.count")
exact(native.get("top1_agreement"), 43, "native.top1_agreement")
hrt = value.get("bpu", {}).get("canonical_hrt_oracle", {})
exact(hrt.get("passed"), True, "hrt.passed")
exact(hrt.get("count"), 43, "hrt.count")
exact(hrt.get("top1_agreement"), 43, "hrt.top1_agreement")
for role in ("fast", "deep"):
    rootmind = value.get("rootmind", {}).get(role, {})
    exact(
        rootmind.get("exact_output"),
        {"authority": False, "status": "READ_ONLY"},
        f"rootmind.{role}.exact_output",
    )
    if not isinstance(rootmind.get("forced_kill"), bool):
        raise SystemExit(f"rootmind.{role}.forced_kill must be boolean")

receipt_names = {
    "00_runtime_bootstrap.json",
    "01_release_verify.json",
    "02_cpu_bm25.json",
    "03_hrt_oracle.json",
    "04_hbm_execution.json",
    "05_native_libdnn.json",
    "06_rootmind_fast_receipt.json",
    "07_rootmind_deep_receipt.json",
    "rootmind_fast_model_binding.json",
    "rootmind_fast_model_page_cache_release.json",
    "rootmind_deep_model_binding.json",
    "rootmind_deep_model_page_cache_release.json",
    "rootmind_precondition_deep_model_binding.json",
    "rootmind_precondition_deep_model_page_cache_release.json",
    "rootmind_precondition_fast_model_binding.json",
    "rootmind_precondition_fast_model_page_cache_release.json",
}
receipt_hashes = value.get("receipts_sha256")
if not isinstance(receipt_hashes, dict) or set(receipt_hashes) != receipt_names:
    raise SystemExit("acceptance receipt hash coverage mismatch")
for name, digest in receipt_hashes.items():
    if not isinstance(digest, str) or sha_re.fullmatch(digest) is None:
        raise SystemExit(f"malformed receipt hash: {name}")
    path = accept_root / name
    if path.is_symlink() or not path.exists() or not stat.S_ISREG(path.lstat().st_mode):
        raise SystemExit(f"acceptance receipt is not a regular non-symlink: {name}")
    if sha256_file(path) != digest:
        raise SystemExit(f"acceptance receipt SHA-256 mismatch: {name}")

release_verify = load_json(accept_root / "01_release_verify.json")
native_receipt = load_json(accept_root / "05_native_libdnn.json")
exact(
    release_verify.get("status"),
    "PASS_X5_STAGED_ZERO_AUTHORITY_LIVE_QUALIFICATION_PENDING",
    "release_verify.status",
)
exact(release_verify.get("candidate_id"), candidate_id, "release_verify.candidate_id")
exact(
    release_verify.get("identity"),
    {
        "hostname": "rootscope-x5",
        "machine_id": "<redacted-device-boot-id>",
        "serial": "3281556110220e0c002bdeab0012004",
        "wlan_mac": "02:00:00:00:00:01",
    },
    "release_verify.identity",
)
exact(
    release_verify.get("authority"),
    {
        "execution_authority": False,
        "external_network": False,
        "gpio_write": False,
        "physical_completion": False,
        "pump_command": False,
        "serial_write": False,
        "state_machine_write": False,
    },
    "release_verify.authority",
)
exact(
    native_receipt.get("schema"),
    "rootscope.v3.x5-native-libdnn-qualification.v1",
    "native_receipt.schema",
)
exact(
    native_receipt.get("status"),
    "PASS_X5_PERSISTENT_NATIVE_LIBDNN",
    "native_receipt.status",
)
exact(native_receipt.get("count"), 43, "native_receipt.count")
exact(
    native_receipt.get("top1_agreement"),
    43,
    "native_receipt.top1_agreement",
)
exact(
    native_receipt.get("model_load_count"),
    1,
    "native_receipt.model_load_count",
)
exact(
    native_receipt.get("worker_lifecycle", {}).get(
        "clean_exit_no_residual_process"
    ),
    True,
    "native_receipt.clean_exit",
)
exact(
    native_receipt.get("eligible_for_zero_authority_shadow_runtime"),
    True,
    "native_receipt.eligible",
)
exact(
    native_receipt.get("selected_for_runtime"),
    False,
    "native_receipt.selected_for_runtime",
)
exact(
    native_receipt.get("selection_effect"),
    "REPORT_ONLY_NO_RUNTIME_CONFIG_MUTATION",
    "native_receipt.selection_effect",
)
exact(
    native_receipt.get("authority"),
    {
        "execution_authority": False,
        "gpio_write": False,
        "physical_completion": False,
        "pump_command": False,
        "serial_write": False,
        "state_machine_write": False,
    },
    "native_receipt.authority",
)

result = {
    "schema": "rootscope.v3.x5-acceptance-activation-binding.v1",
    "status": "PASS_EXACT_ACCEPTANCE_BOUND_TO_CANDIDATE",
    "candidate_id": candidate_id,
    "release_root": str(target),
    "candidate_manifest_sha256": sha256_file(target / "candidate_manifest.json"),
    "acceptance_summary": str(summary),
    "acceptance_summary_sha256": expected_sha,
    "receipts_sha256": dict(sorted(receipt_hashes.items())),
    "authority": release_verify["authority"],
}
print(json.dumps(result, sort_keys=True, allow_nan=False))
PY
  acceptance_validation_sha="$(
    sha256sum "${EVIDENCE}/acceptance_validation.json" | awk '{print $1}'
  )"
fi

if [[ "${ACTIVATE}" == "1" ]]; then
  if [[ -n "${previous}" ]]; then
    ln -s -- "${previous}" "${rollback_link}"
  fi
  ln -s -- "${TARGET}" "${next_link}"
  activation_in_progress=1
  mv -Tf "${next_link}" "${STATE_ROOT}/current"
  if [[ ! -L "${STATE_ROOT}/current" \
        || "$(readlink "${STATE_ROOT}/current")" != "${TARGET}" \
        || "$(readlink -f "${STATE_ROOT}/current")" != "${TARGET}" ]]; then
    echo "atomic current switch verification failed" >&2
    exit 14
  fi
  python3 "${TRUSTED_VERIFIER}" \
    --release-root "${TARGET}" --require-x5 \
    >"${EVIDENCE}/verify_activated_target.json"
fi

python3 - \
  "${EVIDENCE}/stage_receipt.json" "${actual_sha}" "${previous}" \
  "${TARGET}" "${ACTIVATE}" "${CANDIDATE_ID}" \
  "${ACCEPTANCE_SUMMARY}" "${EXPECTED_ACCEPTANCE_SHA}" \
  "${acceptance_validation_sha}" <<'PY'
import json
import os
from pathlib import Path
import sys

(
    path_raw,
    archive_sha,
    previous,
    target,
    activated,
    candidate_id,
    acceptance_summary,
    acceptance_sha,
    acceptance_validation_sha,
) = sys.argv[1:]
path = Path(path_raw)
receipt = {
    "schema": "rootscope.v3.x5-stage-receipt.v2",
    "status": (
        "STAGED_AND_ACCEPTANCE_BOUND_SOFTWARE_ACTIVATED_"
        "LIVE_QUALIFICATION_PENDING"
        if activated == "1"
        else "STAGED_ONLY_CURRENT_UNCHANGED"
    ),
    "candidate_id": candidate_id,
    "archive_sha256": archive_sha,
    "previous_current": previous or None,
    "current": target if activated == "1" else (previous or None),
    "staged_candidate": target,
    "current_symlink_changed": activated == "1",
    "acceptance_summary": acceptance_summary or None,
    "acceptance_summary_sha256": acceptance_sha or None,
    "acceptance_validation_sha256": acceptance_validation_sha or None,
    "activation_transaction_committed": activated == "1",
    "service_started": False,
    "camera_opened": False,
    "serial_opened": False,
    "pump_touched": False,
    "physical_completion": False,
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
directory_fd = os.open(path.parent, os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
print(json.dumps(receipt, sort_keys=True, allow_nan=False))
PY

if [[ "${ACTIVATE}" == "1" ]]; then
  activation_committed=1
fi
