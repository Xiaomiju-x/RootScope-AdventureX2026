#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
umask 077
export LC_ALL=C

RAW_RELEASE_ROOT="${1:?explicit staged candidate root required}"
[[ ! -L "${RAW_RELEASE_ROOT}" ]] || {
  echo "candidate root must not be a symlink (current is never accepted here)" >&2
  exit 31
}
RELEASE_ROOT="$(readlink -f "${RAW_RELEASE_ROOT}")"
[[ -d "${RELEASE_ROOT}" \
   && "${RELEASE_ROOT}" =~ /rootscope_v3_pc_ready_20260724_[0-9a-f]{12}$ ]] || {
  echo "invalid staged v3 candidate root" >&2
  exit 31
}

STATE_ROOT="${HOME}/.local/share/rootscope-v3"
EVIDENCE_PARENT="${STATE_ROOT}/evidence"
RUN_ID="accept-$(date -u +%Y%m%dT%H%M%S%NZ)-$$"
EVIDENCE="${EVIDENCE_PARENT}/${RUN_ID}"
mkdir -p -m 700 "${EVIDENCE_PARENT}"
mkdir -m 700 "${EVIDENCE}"

# This is deliberately a staged-candidate, zero-authority acceptance.  It does
# not select/activate current, start a service, open a camera or touch serial,
# GPIO, a pump, or the physical state machine.
bash "${RELEASE_ROOT}/tools/release_v3/x5_bootstrap_runtime_v3.sh" \
  "${RELEASE_ROOT}" "${EVIDENCE}" \
  >"${EVIDENCE}/00_runtime_bootstrap.stdout.json"

runtime_lines=()
mapfile -t runtime_lines <"${EVIDENCE}/runtime_paths.env"
[[ "${#runtime_lines[@]}" == 2 \
   && "${runtime_lines[0]}" == ROOTSCOPE_CPU_PYTHON=* \
   && "${runtime_lines[1]}" == ROOTSCOPE_BPU_PYTHON=* ]] || {
  echo "candidate runtime_paths.env is not the exact two-line contract" >&2
  exit 32
}
ROOTSCOPE_CPU_PYTHON="${runtime_lines[0]#ROOTSCOPE_CPU_PYTHON=}"
ROOTSCOPE_BPU_PYTHON="${runtime_lines[1]#ROOTSCOPE_BPU_PYTHON=}"
CANDIDATE_ID="$(basename "${RELEASE_ROOT}")"
[[ "${ROOTSCOPE_CPU_PYTHON}" \
      == "${STATE_ROOT}/venvs/${CANDIDATE_ID}-cpu/bin/python3" \
   && "${ROOTSCOPE_BPU_PYTHON}" \
      == "${STATE_ROOT}/venvs/${CANDIDATE_ID}-bpu-system-site/bin/python3" \
   && -x "${ROOTSCOPE_CPU_PYTHON}" \
   && -x "${ROOTSCOPE_BPU_PYTHON}" ]] || {
  echo "candidate runtime interpreter binding changed" >&2
  exit 32
}

"${ROOTSCOPE_CPU_PYTHON}" -I \
  "${RELEASE_ROOT}/tools/release_v3/verify_rootscope_v3_release.py" \
  --release-root "${RELEASE_ROOT}" --require-x5 \
  >"${EVIDENCE}/01_release_verify.json"

PYTHONPATH="${RELEASE_ROOT}/rootscope:${RELEASE_ROOT}" \
  "${ROOTSCOPE_CPU_PYTHON}" \
  "${RELEASE_ROOT}/tools/release_v3/x5_cpu_bm25_accept_v3.py" \
  --release-root "${RELEASE_ROOT}" \
  --output "${EVIDENCE}/02_cpu_bm25.json" \
  >"${EVIDENCE}/02_cpu_bm25.stdout.json"

MODEL="${RELEASE_ROOT}/models/rootscope_seed17_resnet18_224x224_rgb_ddr_r7.bin"
MODEL_SHA="4dc4bc24741332bb6bc6da184dd1d6f954ae539ba740330490339d76eb200285"
ORACLE="${RELEASE_ROOT}/rootscope/configs/competition_v3/hbm_persistent_oracle_43.v1.json"
INPUT_ROOT="${RELEASE_ROOT}/inputs/static43"

# Canonical vendor numerical oracle: all 43 immutable inputs must pass.
PYTHONPATH="${RELEASE_ROOT}/rootscope:${RELEASE_ROOT}" \
  "${ROOTSCOPE_CPU_PYTHON}" \
  "${RELEASE_ROOT}/rootscope/tools/x5_hrt_oracle_qualify_v3.py" \
  --model "${MODEL}" \
  --model-sha256 "${MODEL_SHA}" \
  --input-root "${INPUT_ROOT}" \
  --oracle-manifest "${ORACLE}" \
  --hrt-model-exec /usr/sbin/hrt_model_exec \
  --work-root "${EVIDENCE}/03_hrt_work" \
  --output "${EVIDENCE}/03_hrt_oracle.json" \
  >"${EVIDENCE}/03_hrt_oracle.stdout.json"

# The legacy Python hbm_runtime wrapper is retained only as a truthful,
# non-authoritative forensic observation.  Neither policy can satisfy the
# acceptance gate; their expected FAIL_CLOSED results do not veto the native
# backend that follows.
set +e
PYTHONPATH="${RELEASE_ROOT}/rootscope:${RELEASE_ROOT}" \
  timeout --signal=TERM --kill-after=5s 120s \
  "${ROOTSCOPE_BPU_PYTHON}" \
  "${RELEASE_ROOT}/rootscope/tools/x5_hbm_persistent_qualify_v3.py" \
  --model "${MODEL}" \
  --model-sha256 "${MODEL_SHA}" \
  --input-root "${INPUT_ROOT}" \
  --oracle-manifest "${ORACLE}" \
  --input-policy RAW_UINT8 \
  --output "${EVIDENCE}/04_hbm_raw.json" \
  >"${EVIDENCE}/04_hbm_raw.stdout.json" \
  2>"${EVIDENCE}/04_hbm_raw.stderr.log"
raw_code="$?"
PYTHONPATH="${RELEASE_ROOT}/rootscope:${RELEASE_ROOT}" \
  timeout --signal=TERM --kill-after=5s 120s \
  "${ROOTSCOPE_BPU_PYTHON}" \
  "${RELEASE_ROOT}/rootscope/tools/x5_hbm_persistent_qualify_v3.py" \
  --model "${MODEL}" \
  --model-sha256 "${MODEL_SHA}" \
  --input-root "${INPUT_ROOT}" \
  --oracle-manifest "${ORACLE}" \
  --input-policy RGB128_CENTERED_INT8 \
  --output "${EVIDENCE}/04_hbm_centered.json" \
  >"${EVIDENCE}/04_hbm_centered.stdout.json" \
  2>"${EVIDENCE}/04_hbm_centered.stderr.log"
centered_code="$?"
set -e
"${ROOTSCOPE_CPU_PYTHON}" -I - \
  "${EVIDENCE}/04_hbm_execution.json" \
  "${raw_code}" "${centered_code}" <<'PY'
import json
import hashlib
import math
from pathlib import Path
import sys

output = Path(sys.argv[1])
raw_code = int(sys.argv[2])
centered_code = int(sys.argv[3])


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant forbidden: {value}")


def finite_number_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def observe(policy, code, receipt_name, stderr_name):
    receipt_path = output.parent / receipt_name
    stderr_path = output.parent / stderr_name
    observation = {
        "policy": policy,
        "exit_code": code,
        "receipt_present": receipt_path.is_file(),
        "receipt_sha256": (
            sha256_file(receipt_path) if receipt_path.is_file() else None
        ),
        "stderr_sha256": (
            sha256_file(stderr_path) if stderr_path.is_file() else None
        ),
        "observed_status": "NO_RECEIPT",
        "observed_backend": None,
        "observed_count": None,
        "observed_top1_agreement": None,
        "observed_mean_cosine": None,
        "parse_error": None,
    }
    if receipt_path.is_file():
        try:
            receipt = json.loads(
                receipt_path.read_text(encoding="utf-8"),
                parse_constant=reject_constant,
            )
            observation.update(
                {
                    "observed_status": (
                        receipt.get("status")
                        if isinstance(receipt.get("status"), str)
                        else "MALFORMED_RECEIPT"
                    ),
                    "observed_backend": (
                        receipt.get("backend_actual")
                        if isinstance(receipt.get("backend_actual"), str)
                        else None
                    ),
                    "observed_count": finite_number_or_none(
                        receipt.get("count")
                    ),
                    "observed_top1_agreement": finite_number_or_none(
                        receipt.get("top1_agreement")
                    ),
                    "observed_mean_cosine": finite_number_or_none(
                        receipt.get("mean_cosine")
                    ),
                }
            )
        except BaseException as exc:
            observation["observed_status"] = "MALFORMED_RECEIPT"
            observation["parse_error"] = f"{type(exc).__name__}: {exc}"
    return observation


value = {
    "schema": "rootscope.v3.legacy-python-hbm-observation.v1",
    "authoritative": False,
    "affects_acceptance": False,
    "timeout_seconds_each": 120,
    "observations": {
        "RAW_UINT8": observe(
            "RAW_UINT8",
            raw_code,
            "04_hbm_raw.json",
            "04_hbm_raw.stderr.log",
        ),
        "RGB128_CENTERED_INT8": observe(
            "RGB128_CENTERED_INT8",
            centered_code,
            "04_hbm_centered.json",
            "04_hbm_centered.stderr.log",
        ),
    },
}
with output.open("x", encoding="utf-8") as handle:
    json.dump(value, handle, ensure_ascii=True, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY

# Primary persistent BPU path: a single hash-bound native worker/model load must
# reproduce all 43 canonical HRT outputs and then exit without a residual PID.
PYTHONPATH="${RELEASE_ROOT}/rootscope:${RELEASE_ROOT}" \
  "${ROOTSCOPE_CPU_PYTHON}" \
  "${RELEASE_ROOT}/rootscope/tools/x5_native_libdnn_qualify_v3.py" \
  --model "${MODEL}" \
  --model-sha256 "${MODEL_SHA}" \
  --worker "${RELEASE_ROOT}/bin/rootscope-native-libdnn-worker" \
  --compile-contract \
    "${RELEASE_ROOT}/rootscope/app/runtime_v3/native/compile_contract_x5.v1.json" \
  --input-root "${INPUT_ROOT}" \
  --oracle-manifest "${ORACLE}" \
  --output "${EVIDENCE}/05_native_libdnn.json" \
  >"${EVIDENCE}/05_native_libdnn.stdout.json"

# The full release verifier intentionally hashes every payload, including both
# GGUFs.  On a 4 GB X5 those clean pages can temporarily occupy CMA-reclaimable
# memory before either RootMind role starts.  Precondition only the two
# manifest-bound GGUFs, largest first, then retain the per-role post-inference
# release gates below.  No global cache knob, service, device, or action
# authority is used.
ROOTMIND_CACHE_HELPER="${RELEASE_ROOT}/rootscope/tools/x5_rootmind_cache_release_v3.py"
ROOTMIND_DEEP_MODEL="${RELEASE_ROOT}/models/llm/deep/rootscope-qwen3-1.7b-rootscope-v3-final.Q4_K_M.gguf"
ROOTMIND_FAST_MODEL="${RELEASE_ROOT}/models/llm/fast/qwen2_05b_distill.Q4_K_M.gguf"
for role in deep fast; do
  if [[ "${role}" == "deep" ]]; then
    role_model="${ROOTMIND_DEEP_MODEL}"
  else
    role_model="${ROOTMIND_FAST_MODEL}"
  fi
  binding="${EVIDENCE}/rootmind_precondition_${role}_model_binding.json"
  release="${EVIDENCE}/rootmind_precondition_${role}_model_page_cache_release.json"
  "${ROOTSCOPE_CPU_PYTHON}" -I "${ROOTMIND_CACHE_HELPER}" bind \
    --release-root "${RELEASE_ROOT}" \
    --role "${role}" \
    --model "${role_model}" \
    --output "${binding}" \
    >"${EVIDENCE}/rootmind_precondition_${role}_bind.stdout.json"
  "${ROOTSCOPE_CPU_PYTHON}" -I "${ROOTMIND_CACHE_HELPER}" release \
    --release-root "${RELEASE_ROOT}" \
    --role "${role}" \
    --binding "${binding}" \
    --output "${release}" \
    --observe-seconds 2 \
    >"${EVIDENCE}/rootmind_precondition_${role}_release.stdout.json"
done

bash "${RELEASE_ROOT}/tools/release_v3/x5_rootmind_smoke_v3.sh" \
  "${RELEASE_ROOT}" fast "${EVIDENCE}" "${ROOTSCOPE_CPU_PYTHON}" \
  >"${EVIDENCE}/06_rootmind_fast_receipt.json"
bash "${RELEASE_ROOT}/tools/release_v3/x5_rootmind_smoke_v3.sh" \
  "${RELEASE_ROOT}" deep "${EVIDENCE}" "${ROOTSCOPE_CPU_PYTHON}" \
  >"${EVIDENCE}/07_rootmind_deep_receipt.json"

"${ROOTSCOPE_CPU_PYTHON}" -I - \
  "${EVIDENCE}/08_acceptance_summary.json" \
  "${EVIDENCE}" "${RELEASE_ROOT}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import stat
import sys

output = Path(sys.argv[1])
evidence = Path(sys.argv[2]).resolve(strict=True)
release_root = Path(sys.argv[3]).resolve(strict=True)


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant forbidden: {value}")


def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key forbidden: {key}")
        result[key] = value
    return result


def load(name):
    path = evidence / name
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=reject_constant,
    )


def exact(value, expected, label):
    if value != expected or type(value) is not type(expected):
        raise SystemExit(f"{label} mismatch: actual={value!r} expected={expected!r}")


def exact_int(value, expected, label):
    if type(value) is not int or value != expected:
        raise SystemExit(f"{label} mismatch: actual={value!r} expected={expected}")


def exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise SystemExit(f"{label} keys changed")
    return value


def nonnegative_int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise SystemExit(f"{label} must be an integer >= {minimum}")
    return value


def finite_at_least(value, minimum, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise SystemExit(f"{label} must be finite and >= {minimum}")


def false_authority(value, label):
    if not isinstance(value, dict):
        raise SystemExit(f"{label} authority is missing")
    for key in (
        "execution_authority",
        "serial_write",
        "gpio_write",
        "pump_command",
        "state_machine_write",
        "physical_completion",
    ):
        exact(value.get(key), False, f"{label}.authority.{key}")
    if "external_network" in value:
        exact(
            value.get("external_network"),
            False,
            f"{label}.authority.external_network",
        )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


bootstrap = load("00_runtime_bootstrap.json")
verify = load("01_release_verify.json")
cpu = load("02_cpu_bm25.json")
hrt = load("03_hrt_oracle.json")
hbm_execution = load("04_hbm_execution.json")
native = load("05_native_libdnn.json")
rootmind_fast = load("06_rootmind_fast_receipt.json")
rootmind_deep = load("07_rootmind_deep_receipt.json")

exact(
    bootstrap.get("schema"),
    "rootscope.v3.x5-runtime-bootstrap.v1",
    "bootstrap.schema",
)
exact(
    bootstrap.get("status"),
    "PASS_OFFLINE_CPU_AND_VENDOR_BPU_INTERPRETERS",
    "bootstrap.status",
)
for key in (
    "network_touched",
    "camera_opened",
    "serial_opened",
    "gpio_touched",
    "pump_touched",
    "service_started",
    "physical_completion",
):
    exact(bootstrap.get(key), False, f"bootstrap.{key}")

exact(
    verify.get("schema"),
    "rootscope.v3.release-verification-receipt.v1",
    "verify.schema",
)
exact(
    verify.get("status"),
    "PASS_X5_STAGED_ZERO_AUTHORITY_LIVE_QUALIFICATION_PENDING",
    "verify.status",
)
exact(verify.get("candidate_id"), release_root.name, "verify.candidate_id")
for key in ("hardware_opened", "camera_opened", "serial_opened", "pump_touched"):
    exact(verify.get(key), False, f"verify.{key}")
false_authority(verify.get("authority"), "verify")

exact(
    cpu.get("schema"),
    "rootscope.v3.x5-cpu-bm25-acceptance.v1",
    "cpu_bm25.schema",
)
exact(
    cpu.get("status"),
    "PASS_X5_CPU_ONNX_AND_BM25_READ_ONLY",
    "cpu_bm25.status",
)
exact(
    cpu.get("cpu", {}).get("status"),
    "PASS_CPU_ONNX_SIMULATED_INPUT_NOT_ACCURACY_EVIDENCE",
    "cpu_bm25.cpu.status",
)
exact(cpu.get("rag", {}).get("sqlite_integrity"), "ok", "cpu_bm25.rag.integrity")
exact(
    cpu.get("rag", {}).get("database_open_mode"),
    "URI_MODE_RO_IMMUTABLE_1",
    "cpu_bm25.rag.open_mode",
)
exact_int(len(cpu.get("rag", {}).get("queries", [])), 3, "cpu_bm25.rag.query_count")
for key in (
    "camera_opened",
    "serial_opened",
    "gpio_touched",
    "pump_touched",
    "physical_completion",
):
    exact(cpu.get(key), False, f"cpu_bm25.{key}")

exact(
    hrt.get("schema"),
    "rootscope.v3.x5-hrt-oracle-qualification.v1",
    "hrt.schema",
)
exact(hrt.get("status"), "PASS_X5_CANONICAL_HRT_BPU_ORACLE", "hrt.status")
if not str(hrt.get("backend_actual", "")).startswith(
    "drobotics.hrt_model_exec@"
) or not str(hrt.get("backend_actual", "")).endswith("/cold-load"):
    raise SystemExit("canonical HRT backend identity mismatch")
exact(hrt.get("model_sha256"), "4dc4bc24741332bb6bc6da184dd1d6f954ae539ba740330490339d76eb200285", "hrt.model_sha256")
exact(hrt.get("canonical_numerical_oracle"), True, "hrt.canonical_oracle")
exact(hrt.get("persistent_model"), False, "hrt.persistent_model")
exact(hrt.get("cold_load_per_inference"), True, "hrt.cold_load")
exact_int(hrt.get("count"), 43, "hrt.count")
exact_int(hrt.get("top1_agreement"), 43, "hrt.top1_agreement")
finite_at_least(hrt.get("mean_cosine"), 0.995, "hrt.mean_cosine")
exact(
    hrt.get("eligible_as_fail_closed_bpu_shadow_fallback"),
    True,
    "hrt.fallback_eligibility",
)
exact(hrt.get("selected_for_primary_runtime"), False, "hrt.primary_selection")
false_authority(hrt.get("authority"), "hrt")

exact(
    hbm_execution.get("schema"),
    "rootscope.v3.legacy-python-hbm-observation.v1",
    "legacy_hbm.execution.schema",
)
exact(hbm_execution.get("authoritative"), False, "legacy_hbm.authoritative")
exact(
    hbm_execution.get("affects_acceptance"),
    False,
    "legacy_hbm.affects_acceptance",
)
exact_int(
    hbm_execution.get("timeout_seconds_each"),
    120,
    "legacy_hbm.timeout_seconds_each",
)
hbm_observations = hbm_execution.get("observations")
if not isinstance(hbm_observations, dict) or set(hbm_observations) != {
    "RAW_UINT8",
    "RGB128_CENTERED_INT8",
}:
    raise SystemExit("legacy_hbm observation coverage changed")
for policy, observation in hbm_observations.items():
    if not isinstance(observation, dict):
        raise SystemExit(f"legacy_hbm.{policy} observation is malformed")
    exact(observation.get("policy"), policy, f"legacy_hbm.{policy}.policy")
    if type(observation.get("exit_code")) is not int:
        raise SystemExit(f"legacy_hbm.{policy}.exit_code must be an integer")
    if type(observation.get("receipt_present")) is not bool:
        raise SystemExit(f"legacy_hbm.{policy}.receipt_present must be boolean")

exact(
    native.get("schema"),
    "rootscope.v3.x5-native-libdnn-qualification.v1",
    "native.schema",
)
exact(
    native.get("status"),
    "PASS_X5_PERSISTENT_NATIVE_LIBDNN",
    "native.status",
)
exact(
    native.get("backend_actual"),
    "rootscope.native_libdnn_valid_shape_bridge/libdnn.so",
    "native.backend",
)
exact(native.get("persistent_model"), True, "native.persistent_model")
exact(native.get("cold_load_per_inference"), False, "native.cold_load")
exact_int(native.get("model_load_count"), 1, "native.model_load_count")
exact_int(native.get("count"), 43, "native.count")
exact_int(native.get("top1_agreement"), 43, "native.top1_agreement")
finite_at_least(native.get("mean_cosine"), 0.995, "native.mean_cosine")
exact(
    native.get("numerical_oracle_gate_passed"),
    True,
    "native.numerical_oracle_gate",
)
exact(
    native.get("eligible_for_zero_authority_shadow_runtime"),
    True,
    "native.shadow_eligibility",
)
exact(native.get("selected_for_runtime"), False, "native.runtime_selection")
exact(
    native.get("selection_effect"),
    "REPORT_ONLY_NO_RUNTIME_CONFIG_MUTATION",
    "native.selection_effect",
)
lifecycle = native.get("worker_lifecycle", {})
exact(lifecycle.get("exit_code_after_stdin_eof"), 0, "native.worker.exit_code")
exact(
    lifecycle.get("clean_exit_no_residual_process"),
    True,
    "native.worker.clean_exit",
)
exact(
    lifecycle.get("after_close", {}).get("proc_present"),
    False,
    "native.worker.proc_after_close",
)
false_authority(native.get("authority"), "native")


def validate_rootmind_cache_release(
    cache_release,
    role,
    runtime,
    *,
    evidence_prefix=None,
):
    prefix = evidence_prefix or f"rootmind_{role}_model"
    binding_name = f"{prefix}_binding.json"
    release_name = f"{prefix}_page_cache_release.json"
    binding_path = evidence / binding_name
    external_release = load(release_name)
    exact(
        cache_release,
        external_release,
        f"{role}.cache_release.external_receipt",
    )

    binding = load(binding_name)
    exact_keys(
        binding,
        {
            "schema",
            "status",
            "created_utc",
            "candidate",
            "role",
            "model",
            "integrity",
            "authority",
        },
        f"{role}.model_binding",
    )
    exact(
        binding.get("schema"),
        "rootscope.v3.rootmind-gguf-cache-binding.v1",
        f"{role}.model_binding.schema",
    )
    exact(binding.get("status"), "BOUND", f"{role}.model_binding.status")
    exact(binding.get("role"), role, f"{role}.model_binding.role")
    if not isinstance(binding.get("created_utc"), str) or not binding["created_utc"]:
        raise SystemExit(f"{role}.model_binding.created_utc is missing")
    binding_candidate = exact_keys(
        binding.get("candidate"),
        {"id", "release_root", "manifest_path", "manifest_sha256"},
        f"{role}.model_binding.candidate",
    )
    manifest_path = release_root / "candidate_manifest.json"
    exact(
        binding_candidate.get("id"),
        release_root.name,
        f"{role}.model_binding.candidate.id",
    )
    exact(
        binding_candidate.get("release_root"),
        str(release_root),
        f"{role}.model_binding.candidate.release_root",
    )
    exact(
        binding_candidate.get("manifest_path"),
        str(manifest_path),
        f"{role}.model_binding.candidate.manifest_path",
    )
    exact(
        binding_candidate.get("manifest_sha256"),
        sha256_file(manifest_path),
        f"{role}.model_binding.candidate.manifest_sha256",
    )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=reject_constant,
    )
    exact(
        manifest.get("schema"),
        "rootscope.v3.candidate-manifest.v1",
        f"{role}.model_binding.candidate_manifest.schema",
    )
    exact(
        manifest.get("candidate_id"),
        release_root.name,
        f"{role}.model_binding.candidate_manifest.candidate_id",
    )
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list) or not manifest_files:
        raise SystemExit(f"{role}.model_binding candidate manifest is empty")
    manifest_paths = [
        row.get("path")
        for row in manifest_files
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    ]
    if (
        len(manifest_paths) != len(manifest_files)
        or len(set(manifest_paths)) != len(manifest_paths)
    ):
        raise SystemExit(
            f"{role}.model_binding candidate manifest paths are malformed"
        )
    binding_model = exact_keys(
        binding.get("model"),
        {
            "path",
            "relative_path",
            "category",
            "bytes",
            "sha256",
            "stat_fingerprint",
        },
        f"{role}.model_binding.model",
    )
    expected_category = (
        "ROOTMIND_FAST_MODEL" if role == "fast" else "ROOTMIND_DEEP_MODEL"
    )
    model_relative = binding_model.get("relative_path")
    expected_prefix = f"models/llm/{role}/"
    if (
        not isinstance(model_relative, str)
        or not model_relative.startswith(expected_prefix)
        or "/" in model_relative[len(expected_prefix):]
        or not model_relative.endswith(".gguf")
    ):
        raise SystemExit(f"{role}.model_binding.model relative path is invalid")
    exact(
        binding_model.get("path"),
        str(release_root / model_relative),
        f"{role}.model_binding.model.release_path",
    )
    exact(
        binding_model.get("path"),
        runtime.get("model_path"),
        f"{role}.model_binding.model.path",
    )
    exact(
        binding_model.get("relative_path"),
        runtime.get("model_relative_path"),
        f"{role}.model_binding.model.relative_path",
    )
    exact(
        binding_model.get("category"),
        expected_category,
        f"{role}.model_binding.model.category",
    )
    exact(
        binding_model.get("bytes"),
        runtime.get("model_bytes"),
        f"{role}.model_binding.model.bytes",
    )
    exact(
        binding_model.get("sha256"),
        runtime.get("model_sha256"),
        f"{role}.model_binding.model.sha256",
    )
    model_sha = binding_model.get("sha256")
    if (
        not isinstance(model_sha, str)
        or len(model_sha) != 64
        or any(character not in "0123456789abcdef" for character in model_sha)
    ):
        raise SystemExit(f"{role}.model_binding.model SHA-256 is invalid")
    nonnegative_int(
        binding_model.get("bytes"),
        f"{role}.model_binding.model.bytes",
        1,
    )
    manifest_model_rows = [
        row for row in manifest_files if row.get("path") == model_relative
    ]
    if len(manifest_model_rows) != 1:
        raise SystemExit(f"{role}.model_binding manifest model is not unique")
    manifest_model = manifest_model_rows[0]
    exact(
        manifest_model.get("category"),
        expected_category,
        f"{role}.model_binding.manifest_model.category",
    )
    exact(
        manifest_model.get("bytes"),
        binding_model.get("bytes"),
        f"{role}.model_binding.manifest_model.bytes",
    )
    exact(
        manifest_model.get("sha256"),
        binding_model.get("sha256"),
        f"{role}.model_binding.manifest_model.sha256",
    )
    model_stat = exact_keys(
        binding_model.get("stat_fingerprint"),
        {
            "device",
            "inode",
            "mode",
            "nlink",
            "uid",
            "gid",
            "size",
            "mtime_ns",
            "ctime_ns",
        },
        f"{role}.model_binding.model.stat_fingerprint",
    )
    for key in (
        "device",
        "inode",
        "mode",
        "nlink",
        "uid",
        "gid",
        "size",
        "mtime_ns",
        "ctime_ns",
    ):
        nonnegative_int(
            model_stat.get(key),
            f"{role}.model_binding.model.stat_fingerprint.{key}",
        )
    exact(model_stat.get("nlink"), 1, f"{role}.model_binding.model.nlink")
    if not stat.S_ISREG(model_stat["mode"]):
        raise SystemExit(f"{role}.model_binding.model is not a regular file")
    exact(
        model_stat.get("size"),
        binding_model.get("bytes"),
        f"{role}.model_binding.model.size",
    )
    binding_integrity = exact_keys(
        binding.get("integrity"),
        {
            "manifest_record_count",
            "unique_role_gguf",
            "content_sha256_verified",
            "regular_file",
            "nofollow_open",
        },
        f"{role}.model_binding.integrity",
    )
    exact(
        binding_integrity.get("manifest_record_count"),
        len(manifest_files),
        f"{role}.model_binding.integrity.manifest_record_count",
    )
    for key in (
        "unique_role_gguf",
        "content_sha256_verified",
        "regular_file",
        "nofollow_open",
    ):
        exact(
            binding_integrity.get(key),
            True,
            f"{role}.model_binding.integrity.{key}",
        )

    authority_keys = {
        "execution_authority",
        "physical_authority",
        "external_network",
        "service_started",
        "serial_opened",
        "serial_write",
        "gpio_touched",
        "pump_command",
        "state_machine_write",
        "model_modified",
    }
    binding_authority = exact_keys(
        binding.get("authority"),
        authority_keys,
        f"{role}.model_binding.authority",
    )
    for key in authority_keys:
        exact(
            binding_authority.get(key),
            False,
            f"{role}.model_binding.authority.{key}",
        )

    exact_keys(
        cache_release,
        {
            "schema",
            "status",
            "created_utc",
            "binding_sha256",
            "candidate",
            "role",
            "model",
            "integrity",
            "preconditions",
            "cache",
            "memory",
            "authority",
            "error",
        },
        f"{role}.cache_release",
    )
    exact(
        cache_release.get("schema"),
        "rootscope.v3.rootmind-gguf-cache-release.v1",
        f"{role}.cache_release.schema",
    )
    exact(cache_release.get("status"), "PASS", f"{role}.cache_release.status")
    exact(cache_release.get("role"), role, f"{role}.cache_release.role")
    if (
        not isinstance(cache_release.get("created_utc"), str)
        or not cache_release["created_utc"]
    ):
        raise SystemExit(f"{role}.cache_release.created_utc is missing")
    exact(
        cache_release.get("binding_sha256"),
        sha256_file(binding_path),
        f"{role}.cache_release.binding_sha256",
    )
    exact(
        cache_release.get("candidate"),
        binding_candidate,
        f"{role}.cache_release.candidate",
    )
    exact(
        cache_release.get("model"),
        binding_model,
        f"{role}.cache_release.model",
    )
    exact(cache_release.get("error"), None, f"{role}.cache_release.error")

    release_integrity = exact_keys(
        cache_release.get("integrity"),
        {
            "binding_valid",
            "release_root_unchanged",
            "manifest_path_unchanged",
            "manifest_sha256_unchanged",
            "manifest_record_unchanged",
            "model_path_unchanged",
            "model_stat_unchanged",
            "model_sha256_verified",
            "model_stat_unchanged_after",
            "model_modified",
        },
        f"{role}.cache_release.integrity",
    )
    for key in (
        "binding_valid",
        "release_root_unchanged",
        "manifest_path_unchanged",
        "manifest_sha256_unchanged",
        "manifest_record_unchanged",
        "model_path_unchanged",
        "model_stat_unchanged",
        "model_sha256_verified",
        "model_stat_unchanged_after",
    ):
        exact(
            release_integrity.get(key),
            True,
            f"{role}.cache_release.integrity.{key}",
        )
    exact(
        release_integrity.get("model_modified"),
        False,
        f"{role}.cache_release.integrity.model_modified",
    )

    preconditions = exact_keys(
        cache_release.get("preconditions"),
        {"llama_server_processes", "no_llama_server", "endpoint", "port_closed"},
        f"{role}.cache_release.preconditions",
    )
    exact(
        preconditions.get("llama_server_processes"),
        [],
        f"{role}.cache_release.preconditions.llama_server_processes",
    )
    exact(
        preconditions.get("no_llama_server"),
        True,
        f"{role}.cache_release.preconditions.no_llama_server",
    )
    exact(
        preconditions.get("endpoint"),
        "127.0.0.1:9080",
        f"{role}.cache_release.preconditions.endpoint",
    )
    exact(
        preconditions.get("port_closed"),
        True,
        f"{role}.cache_release.preconditions.port_closed",
    )

    cache = exact_keys(
        cache_release.get("cache"),
        {
            "method",
            "fadvise_applied",
            "exact_file_only",
            "global_drop_caches",
            "sync_called",
            "compact_memory_called",
            "resident_bytes_before",
            "resident_bytes_after",
            "resident_limit_bytes",
            "window_reached",
        },
        f"{role}.cache_release.cache",
    )
    exact(
        cache.get("method"),
        "POSIX_FADV_DONTNEED",
        f"{role}.cache_release.cache.method",
    )
    exact(
        cache.get("fadvise_applied"),
        True,
        f"{role}.cache_release.cache.fadvise_applied",
    )
    exact(
        cache.get("exact_file_only"),
        True,
        f"{role}.cache_release.cache.exact_file_only",
    )
    for key in ("global_drop_caches", "sync_called", "compact_memory_called"):
        exact(
            cache.get(key),
            False,
            f"{role}.cache_release.cache.{key}",
        )
    resident_before = nonnegative_int(
        cache.get("resident_bytes_before"),
        f"{role}.cache_release.cache.resident_bytes_before",
    )
    resident_after = nonnegative_int(
        cache.get("resident_bytes_after"),
        f"{role}.cache_release.cache.resident_bytes_after",
    )
    resident_limit = nonnegative_int(
        cache.get("resident_limit_bytes"),
        f"{role}.cache_release.cache.resident_limit_bytes",
    )
    exact(resident_limit, 4096, f"{role}.cache_release.cache.resident_limit")
    if resident_after > resident_limit:
        raise SystemExit(f"{role}.cache_release residency gate failed")
    exact(
        cache.get("window_reached"),
        True,
        f"{role}.cache_release.cache.window_reached",
    )

    memory = exact_keys(
        cache_release.get("memory"),
        {
            "before",
            "after",
            "samples",
            "observe_seconds",
            "cma_free_minimum_kib",
            "window_reached",
        },
        f"{role}.cache_release.memory",
    )
    observe_seconds = memory.get("observe_seconds")
    if (
        isinstance(observe_seconds, bool)
        or not isinstance(observe_seconds, (int, float))
        or not math.isfinite(float(observe_seconds))
        or float(observe_seconds) != 2.0
    ):
        raise SystemExit(
            f"{role}.cache_release.memory.observe_seconds must be exactly 2"
        )
    exact(
        memory.get("cma_free_minimum_kib"),
        131072,
        f"{role}.cache_release.memory.cma_free_minimum_kib",
    )
    exact(
        memory.get("window_reached"),
        True,
        f"{role}.cache_release.memory.window_reached",
    )
    for label in ("before", "after"):
        snapshot = exact_keys(
            memory.get(label),
            {"mem_available_kib", "cma_free_kib", "cached_kib"},
            f"{role}.cache_release.memory.{label}",
        )
        for key in ("mem_available_kib", "cma_free_kib", "cached_kib"):
            nonnegative_int(
                snapshot.get(key),
                f"{role}.cache_release.memory.{label}.{key}",
            )
    if memory["after"]["cma_free_kib"] < 131072:
        raise SystemExit(f"{role}.cache_release final CMA recovery gate failed")
    samples = memory.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise SystemExit(f"{role}.cache_release memory samples are incomplete")
    for index, sample in enumerate(samples):
        exact_keys(
            sample,
            {
                "elapsed_ms",
                "mem_available_kib",
                "cma_free_kib",
                "cached_kib",
                "resident_bytes",
                "gate_pass",
            },
            f"{role}.cache_release.memory.samples[{index}]",
        )
        for key in (
            "elapsed_ms",
            "mem_available_kib",
            "cma_free_kib",
            "cached_kib",
            "resident_bytes",
        ):
            nonnegative_int(
                sample.get(key),
                f"{role}.cache_release.memory.samples[{index}].{key}",
            )
        exact(
            sample.get("gate_pass"),
            True,
            f"{role}.cache_release.memory.samples[{index}].gate_pass",
        )
        if (
            sample["cma_free_kib"] < 131072
            or sample["resident_bytes"] > resident_limit
        ):
            raise SystemExit(f"{role}.cache_release sampled gate failed")
    exact(
        samples[-1]["resident_bytes"],
        resident_after,
        f"{role}.cache_release final sampled residency",
    )

    release_authority = exact_keys(
        cache_release.get("authority"),
        authority_keys,
        f"{role}.cache_release.authority",
    )
    for key in authority_keys:
        exact(
            release_authority.get(key),
            False,
            f"{role}.cache_release.authority.{key}",
        )
    return {
        "status": cache_release["status"],
        "schema": cache_release["schema"],
        "binding_sha256": cache_release["binding_sha256"],
        "receipt_sha256": sha256_file(evidence / release_name),
        "method": cache["method"],
        "exact_file_only": cache["exact_file_only"],
        "global_drop_caches": cache["global_drop_caches"],
        "resident_bytes_before": resident_before,
        "resident_bytes_after": resident_after,
        "resident_limit_bytes": resident_limit,
        "cma_free_after_kib": memory["after"]["cma_free_kib"],
        "cma_free_minimum_kib": memory["cma_free_minimum_kib"],
        "observe_seconds": observe_seconds,
    }


def validate_rootmind(receipt, role):
    exact(receipt.get("schema"), "rootscope.v3.x5-rootmind-smoke.v3", f"{role}.schema")
    exact(receipt.get("role"), role, f"{role}.role")
    exact(
        receipt.get("candidate", {}).get("candidate_id"),
        release_root.name,
        f"{role}.candidate_id",
    )
    status = receipt.get("status")
    allowed = {
        "PASS_X5_ROOTMIND_CHAT_TEMPLATE_SCHEMA_LOCKED_READ_ONLY",
        (
            "PASS_X5_ROOTMIND_CHAT_TEMPLATE_EXPLICIT_GBNF_EXACT_READ_ONLY_"
            "SCHEMA_RUNTIME_INCOMPATIBLE"
        ),
    }
    if status not in allowed:
        raise SystemExit(f"{role}.status is not an accepted strict mode: {status!r}")
    contract = receipt.get("contract", {})
    transport = receipt.get("transport", {})
    exact(transport.get("loopback_only"), True, f"{role}.loopback_only")
    exact(
        transport.get("external_network_touched"),
        False,
        f"{role}.external_network_touched",
    )
    exact(
        contract.get("exact_output"),
        {"authority": False, "status": "READ_ONLY"},
        f"{role}.exact_output",
    )
    exact(contract.get("tool_interface_supplied"), False, f"{role}.tools_supplied")
    exact(contract.get("tool_calls_observed"), False, f"{role}.tool_calls")
    fallback = contract.get("single_explicit_gbnf_retry_used")
    if status == "PASS_X5_ROOTMIND_CHAT_TEMPLATE_SCHEMA_LOCKED_READ_ONLY":
        exact(fallback, False, f"{role}.fallback")
        exact(contract.get("schema_primary_passed"), True, f"{role}.schema_primary")
        exact(contract.get("json_schema_strict"), True, f"{role}.json_schema")
        exact(contract.get("explicit_gbnf_strict"), False, f"{role}.gbnf")
        enforcement = "JSON_SCHEMA_STRICT_AND_EXACT_POST_PARSE"
    else:
        exact(role, "deep", f"{role}.fallback_role")
        exact(fallback, True, f"{role}.fallback")
        exact(contract.get("schema_primary_passed"), False, f"{role}.schema_primary")
        exact(contract.get("json_schema_strict"), False, f"{role}.json_schema")
        exact(contract.get("explicit_gbnf_strict"), True, f"{role}.gbnf")
        exact(
            contract.get("compatibility_downgrade_reason"),
            "B9637_QWEN3_ASSISTANT_PREFIX_JSON_SCHEMA_GRAMMAR_SAMPLER_INIT",
            f"{role}.compatibility_reason",
        )
        enforcement = (
            "B9637_SCHEMA_RUNTIME_INCOMPATIBLE_EXACT_ONE_GBNF_RETRY_"
            "AND_EXACT_POST_PARSE"
        )
    shutdown = receipt.get("shutdown", {})
    exact(shutdown.get("process_stopped"), True, f"{role}.process_stopped")
    exact(shutdown.get("port_closed_after_stop"), True, f"{role}.port_closed")
    if type(shutdown.get("forced_kill")) is not bool:
        raise SystemExit(f"{role}.forced_kill must be a boolean")
    cache_summary = validate_rootmind_cache_release(
        shutdown.get("model_page_cache_release"),
        role,
        receipt.get("runtime", {}),
    )
    exact(receipt.get("execution_authority"), False, f"{role}.execution_authority")
    exact(receipt.get("physical_authority"), False, f"{role}.physical_authority")
    for key in (
        "service_started",
        "serial_opened",
        "serial_write",
        "gpio_touched",
        "pump_command",
        "state_machine_write",
        "physical_completion",
    ):
        exact(receipt.get(key), False, f"{role}.{key}")
    return {
        "status": status,
        "enforcement": enforcement,
        "schema_primary_passed": contract["schema_primary_passed"],
        "explicit_gbnf_retry_used": fallback,
        "exact_output": contract["exact_output"],
        "model_sha256": receipt["runtime"]["model_sha256"],
        "server_sha256": receipt["runtime"]["server_sha256"],
        "forced_kill": shutdown["forced_kill"],
        "model_page_cache_release": cache_summary,
    }


fast_summary = validate_rootmind(rootmind_fast, "fast")
deep_summary = validate_rootmind(rootmind_deep, "deep")

cache_precondition = {}
for precondition_role in ("deep", "fast"):
    precondition_prefix = f"rootmind_precondition_{precondition_role}_model"
    precondition_binding = load(f"{precondition_prefix}_binding.json")
    precondition_release = load(
        f"{precondition_prefix}_page_cache_release.json"
    )
    precondition_model = precondition_binding.get("model", {})
    cache_precondition[precondition_role] = validate_rootmind_cache_release(
        precondition_release,
        precondition_role,
        {
            "model_path": precondition_model.get("path"),
            "model_relative_path": precondition_model.get("relative_path"),
            "model_bytes": precondition_model.get("bytes"),
            "model_sha256": precondition_model.get("sha256"),
        },
        evidence_prefix=precondition_prefix,
    )

receipt_names = (
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
)
receipt_hashes = {
    name: sha256_file(evidence / name)
    for name in receipt_names
}
value = {
    "schema": "rootscope.v3.x5-software-acceptance.v2",
    "status": (
        "PASS_X5_OFFLINE_ZERO_AUTHORITY_SOFTWARE_NATIVE_LIBDNN_"
        "LIVE_RESOURCE_STM32_PHYSICAL_PENDING"
    ),
    "candidate_id": release_root.name,
    "release_root": str(release_root),
    "evidence_root": str(evidence),
    "receipts_sha256": receipt_hashes,
    "runtime_bootstrap": {
        "status": bootstrap["status"],
        "passed": (
            bootstrap["status"]
            == "PASS_OFFLINE_CPU_AND_VENDOR_BPU_INTERPRETERS"
        ),
    },
    "release_verification": {
        "status": verify["status"],
        "passed": (
            verify["status"]
            == "PASS_X5_STAGED_ZERO_AUTHORITY_LIVE_QUALIFICATION_PENDING"
        ),
    },
    "cpu_onnx_bm25": {
        "status": cpu["status"],
        "passed": cpu["status"] == "PASS_X5_CPU_ONNX_AND_BM25_READ_ONLY",
        "bm25_open_mode": cpu["rag"]["database_open_mode"],
        "frozen_query_count": len(cpu["rag"]["queries"]),
    },
    "bpu": {
        "canonical_hrt_oracle": {
            "status": hrt["status"],
            "backend_actual": hrt["backend_actual"],
            "count": hrt["count"],
            "top1_agreement": hrt["top1_agreement"],
            "mean_cosine": hrt["mean_cosine"],
            "cold_load_per_inference": hrt["cold_load_per_inference"],
            "passed": (
                hrt["status"] == "PASS_X5_CANONICAL_HRT_BPU_ORACLE"
                and hrt["top1_agreement"] == hrt["count"] == 43
            ),
        },
        "qualification_persistent_native_libdnn": {
            "status": native["status"],
            "backend_actual": native["backend_actual"],
            "count": native["count"],
            "top1_agreement": native["top1_agreement"],
            "mean_cosine": native["mean_cosine"],
            "model_load_count": native["model_load_count"],
            "worker_pid": native["single_worker_pid"],
            "clean_worker_exit": lifecycle["clean_exit_no_residual_process"],
            "selected_for_runtime": native["selected_for_runtime"],
            "isolated_post_worker_cma_observation": native[
                "isolated_post_worker_cma_observation"
            ],
            "passed": (
                native["status"] == "PASS_X5_PERSISTENT_NATIVE_LIBDNN"
                and native["top1_agreement"] == native["count"] == 43
                and native["model_load_count"] == 1
                and lifecycle["clean_exit_no_residual_process"] is True
            ),
        },
        "legacy_python_hbm_runtime_non_authoritative": {
            "authoritative": hbm_execution["authoritative"],
            "affects_acceptance": hbm_execution["affects_acceptance"],
            "timeout_seconds_each": hbm_execution[
                "timeout_seconds_each"
            ],
            "observations": hbm_observations,
        },
    },
    "rootmind": {
        "fast": fast_summary,
        "deep": deep_summary,
    },
    "rootmind_cache_precondition": cache_precondition,
    "acceptance_scope": (
        "STAGED_CANDIDATE_OFFLINE_ZERO_AUTHORITY_SOFTWARE_AND_STATIC43_BPU"
    ),
    "pending": {
        "live_camera": "PENDING_NOT_RUN_BY_ACCEPTANCE",
        "resource_soak": "PENDING_NOT_RUN_BY_ACCEPTANCE",
        "thermal_cma_full_stack": "PENDING_SEPARATE_QUALIFICATION",
        "stm32": "PENDING_HARDWARE_NOT_TOUCHED",
        "physical_loop": "PENDING_HARDWARE_NOT_TOUCHED",
    },
    "mutation_boundary": {
        "current_selected_or_modified": False,
        "service_started": False,
        "camera_opened": False,
        "serial_opened": False,
        "serial_write": False,
        "gpio_touched": False,
        "pump_touched": False,
        "physical_completion": False,
    },
}
with output.open("x", encoding="utf-8") as handle:
    json.dump(
        value,
        handle,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    handle.write("\n")
print(json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False))
PY
