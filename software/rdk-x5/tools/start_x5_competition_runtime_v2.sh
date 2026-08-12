#!/usr/bin/env bash
# Foreground-only RootScope competition runtime for one 4 GB RDK X5.
set -euo pipefail

EXPECTED_HOSTNAME="rootscope-x5"
EXPECTED_MACHINE_ID="00000000000000000000000000000001"
EXPECTED_SERIAL="3281556110220e0c002bdeab0012004"
EXPECTED_WLAN_MAC="02:00:00:00:00:01"

EXPECTED_R7_SHA256="4dc4bc24741332bb6bc6da184dd1d6f954ae539ba740330490339d76eb200285"
EXPECTED_LLAMA_SERVER_SHA256="dcb636215243b8911488b8ca96f0c39bedee14e92f44f7d0ef6c599419acf9b9"
EXPECTED_QWEN_SHA256="6c8adc95de81b1d0103fcf900a5f369d5936d234e42ed64881dad56e0104e77b"
EXPECTED_HRT_MODEL_EXEC_SHA256="c3a47c77889bc82c8519b68a86b75f8205c6a4f9695339bb3d01da2713abcb04"
EXPECTED_LIBDNN_SHA256="661bac161124921eb9065fb9cd8d311144ea3f899fee28d61dfd0d7255074ace"
EXPECTED_LIBHBRT_SHA256="8b4719d147a53a4adb215f0307a732a420b4dc2ffaa893f4a8fcd02a7e88fc9a"

RELEASES_PARENT="${HOME}/.local/share/rootscope-competition-runtime/releases"
RUNS_PARENT="${HOME}/.local/share/rootscope-competition-runtime/runs"
DEFAULT_RELEASE_ROOT="${RELEASES_PARENT}/rootscope_competition_runtime_v2_candidate_20260723"
RELEASE_ROOT="${ROOTSCOPE_COMPETITION_RELEASE_ROOT:-${DEFAULT_RELEASE_ROOT}}"
RUN_ROOT=""
LIVE_MODE=0

usage() {
  cat <<'EOF'
usage: start_x5_competition_runtime_v2.sh [options]

Options:
  --release-root ABSOLUTE_PATH  Candidate release (defaults to the frozen v2 candidate).
  --run-root ABSOLUTE_PATH      New run directory below the competition runs parent.
  --live                        Run Competition Live v2 after one three-role LLM call.
  --no-live                     Run BPU/LLM readiness smoke, then cleanly exit.
  -h, --help                    Show this help.

This launcher never registers a service, never enables boot persistence, and
never opens a camera unless --live is explicitly supplied.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --release-root)
      [[ "$#" -ge 2 ]] || { echo "--release-root needs a value" >&2; exit 64; }
      RELEASE_ROOT="$2"
      shift 2
      ;;
    --run-root)
      [[ "$#" -ge 2 ]] || { echo "--run-root needs a value" >&2; exit 64; }
      RUN_ROOT="$2"
      shift 2
      ;;
    --live)
      LIVE_MODE=1
      shift
      ;;
    --no-live)
      LIVE_MODE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

[[ "$(id -un)" == "rootscope" ]] || { echo "runtime user must be rootscope" >&2; exit 65; }
[[ "$(hostname)" == "${EXPECTED_HOSTNAME}" ]] || { echo "hostname mismatch" >&2; exit 65; }
[[ "$(cat /etc/machine-id)" == "${EXPECTED_MACHINE_ID}" ]] || { echo "machine-id mismatch" >&2; exit 65; }
[[ "$(tr -d '\000' </proc/device-tree/serial-number)" == "${EXPECTED_SERIAL}" ]] || {
  echo "device-tree serial mismatch" >&2
  exit 65
}
[[ "$(cat /sys/class/net/wlan0/address)" == "${EXPECTED_WLAN_MAC}" ]] || {
  echo "wlan0 MAC mismatch" >&2
  exit 65
}

[[ -d "${RELEASE_ROOT}" && ! -L "${RELEASE_ROOT}" ]] || {
  echo "candidate release must be an existing non-symlink directory" >&2
  exit 66
}
RELEASE_ROOT="$(readlink -f "${RELEASE_ROOT}")"
case "${RELEASE_ROOT}" in
  "${RELEASES_PARENT}"/*) ;;
  *)
    echo "candidate release escaped the competition release parent" >&2
    exit 66
    ;;
esac

APP_ROOT="${RELEASE_ROOT}/rootscope"
TOOLS_ROOT="${APP_ROOT}/tools"
CONFIG_ROOT="${APP_ROOT}/configs"
R7_MODEL="${RELEASE_ROOT}/models/rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes.bin"
CANDIDATE_MANIFEST="${RELEASE_ROOT}/candidate_manifest.json"
CANDIDATE_SHA256SUMS="${RELEASE_ROOT}/SHA256SUMS"

FIELD_ROOT="${HOME}/.local/share/rootscope-field-v2"
CORE_PYTHON="${FIELD_ROOT}/core_v1/venvs/rootscope_x5_offline_core_v1/bin/python3"
LLAMA_SERVER="${FIELD_ROOT}/staged_components/rootscope_llama_server_arm64_b9637_v1/bin/llama-server"
QWEN_MODEL="${FIELD_ROOT}/readonly_llm/models/qwen2_05b_distill.Q4_K_M.gguf"
BPU_PYTHON="/usr/bin/python3"
HRT_MODEL_EXEC="/usr/sbin/hrt_model_exec"
LIBDNN="/usr/lib/libdnn.so"
LIBHBRT="/usr/lib/libhbrt_bayes_aarch64.so"

[[ -d "${APP_ROOT}/app" && -d "${TOOLS_ROOT}" && -d "${CONFIG_ROOT}" ]] || {
  echo "candidate release Python/config layout is incomplete" >&2
  exit 66
}
[[ -f "${CANDIDATE_MANIFEST}" && ! -L "${CANDIDATE_MANIFEST}" ]] || {
  echo "candidate manifest is missing or a symlink" >&2
  exit 66
}
[[ -f "${CANDIDATE_SHA256SUMS}" && ! -L "${CANDIDATE_SHA256SUMS}" ]] || {
  echo "candidate SHA256SUMS is missing or a symlink" >&2
  exit 66
}

install -d -m 700 "${RUNS_PARENT}"
if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  RUN_ROOT="${RUNS_PARENT}/${RUN_ID}"
fi
case "${RUN_ROOT}" in
  "${RUNS_PARENT}"/*) ;;
  *)
    echo "run root escaped the competition runs parent" >&2
    exit 66
    ;;
esac
[[ ! -e "${RUN_ROOT}" && ! -L "${RUN_ROOT}" ]] || {
  echo "run root must be new" >&2
  exit 66
}
install -d -m 700 "${RUN_ROOT}"
RUN_ROOT="$(readlink -f "${RUN_ROOT}")"

CHECKSUM_LOG="${RUN_ROOT}/candidate_sha256_check.log"
DEPENDENCY_LOG="${RUN_ROOT}/dependency_preflight.json"
LLAMA_HELP="${RUN_ROOT}/llama_server_help.txt"
LLAMA_VERSION="${RUN_ROOT}/llama_server_version.txt"
LLAMA_LOG="${RUN_ROOT}/llama_server.log"
LLAMA_HEALTH="${RUN_ROOT}/llama_health.json"
BPU_LOG="${RUN_ROOT}/bpu_worker.log"
BPU_SOCKET="${RUN_ROOT}/r7.sock"
ROLE_REPORT="${RUN_ROOT}/competition_llm_three_roles.json"
ROLE_STDOUT="${RUN_ROOT}/competition_llm_stdout.log"
ROLE_STDERR="${RUN_ROOT}/competition_llm_stderr.log"
READY_RECEIPT="${RUN_ROOT}/runtime_ready.json"
CLEANUP_RECEIPT="${RUN_ROOT}/runtime_cleanup.json"

LLAMA_PID=""
BPU_PID=""
ROLE_EXIT_CODE=""
STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

stop_pid_bounded() {
  local pid="$1"
  local label="$2"
  [[ -n "${pid}" ]] || return 0
  kill -TERM "${pid}" 2>/dev/null || return 0
  for _ in $(seq 1 10); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 0.2
  done
  printf '%s did not stop after TERM; sending KILL\n' "${label}" >&2
  kill -KILL "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

stop_loopback_9080_listener() {
  local listener_pids
  listener_pids="$(
    ss -H -ltnp 2>/dev/null |
      awk '$4 == "127.0.0.1:9080" {print $0}' |
      sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' |
      sort -u
  )"
  [[ -n "${listener_pids}" ]] || return 0
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    kill -TERM "${pid}" 2>/dev/null || true
  done <<<"${listener_pids}"
  sleep 1
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    kill -0 "${pid}" 2>/dev/null || continue
    printf 'loopback 9080 listener pid %s did not stop after TERM; sending KILL\n' "${pid}" >&2
    kill -KILL "${pid}" 2>/dev/null || true
  done <<<"${listener_pids}"
}

cleanup() {
  local original_rc="$?"
  trap - EXIT INT TERM
  set +e
  stop_pid_bounded "${LLAMA_PID}" "llama-server"
  stop_pid_bounded "${BPU_PID}" "bpu-worker"
  stop_loopback_9080_listener
  if [[ -S "${BPU_SOCKET}" ]]; then
    rm -f -- "${BPU_SOCKET}"
  fi
  local llama_stopped=true
  local bpu_stopped=true
  local port_closed=true
  [[ -z "${LLAMA_PID}" ]] || ! kill -0 "${LLAMA_PID}" 2>/dev/null || llama_stopped=false
  [[ -z "${BPU_PID}" ]] || ! kill -0 "${BPU_PID}" 2>/dev/null || bpu_stopped=false
  if curl --fail --silent --max-time 1 "http://127.0.0.1:9080/health" >/dev/null 2>&1; then
    port_closed=false
  fi
  "${BPU_PYTHON}" - \
    "${CLEANUP_RECEIPT}" "${original_rc}" "${STARTED_AT_UTC}" \
    "${llama_stopped}" "${bpu_stopped}" "${port_closed}" \
    "${ROLE_EXIT_CODE:-null}" <<'PY'
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

path = Path(sys.argv[1])
role_exit = None if sys.argv[7] == "null" else int(sys.argv[7])
payload = {
    "schema": "rootscope.competition-runtime-cleanup.v2",
    "process_exit_code": int(sys.argv[2]),
    "started_at_utc": sys.argv[3],
    "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "llama_server_stopped": sys.argv[4] == "true",
    "bpu_worker_stopped": sys.argv[5] == "true",
    "loopback_port_9080_closed": sys.argv[6] == "true",
    "competition_llm_exit_code": role_exit,
    "service_registered": False,
    "boot_persistence_created": False,
    "camera_opened_by_orchestrator": False,
    "external_network_touched": False,
    "serial_opened": False,
    "gpio_touched": False,
    "pump_touched": False,
    "execution_authority": False,
    "physical_authority": False,
}
path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
  exit "${original_rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

sha256_exact() {
  local path="$1"
  local expected="$2"
  local label="$3"
  [[ -f "${path}" && ! -L "${path}" ]] || {
    printf '%s is missing, non-regular, or a symlink\n' "${label}" >&2
    return 1
  }
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    printf '%s SHA-256 mismatch: %s != %s\n' "${label}" "${actual}" "${expected}" >&2
    return 1
  }
}

(
  cd "${RELEASE_ROOT}"
  sha256sum --strict --check SHA256SUMS
) >"${CHECKSUM_LOG}" 2>&1

sha256_exact "${R7_MODEL}" "${EXPECTED_R7_SHA256}" "r7 BPU model"
sha256_exact "${LLAMA_SERVER}" "${EXPECTED_LLAMA_SERVER_SHA256}" "llama-server"
sha256_exact "${QWEN_MODEL}" "${EXPECTED_QWEN_SHA256}" "Qwen2 GGUF"
sha256_exact "${HRT_MODEL_EXEC}" "${EXPECTED_HRT_MODEL_EXEC_SHA256}" "hrt_model_exec"
sha256_exact "${LIBDNN}" "${EXPECTED_LIBDNN_SHA256}" "libdnn"
sha256_exact "${LIBHBRT}" "${EXPECTED_LIBHBRT_SHA256}" "libhbrt"
[[ -x "${CORE_PYTHON}" ]] || {
  echo "core Python is missing or not executable" >&2
  exit 67
}
[[ -x "${BPU_PYTHON}" ]] || { echo "BPU system Python is unavailable" >&2; exit 67; }

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${APP_ROOT}" \
  "${CORE_PYTHON}" - "${DEPENDENCY_LOG}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

from app.competition_llm.contracts import AUTHORITY
from app.competition_runtime.bpu_shadow_client import BpuShadowClient

connection = sqlite3.connect(":memory:")
try:
    fts5 = connection.execute(
        "SELECT sqlite_compileoption_used('ENABLE_FTS5')"
    ).fetchone()[0] == 1
finally:
    connection.close()
if not fts5:
    raise SystemExit("SQLite FTS5 is unavailable")
if any(AUTHORITY.values()):
    raise SystemExit("competition LLM authority contract changed")
payload = {
    "schema": "rootscope.competition-runtime-dependency-preflight.v2",
    "core_python": sys.executable,
    "sqlite_fts5": True,
    "competition_llm_imported": True,
    "bpu_shadow_client_imported": BpuShadowClient.__name__,
    "authority_all_false": True,
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${APP_ROOT}" \
  "${BPU_PYTHON}" - <<'PY' >>"${RUN_ROOT}/bpu_python_preflight.log" 2>&1
import numpy
from PIL import Image
from app.competition_runtime.bpu_shadow_worker import HashBoundR7BpuBackend
print("BPU_SYSTEM_PYTHON_PREFLIGHT_OK")
print(numpy.__version__)
print(Image.__name__)
print(HashBoundR7BpuBackend.__name__)
PY

timeout 20 "${HRT_MODEL_EXEC}" --version >>"${RUN_ROOT}/bpu_python_preflight.log" 2>&1
timeout 20 "${HRT_MODEL_EXEC}" model_info --model_file "${R7_MODEL}" \
  >>"${RUN_ROOT}/bpu_python_preflight.log" 2>&1

timeout 20 "${LLAMA_SERVER}" --help >"${LLAMA_HELP}" 2>&1
for required_flag in \
  --model --host --port --ctx-size --threads --threads-batch --parallel \
  --batch-size --ubatch-size --no-warmup --no-ui --cache-ram
do
  grep -Fq -- "${required_flag}" "${LLAMA_HELP}" || {
    printf 'llama-server help omits required flag: %s\n' "${required_flag}" >&2
    exit 67
  }
done
timeout 20 "${LLAMA_SERVER}" --version >"${LLAMA_VERSION}" 2>&1
grep -Fq "9637" "${LLAMA_VERSION}"

if ss -H -ltn 2>/dev/null | awk '$4 ~ /:9080$/ { found=1 } END { exit(found ? 0 : 1) }'; then
  echo "TCP port 9080 already has a listener" >&2
  exit 68
fi
[[ ! -e "${BPU_SOCKET}" && ! -L "${BPU_SOCKET}" ]] || {
  echo "run-scoped BPU socket path already exists" >&2
  exit 68
}

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${APP_ROOT}" \
  "${BPU_PYTHON}" -m app.competition_runtime.bpu_shadow_worker \
    --socket "${BPU_SOCKET}" \
    --model-bin "${R7_MODEL}" \
    --expected-model-sha256 "${EXPECTED_R7_SHA256}" \
    --backend canonical_hrt \
    --hrt-model-exec "${HRT_MODEL_EXEC}" \
    --hrt-timeout-s 8.0 \
    --connection-timeout-s 1.0 \
    >"${BPU_LOG}" 2>&1 &
BPU_PID="$!"

bpu_ready=0
for _ in $(seq 1 90); do
  if [[ -S "${BPU_SOCKET}" ]] && kill -0 "${BPU_PID}" 2>/dev/null; then
    bpu_ready=1
    break
  fi
  kill -0 "${BPU_PID}" 2>/dev/null || break
  sleep 1
done
[[ "${bpu_ready}" == 1 ]] || {
  echo "BPU worker did not create its run-scoped AF_UNIX socket" >&2
  exit 69
}

"${LLAMA_SERVER}" \
  --model "${QWEN_MODEL}" \
  --host 127.0.0.1 \
  --port 9080 \
  --ctx-size 512 \
  --threads 2 \
  --threads-batch 2 \
  --parallel 1 \
  --batch-size 32 \
  --ubatch-size 16 \
  --no-warmup \
  --no-ui \
  --cache-ram 0 \
  >"${LLAMA_LOG}" 2>&1 &
LLAMA_PID="$!"

llama_ready=0
for _ in $(seq 1 180); do
  if curl --fail --silent --show-error --max-time 2 \
      --output "${LLAMA_HEALTH}" "http://127.0.0.1:9080/health" 2>/dev/null \
      && grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' "${LLAMA_HEALTH}"; then
    llama_ready=1
    break
  fi
  kill -0 "${LLAMA_PID}" 2>/dev/null || break
  sleep 1
done
[[ "${llama_ready}" == 1 ]] || {
  echo "loopback llama-server did not become ready" >&2
  exit 69
}

mapfile -t llama_listeners < <(
  ss -H -ltn | awk '$4 ~ /:9080$/ {print $4}'
)
[[ "${#llama_listeners[@]}" == 1 && "${llama_listeners[0]}" == "127.0.0.1:9080" ]] || {
  printf 'llama-server listener is not exactly loopback-only: %s\n' \
    "${llama_listeners[*]:-NONE}" >&2
  exit 70
}

set +e
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${APP_ROOT}" timeout 240 \
  "${CORE_PYTHON}" -m app.competition_llm.competition_rag \
    --endpoint "http://127.0.0.1:9080" \
    --model-id "qwen2-0.5b-q4km-rootscope-competition" \
    --model-sha256 "${EXPECTED_QWEN_SHA256}" \
    --timeout 180 \
    --api-mode completion \
    --corpus "${CONFIG_ROOT}/competition/rootscope_rag_corpus.v1.jsonl" \
    --registry "${CONFIG_ROOT}/competition/rootscope_rag_sources.v1.json" \
    --allowlist "${CONFIG_ROOT}/competition/rootscope_rag_citation_allowlist.v1.json" \
    --query "RootScope X5 4GB safety BPU evidence" \
    --output "${ROLE_REPORT}" \
    >"${ROLE_STDOUT}" 2>"${ROLE_STDERR}"
ROLE_EXIT_CODE="$?"
set -e
[[ "${ROLE_EXIT_CODE}" == 0 || "${ROLE_EXIT_CODE}" == 2 ]] || {
  echo "competition LLM client failed outside its accepted/fallback contract" >&2
  exit 71
}

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${APP_ROOT}" \
  "${CORE_PYTHON}" - \
    "${ROLE_REPORT}" "${READY_RECEIPT}" "${ROLE_EXIT_CODE}" \
    "${LLAMA_PID}" "${BPU_PID}" "${RELEASE_ROOT}" \
    "${EXPECTED_LLAMA_SERVER_SHA256}" "${EXPECTED_QWEN_SHA256}" \
    "${EXPECTED_R7_SHA256}" "${LIVE_MODE}" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

report_path = Path(sys.argv[1])
ready_path = Path(sys.argv[2])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report["cluster_topology"]["resident_model_count"] != 1:
    raise SystemExit("resident model count changed")
if report["cluster_topology"]["logical_roles"] != [
    "EVIDENCE_EXPLAINER",
    "SAFETY_AUDITOR",
    "DEFENSE_QA",
]:
    raise SystemExit("logical roles changed")
if report["generation"]["inference_call_budget"] != 1:
    raise SystemExit("inference call budget changed")
if report["generation"]["inference_call_count"] != 1:
    raise SystemExit("competition role projection did not make exactly one call")
if len(report["roles"]) != 3:
    raise SystemExit("role projection count changed")
if any(report["authority"].values()):
    raise SystemExit("LLM authority is not all false")
boundary = report["runtime_boundary"]
if boundary["external_network_touched"] is not False:
    raise SystemExit("external network boundary changed")

payload = {
    "schema": "rootscope.competition-runtime-ready.v2",
    "status": "READY_FOREGROUND_ONLY",
    "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "release_root": sys.argv[6],
    "llama_server_pid": int(sys.argv[4]),
    "bpu_worker_pid": int(sys.argv[5]),
    "llama_server_sha256": sys.argv[7],
    "qwen_model_sha256": sys.argv[8],
    "r7_model_sha256": sys.argv[9],
    "llm_role_exit_code": int(sys.argv[3]),
    "llm_model_output_accepted": report["provenance"]["model_output_accepted"],
    "llm_fallback_reason": report["provenance"]["fallback_reason"],
    "resident_model_count": 1,
    "logical_role_count": 3,
    "inference_call_count": 1,
    "llama_bind": "127.0.0.1:9080",
    "bpu_transport": "AF_UNIX",
    "bpu_qualification": "SHADOW_CANDIDATE_NOT_DEFAULT",
    "selected_bin_changed": False,
    "live_requested": sys.argv[10] == "1",
    "service_registered": False,
    "boot_persistence_created": False,
    "external_network_touched": False,
    "serial_opened": False,
    "gpio_touched": False,
    "pump_touched": False,
    "execution_authority": False,
    "physical_authority": False,
}
ready_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
PY

printf 'RootScope competition runtime ready: %s\n' "${RUN_ROOT}"
printf 'LLM: one Qwen2 0.5B at 127.0.0.1:9080; role call count=1\n'
printf 'BPU: r7 shadow worker at %s\n' "${BPU_SOCKET}"

if [[ "${LIVE_MODE}" == 1 ]]; then
  "${TOOLS_ROOT}/start_x5_competition_live_vision_v2.sh" \
    --release-root "${RELEASE_ROOT}" \
    --run-root "${RUN_ROOT}/live" \
    --bpu-socket "${BPU_SOCKET}"
  exit "$?"
fi

printf 'No camera was opened. Runtime readiness smoke is complete; cleaning up.\n'
exit 0
