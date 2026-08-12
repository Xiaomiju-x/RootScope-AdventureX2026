#!/usr/bin/env bash
# Explicit Competition Live v2 launcher for the candidate release itself.
set -euo pipefail

EXPECTED_HOSTNAME="rootscope-x5"
EXPECTED_MACHINE_ID="00000000000000000000000000000001"
EXPECTED_SERIAL="3281556110220e0c002bdeab0012004"
EXPECTED_WLAN_MAC="02:00:00:00:00:01"

EXPECTED_R7_SHA256="4dc4bc24741332bb6bc6da184dd1d6f954ae539ba740330490339d76eb200285"
EXPECTED_CPU_CAPSULE_SHA256="1e839948e466895e0b416aababb39402380463f9ce125c8aeb59f4087191cb97"
EXPECTED_CPU_MODEL_SHA256="50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad"
EXPECTED_REGISTRY_SHA256="f5328b66c6b385dd081c1a27577d92b5c63fc18dc00d815f7d3d2561bb68e29f"
EXPECTED_CALIBRATION_SHA256="e82b196ab627a935fd571bba2b37aa637f040af5ba81fbab5183c1f15aa2e564"
EXPECTED_MATCHER_SHA256="9952864e50371675e7ea181cc57f2edd9eadd9189bfa0e9eda1e5cdd8f8ca61a"

RELEASES_PARENT="${HOME}/.local/share/rootscope-competition-runtime/releases"
RUNS_PARENT="${HOME}/.local/share/rootscope-competition-runtime/runs"
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DERIVED_RELEASE_ROOT="$(cd -- "${SCRIPT_ROOT}/../.." && pwd -P)"
RELEASE_ROOT="${ROOTSCOPE_COMPETITION_RELEASE_ROOT:-${DERIVED_RELEASE_ROOT}}"
RUN_ROOT=""
BPU_SOCKET=""

usage() {
  cat <<'EOF'
usage: start_x5_competition_live_vision_v2.sh \
  --run-root ABSOLUTE_PATH --bpu-socket ABSOLUTE_PATH [--release-root PATH]

The run root must be new and below the RootScope competition runs parent.
The BPU worker must already exist; this launcher never starts or owns it.
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
    --bpu-socket)
      [[ "$#" -ge 2 ]] || { echo "--bpu-socket needs a value" >&2; exit 64; }
      BPU_SOCKET="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      exit 64
      ;;
  esac
done

[[ -n "${RUN_ROOT}" && -n "${BPU_SOCKET}" ]] || {
  usage >&2
  exit 64
}
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

case "${RUN_ROOT}" in
  "${RUNS_PARENT}"/*) ;;
  *)
    echo "live run root escaped the competition runs parent" >&2
    exit 66
    ;;
esac
[[ ! -e "${RUN_ROOT}" && ! -L "${RUN_ROOT}" ]] || {
  echo "live run root must be new" >&2
  exit 66
}
install -d -m 700 "${RUN_ROOT}"
RUN_ROOT="$(readlink -f "${RUN_ROOT}")"

APP_ROOT="${RELEASE_ROOT}/rootscope"
TOOLS_ROOT="${APP_ROOT}/tools"
CONFIG_ROOT="${APP_ROOT}/configs"
LIVE_SCRIPT="${TOOLS_ROOT}/x5_competition_live_vision_v2.py"
R7_MODEL="${RELEASE_ROOT}/models/rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes.bin"

FIELD_ROOT="${HOME}/.local/share/rootscope-field-v2"
CORE_ROOT="${FIELD_ROOT}/core_v1/releases/rootscope_x5_offline_core_v1/rootscope"
CORE_PYTHON="${FIELD_ROOT}/core_v1/venvs/rootscope_x5_offline_core_v1/bin/python3"
CPU_CAPSULE="${FIELD_ROOT}/core_v1/config/rootscope_x5_offline_core_v1.capsule.json"
CPU_MODEL="${CORE_ROOT}/deploy/x5/models/rootscope_seed17_cpu_experimental_opset11.onnx"
REGISTRY="${CORE_ROOT}/app/vision/known_card_template_registry.frozen.experimental.json"
MATCHER="${CORE_ROOT}/app/vision/card_geometric_matcher.config.example.json"
CALIBRATION="${CONFIG_ROOT}/omega/vision_board_replay_new_x5_20260723.json"
CAMERA="/dev/v4l/by-id/usb-Web_Camera_Web_Camera_202604081837-video-index0"

sha256_exact() {
  local path="$1"
  local expected="$2"
  local label="$3"
  [[ -f "${path}" && ! -L "${path}" ]] || {
    printf '%s is missing, non-regular, or a symlink\n' "${label}" >&2
    return 1
  }
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || {
    printf '%s SHA-256 mismatch\n' "${label}" >&2
    return 1
  }
}

[[ -x "${CORE_PYTHON}" ]] || {
  echo "core Python is missing or not executable" >&2
  exit 67
}
[[ -f "${LIVE_SCRIPT}" && ! -L "${LIVE_SCRIPT}" ]] || {
  echo "Competition Live v2 is absent from the candidate release" >&2
  exit 67
}
[[ -S "${BPU_SOCKET}" && ! -L "${BPU_SOCKET}" ]] || {
  echo "explicit BPU AF_UNIX socket is unavailable" >&2
  exit 67
}
sha256_exact "${R7_MODEL}" "${EXPECTED_R7_SHA256}" "r7 BPU model"
sha256_exact "${CPU_CAPSULE}" "${EXPECTED_CPU_CAPSULE_SHA256}" "CPU capsule"
sha256_exact "${CPU_MODEL}" "${EXPECTED_CPU_MODEL_SHA256}" "CPU ONNX model"
sha256_exact "${REGISTRY}" "${EXPECTED_REGISTRY_SHA256}" "card registry"
sha256_exact "${CALIBRATION}" "${EXPECTED_CALIBRATION_SHA256}" "Omega calibration"
sha256_exact "${MATCHER}" "${EXPECTED_MATCHER_SHA256}" "geometry matcher"

[[ -e "${CAMERA}" || -L "${CAMERA}" ]] || {
  echo "explicit camera alias is missing" >&2
  exit 68
}
DEVICE="$(readlink -f "${CAMERA}")"
[[ -c "${DEVICE}" ]] || {
  echo "camera alias does not resolve to a character device" >&2
  exit 68
}
OWNERS="$(fuser "${DEVICE}" 2>/dev/null || true)"
[[ -z "${OWNERS}" ]] || {
  printf 'camera already has owner(s): %s\n' "${OWNERS}" >&2
  exit 68
}

printf '%s\n' '============================================================'
printf '%s\n' 'RootScope Competition Live Vision v2'
printf '%s\n' 'Code/PYTHONPATH: candidate release itself'
printf '%s\n' 'Primary: CPU audit/fallback + Gray-World/flip TTA/temporal fusion'
printf '%s\n' 'Proposal: r7 BPU via AF_UNIX (SHADOW_CANDIDATE_NOT_DEFAULT)'
printf '%s\n' 'Advisory: Omega OOD + geometry; never suppresses preview'
printf 'Run evidence: %s\n' "${RUN_ROOT}"
printf '%s\n' '============================================================'

set +e
cd "${APP_ROOT}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${APP_ROOT}" \
  "${CORE_PYTHON}" "${LIVE_SCRIPT}" \
    --device "${CAMERA}" \
    --capsule "${CPU_CAPSULE}" \
    --model "${CPU_MODEL}" \
    --registry "${REGISTRY}" \
    --calibration "${CALIBRATION}" \
    --matcher "${MATCHER}" \
    --bpu-socket "${BPU_SOCKET}" \
    --expected-bpu-model-sha256 "${EXPECTED_R7_SHA256}" \
    --bpu-timeout-s 3.0 \
    --bpu-interval-s 2.0 \
    --output-dir "${RUN_ROOT}/evidence" \
    --width 1920 --height 1080 --fps 30 \
  2>&1 | tee "${RUN_ROOT}/terminal.log"
RC="${PIPESTATUS[0]}"
set -e

OWNERS_AFTER="$(fuser "${DEVICE}" 2>/dev/null || true)"
printf 'Process exit code: %s\n' "${RC}"
printf 'Camera owner after exit: %s\n' "${OWNERS_AFTER}"
if [[ -n "${OWNERS_AFTER}" ]]; then
  echo "camera owner remained after Competition Live v2 exit" >&2
  exit 30
fi
exit "${RC}"
