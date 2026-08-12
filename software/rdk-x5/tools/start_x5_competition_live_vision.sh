#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: start_x5_competition_live_vision.sh /absolute/run/root}"
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

EXPECTED_HOSTNAME="rootscope-x5"
EXPECTED_MACHINE_ID="00000000000000000000000000000001"
EXPECTED_SERIAL="3281556110258c1902ab5d9b0012004"
EXPECTED_WLAN_MAC="02:00:00:00:00:01"

test "$(hostname)" = "${EXPECTED_HOSTNAME}"
test "$(cat /etc/machine-id)" = "${EXPECTED_MACHINE_ID}"
test "$(tr -d '\000' </proc/device-tree/serial-number)" = "${EXPECTED_SERIAL}"
test "$(cat /sys/class/net/wlan0/address)" = "${EXPECTED_WLAN_MAC}"

case "${RUN_ROOT}" in
  /opt/rootscope/rootscope_competition_live_runs/*) ;;
  *)
    printf 'ERROR: run root is outside the frozen competition-live parent\n' >&2
    exit 2
    ;;
esac
test -d "${RUN_ROOT}"
test ! -L "${RUN_ROOT}"

V2_ROOT="${HOME}/.local/share/rootscope-field-v2"
OVERLAY_ROOT="${HOME}/.local/share/rootscope-event-vision-overlay/releases/rootscope_event_vision_overlay_v1_2"
APP_ROOT="${OVERLAY_ROOT}/rootscope"
CORE="${V2_ROOT}/core_v1/releases/rootscope_x5_offline_core_v1/rootscope"
PY="${V2_ROOT}/core_v1/venvs/rootscope_x5_offline_core_v1/bin/python3"
CAMERA="/dev/v4l/by-id/usb-Web_Camera_Web_Camera_202604081837-video-index0"
DEVICE="$(readlink -f "${CAMERA}")"

test -x "${PY}"
test -c "${DEVICE}"
OWNERS="$(fuser "${DEVICE}" 2>/dev/null || true)"
test -z "${OWNERS}"

printf '%s\n' '============================================================'
printf '%s\n' 'RootScope Competition Live Vision'
printf '%s\n' 'Primary: ResNet18 CPU + Gray-World + flip TTA + temporal fusion'
printf '%s\n' 'Shadow: Omega OOD + conformal + geometry (display-only, non-blocking)'
printf '%s\n' 'BPU plant model: OFF / not qualified'
printf '%s\n' 'Q or ESC in the preview window exits and releases the camera.'
printf 'Run evidence: %s\n' "${RUN_ROOT}"
printf '%s\n' '============================================================'

cd "${APP_ROOT}"
set +e
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${APP_ROOT}" \
  "${PY}" "${SCRIPT_ROOT}/x5_competition_live_vision.py" \
    --device "${CAMERA}" \
    --capsule "${V2_ROOT}/core_v1/config/rootscope_x5_offline_core_v1.capsule.json" \
    --model "${CORE}/deploy/x5/models/rootscope_seed17_cpu_experimental_opset11.onnx" \
    --registry "${CORE}/app/vision/known_card_template_registry.frozen.experimental.json" \
    --calibration "${APP_ROOT}/configs/omega/vision_board_replay_new_x5_20260723.json" \
    --matcher "${CORE}/app/vision/card_geometric_matcher.config.example.json" \
    --output-dir "${RUN_ROOT}/evidence" \
    --width 1920 --height 1080 --fps 30 \
  2>&1 | tee "${RUN_ROOT}/terminal.log"
RC="${PIPESTATUS[0]}"
set -e

OWNERS_AFTER="$(fuser "${DEVICE}" 2>/dev/null || true)"
printf 'Process exit code: %s\n' "${RC}"
printf 'Camera owner after exit: %s\n' "${OWNERS_AFTER}"
if test -n "${OWNERS_AFTER}"; then
  printf 'ERROR: camera owner remained after exit\n' >&2
  exit 30
fi
exit "${RC}"
