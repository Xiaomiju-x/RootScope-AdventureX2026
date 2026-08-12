#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/rootscope/.local/share/rootscope-auto-irrigation/current"
BUNDLE="/opt/rootscope/.local/share/rootscope-answer-vision/current"
CAMERA="/dev/v4l/by-id/usb-Insta360_Insta360_Link_2C-video-index0"
ORT_SITE="/opt/rootscope/.local/share/rootscope-v3/venvs/rootscope_v3_pc_ready_20260724_bde610b5e429-cpu/lib/python3.10/site-packages"

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/opt/rootscope/.Xauthority}"
export ROOTSCOPE_ORT_SITE="${ORT_SITE}"
export PYTHONPATH="${ROOT}"
export PYTHONUNBUFFERED=1

LOG_DIR="/opt/rootscope/.local/state/rootscope-auto-irrigation/launcher_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/autopilot-$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ ! -e "${CAMERA}" ]]; then
  echo "ERROR: Insta360 Link 2C is not connected: ${CAMERA}" >&2
  read -r -p "Press Enter to close..."
  exit 2
fi
if pgrep -f "x5_answer_card_live.py|x5_visual_irrigation_kiosk.py|x5_visual_irrigation_autopilot.py" >/dev/null; then
  echo "REFUSED: another RootScope camera window is already running." >&2
  echo "Close it with Q/Esc, then double-click this icon again." >&2
  read -r -p "Press Enter to close..."
  exit 3
fi

echo "RootScope hands-free continuous answer demo"
echo "Camera + inference stay live. One confirmed target triggers immediately."
echo "Present each new card only after the probe is physically back at TOP."
echo "Q/Esc = safe stop and quit."
echo "Runtime log: ${LOG_FILE}"

set +e
/usr/bin/python3 \
  "${ROOT}/tools/x5_visual_irrigation_autopilot.py" \
  --bundle "${BUNDLE}" \
  --camera "${CAMERA}" \
  --screen-width 1024 \
  --screen-height 600 \
  --clear-seconds 0.6 \
  --execute \
  --autopilot-token "START ROOTSCOPE HANDS FREE KIOSK" \
  2>&1 | tee "${LOG_FILE}"
rc="${PIPESTATUS[0]}"
set -e

if [[ "${rc}" -ne 0 ]]; then
  echo
  echo "RootScope exited with code ${rc}."
  echo "The error is preserved in ${LOG_FILE}."
  read -r -p "Press Enter to close..."
fi
exit "${rc}"
