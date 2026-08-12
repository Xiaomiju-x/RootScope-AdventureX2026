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

if [[ ! -e "${CAMERA}" ]]; then
  echo "ERROR: Insta360 Link 2C is not connected: ${CAMERA}" >&2
  read -r -p "Press Enter to close..."
  exit 2
fi
if pgrep -f "x5_answer_card_live.py" >/dev/null; then
  echo "REFUSED: the vision-only window still owns the camera." >&2
  echo "Press Q in that window, then double-click this icon again." >&2
  read -r -p "Press Enter to close..."
  exit 3
fi

echo "RootScope continuous answer demo"
echo "SPACE = probe manually returned to TOP; arm exactly one next plant"
echo "Q/Esc = quit"
echo "After each completed cycle, live vision resumes automatically."
echo "The probe never returns upward automatically."

exec /usr/bin/python3 \
  "${ROOT}/tools/x5_visual_irrigation_kiosk.py" \
  --bundle "${BUNDLE}" \
  --camera "${CAMERA}" \
  --execute \
  --kiosk-token "START ROOTSCOPE CONTINUOUS KIOSK"
