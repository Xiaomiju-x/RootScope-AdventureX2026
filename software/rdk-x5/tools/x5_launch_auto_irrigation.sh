#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/rootscope/.local/share/rootscope-auto-irrigation/current"
BUNDLE="/opt/rootscope/.local/share/rootscope-answer-vision/current"
ORT_SITE="/opt/rootscope/.local/share/rootscope-v3/venvs/rootscope_v3_pc_ready_20260724_bde610b5e429-cpu/lib/python3.10/site-packages"

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/opt/rootscope/.Xauthority}"
export ROOTSCOPE_ORT_SITE="${ORT_SITE}"
export PYTHONPATH="${ROOT}"

if pgrep -f "x5_answer_card_live.py" >/dev/null; then
  echo "REFUSED: the vision-only window still owns the camera."
  echo "Press Q in that window, then start this one-cycle program again."
  read -r -p "Press Enter to close..."
  exit 2
fi

exec /usr/bin/python3 \
  "${ROOT}/tools/x5_visual_irrigation_cycle.py" \
  --bundle "${BUNDLE}" \
  --execute \
  --manual-home-observed-at-top \
  --confirm-independent-motor-power \
  --confirm-water-path-safe \
  --confirm-emergency-stop-ready
