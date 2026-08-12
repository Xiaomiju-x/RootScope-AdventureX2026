#!/usr/bin/env bash
set -euo pipefail

BUNDLE="/opt/rootscope/.local/share/rootscope-answer-vision/current"
CAMERA="/dev/v4l/by-id/usb-Insta360_Insta360_Link_2C-video-index0"
ORT_SITE="/opt/rootscope/.local/share/rootscope-v3/venvs/rootscope_v3_pc_ready_20260724_bde610b5e429-cpu/lib/python3.10/site-packages"
SNAPSHOTS="/opt/rootscope/rootscope_answer_snapshots/live_answer_demo"

if [[ ! -e "${CAMERA}" ]]; then
  echo "ERROR: Insta360 Link 2C capture node is not present: ${CAMERA}" >&2
  echo "Reconnect the camera to the RDK X5 USB port and try again." >&2
  exit 2
fi
if [[ ! -f "${BUNDLE}/rootscope_answer_cards_resnet18_opset11.onnx" ]]; then
  echo "ERROR: active RootScope answer-card bundle is incomplete." >&2
  exit 3
fi

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/opt/rootscope/.Xauthority}"
export ROOTSCOPE_ORT_SITE="${ORT_SITE}"

echo "RootScope four-card answer vision"
echo "Camera: ${CAMERA}"
echo "Model:  $(readlink -f "${BUNDLE}")"
echo "Keys: Q/Esc quit, S save annotated evidence"
echo "Safety: vision-only; no serial/GPIO/pump/probe authority"

exec /usr/bin/python3 \
  "${BUNDLE}/runtime/x5_answer_card_live.py" \
  --bundle "${BUNDLE}" \
  --camera "${CAMERA}" \
  --width 1920 \
  --height 1080 \
  --fps 30 \
  --infer-every 3 \
  --snapshot-root "${SNAPSHOTS}"
