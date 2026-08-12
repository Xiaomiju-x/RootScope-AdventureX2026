#!/usr/bin/env bash
# Local-only candidate installer.  It performs no downloads and grants no hardware authority.
set -euo pipefail

PROJECT_ROOT="${ROOTSCOPE_PROJECT_ROOT:-/opt/rootscope/current}"
SYSTEM_PYTHON="${ROOTSCOPE_SYSTEM_PYTHON:-python3}"
VENV_DIR="${ROOTSCOPE_VENV_DIR:-/opt/rootscope/venv}"
WHEELHOUSE="$PROJECT_ROOT/deploy/x5/wheelhouse"
MANIFEST="$WHEELHOUSE/candidate_cp310_aarch64_manifest.json"
LOCK="$WHEELHOUSE/requirements-cp310-aarch64-candidate.txt"
LINKS="$WHEELHOUSE/candidate_cp310_aarch64"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "[fatal] candidate lock requires Linux aarch64" >&2
  exit 2
fi

"$SYSTEM_PYTHON" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"[fatal] candidate lock requires CPython 3.10, got {sys.version}")
PY

cd "$PROJECT_ROOT"
"$SYSTEM_PYTHON" "$WHEELHOUSE/audit_candidate_wheelhouse.py" \
  --manifest "$MANIFEST" \
  --require-wheel-files \
  >/tmp/rootscope-wheelhouse-audit.json
"$SYSTEM_PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python3" -m pip install \
  --no-index \
  --only-binary=:all: \
  --find-links "$LINKS" \
  --require-hashes \
  -r "$LOCK"

"$VENV_DIR/bin/python3" - <<'PY'
import json
import numpy
import onnxruntime
from PIL import Image
print(json.dumps({
    "status": "IMPORT_PASS_CANDIDATE_NOT_X5_QUALIFIED",
    "numpy": numpy.__version__,
    "Pillow": Image.__version__,
    "onnxruntime": onnxruntime.__version__,
    "providers": onnxruntime.get_available_providers(),
    "x5_validated": False,
    "execution_authority": False,
}, sort_keys=True))
PY

echo "[ok] candidate venv installed; run preflight/selftest and preserve receipts before any qualification claim"
