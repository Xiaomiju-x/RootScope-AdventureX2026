#!/usr/bin/env bash
# Read-only local preflight.  No SSH, network, service start, device scan, or device open.
set -euo pipefail

PROJECT_ROOT="${ROOTSCOPE_PROJECT_ROOT:-/opt/rootscope/current}"
PYTHON="${ROOTSCOPE_PYTHON:-/opt/rootscope/venv/bin/python3}"
CONFIG="${ROOTSCOPE_CAPSULE_CONFIG:-/etc/rootscope/capsule_config.json}"

cd "$PROJECT_ROOT"
exec "$PYTHON" -m app.edge.cli preflight --config "$CONFIG"
