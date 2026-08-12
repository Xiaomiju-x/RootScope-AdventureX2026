#!/usr/bin/env bash
# Starts only the locked Dashboard after read-only preflight and simulated self-test.
set -euo pipefail

PROJECT_ROOT="${ROOTSCOPE_PROJECT_ROOT:-/opt/rootscope/current}"
PYTHON="${ROOTSCOPE_PYTHON:-/opt/rootscope/venv/bin/python3}"
CONFIG="${ROOTSCOPE_CAPSULE_CONFIG:-/etc/rootscope/capsule_config.json}"

cd "$PROJECT_ROOT"
# app.edge.service performs both gates in-process before binding the locked
# loopback Dashboard.  Keeping a single gate owner avoids loading ONNX twice.
exec "$PYTHON" -m app.edge.service --config "$CONFIG"
