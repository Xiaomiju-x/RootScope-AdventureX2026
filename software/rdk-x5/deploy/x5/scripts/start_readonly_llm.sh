#!/usr/bin/env bash
# Manual, fail-closed launcher for one hash-frozen external llama-server.
set -euo pipefail

required=(
  ROOTSCOPE_PROJECT_ROOT ROOTSCOPE_PYTHON ROOTSCOPE_LLM_MANIFEST
  ROOTSCOPE_LLM_MODEL ROOTSCOPE_LLM_MODEL_SHA256 ROOTSCOPE_LLAMA_SERVER
  ROOTSCOPE_LLAMA_SERVER_SHA256 ROOTSCOPE_LLM_GATE_FILE ROOTSCOPE_LLM_HOST
  ROOTSCOPE_LLM_PORT ROOTSCOPE_LLM_THREADS ROOTSCOPE_LLM_CONTEXT
  ROOTSCOPE_LLM_READ_ONLY ROOTSCOPE_LLM_EXTERNAL_NETWORK
  ROOTSCOPE_LLM_TOOL_EXECUTION ROOTSCOPE_LLM_ACTUATOR_ACCESS
  ROOTSCOPE_LLM_MANUAL_ACK
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment value: $name" >&2
    exit 64
  fi
done

[[ "$ROOTSCOPE_LLM_HOST" == "127.0.0.1" ]] || { echo "host must be 127.0.0.1" >&2; exit 65; }
[[ "$ROOTSCOPE_LLM_PORT" == "9080" ]] || { echo "port must be 9080" >&2; exit 65; }
[[ "$ROOTSCOPE_LLM_READ_ONLY" == "true" ]] || { echo "read-only gate failed" >&2; exit 65; }
[[ "$ROOTSCOPE_LLM_EXTERNAL_NETWORK" == "false" ]] || { echo "external network must remain false" >&2; exit 65; }
[[ "$ROOTSCOPE_LLM_TOOL_EXECUTION" == "false" ]] || { echo "tool execution must remain false" >&2; exit 65; }
[[ "$ROOTSCOPE_LLM_ACTUATOR_ACCESS" == "false" ]] || { echo "actuator access must remain false" >&2; exit 65; }
[[ "$ROOTSCOPE_LLM_MANUAL_ACK" == "READ_ONLY_EXPLANATION_ONLY" ]] || { echo "manual acknowledgement missing" >&2; exit 65; }
[[ -f "$ROOTSCOPE_LLM_GATE_FILE" ]] || { echo "manual activation gate missing" >&2; exit 65; }
[[ "$(tr -d '\r\n' < "$ROOTSCOPE_LLM_GATE_FILE")" == "READ_ONLY_EXPLANATION_ONLY" ]] || {
  echo "manual activation gate content mismatch" >&2
  exit 65
}
[[ "$ROOTSCOPE_LLM_THREADS" =~ ^[1-8]$ ]] || { echo "threads must be 1..8" >&2; exit 65; }
[[ "$ROOTSCOPE_LLM_CONTEXT" =~ ^[0-9]+$ ]] || { echo "context must be numeric" >&2; exit 65; }
(( ROOTSCOPE_LLM_CONTEXT >= 512 && ROOTSCOPE_LLM_CONTEXT <= 4096 )) || {
  echo "context must be 512..4096" >&2
  exit 65
}

"$ROOTSCOPE_PYTHON" "$ROOTSCOPE_PROJECT_ROOT/deploy/x5/scripts/readonly_llm_preflight.py" \
  --manifest "$ROOTSCOPE_LLM_MANIFEST" \
  --model "$ROOTSCOPE_LLM_MODEL" \
  --llama-server "$ROOTSCOPE_LLAMA_SERVER" \
  --llama-server-sha256 "$ROOTSCOPE_LLAMA_SERVER_SHA256" \
  --host 127.0.0.1 \
  --port 9080

exec "$ROOTSCOPE_LLAMA_SERVER" \
  -m "$ROOTSCOPE_LLM_MODEL" \
  --host 127.0.0.1 \
  --port 9080 \
  -c "$ROOTSCOPE_LLM_CONTEXT" \
  -t "$ROOTSCOPE_LLM_THREADS" \
  -n 384
