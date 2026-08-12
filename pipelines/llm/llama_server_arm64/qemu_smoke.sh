#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT=/release
MODEL=/model/qwen2_05b_distill.Q4_K_M.gguf
SERVER="$RELEASE_ROOT/bin/llama-server"
PORT=19080
LOG="$RELEASE_ROOT/metadata/qemu_server.log"
HEALTH="$RELEASE_ROOT/metadata/qemu_health_response.txt"
COMPLETION="$RELEASE_ROOT/metadata/qemu_completion_response.txt"
HEALTH_HEADERS="$RELEASE_ROOT/metadata/qemu_health_headers.txt"
COMPLETION_HEADERS="$RELEASE_ROOT/metadata/qemu_completion_headers.txt"

test -x "$SERVER"
test -r "$MODEL"

"$SERVER" --version > "$RELEASE_ROOT/metadata/qemu_version.txt" 2>&1
ldd "$SERVER" > "$RELEASE_ROOT/metadata/qemu_ldd.txt" 2>&1

"$SERVER" \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --ctx-size 256 \
  --threads 1 \
  --threads-batch 1 \
  --parallel 1 \
  --batch-size 32 \
  --ubatch-size 32 \
  --no-warmup \
  --no-ui \
  --cache-ram 0 \
  > "$LOG" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 240); do
  if curl --fail --silent --show-error --max-time 5 \
      --dump-header "$HEALTH_HEADERS" \
      --output "$HEALTH" \
      "http://127.0.0.1:$PORT/health" 2>/dev/null \
      && grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' "$HEALTH"; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done

if [[ "$ready" != 1 ]]; then
  printf 'QEMU health smoke did not become ready\n' >&2
  tail -n 80 "$LOG" >&2 || true
  exit 1
fi

payload='{"messages":[{"role":"user","content":"Reply with exactly OK."}],"temperature":0,"max_tokens":1,"stream":false}'
curl --fail --silent --show-error --max-time 180 \
  --dump-header "$COMPLETION_HEADERS" \
  --output "$COMPLETION" \
  --header 'Content-Type: application/json' \
  --data-binary "$payload" \
  "http://127.0.0.1:$PORT/v1/chat/completions"
grep -q '"choices"' "$COMPLETION"

kill "$server_pid"
wait "$server_pid" || true
trap - EXIT
