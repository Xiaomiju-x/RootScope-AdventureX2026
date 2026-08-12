#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 5 ]]; then
  echo "usage: $0 SERVER MODEL SERVER_SHA256 MODEL_SHA256 EVIDENCE_DIR" >&2
  exit 64
fi

server="$1"
model="$2"
expected_server_sha="$3"
expected_model_sha="$4"
evidence_dir="$5"
host=127.0.0.1
port=9080

test -x "$server"
test -r "$model"
test "$(sha256sum "$server" | awk '{print $1}')" = "$expected_server_sha"
test "$(sha256sum "$model" | awk '{print $1}')" = "$expected_model_sha"
install -d -m 700 "$evidence_dir"

health="$evidence_dir/health.json"
completion="$evidence_dir/completion.json"
server_log="$evidence_dir/server.log"
receipt="$evidence_dir/receipt.json"
process_status="$evidence_dir/process_status.txt"

if curl --fail --silent --max-time 1 "http://$host:$port/health" >/dev/null 2>&1; then
  echo "port $port already has a health endpoint" >&2
  exit 1
fi

started_ms="$(date +%s%3N)"
"$server" \
  --model "$model" \
  --host "$host" \
  --port "$port" \
  --ctx-size 256 \
  --threads 2 \
  --threads-batch 2 \
  --parallel 1 \
  --batch-size 32 \
  --ubatch-size 32 \
  --no-warmup \
  --no-ui \
  --cache-ram 0 \
  >"$server_log" 2>&1 &
server_pid=$!

cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 180); do
  if curl --fail --silent --show-error --max-time 3 \
      --output "$health" "http://$host:$port/health" 2>/dev/null \
      && grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' "$health"; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
test "$ready" = 1
ready_ms="$(date +%s%3N)"

payload='{"messages":[{"role":"user","content":"Reply with exactly OK."}],"temperature":0,"max_tokens":4,"stream":false}'
curl --fail --silent --show-error --max-time 180 \
  --output "$completion" \
  --header 'Content-Type: application/json' \
  --data-binary "$payload" \
  "http://$host:$port/v1/chat/completions"
grep -q '"choices"' "$completion"
completed_ms="$(date +%s%3N)"
grep -E '^(Name|State|VmPeak|VmRSS|VmHWM|Threads):' "/proc/$server_pid/status" \
  >"$process_status"

cleanup
trap - EXIT
if curl --fail --silent --max-time 1 "http://$host:$port/health" >/dev/null 2>&1; then
  echo "loopback endpoint remained available after stop" >&2
  exit 1
fi

/usr/bin/python3 - \
  "$health" "$completion" "$receipt" \
  "$started_ms" "$ready_ms" "$completed_ms" \
  "$expected_server_sha" "$expected_model_sha" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

health_path, completion_path, receipt_path = map(Path, sys.argv[1:4])
started_ms, ready_ms, completed_ms = map(int, sys.argv[4:7])
server_sha, model_sha = sys.argv[7:9]
health = json.loads(health_path.read_text(encoding="utf-8"))
completion = json.loads(completion_path.read_text(encoding="utf-8"))
choices = completion.get("choices")
if not isinstance(choices, list) or not choices:
    raise SystemExit("completion has no choices")
message = choices[0].get("message")
if not isinstance(message, dict) or not str(message.get("content", "")).strip():
    raise SystemExit("completion content is empty")

receipt = {
    "schema": "rootscope.x5-llm-loopback-smoke.v1",
    "status": "PASS_X5_FOREGROUND_LOOPBACK_LLM_SMOKE_STOPPED",
    "board_hostname": Path("/etc/hostname").read_text(encoding="utf-8").strip(),
    "machine_id": Path("/etc/machine-id").read_text(encoding="utf-8").strip(),
    "device_tree_serial": Path("/proc/device-tree/serial-number").read_bytes()
        .rstrip(b"\0").decode("ascii"),
    "server_sha256": server_sha,
    "model_sha256": model_sha,
    "bind_host": "127.0.0.1",
    "port": 9080,
    "health": health,
    "response_content": str(message["content"]).strip(),
    "finish_reason": choices[0].get("finish_reason"),
    "usage": completion.get("usage"),
    "server_ready_ms": ready_ms - started_ms,
    "completion_roundtrip_ms": completed_ms - ready_ms,
    "foreground_process_started": True,
    "process_stopped": True,
    "port_closed_after_stop": True,
    "service_started": False,
    "systemctl_invoked": False,
    "activation_gate_created": False,
    "external_network_touched": False,
    "tool_execution": False,
    "actuator_access": False,
    "execution_authority": False,
    "physical_authority": False,
}
receipt_path.write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
PY
