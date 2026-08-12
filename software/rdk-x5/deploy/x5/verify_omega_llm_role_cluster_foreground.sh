#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 6 ]]; then
  echo "usage: $0 SERVER MODEL SERVER_SHA256 MODEL_SHA256 PROJECT_ROOT EVIDENCE_DIR" >&2
  exit 64
fi

server="$1"
model="$2"
expected_server_sha="$3"
expected_model_sha="$4"
project_root="$5"
evidence_dir="$6"
host=127.0.0.1
port=9080

test -x "$server"
test -r "$model"
test -f "$project_root/app/omega_runtime/loopback_llm_cluster.py"
test -f "$project_root/configs/omega/field_knowledge.v1.md"
test "$(sha256sum "$server" | awk '{print $1}')" = "$expected_server_sha"
test "$(sha256sum "$model" | awk '{print $1}')" = "$expected_model_sha"
install -d -m 700 "$evidence_dir"

health="$evidence_dir/health.json"
server_log="$evidence_dir/server.log"
cluster_report="$evidence_dir/role_cluster_report.json"
process_status="$evidence_dir/process_status.txt"
execution_receipt="$evidence_dir/execution_receipt.json"

if curl --fail --silent --max-time 1 "http://$host:$port/health" >/dev/null 2>&1; then
  echo "port $port already has a health endpoint" >&2
  exit 1
fi

started_ms="$(date +%s%3N)"
"$server" \
  --model "$model" \
  --host "$host" \
  --port "$port" \
  --ctx-size 2048 \
  --threads 2 \
  --threads-batch 2 \
  --parallel 1 \
  --batch-size 128 \
  --ubatch-size 64 \
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

grep -E '^(Name|State|VmPeak|VmRSS|VmHWM|Threads):' "/proc/$server_pid/status" \
  >"$process_status"

set +e
PYTHONPATH="$project_root" timeout 600 /usr/bin/python3 \
  -m app.omega_runtime.loopback_llm_cluster \
  --endpoint "http://$host:$port" \
  --model-sha256 "$expected_model_sha" \
  --timeout-seconds 150 \
  --corpus "$project_root/configs/omega/field_knowledge.v1.md" \
  --output "$cluster_report"
cluster_exit=$?
set -e
completed_ms="$(date +%s%3N)"

cleanup
trap - EXIT
if curl --fail --silent --max-time 1 "http://$host:$port/health" >/dev/null 2>&1; then
  echo "loopback endpoint remained available after stop" >&2
  exit 1
fi

/usr/bin/python3 - \
  "$health" "$cluster_report" "$execution_receipt" \
  "$started_ms" "$ready_ms" "$completed_ms" "$cluster_exit" \
  "$expected_server_sha" "$expected_model_sha" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


health_path, cluster_path, receipt_path = map(Path, sys.argv[1:4])
started_ms, ready_ms, completed_ms, cluster_exit = map(int, sys.argv[4:8])
server_sha, model_sha = sys.argv[8:10]
health = json.loads(health_path.read_text(encoding="utf-8"))
cluster = (
    json.loads(cluster_path.read_text(encoding="utf-8"))
    if cluster_path.is_file()
    else None
)
if cluster_exit == 0:
    status = "PASS_X5_THREE_LOGICAL_ROLES_CITED_ZERO_AUTHORITY"
elif cluster_exit == 2 and isinstance(cluster, dict):
    status = "SAFE_FALLBACK_X5_MODEL_OUTPUT_NOT_ACCEPTED"
else:
    status = "FAIL_X5_ROLE_CLUSTER_EXECUTION"

receipt = {
    "schema_version": "rootscope.omega.x5-role-cluster-execution.v1",
    "status": status,
    "cluster_exit_code": cluster_exit,
    "board_hostname": Path("/etc/hostname").read_text(encoding="utf-8").strip(),
    "machine_id": Path("/etc/machine-id").read_text(encoding="utf-8").strip(),
    "device_tree_serial": Path("/proc/device-tree/serial-number").read_bytes()
        .rstrip(b"\0").decode("ascii"),
    "server_sha256": server_sha,
    "model_sha256": model_sha,
    "bind_host": "127.0.0.1",
    "port": 9080,
    "server_ready_ms": ready_ms - started_ms,
    "cluster_roundtrip_ms": completed_ms - ready_ms,
    "cluster_report_sha256": sha256(cluster_path) if cluster_path.is_file() else None,
    "accepted_model_role_count": (
        cluster.get("accepted_model_role_count")
        if isinstance(cluster, dict)
        else None
    ),
    "deterministic_fallback_role_count": (
        cluster.get("deterministic_fallback_role_count")
        if isinstance(cluster, dict)
        else None
    ),
    "resident_model_count": 1,
    "logical_role_count": 3,
    "scheduling": "SERIAL_SHARED_ENDPOINT",
    "foreground_process_started": True,
    "process_stopped": True,
    "port_closed_after_stop": True,
    "service_started": False,
    "systemctl_invoked": False,
    "activation_gate_created": False,
    "external_network_touched": False,
    "tool_execution": False,
    "serial_write": False,
    "gpio_write": False,
    "state_machine_write": False,
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

if [[ "$cluster_exit" = 0 || "$cluster_exit" = 2 ]]; then
  exit "$cluster_exit"
fi
exit 1
