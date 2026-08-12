#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

RELEASE_ROOT="${1:?release root required}"
ROLE="${2:?fast or deep role required}"
EVIDENCE_ROOT="${3:?evidence root required}"
EXPLICIT_CPU_PYTHON="${4:-${ROOTSCOPE_CPU_PYTHON:-}}"
RELEASE_ROOT="$(readlink -f "${RELEASE_ROOT}")"
case "${ROLE}" in
  fast) MODEL_DIR="${RELEASE_ROOT}/models/llm/fast" ;;
  deep) MODEL_DIR="${RELEASE_ROOT}/models/llm/deep" ;;
  *) echo "role must be fast or deep" >&2; exit 21 ;;
esac
models=()
mapfile -t models < <(find "${MODEL_DIR}" -maxdepth 1 -type f -name '*.gguf' -print)
[[ "${#models[@]}" == 1 ]] || {
  echo "RootMind role must contain exactly one GGUF" >&2
  exit 22
}
MODEL="${models[0]}"
SERVER="${RELEASE_ROOT}/bin/llama-server"
MANIFEST="${RELEASE_ROOT}/candidate_manifest.json"
CACHE_HELPER="${RELEASE_ROOT}/rootscope/tools/x5_rootmind_cache_release_v3.py"
[[ -f "${MODEL}" && -x "${SERVER}" && -f "${MANIFEST}" \
   && -f "${CACHE_HELPER}" ]] || {
  echo "RootMind model/runtime missing" >&2
  exit 22
}
mkdir -p "${EVIDENCE_ROOT}"
RUNTIME_PATHS="${EVIDENCE_ROOT}/runtime_paths.env"
[[ -f "${RUNTIME_PATHS}" ]] || {
  echo "candidate runtime_paths.env is missing" >&2
  exit 22
}
cpu_lines=()
mapfile -t cpu_lines < <(grep '^ROOTSCOPE_CPU_PYTHON=' "${RUNTIME_PATHS}" || true)
[[ "${#cpu_lines[@]}" == 1 ]] || {
  echo "runtime_paths.env must contain exactly one CPU interpreter" >&2
  exit 22
}
BOOTSTRAP_CPU_PYTHON="${cpu_lines[0]#ROOTSCOPE_CPU_PYTHON=}"
CPU_PYTHON="${EXPLICIT_CPU_PYTHON:-${BOOTSTRAP_CPU_PYTHON}}"
CANDIDATE_ID="$(basename "${RELEASE_ROOT}")"
EXPECTED_CPU_VENV="${HOME}/.local/share/rootscope-v3/venvs/${CANDIDATE_ID}-cpu"
EXPECTED_CPU_PYTHON="${EXPECTED_CPU_VENV}/bin/python3"
[[ "${BOOTSTRAP_CPU_PYTHON}" == "${EXPECTED_CPU_PYTHON}" \
   && "${CPU_PYTHON}" == "${EXPECTED_CPU_PYTHON}" \
   && -x "${CPU_PYTHON}" ]] || {
  echo "RootMind parser is not bound to the candidate CPU venv" >&2
  exit 22
}

LOG="${EVIDENCE_ROOT}/rootmind_${ROLE}.log"
HEALTH="${EVIDENCE_ROOT}/rootmind_${ROLE}_health.json"
REQUEST="${EVIDENCE_ROOT}/rootmind_${ROLE}_request.json"
FALLBACK_REQUEST="${EVIDENCE_ROOT}/rootmind_${ROLE}_explicit_gbnf_request.json"
RESPONSE="${EVIDENCE_ROOT}/rootmind_${ROLE}_response.json"
PRIMARY_RESPONSE="${EVIDENCE_ROOT}/rootmind_${ROLE}_schema_response.json"
FALLBACK_RESPONSE="${EVIDENCE_ROOT}/rootmind_${ROLE}_explicit_gbnf_response.json"
PRIMARY_CURL_STDERR="${EVIDENCE_ROOT}/rootmind_${ROLE}_schema_curl.stderr"
FALLBACK_CURL_STDERR="${EVIDENCE_ROOT}/rootmind_${ROLE}_explicit_gbnf_curl.stderr"
PRIMARY_HTTP_STATUS="${EVIDENCE_ROOT}/rootmind_${ROLE}_schema_http_status.json"
FALLBACK_HTTP_STATUS="${EVIDENCE_ROOT}/rootmind_${ROLE}_explicit_gbnf_http_status.json"
PRIMARY_LOG_DELTA="${EVIDENCE_ROOT}/rootmind_${ROLE}_schema_log_delta.txt"
COMPATIBILITY_EVIDENCE="${EVIDENCE_ROOT}/rootmind_${ROLE}_grammar_compatibility.json"
SERVER_VERSION_FILE="${EVIDENCE_ROOT}/rootmind_${ROLE}_server_version.txt"
PROC_STATUS="${EVIDENCE_ROOT}/rootmind_${ROLE}_proc_status.txt"
PROC_CMDLINE="${EVIDENCE_ROOT}/rootmind_${ROLE}_proc_cmdline.txt"
LISTENER_FILE="${EVIDENCE_ROOT}/rootmind_${ROLE}_listener.txt"
MODEL_BINDING="${EVIDENCE_ROOT}/rootmind_${ROLE}_model_binding.json"
CACHE_RELEASE_RECEIPT="${EVIDENCE_ROOT}/rootmind_${ROLE}_model_page_cache_release.json"
CACHE_RELEASE_CLEANUP="${EVIDENCE_ROOT}/rootmind_${ROLE}_model_page_cache_cleanup.json"
SERVER_PID=""
FORCED_KILL=false
FALLBACK_USED=false
FALLBACK_REASON="NONE"
FALLBACK_MS="-1"
CACHE_RELEASE_DONE=false

stop_server() {
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi
  local target="${SERVER_PID}"
  if kill -0 "${target}" 2>/dev/null; then
    kill "${target}" 2>/dev/null || true
    local deadline=$((SECONDS + 5))
    while kill -0 "${target}" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 0.1
    done
    if kill -0 "${target}" 2>/dev/null; then
      FORCED_KILL=true
      kill -9 "${target}" 2>/dev/null || true
    fi
  fi
  wait "${target}" 2>/dev/null || true
  SERVER_PID=""
}

cleanup() {
  local original_rc=$?
  trap - EXIT
  set +e
  stop_server
  # Failure paths still make a best-effort, exact-file release attempt after
  # the process and listener are gone.  Its separate receipt can never replace
  # the fail-closed main receipt or the original exit status.
  if [[ "${CACHE_RELEASE_DONE}" != true && -f "${MODEL_BINDING}" \
        && ! -e "${CACHE_RELEASE_CLEANUP}" ]]; then
    if ! ss -H -ltn 2>/dev/null |
        awk '$4 ~ /:9080$/ {found=1} END{exit(found?0:1)}'; then
      "${CPU_PYTHON}" -I "${CACHE_HELPER}" release \
        --release-root "${RELEASE_ROOT}" \
        --role "${ROLE}" \
        --binding "${MODEL_BINDING}" \
        --output "${CACHE_RELEASE_CLEANUP}" \
        --observe-seconds 2 >/dev/null 2>&1
    fi
  fi
  exit "${original_rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ss -H -ltn | awk '$4 ~ /:9080$/ {found=1} END{exit(found?0:1)}'; then
  echo "port 9080 already occupied" >&2
  exit 23
fi

"${CPU_PYTHON}" -I - \
  "${REQUEST}" "${FALLBACK_REQUEST}" "${EXPECTED_CPU_VENV}" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
fallback_output = Path(sys.argv[2])
expected_prefix = Path(sys.argv[3]).resolve(strict=True)
if Path(sys.prefix).resolve(strict=True) != expected_prefix or not sys.flags.isolated:
    raise SystemExit("request builder did not run in the isolated candidate CPU venv")
schema = {
    "type": "object",
    "properties": {
        "authority": {"type": "boolean", "const": False},
        "status": {"type": "string", "const": "READ_ONLY"},
    },
    "required": ["authority", "status"],
    "additionalProperties": False,
}
request = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are the RootMind read-only verifier. Return exactly one JSON "
                "object. Never call tools, issue commands, or claim physical authority."
            ),
        },
        {
            "role": "user",
            "content": "Report authority=false and status=READ_ONLY.",
        },
    ],
    "chat_template_kwargs": {"enable_thinking": False},
    "reasoning_format": "none",
    "parse_tool_calls": False,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "rootscope_read_only_smoke",
            "strict": True,
            "schema": schema,
        },
    },
    "max_tokens": 32,
    "temperature": 0,
    "seed": 17,
    "stream": False,
}
output.write_text(
    json.dumps(request, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
fallback_request = {
    "messages": [
        {
            "role": "user",
            "content": (
                'Output exactly {"authority":false,"status":"READ_ONLY"}. '
                "No tools or actions."
            ),
        },
    ],
    "chat_template_kwargs": {"enable_thinking": False},
    "reasoning_format": "deepseek",
    "parse_tool_calls": False,
    "grammar": (
        'root ::= "{\\"authority\\":false,'
        '\\"status\\":\\"READ_ONLY\\"}"'
    ),
    "grammar_lazy": False,
    "max_tokens": 32,
    "temperature": 0,
    "seed": 17,
    "stream": False,
}
fallback_output.write_text(
    json.dumps(fallback_request, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY

"${CPU_PYTHON}" -I "${CACHE_HELPER}" bind \
  --release-root "${RELEASE_ROOT}" \
  --role "${ROLE}" \
  --model "${MODEL}" \
  --output "${MODEL_BINDING}" >/dev/null

if ! timeout 15 "${SERVER}" --version >"${SERVER_VERSION_FILE}" 2>&1; then
  echo "unable to obtain llama-server build information" >&2
  exit 24
fi
[[ -s "${SERVER_VERSION_FILE}" ]] || {
  echo "llama-server build information is empty" >&2
  exit 24
}

START_NS="$(date +%s%N)"
"${SERVER}" --model "${MODEL}" --host 127.0.0.1 --port 9080 \
  --ctx-size 512 --threads 2 --threads-batch 2 --parallel 1 \
  --batch-size 32 --ubatch-size 16 --no-warmup --no-ui --cache-ram 0 \
  --jinja --reasoning-format none \
  >"${LOG}" 2>&1 &
SERVER_PID="$!"
ready=0
ready_deadline=$((SECONDS + 180))
while (( SECONDS < ready_deadline )); do
  if curl --fail --silent --max-time 2 -o "${HEALTH}" \
      http://127.0.0.1:9080/health \
      && grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' "${HEALTH}"; then
    ready=1
    break
  fi
  kill -0 "${SERVER_PID}" 2>/dev/null || break
  sleep 1
done
[[ "${ready}" == 1 ]] || { echo "RootMind server failed to become ready" >&2; exit 24; }
READY_NS="$(date +%s%N)"
listeners="$(ss -H -ltn | awk '$4 ~ /:9080$/ {print $4}')"
[[ "${listeners}" == "127.0.0.1:9080" ]] || {
  echo "RootMind listener is not exactly loopback" >&2
  exit 25
}
printf '%s\n' "${listeners}" >"${LISTENER_FILE}"

COMPLETION_START_NS="$(date +%s%N)"
PRIMARY_START_NS="${COMPLETION_START_NS}"
PRIMARY_LOG_BYTES_BEFORE="$(stat -c '%s' "${LOG}")"
: >"${PRIMARY_RESPONSE}"
: >"${PRIMARY_CURL_STDERR}"
set +e
PRIMARY_HTTP_CODE="$(
  curl --silent --show-error --connect-timeout 5 --max-time 180 \
    -H 'Content-Type: application/json' \
    --data-binary "@${REQUEST}" \
    -o "${PRIMARY_RESPONSE}" --write-out '%{http_code}' \
    http://127.0.0.1:9080/v1/chat/completions \
    2>"${PRIMARY_CURL_STDERR}"
)"
PRIMARY_CURL_CODE="$?"
set -e
PRIMARY_END_NS="$(date +%s%N)"
[[ "${PRIMARY_HTTP_CODE}" =~ ^[0-9]{3}$ ]] || {
  echo "RootMind schema attempt returned an invalid HTTP status" >&2
  exit 26
}
printf '{"curl_exit_code":%s,"http_status":%s}\n' \
  "${PRIMARY_CURL_CODE}" "${PRIMARY_HTTP_CODE}" >"${PRIMARY_HTTP_STATUS}"
sleep 0.1
PRIMARY_LOG_BYTES_AFTER="$(stat -c '%s' "${LOG}")"
if (( PRIMARY_LOG_BYTES_AFTER > PRIMARY_LOG_BYTES_BEFORE )); then
  tail -c "+$((PRIMARY_LOG_BYTES_BEFORE + 1))" "${LOG}" >"${PRIMARY_LOG_DELTA}"
else
  : >"${PRIMARY_LOG_DELTA}"
fi
PRIMARY_MS="$(( (PRIMARY_END_NS - PRIMARY_START_NS) / 1000000 ))"

if [[ "${PRIMARY_CURL_CODE}" != 0 ]]; then
  echo "RootMind schema attempt failed at loopback transport" >&2
  exit 26
elif [[ "${PRIMARY_HTTP_CODE}" =~ ^2[0-9]{2}$ ]]; then
  cp "${PRIMARY_RESPONSE}" "${RESPONSE}"
else
  # b9637/Qwen3 can reject the response_format-generated grammar when its root
  # redundantly includes the assistant generation prefix.  Only the exact
  # board-observed signature below may unlock one explicit exact-GBNF retry.
  [[ "${ROLE}" == "deep" && "${PRIMARY_HTTP_CODE}" == "400" ]] || {
    echo "RootMind schema request failed without the allowed Deep grammar signature" >&2
    exit 26
  }
  "${CPU_PYTHON}" -I - \
    "${PRIMARY_RESPONSE}" "${PRIMARY_LOG_DELTA}" "${COMPATIBILITY_EVIDENCE}" \
    "${ROLE}" "${PRIMARY_HTTP_CODE}" "${PRIMARY_CURL_CODE}" \
    "${EXPECTED_CPU_VENV}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    response_path,
    log_delta_path,
    output_path,
    role,
    http_status,
    curl_exit,
    expected_prefix_raw,
) = sys.argv[1:]
expected_prefix = Path(expected_prefix_raw).resolve(strict=True)
if Path(sys.prefix).resolve(strict=True) != expected_prefix or not sys.flags.isolated:
    raise SystemExit("compatibility classifier is outside the candidate CPU venv")
if role != "deep" or http_status != "400" or curl_exit != "0":
    raise SystemExit("compatibility retry requires Deep HTTP 400 with intact transport")

response_text = Path(response_path).read_text(encoding="utf-8")
try:
    response = json.loads(response_text)
except json.JSONDecodeError as exc:
    raise SystemExit("schema failure response is not JSON") from exc
error = response.get("error") if isinstance(response, dict) else None
if isinstance(error, dict):
    error_text = " ".join(str(value) for value in error.values())
elif isinstance(error, str):
    error_text = error
else:
    raise SystemExit("schema failure response has no explicit error")
if "failed to initialize samplers" not in error_text.casefold():
    raise SystemExit("HTTP 400 is not the observed sampler initialization failure")

log_delta = Path(log_delta_path).read_text(encoding="utf-8")
checks = {
    "grammar_sampler_init_error": (
        "error initializing grammar sampler for grammar:" in log_delta.casefold()
    ),
    "grammar_root_contains_assistant_prefix": (
        'root ::= "<|im_start|>assistant\\n"' in log_delta
    ),
    "generation_prompt_observed": "Generation prompt:" in log_delta,
    "empty_think_prompt_observed": (
        "<think>" in log_delta and "</think>" in log_delta
    ),
    "sampler_launch_failed": (
        "Failed to initialize samplers:" in log_delta
        and "failed to launch slot with task" in log_delta
    ),
}
if not all(checks.values()):
    raise SystemExit(f"Deep grammar incompatibility signature incomplete: {checks}")

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

receipt = {
    "schema": "rootscope.v3.rootmind-grammar-compatibility.v1",
    "status": "ALLOW_EXACTLY_ONE_EXPLICIT_GBNF_RETRY",
    "reason_code": (
        "B9637_QWEN3_ASSISTANT_PREFIX_JSON_SCHEMA_GRAMMAR_SAMPLER_INIT"
    ),
    "role": role,
    "primary_http_status": 400,
    "primary_curl_exit_code": 0,
    "checks": checks,
    "primary_response_sha256": sha256_file(response_path),
    "primary_log_delta_sha256": sha256_file(log_delta_path),
    "execution_authority": False,
    "physical_authority": False,
}
Path(output_path).write_text(
    json.dumps(receipt, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY
  FALLBACK_USED=true
  FALLBACK_REASON="B9637_QWEN3_ASSISTANT_PREFIX_JSON_SCHEMA_GRAMMAR_SAMPLER_INIT"
  FALLBACK_START_NS="$(date +%s%N)"
  : >"${FALLBACK_RESPONSE}"
  : >"${FALLBACK_CURL_STDERR}"
  set +e
  FALLBACK_HTTP_CODE="$(
    curl --silent --show-error --connect-timeout 5 --max-time 180 \
      -H 'Content-Type: application/json' \
      --data-binary "@${FALLBACK_REQUEST}" \
      -o "${FALLBACK_RESPONSE}" --write-out '%{http_code}' \
      http://127.0.0.1:9080/v1/chat/completions \
      2>"${FALLBACK_CURL_STDERR}"
  )"
  FALLBACK_CURL_CODE="$?"
  set -e
  FALLBACK_END_NS="$(date +%s%N)"
  [[ "${FALLBACK_HTTP_CODE}" =~ ^[0-9]{3}$ ]] || {
    echo "RootMind explicit-GBNF retry returned an invalid HTTP status" >&2
    exit 26
  }
  printf '{"curl_exit_code":%s,"http_status":%s}\n' \
    "${FALLBACK_CURL_CODE}" "${FALLBACK_HTTP_CODE}" >"${FALLBACK_HTTP_STATUS}"
  [[ "${FALLBACK_CURL_CODE}" == 0 && "${FALLBACK_HTTP_CODE}" =~ ^2[0-9]{2}$ ]] || {
    echo "RootMind single explicit-GBNF retry failed closed" >&2
    exit 26
  }
  FALLBACK_MS="$(( (FALLBACK_END_NS - FALLBACK_START_NS) / 1000000 ))"
  cp "${FALLBACK_RESPONSE}" "${RESPONSE}"
fi
COMPLETION_END_NS="$(date +%s%N)"
kill -0 "${SERVER_PID}" 2>/dev/null || {
  echo "RootMind server exited before evidence capture" >&2
  exit 26
}
cp "/proc/${SERVER_PID}/status" "${PROC_STATUS}"
tr '\0' '\n' <"/proc/${SERVER_PID}/cmdline" >"${PROC_CMDLINE}"
PEAK_RSS_KIB="$(awk '$1=="VmHWM:" {print $2}' "${PROC_STATUS}")"
CURRENT_RSS_KIB="$(awk '$1=="VmRSS:" {print $2}' "${PROC_STATUS}")"
[[ "${PEAK_RSS_KIB}" =~ ^[0-9]+$ && "${PEAK_RSS_KIB}" -gt 0 \
   && "${CURRENT_RSS_KIB}" =~ ^[0-9]+$ && "${CURRENT_RSS_KIB}" -gt 0 ]] || {
  echo "RootMind RSS evidence is unavailable" >&2
  exit 26
}

stop_server
if ss -H -ltn | awk '$4 ~ /:9080$/ {found=1} END{exit(found?0:1)}'; then
  echo "RootMind loopback listener remained open after stop" >&2
  exit 26
fi
STOPPED_NS="$(date +%s%N)"
"${CPU_PYTHON}" -I "${CACHE_HELPER}" release \
  --release-root "${RELEASE_ROOT}" \
  --role "${ROLE}" \
  --binding "${MODEL_BINDING}" \
  --output "${CACHE_RELEASE_RECEIPT}" \
  --observe-seconds 2 >/dev/null
CACHE_RELEASE_DONE=true
READY_MS="$(( (READY_NS - START_NS) / 1000000 ))"
COMPLETION_MS="$(( (COMPLETION_END_NS - COMPLETION_START_NS) / 1000000 ))"
LIFECYCLE_MS="$(( (STOPPED_NS - START_NS) / 1000000 ))"

"${CPU_PYTHON}" -I - \
  "${RESPONSE}" "${REQUEST}" "${HEALTH}" "${LOG}" \
  "${SERVER_VERSION_FILE}" "${PROC_STATUS}" "${PROC_CMDLINE}" "${LISTENER_FILE}" \
  "${ROLE}" "${RELEASE_ROOT}" "${SERVER}" "${MANIFEST}" \
  "${EXPECTED_CPU_VENV}" "${READY_MS}" "${COMPLETION_MS}" "${LIFECYCLE_MS}" \
  "${PEAK_RSS_KIB}" "${CURRENT_RSS_KIB}" "${FORCED_KILL}" \
  "${PRIMARY_RESPONSE}" "${FALLBACK_REQUEST}" "${FALLBACK_RESPONSE}" \
  "${PRIMARY_CURL_STDERR}" "${FALLBACK_CURL_STDERR}" \
  "${PRIMARY_HTTP_STATUS}" "${FALLBACK_HTTP_STATUS}" \
  "${PRIMARY_LOG_DELTA}" "${COMPATIBILITY_EVIDENCE}" \
  "${FALLBACK_USED}" "${FALLBACK_REASON}" "${PRIMARY_MS}" "${FALLBACK_MS}" \
  "${MODEL_BINDING}" "${CACHE_RELEASE_RECEIPT}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import sys

(
    response_path,
    request_path,
    health_path,
    log_path,
    version_path,
    proc_status_path,
    proc_cmdline_path,
    listener_path,
    role,
    release_root_raw,
    server_path_raw,
    manifest_path_raw,
    expected_cpu_venv_raw,
    ready_ms_raw,
    completion_ms_raw,
    lifecycle_ms_raw,
    peak_rss_raw,
    current_rss_raw,
    forced_kill_raw,
    primary_response_path,
    fallback_request_path,
    fallback_response_path,
    primary_curl_stderr_path,
    fallback_curl_stderr_path,
    primary_http_status_path,
    fallback_http_status_path,
    primary_log_delta_path,
    compatibility_evidence_path,
    fallback_used_raw,
    fallback_reason_raw,
    primary_ms_raw,
    fallback_ms_raw,
    model_binding_path,
    cache_release_receipt_path,
) = sys.argv[1:]


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant forbidden: {value}")


def reject_duplicate_pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key forbidden: {key}")
        output[key] = value
    return output


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=reject_constant,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise SystemExit(f"{label} must be an integer >= {minimum}")
    return value


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise SystemExit(f"{label} must be finite and non-negative")
    return value


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise SystemExit(f"{label} keys changed")
    return value


def exact_bool(value, expected, label):
    if type(value) is not bool or value is not expected:
        raise SystemExit(f"{label} must be {expected}")
    return value


def require_false_authority(value, label):
    expected_keys = {
        "execution_authority",
        "physical_authority",
        "external_network",
        "service_started",
        "serial_opened",
        "serial_write",
        "gpio_touched",
        "pump_command",
        "state_machine_write",
        "model_modified",
    }
    exact_keys(value, expected_keys, label)
    for key in sorted(expected_keys):
        exact_bool(value[key], False, f"{label}.{key}")
    return value


release_root = Path(release_root_raw).resolve(strict=True)
server_path = Path(server_path_raw).resolve(strict=True)
manifest_path = Path(manifest_path_raw).resolve(strict=True)
expected_cpu_venv = Path(expected_cpu_venv_raw).resolve(strict=True)
if Path(sys.prefix).resolve(strict=True) != expected_cpu_venv or not sys.flags.isolated:
    raise SystemExit("receipt parser did not run in the isolated candidate CPU venv")
if not re.fullmatch(r"rootscope_v3_pc_ready_20260724_[0-9a-f]{12}", release_root.name):
    raise SystemExit("unexpected RootScope v3 candidate id")

request = load_json(request_path)
expected_schema = {
    "type": "object",
    "properties": {
        "authority": {"type": "boolean", "const": False},
        "status": {"type": "string", "const": "READ_ONLY"},
    },
    "required": ["authority", "status"],
    "additionalProperties": False,
}
if request.get("chat_template_kwargs") != {"enable_thinking": False}:
    raise SystemExit("enable_thinking=false is not bound in the request")
if request.get("reasoning_format") != "none" or request.get("parse_tool_calls") is not False:
    raise SystemExit("RootMind request reasoning/tool boundary changed")
if "tools" in request or "tool_choice" in request:
    raise SystemExit("RootMind smoke must not supply a tool interface")
response_format = request.get("response_format")
if (
    not isinstance(response_format, dict)
    or response_format.get("type") != "json_schema"
    or response_format.get("json_schema", {}).get("strict") is not True
    or response_format.get("json_schema", {}).get("schema") != expected_schema
):
    raise SystemExit("RootMind exact JSON schema changed")
fallback_request = load_json(fallback_request_path)
expected_fallback_request = {
    "messages": [
        {
            "role": "user",
            "content": (
                'Output exactly {"authority":false,"status":"READ_ONLY"}. '
                "No tools or actions."
            ),
        },
    ],
    "chat_template_kwargs": {"enable_thinking": False},
    "reasoning_format": "deepseek",
    "parse_tool_calls": False,
    "grammar": (
        'root ::= "{\\"authority\\":false,'
        '\\"status\\":\\"READ_ONLY\\"}"'
    ),
    "grammar_lazy": False,
    "max_tokens": 32,
    "temperature": 0,
    "seed": 17,
    "stream": False,
}
if fallback_request != expected_fallback_request:
    raise SystemExit("RootMind explicit-GBNF retry request changed")
if "response_format" in fallback_request or "json_schema" in fallback_request:
    raise SystemExit("RootMind explicit-GBNF retry must bypass response_format")
if fallback_request.get("grammar_lazy") is not False:
    raise SystemExit("RootMind explicit-GBNF retry must be eager")
if "tools" in fallback_request or "tool_choice" in fallback_request:
    raise SystemExit("RootMind explicit-GBNF retry must not supply tools")

if fallback_used_raw not in {"true", "false"}:
    raise SystemExit("fallback-used evidence is invalid")
fallback_used = fallback_used_raw == "true"
primary_http_status = load_json(primary_http_status_path)
if set(primary_http_status) != {"curl_exit_code", "http_status"}:
    raise SystemExit("primary HTTP status evidence keys changed")
primary_curl_exit = exact_int(
    primary_http_status["curl_exit_code"], "primary.curl_exit_code"
)
primary_http_code = exact_int(
    primary_http_status["http_status"], "primary.http_status", 100
)
primary_response_sha = sha256_file(primary_response_path)
primary_log_delta = Path(primary_log_delta_path).read_text(encoding="utf-8")
fallback_http_status = None
compatibility_evidence = None
if fallback_used:
    if (
        role != "deep"
        or fallback_reason_raw
        != "B9637_QWEN3_ASSISTANT_PREFIX_JSON_SCHEMA_GRAMMAR_SAMPLER_INIT"
        or primary_curl_exit != 0
        or primary_http_code != 400
    ):
        raise SystemExit(
            "explicit-GBNF retry is outside its exact compatibility gate"
        )
    primary_error_response = load_json(primary_response_path)
    primary_error = (
        primary_error_response.get("error")
        if isinstance(primary_error_response, dict)
        else None
    )
    if isinstance(primary_error, dict):
        primary_error_text = " ".join(str(value) for value in primary_error.values())
    elif isinstance(primary_error, str):
        primary_error_text = primary_error
    else:
        raise SystemExit("primary HTTP 400 has no explicit error")
    signature_checks = {
        "response_sampler_failure": (
            "failed to initialize samplers" in primary_error_text.casefold()
        ),
        "grammar_sampler_init_error": (
            "error initializing grammar sampler for grammar:"
            in primary_log_delta.casefold()
        ),
        "grammar_root_contains_assistant_prefix": (
            'root ::= "<|im_start|>assistant\\n"' in primary_log_delta
        ),
        "generation_prompt_observed": "Generation prompt:" in primary_log_delta,
        "empty_think_prompt_observed": (
            "<think>" in primary_log_delta and "</think>" in primary_log_delta
        ),
        "sampler_launch_failed": (
            "Failed to initialize samplers:" in primary_log_delta
            and "failed to launch slot with task" in primary_log_delta
        ),
    }
    if not all(signature_checks.values()):
        raise SystemExit(
            f"recorded grammar compatibility signature is incomplete: {signature_checks}"
        )
    compatibility_evidence = load_json(compatibility_evidence_path)
    if (
        compatibility_evidence.get("status")
        != "ALLOW_EXACTLY_ONE_EXPLICIT_GBNF_RETRY"
        or compatibility_evidence.get("reason_code") != fallback_reason_raw
        or compatibility_evidence.get("role") != "deep"
        or compatibility_evidence.get("primary_http_status") != 400
        or compatibility_evidence.get("primary_curl_exit_code") != 0
        or compatibility_evidence.get("checks")
        != {
            key: value
            for key, value in signature_checks.items()
            if key != "response_sampler_failure"
        }
        or compatibility_evidence.get("primary_response_sha256")
        != primary_response_sha
        or compatibility_evidence.get("primary_log_delta_sha256")
        != sha256_file(primary_log_delta_path)
    ):
        raise SystemExit("grammar compatibility receipt is not bound to raw evidence")
    fallback_http_status = load_json(fallback_http_status_path)
    if (
        set(fallback_http_status) != {"curl_exit_code", "http_status"}
        or fallback_http_status.get("curl_exit_code") != 0
        or not 200 <= fallback_http_status.get("http_status", 0) <= 299
    ):
        raise SystemExit("single explicit-GBNF retry did not return HTTP success")
    if (
        not Path(fallback_response_path).is_file()
        or sha256_file(response_path) != sha256_file(fallback_response_path)
    ):
        raise SystemExit(
            "final response is not the single explicit-GBNF retry response"
        )
else:
    if fallback_reason_raw != "NONE":
        raise SystemExit("primary schema success cannot carry a fallback reason")
    if primary_curl_exit != 0 or not 200 <= primary_http_code <= 299:
        raise SystemExit("primary schema request did not return HTTP success")
    if sha256_file(response_path) != primary_response_sha:
        raise SystemExit("final response is not the schema-constrained response")
    for unexpected_path in (
        fallback_response_path,
        fallback_curl_stderr_path,
        fallback_http_status_path,
        compatibility_evidence_path,
    ):
        if Path(unexpected_path).exists():
            raise SystemExit("unused fallback evidence unexpectedly exists")

health = load_json(health_path)
if health.get("status") != "ok":
    raise SystemExit("RootMind health contract failed")
response = load_json(response_path)
choices = response.get("choices")
if not isinstance(choices, list) or len(choices) != 1:
    raise SystemExit("RootMind response must contain exactly one choice")
choice = choices[0]
if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
    raise SystemExit("RootMind response was truncated or did not stop cleanly")
message = choice.get("message")
if not isinstance(message, dict) or message.get("role") != "assistant":
    raise SystemExit("RootMind assistant message is missing")
if message.get("tool_calls") not in (None, []):
    raise SystemExit("RootMind attempted a tool call")
content = message.get("content")
if not isinstance(content, str):
    raise SystemExit("RootMind response content must be text")
try:
    decoded = json.loads(
        content,
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=reject_constant,
    )
except (TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"RootMind content is not one strict JSON object: {exc}") from exc
if type(decoded) is not dict or set(decoded) != {"authority", "status"}:
    raise SystemExit("RootMind content keys must be exactly authority/status")
if type(decoded["authority"]) is not bool or decoded["authority"] is not False:
    raise SystemExit("RootMind authority must be the boolean false")
if type(decoded["status"]) is not str or decoded["status"] != "READ_ONLY":
    raise SystemExit("RootMind status must be exactly READ_ONLY")

usage = response.get("usage")
if not isinstance(usage, dict):
    raise SystemExit("RootMind response usage is missing")
prompt_tokens = exact_int(usage.get("prompt_tokens"), "usage.prompt_tokens", 1)
completion_tokens = exact_int(
    usage.get("completion_tokens"), "usage.completion_tokens", 1
)
total_tokens = exact_int(usage.get("total_tokens"), "usage.total_tokens", 2)
if total_tokens != prompt_tokens + completion_tokens:
    raise SystemExit("RootMind usage token totals are inconsistent")
timings = response.get("timings")
if not isinstance(timings, dict):
    raise SystemExit("RootMind response timings are missing")
for key in (
    "prompt_n",
    "prompt_ms",
    "prompt_per_token_ms",
    "prompt_per_second",
    "predicted_n",
    "predicted_ms",
    "predicted_per_token_ms",
    "predicted_per_second",
):
    finite_number(timings.get(key), f"timings.{key}")

binding = load_json(model_binding_path)
exact_keys(
    binding,
    {
        "schema",
        "status",
        "created_utc",
        "candidate",
        "role",
        "model",
        "integrity",
        "authority",
    },
    "model_binding",
)
if (
    binding["schema"] != "rootscope.v3.rootmind-gguf-cache-binding.v1"
    or binding["status"] != "BOUND"
    or binding["role"] != role
    or not isinstance(binding["created_utc"], str)
    or not binding["created_utc"]
):
    raise SystemExit("model binding identity changed")
binding_candidate = exact_keys(
    binding["candidate"],
    {"id", "release_root", "manifest_path", "manifest_sha256"},
    "model_binding.candidate",
)
if (
    binding_candidate["id"] != release_root.name
    or binding_candidate["release_root"] != str(release_root)
    or binding_candidate["manifest_path"] != str(manifest_path)
    or binding_candidate["manifest_sha256"] != sha256_file(manifest_path)
):
    raise SystemExit("model binding candidate/manifest identity changed")
binding_model = exact_keys(
    binding["model"],
    {
        "path",
        "relative_path",
        "category",
        "bytes",
        "sha256",
        "stat_fingerprint",
    },
    "model_binding.model",
)
model_relative = binding_model["relative_path"]
expected_model_prefix = f"models/llm/{role}/"
if (
    not isinstance(model_relative, str)
    or not model_relative.startswith(expected_model_prefix)
    or "/" in model_relative[len(expected_model_prefix):]
    or not model_relative.endswith(".gguf")
    or binding_model["path"] != str(release_root / model_relative)
    or binding_model["category"]
    != ("ROOTMIND_FAST_MODEL" if role == "fast" else "ROOTMIND_DEEP_MODEL")
    or type(binding_model["bytes"]) is not int
    or binding_model["bytes"] <= 0
    or not isinstance(binding_model["sha256"], str)
    or not re.fullmatch(r"[0-9a-f]{64}", binding_model["sha256"])
    or not isinstance(binding_model["stat_fingerprint"], dict)
    or not binding_model["stat_fingerprint"]
):
    raise SystemExit("model binding model identity changed")
model_stat = exact_keys(
    binding_model["stat_fingerprint"],
    {
        "device",
        "inode",
        "mode",
        "nlink",
        "uid",
        "gid",
        "size",
        "mtime_ns",
        "ctime_ns",
    },
    "model_binding.model.stat_fingerprint",
)
for key in ("device", "inode", "mode", "uid", "gid", "mtime_ns", "ctime_ns"):
    exact_int(model_stat[key], f"model_binding.model.stat_fingerprint.{key}")
exact_int(
    model_stat["nlink"], "model_binding.model.stat_fingerprint.nlink", 1
)
if (
    exact_int(
        model_stat["size"], "model_binding.model.stat_fingerprint.size", 1
    )
    != binding_model["bytes"]
):
    raise SystemExit("model binding stat size differs from bound bytes")
binding_integrity = exact_keys(
    binding["integrity"],
    {
        "manifest_record_count",
        "unique_role_gguf",
        "content_sha256_verified",
        "regular_file",
        "nofollow_open",
    },
    "model_binding.integrity",
)
exact_int(
    binding_integrity["manifest_record_count"],
    "model_binding.integrity.manifest_record_count",
    1,
)
for key in (
    "unique_role_gguf",
    "content_sha256_verified",
    "regular_file",
    "nofollow_open",
):
    exact_bool(
        binding_integrity[key], True, f"model_binding.integrity.{key}"
    )
require_false_authority(binding["authority"], "model_binding.authority")

cache_release = load_json(cache_release_receipt_path)
exact_keys(
    cache_release,
    {
        "schema",
        "status",
        "created_utc",
        "binding_sha256",
        "candidate",
        "role",
        "model",
        "integrity",
        "preconditions",
        "cache",
        "memory",
        "authority",
        "error",
    },
    "model_page_cache_release",
)
if (
    cache_release["schema"]
    != "rootscope.v3.rootmind-gguf-cache-release.v1"
    or cache_release["status"] != "PASS"
    or cache_release["role"] != role
    or not isinstance(cache_release["created_utc"], str)
    or not cache_release["created_utc"]
    or cache_release["binding_sha256"] != sha256_file(model_binding_path)
    or cache_release["candidate"] != binding_candidate
    or cache_release["model"] != binding_model
    or cache_release["error"] is not None
):
    raise SystemExit("model page-cache release identity changed or failed")
release_integrity = exact_keys(
    cache_release["integrity"],
    {
        "binding_valid",
        "release_root_unchanged",
        "manifest_path_unchanged",
        "manifest_sha256_unchanged",
        "manifest_record_unchanged",
        "model_path_unchanged",
        "model_stat_unchanged",
        "model_sha256_verified",
        "model_stat_unchanged_after",
        "model_modified",
    },
    "model_page_cache_release.integrity",
)
for key in (
    "binding_valid",
    "release_root_unchanged",
    "manifest_path_unchanged",
    "manifest_sha256_unchanged",
    "manifest_record_unchanged",
    "model_path_unchanged",
    "model_stat_unchanged",
    "model_sha256_verified",
    "model_stat_unchanged_after",
):
    exact_bool(
        release_integrity[key],
        True,
        f"model_page_cache_release.integrity.{key}",
    )
exact_bool(
    release_integrity["model_modified"],
    False,
    "model_page_cache_release.integrity.model_modified",
)
release_preconditions = exact_keys(
    cache_release["preconditions"],
    {"llama_server_processes", "no_llama_server", "endpoint", "port_closed"},
    "model_page_cache_release.preconditions",
)
if release_preconditions["llama_server_processes"] != []:
    raise SystemExit("llama-server remained during model page-cache release")
exact_bool(
    release_preconditions["no_llama_server"],
    True,
    "model_page_cache_release.preconditions.no_llama_server",
)
if release_preconditions["endpoint"] != "127.0.0.1:9080":
    raise SystemExit("model page-cache release endpoint changed")
exact_bool(
    release_preconditions["port_closed"],
    True,
    "model_page_cache_release.preconditions.port_closed",
)
release_cache = exact_keys(
    cache_release["cache"],
    {
        "method",
        "fadvise_applied",
        "resident_bytes_before",
        "resident_bytes_after",
        "resident_limit_bytes",
        "window_reached",
        "exact_file_only",
        "global_drop_caches",
        "sync_called",
        "compact_memory_called",
    },
    "model_page_cache_release.cache",
)
if release_cache["method"] != "POSIX_FADV_DONTNEED":
    raise SystemExit("model page-cache release method changed")
exact_bool(
    release_cache["fadvise_applied"],
    True,
    "model_page_cache_release.cache.fadvise_applied",
)
resident_before = exact_int(
    release_cache["resident_bytes_before"],
    "model_page_cache_release.cache.resident_bytes_before",
)
resident_after = exact_int(
    release_cache["resident_bytes_after"],
    "model_page_cache_release.cache.resident_bytes_after",
)
resident_limit = exact_int(
    release_cache["resident_limit_bytes"],
    "model_page_cache_release.cache.resident_limit_bytes",
)
if resident_limit != 4096 or resident_after > resident_limit:
    raise SystemExit("model page-cache residency release gate failed")
exact_bool(
    release_cache["window_reached"],
    True,
    "model_page_cache_release.cache.window_reached",
)
exact_bool(
    release_cache["exact_file_only"],
    True,
    "model_page_cache_release.cache.exact_file_only",
)
for key in ("global_drop_caches", "sync_called", "compact_memory_called"):
    exact_bool(
        release_cache[key],
        False,
        f"model_page_cache_release.cache.{key}",
    )
release_memory = exact_keys(
    cache_release["memory"],
    {
        "before",
        "after",
        "samples",
        "observe_seconds",
        "cma_free_minimum_kib",
        "window_reached",
    },
    "model_page_cache_release.memory",
)
if (
    not isinstance(release_memory["samples"], list)
    or len(release_memory["samples"]) < 2
    or finite_number(
        release_memory["observe_seconds"],
        "model_page_cache_release.memory.observe_seconds",
    )
    != 2
    or exact_int(
        release_memory["cma_free_minimum_kib"],
        "model_page_cache_release.memory.cma_free_minimum_kib",
    )
    != 131072
):
    raise SystemExit("model page-cache memory recovery evidence changed")
for label in ("before", "after"):
    snapshot = exact_keys(
        release_memory[label],
        {"mem_available_kib", "cma_free_kib", "cached_kib"},
        f"model_page_cache_release.memory.{label}",
    )
    for key in ("mem_available_kib", "cma_free_kib", "cached_kib"):
        exact_int(
            snapshot[key], f"model_page_cache_release.memory.{label}.{key}"
        )
for index, sample in enumerate(release_memory["samples"]):
    exact_keys(
        sample,
        {
            "elapsed_ms",
            "mem_available_kib",
            "cma_free_kib",
            "cached_kib",
            "resident_bytes",
            "gate_pass",
        },
        f"model_page_cache_release.memory.samples[{index}]",
    )
    for key in (
        "elapsed_ms",
        "mem_available_kib",
        "cma_free_kib",
        "cached_kib",
        "resident_bytes",
    ):
        exact_int(
            sample[key],
            f"model_page_cache_release.memory.samples[{index}].{key}",
        )
    exact_bool(
        sample["gate_pass"],
        True,
        f"model_page_cache_release.memory.samples[{index}].gate_pass",
    )
    if (
        sample["cma_free_kib"] < 131072
        or sample["resident_bytes"] > resident_limit
    ):
        raise SystemExit(
            f"model page-cache sample {index} violates the resource gate"
        )
last_sample = release_memory["samples"][-1]
if (
    last_sample["resident_bytes"] != resident_after
    or release_memory["after"]
    != {
        key: last_sample[key]
        for key in ("mem_available_kib", "cma_free_kib", "cached_kib")
    }
):
    raise SystemExit("final sampled residency differs from release receipt")
exact_bool(
    release_memory["window_reached"],
    True,
    "model_page_cache_release.memory.window_reached",
)
require_false_authority(
    cache_release["authority"], "model_page_cache_release.authority"
)

manifest = load_json(manifest_path)
if manifest.get("candidate_id") != release_root.name:
    raise SystemExit("candidate manifest id mismatch")
manifest_files = manifest.get("files")
if not isinstance(manifest_files, list):
    raise SystemExit("candidate manifest files are missing")
files_by_path = {}
for row in manifest_files:
    if not isinstance(row, dict) or not isinstance(row.get("path"), str):
        raise SystemExit("candidate manifest contains an invalid file row")
    if row["path"] in files_by_path:
        raise SystemExit("candidate manifest contains duplicate paths")
    files_by_path[row["path"]] = row
if binding_integrity["manifest_record_count"] != len(manifest_files):
    raise SystemExit("model binding manifest record count changed")
server_relative = server_path.relative_to(release_root).as_posix()
model_sha = binding_model["sha256"]
model_bytes = binding_model["bytes"]
server_sha = sha256_file(server_path)
model_row = files_by_path.get(model_relative)
if (
    not isinstance(model_row, dict)
    or model_row.get("sha256") != model_sha
    or model_row.get("bytes") != model_bytes
    or model_row.get("category") != binding_model["category"]
):
    raise SystemExit(f"candidate manifest binding failed for {model_relative}")
server_row = files_by_path.get(server_relative)
if (
    not isinstance(server_row, dict)
    or server_row.get("sha256") != server_sha
    or server_row.get("bytes") != server_path.stat().st_size
):
    raise SystemExit(f"candidate manifest binding failed for {server_relative}")

ready_ms = exact_int(int(ready_ms_raw), "elapsed.server_ready_ms")
completion_ms = exact_int(int(completion_ms_raw), "elapsed.completion_roundtrip_ms")
lifecycle_ms = exact_int(int(lifecycle_ms_raw), "elapsed.server_lifecycle_ms")
primary_ms = exact_int(int(primary_ms_raw), "elapsed.schema_primary_ms")
if fallback_used:
    fallback_ms = exact_int(int(fallback_ms_raw), "elapsed.explicit_gbnf_retry_ms")
else:
    if fallback_ms_raw != "-1":
        raise SystemExit("unused fallback must not report a duration")
    fallback_ms = None
peak_rss_kib = exact_int(int(peak_rss_raw), "resources.peak_rss_kib", 1)
current_rss_kib = exact_int(int(current_rss_raw), "resources.current_rss_kib", 1)
if forced_kill_raw not in {"true", "false"}:
    raise SystemExit("forced-kill evidence is invalid")
forced_kill = forced_kill_raw == "true"
listener = Path(listener_path).read_text(encoding="ascii").strip()
if listener != "127.0.0.1:9080":
    raise SystemExit("listener evidence is not exact loopback")
server_build = Path(version_path).read_text(encoding="utf-8").strip()
if not server_build:
    raise SystemExit("llama-server build evidence is empty")
if fallback_used and "9637" not in server_build:
    raise SystemExit("grammar compatibility downgrade is restricted to llama-server b9637")

status = (
    "PASS_X5_ROOTMIND_CHAT_TEMPLATE_EXPLICIT_GBNF_EXACT_READ_ONLY_"
    "SCHEMA_RUNTIME_INCOMPATIBLE"
    if fallback_used
    else "PASS_X5_ROOTMIND_CHAT_TEMPLATE_SCHEMA_LOCKED_READ_ONLY"
)
receipt = {
    "schema": "rootscope.v3.x5-rootmind-smoke.v3",
    "status": status,
    "role": role,
    "candidate": {
        "candidate_id": release_root.name,
        "release_root": str(release_root),
        "manifest_sha256": sha256_file(manifest_path),
    },
    "runtime": {
        "cpu_python": sys.executable,
        "cpu_prefix": str(expected_cpu_venv),
        "isolated_python": True,
        "server_path": str(server_path),
        "server_relative_path": server_relative,
        "server_sha256": server_sha,
        "server_bytes": server_path.stat().st_size,
        "server_build": server_build,
        "model_path": binding_model["path"],
        "model_relative_path": model_relative,
        "model_sha256": model_sha,
        "model_bytes": model_bytes,
        "resident_model_count": 1,
    },
    "transport": {
        "endpoint": "http://127.0.0.1:9080/v1/chat/completions",
        "listener": listener,
        "loopback_only": True,
        "external_network_touched": False,
        "request_sha256": sha256_file(request_path),
        "response_sha256": sha256_file(response_path),
        "schema_attempt": {
            "curl_exit_code": primary_curl_exit,
            "http_status": primary_http_code,
            "request_sha256": sha256_file(request_path),
            "response_sha256": primary_response_sha,
            "curl_stderr_sha256": sha256_file(primary_curl_stderr_path),
            "server_log_delta_sha256": sha256_file(primary_log_delta_path),
        },
        "explicit_gbnf_retry": (
            {
                "attempted": True,
                "curl_exit_code": fallback_http_status["curl_exit_code"],
                "http_status": fallback_http_status["http_status"],
                "request_sha256": sha256_file(fallback_request_path),
                "response_sha256": sha256_file(fallback_response_path),
                "curl_stderr_sha256": sha256_file(fallback_curl_stderr_path),
                "compatibility_evidence_sha256": sha256_file(
                    compatibility_evidence_path
                ),
            }
            if fallback_used
            else {
                "attempted": False,
                "request_sha256": sha256_file(fallback_request_path),
            }
        ),
        "health_sha256": sha256_file(health_path),
        "server_log_sha256": sha256_file(log_path),
        "server_version_sha256": sha256_file(version_path),
        "proc_status_sha256": sha256_file(proc_status_path),
        "proc_cmdline_sha256": sha256_file(proc_cmdline_path),
    },
    "contract": {
        "chat_template_applied": True,
        "enable_thinking": False,
        "schema_primary_attempted": True,
        "schema_primary_passed": not fallback_used,
        "json_schema_strict": not fallback_used,
        "explicit_gbnf_strict": fallback_used,
        "final_enforcement": (
            "EXPLICIT_EXACT_GBNF_AND_STRICT_EXACT_POST_PARSE_AFTER_PROVEN_"
            "B9637_RESPONSE_FORMAT_INCOMPATIBILITY"
            if fallback_used
            else "LLAMA_SERVER_JSON_SCHEMA_GRAMMAR_AND_STRICT_EXACT_POST_PARSE"
        ),
        "single_explicit_gbnf_retry_used": fallback_used,
        "compatibility_downgrade_reason": (
            fallback_reason_raw if fallback_used else None
        ),
        "compatibility_evidence": compatibility_evidence,
        "exact_output": decoded,
        "finish_reason": choice["finish_reason"],
        "tool_interface_supplied": False,
        "tool_calls_observed": False,
    },
    "performance": {
        "elapsed_ms": {
            "server_ready": ready_ms,
            "completion_roundtrip": completion_ms,
            "schema_primary": primary_ms,
            "explicit_gbnf_retry": fallback_ms,
            "server_lifecycle": lifecycle_ms,
        },
        "usage": usage,
        "timings": timings,
        "process_peak_rss_kib": peak_rss_kib,
        "process_rss_after_completion_kib": current_rss_kib,
    },
    "shutdown": {
        "process_stopped": True,
        "port_closed_after_stop": True,
        "forced_kill": forced_kill,
        "model_page_cache_release": cache_release,
    },
    "execution_authority": False,
    "physical_authority": False,
    "service_started": False,
    "serial_opened": False,
    "serial_write": False,
    "gpio_touched": False,
    "pump_command": False,
    "state_machine_write": False,
    "physical_completion": False,
}
print(json.dumps(receipt, ensure_ascii=True, sort_keys=True, allow_nan=False))
PY
