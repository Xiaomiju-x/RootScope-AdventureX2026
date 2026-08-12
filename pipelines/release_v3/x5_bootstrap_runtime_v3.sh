#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="${1:?verified candidate root required}"
EVIDENCE="${2:?evidence directory required}"
RELEASE_ROOT="$(readlink -f "${RELEASE_ROOT}")"
[[ "${RELEASE_ROOT}" =~ /rootscope_v3_pc_ready_20260724_[0-9a-f]{12}$ ]] || {
  echo "invalid v3 candidate root" >&2
  exit 41
}
SYSTEM_PYTHON="/usr/bin/python3"
STATE_ROOT="${HOME}/.local/share/rootscope-v3"
CANDIDATE_ID="$(basename "${RELEASE_ROOT}")"
CPU_VENV="${STATE_ROOT}/venvs/${CANDIDATE_ID}-cpu"
BPU_VENV="${STATE_ROOT}/venvs/${CANDIDATE_ID}-bpu-system-site"
WHEELHOUSE="${RELEASE_ROOT}/wheelhouse"
LOCK="${RELEASE_ROOT}/tools/release_v3/x5_wheelhouse_lock.v1.json"
REQUIREMENTS="${RELEASE_ROOT}/tools/release_v3/requirements-x5-cpu-v3.txt"
mkdir -p -m 700 "${STATE_ROOT}/venvs" "${EVIDENCE}"

"${SYSTEM_PYTHON}" - "${WHEELHOUSE}" "${LOCK}" <<'PY'
import hashlib
import json
import platform
from pathlib import Path
import sys

wheelhouse, lock_path = map(Path, sys.argv[1:])
if platform.machine() != "aarch64" or sys.version_info[:2] != (3, 10):
    raise SystemExit("RootScope v3 runtime requires Linux aarch64 CPython 3.10")
lock = json.loads(lock_path.read_text(encoding="utf-8"))
expected = lock["wheels"]
actual = {path.name: path for path in wheelhouse.glob("*.whl")}
if set(actual) != set(expected):
    raise SystemExit(
        f"wheelhouse coverage mismatch missing={sorted(set(expected)-set(actual))} "
        f"extra={sorted(set(actual)-set(expected))}"
    )
for name, expected_sha in expected.items():
    digest = hashlib.sha256(actual[name].read_bytes()).hexdigest()
    if digest != expected_sha:
        raise SystemExit(f"wheel hash mismatch: {name}")
PY

# The content-addressed candidate owns both venvs.  Recreate them for every
# acceptance so an old same-name directory cannot add untracked packages.
rm -rf "${CPU_VENV}" "${BPU_VENV}"
"${SYSTEM_PYTHON}" -m venv "${CPU_VENV}"
PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INDEX=1 \
  "${CPU_VENV}/bin/python3" -m pip install \
  --no-index --only-binary=:all: --find-links "${WHEELHOUSE}" \
  --require-hashes -r "${REQUIREMENTS}" >/dev/null

"${SYSTEM_PYTHON}" -m venv --system-site-packages "${BPU_VENV}"

"${CPU_VENV}/bin/python3" -I - "${CPU_VENV}" "${BPU_VENV}/bin/python3" "${EVIDENCE}/00_runtime_bootstrap.json" <<'PY'
import json
import importlib.metadata
from pathlib import Path
import subprocess
import sys

expected_cpu_root = Path(sys.argv[1]).resolve(strict=True)
bpu_python_entry = Path(sys.argv[2]).absolute()
if not bpu_python_entry.is_file():
    raise SystemExit("BPU venv interpreter entry is missing")
resolved_interpreter = bpu_python_entry.resolve(strict=True)
bpu_root = str(bpu_python_entry.parent.parent.resolve(strict=True))
output = Path(sys.argv[3])
import cv2
import numpy
import onnxruntime
from PIL import Image

cpu_root = Path(sys.prefix).resolve(strict=True)
if cpu_root != expected_cpu_root:
    raise SystemExit("CPU probe did not execute inside the isolated venv")
if Path(sys.base_prefix).resolve(strict=True) != Path("/usr"):
    raise SystemExit("CPU venv base interpreter is not the RDK system Python")
cpu_origins = {
    "numpy_origin": numpy.__file__,
    "cv2_origin": cv2.__file__,
    "onnxruntime_origin": onnxruntime.__file__,
    "pillow_origin": Image.__file__,
}
for key, value in cpu_origins.items():
    if not Path(value).resolve(strict=True).is_relative_to(cpu_root):
        raise SystemExit(f"{key} does not resolve inside the CPU venv")

required = {
    "coloredlogs": "15.0.1",
    "flatbuffers": "25.12.19",
    "humanfriendly": "10.0",
    "mpmath": "1.3.0",
    "numpy": "2.2.6",
    "onnxruntime": "1.22.1",
    "opencv-python-headless": "4.13.0.92",
    "packaging": "26.2",
    "pillow": "11.3.0",
    "protobuf": "7.35.1",
    "sympy": "1.14.0",
}
installed = {
    distribution.metadata["Name"].casefold().replace("_", "-"): distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
unexpected = set(installed) - set(required) - {"pip", "setuptools"}
if unexpected or any(installed.get(name) != version for name, version in required.items()):
    raise SystemExit(
        f"CPU venv distribution audit failed unexpected={sorted(unexpected)} "
        f"installed={installed}"
    )
probe = subprocess.run(
    [
        str(bpu_python_entry),
        "-I",
        "-c",
        "import cv2,hbm_runtime,json,numpy,sys;"
        "print(json.dumps({'python':sys.executable,'python_version':list(sys.version_info[:3]),"
        "'cv2':cv2.__version__,'cv2_origin':cv2.__file__,'numpy':numpy.__version__,"
        "'numpy_origin':numpy.__file__,'hbm_runtime_origin':hbm_runtime.__file__,"
        "'hbm_runtime_version':getattr(hbm_runtime,'__version__',None),"
        "'prefix':sys.prefix,'base_prefix':sys.base_prefix},sort_keys=True))",
    ],
    check=False,
    capture_output=True,
    text=True,
    timeout=60,
)
if probe.returncode != 0:
    raise SystemExit("BPU system-site runtime import probe failed: " + probe.stderr[-1000:])
bpu = json.loads(probe.stdout)
if Path(bpu["python"]).resolve(strict=True) != resolved_interpreter:
    raise SystemExit("BPU venv interpreter target changed")
if str(Path(bpu["prefix"]).resolve(strict=True)) != bpu_root:
    raise SystemExit("BPU probe did not execute inside the isolated venv")
if str(Path(bpu["base_prefix"]).resolve(strict=True)) != "/usr":
    raise SystemExit("BPU venv base interpreter is not the RDK system Python")
for key in ("cv2_origin", "numpy_origin", "hbm_runtime_origin"):
    origin = Path(bpu[key]).resolve(strict=True)
    if origin.is_relative_to(Path(bpu_root)):
        raise SystemExit(f"{key} unexpectedly resolves inside BPU venv")
    if not origin.is_relative_to(Path("/usr")):
        raise SystemExit(f"{key} does not resolve from the RDK vendor system tree")
receipt = {
    "schema": "rootscope.v3.x5-runtime-bootstrap.v1",
    "status": "PASS_OFFLINE_CPU_AND_VENDOR_BPU_INTERPRETERS",
    "cpu": {
        "python": sys.executable,
        "python_version": list(sys.version_info[:3]),
        "prefix": str(cpu_root),
        **cpu_origins,
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "Pillow": Image.__version__,
        "onnxruntime": onnxruntime.__version__,
        "providers": onnxruntime.get_available_providers(),
        "installed_distributions": installed,
        "source": "HASH_LOCKED_CANDIDATE_WHEELHOUSE",
    },
    "bpu": {
        **bpu,
        "source": "RDK_VENDOR_SYSTEM_SITE_PACKAGES_VIA_ISOLATED_VENV",
    },
    "network_touched": False,
    "camera_opened": False,
    "serial_opened": False,
    "gpio_touched": False,
    "pump_touched": False,
    "service_started": False,
    "physical_completion": False,
}
output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True))
PY

cat >"${EVIDENCE}/runtime_paths.env" <<EOF
ROOTSCOPE_CPU_PYTHON=${CPU_VENV}/bin/python3
ROOTSCOPE_BPU_PYTHON=${BPU_VENV}/bin/python3
EOF
chmod 600 "${EVIDENCE}/runtime_paths.env"
