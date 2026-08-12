#!/usr/bin/env python3
"""Read-only host preflight for the RootScope seed-17 BPU staging package.

No process is launched and Docker/WSL/hb_mapper/device state is not queried.
The command for a later, explicitly authorized container run is printed only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
AUDITOR_PATH = Path(__file__).with_name("audit_rootscope_seed17_bpu_staging.py")
SPEC = importlib.util.spec_from_file_location("rootscope_bpu_staging_auditor", AUDITOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load independent BPU staging auditor")
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def preflight(staging: Path) -> dict[str, Any]:
    report = AUDITOR.audit_staging(staging)
    resolved = staging.resolve(strict=True)
    # This is display-only.  Running it requires a separate, explicit decision
    # because starting Docker Desktop may create host virtual network adapters.
    suggested = (
        'docker run --rm -v "'
        + str(resolved)
        + ':/workspace" -w /workspace '
        + AUDITOR.TOOLCHAIN_IMAGE
        + " bash scripts/inside_toolchain_mapper.sh"
    )
    return {
        "schema_version": "rootscope.seed17.bayes_e.readonly_host_preflight.v1",
        "status": "PASS_READ_ONLY_STAGING_PREFLIGHT",
        "staging_status": report["staging_status"],
        "staging_path": str(resolved),
        "python": platform.python_version(),
        "host_platform": platform.platform(),
        "staging_receipt_sha256": report["staging_receipt_sha256"],
        "staging_sha256sums_sha256": report["staging_sha256sums_sha256"],
        "calibration_sample_count": report["calibration"]["sample_count"],
        "unique_train_sources": report["calibration"]["unique_train_sources_covered"],
        "model_sha256": report["model"]["sha256"],
        "read_only": True,
        "processes_launched": 0,
        "docker_wsl_hb_mapper_device_queries_or_invocations": 0,
        "docker_daemon_state": "NOT_QUERIED",
        "hb_mapper_checker_executed": False,
        "hb_mapper_makertbin_executed": False,
        "bpu_compiled": False,
        "x5_ready": False,
        "model_qualified": False,
        "suggested_later_command_not_executed": suggested,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=AUDITOR.DEFAULT_STAGING)
    return parser.parse_args()


def main() -> int:
    result = preflight(parse_args().staging)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
