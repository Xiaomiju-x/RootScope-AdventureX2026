"""Start the locked RootScope dashboard after offline capsule gates pass.

The service registers no action endpoints.  It may run one deterministic CPU
ONNX simulated-input check, but never opens configured RGB/depth devices,
serial, a BPU runtime, or an LLM endpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

from .capsule import CapsuleConfig
from .preflight import run_preflight
from .selftest import run_simulated_selftest
from ..web.server import DashboardServer
from ..web.state_store import SnapshotStore, default_snapshot


def locked_snapshot(config: CapsuleConfig) -> dict:
    snapshot = default_snapshot()
    snapshot.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "SIMULATED_ONLY",
            "state": "BOOT_LOCKED",
            "backend_actual": "clean_x5_capsule_locked_no_io",
            "capsule": {
                "status": config.status,
                "x5_validated": False,
                "model_enabled": config.model.enabled,
                "model_candidate": False,
                "model_qualified": False,
                "bpu_ready": False,
                "bpu_used": False,
                "rgb_enabled_but_not_opened": config.rgb.enabled,
                "depth_enabled_but_not_opened": config.depth.enabled,
                "llm_enabled_but_not_queried": config.llm.enabled,
            },
            "alerts": [
                "CLEAN_X5_CAPSULE_LOCKED_NO_HARDWARE_IO",
                "NO_ACTION_ENDPOINTS_REGISTERED",
                "CPU_ONNX_SELFTEST_IS_NOT_ACCURACY_OR_BPU_EVIDENCE",
            ],
        }
    )
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = CapsuleConfig.from_json_file(args.config)
    preflight = run_preflight(config)
    print(json.dumps(preflight, ensure_ascii=False, sort_keys=True))
    if preflight["status"] == "FAIL":
        return 2
    selftest = run_simulated_selftest(config)
    print(json.dumps(selftest, ensure_ascii=False, sort_keys=True))
    server = DashboardServer(
        SnapshotStore(locked_snapshot(config)),
        host=config.dashboard_host,
        port=config.dashboard_port,
        actions={},
    )
    host, port = server.address
    print(f"RootScope locked capsule dashboard: http://{host}:{port}")
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
