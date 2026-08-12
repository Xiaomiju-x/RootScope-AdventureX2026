"""Serve the RootScope dashboard with an explicitly in-memory fixture action.

This entry point can only load ``SIMULATION_ONLY / FAKE_F407`` configuration.
It never imports a physical serial adapter.  A successful internal fixture run
is still exported to the page as ``SIMULATED_ONLY``.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any, Mapping

from ..simulation import run_simulated_once
from .server import DashboardServer
from .state_store import SnapshotStore, default_snapshot


class FixtureActions:
    """Serialized callbacks for one local dashboard process."""

    def __init__(
        self,
        store: SnapshotStore,
        *,
        config_path: Path,
        evidence_path: Path,
    ) -> None:
        self.store = store
        self.config_path = Path(config_path)
        self.evidence_path = Path(evidence_path)
        self._lock = threading.Lock()

    @staticmethod
    def _profile(payload: Mapping[str, Any]) -> str:
        unknown = set(payload) - {"profile"}
        if unknown:
            raise ValueError(f"unknown fixture fields: {sorted(unknown)}")
        profile = payload.get("profile", "Profile-B-SIM")
        if not isinstance(profile, str) or profile not in {
            "Profile-A-SIM",
            "Profile-B-SIM",
            "Profile-C-SIM",
        }:
            raise ValueError("profile must be Profile-A-SIM, Profile-B-SIM or Profile-C-SIM")
        return profile

    def start(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        profile = self._profile(payload)
        if not self._lock.acquire(blocking=False):
            raise ValueError("SIMULATION_ALREADY_RUNNING")
        try:
            result = run_simulated_once(
                self.evidence_path,
                self.config_path,
                profile_id=profile,
            )
            self.store.replace(result.dashboard_snapshot)
            report = result.report
            return {
                "mode": "SIMULATED_ONLY",
                "hardware_touched": False,
                "physical_completion_claim": False,
                "task_id": report["task_id"],
                "task_seq": report["task_seq"],
                "simulated_pipeline_state": report["simulated_pipeline_state"],
                "evidence_terminal_hash": report["evidence_terminal_hash"],
            }
        finally:
            self._lock.release()

    def reset_view(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if payload:
            raise ValueError("reset view accepts an empty JSON object")
        self.store.replace(default_snapshot())
        return {
            "mode": "SIMULATED_ONLY",
            "state": "BOOT_LOCKED",
            "hardware_touched": False,
            "view_only_reset": True,
        }


def build_fixture_server(
    *,
    host: str,
    port: int,
    config_path: Path,
    evidence_path: Path,
) -> DashboardServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the SIMULATED_ONLY fixture server may listen only on loopback")
    store = SnapshotStore()
    actions = FixtureActions(
        store,
        config_path=config_path,
        evidence_path=evidence_path,
    )
    return DashboardServer(
        store,
        host=host,
        port=port,
        actions={
            "/api/simulate/start": actions.start,
            "/api/simulate/reset": actions.reset_view,
        },
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "h12_simulation_config.json",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=project_root
        / "evidence"
        / "local_h12"
        / "dashboard_simulation_run.jsonl",
    )
    args = parser.parse_args()
    server = build_fixture_server(
        host=args.host,
        port=args.port,
        config_path=args.config,
        evidence_path=args.evidence,
    )
    host, port = server.address
    print(f"RootScope SIMULATED_ONLY dashboard: http://{host}:{port}")
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
