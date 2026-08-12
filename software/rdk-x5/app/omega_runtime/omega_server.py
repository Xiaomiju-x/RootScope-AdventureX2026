"""Loopback-only, read-only RootScope-Ω Truth Ribbon dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.web.server import DashboardServer
from app.web.state_store import SnapshotStore

from .replay import run_locked_replay


def build_omega_server(
    *,
    host: str,
    port: int,
    cases_path: Path,
    profiles_path: Path,
    corpus_path: Path,
) -> DashboardServer:
    # DashboardServer currently uses the IPv4 ThreadingHTTPServer.  Accept one
    # numeric address only: this avoids depending on mutable hostname
    # resolution and avoids claiming IPv6 support that the server does not
    # provide.
    if host != "127.0.0.1":
        raise ValueError(
            "RootScope-Ω replay dashboard must listen on numeric IPv4 loopback "
            "127.0.0.1"
        )
    report = run_locked_replay(
        cases_path=cases_path,
        profiles_path=profiles_path,
        corpus_path=corpus_path,
    )
    return DashboardServer(
        SnapshotStore(report),
        host=host,
        port=port,
        static_root=Path(__file__).with_name("static"),
        actions={},
    )


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--cases",
        type=Path,
        default=root / "configs" / "omega" / "locked_replay_cases.v1.json",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=root / "configs" / "omega" / "edge_profiles.v1.json",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=root / "configs" / "omega" / "field_knowledge.v1.md",
    )
    args = parser.parse_args()
    server = build_omega_server(
        host=args.host,
        port=args.port,
        cases_path=args.cases,
        profiles_path=args.profiles,
        corpus_path=args.corpus,
    )
    host, port = server.address
    print(f"RootScope-Ω read-only dashboard: http://{host}:{port}")
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
