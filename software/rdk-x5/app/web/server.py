"""Small offline HTTP server for the H12 RootScope dashboard baseline."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .state_store import SnapshotStore


Action = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]


class DashboardServer:
    def __init__(
        self,
        store: SnapshotStore,
        host: str = "127.0.0.1",
        port: int = 8765,
        static_root: str | Path | None = None,
        actions: Mapping[str, Action] | None = None,
    ) -> None:
        self.store = store
        self.static_root = Path(static_root or Path(__file__).parent).resolve()
        self.actions = dict(actions or {})
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "RootScopeDashboard/0.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                return None

            def _json(self, status: int, payload: Mapping[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def _html(self, path: Path) -> None:
                if not path.is_file() or path.parent != outer.static_root:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = urlsplit(self.path).path
                if path == "/api/health":
                    self._json(HTTPStatus.OK, {"ok": True, "service": "rootscope-dashboard", "hardware_touched": False})
                elif path == "/api/status":
                    self._json(HTTPStatus.OK, outer.store.snapshot())
                elif path in {"/", "/index.html"}:
                    self._html(outer.static_root / "index.html")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                path = urlsplit(self.path).path
                action = outer.actions.get(path)
                if action is None:
                    self._json(HTTPStatus.CONFLICT, {"ok": False, "error": "ACTION_UNAVAILABLE_IN_CURRENT_MODE"})
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size < 0 or size > 16_384:
                        raise ValueError("request length must be within 0..16384 bytes")
                    payload = json.loads(self.rfile.read(size).decode("utf-8")) if size else {}
                    if not isinstance(payload, dict):
                        raise ValueError("JSON body must be an object")
                    result = action(payload) or {}
                    self._json(HTTPStatus.OK, {"ok": True, **dict(result)})
                except (ValueError, json.JSONDecodeError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                except Exception:
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "ACTION_FAILED"})

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="rootscope-http", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread:
            self._thread.join(timeout=3.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = DashboardServer(SnapshotStore(), host=args.host, port=args.port)
    host, port = server.address
    print(f"RootScope dashboard fixture: http://{host}:{port}")
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
