#!/usr/bin/env python3
"""Strict read-only static server for the public RootScope presentation."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent / "public"
ALLOWED_METHODS = "GET, HEAD, OPTIONS"
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; "
        "form-action 'none'; connect-src 'self'; img-src 'self' data:; "
        "style-src 'self'; script-src 'self'; font-src 'self'; "
        "manifest-src 'self'; worker-src 'self'"
    ),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": (
        "accelerometer=(), autoplay=(), camera=(), display-capture=(), geolocation=(), "
        "gyroscope=(), microphone=(), payment=(), usb=()"
    ),
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


class RootScopeHandler(SimpleHTTPRequestHandler):
    server_version = "RootScopePublic/1"
    sys_version = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print(json.dumps({
            "remote": self.client_address[0],
            "request": fmt % args,
        }, ensure_ascii=False), flush=True)

    def end_headers(self) -> None:
        for key, value in SECURITY_HEADERS.items():
            self.send_header(key, value)
        self.send_header("Cache-Control", self._cache_policy())
        self.send_header("Vary", "Accept-Encoding")
        super().end_headers()

    def _cache_policy(self) -> str:
        path = urlsplit(self.path).path
        if path in {"/", "/index.html", "/sw.js", "/healthz"}:
            return "no-store"
        return "public, max-age=3600, stale-while-revalidate=86400"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _safe_public_path(self) -> bool:
        raw = unquote(urlsplit(self.path).path)
        if "\x00" in raw or "\\" in raw:
            return False
        candidate = (ROOT / raw.lstrip("/")).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
            return True
        except ValueError:
            return False

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", ALLOWED_METHODS)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._send_json(HTTPStatus.OK, {
                "status": "ok",
                "service": "rootscope-public",
                "release": "rootscope-public-v1",
                "mode": "read-only",
            })
            return
        if not self._safe_public_path():
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_HEAD(self) -> None:
        self.do_GET()

    def _method_not_allowed(self) -> None:
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", ALLOWED_METHODS)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        body = b'{"error":"method_not_allowed","mode":"read-only"}'
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_CONNECT = _method_not_allowed
    do_TRACE = _method_not_allowed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("ROOTSCOPE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ROOTSCOPE_PORT", "29200")))
    args = parser.parse_args()
    mimetypes.add_type("image/svg+xml", ".svg")
    mimetypes.add_type("image/webp", ".webp")
    server = ThreadingHTTPServer((args.host, args.port), RootScopeHandler)
    print(json.dumps({"listening": f"{args.host}:{args.port}", "root": str(ROOT)}, ensure_ascii=False), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
