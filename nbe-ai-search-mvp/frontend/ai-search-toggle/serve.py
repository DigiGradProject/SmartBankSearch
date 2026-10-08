"""Serve the built frontend with no-store caching + /v1 API proxy.

The plain `python -m http.server` sends Last-Modified but no Cache-Control,
so browsers heuristically cache index.html and keep running a stale JS/CSS
bundle even after a rebuild — which caused repeated "my fix isn't live"
confusion during development. This server always revalidates.

It also proxies `/v1/*` requests to the FastAPI backend (default
http://127.0.0.1:7000). The frontend bundle is built with a relative API
base, so the UI works from any origin (localhost, forwarded port, LAN IP)
without CORS or hardcoded-host issues.
"""

from __future__ import annotations

import http.server
import json
import functools
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND = f"http://127.0.0.1:{sys.argv[2] if len(sys.argv) > 2 else 7000}"
PROXY_PREFIX = "/v1/"


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:  # noqa: D102 — stdlib hook
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    # ------------------------------------------------------------------
    # /v1/* → FastAPI backend proxy (same-origin for the browser)
    # ------------------------------------------------------------------

    def _is_proxy_path(self) -> bool:
        return self.path.startswith(PROXY_PREFIX) or self.path == "/v1"

    def _proxy(self) -> None:
        url = BACKEND + self.path
        body: bytes | None = None
        if "Content-Length" in self.headers:
            length = int(self.headers["Content-Length"])
            body = self.rfile.read(length)
        request = urllib.request.Request(url, data=body, method=self.command)
        for header in ("Content-Type", "Accept"):
            if self.headers.get(header):
                request.add_header(header, self.headers[header])
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
                self.send_response(response.status)
                content_type = response.headers.get("Content-Type", "application/json")
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (urllib.error.URLError, OSError) as exc:
            payload = json.dumps(
                {"detail": f"backend unreachable on {BACKEND}: {exc}"}
            ).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 — stdlib naming
        if self._is_proxy_path():
            self._proxy()
        else:
            super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 — stdlib naming
        if self._is_proxy_path():
            self._proxy()
        else:
            self.send_response(405)
            self.end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5173
    directory = Path(__file__).resolve().parent / "dist"
    handler = functools.partial(NoCacheHandler, directory=str(directory))
    print(
        f"Serving {directory} on http://0.0.0.0:{port} "
        f"(Cache-Control: no-store, /v1/* proxied to {BACKEND})"
    )
    http.server.ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()


if __name__ == "__main__":
    main()
