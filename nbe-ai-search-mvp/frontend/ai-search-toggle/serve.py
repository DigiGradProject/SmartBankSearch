"""Serve the built frontend with no-store caching.

The plain `python -m http.server` sends Last-Modified but no Cache-Control,
so browsers heuristically cache index.html and keep running a stale JS/CSS
bundle even after a rebuild — which caused repeated "my fix isn't live"
confusion during development. This server always revalidates.
"""

from __future__ import annotations

import http.server
import functools
import sys
from pathlib import Path


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:  # noqa: D102 — stdlib hook
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5173
    directory = Path(__file__).resolve().parent / "dist"
    handler = functools.partial(NoCacheHandler, directory=str(directory))
    print(f"Serving {directory} on http://0.0.0.0:{port} (Cache-Control: no-store)")
    http.server.ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()


if __name__ == "__main__":
    main()
