from __future__ import annotations

from pathlib import Path

import httpx
from pypdf import PdfReader

from shared.logging import get_logger

logger = get_logger(__name__)


def download_pdf(url: str, dest: Path, timeout: float = 60.0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return _is_pdf_file(dest)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            content = response.content
            if not content.startswith(b"%PDF"):
                logger.warning("pdf_download_not_pdf", url=url)
                return False
            dest.write_bytes(content)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdf_download_failed", url=url, error=str(exc))
        return False


def _is_pdf_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"%PDF"
    except OSError:
        return False


def extract_pdf_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages[:40]:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdf_extract_failed", path=str(path), error=str(exc))
        return ""
