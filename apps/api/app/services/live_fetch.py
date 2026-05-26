from __future__ import annotations

import re

import httpx


def fetch_page_excerpt(url: str, *, max_chars: int = 3500, timeout: float = 8.0) -> str | None:
    """Fetch a URL and return a stripped plain-text excerpt, or None on any failure."""
    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "W3CProcessBot/1.0"},
        )
        resp.raise_for_status()
    except Exception:
        return None
    html = resp.text
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text[:max_chars] if text else None
