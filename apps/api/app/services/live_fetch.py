from __future__ import annotations

import re
from ipaddress import ip_address
from urllib.parse import urlparse

import httpx

from app.services.source_allowlist import is_allowed_source


def fetch_page_excerpt(
    url: str,
    *,
    max_chars: int = 3500,
    timeout: float = 8.0,
    allowlist: list[str] | None = None,
) -> str | None:
    """Fetch a URL and return a stripped plain-text excerpt, or None on any failure.

    Defense in depth against SSRF:
    - Reject anything that isn't http(s).
    - Reject private / loopback / link-local hosts before the request leaves.
    - If an ``allowlist`` is provided, the URL host+path must match it.
    - ``follow_redirects`` is OFF so a 30x to a private host cannot smuggle past
      the host check.
    """
    if not _is_safe_public_url(url):
        return None
    if allowlist is not None and not is_allowed_source(url, allowlist):
        return None
    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=False,
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


def _is_safe_public_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in {"localhost", "ip6-localhost", "ip6-loopback"}:
        return False
    if host.endswith(".local") or host.endswith(".internal"):
        return False
    if host.endswith(".w3.internal"):
        return False
    try:
        addr = ip_address(host)
    except ValueError:
        return True
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )
