from urllib.parse import urlparse


def is_allowed_source(url: str, allowlist: list[str]) -> bool:
    """Match a URL against the allowlist.

    Each allowlist entry is either:
      - a bare host ("w3.org") → matches that host or any subdomain
      - a host + path prefix ("github.com/w3c") → matches the host AND requires
        the URL path to be EXACTLY ``/w3c`` or BEGIN with ``/w3c/``.

    The path-prefix branch previously used ``startswith`` directly, which
    silently let ``github.com/w3cmalicious/repo`` through because that string
    starts with ``github.com/w3c``. The fix is to require the prefix end at a
    path separator (or at end-of-string).
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path_host = f"{host}{parsed.path}".lower().rstrip("/")

    for entry in allowlist:
        normalized = entry.lower().removeprefix("https://").removeprefix("http://").rstrip("/")
        if not normalized:
            continue
        if host == normalized or host.endswith(f".{normalized}"):
            return True
        # Require a '/' boundary after the prefix so the allowlist entry
        # "github.com/w3c" does NOT match "github.com/w3cmalicious/...".
        if path_host == normalized or path_host.startswith(f"{normalized}/"):
            return True
    return False
