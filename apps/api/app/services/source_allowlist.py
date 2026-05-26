from urllib.parse import urlparse


def is_allowed_source(url: str, allowlist: list[str]) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path_host = f"{host}{parsed.path}".lower().rstrip("/")

    for entry in allowlist:
        normalized = entry.lower().removeprefix("https://").removeprefix("http://").rstrip("/")
        if host == normalized or host.endswith(f".{normalized}") or path_host.startswith(normalized):
            return True
    return False

