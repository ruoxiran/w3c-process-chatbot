"""Validate and use per-request LLM provider overrides.

When a user supplies their own LLM endpoint and API key in the chat request,
we still need to defend the W3C server against:

- SSRF into private / loopback / link-local / cloud-metadata addresses
- Schemes other than http / https
- Production deployments accidentally allowing http://

The api_key is held in a Pydantic ``SecretStr`` upstream so it doesn't leak
into log lines or audit blobs. This module only validates the URL and
constructs the one-shot client; the workflow never stores the override on
``self.*``.
"""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlparse

from app.core.config import Settings
from app.models.schemas import ProviderOverride
from app.services.ollama import OllamaClient
from app.services.openai_compatible import OpenAICompatibleClient


# Hostnames that always indicate "do not let the server reach this on behalf
# of an untrusted caller", independent of DNS resolution.
_BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".w3.internal")
_LOOPBACK_HOSTS = frozenset({"localhost", "ip6-localhost", "ip6-loopback"})


class ProviderOverrideError(ValueError):
    """Raised when a user-supplied provider URL fails validation."""


def _is_public_internet_host(host: str) -> bool:
    if not host:
        return False
    if host in _LOOPBACK_HOSTS:
        return False
    if any(host.endswith(suffix) for suffix in _BLOCKED_HOST_SUFFIXES):
        return False
    try:
        addr = ip_address(host)
    except ValueError:
        # Hostname rather than literal IP; trust DNS resolution downstream.
        # We still block obvious internal suffixes above.
        return True
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def validate_provider_override(override: ProviderOverride, settings: Settings) -> None:
    """Raise ProviderOverrideError if the override is not safe to call.

    Policy:
      - scheme must be http or https
      - in production (app_env != development), https is required
      - openai-compatible: must be a PUBLIC internet host. Loopback / private
        ranges / .internal / .local are rejected.
      - ollama: loopback / private ranges are allowed (self-hosters), but
        cloud-metadata-style link-local and .internal suffixes still rejected.
        Operators on public infra should set
        ``provider_override_allow_ollama=False`` to reject this kind entirely
        and close the residual DNS-rebinding gap (validate-time DNS lookup
        cannot prevent a hostname rebinding to a private IP at connect time).
    """
    if override.kind == "ollama" and not settings.provider_override_allow_ollama:
        raise ProviderOverrideError(
            "ollama provider overrides are disabled on this deployment; "
            "use an openai-compatible endpoint instead"
        )
    parsed = urlparse(str(override.base_url))
    if parsed.scheme not in {"http", "https"}:
        raise ProviderOverrideError("base_url must use http or https")
    if settings.app_env != "development" and parsed.scheme != "https":
        raise ProviderOverrideError("https is required for provider overrides in this environment")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ProviderOverrideError("base_url is missing a host")
    if any(host.endswith(suffix) for suffix in _BLOCKED_HOST_SUFFIXES):
        raise ProviderOverrideError(f"host suffix not allowed: {host}")

    # Cloud-metadata IPv4 link-local is always blocked, even for Ollama.
    try:
        addr = ip_address(host)
    except ValueError:
        addr = None
    if addr is not None and (addr.is_link_local or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
        raise ProviderOverrideError(f"address range not allowed: {host}")

    if override.kind == "openai-compatible" and not _is_public_internet_host(host):
        raise ProviderOverrideError(
            "openai-compatible overrides must point at a public host; loopback / "
            "private ranges / .internal / .local are not permitted"
        )


def build_override_client(
    override: ProviderOverride, settings: Settings
) -> OllamaClient | OpenAICompatibleClient:
    """Return a one-shot client for the override. Caller must not cache it."""
    validate_provider_override(override, settings)
    base_url = str(override.base_url)
    if override.kind == "ollama":
        return OllamaClient(base_url, settings.ollama_timeout_seconds)
    api_key = override.api_key.get_secret_value() if override.api_key else None
    return OpenAICompatibleClient(base_url, api_key, settings.openai_compatible_timeout_seconds)
