"""Regression tests for the security-hardening guards added in round 6.

Each test pins a specific guard so a future refactor can't silently
remove it. None of these exercise the network — they hit the helper
function directly.
"""

from __future__ import annotations

import pytest

from app.services import openai_compatible
from app.services.cross_encoder_reranker import (
    MissingDependencyError,
    _validate_model_name,
)


# ---------- Retry-After cap ------------------------------------------------


def test_parse_retry_after_caps_giant_values() -> None:
    """A hostile upstream returning ``Retry-After: 86400`` (24 h) must
    not be allowed to pin a worker thread for hours. The cap is 60 s."""
    assert openai_compatible._parse_retry_after("86400") == 60.0
    assert openai_compatible._parse_retry_after("3600") == 60.0
    assert openai_compatible._parse_retry_after("999999") == 60.0


def test_parse_retry_after_passes_through_small_values() -> None:
    assert openai_compatible._parse_retry_after("0") == 0.0
    assert openai_compatible._parse_retry_after("5") == 5.0
    assert openai_compatible._parse_retry_after("59.5") == 59.5


def test_parse_retry_after_rejects_negative_and_garbage() -> None:
    assert openai_compatible._parse_retry_after("-5") == 0.0
    assert openai_compatible._parse_retry_after("not-a-number") is None
    assert openai_compatible._parse_retry_after("") is None
    assert openai_compatible._parse_retry_after(None) is None


# ---------- Reranker model allowlist ---------------------------------------


def test_reranker_allowlist_accepts_known_models() -> None:
    _validate_model_name("BAAI/bge-reranker-v2-m3")  # default
    _validate_model_name("cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_reranker_allowlist_rejects_unknown_models() -> None:
    """Untrusted HuggingFace repos can ship PyTorch ``.bin`` files
    whose unpickling executes arbitrary code at load time."""
    with pytest.raises(MissingDependencyError, match="not in the allowlist"):
        _validate_model_name("attacker/malicious-reranker")
    with pytest.raises(MissingDependencyError, match="not in the allowlist"):
        _validate_model_name("../../etc/passwd")


# ---------- Provider-id Literal --------------------------------------------


def test_llm_provider_literal_rejects_unknown_values() -> None:
    """``llm_provider`` is a Literal so typos like ``"openais"`` fail
    at config load instead of silently routing to Ollama at runtime."""
    from pydantic import ValidationError

    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(llm_provider="openais")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Settings(llm_provider="claude")  # type: ignore[arg-type]
    # Spot-check the known-good values still work.
    Settings(llm_provider="ollama")
    Settings(llm_provider="openai-compatible")
    Settings(llm_provider="template")
