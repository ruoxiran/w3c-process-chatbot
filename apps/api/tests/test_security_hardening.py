"""Regression tests for the security and observability guards.

Each test pins a specific guard so a future refactor can't silently
remove it. None of these exercise the network — they hit the helper
function directly.
"""

from __future__ import annotations

import json
import logging

import pytest

from app.core.logging_setup import (
    JsonFormatter,
    log_event,
    new_request_id,
    set_request_id,
    setup_logging,
)
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


# ---------- Structured logging ---------------------------------------------


def _capture_json_record(emit_fn) -> dict:
    """Invoke ``emit_fn`` with a handler that captures one JSON record."""
    captured: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(JsonFormatter().format(record))

    test_logger = logging.getLogger(f"test.{new_request_id()}")
    test_logger.addHandler(_CaptureHandler())
    test_logger.setLevel(logging.INFO)
    try:
        emit_fn(test_logger)
    finally:
        for handler in list(test_logger.handlers):
            test_logger.removeHandler(handler)
    assert captured, "no log line was emitted"
    return json.loads(captured[-1])


def test_log_event_emits_structured_json_with_request_id() -> None:
    set_request_id("abc123xyz789")
    record = _capture_json_record(
        lambda log: log_event(log, "retriever", duration_ms=42.1, citations=12)
    )
    assert record["request_id"] == "abc123xyz789"
    assert record["stage"] == "retriever"
    assert record["status"] == "ok"
    assert record["duration_ms"] == 42.1
    assert record["citations"] == 12


def test_log_event_promotes_known_fields_and_prefixes_unknown_ones() -> None:
    set_request_id("rid-known-unknown")
    record = _capture_json_record(
        lambda log: log_event(log, "smoke", model="gpt-4.1", top_k=8)
    )
    # ``model`` is a known structured field, so it stays at top level.
    assert record["model"] == "gpt-4.1"
    # ``top_k`` is unknown, so it lands as ``top_k`` (stripped of the
    # ``ev_`` storage prefix) at top level — that's what the formatter
    # does for any record attribute starting with ``ev_``.
    assert record["top_k"] == 8


def test_setup_logging_is_idempotent() -> None:
    """A second call must not stack handlers — otherwise log lines get
    duplicated every time the module reloads."""
    setup_logging()
    setup_logging()
    setup_logging()
    json_handlers = [
        h for h in logging.getLogger().handlers if isinstance(h.formatter, JsonFormatter)
    ]
    assert len(json_handlers) == 1
