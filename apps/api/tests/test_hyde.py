"""Unit tests for the HyDE hypothetical-passage helper."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.config import Settings
from app.services.hyde import (
    HypotheticalPassageResult,
    _clear_cache_for_tests,
    generate_hypothetical_passage,
)


def _settings() -> Settings:
    return Settings(llm_provider="ollama", w3c_api_enabled=False)


@pytest.fixture(autouse=True)
def _isolate_cache():
    """HyDE caches by (model, query). Tests must not see each other's results."""
    _clear_cache_for_tests()
    yield
    _clear_cache_for_tests()


class _FakeClient:
    """Spy that records calls and returns a canned JSON dict."""

    def __init__(self, payload: dict[str, Any] | None = None, raises: Exception | None = None) -> None:
        self.payload = payload
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, *, model: str, prompt: str, num_predict: int) -> dict[str, Any]:
        self.calls.append({"model": model, "prompt": prompt, "num_predict": num_predict})
        if self.raises:
            raise self.raises
        return self.payload or {}


def test_returns_passage_from_well_formed_json() -> None:
    client = _FakeClient(payload={"passage": "Specs advance from CR to PR after Wide Review and exit criteria are met."})
    result = generate_hypothetical_passage(
        "from CR to REC?", settings=_settings(), client=client, model="gpt-test"
    )
    assert "CR to PR" in result.passage
    assert result.error is None
    assert result.model == "gpt-test"


def test_accepts_alternative_json_field_names() -> None:
    """Some models return ``answer`` or ``draft`` instead of ``passage``."""
    for field in ("answer", "draft"):
        _clear_cache_for_tests()
        client = _FakeClient(payload={field: "Hypothetical answer."})
        result = generate_hypothetical_passage(
            f"test {field}", settings=_settings(), client=client, model="m"
        )
        assert result.passage == "Hypothetical answer."


def test_empty_passage_on_unparseable_payload() -> None:
    """Garbage JSON returns an empty passage — never raises out of the helper."""
    client = _FakeClient(payload={"unrelated_field": "noise"})
    result = generate_hypothetical_passage(
        "anything", settings=_settings(), client=client, model="m"
    )
    assert result.passage == ""
    assert result.error is None


def test_collapses_whitespace_and_caps_length() -> None:
    """Multi-line, padded model output is normalised so it doesn't blow up
    the downstream retrieval query."""
    long_text = "a" * 2000  # 2000 chars
    client = _FakeClient(payload={"passage": long_text})
    result = generate_hypothetical_passage(
        "long", settings=_settings(), client=client, model="m"
    )
    assert len(result.passage) <= 600  # _MAX_CHARS


def test_normalises_internal_whitespace() -> None:
    client = _FakeClient(payload={"passage": "Line 1.\n\n  Line 2.  \tLine 3."})
    result = generate_hypothetical_passage(
        "ws", settings=_settings(), client=client, model="m"
    )
    assert result.passage == "Line 1. Line 2. Line 3."


def test_llm_failure_yields_empty_passage_with_error_tag() -> None:
    client = _FakeClient(raises=RuntimeError("upstream 503"))
    result = generate_hypothetical_passage(
        "x", settings=_settings(), client=client, model="m"
    )
    assert result.passage == ""
    assert result.error == "RuntimeError"


def test_cached_by_message_and_model() -> None:
    """Repeated calls with the same (message, model) hit the cache —
    one LLM call across both invocations."""
    client = _FakeClient(payload={"passage": "p"})
    first = generate_hypothetical_passage("same query", settings=_settings(), client=client, model="m")
    second = generate_hypothetical_passage("same query", settings=_settings(), client=client, model="m")
    assert first.passage == second.passage
    assert len(client.calls) == 1  # second call served from cache


def test_different_models_get_different_cache_entries() -> None:
    """A model switch must NOT serve a stale result from a previous model."""
    client_a = _FakeClient(payload={"passage": "result-from-model-a"})
    client_b = _FakeClient(payload={"passage": "result-from-model-b"})
    first = generate_hypothetical_passage("q", settings=_settings(), client=client_a, model="model-a")
    second = generate_hypothetical_passage("q", settings=_settings(), client=client_b, model="model-b")
    assert first.passage == "result-from-model-a"
    assert second.passage == "result-from-model-b"


def test_dataclass_is_frozen() -> None:
    """The result is frozen — defensive against accidental mutation
    of cached entries."""
    result = HypotheticalPassageResult(passage="p", model="m")
    with pytest.raises(Exception):
        result.passage = "mutated"  # type: ignore[misc]
