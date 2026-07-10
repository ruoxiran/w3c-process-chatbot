"""Tests for the streaming-path retry behaviour of OpenAICompatibleClient.

The sync path already retries transient upstream statuses via
``_post_with_backoff``; the UI, however, always uses ``stream_answer``.
These tests pin the contract that connection-time 429/5xx responses are
retried with backoff BEFORE any token is yielded, and that a non-retryable
error still raises ``httpx.HTTPStatusError``.
"""
from __future__ import annotations

from contextlib import contextmanager

import httpx
import pytest

import app.services.openai_compatible as oc
from app.services.openai_compatible import OpenAICompatibleClient


_SSE_BODY = (
    'data: {"choices":[{"delta":{"content":"Hello "}}]}\n'
    'data: {"choices":[{"delta":{"content":"world."}}]}\n'
    "data: [DONE]\n"
)


def _fake_stream_factory(responses: list[httpx.Response], calls: list[int]):
    @contextmanager
    def fake_stream(method: str, url: str, **kwargs):
        calls.append(1)
        response = responses.pop(0)
        response.request = httpx.Request(method, url)
        yield response

    return fake_stream


def _stream_kwargs() -> dict[str, object]:
    return dict(
        model="test-model",
        question="What is a WD?",
        locale="en",
        citations=[],
        fallback_answer="fallback",
        fallback_next_steps=[],
    )


def test_stream_answer_retries_transient_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    responses = [
        httpx.Response(429, json={"error": {"type": "rate_limit_reached_error"}}),
        httpx.Response(200, text=_SSE_BODY),
    ]
    monkeypatch.setattr(oc.httpx, "stream", _fake_stream_factory(responses, calls))
    monkeypatch.setattr(oc.time, "sleep", lambda _s: None)

    client = OpenAICompatibleClient("https://example.test/v1", "key")
    deltas = list(client.stream_answer(**_stream_kwargs()))

    assert deltas == ["Hello ", "world."]
    assert len(calls) == 2


def test_stream_answer_raises_after_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    responses = [
        httpx.Response(429, json={"error": {"type": "engine_overloaded_error"}})
        for _ in range(oc._MAX_RETRIES + 1)
    ]
    monkeypatch.setattr(oc.httpx, "stream", _fake_stream_factory(responses, calls))
    monkeypatch.setattr(oc.time, "sleep", lambda _s: None)

    client = OpenAICompatibleClient("https://example.test/v1", "key")
    with pytest.raises(httpx.HTTPStatusError):
        list(client.stream_answer(**_stream_kwargs()))

    assert len(calls) == oc._MAX_RETRIES + 1


def test_stream_answer_does_not_retry_non_transient_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    responses = [httpx.Response(401, json={"error": {"type": "auth_error"}})]
    monkeypatch.setattr(oc.httpx, "stream", _fake_stream_factory(responses, calls))
    monkeypatch.setattr(oc.time, "sleep", lambda _s: None)

    client = OpenAICompatibleClient("https://example.test/v1", "key")
    with pytest.raises(httpx.HTTPStatusError):
        list(client.stream_answer(**_stream_kwargs()))

    assert len(calls) == 1


def test_stream_answer_drops_temperature_on_400_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kimi k2.5 pins temperature to 1; our 0.1 gets a 400. The client must
    retry once without the temperature key and succeed."""
    calls: list[int] = []
    payloads: list[dict[str, object]] = []
    responses = [
        httpx.Response(
            400,
            json={"error": {"message": "invalid temperature: only 1 is allowed for this model", "type": "invalid_request_error"}},
        ),
        httpx.Response(200, text=_SSE_BODY),
    ]

    from contextlib import contextmanager

    @contextmanager
    def fake_stream(method: str, url: str, **kwargs):
        calls.append(1)
        payloads.append(dict(kwargs.get("json") or {}))
        response = responses.pop(0)
        response.request = httpx.Request(method, url)
        yield response

    monkeypatch.setattr(oc.httpx, "stream", fake_stream)
    monkeypatch.setattr(oc.time, "sleep", lambda _s: None)

    client = OpenAICompatibleClient("https://example.test/v1", "key")
    deltas = list(client.stream_answer(**_stream_kwargs()))

    assert deltas == ["Hello ", "world."]
    assert len(calls) == 2
    assert "temperature" in payloads[0]
    assert "temperature" not in payloads[1]


def test_chat_drops_temperature_on_400_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict[str, object]] = []
    responses = [
        httpx.Response(
            400,
            json={"error": {"message": "invalid temperature: only 1 is allowed for this model"}},
        ),
        httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}),
    ]

    def fake_post(url: str, **kwargs):
        payloads.append(dict(kwargs.get("json") or {}))
        response = responses.pop(0)
        response.request = httpx.Request("POST", url)
        return response

    monkeypatch.setattr(oc.httpx, "post", fake_post)
    monkeypatch.setattr(oc.time, "sleep", lambda _s: None)

    client = OpenAICompatibleClient("https://example.test/v1", "key")
    text = client._chat(model="kimi-k2.5", messages=[{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=10)

    assert text == "ok"
    assert "temperature" in payloads[0]
    assert "temperature" not in payloads[1]


def test_stream_answer_respects_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    calls: list[int] = []
    responses = [
        httpx.Response(429, headers={"retry-after": "2"}, json={"error": {}}),
        httpx.Response(200, text=_SSE_BODY),
    ]
    monkeypatch.setattr(oc.httpx, "stream", _fake_stream_factory(responses, calls))
    monkeypatch.setattr(oc.time, "sleep", sleeps.append)

    client = OpenAICompatibleClient("https://example.test/v1", "key")
    deltas = list(client.stream_answer(**_stream_kwargs()))

    assert deltas == ["Hello ", "world."]
    assert sleeps == [2.0]
