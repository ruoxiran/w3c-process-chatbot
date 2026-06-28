"""Unit tests for the chain-of-verification claim auditor.

Each test pins one slice of behaviour with a fake JSON client — no
network. Tests deliberately don't try the regex sentence splitter
in isolation; we test through the public ``verify_claims`` entry
point with realistic-looking answers.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.config import Settings
from app.models.schemas import Citation
from app.services.claim_verifier import (
    ClaimVerificationResult,
    _clear_cache_for_tests,
    verify_claims,
)


def _settings() -> Settings:
    return Settings(llm_provider="ollama", w3c_api_enabled=False)


def _citation(quote: str, url: str = "https://www.w3.org/policies/process/") -> Citation:
    return Citation(
        title="Process Document",
        url=url,
        source_type="process",
        heading_path="Process > 6. Technical Reports",
        quote=quote,
    )


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache_for_tests()
    yield
    _clear_cache_for_tests()


class _FakeClient:
    def __init__(self, payload: dict[str, Any] | None = None, raises: Exception | None = None):
        self.payload = payload
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, *, model: str, prompt: str, num_predict: int) -> dict[str, Any]:
        self.calls.append({"model": model, "prompt": prompt, "num_predict": num_predict})
        if self.raises:
            raise self.raises
        return self.payload or {}


def test_skips_when_no_citations() -> None:
    """No citations = nothing to verify against. Pure no-op path."""
    client = _FakeClient()
    result = verify_claims("answer text [S1].", [], settings=_settings(), client=client)
    assert result.annotated_answer == "answer text [S1]."
    assert result.skipped_reason == "no_input"
    assert client.calls == []


def test_skips_when_no_candidate_unsourced_sentences() -> None:
    """An answer where every claim already carries a ``[Sn]`` tag has
    no candidates — the LLM is never called."""
    answer = "Wide Review is required before CR transition [S1]."
    client = _FakeClient(payload={"unsupported": [1]})
    result = verify_claims(
        answer, [_citation("Wide Review before CR")], settings=_settings(), client=client
    )
    assert result.skipped_reason == "no_candidates"
    assert result.annotated_answer == answer
    assert client.calls == []


def test_flags_an_unsupported_claim_with_inline_marker() -> None:
    answer = (
        "Wide Review is required before CR transition [S1]. "
        "The Process also requires an AC vote of 80 percent for approval."
    )
    client = _FakeClient(payload={"unsupported": [1]})
    result = verify_claims(
        answer,
        [_citation("Wide Review must precede CR transition")],
        settings=_settings(),
        client=client,
    )
    assert "[unverified]" in result.annotated_answer
    assert "80 percent" in result.annotated_answer
    assert len(result.unsupported_claims) == 1
    assert "80 percent" in result.unsupported_claims[0].sentence


def test_does_not_modify_answer_when_llm_returns_empty_list() -> None:
    """LLM says nothing was unsupported — no inline annotation, no audit
    entries, but no skipped_reason either (we did try)."""
    answer = (
        "Wide Review is required before CR transition [S1]. "
        "The Working Group must also document implementation experience."
    )
    client = _FakeClient(payload={"unsupported": []})
    result = verify_claims(
        answer,
        [_citation("Wide Review before CR; implementation experience required")],
        settings=_settings(),
        client=client,
    )
    assert "[unverified]" not in result.annotated_answer
    assert result.annotated_answer == answer
    assert result.unsupported_claims == ()
    assert result.error is None


def test_llm_error_falls_back_to_original_answer() -> None:
    answer = "Wide Review is required before CR [S1]. The AC requires 80 percent approval."
    client = _FakeClient(raises=RuntimeError("upstream 503"))
    result = verify_claims(
        answer, [_citation("...")], settings=_settings(), client=client
    )
    assert result.annotated_answer == answer
    assert result.error == "RuntimeError"
    assert result.skipped_reason == "llm_error"


def test_cache_hit_on_repeated_call() -> None:
    answer = "Wide Review is required before CR [S1]. The AC requires 80 percent."
    citations = [_citation("Wide Review precedes CR")]
    client = _FakeClient(payload={"unsupported": [1]})
    first = verify_claims(answer, citations, settings=_settings(), client=client)
    second = verify_claims(answer, citations, settings=_settings(), client=client)
    assert first.annotated_answer == second.annotated_answer
    assert len(client.calls) == 1  # second call served from cache


def test_caps_candidate_sentences_to_avoid_prompt_bloat() -> None:
    """An answer with 20 uncited claim-shaped sentences must still
    produce a bounded prompt — the verifier truncates to a fixed cap."""
    parts = [
        f"The Working Group must follow procedure number {i} carefully."
        for i in range(20)
    ]
    answer = " ".join(parts)
    client = _FakeClient(payload={"unsupported": []})
    verify_claims(answer, [_citation("...")], settings=_settings(), client=client)
    assert len(client.calls) == 1
    # The prompt should list at most 8 sentences (the implementation cap).
    prompt = client.calls[0]["prompt"]
    numbered = [line for line in prompt.split("\n") if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."))]
    # 8 numbered lines for the sentences (1-8); 9. should not appear.
    assert "9." not in prompt
    assert "8." in prompt


def test_ignores_garbage_indices_from_llm() -> None:
    """If the LLM hallucinates indices out of range (or non-int values),
    drop them quietly instead of crashing."""
    answer = (
        "Wide Review is required before CR [S1]. "
        "The Working Group must document implementation experience."
    )
    client = _FakeClient(payload={"unsupported": [99, "foo", -1, 1]})
    result = verify_claims(
        answer, [_citation("Wide Review before CR")], settings=_settings(), client=client
    )
    # Only index 1 survives; the others are garbage.
    assert len(result.unsupported_claims) == 1


def test_dataclass_is_frozen() -> None:
    result = ClaimVerificationResult(
        annotated_answer="x", unsupported_claims=(), model="m"
    )
    with pytest.raises(Exception):
        result.model = "y"  # type: ignore[misc]
