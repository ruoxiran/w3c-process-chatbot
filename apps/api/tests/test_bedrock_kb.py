"""Tests for AWS Bedrock Knowledge Base retrieval (Option B).

Two layers:

1. ``BedrockKnowledgeBaseClient`` maps the ``bedrock-agent-runtime`` Retrieve
   response shape into ``Citation`` objects (verified against a fake runtime).
2. The workflow merges KB passages into the retrieved citation pool, records a
   trace step, and degrades gracefully when the KB call fails.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.models.schemas import ChatRequest, Citation, SourceType
from app.services.bedrock_kb import BedrockKnowledgeBaseClient
from app.workflows.chat_workflow import ChatWorkflow


# ---------- Fake bedrock-agent-runtime -----------------------------------


class _FakeAgentRuntime:
    def __init__(self, results: list[dict]) -> None:
        self._results = results
        self.calls: list[dict] = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return {"retrievalResults": self._results}


def _client_with(results: list[dict], **kw) -> BedrockKnowledgeBaseClient:
    client = BedrockKnowledgeBaseClient("kb-123", "us-east-1", **kw)
    client._runtime = _FakeAgentRuntime(results)
    return client


# ---------- Client-level mapping -----------------------------------------


def test_retrieve_maps_web_and_s3_results_and_skips_empty() -> None:
    results = [
        {
            "content": {"text": "The W3C Patent Policy governs royalty-free licensing."},
            "location": {"type": "WEB", "webLocation": {"url": "https://www.w3.org/policies/patent-policy/"}},
            "metadata": {"title": "W3C Patent Policy"},
            "score": 0.9,
        },
        {
            "content": {"text": "FAQ answer about exclusion opportunities."},
            "location": {"type": "S3", "s3Location": {"uri": "s3://kb-bucket/patent-faq.html"}},
            "metadata": {},
            "score": 0.8,
        },
        {"content": {"text": "   "}},  # empty text → skipped
    ]
    citations = _client_with(results, max_results=5).retrieve("patent policy exclusions")

    assert len(citations) == 2
    assert all(c.source_type == SourceType.related_policy for c in citations)
    # Web result: metadata title + web url.
    assert citations[0].title == "W3C Patent Policy"
    assert str(citations[0].url) == "https://www.w3.org/policies/patent-policy/"
    assert "royalty-free" in (citations[0].quote or "")
    # S3 result: title falls back to the file tail, url is the s3 uri.
    assert citations[1].title == "patent-faq.html"
    assert str(citations[1].url) == "s3://kb-bucket/patent-faq.html"


def test_retrieve_passes_query_and_result_count() -> None:
    client = _client_with([], max_results=4)
    client.retrieve("how do exclusions work")
    call = client._runtime.calls[0]
    assert call["knowledgeBaseId"] == "kb-123"
    assert call["retrievalQuery"] == {"text": "how do exclusions work"}
    assert call["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"] == 4


def test_quote_is_truncated() -> None:
    long_text = "x" * 5000
    citations = _client_with([{"content": {"text": long_text}}]).retrieve("q")
    assert len(citations[0].quote) == 1500


def test_url_falls_back_to_kb_urn_when_no_location() -> None:
    citations = _client_with([{"content": {"text": "orphan passage"}}]).retrieve("q")
    assert str(citations[0].url) == "bedrock-kb://kb-123"


def test_managed_kb_metadata_shape() -> None:
    # The shape a managed KB actually returns: _document_title / _source_uri
    # metadata plus an s3Location; _source_uri (clickable) wins over s3 uri.
    results = [
        {
            "content": {"text": "A Patent Review Draft is a version of a W3C Specification..."},
            "location": {"type": "S3", "s3Location": {"uri": "https://bucket.s3.amazonaws.com/kb/patent-policy-faq.md"}},
            "metadata": {
                "_document_title": "patent-policy-faq.md",
                "_source_uri": "https://bucket.s3.amazonaws.com/kb/patent-policy-faq.md",
                "_file_type": "PLAIN_TEXT",
            },
            "score": 0.64,
        },
    ]
    citations = _client_with(results).retrieve("patent review draft")
    assert citations[0].title == "patent-policy-faq.md"
    assert str(citations[0].url) == "https://bucket.s3.amazonaws.com/kb/patent-policy-faq.md"


class _ManagedFakeRuntime:
    """Rejects vectorSearchConfiguration like a managed KB; succeeds without it."""

    def __init__(self, results: list[dict]) -> None:
        self._results = results
        self.calls: list[dict] = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        if "retrievalConfiguration" in kwargs:
            raise RuntimeError(
                "ValidationException: vectorSearchConfiguration is not supported for "
                "managed knowledge bases. Use managedSearchConfiguration instead."
            )
        return {"retrievalResults": self._results}


def test_managed_kb_retries_without_configuration() -> None:
    client = BedrockKnowledgeBaseClient("kb-managed", "us-east-1")
    client._runtime = _ManagedFakeRuntime([{"content": {"text": "managed passage"}}])

    citations = client.retrieve("q")

    assert len(citations) == 1
    # First attempt carried the config (rejected), retry dropped it.
    assert "retrievalConfiguration" in client._runtime.calls[0]
    assert "retrievalConfiguration" not in client._runtime.calls[1]


# ---------- Workflow merge -----------------------------------------------


class _FakeKB:
    def __init__(self, citations: list[Citation]) -> None:
        self._citations = citations
        self.calls: list[str] = []

    def retrieve(self, query: str) -> list[Citation]:
        self.calls.append(query)
        return self._citations


def _kb_settings() -> Settings:
    return Settings(
        app_env="development",
        llm_provider="template",  # no real LLM needed; retrieval still runs
        w3c_api_enabled=False,
        llm_router_enabled=False,
        hyde_enabled=False,
    )


def test_workflow_merges_kb_citations() -> None:
    kb_citation = Citation(
        title="Patent Policy FAQ",
        url="https://www.w3.org/policies/patent-policy/faq",
        source_type=SourceType.related_policy,
        quote="The Patent Policy FAQ explains exclusion opportunities.",
    )
    fake_kb = _FakeKB([kb_citation])
    workflow = ChatWorkflow(_kb_settings(), bedrock_kb_client=fake_kb)

    response = workflow.run(ChatRequest(message="How do W3C patent exclusions work?"))

    assert fake_kb.calls == ["How do W3C patent exclusions work?"]
    assert response.audit["bedrock_kb_hits"] == 1
    assert any(str(c.url) == "https://www.w3.org/policies/patent-policy/faq" for c in response.citations)
    assert any(step.id == "bedrock_kb" for step in response.workflow_trace)


def test_workflow_survives_kb_failure() -> None:
    class _BoomKB:
        def retrieve(self, query: str) -> list[Citation]:
            raise RuntimeError("AccessDeniedException")

    workflow = ChatWorkflow(_kb_settings(), bedrock_kb_client=_BoomKB())

    response = workflow.run(ChatRequest(message="How do W3C patent exclusions work?"))

    # Non-fatal: request still returns, error recorded, no KB hits merged.
    assert response.audit["bedrock_kb_error"] == "RuntimeError"
    assert "bedrock_kb_hits" not in response.audit
