"""Tests for server-side ``provider_choice`` selection.

Unlike ``provider_override`` (a browser-supplied endpoint + key), a
``provider_choice`` just names one of the server's own configured providers
("kimi" / "bedrock"). The server uses its own .env credentials; no secret
crosses the wire. These tests lock in that the choice picks the right
server-side client and default model, and takes precedence over the
server default provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import Settings
from app.models.schemas import ChatRequest
from app.workflows.chat_workflow import ChatWorkflow


def _settings() -> Settings:
    return Settings(
        app_env="development",
        llm_provider="bedrock",
        llm_model="us.anthropic.claude-sonnet-5",
        openai_compatible_model="moonshot-v1-32k",
        w3c_api_enabled=False,
        llm_router_enabled=False,
    )


@dataclass
class _Capture:
    label: str
    text: str
    captured_model: str | None = field(default=None)

    def generate_answer(self, *, model, history, **_kwargs):
        self.captured_model = model
        from app.services.bedrock import BedrockGeneration

        return BedrockGeneration(text=f"{self.text} ({model})", model=model)

    # The workflow may probe for streaming support; force the sync path.
    def __contains__(self, _item):  # pragma: no cover - defensive
        return False


def _workflow(bedrock: _Capture, kimi: _Capture) -> ChatWorkflow:
    return ChatWorkflow(_settings(), bedrock_client=bedrock, openai_compatible_client=kimi)


def test_provider_choice_bedrock_uses_bedrock_client_and_default_model() -> None:
    bedrock = _Capture("BEDROCK", "answer from bedrock")
    kimi = _Capture("KIMI", "answer from kimi")
    workflow = _workflow(bedrock, kimi)

    response = workflow.run(
        ChatRequest(message="What does a Staff Contact do?", provider_choice="bedrock")
    )

    assert bedrock.captured_model == "us.anthropic.claude-sonnet-5"
    assert kimi.captured_model is None
    assert response.audit["model_provider_source"] == "choice"


def test_provider_choice_kimi_uses_openai_compatible_client() -> None:
    bedrock = _Capture("BEDROCK", "answer from bedrock")
    kimi = _Capture("KIMI", "answer from kimi")
    workflow = _workflow(bedrock, kimi)

    response = workflow.run(
        ChatRequest(message="What does a Staff Contact do?", provider_choice="kimi")
    )

    assert kimi.captured_model == "moonshot-v1-32k"
    assert bedrock.captured_model is None
    assert response.audit["model_provider_source"] == "choice"


def test_provider_choice_honours_explicit_model() -> None:
    bedrock = _Capture("BEDROCK", "answer from bedrock")
    kimi = _Capture("KIMI", "answer from kimi")
    workflow = _workflow(bedrock, kimi)

    workflow.run(
        ChatRequest(
            message="What does a Staff Contact do?",
            provider_choice="bedrock",
            model="amazon.nova-pro-v1:0",
        )
    )

    assert bedrock.captured_model == "amazon.nova-pro-v1:0"
